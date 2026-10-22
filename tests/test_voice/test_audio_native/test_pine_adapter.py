"""Key-free Pine tests exercise the same tick/SDK boundary as live calls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.data_model.message import AssistantMessage
from tau2.data_model.pine_config import PineConfig
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.transcript import LiveTranscriptTick
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.pine.discrete_time_adapter import DiscreteTimePineAdapter
from tau2.voice.audio_native.pine.provider import AudioFrame, PineProvider
from tau2.voice.audio_native.tick_result import TickResult


def config():
    return PineConfig(
        frontend_model="frontend-model",
        backend_model="backend-model",
        frontend_instructions="Speak to the caller.",
        backend_instructions="Use tools and report verified results.",
        voice_id="voice_test",
    )


def event(kind, data, event_id="evt_test"):
    return SimpleNamespace(
        id=event_id,
        event=kind,
        data={"id": event_id, "type": kind, "data": data},
    )


def result():
    return TickResult(
        tick_number=1,
        audio_sent_bytes=1600,
        audio_sent_duration_ms=200,
        live_transcript=LiveTranscriptTick(session_id="call_test"),
    )


def test_provider_config_is_required_and_serializes_prompts():
    with pytest.raises(ValueError, match="pine_config"):
        AudioNativeConfig(provider="pine")
    with pytest.raises(ValueError, match="only valid"):
        AudioNativeConfig(provider="openai", pine_config=config())
    with pytest.raises(ValueError, match="realtime_generation"):
        AudioNativeConfig(
            provider="pine", pine_config=config(), realtime_generation=False
        )
    run = AudioNativeConfig(provider="pine", pine_config=config())
    assert run.realtime_generation_enabled
    assert run.model == "frontend-model"
    assert run.model_dump()["pine_config"]["backend_model"] == "backend-model"
    assert config().render_backend_instructions("POLICY") == (
        "Use tools and report verified results.\n\nPOLICY"
    )


def test_factory_uses_config_model_and_realtime_pacing():
    adapter, model = create_adapter("pine", tick_duration_ms=20, pine_config=config())
    assert isinstance(adapter, DiscreteTimePineAdapter)
    assert adapter.realtime_pacing
    assert model == "frontend-model"


def test_replayed_tool_request_only_executes_once():
    adapter = DiscreteTimePineAdapter(tick_duration_ms=20, config=config())
    tick = result()
    call = {"id": "tc_1", "name": "get_order", "arguments": {"id": "123"}}
    adapter._process_event(tick, event("call.tool_call.requested", call))
    adapter._process_event(tick, event("call.tool_call.requested", call))
    adapter._process_event(tick, event("call.tool_call.requested", call, "evt_other"))
    assert len(tick.tool_calls) == 1
    assert tick.tool_calls[0].id == "tc_1"
    assert tick.tool_calls[0].arguments == {"id": "123"}


def test_final_transcript_keeps_server_timing_and_ignores_unspoken_deltas():
    adapter = DiscreteTimePineAdapter(tick_duration_ms=20, config=config())
    tick = result()
    adapter._process_event(
        tick,
        event(
            "call.transcript.delta",
            {"index": 0, "role": "agent", "delta": "Hello, I can help you."},
            "evt_delta",
        ),
    )
    adapter._process_event(
        tick,
        event(
            "call.transcript.done",
            {
                "index": 0,
                "role": "agent",
                "text": "Hello,",
                "start_ms": 200,
                "end_ms": 400,
            },
            "evt_done",
        ),
    )
    assert len(tick.live_transcript.fragments) == 1
    fragment = tick.live_transcript.fragments[0]
    assert fragment.text == "Hello,"
    assert fragment.start_ms == 200
    assert fragment.end_ms == 400
    assert fragment.ingress_audio_bytes is None


def test_tool_results_use_sdk_and_preserve_failed_submission_for_retry():
    async def run():
        client = SimpleNamespace(calls=SimpleNamespace(submit_tool_result=AsyncMock()))
        provider = PineProvider(config(), client=client)
        provider.call_id = "call_test"
        adapter = DiscreteTimePineAdapter(
            tick_duration_ms=20, config=config(), provider=provider
        )
        adapter.send_tool_result("tc_1", '{"delivered":true}')
        client.calls.submit_tool_result.side_effect = RuntimeError("offline")
        with pytest.raises(RuntimeError, match="offline"):
            await adapter._flush_pending_tool_results()
        assert len(adapter._pending_tool_results) == 1
        client.calls.submit_tool_result.side_effect = None
        await adapter._flush_pending_tool_results()
        client.calls.submit_tool_result.assert_awaited_with(
            "call_test", "tc_1", body={"output": '{"delivered":true}'}
        )
        assert not adapter._pending_tool_results

    asyncio.run(run())


def test_full_tick_routes_real_tau_tool_call_and_preserves_audio():
    async def run():
        provider = SimpleNamespace(
            call_id="call_test",
            events=asyncio.Queue(),
            check_health=lambda: None,
            send_audio=AsyncMock(),
            submit_result=AsyncMock(),
            tool_call_pending=AsyncMock(return_value=True),
        )
        adapter = DiscreteTimePineAdapter(
            tick_duration_ms=20, config=config(), provider=provider
        )
        provider.events.put_nowait(AudioFrame(b"\x00\x00" * 480))
        provider.events.put_nowait(
            event(
                "call.tool_call.requested",
                {"id": "tc_1", "name": "get_order", "arguments": {"order_id": "123"}},
            )
        )
        tick = await adapter._async_run_tick(b"\xff" * 160, 1)
        assert tick.audio_sent_duration_ms == 20
        assert len(tick.get_played_agent_audio()) == 160
        assert tick.tool_calls[0].name == "get_order"
        assert len(provider.send_audio.await_args.args[0]) > 900
        # Results originate from the orchestrator, never direct tool invocation.
        assert provider.submit_result.await_count == 0

    asyncio.run(run())


def test_remote_hangup_stops_agent_without_synthesizing_transcript():
    adapter = DiscreteTimePineAdapter(tick_duration_ms=20, config=config())
    tick = result()
    adapter._process_event(tick, event("call.ended", {}))
    assert adapter._ended
    message = AssistantMessage(
        role="assistant", content=None, raw_data={"session_ended": True}
    )
    assert DiscreteTimeAudioNativeAgent.is_stop(message)
    assert message.content is None
    assert not DiscreteTimeAudioNativeAgent.is_stop(
        AssistantMessage(role="assistant", content=None)
    )


def test_background_failure_is_not_silently_treated_as_silence():
    async def run():
        provider = PineProvider(config(), client=object())

        async def fail():
            raise ValueError("bad stream")

        provider._spawn(fail())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="transport failed"):
            provider.check_health()

    asyncio.run(run())


def test_replayed_expired_tool_is_not_executed():
    async def run():
        provider = SimpleNamespace(tool_call_pending=AsyncMock(return_value=False))
        adapter = DiscreteTimePineAdapter(
            tick_duration_ms=20, config=config(), provider=provider
        )
        tick = result()
        await adapter._receive_event(
            tick,
            event(
                "call.tool_call.requested",
                {"id": "tc_old", "name": "get_order", "arguments": {}},
            ),
        )
        assert not tick.tool_calls
        provider.tool_call_pending.assert_awaited_once_with("tc_old")

    asyncio.run(run())


def test_terminal_tick_drains_final_transcript_and_buffered_media():
    provider = SimpleNamespace(
        call_id="call_test",
        events=None,
        check_health=lambda: None,
        send_audio=AsyncMock(),
        submit_result=AsyncMock(),
    )
    adapter = DiscreteTimePineAdapter(
        tick_duration_ms=20, config=config(), provider=provider
    )
    adapter._bg_loop.start()

    async def enqueue():
        provider.events = asyncio.Queue()
        provider.events.put_nowait(
            event(
                "call.transcript.done",
                {
                    "index": 0,
                    "role": "agent",
                    "text": "Goodbye.",
                    "start_ms": 10,
                    "end_ms": 40,
                },
                "evt_final",
            )
        )
        provider.events.put_nowait(event("call.ended", {}, "evt_end"))
        # Accepted final media can follow the terminal control event.
        provider.events.put_nowait(AudioFrame(b"\x10\x01" * 1440))

    adapter._bg_loop.run_coroutine(enqueue())
    try:
        first = adapter.run_tick(b"\xff" * 160, tick_number=0)
        assert first.tick_number == 0
        assert first.proportional_transcript == "Goodbye."
        assert not first.session_ended
        second = adapter.run_tick(b"\xff" * 160, tick_number=1)
        assert not second.session_ended
        third = adapter.run_tick(b"\xff" * 160, tick_number=2)
        assert third.session_ended
        assert sum(t.agent_audio_bytes for t in [first, second, third]) == 480
    finally:
        adapter._bg_loop.stop()
