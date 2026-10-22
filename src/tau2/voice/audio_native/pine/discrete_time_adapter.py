"""Clock Pine's continuous WebRTC media with Tau's real-time tick lifecycle."""

import asyncio
import audioop
from typing import Any

from tau2.config import PINE_MEDIA_SAMPLE_RATE
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.data_model.message import ToolCall
from tau2.data_model.pine_config import PineConfig
from tau2.data_model.transcript import LiveTranscriptFragment, LiveTranscriptTick
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.pine.provider import AudioFrame, PineProvider
from tau2.voice.audio_native.tick_result import TickResult


class DiscreteTimePineAdapter(DiscreteTimeAdapter):
    """Use SDK SSE tools and transcripts independently of room audio."""

    def __init__(
        self,
        *,
        tick_duration_ms: int,
        config: PineConfig,
        audio_format: AudioFormat | None = None,
        language: str = "en",
        provider: PineProvider | None = None,
    ) -> None:
        if audio_format is not None and audio_format != TELEPHONY_AUDIO_FORMAT:
            raise ValueError("Pine expects 8 kHz mu-law telephony audio")
        super().__init__(tick_duration_ms, audio_format, send_audio_instant=True)
        self.realtime_pacing = True
        self.provider = provider or PineProvider(config, language=language)
        self._bg_loop = BackgroundAsyncLoop()
        self._converter = StreamingTelephonyConverter(
            input_sample_rate=PINE_MEDIA_SAMPLE_RATE,
            output_sample_rate=PINE_MEDIA_SAMPLE_RATE,
        )
        self._seen_events: set[str] = set()
        self._seen_tool_calls: set[str] = set()
        self._transcript_text: dict[int, str] = {}
        self._ended = False

    @property
    def is_connected(self) -> bool:
        return self.provider.is_connected

    def connect(self, system_prompt, tools, vad_config=None, modality="audio") -> None:
        """Connect without overriding the server's speech detection settings."""
        if modality != "audio":
            raise ValueError("Pine supports only full-duplex audio modality")
        if self.is_connected:
            raise RuntimeError("Pine adapter is already connected")
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self.provider.connect(system_prompt, tools), timeout=120
            )
        except BaseException:
            self._bg_loop.stop()
            raise

    def disconnect(self) -> None:
        """Close the remote call and release background resources."""
        try:
            if self._bg_loop.is_running:
                self._bg_loop.run_coroutine(self.provider.close(), timeout=30)
        finally:
            self._bg_loop.stop()
            self.clear_buffers()
            self._converter.reset()
            self._seen_events.clear()
            self._seen_tool_calls.clear()
            self._transcript_text.clear()
            self._ended = False

    def run_tick(self, user_audio: bytes, tick_number: int | None = None) -> TickResult:
        self._tick_count = (
            tick_number if tick_number is not None else self._tick_count + 1
        )
        result = self._bg_loop.run_coroutine(
            self._async_run_tick(user_audio, self._tick_count), timeout=30
        )
        # Transcripts arrive independently of media. Keep their own evidence,
        # never manufacture per-audio-item alignment or text for silence.
        if result.live_transcript:
            result.proportional_transcript = "".join(
                fragment.text
                for fragment in result.live_transcript.fragments
                if fragment.speaker == "agent"
            )
        pcm = audioop.ulaw2lin(result.agent_audio_data, 2)
        result.contains_speech = bool(pcm) and audioop.max(pcm, 2) > 32
        if result.live_transcript is not None:
            frame_bytes = self.audio_format.sample_rate * 2 // 50
            for offset in range(0, len(pcm), frame_bytes):
                frame = pcm[offset : offset + frame_bytes]
                if audioop.max(frame, 2) > 32:
                    result.live_transcript.speech_intervals_ms.append(
                        (
                            offset * 500 / self.audio_format.sample_rate,
                            (offset + len(frame)) * 500 / self.audio_format.sample_rate,
                        )
                    )
        # Deliver any buffered last audio before reporting remote termination.
        result.session_ended = (
            self._ended
            and not self._buffered_agent_audio
            and self.provider.events.empty()
        )
        return result

    async def _flush_pending_tool_results(self) -> None:
        while self._pending_tool_results:
            call_id, result, _, _ = self._pending_tool_results[0]
            await self.provider.submit_result(call_id, result)
            self._pending_tool_results.pop(0)

    async def _execute_tick(
        self, user_audio: bytes, tick_number: int, result: TickResult, tick_start: float
    ) -> None:
        self.provider.check_health()
        result.live_transcript = LiveTranscriptTick(session_id=self.provider.call_id)
        deadline = tick_start + self.tick_duration_ms / 1000

        async def receive():
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    # Process already received events even when generation lagged.
                    while not self.provider.events.empty():
                        await self._receive_event(
                            result, self.provider.events.get_nowait()
                        )
                    break
                try:
                    event = await asyncio.wait_for(
                        self.provider.events.get(), timeout=remaining
                    )
                except TimeoutError:
                    break
                await self._receive_event(result, event)

        await asyncio.gather(
            self.provider.send_audio(self._converter.convert_input(user_audio)),
            receive(),
        )
        self.provider.check_health()

    async def _receive_event(self, result: TickResult, event: Any) -> None:
        if (
            not isinstance(event, AudioFrame)
            and event.event == "call.tool_call.requested"
        ):
            call_id = event.data["data"]["id"]
            if call_id not in self._seen_tool_calls:
                if not await self.provider.tool_call_pending(call_id):
                    self._seen_tool_calls.add(call_id)
        self._process_event(result, event)

    def _process_event(self, result: TickResult, event: Any) -> None:
        if isinstance(event, AudioFrame):
            result.agent_audio_chunks.append(
                (self._converter.convert_output(event.data), None)
            )
            return
        envelope = event.data
        event_id = envelope.get("id", event.id)
        if event_id is not None:
            if event_id in self._seen_events:
                return
            self._seen_events.add(event_id)
        result.events.append(envelope)
        data = envelope.get("data", {})
        if event.event == "call.tool_call.requested":
            if data["id"] not in self._seen_tool_calls:
                self._seen_tool_calls.add(data["id"])
                result.tool_calls.append(
                    ToolCall(
                        id=data["id"], name=data["name"], arguments=data["arguments"]
                    )
                )
        elif event.event == "call.transcript.done":
            # Finalized text is authoritative (including interrupted utterances).
            # Keep reported timing on the server answered_at clock. These are
            # server transcript offsets, not sample-accurate acoustic boundaries
            # or browser playback acknowledgements.
            if data["role"] not in {"agent", "user"}:
                return
            index = data["index"]
            if index in self._transcript_text:
                return
            self._transcript_text[index] = data["text"]
            result.live_transcript.fragments.append(
                LiveTranscriptFragment(
                    event_id=event_id,
                    speaker="agent" if data["role"] == "agent" else "caller",
                    text=data["text"],
                    start_ms=data["start_ms"],
                    end_ms=data["end_ms"],
                )
            )
        elif event.event == "call.ended":
            self._ended = True
