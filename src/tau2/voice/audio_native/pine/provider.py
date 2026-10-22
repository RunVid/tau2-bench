"""Pine SDK control/events with a separate LiveKit audio connection.

No model protocol is emulated. Tool calls leave this provider for execution by
Tau's environment, then return through the public SDK result endpoint.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tau2.config import PINE_MEDIA_SAMPLE_RATE
from tau2.data_model.pine_config import PineConfig
from tau2.environment.tool import Tool


@dataclass(frozen=True)
class AudioFrame:
    """Mono PCM16 received from the remote media track."""

    data: bytes


class PineProvider:
    """Own one custom agent, call, SSE subscription and media participant."""

    def __init__(
        self, config: PineConfig, *, language: str = "en", client: Any = None
    ) -> None:
        self.config = config
        self.language = language
        self.client = client
        self._owns_client = client is None
        self.call_id: str | None = None
        self.agent_id: str | None = None
        self.events: asyncio.Queue = asyncio.Queue(maxsize=4096)
        self._tasks: set[asyncio.Task] = set()
        self._failure: BaseException | None = None
        self._room = None
        self._source = None
        self._connected = False
        self._media_ready: asyncio.Event | None = None
        self._closing = False
        self._media_disconnected_at: float | None = None
        self.ended = False

    @property
    def is_connected(self) -> bool:
        """Whether call setup and media connection completed."""
        return self._connected

    def _spawn(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)

        def finished(task):
            self._tasks.discard(task)
            if not task.cancelled() and task.exception() is not None:
                self._failure = task.exception()

        task.add_done_callback(finished)

    def check_health(self) -> None:
        """Surface background transport failures instead of silently stalling."""
        if self._failure is not None:
            raise RuntimeError("Pine provider transport failed") from self._failure
        if (
            self._media_disconnected_at is not None
            and not self.ended
            and asyncio.get_running_loop().time() - self._media_disconnected_at > 5
        ):
            raise RuntimeError("Pine media disconnected without a terminal call event")

    async def connect(self, system_prompt: str, tools: list[Tool]) -> None:
        """Create an episode-specific agent/call and join its audio room."""
        try:
            from livekit import rtc
            from pinevoice import AsyncPineVoice
        except ImportError as exc:
            raise ImportError(
                "Pine requires the voice extra and the custom-agent pinevoice SDK; "
                "see docs/pine-provider.md"
            ) from exc

        self.ended = False
        self._closing = False
        self._media_disconnected_at = None
        self._failure = None
        self.events = asyncio.Queue(maxsize=4096)
        if self.client is None:
            self.client = AsyncPineVoice()
        try:
            tool_specs = []
            for tool in tools:
                function = tool.openai_schema["function"]
                tool_specs.append(
                    {
                        "name": function["name"],
                        "description": function.get("description", ""),
                        "parameters": function["parameters"],
                        "execution": {
                            "type": "client",
                            "timeout_seconds": self.config.tool_timeout_seconds,
                        },
                    }
                )
            agent = await self.client.agents.create(
                body={
                    "name": "Tau benchmark episode",
                    "voice": {"id": self.config.voice_id},
                    "language": self.language,
                    "frontend": {
                        "instructions": self.config.frontend_instructions,
                        "model": self.config.frontend_model,
                    },
                    "backend": {
                        "instructions": self.config.render_backend_instructions(
                            system_prompt
                        ),
                        "model": self.config.backend_model,
                        "tools": tool_specs,
                    },
                }
            )
            self.agent_id = agent.id
            call_body = {
                "agent_id": agent.id,
                "channel": "web",
                "goal": "Assist the caller according to the supplied agent instructions.",
                "max_duration_seconds": self.config.max_duration_seconds,
            }
            if self.config.voice_id is not None:
                call_body["voice_id"] = self.config.voice_id
            call = await self.client.calls.create(body=call_body)
            self.call_id = call.id
            if call.connection is None:
                raise RuntimeError("Pine web call did not return media credentials")
            self._spawn(self._read_events())
            self._room = rtc.Room()
            self._media_ready = asyncio.Event()

            @self._room.on("disconnected")
            def disconnected(reason):
                if not self._closing:
                    self._media_disconnected_at = asyncio.get_running_loop().time()

            @self._room.on("track_subscribed")
            def track_subscribed(track, publication, participant):
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    self._spawn(self._read_audio(track))
                    self._media_ready.set()

            await self._room.connect(call.connection.url, call.connection.token)
            self._source = rtc.AudioSource(PINE_MEDIA_SAMPLE_RATE, 1, queue_size_ms=100)
            track = rtc.LocalAudioTrack.create_audio_track("tau-caller", self._source)
            await self._room.local_participant.publish_track(
                track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            await self._wait_for_media()
            self._connected = True
        except BaseException:
            await self.close()
            raise

    async def _wait_for_media(self) -> None:
        """Do not start the caller clock before the engine publishes its audio track."""
        async with asyncio.timeout(60):
            while not self._media_ready.is_set():
                self.check_health()
                if self.ended:
                    raise RuntimeError("Pine call ended before media became ready")
                try:
                    await asyncio.wait_for(self._media_ready.wait(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

    async def _read_events(self) -> None:
        async for event in self.client.calls.stream(self.call_id, max_reconnects=10):
            self.events.put_nowait(event)
            if event.event == "call.ended":
                self.ended = True
        if not self.ended:
            raise RuntimeError("Pine event stream ended before call.ended")

    async def _read_audio(self, track) -> None:
        from livekit import rtc

        stream = rtc.AudioStream(
            track,
            sample_rate=PINE_MEDIA_SAMPLE_RATE,
            num_channels=1,
            frame_size_ms=20,
        )
        try:
            async for frame in stream:
                self.events.put_nowait(AudioFrame(bytes(frame.frame.data)))
        finally:
            await stream.aclose()

    async def send_audio(self, pcm: bytes) -> None:
        """Publish a tick; LiveKit clocks frames without a second pacing loop."""
        from livekit import rtc

        self.check_health()
        if self.ended:
            return
        frame_bytes = PINE_MEDIA_SAMPLE_RATE * 2 // 50
        for offset in range(0, len(pcm), frame_bytes):
            data = pcm[offset : offset + frame_bytes]
            await self._source.capture_frame(
                rtc.AudioFrame(data, PINE_MEDIA_SAMPLE_RATE, 1, len(data) // 2)
            )

    async def tool_call_pending(self, tool_call_id: str) -> bool:
        """Check durable state before executing a potentially replayed request."""
        invocation = await self.client.calls.get_tool_call(self.call_id, tool_call_id)
        return invocation.status == "pending"

    async def submit_result(self, tool_call_id: str, output: str) -> None:
        """Return the environment's result without executing tools locally."""
        from pinevoice import ConflictError

        try:
            await self.client.calls.submit_tool_result(
                self.call_id, tool_call_id, body={"output": output}
            )
        except ConflictError:
            invocation = await self.client.calls.get_tool_call(
                self.call_id, tool_call_id
            )
            if invocation.status not in {"expired", "canceled"}:
                raise
            # The engine already received the deadline/cancellation outcome.
            # A late environment result must not revive that invocation or turn
            # a normal hangup into an infrastructure failure.
            logger.debug("Discarding result for terminal Pine tool invocation")

    async def close(self) -> None:
        """Release media and remote resources, including after partial setup."""
        self._connected = False
        self._closing = True
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception:
                logger.exception("Could not disconnect Pine benchmark media")
            finally:
                self._room = None
        if self._source is not None:
            try:
                await self._source.aclose()
            except Exception:
                logger.exception("Could not release Pine benchmark audio source")
            finally:
                self._source = None
        if self.client is not None:
            if self.call_id is not None and not self.ended:
                try:
                    await self.client.calls.hangup(self.call_id)
                except Exception:
                    logger.exception("Could not hang up Pine benchmark call")
            if self.agent_id is not None:
                try:
                    await self.client.agents.delete(self.agent_id)
                except Exception:
                    logger.exception("Could not delete Pine benchmark agent")
            if self._owns_client:
                await self.client.close()
                self.client = None
        self.call_id = None
        self.agent_id = None
