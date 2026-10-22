# Pine custom-agent voice provider

The `pine` provider connects the ordinary audio-native agent to a Pine Portal
custom Agent through the **Pine Python SDK**. It does not emulate an OpenAI
Realtime endpoint. Media travels through the returned LiveKit room URL/token;
SSE delivers tool requests and transcript events, and HTTP returns tool results.
Tau's environment remains the only executor of benchmark tools.

## Installation

Install the voice/dev dependencies from the checked-in lock, then install the
Pine SDK revision containing custom Agents and client tool execution:

```bash
UV_FROZEN=1 uv sync --extra voice --extra dev
uv pip install --python .venv/bin/python -e /path/to/agent3/services/portal/sdks/python
```

The SDK feature must be available before using this provider. An older SDK with
the same development version is not sufficient. Until that SDK revision is
published, use the reviewed Portal worktree; there is no dependency on a private
Git URL or an undeclared local path in the benchmark lock. Use `uv run --no-sync`
after the editable install so dependency synchronization does not remove it.
The RTC client is provided by the voice extra. The minimum LiveKit RTC version is
1.1.5 (paired with Agents 1.5.2): its upstream fix removes the hidden native
100 ms queue that could drop audio even while Python's queue was unbounded.
The adapter eagerly drains SDK audio into its bounded 4096-event queue and fails
on overflow. LiveKit's internal native queue is upstream-managed and unbounded;
the public Python API does not expose a native queue limit. This is not a strict
whole-process memory bound during a stalled Python runtime.

Set `PINE_VOICE_API_KEY` and optionally `PINE_VOICE_BASE_URL` (including `/v1`).
Credentials are read by the SDK and are never part of the serialized run config.
The caller simulator still requires its usual model/STT/TTS credentials and
account-owned caller voices; see [running the benchmark](running-the-benchmark.md).

## Configuration and smoke run

Create a local `pine-config.json` (use actual Portal model/voice catalog IDs):

```json
{
  "frontend_model": "gemini-3.6-flash",
  "backend_model": "gemini-3.6-flash",
  "voice_id": "voice_REPLACE",
  "frontend_instructions": "Speak naturally with the caller. Use verified backend information for business decisions. Do not invent tool results. When waiting, output <wait>. End the call only when the caller is finished, using <press-phone-key>End Call Button</press-phone-key>.",
  "backend_instructions": "Follow the supplied task policy. Use the business tools to investigate and perform requested actions. Use send_to_frontend to provide verified findings or necessary questions. When more conversation is needed, use wait_for_conversation_update.",
  "tool_timeout_seconds": 30,
  "max_duration_seconds": 1800
}
```

Instructions are explicit and serialized with the run. The backend also receives
the complete standard benchmark agent instructions and domain policy, unchanged.
No hidden caller goal, task solution, database state or grading information is
sent to Pine. The resolved run language is used for the Agent language.

```bash
uv run --no-sync tau2 run --audio-native --audio-native-provider pine \
  --pine-config "$(cat pine-config.json)" --realtime-generation \
  --benchmark none --domain retail --language en \
  --num-tasks 1 --num-trials 1 --save-to pine-smoke
```

A custom Agent and call are created for each episode. The participant publishes
mono PCM16 microphone audio and subscribes to the remote agent's audio. The
shared streaming converter handles Tau's 8 kHz mu-law interface. The adapter sets
`realtime_pacing=True`; real-time caller generation is required so model/TTS
latency does not freeze live room playback. Tool execution uses ordinary Tau
`ToolCall` / `ToolMessage` exchange, not direct calls to Python tool functions.

After the episode the adapter disconnects the room, hangs up a still-active call,
and deletes the temporary Agent. Call records are retained for debugging. Setup
failures also attempt cleanup. Cleanup failures are logged with the SDK error;
check the Portal project for resources if a network outage prevented cleanup.

## Recovery and evidence

- The SDK reconnects SSE with its last event ID. The adapter deduplicates event
  IDs and tool call IDs within the episode so replay does not re-execute tools.
  This is not recovery of a killed benchmark process's in-flight call.
- Transport failures fail the episode; they are not treated as silent audio or
  successful completion. Queue overflow also fails rather than dropping media.
- Finalized transcript events are authoritative. Interim text is not treated as
  spoken text; interrupted utterances use the finalized text supplied by Portal.
- Transcript offsets are preserved on Portal's `answered_at` clock. They can be
  derived from engine speech events / transcript emission and are **not** claimed
  to be sample-accurate acoustic boundaries or client playback acknowledgements.
  Room connection and first Tau tick have different origins; no guessed offset
  is applied. Audio speech intervals are computed separately on actual delivered
  tick audio, using the same activity threshold as the OpenAI Live adapter.
- `call.ended` stops the agent through explicit provider termination, not a fake
  spoken stop token. Final transcript events must precede that terminal event.
  Termination also waits for remote audio track removal and 200 ms without new
  audio; queued tick audio is drained. Failure to reach this barrier within five
  seconds fails the episode rather than silently clipping trailing media.
- Usage costs are not fabricated when Portal provides no per-model usage ledger.
  Review transcript/audio alignment and judge coverage before a public submission.

## Full submission requirements

A smoke run is not a leaderboard result. At the repository revision on which this
provider is based, the standard 2.0 voice matrix is **900 calls**: six languages
(`en`, `es`, `pt`, `hi`, `zh`, `ko`), three domains, 50 frozen tasks each, one trial.
Use one configuration throughout, regular speech complexity, and account-owned
caller voices. Banking and Entity are separate optional packages (582 and 1200
calls respectively); they do not enter the voice-v2 overall.

```bash
uv run --no-sync tau2 run --audio-native --benchmark voice \
  --audio-native-provider pine --pine-config "$(cat pine-config.json)" \
  --realtime-generation --speech-complexity regular --num-trials 1 \
  --save-to pine-voice-v2
uv run --no-sync tau2 judges suite data/simulations/pine-voice-v2
uv run --no-sync tau2 submit prepare data/simulations/pine-voice-v2 \
  --output ./submissions --metadata-file ./submitter.json
uv run --no-sync tau2 submit validate ./submissions/REPLACE_PACKAGE \
  --results data/simulations/pine-voice-v2
```

Do not use `--num-tasks` / `--task-ids` for a submission. Follow the canonical
[submission guide](leaderboard-submission.md) for completeness, required judge
coverage, methodology disclosure, voice overrides and uploads. The commands above
are a checklist, not a claim that a full matrix or judge suite has been run.
