---
name: stt-studio
description: Transcribe audio or video locally with STT Studio, producing TXT, SRT, VTT or JSON transcripts. Use when the user asks to transcribe a recording, caption or subtitle a video, turn speech into text, batch a folder of media, or drive the STT Studio app from an agent or script. Runs on the user's own machine over loopback only.
---

# STT Studio

STT Studio is a local speech-to-text app for macOS. A FastAPI backend runs `faster-whisper`
(or `mlx-audio` on Apple Silicon) and exposes an HTTP API on `http://127.0.0.1:8765`.
Audio never leaves the machine.

## Read this first: where the agent runs

The API listens on loopback only. The backend rejects non-loopback clients, non-loopback
`Host` headers, and mutations from a non-local `Origin` with `403`.

- **Agent with shell access** (Claude Code, Cursor, Codex CLI, a terminal, a local script):
  use `scripts/stt_cli.py`. It is the supported path and handles video, chunking and retries.
- **Hosted chat agent** (ChatGPT, Claude in a browser, any cloud agent): it **cannot reach
  `http://127.0.0.1`** on the user's computer. Do not pretend otherwise and do not ask the
  user to expose the port. Ask them to run the CLI or the app locally and paste the transcript,
  or to give the task to a locally installed agent.
- **Other languages/tools on the same machine**: call the HTTP API directly; see
  [API_REFERENCE.md](API_REFERENCE.md).

Never bind the server to `0.0.0.0`, add a tunnel, or publish the port. The service has no
authentication and can download and run model files.

## Is the backend running?

```sh
python3 scripts/stt_cli.py status        # {"status": "ok", "active_jobs": 0, ...}
```

If that fails, start it from the repository root:

```sh
./scripts/dev.sh                                              # backend + web UI
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8765   # backend only
```

`ffmpeg` and `ffprobe` must be on `PATH` (the CLI also needs them to plan chunking).
Set `STT_STUDIO_URL` if the backend uses a different loopback port.

## Workflow

1. **`status`** — confirm the service is up and no job is running.
2. **`capabilities`** — limits, accepted formats, runtimes and the endpoint list. Read this
   instead of assuming a limit; it reflects the server's environment variables.
3. **`probe <file>`** — check `has_audio`, `duration_seconds`, `transcribable` and `reason`
   before spending time on a model. Video with no audio track is refused here.
4. **`transcribe`** — one file, several files, or `batch submit` for fire-and-forget.
5. **Verify** — check the exit code, read the output, and confirm the text looks like speech
   rather than a repetition loop or silence.

```sh
# One video to SRT, chunked every 20 minutes (the default)
python3 scripts/stt_cli.py transcribe meeting.mp4 --format srt -o meeting.srt

# Timestamped JSON on stdout, for an agent to read
python3 scripts/stt_cli.py transcribe interview.m4a --json

# Several files, one output per input
python3 scripts/stt_cli.py transcribe *.wav --output-dir transcripts --format txt

# Force a language and give vocabulary hints
python3 scripts/stt_cli.py transcribe panel.mp4 --language en --prompt "ANZ, Kubernetes, Leichhardt"

# Queue now, collect later
python3 scripts/stt_cli.py batch submit a.wav b.mp4
python3 scripts/stt_cli.py batch status <batch-id>
python3 scripts/stt_cli.py batch export <batch-id> --format json -o batch.json

# One job from the HTTP API
python3 scripts/stt_cli.py job status <job-id> --wait 60
python3 scripts/stt_cli.py job export <job-id> --format vtt -o captions.vtt
```

Exit codes: `0` success, `1` a request or transcription failed (message on stderr), `2` invalid
usage. With `--json`, every record has a stable shape: `source`, `status`
(`completed`/`incomplete`/`failed`), `error` (null on success), `duration_seconds`, `language`,
`segments`, `output`, and `text`.

## Choosing a model

- Default: **`small`**, `int8`, CPU. Good accuracy/speed balance and a modest download.
- `tiny` and `base` are faster but mishear names and numbers. Use them for drafts, not for
  anything the user will publish. `medium`, `large-v3` and `large-v3-turbo` are more accurate
  and much slower on CPU; check what is already cached with `models`.
- **`Qwen3-ASR (MLX conversion)`** — Apple Silicon only, needs the optional `mlx-audio`
  runtime. It is fast and accurate but returns **one untimed segment**, so SRT/VTT contain a
  single caption spanning the whole file. Say this to the user before they choose it.
- **Parakeet MLX is not available.** The installed `mlx-audio` cannot load its config. Do not
  offer it or claim it works.
- A custom Hugging Face `owner/model` is only usable if it contains a CTranslate2
  faster-whisper model. Transformers checkpoints are not interchangeable.

Do not claim a model works unless `/api/models` lists it with `supported: true` for an
installed runtime. Check `runtime_capabilities` before sending `beam_size`, `vad_filter` or
`initial_prompt`: the MLX runtime ignores them.

## Video files

Video is accepted directly; the backend runs ffmpeg and transcribes the audio track. You do not
need to extract audio yourself. Always `probe` first — a screen recording without a microphone
track has `has_audio: false` and `transcribable: false`, and the API will not invent speech.
The CLI refuses it with exit code 1 and a clear message.

## Long audio

One job is capped at `WHISPER_MAX_AUDIO_SECONDS` (default 7,200 s = 2 hours). The CLI splits
anything longer than `--chunk-seconds` (default 1,200 s) into 16 kHz mono WAV chunks:

- cut points are moved to the middle of a detected pause (`silencedetect`, −35 dB, 0.4 s) within
  a window near each boundary;
- a final sliver under `min(5 s, 25% of the chunk)` is folded into the previous chunk;
- chunks are uploaded in batches of 10, transcribed in order, and each segment's times are
  offset by the chunk start; ends are clamped to the chunk so captions never overlap;
- if the whole-command `--timeout` (default 3,600 s) expires, the batch is cancelled, finished
  chunks are kept, and the exit code is 1.

**Boundaries can still split a word** when speech runs across the cut with no pause. Expect an
occasional clipped or duplicated word at a join; a larger `--chunk-seconds` means fewer joins.
Do not promise seamless stitching.

## Errors

| Code | Meaning | What to do |
| --- | --- | --- |
| 403 | Non-loopback client, `Host` or `Origin` | Call from `127.0.0.1`/`localhost`; never expose the port |
| 404 | Unknown job or batch id | Ids are per server run; re-submit after a restart |
| 409 | Export before the job finished, export of a failed job, unload while running, cancel of a finished job | Wait for a terminal status, or cancel first |
| 413 | Upload over `max_upload_bytes` (2 GiB) or batch over 8 GiB | Split the file; the CLI's chunking avoids this for long audio |
| 422 | Bad model, runtime, compute type, `beam_size`, format, empty upload, undecodable media | Read `detail`; check `capabilities` and `models` |
| 429 | Another job or batch is active (`max_active_jobs: 1`) | Wait, cancel, or retry; the CLI retries every 3 s |
| 503 | Model could not be downloaded, loaded or warmed | Check network, disk space and the model id; retry |

The CLI prints `Cannot reach STT Studio at …` when the backend is down, and `error: …` with
exit code 1 for every API failure. It never prints a traceback for expected failures.

## Limits to state honestly

- One worker: one job or one batch at a time. A second submission gets `429`.
- Jobs, batches and transcripts live in memory and are lost when the server restarts.
- Cancellation is cooperative. faster-whisper stops between segments; the MLX runtime cannot
  be interrupted mid-`generate`, so cancellation takes effect only at the next step.
- Diarisation (who spoke when) is **not implemented**. Do not offer speaker labels.
- Translation is not offered; transcripts keep the spoken language.
- No watch folders, no automatic file-system monitoring.
- The browser UI does not yet use every endpoint; the API and CLI are the complete surface.

## Output formats

`txt` (one line per segment), `srt` (`HH:MM:SS,mmm`), `vtt` (`HH:MM:SS.mmm`), and `json`
(an array of `{"start", "end", "text"}`). Batch export supports `txt` (headed sections) and
`json` (`format: "stt-studio-batch-results"`, `version: 1`). Timestamps are rounded the same way
as the browser export.

Full request/response detail: [API_REFERENCE.md](API_REFERENCE.md).
