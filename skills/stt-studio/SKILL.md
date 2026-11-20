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
# Who spoke when (needs HF_TOKEN and the diarizer; see "Speaker labels")
python3 scripts/stt_cli.py transcribe call.mp4 --diarize --format srt -o call.srt --json

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

## Speaker labels (`--diarize`)

Use `--diarize` (CLI) or `diarize=true` (API) only when the user wants to know who spoke when.

- **Check first:** `capabilities.diarization.available`. It is false when the diarizer environment
  is missing (run `scripts/setup-diarizer.sh`) or `HF_TOKEN` is not set in the project's `.env`.
  The account behind the token must have accepted the terms of `pyannote/speaker-diarization-3.1`
  and `pyannote/segmentation-3.0` on Hugging Face. Never ask the user to paste the token into chat.
- **Runs locally.** pyannote runs on this machine, so the audio does not leave it. The model
  (MIT licence on its card) is cached in `models/`.
- **One job per file.** A file is never split when diarizing, because each chunk would number its
  speakers separately. A file longer than the single-job limit is refused with exit 1.
- **Labels are anonymous**: `Speaker 1` is whoever talks first, then `Speaker 2`, and so on. They
  are per job. Output: TXT and SRT prefix each line `Speaker 1: `; VTT uses `<v Speaker 1>`; JSON
  and batch JSON add a `speaker` key. A segment takes the speaker who talks longest inside it.
- **Read the outcome.** Diarization runs after transcription. A failure keeps the transcript, drops
  the labels and records `info.diarization.status = "failed"` with a `reason`. The CLI still exits
  0 and prints a warning, so check `diarization.status` in `--json` output before relying on labels.
- **Quality, measured on this Mac:**
  - Synthetic two-voice speech (macOS voices, 10 turns): all 10 segments got the right speaker.
  - An 81-minute recording of a two-person call: two main voices held 95% of talk time, and about
    7% of segments went to three small labels. Those small labels may be fragments of the main
    voices, or real people. This was not verified.
  - A 19-minute Vietnamese meeting with an unknown number of speakers: four speakers at the default.
  - Overlapping speech is not separated. Expect errors when people talk over each other, and in
    very large groups (a nine-person hearing needed a lower threshold in another project).
  - Tell the user these figures and the uncertainty. Do not describe the labels as accurate.
- **Speed:** about 3 to 4 minutes of diarization for an 81-minute recording (on the GPU via `mps`),
  on top of transcription time.
- **Tuning:** `STT_STUDIO_DIARIZER_THRESHOLD` (default `0.7046`) sets the clustering threshold.
  `0.5` splits more and suits large groups; on a two-person call it created about twelve small
  spurious speakers. Forcing a speaker count was tested and made the two-person split worse, so
  it is not offered.

## Errors

| Code | Meaning | What to do |
| --- | --- | --- |
| 403 | Non-loopback client, `Host` or `Origin` | Call from `127.0.0.1`/`localhost`; never expose the port |
| 404 | Unknown job or batch id | Ids are per server run; re-submit after a restart |
| 409 | Export before the job finished, export of a failed job, unload while running, cancel of a finished job | Wait for a terminal status, or cancel first |
| 413 | Upload over `max_upload_bytes` (2 GiB) or batch over 8 GiB | Split the file; the CLI's chunking avoids this for long audio |
| 422 | Bad model, runtime, compute type, `beam_size`, format, empty upload, undecodable media, or `diarize` when speaker labels are unavailable | Read `detail`; check `capabilities` and `models` (`capabilities.diarization.reason` says what is missing) |
| 429 | Another job or batch is active (`max_active_jobs: 1`) | Wait, cancel, or retry; the CLI retries every 3 s |
| 503 | Model could not be downloaded, loaded or warmed | Check network, disk space and the model id; retry |

The CLI prints `Cannot reach STT Studio at …` when the backend is down, and `error: …` with
exit code 1 for every API failure. It never prints a traceback for expected failures.

## Limits to state honestly

- One worker: one job or one batch at a time. A second submission gets `429`.
- Jobs, batches and transcripts live in memory and are lost when the server restarts.
- Cancellation is cooperative. faster-whisper stops between segments; the MLX runtime cannot
  be interrupted mid-`generate`, so cancellation takes effect only at the next step.
- Speaker labels are **anonymous** ("Speaker 1", "Speaker 2"). They never give real names; say so
  if the user asks who a speaker is. Labels come from a local pyannote model and are only as good
  as the recording (see "Speaker labels" above).
- Translation is not offered; transcripts keep the spoken language.
- No watch folders, no automatic file-system monitoring.
- The browser UI does not yet use every endpoint; the API and CLI are the complete surface.

## Output formats

`txt` (one line per segment), `srt` (`HH:MM:SS,mmm`), `vtt` (`HH:MM:SS.mmm`), and `json`
(an array of `{"start", "end", "text"}`). Batch export supports `txt` (headed sections) and
`json` (`format: "stt-studio-batch-results"`, `version: 1`). Timestamps are rounded the same way
as the browser export.

Full request/response detail: [API_REFERENCE.md](API_REFERENCE.md).
