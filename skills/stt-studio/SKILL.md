---
name: stt-studio
description: Transcribe audio or video locally with STT Studio, producing TXT, SRT, VTT or JSON transcripts. Use when the user asks to transcribe a recording, caption or subtitle a video, turn speech into text, batch a folder of media, or drive the STT Studio app from an agent or script. Runs on the user's own machine over loopback only.
---

# STT Studio

STT Studio is a local speech-to-text app for macOS. A FastAPI backend runs `faster-whisper`
(or `mlx-audio` on Apple Silicon) and exposes an HTTP API on `http://127.0.0.1:8765`.
Audio never leaves the machine. Transcript text leaves it only when the user asks for an AI summary or
action items; see [AI summaries and action items](#ai-summaries-and-action-items-optional).

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
   before spending time on a model. `probe` exits 0 even when `transcribable` is false, so read
   the field rather than the exit code.
4. **`transcribe`** — one file, several files, or `batch submit` for fire-and-forget.
5. **Verify** — check the exit code, read the output, and confirm the text looks like speech
   rather than a repetition loop or silence.

```sh
# One video to SRT (long files are split at pauses; see "Long audio")
python3 scripts/stt_cli.py transcribe meeting.mp4 --format srt -o meeting.srt

# Timestamped JSON on stdout, for an agent to read (read files[0], see "Exit codes and JSON")
python3 scripts/stt_cli.py transcribe interview.m4a --json

# Several files, one output per input
python3 scripts/stt_cli.py transcribe *.wav --output-dir transcripts --format txt

# Choose a model (list the runnable ones with `models` first; `medium` is an example)
python3 scripts/stt_cli.py transcribe clip.wav --model medium --format txt -o clip.txt

# Force a language and give vocabulary hints
python3 scripts/stt_cli.py transcribe panel.mp4 --language en --prompt "ANZ, Kubernetes, Leichhardt"

# Who spoke when (needs HF_TOKEN and the diarizer; see "Speaker labels")
python3 scripts/stt_cli.py transcribe call.mp4 --diarize --format srt -o call.srt

# Queue now, collect later. A batch has no wait command: poll `batch status` until "done" is true.
python3 scripts/stt_cli.py batch submit a.wav b.mp4
python3 scripts/stt_cli.py batch status <batch-id>
python3 scripts/stt_cli.py batch export <batch-id> --format json -o batch.json

# One job from the HTTP API
python3 scripts/stt_cli.py job status <job-id> --wait 60
python3 scripts/stt_cli.py job export <job-id> --format vtt -o captions.vtt
```

## CLI commands

Every command is `python3 scripts/stt_cli.py COMMAND`. `python3 scripts/stt_cli.py COMMAND --help` lists all flags.
The complete table is in [API_REFERENCE.md](API_REFERENCE.md#command-line-client).

| Command | Purpose and key flags |
| --- | --- |
| `status`, `capabilities`, `models`, `unload` | Health, limits, model list and cache state, free resident models |
| `load [MODEL] [--compute-type T] [--runtime R]` | Download if needed, load and warm a model |
| `probe FILE...` | Duration and streams on the server, without transcribing |
| `transcribe FILE... [--format txt\|srt\|vtt\|json] [-o F \| --output-dir D] [--json] [--chunk-seconds N] [--timeout S] [--diarize]` | Transcribe and wait; long files are split |
| `batch submit FILE... [--diarize]` | Queue files and return the batch id at once |
| `batch status ID`, `batch cancel ID`, `batch export ID [--format txt\|json] [-o F]` | Poll, cancel or export a batch |
| `job status ID [--wait S]`, `job cancel ID`, `job export ID [--format F] [-o F]` | Inspect, cancel or export one job |
| `jobs list [--limit 1-200] [--offset N] [--json]` | Stored jobs, newest first, one page at a time |
| `jobs delete ID [--json]` | Delete a finished job and its transcript |
| `batches list [--limit 1-200] [--offset N] [--json]` | Stored batches, newest first, one page at a time |
| `batches delete ID [--json]` | Delete a finished batch and all of its jobs |
| `ai summary ID`, `ai action-items ID` `[--json]` | Summary or action items for a completed job. Sends transcript text to NVIDIA NIM |
| `storage [--json]` | Job and batch counts and disk use |

Model options for `transcribe` and `batch submit`: `--model` (default `small`), `--runtime`, `--compute-type`
(default `int8`), `--language` (default `auto`), `--beam-size` (default 5), `--no-vad`, `--prompt`.

Upload rule: videos, and non-WAV audio over 1 GiB, are sent as extracted 16 kHz mono WAV by both `transcribe`
and `batch submit`. There is no CLI command to delete a cached model; use `DELETE /api/models/cache/{id}`.

## Exit codes and JSON

Exit codes: `0` success, `1` a request or transcription failed (message on stderr), `2` invalid
usage, `130` interrupted by Ctrl-C (the CLI cancels the batches it started first). `job status`
exits `1` when the job has `failed`; the failed record is still printed, so read its `error` field.

Under `--json`, a run that succeeds prints the `files` wrapper described below. A run that fails
before any file is processed (for example a missing file) prints `{"error": "...", "status": ...}`
with no `files` key, and exits 1 or 2. Check the exit code first, then read whichever shape came back.
A timeout under `--json` prints the wrapper, with the unfinished files marked `incomplete` or
`cancelled`; the batch id is on stderr.

A single-file `transcribe` runs as a batch of one, so it uses the same single worker. If another
job or batch is running, the CLI waits and retries every 3 s until `--timeout` runs out. If the run is interrupted (Ctrl-C), the CLI cancels the batches it started, so the server is not left busy. A process started with `&` in a non-interactive shell may ignore SIGINT; if a batch is still running, cancel it with `batch cancel <batch-id>`.

With `--json`, stdout is `{"format": "...", "files": [record, ...]}`. Read `files[0]` for one file.
Each record has `source`, `status` (`completed`/`incomplete`/`failed`), `error` (null on success),
`duration_seconds`, `language`, `segments` and `text`. It has `output` only when `-o` or
`--output-dir` is given, and `diarization` when `--diarize` was used. The CLI prints
`[stt] batch <id> started` to stderr in every mode, so a run that times out can still be recovered
with `batch export`.

## Choosing a model

- Default: **`small`**, `int8`, CPU. Good accuracy/speed balance and a modest download.
- **Language:** automatic detection can be wrong. On a 60 s Vietnamese meeting clip, `small` with
  auto-detection reported `en` and produced English-looking text; the same audio with `--language vi`
  gave Vietnamese. When the user knows the language, pass `--language <code>` and say so in the
  reply. When it is unknown, check the detected `language` in the output before trusting the text.
- `tiny` and `base` are faster but mishear names and numbers. Use them for drafts, not for
  anything the user will publish. `medium`, `large-v3` and `large-v3-turbo` are more accurate
  and much slower on CPU; check what is already cached with `models`.
- **`Qwen3-ASR (MLX conversion)`** — Apple Silicon only, needs the optional `mlx-audio`
  runtime. It is fast and accurate but returns **one untimed segment**, so SRT/VTT contain a
  single caption spanning the whole file. Say this to the user before they choose it. For timed captions, use faster-whisper (its `small` and larger models give timed segments and support speaker labels); the MLX route stays untimed by decision.
- **Parakeet MLX is not available.** The installed `mlx-audio` cannot load its config. Do not
  offer it or claim it works.
- A custom Hugging Face `owner/model` is only usable if it contains a CTranslate2
  faster-whisper model. Transformers checkpoints are not interchangeable.

Do not claim a model works unless `/api/models` lists it with `supported: true` for an
installed runtime. Each model's `controls` say which options its runtime applies. The MLX runtime accepts
`beam_size`, `vad_filter` and `initial_prompt` but ignores them; the job's `ignored_controls` names any it dropped.

## Video files

Video is accepted directly; the backend runs ffmpeg and transcribes the audio track. You do not
need to extract audio yourself. Always `probe` first. A screen recording without a microphone
track has `has_audio: false` and `transcribable: false`, and the API will not invent speech.
The CLI refuses such a file with exit code 1. The HTTP API also refuses it at submission with
`422` and the message `The file has no audio track to transcribe.`, so no job is queued.

## Long audio

One job is capped at `WHISPER_MAX_AUDIO_SECONDS` (default 7,200 s = 2 hours). The CLI splits
anything longer than `--chunk-seconds` (default 1,200 s) into 16 kHz mono WAV chunks:

- cut targets fall every `--chunk-seconds`, and each cut moves to the middle of a detected pause
  (`silencedetect`, −35 dB, 0.4 s) within `min(120 s, 25% of the target)` of its target. With no
  pause in that window, the cut stays on the target;
- a final sliver under `min(5 s, 25% of the target)` is folded into the previous chunk, so the last
  chunk can run up to about 25% past the target. A 46 s file with `--chunk-seconds 20` gave cuts at
  20.0 s and a final chunk of about 26 s;
- chunks are uploaded in groups of 10, transcribed in order, and each segment's times are offset by
  the chunk start; ends are clamped to the chunk so captions never overlap;
- if the whole-command `--timeout` (default 3,600 s) expires, the CLI cancels the remaining chunks,
  writes no `-o` file, and exits 1. Chunks that already finished stay on the server; export them with
  `batch export <batch-id> --format json`, using the id printed at the start of the run. If nothing
  finished, the export exits 0 with every file `cancelled` and no segments, so check the statuses
  rather than the exit code.

**Boundaries can still split a word** when speech runs across the cut with no pause. In a test
run, "chunk limit" came out as "chunk limiting." at a 20 s join. Expect an occasional clipped or
duplicated word at a join; a larger `--chunk-seconds` means fewer joins. Do not promise seamless
stitching.

## Watch a folder

`scripts/watch_folder.py` transcribes recordings as they land in a folder, for a user who does not
want to type commands. Drop files into the watch folder; transcripts appear in the output folder.

```sh
python3 scripts/watch_folder.py ~/Recordings --output-dir ~/Transcripts --format srt   # runs until Ctrl-C
python3 scripts/watch_folder.py ~/Recordings --output-dir ~/Transcripts --once         # one pass, then exits
python3 scripts/watch_folder.py ~/Recordings --output-dir ~/Transcripts --once --retry-failed
```

Options: `--output-dir` (required), `--format txt|srt|vtt|json` (default `txt`), `--model`, `--language`,
`--diarize`, `--interval` (seconds between scans, default 5), `--once`, `--file-timeout` (seconds allowed per
file, default 3600; a file that runs longer is cancelled and counts as a failed attempt), `--retry-failed`.

What it does:

- Scans the watch folder, not recursively, every `--interval` seconds. It takes the extensions the
  server accepts from `/api/capabilities`. It ignores hidden files, names starting with `~`, and
  downloads still in progress (`.part`, `.crdownload`, `.download`, `.tmp`).
- Transcribes a file only when its size and modification time are the same in two consecutive scans,
  so a file still being copied waits. `--once` scans twice, `--interval` apart, then exits.
- Runs the same work as `transcribe`: a file with no audio track fails without being sent, long files
  are split at pauses, and `--diarize` follows the same rules. Output is `<stem>.<format>` in the output
  folder. If that name is taken, it writes `<stem>-2.<format>`, `-3`, and so on. It never overwrites.
- Records every source in `<output folder>/.watch-state.json`, keyed by name, size and modification
  time, with its status (`completed`, `retrying` or `failed`), output path or error, attempt count and
  time. A completed source is never transcribed again.
- Retries a file that fails (transcription error, timeout or a server `5xx`) up to 3 attempts in total.
  The file is then marked `failed` and skipped until `--retry-failed`. A `429` (server busy) or an
  unreachable server pauses the watcher without charging the file an attempt.
- Holds `<output folder>/.watch.lock` for the whole run. A second watcher, or a manual `--once` while the
  background agent runs, exits with an error instead of writing duplicate transcripts.
  Do not delete the state
  file casually: without it, every file in the watch folder is transcribed again.
- Logs one timestamped line per event to stderr: `started`, `skipped`, `retrying`, `completed`, `failed`, and
  `waiting` when the server is unreachable or busy.

Exit codes: under `--once`, `0` when no attempt failed in this run; `1` when any attempt failed in this run
(including one that will be retried), or the server was unavailable. Without `--once` the watcher keeps running
through server outages and exits only on Ctrl-C. Ctrl-C exits `130` after cancelling the running batch,
and nothing is written for the file being transcribed. Startup errors (no `ffmpeg`, an unreadable state
file, a second watcher on the same output folder, a non-loopback URL) exit `1` or `2` before anything is processed.

What it does not do:

- It polls; it does not use file-system events. It does not look inside subfolders and does not follow
  symbolic links (it logs a link once and skips it).
- It is not a service by default. It does not start the backend, and it does not start at login. On macOS
  you can install a launch agent for it (see "Run in the background"); nothing is loaded until you run the
  `launchctl` command that the install script prints.
- It does not move, rename or delete recordings, and it does not edit a transcript after writing it.
- It transcribes one file at a time, because the server has one worker.
- Run one watcher per output folder. The output folder lock refuses a second watcher, so the same file is
  not transcribed twice.

Privacy: the only upload is the media file, to the backend at `STT_STUDIO_URL` (default
`http://127.0.0.1:8765`). The watcher refuses any non-loopback host, so audio stays on this machine.
Apart from that upload it sends only the status and cancel calls `transcribe` makes. Logs contain file
names, counts and error messages, never transcript text. Symbolic links are never followed, so nothing
outside the watch folder is read.

### Run in the background (macOS)

`scripts/watch_service/install.sh WATCH_DIR OUTPUT_DIR` writes a launch agent,
`~/Library/LaunchAgents/com.sttstudio.watch.plist`, that runs the watcher in its loop. It takes the same
options as the watcher (`--format`, `--model`, `--language`, `--diarize`, `--interval`, default 10 s) and
`--url` for a loopback backend. Both folders must be absolute, and the output folder must not be inside the
watch folder. Install does not load the agent. It prints the `launchctl bootstrap` command to run; after
that the agent starts at each login. Logs go to `~/Library/Logs/stt-studio-watch.log`.

- `scripts/watch_service/status.sh` shows the settings and the last 20 log lines. It never calls `launchctl`.
- `scripts/watch_service/uninstall.sh` removes the plist only if this tool wrote it, and prints the
  `launchctl bootout` command that stops the agent.
- Stop it with `bootout` rather than `kill`: `KeepAlive` restarts the watcher after a non-zero exit.
- A file being transcribed when the agent stops is transcribed again after the next start, and the server
  finishes the batch it already started. A watcher that exits with an error (for example, `ffmpeg` missing)
  is restarted every 30 s; read the log with `status.sh`.

## Speaker labels (`--diarize`)

Use `--diarize` (CLI) or `diarize=true` (API) only when the user wants to know who spoke when.

- **Check first:** `capabilities.diarization.available`. It is false when the diarizer environment
  is missing (run `scripts/setup-diarizer.sh`) or `HF_TOKEN` is not set in the project's `.env`.
  The account behind the token must have accepted the terms of `pyannote/speaker-diarization-3.1`
  and `pyannote/segmentation-3.0` on Hugging Face. If those gated repositories refuse the token, the
  failure `reason` names both of them. Never ask the user to paste the token into chat.
- **Runs locally.** pyannote runs on this machine, so the audio does not leave it. The model
  (the pipeline's card says MIT). It also loads `pyannote/wespeaker-voxceleb-resnet34-LM`, which is **CC-BY-4.0** and needs attribution (checked on Hugging Face). Weights are cached in `models/`.
- **One job per file.** A file is never split when diarizing, because each chunk would number its
  speakers separately. A file longer than the single-job limit is refused with exit 1. To label a
  file over 2 hours, the server must run with a higher `WHISPER_MAX_AUDIO_SECONDS`; the CLI will not
  split it.
- **Labels are anonymous**: `Speaker 1` is whoever talks first, then `Speaker 2`, and so on. They
  are per job. Output: TXT and SRT prefix each line `Speaker 1: `; VTT uses `<v Speaker 1>`; JSON
  and batch JSON add a `speaker` key. A segment takes the speaker who talks longest inside it.
- **Read the outcome.** Diarization runs after transcription. Job and batch exports carry
  `info.diarization`; CLI `--json` output carries `diarization` on the record. Both have `status`
  (`completed` or `failed`). A failure keeps the transcript and drops the labels. A `failed` status
  also has a `reason`; that failure path is covered by tests but I have not seen it on the real
  model. The CLI still exits 0 and prints a warning, so check `status` before relying on labels.
- **Quality, measured on this Mac:**
  - Synthetic two-voice speech (a 10-turn clip made with macOS voices; the clip is not kept in the repo, and the opt-in test uses 6 turns): all 10 segments got the right speaker.
  - An 81-minute recording of a two-person call: two main voices held 95% of talk time, and about
    7% of segments went to three small labels. Those small labels may be fragments of the main
    voices, or real people. This was not verified.
  - A 19-minute Vietnamese meeting with an unknown number of speakers: four speakers at the default.
  - Overlapping speech is not separated. Expect errors when people talk over each other, and in
    very large groups (a nine-person hearing needed a lower threshold in another project).
- **How to say it:** if the user asks how accurate the labels are, say: "Speaker labels are
  anonymous and come from a local model. On a two-voice test they matched the turns. On a long
  two-person call a few short extra labels appeared, and overlapping speech is not separated. I have
  not checked them against a ground-truth transcript." Do not call them accurate, and do not give
  anyone's real name from the labels.
- **Speed:** about 3 to 4 minutes of diarization for an 81-minute recording (on the GPU via `mps`),
  on top of transcription time.
- **Tuning:** `STT_STUDIO_DIARIZER_THRESHOLD` (default `0.7046`) sets the clustering threshold.
  `0.5` splits more and suits large groups; on a two-person call it created about twelve small
  spurious speakers. Forcing a speaker count was tested and made the two-person split worse, so
  it is not offered.

## AI summaries and action items (optional)

`POST /api/jobs/{id}/ai` with `{"mode": "summary"}` or `{"mode": "action_items"}` asks NVIDIA NIM
(`nvidia/nemotron-3-ultra-550b-a55b`) to summarise a finished transcript or list its action items.
The result is stored on the job as `ai_results`. It is the only call in this skill that sends
transcript text to a hosted service.

- **Only when the user asks.** Call it only when the user asks for a summary or action items of a
  transcript. Never call it as part of transcribing, batching, watching a folder, or to "be helpful".
- **Say it first.** Before the first call, tell the user that the transcript text (not the audio) will
  leave this Mac and go to NVIDIA NIM, and get their agreement for that transcript. Say the same when
  a later request sends a different transcript.
- **Check first:** `GET /api/capabilities` → `ai.available`. When it is false, read `ai.reason` to the
  user (usually: no `NVIDIA_NIM_API_KEY_1` to `_9` in `.env` or the environment). Do not work around it,
  and never ask the user to paste a key into the chat.
- **Keys stay out of output.** Never print, echo, store or ask for a key. The API never returns one.
- **Completed transcripts only.** A job that is queued, running or failed gives `409`. Empty
  transcripts and transcripts over 150,000 characters give `422`, and the detail says why.
- **Check the output.** Results are generated text. Summaries can miss or invent detail, and the
  action-item owner and due date are only as good as the transcript. Say so when you show one.
- **Failures are expected.** `502` means NVIDIA NIM refused every key or could not be reached. Report
  it and do not retry in a loop.

```sh
curl -s -X POST http://127.0.0.1:8765/api/jobs/<job-id>/ai -H 'Content-Type: application/json' \
  -d '{"mode": "action_items"}'
```

## Errors

| Code | Meaning | What to do |
| --- | --- | --- |
| 403 | Non-loopback client, `Host` or `Origin` | Call from `127.0.0.1`/`localhost`; never expose the port |
| 404 | Unknown job or batch id, or one that was deleted | Ids stay valid across restarts; if the id is gone it was deleted or never existed. Re-submit the file |
| 409 | Export before the job or batch has finished | Wait for a terminal status (`job status --wait`, or poll `batch status` until `done`) |
| 409 | Export of a failed job; cancel of a finished job; unload while a job runs; delete while queued or running | A finished job cannot be cancelled. Read `error`, or re-submit the file. For unload, wait for the running job or cancel it first. Delete only finished jobs and batches |
| 413 | Upload over `max_upload_bytes` (2 GiB) or batch over 8 GiB | Split the file; the CLI's chunking avoids this for long audio |
| 422 | Bad model name or format (a well-formed `owner/model` that does not exist is accepted with `202` and then fails as a job; check the job's `error`), runtime, compute type, `beam_size`, format, empty upload, undecodable media, a file with no audio track, or `diarize` when speaker labels are unavailable | Read `detail`; check `capabilities` and `models` (`capabilities.diarization.reason` says what is missing) |
| 429 | Another job or batch is active (`max_active_jobs: 1`) | The CLI retries every 3 s silently, so a waiting `transcribe` can look hung; wait, or cancel the other work. Raw API callers get the 429 and must retry themselves |
| 503 | Model could not be downloaded, loaded or warmed | Check network, disk space and the model id; retry |

The CLI prints `Cannot reach STT Studio at …` when the backend is down, and `error: …` with
exit code 1 for every API failure. It never prints a traceback for expected failures.

## Limits to state honestly

- One worker: one job or one batch at a time. A second submission gets `429`.
- Jobs, batches and transcripts persist in the server's local database until they are deleted
  (`jobs delete` / `batches delete`, or `DELETE /api/jobs/{id}` / `DELETE /api/batches/{id}`, once finished). Audio is never stored. Ids stay
  valid across restarts. A job that was queued or running when the server stopped is marked `failed` with
  `Interrupted by a server restart`; re-submit that file. `GET /api/storage` shows where results are kept
  (a folder name, never an absolute path).
- Cancellation is cooperative. faster-whisper stops between segments; the MLX runtime cannot
  be interrupted mid-`generate`, so cancellation takes effect only at the next step.
- Speaker labels are **anonymous** ("Speaker 1", "Speaker 2"). They never give real names; say so
  if the user asks who a speaker is. Labels come from a local pyannote model and are only as good
  as the recording (see "Speaker labels" above).
- Translation is not offered; transcripts keep the spoken language.
- The watch folder (see "Watch a folder") polls rather than watching file-system events. It runs in the
  background only after you install the launch agent yourself (see "Run in the background").
- The browser UI covers the main workflows, including batch history and model cache deletion. The API is the
  complete surface; the CLI has no command for cache deletion.
- Word error rate is offline: `python -m evaluation.wer reference.txt hypothesis.txt` compares two local
  transcripts and does not call the backend. See `evaluation/README.md`.

## Output formats

`txt` (one line per segment), `srt` (`HH:MM:SS,mmm`), `vtt` (`HH:MM:SS.mmm`), and `json`
(an array of `{"start", "end", "text"}`). Batch export supports `txt` (headed sections) and
`json` (`format: "stt-studio-batch-results"`, `version: 1`). Timestamps are rounded to the millisecond.

Full request/response detail: [API_REFERENCE.md](API_REFERENCE.md).
