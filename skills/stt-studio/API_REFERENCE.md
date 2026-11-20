# STT Studio API reference

Local HTTP API for STT Studio. Base URL `http://127.0.0.1:8765`. JSON in, JSON out, except
media uploads (multipart) and exports (file download).

The server accepts loopback clients only. Non-loopback peers, non-loopback `Host` headers, and
mutating requests with a non-local `Origin` are rejected with `403`. There is no authentication;
do not expose the port.

## Contents

- [Start the server](#start-the-server)
- [GET /api/health](#get-apihealth)
- [GET /api/capabilities](#get-apicapabilities)
- [GET /api/models](#get-apimodels)
- [POST /api/models/load](#post-apimodelsload)
- [POST /api/models/unload](#post-apimodelsunload)
- [POST /api/probe](#post-apiprobe)
- [POST /api/jobs](#post-apijobs)
- [GET /api/jobs](#get-apijobs)
- [GET /api/jobs/{job_id}](#get-apijobsjob_id)
- [POST /api/jobs/{job_id}/cancel](#post-apijobsjob_idcancel)
- [GET /api/jobs/{job_id}/export](#get-apijobsjob_idexport)
- [DELETE /api/jobs/{job_id}](#delete-apijobsjob_id)
- [POST /api/batches](#post-apibatches)
- [GET /api/batches/{batch_id}](#get-apibatchesbatch_id)
- [POST /api/batches/{batch_id}/cancel](#post-apibatchesbatch_idcancel)
- [GET /api/batches/{batch_id}/export](#get-apibatchesbatch_idexport)
- [DELETE /api/batches/{batch_id}](#delete-apibatchesbatch_id)
- [GET /api/storage](#get-apistorage)
- [Data shapes](#data-shapes)
- [Errors](#errors)
- [Environment variables](#environment-variables)
- [Command-line client](#command-line-client)

## Start the server

```sh
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

`./scripts/dev.sh` starts this backend plus the Vite UI. `ffmpeg` and `ffprobe` are required.

## GET /api/health

```json
{"status": "ok", "active_jobs": 0, "loaded_models": 0}
```

`active_jobs` counts queued and running jobs. `loaded_models` counts resident models.

## GET /api/capabilities

Everything an agent needs to plan work. Values come from the running server, so read them
rather than hard-coding limits.

```json
{
  "service": "stt-studio",
  "api_version": "1.0.0",
  "default_model": "small",
  "compute_types": ["float16", "float32", "int8", "int8_float16"],
  "runtimes": ["faster-whisper", "mlx-audio"],
  "runtime_capabilities": {
    "faster-whisper": {"computeTypes": true, "beamSize": true, "vad": true,
                        "initialPrompt": true, "languageHint": "codes", "timedSegments": true},
    "mlx-audio": {"computeTypes": false, "beamSize": false, "vad": false,
                   "initialPrompt": false, "languageHint": "auto", "timedSegments": false}
  },
  "accepted_extensions": ["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma",
                           "mp4", "m4v", "mov", "mkv", "webm"],
  "accepts_any_ffmpeg_decodable_media": true,
  "diarization": {"available": true, "model": "pyannote/speaker-diarization-3.1", "license": "MIT", "reason": null},
  "export_formats": ["txt", "srt", "vtt", "json"],
  "batch_export_formats": ["txt", "json"],
  "limits": {
    "max_upload_bytes": 2147483648,
    "max_batch_files": 25,
    "max_batch_upload_bytes": 8589934592,
    "max_audio_seconds": 7200.0,
    "max_active_jobs": 1,
    "max_language_length": 20,
    "beam_size": [1, 20]
  },
  "notes": ["..."],
  "endpoints": [{"method": "GET", "path": "/api/health", "purpose": "..."}]
}
```

`accepted_extensions` is advisory. Any container ffmpeg can decode is accepted; an undecodable
file fails at probe/transcription time with `422`.

## GET /api/models

Runnable presets, cache state and loaded models.

```json
{
  "default_model": "small",
  "compute_types": ["float16", "float32", "int8", "int8_float16"],
  "models": [{"id": "...", "name": "...", "runtime": "faster-whisper", "supported": true,
              "repository": "Systran/faster-whisper-small", "cached": true, "...": "..."}],
  "cached_models": ["models--Systran--faster-whisper-small"],
  "loaded_models": [{"model": "small", "compute_type": "int8", "runtime": "faster-whisper"}],
  "runtimes": ["faster-whisper", "mlx-audio"],
  "runtime_capabilities": {"...": "..."},
  "custom_model_runtime": "CTranslate2 / faster-whisper",
  "custom_model_help": "Custom Hugging Face repositories must contain a compatible CTranslate2 Whisper model."
}
```

Only entries with `supported: true` for an installed runtime are listed. A custom
`owner/model` must contain CTranslate2 weights.

## POST /api/models/load

JSON body:

```json
{"model": "small", "compute_type": "int8", "runtime": "faster-whisper"}
```

`runtime` is optional for built-in presets. Downloads the model if it is not cached, loads it,
and runs a one-second silent warmup. Returns `{"model", "compute_type", "loaded": true,
"runtime", "warmup": "complete", "device"}`. A failed download or load returns `503`. The call
can take minutes on a cold model and runs in the worker thread pool, so health checks stay
responsive.

## POST /api/models/unload

No body. Releases resident models, runs garbage collection, and clears the MLX cache when that
runtime was used.

```json
{"unloaded": [{"model": "tiny", "compute_type": "int8", "runtime": "faster-whisper"}],
 "loaded_models": []}
```

`409` if a job is running. Model swaps release the previous model automatically.

## POST /api/probe

`multipart/form-data` with one field, `file`. Streams the upload to a temporary file, runs
ffprobe, and deletes it. Bounded by `max_upload_bytes`.

```json
{
  "container": "mov,mp4,m4a,3gp,3g2,mj2",
  "duration_seconds": 45.914014,
  "has_audio": true,
  "has_video": true,
  "streams": [
    {"index": 0, "type": "video", "codec": "mpeg4", "width": 640, "height": 360, "duration_seconds": 45.88},
    {"index": 1, "type": "audio", "codec": "aac", "channels": 1, "sample_rate": 22050.0, "duration_seconds": 45.914014}
  ],
  "filename": "talk.mp4",
  "max_audio_seconds": 7200.0,
  "exceeds_audio_limit": false,
  "transcribable": true
}
```

When `transcribable` is false, `reason` explains why:

- no audio track — `"The file has no audio track to transcribe."`
- longer than one job — `"Longer than the single-job limit; split it into chunks (scripts/stt_cli.py does this). Speaker labels need one job, so raise WHISPER_MAX_AUDIO_SECONDS to label this file."`

`422` when the file is empty or ffprobe cannot read it.

## POST /api/jobs

`multipart/form-data`:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `file` | file | required | Audio or video; ffmpeg extracts the audio |
| `model` | string | `small` | Preset or Hugging Face `owner/model` |
| `runtime` | string | inferred | `faster-whisper` or `mlx-audio` |
| `compute_type` | string | `int8` | `int8`, `int8_float16`, `float16`, `float32` |
| `language` | string | auto | Code such as `en`; max 20 characters |
| `beam_size` | int | `5` | 1–20 |
| `vad_filter` | bool | `true` | Voice-activity filtering |
| `initial_prompt` | string | empty | Vocabulary hint (names, jargon) |
| `diarize` | bool | `false` | Add anonymous speaker labels (local pyannote). Refused with `422` when unavailable, or when the runtime returns untimed segments (`mlx-audio`) |

`202` response: `{"id": "<uuid>", "status": "queued"}`. A file with no audio track is refused with `422` and `The file has no audio track to transcribe.` before the job is queued.

One job runs at a time (`max_active_jobs: 1`). A second submission returns `429`; the CLI waits
and retries. Uploads are capped at `max_upload_bytes` plus a 1 MiB multipart allowance, enforced
before parsing from both `Content-Length` and streamed chunks.

## GET /api/jobs/{job_id}

Query `wait`: seconds to long-poll, `0`–`60`, default `0`. The request returns as soon as the
job reaches `completed`, `failed` or `cancelled`, or when the wait expires.

```json
{
  "id": "3ac69a80-3b6e-4e39-a223-2b1f77ebfae5",
  "status": "completed",
  "created_at": 1795146532.5,
  "model": "tiny",
  "runtime": "faster-whisper",
  "compute_type": "int8",
  "language": null,
  "beam_size": 5,
  "vad_filter": true,
  "segments": [{"start": 0.0, "end": 2.24, "text": "Welcome to STT Studio.", "avg_logprob": -0.2, "no_speech_prob": 0.01}],
  "info": {"language": "en", "language_probability": 0.99, "duration": 45.9},
  "error": null,
  "progress": 100.0,
  "cancel_requested": false,
  "filename": "talk.mp4",
  "batch_id": null,
  "diarize": false
}
```

Statuses: `queued`, `running`, `completed`, `failed`, `cancelled`. `segments` grows while a job
runs (a cancelled job keeps what it produced). Finished jobs stay available until deleted, including
after a server restart, with the same `id`. A job that was `queued` or `running` when the server stopped
returns `failed` with `error: "Interrupted by a server restart"`. `progress` is a measured percentage for
faster-whisper, or `null` when the runtime cannot report it. `error` is a readable message.

## POST /api/jobs/{job_id}/cancel

No body. A queued job becomes `cancelled` immediately; a running job sets `cancel_requested` and
stops at the next checkpoint. Returns the job object. `409` if it has already finished, `404` if
unknown.

The MLX runtime cannot be interrupted inside `generate`, so cancellation can take as long as the
current step.

## GET /api/jobs/{job_id}/export

Query `format`: `txt` (default), `srt`, `vtt`, `json`.

- `200` with `Content-Disposition: attachment; filename="<upload stem>.<format>"`.
- `409` while queued/running, or if the job failed.
- `422` for an unknown format.
- `404` for an unknown job.

A cancelled job exports its partial transcript.

## DELETE /api/jobs/{job_id}

No body. Deletes a finished job (`completed`, `failed` or `cancelled`) and its stored transcript.
Returns `{"id": "<uuid>", "deleted": true}`. `409` while the job is `queued` or `running`; `404` if
unknown or already deleted. If the job belongs to a batch, the batch keeps its record and the job is
removed from it, so the batch's `total`, `jobs` and export no longer include that file.

## POST /api/batches

`multipart/form-data` with repeated `files` fields (1–25) plus the same model fields as
`/api/jobs`. Files run one after another in submission order on the single worker.

`202` returns the batch summary (see below). `429` if any job is already queued or running,
`422` for no files, too many files, invalid settings, or any file with no audio track (nothing is queued). The combined upload is capped at
`max_batch_upload_bytes` plus the multipart allowance.

## GET /api/batches/{batch_id}

```json
{
  "id": "c7933540-9194-4c88-9d35-5b29295c47be",
  "created_at": 1795146532.5,
  "total": 1,
  "counts": {"queued": 0, "running": 0, "completed": 1, "failed": 0, "cancelled": 0},
  "done": true,
  "jobs": [{"...": "job objects, in submission order"}]
}
```

`done` is true when every job is terminal.

## POST /api/batches/{batch_id}/cancel

No body. Cancels queued jobs immediately and asks the running job to stop. Returns the summary.
`404` if unknown.

## GET /api/batches/{batch_id}/export

Query `format`: `json` (default) or `txt`. Requires every file to be finished (`409` otherwise);
`422` for an unknown format.

`txt` writes one headed section per file:

```text
== talk.mp4 (completed) ==
Welcome to STT Studio.
...
```

`json`:

```json
{
  "format": "stt-studio-batch-results",
  "version": 1,
  "batch_id": "c7933540-...",
  "exported_at": "2026-11-20T03:48:57.363978+00:00",
  "files": [
    {"filename": "talk.mp4", "status": "completed", "error": null,
     "info": {"language": "en", "duration": 45.9},
     "segments": [{"start": 0.0, "end": 2.24, "text": "Welcome to STT Studio."}]}
  ]
}
```

## DELETE /api/batches/{batch_id}

No body. Deletes a finished batch, its record and all of its jobs. Returns
`{"id", "deleted": true, "jobs_deleted": n}`. `409` while any file in the batch is `queued` or `running`
(cancel the batch and wait first); `404` if unknown.

## GET /api/jobs

Lists stored jobs, newest first, as metadata only (no `segments`; fetch a transcript with
`GET /api/jobs/{job_id}`). Query: `limit` (1–200, default 50) and `offset` (default 0). Out-of-range values
return `422`.

```json
{"jobs": [{"id": "...", "status": "completed", "created_at": 1795146532.5, "model": "tiny",
           "runtime": "faster-whisper", "compute_type": "int8", "language": null, "beam_size": 5,
           "vad_filter": true, "info": {"language": "en", "language_probability": 0.99, "duration": 45.9},
           "error": null, "progress": 100.0, "cancel_requested": false, "filename": "talk.mp4",
           "batch_id": null, "diarize": false}],
 "total": 1, "limit": 50, "offset": 0}
```

## GET /api/storage

Where results are kept and how much space they take. Audio is never stored here.

```json
{"data_dir": "data", "jobs": 12, "batches": 2, "bytes": 3145728}
```

`data_dir` is the folder name only (relative to the repository when it is inside it, otherwise the
last path component). It is never an absolute path. `bytes` counts the SQLite database and its WAL and
SHM files. Set the folder with `WHISPER_DATA_DIR`; the default is `data/` in the repository.

## Speaker labels (`diarize`)

Set `diarize=true` on `POST /api/jobs` or `POST /api/batches`. Labelling runs locally after
transcription, with the pyannote `speaker-diarization-3.1` model (MIT on its card). The pipeline also loads `pyannote/wespeaker-voxceleb-resnet34-LM`, which is CC-BY-4.0 and needs attribution.

- **Availability:** `GET /api/capabilities` returns `diarization: {available, model, license, reason}`.
  A request with `diarize=true` is refused with `422` and the same `reason` when it is unavailable. It is also refused for a runtime whose `timedSegments` capability is `false` (`mlx-audio`), because its single untimed segment cannot be matched to speakers.
- **Requirements:** the diarizer environment (`scripts/setup-diarizer.sh`) and `HF_TOKEN` in the
  project's `.env` or the server environment. The token's account must have accepted the model terms.
- **Result:** each segment takes the speaker who talks longest inside it, with labels `Speaker 1`,
  `Speaker 2`, ... in order of first turn. The job's `info.diarization` reports
  `{"model", "license", "status", "speakers"}`; a failure has `"status": "failed"` and a `reason`
  with no local paths. A cancel during labelling gives `"status": "cancelled"`. The transcript is kept in every case.
- **Exports:** TXT and SRT prefix each line `Speaker 1: `; VTT uses `<v Speaker 1>`; JSON and
  batch JSON add a `speaker` key to each segment.
- **Limits:** labels are per job, so the CLI never splits a file when `--diarize` is used. Overlapping
  speech is not separated. Accuracy varies with the recording; see SKILL.md for the measured results.

## Data shapes

Segment: `{"start": <seconds>, "end": <seconds>, "text": "<string>"}`, plus `"speaker": "Speaker 1"` when
diarization labelled it, plus runtime extras
(`avg_logprob`, `no_speech_prob`). Exporters drop extras and empty text.

Export formats:

- `txt` — one segment per line, trailing newline.
- `srt` — numbered blocks, `HH:MM:SS,mmm`.
- `vtt` — `WEBVTT` header, `HH:MM:SS.mmm`.
- `json` — array of `{"start", "end", "text"}`.

Milliseconds are rounded identically to the browser export.

## Errors

`segments` is always a list (empty until the job produces segments). A well-formed `owner/model` that does not exist is accepted with `202`; the job then fails, and its `error` says what went wrong.

Most error bodies are `{"detail": "<message>"}`. Validation errors (422 from a bad query or form value) return `detail` as a list of objects with `type`, `loc`, `msg`, `input` and `ctx`. Malformed HTTP requests (400) can return plain text.

| Code | When |
| --- | --- |
| `400` | Malformed `Content-Length` on an upload |
| `403` | Non-loopback peer, non-loopback `Host`, or non-local `Origin` on a mutation |
| `404` | Unknown job or batch id, or one that was deleted |
| `409` | Export before completion, export of a failed job, unload while running, cancel after completion, delete while queued or running |
| `413` | Upload over the configured byte limit |
| `422` | Invalid model/runtime/compute type/`beam_size`/format/language length, empty upload, undecodable media, a file with no audio track, `wait` outside 0–60, `diarize=true` when speaker labels are unavailable or the runtime is untimed |
| `429` | Another job or batch is active |
| `503` | Model could not be downloaded, loaded or warmed; or local job storage is unavailable (the body names no paths) |

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `WHISPER_MAX_UPLOAD_BYTES` | `2147483648` (2 GiB) | Per-file upload cap |
| `WHISPER_MAX_BATCH_UPLOAD_BYTES` | `8589934592` (8 GiB) | Combined batch upload cap |
| `WHISPER_MULTIPART_OVERHEAD_BYTES` | `1048576` (1 MiB) | Allowance for multipart headers |
| `WHISPER_MAX_ACTIVE_JOBS` | `1` | Concurrent jobs (the worker is serial) |
| `WHISPER_MAX_BATCH_FILES` | `25` | Files per batch |
| `WHISPER_MAX_AUDIO_SECONDS` | `7200` | Decoded audio per job |
| `WHISPER_MODEL_CACHE` | `<repo>/models` | Model cache directory |
| `WHISPER_DATA_DIR` | `<repo>/data` | Folder for the job and transcript database (`stt-studio.sqlite`); audio is never stored there |
| `WHISPER_DEVICE` | `cpu` | CTranslate2 device |
| `WHISPER_DISABLE_MLX` | unset | `1` disables the MLX runtime |
| `HF_TOKEN` | unset (also read from `.env`) | Hugging Face token for the gated pyannote model; never returned by the API |
| `STT_STUDIO_DIARIZER_PYTHON` | `.local/diarizer-venv/bin/python` | Interpreter for speaker labels; a configured path is used as-is |
| `STT_STUDIO_DIARIZER_THRESHOLD` | `0.7046` | pyannote clustering threshold; `0.5` splits more, for large groups |

## Command-line client

`scripts/stt_cli.py` (standard library only) wraps this API, extracts video audio, splits long
files at pauses, retries `429`, and writes exports:

```sh
python3 scripts/stt_cli.py status
python3 scripts/stt_cli.py capabilities
python3 scripts/stt_cli.py models
python3 scripts/stt_cli.py load small --compute-type int8
python3 scripts/stt_cli.py unload
python3 scripts/stt_cli.py probe clip.mp4
python3 scripts/stt_cli.py transcribe clip.mp4 --format srt -o clip.srt --chunk-seconds 1200
python3 scripts/stt_cli.py transcribe meeting.mp4 --diarize --format vtt -o meeting.vtt
python3 scripts/stt_cli.py transcribe a.wav b.mp4 --output-dir out --format vtt
python3 scripts/stt_cli.py batch submit a.wav b.wav
python3 scripts/stt_cli.py batch status <batch-id>
python3 scripts/stt_cli.py batch cancel <batch-id>
python3 scripts/stt_cli.py batch export <batch-id> --format json -o batch.json
python3 scripts/stt_cli.py job status <job-id> --wait 60
python3 scripts/stt_cli.py job cancel <job-id>
python3 scripts/stt_cli.py job export <job-id> --format vtt -o captions.vtt
python3 scripts/stt_cli.py transcribe call.mp4 --diarize --json        # adds speaker labels, one job per file
```

Environment: `STT_STUDIO_URL` (default `http://127.0.0.1:8765`). Exit codes: `0` success,
`1` failure, `2` usage error. `job status` exits `1` when the job has `failed`. `transcribe` has a
`--timeout` budget (default 3,600 s). On expiry the CLI cancels the remaining chunks, writes no `-o`
file and exits 1. Chunks that finished stay on the server: the batch id is printed to stderr in every
mode, so export them with `batch export <batch-id>`. There is no `batch wait` command; poll
`batch status` until `done` is true.

Known limits: jobs and batches persist until deleted, but a job running at a restart is marked failed rather than resumed; one job runs at a time;
chunk joins can split a word where no pause exists; the supported MLX model (Qwen3-ASR) returns
one untimed segment, so SRT and VTT hold a single caption. Speaker labels are anonymous and
measured on few recordings (see SKILL.md, "Speaker labels"); overlapping speech is not separated.
