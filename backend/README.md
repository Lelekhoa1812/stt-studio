# STT Studio Local Inference API

This service runs faster-whisper on the same Mac as its caller. Audio is written to a temporary file, passed to the local inference model, and deleted when the job finishes or fails. The API does not upload audio to a hosted service. The one outbound request that carries transcript text is the optional AI mode (see [AI modes](#ai-modes-optional)), which runs only when a caller asks for it and sends text, never audio. Model downloads come from Hugging Face when a model is not already cached; the cache defaults to the repository's `models/` directory.

The default is `small` with `int8` compute on CPU, which keeps the first download and memory use modest. You can select a built-in model preset or, for a custom model, a Hugging Face `owner/model` identifier and compute type (`int8`, `int8_float16`, `float16`, or `float32`). A custom repository must contain a model format supported by faster-whisper/CTranslate2; arbitrary ASR repositories and Transformers checkpoints are not interchangeable. The [model catalog](../models/README.md) records verified current choices and future runtime candidates. Check the exact repository's card and license before use. The default language is automatic detection; pass a language code to force one. Set `WHISPER_DEVICE` to change the CTranslate2 device. Device support depends on the installed CTranslate2 build and hardware. Only one model is kept resident at a time to bound memory use.

## Run

From the repository root, install the backend requirements and start the server bound to loopback:

```sh
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

The API only listens on localhost in this command and rejects non-loopback client connections and non-loopback `Host` headers. Mutating requests carrying an `Origin` header are accepted only from localhost origins; no broad CORS policy is enabled. Health and model status snapshots stay responsive while the inference worker loads or runs a model. Treat local callers as trusted because the service can download and run model files and consume CPU and disk space. Upload files are limited to 2 GiB by default (`WHISPER_MAX_UPLOAD_BYTES`), audio is limited to 7,200 seconds by default (`WHISPER_MAX_AUDIO_SECONDS`), and one queued or running transcription is allowed (`WHISPER_MAX_ACTIVE_JOBS`). Before FastAPI parses multipart data, the request body is capped at the file limit plus 1 MiB for boundaries, headers, and form fields; change that allowance with `WHISPER_MULTIPART_OVERHEAD_BYTES` if needed. The route separately checks the exact uploaded file size. This cap is enforced from both `Content-Length` and streamed body chunks when that header is absent or inaccurate.

The server requires both `ffprobe` and `ffmpeg`. It probes duration before decoding or loading a model and rejects invalid media or files over the configured limit. ffmpeg writes 16 kHz mono float PCM to a temporary disk file that is read through a NumPy memory map, avoiding an in-memory PCM copy and unbounded captured stdout. Its output is capped at five seconds beyond the configured duration, then the decoded sample count is checked so an inaccurate probe cannot silently truncate a long recording. The temporary PCM is removed after success or failure. The duration cap also bounds faster-whisper's in-memory spectrogram work; a two-hour recording can still require several gigabytes of RAM during feature extraction. Lower `WHISPER_MAX_AUDIO_SECONDS` on machines with less memory. The server checks available disk space before decoding.

## API

- `GET /api/health` reports service status, active jobs, and loaded models.
- `GET /api/capabilities` reports limits, accepted extensions, runtimes and their capabilities, export formats, notes, and the endpoint list. Read it instead of hard-coding limits; the values come from the running server's environment.
- `GET /api/models` reports runnable presets, their exact source cards and licenses, approximate download sizes, cached state, supported compute types, available runtimes, and models loaded into memory. Each model also carries `controls` (which request fields its runtime applies), `emits` (which outputs it produces) and `cancellation` (whether a running job can stop inside generation); see [capability metadata](../models/README.md#capability-metadata).
- `POST /api/models/load` with JSON `{"model":"small","compute_type":"int8","runtime":"faster-whisper"}` downloads if needed, loads the model, and performs a one-second silent inference warmup. `runtime` may be omitted for built-in presets. It runs in FastAPI's worker thread pool, so model loading does not block the async event loop.
- `POST /api/models/unload` releases resident models, collects garbage, and clears the MLX cache when that runtime was used. It also stops the resident speaker-label worker, which the next labelling job starts again. It returns `409` while a job is running; model swaps release the previous model automatically.
- `POST /api/probe` accepts one multipart `file` and returns `container`, `duration_seconds`, `has_audio`, `has_video`, `streams`, `filename`, `exceeds_audio_limit`, `transcribable`, and a `reason` when it is not transcribable. It runs ffprobe only, without loading a model, and is bounded by the same upload limit.
- `GET /api/jobs?limit=50&offset=0` lists stored jobs newest first, with `total`, as metadata only (no `segments`). `limit` is 1–200.
- `DELETE /api/jobs/{id}` deletes a finished job and its transcript (`409` while it is queued or running, `404` if unknown). If the job belongs to a batch, the batch keeps its record without that file.
- `POST /api/jobs` accepts multipart fields `file` (required), `model`, `runtime`, `compute_type`, `language`, `beam_size`, `vad_filter`, `initial_prompt`, and `diarize`. It responds `202` with `{ "id", "status": "queued", "ignored_controls": [] }`. `ignored_controls` lists, by field name, the Whisper-only controls the chosen runtime accepts but cannot apply: `beam_size` (any value other than 5), `vad_filter` (when true), and `initial_prompt` (when non-empty). The MLX runtime ignores all three, so an MLX job's request is accepted rather than refused. Those controls are stored as `NULL` on the job, and job responses carry the same `ignored_controls` list. Audio is decoded by local ffmpeg to 16 kHz mono PCM before inference, avoiding dependence on PyAV compatibility. Video files are accepted directly.
- `GET /api/jobs/{id}` returns job state. States are `queued`, `running`, `completed`, `failed`, and `cancelled`. Completed jobs include `segments` (`start`, `end`, `text`, `avg_logprob`, `no_speech_prob`) and detected-language/duration `info`; failed jobs include a readable `error`. Add `?wait=SECONDS` (0–60) to long-poll until the job finishes.
- `POST /api/jobs/{id}/cancel` cancels a queued job immediately or asks a running job to stop at its next checkpoint. Partial segments are kept, and `GET /api/jobs/{id}/export?format=txt|srt|vtt|json` downloads a completed or cancelled transcript as an attachment. Export returns `409` before completion and `422` for an unknown format.
- `POST /api/jobs/{id}/ai` with JSON `{"mode": "summary"}` or `{"mode": "action_items"}` runs an optional AI mode on a completed or cancelled transcript through NVIDIA NIM. It returns `{"mode", "text", "model", "created_at"}`, stores the result in the job's `ai_results`, and replaces any earlier result for the same mode. `capabilities.ai` reports whether it can run and why not. Details are in [AI modes](#ai-modes-optional).
- `POST /api/batches` queues 1–25 files with the same model fields as `/api/jobs`; they run one after another on the single worker. `GET /api/batches/{id}` returns per-file status and counts, `POST /api/batches/{id}/cancel` cancels the rest, and `GET /api/batches/{id}/export?format=txt|json` downloads the finished batch (headed text sections, or a versioned `stt-studio-batch-results` JSON document). `DELETE /api/batches/{id}` deletes a finished batch and all of its jobs (`409` while any file is queued or running). `GET /api/storage` returns `{"data_dir", "jobs", "batches", "bytes"}`, where `data_dir` is the folder name only.

`scripts/stt_cli.py` wraps this API for scripts and AI agents: it probes media, extracts video audio, splits long recordings at pauses into chunks, retries when the worker is busy, and writes the same export formats. `skills/stt-studio/` contains agent instructions and a full [API reference](../skills/stt-studio/API_REFERENCE.md). The CLI needs only the Python standard library plus `ffmpeg`/`ffprobe` on `PATH`, and reads `STT_STUDIO_URL` (default `http://127.0.0.1:8765`).

Invalid job parameters return `422`, missing jobs `404`, uploads over the configured size limit `413`, busy job queue `429`, rejected remote connections/origins `403`, and model-load failures `503`. Audio inspection and decode errors appear on the job's `error` field. A `404` means the id is unknown or was deleted; `503` means local job storage is unavailable.

## Storage

Jobs, batches and their transcripts (segments, detected-language info, status and error) are stored in `data/stt-studio.sqlite` inside the repository, or in the folder named by `WHISPER_DATA_DIR`. The database uses SQLite in WAL mode with a schema version table (now version 3: `beam_size` and `vad_filter` accept NULL for runtimes that ignore them, and `ignored_controls` is stored per job). Opening an older database migrates it in place, rebuilding the jobs table once and keeping every row. Records persist until you delete them with `DELETE /api/jobs/{id}` (a finished job) or `DELETE /api/batches/{id}` (a finished batch and its jobs); there is no automatic expiry. Audio is never written to this folder: uploads go to temporary files that are removed when processing ends, and only metadata and transcript text are stored. `GET /api/storage` reports the folder name (never an absolute path), the number of jobs and batches, and the bytes used.

Job and batch ids are UUIDs and stay valid across restarts. On startup, any job still queued or running in the database is marked `failed` with the error `Interrupted by a server restart`, and it keeps whatever was saved before the stop. This recovery runs once per server start. Finished jobs and batches from earlier runs are returned as they were stored, and exports work for them. The `data/` folder is ignored by git; the database file is readable only by the current user.

Only one job or batch runs at a time (`WHISPER_MAX_ACTIVE_JOBS`), so a second submission returns `429`. A single job is capped at `WHISPER_MAX_AUDIO_SECONDS` (default 7,200); longer recordings must be split by the caller, which the CLI does automatically. The MLX runtime returns one untimed segment for the supported Qwen3-ASR model, so SRT and VTT exports contain a single caption, and it cannot be interrupted inside a generation call. Diarisation is not implemented.

Inference is separated behind an adapter contract (`load`, `warmup`, `transcribe`) managed by `ModelManager`. The current implementation registers only `faster-whisper`; the API rejects built-in model/runtime mismatches and custom runtime IDs that are not installed. Adapters normalize their output to timestamped segment rows and detected-language metadata. The contract is an integration boundary, not a claim that catalog candidate families already run.

## Speaker labels (optional)

`diarize=true` on `POST /api/jobs` or `POST /api/batches` adds anonymous speaker labels (`Speaker 1`, `Speaker 2`, ...) using the pyannote `speaker-diarization-3.1` model. It runs locally after transcription, in its own Python 3.12 environment that is kept separate from this backend:

```sh
scripts/setup-diarizer.sh          # creates .local/diarizer-venv (torch 2.8, pyannote.audio 3.4.0)
```

Add `HF_TOKEN` to the project's `.env`. The Hugging Face account behind it must have accepted the terms of `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`. `GET /api/capabilities` reports `diarization.available` and the reason when it is unavailable; a `diarize=true` request is refused with `422` until it is. A failed labelling run keeps the transcript and records `info.diarization.status = "failed"` with a path-free reason. Labels are per job, so the CLI does not split files when diarizing. Overlapping speech is not separated. `STT_STUDIO_DIARIZER_PYTHON` and `STT_STUDIO_DIARIZER_THRESHOLD` (default `0.7046`) tune the runner; see the skill's API reference for measured results.

## AI modes (optional)

`POST /api/jobs/{id}/ai` asks NVIDIA NIM for a summary or a list of action items. It is off unless a caller makes the request, and the job's transcript is the only input.

- **What leaves the machine.** The transcript text, as lines with optional speaker labels and no timestamps. Audio, file names and job ids are never sent. A client may send the transcript as the user sees it (with edits and speaker names) in a `segments` field; otherwise the stored segments are used.
- **Model.** `nvidia/nemotron-3-ultra-550b-a55b`, through the OpenAI-compatible chat completions endpoint at `https://integrate.api.nvidia.com/v1/chat/completions`.
- **Keys.** `NVIDIA_NIM_API_KEY_1` to `NVIDIA_NIM_API_KEY_9`, read from the environment or the project's `.env`, in order. A `401`, `403`, `429` or `5xx` response moves to the next key; when every key is refused the request returns `502`. Other refusals and network errors are not retried. No key is returned, logged or stored, and no error body contains a key, the URL or a local path. `GET /api/capabilities` reports `ai: {available, model, modes, reason}`; with no key the endpoint returns `503` and the reason.
- **Limits.** The job must be completed or cancelled (`409` otherwise), the transcript must not be empty and must be at most 150,000 characters (`422`), and each model call has a 120-second timeout.
- **Prompts.** Two short system prompts (`backend/nim.py`) ask for one overview sentence and 5 to 8 bullets, or for action items with the owner only when named and the due date only when stated. Both tell the model to use only the transcript and to say so when the transcript does not answer the question.
- **Storage.** Results are stored in the job row (schema version 2 adds the `ai_results` column; version 1 databases migrate on open) and survive restarts. Deleting the job deletes them.

The opt-in real call is `STT_STUDIO_REAL_NIM=1 python -m pytest backend/tests/test_ai_modes.py -k real`. It sends a short synthetic meeting transcript and needs a key in the environment or `.env`; the default test run uses a fake HTTP function and never calls the network.

## Tests

```sh
python3 -m pip install -r backend/requirements-dev.txt
python3 -m pytest backend/tests
```
