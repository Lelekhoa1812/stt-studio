# STT Studio

STT Studio is an early-stage local speech-to-text workspace for Mac. It combines a FastAPI service running `faster-whisper` (and optionally `mlx-audio` on Apple Silicon) with a Vite/React interface. Select an audio file, choose a model and transcription settings, run a local job, then review, edit, search, copy, or export the transcript. It is a development project, not a packaged or production-ready application.

The interface supports drag-and-drop or file browsing, audio playback with a waveform timeline (click or drag to seek), model loading and warmup, language and compute settings, vocabulary hints, job status, editable timestamped segments, search with find and replace across segments, and TXT/SRT/JSON export. Keyboard shortcuts: Space plays or pauses, ArrowUp and ArrowDown move the active segment, Cmd/Ctrl+F focuses search, Escape clears the search when focus is outside a text field, and Cmd/Ctrl+Enter starts a transcription. The edited transcript is autosaved in the browser, and a recovery notice offers to restore it after a reload. Batches retry only their failed files, keep a history you can delete from, and the model storage panel deletes cached models. AI summary and action-item requests can be cancelled.

The runnable catalog includes Whisper size/language variants, compatible Distil-Whisper conversions, and Qwen3-ASR 0.6B (8-bit MLX) on Apple Silicon when `mlx-audio` is installed; custom CTranslate2 Whisper repositories are supported too. Other ASR families are documented as adapter candidates and are not presented as runnable yet. You can explicitly save and reopen projects in the browser, including their audio, edited transcript, and settings, or export a portable `.sttproj` file. Transcripts appear after inference completes; incremental token or percentage progress is not available from the current backend.

## Run locally

Requirements: Python 3.12 (pinned in `.python-version`), Node.js 22, npm, and `ffmpeg` and `ffprobe` available on `PATH`. The first model use downloads weights from Hugging Face.

The simplest way to run both local services on macOS is:

```sh
./scripts/dev.sh
```

The launcher creates the Python environment, installs backend and frontend dependencies when needed, and binds the services to loopback. On Apple Silicon it also tries to install the optional MLX-Audio runtime. It requires Python 3.12, Node.js, npm, `ffmpeg`, and `ffprobe`; on macOS you can install missing tools with `brew install python@3.12 node ffmpeg`. Keep the terminal open while using the app and press Ctrl+C to stop both services.

To start the services manually, run the backend from the repository root:

```sh
python3.12 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

Then, in a second terminal, install and start the frontend with `cd frontend && npm ci && npm run dev`. Open the local URL printed by Vite. Its development proxy sends `/api` requests to the backend on port 8765.

The API lists models with `GET /api/models`, shows the local cache with `GET /api/models/cache`, and removes a cached model with `DELETE /api/models/cache/{model_id}`. Models are loaded and unloaded with `POST /api/models/load` and `POST /api/models/unload`. Limits and runtimes come from `GET /api/capabilities`, `GET /api/health` reports service state, and `POST /api/probe` inspects media without transcribing. Transcription jobs use `POST /api/jobs`, `GET /api/jobs` (paginated, newest first), `GET /api/jobs/{id}` (with `?wait=`), job cancel, export, `POST /api/jobs/{id}/ai` and `DELETE /api/jobs/{id}`. Batches use `POST /api/batches`, `GET /api/batches` (paginated, newest first), `GET /api/batches/{id}`, batch cancel, export and `DELETE /api/batches/{id}`. `GET /api/storage` reports stored counts and disk use. Finished jobs and batches stay until deleted. The default model is `small` with `int8` compute. [backend/README.md](backend/README.md) covers settings, fields, limits, and error responses; [API_REFERENCE.md](skills/stt-studio/API_REFERENCE.md) lists every request and response, and [SKILL.md](skills/stt-studio/SKILL.md) gives agent instructions.

Optional speaker labels (`--diarize` or `diarize=true`, on faster-whisper jobs only) run a local pyannote model. Set them up once with `scripts/setup-diarizer.sh` and put `HF_TOKEN` in `.env`, after accepting the terms of `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on Hugging Face. If the token cannot access those gated repositories, the job's diarization status says so. Labels are anonymous and their accuracy depends on the recording; see [backend/README.md](backend/README.md#speaker-labels-optional).

## Command-line client

`scripts/stt_cli.py` drives the same API from a terminal. It uses `STT_STUDIO_URL` (default `http://127.0.0.1:8765`). Run `python scripts/stt_cli.py COMMAND --help` for every flag.

| Command | Purpose |
| --- | --- |
| `transcribe FILE... [--format txt\|srt\|vtt\|json] [-o PATH \| --output-dir DIR] [--json] [--chunk-seconds 1200] [--timeout 3600] [--diarize]` | Transcribe audio or video. Inputs longer than the chunk size are split automatically |
| `batch submit FILE... [--diarize]` | Queue files and print the batch id without waiting |
| `batch status BATCH_ID`, `batch cancel BATCH_ID` | Show a batch, or cancel its remaining files |
| `batch export BATCH_ID [--format txt\|json] [-o PATH]` | Download a finished batch |
| `job status JOB_ID [--wait SECONDS]`, `job cancel JOB_ID` | Show a job (`--wait` long-polls, max 60 s), or cancel it |
| `job export JOB_ID [--format txt\|srt\|vtt\|json] [-o PATH]` | Download a finished job |
| `jobs list [--limit 1-200] [--offset N] [--json]`, `jobs delete JOB_ID [--json]` | List stored jobs, newest first (`--limit` defaults to 50), or delete a finished job |
| `batches list [--limit 1-200] [--offset N] [--json]`, `batches delete BATCH_ID [--json]` | List stored batches, or delete a finished batch and its jobs |
| `ai summary JOB_ID [--json]`, `ai action-items JOB_ID [--json]` | Summary or action items for a finished job via NVIDIA NIM (sends transcript text only) |
| `storage [--json]` | Count stored jobs and batches and show disk use |
| `status`, `capabilities`, `models`, `load [MODEL] [--compute-type] [--runtime]`, `unload`, `probe FILE...` | Service health, limits, model list, model load/unload, and media inspection |

`transcribe` and `batch submit` also accept `--model` (default `small`), `--runtime`, `--compute-type` (`int8` by default), `--language` (default `auto`), `--beam-size` (1-20, default 5), `--no-vad`, and `--prompt`. Videos are uploaded as their 16 kHz mono WAV audio track, and non-WAV audio larger than 1 GiB is converted to WAV before upload.

To transcribe recordings as they land in a folder, run `scripts/watch_folder.py WATCH_DIR --output-dir OUT_DIR` while the backend is running. It waits while the server is down or answers `429`. A file that fails with an HTTP 5xx error or times out is retried, and after three failed attempts it is marked failed until `--retry-failed` is given. `--file-timeout` (default 3600 seconds) limits each attempt, and `--once` scans twice and exits. An exclusive `.watch.lock` in the output folder allows one writer at a time, so a second watcher refuses to start. On macOS, `scripts/watch_service/install.sh WATCH_DIR OUT_DIR` writes a launch agent that runs the watcher in the background. The script does not load the agent: it prints the `launchctl bootstrap` command to run. `status.sh` and `uninstall.sh` in the same folder show and remove it. See [SKILL.md](skills/stt-studio/SKILL.md#watch-a-folder) for details.

## Privacy and data handling

Inference runs on this machine; the project does not send recordings to a hosted transcription service. The optional AI modes are the one exception to "nothing leaves this machine": when you select Summary or Action items for a finished transcript, the app sends that transcript's text to NVIDIA NIM (model `nvidia/nemotron-3-ultra-550b-a55b`). The text is the transcript as shown, with speaker labels or names you have entered, and no timestamps, file names or audio. Nothing is sent unless you select a mode, and no mode runs automatically. The NVIDIA keys (`NVIDIA_NIM_API_KEY_1` to `_9`, read from the environment or `.env`) are never shown in the interface, returned by the API, or written to logs. AI results are kept with the job until you delete it. The backend writes uploads and decoded audio to local temporary files and removes them when processing finishes or fails normally. A forced termination can leave temporary files behind. Model weights may be downloaded from Hugging Face on first use; cached weights can be listed and deleted from the model storage panel or through the cache endpoints. Backend jobs, batches and transcripts are kept in a local SQLite database in `data/` (or the folder named by `WHISPER_DATA_DIR`) until you delete them; there is no automatic expiry. Audio is never written there. A job that was queued or running when the server stopped is marked failed with "Interrupted by a server restart", and job ids stay valid across restarts. The current session's transcript is autosaved in this browser's local storage, and projects are saved only when you choose to save them, in the same storage or as `.sttproj` files. Clearing browser site data can remove drafts and saved projects. Export transcripts separately for backup; exports do not include the audio or settings. Bind the service to `127.0.0.1` as shown. Authentication and protections for exposing the API to a network are not implemented.

Keep recordings, transcripts, logs, model weights, and virtual environments out of source control. The ignore rules cover common media, meeting files, outputs, logs, and model caches, but verify `git status` before sharing changes.

## Architecture

- `backend/app.py` implements the local FastAPI API, inference-adapter contract, serialized transcription worker, upload handling, and job and batch endpoints. `backend/diarization.py` runs the optional speaker labelling, and `backend/nim.py` the optional AI modes.
- `backend/tests/` contains API tests using a fake transcription model.
- `frontend/src/App.tsx` implements the single-workspace interface, keyboard handling, autosave, and client-side transcript editing and export.
- `scripts/stt_cli.py`, `scripts/watch_folder.py` and `scripts/watch_service/` are the command-line client, folder watcher, and macOS launch agent.
- `models/catalog.json` and [`models/README.md`](models/README.md) document model families, verified runtime formats, source cards, and license metadata. The same directory is used as the local model cache; downloaded weights are ignored and must not be committed.

## Testing

Run the backend suite with:

```sh
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend/tests
```

Run the evaluation tests with:

```sh
python -m unittest discover -s evaluation/tests
```

Lint the Python code with the pyflakes rules configured in `ruff.toml`:

```sh
python -m pip install ruff==0.17.6
python -m ruff check backend scripts evaluation
```

Build, type-check and browser-test the frontend from `frontend/`:

```sh
cd frontend
npm ci
npm run build
npm run typecheck
npx playwright install chromium
npm run test:e2e
```

An offline word error rate evaluator and calibration guidance are available in [evaluation/README.md](evaluation/README.md).

CI (`.github/workflows/ci.yml`) runs these same checks on pushes and pull requests, but it is currently paused: the repository owner disabled the workflow. Re-enable it with `gh workflow enable ci.yml`.

## Current limits and planned work

The backend serializes work to one active transcription at a time, defaults to a two-hour audio limit, and shows partial segments only for faster-whisper, and progress is measured by segment for that runtime alone; the MLX runtime shows an indeterminate state. Long recordings can still require substantial memory; see [backend/README.md](backend/README.md) for resource limits. Browser-saved projects are local to this browser; exported `.sttproj` files are portable but do not resume backend jobs. Jobs, batches and transcripts persist until deleted, and a job running at a restart is marked failed rather than resumed. Live microphone capture and translation are not implemented.

## Attribution

STT Studio uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for inference with OpenAI's [Whisper](https://github.com/openai/whisper) model and [FFmpeg](https://ffmpeg.org/) for audio decoding. See upstream projects for their licenses and model terms.

## License

The project is licensed under the MIT License; see [LICENSE](LICENSE).
