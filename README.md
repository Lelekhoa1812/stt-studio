# STT Studio

STT Studio is an early-stage local speech-to-text workspace for Mac. It combines a FastAPI service running `faster-whisper` with a Vite/React interface. Select an audio file, choose a model and transcription settings, run a local job, then review, edit, search, copy, or export the transcript. It is a development project, not a packaged or production-ready application.

The interface supports drag-and-drop or file browsing, audio playback, model loading and warmup, language and compute settings, vocabulary hints, job status, editable timestamped segments, transcript search, and TXT/SRT/JSON export. The runnable catalog includes Whisper size/language variants and compatible Distil-Whisper conversions, with support for custom CTranslate2 Whisper repositories. Other ASR families are documented as adapter candidates and are not presented as runnable yet. You can explicitly save and reopen projects in the browser, including their audio, edited transcript, and settings. Transcripts appear after inference completes; incremental token or percentage progress is not available from the current backend.

## Run locally

Requirements: Python 3.12, Node.js 22, npm, and `ffmpeg` and `ffprobe` available on `PATH`. The first model use downloads weights from Hugging Face.

The simplest way to run both local services on macOS is:

```sh
./scripts/dev.sh
```

The launcher creates the Python environment, installs backend and frontend dependencies when needed, and binds the services to loopback. It requires Python 3.12, Node.js, npm, `ffmpeg`, and `ffprobe`; on macOS you can install missing tools with `brew install python@3.12 node ffmpeg`. Keep the terminal open while using the app and press Ctrl+C to stop both services.

To start the services manually, run the backend from the repository root:

```sh
python3.12 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

Then, in a second terminal, install and start the frontend with `cd frontend && npm ci && npm run dev`. Open the local URL printed by Vite. Its development proxy sends `/api` requests to the backend on port 8765.

The API exposes `GET /api/health`, `GET /api/capabilities`, `GET /api/models`, `POST /api/models/load`, `POST /api/models/unload`, `POST /api/probe`, `POST /api/jobs`, `GET /api/jobs/{id}` (with `?wait=`), job cancel and export, and a batch queue (`POST /api/batches` plus status, cancel and export). Finished jobs and batches are stored locally until deleted, with `GET /api/jobs` (paginated list), `DELETE /api/jobs/{id}`, `DELETE /api/batches/{id}` and `GET /api/storage`. The default model is `small` with `int8` compute. See [backend/README.md](backend/README.md) for settings, API fields, limits, and error responses, and [skills/stt-studio/SKILL.md](skills/stt-studio/SKILL.md) for agent instructions. `scripts/stt_cli.py` drives the whole workflow from a terminal, including video input and automatic chunking for long recordings.

Optional speaker labels (`--diarize` or `diarize=true`) run a local pyannote model. Set them up once with `scripts/setup-diarizer.sh` and put `HF_TOKEN` in `.env`, after accepting the model terms on Hugging Face. Labels are anonymous and their accuracy depends on the recording; see [backend/README.md](backend/README.md#speaker-labels-optional).

To transcribe recordings as they land in a folder, run `scripts/watch_folder.py WATCH_DIR --output-dir OUT_DIR` while the backend is running; it waits while the backend is down. On macOS, `scripts/watch_service/install.sh WATCH_DIR OUT_DIR` writes a launch agent that runs the watcher in the background. The script does not load the agent: it prints the `launchctl bootstrap` command to run. `status.sh` and `uninstall.sh` in the same folder show and remove it. See [SKILL.md](skills/stt-studio/SKILL.md#watch-a-folder) for details.

## Privacy and data handling

Inference runs on this machine; the project does not send recordings to a hosted transcription service. The backend writes uploads and decoded audio to local temporary files and removes them when processing finishes or fails normally. A forced termination can leave temporary files behind. Model weights may be downloaded from Hugging Face on first use. Backend jobs, batches and transcripts are kept in a local SQLite database in `data/` (or the folder named by `WHISPER_DATA_DIR`) until you delete them; there is no automatic expiry. Audio is never written there. A job that was queued or running when the server stopped is marked failed with "Interrupted by a server restart", and job ids stay valid across restarts. Projects are saved only when you choose to save them, in the current browser's storage for this local app address; clearing browser site data can remove them. Export transcripts separately for backup; exports do not include the audio or settings. Bind the service to `127.0.0.1` as shown. Authentication and protections for exposing the API to a network are not implemented.

Keep recordings, transcripts, logs, model weights, and virtual environments out of source control. The ignore rules cover common media, meeting files, outputs, logs, and model caches, but verify `git status` before sharing changes.

## Architecture

- `backend/app.py` implements the local FastAPI API, inference-adapter contract, serialized transcription worker, upload handling, and job status endpoints.
- `backend/tests/` contains API tests using a fake transcription model.
- `frontend/src/App.tsx` implements the single-workspace interface and client-side transcript editing/export.
- `models/catalog.json` and [`models/README.md`](models/README.md) document model families, verified runtime formats, source cards, and license metadata. The same directory is used as the local model cache; downloaded weights are ignored and must not be committed.

## Testing

Run the backend suite with:

```sh
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend/tests
```

Build the frontend with:

```sh
cd frontend
npm ci
npm run build
npx playwright install chromium
npm run test:e2e
```

An offline word error rate evaluator and calibration guidance are available in [evaluation/README.md](evaluation/README.md). Run its synthetic tests with `python -m unittest discover -s evaluation/tests`.

CI runs the backend and evaluation tests, frontend build, and browser workflow tests on pushes and pull requests.

## Current limits and planned work

The backend serializes work to one active transcription at a time, defaults to a two-hour audio limit, and shows partial segments only for faster-whisper, and progress is measured by segment for that runtime alone; the MLX runtime shows an indeterminate state. Long recordings can still require substantial memory; see [backend/README.md](backend/README.md) for resource limits. Browser-saved projects are not portable project files and do not resume backend jobs. Jobs, batches and transcripts persist until deleted, and a job running at a restart is marked failed rather than resumed. Live microphone capture and translation are not implemented.

## Attribution

STT Studio uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for inference with OpenAI's [Whisper](https://github.com/openai/whisper) model and [FFmpeg](https://ffmpeg.org/) for audio decoding. See upstream projects for their licenses and model terms.

## License

The project is licensed under the MIT License; see [LICENSE](LICENSE).
