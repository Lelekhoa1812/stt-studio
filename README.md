# STT Studio

STT Studio is an early-stage local speech-to-text workspace for Mac. It combines a FastAPI service running `faster-whisper` with a Vite/React interface. Select an audio file, choose a model and transcription settings, run a local job, then review, edit, search, copy, or export the transcript. It is a development project, not a packaged or production-ready application.

The interface supports drag-and-drop or file browsing, audio playback, model loading and warmup, language and compute settings, vocabulary hints, job status, editable timestamped segments, transcript search, and TXT/SRT/JSON export. The runnable catalog includes Whisper size/language variants and compatible Distil-Whisper conversions, with support for custom CTranslate2 Whisper repositories. Other ASR families are documented as adapter candidates and are not presented as runnable yet. You can explicitly save and reopen projects in the browser, including their audio, edited transcript, and settings. Transcripts appear after inference completes; incremental token or percentage progress is not available from the current backend.

## Run locally

Requirements: Python 3.12, Node.js 22, npm, and `ffmpeg` and `ffprobe` available on `PATH`. The first model use downloads weights from Hugging Face.

In one terminal, install and start the backend from the repository root:

```sh
python3.12 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

In another terminal, install and start the frontend:

```sh
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite. The development proxy sends `/api` requests to the loopback backend on port 8765. Keep both processes running while using the interface.

The API exposes `GET /api/health`, `GET /api/models`, `POST /api/models/load`, `POST /api/jobs`, and `GET /api/jobs/{id}`. The default model is `small` with `int8` compute. See [backend/README.md](backend/README.md) for settings, API fields, limits, and error responses.

## Privacy and data handling

Inference runs on this machine; the project does not send recordings to a hosted transcription service. The backend writes uploads and decoded audio to local temporary files and removes them when processing finishes or fails normally. A forced termination can leave temporary files behind. Model weights may be downloaded from Hugging Face on first use. Backend jobs are held in memory and cleared on restart. Projects are saved only when you choose to save them, in the current browser's storage for this local app address; clearing browser site data can remove them. Export transcripts separately for backup; exports do not include the audio or settings. Bind the service to `127.0.0.1` as shown. Authentication and protections for exposing the API to a network are not implemented.

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

The backend serializes work to one active transcription at a time, defaults to a two-hour audio limit, and does not expose partial segments or measured progress. Long recordings can still require substantial memory; see [backend/README.md](backend/README.md) for resource limits. Browser-saved projects are not portable project files and do not resume backend jobs. Live microphone capture, speaker diarization, translation, and batch processing are not implemented.

## Attribution

STT Studio uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for inference with OpenAI's [Whisper](https://github.com/openai/whisper) model and [FFmpeg](https://ffmpeg.org/) for audio decoding. See upstream projects for their licenses and model terms.

## License

The project is licensed under the MIT License; see [LICENSE](LICENSE).
