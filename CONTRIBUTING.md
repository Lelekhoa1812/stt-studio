# Contributing

STT Studio is an early-stage project. Contributions should keep the local-first scope clear and avoid presenting planned work as available functionality.

## Before you start

- Check the existing implementation and open issues before starting a large change.
- Keep recordings, generated transcripts, logs, model files, and virtual environments out of commits.
- Do not add audio uploads to external services, telemetry, or remote inference without documenting the data flow and getting agreement from the project maintainers.

## Development setup

The backend uses FastAPI and `faster-whisper`:

```sh
python3.12 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements-dev.txt
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

The frontend uses Vite, React, and TypeScript:

```sh
cd frontend
npm ci
npm run dev
```

The model downloads on first use. See [README.md](README.md) for local data-handling details.

## Checks

Run the backend tests and build the frontend before opening a change:

```sh
python -m pytest backend/tests
cd frontend && npm ci && npm run build
```

Do not commit private recordings as test fixtures; use short synthetic or openly licensed samples where audio fixtures are necessary. Include clear reproduction steps for bug fixes and note platform, Python, model, and dependency versions when they affect the result.

## Pull requests

Describe the problem, the resulting behavior, how you checked it, and any known limits. Do not claim model accuracy, privacy guarantees, or supported features beyond what the implementation demonstrates.
