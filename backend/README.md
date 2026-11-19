# Local Whisper API

This service runs faster-whisper on the same Mac as its caller. Audio is written to a temporary file, passed to the local inference model, and deleted when the job finishes or fails. The API does not upload audio to a hosted service. Model downloads come from Hugging Face when a model is not already cached; the cache defaults to the repository's `models/` directory.

The default is `small` with `int8` compute on CPU, which keeps the first download and memory use modest. You can select a supported standard model name or a Hugging Face `owner/model` identifier and compute type (`int8`, `int8_float16`, `float16`, or `float32`). The default language is automatic detection; pass a language code to force one. Set `WHISPER_DEVICE` to change the CTranslate2 device. Device support depends on the installed CTranslate2 build and hardware. Only one model is kept resident at a time to bound memory use.

## Run

From the repository root, install the backend requirements and start the server bound to loopback:

```sh
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

The API only listens on localhost in this command and rejects non-loopback client connections and non-loopback `Host` headers. Mutating requests carrying an `Origin` header are accepted only from localhost origins; no broad CORS policy is enabled. Health and model status snapshots stay responsive while the inference worker loads or runs a model. Treat local callers as trusted because the service can download and run model files and consume CPU and disk space. Uploads are limited to 2 GiB by default (`WHISPER_MAX_UPLOAD_BYTES`), and one queued or running transcription is allowed (`WHISPER_MAX_ACTIVE_JOBS`).

## API

- `GET /api/health` reports service status and active jobs.
- `GET /api/models` reports the default, supported compute types, locally cached models, and models loaded into memory.
- `POST /api/models/load` with JSON `{"model":"small","compute_type":"int8"}` downloads if needed, loads the model, and performs a one-second silent inference warmup. It runs in FastAPI's worker thread pool, so model loading does not block the async event loop.
- `POST /api/jobs` accepts multipart fields `file` (required), `model`, `compute_type`, `language`, `beam_size`, `vad_filter`, and `initial_prompt`. It responds `202` with `{ "id", "status": "queued" }`. Audio is decoded by local ffmpeg to 16 kHz mono PCM before inference, avoiding dependence on PyAV compatibility.
- `GET /api/jobs/{id}` returns job state. States are `queued`, `running`, `completed`, and `failed`. Completed jobs include `segments` (`start`, `end`, `text`, `avg_logprob`, `no_speech_prob`) and detected-language/duration `info`; failed jobs include a readable `error`.

Invalid input returns `422`, missing jobs `404`, uploads over the configured limit `413`, busy job queue `429`, rejected remote origins `403`, and model-load failures `503`. Jobs and transcripts live in memory and are not persisted across a server restart.

## Tests

```sh
python3 -m pip install -r backend/requirements-dev.txt
python3 -m pytest backend/tests
```
