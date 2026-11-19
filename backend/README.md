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

The API only listens on localhost in this command and rejects non-loopback client connections and non-loopback `Host` headers. Mutating requests carrying an `Origin` header are accepted only from localhost origins; no broad CORS policy is enabled. Health and model status snapshots stay responsive while the inference worker loads or runs a model. Treat local callers as trusted because the service can download and run model files and consume CPU and disk space. Upload files are limited to 2 GiB by default (`WHISPER_MAX_UPLOAD_BYTES`), audio is limited to 7,200 seconds by default (`WHISPER_MAX_AUDIO_SECONDS`), and one queued or running transcription is allowed (`WHISPER_MAX_ACTIVE_JOBS`). Before FastAPI parses multipart data, the request body is capped at the file limit plus 1 MiB for boundaries, headers, and form fields; change that allowance with `WHISPER_MULTIPART_OVERHEAD_BYTES` if needed. The route separately checks the exact uploaded file size. This cap is enforced from both `Content-Length` and streamed body chunks when that header is absent or inaccurate.

The server requires both `ffprobe` and `ffmpeg`. It probes duration before decoding or loading a model and rejects invalid media or files over the configured limit. ffmpeg writes 16 kHz mono float PCM to a temporary disk file that is read through a NumPy memory map, avoiding an in-memory PCM copy and unbounded captured stdout. Its output is capped at five seconds beyond the configured duration, then the decoded sample count is checked so an inaccurate probe cannot silently truncate a long recording. The temporary PCM is removed after success or failure. The duration cap also bounds faster-whisper's in-memory spectrogram work; a two-hour recording can still require several gigabytes of RAM during feature extraction. Lower `WHISPER_MAX_AUDIO_SECONDS` on machines with less memory. The server checks available disk space before decoding.

## API

- `GET /api/health` reports service status and active jobs.
- `GET /api/models` reports the default, supported compute types, locally cached models, and models loaded into memory.
- `POST /api/models/load` with JSON `{"model":"small","compute_type":"int8"}` downloads if needed, loads the model, and performs a one-second silent inference warmup. It runs in FastAPI's worker thread pool, so model loading does not block the async event loop.
- `POST /api/jobs` accepts multipart fields `file` (required), `model`, `compute_type`, `language`, `beam_size`, `vad_filter`, and `initial_prompt`. It responds `202` with `{ "id", "status": "queued" }`. Audio is decoded by local ffmpeg to 16 kHz mono PCM before inference, avoiding dependence on PyAV compatibility.
- `GET /api/jobs/{id}` returns job state. States are `queued`, `running`, `completed`, and `failed`. Completed jobs include `segments` (`start`, `end`, `text`, `avg_logprob`, `no_speech_prob`) and detected-language/duration `info`; failed jobs include a readable `error`.

Invalid job parameters return `422`, missing jobs `404`, uploads over the configured size limit `413`, busy job queue `429`, rejected remote connections/origins `403`, and model-load failures `503`. Audio inspection and decode errors appear on the job's `error` field. Jobs and transcripts live in memory and are not persisted across a server restart.

## Tests

```sh
python3 -m pip install -r backend/requirements-dev.txt
python3 -m pytest backend/tests
```
