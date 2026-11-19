"""Local FastAPI wrapper around faster-whisper."""
from __future__ import annotations

import os
import ipaddress
import re
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

DEFAULT_MODEL = "small"
KNOWN_MODELS = {"tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
                "large-v1", "large-v2", "large-v3", "large-v3-turbo"}
COMPUTE_TYPES = {"int8", "int8_float16", "float16", "float32"}
MAX_UPLOAD_BYTES = int(os.environ.get("WHISPER_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
MAX_ACTIVE_JOBS = int(os.environ.get("WHISPER_MAX_ACTIVE_JOBS", "1"))
MODEL_CACHE = Path(os.environ.get("WHISPER_MODEL_CACHE", str(Path(__file__).parent.parent / "models")))


def _model_factory(model: str, compute_type: str):
    # Keep the native dependency lazy so API utilities and tests can load without it.
    from faster_whisper import WhisperModel

    device = os.environ.get("WHISPER_DEVICE", "cpu")
    return WhisperModel(model, device=device, compute_type=compute_type, download_root=str(MODEL_CACHE))


@dataclass
class Job:
    id: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    model: str = DEFAULT_MODEL
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    initial_prompt: str | None = None
    segments: list[dict[str, Any]] | None = None
    info: dict[str, Any] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("id", "status", "created_at", "model", "compute_type", "language", "beam_size", "vad_filter", "segments", "info", "error")}


class ModelManager:
    def __init__(self, factory: Callable[[str, str], Any] = _model_factory):
        self.factory = factory
        self._models: dict[tuple[str, str], Any] = {}
        self._state_lock = threading.Lock()
        self._operation_lock = threading.RLock()

    def get(self, model: str, compute_type: str):
        key = (model, compute_type)
        with self._operation_lock:
            with self._state_lock:
                if key in self._models:
                    return self._models[key]
                # Drop the previous resident reference before allocating another model.
                self._models.clear()
            instance = self.factory(model, compute_type)
            with self._state_lock:
                self._models[key] = instance
            return instance

    def load(self, model: str, compute_type: str) -> dict[str, Any]:
        # Keep the lock through replacement and warmup so another request cannot
        # evict this model while its native runtime is initializing.
        with self._operation_lock:
            instance = self.get(model, compute_type)
            import numpy as np
            segments, _ = instance.transcribe(np.zeros(16000, dtype=np.float32), language="en",
                                               beam_size=1, vad_filter=False)
            list(segments)
            return {"model": model, "compute_type": compute_type, "loaded": True,
                    "warmup": "complete", "device": os.environ.get("WHISPER_DEVICE", "cpu")}

    def transcribe(self, path: Path, *, model: str, compute_type: str, language: str | None,
                   beam_size: int, vad_filter: bool, initial_prompt: str | None):
        # Decode with ffmpeg so operation does not depend on PyAV's compatibility with
        # the installed faster-whisper/CTranslate2 combination.
        try:
            decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
                                      "-ac", "1", "-ar", "16000", "-"],
                                     capture_output=True, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to decode audio; install it and retry") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Audio decoding failed: {detail or 'ffmpeg could not decode this file'}") from exc
        import numpy as np
        audio = np.frombuffer(decoded.stdout, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("Audio decoder produced no samples; check the input file")
        with self._operation_lock:
            instance = self.get(model, compute_type)
            segments, info = instance.transcribe(
                audio, language=language, beam_size=beam_size,
                vad_filter=vad_filter, initial_prompt=initial_prompt,
            )
            rows = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip(),
                     "avg_logprob": float(s.avg_logprob), "no_speech_prob": float(s.no_speech_prob)}
                    for s in segments]
        meta = {"language": getattr(info, "language", language),
                "language_probability": getattr(info, "language_probability", None),
                "duration": getattr(info, "duration", None)}
        return rows, meta

    def snapshot(self) -> list[dict[str, str]]:
        with self._state_lock:
            return [{"model": model, "compute_type": compute_type}
                    for model, compute_type in self._models]


def _public_error(exc: Exception) -> str:
    """Keep useful diagnostics without returning local server paths."""
    message = str(exc).replace(str(MODEL_CACHE), "<model cache>")
    message = re.sub(r"/(?:private/)?tmp/whisper-upload-[^\s:'\"]+", "<uploaded audio>", message)
    return message[:1000] or exc.__class__.__name__


class LoadRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    compute_type: str = "int8"


def _validate_model_settings(model: str, compute_type: str, beam_size: int) -> None:
    if not model.strip() or len(model) > 200:
        raise HTTPException(422, "model must be a non-empty model name (max 200 characters)")
    if model not in KNOWN_MODELS and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model):
        raise HTTPException(422, "model must be a supported model name or Hugging Face owner/model identifier")
    if any(part in {".", ".."} for part in model.split("/")):
        raise HTTPException(422, "model identifier is invalid")
    if compute_type not in COMPUTE_TYPES:
        raise HTTPException(422, f"compute_type must be one of: {', '.join(sorted(COMPUTE_TYPES))}")
    if not 1 <= beam_size <= 20:
        raise HTTPException(422, "beam_size must be between 1 and 20")


def create_app(manager: ModelManager | None = None) -> FastAPI:
    api = FastAPI(title="STT Studio Local API", version="1.0.0")

    class LocalOriginGuard(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            peer = request.client.host if request.client else ""
            try:
                if not ipaddress.ip_address(peer).is_loopback:
                    from fastapi.responses import JSONResponse
                    return JSONResponse({"detail": "This API only accepts local connections"}, status_code=403)
            except ValueError:
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "This API only accepts local connections"}, status_code=403)
            from urllib.parse import urlsplit
            host_header = request.headers.get("host", "")
            try:
                hostname = (urlsplit("//" + host_header).hostname or "").lower()
                local_host = hostname == "localhost" or ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                local_host = False
            if not local_host:
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Host must be localhost or a loopback address"}, status_code=403)
            if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
                origin = request.headers.get("origin")
                if origin:
                    host = (urlsplit(origin).hostname or "").lower()
                    if host not in {"localhost", "127.0.0.1", "::1"}:
                        from fastapi.responses import JSONResponse
                        return JSONResponse({"detail": "Mutations are only allowed from a local origin"}, status_code=403)
            return await call_next(request)

    api.add_middleware(LocalOriginGuard)
    api.state.models = manager or ModelManager()
    api.state.jobs = {}
    api.state.jobs_lock = threading.Lock()
    api.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-job")

    @api.get("/api/health")
    def health():
        with api.state.jobs_lock:
            active = sum(j.status in ("queued", "running") for j in api.state.jobs.values())
        return {"status": "ok", "active_jobs": active, "loaded_models": len(api.state.models.snapshot())}

    @api.get("/api/models")
    def models():
        cached = sorted({p.name for p in MODEL_CACHE.glob("models--*") if p.is_dir()}) if MODEL_CACHE.exists() else []
        loaded = api.state.models.snapshot()
        return {"default_model": DEFAULT_MODEL, "compute_types": sorted(COMPUTE_TYPES),
                "cached_models": cached, "loaded_models": loaded}

    @api.post("/api/models/load")
    def load_model(request: LoadRequest):
        _validate_model_settings(request.model, request.compute_type, 1)
        try:
            return api.state.models.load(request.model, request.compute_type)
        except Exception as exc:
            raise HTTPException(503, f"Could not load or warm up model: {_public_error(exc)}") from exc

    def run_job(job: Job, path: Path):
        job.status = "running"
        try:
            rows, info = api.state.models.transcribe(
                path, model=job.model, compute_type=job.compute_type, language=job.language,
                beam_size=job.beam_size, vad_filter=job.vad_filter, initial_prompt=job.initial_prompt)
            job.segments, job.info, job.status = rows, info, "completed"
        except Exception as exc:
            job.error = f"Transcription failed: {_public_error(exc)}"
            job.status = "failed"
        finally:
            path.unlink(missing_ok=True)

    @api.post("/api/jobs", status_code=202)
    async def create_job(file: UploadFile = File(...),
                         model: str = Form(DEFAULT_MODEL), compute_type: str = Form("int8"),
                         language: str | None = Form(None), beam_size: int = Form(5),
                         vad_filter: bool = Form(True), initial_prompt: str | None = Form(None)):
        _validate_model_settings(model, compute_type, beam_size)
        if language is not None and len(language) > 20:
            raise HTTPException(422, "language must be at most 20 characters")
        suffix = Path(file.filename or "audio").suffix[:16] or ".audio"
        fd, filename = tempfile.mkstemp(prefix="whisper-upload-", suffix=suffix)
        path = Path(filename)
        total = 0
        try:
            with os.fdopen(fd, "wb") as target:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES} byte limit")
                    target.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        if total == 0:
            path.unlink(missing_ok=True)
            raise HTTPException(422, "Uploaded audio file is empty")
        job = Job(id=str(uuid.uuid4()), model=model, compute_type=compute_type, language=language,
                  beam_size=beam_size, vad_filter=vad_filter, initial_prompt=initial_prompt)
        with api.state.jobs_lock:
            active = sum(j.status in ("queued", "running") for j in api.state.jobs.values())
            if active >= MAX_ACTIVE_JOBS:
                path.unlink(missing_ok=True)
                raise HTTPException(429, f"At most {MAX_ACTIVE_JOBS} transcription job(s) may be active")
            api.state.jobs[job.id] = job
        api.state.executor.submit(run_job, job, path)
        return {"id": job.id, "status": "queued"}

    @api.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        with api.state.jobs_lock:
            job = api.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job.as_dict()

    return api


app = create_app()
