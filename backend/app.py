"""Local FastAPI wrapper around faster-whisper."""
from __future__ import annotations

import importlib.util
import json
import os
import ipaddress
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

DEFAULT_MODEL = "small"
MODEL_CATALOG_PATH = Path(__file__).resolve().parent.parent / "models" / "catalog.json"
# What each runtime can accept. Family-level catalog entries may override only what differs
# (for example, Qwen3-ASR takes language names while Parakeet v3 is auto-detect only).
RUNTIME_CAPABILITIES: dict[str, dict[str, Any]] = {
    "faster-whisper": {"computeTypes": True, "beamSize": True, "vad": True,
                       "initialPrompt": True, "languageHint": "codes"},
    "mlx-audio": {"computeTypes": False, "beamSize": False, "vad": False,
                  "initialPrompt": False, "languageHint": "auto"},
}
# MLX-Audio language models expect English names; map the UI's ISO codes to them.
MLX_LANGUAGE_NAMES = {
    "zh": "Chinese", "yue": "Cantonese", "en": "English", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fa": "Persian",
    "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}
MLX_LANGUAGE_CODES = {name.lower(): code for code, name in MLX_LANGUAGE_NAMES.items()}


def model_capabilities(runtime: str, family_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    capabilities = dict(RUNTIME_CAPABILITIES.get(runtime, RUNTIME_CAPABILITIES["faster-whisper"]))
    capabilities.update(family_overrides or {})
    return capabilities
_MODEL_SIZES = {"tiny": "≈ 75 MB", "base": "≈ 145 MB", "small": "≈ 480 MB", "medium": "≈ 1.5 GB",
                "large-v1": "≈ 2.9 GB", "large-v2": "≈ 2.9 GB", "large-v3": "≈ 2.9 GB",
                "large-v3-turbo": "≈ 1.6 GB", "distil-small.en": "≈ 360 MB",
                "distil-medium.en": "≈ 800 MB", "distil-large-v2": "≈ 1.5 GB", "distil-large-v3": "≈ 1.5 GB"}


def _load_model_catalog() -> tuple[dict[str, Any], ...]:
    """Build the runnable model switcher from the source-linked catalog."""
    document = json.loads(MODEL_CATALOG_PATH.read_text(encoding="utf-8"))
    rows = []
    for family in document["families"]:
        for source in family.get("models", []):
            source_id = source["id"]
            # Keep faster-whisper's standard aliases, while custom CT2 repositories
            # are passed to the runtime by their exact repository ID.
            runtime_id = family.get("runtimeId", "faster-whisper")
            expected_repo = f"Systran/faster-whisper-{source_id}"
            model_id = source_id if runtime_id == "faster-whisper" and source.get("repository") == expected_repo else source["repository"]
            rows.append({"id": model_id, "name": source.get("label", source_id), "family": family["name"],
                         "size": _MODEL_SIZES.get(source_id, _MODEL_SIZES.get(source_id.removesuffix(".en"),
                                                                               source.get("downloadSize"))),
                         "repository": source["repository"],
                         "languages": source.get("languages"),
                         "description": family.get("runtimeNote"),
                         "publisher": source.get("repositoryPublisher", family.get("publisher")),
                         "publisher_slug": source.get("repositoryPublisherOrganization", family.get("publisherOrganization")),
                         "publisher_url": source.get("repositoryPublisherProfile", family.get("publisherProfile")),
                         "family_publisher": family.get("publisher"), "source_url": source.get("repositorySource"),
                         "license": source.get("license"), "runtime": runtime_id,
                         "runtime_status": family.get("runtimeStatus", "candidate"),
                         "supported": family.get("runtimeStatus") == "supported",
                         "capabilities": model_capabilities(runtime_id, family.get("capabilities"))})
    return tuple(rows)


MODEL_CATALOG = _load_model_catalog()
MODEL_CATALOG_BY_ID = {model["id"]: model for model in MODEL_CATALOG}
MODEL_ALIASES = {"large-v3-turbo": "dropbox-dash/faster-whisper-large-v3-turbo"}
KNOWN_MODELS = {model["id"] for model in MODEL_CATALOG} | set(MODEL_ALIASES)
COMPUTE_TYPES = {"int8", "int8_float16", "float16", "float32"}
MAX_UPLOAD_BYTES = int(os.environ.get("WHISPER_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
MAX_MULTIPART_OVERHEAD_BYTES = int(os.environ.get("WHISPER_MULTIPART_OVERHEAD_BYTES", str(1024 * 1024)))
MAX_ACTIVE_JOBS = int(os.environ.get("WHISPER_MAX_ACTIVE_JOBS", "1"))
MAX_AUDIO_SECONDS = float(os.environ.get("WHISPER_MAX_AUDIO_SECONDS", "7200"))
MODEL_CACHE = Path(os.environ.get("WHISPER_MODEL_CACHE", str(Path(__file__).parent.parent / "models")))


def _model_is_cached(repository: str) -> bool:
    snapshots = MODEL_CACHE / f"models--{repository.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    weight_suffixes = (".safetensors", ".gguf", ".onnx")
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir():
            continue
        for artifact in snapshot.iterdir():
            name = artifact.name
            if artifact.is_file() and (name == "model.bin" or name.endswith(weight_suffixes) or
                                       (name.startswith("pytorch_model") and name.endswith(".bin"))):
                return True
    return False


def _model_factory(model: str, compute_type: str):
    # Keep the native dependency lazy so API utilities and tests can load without it.
    from faster_whisper import WhisperModel

    device = os.environ.get("WHISPER_DEVICE", "cpu")
    return WhisperModel(MODEL_ALIASES.get(model, model), device=device, compute_type=compute_type,
                        download_root=str(MODEL_CACHE))


class InferenceAdapter(Protocol):
    """Runtime contract: load, warm, and return normalized transcript data."""

    runtime_id: str

    def load(self, model: str, compute_type: str) -> Any: ...
    def warmup(self, instance: Any) -> None: ...
    def transcribe(self, instance: Any, audio: Any, *, language: str | None, beam_size: int,
                   vad_filter: bool, initial_prompt: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


class FasterWhisperAdapter:
    runtime_id = "faster-whisper"

    def __init__(self, factory: Callable[[str, str], Any] = _model_factory):
        self.factory = factory

    def load(self, model: str, compute_type: str) -> Any:
        return self.factory(model, compute_type)

    def warmup(self, instance: Any) -> None:
        import numpy as np
        segments, _ = instance.transcribe(np.zeros(16000, dtype=np.float32), language="en",
                                           beam_size=1, vad_filter=False)
        list(segments)

    def transcribe(self, instance: Any, audio: Any, *, language: str | None, beam_size: int,
                   vad_filter: bool, initial_prompt: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        segments, info = instance.transcribe(audio, language=language, beam_size=beam_size,
                                             vad_filter=vad_filter, initial_prompt=initial_prompt)
        rows = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip(),
                 "avg_logprob": float(s.avg_logprob), "no_speech_prob": float(s.no_speech_prob)}
                for s in segments]
        metadata = {"language": getattr(info, "language", language),
                    "language_probability": getattr(info, "language_probability", None),
                    "duration": getattr(info, "duration", None)}
        return rows, metadata


def _mlx_audio_available() -> bool:
    """MLX-Audio is optional: it needs Apple Silicon and the mlx-audio package."""
    if os.environ.get("WHISPER_DISABLE_MLX") == "1":
        return False
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    return importlib.util.find_spec("mlx_audio") is not None


def _mlx_snapshot(repository: str) -> str:
    # Download into MODEL_CACHE so _model_is_cached sees MLX checkpoints like other runtimes.
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repository, cache_dir=str(MODEL_CACHE))


def _mlx_loader(path: str) -> Any:
    from mlx_audio.stt import load

    return load(path)


def _mlx_to_array(audio: Any) -> Any:
    import mlx.core as mx
    import numpy as np

    return mx.array(np.asarray(audio, dtype=np.float32))


def _mlx_rows(result: Any, duration: float) -> list[dict[str, Any]]:
    """Normalize MLX-Audio output (timed sentences, segments, or plain text) to job segments."""

    def row(start: Any, end: Any, text: Any) -> dict[str, Any]:
        return {"start": float(start), "end": float(end), "text": str(text).strip(),
                "avg_logprob": None, "no_speech_prob": None}

    sentences = [s for s in (getattr(result, "sentences", None) or []) if str(getattr(s, "text", "")).strip()]
    if sentences:
        return [row(s.start, s.end, s.text) for s in sentences]
    rows = []
    for segment in getattr(result, "segments", None) or []:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            rows.append(row(segment.get("start", segment.get("start_time", 0.0)),
                            segment.get("end", segment.get("end_time", duration)), text))
    if rows:
        return rows
    text = str(getattr(result, "text", "") or "").strip()
    return [row(0.0, duration, text)] if text else []


@dataclass
class _MlxInstance:
    repository: str
    model: Any
    language_hint: str


class MlxAudioAdapter:
    """Optional Apple Silicon runtime for MLX-Audio speech models (Parakeet, Qwen3-ASR).

    Whisper-only controls (compute type, beam size, VAD, vocabulary prompt) do not apply here
    and are ignored; the API reports them as unsupported through the runtime capabilities.
    """

    runtime_id = "mlx-audio"
    WARMUP_SAMPLES = 16000
    SAMPLE_RATE = 16000

    def __init__(self, snapshot: Callable[[str], str] = _mlx_snapshot,
                 loader: Callable[[str], Any] = _mlx_loader,
                 to_array: Callable[[Any], Any] = _mlx_to_array):
        self.snapshot = snapshot
        self.loader = loader
        self.to_array = to_array

    def load(self, model: str, compute_type: str) -> Any:
        del compute_type  # CTranslate2 quantization does not apply to MLX checkpoints.
        loaded = self.loader(self.snapshot(model))
        capabilities = model_capabilities(self.runtime_id, (MODEL_CATALOG_BY_ID.get(model) or {}).get("capabilities"))
        return _MlxInstance(repository=model, model=loaded, language_hint=capabilities["languageHint"])

    def warmup(self, instance: Any) -> None:
        import numpy as np

        instance.model.generate(self.to_array(np.zeros(self.WARMUP_SAMPLES, dtype=np.float32)))

    @staticmethod
    def _language_argument(instance: _MlxInstance, language: str | None) -> str | None:
        if instance.language_hint != "names" or not language or language.lower() == "auto":
            return None
        name = MLX_LANGUAGE_NAMES.get(language.lower())
        if name is None:
            raise ValueError(f"Language {language} is not supported by {instance.repository}")
        supported = getattr(getattr(instance.model, "config", None), "support_languages", None) or []
        if supported and name.lower() not in {entry.lower() for entry in supported}:
            raise ValueError(f"Language {name} is not supported by {instance.repository}")
        return name

    def transcribe(self, instance: Any, audio: Any, *, language: str | None, beam_size: int,
                   vad_filter: bool, initial_prompt: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del beam_size, vad_filter, initial_prompt  # Whisper-only controls do not apply to MLX models.
        kwargs = {}
        language_name = self._language_argument(instance, language)
        if language_name:
            kwargs["language"] = language_name
        duration = len(audio) / self.SAMPLE_RATE
        result = instance.model.generate(self.to_array(audio), **kwargs)
        detected = getattr(result, "language", None)
        if isinstance(detected, (list, tuple)):
            detected = detected[0] if detected else None
        if isinstance(detected, str) and detected:
            detected = MLX_LANGUAGE_CODES.get(detected.lower(), detected.lower())
        metadata = {"language": detected or language, "language_probability": None, "duration": duration}
        return _mlx_rows(result, duration), metadata


def _default_adapters(factory: Callable[[str, str], Any]) -> dict[str, InferenceAdapter]:
    adapters: dict[str, InferenceAdapter] = {FasterWhisperAdapter.runtime_id: FasterWhisperAdapter(factory)}
    if _mlx_audio_available():
        adapters[MlxAudioAdapter.runtime_id] = MlxAudioAdapter()
    return adapters


@dataclass
class Job:
    id: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    model: str = DEFAULT_MODEL
    runtime: str = "faster-whisper"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    initial_prompt: str | None = None
    segments: list[dict[str, Any]] | None = None
    info: dict[str, Any] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("id", "status", "created_at", "model", "runtime", "compute_type", "language", "beam_size", "vad_filter", "segments", "info", "error")}


class ModelManager:
    def __init__(self, factory: Callable[[str, str], Any] = _model_factory,
                 adapters: dict[str, InferenceAdapter] | None = None):
        self.adapters = adapters if adapters is not None else _default_adapters(factory)
        self._models: dict[tuple[str, str, str], Any] = {}
        self._state_lock = threading.Lock()
        self._operation_lock = threading.RLock()

    def get(self, model: str, compute_type: str, runtime: str = "faster-whisper"):
        adapter = self.adapters.get(runtime)
        if adapter is None:
            raise ValueError(f"Inference runtime is unavailable: {runtime}")
        key = (model, compute_type, runtime)
        with self._operation_lock:
            with self._state_lock:
                if key in self._models:
                    return self._models[key]
                # Drop the previous resident reference before allocating another model.
                self._models.clear()
            instance = adapter.load(model, compute_type)
            with self._state_lock:
                self._models[key] = instance
            return instance

    def load(self, model: str, compute_type: str, runtime: str = "faster-whisper") -> dict[str, Any]:
        # Keep the lock through replacement and warmup so another request cannot
        # evict this model while its native runtime is initializing.
        with self._operation_lock:
            adapter = self.adapters.get(runtime)
            if adapter is None:
                raise ValueError(f"Inference runtime is unavailable: {runtime}")
            instance = self.get(model, compute_type, runtime)
            adapter.warmup(instance)
            return {"model": model, "compute_type": compute_type, "loaded": True,
                    "runtime": runtime, "warmup": "complete", "device": os.environ.get("WHISPER_DEVICE", "cpu")}

    def transcribe(self, path: Path, *, model: str, compute_type: str, language: str | None,
                   beam_size: int, vad_filter: bool, initial_prompt: str | None,
                   runtime: str = "faster-whisper"):
        duration = _probe_audio_duration(path)
        if duration > MAX_AUDIO_SECONDS:
            raise RuntimeError(
                f"Audio is {duration:.0f} seconds; the configured limit is {MAX_AUDIO_SECONDS:.0f} seconds "
                "(WHISPER_MAX_AUDIO_SECONDS)"
            )

        # ffmpeg writes PCM to disk. A memmap avoids holding a second full PCM copy
        # in Python while faster-whisper builds its spectrogram.
        decode_limit_seconds = MAX_AUDIO_SECONDS + 5
        expected_bytes = int(min(duration, decode_limit_seconds) * 16000 * 4)
        free_bytes = shutil.disk_usage(tempfile.gettempdir()).free
        if free_bytes < expected_bytes + 64 * 1024 * 1024:
            raise RuntimeError("Not enough free disk space to decode this audio; free space and retry")
        try:
            temp_dir = tempfile.TemporaryDirectory(prefix="stt-pcm-")
        except OSError as exc:
            raise RuntimeError("Could not create temporary storage for audio decoding") from exc
        with temp_dir as temp_name:
            pcm_path = Path(temp_name) / "audio.f32"
            timeout = max(120, min(duration, decode_limit_seconds) * 3)
            try:
                with tempfile.TemporaryFile(mode="w+b") as error_log:
                    try:
                        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror",
                                        "-protocol_whitelist", "file,crypto,data", "-i", str(path),
                                        "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000",
                                        "-t", str(decode_limit_seconds), "-y", str(pcm_path)],
                                       stdout=subprocess.DEVNULL, stderr=error_log, check=True, timeout=timeout)
                    except FileNotFoundError as exc:
                        raise RuntimeError("ffmpeg is required to decode audio; install it and retry") from exc
                    except subprocess.TimeoutExpired as exc:
                        raise RuntimeError("Audio decoding timed out; try a shorter or simpler audio file") from exc
                    except subprocess.CalledProcessError as exc:
                        error_log.seek(max(0, error_log.tell() - 8192))
                        detail = error_log.read(8192).decode("utf-8", errors="replace").strip()
                        detail = detail.replace(str(path), "uploaded audio")
                        raise RuntimeError(f"Audio decoding failed: {detail or 'ffmpeg could not decode this file'}") from exc
            except OSError as exc:
                raise RuntimeError("Could not write temporary decoded audio; check available disk space") from exc
            try:
                pcm_bytes = pcm_path.stat().st_size
            except FileNotFoundError as exc:
                raise RuntimeError("ffmpeg did not produce decoded audio; check the input file") from exc
            if pcm_bytes == 0:
                raise RuntimeError("Audio decoder produced no samples; check the input file")
            if pcm_bytes % 4:
                raise RuntimeError("Audio decoder produced invalid PCM data")
            decoded_seconds = pcm_bytes / (16000 * 4)
            if decoded_seconds > MAX_AUDIO_SECONDS:
                raise RuntimeError(
                    f"Decoded audio exceeds the configured limit of {MAX_AUDIO_SECONDS:.0f} seconds "
                    "(WHISPER_MAX_AUDIO_SECONDS)"
                )
            import numpy as np
            try:
                audio = np.memmap(pcm_path, dtype=np.float32, mode="r", shape=(pcm_bytes // 4,))
            except (OSError, ValueError) as exc:
                raise RuntimeError("Could not read the temporary decoded audio; free disk space and retry") from exc
            try:
                with self._operation_lock:
                    adapter = self.adapters.get(runtime)
                    if adapter is None:
                        raise ValueError(f"Inference runtime is unavailable: {runtime}")
                    instance = self.get(model, compute_type, runtime)
                    rows, meta = adapter.transcribe(
                        instance, audio, language=language, beam_size=beam_size,
                        vad_filter=vad_filter, initial_prompt=initial_prompt,
                    )
            finally:
                audio._mmap.close()
        return rows, meta

    def snapshot(self) -> list[dict[str, str]]:
        with self._state_lock:
            return [{"model": model, "compute_type": compute_type, "runtime": runtime}
                    for model, compute_type, runtime in self._models]


def _public_error(exc: Exception) -> str:
    """Keep useful diagnostics without returning local server paths."""
    message = str(exc).replace(str(MODEL_CACHE), "<model cache>")
    message = re.sub(r"/(?:private/)?tmp/whisper-upload-[^\s:'\"]+", "<uploaded audio>", message)
    return message[:1000] or exc.__class__.__name__


def _probe_audio_duration(path: Path) -> float:
    """Read container duration before decoding the full waveform."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-protocol_whitelist", "file,crypto,data",
             "-select_streams", "a:0", "-show_entries",
             "format=duration:stream=duration", "-of", "json", str(path)],
            capture_output=True, check=True, timeout=20,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to inspect audio; install ffmpeg and retry") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Audio inspection timed out; check the input file") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        detail = detail.replace(str(path), "uploaded audio")
        raise RuntimeError(f"Could not inspect audio: {detail or 'ffprobe could not read this file'}") from exc
    try:
        metadata = json.loads(result.stdout)
        value = metadata.get("format", {}).get("duration")
        if value in (None, "N/A") and metadata.get("streams"):
            value = metadata["streams"][0].get("duration")
        duration = float(value)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not determine audio duration; use a valid audio file") from exc
    if not (duration > 0):
        raise RuntimeError("Audio duration must be greater than zero")
    return duration


class LocalAndBoundedUploadMiddleware:
    """Enforce loopback access and cap request bodies before multipart parsing."""

    def __init__(self, app):
        self.app = app

    @staticmethod
    async def _reject(send, status: int, detail: str):
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode("ascii"))]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        peer = (scope.get("client") or ("", 0))[0]
        try:
            if not ipaddress.ip_address(peer).is_loopback:
                await self._reject(send, 403, "This API only accepts local connections")
                return
        except ValueError:
            await self._reject(send, 403, "This API only accepts local connections")
            return

        from urllib.parse import urlsplit
        try:
            hostname = (urlsplit("//" + headers.get(b"host", b"").decode("latin-1")).hostname or "").lower()
            local_host = hostname == "localhost" or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            local_host = False
        if not local_host:
            await self._reject(send, 403, "Host must be localhost or a loopback address")
            return

        if scope.get("method") in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = headers.get(b"origin")
            if origin:
                try:
                    origin_host = (urlsplit(origin.decode("latin-1")).hostname or "").lower()
                except ValueError:
                    origin_host = ""
                if origin_host not in {"localhost", "127.0.0.1", "::1"}:
                    await self._reject(send, 403, "Mutations are only allowed from a local origin")
                    return

        is_upload = (scope.get("method") == "POST" and scope.get("path") == "/api/jobs"
                     and headers.get(b"content-type", b"").lower().startswith(b"multipart/form-data"))
        if is_upload:
            request_limit = MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
            declared = headers.get(b"content-length")
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError:
                    await self._reject(send, 400, "Invalid Content-Length header")
                    return
                if declared_size > request_limit:
                    await self._reject(send, 413, f"Multipart request exceeds {request_limit} byte limit")
                    return
            received = 0

            async def bounded_receive():
                nonlocal received
                message = await receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > request_limit:
                        raise HTTPException(413, f"Multipart request exceeds {request_limit} byte limit")
                return message

            await self.app(scope, bounded_receive, send)
            return

        await self.app(scope, receive, send)


class LoadRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    compute_type: str = "int8"
    runtime: str | None = None


def _validate_model_settings(model: str, compute_type: str, beam_size: int, runtime: str | None = None,
                             available_runtimes: set[str] | None = None) -> tuple[str, str]:
    """Return the (runtime, compute_type) pair the runtime will actually use."""
    if not model.strip() or len(model) > 200:
        raise HTTPException(422, "model must be a non-empty model name (max 200 characters)")
    if model not in KNOWN_MODELS and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model):
        raise HTTPException(422, "model must be a supported model name or Hugging Face owner/model identifier")
    if any(part in {".", ".."} for part in model.split("/")):
        raise HTTPException(422, "model identifier is invalid")
    if not 1 <= beam_size <= 20:
        raise HTTPException(422, "beam_size must be between 1 and 20")
    catalog_id = MODEL_ALIASES.get(model, model)
    catalog_model = MODEL_CATALOG_BY_ID.get(catalog_id)
    if catalog_model and not catalog_model["supported"]:
        raise HTTPException(422, f"Model family is catalogued but not supported yet: {catalog_model['family']}")
    expected_runtime = catalog_model.get("runtime") if catalog_model else None
    runtime = runtime or expected_runtime or "faster-whisper"
    if expected_runtime and runtime != expected_runtime:
        raise HTTPException(422, f"Model {model} requires the {expected_runtime} runtime")
    if available_runtimes is None:
        available_runtimes = {FasterWhisperAdapter.runtime_id}
    if runtime not in available_runtimes:
        raise HTTPException(422, f"Inference runtime is not installed: {runtime}")
    capabilities = model_capabilities(runtime, catalog_model.get("capabilities") if catalog_model else None)
    if capabilities["computeTypes"]:
        if compute_type not in COMPUTE_TYPES:
            raise HTTPException(422, f"compute_type must be one of: {', '.join(sorted(COMPUTE_TYPES))}")
    else:
        # CTranslate2 quantization does not apply; keep a stable sentinel for job and cache keys.
        compute_type = "native"
    return runtime, compute_type


def create_app(manager: ModelManager | None = None) -> FastAPI:
    api = FastAPI(title="STT Studio Local API", version="1.0.0")
    api.add_middleware(LocalAndBoundedUploadMiddleware)
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
        available_runtimes = set(getattr(api.state.models, "adapters", {"faster-whisper": None}))
        available_models = [model for model in MODEL_CATALOG
                            if model["supported"] and model["runtime"] in available_runtimes]
        return {"default_model": DEFAULT_MODEL, "compute_types": sorted(COMPUTE_TYPES),
                "models": [{**model, "cached": _model_is_cached(model["repository"])} for model in available_models],
                "cached_models": cached, "loaded_models": loaded,
                "runtimes": sorted(getattr(api.state.models, "adapters", {"faster-whisper": None})),
                "runtime_capabilities": {name: model_capabilities(name) for name in sorted(available_runtimes)},
                "custom_model_runtime": "CTranslate2 / faster-whisper",
                "custom_model_help": "Custom Hugging Face repositories must contain a compatible CTranslate2 Whisper model."}

    @api.post("/api/models/load")
    def load_model(request: LoadRequest):
        runtimes = set(getattr(api.state.models, "adapters", {"faster-whisper": None}))
        runtime, compute_type = _validate_model_settings(request.model, request.compute_type, 1, request.runtime,
                                                         available_runtimes=runtimes)
        try:
            return api.state.models.load(request.model, compute_type, runtime)
        except Exception as exc:
            raise HTTPException(503, f"Could not load or warm up model: {_public_error(exc)}") from exc

    def run_job(job: Job, path: Path):
        job.status = "running"
        try:
            rows, info = api.state.models.transcribe(
                path, model=job.model, compute_type=job.compute_type, language=job.language,
                beam_size=job.beam_size, vad_filter=job.vad_filter, initial_prompt=job.initial_prompt,
                runtime=job.runtime)
            job.segments, job.info, job.status = rows, info, "completed"
        except Exception as exc:
            job.error = f"Transcription failed: {_public_error(exc)}"
            job.status = "failed"
        finally:
            path.unlink(missing_ok=True)

    @api.post("/api/jobs", status_code=202)
    async def create_job(file: UploadFile = File(...),
                         model: str = Form(DEFAULT_MODEL), compute_type: str = Form("int8"),
                         runtime: str | None = Form(None),
                         language: str | None = Form(None), beam_size: int = Form(5),
                         vad_filter: bool = Form(True), initial_prompt: str | None = Form(None)):
        runtimes = set(getattr(api.state.models, "adapters", {"faster-whisper": None}))
        runtime, compute_type = _validate_model_settings(model, compute_type, beam_size, runtime,
                                                         available_runtimes=runtimes)
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
        job = Job(id=str(uuid.uuid4()), model=model, runtime=runtime, compute_type=compute_type, language=language,
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
