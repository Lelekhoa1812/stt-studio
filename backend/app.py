"""Local FastAPI wrapper around faster-whisper."""
from __future__ import annotations

import errno
import fcntl
import asyncio
import gc
import importlib.util
import json
import logging
import math
import os
import ipaddress
import platform
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from backend import diarization, exporters, nim

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "small"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
MODEL_CATALOG_PATH = Path(__file__).resolve().parent.parent / "models" / "catalog.json"
# What each runtime can accept. Family-level catalog entries may override only what differs
# (for example, Qwen3-ASR takes language names while Parakeet v3 is auto-detect only).
RUNTIME_CAPABILITIES: dict[str, dict[str, Any]] = {
    "faster-whisper": {"computeTypes": True, "beamSize": True, "vad": True,
                       "initialPrompt": True, "languageHint": "codes", "timedSegments": True},
    "mlx-audio": {"computeTypes": False, "beamSize": False, "vad": False,
                  "initialPrompt": False, "languageHint": "auto", "timedSegments": False},
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
                         "capabilities": model_capabilities(runtime_id, family.get("capabilities")),
                         # Per-model controls and outputs as the runtime actually applies them.
                         "controls": source.get("controls"), "emits": source.get("emits"),
                         "cancellation": source.get("cancellation")})
    return tuple(rows)


MODEL_CATALOG = _load_model_catalog()
MODEL_CATALOG_BY_ID = {model["id"]: model for model in MODEL_CATALOG}
MODEL_ALIASES = {"large-v3-turbo": "dropbox-dash/faster-whisper-large-v3-turbo"}
KNOWN_MODELS = {model["id"] for model in MODEL_CATALOG} | set(MODEL_ALIASES)
COMPUTE_TYPES = {"int8", "int8_float16", "float16", "float32"}
# Language codes faster-whisper accepts (its tokenizer table). A constant keeps faster_whisper out of import time.
WHISPER_LANGUAGE_CODES = frozenset({
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da",
    "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi",
    "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
    "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn",
    "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
    "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
    "yi", "yo", "zh", "yue",
})
MAX_INITIAL_PROMPT_CHARS = 1000
DEFAULT_BEAM_SIZE = 5
# The data protocol can smuggle another input. Allow only local files.
_MEDIA_PROTOCOL_WHITELIST = "file,crypto"
# ffprobe stdout is stopped at this size. A longer stream is refused and is not kept.
_FFPROBE_STDOUT_LIMIT = 1024 * 1024
def _env_int(name: str, default: str) -> int:
    """Read a count or byte limit from the environment. A bad value stops startup with the variable's name."""
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{name} must be a whole number of at least 1; got {raw!r}")
    return value


def _env_seconds(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if not (value > 0 and value != float("inf")):
        raise ValueError(f"{name} must be a positive number of seconds; got {raw!r}")
    return value


MAX_UPLOAD_BYTES = _env_int("WHISPER_MAX_UPLOAD_BYTES", str(2 * 1024**3))
MAX_MULTIPART_OVERHEAD_BYTES = _env_int("WHISPER_MULTIPART_OVERHEAD_BYTES", str(1024 * 1024))
MAX_ACTIVE_JOBS = _env_int("WHISPER_MAX_ACTIVE_JOBS", "1")
MAX_BATCH_FILES = _env_int("WHISPER_MAX_BATCH_FILES", "25")
MAX_BATCH_UPLOAD_BYTES = _env_int("WHISPER_MAX_BATCH_UPLOAD_BYTES", str(8 * 1024**3))
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
MAX_AUDIO_SECONDS = _env_seconds("WHISPER_MAX_AUDIO_SECONDS", "7200")
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


def _cache_folder(repository: str) -> Path:
    """The Hugging Face cache folder for a repository. Refuses any path that resolves outside MODEL_CACHE."""
    root = MODEL_CACHE.resolve()
    name = f"models--{repository.replace('/', '--')}"
    folder = (root / name).resolve()
    if folder.parent != root or folder.name != name:
        raise ValueError("Refusing to touch a path outside the model cache")
    return folder


def _folder_bytes(folder: Path) -> int:
    """Bytes stored under a cache folder. Symlinks (snapshot entries) are skipped; the blobs they point to are counted."""
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            path = Path(root, name)
            if not path.is_symlink():
                total += _file_size(path)
    return total


def _remove_model_folder(folder: Path) -> tuple[int, bool]:
    """Delete one cache folder. Returns the bytes it held and whether anything was there."""
    if not folder.is_dir():
        return 0, False
    freed = _folder_bytes(folder)
    try:
        shutil.rmtree(folder)
    except OSError:
        logger.exception("Could not remove a cached model folder")
        raise HTTPException(500, "Could not remove the cached model files; check file permissions") from None
    return freed, True


def _model_factory(model: str, compute_type: str):
    # Keep the native dependency lazy so API utilities and tests can load without it.
    from faster_whisper import WhisperModel

    device = os.environ.get("WHISPER_DEVICE", "cpu")
    return WhisperModel(MODEL_ALIASES.get(model, model), device=device, compute_type=compute_type,
                        download_root=str(MODEL_CACHE))


class JobCancelled(Exception):
    """Raised inside a running transcription when the user cancels the job."""


class JobControl:
    """Cooperative cancellation and segment reporting for one job.

    Adapters call check() between units of work and report() for each finished segment.
    Cancellation takes effect at the next check, so a segment already being decoded finishes first.
    finish_part() runs after each transcribed part so the partial transcript can be saved.
    """

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self.on_segment: Callable[[dict[str, Any], float], None] | None = None
        self.on_part: Callable[[], None] | None = None

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def check(self) -> None:
        if self._cancelled.is_set():
            raise JobCancelled("Transcription was cancelled")

    def report(self, row: dict[str, Any], duration: float) -> None:
        if self.on_segment is not None:
            self.on_segment(row, duration)

    def finish_part(self) -> None:
        if self.on_part is not None:
            self.on_part()


def _note_transcribed_part(control: JobControl | None) -> None:
    if control is not None:
        control.finish_part()


class _TimelineControl:
    """Report one part's segments on the full recording so progress does not restart at each part."""

    def __init__(self, control: JobControl | None, offset: float, total_duration: float):
        self._control = control
        self._offset = offset
        self._total_duration = total_duration

    def check(self) -> None:
        if self._control is not None:
            self._control.check()

    def report(self, row: dict[str, Any], duration: float) -> None:
        del duration
        if self._control is None:
            return
        text = str(row.get("text") or "").strip()
        if not text:
            return
        shifted = {**row, "start": float(row["start"]) + self._offset,
                   "end": float(row["end"]) + self._offset, "text": text}
        # A part-relative end would make progress fall back to 0 when the next part starts.
        self._control.report(shifted, self._total_duration)

    @property
    def cancelled(self) -> bool:
        return self._control is not None and self._control.cancelled


class InferenceAdapter(Protocol):
    """Runtime contract: load, warm, and return normalized transcript data."""

    runtime_id: str

    def load(self, model: str, compute_type: str) -> Any: ...
    def warmup(self, instance: Any) -> None: ...
    def transcribe(self, instance: Any, audio: Any, *, language: str | None, beam_size: int,
                   vad_filter: bool, initial_prompt: str | None,
                   control: JobControl | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


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
                   vad_filter: bool, initial_prompt: str | None,
                   control: JobControl | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        segments, info = instance.transcribe(audio, language=language, beam_size=beam_size,
                                             vad_filter=vad_filter, initial_prompt=initial_prompt)
        duration = float(getattr(info, "duration", 0) or 0)
        rows: list[dict[str, Any]] = []
        for s in segments:
            if control is not None:
                control.check()
            row = {"start": float(s.start), "end": float(s.end), "text": s.text.strip(),
                   "avg_logprob": float(s.avg_logprob), "no_speech_prob": float(s.no_speech_prob)}
            rows.append(row)
            if control is not None:
                control.report(row, duration)
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
    device = "metal"  # MLX runs on Apple's Metal GPU backend
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
                   vad_filter: bool, initial_prompt: str | None,
                   control: JobControl | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del beam_size, vad_filter, initial_prompt  # Whisper-only controls do not apply to MLX models.
        kwargs = {}
        language_name = self._language_argument(instance, language)
        if language_name:
            kwargs["language"] = language_name
        duration = len(audio) / self.SAMPLE_RATE
        if control is not None:
            control.check()
        # generate() is a single call with no per-segment hook, so cancellation can only be
        # honoured before and after it. Progress is reported as indeterminate for this runtime.
        result = instance.model.generate(self.to_array(audio), **kwargs)
        if control is not None:
            control.check()
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
    # None when the runtime ignores the control; the job row stores NULL in that case.
    beam_size: int | None = DEFAULT_BEAM_SIZE
    vad_filter: bool | None = True
    initial_prompt: str | None = None
    # Whisper-only controls the runtime accepted but ignored, by request field name.
    ignored_controls: list[str] = field(default_factory=list)
    segments: list[dict[str, Any]] | None = None
    info: dict[str, Any] | None = None
    error: str | None = None
    # Measured percent (0-100) when the runtime reports segment timings; None means indeterminate.
    progress: float | None = None
    cancel_requested: bool = False
    filename: str | None = None
    batch_id: str | None = None
    diarize: bool = False
    ai_results: list[dict[str, Any]] = field(default_factory=list)
    control: JobControl = field(default_factory=JobControl, repr=False)

    def as_dict(self) -> dict[str, Any]:
        keys = ("id", "status", "created_at", "model", "runtime", "compute_type", "language", "beam_size",
                "vad_filter", "segments", "info", "error", "progress", "cancel_requested", "filename", "batch_id",
                "diarize", "ai_results", "ignored_controls")
        data = {k: getattr(self, k) for k in keys}
        # Copy partial segments so the response never shares a list the worker is still appending to.
        data["segments"] = list(data["segments"] or [])
        data["ai_results"] = list(data["ai_results"] or [])
        data["ignored_controls"] = list(data["ignored_controls"] or [])
        return data


STORAGE_UNAVAILABLE = "Local job storage is unavailable; check disk space and file permissions"
INTERRUPTED_MESSAGE = "Interrupted by a server restart"
DATABASE_NAME = "stt-studio.sqlite"
SCHEMA_VERSION = 3
SCHEMA_STATEMENTS = (
    "CREATE TABLE batches (id TEXT PRIMARY KEY, created_at REAL NOT NULL, job_ids TEXT NOT NULL)",
    "CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at REAL NOT NULL, model TEXT NOT NULL, "
    "runtime TEXT NOT NULL, compute_type TEXT NOT NULL, language TEXT, beam_size INTEGER NOT NULL, "
    "vad_filter INTEGER NOT NULL, segments TEXT NOT NULL, info TEXT NOT NULL, error TEXT, progress REAL, "
    "cancel_requested INTEGER NOT NULL, filename TEXT, batch_id TEXT, diarize INTEGER NOT NULL)",
    "CREATE INDEX jobs_created_at ON jobs (created_at)",
)
# Version 1 is SCHEMA_STATEMENTS. Each later version lists the statements that move the previous version forward.
SCHEMA_MIGRATIONS = {
    2: ("ALTER TABLE jobs ADD COLUMN ai_results TEXT NOT NULL DEFAULT '[]'",),
    # Runtimes that ignore beam size or VAD store NULL there, which the version 1 NOT NULL columns refuse.
    # SQLite cannot drop NOT NULL in place, so the table is rebuilt and its rows copied across.
    3: (
        "CREATE TABLE jobs_v3 (id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at REAL NOT NULL, "
        "model TEXT NOT NULL, runtime TEXT NOT NULL, compute_type TEXT NOT NULL, language TEXT, "
        "beam_size INTEGER, vad_filter INTEGER, segments TEXT NOT NULL, info TEXT NOT NULL, error TEXT, "
        "progress REAL, cancel_requested INTEGER NOT NULL, filename TEXT, batch_id TEXT, diarize INTEGER NOT NULL, "
        "ai_results TEXT NOT NULL DEFAULT '[]', ignored_controls TEXT NOT NULL DEFAULT '[]')",
        "INSERT INTO jobs_v3 (id, status, created_at, model, runtime, compute_type, language, beam_size, vad_filter, "
        "segments, info, error, progress, cancel_requested, filename, batch_id, diarize, ai_results) "
        "SELECT id, status, created_at, model, runtime, compute_type, language, beam_size, vad_filter, "
        "segments, info, error, progress, cancel_requested, filename, batch_id, diarize, ai_results FROM jobs",
        "DROP TABLE jobs",
        "ALTER TABLE jobs_v3 RENAME TO jobs",
        "CREATE INDEX jobs_created_at ON jobs (created_at)",
    ),
}
JOB_UPSERT = ("INSERT OR REPLACE INTO jobs (id, status, created_at, model, runtime, compute_type, language, "
              "beam_size, vad_filter, segments, info, error, progress, cancel_requested, filename, batch_id, diarize, "
              "ai_results, ignored_controls) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
SUMMARY_SELECT = ("SELECT id, status, created_at, model, runtime, compute_type, language, beam_size, vad_filter, "
                  "'[]' AS segments, info, error, progress, cancel_requested, filename, batch_id, diarize, "
                  "'[]' AS ai_results, ignored_controls FROM jobs")


class StorageError(Exception):
    """Local job storage failed. The message never includes a filesystem path."""


def _data_dir() -> Path:
    return Path(os.environ.get("WHISPER_DATA_DIR") or DEFAULT_DATA_DIR).expanduser().resolve()


def _data_dir_label(data_dir: Path) -> str:
    try:
        return data_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return data_dir.name


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _job_row(job: Job) -> tuple:
    return (job.id, job.status, job.created_at, job.model, job.runtime, job.compute_type, job.language,
            job.beam_size, None if job.vad_filter is None else int(job.vad_filter),
            json.dumps(list(job.segments or [])), json.dumps(job.info),
            job.error, job.progress, int(job.cancel_requested), job.filename, job.batch_id, int(job.diarize),
            json.dumps(list(job.ai_results or [])), json.dumps(list(job.ignored_controls or [])))


def _batch_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {"id": row["id"], "created_at": row["created_at"], "job_ids": json.loads(row["job_ids"])}


def _job_from_row(row: sqlite3.Row) -> Job:
    vad_filter = row["vad_filter"]
    return Job(id=row["id"], status=row["status"], created_at=row["created_at"], model=row["model"],
               runtime=row["runtime"], compute_type=row["compute_type"], language=row["language"],
               beam_size=row["beam_size"], vad_filter=None if vad_filter is None else bool(vad_filter),
               segments=json.loads(row["segments"]), info=json.loads(row["info"]), error=row["error"],
               progress=row["progress"], cancel_requested=bool(row["cancel_requested"]),
               filename=row["filename"], batch_id=row["batch_id"], diarize=bool(row["diarize"]),
               ai_results=json.loads(row["ai_results"]), ignored_controls=json.loads(row["ignored_controls"]))


class JobStore:
    """SQLite record of jobs and batches in the data directory. Audio is never written here."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.path = data_dir / DATABASE_NAME
        self._lock = threading.Lock()
        try:
            data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Incremental auto-vacuum lets deletions return space to the filesystem; it only applies to new files.
            self._conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._migrate()
            self._restrict_files()
        except (OSError, sqlite3.Error):
            raise StorageError(STORAGE_UNAVAILABLE) from None

    def _restrict_files(self) -> None:
        """Keep the database and its WAL and shared-memory files readable only by this user."""
        for path in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
            try:
                path.chmod(0o600)
            except FileNotFoundError:
                continue

    def _migrate(self) -> None:
        conn = self._conn
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        version = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise StorageError("The local job database was written by a newer STT Studio; upgrade the server")
        if version < SCHEMA_VERSION:
            conn.execute("BEGIN IMMEDIATE")
            if version == 0:
                for statement in SCHEMA_STATEMENTS:
                    conn.execute(statement)
            for target in range(max(version, 1) + 1, SCHEMA_VERSION + 1):
                for statement in SCHEMA_MIGRATIONS[target]:
                    conn.execute(statement)
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            conn.execute("COMMIT")

    @contextmanager
    def _db(self):
        with self._lock:
            try:
                yield self._conn
            except sqlite3.Error:
                raise StorageError(STORAGE_UNAVAILABLE) from None
            finally:
                # SQLite creates -wal and -shm with the process umask, so tighten them after each use.
                try:
                    self._restrict_files()
                except OSError:
                    logger.warning("Could not restrict the permissions of the local job database")

    def _reclaim(self) -> None:
        with self._db() as conn:
            conn.execute("PRAGMA incremental_vacuum").fetchall()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()

    def mark_interrupted(self) -> int:
        """Fail every job that was queued or running when the previous server process stopped.

        Segments already stored for the job stay in the row. The audio is not resumed.
        """
        with self._db() as conn, conn:
            cursor = conn.execute("UPDATE jobs SET status = 'failed', error = ? WHERE status IN ('queued', 'running')",
                                  (INTERRUPTED_MESSAGE,))
            return cursor.rowcount

    def save_job(self, job: Job) -> None:
        with self._db() as conn, conn:
            conn.execute(JOB_UPSERT, _job_row(job))

    def create_batch(self, batch_id: str, created_at: float, jobs: list[Job]) -> None:
        with self._db() as conn, conn:
            conn.execute("INSERT INTO batches (id, created_at, job_ids) VALUES (?, ?, ?)",
                         (batch_id, created_at, json.dumps([job.id for job in jobs])))
            conn.executemany(JOB_UPSERT, [_job_row(job) for job in jobs])

    def load_job(self, job_id: str) -> Job | None:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _job_from_row(row) if row is not None else None

    def list_jobs(self, limit: int, offset: int) -> tuple[list[Job], int]:
        with self._db() as conn:
            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            rows = conn.execute(f"{SUMMARY_SELECT} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                                (limit, offset)).fetchall()
        return [_job_from_row(row) for row in rows], total

    def load_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._db() as conn:
            row = conn.execute("SELECT id, created_at, job_ids FROM batches WHERE id = ?", (batch_id,)).fetchone()
        if row is None:
            return None
        return _batch_from_row(row)

    def list_batches(self, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        with self._db() as conn:
            total = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
            rows = conn.execute("SELECT id, created_at, job_ids FROM batches ORDER BY created_at DESC, rowid DESC "
                                "LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [_batch_from_row(row) for row in rows], total

    def delete_job(self, job_id: str) -> None:
        with self._db() as conn:
            with conn:
                row = conn.execute("SELECT batch_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
                conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                if row is not None and row["batch_id"]:
                    batch = conn.execute("SELECT job_ids FROM batches WHERE id = ?", (row["batch_id"],)).fetchone()
                    if batch is not None:
                        remaining = [item for item in json.loads(batch["job_ids"]) if item != job_id]
                        conn.execute("UPDATE batches SET job_ids = ? WHERE id = ?",
                                     (json.dumps(remaining), row["batch_id"]))

    def delete_batch(self, batch_id: str) -> None:
        with self._db() as conn:
            with conn:
                conn.execute("DELETE FROM jobs WHERE batch_id = ?", (batch_id,))
                conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))

    def reclaim(self) -> None:
        """Return freed pages to the disk after a delete. Failing here only wastes space, so it is logged, not raised."""
        try:
            self._reclaim()
        except (OSError, sqlite3.Error, StorageError):
            logger.exception("Could not reclaim disk space after deleting from the job database")

    def stats(self) -> dict[str, int]:
        with self._db() as conn:
            jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            batches = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        files = (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm"))
        return {"jobs": jobs, "batches": batches, "bytes": sum(_file_size(path) for path in files)}


class ModelBusy(Exception):
    """Another model load or inference holds the model manager."""


_PCM_RATE = 16000
_PCM_BYTES = 4


def _require_decode_space(seconds: float) -> None:
    expected_bytes = int(seconds * _PCM_RATE * _PCM_BYTES)
    free_bytes = shutil.disk_usage(tempfile.gettempdir()).free
    if free_bytes < expected_bytes + 64 * 1024 * 1024:
        raise RuntimeError("Not enough free disk space to decode this audio; free space and retry")


@contextmanager
def _pcm_directory():
    """A temporary directory for one decode. The sweep leaves it alone while this process holds it."""
    try:
        temporary = tempfile.TemporaryDirectory(prefix="stt-pcm-")
    except OSError as exc:
        raise RuntimeError("Could not create temporary storage for audio decoding") from exc
    path = Path(temporary.name)
    tracked = [path]
    try:
        resolved = path.resolve()
        if resolved != path:
            tracked.append(resolved)
    except OSError:
        pass
    for item in tracked:
        _track_upload(item)
    try:
        yield temporary.name
    finally:
        try:
            temporary.cleanup()
        finally:
            for item in tracked:
                _untrack_upload(item)


def _terminate_ffmpeg_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=5)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            process.wait(timeout=5)
        except Exception:
            pass


def _run_ffmpeg(command: list[str], path: Path, timeout: float, control: JobControl | None = None) -> None:
    """Run one ffmpeg argument list. Local paths in the decoder error are redacted."""
    try:
        with tempfile.TemporaryFile(mode="w+b") as error_log:
            try:
                if control is None:
                    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=error_log, check=True, timeout=timeout)
                else:
                    process = subprocess.Popen(
                        command, stdout=subprocess.DEVNULL, stderr=error_log, start_new_session=True,
                    )
                    deadline = time.monotonic() + timeout
                    try:
                        while True:
                            code = process.poll()
                            if code is not None:
                                if code != 0:
                                    raise subprocess.CalledProcessError(code, command)
                                break
                            if control.cancelled:
                                _terminate_ffmpeg_group(process)
                                raise JobCancelled("Transcription was cancelled")
                            if time.monotonic() >= deadline:
                                _terminate_ffmpeg_group(process)
                                raise subprocess.TimeoutExpired(command, timeout)
                            time.sleep(0.05)
                    except JobCancelled:
                        raise
                    except subprocess.TimeoutExpired:
                        raise
                    except subprocess.CalledProcessError:
                        raise
            except FileNotFoundError as exc:
                raise RuntimeError("ffmpeg is required to decode audio; install it and retry") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Audio decoding timed out; try a shorter or simpler audio file") from exc
            except subprocess.CalledProcessError as exc:
                error_log.seek(max(0, error_log.tell() - 8192))
                detail = error_log.read(8192).decode("utf-8", errors="replace").strip()
                detail = detail.replace(str(path), "uploaded audio")
                raise RuntimeError(f"Audio decoding failed: {detail or 'ffmpeg could not decode this file'}") from exc
    except JobCancelled:
        raise
    except OSError as exc:
        raise RuntimeError("Could not write temporary decoded audio; check available disk space") from exc


def _bounded_decode_command(path: Path, pcm_path: Path, decode_limit_seconds: float) -> list[str]:
    return ["ffmpeg", "-nostdin", "-v", "error", "-xerror",
            "-protocol_whitelist", _MEDIA_PROTOCOL_WHITELIST, "-i", str(path),
            "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000",
            "-t", str(decode_limit_seconds), "-y", str(pcm_path)]


def _part_decode_command(path: Path, pcm_path: Path, start: float, length: float) -> list[str]:
    # -ss and -t are input options so ffmpeg does not decode the rest of the file into this part.
    return ["ffmpeg", "-nostdin", "-v", "error", "-xerror",
            "-protocol_whitelist", _MEDIA_PROTOCOL_WHITELIST,
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(path),
            "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000",
            "-y", str(pcm_path)]


def _map_pcm(pcm_path: Path):
    import numpy as np

    try:
        pcm_bytes = pcm_path.stat().st_size
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg did not produce decoded audio; check the input file") from exc
    if pcm_bytes == 0:
        raise RuntimeError("Audio decoder produced no samples; check the input file")
    if pcm_bytes % _PCM_BYTES:
        raise RuntimeError("Audio decoder produced invalid PCM data")
    try:
        audio = np.memmap(pcm_path, dtype=np.float32, mode="r", shape=(pcm_bytes // _PCM_BYTES,))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Could not read the temporary decoded audio; free disk space and retry") from exc
    return audio, pcm_bytes / (_PCM_RATE * _PCM_BYTES)


_SILENCE_STDERR_CAP = 1024 * 1024


def _parse_silence_midpoints(stderr: bytes, path: Path) -> list[float]:
    detail = stderr.decode("utf-8", errors="replace").replace(str(path), "uploaded audio")
    pauses: list[float] = []
    pending: float | None = None
    for line in detail.splitlines():
        start = re.search(r"silence_start: (-?[\d.]+)", line)
        if start:
            pending = float(start.group(1))
            continue
        end = re.search(r"silence_end: (-?[\d.]+)", line)
        if end and pending is not None:
            pauses.append((pending + float(end.group(1))) / 2)
            pending = None
    return pauses


def _accumulate_silence_stderr(chunk: bytes, stored: bytearray, *, cap: int, overflow: list[bool]) -> None:
    if not chunk or overflow[0]:
        return
    room = cap - len(stored)
    if len(chunk) <= room:
        stored.extend(chunk)
    else:
        stored.extend(chunk[:room])
        overflow[0] = True


def _read_available_silence_stderr(stderr) -> bytes:
    import select

    collected = bytearray()
    while stderr is not None:
        try:
            ready, _, _ = select.select([stderr], [], [], 0)
        except (ValueError, OSError):
            break
        if not ready:
            break
        chunk = stderr.read(4096)
        if not chunk:
            break
        collected.extend(chunk)
    return bytes(collected)


def _silence_midpoints(path: Path, duration: float, control: JobControl | None = None) -> list[float]:
    """Midpoints of pauses at least 0.4 s long, below -35 dB. An empty list means fixed boundaries."""
    command = ["ffmpeg", "-nostdin", "-hide_banner",
               "-protocol_whitelist", _MEDIA_PROTOCOL_WHITELIST, "-vn", "-i", str(path),
               "-af", "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"]
    timeout = max(300.0, duration * 2)
    if control is None:
        try:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
        except Exception:
            # A missing pause list still leaves fixed boundaries. The part decode reports a real ffmpeg failure.
            logger.debug("Silence detection failed; using fixed part boundaries", exc_info=True)
            return []
        return _parse_silence_midpoints(result.stderr or b"", path)

    stored = bytearray()
    overflow = [False]
    try:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   start_new_session=True)
    except Exception:
        logger.debug("Silence detection failed; using fixed part boundaries", exc_info=True)
        return []

    deadline = time.monotonic() + timeout
    try:
        while True:
            if process.stderr is not None:
                chunk = _read_available_silence_stderr(process.stderr)
                _accumulate_silence_stderr(chunk, stored, cap=_SILENCE_STDERR_CAP, overflow=overflow)

            code = process.poll()
            if code is not None:
                if process.stderr is not None:
                    while True:
                        chunk = process.stderr.read(4096)
                        if not chunk:
                            break
                        if overflow[0]:
                            continue
                        _accumulate_silence_stderr(chunk, stored, cap=_SILENCE_STDERR_CAP, overflow=overflow)
                if overflow[0] or code != 0:
                    logger.debug("Silence detection failed; using fixed part boundaries")
                    return []
                return _parse_silence_midpoints(bytes(stored), path)

            if control.cancelled:
                _terminate_ffmpeg_group(process)
                raise JobCancelled("Transcription was cancelled")
            if time.monotonic() >= deadline:
                _terminate_ffmpeg_group(process)
                logger.debug("Silence detection failed; using fixed part boundaries", exc_info=True)
                return []
            time.sleep(0.05)
    except JobCancelled:
        raise
    except Exception:
        logger.debug("Silence detection failed; using fixed part boundaries", exc_info=True)
        try:
            _terminate_ffmpeg_group(process)
        except Exception:
            pass
        return []


def _find_cut_points(path: Path, duration: float, chunk_seconds: float,
                     control: JobControl | None = None) -> list[float]:
    """Choose a cut near each boundary, preferring the middle of a pause, and never past the limit."""
    targets = [index * chunk_seconds for index in range(1, math.ceil(duration / chunk_seconds))]
    if not targets:
        return []
    window = min(120.0, chunk_seconds * 0.25)
    pauses = _silence_midpoints(path, duration, control)
    cuts: list[float] = []
    for target in targets:
        previous = cuts[-1] if cuts else 0.0
        latest = previous + chunk_seconds
        candidates = [pause for pause in pauses
                      if abs(pause - target) <= window and previous + 1.0 < pause <= latest and pause < duration]
        chosen = min(candidates, key=lambda pause: abs(pause - target)) if candidates else min(target, latest)
        if cuts and chosen <= cuts[-1] + 1.0:
            chosen = min(target, latest)
        chosen = min(chosen, latest)
        if chosen <= previous or chosen >= duration:
            continue
        cuts.append(chosen)
    return cuts


def _part_spans(cuts: list[float], duration: float, chunk_seconds: float) -> list[tuple[float, float]]:
    """Return (start, length) pairs. Each length is at most chunk_seconds and the spans cover the file."""
    points = [0.0]
    for cut in cuts:
        if points[-1] + 1e-3 < cut < duration:
            points.append(cut)
    if duration - points[-1] > 1e-3:
        points.append(duration)
    spans: list[tuple[float, float]] = []
    for start, end in zip(points, points[1:]):
        cursor = start
        while end - cursor > chunk_seconds + 1e-3:
            spans.append((cursor, chunk_seconds))
            cursor += chunk_seconds
        remaining = end - cursor
        if remaining > 1e-3:
            spans.append((cursor, remaining))
    return spans


def _with_offset(rows: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        kept.append({**row, "start": float(row["start"]) + offset,
                     "end": float(row["end"]) + offset, "text": text})
    return kept


def _take_language(chosen: tuple[Any, Any], meta: dict[str, Any]) -> tuple[Any, Any]:
    language, probability = chosen
    detected = meta.get("language")
    if language is None and detected:
        return detected, meta.get("language_probability")
    return chosen


def _chunk_meta(language_state: tuple[Any, Any], duration: float, chunk_count: int) -> dict[str, Any]:
    language, probability = language_state
    return {"language": language, "language_probability": probability,
            "duration": duration, "chunk_count": chunk_count}


class ModelManager:
    def __init__(self, factory: Callable[[str, str], Any] = _model_factory,
                 adapters: dict[str, InferenceAdapter] | None = None):
        self.adapters = adapters if adapters is not None else _default_adapters(factory)
        self._models: dict[tuple[str, str, str], Any] = {}
        self._state_lock = threading.Lock()
        self._operation_lock = threading.RLock()

    @contextmanager
    def exclusive(self):
        """Hold the operation lock without waiting. Raises ModelBusy if a load or inference is in progress."""
        if not self._operation_lock.acquire(blocking=False):
            raise ModelBusy()
        try:
            yield
        finally:
            self._operation_lock.release()

    def evict(self, model: str) -> list[dict[str, str]]:
        """Drop the resident instances of one model and leave the others loaded. Call inside exclusive()."""
        with self._state_lock:
            keys = [key for key in self._models if key[0] == model]
            released = [{"model": m, "compute_type": c, "runtime": r} for m, c, r in keys]
            for key in keys:
                del self._models[key]
        if released:
            gc.collect()
            _release_accelerator_cache()
        return released

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
            _release_accelerator_cache()
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
            try:
                adapter.warmup(instance)
            except BaseException:
                # A model that failed warmup must not stay resident or be reported as loaded.
                instance = None
                with self._state_lock:
                    self._models.pop((model, compute_type, runtime), None)
                gc.collect()
                _release_accelerator_cache()
                raise
            return {"model": model, "compute_type": compute_type, "loaded": True,
                    "runtime": runtime, "warmup": "complete",
                    # An adapter that knows its accelerator reports it; the rest use WHISPER_DEVICE.
                    "device": getattr(adapter, "device", None) or os.environ.get("WHISPER_DEVICE", "cpu")}

    def transcribe(self, path: Path, *, model: str, compute_type: str, language: str | None,
                   beam_size: int, vad_filter: bool, initial_prompt: str | None,
                   runtime: str = "faster-whisper", control: JobControl | None = None):
        if control is not None:
            control.check()
        duration = _probe_audio_duration(path, control)
        kwargs = dict(model=model, compute_type=compute_type, language=language, beam_size=beam_size,
                      vad_filter=vad_filter, initial_prompt=initial_prompt, runtime=runtime, control=control)
        if duration <= MAX_AUDIO_SECONDS:
            return self._transcribe_within_limit(path, duration=duration, **kwargs)
        return self._transcribe_in_parts(path, duration=duration, **kwargs)

    def _open_runtime(self, runtime: str, model: str, compute_type: str):
        adapter = self.adapters.get(runtime)
        if adapter is None:
            raise ValueError(f"Inference runtime is unavailable: {runtime}")
        return adapter, self.get(model, compute_type, runtime)

    def _infer(self, adapter, instance, audio, *, language, beam_size, vad_filter, initial_prompt, control):
        if control is not None:
            control.check()
        return adapter.transcribe(
            instance, audio, language=language, beam_size=beam_size,
            vad_filter=vad_filter, initial_prompt=initial_prompt,
            **({"control": control} if control is not None else {}),
        )

    def _transcribe_within_limit(self, path: Path, *, duration: float, model: str, compute_type: str,
                                 language: str | None, beam_size: int, vad_filter: bool,
                                 initial_prompt: str | None, runtime: str, control: JobControl | None):
        # One decode, one inference. The -t cap stays so a wrong probe cannot fill the disk.
        decode_limit_seconds = MAX_AUDIO_SECONDS + 5
        _require_decode_space(min(duration, decode_limit_seconds))
        with _pcm_directory() as temp_name:
            pcm_path = Path(temp_name) / "audio.f32"
            timeout = max(120, min(duration, decode_limit_seconds) * 3)
            _run_ffmpeg(_bounded_decode_command(path, pcm_path, decode_limit_seconds), path, timeout, control)
            audio, decoded_seconds = _map_pcm(pcm_path)
            try:
                if decoded_seconds <= MAX_AUDIO_SECONDS:
                    with self._operation_lock:
                        adapter, instance = self._open_runtime(runtime, model, compute_type)
                        rows, meta = self._infer(
                            adapter, instance, audio, language=language, beam_size=beam_size,
                            vad_filter=vad_filter, initial_prompt=initial_prompt, control=control)
                    _note_transcribed_part(control)
                    return rows, {**meta, "chunk_count": 1}
                collected, language_state, parts = self._transcribe_slices(
                    audio, total_duration=decoded_seconds, model=model, compute_type=compute_type,
                    language=language, beam_size=beam_size, vad_filter=vad_filter,
                    initial_prompt=initial_prompt, runtime=runtime, control=control)
            finally:
                audio._mmap.close()
            if decoded_seconds + 0.05 < decode_limit_seconds:
                return collected, _chunk_meta(language_state, decoded_seconds, parts)
            # The probe was short and the safety cap was filled, so keep reading one part at a time.
            pcm_path.unlink(missing_ok=True)
            tail, language_state, tail_parts, covered = self._transcribe_until_end(
                path, pcm_path, start=decoded_seconds, language_state=language_state,
                model=model, compute_type=compute_type, language=language, beam_size=beam_size,
                vad_filter=vad_filter, initial_prompt=initial_prompt, runtime=runtime, control=control)
            return collected + tail, _chunk_meta(language_state, covered, parts + tail_parts)

    def _transcribe_slices(self, audio, *, total_duration: float, model: str, compute_type: str,
                           language: str | None, beam_size: int, vad_filter: bool,
                           initial_prompt: str | None, runtime: str, control: JobControl | None):
        max_samples = max(1, int(MAX_AUDIO_SECONDS * _PCM_RATE))
        collected: list[dict[str, Any]] = []
        language_state: tuple[Any, Any] = (None, None)
        parts = 0
        with self._operation_lock:
            adapter, instance = self._open_runtime(runtime, model, compute_type)
            offset_samples = 0
            total_samples = len(audio)
            while offset_samples < total_samples:
                if control is not None:
                    control.check()
                offset = offset_samples / _PCM_RATE
                timeline = _TimelineControl(control, offset, total_duration)
                rows, meta = self._infer(
                    adapter, instance, audio[offset_samples:offset_samples + max_samples],
                    language=language, beam_size=beam_size, vad_filter=vad_filter,
                    initial_prompt=initial_prompt, control=timeline)
                collected.extend(_with_offset(rows, offset))
                language_state = _take_language(language_state, meta)
                parts += 1
                _note_transcribed_part(control)
                offset_samples += max_samples
        return collected, language_state, parts

    def _transcribe_in_parts(self, path: Path, *, duration: float, model: str, compute_type: str,
                             language: str | None, beam_size: int, vad_filter: bool,
                             initial_prompt: str | None, runtime: str, control: JobControl | None):
        spans = _part_spans(_find_cut_points(path, duration, MAX_AUDIO_SECONDS, control),
                            duration, MAX_AUDIO_SECONDS)
        if not spans:
            raise RuntimeError("Audio duration must be greater than zero")
        collected: list[dict[str, Any]] = []
        language_state: tuple[Any, Any] = (None, None)
        with _pcm_directory() as temp_name:
            pcm_path = Path(temp_name) / "audio.f32"
            with self._operation_lock:
                adapter, instance = self._open_runtime(runtime, model, compute_type)
                for start, length in spans:
                    if control is not None:
                        control.check()
                    _require_decode_space(length)
                    _run_ffmpeg(_part_decode_command(path, pcm_path, start, length), path, max(120, length * 3),
                                control)
                    audio, _decoded = _map_pcm(pcm_path)
                    try:
                        timeline = _TimelineControl(control, start, duration)
                        rows, meta = self._infer(
                            adapter, instance, audio, language=language, beam_size=beam_size,
                            vad_filter=vad_filter, initial_prompt=initial_prompt, control=timeline)
                    finally:
                        audio._mmap.close()
                        pcm_path.unlink(missing_ok=True)
                    collected.extend(_with_offset(rows, start))
                    language_state = _take_language(language_state, meta)
                    _note_transcribed_part(control)
        return collected, _chunk_meta(language_state, duration, len(spans))

    def _transcribe_until_end(self, path: Path, pcm_path: Path, *, start: float,
                              language_state: tuple[Any, Any], model: str, compute_type: str,
                              language: str | None, beam_size: int, vad_filter: bool,
                              initial_prompt: str | None, runtime: str, control: JobControl | None):
        """Decode the rest of a file whose probe was too short, one limited part at a time."""
        collected: list[dict[str, Any]] = []
        parts = 0
        cursor = start
        length = MAX_AUDIO_SECONDS
        with self._operation_lock:
            adapter, instance = self._open_runtime(runtime, model, compute_type)
            while True:
                if control is not None:
                    control.check()
                _require_decode_space(length)
                try:
                    _run_ffmpeg(_part_decode_command(path, pcm_path, cursor, length), path, max(120, length * 3),
                                control)
                except RuntimeError as exc:
                    # Seeking past the end can fail under -xerror. Keep the audio already transcribed.
                    message = str(exc).lower()
                    if "disk" in message or "temporary" in message or "timed out" in message:
                        raise
                    break
                try:
                    audio, decoded = _map_pcm(pcm_path)
                except RuntimeError as exc:
                    if "no samples" in str(exc) or "did not produce" in str(exc):
                        break
                    raise
                try:
                    if decoded <= 0.01:
                        break
                    timeline = _TimelineControl(control, cursor, max(start, cursor + decoded))
                    rows, meta = self._infer(
                        adapter, instance, audio, language=language, beam_size=beam_size,
                        vad_filter=vad_filter, initial_prompt=initial_prompt, control=timeline)
                finally:
                    audio._mmap.close()
                    pcm_path.unlink(missing_ok=True)
                collected.extend(_with_offset(rows, cursor))
                language_state = _take_language(language_state, meta)
                parts += 1
                _note_transcribed_part(control)
                if decoded < 0.01:
                    break
                cursor += decoded
                if decoded + 0.05 < length:
                    break
        return collected, language_state, parts, cursor

    def snapshot(self) -> list[dict[str, str]]:
        with self._state_lock:
            return [{"model": model, "compute_type": compute_type, "runtime": runtime}
                    for model, compute_type, runtime in self._models]

    def unload(self) -> dict[str, Any]:
        """Drop every resident model and release accelerator caches. Waits for any running inference."""
        with self._operation_lock:
            with self._state_lock:
                released = [{"model": model, "compute_type": compute_type, "runtime": runtime}
                            for model, compute_type, runtime in self._models]
                self._models.clear()
            gc.collect()
            _release_accelerator_cache()
        return {"unloaded": released, "loaded_models": []}


def _release_accelerator_cache() -> None:
    """Return MLX's cached Metal buffers after a model is dropped. No-op when MLX was never imported."""
    if "mlx.core" not in sys.modules:
        return
    try:
        import mlx.core as mx
        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if callable(clear):
            clear()
    except Exception:
        logger.debug("Could not release MLX cached buffers", exc_info=True)


def _public_error(exc: Exception) -> str:
    """Keep useful diagnostics without returning local server paths."""
    message = str(exc).replace(str(MODEL_CACHE), "<model cache>")
    # The temporary directory is whatever TMPDIR says (on macOS, a per-user folder), so redact it too.
    temp_roots = {tempfile.gettempdir(), str(Path(tempfile.gettempdir()).resolve()), "/tmp", "/private/tmp"}
    alternatives = "|".join(re.escape(root.rstrip("/")) for root in sorted(temp_roots, key=len, reverse=True))
    message = re.sub(rf"(?:{alternatives})/whisper-upload-[^\s:'\"]+", "<uploaded audio>", message)
    message = re.sub(rf"(?:{alternatives})/(?:stt-pcm|stt-diarize)-[^\s:'\"]+", "<path>", message)
    # Any other absolute path, including ones outside the temp roots.
    message = diarization.redact_paths(message)
    return message[:1000] or exc.__class__.__name__


def _require_audio(paths: list[Path]) -> None:
    """Refuse uploads with no audio track before they are queued, so the failure is immediate and clear."""
    for path in paths:
        try:
            info = _ffprobe_media(path)
        except RuntimeError as exc:
            for other in paths:
                other.unlink(missing_ok=True)
            raise HTTPException(422, _public_error(exc)) from exc
        if not info["has_audio"]:
            for other in paths:
                other.unlink(missing_ok=True)
            raise HTTPException(422, "The file has no audio track to transcribe.")


def _normalize_language(language: str | None) -> str | None:
    """Return a Whisper language code, or None for auto-detection. Unknown codes are refused before any upload is stored."""
    if language is None:
        return None
    if len(language) > 20:
        raise HTTPException(422, "language must be at most 20 characters")
    code = language.strip().lower()
    if code in ("", "auto"):
        return None
    if code not in WHISPER_LANGUAGE_CODES:
        raise HTTPException(422, f"Unknown language code '{code}'. Use a code such as en or vi, or auto to detect it.")
    return code


def _require_model_language(model: str, runtime: str, language: str | None) -> None:
    """Refuse a language this model cannot accept, before any upload is stored."""
    if language is None:
        return
    catalog_model = MODEL_CATALOG_BY_ID.get(MODEL_ALIASES.get(model, model))
    if catalog_model and catalog_model.get("languages") == "English" and language != "en":
        raise HTTPException(422, "This model accepts only English. Use en, or auto to detect the language.")
    if runtime == "mlx-audio" and language not in MLX_LANGUAGE_NAMES:
        raise HTTPException(422, f"Language '{language}' is not supported by this model. "
                                  "Use a supported code, or auto to detect the language.")


def _check_initial_prompt(initial_prompt: str | None) -> None:
    if initial_prompt is not None and len(initial_prompt) > MAX_INITIAL_PROMPT_CHARS:
        raise HTTPException(422, f"initial_prompt must be at most {MAX_INITIAL_PROMPT_CHARS} characters")


@dataclass(frozen=True)
class AcceptedControls:
    beam_size: int | None
    vad_filter: bool | None
    initial_prompt: str | None
    ignored: list[str]


def _accept_controls(model: str, runtime: str, beam_size: int, vad_filter: bool,
                     initial_prompt: str | None) -> AcceptedControls:
    """Accept Whisper-only controls the runtime cannot apply. They are stored as NULL and reported by name."""
    catalog_model = MODEL_CATALOG_BY_ID.get(MODEL_ALIASES.get(model, model))
    capabilities = model_capabilities(runtime, catalog_model.get("capabilities") if catalog_model else None)
    ignored: list[str] = []
    if not capabilities["beamSize"]:
        if beam_size != DEFAULT_BEAM_SIZE:
            ignored.append("beam_size")
        beam_size = None
    if not capabilities["vad"]:
        if vad_filter:
            ignored.append("vad_filter")
        vad_filter = None
    if not capabilities["initialPrompt"]:
        if initial_prompt:
            ignored.append("initial_prompt")
        initial_prompt = None
    return AcceptedControls(beam_size, vad_filter, initial_prompt, ignored)


def _require_diarization(requested: bool, runtime: str) -> None:
    if not requested:
        return
    if not RUNTIME_CAPABILITIES.get(runtime, {}).get("timedSegments", False):
        # Labels are matched to segment times; an untimed result would give every segment one speaker.
        raise HTTPException(422, f"Speaker labels need timed segments, and the {runtime} runtime returns one "
                                 "untimed segment. Use a faster-whisper model.")
    state = diarization.availability()
    if not state["available"]:
        raise HTTPException(422, state["reason"])


def _run_diarization(job: Job, path: Path, duration: float | None) -> dict[str, Any]:
    """Attach speaker labels to a finished transcript. A failure keeps the transcript and records the reason."""
    report: dict[str, Any] = {"model": diarization.MODEL_ID, "license": diarization.MODEL_LICENSE}
    try:
        turns = diarization.diarize_file(path, seconds=duration, should_stop=lambda: job.cancel_requested,
                                         cache_dir=MODEL_CACHE)
    except Exception as exc:
        if job.cancel_requested:
            return {**report, "status": "cancelled", "speakers": 0}
        logger.exception("Speaker labelling failed for job %s; keeping the transcript", job.id)
        reason = diarization.redact_paths(_public_error(exc))
        return {**report, "status": "failed", "speakers": 0, "reason": reason}
    return {**report, "status": "completed", "speakers": diarization.attach_speakers(job.segments, turns)}


def _read_pipe_bytes(pipe, limit: int) -> tuple[bytes, bool]:
    """Read at most limit bytes. The second value is true when more data was waiting."""
    if pipe is None:
        return b"", False
    chunks: list[bytes] = []
    total = 0
    while total < limit:
        piece = pipe.read(min(65536, limit - total))
        if not piece:
            break
        chunks.append(piece)
        total += len(piece)
    extra = pipe.read(1)
    if extra:
        return b"".join(chunks), True
    return b"".join(chunks), False


def _stop_ffprobe(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _run_ffprobe(command: list[str], timeout: float, control: JobControl | None = None) -> subprocess.CompletedProcess:
    """Run ffprobe and stop after 1 MiB of stdout. The refusal does not include a local path."""
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise
    stdout_box: list[tuple[bytes, bool]] = []
    stderr_box: list[tuple[bytes, bool]] = []

    def read_stdout() -> None:
        data, over = _read_pipe_bytes(process.stdout, _FFPROBE_STDOUT_LIMIT)
        stdout_box.append((data, over))
        if over:
            _stop_ffprobe(process)

    def read_stderr() -> None:
        data, over = _read_pipe_bytes(process.stderr, _FFPROBE_STDOUT_LIMIT)
        stderr_box.append((data, over))
        if over:
            _stop_ffprobe(process)

    threads = [threading.Thread(target=read_stdout), threading.Thread(target=read_stderr)]
    for thread in threads:
        thread.start()
    timed_out: subprocess.TimeoutExpired | None = None
    try:
        if control is None:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                timed_out = exc
                _stop_ffprobe(process)
                returncode = process.poll() if process.poll() is not None else -1
        else:
            deadline = time.monotonic() + timeout
            returncode: int | None = None
            while True:
                code = process.poll()
                if code is not None:
                    returncode = code
                    break
                if control.cancelled:
                    _stop_ffprobe(process)
                    raise JobCancelled("Transcription was cancelled")
                if time.monotonic() >= deadline:
                    timed_out = subprocess.TimeoutExpired(command, timeout)
                    _stop_ffprobe(process)
                    returncode = process.poll() if process.poll() is not None else -1
                    break
                time.sleep(0.05)
    except JobCancelled:
        for thread in threads:
            thread.join()
        raise
    for thread in threads:
        thread.join()
    stdout, stdout_over = stdout_box[0] if stdout_box else (b"", False)
    stderr, stderr_over = stderr_box[0] if stderr_box else (b"", False)
    if stdout_over or stderr_over or len(stdout) > _FFPROBE_STDOUT_LIMIT:
        raise RuntimeError("ffprobe output exceeded 1 MiB")
    if timed_out is not None:
        raise timed_out
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, stdout, stderr)
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _probe_audio_duration(path: Path, control: JobControl | None = None) -> float:
    """Read container duration before decoding the full waveform."""
    try:
        result = _run_ffprobe(
            ["ffprobe", "-v", "error", "-protocol_whitelist", _MEDIA_PROTOCOL_WHITELIST,
             "-select_streams", "a:0", "-show_entries",
             "format=duration:stream=duration", "-of", "json", str(path)],
            20,
            control,
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


def _ffprobe_media(path: Path) -> dict[str, Any]:
    """Describe every stream in a media file (audio, video, subtitles) without decoding it."""
    try:
        result = _run_ffprobe(
            ["ffprobe", "-v", "error", "-protocol_whitelist", _MEDIA_PROTOCOL_WHITELIST,
             "-show_format", "-show_streams", "-of", "json", str(path)],
            30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to inspect media; install ffmpeg and retry") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Media inspection timed out; check the input file") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip().replace(str(path), "uploaded file")
        raise RuntimeError(f"Could not inspect media: {detail or 'ffprobe could not read this file'}") from exc
    try:
        metadata = json.loads(result.stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Media inspection returned unreadable output") from exc

    def number(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed == parsed and parsed >= 0 else None

    streams = []
    for raw in metadata.get("streams") or []:
        kind = raw.get("codec_type") or "other"
        item: dict[str, Any] = {"index": raw.get("index"), "type": kind, "codec": raw.get("codec_name")}
        if kind == "audio":
            item["channels"] = raw.get("channels")
            item["sample_rate"] = number(raw.get("sample_rate"))
        if kind == "video":
            item["width"] = raw.get("width")
            item["height"] = raw.get("height")
        item["duration_seconds"] = number(raw.get("duration"))
        streams.append(item)
    fmt = metadata.get("format") or {}
    duration = number(fmt.get("duration"))
    if duration is None:
        durations = [s["duration_seconds"] for s in streams if s["duration_seconds"]]
        duration = max(durations) if durations else None
    return {
        "container": fmt.get("format_name"),
        "duration_seconds": duration,
        "has_audio": any(s["type"] == "audio" for s in streams),
        "has_video": any(s["type"] == "video" for s in streams),
        "streams": streams,
    }


API_ENDPOINTS = (
    {"method": "GET", "path": "/api/capabilities", "purpose": "Limits, accepted formats, runtimes and this endpoint list"},
    {"method": "GET", "path": "/api/health", "purpose": "Service status, active jobs, loaded models"},
    {"method": "GET", "path": "/api/models", "purpose": "Runnable models, cache state, runtimes"},
    {"method": "POST", "path": "/api/models/load", "purpose": "Download (if needed), load and warm a model"},
    {"method": "POST", "path": "/api/models/unload", "purpose": "Release resident models and accelerator memory"},
    {"method": "GET", "path": "/api/models/cache", "purpose": "Every catalog model with its cached flag and bytes on disk"},
    {"method": "DELETE", "path": "/api/models/cache/{model_id:path}", "purpose": "Delete one model's cached files (404 unknown id, 409 while in use)"},
    {"method": "POST", "path": "/api/probe", "purpose": "Inspect a media file's streams and duration (no transcription)"},
    {"method": "POST", "path": "/api/jobs", "purpose": "Queue one transcription from an uploaded file"},
    {"method": "GET", "path": "/api/jobs", "purpose": "Stored jobs, newest first, metadata only without segments (?limit=&offset=)"},
    {"method": "GET", "path": "/api/jobs/{job_id}", "purpose": "Job state, progress and segments; ?wait=SECONDS long-polls until finished"},
    {"method": "DELETE", "path": "/api/jobs/{job_id}", "purpose": "Delete a finished job and its transcript (409 while queued or running)"},
    {"method": "POST", "path": "/api/jobs/{job_id}/cancel", "purpose": "Cancel a queued or running job (partial text is kept)"},
    {"method": "POST", "path": "/api/jobs/{job_id}/ai", "purpose": "Summary or action items for a finished job via NVIDIA NIM; sends transcript text only (503 without a key)"},
    {"method": "GET", "path": "/api/jobs/{job_id}/export", "purpose": "Download a finished job as txt, srt, vtt or json (?format=)"},
    {"method": "POST", "path": "/api/batches", "purpose": "Queue up to N files that run one after another"},
    {"method": "GET", "path": "/api/batches", "purpose": "Stored batches, newest first, paginated like GET /api/jobs (limit, offset)"},
    {"method": "GET", "path": "/api/batches/{batch_id}", "purpose": "Batch counts and per-file status"},
    {"method": "DELETE", "path": "/api/batches/{batch_id}", "purpose": "Delete a finished batch and its jobs (409 while any file is queued or running)"},
    {"method": "POST", "path": "/api/batches/{batch_id}/cancel", "purpose": "Cancel remaining files in a batch"},
    {"method": "GET", "path": "/api/batches/{batch_id}/export", "purpose": "Download a finished batch as txt or json (?format=)"},
    {"method": "GET", "path": "/api/storage", "purpose": "Where jobs and transcripts are stored, with counts and bytes (audio is never stored)"},
)
MAX_JSON_BODY_BYTES = 1024 * 1024
MAX_AI_BODY_BYTES = 4 * 1024 * 1024
AI_ROUTE = re.compile(r"/api/jobs/[^/]+/ai")
ACCEPTED_EXTENSIONS = ("wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma", "mp4", "m4v", "mov", "mkv", "webm")


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

        is_upload = (scope.get("method") == "POST" and scope.get("path") in {"/api/jobs", "/api/batches", "/api/probe"}
                     and headers.get(b"content-type", b"").lower().startswith(b"multipart/form-data"))
        if is_upload:
            total_limit = MAX_BATCH_UPLOAD_BYTES if scope.get("path") == "/api/batches" else MAX_UPLOAD_BYTES
            request_limit = total_limit + MAX_MULTIPART_OVERHEAD_BYTES
            label = "Multipart request"
        else:
            # Every other request is small JSON or an empty GET. The AI route carries transcript text, so it gets more room.
            is_ai = scope.get("method") == "POST" and AI_ROUTE.fullmatch(scope.get("path") or "") is not None
            request_limit = MAX_AI_BODY_BYTES if is_ai else MAX_JSON_BODY_BYTES
            label = "Request body"
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError:
                await self._reject(send, 400, "Invalid Content-Length header")
                return
            if declared_size > request_limit:
                await self._reject(send, 413, f"{label} exceeds {request_limit} byte limit")
                return
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > request_limit:
                    raise HTTPException(413, f"{label} exceeds {request_limit} byte limit")
            return message

        await self.app(scope, bounded_receive, send)


class AiSegment(BaseModel):
    speaker: str | None = Field(default=None, max_length=200)
    text: str


class AiRequest(BaseModel):
    mode: Literal["summary", "action_items"]
    # The transcript as the user sees it (edits and speaker names included). Omitted: the stored segments.
    segments: list[AiSegment] | None = Field(default=None, max_length=50_000)


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


async def _store_upload(file: UploadFile) -> Path:
    """Stream one uploaded file to a temporary path, enforcing the per-file size limit."""
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
    except OSError as exc:
        path.unlink(missing_ok=True)
        if exc.errno == errno.ENOSPC:
            # The message names no path: the temporary location is a server detail.
            raise HTTPException(503, "Not enough disk space") from None
        raise
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    if total == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(422, "Uploaded audio file is empty")
    return path


# Uploads and decode directories that queued or running jobs still need.
# The startup sweep must never delete these.
_LIVE_UPLOADS: set[str] = set()
_LIVE_UPLOADS_LOCK = threading.Lock()


def _track_upload(path: Path) -> None:
    with _LIVE_UPLOADS_LOCK:
        _LIVE_UPLOADS.add(str(path))


def _untrack_upload(path: Path) -> None:
    with _LIVE_UPLOADS_LOCK:
        _LIVE_UPLOADS.discard(str(path))


def _descriptor_path(fd: int) -> str | None:
    """The path this process has open on one descriptor, when the platform can say."""
    getpath = getattr(fcntl, "F_GETPATH", None)
    if getpath is not None:
        try:
            # macOS F_GETPATH accepts at most PATH_MAX (1024) bytes.
            raw = fcntl.fcntl(fd, getpath, bytes(1024))
        except (OSError, ValueError):
            raw = b""
        if isinstance(raw, bytes):
            text = raw.split(b"\x00", 1)[0].decode("utf-8", "surrogateescape")
            if text:
                return text
    for root in ("/proc/self/fd", "/dev/fd"):
        try:
            return os.readlink(f"{root}/{fd}")
        except OSError:
            continue
    return None


def _process_has_open(path: Path) -> bool:
    """True when this process has the directory, or a file inside it, open."""
    try:
        target = str(path.resolve())
    except OSError:
        return True
    prefix = target + os.sep
    try:
        names = os.listdir("/dev/fd")
    except OSError:
        names = []
    if not names:
        try:
            names = os.listdir("/proc/self/fd")
        except OSError:
            return False
    for name in names:
        try:
            fd = int(name)
        except ValueError:
            continue
        opened = _descriptor_path(fd)
        if not opened:
            continue
        if opened == target or opened.startswith(prefix):
            return True
        try:
            resolved = str(Path(opened).resolve())
        except OSError:
            continue
        if resolved == target or resolved.startswith(prefix):
            return True
    return False


def _sweep_stale_temp_files(max_age_seconds: float = 3600) -> None:
    """Remove uploads, diarization copies and decoded audio left by a crash, so a recording does not linger in /tmp."""
    cutoff = time.time() - max_age_seconds
    with _LIVE_UPLOADS_LOCK:
        live = set(_LIVE_UPLOADS)
    root = Path(tempfile.gettempdir())
    for pattern in ("whisper-upload-*", "stt-diarize-*", "stt-pcm-*"):
        for path in root.glob(pattern):
            if str(path) in live:
                continue
            try:
                if str(path.resolve()) in live:
                    continue
                if path.stat().st_mtime > cutoff:
                    continue
                # A directory this process still has open belongs to a running decode.
                if path.is_dir() and _process_has_open(path):
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError:
                continue


def create_app(manager: ModelManager | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Restart recovery and temp cleanup run when the server starts, never when this module is imported.
        api.state.store.mark_interrupted()
        _sweep_stale_temp_files()
        yield

    api = FastAPI(title="STT Studio Local API", version="1.0.0", lifespan=lifespan)
    api.add_middleware(LocalAndBoundedUploadMiddleware)
    api.state.models = manager or ModelManager()
    # Jobs still queued or running live here; finished jobs are read back from the store.
    api.state.jobs = {}
    api.state.store = JobStore(_data_dir())
    api.state.jobs_lock = threading.Lock()
    api.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-job")
    data_label = _data_dir_label(api.state.store.data_dir)

    @api.exception_handler(StorageError)
    def storage_unavailable(_request, exc: StorageError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    def find_job(job_id: str) -> Job | None:
        # Caller holds jobs_lock.
        job = api.state.jobs.get(job_id)
        return job if job is not None else api.state.store.load_job(job_id)

    def save_with_retry(job: Job) -> bool:
        # Caller holds jobs_lock. One retry covers a transient failure; a second failure means no durable row.
        for attempt in (1, 2):
            try:
                api.state.store.save_job(job)
                return True
            except StorageError:
                logger.warning("Could not save the state of job %s (attempt %d)", job.id, attempt)
        return False

    # Last time a partial transcript was written, per job. Saves stay about a second apart.
    _segment_checkpoint_at: dict[str, float] = {}

    def checkpoint_segments(job: Job, *, force: bool = False) -> None:
        """Write the segments produced so far. A crash then leaves those rows in SQLite.

        Tiny segments do not each write a row. A part end always writes.
        """
        now = time.monotonic()
        previous = _segment_checkpoint_at.get(job.id)
        if not force and previous is not None and now - previous < 1.0:
            return
        with api.state.jobs_lock:
            if not save_with_retry(job):
                return
        _segment_checkpoint_at[job.id] = now

    def finish(job: Job, status: str, error: str | None = None, progress: float | None = None) -> None:
        """Persist the final state, then apply it in memory. Readers hold jobs_lock, so they never see a
        status the database does not hold."""
        _segment_checkpoint_at.pop(job.id, None)
        changes: dict[str, Any] = {"status": status, "error": error}
        if progress is not None:
            changes["progress"] = progress
        with api.state.jobs_lock:
            if save_with_retry(replace(job, **changes)):
                job.status, job.error = status, error
                if progress is not None:
                    job.progress = progress
                api.state.jobs.pop(job.id, None)
                return
            # Never report a job as completed without a durable row.
            failure = "Transcript could not be saved"
            saved = save_with_retry(replace(job, status="failed", error=failure))
            job.status, job.error = "failed", failure
            if saved:
                api.state.jobs.pop(job.id, None)
            else:
                # Kept in memory so readers see the failure instead of a stale status.
                logger.error("Job %s stays in memory as failed because its row could not be saved", job.id)

    def cancel_queued(job: Job) -> None:
        # Caller holds jobs_lock. The row is saved before memory changes, so a failed save leaves both as they were.
        api.state.store.save_job(replace(job, cancel_requested=True, status="cancelled"))
        job.cancel_requested = True
        job.status = "cancelled"
        api.state.jobs.pop(job.id, None)

    @api.get("/api/capabilities")
    def capabilities():
        """Everything an agent needs to plan a call: limits, formats, runtimes and endpoints."""
        runtimes = sorted(getattr(api.state.models, "adapters", {"faster-whisper": None}))
        return {
            "service": "stt-studio",
            "api_version": api.version,
            "default_model": DEFAULT_MODEL,
            "compute_types": sorted(COMPUTE_TYPES),
            "runtimes": runtimes,
            "runtime_capabilities": {name: model_capabilities(name) for name in runtimes},
            "diarization": diarization.availability(),
            "ai": nim.availability(),
            "accepted_extensions": list(ACCEPTED_EXTENSIONS),
            "accepts_any_ffmpeg_decodable_media": True,
            "export_formats": list(exporters.EXPORT_FORMATS),
            "batch_export_formats": list(exporters.BATCH_EXPORT_FORMATS),
            "limits": {
                "max_upload_bytes": MAX_UPLOAD_BYTES,
                "max_batch_files": MAX_BATCH_FILES,
                "max_batch_upload_bytes": MAX_BATCH_UPLOAD_BYTES,
                "max_audio_seconds": MAX_AUDIO_SECONDS,
                "max_active_jobs": MAX_ACTIVE_JOBS,
                "max_language_length": 20,
                "beam_size": [1, 20],
            },
            "notes": [
                "The server processes a long recording in parts when it is longer than max_audio_seconds.",
                "Video files are accepted directly; ffmpeg extracts the audio track.",
                "The supported MLX model (Qwen3-ASR) returns one untimed segment, so srt and vtt exports "
                "contain a single caption.",
                "diarize=true adds speaker labels (pyannote, local). It needs HF_TOKEN and the diarizer "
                "environment, runs after transcription, and a failure keeps the transcript with a reason in "
                "info.diarization. Speaker numbering is per job, so do not split a file when diarizing.",
                "Jobs, batches and transcripts persist in the local data directory until they are deleted, and ids "
                "stay valid across restarts. Audio is never stored. A job still queued or running at a restart is "
                "marked failed with 'Interrupted by a server restart'.",
                "POST /api/jobs/{id}/ai (summary or action_items) sends transcript text to NVIDIA NIM, never audio, "
                "and only when called. It needs NVIDIA_NIM_API_KEY_1 to _9 in .env or the environment; see capabilities.ai.",
            ],
            "endpoints": list(API_ENDPOINTS),
        }

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

    @api.post("/api/models/unload")
    def unload_models():
        with api.state.jobs_lock:
            if any(j.status == "running" for j in api.state.jobs.values()):
                raise HTTPException(409, "A transcription is running; wait for it or cancel it before unloading")
        diarization.stop_diarizer()
        return api.state.models.unload()

    @api.get("/api/models/cache")
    def model_cache():
        """Every catalog entry with whether its files are cached and how many bytes they take on disk."""
        models = []
        for model in MODEL_CATALOG:
            try:
                folder = _cache_folder(model["repository"])
            except ValueError:
                models.append({"id": model["id"], "cached": False, "sizeBytes": None})
                continue
            present = folder.is_dir()
            models.append({"id": model["id"], "cached": _model_is_cached(model["repository"]),
                           "sizeBytes": _folder_bytes(folder) if present else None})
        return {"models": models}

    @api.delete("/api/models/cache/{model_id:path}")
    def delete_cached_model(model_id: str):
        """Delete one catalog model's cached files. Other models and their files are never touched."""
        model = MODEL_CATALOG_BY_ID.get(model_id)
        if model is None:
            raise HTTPException(404, "Model not found in the catalog")
        try:
            folder = _cache_folder(model["repository"])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        with api.state.jobs_lock:
            jobs = list(api.state.jobs.values())
            running = any(j.status == "running" for j in jobs)
            # A queued job for this model would download into the folder as it is removed.
            wanted = any(j.model == model_id and j.status in ("queued", "running") for j in jobs)
            loaded = any(entry["model"] == model_id for entry in api.state.models.snapshot())
            if wanted or (running and loaded):
                raise HTTPException(409, "This model is loaded and a transcription is running, or a job is queued "
                                         "for it; wait for the jobs to finish before deleting its files")
        if running:
            # Another model's transcription holds the manager, so it cannot be waited on; this model is not in use.
            freed, deleted = _remove_model_folder(folder)
        else:
            try:
                with api.state.models.exclusive():
                    api.state.models.evict(model_id)
                    freed, deleted = _remove_model_folder(folder)
            except ModelBusy:
                raise HTTPException(409, "A model is being loaded; try again when it finishes") from None
        return {"id": model_id, "deleted": deleted, "freedBytes": freed}

    @api.post("/api/probe")
    async def probe(file: UploadFile = File(...)):
        path = await _store_upload(file)
        try:
            info = await asyncio.to_thread(_ffprobe_media, path)
        except RuntimeError as exc:
            raise HTTPException(422, _public_error(exc)) from exc
        finally:
            path.unlink(missing_ok=True)
        duration = info["duration_seconds"]
        info["filename"] = Path(file.filename or "media").name[:255]
        info["max_audio_seconds"] = MAX_AUDIO_SECONDS
        info["exceeds_audio_limit"] = bool(duration and duration > MAX_AUDIO_SECONDS)
        info["transcribable"] = bool(info["has_audio"])
        if not info["has_audio"]:
            info["reason"] = "The file has no audio track to transcribe."
        elif info["exceeds_audio_limit"]:
            info["reason"] = ("This file is longer than max_audio_seconds. "
                              "The server will process it in parts.")
        return info

    @api.post("/api/models/load")
    def load_model(request: LoadRequest):
        runtimes = set(getattr(api.state.models, "adapters", {"faster-whisper": None}))
        runtime, compute_type = _validate_model_settings(request.model, request.compute_type, 1, request.runtime,
                                                         available_runtimes=runtimes)
        with api.state.jobs_lock:
            if any(j.status == "running" for j in api.state.jobs.values()):
                raise HTTPException(409, "A transcription is running; wait for it or cancel it before loading a model")
        try:
            return api.state.models.load(request.model, compute_type, runtime)
        except Exception as exc:
            raise HTTPException(503, f"Could not load or warm up model: {_public_error(exc)}") from exc

    def record_segment(job: Job, row: dict[str, Any], duration: float) -> None:
        # Append in place; as_dict copies the list for readers.
        job.segments.append(row)
        if duration > 0:
            job.progress = round(min(99.9, 100.0 * float(row["end"]) / duration), 1)
        checkpoint_segments(job)

    def run_job(job: Job, path: Path):
        outcome: tuple[str, str | None, float | None] = ("failed", "Transcription did not finish", None)
        cancelled_in_queue = False
        try:
            with api.state.jobs_lock:
                if job.cancel_requested:
                    cancelled_in_queue = True  # cancel_queued already saved the cancelled state
                else:
                    api.state.store.save_job(replace(job, status="running"))
                    job.status = "running"
            if not cancelled_in_queue:
                job.segments = []
                job.control.on_segment = lambda row, duration: record_segment(job, row, duration)
                job.control.on_part = lambda: checkpoint_segments(job, force=True)
                rows, info = api.state.models.transcribe(
                    path, model=job.model, compute_type=job.compute_type, language=job.language,
                    beam_size=job.beam_size, vad_filter=job.vad_filter, initial_prompt=job.initial_prompt,
                    runtime=job.runtime, control=job.control)
                # Returned rows are the transcript of record. Earlier checkpoints already saved segments.
                job.segments, job.info = rows, info
                if job.diarize:
                    job.info = {**info, "diarization": _run_diarization(job, path, info.get("duration"))}
                outcome = ("cancelled" if job.cancel_requested else "completed", None, 100.0)
        except JobCancelled:
            # Keep whatever was transcribed so far so the user can review or copy it.
            outcome = ("cancelled", None, None)
        except Exception as exc:
            logger.exception("Transcription failed for job %s", job.id)
            outcome = ("failed", f"Transcription failed: {_public_error(exc)}", None)
        finally:
            path.unlink(missing_ok=True)
            _untrack_upload(path)
            if not cancelled_in_queue:
                finish(job, *outcome)

    def ensure_job_slot() -> None:
        with api.state.jobs_lock:
            active = sum(j.status in ("queued", "running") for j in api.state.jobs.values())
        if active >= MAX_ACTIVE_JOBS:
            raise HTTPException(429, f"At most {MAX_ACTIVE_JOBS} transcription job(s) may be active")

    @api.post("/api/jobs", status_code=202)
    async def create_job(file: UploadFile = File(...),
                         model: str = Form(DEFAULT_MODEL), compute_type: str = Form("int8"),
                         runtime: str | None = Form(None),
                         language: str | None = Form(None), beam_size: int = Form(5),
                         vad_filter: bool = Form(True), initial_prompt: str | None = Form(None),
                         diarize: bool = Form(False)):
        runtimes = set(getattr(api.state.models, "adapters", {"faster-whisper": None}))
        runtime, compute_type = _validate_model_settings(model, compute_type, beam_size, runtime,
                                                         available_runtimes=runtimes)
        language = _normalize_language(language)
        _require_model_language(model, runtime, language)
        _check_initial_prompt(initial_prompt)
        _require_diarization(diarize, runtime)
        ensure_job_slot()  # refuse before the upload is written to disk
        path = await _store_upload(file)
        await asyncio.to_thread(_require_audio, [path])
        accepted = _accept_controls(model, runtime, beam_size, vad_filter, initial_prompt)
        job = Job(id=str(uuid.uuid4()), model=model, runtime=runtime, compute_type=compute_type, language=language,
                  beam_size=accepted.beam_size, vad_filter=accepted.vad_filter, initial_prompt=accepted.initial_prompt,
                  ignored_controls=accepted.ignored,
                  filename=Path(file.filename or "audio").name[:255], diarize=diarize)
        try:
            with api.state.jobs_lock:
                active = sum(j.status in ("queued", "running") for j in api.state.jobs.values())
                if active >= MAX_ACTIVE_JOBS:
                    raise HTTPException(429, f"At most {MAX_ACTIVE_JOBS} transcription job(s) may be active")
                api.state.store.save_job(job)
                api.state.jobs[job.id] = job
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        _track_upload(path)
        api.state.executor.submit(run_job, job, path)
        return {"id": job.id, "status": "queued", "ignored_controls": list(job.ignored_controls)}

    @api.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        with api.state.jobs_lock:
            job = find_job(job_id)
            if job is None:
                raise HTTPException(404, "Job not found")
            if job.status == "queued":
                cancel_queued(job)
            elif job.status == "running":
                if not job.cancel_requested:
                    job.cancel_requested = True
                    job.control.cancel()
            else:
                raise HTTPException(409, f"Job has already {job.status}")
            return job.as_dict()

    @api.post("/api/jobs/{job_id}/ai")
    def run_ai_mode(job_id: str, request: AiRequest):
        """Summary or action items for a finished transcript. Only transcript text goes to NVIDIA NIM."""
        with api.state.jobs_lock:
            job = find_job(job_id)
            if job is None:
                raise HTTPException(404, "Job not found")
            status = job.status
            stored_rows = list(job.segments or [])
        if status not in ("completed", "cancelled"):
            raise HTTPException(409, f"The job is {status}; AI modes are available once it completes or is cancelled")
        if request.segments is not None and any(len(segment.text) > nim.MAX_TRANSCRIPT_CHARS for segment in request.segments):
            # Checked here, not in the schema, so the refusal keeps the same reason text as the whole-transcript limit.
            raise HTTPException(422, f"A transcript segment is longer than {nim.MAX_TRANSCRIPT_CHARS:,} characters; "
                                     f"AI modes accept at most {nim.MAX_TRANSCRIPT_CHARS:,} characters per segment")
        rows = [segment.model_dump() for segment in request.segments] if request.segments is not None else stored_rows
        transcript = nim.transcript_text(rows)
        if not transcript:
            raise HTTPException(422, "The transcript is empty, so there is nothing to summarise")
        if len(transcript) > nim.MAX_TRANSCRIPT_CHARS:
            raise HTTPException(422, f"The transcript is {len(transcript):,} characters; AI modes accept at most "
                                     f"{nim.MAX_TRANSCRIPT_CHARS:,} characters")
        keys = nim.configured_keys()
        if not keys:
            raise HTTPException(503, nim.NO_KEY_REASON)
        try:
            text = nim.generate(request.mode, transcript, keys)
        except nim.NimRequestError as exc:
            raise HTTPException(502, str(exc)) from None
        result = {"mode": request.mode, "text": text, "model": nim.MODEL_ID, "created_at": time.time()}
        with api.state.jobs_lock:
            job = find_job(job_id)
            if job is None:
                raise HTTPException(404, "Job not found")
            job.ai_results = [item for item in job.ai_results if item["mode"] != request.mode] + [result]
            api.state.store.save_job(job)
        return result

    def batch_view(batch: dict[str, Any]) -> dict[str, Any]:
        """A stored batch with its files' current state. The caller must hold jobs_lock."""
        found = [find_job(job_id) for job_id in batch["job_ids"]]
        jobs = [job.as_dict() for job in found if job is not None]
        counts = {status: 0 for status in ("queued", "running", "completed", "failed", "cancelled")}
        for job in jobs:
            counts[job["status"]] = counts.get(job["status"], 0) + 1
        return {"id": batch["id"], "created_at": batch["created_at"], "total": len(jobs), "counts": counts,
                "done": all(job["status"] in TERMINAL_STATUSES for job in jobs), "jobs": jobs}

    def batch_summary(batch_id: str) -> dict[str, Any]:
        with api.state.jobs_lock:
            batch = api.state.store.load_batch(batch_id)
            if batch is None:
                raise HTTPException(404, "Batch not found")
            return batch_view(batch)

    @api.post("/api/batches", status_code=202)
    async def create_batch(files: list[UploadFile] = File(...),
                           model: str = Form(DEFAULT_MODEL), compute_type: str = Form("int8"),
                           runtime: str | None = Form(None), language: str | None = Form(None),
                           beam_size: int = Form(5), vad_filter: bool = Form(True),
                           initial_prompt: str | None = Form(None), diarize: bool = Form(False)):
        if not files:
            raise HTTPException(422, "Choose at least one audio file")
        if len(files) > MAX_BATCH_FILES:
            raise HTTPException(422, f"A batch may contain at most {MAX_BATCH_FILES} files")
        runtimes = set(getattr(api.state.models, "adapters", {"faster-whisper": None}))
        runtime, compute_type = _validate_model_settings(model, compute_type, beam_size, runtime,
                                                         available_runtimes=runtimes)
        language = _normalize_language(language)
        _require_model_language(model, runtime, language)
        _check_initial_prompt(initial_prompt)
        _require_diarization(diarize, runtime)

        def another_job_active() -> bool:
            return any(j.status in ("queued", "running") for j in api.state.jobs.values())

        with api.state.jobs_lock:
            if another_job_active():
                raise HTTPException(429, "Another transcription is active; wait for it to finish or cancel it before starting a batch")
        stored: list[tuple[str, Path]] = []
        try:
            for upload in files:
                stored.append((Path(upload.filename or "audio").name[:255], await _store_upload(upload)))
        except Exception:
            for _, stored_path in stored:
                stored_path.unlink(missing_ok=True)
            raise
        await asyncio.to_thread(_require_audio, [stored_path for _, stored_path in stored])
        batch_id = str(uuid.uuid4())
        accepted = _accept_controls(model, runtime, beam_size, vad_filter, initial_prompt)
        jobs = [Job(id=str(uuid.uuid4()), model=model, runtime=runtime, compute_type=compute_type, language=language,
                    beam_size=accepted.beam_size, vad_filter=accepted.vad_filter, initial_prompt=accepted.initial_prompt,
                    ignored_controls=accepted.ignored,
                    filename=name, batch_id=batch_id, diarize=diarize) for name, _ in stored]
        try:
            with api.state.jobs_lock:
                if another_job_active():
                    raise HTTPException(429, "Another transcription is active; wait for it to finish or cancel it before starting a batch")
                api.state.store.create_batch(batch_id, time.time(), jobs)
                for job in jobs:
                    api.state.jobs[job.id] = job
        except BaseException:
            for _, stored_path in stored:
                stored_path.unlink(missing_ok=True)
            raise
        # One worker runs the batch in submission order; each file keeps its own job status.
        for job, (_, stored_path) in zip(jobs, stored):
            _track_upload(stored_path)
            api.state.executor.submit(run_job, job, stored_path)
        return batch_summary(batch_id)

    @api.get("/api/batches")
    def list_batches(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        """Stored batches, newest first, each with its files' metadata but no segments. Read GET /api/batches/{id} for transcripts."""
        with api.state.jobs_lock:
            stored, total = api.state.store.list_batches(limit, offset)
            summaries = []
            for batch in stored:
                summary = batch_view(batch)
                for job in summary["jobs"]:
                    del job["segments"], job["ai_results"]
                summaries.append(summary)
        return {"batches": summaries, "total": total, "limit": limit, "offset": offset}

    @api.get("/api/batches/{batch_id}")
    def get_batch(batch_id: str):
        return batch_summary(batch_id)

    @api.post("/api/batches/{batch_id}/cancel")
    def cancel_batch(batch_id: str):
        with api.state.jobs_lock:
            batch = api.state.store.load_batch(batch_id)
            if batch is None:
                raise HTTPException(404, "Batch not found")
            for job_id in batch["job_ids"]:
                job = find_job(job_id)
                if job is None:
                    continue
                if job.status == "queued":
                    cancel_queued(job)
                elif job.status == "running" and not job.cancel_requested:
                    job.cancel_requested = True
                    job.control.cancel()
        return batch_summary(batch_id)

    @api.get("/api/jobs")
    def list_jobs(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        """Stored jobs, newest first, without segments. Read GET /api/jobs/{id} for a transcript."""
        with api.state.jobs_lock:
            stored, total = api.state.store.list_jobs(limit, offset)
            items = [api.state.jobs.get(job.id, job) for job in stored]
            summaries = []
            for job in items:
                summary = job.as_dict()
                del summary["segments"], summary["ai_results"]
                summaries.append(summary)
        return {"jobs": summaries, "total": total, "limit": limit, "offset": offset}

    @api.get("/api/jobs/{job_id}")
    def get_job(job_id: str, wait: float = Query(0, ge=0, le=60)):
        """Return the job. With ?wait=N, hold the request until the job finishes or N seconds pass."""
        deadline = time.monotonic() + wait
        while True:
            with api.state.jobs_lock:
                job = find_job(job_id)
                if job is None:
                    raise HTTPException(404, "Job not found")
                snapshot = job.as_dict()
            remaining = deadline - time.monotonic()
            if snapshot["status"] in TERMINAL_STATUSES or remaining <= 0:
                return snapshot
            time.sleep(min(0.25, remaining))

    @api.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        """Remove a finished job and its transcript. A batch keeps its record without this file."""
        with api.state.jobs_lock:
            job = find_job(job_id)
            if job is None:
                raise HTTPException(404, "Job not found")
            if job.status not in TERMINAL_STATUSES:
                raise HTTPException(409, f"The job is {job.status}; cancel it and wait for it to finish before deleting it")
            api.state.store.delete_job(job_id)
            api.state.jobs.pop(job_id, None)
        api.state.store.reclaim()
        return {"id": job_id, "deleted": True}

    @api.delete("/api/batches/{batch_id}")
    def delete_batch(batch_id: str):
        """Remove a finished batch and all of its jobs."""
        with api.state.jobs_lock:
            batch = api.state.store.load_batch(batch_id)
            if batch is None:
                raise HTTPException(404, "Batch not found")
            found = [find_job(job_id) for job_id in batch["job_ids"]]
            if any(job is not None and job.status not in TERMINAL_STATUSES for job in found):
                raise HTTPException(409, "A file in this batch is queued or running; cancel the batch and wait before deleting it")
            api.state.store.delete_batch(batch_id)
            for job_id in batch["job_ids"]:
                api.state.jobs.pop(job_id, None)
        api.state.store.reclaim()
        return {"id": batch_id, "deleted": True, "jobs_deleted": len(batch["job_ids"])}

    @api.get("/api/storage")
    def storage():
        """Where results are kept and how much space they use. The path is reported as a name, never absolute."""
        return {"data_dir": data_label, **api.state.store.stats()}

    def _download(body: str, media_type: str, stem: str, fmt: str) -> Response:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:120] or "transcript"
        return Response(body, media_type=media_type,
                        headers={"Content-Disposition": f'attachment; filename="{safe}.{fmt}"'})

    @api.get("/api/jobs/{job_id}/export")
    def export_job(job_id: str, format: str = Query("txt")):
        fmt = format.lower()
        if fmt not in exporters.EXPORT_FORMATS:
            raise HTTPException(422, f"format must be one of: {', '.join(exporters.EXPORT_FORMATS)}")
        with api.state.jobs_lock:
            job = find_job(job_id)
            if job is None:
                raise HTTPException(404, "Job not found")
            status, segments, filename = job.status, list(job.segments or []), job.filename
        if status == "failed":
            raise HTTPException(409, "The job failed, so there is no transcript to export")
        if status not in ("completed", "cancelled"):
            raise HTTPException(409, f"The job is {status}; export is available once it completes or is cancelled")
        return _download(exporters.render(segments, fmt), exporters.MEDIA_TYPES[fmt],
                         Path(filename or "transcript").stem, fmt)

    @api.get("/api/batches/{batch_id}/export")
    def export_batch(batch_id: str, format: str = Query("json")):
        fmt = format.lower()
        if fmt not in exporters.BATCH_EXPORT_FORMATS:
            raise HTTPException(422, f"format must be one of: {', '.join(exporters.BATCH_EXPORT_FORMATS)}")
        summary = batch_summary(batch_id)
        if not summary["done"]:
            raise HTTPException(409, "The batch is still running; export once every file has finished")
        files = summary["jobs"]
        if fmt == "txt":
            body = exporters.batch_text(files)
        else:
            body = exporters.batch_json(batch_id, datetime.now(timezone.utc).isoformat(), files)
        return _download(body, exporters.MEDIA_TYPES[fmt], f"batch-results-{batch_id[:8]}", fmt)

    return api


_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    # `uvicorn backend.app:app` looks this up, so the app is built when a server asks for it, never on import.
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
