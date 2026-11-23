"""Local FastAPI wrapper around faster-whisper."""
from __future__ import annotations

import errno
import asyncio
import gc
import importlib.util
import json
import logging
import os
import ipaddress
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
                         "capabilities": model_capabilities(runtime_id, family.get("capabilities"))})
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
    """

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self.on_segment: Callable[[dict[str, Any], float], None] | None = None

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
    beam_size: int = 5
    vad_filter: bool = True
    initial_prompt: str | None = None
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
                "diarize", "ai_results")
        data = {k: getattr(self, k) for k in keys}
        # Copy partial segments so the response never shares a list the worker is still appending to.
        data["segments"] = list(data["segments"] or [])
        data["ai_results"] = list(data["ai_results"] or [])
        return data


STORAGE_UNAVAILABLE = "Local job storage is unavailable; check disk space and file permissions"
INTERRUPTED_MESSAGE = "Interrupted by a server restart"
DATABASE_NAME = "stt-studio.sqlite"
SCHEMA_VERSION = 2
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
}
JOB_UPSERT = ("INSERT OR REPLACE INTO jobs (id, status, created_at, model, runtime, compute_type, language, "
              "beam_size, vad_filter, segments, info, error, progress, cancel_requested, filename, batch_id, diarize, "
              "ai_results) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
SUMMARY_SELECT = ("SELECT id, status, created_at, model, runtime, compute_type, language, beam_size, vad_filter, "
                  "'[]' AS segments, info, error, progress, cancel_requested, filename, batch_id, diarize, "
                  "'[]' AS ai_results FROM jobs")


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
            job.beam_size, int(job.vad_filter), json.dumps(list(job.segments or [])), json.dumps(job.info),
            job.error, job.progress, int(job.cancel_requested), job.filename, job.batch_id, int(job.diarize),
            json.dumps(list(job.ai_results or [])))


def _job_from_row(row: sqlite3.Row) -> Job:
    return Job(id=row["id"], status=row["status"], created_at=row["created_at"], model=row["model"],
               runtime=row["runtime"], compute_type=row["compute_type"], language=row["language"],
               beam_size=row["beam_size"], vad_filter=bool(row["vad_filter"]),
               segments=json.loads(row["segments"]), info=json.loads(row["info"]), error=row["error"],
               progress=row["progress"], cancel_requested=bool(row["cancel_requested"]),
               filename=row["filename"], batch_id=row["batch_id"], diarize=bool(row["diarize"]),
               ai_results=json.loads(row["ai_results"]))


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
        """Fail every job that was queued or running when the previous server process stopped."""
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
        return {"id": row["id"], "created_at": row["created_at"], "job_ids": json.loads(row["job_ids"])}

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
                    "runtime": runtime, "warmup": "complete", "device": os.environ.get("WHISPER_DEVICE", "cpu")}

    def transcribe(self, path: Path, *, model: str, compute_type: str, language: str | None,
                   beam_size: int, vad_filter: bool, initial_prompt: str | None,
                   runtime: str = "faster-whisper", control: JobControl | None = None):
        if control is not None:
            control.check()
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
                    if control is not None:
                        control.check()
                    rows, meta = adapter.transcribe(
                        instance, audio, language=language, beam_size=beam_size,
                        vad_filter=vad_filter, initial_prompt=initial_prompt,
                        **({"control": control} if control is not None else {}),
                    )
            finally:
                audio._mmap.close()
        return rows, meta

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


def _check_initial_prompt(initial_prompt: str | None) -> None:
    if initial_prompt is not None and len(initial_prompt) > MAX_INITIAL_PROMPT_CHARS:
        raise HTTPException(422, f"initial_prompt must be at most {MAX_INITIAL_PROMPT_CHARS} characters")


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
        reason = re.sub(r"/[^\s'\"]+", "<path>", _public_error(exc))
        return {**report, "status": "failed", "speakers": 0, "reason": reason}
    return {**report, "status": "completed", "speakers": diarization.attach_speakers(job.segments, turns)}


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


def _ffprobe_media(path: Path) -> dict[str, Any]:
    """Describe every stream in a media file (audio, video, subtitles) without decoding it."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-protocol_whitelist", "file,crypto,data",
             "-show_format", "-show_streams", "-of", "json", str(path)],
            capture_output=True, check=True, timeout=30,
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
    {"method": "POST", "path": "/api/probe", "purpose": "Inspect a media file's streams and duration (no transcription)"},
    {"method": "POST", "path": "/api/jobs", "purpose": "Queue one transcription from an uploaded file"},
    {"method": "GET", "path": "/api/jobs", "purpose": "Stored jobs, newest first, metadata only without segments (?limit=&offset=)"},
    {"method": "GET", "path": "/api/jobs/{job_id}", "purpose": "Job state, progress and segments; ?wait=SECONDS long-polls until finished"},
    {"method": "DELETE", "path": "/api/jobs/{job_id}", "purpose": "Delete a finished job and its transcript (409 while queued or running)"},
    {"method": "POST", "path": "/api/jobs/{job_id}/cancel", "purpose": "Cancel a queued or running job (partial text is kept)"},
    {"method": "POST", "path": "/api/jobs/{job_id}/ai", "purpose": "Summary or action items for a finished job via NVIDIA NIM; sends transcript text only (503 without a key)"},
    {"method": "GET", "path": "/api/jobs/{job_id}/export", "purpose": "Download a finished job as txt, srt, vtt or json (?format=)"},
    {"method": "POST", "path": "/api/batches", "purpose": "Queue up to N files that run one after another"},
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


def _sweep_stale_temp_files(max_age_seconds: float = 3600) -> None:
    """Remove uploads and diarization copies left by a crash or kill, so a recording does not linger in /tmp."""
    cutoff = time.time() - max_age_seconds
    root = Path(tempfile.gettempdir())
    for pattern in ("whisper-upload-*", "stt-diarize-*"):
        for path in root.glob(pattern):
            try:
                if path.stat().st_mtime > cutoff:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError:
                continue


def create_app(manager: ModelManager | None = None) -> FastAPI:
    _sweep_stale_temp_files()
    api = FastAPI(title="STT Studio Local API", version="1.0.0")
    api.add_middleware(LocalAndBoundedUploadMiddleware)
    api.state.models = manager or ModelManager()
    # Jobs still queued or running live here; finished jobs are read back from the store.
    api.state.jobs = {}
    api.state.store = JobStore(_data_dir())
    api.state.store.mark_interrupted()
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

    def finish(job: Job, status: str, error: str | None = None, progress: float | None = None) -> None:
        """Persist the final state, then apply it in memory. Readers hold jobs_lock, so they never see a
        status the database does not hold."""
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
                "Long recordings above max_audio_seconds must be split into chunks by the caller; "
                "scripts/stt_cli.py does this automatically.",
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
        info["transcribable"] = info["has_audio"] and not info["exceeds_audio_limit"]
        if not info["has_audio"]:
            info["reason"] = "The file has no audio track to transcribe."
        elif info["exceeds_audio_limit"]:
            info["reason"] = ("Longer than the single-job limit; split it into chunks (scripts/stt_cli.py does this). "
                                 "Speaker labels need one job, so raise WHISPER_MAX_AUDIO_SECONDS to label this file.")
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
                rows, info = api.state.models.transcribe(
                    path, model=job.model, compute_type=job.compute_type, language=job.language,
                    beam_size=job.beam_size, vad_filter=job.vad_filter, initial_prompt=job.initial_prompt,
                    runtime=job.runtime, control=job.control)
                # Transcript stays in memory until the final state is saved with it.
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
        _check_initial_prompt(initial_prompt)
        _require_diarization(diarize, runtime)
        ensure_job_slot()  # refuse before the upload is written to disk
        path = await _store_upload(file)
        await asyncio.to_thread(_require_audio, [path])
        job = Job(id=str(uuid.uuid4()), model=model, runtime=runtime, compute_type=compute_type, language=language,
                  beam_size=beam_size, vad_filter=vad_filter, initial_prompt=initial_prompt,
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
        api.state.executor.submit(run_job, job, path)
        return {"id": job.id, "status": "queued"}

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

    def batch_summary(batch_id: str) -> dict[str, Any]:
        with api.state.jobs_lock:
            batch = api.state.store.load_batch(batch_id)
            if batch is None:
                raise HTTPException(404, "Batch not found")
            found = [find_job(job_id) for job_id in batch["job_ids"]]
            jobs = [job.as_dict() for job in found if job is not None]
        counts = {status: 0 for status in ("queued", "running", "completed", "failed", "cancelled")}
        for job in jobs:
            counts[job["status"]] = counts.get(job["status"], 0) + 1
        return {"id": batch_id, "created_at": batch["created_at"], "total": len(jobs), "counts": counts,
                "done": all(job["status"] in TERMINAL_STATUSES for job in jobs), "jobs": jobs}

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
        jobs = [Job(id=str(uuid.uuid4()), model=model, runtime=runtime, compute_type=compute_type, language=language,
                    beam_size=beam_size, vad_filter=vad_filter, initial_prompt=initial_prompt,
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
            api.state.executor.submit(run_job, job, stored_path)
        return batch_summary(batch_id)

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


app = create_app()
