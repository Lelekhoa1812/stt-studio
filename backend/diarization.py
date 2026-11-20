"""Speaker labels from a local pyannote pipeline, run in its own interpreter (scripts/setup-diarizer.sh).

Audio is converted to 16 kHz mono WAV in a temporary directory. A failure is raised as DiarizationError; the
job keeps its transcript and reports the reason in info.diarization.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_ID = "pyannote/speaker-diarization-3.1"
MODEL_LICENSE = "MIT"  # Read from the Hugging Face model card; re-check it before changing MODEL_ID.
CLI_PATH = Path(__file__).with_name("diarize_cli.py")
SAMPLE_RATE = 16000


class DiarizationError(Exception):
    """A diarization run that could not produce speaker turns. The message is safe to show."""


def _read_env_file(name: str) -> str | None:
    """Read one KEY=value from the project's .env. Accepts export, quotes and trailing comments; an unreadable file counts as absent."""
    try:
        lines = (ROOT_DIR / ".env").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, raw = line.partition("=")
        if not separator or key.strip() != name:
            continue
        value = raw.strip()
        if value[:1] in ("'", '"') and value.count(value[0]) >= 2:
            value = value[1:value.index(value[0], 1)]
        else:
            value = value.split(" #", 1)[0].strip()
        return value or None
    return None


def hf_token() -> str | None:
    """HF_TOKEN from the environment, or from the project's .env file. Never logged or returned to clients."""
    return os.environ.get("HF_TOKEN", "").strip() or _read_env_file("HF_TOKEN")


def diarizer_python() -> Path | None:
    # An explicitly configured interpreter is authoritative; a bad path must not silently fall back.
    configured = os.environ.get("STT_STUDIO_DIARIZER_PYTHON", "").strip()
    candidate = Path(configured) if configured else ROOT_DIR / ".local" / "diarizer-venv" / "bin" / "python"
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def availability() -> dict[str, Any]:
    """Whether speaker labels can be produced right now, and what to do if not."""
    reason = None
    if diarizer_python() is None:
        reason = "The diarizer environment is missing; run scripts/setup-diarizer.sh."
    elif hf_token() is None:
        reason = "HF_TOKEN is not set; add it to .env (the account must have accepted the model terms)."
    return {"available": reason is None, "model": MODEL_ID, "license": MODEL_LICENSE, "reason": reason}


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=5)
    except Exception:
        pass


def _to_wav(source: Path, target: Path) -> None:
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-c:a", "pcm_s16le", "-y", str(target)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as exc:
        raise DiarizationError("Preparing audio for speaker labels timed out") from exc
    if result.returncode != 0 or not target.is_file():
        raise DiarizationError("Could not prepare audio for speaker labels")


def diarize_file(source: Path, *, seconds: float | None = None,
                 should_stop: Callable[[], bool] | None = None,
                 cache_dir: Path | None = None) -> list[tuple[float, float, str]]:
    """Speaker turns as (start_seconds, end_seconds, raw_label). Raises DiarizationError on any failure."""
    python = diarizer_python()
    token = hf_token()
    if python is None or token is None:
        raise DiarizationError(availability()["reason"] or "Speaker labels are not configured")
    limit = max(300.0, (seconds or 0.0) * 1.5)
    env = {**os.environ, "HF_TOKEN": token}
    if cache_dir is not None:
        env["HF_HUB_CACHE"] = str(cache_dir)
    with tempfile.TemporaryDirectory(prefix="stt-diarize-") as directory:
        wav_path = Path(directory) / "audio.wav"
        out_path = Path(directory) / "turns.json"
        log_path = Path(directory) / "diarizer.log"
        _to_wav(source, wav_path)
        try:
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    [str(python), str(CLI_PATH), str(wav_path), str(out_path)],
                    stdout=subprocess.DEVNULL, stderr=log, stdin=subprocess.DEVNULL, env=env,
                    start_new_session=True,  # its own process group, so a cancel or timeout stops all of it
                )
        except OSError as exc:
            raise DiarizationError(f"Could not start the diarizer ({type(exc).__name__})") from exc
        deadline = time.monotonic() + limit
        while True:
            try:
                process.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() > deadline:
                    _kill_group(process)
                    raise DiarizationError("Speaker labelling timed out")
                if should_stop is not None and should_stop():
                    _kill_group(process)
                    raise DiarizationError("Speaker labelling was cancelled")
        if process.returncode != 0 or not out_path.is_file():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            except OSError:
                lines = []
            # The last log line is usually an exception. Remove the token and any file paths before it can reach a client.
            tail = lines[-1].replace(token, "<token>") if lines else ""
            tail = re.sub(r"/[^\s'\"]+", "<path>", tail)[:200]
            raise DiarizationError(f"Speaker labelling failed (exit {process.returncode})" + (f": {tail}" if tail else ""))
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DiarizationError("Speaker labelling returned unreadable output") from exc
    raw = payload.get("turns") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        raise DiarizationError("Speaker labelling returned unexpected output")
    turns = []
    for item in raw:
        if isinstance(item, list) and len(item) == 3 and isinstance(item[2], str):
            try:
                start, end = int(item[0]), int(item[1])
            except (TypeError, ValueError):
                continue
            if end > start >= 0:
                turns.append((start / 1000.0, end / 1000.0, item[2]))
    return turns


def attach_speakers(segments: list[dict[str, Any]], turns: list[tuple[float, float, str]]) -> int:
    """Label each segment with the speaker who talks longest inside it. Returns the number of speakers applied.

    Labels are numbered "Speaker 1", "Speaker 2", ... in order of each speaker's first turn. A speaker whose
    turns overlap no segment is not counted.
    """
    if not turns:
        return 0
    first_seen: dict[str, float] = {}
    for start, _end, raw in sorted(turns):
        first_seen.setdefault(raw, start)
    names = {raw: f"Speaker {index}" for index, raw in enumerate(sorted(first_seen, key=first_seen.get), start=1)}
    applied: set[str] = set()
    for segment in segments:
        talk: dict[str, float] = {}
        for start, end, raw in turns:
            overlap = min(end, float(segment["end"])) - max(start, float(segment["start"]))
            if overlap > 0:
                talk[raw] = talk.get(raw, 0.0) + overlap
        if talk:
            label = names[max(talk, key=talk.get)]
            segment["speaker"] = label
            applied.add(label)
    return len(applied)


