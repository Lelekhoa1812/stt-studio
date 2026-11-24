"""Speaker labels from a local pyannote pipeline, run in its own interpreter (scripts/setup-diarizer.sh).

Audio is converted to 16 kHz mono WAV in a temporary directory. A failure is raised as DiarizationError; the
job keeps its transcript and reports the reason in info.diarization.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
import uuid
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


_lock = threading.Lock()  # one request at a time, and worker starts and stops
_worker: _Worker | None = None
DEFAULT_IDLE_SECONDS = 300.0
_idle_timer: threading.Timer | None = None
_idle_token: object | None = None


def _idle_seconds() -> float:
    """How long the resident diarizer may wait between requests. STT_STUDIO_DIARIZER_IDLE_SECONDS overrides it."""
    try:
        value = float(os.environ.get("STT_STUDIO_DIARIZER_IDLE_SECONDS", "").strip() or DEFAULT_IDLE_SECONDS)
    except ValueError:
        return DEFAULT_IDLE_SECONDS
    return value if value > 0 else DEFAULT_IDLE_SECONDS


def _cancel_idle_timer() -> None:
    global _idle_timer, _idle_token
    if _idle_timer is not None:
        _idle_timer.cancel()
    _idle_timer = None
    _idle_token = None


def _arm_idle_timer() -> None:
    """Call with _lock held. After the idle limit with no request, the resident diarizer is stopped."""
    global _idle_timer, _idle_token
    _cancel_idle_timer()
    token = object()
    timer = threading.Timer(_idle_seconds(), _idle_expired, args=(token,))
    timer.daemon = True
    _idle_timer, _idle_token = timer, token
    timer.start()


def _idle_expired(token: object) -> None:
    with _lock:
        if _idle_token is not token:
            return  # a request started or re-armed the timer after this one fired
        stop_diarizer()


class _Worker:
    def __init__(self, process: subprocess.Popen, python: Path, token: str, cache_dir: Path | None) -> None:
        self.process = process
        self.token = token
        self.key = _worker_key(python, token, cache_dir)
        self.replies: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read_replies, daemon=True).start()

    def _read_replies(self) -> None:
        for line in self.process.stdout:
            self.replies.put(line)
        self.replies.put(None)


def _worker_key(python: Path, token: str, cache_dir: Path | None) -> tuple[str, str, str]:
    return (str(python), token, str(cache_dir))


def _discard(worker: _Worker) -> int | None:
    global _worker
    if _worker is worker:
        _worker = None
    _kill_group(worker.process)
    try:
        worker.process.stdin.close()
    except (OSError, ValueError):
        pass
    return worker.process.returncode


def stop_diarizer() -> None:
    """Stop the resident diarizer. Unload, process exit and the idle limit call it; the next job starts a new one."""
    _cancel_idle_timer()
    worker = _worker
    if worker is not None:
        _discard(worker)


atexit.register(stop_diarizer)


GATED_MESSAGE = (
    "Speaker labelling failed: the Hugging Face token cannot access the gated pyannote models. Accept the terms of "
    "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 on Hugging Face with the account that owns "
    "HF_TOKEN, then try again."
)
_GATED_ERROR = re.compile(r"GatedRepoError|\b40[13]\b|\bgated\b|\brestricted\b", re.IGNORECASE)


def _is_gated_refusal(text: str) -> bool:
    """Whether diarizer output says the token lacks access to the gated model repositories."""
    return _GATED_ERROR.search(text) is not None


# An absolute path starts at a slash that opens a word and runs to the end of its line, so folder names with spaces
# (for example "/Users/me/My Music/model.bin") are replaced as a whole and none of them survives.
_ABSOLUTE_PATH = re.compile(r"""(^|[\s'"(\[=])/.*""", re.MULTILINE)


def redact_paths(text: str) -> str:
    """Replace each absolute path, through the end of its line, with <path>."""
    return _ABSOLUTE_PATH.sub(lambda match: match.group(1) + "<path>", text)


def _safe_tail(text: str, token: str) -> str:
    """The last line of diarizer output, without the token or any local path, for a client message."""
    lines = text.strip().splitlines()
    tail = lines[-1].replace(token, "<token>") if lines else ""
    return redact_paths(tail)[:200]


def _read_reply(worker: _Worker, deadline: float, should_stop: Callable[[], bool] | None) -> dict[str, Any]:
    while True:
        try:
            line = worker.replies.get(timeout=1.0)
        except queue.Empty:
            if time.monotonic() > deadline:
                _discard(worker)
                raise DiarizationError("Speaker labelling timed out")
            if should_stop is not None and should_stop():
                _discard(worker)
                raise DiarizationError("Speaker labelling was cancelled")
            continue
        break
    if line is None:
        code = _discard(worker)
        raise DiarizationError(f"Speaker labelling failed (the diarizer exited with code {code})")
    try:
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise ValueError("reply is not an object")
    except ValueError as exc:
        _discard(worker)
        raise DiarizationError("Speaker labelling failed (unreadable reply from the diarizer)") from exc
    return reply


_WORKER_ENV_NAMES = frozenset({
    "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "HF_HOME", "HF_HUB_CACHE", "PYTHONPATH",
    "STT_STUDIO_DIARIZER_THRESHOLD",
    # Hooks the existing fake-worker tests in backend/tests/test_diarization.py pass to their child process.
    "FAKE_CONTROL", "FAKE_LOG", "FAKE_CHILD",
})


def _worker_environment(token: str, cache_dir: Path | None) -> dict[str, str]:
    """The diarizer's environment: an allowlist, so backend secrets such as NVIDIA_NIM_API_KEY_* never reach it."""
    env = {name: value for name, value in os.environ.items() if name in _WORKER_ENV_NAMES}
    env["HF_TOKEN"] = token
    if cache_dir is not None:
        env["HF_HUB_CACHE"] = str(cache_dir)
    return env


def _start_worker(python: Path, token: str, cache_dir: Path | None, deadline: float,
                  should_stop: Callable[[], bool] | None) -> _Worker:
    env = _worker_environment(token, cache_dir)
    try:
        process = subprocess.Popen(
            [str(python), str(CLI_PATH)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, text=True, encoding="utf-8",
            start_new_session=True,  # its own process group, so a cancel, timeout or stop reaches all of it
        )
    except OSError as exc:
        raise DiarizationError(f"Could not start the diarizer ({type(exc).__name__})") from exc
    worker = _Worker(process, python, token, cache_dir)
    ready = _read_reply(worker, deadline, should_stop)
    if ready.get("ready") is not True:
        _discard(worker)
        raw_error = str(ready.get("error", ""))
        if _is_gated_refusal(raw_error):
            raise DiarizationError(GATED_MESSAGE)
        detail = _safe_tail(raw_error, token)
        raise DiarizationError("Speaker labelling failed" + (f": {detail}" if detail else ""))
    return worker


def _worker_for(python: Path, token: str, cache_dir: Path | None, deadline: float,
                should_stop: Callable[[], bool] | None) -> _Worker:
    global _worker
    worker = _worker
    if worker is not None and (worker.key != _worker_key(python, token, cache_dir) or worker.process.poll() is not None):
        _discard(worker)
        worker = None
    if worker is None:
        worker = _start_worker(python, token, cache_dir, deadline, should_stop)
        _worker = worker
    return worker


def _request(worker: _Worker, audio: Path, out: Path, log: Path, deadline: float,
             should_stop: Callable[[], bool] | None) -> None:
    request_id = uuid.uuid4().hex
    message = json.dumps({"id": request_id, "audio": str(audio), "out": str(out), "log": str(log)})
    try:
        worker.process.stdin.write(message + "\n")
        worker.process.stdin.flush()
    except (OSError, ValueError) as exc:
        _discard(worker)
        raise DiarizationError("Speaker labelling failed (the diarizer is not running)") from exc
    reply = _read_reply(worker, deadline, should_stop)
    if reply.get("id") != request_id:
        _discard(worker)
        raise DiarizationError("Speaker labelling failed (unexpected reply from the diarizer)")
    if reply.get("ok") is not True:
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if _is_gated_refusal(text):
            raise DiarizationError(GATED_MESSAGE)
        detail = _safe_tail(text, worker.token)
        raise DiarizationError("Speaker labelling failed" + (f": {detail}" if detail else ""))


def _wav_duration_cap() -> float:
    """Seconds of audio ffmpeg may write. Same ceiling as transcription, plus its five-second margin."""
    from backend.app import MAX_AUDIO_SECONDS
    return MAX_AUDIO_SECONDS + 5


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


def _to_wav(source: Path, target: Path, should_stop: Callable[[], bool] | None = None) -> None:
    # file,crypto only: the data protocol can smuggle another input. -t stops a huge decode.
    command = ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,crypto",
               "-i", str(source), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-c:a", "pcm_s16le", "-t", str(_wav_duration_cap()), "-y", str(target)]
    timeout = max(600, int(_wav_duration_cap() * 3))
    if should_stop is None:
        try:
            # The process budget must cover the duration cap. A fixed 600 s kills a long, allowed file.
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise DiarizationError("Preparing audio for speaker labels timed out") from exc
        if result.returncode != 0 or not target.is_file():
            raise DiarizationError("Could not prepare audio for speaker labels")
        return
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            code = process.poll()
            if code is not None:
                if code != 0 or not target.is_file():
                    raise DiarizationError("Could not prepare audio for speaker labels")
                return
            if should_stop():
                _terminate_ffmpeg_group(process)
                raise DiarizationError("Preparing audio for speaker labels was cancelled")
            if time.monotonic() >= deadline:
                _terminate_ffmpeg_group(process)
                raise DiarizationError("Preparing audio for speaker labels timed out")
            time.sleep(0.05)
    except DiarizationError:
        raise


def diarize_file(source: Path, *, seconds: float | None = None,
                 should_stop: Callable[[], bool] | None = None,
                 cache_dir: Path | None = None) -> list[tuple[float, float, str]]:
    """Speaker turns as (start_seconds, end_seconds, raw_label). Raises DiarizationError on any failure."""
    python = diarizer_python()
    token = hf_token()
    if python is None or token is None:
        raise DiarizationError(availability()["reason"] or "Speaker labels are not configured")
    limit = max(300.0, (seconds or 0.0) * 1.5)
    with tempfile.TemporaryDirectory(prefix="stt-diarize-") as directory:
        wav_path = Path(directory) / "audio.wav"
        out_path = Path(directory) / "turns.json"
        log_path = Path(directory) / "diarizer.log"
        _to_wav(source, wav_path, should_stop)
        deadline = time.monotonic() + limit
        with _lock:
            _cancel_idle_timer()  # a request is in progress; the idle clock restarts when it finishes
            try:
                worker = _worker_for(python, token, cache_dir, deadline, should_stop)
                _request(worker, wav_path, out_path, log_path, deadline, should_stop)
            finally:
                _arm_idle_timer()
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

    Labels are numbered "Speaker 1", "Speaker 2", ... in order of each speaker's first labelled segment. Only
    speakers who win at least one segment are numbered, so a turn that falls in silence does not take a number.
    """
    if not turns:
        return 0
    winners: list[tuple[dict[str, Any], str]] = []
    for segment in segments:
        talk: dict[str, float] = {}
        for start, end, raw in turns:
            overlap = min(end, float(segment["end"])) - max(start, float(segment["start"]))
            if overlap > 0:
                talk[raw] = talk.get(raw, 0.0) + overlap
        if talk:
            winners.append((segment, max(talk, key=talk.get)))
    first_labelled: dict[str, float] = {}
    for segment, raw in winners:
        start = float(segment["start"])
        first_labelled[raw] = min(start, first_labelled.get(raw, start))
    ordered = sorted(first_labelled, key=lambda raw: (first_labelled[raw], raw))
    names = {raw: f"Speaker {index}" for index, raw in enumerate(ordered, start=1)}
    for segment, raw in winners:
        segment["speaker"] = names[raw]
    return len(names)


