#!/usr/bin/env python3
"""Command-line client for a running STT Studio backend, built for scripts and AI agents.

Standard library only. It talks to the local HTTP API and uses ffmpeg/ffprobe on PATH for media
planning. Start the backend first (./scripts/dev.sh, or uvicorn on 127.0.0.1:8765). Set
STT_STUDIO_URL to use another loopback address.

Examples:
  python3 scripts/stt_cli.py status
  python3 scripts/stt_cli.py transcribe meeting.mp4 --format srt --output meeting.srt
  python3 scripts/stt_cli.py transcribe interview.m4a --json            # segments with timestamps on stdout
  python3 scripts/stt_cli.py transcribe *.wav --output-dir transcripts --format txt
  python3 scripts/stt_cli.py batch submit a.wav b.wav                   # returns a batch id immediately
  python3 scripts/stt_cli.py job export <job-id> --format vtt -o captions.vtt

Exit codes: 0 success, 1 a request or transcription failed, 2 invalid usage,
130 interrupted (active batches are cancelled first).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from backend import exporters  # noqa: E402  (one formatter shared with the HTTP API)

DEFAULT_URL = "http://127.0.0.1:8765"
DEFAULT_CHUNK_SECONDS = 1200     # 20 minutes: bounded memory per job and cheap retries
BATCH_GROUP_SIZE = 10            # files per batch request; keeps each upload modest
DEFAULT_TIMEOUT = 3600           # whole-command budget in seconds for transcribe and batch waits
POLL_SECONDS = 1.0
CANCEL_GRACE_SECONDS = 60        # after a timeout, how long to wait for a cancelled batch to settle
TERMINAL = frozenset({"completed", "failed", "cancelled"})
ACTIVE_BATCHES: list[str] = []   # batches this run submitted, cancelled if the user interrupts
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})  # same rule as watch_folder.py and watch_service
LARGE_UPLOAD_BYTES = 1024 ** 3   # non-WAV inputs above this are extracted to WAV before upload
TOOL_OUTPUT_BYTE_CAP = 1024 ** 2   # ffprobe stdout/stderr: refuse unbounded reads


class CliError(Exception):
    """A failure to report to the caller. The message is safe to show directly."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ServerUnreachable(CliError):
    """The backend did not answer at all. Retrying later can succeed; the file itself is not at fault."""


# ---------------------------------------------------------------- HTTP helpers

def base_url() -> str:
    return os.environ.get("STT_STUDIO_URL", DEFAULT_URL).rstrip("/")


def _require_loopback(url: str) -> None:
    """Audio must stay on this machine, so every request goes to a loopback host only."""
    if urllib.parse.urlsplit(url).hostname not in LOOPBACK_HOSTS:
        raise CliError("STT_STUDIO_URL must be a loopback address (127.0.0.1, localhost or ::1) so that "
                       "audio stays on this machine")


def _detail(body: bytes) -> str:
    try:
        data = json.loads(body or b"{}")
    except json.JSONDecodeError:
        return body.decode("utf-8", errors="replace").strip()[:500]
    detail = data.get("detail", data) if isinstance(data, dict) else data
    return detail if isinstance(detail, str) else json.dumps(detail)[:500]


def _multipart(fields: dict[str, object], files: list[tuple[str, Path, str]]):
    """Stream a multipart body to a temporary file so large media never sits in memory."""
    boundary = uuid.uuid4().hex
    spool = tempfile.TemporaryFile()
    for name, value in fields.items():
        if value is None:
            continue
        text = ("true" if value is True else "false" if value is False else str(value))
        spool.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{text}\r\n'.encode())
    for field_name, path, upload_name in files:
        safe_name = upload_name.replace("\\", "_").replace('"', "_")
        content_type = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "video/mp4",
                        ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".webm": "video/webm",
                        ".flac": "audio/flac", ".ogg": "audio/ogg", ".opus": "audio/opus"}.get(
            path.suffix.lower(), "application/octet-stream")
        spool.write((f'--{boundary}\r\nContent-Disposition: form-data; name="{field_name}"; '
                     f'filename="{safe_name}"\r\nContent-Type: {content_type}\r\n\r\n').encode())
        with path.open("rb") as source:
            shutil.copyfileobj(source, spool, length=1024 * 1024)
        spool.write(b"\r\n")
    spool.write(f"--{boundary}--\r\n".encode())
    size = spool.tell()
    spool.seek(0)
    return spool, size, f"multipart/form-data; boundary={boundary}"


def call(method: str, path: str, *, params: dict[str, object] | None = None, fields: dict[str, object] | None = None,
         files: list[tuple[str, Path, str]] | None = None, json_body: dict | None = None,
         timeout: float = 60) -> bytes:
    base = base_url()
    _require_loopback(base)  # before any body is built or any byte is sent
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    headers: dict[str, str] = {}
    body = None
    if files is not None:
        body, size, content_type = _multipart(fields or {}, files)
        headers = {"Content-Type": content_type, "Content-Length": str(size)}
    elif json_body is not None:
        raw = json.dumps({k: v for k, v in json_body.items() if v is not None}).encode("utf-8")
        body = tempfile.TemporaryFile()
        body.write(raw)
        body.seek(0)
        headers = {"Content-Type": "application/json", "Content-Length": str(len(raw))}
    try:
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise CliError(_detail(exc.read()) or f"HTTP {exc.code} from {path}", status=exc.code) from None
    except urllib.error.URLError as exc:
        raise ServerUnreachable(f"Cannot reach STT Studio at {base_url()} ({exc.reason}). Start it with "
                                "./scripts/dev.sh or set STT_STUDIO_URL.") from None
    except OSError as exc:  # socket timeouts and resets
        raise ServerUnreachable(f"Request to {path} failed: {exc}") from None
    finally:
        if body is not None:
            body.close()


def _cancel_batch(batch_id: str) -> None:
    """Best effort: a cancel that fails must never hide the error that caused it."""
    try:
        call("POST", f"/api/batches/{batch_id}/cancel", timeout=10)
    except Exception:
        pass


def cancel_active_batches() -> None:
    """Ask the server to stop the batches this run started. Used on Ctrl-C so nothing is left running."""
    for batch_id in ACTIVE_BATCHES:
        _cancel_batch(batch_id)


def get_json(path: str, params: dict[str, object] | None = None, timeout: float = 30) -> dict:
    return json.loads(call("GET", path, params=params, timeout=timeout) or b"{}")


def post_json(path: str, json_body: dict | None = None, timeout: float = 60) -> dict:
    return json.loads(call("POST", path, json_body=json_body, timeout=timeout) or b"{}")


def emit(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- media helpers

def _need_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise CliError(f"{' and '.join(missing)} must be on PATH. On macOS: brew install ffmpeg")


def _protocol_whitelist_args() -> list[str]:
    return ["-protocol_whitelist", "file,crypto"]


def _read_pipe_bytes(pipe, limit: int) -> tuple[bytes, bool]:
    """Read at most limit bytes from a binary pipe. The second value is true if more data was waiting."""
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


def _read_pipe_all(pipe) -> bytes:
    """Read a binary pipe until EOF so a full pipe cannot stall the tool."""
    chunks: list[bytes] = []
    while True:
        piece = pipe.read(65536)
        if not piece:
            return b"".join(chunks)
        chunks.append(piece)


def _read_tool_output(pipe, cap_output_bytes: int | None) -> tuple[bytes, bool]:
    if cap_output_bytes is None:
        return _read_pipe_all(pipe), False
    return _read_pipe_bytes(pipe, cap_output_bytes)


def _signal_tool_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the tool's process group. The caller must not be in that group."""
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _terminate_tool_group(proc: subprocess.Popen) -> None:
    """Stop the tool process group. SIGTERM, then SIGKILL if it is still alive."""
    _signal_tool_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_tool_group(proc, signal.SIGKILL)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run_tool(command: list[str], timeout: float = 600, *, cap_output_bytes: int | None = None) -> subprocess.CompletedProcess:
    """Run ffmpeg or ffprobe in its own process group.

    A timeout stops that group and raises CliError. KeyboardInterrupt stops the group and is raised
    again, so the caller can cancel server work. Pipes are read while the tool runs. A capped read
    stops the group once the cap is hit, so the pipe cannot fill until the timeout.
    """
    # Its own session: a watcher SIGTERM becomes KeyboardInterrupt here and does not reach the tool.
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    ) as proc:
        stdout_holder: list[tuple[bytes, bool]] = []
        stderr_holder: list[tuple[bytes, bool]] = []

        def read_stdout() -> None:
            item = _read_tool_output(proc.stdout, cap_output_bytes)
            stdout_holder.append(item)
            if item[1]:
                # Stop here. Waiting for the timeout would fill the pipe and report the wrong error.
                _signal_tool_group(proc, signal.SIGTERM)

        def read_stderr() -> None:
            item = _read_tool_output(proc.stderr, cap_output_bytes)
            stderr_holder.append(item)
            if item[1]:
                _signal_tool_group(proc, signal.SIGTERM)

        threads = [threading.Thread(target=read_stdout), threading.Thread(target=read_stderr)]
        for thread in threads:
            thread.start()
        stopped = False
        try:
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                stopped = True
                _terminate_tool_group(proc)
                raise CliError(f"{command[0]} timed out on {Path(command[-1]).name}") from exc
            except KeyboardInterrupt:
                stopped = True
                _terminate_tool_group(proc)
                raise
        finally:
            # After a kill, do not wait forever for a reader. On success, wait until the pipes are drained.
            for thread in threads:
                thread.join(timeout=5 if stopped else None)
        stdout_bytes, stdout_trunc = stdout_holder[0]
        stderr_bytes, stderr_trunc = stderr_holder[0]
        if cap_output_bytes is not None and (stdout_trunc or stderr_trunc):
            raise CliError(f"{command[0]} output exceeded {cap_output_bytes // (1024 ** 2)} MiB on {Path(command[-1]).name}")
        return subprocess.CompletedProcess(
            command, returncode,
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
        )


def probe_local(path: Path) -> dict:
    _need_tools()
    result = _run_tool(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json",
                        *_protocol_whitelist_args(), "-i", str(path)],
                       60, cap_output_bytes=TOOL_OUTPUT_BYTE_CAP)
    if result.returncode != 0:
        raise CliError(f"{path.name}: ffprobe could not read this file ({result.stderr.strip()[:200]})")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    duration = None
    try:
        duration = float((data.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        durations = [float(s["duration"]) for s in streams if s.get("duration") not in (None, "N/A")]
        duration = max(durations) if durations else None
    return {
        "duration": duration if duration and duration > 0 else None,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        # Cover art on an MP3 is reported as a video stream; it does not make the input a video.
        "has_video": any(s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
                         for s in streams),
    }


def find_cut_points(path: Path, duration: float, chunk_seconds: float) -> list[float]:
    """Choose cut points near each target boundary, preferring the middle of a pause."""
    targets = [k * chunk_seconds for k in range(1, math.ceil(duration / chunk_seconds))]
    if not targets:
        return []
    window = min(120.0, chunk_seconds * 0.25)
    result = _run_tool(["ffmpeg", "-nostdin", "-hide_banner", "-vn", *_protocol_whitelist_args(), "-i", str(path),
                        "-af", "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"],
                       timeout=max(300.0, duration * 2))
    pauses: list[float] = []
    pending_start: float | None = None
    for line in result.stderr.splitlines():
        start = re.search(r"silence_start: (-?[\d.]+)", line)
        if start:
            pending_start = float(start.group(1))
            continue
        end = re.search(r"silence_end: (-?[\d.]+)", line)
        if end and pending_start is not None:
            pauses.append((pending_start + float(end.group(1))) / 2)
            pending_start = None
    cuts: list[float] = []
    for target in targets:
        candidates = [p for p in pauses if abs(p - target) <= window and 0 < p < duration]
        chosen = min(candidates, key=lambda p: abs(p - target)) if candidates else target
        if cuts and chosen <= cuts[-1] + 1.0:
            chosen = target
        cuts.append(chosen)
    return cuts


def _unique_output_path(directory: Path, stem: str, extension: str, taken: set[str]) -> Path:
    """Name an output so two inputs with the same stem never overwrite each other: talk.txt, talk-2.txt, ..."""
    number = 1
    while True:
        name = f"{stem}.{extension}" if number == 1 else f"{stem}-{number}.{extension}"
        if name.casefold() not in taken:  # case-insensitive, because macOS folders usually are
            taken.add(name.casefold())
            return directory / name
        number += 1


def _check_output_file(target: Path) -> None:
    """Fail before any upload if the result file could not be written."""
    if target.is_dir():
        raise CliError(f"Cannot write {target}: it is a folder")
    if target.exists():
        if not os.access(target, os.W_OK):
            raise CliError(f"Cannot write {target}: permission denied")
    elif not target.parent.is_dir():
        raise CliError(f"Cannot write {target}: folder {target.parent} does not exist")
    elif not os.access(target.parent, os.W_OK):
        raise CliError(f"Cannot write {target}: permission denied")


def _check_output_dir(directory: Path) -> None:
    """Fail before any upload if the output folder is missing and cannot be created, or is not writable."""
    existing = directory
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.is_dir() or not os.access(existing, os.W_OK):
        raise CliError(f"Cannot write into {directory}: it is not a writable folder")


def extract_chunk(source: Path, start: float, length: float | None, target: Path) -> None:
    """Write mono 16 kHz WAV audio from source. A length of None means 'to the end of the file'."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}"]
    if length is not None:
        command += ["-t", f"{length:.3f}"]
    command += [*_protocol_whitelist_args(), "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                "-y", str(target)]
    result = _run_tool(command)
    if result.returncode != 0 or not target.exists():
        raise CliError(f"{source.name}: could not extract audio from {start:.0f}s ({result.stderr.strip()[:200]})")


# ---------------------------------------------------------------- batch orchestration

@dataclass
class WorkItem:
    source_index: int
    chunk_index: int
    path: Path          # file to upload (original media, or an extracted WAV chunk)
    upload_name: str
    offset: float       # seconds to add to this chunk's segment times
    length: float | None = None  # chunk duration; segment times are clamped to it


def whole_file_item(source: Path, info: dict, index: int, work_dir: str) -> WorkItem:
    """Plan the upload for a file that is not split. Videos and very large non-WAV files go up as WAV audio."""
    length = info["duration"] or None
    if info["has_video"] or (source.suffix.lower() != ".wav" and source.stat().st_size > LARGE_UPLOAD_BYTES):
        # Videos are uploaded as their audio track only, so a short clip cannot hit the upload cap.
        wav_path = Path(work_dir) / f"{index:03d}-whole.wav"
        extract_chunk(source, 0.0, None, wav_path)
        return WorkItem(index, 0, wav_path, f"{source.stem}.wav", 0.0, length)
    return WorkItem(index, 0, source, source.name, 0.0, length)


def _model_fields(args) -> dict[str, object]:
    language = None if not args.language or args.language.lower() == "auto" else args.language
    return {
        "model": args.model,
        "runtime": args.runtime,
        "compute_type": args.compute_type,
        "language": language,
        "beam_size": args.beam_size,
        "vad_filter": not args.no_vad,
        "initial_prompt": args.prompt or "",
        "diarize": bool(getattr(args, "diarize", False)),
    }


def submit_batch(items: list[WorkItem], args, deadline: float) -> dict:
    files = [("files", item.path, item.upload_name) for item in items]
    while True:
        try:
            return json.loads(call("POST", "/api/batches", fields=_model_fields(args), files=files, timeout=900))
        except CliError as exc:
            # 429 means another transcription is running. Wait for it instead of failing.
            if exc.status != 429 or time.monotonic() >= deadline:
                raise
            time.sleep(3)


def wait_for_batch(batch_id: str, total: int, deadline: float, quiet: bool) -> tuple[dict, bool]:
    """Poll until the batch finishes. On timeout, cancel it and return (state, timed_out)."""
    last_done = -1
    cancel_grace: float | None = None
    while True:
        state = get_json(f"/api/batches/{batch_id}")
        done = sum(job["status"] in TERMINAL for job in state["jobs"])
        if done != last_done and not quiet:
            print(f"[stt] {done}/{total} files finished", file=sys.stderr)
            last_done = done
        if state["done"]:
            return state, cancel_grace is not None
        now = time.monotonic()
        if cancel_grace is None and now >= deadline:
            call("POST", f"/api/batches/{batch_id}/cancel")
            cancel_grace = now + CANCEL_GRACE_SECONDS
        if cancel_grace is not None and now >= cancel_grace:
            return state, True
        time.sleep(POLL_SECONDS)


def run_work_items(items: list[WorkItem], args, deadline: float) -> dict[int, dict]:
    """Run items in order, in groups. Returns the job record for each item index that ran."""
    results: dict[int, dict] = {}
    for start in range(0, len(items), BATCH_GROUP_SIZE):
        group = items[start:start + BATCH_GROUP_SIZE]
        batch = submit_batch(group, args, deadline)
        ACTIVE_BATCHES.append(batch["id"])
        # Always on stderr, so a --json run that times out still tells the caller which batch to export.
        print(f"[stt] batch {batch['id']} started with {len(group)} file(s)", file=sys.stderr)
        finished = False
        try:
            state, timed_out = wait_for_batch(batch["id"], len(group), deadline, quiet=args.json)
            finished = True
        finally:
            if not finished:
                # Any failure while waiting (not only Ctrl-C) must not leave the server busy for later runs.
                _cancel_batch(batch["id"])
        for offset, job in enumerate(state["jobs"]):
            results[start + offset] = job
        if timed_out:
            if not args.json:
                print("[stt] time budget reached; remaining chunks were not run", file=sys.stderr)
            break
    return results


def cmd_transcribe(args) -> int:
    sources = [Path(name).expanduser() for name in args.files]
    for source in sources:
        if not source.is_file():
            raise CliError(f"No such file: {source}")
    if len(sources) > 1 and not (args.output_dir or args.json):
        raise CliError("Several files need --output-dir (or --json). Use one of them.", status=None)
    if args.output and len(sources) > 1:
        raise CliError("--output applies to one file; use --output-dir for several.")
    _need_tools()  # a missing ffmpeg is a setup problem, not a per-file failure
    # Check destinations now, so a bad path fails before any upload rather than after a long transcription.
    if args.output:
        _check_output_file(Path(args.output))
    if args.output_dir:
        _check_output_dir(Path(args.output_dir))

    deadline = time.monotonic() + args.timeout
    items: list[WorkItem] = []
    per_source: list[dict] = []
    if args.diarize:
        # Speaker numbers are per job, so a file must go through one job, not be split into chunks.
        state = get_json("/api/capabilities")
        if not state["diarization"]["available"]:
            raise CliError(state["diarization"]["reason"] or "Speaker labels are not available on this server")
    with tempfile.TemporaryDirectory(prefix="stt-cli-") as work_dir:
        for index, source in enumerate(sources):
            mark = len(items)
            record = {"source": str(source), "status": "pending", "error": None, "duration_seconds": None,
                      "language": None, "segments": [], "output": None}
            per_source.append(record)
            try:
                info = probe_local(source)
                record["duration_seconds"] = info["duration"]
                if not info["has_audio"]:
                    raise CliError("The file has no audio track to transcribe.")
                duration = info["duration"] or 0.0
                if duration > args.chunk_seconds and not args.diarize:
                    cuts = find_cut_points(source, duration, args.chunk_seconds)
                    bounds = [0.0, *cuts, duration]
                    # A sliver at the end (encoder padding) is folded into the previous chunk.
                    if len(bounds) > 2 and bounds[-1] - bounds[-2] < min(5.0, args.chunk_seconds * 0.25):
                        del bounds[-2]
                    if not args.json and len(bounds) > 2:
                        print(f"[stt] {source.name}: {duration / 60:.1f} min, splitting into {len(bounds) - 1} chunks",
                              file=sys.stderr)
                    for chunk_index in range(len(bounds) - 1):
                        chunk_path = Path(work_dir) / f"{index:03d}-{chunk_index:03d}.wav"
                        chunk_length = bounds[chunk_index + 1] - bounds[chunk_index]
                        extract_chunk(source, bounds[chunk_index], chunk_length, chunk_path)
                        items.append(WorkItem(index, chunk_index, chunk_path,
                                              f"{source.stem}-part{chunk_index + 1:02d}.wav",
                                              bounds[chunk_index], chunk_length))
                else:
                    items.append(whole_file_item(source, info, index, work_dir))
            except CliError as exc:
                # One bad file must not stop the others: drop anything planned for it and record the reason.
                del items[mark:]
                record.update(status="failed", error=str(exc))

        results = run_work_items(items, args, deadline) if items else {}

        for index, record in enumerate(per_source):
            if record["status"] == "failed":
                continue
            mine = [(n, item) for n, item in enumerate(items) if item.source_index == index]
            problems = []
            segments = []
            for position, item in mine:
                job = results.get(position)
                if job is None or job["status"] != "completed":
                    reason = (job or {}).get("error") or f"file {job['status'] if job else 'was not started'}"
                    problems.append(f"chunk {item.chunk_index + 1}: {reason}")
                    continue
                job_info = job.get("info") or {}
                record["language"] = record["language"] or job_info.get("language")
                if job_info.get("diarization") is not None:
                    record["diarization"] = job_info["diarization"]
                for row in job.get("segments") or []:
                    # Whisper can report an end a little past the chunk; clamp so captions never overlap.
                    start = row["start"]
                    end = row["end"]
                    if item.length is not None:
                        if start >= item.length - 0.01:
                            continue
                        end = min(end, item.length)
                        start = min(start, end)
                    segment = {"start": start + item.offset, "end": end + item.offset, "text": row["text"]}
                    if row.get("speaker"):
                        segment["speaker"] = row["speaker"]
                    segments.append(segment)
            record["segments"] = segments
            record["status"] = "completed" if not problems else "incomplete"
            record["error"] = "; ".join(problems) or None

    for record in per_source:
        report = record.get("diarization") or {}
        if report.get("status") == "failed":
            # The transcript is still valid; only the speaker labels are missing.
            print(f"warning: {record['source']}: no speaker labels ({report.get('reason')})", file=sys.stderr)
    exit_code = 0
    taken_names: set[str] = set()
    for index, record in enumerate(per_source):
        if record["status"] != "completed":
            exit_code = 1
            continue
        text = exporters.render(record["segments"], args.format)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            record["output"] = str(Path(args.output).resolve())
        elif args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(record["source"]).stem
            target = _unique_output_path(out_dir, stem, args.format, taken_names)
            if target.name != f"{stem}.{args.format}" and not args.json:
                print(f"[stt] {record['source']} is written as {target.name}; {stem}.{args.format} was already used",
                      file=sys.stderr)
            target.write_text(text, encoding="utf-8")
            record["output"] = str(target.resolve())
        elif not args.json:
            sys.stdout.write(text)
        if args.json:
            record["text"] = exporters.to_txt(record["segments"]).rstrip("\n")

    if args.json:
        # Keep the schema stable for agents: "error" is always present, null on success.
        emit({"format": args.format, "files": [
            {k: v for k, v in record.items() if v is not None or k == "error"} for record in per_source]})
    elif exit_code:
        failed = [record for record in per_source if record["status"] != "completed"]
        for record in failed:
            print(f"error: {record['source']}: {record['error']}", file=sys.stderr)
        print(f"[stt] summary: {len(per_source) - len(failed)} of {len(per_source)} file(s) succeeded",
              file=sys.stderr)
    return exit_code


# ---------------------------------------------------------------- command handlers

def cmd_status(args) -> int:
    emit(get_json("/api/health"))
    return 0


def cmd_capabilities(args) -> int:
    emit(get_json("/api/capabilities"))
    return 0


def cmd_models(args) -> int:
    emit(get_json("/api/models"))
    return 0


def cmd_models_cache_list(args) -> int:
    data = get_json("/api/models/cache")
    if args.json:
        emit(data)
        return 0
    models = data.get("models") or []
    if not models:
        print("No catalog models.")
        return 0
    for entry in models:
        cached = "yes" if entry.get("cached") else "no"
        print(f"{entry.get('id', '-')}  cached {cached}  {_human_size(entry.get('sizeBytes'))}")
    return 0


def cmd_models_cache_delete(args) -> int:
    result = json.loads(call("DELETE", f"/api/models/cache/{_path_id(args.model_id)}") or b"{}")
    if args.json:
        emit(result)
    else:
        print(f"Freed {_human_size(result.get('freedBytes'))}.")
    return 0


def cmd_load(args) -> int:
    body = {"model": args.model, "compute_type": args.compute_type, "runtime": args.runtime}
    emit(post_json("/api/models/load", body, timeout=1800))
    return 0


def cmd_unload(args) -> int:
    emit(post_json("/api/models/unload"))
    return 0


def cmd_probe(args) -> int:
    results = []
    for name in args.files:
        path = Path(name).expanduser()
        if not path.is_file():
            raise CliError(f"No such file: {path}")
        results.append(json.loads(call("POST", "/api/probe", files=[("file", path, path.name)], timeout=900)))
    emit(results[0] if len(results) == 1 else results)
    return 0


def cmd_batch_submit(args) -> int:
    paths = [Path(name).expanduser() for name in args.files]
    for path in paths:
        if not path.is_file():
            raise CliError(f"No such file: {path}")
    _need_tools()
    # The server rejects the whole batch if any one file has no audio, so check them all first.
    problems = []
    probed: list[tuple[Path, dict]] = []
    for path in paths:
        try:
            info = probe_local(path)
        except CliError as exc:
            problems.append(str(exc))
            continue
        if not info["has_audio"]:
            problems.append(f"{path.name}: the file has no audio track to transcribe")
        else:
            probed.append((path, info))
    if problems:
        raise CliError("Nothing was submitted. Remove or fix these files and try again:\n  " + "\n  ".join(problems))
    # Same upload rules as transcribe: videos go up as their WAV audio track, not as the whole container.
    with tempfile.TemporaryDirectory(prefix="stt-cli-") as work_dir:
        items = [whole_file_item(path, info, index, work_dir) for index, (path, info) in enumerate(probed)]
        batch = submit_batch(items, args, time.monotonic() + 600)
    emit(batch)
    return 0


def cmd_batch_status(args) -> int:
    path = f"/api/batches/{args.batch_id}"
    if args.wait is None:
        emit(get_json(path))
        return 0
    deadline = time.monotonic() + args.wait
    state = None
    while True:
        state = get_json(path)
        if state.get("done"):
            emit(state)
            return 0
        if time.monotonic() >= deadline:
            emit(state)
            return 1
        time.sleep(1)


def cmd_batch_cancel(args) -> int:
    emit(json.loads(call("POST", f"/api/batches/{args.batch_id}/cancel") or b"{}"))
    return 0


def _write_export(body: bytes, output: str | None) -> None:
    if output:
        Path(output).write_bytes(body)
        print(f"Wrote {Path(output).resolve()}", file=sys.stderr)
    else:
        sys.stdout.buffer.write(body)


def cmd_batch_export(args) -> int:
    body = call("GET", f"/api/batches/{args.batch_id}/export", params={"format": args.format})
    _write_export(body, args.output)
    return 0


def cmd_job_status(args) -> int:
    state = get_json(f"/api/jobs/{args.job_id}", params={"wait": args.wait}, timeout=args.wait + 30)
    emit(state)
    return 1 if state.get("status") == "failed" else 0


def cmd_job_cancel(args) -> int:
    emit(json.loads(call("POST", f"/api/jobs/{args.job_id}/cancel") or b"{}"))
    return 0


def cmd_job_export(args) -> int:
    body = call("GET", f"/api/jobs/{args.job_id}/export", params={"format": args.format})
    _write_export(body, args.output)
    return 0


def _path_id(value: str) -> str:
    """Keep an id from changing the route it is sent to (a slash must not reach another endpoint)."""
    return urllib.parse.quote(value, safe="")


def _when(timestamp: object) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(timestamp)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "-"


def _human_size(count: object) -> str:
    size = float(count or 0)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def cmd_jobs_list(args) -> int:
    data = get_json("/api/jobs", params={"limit": args.limit, "offset": args.offset})
    if args.json:
        emit(data)
        return 0
    jobs = data.get("jobs") or []
    total = data.get("total", len(jobs))
    if not jobs:
        print("No jobs stored." if not total else f"No jobs at offset {args.offset}; {total} job(s) in total.")
        return 0
    print(f"{'ID':<36}  {'STATUS':<10}  {'CREATED':<16}  FILE")
    for job in jobs:
        print(f"{job.get('id', '-'):<36}  {job.get('status', '-'):<10}  {_when(job.get('created_at')):<16}  "
              f"{job.get('filename') or '-'}")
    print(f"Showing {args.offset + 1}-{args.offset + len(jobs)} of {total} job(s).")
    return 0


def cmd_jobs_delete(args) -> int:
    result = json.loads(call("DELETE", f"/api/jobs/{_path_id(args.job_id)}") or b"{}")
    if args.json:
        emit(result)
    else:
        print(f"Deleted job {args.job_id}.")
    return 0


def cmd_batches_list(args) -> int:
    # GET /api/batches is the batch-history route; the response is {"batches": [...]} or a bare list.
    data = get_json("/api/batches", params={"limit": args.limit, "offset": args.offset})
    if args.json:
        emit(data)
        return 0
    if isinstance(data, list):
        batches, total = data, len(data)
    else:
        batches = data.get("batches") or []
        total = data.get("total", len(batches))
    if not batches:
        print("No batches stored." if not total else f"No batches at offset {args.offset}; {total} batch(es) in total.")
        return 0
    print(f"{'ID':<36}  {'CREATED':<16}  {'FILES':>5}  STATUS")
    for batch in batches:
        counts = batch.get("counts") or {}
        state = "done" if batch.get("done") else "running"
        summary = ", ".join(f"{name} {number}" for name, number in counts.items() if number) or "-"
        print(f"{batch.get('id', '-'):<36}  {_when(batch.get('created_at')):<16}  {batch.get('total', '-'):>5}  "
              f"{state} ({summary})")
    print(f"Showing {args.offset + 1}-{args.offset + len(batches)} of {total} batch(es).")
    return 0


def cmd_batches_delete(args) -> int:
    result = json.loads(call("DELETE", f"/api/batches/{_path_id(args.batch_id)}") or b"{}")
    if args.json:
        emit(result)
    else:
        print(f"Deleted batch {args.batch_id} and its {result.get('jobs_deleted', 0)} job(s).")
    return 0


def cmd_ai(args) -> int:
    # Only transcript text is sent to the AI provider, never audio. The server answers 503 with the reason if no key is set.
    # The API also accepts a cancelled job. The CLI does not: a partial transcript is not summarised.
    job_path = f"/api/jobs/{_path_id(args.job_id)}"
    status = get_json(job_path).get("status")
    if status != "completed":
        raise CliError(f"Job {args.job_id} is {status or 'unknown'}; AI modes need a completed job.")
    result = post_json(f"{job_path}/ai", {"mode": args.mode}, timeout=900)
    if args.json:
        emit(result)
    else:
        print((result.get("text") or "").rstrip())
    return 0


def cmd_storage(args) -> int:
    data = get_json("/api/storage")
    if args.json:
        emit(data)
        return 0
    print(f"Data folder: {data.get('data_dir', '-')}")
    print(f"Jobs:        {data.get('jobs', 0)}")
    print(f"Batches:     {data.get('batches', 0)}")
    print(f"Disk used:   {_human_size(data.get('bytes'))}")
    return 0


# ---------------------------------------------------------------- argument parsing

def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="small", help="Model preset or Hugging Face owner/model (default: small)")
    parser.add_argument("--runtime", default=None, help="faster-whisper or mlx-audio; usually inferred from the model")
    parser.add_argument("--compute-type", default="int8", help="int8, int8_float16, float16 or float32 (default: int8)")
    parser.add_argument("--language", default=None, help="Language code such as en, or auto (default: auto)")
    parser.add_argument("--beam-size", type=int, default=5, help="1-20 (default: 5)")
    parser.add_argument("--no-vad", action="store_true", help="Disable voice-activity filtering")
    parser.add_argument("--prompt", default=None, help="Vocabulary hint, e.g. names and jargon")


def _positive_seconds(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value}") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def _wait_seconds(value: str) -> float:
    """Seconds for job status --wait. The server rejects anything outside 0 to 60."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value}") from None
    if not math.isfinite(number) or number < 0 or number > 60:
        raise argparse.ArgumentTypeError("must be between 0 and 60")
    return number


def _int_in_range(low: int, high: int | None = None):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a whole number: {value}") from None
        if number < low or (high is not None and number > high):
            bound = f"between {low} and {high}" if high is not None else f"{low} or more"
            raise argparse.ArgumentTypeError(f"must be {bound}")
        return number
    return parse


def _format_option(parser: argparse.ArgumentParser, default: str = "txt") -> None:
    parser.add_argument("--format", choices=exporters.EXPORT_FORMATS, default=default,
                        help=f"Output format (default: {default})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stt_cli.py", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sub.add_parser("status", help="Service health and active jobs").set_defaults(func=cmd_status)
    sub.add_parser("capabilities", help="Limits, formats, runtimes and endpoint list").set_defaults(func=cmd_capabilities)
    models = sub.add_parser("models", help="Runnable models and cache state")
    models_sub = models.add_subparsers(dest="models_command", metavar="ACTION")
    cache = models_sub.add_parser("cache", help="Inspect or delete cached model weights on disk")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True, metavar="ACTION")
    cache_list = cache_sub.add_parser("list", help="List catalog models and whether weights are cached")
    cache_list.add_argument("--json", action="store_true", help="Print the JSON response instead of a table")
    cache_list.set_defaults(func=cmd_models_cache_list)
    cache_delete = cache_sub.add_parser("delete", help="Delete one model's cached weights")
    cache_delete.add_argument("model_id", help="Catalog model id (slashes are allowed)")
    cache_delete.add_argument("--json", action="store_true", help="Print the JSON response")
    cache_delete.set_defaults(func=cmd_models_cache_delete)
    models.set_defaults(func=cmd_models)

    load = sub.add_parser("load", help="Download (if needed) and load a model")
    load.add_argument("model", nargs="?", default="small")
    load.add_argument("--compute-type", default="int8")
    load.add_argument("--runtime", default=None)
    load.set_defaults(func=cmd_load)

    sub.add_parser("unload", help="Free resident models and accelerator memory").set_defaults(func=cmd_unload)

    probe = sub.add_parser("probe", help="Inspect media streams and duration on the server (no transcription)")
    probe.add_argument("files", nargs="+")
    probe.set_defaults(func=cmd_probe)

    transcribe = sub.add_parser(
        "transcribe", help="Transcribe files (audio or video); long files are split automatically")
    transcribe.add_argument("files", nargs="+")
    _add_model_options(transcribe)
    _format_option(transcribe)
    transcribe.add_argument("--output", "-o", help="Write one file to this path")
    transcribe.add_argument("--output-dir", help="Write one file per input into this directory")
    transcribe.add_argument("--json", action="store_true",
                            help="Print a JSON summary with timestamped segments instead of plain text")
    transcribe.add_argument("--chunk-seconds", type=_positive_seconds, default=DEFAULT_CHUNK_SECONDS,
                            help=f"Split inputs longer than this (default: {DEFAULT_CHUNK_SECONDS})")
    transcribe.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                            help=f"Whole-command budget in seconds (default: {DEFAULT_TIMEOUT})")
    transcribe.add_argument("--diarize", action="store_true",
                            help="Label speakers (local pyannote). Needs HF_TOKEN on the server; files are not split")
    transcribe.set_defaults(func=cmd_transcribe)

    batch = sub.add_parser("batch", help="Queue several files and manage the batch")
    batch_sub = batch.add_subparsers(dest="batch_command", required=True, metavar="ACTION")
    submit = batch_sub.add_parser("submit", help="Queue files and return the batch id without waiting")
    submit.add_argument("files", nargs="+")
    _add_model_options(submit)
    submit.add_argument("--diarize", action="store_true",
                        help="Label speakers in each file (local pyannote). Needs HF_TOKEN on the server")
    submit.set_defaults(func=cmd_batch_submit)
    status = batch_sub.add_parser("status", help="Show a batch")
    status.add_argument("batch_id")
    status.add_argument("--wait", type=_positive_seconds, default=None,
                        help="Poll about once a second until the batch finishes or this budget expires")
    status.set_defaults(func=cmd_batch_status)
    cancel = batch_sub.add_parser("cancel", help="Cancel remaining files in a batch")
    cancel.add_argument("batch_id")
    cancel.set_defaults(func=cmd_batch_cancel)
    export = batch_sub.add_parser("export", help="Download a finished batch")
    export.add_argument("batch_id")
    export.add_argument("--format", choices=exporters.BATCH_EXPORT_FORMATS, default="json")
    export.add_argument("--output", "-o")
    export.set_defaults(func=cmd_batch_export)

    job = sub.add_parser("job", help="Inspect, cancel or export one transcription job")
    job_sub = job.add_subparsers(dest="job_command", required=True, metavar="ACTION")
    job_status = job_sub.add_parser("status", help="Show a job; --wait long-polls until it finishes")
    job_status.add_argument("job_id")
    job_status.add_argument("--wait", type=_wait_seconds, default=0,
                            help="Seconds to wait for completion, 0-60 (default: 0)")
    job_status.set_defaults(func=cmd_job_status)
    job_cancel = job_sub.add_parser("cancel", help="Cancel a queued or running job")
    job_cancel.add_argument("job_id")
    job_cancel.set_defaults(func=cmd_job_cancel)
    job_export = job_sub.add_parser("export", help="Download a finished job")
    job_export.add_argument("job_id")
    _format_option(job_export)
    job_export.add_argument("--output", "-o")
    job_export.set_defaults(func=cmd_job_export)

    jobs = sub.add_parser("jobs", help="List or delete stored jobs")
    jobs_sub = jobs.add_subparsers(dest="jobs_command", required=True, metavar="ACTION")
    jobs_list = jobs_sub.add_parser("list", help="List stored jobs, newest first")
    jobs_list.add_argument("--limit", type=_int_in_range(1, 200), default=50,
                           help="Jobs per page, 1-200 (default: 50)")
    jobs_list.add_argument("--offset", type=_int_in_range(0), default=0, help="Jobs to skip (default: 0)")
    jobs_list.add_argument("--json", action="store_true", help="Print the JSON response instead of a table")
    jobs_list.set_defaults(func=cmd_jobs_list)
    jobs_delete = jobs_sub.add_parser("delete", help="Delete a finished job and its transcript")
    jobs_delete.add_argument("job_id")
    jobs_delete.add_argument("--json", action="store_true", help="Print the JSON response")
    jobs_delete.set_defaults(func=cmd_jobs_delete)

    batches_history = sub.add_parser("batches", help="List or delete stored batches")
    batches_history_sub = batches_history.add_subparsers(dest="batches_command", required=True, metavar="ACTION")
    batches_list = batches_history_sub.add_parser("list", help="List stored batches")
    batches_list.add_argument("--limit", type=_int_in_range(1, 200), default=50,
                              help="Batches per page, 1-200 (default: 50)")
    batches_list.add_argument("--offset", type=_int_in_range(0), default=0, help="Batches to skip (default: 0)")
    batches_list.add_argument("--json", action="store_true", help="Print the JSON response instead of a table")
    batches_list.set_defaults(func=cmd_batches_list)
    batches_delete = batches_history_sub.add_parser("delete", help="Delete a finished batch and all of its jobs")
    batches_delete.add_argument("batch_id")
    batches_delete.add_argument("--json", action="store_true", help="Print the JSON response")
    batches_delete.set_defaults(func=cmd_batches_delete)

    ai = sub.add_parser("ai", help="Summary or action items for a finished job (sends transcript text to NVIDIA NIM)",
                        description="Summary or action items for a completed job, via NVIDIA NIM. Only the "
                                    "transcript text is sent, never the audio. Needs NVIDIA_NIM_API_KEY_1 on the server.")
    ai_sub = ai.add_subparsers(dest="ai_command", required=True, metavar="MODE")
    ai_summary = ai_sub.add_parser("summary", help="Summarise a completed job")
    ai_summary.add_argument("job_id")
    ai_summary.add_argument("--json", action="store_true", help="Print the JSON response instead of the text")
    ai_summary.set_defaults(func=cmd_ai, mode="summary")
    ai_items = ai_sub.add_parser("action-items", help="List the action items in a completed job")
    ai_items.add_argument("job_id")
    ai_items.add_argument("--json", action="store_true", help="Print the JSON response instead of the text")
    ai_items.set_defaults(func=cmd_ai, mode="action_items")

    storage = sub.add_parser("storage", help="Count stored jobs and batches and show disk use")
    storage.add_argument("--json", action="store_true", help="Print the JSON response instead of totals")
    storage.set_defaults(func=cmd_storage)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    wants_json = bool(getattr(args, "json", False))
    ACTIVE_BATCHES.clear()
    try:
        return args.func(args)
    except (CliError, OSError) as exc:
        # OSError covers unwritable outputs and full disks: one clean line, never a traceback.
        status = exc.status if isinstance(exc, CliError) else None
        if wants_json:
            emit({"error": str(exc), "status": status})
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Ctrl-C must not leave the server working on files nobody will collect (it would block new work with 429).
        cancel_active_batches()
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
