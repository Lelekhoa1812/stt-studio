#!/usr/bin/env python3
"""Watch a folder and transcribe each recording that lands in it, with no commands to type.

Standard library only. It imports scripts/stt_cli.py and runs the same transcribe code in-process, so each
file gets the same probe, chunking, speaker-label and error rules as `stt_cli.py transcribe`. The only upload
is the media file, sent to the local backend (the URL must be loopback). Transcripts go to OUT_DIR, and
OUT_DIR/.watch-state.json records what was done so nothing is transcribed twice.

Examples:
  python3 scripts/watch_folder.py ~/Recordings --output-dir ~/Transcripts --format srt
  python3 scripts/watch_folder.py ~/Recordings --output-dir ~/Transcripts --once
  python3 scripts/watch_folder.py ~/Recordings --output-dir ~/Transcripts --once --retry-failed

Exit codes: 0 done, 1 a file failed in this run or the server was unavailable under --once,
2 invalid usage, 130 interrupted by Ctrl-C (any running batch is cancelled first).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import io
import json
import math
import os
import signal
import stat
import sys
import tempfile
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stt_cli  # noqa: E402  (also puts the repository root on sys.path for backend.exporters)

STATE_NAME = ".watch-state.json"
LOCK_NAME = ".watch.lock"
TEMP_SUFFIXES = (".part", ".crdownload", ".download", ".tmp")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

Signature = tuple[int, int]     # (size in bytes, mtime in nanoseconds)
SourceKey = tuple[str, int, int]  # (file name, size, mtime in nanoseconds)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str, stream=None) -> None:
    print(f"{now_iso()} {message}", file=stream or sys.stderr, flush=True)


class _LineRelay(io.TextIOBase):
    """Stands in for stderr while a file is transcribed and logs each complete line as it is written, so a long
    file shows progress in the log instead of nothing until it finishes. `stream` is the real stderr."""

    def __init__(self, stream):
        self.stream = stream
        self.partial = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.partial += text
        *lines, self.partial = self.partial.split("\n")
        for line in lines:
            if line.strip():
                log(line, self.stream)
        return len(text)

    def finish(self) -> None:
        if self.partial.strip():
            log(self.partial, self.stream)
        self.partial = ""


class _Pause(Exception):
    """Stop this pass and try again on the next one: the server is down or busy, the file is fine."""


class _PublishError(Exception):
    """The transcript could not be written to OUT_DIR. The file is retried; the state is still consistent."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


MAX_ATTEMPTS = 3  # failed attempts on one file before it is marked failed (until --retry-failed)
DEFAULT_FILE_TIMEOUT = 3600.0  # seconds; same as the transcribe command's default budget


def _waits_for_server(exc: stt_cli.CliError) -> bool:
    """Server-wide conditions: wait and try later, without charging the file an attempt. Anything else (including
    a 5xx) is about this file, so it counts as an attempt against that file only."""
    return isinstance(exc, stt_cli.ServerUnreachable) or exc.status == 429


class WatchState:
    """Outcome of each source, persisted in the output folder so a restart does not redo finished work."""

    def __init__(self, path: Path, sources: dict[SourceKey, dict]):
        self.path = path
        self.sources = sources

    @classmethod
    def load(cls, path: Path) -> WatchState:
        if not path.exists():
            return cls(path, {})
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))["sources"]
            sources = {(row["name"], int(row["size"]), int(row["mtime_ns"])): row for row in rows}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise stt_cli.CliError(f"{path} cannot be read ({exc}). Fix or remove it; removing it transcribes "
                                   "every file again.") from None
        return cls(path, sources)

    def get(self, key: SourceKey) -> dict | None:
        return self.sources.get(key)

    def record(self, key: SourceKey, status: str, output: str | None = None, error: str | None = None,
               digest: str | None = None, attempts: int | None = None) -> None:
        name, size, mtime_ns = key
        self.sources[key] = {"name": name, "size": size, "mtime_ns": mtime_ns, "status": status,
                             "output": output, "error": error, "digest": digest, "attempts": attempts,
                             "time": now_iso()}
        self.save()

    def save(self) -> None:
        # Serialise first, then write a temporary file and rename it over the old state, so a crash never
        # leaves a half-written state file.
        text = json.dumps({"version": 1, "sources": list(self.sources.values())}, indent=2, ensure_ascii=False)
        fd, staged = tempfile.mkstemp(dir=self.path.parent, prefix=".watch-state-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged, self.path)
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise


def output_name(stem: str, number: int, fmt: str) -> str:
    """stem.fmt for the first transcript, then stem-2.fmt, stem-3.fmt, ..."""
    return f"{stem}.{fmt}" if number == 1 else f"{stem}-{number}.{fmt}"


def publish_exact(text: str, target: Path) -> bool:
    """Write a transcript to exactly this name. Returns False if the name is already taken. Never overwrites."""
    staged = target.parent / f".{target.stem}.{uuid.uuid4().hex}.tmp"
    try:
        staged.write_text(text, encoding="utf-8")
        try:
            os.link(staged, target)  # atomic, and fails if the name is already taken
        except FileExistsError:
            return False
        except OSError:  # volumes without hard links: claim the name exclusively instead
            try:
                with target.open("x", encoding="utf-8") as handle:
                    handle.write(text)
            except FileExistsError:
                return False
            except BaseException:
                target.unlink(missing_ok=True)  # do not leave a partial transcript behind
                raise
        return True
    finally:
        staged.unlink(missing_ok=True)


@dataclass(frozen=True)
class _Candidate:
    name: str
    path: Path
    size: int
    mtime_ns: int

    @property
    def signature(self) -> Signature:
        return (self.size, self.mtime_ns)

    @property
    def key(self) -> SourceKey:
        return (self.name, self.size, self.mtime_ns)


def transcribe_args(args: argparse.Namespace) -> argparse.Namespace:
    """The options `stt_cli.py transcribe` would build for these settings; the file is filled in per call."""
    argv = ["transcribe", "placeholder", "--json", "--format", args.format, "--timeout", str(args.file_timeout)]
    if args.model:
        argv += ["--model", args.model]
    if args.language:
        argv += ["--language", args.language]
    if args.diarize:
        argv.append("--diarize")
    return stt_cli.build_parser().parse_args(argv)


class Watcher:
    """Scans the watch folder and transcribes files that were unchanged between two scans."""

    def __init__(self, watch_dir: Path, out_dir: Path, args: argparse.Namespace, template: argparse.Namespace):
        self.watch_dir = watch_dir
        self.out_dir = out_dir
        self.format = args.format
        self.diarize = args.diarize
        self.retry_failed = args.retry_failed
        self.file_timeout = args.file_timeout  # seconds of transcription budget per file
        self.template = template
        self.state = WatchState.load(out_dir / STATE_NAME)
        self.extensions: frozenset[str] = frozenset()
        self.previous: dict[str, Signature] = {}
        self.blocked: str | None = None
        self.failed: set[SourceKey] = set()  # stable sources whose latest outcome is a failure
        self.run_failures = 0  # failures that happened in this run; earlier failures do not count
        self._last_pause: str | None = None
        self._announced: set[SourceKey] = set()
        self._logged_links: set[str] = set()
        self._recover_interrupted()

    def _recover_interrupted(self) -> None:
        """A run that stopped after naming a transcript but before recording it leaves 'publishing' behind.
        If that transcript is on disk with the recorded digest it was published: record it as completed.
        Otherwise nothing was published for it and it is transcribed again."""
        for key, row in list(self.state.sources.items()):
            if row["status"] != "publishing":
                continue
            output = Path(row["output"]) if row.get("output") else None
            if output is not None and output.is_file() and _sha256(output.read_bytes()) == row.get("digest"):
                self.state.record(key, "completed", output=row["output"], digest=row["digest"])
                log(f"recovered: {row['name']} was published before the last stop")
            else:  # the row stays 'publishing', which _wanted treats as work still to do
                log(f"will transcribe again: {row['name']} was not published before the last stop")

    def cycle(self) -> None:
        """One pass: check the server, scan, and transcribe the files that stopped changing."""
        self.blocked = None
        try:
            capabilities = stt_cli.get_json("/api/capabilities")
        except stt_cli.CliError as exc:
            self._pause(str(exc))
            return
        if self._last_pause:
            log("resumed: STT Studio is answering again")
            self._last_pause = None
        if self.diarize and not capabilities["diarization"]["available"]:
            self._pause(capabilities["diarization"]["reason"] or "speaker labels are not available on the server")
            return
        self.extensions = frozenset(ext.lower() for ext in capabilities["accepted_extensions"])
        try:
            stable = self._scan()
        except OSError as exc:
            self._pause(f"cannot read {self.watch_dir}: {exc.strerror or exc}")
            return
        for candidate in stable:
            if not self._wanted(candidate) or not self._unchanged(candidate):
                continue
            try:
                self._process(candidate)
            except _Pause as exc:
                self._pause(str(exc))
                return

    def _pause(self, reason: str) -> None:
        self.blocked = reason
        if reason != self._last_pause:
            self._last_pause = reason
            log(f"waiting: {reason}")

    def _scan(self) -> list[_Candidate]:
        """Return the files whose size and mtime match the previous scan. Not recursive; links are skipped."""
        current: dict[str, _Candidate] = {}
        with os.scandir(self.watch_dir) as entries:
            for entry in entries:
                name = entry.name
                if name.startswith((".", "~")) or name.lower().endswith(TEMP_SUFFIXES):
                    continue
                if entry.is_symlink():
                    if name not in self._logged_links:
                        self._logged_links.add(name)
                        log(f"skipped: {name} is a symbolic link; links are not followed")
                    continue
                if not entry.is_file(follow_symlinks=False) or Path(name).suffix[1:].lower() not in self.extensions:
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:  # removed since the listing
                    continue
                current[name] = _Candidate(name, self.watch_dir / name, info.st_size, info.st_mtime_ns)
        stable = [c for name, c in sorted(current.items()) if self.previous.get(name) == c.signature]
        self.previous = {name: c.signature for name, c in current.items()}
        return stable

    def _wanted(self, candidate: _Candidate) -> bool:
        record = self.state.get(candidate.key)
        if record is None or record["status"] in ("publishing", "retrying") or (
                record["status"] == "failed" and self.retry_failed):
            return True
        if record["status"] == "failed":
            self.failed.add(candidate.key)
        if candidate.key not in self._announced:
            self._announced.add(candidate.key)
            reason = ("already transcribed" if record["status"] == "completed"
                      else "failed earlier; use --retry-failed to try again")
            log(f"skipped: {candidate.name} ({reason})")
        return False

    def _unchanged(self, candidate: _Candidate) -> bool:
        # Check again right before work starts: a file can change, or be swapped for a link, after the scan.
        try:
            info = os.lstat(candidate.path)
        except OSError:
            return False
        return stat.S_ISREG(info.st_mode) and (info.st_size, info.st_mtime_ns) == candidate.signature

    def _process(self, candidate: _Candidate) -> None:
        log(f"started: {candidate.name}")
        started = time.monotonic()
        try:
            record = self._transcribe(candidate.path)
        except KeyboardInterrupt:
            raise
        except stt_cli.CliError as exc:
            if _waits_for_server(exc):
                raise _Pause(str(exc)) from None
            self._attempt_failed(candidate, str(exc))
            return
        except Exception as exc:  # one bad file must not stop the watcher
            self._attempt_failed(candidate, f"{exc.__class__.__name__}: {exc}")
            return
        if record["status"] != "completed":
            if time.monotonic() - started >= self.file_timeout:
                # Out of budget, not broken: it is retried like any other attempt, and is only marked failed
                # after MAX_ATTEMPTS (or on --retry-failed).
                self._attempt_failed(candidate, f"timed out after {self.file_timeout:g} s")
            else:
                self._attempt_failed(candidate, record.get("error") or f"transcription {record['status']}")
            return
        # State-save errors are not this file's fault: they propagate and stop the run, so nothing is retried
        # from a state the disk does not hold.
        try:
            self._complete(candidate, record)
        except _PublishError as exc:
            self._attempt_failed(candidate, str(exc))

    def _transcribe(self, path: Path) -> dict:
        args = copy.copy(self.template)
        args.files = [str(path)]
        stt_cli.ACTIVE_BATCHES.clear()
        stdout, relay = io.StringIO(), _LineRelay(sys.stderr)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(relay):
                stt_cli.cmd_transcribe(args)
        finally:
            relay.finish()
        return json.loads(stdout.getvalue())["files"][0]

    def _complete(self, candidate: _Candidate, record: dict) -> None:
        text = stt_cli.exporters.render(record["segments"], self.format)
        digest = _sha256(text.encode("utf-8"))
        target = self._publish(candidate, text, digest)
        self.state.record(candidate.key, "completed", output=str(target), digest=digest)
        self.failed.discard(candidate.key)
        log(f"completed: {candidate.name} -> {target.name} ({len(record['segments'])} segments)")

    def _publish(self, candidate: _Candidate, text: str, digest: str) -> Path:
        """Record the chosen name as 'publishing' BEFORE writing, then write it. If the state save fails, nothing
        has been written yet, so the retry cannot create a second copy. Names that are taken are skipped."""
        number = 1
        while True:
            target = self.out_dir / output_name(candidate.path.stem, number, self.format)
            number += 1
            if os.path.lexists(target):
                continue
            self.state.record(candidate.key, "publishing", output=str(target), digest=digest,
                              attempts=self._attempts_so_far(candidate.key))
            try:
                if publish_exact(text, target):
                    return target
            except OSError as exc:
                raise _PublishError(f"could not write {target.name}: {exc.strerror or exc}") from None

    def _attempts_so_far(self, key: SourceKey) -> int:
        record = self.state.get(key)
        if record is not None and record["status"] in ("retrying", "publishing"):
            return int(record.get("attempts") or 0)
        return 0  # new, or retried on request after a permanent failure

    def _attempt_failed(self, candidate: _Candidate, error: str) -> None:
        """One failed attempt against this file only. It is retried until MAX_ATTEMPTS, then marked failed."""
        attempts = self._attempts_so_far(candidate.key) + 1
        self.run_failures += 1
        if attempts >= MAX_ATTEMPTS:
            self.state.record(candidate.key, "failed", error=error, attempts=attempts)
            self.failed.add(candidate.key)
            log(f"failed: {candidate.name}: {error} (gave up after {attempts} attempts)")
        else:
            self.state.record(candidate.key, "retrying", error=error, attempts=attempts)
            log(f"retrying: {candidate.name}: {error} (attempt {attempts} of {MAX_ATTEMPTS} failed)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="watch_folder.py", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("watch_dir", help="Folder to watch. Subfolders and symbolic links are ignored")
    parser.add_argument("--output-dir", required=True, help="Folder for transcripts and " + STATE_NAME)
    parser.add_argument("--format", choices=stt_cli.exporters.EXPORT_FORMATS, default="txt",
                        help="Transcript format (default: txt)")
    parser.add_argument("--model", default=None, help="Model preset or Hugging Face owner/model (default: small)")
    parser.add_argument("--language", default=None, help="Language code such as en, or auto (default: auto)")
    parser.add_argument("--diarize", action="store_true",
                        help="Label speakers (local pyannote). Needs HF_TOKEN on the server; files are not split")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between scans (default: 5)")
    parser.add_argument("--once", action="store_true",
                        help="Scan twice, --interval apart, transcribe what is stable, then exit")
    parser.add_argument("--file-timeout", type=float, default=DEFAULT_FILE_TIMEOUT,
                        help="Seconds allowed per file before it is cancelled and retried (default: 3600)")
    parser.add_argument("--retry-failed", action="store_true", help="Try again the files that failed earlier")
    return parser


@contextlib.contextmanager
def single_writer(out_dir: Path):
    """Hold an exclusive lock on OUT_DIR/.watch.lock for the whole run. A second watcher, or a manual --once while
    the background agent runs, would otherwise read the same state and write duplicate transcripts."""
    lock_path = out_dir / LOCK_NAME
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            holder = handle.read().strip() or "unknown"
            raise stt_cli.CliError(f"another watcher is already writing to {out_dir} (pid {holder}); "
                                   f"wait for it to finish or stop it first") from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield


def run(args: argparse.Namespace, watch_dir: Path) -> int:
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    with single_writer(out_dir):
        watcher = Watcher(watch_dir, out_dir, args, transcribe_args(args))
        mode = "once" if args.once else f"every {args.interval:g} s"
        log(f"watching {watch_dir} -> {out_dir} ({args.format}, {mode})")
        if args.once:
            watcher.cycle()
            if watcher.blocked:
                return 1
            time.sleep(args.interval)
            watcher.cycle()
            return 1 if watcher.blocked or watcher.run_failures else 0
        while True:
            watcher.cycle()
            time.sleep(args.interval)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be greater than 0")
    if not math.isfinite(args.file_timeout) or args.file_timeout <= 0:
        parser.error("--file-timeout must be greater than 0 seconds")
    watch_dir = Path(args.watch_dir).expanduser()
    if not watch_dir.is_dir():
        parser.error(f"No such folder: {watch_dir}")
    if urllib.parse.urlsplit(stt_cli.base_url()).hostname not in LOOPBACK_HOSTS:
        parser.error("STT_STUDIO_URL must be a loopback address so that audio stays on this machine")
    def _terminate(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        stt_cli._need_tools()
        return run(args, watch_dir)
    except (stt_cli.CliError, OSError) as exc:
        log(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        stt_cli.cancel_active_batches()
        log("interrupted; any running batch was cancelled and nothing was written for it")
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
