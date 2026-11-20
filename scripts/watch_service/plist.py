#!/usr/bin/env python3
"""Build, check and read the launchd agent that runs scripts/watch_folder.py in the background.

Standard library only. The shell scripts in this folder call it, so every value in the plist is escaped by
plistlib and never by the shell. Nothing here starts, loads or stops a job; that is left to launchctl, which
the user runs.

  plist.py write --plist PATH --log-dir DIR --watch DIR --output DIR --format F --interval S
                 [--model M] [--language L] [--diarize] [--url URL]
      Validates the settings, writes the plist to a staging file next to PATH and prints the staging path.
  plist.py state PATH   Prints missing, generated, or foreign (a plist this tool did not write).
  plist.py show PATH    Prints the settings of a generated plist.
  plist.py log PATH     Prints the log file of a generated plist.
"""
from __future__ import annotations

import argparse
import math
import os
import plistlib
import shlex
import sys
import tempfile
import urllib.parse
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import watch_folder  # noqa: E402  (the loopback rule and the export formats come from the watcher and the CLI)

LABEL = "com.sttstudio.watch"
MARKER_KEY = "com.sttstudio.generated-by"
MARKER_VALUE = "stt-studio/scripts/watch_service"
LOG_NAME = "stt-studio-watch.log"
PATH_VALUE = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
THROTTLE_SECONDS = 30
DEFAULT_URL = "http://127.0.0.1:8765"


class Refusal(Exception):
    """A setting that must not be installed. The message is shown to the user as it is."""


def _require_absolute(value: str, name: str) -> str:
    if not value.startswith("/"):
        raise Refusal(f"{name} must be an absolute path (start with /), got {value!r}")
    return value


def _absolute(value: str, name: str) -> str:
    return os.path.realpath(_require_absolute(value, name))


def _inside(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def _loopback(url: str) -> str:
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
    except ValueError as exc:
        raise Refusal(f"--url is not a valid URL: {exc}") from None
    if parts.scheme not in ("http", "https") or host not in watch_folder.LOOPBACK_HOSTS:
        raise Refusal("--url must be a loopback address (http or https on 127.0.0.1, localhost or ::1) "
                      "so that audio stays on this machine")
    return url


def _interval(text: str) -> str:
    try:
        value = float(text)
    except ValueError:
        raise Refusal(f"--interval must be a number of seconds, got {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise Refusal("--interval must be greater than 0")
    return f"{value:g}"


def _option(value: str, name: str) -> str:
    if not value or value.startswith("-"):
        raise Refusal(f"{name} needs a value that does not start with '-'")
    return value


def plist_state(path: Path) -> str:
    if not os.path.lexists(path):
        return "missing"
    try:
        data = plistlib.loads(path.read_bytes())
    except Exception:
        return "foreign"
    generated = isinstance(data, dict) and data.get(MARKER_KEY) == MARKER_VALUE
    return "generated" if generated else "foreign"


def _load_generated(path: Path) -> dict:
    if plist_state(path) != "generated":
        raise Refusal(f"{path} was not written by this tool")
    return plistlib.loads(path.read_bytes())


def build_plist(*, watch: str, output: str, log: str, fmt: str, interval: str, model: str | None,
                language: str | None, diarize: bool, url: str | None) -> dict:
    command = ["/usr/bin/env", "python3", str(SCRIPTS / "watch_folder.py"), watch,
               "--output-dir", output, "--format", fmt]
    if model:
        command += ["--model", model]
    if language:
        command += ["--language", language]
    if diarize:
        command.append("--diarize")
    command += ["--interval", interval]
    environment = {"PATH": PATH_VALUE}
    if url:
        environment["STT_STUDIO_URL"] = url
    return {
        "Label": LABEL,
        MARKER_KEY: MARKER_VALUE,
        "ProgramArguments": command,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": THROTTLE_SECONDS,
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "EnvironmentVariables": environment,
    }


def cmd_write(args: argparse.Namespace) -> str:
    plist = Path(_require_absolute(args.plist, "--plist"))
    watch = _absolute(args.watch, "WATCH_DIR")
    output = _absolute(args.output, "OUTPUT_DIR")
    log_dir = _absolute(args.log_dir, "STT_WATCH_LOG_DIR")
    if not Path(watch).is_dir():
        raise Refusal(f"WATCH_DIR is not a folder: {args.watch}")
    if _inside(output, watch):
        raise Refusal("OUTPUT_DIR must not be inside WATCH_DIR (or be the same folder): the watcher would read its own transcripts")
    if args.format not in watch_folder.stt_cli.exporters.EXPORT_FORMATS:
        raise Refusal(f"--format must be one of {', '.join(watch_folder.stt_cli.exporters.EXPORT_FORMATS)}")
    interval = _interval(args.interval)
    url = _loopback(args.url) if args.url else None
    model = _option(args.model, "--model") if args.model else None
    language = _option(args.language, "--language") if args.language else None
    if plist_state(plist) == "foreign":
        raise Refusal(f"{plist} exists and was not written by this tool; move it aside first")
    data = build_plist(watch=watch, output=output, log=str(Path(log_dir) / LOG_NAME), fmt=args.format,
                       interval=interval, model=model, language=language, diarize=args.diarize, url=url)
    plist.parent.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(dir=plist.parent, prefix=f".{plist.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            plistlib.dump(data, handle, fmt=plistlib.FMT_XML)
        os.chmod(staged, 0o644)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return staged


def cmd_state(args: argparse.Namespace) -> str:
    return plist_state(Path(args.plist))


def cmd_show(args: argparse.Namespace) -> str:
    data = _load_generated(Path(args.plist))
    keep_alive = data["KeepAlive"]
    keep_text = ("restart after any non-zero exit (SuccessfulExit false)" if keep_alive == {"SuccessfulExit": False}
                 else repr(keep_alive))
    url = data["EnvironmentVariables"].get("STT_STUDIO_URL", f"{DEFAULT_URL} (default)")
    lines = [
        f"Plist: {args.plist}",
        f"Label: {data['Label']}",
        f"Command: {shlex.join(data['ProgramArguments'])}",
        f"RunAtLoad: {'yes' if data['RunAtLoad'] else 'no'}",
        f"KeepAlive: {keep_text}",
        f"ThrottleInterval: {data['ThrottleInterval']} s",
        f"STT_STUDIO_URL: {url}",
        f"Log: {data['StandardErrorPath']}",
        "Loaded in launchd: not checked (this tool never calls launchctl)",
    ]
    return "\n".join(lines)


def cmd_log(args: argparse.Namespace) -> str:
    return _load_generated(Path(args.plist))["StandardErrorPath"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plist.py", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    write = sub.add_parser("write", help="validate the settings and write a staging plist")
    write.add_argument("--plist", required=True)
    write.add_argument("--log-dir", required=True)
    write.add_argument("--watch", required=True)
    write.add_argument("--output", required=True)
    write.add_argument("--format", required=True)
    write.add_argument("--interval", required=True)
    write.add_argument("--model")
    write.add_argument("--language")
    write.add_argument("--diarize", action="store_true")
    write.add_argument("--url")
    write.set_defaults(handler=cmd_write)
    for name, handler in (("state", cmd_state), ("show", cmd_show), ("log", cmd_log)):
        single = sub.add_parser(name)
        single.add_argument("plist")
        single.set_defaults(handler=handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(args.handler(args))
    except Refusal as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
