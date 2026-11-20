"""The folder watcher against a real loopback server, with real ffmpeg media and fake models."""
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn

from backend.app import ModelManager, create_app
from test_api import FakeModel

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                reason="ffmpeg and ffprobe are required")


def _load_watch():
    spec = importlib.util.spec_from_file_location("watch_folder", REPO / "scripts" / "watch_folder.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules, so register before exec.
    sys.modules["watch_folder"] = module
    spec.loader.exec_module(module)
    return module


watch = _load_watch()


def _make_media(path: Path, seconds: int, with_audio=True):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10"]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000"]
    command += ["-t", str(seconds), "-c:v", "mpeg4"]
    command += ["-c:a", "aac"] if with_audio else ["-an"]
    command += ["-y", str(path)]
    subprocess.run(command, check=True)
    return path


def _is_warmup(kwargs) -> bool:
    # The loader warms models with a silent array and no initial_prompt; real jobs always pass one.
    return "initial_prompt" not in kwargs


class FlakyModel(FakeModel):
    """Fails the first real transcription, then behaves like FakeModel."""

    def __init__(self, failures=1):
        super().__init__()
        self.failures = failures

    def transcribe(self, source, **kwargs):
        if not _is_warmup(kwargs) and self.failures > 0:
            self.failures -= 1
            raise RuntimeError("model crashed")
        return super().transcribe(source, **kwargs)


class SlowModel(FakeModel):
    """Yields a segment every 50 ms for 30 s, so a job is still running when the test interrupts it."""

    def transcribe(self, source, **kwargs):
        if _is_warmup(kwargs):
            return super().transcribe(source, **kwargs)

        def segments():
            for index in range(600):
                time.sleep(0.05)
                yield SimpleNamespace(start=index * 0.5, end=index * 0.5 + 0.4, text=" slow ",
                                      avg_logprob=-0.2, no_speech_prob=0.03)

        return segments(), SimpleNamespace(language="en", language_probability=0.98, duration=300.0)


@pytest.fixture()
def serve(monkeypatch):
    started = []

    def start(model=None):
        model = model or FakeModel()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        api = create_app(ModelManager(lambda *_: model))
        instance = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=instance.run, daemon=True)
        thread.start()
        for _ in range(200):
            if instance.started:
                break
            time.sleep(0.05)
        monkeypatch.setenv("STT_STUDIO_URL", f"http://127.0.0.1:{port}")
        started.append((instance, thread))
        return SimpleNamespace(api=api, model=model, url=f"http://127.0.0.1:{port}")

    yield start
    for instance, thread in started:
        instance.should_exit = True
        thread.join(timeout=5)


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _folders(tmp_path):
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    return watch_dir, tmp_path / "out"


def _watcher(watch_dir, out_dir, *extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    args = watch.build_parser().parse_args([str(watch_dir), "--output-dir", str(out_dir), *extra])
    return watch.Watcher(watch_dir, out_dir, args, watch.transcribe_args(args))


def _get_json(live, path):
    import urllib.request

    with urllib.request.urlopen(live.url + path, timeout=10) as response:
        return json.loads(response.read())


def _batch_count(live):
    return _get_json(live, "/api/storage")["batches"]


def _jobs(live):
    return _get_json(live, "/api/jobs?limit=200")["jobs"]


def _state(out_dir):
    return json.loads((out_dir / ".watch-state.json").read_text(encoding="utf-8"))["sources"]


def test_a_file_is_transcribed_only_after_two_scans_agree_on_its_size(serve, tmp_path):
    live = serve()
    watch_dir, out_dir = _folders(tmp_path)
    full = _make_media(tmp_path / "source.mp4", seconds=2)
    target = watch_dir / "talk.mp4"
    target.write_bytes(full.read_bytes()[:500])  # a copy still in progress
    watcher = _watcher(watch_dir, out_dir, "--format", "srt")
    watcher.cycle()  # first sighting
    target.write_bytes(full.read_bytes())  # the copy finishes, so the size changes
    watcher.cycle()  # differs from the previous scan: not stable yet
    assert _batch_count(live) == 0 and not (out_dir / "talk.srt").exists()
    watcher.cycle()  # unchanged since the previous scan: stable, so it is transcribed
    assert (out_dir / "talk.srt").read_text(encoding="utf-8").startswith("1\n00:00:00,250 --> 00:00:01,500")
    assert _batch_count(live) == 1


def test_completed_sources_are_not_reprocessed_and_state_is_written_atomically(serve, tmp_path, monkeypatch):
    live = serve()
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "talk.mp4")
    first = _watcher(watch_dir, out_dir)
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(watch.os, "replace", lambda src, dst: (replaced.append((Path(src).parent, Path(dst))),
                                                              real_replace(src, dst)))
    first.cycle()
    first.cycle()
    assert replaced == [(out_dir, out_dir / ".watch-state.json")]  # temp file in the same folder, then rename
    before = (out_dir / ".watch-state.json").read_text(encoding="utf-8")
    assert _state(out_dir)[0]["status"] == "completed" and _state(out_dir)[0]["output"] == str(out_dir / "talk.txt")

    monkeypatch.setattr(watch.os, "replace", real_replace)
    again = _watcher(watch_dir, out_dir)  # a fresh run reads the state file
    again.cycle()
    again.cycle()
    assert _batch_count(live) == 1
    assert (out_dir / ".watch-state.json").read_text(encoding="utf-8") == before
    assert [p.name for p in out_dir.iterdir() if p.name.endswith(".tmp")] == []


def test_a_failed_state_save_leaves_the_previous_state_intact(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = watch.WatchState(out_dir / ".watch-state.json", {})
    state.record(("a.mp4", 1, 2), "failed", error="first")
    before = (out_dir / ".watch-state.json").read_text(encoding="utf-8")

    def broken_dumps(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(watch.json, "dumps", broken_dumps)
    with pytest.raises(RuntimeError):
        state.record(("b.mp4", 3, 4), "completed", output="x")
    assert (out_dir / ".watch-state.json").read_text(encoding="utf-8") == before
    assert sorted(p.name for p in out_dir.iterdir()) == [".watch-state.json"]


def test_a_silent_video_is_recorded_as_failed_and_retried_only_on_request(serve, tmp_path, capsys):
    serve()
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "screen.mp4", seconds=1, with_audio=False), watch_dir / "screen.mp4")
    first = _watcher(watch_dir, out_dir)
    first.cycle()
    first.cycle()
    assert _state(out_dir)[0]["status"] == "failed"
    assert "no audio track" in _state(out_dir)[0]["error"]
    assert first.failed
    capsys.readouterr()

    again = _watcher(watch_dir, out_dir)
    again.cycle()
    again.cycle()
    err = capsys.readouterr().err
    assert "skipped: screen.mp4 (failed earlier" in err and "started: screen.mp4" not in err
    assert again.failed  # still failing, so the exit code stays non-zero

    retry = _watcher(watch_dir, out_dir, "--retry-failed")
    retry.cycle()
    retry.cycle()
    err = capsys.readouterr().err
    assert "started: screen.mp4" in err and "failed: screen.mp4" in err
    assert _state(out_dir)[0]["status"] == "failed"


def test_a_failed_transcription_is_completed_when_retried(serve, tmp_path):
    serve(FlakyModel(failures=1))
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "talk.mp4")
    first = _watcher(watch_dir, out_dir)
    first.cycle()
    first.cycle()
    assert _state(out_dir)[0]["status"] == "failed" and "model crashed" in _state(out_dir)[0]["error"]
    assert not (out_dir / "talk.txt").exists()

    retry = _watcher(watch_dir, out_dir, "--retry-failed")
    retry.cycle()
    retry.cycle()
    assert _state(out_dir)[0]["status"] == "completed"
    assert (out_dir / "talk.txt").read_text(encoding="utf-8").strip() == "hello"
    assert not retry.failed


def test_outputs_never_overwrite_and_get_numbered_suffixes(serve, tmp_path):
    serve()
    watch_dir, out_dir = _folders(tmp_path)
    out_dir.mkdir()
    (out_dir / "talk.srt").write_text("MINE", encoding="utf-8")
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "talk.mp4")
    shutil.copy(_make_media(tmp_path / "source.mkv", seconds=1), watch_dir / "talk.mkv")
    watcher = _watcher(watch_dir, out_dir, "--format", "srt")
    watcher.cycle()
    watcher.cycle()
    outputs = {row["name"]: Path(row["output"]).name for row in _state(out_dir)}
    assert outputs == {"talk.mkv": "talk-2.srt", "talk.mp4": "talk-3.srt"}
    assert (out_dir / "talk.srt").read_text(encoding="utf-8") == "MINE"
    assert (out_dir / "talk-2.srt").exists() and (out_dir / "talk-3.srt").exists()


def test_once_exits_zero_on_success_and_one_while_a_file_is_failed(serve, tmp_path):
    serve()
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "talk.mp4")
    base = [str(watch_dir), "--output-dir", str(out_dir), "--once", "--interval", "0.1", "--format", "srt"]
    assert watch.main(base) == 0
    assert (out_dir / "talk.srt").exists()

    shutil.copy(_make_media(tmp_path / "silent.mp4", seconds=1, with_audio=False), watch_dir / "screen.mp4")
    assert watch.main(base) == 1  # the new silent file fails
    assert watch.main(base) == 1  # it is still failed on the next run, so the run still reports it


def test_once_reports_an_unreachable_server_and_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STT_STUDIO_URL", "http://127.0.0.1:9")
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "talk.mp4")
    assert watch.main([str(watch_dir), "--output-dir", str(out_dir), "--once", "--interval", "0.05"]) == 1
    err = capsys.readouterr().err
    assert "waiting: Cannot reach STT Studio" in err
    assert not (out_dir / ".watch-state.json").exists()  # nothing is recorded as failed


def test_audio_is_refused_unless_the_server_is_on_loopback(tmp_path, monkeypatch):
    monkeypatch.setenv("STT_STUDIO_URL", "http://192.168.1.20:8765")
    watch_dir, out_dir = _folders(tmp_path)
    with pytest.raises(SystemExit) as refused:
        watch.main([str(watch_dir), "--output-dir", str(out_dir), "--once"])
    assert refused.value.code == 2


def test_symbolic_links_are_never_followed(serve, tmp_path, capsys):
    live = serve()
    watch_dir, out_dir = _folders(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), elsewhere / "private.mp4")
    shutil.copy(_make_media(tmp_path / "other.mp4", seconds=1), watch_dir / "talk.mp4")
    (watch_dir / "linked.mp4").symlink_to(elsewhere / "private.mp4")
    (watch_dir / "alias.mp4").symlink_to(watch_dir / "talk.mp4")
    assert watch.main([str(watch_dir), "--output-dir", str(out_dir), "--once", "--interval", "0.1"]) == 0
    assert [row["name"] for row in _state(out_dir)] == ["talk.mp4"]
    assert _batch_count(live) == 1
    assert "symbolic link" in capsys.readouterr().err


def test_hidden_temporary_and_unsupported_names_are_ignored(serve, tmp_path):
    live = serve()
    watch_dir, out_dir = _folders(tmp_path)
    for name in (".hidden.mp4", "~lock.mp4", "download.mp4.part", "download.mp4.crdownload", "download.mp4.download",
                 "partial.mp4.tmp", "notes.txt"):
        (watch_dir / name).write_bytes(b"not media")
    watcher = _watcher(watch_dir, out_dir)
    watcher.cycle()
    watcher.cycle()
    assert _batch_count(live) == 0 and list(out_dir.iterdir()) == []


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["ctrl-c", "launchctl-stop"])
def test_stop_signals_cancel_the_running_batch_and_exit_130(serve, tmp_path, sig):
    live = serve(SlowModel())
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=2), watch_dir / "talk.mp4")
    log_path = tmp_path / "watch.err"
    with log_path.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen([sys.executable, str(REPO / "scripts" / "watch_folder.py"), str(watch_dir),
                                 "--output-dir", str(out_dir), "--interval", "0.2", "--format", "srt"],
                                stdout=subprocess.DEVNULL, stderr=err, env=os.environ.copy())
        try:
            # Wait for a segment, not just "running": the client records the batch id a moment after the
            # server starts the job, and a segment proves that moment has passed.
            assert _wait_for(lambda: any(job.segments for job in live.api.state.jobs.values()), 30)
            proc.send_signal(sig)
            code = proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
    assert code == 130
    log = log_path.read_text(encoding="utf-8")
    assert "started: talk.mp4" in log and "interrupted" in log and "completed" not in log
    assert _wait_for(lambda: _jobs(live) and all(job["status"] == "cancelled" for job in _jobs(live)))
    assert not list(out_dir.glob("*.srt"))
    assert not (out_dir / ".watch-state.json").exists()


def test_transcribe_options_are_passed_through_as_the_cli_would_read_them():
    args = watch.build_parser().parse_args(["W", "--output-dir", "O", "--model", "tiny", "--language", "en",
                                            "--format", "vtt", "--diarize"])
    template = watch.transcribe_args(args)
    assert (template.model, template.language, template.format, template.json, template.diarize) == (
        "tiny", "en", "vtt", True, True)
