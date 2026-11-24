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
    # Two saves (the name is recorded before writing, then completed): each is a temp file then a rename.
    assert replaced and all(entry == (out_dir, out_dir / ".watch-state.json") for entry in replaced)
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
    for _ in range(4):  # first sighting, then three attempts
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
    for _ in range(4):
        retry.cycle()
    err = capsys.readouterr().err
    assert "started: screen.mp4" in err and "failed: screen.mp4" in err
    assert _state(out_dir)[0]["status"] == "failed"


def test_a_failed_transcription_is_completed_when_retried(serve, tmp_path):
    serve(FlakyModel(failures=3))  # fails all three automatic attempts
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "talk.mp4")
    first = _watcher(watch_dir, out_dir)
    for _ in range(4):  # first sighting, then three attempts
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


def test_once_exits_one_only_for_failures_that_happen_in_this_run(serve, tmp_path):
    serve()
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "talk.mp4")
    base = [str(watch_dir), "--output-dir", str(out_dir), "--once", "--interval", "0.1", "--format", "srt"]
    assert watch.main(base) == 0
    assert (out_dir / "talk.srt").exists()

    shutil.copy(_make_media(tmp_path / "silent.mp4", seconds=1, with_audio=False), watch_dir / "screen.mp4")
    assert watch.main(base) == 1  # the new silent file fails attempt 1 of 3 in this run
    assert watch.main(base) == 1  # attempt 2 of 3
    assert watch.main(base) == 1  # attempt 3 of 3: given up in this run
    assert watch.main(base) == 0  # it failed in an earlier run, so this run has nothing to report


def test_once_exits_zero_when_the_only_failure_is_from_an_earlier_run(serve, tmp_path):
    serve()
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "silent.mp4", seconds=1, with_audio=False), watch_dir / "screen.mp4")
    earlier = _watcher(watch_dir, out_dir)
    for _ in range(4):  # first sighting, then three attempts: the file fails for good
        earlier.cycle()
    assert _state(out_dir)[0]["status"] == "failed"
    base = [str(watch_dir), "--output-dir", str(out_dir), "--once", "--interval", "0.1"]
    assert watch.main(base) == 0


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


def test_tuning_flags_are_forwarded_only_when_set(monkeypatch):
    captured = []
    real_build = watch.stt_cli.build_parser

    def build_parser():
        parser = real_build()
        parse_args = parser.parse_args

        def recording(argv=None):
            captured.append(list(argv))
            return parse_args(argv)

        parser.parse_args = recording
        return parser

    monkeypatch.setattr(watch.stt_cli, "build_parser", build_parser)
    flags = ("--beam-size", "--no-vad", "--prompt", "--compute-type", "--chunk-seconds", "--runtime")
    plain = watch.build_parser().parse_args(["W", "--output-dir", "O"])
    template = watch.transcribe_args(plain)
    for flag in flags:
        assert flag not in captured[-1]
    assert (template.beam_size, template.no_vad, template.prompt, template.compute_type, template.runtime) == (
        5, False, None, "int8", None)
    assert template.chunk_seconds == watch.stt_cli.DEFAULT_CHUNK_SECONDS

    tuned = watch.build_parser().parse_args([
        "W", "--output-dir", "O", "--beam-size", "1", "--no-vad", "--prompt", "names",
        "--compute-type", "float16", "--chunk-seconds", "30", "--runtime", "faster-whisper"])
    template = watch.transcribe_args(tuned)
    argv = captured[-1]
    assert argv[argv.index("--beam-size") + 1] == "1"
    assert "--no-vad" in argv
    assert argv[argv.index("--prompt") + 1] == "names"
    assert argv[argv.index("--compute-type") + 1] == "float16"
    assert argv[argv.index("--chunk-seconds") + 1] == "30"
    assert argv[argv.index("--runtime") + 1] == "faster-whisper"
    assert (template.beam_size, template.no_vad, template.prompt, template.compute_type,
            template.chunk_seconds, template.runtime) == (1, True, "names", "float16", 30.0, "faster-whisper")


@pytest.mark.parametrize("value", ["0", "-5", "nan"])
def test_chunk_seconds_must_be_positive(value, capsys):
    with pytest.raises(SystemExit) as exit_info:
        watch.build_parser().parse_args(["W", "--output-dir", "O", "--chunk-seconds", value])
    assert exit_info.value.code == 2
    assert "must be greater than 0" in capsys.readouterr().err


def test_a_failed_state_save_before_writing_publishes_nothing_and_the_retry_makes_one_copy(serve, tmp_path,
                                                                                         monkeypatch):
    serve()
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "talk.mp4")
    first = _watcher(watch_dir, out_dir)
    first.cycle()  # first sighting
    real_save = watch.WatchState.save

    def disk_full(self):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(watch.WatchState, "save", disk_full)
    with pytest.raises(OSError):
        first.cycle()
    assert list(out_dir.iterdir()) == []  # nothing written and nothing recorded

    monkeypatch.setattr(watch.WatchState, "save", real_save)
    restarted = _watcher(watch_dir, out_dir)  # launchd restarts the watcher
    restarted.cycle()
    restarted.cycle()
    assert sorted(p.name for p in out_dir.glob("talk*")) == ["talk.txt"]
    assert _state(out_dir)[0]["status"] == "completed"


def test_a_failed_save_after_publishing_does_not_duplicate_the_transcript_on_restart(serve, tmp_path, monkeypatch):
    serve()
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "talk.mp4")
    real_save = watch.WatchState.save

    def save_unless_completing(self):
        # Fail only the save that records 'completed', which comes after the transcript is on disk.
        if any(row["status"] == "completed" for row in self.sources.values()):
            raise OSError(28, "No space left on device")
        real_save(self)

    monkeypatch.setattr(watch.WatchState, "save", save_unless_completing)
    first = _watcher(watch_dir, out_dir)
    first.cycle()
    with pytest.raises(OSError):
        first.cycle()
    assert (out_dir / "talk.txt").exists()
    assert _state(out_dir)[0]["status"] == "publishing"

    monkeypatch.setattr(watch.WatchState, "save", real_save)
    restarted = _watcher(watch_dir, out_dir)  # launchd restarts the watcher
    restarted.cycle()
    restarted.cycle()
    assert sorted(p.name for p in out_dir.glob("talk*")) == ["talk.txt"]
    assert _state(out_dir)[0]["status"] == "completed"


def test_an_interrupted_publish_is_recognised_only_when_the_transcript_matches(tmp_path):
    watch_dir, out_dir = _folders(tmp_path)
    out_dir.mkdir()
    (out_dir / "ok.txt").write_text("full transcript", encoding="utf-8")
    (out_dir / "talk.txt").write_text("half a tra", encoding="utf-8")  # partial, not the recorded text
    state = watch.WatchState(out_dir / ".watch-state.json", {})
    state.record(("ok.mp4", 1, 2), "publishing", output=str(out_dir / "ok.txt"),
                 digest=watch._sha256(b"full transcript"))
    state.record(("talk.mp4", 3, 4), "publishing", output=str(out_dir / "talk.txt"),
                 digest=watch._sha256(b"full transcript"))
    restarted = _watcher(watch_dir, out_dir)
    assert restarted.state.get(("ok.mp4", 1, 2))["status"] == "completed"
    assert restarted.state.get(("talk.mp4", 3, 4))["status"] == "publishing"  # will be transcribed again


def _row(out_dir, name):
    return next(row for row in _state(out_dir) if row["name"] == name)


def test_a_file_that_keeps_getting_500_does_not_block_the_others_and_gives_up_after_three_attempts(serve, tmp_path,
                                                                                               monkeypatch):
    serve()
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "a_bad.mp4")  # sorted first
    shutil.copy(_make_media(tmp_path / "other.mp4", seconds=1), watch_dir / "b_good.mp4")
    real_submit = watch.stt_cli.submit_batch

    def server_error_for_one_file(items, args, deadline):
        if any(item.upload_name.startswith("a_bad.") for item in items):  # video is uploaded as a_bad.wav
            raise watch.stt_cli.CliError("HTTP 500 from /api/batches", status=500)
        return real_submit(items, args, deadline)

    monkeypatch.setattr(watch.stt_cli, "submit_batch", server_error_for_one_file)
    watcher = _watcher(watch_dir, out_dir, "--format", "srt")
    watcher.cycle()  # first sighting
    watcher.cycle()
    assert (out_dir / "b_good.srt").exists()  # not held up behind the file that failed
    assert watcher.blocked is None
    assert _row(out_dir, "a_bad.mp4")["status"] == "retrying" and _row(out_dir, "a_bad.mp4")["attempts"] == 1

    restarted = _watcher(watch_dir, out_dir, "--format", "srt")  # the count survives a restart
    restarted.cycle()
    restarted.cycle()
    assert _row(out_dir, "a_bad.mp4")["attempts"] == 2
    restarted.cycle()
    assert _row(out_dir, "a_bad.mp4")["status"] == "failed" and _row(out_dir, "a_bad.mp4")["attempts"] == 3
    assert "HTTP 500" in _row(out_dir, "a_bad.mp4")["error"]
    assert _row(out_dir, "b_good.mp4")["status"] == "completed"
    assert restarted.blocked is None and restarted.failed


def test_a_busy_server_429_waits_and_does_not_count_as_an_attempt(serve, tmp_path, monkeypatch):
    serve()
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=1), watch_dir / "talk.mp4")

    def busy(items, args, deadline):
        raise watch.stt_cli.CliError("another transcription is running", status=429)

    monkeypatch.setattr(watch.stt_cli, "submit_batch", busy)
    watcher = _watcher(watch_dir, out_dir)
    watcher.cycle()
    watcher.cycle()
    watcher.cycle()
    assert watcher.blocked == "another transcription is running"
    assert not (out_dir / ".watch-state.json").exists()  # no attempt was recorded against the file
    assert watcher.run_failures == 0


def test_a_second_watcher_refuses_to_start_while_the_first_holds_the_output_lock(serve, tmp_path):
    serve()
    watch_dir, out_dir = _folders(tmp_path)
    out_dir.mkdir()
    log_path = tmp_path / "first.err"
    with log_path.open("w", encoding="utf-8") as err:
        first = subprocess.Popen([sys.executable, str(REPO / "scripts" / "watch_folder.py"), str(watch_dir),
                                  "--output-dir", str(out_dir), "--interval", "0.2"],
                                 stdout=subprocess.DEVNULL, stderr=err, env=os.environ.copy())
    try:
        assert _wait_for(lambda: "watching" in log_path.read_text(encoding="utf-8"), 30)
        second = subprocess.run([sys.executable, str(REPO / "scripts" / "watch_folder.py"), str(watch_dir),
                                 "--output-dir", str(out_dir), "--once", "--interval", "0.1"],
                                capture_output=True, text=True, timeout=60, env=os.environ.copy())
        assert second.returncode == 1
        assert "already writing to" in second.stderr and f"pid {first.pid}" in second.stderr
        assert first.poll() is None  # the first watcher is still running and was not disturbed
    finally:
        first.terminate()
        first.wait(timeout=30)


def test_the_output_lock_is_released_when_the_run_ends(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with watch.single_writer(out_dir):
        with pytest.raises(watch.stt_cli.CliError, match="already writing"):
            with watch.single_writer(out_dir):
                pass
    with watch.single_writer(out_dir):  # free again once the first run has ended
        pass


def test_file_timeout_is_validated_and_passed_to_the_transcriber(tmp_path):
    watch_dir, out_dir = _folders(tmp_path)
    for bad in ("0", "-5", "nan", "inf", "soon"):
        with pytest.raises(SystemExit) as refused:
            watch.main([str(watch_dir), "--output-dir", str(out_dir), "--file-timeout", bad])
        assert refused.value.code == 2, bad
    default = watch.build_parser().parse_args([str(watch_dir), "--output-dir", str(out_dir)])
    assert default.file_timeout == 3600.0 and watch.transcribe_args(default).timeout == 3600.0
    custom = watch.build_parser().parse_args([str(watch_dir), "--output-dir", str(out_dir), "--file-timeout", "7.5"])
    assert custom.file_timeout == 7.5 and watch.transcribe_args(custom).timeout == 7.5


def test_a_file_over_its_time_budget_is_cancelled_and_retried_not_failed(serve, tmp_path):
    serve(SlowModel())
    watch_dir, out_dir = _folders(tmp_path)
    shutil.copy(_make_media(tmp_path / "source.mp4", seconds=2), watch_dir / "talk.mp4")
    watcher = _watcher(watch_dir, out_dir, "--file-timeout", "1")
    watcher.cycle()  # first sighting
    watcher.cycle()
    row = _row(out_dir, "talk.mp4")
    assert row["status"] == "retrying" and row["attempts"] == 1
    assert row["error"] == "timed out after 1 s"
    assert not (out_dir / "talk.txt").exists() and not watcher.failed


def test_transcriber_progress_reaches_the_log_while_the_file_is_still_running(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(watch, "log", lambda message, stream=None: events.append(message))

    def fake_transcriber(args):
        print("[stt] batch b1 started with 1 file(s)", file=sys.stderr)
        events.append("[still transcribing]")
        print("[stt] half", end="", file=sys.stderr)  # a partial line waits until it is complete
        print(" done", file=sys.stderr)
        print(json.dumps({"files": [{"status": "completed", "segments": []}]}))

    monkeypatch.setattr(watch.stt_cli, "cmd_transcribe", fake_transcriber)
    watch_dir, out_dir = _folders(tmp_path)
    record = _watcher(watch_dir, out_dir)._transcribe(tmp_path / "talk.mp4")
    assert record["status"] == "completed"
    assert events == ["[stt] batch b1 started with 1 file(s)", "[still transcribing]", "[stt] half done"]
