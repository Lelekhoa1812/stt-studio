"""ffmpeg decode respects JobControl cancellation."""

import threading
import time
from pathlib import Path

from backend import app as app_module
from backend.app import JobCancelled, JobControl, _run_ffmpeg


class _HangingPopen:
    instances: list["_HangingPopen"] = []

    def __init__(self, command, **kwargs):
        assert kwargs.get("start_new_session") is True
        self.command = command
        self.pid = 4242
        self._code: int | None = None
        self.terminated = threading.Event()
        _HangingPopen.instances.append(self)

    def poll(self):
        return self._code

    def wait(self, timeout=None):
        if self._code is None:
            self.terminated.wait(timeout)
        return self._code

    def kill_group(self):
        self._code = -15
        self.terminated.set()


class _SuccessPopen:
    def __init__(self, command, **kwargs):
        assert kwargs.get("start_new_session") is True
        self.pid = 5151
        self._code: int | None = None

    def poll(self):
        if self._code is None:
            self._code = 0
        return self._code

    def wait(self, timeout=None):
        self.poll()
        return self._code


def test_run_ffmpeg_terminates_when_control_is_cancelled(monkeypatch, tmp_path):
    _HangingPopen.instances.clear()
    monkeypatch.setattr(app_module.subprocess, "Popen", _HangingPopen)
    killed = {"done": False}

    def killpg(pid, sig):
        assert pid == 4242
        proc = _HangingPopen.instances[0]
        proc.kill_group()
        killed["done"] = True

    monkeypatch.setattr(app_module.os, "killpg", killpg)
    control = JobControl()
    path = tmp_path / "secret.wav"
    error: list[BaseException] = []

    def run():
        try:
            _run_ffmpeg(["ffmpeg", "-i", str(path)], path, timeout=30.0, control=control)
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not _HangingPopen.instances:
        time.sleep(0.02)
    assert _HangingPopen.instances
    control.cancel()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(error) == 1 and isinstance(error[0], JobCancelled)
    assert killed["done"]


def test_run_ffmpeg_waits_for_success_when_not_cancelled(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.subprocess, "Popen", _SuccessPopen)
    path = tmp_path / "audio.wav"
    control = JobControl()
    _run_ffmpeg(["ffmpeg", "-i", str(path)], path, timeout=5.0, control=control)


def test_run_ffmpeg_without_control_uses_blocking_run(monkeypatch, tmp_path):
    calls: list[tuple] = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return None

    monkeypatch.setattr(app_module.subprocess, "run", run)
    path = tmp_path / "audio.wav"
    _run_ffmpeg(["ffmpeg", "-i", str(path)], path, timeout=5.0, control=None)
    assert len(calls) == 1
    assert calls[0][0][0] == "ffmpeg"