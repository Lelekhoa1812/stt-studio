"""ffmpeg decode respects JobControl cancellation."""

import io
import subprocess
import threading
import time
from pathlib import Path

from backend import app as app_module
from backend.app import JobCancelled, JobControl, _run_ffmpeg, _silence_midpoints


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


class _HangingSilencePopen:
    instances: list["_HangingSilencePopen"] = []

    def __init__(self, command, **kwargs):
        assert kwargs.get("stderr") == subprocess.PIPE
        assert kwargs.get("start_new_session") is True
        self.command = command
        self.pid = 4343
        self._code: int | None = None
        self.terminated = threading.Event()
        self.stderr = io.BytesIO()
        _HangingSilencePopen.instances.append(self)

    def poll(self):
        return self._code

    def kill_group(self):
        self._code = -15
        self.terminated.set()


class _SilenceSuccessPopen:
    def __init__(self, command, **kwargs):
        assert kwargs.get("stderr") == subprocess.PIPE
        self.pid = 5252
        self._code: int | None = None
        self.stderr = io.BytesIO(b"[silencedetect @ 0x0] silence_start: 2.0\n"
                               b"[silencedetect @ 0x0] silence_end: 4.0\n")

    def poll(self):
        if self._code is None:
            self._code = 0
        return self._code


class _SilenceFailPopen:
    def __init__(self, command, **kwargs):
        self.pid = 5353
        self.stderr = io.BytesIO(b"decode error\n")

    def poll(self):
        return 1


def test_silence_midpoints_terminates_when_control_is_cancelled(monkeypatch, tmp_path):
    _HangingSilencePopen.instances.clear()
    monkeypatch.setattr(app_module.subprocess, "Popen", _HangingSilencePopen)
    killed = {"done": False}

    def killpg(pid, sig):
        assert pid == 4343
        proc = _HangingSilencePopen.instances[0]
        proc.kill_group()
        killed["done"] = True

    monkeypatch.setattr(app_module.os, "killpg", killpg)
    control = JobControl()
    path = tmp_path / "long.wav"
    error: list[BaseException] = []

    def run():
        try:
            _silence_midpoints(path, duration=600.0, control=control)
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not _HangingSilencePopen.instances:
        time.sleep(0.02)
    assert _HangingSilencePopen.instances
    control.cancel()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(error) == 1 and isinstance(error[0], JobCancelled)
    assert killed["done"]


def test_silence_midpoints_parses_stderr_when_not_cancelled(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.subprocess, "Popen", _SilenceSuccessPopen)
    path = tmp_path / "long.wav"
    control = JobControl()
    pauses = _silence_midpoints(path, duration=120.0, control=control)
    assert pauses == [3.0]


def test_silence_midpoints_returns_empty_on_scan_failure_with_control(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.subprocess, "Popen", _SilenceFailPopen)
    path = tmp_path / "long.wav"
    control = JobControl()
    assert _silence_midpoints(path, duration=120.0, control=control) == []


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