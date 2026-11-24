"""The background service for the folder watcher: install, uninstall and status, in a temporary HOME.

Nothing here loads or starts a launch agent. A fake launchctl is first on PATH and leaves a marker file if it is
ever run, so the tests can assert that the scripts never call it.
"""
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
SERVICE = REPO / "scripts" / "watch_service"
WATCHER = REPO / "scripts" / "watch_folder.py"
PLIST_NAME = "com.sttstudio.watch.plist"
MARKER_KEY = "com.sttstudio.generated-by"
LOG_NAME = "stt-studio-watch.log"

requires_macos = pytest.mark.skipif(not (shutil.which("plutil") and shutil.which("bash")),
                                    reason="the service scripts need macOS plutil and bash")


@pytest.fixture()
def ctx(tmp_path):
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "launchctl-was-called"
    fake = fake_bin / "launchctl"
    fake.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 0\n')
    fake.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key != "STT_STUDIO_URL"}
    env.update({
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "STT_WATCH_AGENT_DIR": str(tmp_path / "svc" / "agents"),
        "STT_WATCH_LOG_DIR": str(tmp_path / "svc" / "logs"),
    })
    watch = tmp_path / "Recordings & Notes"
    watch.mkdir()
    agents = tmp_path / "svc" / "agents"
    return SimpleNamespace(env=env, fake=fake, marker=marker, watch=watch, output=tmp_path / "Transcripts & Out",
                           agents=agents, logs=tmp_path / "svc" / "logs", plist=agents / PLIST_NAME,
                           tmp=tmp_path)


def run(ctx, script, *args):
    return subprocess.run(["bash", str(SERVICE / script), *args], env=ctx.env, capture_output=True, text=True,
                          timeout=120)


def install(ctx, *extra):
    result = run(ctx, "install.sh", str(ctx.watch), str(ctx.output), *extra)
    assert result.returncode == 0, result.stderr
    return result


def load(ctx):
    return plistlib.loads(ctx.plist.read_bytes())


@pytest.mark.parametrize("script", ["install.sh", "uninstall.sh", "status.sh"])
def test_scripts_parse(script):
    result = subprocess.run(["bash", "-n", str(SERVICE / script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@requires_macos
def test_install_writes_the_expected_agent(ctx):
    python = shutil.which("python3", path=ctx.env["PATH"])
    result = install(ctx, "--format", "vtt", "--model", "small", "--language", "en", "--diarize", "--interval", "7")
    data = load(ctx)
    assert data["ProgramArguments"] == [
        python, str(WATCHER), os.path.realpath(ctx.watch), "--output-dir",
        os.path.realpath(ctx.output), "--format", "vtt", "--model", "small", "--language", "en", "--diarize",
        "--interval", "7"]
    assert data["Label"] == "com.sttstudio.watch"
    assert MARKER_KEY in data
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["ThrottleInterval"] == 30
    log = os.path.realpath(ctx.logs) + "/" + LOG_NAME
    assert data["StandardOutPath"] == log
    assert data["StandardErrorPath"] == log
    environment = data["EnvironmentVariables"]
    assert {"/opt/homebrew/bin", "/usr/local/bin"} <= set(environment["PATH"].split(":"))
    assert "STT_STUDIO_URL" not in environment
    assert sorted(path.name for path in ctx.agents.iterdir()) == [PLIST_NAME]
    assert f"launchctl bootstrap gui/{os.getuid()} {ctx.plist}" in result.stdout


@requires_macos
def test_defaults_are_txt_every_10_seconds(ctx):
    install(ctx)
    data = load(ctx)
    assert data["ProgramArguments"][-4:] == ["--format", "txt", "--interval", "10"]
    for flag in ("--model", "--language", "--diarize", "--once", "--beam-size", "--no-vad", "--prompt",
                 "--compute-type", "--chunk-seconds", "--runtime"):
        assert flag not in data["ProgramArguments"]
    assert "STT_STUDIO_URL" not in data["EnvironmentVariables"]


@requires_macos
def test_format_srt_is_still_requested(ctx):
    install(ctx, "--format", "srt")
    arguments = load(ctx)["ProgramArguments"]
    assert arguments[arguments.index("--format") + 1] == "srt"


@requires_macos
def test_beam_size_is_put_on_the_program_arguments(ctx):
    install(ctx, "--beam-size", "1")
    arguments = load(ctx)["ProgramArguments"]
    assert arguments[arguments.index("--beam-size") + 1] == "1"


@requires_macos
def test_tuning_flags_are_passed_only_when_set(ctx):
    install(ctx, "--no-vad", "--prompt", "names", "--compute-type", "float16", "--chunk-seconds", "30",
            "--runtime", "faster-whisper")
    arguments = load(ctx)["ProgramArguments"]
    assert "--no-vad" in arguments
    assert arguments[arguments.index("--prompt") + 1] == "names"
    assert arguments[arguments.index("--compute-type") + 1] == "float16"
    assert arguments[arguments.index("--chunk-seconds") + 1] == "30"
    assert arguments[arguments.index("--runtime") + 1] == "faster-whisper"
    assert "--beam-size" not in arguments


@requires_macos
def test_paths_with_spaces_and_ampersands_round_trip(ctx):
    model = "owner/model&<tag>"
    install(ctx, "--model", model)
    arguments = load(ctx)["ProgramArguments"]
    assert arguments[2] == os.path.realpath(ctx.watch)
    assert arguments[4] == os.path.realpath(ctx.output)
    assert arguments[arguments.index("--model") + 1] == model
    assert b"&amp;" in ctx.plist.read_bytes()


@requires_macos
def test_the_interpreter_is_pinned_to_an_absolute_path_and_printed(ctx):
    result = install(ctx)
    interpreter = load(ctx)["ProgramArguments"][0]
    assert os.path.isabs(interpreter) and os.access(interpreter, os.X_OK)
    assert interpreter == shutil.which("python3", path=ctx.env["PATH"])
    assert f"Python: {interpreter}" in result.stdout


@requires_macos
def test_an_explicit_interpreter_is_used_as_given(ctx):
    ctx.env["STT_WATCH_PYTHON"] = sys.executable
    install(ctx)
    assert load(ctx)["ProgramArguments"][0] == sys.executable


@requires_macos
def test_a_missing_relative_or_non_executable_interpreter_is_refused(ctx):
    not_executable = ctx.tmp / "not-executable-python"
    not_executable.write_text("#!/bin/sh\n")
    not_executable.chmod(0o644)
    for python in ("relative/python3", str(ctx.tmp / "missing-python"), str(not_executable)):
        ctx.env["STT_WATCH_PYTHON"] = python
        result = run(ctx, "install.sh", str(ctx.watch), str(ctx.output))
        assert result.returncode != 0, python
        assert "Python interpreter" in result.stderr or "absolute path" in result.stderr, python
    assert not ctx.plist.exists()


@requires_macos
def test_refuses_relative_folders(ctx):
    for watch, output in (("relative/watch", str(ctx.output)), (str(ctx.watch), "transcripts")):
        result = run(ctx, "install.sh", watch, output)
        assert result.returncode != 0
        assert "absolute" in result.stderr
    assert not ctx.plist.exists()


@requires_macos
def test_refuses_output_inside_watch(ctx):
    link = ctx.tmp / "link-to-watch"
    link.symlink_to(ctx.watch, target_is_directory=True)
    for output in (ctx.watch / "out", ctx.watch, link / "out"):
        result = run(ctx, "install.sh", str(ctx.watch), str(output))
        assert result.returncode != 0, output
        assert "inside WATCH_DIR" in result.stderr
    assert not ctx.plist.exists()


@requires_macos
def test_refuses_a_missing_watch_folder(ctx):
    result = run(ctx, "install.sh", str(ctx.tmp / "missing"), str(ctx.output))
    assert result.returncode != 0
    assert "not a folder" in result.stderr
    assert not ctx.plist.exists()


@requires_macos
@pytest.mark.parametrize("url", ["http://example.com:8765", "http://192.168.1.20:8765", "http://0.0.0.0:8765",
                                 "http://127.0.0.1.example.com:8765", "file:///tmp/x"])
def test_refuses_non_loopback_url(ctx, url):
    result = run(ctx, "install.sh", str(ctx.watch), str(ctx.output), "--url", url)
    assert result.returncode != 0
    assert "loopback" in result.stderr
    assert not ctx.plist.exists()


@requires_macos
@pytest.mark.parametrize("url", ["http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"])
def test_accepts_loopback_url_and_passes_it_to_the_watcher(ctx, url):
    install(ctx, "--url", url)
    assert load(ctx)["EnvironmentVariables"]["STT_STUDIO_URL"] == url


@requires_macos
@pytest.mark.parametrize("extra", [["--format", "docx"], ["--interval", "0"], ["--interval", "nan"],
                                   ["--interval", "soon"]])
def test_refuses_bad_format_or_interval(ctx, extra):
    result = run(ctx, "install.sh", str(ctx.watch), str(ctx.output), *extra)
    assert result.returncode != 0
    assert not ctx.plist.exists()


@requires_macos
def test_install_does_not_overwrite_a_plist_it_did_not_write(ctx):
    ctx.agents.mkdir(parents=True)
    ctx.plist.write_bytes(plistlib.dumps({"Label": "com.example.other", "ProgramArguments": ["/bin/true"]}))
    before = ctx.plist.read_bytes()
    result = run(ctx, "install.sh", str(ctx.watch), str(ctx.output))
    assert result.returncode != 0
    assert "not written by this tool" in result.stderr
    assert ctx.plist.read_bytes() == before


@requires_macos
def test_uninstall_removes_only_a_generated_plist(ctx):
    install(ctx)
    result = run(ctx, "uninstall.sh")
    assert result.returncode == 0, result.stderr
    assert not ctx.plist.exists()
    assert f"launchctl bootout gui/{os.getuid()}/com.sttstudio.watch" in result.stdout


@requires_macos
def test_uninstall_refuses_a_plist_without_the_marker(ctx):
    ctx.agents.mkdir(parents=True)
    ctx.plist.write_bytes(plistlib.dumps({"Label": "com.sttstudio.watch"}))
    result = run(ctx, "uninstall.sh")
    assert result.returncode != 0
    assert "was not written by install.sh" in result.stderr
    assert ctx.plist.exists()


@requires_macos
def test_uninstall_when_nothing_is_installed(ctx):
    result = run(ctx, "uninstall.sh")
    assert result.returncode == 0
    assert "Nothing to remove" in result.stdout


@requires_macos
def test_status_shows_settings_and_the_last_20_log_lines(ctx):
    install(ctx, "--model", "small")
    ctx.logs.mkdir(parents=True, exist_ok=True)
    (ctx.logs / LOG_NAME).write_text("".join(f"entry {n:02d}\n" for n in range(1, 26)))
    result = run(ctx, "status.sh")
    assert result.returncode == 0, result.stderr
    assert "Label: com.sttstudio.watch" in result.stdout
    assert f"Python: {load(ctx)['ProgramArguments'][0]}" in result.stdout
    assert "--model small" in result.stdout
    assert "entry 25" in result.stdout
    assert "entry 06" in result.stdout
    assert "entry 05" not in result.stdout
    assert "not checked" in result.stdout


@requires_macos
def test_status_when_not_installed(ctx):
    result = run(ctx, "status.sh")
    assert result.returncode == 0
    assert "Not installed" in result.stdout


@requires_macos
def test_scripts_never_call_launchctl(ctx):
    assert shutil.which("launchctl", path=ctx.env["PATH"]) == str(ctx.fake)
    install(ctx, "--diarize")
    assert run(ctx, "status.sh").returncode == 0
    assert run(ctx, "uninstall.sh").returncode == 0
    assert not ctx.marker.exists()


def test_dev_script_parses():
    result = subprocess.run(["bash", "-n", str(REPO / "scripts" / "dev.sh")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dev_script_refuses_to_start_when_the_backend_port_is_busy(tmp_path):
    import socket

    # Hold the port ourselves when it is free. If it is already busy (for example a running STT Studio backend),
    # that is the state under test too. The minimal PATH stops the script before any install if the port check
    # ever misses a busy port; the assertion on the message then fails instead.
    holder = socket.socket()
    try:
        holder.bind(("127.0.0.1", 8765))
        holder.listen(1)
    except OSError:
        pass
    try:
        result = subprocess.run(["bash", str(REPO / "scripts" / "dev.sh")], cwd=tmp_path,
                                env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
                                capture_output=True, text=True, timeout=60)
    finally:
        holder.close()
    assert result.returncode != 0
    assert "already in use" in result.stderr
    assert "uvicorn" not in result.stdout + result.stderr
