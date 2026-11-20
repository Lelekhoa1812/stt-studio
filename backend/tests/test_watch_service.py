"""The background service for the folder watcher: install, uninstall and status, in a temporary HOME.

Nothing here loads or starts a launch agent. A fake launchctl is first on PATH and leaves a marker file if it is
ever run, so the tests can assert that the scripts never call it.
"""
import os
import plistlib
import shutil
import subprocess
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
    result = install(ctx, "--format", "vtt", "--model", "small", "--language", "en", "--diarize", "--interval", "7")
    data = load(ctx)
    assert data["ProgramArguments"] == [
        "/usr/bin/env", "python3", str(WATCHER), os.path.realpath(ctx.watch), "--output-dir",
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
def test_defaults_are_srt_every_10_seconds(ctx):
    install(ctx)
    data = load(ctx)
    assert data["ProgramArguments"][-4:] == ["--format", "srt", "--interval", "10"]
    for flag in ("--model", "--language", "--diarize", "--once"):
        assert flag not in data["ProgramArguments"]
    assert "STT_STUDIO_URL" not in data["EnvironmentVariables"]


@requires_macos
def test_paths_with_spaces_and_ampersands_round_trip(ctx):
    model = "owner/model&<tag>"
    install(ctx, "--model", model)
    arguments = load(ctx)["ProgramArguments"]
    assert arguments[3] == os.path.realpath(ctx.watch)
    assert arguments[5] == os.path.realpath(ctx.output)
    assert arguments[arguments.index("--model") + 1] == model
    assert b"&amp;" in ctx.plist.read_bytes()


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
