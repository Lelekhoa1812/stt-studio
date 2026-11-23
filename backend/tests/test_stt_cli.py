"""The agent CLI against a real loopback HTTP server, with real ffmpeg media and a fake model."""
import importlib.util
import io
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from backend.app import ModelManager, create_app
from test_api import FakeModel

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                reason="ffmpeg and ffprobe are required")


def _load_cli():
    spec = importlib.util.spec_from_file_location("stt_cli", REPO / "scripts" / "stt_cli.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules, so register before exec.
    sys.modules["stt_cli"] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


@pytest.fixture()
def server(monkeypatch):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    api = create_app(ModelManager(lambda model, compute_type: FakeModel()))
    instance = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    for _ in range(200):
        if instance.started:
            break
        time.sleep(0.05)
    monkeypatch.setenv("STT_STUDIO_URL", f"http://127.0.0.1:{port}")
    yield f"http://127.0.0.1:{port}"
    instance.should_exit = True
    thread.join(timeout=5)


def _make_media(path: Path, seconds: int, with_audio=True):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10"]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000"]
    command += ["-t", str(seconds), "-c:v", "mpeg4"]
    command += ["-c:a", "aac"] if with_audio else ["-an"]
    command += ["-y", str(path)]
    subprocess.run(command, check=True)
    return path


def test_cli_status_and_capabilities_print_json(server, capsys):
    assert cli.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert cli.main(["capabilities"]) == 0
    assert "/api/batches/{batch_id}/export" in capsys.readouterr().out


def test_cli_reports_unreachable_server_without_traceback(monkeypatch, capsys):
    monkeypatch.setenv("STT_STUDIO_URL", "http://127.0.0.1:9")
    assert cli.main(["status"]) == 1
    assert "Cannot reach STT Studio" in capsys.readouterr().err


def test_video_is_split_into_chunks_and_offsets_are_restored(server, tmp_path, capsys):
    video = _make_media(tmp_path / "talk.mp4", seconds=3)
    code = cli.main(["transcribe", str(video), "--chunk-seconds", "1", "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    record = summary["files"][0]
    assert record["status"] == "completed" and record["error"] is None
    assert 2.5 <= record["duration_seconds"] <= 3.5
    # FakeModel returns one segment per chunk at 0.25 s into that chunk.
    assert [round(row["start"], 2) for row in record["segments"]] == [0.25, 1.25, 2.25]
    assert record["text"] == "hello\nhello\nhello"


def test_output_dir_writes_srt_per_input_with_stitched_timestamps(server, tmp_path, capsys):
    first = _make_media(tmp_path / "one.mp4", seconds=2)
    second = _make_media(tmp_path / "two.mp4", seconds=1)
    out = tmp_path / "out"
    assert cli.main(["transcribe", str(first), str(second), "--format", "srt", "--output-dir", str(out),
                     "--chunk-seconds", "1"]) == 0
    # FakeModel reports 0.25 -> 1.5 in every chunk; ends are clamped to the chunk, so captions cannot overlap.
    text = (out / "one.srt").read_text(encoding="utf-8")
    assert text.startswith("1\n00:00:00,250 --> 00:00:01,000\nhello\n\n2\n00:00:01,250 --> 00:00:02,000\nhello")
    assert (out / "two.srt").read_text(encoding="utf-8").startswith("1\n00:00:00,250 --> 00:00:01,000\nhello")


def test_media_without_audio_is_reported_not_sent(server, tmp_path, capsys):
    silent = _make_media(tmp_path / "screen.mp4", seconds=1, with_audio=False)
    assert cli.main(["transcribe", str(silent), "--json"]) == 1
    record = json.loads(capsys.readouterr().out)["files"][0]
    assert record["status"] == "failed" and "no audio track" in record["error"]


def test_batch_submit_then_export_through_the_cli(server, tmp_path, capsys):
    clip = _make_media(tmp_path / "clip.mp4", seconds=1)
    assert cli.main(["batch", "submit", str(clip)]) == 0
    batch_id = json.loads(capsys.readouterr().out)["id"]
    for _ in range(300):
        assert cli.main(["batch", "status", batch_id]) == 0
        if json.loads(capsys.readouterr().out)["done"]:
            break
        time.sleep(0.02)
    target = tmp_path / "batch.json"
    assert cli.main(["batch", "export", batch_id, "--output", str(target)]) == 0
    capsys.readouterr()
    assert json.loads(target.read_text())["files"][0]["segments"][0]["text"] == "hello"


def test_multipart_body_is_well_formed(tmp_path):
    payload = tmp_path / "a.wav"
    payload.write_bytes(b"RIFFdata")
    body, size, content_type = cli._multipart({"model": "small", "vad_filter": True, "language": None},
                                              [("files", payload, 'odd"name.wav')])
    try:
        raw = body.read()
    finally:
        body.close()
    boundary = content_type.split("boundary=")[1]
    assert size == len(raw)
    assert f"--{boundary}".encode() in raw and b'name="model"\r\n\r\nsmall' in raw
    assert b'name="vad_filter"\r\n\r\ntrue' in raw and b"language" not in raw
    assert b'filename="odd_name.wav"' in raw and raw.endswith(f"--{boundary}--\r\n".encode())


def test_cut_points_prefer_the_middle_of_a_pause(monkeypatch, tmp_path):
    stderr = ("[silencedetect @ 0x1] silence_start: 9.6\n"
              "[silencedetect @ 0x1] silence_end: 10.4 | silence_duration: 0.8\n")
    monkeypatch.setattr(cli, "_run_tool", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", stderr))
    cuts = cli.find_cut_points(tmp_path / "x.wav", duration=25.0, chunk_seconds=10.0)
    assert cuts[0] == pytest.approx(10.0)  # pause midpoint at 10.0 sits exactly on the target
    assert cuts[1] == pytest.approx(20.0)  # no pause near 20 s, so the target is used


def test_diarize_is_never_split_and_labels_reach_the_json(server, tmp_path, capsys, monkeypatch):
    from backend import diarization

    monkeypatch.setattr(diarization, "availability", lambda: {"available": True, "model": diarization.MODEL_ID,
                                                              "license": "MIT", "reason": None})
    monkeypatch.setattr(diarization, "diarize_file", lambda *a, **k: [(0.0, 3.0, "spk_0")])
    video = _make_media(tmp_path / "talk.mp4", seconds=3)
    code = cli.main(["transcribe", str(video), "--diarize", "--chunk-seconds", "1", "--json"])
    record = json.loads(capsys.readouterr().out)["files"][0]
    assert code == 0 and record["status"] == "completed"
    # One job, one transcript: a split would have produced three segments.
    assert [row["text"] for row in record["segments"]] == ["hello"]
    assert record["segments"][0]["speaker"] == "Speaker 1"
    assert record["diarization"]["status"] == "completed"


def test_diarize_refuses_files_longer_than_one_job(server, tmp_path, capsys, monkeypatch):
    from backend import app as app_module
    from backend import diarization

    monkeypatch.setattr(diarization, "availability", lambda: {"available": True, "model": diarization.MODEL_ID,
                                                              "license": "MIT", "reason": None})
    monkeypatch.setattr(app_module, "MAX_AUDIO_SECONDS", 1.0)
    video = _make_media(tmp_path / "long.mp4", seconds=3)
    assert cli.main(["transcribe", str(video), "--diarize"]) == 1
    assert "single-job limit" in capsys.readouterr().err


def test_diarize_reports_the_reason_when_the_server_cannot_label(server, tmp_path, capsys, monkeypatch):
    from backend import diarization

    monkeypatch.setattr(diarization, "availability", lambda: {"available": False, "model": diarization.MODEL_ID,
                                                              "license": "MIT", "reason": "HF_TOKEN is not set"})
    video = _make_media(tmp_path / "talk.mp4", seconds=1)
    assert cli.main(["transcribe", str(video), "--diarize"]) == 1
    assert "HF_TOKEN is not set" in capsys.readouterr().err



def test_job_status_exits_nonzero_only_for_a_failed_job(monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_json", lambda *a, **k: {"id": "j", "status": "failed", "error": "boom"})
    assert cli.main(["job", "status", "j"]) == 1
    monkeypatch.setattr(cli, "get_json", lambda *a, **k: {"id": "j", "status": "completed", "segments": []})
    assert cli.main(["job", "status", "j"]) == 0
    capsys.readouterr()


def test_single_chunk_after_sliver_fold_is_not_reported_as_a_split(server, tmp_path, capsys):
    # 2.2 s with a 2 s target leaves a 0.2 s sliver, which is folded, so the file is one chunk.
    clip = _make_media(tmp_path / "short.mp4", seconds=2.2)
    code = cli.main(["transcribe", str(clip), "--chunk-seconds", "2"])
    captured = capsys.readouterr()
    assert code == 0
    assert "splitting" not in captured.err


def test_json_runs_still_announce_the_batch_id_on_stderr(server, tmp_path, capsys):
    clip = _make_media(tmp_path / "clip.mp4", seconds=1)
    assert cli.main(["transcribe", str(clip), "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["files"][0]["status"] == "completed"
    assert "[stt] batch " in captured.err


def test_ctrl_c_cancels_the_batch_so_the_server_is_not_left_busy(server, tmp_path, capsys, monkeypatch):
    clip = _make_media(tmp_path / "clip.mp4", seconds=1)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "wait_for_batch", interrupted)
    assert cli.main(["transcribe", str(clip), "--json"]) == 130
    captured = capsys.readouterr()
    batch_id = re.search(r"batch ([0-9a-f-]{36}) started", captured.err).group(1)
    # Cancellation is cooperative: the running file stops at its next checkpoint, so wait for the batch to settle.
    for _ in range(500):
        if cli.get_json(f"/api/batches/{batch_id}")["done"]:
            break
        time.sleep(0.02)
    assert cli.get_json(f"/api/batches/{batch_id}")["done"] is True
    assert cli.get_json("/api/health")["active_jobs"] == 0


def _spy_calls(monkeypatch) -> list[tuple[str, str]]:
    """Record every (method, path) the CLI sends, while still sending it to the real server."""
    calls: list[tuple[str, str]] = []
    real_call = cli.call

    def spy(method, path, **kwargs):
        calls.append((method, path))
        return real_call(method, path, **kwargs)

    monkeypatch.setattr(cli, "call", spy)
    return calls


def test_remote_stt_studio_url_is_refused_before_any_network_call(tmp_path, capsys, monkeypatch):
    clip = _make_media(tmp_path / "clip.mp4", seconds=1)
    monkeypatch.setenv("STT_STUDIO_URL", "http://example.com")

    def no_network(*args, **kwargs):
        raise AssertionError("a request was sent to a non-loopback host")

    monkeypatch.setattr(cli.urllib.request, "urlopen", no_network)
    assert cli.main(["transcribe", str(clip)]) == 1
    assert "loopback" in capsys.readouterr().err
    assert cli.main(["status"]) == 1
    assert "loopback" in capsys.readouterr().err


def test_short_video_is_uploaded_as_wav_not_the_original(server, tmp_path, capsys, monkeypatch):
    video = _make_media(tmp_path / "talk.mp4", seconds=2)
    uploads = []
    real_submit = cli.submit_batch

    def record_upload(items, args, deadline):
        uploads.extend((item.upload_name, item.path.read_bytes()[:4]) for item in items)
        return real_submit(items, args, deadline)

    monkeypatch.setattr(cli, "submit_batch", record_upload)
    assert cli.main(["transcribe", str(video), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["files"][0]["status"] == "completed"
    assert uploads == [("talk.wav", b"RIFF")]


def test_large_non_wav_audio_is_extracted_to_wav_before_upload(server, tmp_path, capsys, monkeypatch):
    voice = tmp_path / "voice.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
                    "-t", "1", "-c:a", "aac", "-y", str(voice)], check=True)
    uploads = []
    real_submit = cli.submit_batch

    def record_upload(items, args, deadline):
        uploads.extend(item.upload_name for item in items)
        return real_submit(items, args, deadline)

    monkeypatch.setattr(cli, "submit_batch", record_upload)
    assert cli.main(["transcribe", str(voice), "--json"]) == 0
    assert uploads == ["voice.m4a"]  # small audio is still sent as it is
    monkeypatch.setattr(cli, "LARGE_UPLOAD_BYTES", 1)
    assert cli.main(["transcribe", str(voice), "--json"]) == 0
    assert uploads[-1] == "voice.wav"


def test_one_undecodable_file_does_not_stop_the_others(server, tmp_path, capsys):
    good = _make_media(tmp_path / "good.mp4", seconds=1)
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not media")
    out = tmp_path / "out"
    code = cli.main(["transcribe", str(broken), str(good), "--output-dir", str(out)])
    err = capsys.readouterr().err
    assert code == 1
    assert (out / "good.txt").read_text(encoding="utf-8").strip() == "hello"
    assert "broken.mp4" in err and "summary: 1 of 2 file(s) succeeded" in err
    assert not (out / "broken.txt").exists()


@pytest.mark.parametrize("value", ["0", "-5", "nan"])
def test_chunk_seconds_must_be_positive_and_exits_2(tmp_path, capsys, value):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["transcribe", str(tmp_path / "clip.mp4"), "--chunk-seconds", value])
    assert exit_info.value.code == 2
    assert "must be greater than 0" in capsys.readouterr().err


def test_a_failed_poll_cancels_the_server_batch(server, tmp_path, capsys, monkeypatch):
    clip = _make_media(tmp_path / "clip.mp4", seconds=1)
    calls = _spy_calls(monkeypatch)

    def broken_poll(*args, **kwargs):
        raise cli.CliError("poll exploded")

    monkeypatch.setattr(cli, "wait_for_batch", broken_poll)
    assert cli.main(["transcribe", str(clip)]) == 1
    captured = capsys.readouterr()
    assert "error: poll exploded" in captured.err
    batch_id = re.search(r"batch ([0-9a-f-]{36}) started", captured.err).group(1)
    assert ("POST", f"/api/batches/{batch_id}/cancel") in calls
    for _ in range(500):
        if cli.get_json(f"/api/batches/{batch_id}")["done"]:
            break
        time.sleep(0.02)
    assert cli.get_json(f"/api/batches/{batch_id}")["done"] is True


def test_a_refused_cancel_does_not_hide_the_poll_error(server, tmp_path, capsys, monkeypatch):
    clip = _make_media(tmp_path / "clip.mp4", seconds=1)
    real_call = cli.call

    def refuse_cancel(method, path, **kwargs):
        if path.endswith("/cancel"):
            raise cli.CliError("cancel refused")
        return real_call(method, path, **kwargs)

    def broken_poll(*args, **kwargs):
        raise cli.CliError("poll exploded")

    monkeypatch.setattr(cli, "call", refuse_cancel)
    monkeypatch.setattr(cli, "wait_for_batch", broken_poll)
    assert cli.main(["transcribe", str(clip)]) == 1
    assert "error: poll exploded" in capsys.readouterr().err


def test_batch_submit_reports_a_silent_file_and_sends_no_batch(server, tmp_path, capsys, monkeypatch):
    good = _make_media(tmp_path / "good.mp4", seconds=1)
    silent = _make_media(tmp_path / "screen.mp4", seconds=1, with_audio=False)
    calls = _spy_calls(monkeypatch)
    assert cli.main(["batch", "submit", str(good), str(silent)]) == 1
    err = capsys.readouterr().err
    assert "screen.mp4" in err and "no audio track" in err and "Nothing was submitted" in err
    assert not any(path == "/api/batches" for _, path in calls)


def test_batch_submit_reports_a_file_over_the_limit_and_sends_no_batch(server, tmp_path, capsys, monkeypatch):
    from backend import app as app_module

    monkeypatch.setattr(app_module, "MAX_AUDIO_SECONDS", 1.0)
    long_clip = _make_media(tmp_path / "long.mp4", seconds=3)
    calls = _spy_calls(monkeypatch)
    assert cli.main(["batch", "submit", str(long_clip)]) == 1
    err = capsys.readouterr().err
    assert "long.mp4" in err and "per-file limit" in err
    assert not any(path == "/api/batches" for _, path in calls)


def test_loopback_check_accepts_the_three_local_hosts():
    for url in ("http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"):
        cli._require_loopback(url)
    with pytest.raises(cli.CliError):
        cli._require_loopback("http://192.168.1.20:8765")


def test_inputs_with_the_same_stem_get_unique_output_names(server, tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _make_media(tmp_path / "a" / "talk.mp4", seconds=1)
    second = _make_media(tmp_path / "b" / "talk.mp4", seconds=1)
    out = tmp_path / "out"
    assert cli.main(["transcribe", str(first), str(second), "--output-dir", str(out)]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["talk-2.txt", "talk.txt"]
    assert (out / "talk.txt").read_text(encoding="utf-8").strip() == "hello"
    assert "talk-2.txt" in capsys.readouterr().err


def test_unwritable_output_fails_before_any_upload(server, tmp_path, capsys, monkeypatch):
    clip = _make_media(tmp_path / "clip.mp4", seconds=1)
    calls = _spy_calls(monkeypatch)
    assert cli.main(["transcribe", str(clip), "-o", str(tmp_path / "missing" / "out.txt")]) == 1
    assert "does not exist" in capsys.readouterr().err
    assert calls == []  # no request of any kind was made


def test_write_errors_print_one_line_without_a_traceback(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "call", lambda *args, **kwargs: b"{}")
    assert cli.main(["job", "export", "job-1", "-o", str(tmp_path / "missing" / "x.txt")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err and err.count("\n") == 1


def test_batch_submit_uploads_a_video_as_wav(server, tmp_path, capsys, monkeypatch):
    video = _make_media(tmp_path / "clip.mp4", seconds=1)
    uploads = []
    real_submit = cli.submit_batch

    def record_upload(items, args, deadline):
        uploads.extend((item.upload_name, item.path.read_bytes()[:4]) for item in items)
        return real_submit(items, args, deadline)

    monkeypatch.setattr(cli, "submit_batch", record_upload)
    assert cli.main(["batch", "submit", str(video)]) == 0
    batch_id = json.loads(capsys.readouterr().out)["id"]
    # The uploaded bytes are a WAV file, and the server stored the job under the WAV name.
    assert uploads == [("clip.wav", b"RIFF")]
    assert cli.get_json(f"/api/batches/{batch_id}")["jobs"][0]["filename"] == "clip.wav"
