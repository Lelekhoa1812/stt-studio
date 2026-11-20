"""Agent-facing endpoints: capabilities, exports, long-poll, probe, unload and batch export."""
import io
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import exporters
from backend.app import ModelManager, create_app
from test_api import FakeModel, make_client, mock_audio_tools


class BlockingModel:
    """First segment is available at once; the second waits until the test releases it."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, source, **kwargs):
        def segments():
            yield SimpleNamespace(start=0.0, end=1.0, text="first", avg_logprob=-0.1, no_speech_prob=0.0)
            self.started.set()
            self.release.wait(timeout=5)
            yield SimpleNamespace(start=1.0, end=2.0, text="second", avg_logprob=-0.1, no_speech_prob=0.0)
        return segments(), SimpleNamespace(language="en", language_probability=1.0, duration=2.0)


def _blocking_client():
    model = BlockingModel()
    manager = ModelManager(lambda *_args: model)
    api = create_app(manager)
    return TestClient(api, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)), model


def _upload(client, name="talk.wav", data=b"audio bytes"):
    return client.post("/api/jobs", files={"file": (name, io.BytesIO(data), "audio/wav")},
                       data={"model": "tiny", "compute_type": "int8"})


def _fake_ffprobe(monkeypatch, streams, duration="12.5"):
    from backend import app as app_module

    body = {"format": {"duration": duration, "format_name": "mov,mp4,m4a"}, "streams": streams}

    def run(command, **kwargs):
        assert command[0] == "ffprobe"
        return SimpleNamespace(stdout=json.dumps(body).encode(), stderr=b"")

    monkeypatch.setattr(app_module.subprocess, "run", run)


def _route_paths(api):
    return {route.path for route in api.routes if hasattr(route, "path")}


def test_capabilities_lists_limits_formats_and_only_real_routes(monkeypatch):
    client, _ = make_client()
    body = client.get("/api/capabilities").json()
    assert body["service"] == "stt-studio"
    assert body["export_formats"] == ["txt", "srt", "vtt", "json"]
    assert body["batch_export_formats"] == ["txt", "json"]
    assert "mp4" in body["accepted_extensions"] and "wav" in body["accepted_extensions"]
    assert body["limits"]["max_audio_seconds"] > 0 and body["limits"]["max_batch_files"] >= 1
    documented = {endpoint["path"] for endpoint in body["endpoints"]}
    assert {"/api/probe", "/api/models/unload", "/api/jobs/{job_id}/export", "/api/batches/{batch_id}/export"} <= documented
    # Every documented path must be a real route, so the agent-facing list cannot drift from the server.
    assert documented <= _route_paths(client.app)


def test_exporters_use_browser_timestamp_rules():
    segments = [{"start": 3723.5, "end": 3725.0, "text": " hello "}, {"start": 0, "end": 1.2, "text": "   "},
                {"start": 4.0, "end": 5.25, "text": "world"}]
    assert exporters.to_srt(segments) == "1\n01:02:03,500 --> 01:02:05,000\nhello\n\n2\n00:00:04,000 --> 00:00:05,250\nworld\n"
    assert exporters.to_vtt(segments).startswith("WEBVTT\n\n01:02:03.500 --> 01:02:05.000\nhello\n")
    assert exporters.to_txt(segments) == "hello\nworld\n"
    assert json.loads(exporters.to_json(segments)) == [
        {"start": 3723.5, "end": 3725.0, "text": "hello"}, {"start": 4.0, "end": 5.25, "text": "world"}]
    assert exporters.to_srt([]) == "" and exporters.to_vtt([]) == "WEBVTT\n"
    with pytest.raises(ValueError):
        exporters.render([], "docx")


def test_job_export_serves_srt_and_vtt_as_attachments(monkeypatch):
    mock_audio_tools(monkeypatch)
    client, _ = make_client()
    job_id = _upload(client, name="Board call Q3.wav").json()["id"]
    for _ in range(200):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "completed":
            break
        time.sleep(0.01)
    srt = client.get(f"/api/jobs/{job_id}/export", params={"format": "srt"})
    assert srt.status_code == 200
    assert srt.headers["content-disposition"] == 'attachment; filename="Board_call_Q3.srt"'
    assert srt.text == "1\n00:00:00,250 --> 00:00:01,500\nhello\n"
    vtt = client.get(f"/api/jobs/{job_id}/export", params={"format": "vtt"})
    assert vtt.text.startswith("WEBVTT") and "00:00:00.250 --> 00:00:01.500" in vtt.text
    assert client.get(f"/api/jobs/{job_id}/export").text == "hello\n"
    assert client.get(f"/api/jobs/{job_id}/export", params={"format": "pdf"}).status_code == 422
    assert client.get("/api/jobs/missing/export").status_code == 404


def test_export_is_refused_while_running_and_after_failure(monkeypatch):
    mock_audio_tools(monkeypatch)
    client, model = _blocking_client()
    job_id = _upload(client).json()["id"]
    assert model.started.wait(timeout=5)
    running = client.get(f"/api/jobs/{job_id}/export", params={"format": "txt"})
    assert running.status_code == 409 and "running" in running.json()["detail"]
    model.release.set()
    for _ in range(200):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "completed":
            break
        time.sleep(0.01)
    assert client.get(f"/api/jobs/{job_id}/export").text == "first\nsecond\n"

    # Swap in a decoder that rejects the audio, so this job fails deterministically.
    mock_audio_tools(monkeypatch, decode_error=True)
    failed_id = _upload(client, name="bad.wav", data=b"not media").json()["id"]
    state = client.get(f"/api/jobs/{failed_id}", params={"wait": 10}).json()
    assert state["status"] == "failed"
    assert client.get(f"/api/jobs/{failed_id}/export").status_code == 409


def test_get_job_long_poll_returns_when_finished_or_after_timeout(monkeypatch):
    mock_audio_tools(monkeypatch)
    client, model = _blocking_client()
    job_id = _upload(client).json()["id"]
    assert model.started.wait(timeout=5)
    started = time.monotonic()
    still_running = client.get(f"/api/jobs/{job_id}", params={"wait": 0.3}).json()
    assert still_running["status"] == "running"
    assert time.monotonic() - started >= 0.25
    threading.Timer(0.2, model.release.set).start()
    finished = client.get(f"/api/jobs/{job_id}", params={"wait": 10}).json()
    assert finished["status"] == "completed" and finished["progress"] == 100.0
    assert client.get(f"/api/jobs/{job_id}", params={"wait": 61}).status_code == 422


def test_probe_reports_streams_and_whether_the_file_can_be_transcribed(monkeypatch):
    _fake_ffprobe(monkeypatch, [
        {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080},
        {"index": 1, "codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000"},
    ])
    client, _ = make_client()
    body = client.post("/api/probe", files={"file": ("meeting.mp4", io.BytesIO(b"video"), "video/mp4")}).json()
    assert body["has_audio"] and body["has_video"] and body["transcribable"]
    assert body["duration_seconds"] == 12.5 and body["filename"] == "meeting.mp4"
    assert body["streams"][1] == {"index": 1, "type": "audio", "codec": "aac", "channels": 2,
                                  "sample_rate": 48000.0, "duration_seconds": None}

    _fake_ffprobe(monkeypatch, [{"index": 0, "codec_type": "video", "codec_name": "h264"}], duration="30")
    silent = client.post("/api/probe", files={"file": ("screen.mp4", io.BytesIO(b"video"), "video/mp4")}).json()
    assert silent["has_audio"] is False and silent["transcribable"] is False
    assert "no audio track" in silent["reason"]

    _fake_ffprobe(monkeypatch, [{"index": 0, "codec_type": "audio", "codec_name": "mp3"}], duration="9000")
    long_file = client.post("/api/probe", files={"file": ("lecture.mp3", io.BytesIO(b"audio"), "audio/mpeg")}).json()
    assert long_file["exceeds_audio_limit"] is True and long_file["transcribable"] is False


def test_real_ffmpeg_video_is_probed_and_its_audio_track_is_transcribed(tmp_path):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg is not installed")
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-t", "2",
                    "-c:v", "mpeg4", "-c:a", "aac", "-y", str(video)], check=True)
    client, made = make_client()
    probed = client.post("/api/probe", files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")}).json()
    assert probed["has_video"] and probed["has_audio"] and probed["transcribable"]
    assert 1.5 <= probed["duration_seconds"] <= 2.5

    job_id = client.post("/api/jobs", files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    for _ in range(500):
        state = client.get(f"/api/jobs/{job_id}", params={"wait": 0.5}).json()
        if state["status"] in {"completed", "failed"}:
            break
    assert state["status"] == "completed", state.get("error")
    audio, _options = made[("tiny", "int8")].calls[0]
    # About two seconds at 16 kHz; a silent or video-only decode would be empty.
    assert len(audio) > 20000


def test_unload_releases_resident_models_and_waits_for_no_running_job(monkeypatch):
    mock_audio_tools(monkeypatch)
    client, _ = make_client()
    client.post("/api/models/load", json={"model": "tiny", "compute_type": "int8"})
    assert client.get("/api/health").json()["loaded_models"] == 1
    body = client.post("/api/models/unload").json()
    assert body["loaded_models"] == [] and body["unloaded"][0]["model"] == "tiny"
    assert client.get("/api/health").json()["loaded_models"] == 0

    blocking_client, model = _blocking_client()
    _upload(blocking_client)
    assert model.started.wait(timeout=5)
    refused = blocking_client.post("/api/models/unload")
    assert refused.status_code == 409
    model.release.set()


def test_batch_export_writes_text_sections_and_json_once_finished(monkeypatch):
    mock_audio_tools(monkeypatch)
    client, _ = make_client()
    response = client.post("/api/batches",
                           files=[("files", ("one.wav", io.BytesIO(b"a"), "audio/wav")),
                                  ("files", ("two.wav", io.BytesIO(b"b"), "audio/wav"))],
                           data={"model": "tiny", "compute_type": "int8"})
    batch_id = response.json()["id"]
    for _ in range(300):
        if client.get(f"/api/batches/{batch_id}").json()["done"]:
            break
        time.sleep(0.01)
    text = client.get(f"/api/batches/{batch_id}/export", params={"format": "txt"})
    assert text.status_code == 200
    assert "== one.wav (completed) ==\nhello\n" in text.text and "== two.wav (completed) ==\nhello\n" in text.text
    payload = client.get(f"/api/batches/{batch_id}/export").json()
    assert payload["format"] == "stt-studio-batch-results" and payload["version"] == 1
    assert [f["filename"] for f in payload["files"]] == ["one.wav", "two.wav"]
    assert payload["files"][0]["segments"][0]["text"] == "hello"
    assert client.get(f"/api/batches/{batch_id}/export", params={"format": "srt"}).status_code == 422
    assert client.get("/api/batches/nope/export").status_code == 404


def test_upload_without_audio_is_refused_before_it_is_queued(monkeypatch):
    _fake_ffprobe(monkeypatch, [{"codec_type": "video", "index": 0, "width": 2, "height": 2}])
    client, made = make_client()
    single = client.post("/api/jobs", files={"file": ("screen.mp4", b"video", "video/mp4")},
                         data={"model": "tiny", "compute_type": "int8"})
    assert single.status_code == 422
    assert single.json()["detail"] == "The file has no audio track to transcribe."
    batch = client.post("/api/batches", files=[("files", ("screen.mp4", b"video", "video/mp4"))],
                        data={"model": "tiny", "compute_type": "int8"})
    assert batch.status_code == 422
    assert client.get("/api/health").json()["active_jobs"] == 0
    assert made == {}
