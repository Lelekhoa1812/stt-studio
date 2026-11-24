import asyncio
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app import ModelManager, create_app


def _readable_pipe(value, default: bytes = b""):
    """A Popen stand-in must expose read(). subprocess.PIPE is -1, which is truthy and is not a stream."""
    if hasattr(value, "read"):
        return value
    return io.BytesIO(default)


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, source, **kwargs):
        self.calls.append((source, kwargs))
        sample = SimpleNamespace(start=0.25, end=1.5, text=" hello ",
                                  avg_logprob=-0.2, no_speech_prob=0.03)
        return iter([sample]), SimpleNamespace(language=kwargs.get("language") or "en",
                                                 language_probability=0.98, duration=1.5)


def mock_audio_tools(monkeypatch, duration=1.0, decode_error=False, decoded_duration=0.000125):
    from backend import app as app_module

    calls = []

    def probe_body() -> bytes:
        body = {"format": {"duration": str(duration)}, "streams": [{"codec_type": "audio", "duration": str(duration)}]}
        return json.dumps(body).encode()

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout=probe_body(), stderr=b"", returncode=0)
        if command[-1] == "-":
            # silencedetect writes no file; a long recording asks for pauses before it decodes parts.
            return SimpleNamespace(stdout=b"", stderr=b"")
        if decode_error:
            kwargs["stderr"].write(b"corrupt audio stream")
            import subprocess
            raise subprocess.CalledProcessError(1, command)
        Path(command[-1]).write_bytes(b"\0" * (int(decoded_duration * 16000) * 4))
        return SimpleNamespace(stdout=b"", stderr=b"")

    class _Probe:
        """Stand-in for ffprobe pipes and cancellable ffmpeg decode."""

        def __init__(self, command, **kwargs):
            calls.append(command)
            self.returncode = 0
            self.pid = 4242
            if command and command[0] == "ffprobe":
                self.stdout = io.BytesIO(probe_body())
                self.stderr = io.BytesIO(b"")
            else:
                self.stdout = _readable_pipe(kwargs.get("stdout"))
                self.stderr = _readable_pipe(kwargs.get("stderr"))
                if command and command[0] == "ffmpeg" and command[-1] != "-":
                    if decode_error:
                        if hasattr(self.stderr, "write"):
                            self.stderr.write(b"corrupt audio stream")
                        self.returncode = 1
                    else:
                        Path(command[-1]).write_bytes(b"\0" * (int(decoded_duration * 16000) * 4))

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(app_module.subprocess, "run", run)
    monkeypatch.setattr(app_module.subprocess, "Popen", _Probe)
    return calls


def make_client():
    made = {}

    def factory(model, compute_type):
        made[(model, compute_type)] = FakeModel()
        return made[(model, compute_type)]

    manager = ModelManager(factory)
    api = create_app(manager)
    return TestClient(api, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)), made


def multipart_body(file_bytes: bytes, boundary="test-boundary"):
    return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()


def asgi_upload(app, body: bytes, *, content_length=True, chunk_size=65536):
    async def run():
        headers = [(b"host", b"127.0.0.1:8765"),
                   (b"content-type", b"multipart/form-data; boundary=test-boundary")]
        if content_length:
            headers.append((b"content-length", str(len(body)).encode()))
        chunks = [body[i:i + chunk_size] for i in range(0, len(body), chunk_size)] or [b""]
        received_chunks = 0
        sent = []

        async def receive():
            nonlocal received_chunks
            if received_chunks < len(chunks):
                index = received_chunks
                received_chunks += 1
                return {"type": "http.request", "body": chunks[index],
                        "more_body": index < len(chunks) - 1}
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/api/jobs", "raw_path": b"/api/jobs",
                 "query_string": b"", "headers": headers, "client": ("127.0.0.1", 4567),
                 "server": ("127.0.0.1", 8765)}
        await app(scope, receive, send)
        status = next(message["status"] for message in sent if message["type"] == "http.response.start")
        response_body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
        return status, json.loads(response_body) if response_body.startswith(b"{") else {}, received_chunks

    return asyncio.run(run())


class NoOpManager:
    def snapshot(self):
        return []

    def transcribe(self, *_args, **_kwargs):
        return [], {"language": "en", "language_probability": 1.0, "duration": 1.0}


def test_health_model_list_and_load_warmup(monkeypatch):
    # Keep this test independent of whether mlx-audio is installed on the host.
    monkeypatch.setenv("WHISPER_DISABLE_MLX", "1")
    client, made = make_client()
    assert client.get("/api/health").json() == {"status": "ok", "active_jobs": 0, "loaded_models": 0}
    models = client.get("/api/models").json()
    assert models["default_model"] == "small"
    assert "int8" in models["compute_types"]
    assert models["runtimes"] == ["faster-whisper"]
    catalog = {entry["id"]: entry for entry in models["models"]}
    assert catalog["large-v3"]["family"] == "Whisper"
    assert catalog["Systran/faster-distil-whisper-large-v3"]["family"] == "Distil-Whisper"
    assert catalog["small"]["publisher"] == "SYSTRAN"
    assert catalog["small"]["family_publisher"] == "OpenAI"
    assert catalog["small"]["publisher_url"] == "https://huggingface.co/Systran"
    assert catalog["Systran/faster-distil-whisper-small.en"]["runtime"] == "faster-whisper"
    assert not any(model["id"] == "Qwen/Qwen3-ASR-0.6B" for model in models["models"])
    result = client.post("/api/models/load", json={"model": "tiny", "compute_type": "int8"})
    assert result.status_code == 200
    assert result.json()["warmup"] == "complete"
    assert result.json()["runtime"] == "faster-whisper"
    assert len(made[("tiny", "int8")].calls) == 1


def test_model_manager_accepts_additional_runtime_adapters():
    class FakeAdapter:
        runtime_id = "fake-runtime"

        def load(self, model, compute_type):
            return {"model": model, "compute_type": compute_type}

        def warmup(self, _instance):
            return None

        def transcribe(self, _instance, _audio, **_options):
            return [], {"language": "en", "duration": 0.0}

    client = TestClient(create_app(ModelManager(adapters={"fake-runtime": FakeAdapter()})),
                        base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    loaded = client.post("/api/models/load", json={"model": "vendor/custom-model", "runtime": "fake-runtime"})
    assert loaded.status_code == 200
    assert loaded.json()["runtime"] == "fake-runtime"
    candidate = client.post("/api/models/load", json={"model": "Qwen/Qwen3-ASR-0.6B", "runtime": "fake-runtime"})
    assert candidate.status_code == 422
    assert "not supported yet" in candidate.json()["detail"]
    unavailable = client.post("/api/models/load", json={"model": "vendor/custom-model", "runtime": "missing"})
    assert unavailable.status_code == 422


def test_model_manager_can_run_with_no_installed_adapters():
    client = TestClient(create_app(ModelManager(adapters={})),
                        base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    models = client.get("/api/models").json()
    assert models["runtimes"] == []
    assert models["models"] == []
    result = client.post("/api/models/load", json={"model": "vendor/custom-model"})
    assert result.status_code == 422
    assert "runtime is not installed" in result.json()["detail"]


def test_audio_tools_use_local_only_ffmpeg_protocols(monkeypatch, tmp_path):
    calls = mock_audio_tools(monkeypatch)
    manager = ModelManager(lambda *_: FakeModel())
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    try:
        manager.transcribe(audio_path, model="tiny", compute_type="int8", language="en",
                           beam_size=1, vad_filter=False, initial_prompt=None)
    finally:
        audio_path.unlink(missing_ok=True)
    ffprobe = next(call for call in calls if call[0] == "ffprobe")
    ffmpeg = next(call for call in calls if call[0] == "ffmpeg")
    assert ffprobe[ffprobe.index("-protocol_whitelist") + 1] == "file,crypto"
    assert ffmpeg[ffmpeg.index("-protocol_whitelist") + 1] == "file,crypto"
    assert "data" not in ffprobe[ffprobe.index("-protocol_whitelist") + 1].split(",")
    assert "data" not in ffmpeg[ffmpeg.index("-protocol_whitelist") + 1].split(",")


def test_ffprobe_helpers_reject_stdout_longer_than_one_mebibyte(monkeypatch, tmp_path):
    from backend import app as app_module

    secret = tmp_path / "do-not-leak-this-path.wav"
    huge = b"x" * (app_module._FFPROBE_STDOUT_LIMIT + 4096)
    opened: list[io.BytesIO] = []

    class _Stream(io.BytesIO):
        def __init__(self, payload: bytes):
            super().__init__(payload)
            self.read_total = 0

        def read(self, size=-1):
            chunk = super().read(size)
            self.read_total += len(chunk)
            return chunk

    class _Process:
        """Fake ffprobe process. stdout is longer than the 1 MiB cap."""

        def __init__(self, command, **kwargs):
            self.stdout = _Stream(huge)
            self.stderr = _Stream(b"")
            self.returncode = None
            opened.append(self.stdout)

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(app_module.subprocess, "Popen", _Process)
    for helper in (app_module._probe_audio_duration, app_module._ffprobe_media):
        with pytest.raises(RuntimeError) as caught:
            helper(secret)
        message = str(caught.value)
        assert message == "ffprobe output exceeded 1 MiB"
        assert str(secret) not in message
        assert secret.name not in message
        assert opened[-1].read_total <= app_module._FFPROBE_STDOUT_LIMIT + 1
        assert opened[-1].read_total < len(huge)


def test_catalog_reports_downloaded_models_from_hugging_face_cache(monkeypatch, tmp_path):
    from backend import app as app_module

    monkeypatch.setattr(app_module, "MODEL_CACHE", tmp_path)
    model_dir = tmp_path / "models--Systran--faster-whisper-small" / "snapshots" / "revision"
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(b"cached")
    assert app_module._model_is_cached("Systran/faster-whisper-small")
    assert not app_module._model_is_cached("Systran/faster-whisper-large-v3")
    mlx_dir = tmp_path / "models--mlx-community--Qwen3-ASR-0.6B-8bit" / "snapshots" / "revision"
    mlx_dir.mkdir(parents=True)
    (mlx_dir / "model.safetensors").write_bytes(b"cached")
    assert app_module._model_is_cached("mlx-community/Qwen3-ASR-0.6B-8bit")


def test_job_transcribes_upload_and_returns_segment_metadata(monkeypatch):
    mock_audio_tools(monkeypatch)
    client, made = make_client()
    response = client.post("/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio bytes"), "audio/wav")},
                           data={"model": "tiny", "compute_type": "int8", "language": "vi",
                                 "beam_size": "3", "vad_filter": "true", "initial_prompt": "names"})
    assert response.status_code == 202
    job_id = response.json()["id"]
    for _ in range(100):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert result["status"] == "completed"
    assert result["segments"] == [{"start": 0.25, "end": 1.5, "text": "hello",
                                    "avg_logprob": -0.2, "no_speech_prob": 0.03}]
    assert result["info"]["language"] == "vi"
    audio, options = made[("tiny", "int8")].calls[0]
    assert len(audio) == 2
    assert type(audio).__name__ == "memmap"
    assert options == {"language": "vi", "beam_size": 3, "vad_filter": True, "initial_prompt": "names"}


def test_invalid_settings_empty_upload_unknown_job_and_remote_origin():
    client, _ = make_client()
    assert client.post("/api/models/load", json={"model": "tiny", "compute_type": "bogus"}).status_code == 422
    assert client.post("/api/models/load", json={"model": "tiny", "compute_type": "int8", "runtime": "nemo"}).status_code == 422
    assert client.post("/api/models/load", json={"model": "../../etc/passwd", "compute_type": "int8"}).status_code == 422
    assert client.get("/api/jobs/missing").status_code == 404
    empty = client.post("/api/jobs", files={"file": ("empty.wav", b"", "audio/wav")})
    assert empty.status_code == 422
    denied = client.post("/api/models/load", json={"model": "tiny", "compute_type": "int8"},
                         headers={"Origin": "https://attacker.example"})
    assert denied.status_code == 403


def test_asgi_upload_body_limit_allows_multipart_overhead_and_missing_content_length(monkeypatch):
    from backend import app as app_module
    mock_audio_tools(monkeypatch)
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 128)
    monkeypatch.setattr(app_module, "MAX_MULTIPART_OVERHEAD_BYTES", 512)
    api = create_app(NoOpManager())
    body = multipart_body(b"valid audio")
    status, response, _ = asgi_upload(api, body, content_length=False, chunk_size=7)
    assert status == 202
    assert response["status"] == "queued"


def test_asgi_upload_body_limit_rejects_declared_and_chunked_oversize(monkeypatch):
    from backend import app as app_module
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 128)
    monkeypatch.setattr(app_module, "MAX_MULTIPART_OVERHEAD_BYTES", 512)
    api = create_app(NoOpManager())
    too_large = multipart_body(b"x" * 1000)

    status, response, received_chunks = asgi_upload(api, too_large, content_length=True)
    assert status == 413
    assert "Multipart request exceeds" in response["detail"]
    assert received_chunks == 0

    status, response, received_chunks = asgi_upload(api, too_large, content_length=False, chunk_size=64)
    assert status == 413
    assert "Multipart request exceeds" in response["detail"]
    assert received_chunks <= 11


def test_chunked_limit_closes_multipart_file_spool_created_before_rejection(monkeypatch):
    from backend import app as app_module
    import starlette.formparsers

    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 128)
    monkeypatch.setattr(app_module, "MAX_MULTIPART_OVERHEAD_BYTES", 512)
    created_spools = []
    real_spooled_temp_file = starlette.formparsers.SpooledTemporaryFile

    def tracked_spooled_temp_file(*args, **kwargs):
        stream = real_spooled_temp_file(*args, **kwargs)
        created_spools.append(stream)
        return stream

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", tracked_spooled_temp_file)
    status, response, received_chunks = asgi_upload(
        create_app(NoOpManager()), multipart_body(b"x" * 1200), content_length=False, chunk_size=64)

    assert status == 413
    assert "Multipart request exceeds" in response["detail"]
    assert 0 < received_chunks < 20
    assert len(created_spools) == 1
    assert created_spools[0].closed


def test_remote_peer_is_rejected_even_without_origin():
    from backend.app import create_app
    client = TestClient(create_app(ModelManager(lambda *_: FakeModel())), base_url="http://127.0.0.1",
                         client=("192.0.2.4", 1234))
    assert client.get("/api/health").status_code == 403


def test_failed_job_reports_safe_error_and_removes_temporary_upload(monkeypatch):
    calls = mock_audio_tools(monkeypatch)
    from backend import app as app_module
    created = []
    real_mkstemp = app_module.tempfile.mkstemp

    def capture_mkstemp(*args, **kwargs):
        result = real_mkstemp(*args, **kwargs)
        created.append(Path(result[1]))
        return result

    monkeypatch.setattr(app_module.tempfile, "mkstemp", capture_mkstemp)

    class BrokenModel:
        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError("bad audio at /tmp/whisper-upload-secret.wav")

    client = TestClient(create_app(ModelManager(lambda *_: BrokenModel())), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    response = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")})
    job_id = response.json()["id"]
    for _ in range(100):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] in {"failed", "completed"}:
            break
        time.sleep(0.01)
    assert result["status"] == "failed"
    assert "bad audio" in result["error"]
    assert "whisper-upload-secret" not in result["error"]
    assert created and all(not path.exists() for path in created)
    assert not Path(_decode_command(calls)[-1]).exists()


def _decode_command(calls):
    return next(command for command in calls if command[0] == "ffmpeg")


class PartedModel:
    """One relative segment per part, plus an empty one that must be dropped. Language arrives on part two."""

    def __init__(self):
        self.calls = 0
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def transcribe(self, source, **kwargs):
        self.calls += 1
        call = self.calls

        def segments():
            if call == 2:
                self.second_started.set()
                assert self.release_second.wait(5)
            yield SimpleNamespace(start=0.0, end=1.0, text=" hello ", avg_logprob=-0.2, no_speech_prob=0.01)
            yield SimpleNamespace(start=1.0, end=1.1, text="   ", avg_logprob=-0.2, no_speech_prob=0.9)

        return segments(), SimpleNamespace(language=None if call == 1 else "fr",
                                           language_probability=None if call == 1 else 0.4,
                                           duration=len(source) / 16000)


def _pcm_decodes(calls):
    return [command for command in calls if command[0] == "ffmpeg" and "f32le" in command]


def test_long_audio_is_transcribed_in_parts(monkeypatch):
    from backend import app as app_module
    from backend import diarization

    calls = mock_audio_tools(monkeypatch, duration=11)
    monkeypatch.setattr(app_module, "MAX_AUDIO_SECONDS", 10)
    monkeypatch.setattr(diarization, "availability", lambda: {"available": True, "model": diarization.MODEL_ID,
                                                              "license": "MIT", "reason": None})
    seen = {}

    def fake_diarize(source, *, seconds=None, should_stop=None, cache_dir=None):
        seen["seconds"] = seconds
        seen["source_exists"] = Path(source).is_file()
        return [(10.0, 11.0, "spk_late")]

    monkeypatch.setattr(diarization, "diarize_file", fake_diarize)
    model = PartedModel()
    client = TestClient(create_app(ModelManager(lambda *_: model)), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    response = client.post("/api/jobs", files={"file": ("long.wav", b"audio", "audio/wav")},
                           data={"model": "tiny", "compute_type": "int8", "diarize": "true"})
    job_id = response.json()["id"]
    assert model.second_started.wait(5)
    running = client.get(f"/api/jobs/{job_id}").json()
    # Part one ended at 1.0 s of an 11 s file. Part two must not send progress back to 0.
    assert running["status"] == "running"
    assert running["progress"] == pytest.approx(9.1, abs=0.1)
    model.release_second.set()
    for _ in range(100):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] in {"failed", "completed", "cancelled"}:
            break
        time.sleep(0.01)
    assert result["status"] == "completed", result.get("error")
    assert result["info"]["chunk_count"] == 2
    assert result["info"]["duration"] == 11
    assert result["info"]["language"] == "fr"
    assert result["info"]["language_probability"] == 0.4
    assert [(row["start"], row["end"], row["text"]) for row in result["segments"]] == [
        (0.0, 1.0, "hello"), (10.0, 11.0, "hello")]
    assert "speaker" not in result["segments"][0]
    assert result["segments"][1]["speaker"] == "Speaker 1"
    assert seen == {"seconds": 11, "source_exists": True}
    assert model.calls == 2
    decodes = _pcm_decodes(calls)
    assert len(decodes) == 2
    assert [float(command[command.index("-t") + 1]) for command in decodes] == [10.0, 1.0]
    assert len({command[-1] for command in decodes}) == 1
    assert not Path(decodes[0][-1]).exists()
    assert all(command[command.index("-protocol_whitelist") + 1] == "file,crypto"
               for command in calls if command[0] in {"ffmpeg", "ffprobe"})
    assert all("data" not in command[command.index("-protocol_whitelist") + 1].split(",")
               for command in calls if command[0] in {"ffmpeg", "ffprobe"})


def test_cancel_after_the_first_part_keeps_that_part(monkeypatch):
    from backend import app as app_module

    calls = mock_audio_tools(monkeypatch, duration=11)
    monkeypatch.setattr(app_module, "MAX_AUDIO_SECONDS", 10)

    class FirstPartModel:
        def __init__(self):
            self.calls = 0
            self.ready = threading.Event()
            self.release = threading.Event()

        def transcribe(self, source, **kwargs):
            self.calls += 1

            def segments():
                yield SimpleNamespace(start=0.2, end=0.8, text="kept", avg_logprob=-0.1, no_speech_prob=0.0)
                self.ready.set()
                assert self.release.wait(5)

            return segments(), SimpleNamespace(language="en", language_probability=1.0, duration=len(source) / 16000)

    model = FirstPartModel()
    client = TestClient(create_app(ModelManager(lambda *_: model)), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    job_id = client.post("/api/jobs", files={"file": ("long.wav", b"audio", "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert model.ready.wait(5)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    model.release.set()
    assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled")
    final = client.get(f"/api/jobs/{job_id}").json()
    assert [(row["start"], row["end"], row["text"]) for row in final["segments"]] == [(0.2, 0.8, "kept")]
    assert model.calls == 1
    assert len(_pcm_decodes(calls)) == 1


def test_decode_error_is_actionable_and_removes_pcm_temp(monkeypatch):
    calls = mock_audio_tools(monkeypatch, decode_error=True)
    client, _ = make_client()
    response = client.post("/api/jobs", files={"file": ("broken.wav", b"not audio", "audio/wav")})
    job_id = response.json()["id"]
    for _ in range(100):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] in {"failed", "completed"}:
            break
        time.sleep(0.01)
    assert result["status"] == "failed"
    assert "Audio decoding failed: corrupt audio stream" in result["error"]
    assert [command[0] for command in calls] == ["ffprobe", "ffprobe", "ffmpeg"]
    assert not Path(calls[-1][-1]).exists()


def test_underreported_duration_is_transcribed_in_parts(monkeypatch):
    calls = mock_audio_tools(monkeypatch, duration=0.5, decoded_duration=1.2)
    from backend import app as app_module
    monkeypatch.setattr(app_module, "MAX_AUDIO_SECONDS", 1)
    model = FakeModel()
    client = TestClient(create_app(ModelManager(lambda *_: model)), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    response = client.post("/api/jobs", files={"file": ("long.wav", b"audio", "audio/wav")})
    job_id = response.json()["id"]
    for _ in range(100):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] in {"failed", "completed"}:
            break
        time.sleep(0.01)
    assert result["status"] == "completed", result.get("error")
    assert result["info"]["chunk_count"] == 2
    assert [(row["start"], row["end"], row["text"]) for row in result["segments"]] == [
        (0.25, 1.5, "hello"), (1.25, 2.5, "hello")]
    assert len(model.calls) == 2
    decode = _decode_command(calls)
    # The first decode is still capped, so a short probe cannot write an unbounded PCM file.
    assert decode[decode.index("-t") + 1] == "6"
    assert "-ss" not in decode
    assert len(_pcm_decodes(calls)) == 1
    assert not Path(decode[-1]).exists()


def test_a_filled_decode_cap_continues_with_later_parts(monkeypatch):
    from backend import app as app_module

    monkeypatch.setattr(app_module, "MAX_AUDIO_SECONDS", 1)
    calls = []

    probe_stdout = json.dumps(
        {"format": {"duration": "0.5"}, "streams": [{"codec_type": "audio", "duration": "0.5"}]}).encode()

    class _Probe:
        def __init__(self, command, **kwargs):
            calls.append(command)
            self.returncode = 0
            self.pid = 4242
            self.stdout = io.BytesIO(probe_stdout if command and command[0] == "ffprobe" else b"")
            self.stderr = _readable_pipe(kwargs.get("stderr"))
            if command and command[0] == "ffmpeg":
                if "-ss" in command:
                    start = float(command[command.index("-ss") + 1])
                    seconds = 0.4 if start < 7 else 0
                else:
                    seconds = 6
                Path(command[-1]).write_bytes(b"\0" * (int(seconds * 16000) * 4))

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout=probe_stdout, stderr=b"", returncode=0)
        if "-ss" in command:
            start = float(command[command.index("-ss") + 1])
            seconds = 0.4 if start < 7 else 0
        else:
            seconds = 6
        Path(command[-1]).write_bytes(b"\0" * (int(seconds * 16000) * 4))
        return SimpleNamespace(stdout=b"", stderr=b"")

    monkeypatch.setattr(app_module.subprocess, "run", run)
    monkeypatch.setattr(app_module.subprocess, "Popen", _Probe)
    model = FakeModel()
    client = TestClient(create_app(ModelManager(lambda *_: model)), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    job_id = client.post("/api/jobs", files={"file": ("long.wav", b"audio", "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    for _ in range(100):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] in {"failed", "completed"}:
            break
        time.sleep(0.01)
    assert result["status"] == "completed", result.get("error")
    assert result["info"]["chunk_count"] == 7
    assert result["segments"][-1]["start"] >= 6
    assert len(model.calls) == 7
    assert any("-ss" in command for command in _pcm_decodes(calls))


def test_cut_points_prefer_a_pause_and_stay_within_the_part_limit(monkeypatch, tmp_path):
    from backend import app as app_module

    def run(command, **kwargs):
        stderr = ("silence_start: 9.2\nsilence_end: 10.0 | silence_duration: 0.8\n"
                  "silence_start: 10.4\nsilence_end: 11.0 | silence_duration: 0.6\n")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=stderr.encode())

    monkeypatch.setattr(app_module.subprocess, "run", run)
    cuts = app_module._find_cut_points(tmp_path / "talk.wav", duration=25, chunk_seconds=10)
    # 9.6 is the middle of the pause that ends on the boundary. 10.7 is past the 10 s limit, so it is not used.
    assert cuts[0] == pytest.approx(9.6)
    assert all(cut <= 10 or cut <= (cuts[index - 1] + 10 if index else 10) for index, cut in enumerate(cuts))
    spans = app_module._part_spans(cuts, 25, 10)
    assert spans[0] == pytest.approx((0.0, 9.6))
    assert all(length <= 10 + 1e-6 for _start, length in spans)
    assert spans[-1][0] + spans[-1][1] == pytest.approx(25)


def test_status_routes_stay_responsive_during_inference_and_host_is_validated(monkeypatch):
    mock_audio_tools(monkeypatch)
    entered = threading.Event()
    release = threading.Event()

    class SlowModel(FakeModel):
        def transcribe(self, source, **kwargs):
            if isinstance(source, str):
                return super().transcribe(source, **kwargs)
            entered.set()
            assert release.wait(2)
            return super().transcribe(source, **kwargs)

    client = TestClient(create_app(ModelManager(lambda *_: SlowModel())), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    response = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")})
    job_id = response.json()["id"]
    assert entered.wait(1)
    start = time.monotonic()
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/models").status_code == 200
    assert time.monotonic() - start < 0.5
    assert client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 403
    release.set()
    for _ in range(100):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert result["status"] == "completed"


def test_model_switch_is_serialized_and_only_one_model_stays_loaded(monkeypatch):
    mock_audio_tools(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    created = []

    class BlockingModel(FakeModel):
        def transcribe(self, source, **kwargs):
            if kwargs.get("language") == "vi":
                entered.set()
                assert release.wait(2)
            return super().transcribe(source, **kwargs)

    def factory(model, compute_type):
        created.append(model)
        return BlockingModel()

    manager = ModelManager(factory)
    with ThreadPoolExecutor(max_workers=2) as pool:
        inference = pool.submit(manager.transcribe, Path("unused"), model="tiny", compute_type="int8",
                                language="vi", beam_size=1, vad_filter=False, initial_prompt=None)
        assert entered.wait(1)
        load = pool.submit(manager.load, "base", "int8")
        time.sleep(0.05)
        assert created == ["tiny"]
        release.set()
        inference.result(timeout=2)
        assert load.result(timeout=2)["loaded"] is True
    assert created == ["tiny", "base"]
    assert manager.snapshot() == [{"model": "base", "compute_type": "int8", "runtime": "faster-whisper"}]


class FakeMlxModel:
    def __init__(self, result=None, support_languages=None):
        self.calls = []
        self.result = result
        self.config = SimpleNamespace(support_languages=support_languages or [])

    def generate(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return self.result if self.result is not None else SimpleNamespace(text="", segments=None, sentences=None)


def _fake_mlx_adapter(model):
    from backend import app as app_module

    return app_module.MlxAudioAdapter(snapshot=lambda repo: f"/cache/{repo}",
                                      loader=lambda path: model,
                                      to_array=lambda audio: ("array", len(audio)))


def test_mlx_adapter_normalizes_sentences_and_ignores_whisper_only_options():
    import numpy as np

    result = SimpleNamespace(text="hello there", language=None,
                             sentences=[SimpleNamespace(start=0.0, end=1.0, text=" hello "),
                                        SimpleNamespace(start=1.0, end=2.0, text="  ")])
    model = FakeMlxModel(result)
    adapter = _fake_mlx_adapter(model)
    instance = adapter.load("mlx-community/parakeet-tdt-0.6b-v3", "native")
    adapter.warmup(instance)
    rows, meta = adapter.transcribe(instance, np.zeros(32000, dtype=np.float32), language="fr",
                                    beam_size=5, vad_filter=True, initial_prompt="names")
    assert rows == [{"start": 0.0, "end": 1.0, "text": "hello", "avg_logprob": None, "no_speech_prob": None}]
    assert meta["duration"] == 2.0 and meta["language"] == "fr"
    # Parakeet is auto-detect only: no language or Whisper options reach generate().
    assert model.calls[-1][1] == {}
    assert model.calls[0][1] == {}  # warmup


def test_mlx_adapter_maps_language_codes_for_name_based_models():
    import numpy as np

    model = FakeMlxModel(SimpleNamespace(text="bonjour", segments=None, sentences=None, language=["French"]),
                         support_languages=["French", "English"])
    adapter = _fake_mlx_adapter(model)
    from backend import app as app_module

    instance = app_module._MlxInstance(repository="mlx-community/Qwen3-ASR-0.6B-8bit", model=model,
                                       language_hint="names")
    rows, meta = adapter.transcribe(instance, np.zeros(16000, dtype=np.float32), language="fr",
                                    beam_size=1, vad_filter=False, initial_prompt=None)
    assert model.calls[-1][1] == {"language": "French"}
    assert meta["language"] == "fr"
    assert rows == [{"start": 0.0, "end": 1.0, "text": "bonjour", "avg_logprob": None, "no_speech_prob": None}]


def test_mlx_adapter_rejects_language_the_model_does_not_support():
    import numpy as np
    import pytest
    from backend import app as app_module

    model = FakeMlxModel(support_languages=["English"])
    instance = app_module._MlxInstance(repository="x/y", model=model, language_hint="names")
    with pytest.raises(ValueError, match="not supported"):
        _fake_mlx_adapter(model).transcribe(instance, np.zeros(16000, dtype=np.float32), language="ja",
                                            beam_size=1, vad_filter=False, initial_prompt=None)


def test_mlx_segments_fall_back_to_plain_text_and_empty_output():
    from backend import app as app_module

    assert app_module._mlx_rows(SimpleNamespace(text=" plain ", segments=None, sentences=None), 3.0) == [
        {"start": 0.0, "end": 3.0, "text": "plain", "avg_logprob": None, "no_speech_prob": None}]
    assert app_module._mlx_rows(SimpleNamespace(text="  ", segments=None, sentences=None), 3.0) == []


def test_compute_type_is_normalized_for_mlx_and_validated_for_faster_whisper(monkeypatch):
    from backend import app as app_module

    repo = "mlx-community/Qwen3-ASR-0.6B-8bit"
    supported_row = {"id": repo, "family": "test", "runtime": "mlx-audio", "supported": True,
                     "capabilities": {"languageHint": "names"}}
    monkeypatch.setitem(app_module.MODEL_CATALOG_BY_ID, repo, supported_row)
    assert app_module._validate_model_settings(repo, "int8", 5, None,
                                               available_runtimes={"faster-whisper", "mlx-audio"}) == (
        "mlx-audio", "native")
    assert app_module._validate_model_settings("small", "int8", 5, None,
                                               available_runtimes={"faster-whisper"}) == ("faster-whisper", "int8")


def test_models_endpoint_reports_runtime_capabilities_and_hides_candidates(monkeypatch):
    from backend import app as app_module

    client = TestClient(create_app(ModelManager(lambda *_: FakeModel(), adapters={
        "faster-whisper": app_module.FasterWhisperAdapter(lambda *_: FakeModel()),
        "mlx-audio": _fake_mlx_adapter(FakeMlxModel()),
    })), base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    body = client.get("/api/models").json()
    assert body["runtime_capabilities"]["mlx-audio"]["computeTypes"] is False
    assert body["runtime_capabilities"]["faster-whisper"]["computeTypes"] is True
    ids = {model["id"] for model in body["models"]}
    assert "small" in ids
    # Only the validated MLX conversion (Qwen3-ASR) is offered; the Parakeet candidate stays hidden.
    mlx_families = {model["family"] for model in body["models"] if model["runtime"] == "mlx-audio"}
    assert mlx_families == {"Qwen3-ASR (MLX conversion)"}
    small = next(model for model in body["models"] if model["id"] == "small")
    assert small["capabilities"]["vad"] is True


def test_mlx_adapter_is_only_registered_on_apple_silicon_with_package(monkeypatch):
    from backend import app as app_module

    monkeypatch.setattr(app_module.platform, "machine", lambda: "x86_64")
    assert app_module._default_adapters(lambda *_: None).keys() == {"faster-whisper"}
    monkeypatch.setattr(app_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(app_module.sys, "platform", "darwin")
    monkeypatch.setattr(app_module.importlib.util, "find_spec", lambda name: None)
    assert app_module._default_adapters(lambda *_: None).keys() == {"faster-whisper"}
    monkeypatch.setattr(app_module.importlib.util, "find_spec", lambda name: object())
    assert app_module._default_adapters(lambda *_: None).keys() == {"faster-whisper", "mlx-audio"}
    monkeypatch.setenv("WHISPER_DISABLE_MLX", "1")
    assert app_module._default_adapters(lambda *_: None).keys() == {"faster-whisper"}


class PausingModel:
    """Yields three timed segments and pauses after the first until the test releases it."""

    def __init__(self):
        self.paused = threading.Event()
        self.resume = threading.Event()

    def transcribe(self, source, **kwargs):
        def segments():
            for index in range(3):
                yield SimpleNamespace(start=float(index), end=float(index + 1), text=f"part {index}",
                                      avg_logprob=-0.1, no_speech_prob=0.0)
                if index == 0:
                    self.paused.set()
                    self.resume.wait(timeout=5)
        return segments(), SimpleNamespace(language="en", language_probability=1.0, duration=3.0)


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_running_job_reports_measured_progress_and_can_be_cancelled(monkeypatch):
    mock_audio_tools(monkeypatch, duration=3.0)
    model = PausingModel()
    client = TestClient(create_app(ModelManager(lambda *_: model)),
                        base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    job_id = client.post("/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: model.paused.is_set())
    running = client.get(f"/api/jobs/{job_id}").json()
    assert running["status"] == "running"
    assert running["progress"] == pytest.approx(33.3, abs=0.1)  # 1.0 s of 3.0 s measured
    assert [row["text"] for row in running["segments"]] == ["part 0"]  # partial results are visible

    cancelled = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["cancel_requested"] is True
    model.resume.set()

    assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled")
    final = client.get(f"/api/jobs/{job_id}").json()
    assert [row["text"] for row in final["segments"]] == ["part 0"]  # the segment after cancel is never kept
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409  # already finished
    assert client.post("/api/jobs/missing/cancel").status_code == 404


def test_completed_job_reports_full_progress_and_refuses_cancel(monkeypatch):
    mock_audio_tools(monkeypatch)
    client, _made = make_client()
    job_id = client.post("/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] == "completed")
    result = client.get(f"/api/jobs/{job_id}").json()
    assert result["progress"] == 100.0
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_batch_runs_each_file_in_order_with_its_own_status(monkeypatch):
    mock_audio_tools(monkeypatch)
    client, _made = make_client()
    response = client.post("/api/batches",
                           files=[("files", ("one.wav", io.BytesIO(b"one"), "audio/wav")),
                                  ("files", ("two.wav", io.BytesIO(b"two"), "audio/wav"))],
                           data={"model": "tiny", "compute_type": "int8"})
    assert response.status_code == 202
    batch_id = response.json()["id"]
    assert response.json()["total"] == 2
    assert _wait_for(lambda: client.get(f"/api/batches/{batch_id}").json()["done"])
    batch = client.get(f"/api/batches/{batch_id}").json()
    assert batch["counts"]["completed"] == 2
    assert [job["filename"] for job in batch["jobs"]] == ["one.wav", "two.wav"]
    assert all(job["batch_id"] == batch_id and job["segments"] for job in batch["jobs"])
    assert client.get("/api/batches/missing").status_code == 404


def test_batch_rejects_too_many_files(monkeypatch):
    from backend import app as app_module
    mock_audio_tools(monkeypatch)
    monkeypatch.setattr(app_module, "MAX_BATCH_FILES", 1)
    client, _made = make_client()
    response = client.post("/api/batches",
                           files=[("files", ("one.wav", io.BytesIO(b"one"), "audio/wav")),
                                  ("files", ("two.wav", io.BytesIO(b"two"), "audio/wav"))],
                           data={"model": "tiny", "compute_type": "int8"})
    assert response.status_code == 422
    assert "at most 1 files" in response.json()["detail"]


def test_cancelling_a_batch_stops_the_running_file_and_skips_the_queued_one(monkeypatch):
    mock_audio_tools(monkeypatch, duration=3.0)

    class CountingPausingModel(PausingModel):
        calls = 0

        def transcribe(self, source, **kwargs):
            CountingPausingModel.calls += 1
            return super().transcribe(source, **kwargs)

    model = CountingPausingModel()
    client = TestClient(create_app(ModelManager(lambda *_: model)),
                        base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    batch_id = client.post("/api/batches",
                           files=[("files", ("one.wav", io.BytesIO(b"one"), "audio/wav")),
                                  ("files", ("two.wav", io.BytesIO(b"two"), "audio/wav"))],
                           data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: model.paused.is_set())
    cancelled = client.post(f"/api/batches/{batch_id}/cancel").json()
    assert cancelled["counts"]["cancelled"] == 1  # the queued file is cancelled at once
    model.resume.set()
    assert _wait_for(lambda: client.get(f"/api/batches/{batch_id}").json()["done"])
    batch = client.get(f"/api/batches/{batch_id}").json()
    assert batch["counts"] == {"queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 2}
    assert CountingPausingModel.calls == 1  # the second file never reached the model
    assert [row["text"] for row in batch["jobs"][0]["segments"]] == ["part 0"]
    assert client.post("/api/batches/missing/cancel").status_code == 404


def test_auto_and_blank_language_mean_detection(monkeypatch):
    mock_audio_tools(monkeypatch)
    client, made = make_client()
    for language in ("auto", ""):
        response = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")},
                               data={"model": "tiny", "compute_type": "int8", "language": language})
        assert response.status_code == 202
        job_id = response.json()["id"]
        for _ in range(100):
            result = client.get(f"/api/jobs/{job_id}").json()
            if result["status"] == "completed":
                break
            time.sleep(0.01)
        assert result["language"] is None
    assert [options["language"] for _, options in made[("tiny", "int8")].calls] == [None, None]


def test_english_only_model_rejects_other_languages_before_the_upload(monkeypatch, tmp_path):
    import tempfile
    from backend import app as app_module

    assert app_module.MODEL_CATALOG_BY_ID["tiny.en"]["languages"] == "English"
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client, _ = make_client()
    form = {"model": "tiny.en", "compute_type": "int8", "language": "vi"}
    response = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")}, data=form)
    assert response.status_code == 422
    batch = client.post("/api/batches", files=[("files", ("one.wav", b"one", "audio/wav"))], data=form)
    assert batch.status_code == 422
    assert list(tmp_path.glob("whisper-upload-*")) == []


def test_english_only_model_accepts_english_auto_and_omitted_language(monkeypatch):
    mock_audio_tools(monkeypatch)
    client, _ = make_client()
    base = {"model": "tiny.en", "compute_type": "int8"}
    for extra in ({"language": "en"}, {"language": "auto"}, {}):
        response = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")},
                               data={**base, **extra})
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] == "completed")


def test_multilingual_model_still_accepts_a_non_english_code(monkeypatch):
    from backend import app as app_module

    assert app_module.MODEL_CATALOG_BY_ID["tiny"]["languages"] == "multilingual"
    mock_audio_tools(monkeypatch)
    client, _ = make_client()
    response = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")},
                           data={"model": "tiny", "compute_type": "int8", "language": "vi"})
    assert response.status_code == 202


def test_mlx_language_outside_its_names_is_422_before_the_upload(monkeypatch, tmp_path):
    import tempfile
    from backend import app as app_module

    assert "he" in app_module.WHISPER_LANGUAGE_CODES
    assert "he" not in app_module.MLX_LANGUAGE_NAMES
    assert "fr" in app_module.MLX_LANGUAGE_NAMES
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client = TestClient(create_app(ModelManager(adapters={
        "mlx-audio": _fake_mlx_adapter(FakeMlxModel()),
    })), base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    form = {"model": "mlx-community/Qwen3-ASR-0.6B-8bit", "language": "he"}
    response = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")}, data=form)
    assert response.status_code == 422
    assert "he" in response.json()["detail"]
    batch = client.post("/api/batches", files=[("files", ("one.wav", b"one", "audio/wav"))], data=form)
    assert batch.status_code == 422
    assert list(tmp_path.glob("whisper-upload-*")) == []

    mock_audio_tools(monkeypatch)
    for extra in ({"language": "fr"}, {"language": "auto"}, {}):
        accepted = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")},
                               data={"model": "mlx-community/Qwen3-ASR-0.6B-8bit", **extra})
        assert accepted.status_code == 202
        job_id = accepted.json()["id"]
        assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] == "completed")


def test_unknown_language_is_422_and_nothing_is_stored_in_the_upload_dir(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client, _ = make_client()
    response = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")},
                           data={"model": "tiny", "compute_type": "int8", "language": "xx"})
    assert response.status_code == 422
    assert "'xx'" in response.json()["detail"]
    batch = client.post("/api/batches", files=[("files", ("one.wav", b"one", "audio/wav"))],
                        data={"model": "tiny", "compute_type": "int8", "language": "xx"})
    assert batch.status_code == 422
    assert list(tmp_path.glob("whisper-upload-*")) == []


def test_initial_prompt_is_capped_at_1000_characters(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    mock_audio_tools(monkeypatch)
    client, _ = make_client()
    form = {"model": "tiny", "compute_type": "int8"}
    too_long = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")},
                           data={**form, "initial_prompt": "a" * 1001})
    assert too_long.status_code == 422
    batch = client.post("/api/batches", files=[("files", ("one.wav", b"one", "audio/wav"))],
                        data={**form, "initial_prompt": "a" * 1001})
    assert batch.status_code == 422
    assert list(tmp_path.glob("whisper-upload-*")) == []
    accepted = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")},
                           data={**form, "initial_prompt": "a" * 1000})
    assert accepted.status_code == 202
    assert _wait_for(lambda: client.get(f"/api/jobs/{accepted.json()['id']}").json()["status"] == "completed")


def test_language_set_matches_faster_whisper_and_covers_mlx_names():
    from backend import app as app_module

    assert len(app_module.WHISPER_LANGUAGE_CODES) == 100
    assert {"yue", "vi", "en"} <= app_module.WHISPER_LANGUAGE_CODES
    assert set(app_module.MLX_LANGUAGE_NAMES) <= app_module.WHISPER_LANGUAGE_CODES
    assert app_module._normalize_language("  VI ") == "vi"
    assert app_module._normalize_language(None) is None


def asgi_json_post(app, path: str, body: bytes, *, content_length=True, chunk_size=65536):
    async def run():
        headers = [(b"host", b"127.0.0.1:8765"), (b"content-type", b"application/json")]
        if content_length:
            headers.append((b"content-length", str(len(body)).encode()))
        chunks = [body[i:i + chunk_size] for i in range(0, len(body), chunk_size)] or [b""]
        received_chunks = 0
        sent = []

        async def receive():
            nonlocal received_chunks
            if received_chunks < len(chunks):
                index = received_chunks
                received_chunks += 1
                return {"type": "http.request", "body": chunks[index], "more_body": index < len(chunks) - 1}
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
                 "headers": headers, "client": ("127.0.0.1", 4567), "server": ("127.0.0.1", 8765)}
        await app(scope, receive, send)
        status = next(message["status"] for message in sent if message["type"] == "http.response.start")
        response_body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
        return status, json.loads(response_body) if response_body.startswith(b"{") else {}, received_chunks

    return asyncio.run(run())


def test_two_mebibyte_json_body_is_413_on_a_non_multipart_route():
    client, _ = make_client()
    body = {"model": "tiny", "compute_type": "int8", "runtime": "x" * (2 * 1024 * 1024)}
    response = client.post("/api/models/load", json=body)
    assert response.status_code == 413
    assert "Request body exceeds" in response.json()["detail"]


def test_undeclared_json_body_is_capped_while_streaming():
    api = create_app(NoOpManager())
    payload = json.dumps({"model": "tiny", "runtime": "x" * (2 * 1024 * 1024)}).encode()
    status, response, received = asgi_json_post(api, "/api/models/load", payload, content_length=False, chunk_size=65536)
    assert status == 413
    assert "Request body exceeds" in response["detail"]
    assert received < len(payload) // 65536


def test_ai_route_accepts_up_to_four_mebibytes_but_not_more():
    client, _ = make_client()
    # Missing job: the request gets past the body cap and reaches the route, which answers 404.
    segments = [{"text": "word " * 20_000} for _ in range(20)]  # about 2 MiB, each segment under the per-segment cap
    accepted = client.post("/api/jobs/missing/ai", json={"mode": "summary", "segments": segments})
    assert accepted.status_code == 404
    too_big = client.post("/api/jobs/missing/ai", json={"mode": "summary", "segments": segments * 2 + [{"text": "x" * 1_000_000}]})
    assert too_big.status_code == 413


def test_ai_segment_text_longer_than_the_transcript_limit_is_refused(monkeypatch):
    from backend import nim

    mock_audio_tools(monkeypatch)
    client, _ = make_client()
    job_id = client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    for _ in range(100):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "completed":
            break
        time.sleep(0.01)
    response = client.post(f"/api/jobs/{job_id}/ai",
                           json={"mode": "summary", "segments": [{"text": "x" * (nim.MAX_TRANSCRIPT_CHARS + 1)}]})
    assert response.status_code == 422
    assert f"{nim.MAX_TRANSCRIPT_CHARS:,}" in response.json()["detail"]
