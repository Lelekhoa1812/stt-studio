import io
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app import ModelManager, create_app


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, source, **kwargs):
        self.calls.append((source, kwargs))
        sample = SimpleNamespace(start=0.25, end=1.5, text=" hello ",
                                  avg_logprob=-0.2, no_speech_prob=0.03)
        return iter([sample]), SimpleNamespace(language=kwargs.get("language") or "en",
                                                 language_probability=0.98, duration=1.5)


def make_client():
    made = {}

    def factory(model, compute_type):
        made[(model, compute_type)] = FakeModel()
        return made[(model, compute_type)]

    manager = ModelManager(factory)
    api = create_app(manager)
    return TestClient(api, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)), made


def test_health_model_list_and_load_warmup():
    client, made = make_client()
    assert client.get("/api/health").json() == {"status": "ok", "active_jobs": 0, "loaded_models": 0}
    models = client.get("/api/models").json()
    assert models["default_model"] == "small"
    assert "int8" in models["compute_types"]
    result = client.post("/api/models/load", json={"model": "tiny", "compute_type": "int8"})
    assert result.status_code == 200
    assert result.json()["warmup"] == "complete"
    assert len(made[("tiny", "int8")].calls) == 1


def test_job_transcribes_upload_and_returns_segment_metadata(monkeypatch):
    from backend import app as app_module
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=struct.pack("f", 0.1)))
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
    assert len(audio) == 1
    assert options == {"language": "vi", "beam_size": 3, "vad_filter": True, "initial_prompt": "names"}


def test_invalid_settings_empty_upload_unknown_job_and_remote_origin():
    client, _ = make_client()
    assert client.post("/api/models/load", json={"model": "tiny", "compute_type": "bogus"}).status_code == 422
    assert client.post("/api/models/load", json={"model": "../../etc/passwd", "compute_type": "int8"}).status_code == 422
    assert client.get("/api/jobs/missing").status_code == 404
    empty = client.post("/api/jobs", files={"file": ("empty.wav", b"", "audio/wav")})
    assert empty.status_code == 422
    denied = client.post("/api/models/load", json={"model": "tiny", "compute_type": "int8"},
                         headers={"Origin": "https://attacker.example"})
    assert denied.status_code == 403


def test_remote_peer_is_rejected_even_without_origin():
    from backend.app import create_app
    client = TestClient(create_app(ModelManager(lambda *_: FakeModel())), base_url="http://127.0.0.1",
                         client=("192.0.2.4", 1234))
    assert client.get("/api/health").status_code == 403


def test_failed_job_reports_safe_error_and_removes_temporary_upload(monkeypatch):
    from backend import app as app_module
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=struct.pack("f", 0.1)))
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


def test_status_routes_stay_responsive_during_inference_and_host_is_validated(monkeypatch):
    from backend import app as app_module
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=struct.pack("f", 0.1)))
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
    from backend import app as app_module
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=struct.pack("f", 0.1)))
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
    assert manager.snapshot() == [{"model": "base", "compute_type": "int8"}]
