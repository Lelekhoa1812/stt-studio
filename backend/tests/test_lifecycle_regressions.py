"""Regression tests for model lifecycle, job state and startup behaviour."""
import io

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.app import ModelManager, create_app
from test_api import FakeModel, _wait_for, mock_audio_tools


class WarmupFailsAdapter:
    runtime_id = "fake-runtime"

    def load(self, model, compute_type):
        return {"model": model, "compute_type": compute_type}

    def warmup(self, _instance):
        raise RuntimeError("warmup exploded")

    def transcribe(self, _instance, _audio, **_options):
        return [], {"language": "en", "duration": 0.0}


def test_failed_warmup_leaves_no_resident_model_and_reports_nothing_loaded():
    manager = ModelManager(adapters={"fake-runtime": WarmupFailsAdapter()})
    with pytest.raises(RuntimeError, match="warmup exploded"):
        manager.load("tiny", "int8", "fake-runtime")
    assert manager.snapshot() == []

    client = TestClient(create_app(manager), base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    assert client.get("/api/health").json()["loaded_models"] == 0
    failed = client.post("/api/models/load", json={"model": "vendor/custom", "runtime": "fake-runtime"})
    assert failed.status_code == 503
    assert client.get("/api/health").json()["loaded_models"] == 0


# --- Final save failures -------------------------------------------------------------------------

def _upload_and_wait(client, timeout=5.0):
    job_id = client.post("/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"]
                     in {"completed", "failed", "cancelled"}, timeout=timeout)
    return job_id, client.get(f"/api/jobs/{job_id}").json()


def _client():
    return TestClient(create_app(ModelManager(lambda *_: FakeModel())), base_url="http://127.0.0.1",
                      client=("127.0.0.1", 12345))


def test_final_save_is_retried_once_and_completed_job_is_durable(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    real_save = app_module.JobStore.save_job
    failures = {"left": 1}

    def flaky_save(self, job):
        if job.status == "completed" and failures["left"] > 0:
            failures["left"] -= 1
            raise app_module.StorageError(app_module.STORAGE_UNAVAILABLE)
        return real_save(self, job)

    monkeypatch.setattr(app_module.JobStore, "save_job", flaky_save)
    client = _client()
    job_id, result = _upload_and_wait(client)
    assert result["status"] == "completed"
    assert failures["left"] == 0
    assert client.app.state.store.load_job(job_id).status == "completed"


def test_job_is_failed_not_completed_when_the_final_save_keeps_failing(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    real_save = app_module.JobStore.save_job

    def broken_final_save(self, job):
        if job.status == "completed":
            raise app_module.StorageError(app_module.STORAGE_UNAVAILABLE)
        return real_save(self, job)

    monkeypatch.setattr(app_module.JobStore, "save_job", broken_final_save)
    client = _client()
    job_id, result = _upload_and_wait(client)
    assert result["status"] == "failed"
    assert result["error"] == "Transcript could not be saved"
    durable = client.app.state.store.load_job(job_id)
    assert durable.status == "failed" and durable.error == "Transcript could not be saved"
