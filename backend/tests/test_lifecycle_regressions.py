"""Regression tests for model lifecycle, job state and startup behaviour."""
import pytest
from fastapi.testclient import TestClient

from backend.app import ModelManager, create_app


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
