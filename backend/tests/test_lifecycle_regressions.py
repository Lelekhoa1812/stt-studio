"""Regression tests for model lifecycle, job state and startup behaviour."""
import io
import threading
import time
import logging
import sys
import types

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.app import ModelManager, create_app
from test_api import FakeModel, PausingModel, _wait_for, mock_audio_tools


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


# --- Failure logging -------------------------------------------------------------------------------


def test_transcription_failure_is_logged_with_traceback(monkeypatch, caplog, data_dir):
    mock_audio_tools(monkeypatch)

    class BrokenModel:
        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError("decoder exploded")

    client = TestClient(create_app(ModelManager(lambda *_: BrokenModel())), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    with caplog.at_level(logging.DEBUG, logger="backend.app"):
        job_id, result = _upload_and_wait(client)
    assert result["status"] == "failed"
    failures = [record for record in caplog.records if record.levelno == logging.ERROR
                and "Transcription failed" in record.getMessage()]
    assert failures and failures[0].exc_info is not None
    assert failures[0].exc_info[0] is RuntimeError


def test_speaker_label_failure_is_logged_with_traceback_and_transcript_kept(monkeypatch, caplog, tmp_path):
    from backend import diarization

    def broken_diarizer(*_args, **_kwargs):
        raise RuntimeError("pyannote crashed")

    monkeypatch.setattr(diarization, "diarize_file", broken_diarizer)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    job = app_module.Job(id="job-1", segments=[{"start": 0.0, "end": 1.0, "text": "hi"}])
    with caplog.at_level(logging.DEBUG, logger="backend.app"):
        report = app_module._run_diarization(job, audio, 1.0)
    assert report["status"] == "failed"
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert errors and errors[0].exc_info is not None and errors[0].exc_info[0] is RuntimeError


def test_accelerator_cache_failure_is_logged_at_debug_not_raised(monkeypatch, caplog):
    core = types.ModuleType("mlx.core")

    def clear_cache():
        raise RuntimeError("metal busy")

    core.clear_cache = clear_cache
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    with caplog.at_level(logging.DEBUG, logger="backend.app"):
        app_module._release_accelerator_cache()
    assert any(record.levelno == logging.DEBUG and "MLX" in record.getMessage() for record in caplog.records)


# --- Atomic job state ---------------------------------------------------------------------------------


def test_failed_cancel_save_keeps_queued_job_consistent_in_memory_and_database(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch, duration=3.0)
    model = PausingModel()
    client = TestClient(create_app(ModelManager(lambda *_: model)), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    batch_id = client.post("/api/batches",
                           files=[("files", ("one.wav", io.BytesIO(b"one"), "audio/wav")),
                                  ("files", ("two.wav", io.BytesIO(b"two"), "audio/wav"))],
                           data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: model.paused.is_set())
    queued_id = client.get(f"/api/batches/{batch_id}").json()["jobs"][1]["id"]

    real_save = app_module.JobStore.save_job
    broken = {"on": True}

    def failing_cancel_save(self, job):
        if broken["on"] and job.status == "cancelled":
            raise app_module.StorageError(app_module.STORAGE_UNAVAILABLE)
        return real_save(self, job)

    monkeypatch.setattr(app_module.JobStore, "save_job", failing_cancel_save)
    refused = client.post(f"/api/jobs/{queued_id}/cancel")
    assert refused.status_code == 503
    assert client.get(f"/api/jobs/{queued_id}").json()["status"] == "queued"
    assert client.app.state.store.load_job(queued_id).status == "queued"

    broken["on"] = False
    assert client.post(f"/api/jobs/{queued_id}/cancel").status_code == 200
    assert client.get(f"/api/jobs/{queued_id}").json()["status"] == "cancelled"
    assert client.app.state.store.load_job(queued_id).status == "cancelled"
    model.resume.set()
    assert _wait_for(lambda: client.get(f"/api/batches/{batch_id}").json()["done"])


def test_running_job_never_persists_full_progress_before_its_final_state(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    saved = []
    real_save = app_module.JobStore.save_job

    def recording_save(self, job):
        saved.append((job.status, job.progress))
        return real_save(self, job)

    monkeypatch.setattr(app_module.JobStore, "save_job", recording_save)
    client = _client()
    _, result = _upload_and_wait(client)
    assert result["status"] == "completed"
    assert all(not (status == "running" and progress == 100.0) for status, progress in saved)
    assert saved[-1] == ("completed", 100.0)


# --- Upload admission ----------------------------------------------------------------------------------


def test_second_job_is_refused_before_its_upload_is_stored(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch, duration=3.0)
    model = PausingModel()
    client = TestClient(create_app(ModelManager(lambda *_: model)), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    first_id = client.post("/api/jobs", files={"file": ("one.wav", io.BytesIO(b"one"), "audio/wav")},
                           data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: model.paused.is_set())

    stored = []
    real_store = app_module._store_upload

    async def recording_store(file):
        stored.append(file.filename)
        return await real_store(file)

    monkeypatch.setattr(app_module, "_store_upload", recording_store)
    refused = client.post("/api/jobs", files={"file": ("two.wav", io.BytesIO(b"two"), "audio/wav")},
                          data={"model": "tiny", "compute_type": "int8"})
    assert refused.status_code == 429
    assert stored == []
    model.resume.set()
    assert _wait_for(lambda: client.get(f"/api/jobs/{first_id}").json()["status"] == "completed")


def test_disk_full_during_upload_is_503_without_paths_and_removes_partial_file(monkeypatch, data_dir):
    import errno
    import os
    import tempfile
    from pathlib import Path

    created = []
    real_mkstemp = tempfile.mkstemp

    def capture_mkstemp(*args, **kwargs):
        result = real_mkstemp(*args, **kwargs)
        created.append(Path(result[1]))
        return result

    class FullDisk(io.BytesIO):
        def write(self, _data):
            raise OSError(errno.ENOSPC, "No space left on device", str(created[-1]))

    def full_disk_fdopen(fd, _mode):
        os.close(fd)
        return FullDisk()

    monkeypatch.setattr(app_module.tempfile, "mkstemp", capture_mkstemp)
    monkeypatch.setattr(app_module.os, "fdopen", full_disk_fdopen)
    client = TestClient(create_app(ModelManager(lambda *_: FakeModel())), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    response = client.post("/api/jobs", files={"file": ("clip.wav", b"audio bytes", "audio/wav")},
                           data={"model": "tiny", "compute_type": "int8"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Not enough disk space"}
    assert created and not any(path.exists() for path in created)
    assert "whisper-upload" not in response.text and str(created[0].parent) not in response.text


# --- Blocking media inspection ------------------------------------------------------------------------


def _slow_probe_result(*_args, **_kwargs):
    time.sleep(1.0)
    return {"container": "wav", "duration_seconds": 1.0, "has_audio": True, "has_video": False, "streams": []}


def _health_latency_during(client, start_request):
    worker = threading.Thread(target=start_request)
    worker.start()
    time.sleep(0.2)  # let the slow inspection begin
    started = time.monotonic()
    assert client.get("/api/health").status_code == 200
    elapsed = time.monotonic() - started
    worker.join()
    return elapsed


def test_slow_probe_does_not_block_health(monkeypatch, data_dir):
    monkeypatch.setattr(app_module, "_ffprobe_media", _slow_probe_result)
    with TestClient(create_app(ModelManager(lambda *_: FakeModel())), base_url="http://127.0.0.1",
                    client=("127.0.0.1", 12345)) as client:
        elapsed = _health_latency_during(client, lambda: client.post(
            "/api/probe", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")}))
    assert elapsed < 0.5


def test_slow_media_check_on_job_upload_does_not_block_health(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    monkeypatch.setattr(app_module, "_ffprobe_media", _slow_probe_result)
    with TestClient(create_app(ModelManager(lambda *_: FakeModel())), base_url="http://127.0.0.1",
                    client=("127.0.0.1", 12345)) as client:
        elapsed = _health_latency_during(client, lambda: client.post(
            "/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")},
            data={"model": "tiny", "compute_type": "int8"}))
    assert elapsed < 0.5


# --- Model switching during a run ---------------------------------------------------------------------


def test_model_load_is_refused_with_409_while_a_transcription_runs(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch, duration=3.0)
    model = PausingModel()
    client = TestClient(create_app(ModelManager(lambda *_: model)), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    job_id = client.post("/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: model.paused.is_set())
    refused = client.post("/api/models/load", json={"model": "base", "compute_type": "int8"})
    assert refused.status_code == 409
    assert "transcription is running" in refused.json()["detail"]
    model.resume.set()
    assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] == "completed")
    assert client.post("/api/models/load", json={"model": "base", "compute_type": "int8"}).status_code == 200


# --- Startup limits -----------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["0", "-3", "two", "1.5", ""])
def test_integer_limits_must_be_whole_numbers_of_at_least_one(monkeypatch, raw):
    monkeypatch.setenv("WHISPER_MAX_ACTIVE_JOBS", raw)
    with pytest.raises(ValueError, match="WHISPER_MAX_ACTIVE_JOBS must be a whole number of at least 1"):
        app_module._env_int("WHISPER_MAX_ACTIVE_JOBS", "1")
    monkeypatch.setenv("WHISPER_MAX_ACTIVE_JOBS", "3")
    assert app_module._env_int("WHISPER_MAX_ACTIVE_JOBS", "1") == 3


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "soon"])
def test_audio_limit_must_be_a_positive_number_of_seconds(monkeypatch, raw):
    monkeypatch.setenv("WHISPER_MAX_AUDIO_SECONDS", raw)
    with pytest.raises(ValueError, match="WHISPER_MAX_AUDIO_SECONDS must be a positive number of seconds"):
        app_module._env_seconds("WHISPER_MAX_AUDIO_SECONDS", "7200")


@pytest.mark.parametrize("raw", ["0", "many"])
def test_bad_limit_stops_the_module_at_import_with_a_clear_message(tmp_path, raw):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "WHISPER_MAX_ACTIVE_JOBS": raw, "WHISPER_DATA_DIR": str(tmp_path / "data")}
    result = subprocess.run([sys.executable, "-c", "import backend.app"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "WHISPER_MAX_ACTIVE_JOBS must be a whole number of at least 1" in result.stderr
