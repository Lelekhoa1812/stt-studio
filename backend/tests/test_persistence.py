"""Durable jobs and transcripts: restarts, interrupted work, deletion, listing and path privacy."""
import io
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend import app as app_module
from backend.app import JobStore, ModelManager, create_app
from test_api import PausingModel, _wait_for, mock_audio_tools


class BulkModel:
    """Returns a long transcript so that deleting it visibly frees space."""

    def transcribe(self, source, **kwargs):
        rows = [SimpleNamespace(start=float(i), end=float(i) + 0.5, text=f"segment {i} " + "lorem ipsum " * 20,
                                avg_logprob=-0.1, no_speech_prob=0.0) for i in range(3000)]
        return iter(rows), SimpleNamespace(language="en", language_probability=1.0, duration=3000.0)


class ScriptedModel:
    def transcribe(self, source, **kwargs):
        rows = [SimpleNamespace(start=0.25, end=1.5, text=" hello ", avg_logprob=-0.2, no_speech_prob=0.03)]
        return iter(rows), SimpleNamespace(language="en", language_probability=0.98, duration=1.5)


def _client(model_factory=lambda *_: ScriptedModel()):
    return TestClient(create_app(ModelManager(model_factory)), base_url="http://127.0.0.1", client=("127.0.0.1", 12345))


def _upload(client, name="talk.wav", data=b"audio bytes"):
    return client.post("/api/jobs", files={"file": (name, io.BytesIO(data), "audio/wav")},
                       data={"model": "tiny", "compute_type": "int8"})


def _batch(client, *names):
    return client.post("/api/batches", files=[("files", (name, io.BytesIO(b"x"), "audio/wav")) for name in names],
                       data={"model": "tiny", "compute_type": "int8"})


def _finish(client, job_id):
    assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] in {"completed", "failed", "cancelled"})
    return client.get(f"/api/jobs/{job_id}").json()


def _batch_done(client, batch_id):
    return _wait_for(lambda: client.get(f"/api/batches/{batch_id}").json()["done"])


def test_completed_job_survives_a_restart_with_identical_segments_and_info(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    first = _client()
    job_id = _upload(first, name="Board call.wav").json()["id"]
    before = _finish(first, job_id)
    del first

    second = _client()
    after = second.get(f"/api/jobs/{job_id}").json()
    assert after == before
    assert after["segments"] == [{"start": 0.25, "end": 1.5, "text": "hello", "avg_logprob": -0.2, "no_speech_prob": 0.03}]
    assert after["info"] == {"language": "en", "language_probability": 0.98, "duration": 1.5, "chunk_count": 1}
    assert second.get(f"/api/jobs/{job_id}/export").text == "hello\n"
    assert second.post(f"/api/jobs/{job_id}/cancel").status_code == 409
    assert second.post("/api/jobs/missing/cancel").status_code == 404


def test_jobs_running_or_queued_at_restart_fail_with_the_restart_message(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch, duration=3.0)
    model = PausingModel()
    first = _client(lambda *_: model)
    batch_id = _batch(first, "one.wav", "two.wav").json()["id"]
    assert _wait_for(lambda: model.paused.is_set())
    abandoned_worker = first.app.state.executor
    del first  # the worker stays blocked, as a killed process would leave it

    with _client() as second:  # entering the client runs the startup recovery, as uvicorn would
        batch = second.get(f"/api/batches/{batch_id}").json()
        assert [job["status"] for job in batch["jobs"]] == ["failed", "failed"]
        assert all(job["error"] == "Interrupted by a server restart" for job in batch["jobs"])
        assert batch["jobs"][0]["segments"] == []
        assert batch["done"] is True
        assert second.get(f"/api/jobs/{batch['jobs'][0]['id']}/export").status_code == 409
    model.resume.set()
    abandoned_worker.shutdown(wait=True)  # drain the abandoned worker so it finishes inside this test


def test_batch_export_is_unchanged_after_a_restart(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    first = _client()
    batch_id = _batch(first, "one.wav", "two.wav").json()["id"]
    assert _batch_done(first, batch_id)
    text_before = first.get(f"/api/batches/{batch_id}/export", params={"format": "txt"}).text
    json_before = first.get(f"/api/batches/{batch_id}/export").json()
    del first

    second = _client()
    assert second.get(f"/api/batches/{batch_id}/export", params={"format": "txt"}).text == text_before
    json_after = second.get(f"/api/batches/{batch_id}/export").json()
    assert json_after["files"] == json_before["files"]
    assert [item["filename"] for item in json_after["files"]] == ["one.wav", "two.wav"]


def test_deleting_a_finished_job_removes_its_row_and_its_bytes(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    client = _client(lambda *_: BulkModel())
    job_id = _upload(client).json()["id"]
    _finish(client, job_id)
    before = client.get("/api/storage").json()
    assert client.delete(f"/api/jobs/{job_id}").json() == {"id": job_id, "deleted": True}
    after = client.get("/api/storage").json()
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.delete(f"/api/jobs/{job_id}").status_code == 404
    assert after["jobs"] == before["jobs"] - 1
    assert after["bytes"] < before["bytes"] // 2


def test_running_and_queued_work_cannot_be_deleted(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch, duration=3.0)
    model = PausingModel()
    client = _client(lambda *_: model)
    batch_id = _batch(client, "one.wav", "two.wav").json()["id"]
    assert _wait_for(lambda: model.paused.is_set())
    running_id, queued_id = [job["id"] for job in client.get(f"/api/batches/{batch_id}").json()["jobs"]]
    refused = client.delete(f"/api/jobs/{running_id}")
    assert refused.status_code == 409 and "running" in refused.json()["detail"]
    assert client.delete(f"/api/jobs/{queued_id}").status_code == 409
    assert client.delete(f"/api/batches/{batch_id}").status_code == 409
    assert client.delete("/api/jobs/missing").status_code == 404
    assert client.delete("/api/batches/missing").status_code == 404
    model.resume.set()
    assert _batch_done(client, batch_id)


def test_deleting_a_batch_removes_its_jobs_and_one_file_can_be_removed_from_it(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    client = _client()
    batch_id = _batch(client, "one.wav", "two.wav").json()["id"]
    assert _batch_done(client, batch_id)
    first_job, second_job = [job["id"] for job in client.get(f"/api/batches/{batch_id}").json()["jobs"]]

    assert client.delete(f"/api/jobs/{first_job}").json() == {"id": first_job, "deleted": True}
    batch = client.get(f"/api/batches/{batch_id}").json()
    assert batch["total"] == 1 and [job["id"] for job in batch["jobs"]] == [second_job]
    export = client.get(f"/api/batches/{batch_id}/export", params={"format": "txt"}).text
    assert "one.wav" not in export and "== two.wav (completed) ==" in export

    deleted = client.delete(f"/api/batches/{batch_id}").json()
    assert deleted == {"id": batch_id, "deleted": True, "jobs_deleted": 1}
    assert client.get(f"/api/batches/{batch_id}").status_code == 404
    assert client.get(f"/api/jobs/{second_job}").status_code == 404
    assert client.get("/api/storage").json()["batches"] == 0
    assert client.get("/api/storage").json()["jobs"] == 0


def test_job_listing_is_paginated_newest_first_and_omits_segments(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    client = _client()
    ids = []
    for name in ("first.wav", "second.wav", "third.wav"):
        job_id = _upload(client, name=name).json()["id"]
        _finish(client, job_id)
        ids.append(job_id)
    page = client.get("/api/jobs", params={"limit": 2}).json()
    assert page["total"] == 3 and page["limit"] == 2 and page["offset"] == 0
    assert [job["id"] for job in page["jobs"]] == [ids[2], ids[1]]
    assert page["jobs"][0]["filename"] == "third.wav" and page["jobs"][0]["status"] == "completed"
    assert all("segments" not in job for job in page["jobs"])
    rest = client.get("/api/jobs", params={"limit": 2, "offset": 2}).json()
    assert [job["id"] for job in rest["jobs"]] == [ids[0]] and rest["total"] == 3
    assert client.get("/api/jobs", params={"limit": 0}).status_code == 422
    assert client.get("/api/jobs", params={"limit": 201}).status_code == 422


def test_storage_and_error_bodies_never_contain_absolute_paths(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    client = _client()
    job_id = _upload(client).json()["id"]
    _finish(client, job_id)

    storage = client.get("/api/storage")
    assert storage.json()["data_dir"] == "persist-store"
    assert set(storage.json()) == {"data_dir", "jobs", "batches", "bytes"}
    assert storage.json()["jobs"] == 1 and storage.json()["bytes"] > 0
    bodies = [storage.text,
              client.get("/api/jobs/missing").text,
              client.delete("/api/jobs/missing").text,
              client.delete("/api/batches/missing").text,
              client.get("/api/batches/missing/export").text,
              client.post(f"/api/jobs/{job_id}/cancel").text,
              client.get(f"/api/jobs/{job_id}/export", params={"format": "pdf"}).text]
    forbidden = {str(data_dir), str(data_dir.resolve()), str(data_dir.parent.resolve())}
    for body in bodies:
        assert not any(path in body for path in forbidden)

    client.app.state.store._conn.close()
    unavailable = client.get("/api/jobs")
    assert unavailable.status_code == 503
    assert "Local job storage is unavailable" in unavailable.json()["detail"]
    assert not any(path in unavailable.text for path in forbidden)


def test_audio_is_never_written_to_the_data_directory(monkeypatch, data_dir):
    mock_audio_tools(monkeypatch)
    client = _client()
    marker = b"AUDIO-MARKER-" * 500
    job_id = client.post("/api/jobs", files={"file": ("secret.wav", io.BytesIO(marker), "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    _finish(client, job_id)
    names = {path.name for path in data_dir.iterdir()}
    assert "stt-studio.sqlite" in names
    assert names <= {"stt-studio.sqlite", "stt-studio.sqlite-wal", "stt-studio.sqlite-shm"}
    assert all(b"AUDIO-MARKER" not in path.read_bytes() for path in data_dir.iterdir())


def test_store_uses_wal_and_records_its_schema_version(data_dir):
    _client()
    db = sqlite3.connect(data_dir / "stt-studio.sqlite")
    try:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("SELECT version FROM schema_version").fetchall() == [(3,)]
    finally:
        db.close()


def test_restart_recovery_runs_once_per_startup_and_never_during_a_run(monkeypatch, data_dir):
    calls = []
    real_mark_interrupted = JobStore.mark_interrupted

    def counting_mark_interrupted(self):
        calls.append(self.data_dir)
        return real_mark_interrupted(self)

    monkeypatch.setattr(JobStore, "mark_interrupted", counting_mark_interrupted)
    mock_audio_tools(monkeypatch, duration=3.0)
    model = PausingModel()
    with _client(lambda *_: model) as client:
        assert len(calls) == 1
        job_id = _upload(client).json()["id"]
        assert _wait_for(lambda: model.paused.is_set())
        for _ in range(5):
            client.get("/api/health")
            client.get("/api/jobs")
            client.get(f"/api/jobs/{job_id}")
        assert len(calls) == 1
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "running"
        model.resume.set()
        assert _finish(client, job_id)["status"] == "completed"

    with _client():
        assert len(calls) == 2


def test_default_data_directory_is_the_repository_data_folder(monkeypatch):
    monkeypatch.delenv("WHISPER_DATA_DIR")
    assert app_module._data_dir() == app_module.DEFAULT_DATA_DIR == app_module.REPO_ROOT / "data"
    assert app_module._data_dir_label(app_module.DEFAULT_DATA_DIR) == "data"


def test_wal_and_shm_files_are_private_like_the_database_after_the_first_save(monkeypatch, data_dir):
    import os
    import stat
    mock_audio_tools(monkeypatch)
    client = _client()
    database = data_dir / "stt-studio.sqlite"
    files = [database, Path(f"{database}-wal"), Path(f"{database}-shm")]
    assert all(path.exists() for path in files)
    for path in files[1:]:
        os.chmod(path, 0o644)  # what a permissive umask leaves behind
    job_id = _upload(client).json()["id"]
    _finish(client, job_id)
    for path in files:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name


def test_failing_reclaim_after_a_delete_is_logged_and_the_job_stays_deleted(monkeypatch, caplog):
    import logging
    mock_audio_tools(monkeypatch)
    client = _client()
    job_id = _upload(client).json()["id"]
    _finish(client, job_id)
    batch_id = _batch(client, "one.wav").json()["id"]
    _batch_done(client, batch_id)

    def broken_reclaim(self):
        raise OSError("disk I/O error")

    monkeypatch.setattr(JobStore, "_reclaim", broken_reclaim)
    with caplog.at_level(logging.ERROR):
        deleted_job = client.delete(f"/api/jobs/{job_id}")
        deleted_batch = client.delete(f"/api/batches/{batch_id}")
    assert deleted_job.status_code == 200 and deleted_job.json()["deleted"] is True
    assert deleted_batch.status_code == 200 and deleted_batch.json()["deleted"] is True
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.get(f"/api/batches/{batch_id}").status_code == 404
    assert "reclaim disk space" in caplog.text
