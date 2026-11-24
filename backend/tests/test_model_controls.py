import io
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.app import FasterWhisperAdapter, MlxAudioAdapter, ModelManager, create_app

MLX_MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"


class FakeWhisper:
    def transcribe(self, source, **kwargs):
        sample = SimpleNamespace(start=0.0, end=1.0, text=" hi ", avg_logprob=-0.1, no_speech_prob=0.0)
        return iter([sample]), SimpleNamespace(language="en", language_probability=1.0, duration=1.0)


class FakeMlx:
    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(support_languages=[])

    def generate(self, audio, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="hello from mlx", segments=None, sentences=None)


def _media_tools(monkeypatch, duration=1.0):
    """Stand-in for ffprobe and ffmpeg: a one-second mono clip, no real media touched."""
    body = {"format": {"duration": str(duration), "format_name": "wav"},
            "streams": [{"codec_type": "audio", "duration": str(duration)}]}
    payload = json.dumps(body).encode()

    def run(command, **kwargs):
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout=payload, stderr=b"", returncode=0)
        Path(command[-1]).write_bytes(b"\0" * (int(duration * 16000) * 4))
        return SimpleNamespace(stdout=b"", stderr=b"")

    class _Probe:
        def __init__(self, command, **kwargs):
            self.stdout = io.BytesIO(payload if command and command[0] == "ffprobe" else b"")
            self.stderr = io.BytesIO(b"")
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(app_module.subprocess, "run", run)
    monkeypatch.setattr(app_module.subprocess, "Popen", _Probe)


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _mlx_client(model=None):
    fake = model or FakeMlx()
    adapter = MlxAudioAdapter(snapshot=lambda repo: f"/cache/{repo}", loader=lambda path: fake,
                              to_array=lambda audio: ("array", len(audio)))
    client = TestClient(create_app(ModelManager(adapters={"mlx-audio": adapter})),
                        base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    return client, fake


def _whisper_client():
    adapters = {"faster-whisper": FasterWhisperAdapter(lambda *_: FakeWhisper())}
    return TestClient(create_app(ModelManager(adapters=adapters)),
                      base_url="http://127.0.0.1", client=("127.0.0.1", 12345))


def _post_job(client, **form):
    data = {"model": MLX_MODEL, **form}
    return client.post("/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")}, data=data)


def _finished(client, job_id):
    assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] in {"completed", "failed"})
    return client.get(f"/api/jobs/{job_id}").json()


def _stored_row(data_dir: Path, job_id: str):
    db = sqlite3.connect(data_dir / "stt-studio.sqlite")
    try:
        return db.execute("SELECT beam_size, vad_filter FROM jobs WHERE id = ?", (job_id,)).fetchone()
    finally:
        db.close()


def test_mlx_job_with_initial_prompt_is_accepted_and_reports_it_as_ignored(monkeypatch, data_dir):
    _media_tools(monkeypatch)
    client, fake = _mlx_client()
    response = _post_job(client, initial_prompt="Acme, Dropbox", vad_filter="false")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["ignored_controls"] == ["initial_prompt"]

    job = _finished(client, body["id"])
    assert job["status"] == "completed"
    assert job["ignored_controls"] == ["initial_prompt"]
    assert job["segments"][0]["text"] == "hello from mlx"
    # The prompt never reaches the runtime.
    assert fake.calls == [{}]
    # Ignored controls are stored as NULL on the job row.
    assert _stored_row(data_dir, body["id"]) == (None, None)


def test_mlx_job_lists_each_ignored_whisper_control_by_name(monkeypatch, data_dir):
    _media_tools(monkeypatch)
    client, _ = _mlx_client()
    body = _post_job(client, beam_size="3", vad_filter="true", initial_prompt="names").json()
    assert body["ignored_controls"] == ["beam_size", "vad_filter", "initial_prompt"]


def test_mlx_default_beam_size_and_empty_prompt_are_not_reported(monkeypatch, data_dir):
    _media_tools(monkeypatch)
    client, _ = _mlx_client()
    body = _post_job(client, beam_size="5", vad_filter="false", initial_prompt="").json()
    assert body["ignored_controls"] == []
    assert _finished(client, body["id"])["ignored_controls"] == []


def test_faster_whisper_job_keeps_every_control_and_reports_none(monkeypatch, data_dir):
    _media_tools(monkeypatch)
    client = _whisper_client()
    response = client.post("/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")},
                           data={"model": "tiny", "compute_type": "int8", "beam_size": "3",
                                 "vad_filter": "true", "initial_prompt": "names"})
    assert response.status_code == 202
    assert response.json()["ignored_controls"] == []
    job = _finished(client, response.json()["id"])
    assert job["ignored_controls"] == [] and job["beam_size"] == 3 and job["vad_filter"] is True
    assert _stored_row(data_dir, job["id"]) == (3, 1)


def test_batch_members_report_ignored_controls_per_file(monkeypatch, data_dir):
    _media_tools(monkeypatch)
    client, _ = _mlx_client()
    response = client.post("/api/batches",
                           files=[("files", ("one.wav", io.BytesIO(b"one"), "audio/wav")),
                                  ("files", ("two.wav", io.BytesIO(b"two"), "audio/wav"))],
                           data={"model": MLX_MODEL, "vad_filter": "false", "initial_prompt": "names"})
    assert response.status_code == 202
    jobs = response.json()["jobs"]
    assert [job["ignored_controls"] for job in jobs] == [["initial_prompt"], ["initial_prompt"]]


def test_ignored_controls_survive_a_server_restart(monkeypatch, data_dir):
    _media_tools(monkeypatch)
    client, _ = _mlx_client()
    job_id = _post_job(client, initial_prompt="names", vad_filter="false").json()["id"]
    assert _finished(client, job_id)["status"] == "completed"

    restarted, _ = _mlx_client()
    job = restarted.get(f"/api/jobs/{job_id}").json()
    assert job["ignored_controls"] == ["initial_prompt"]
    assert job["beam_size"] is None and job["vad_filter"] is None


def test_version_two_database_is_rebuilt_and_keeps_its_jobs(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(data_dir / "stt-studio.sqlite")
    db.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for statement in app_module.SCHEMA_STATEMENTS + app_module.SCHEMA_MIGRATIONS[2]:
        db.execute(statement)
    db.execute("INSERT INTO schema_version (version) VALUES (2)")
    segments = json.dumps([{"start": 0.0, "end": 1.0, "text": "kept", "avg_logprob": None, "no_speech_prob": None}])
    db.execute("INSERT INTO jobs (id, status, created_at, model, runtime, compute_type, language, beam_size, vad_filter, "
               "segments, info, error, progress, cancel_requested, filename, batch_id, diarize, ai_results) "
               "VALUES ('old-job', 'completed', 1.0, 'tiny', 'faster-whisper', 'int8', NULL, 3, 0, ?, '{}', NULL, "
               "100.0, 0, 'old.wav', NULL, 0, '[]')", (segments,))
    db.commit()
    db.close()

    client = _whisper_client()
    job = client.get("/api/jobs/old-job").json()
    assert job["segments"][0]["text"] == "kept"
    assert job["beam_size"] == 3 and job["vad_filter"] is False and job["ignored_controls"] == []

    db = sqlite3.connect(data_dir / "stt-studio.sqlite")
    try:
        assert db.execute("SELECT version FROM schema_version").fetchall() == [(3,)]
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert db.execute("SELECT name FROM sqlite_master WHERE name = 'jobs_created_at'").fetchone() is not None
        # NULL controls must be accepted once the table has been rebuilt.
        db.execute("INSERT INTO jobs (id, status, created_at, model, runtime, compute_type, language, beam_size, "
                   "vad_filter, segments, info, error, progress, cancel_requested, filename, batch_id, diarize, "
                   "ai_results, ignored_controls) VALUES ('null-job', 'completed', 2.0, 'x', 'mlx-audio', 'native', "
                   "NULL, NULL, NULL, '[]', '{}', NULL, 100.0, 0, 'n.wav', NULL, 0, '[]', '[]')")
        db.commit()
    finally:
        db.close()


CONTROL_KEYS = {"language", "vad", "beamSize", "initialPrompt", "wordTimestamps", "diarization", "hotwords",
                "translation"}
EMIT_KEYS = {"segments", "wordTimestamps", "detectedLanguage"}
CATALOG_PATH = Path(__file__).resolve().parents[2] / "models" / "catalog.json"


def _catalog_entries():
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return [(family, model) for family in document["families"] for model in family["models"]]


def test_every_catalog_entry_declares_controls_emits_cancellation_and_size():
    entries = _catalog_entries()
    assert len(entries) == 25
    for _family, model in entries:
        assert set(model["controls"]) == CONTROL_KEYS, model["id"]
        assert all(isinstance(value, bool) for value in model["controls"].values()), model["id"]
        assert set(model["emits"]) == EMIT_KEYS, model["id"]
        assert all(isinstance(value, bool) for value in model["emits"].values()), model["id"]
        assert isinstance(model["cancellation"], bool), model["id"]
        assert model["downloadSize"] is None or isinstance(model["downloadSize"], str), model["id"]


def test_runtime_flags_match_what_each_adapter_applies():
    entries = {model["id"]: model for _family, model in _catalog_entries()}
    whisper = entries["small"]
    assert whisper["controls"]["vad"] and whisper["controls"]["beamSize"] and whisper["controls"]["initialPrompt"]
    assert whisper["controls"]["diarization"] and whisper["cancellation"]
    assert whisper["emits"]["detectedLanguage"] and whisper["emits"]["segments"]
    qwen = entries["qwen3-asr-0.6b-mlx-8bit"]
    assert qwen["controls"]["language"] and not qwen["controls"]["vad"] and not qwen["controls"]["initialPrompt"]
    assert not qwen["cancellation"]
    # Nothing in this catalog asks the runtimes for word timestamps, hotwords or translation.
    assert not any(model["controls"]["wordTimestamps"] or model["controls"]["hotwords"]
                   or model["controls"]["translation"] for _family, model in _catalog_entries())


def test_models_endpoint_exposes_controls_and_emits_as_additive_fields():
    client = TestClient(create_app(ModelManager(adapters={
        "faster-whisper": FasterWhisperAdapter(lambda *_: FakeWhisper())})),
        base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    body = client.get("/api/models").json()
    assert body["models"]
    repository_sizes = {model["repository"]: model["downloadSize"] for _family, model in _catalog_entries()}
    for row in body["models"]:
        assert {"controls", "emits", "cancellation"} <= set(row), row["id"]
        assert set(row["controls"]) == CONTROL_KEYS and set(row["emits"]) == EMIT_KEYS
        # Existing fields keep their names and values.
        assert {"id", "name", "family", "size", "repository", "capabilities", "cached", "runtime"} <= set(row)
    small = next(row for row in body["models"] if row["id"] == "small")
    assert small["controls"]["vad"] is True and small["cancellation"] is True
    assert all(row["size"] == repository_sizes[row["repository"]] for row in body["models"]
               if row["runtime"] == "faster-whisper")


def _load_verify_script():
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "verify_model_catalog.py"
    spec = importlib.util.spec_from_file_location("verify_model_catalog", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_catalog_check_passes_without_any_network_request(monkeypatch):
    verify = _load_verify_script()

    def no_network(*_args, **_kwargs):
        raise AssertionError("the offline check must not touch the network")

    monkeypatch.setattr(verify, "urlopen", no_network)
    assert verify.check_capability_metadata() == []
    assert verify.main(["--offline"]) == 0


def test_offline_catalog_check_reports_missing_and_non_boolean_flags(tmp_path):
    verify = _load_verify_script()
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    first = document["families"][0]["models"][0]
    del first["cancellation"]
    del first["controls"]["vad"]
    first["emits"]["segments"] = "yes"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    issues = verify.check_capability_metadata(path)
    assert any("cancellation must be true or false" in issue for issue in issues)
    assert any("controls.vad is missing" in issue for issue in issues)
    assert any("emits.segments must be true or false" in issue for issue in issues)


def test_verify_script_help_exits_cleanly(capsys):
    verify = _load_verify_script()
    with pytest.raises(SystemExit) as exit_info:
        verify.main(["--help"])
    assert exit_info.value.code == 0
    assert "--offline" in capsys.readouterr().out


def _use_cache(monkeypatch, tmp_path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(app_module, "MODEL_CACHE", cache)
    return cache


def _cached_model(cache: Path, repository: str, blob_bytes: int = 100) -> Path:
    """A complete Hugging Face cache folder: one blob, with a snapshot symlink to it."""
    folder = cache / f"models--{repository.replace('/', '--')}"
    blob = folder / "blobs" / "abc123"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * blob_bytes)
    snapshot = folder / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    os.symlink(blob, snapshot / "model.bin")
    return folder


def _partial_model(cache: Path, repository: str, blob_bytes: int = 50) -> Path:
    folder = cache / f"models--{repository.replace('/', '--')}"
    (folder / "blobs").mkdir(parents=True)
    (folder / "blobs" / "partial.incomplete").write_bytes(b"x" * blob_bytes)
    return folder


class PausingWhisper:
    """Yields one segment, then waits until the test releases it, so a job stays running."""

    def __init__(self):
        self.paused = threading.Event()
        self.resume = threading.Event()

    def transcribe(self, source, **kwargs):
        def segments():
            yield SimpleNamespace(start=0.0, end=1.0, text="part", avg_logprob=-0.1, no_speech_prob=0.0)
            self.paused.set()
            self.resume.wait(timeout=5)
            yield SimpleNamespace(start=1.0, end=2.0, text="more", avg_logprob=-0.1, no_speech_prob=0.0)
        return segments(), SimpleNamespace(language="en", language_probability=1.0, duration=2.0)


def _pausing_client(model):
    return TestClient(create_app(ModelManager(adapters={
        "faster-whisper": FasterWhisperAdapter(lambda *_: model)})), base_url="http://127.0.0.1",
        client=("127.0.0.1", 12345))


def test_model_cache_lists_every_catalog_entry_with_cached_flag_and_bytes(monkeypatch, tmp_path):
    cache = _use_cache(monkeypatch, tmp_path)
    _cached_model(cache, "Systran/faster-whisper-small", blob_bytes=100)
    _partial_model(cache, "Systran/faster-whisper-tiny", blob_bytes=50)
    client = _whisper_client()
    body = client.get("/api/models/cache").json()
    ids = [entry["id"] for entry in body["models"]]
    assert ids == [model["id"] for model in app_module.MODEL_CATALOG]
    assert len(ids) == len(set(ids)) == 25
    entries = {entry["id"]: entry for entry in body["models"]}
    assert entries["small"] == {"id": "small", "cached": True, "sizeBytes": 100}
    # A partial download is not cached, but its bytes are still on disk.
    assert entries["tiny"] == {"id": "tiny", "cached": False, "sizeBytes": 50}
    assert entries["medium"] == {"id": "medium", "cached": False, "sizeBytes": None}


def test_delete_removes_only_the_requested_model_folder(monkeypatch, tmp_path):
    cache = _use_cache(monkeypatch, tmp_path)
    small = _cached_model(cache, "Systran/faster-whisper-small", blob_bytes=100)
    tiny = _cached_model(cache, "Systran/faster-whisper-tiny", blob_bytes=40)
    client = _whisper_client()
    response = client.delete("/api/models/cache/small")
    assert response.status_code == 200
    assert response.json() == {"id": "small", "deleted": True, "freedBytes": 100}
    assert not small.exists()
    assert tiny.is_dir() and (tiny / "blobs" / "abc123").read_bytes() == b"x" * 40
    again = client.delete("/api/models/cache/small").json()
    assert again == {"id": "small", "deleted": False, "freedBytes": 0}


def test_delete_handles_repositories_whose_ids_contain_a_slash(monkeypatch, tmp_path):
    cache = _use_cache(monkeypatch, tmp_path)
    folder = _cached_model(cache, "mlx-community/Qwen3-ASR-0.6B-8bit", blob_bytes=10)
    client = _whisper_client()
    response = client.delete("/api/models/cache/mlx-community/Qwen3-ASR-0.6B-8bit")
    assert response.status_code == 200 and response.json()["deleted"] is True
    assert response.json()["freedBytes"] == 10
    assert not folder.exists()


def test_delete_unknown_model_is_404_and_touches_nothing(monkeypatch, tmp_path):
    cache = _use_cache(monkeypatch, tmp_path)
    small = _cached_model(cache, "Systran/faster-whisper-small")
    client = _whisper_client()
    assert client.delete("/api/models/cache/not-in-the-catalog").status_code == 404
    assert small.is_dir()


def test_delete_unloads_the_model_it_removes(monkeypatch, tmp_path):
    cache = _use_cache(monkeypatch, tmp_path)
    small = _cached_model(cache, "Systran/faster-whisper-small")
    client = _whisper_client()
    assert client.post("/api/models/load", json={"model": "small", "compute_type": "int8"}).status_code == 200
    assert client.get("/api/health").json()["loaded_models"] == 1
    assert client.delete("/api/models/cache/small").json()["deleted"] is True
    assert client.get("/api/health").json()["loaded_models"] == 0
    assert not small.exists()


def test_evict_drops_only_the_named_model():
    manager = ModelManager(adapters={})
    manager._models[("small", "int8", "faster-whisper")] = object()
    manager._models[("tiny", "int8", "faster-whisper")] = object()
    with manager.exclusive():
        released = manager.evict("small")
    assert released == [{"model": "small", "compute_type": "int8", "runtime": "faster-whisper"}]
    assert manager.snapshot() == [{"model": "tiny", "compute_type": "int8", "runtime": "faster-whisper"}]


def test_delete_is_refused_while_the_model_is_loaded_and_a_job_runs(monkeypatch, tmp_path):
    _media_tools(monkeypatch, duration=2.0)
    cache = _use_cache(monkeypatch, tmp_path)
    tiny = _cached_model(cache, "Systran/faster-whisper-tiny")
    model = PausingWhisper()
    client = _pausing_client(model)
    job_id = client.post("/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: model.paused.is_set())
    refused = client.delete("/api/models/cache/tiny")
    assert refused.status_code == 409
    assert tiny.is_dir()
    model.resume.set()
    assert _finished(client, job_id)["status"] == "completed"
    assert client.delete("/api/models/cache/tiny").json()["deleted"] is True


def test_delete_of_another_model_is_allowed_while_a_job_runs(monkeypatch, tmp_path):
    _media_tools(monkeypatch, duration=2.0)
    cache = _use_cache(monkeypatch, tmp_path)
    tiny = _cached_model(cache, "Systran/faster-whisper-tiny", blob_bytes=40)
    model = PausingWhisper()
    client = _pausing_client(model)
    job_id = client.post("/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")},
                         data={"model": "small", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: model.paused.is_set())
    response = client.delete("/api/models/cache/tiny")
    assert response.status_code == 200
    assert response.json() == {"id": "tiny", "deleted": True, "freedBytes": 40}
    assert not tiny.exists()
    model.resume.set()
    assert _finished(client, job_id)["status"] == "completed"


def test_a_queued_job_for_the_model_blocks_its_deletion(monkeypatch, tmp_path):
    cache = _use_cache(monkeypatch, tmp_path)
    tiny = _cached_model(cache, "Systran/faster-whisper-tiny")
    client = _whisper_client()
    with client.app.state.jobs_lock:
        client.app.state.jobs["queued-job"] = app_module.Job(id="queued-job", model="tiny", status="queued")
    assert client.delete("/api/models/cache/tiny").status_code == 409
    assert tiny.is_dir()


def test_delete_returns_409_while_a_load_holds_the_model_manager(monkeypatch, tmp_path):
    cache = _use_cache(monkeypatch, tmp_path)
    small = _cached_model(cache, "Systran/faster-whisper-small")
    api = create_app(ModelManager(adapters={"faster-whisper": FasterWhisperAdapter(lambda *_: FakeWhisper())}))
    client = TestClient(api, base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    operation_lock = api.state.models._operation_lock
    operation_lock.acquire()
    try:
        response = client.delete("/api/models/cache/small")
    finally:
        operation_lock.release()
    assert response.status_code == 409
    assert small.is_dir()


def test_a_symlinked_cache_folder_that_points_outside_the_cache_is_refused(monkeypatch, tmp_path):
    cache = _use_cache(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.bin").write_bytes(b"not a model")
    os.symlink(outside, cache / "models--Systran--faster-whisper-small")
    client = _whisper_client()
    response = client.delete("/api/models/cache/small")
    assert response.status_code == 422
    assert (outside / "keep.bin").read_bytes() == b"not a model"
    entry = next(e for e in client.get("/api/models/cache").json()["models"] if e["id"] == "small")
    assert entry == {"id": "small", "cached": False, "sizeBytes": None}
    with pytest.raises(ValueError):
        app_module._cache_folder("Systran/faster-whisper-small")


def test_batch_list_is_empty_on_a_fresh_store(data_dir):
    client = _whisper_client()
    assert client.get("/api/batches").json() == {"batches": [], "total": 0, "limit": 50, "offset": 0}


def test_batch_list_is_newest_first_and_paginated(monkeypatch, data_dir):
    _media_tools(monkeypatch)
    client = _whisper_client()
    ids = []
    for name in ("first", "second", "third"):
        response = client.post("/api/batches", files=[("files", (f"{name}.wav", io.BytesIO(b"x"), "audio/wav"))],
                               data={"model": "tiny", "compute_type": "int8"})
        assert response.status_code == 202
        batch_id = response.json()["id"]
        assert _wait_for(lambda bid=batch_id: client.get(f"/api/batches/{bid}").json()["done"])
        ids.append(batch_id)
    newest_first = list(reversed(ids))

    page = client.get("/api/batches").json()
    assert page["total"] == 3 and page["limit"] == 50 and page["offset"] == 0
    assert [batch["id"] for batch in page["batches"]] == newest_first

    page = client.get("/api/batches", params={"limit": 2}).json()
    assert page["total"] == 3 and page["limit"] == 2 and page["offset"] == 0
    assert [batch["id"] for batch in page["batches"]] == newest_first[:2]

    page = client.get("/api/batches", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 3 and page["offset"] == 2
    assert [batch["id"] for batch in page["batches"]] == newest_first[2:]

    assert client.get("/api/batches", params={"offset": 10}).json()["batches"] == []
    assert client.get("/api/batches", params={"limit": 0}).status_code == 422
    assert client.get("/api/batches", params={"limit": 201}).status_code == 422


def test_batch_list_entries_carry_counts_and_files_without_transcripts(monkeypatch, data_dir):
    _media_tools(monkeypatch)
    client = _whisper_client()
    batch_id = client.post("/api/batches", files=[("files", ("one.wav", io.BytesIO(b"x"), "audio/wav"))],
                           data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: client.get(f"/api/batches/{batch_id}").json()["done"])

    listed = client.get("/api/batches").json()["batches"][0]
    assert {"id", "created_at", "total", "counts", "done", "jobs"} <= set(listed)
    assert listed["id"] == batch_id and listed["total"] == 1 and listed["counts"]["completed"] == 1
    assert "segments" not in listed["jobs"][0] and "ai_results" not in listed["jobs"][0]
    # The per-batch route keeps the transcript.
    assert "segments" in client.get(f"/api/batches/{batch_id}").json()["jobs"][0]
