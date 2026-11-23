import io
import json
import sqlite3
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
    def run(command, **kwargs):
        if command[0] == "ffprobe":
            body = {"format": {"duration": str(duration), "format_name": "wav"},
                    "streams": [{"codec_type": "audio", "duration": str(duration)}]}
            return SimpleNamespace(stdout=json.dumps(body).encode(), stderr=b"")
        Path(command[-1]).write_bytes(b"\0" * (int(duration * 16000) * 4))
        return SimpleNamespace(stdout=b"", stderr=b"")

    monkeypatch.setattr(app_module.subprocess, "run", run)


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
