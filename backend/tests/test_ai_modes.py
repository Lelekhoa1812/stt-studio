"""Optional AI modes: prompt content, key rotation, refusals, persistence, and an opt-in call to NVIDIA NIM."""
import io
import json
import logging
import os
import sqlite3
import urllib.error
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend import nim
from backend.app import ModelManager, create_app
from test_api import PausingModel, _wait_for, mock_audio_tools

KEY_ONE = "nvapi-test-key-one-0001"
KEY_TWO = "nvapi-test-key-two-0002"
REAL_NIM = os.environ.get("STT_STUDIO_REAL_NIM") == "1"


class MeetingModel:
    def transcribe(self, source, **kwargs):
        rows = [SimpleNamespace(start=12.5, end=15.0, text=" We agreed to ship the beta on Friday. ",
                                avg_logprob=-0.1, no_speech_prob=0.0),
                SimpleNamespace(start=61.25, end=66.0, text="Priya will write the release notes.",
                                avg_logprob=-0.1, no_speech_prob=0.0)]
        return iter(rows), SimpleNamespace(language="en", language_probability=1.0, duration=66.0)


class SilentModel:
    def transcribe(self, source, **kwargs):
        return iter([]), SimpleNamespace(language="en", language_probability=1.0, duration=1.0)


class FakeNim:
    """Replaces the HTTP call. Each outcome is reply text, an HTTP status to refuse with, or an exception to raise."""

    def __init__(self, monkeypatch, outcomes):
        self.outcomes = outcomes
        self.calls = []
        monkeypatch.setattr(nim, "send_chat", self.send)

    def send(self, api_key, messages):
        self.calls.append((api_key, messages))
        outcome = self.outcomes[api_key]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            raise nim.NimRequestError(outcome, f"NVIDIA NIM refused the request (HTTP {outcome})")
        return outcome


@pytest.fixture
def keys(monkeypatch):
    """Keys from the environment only. The project's .env is ignored so a developer's file cannot change results."""
    monkeypatch.setattr(nim, "_read_env_file", lambda name: None)
    for number in range(1, 10):
        monkeypatch.delenv(f"NVIDIA_NIM_API_KEY_{number}", raising=False)
    monkeypatch.setenv("NVIDIA_NIM_API_KEY_1", KEY_ONE)
    monkeypatch.setenv("NVIDIA_NIM_API_KEY_2", KEY_TWO)
    return [KEY_ONE, KEY_TWO]


@pytest.fixture
def no_keys(monkeypatch):
    monkeypatch.setattr(nim, "_read_env_file", lambda name: None)
    for number in range(1, 10):
        monkeypatch.delenv(f"NVIDIA_NIM_API_KEY_{number}", raising=False)


def _api(model=None):
    return TestClient(create_app(ModelManager(lambda *_: model or MeetingModel())),
                      base_url="http://127.0.0.1", client=("127.0.0.1", 12345))


def _completed_job(monkeypatch, model=None):
    mock_audio_tools(monkeypatch)
    client = _api(model)
    job_id = client.post("/api/jobs", files={"file": ("board call.wav", io.BytesIO(b"audio bytes"), "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] in {"completed", "failed", "cancelled"})
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "completed"
    return client, job_id


def _assert_no_secrets(body, caplog):
    for secret in (KEY_ONE, KEY_TWO):
        assert secret not in body
        assert secret not in caplog.text
    assert "We agreed" not in caplog.text
    assert "Priya" not in caplog.text


def test_prompts_carry_transcript_text_and_no_timestamps():
    segments = [{"start": 12.5, "end": 15.0, "text": " We agreed to ship the beta on Friday. ", "speaker": "Speaker 1"},
                {"start": 61.25, "end": 66.0, "text": "Priya will write the release notes."}]
    transcript = nim.transcript_text(segments)
    assert transcript == "Speaker 1: We agreed to ship the beta on Friday.\nPriya will write the release notes."
    messages = nim.build_messages("action_items", transcript)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Priya will write the release notes." in messages[1]["content"]
    prompt = messages[0]["content"] + messages[1]["content"]
    for stamp in ("12.5", "15.0", "61.25", "66.0", "00:"):
        assert stamp not in prompt
    assert "Use only the transcript" in nim.SYSTEM_PROMPTS["summary"]
    with pytest.raises(ValueError):
        nim.build_messages("translate", transcript)


def test_summary_sends_only_transcript_text_and_stores_the_result(monkeypatch, keys, caplog):
    fake = FakeNim(monkeypatch, {KEY_ONE: "Overview.\n- The beta ships on Friday."})
    client, job_id = _completed_job(monkeypatch)
    with caplog.at_level(logging.DEBUG):
        response = client.post(f"/api/jobs/{job_id}/ai", json={"mode": "summary"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "summary" and body["text"] == "Overview.\n- The beta ships on Friday."
    assert body["model"] == "nvidia/nemotron-3-ultra-550b-a55b" and isinstance(body["created_at"], float)
    key_used, messages = fake.calls[0]
    assert key_used == KEY_ONE
    user_message = messages[-1]["content"]
    assert "We agreed to ship the beta on Friday." in user_message
    assert "Priya will write the release notes." in user_message
    assert "12.5" not in user_message and "61.25" not in user_message and "00:" not in user_message
    assert client.get(f"/api/jobs/{job_id}").json()["ai_results"] == [body]
    _assert_no_secrets(response.text, caplog)


def test_stored_result_survives_a_restart_and_a_repeat_replaces_the_same_mode(monkeypatch, keys):
    fake = FakeNim(monkeypatch, {KEY_ONE: "First summary."})
    client, job_id = _completed_job(monkeypatch)
    assert client.post(f"/api/jobs/{job_id}/ai", json={"mode": "summary"}).status_code == 200
    fake.outcomes[KEY_ONE] = "Action: Priya writes the release notes."
    assert client.post(f"/api/jobs/{job_id}/ai", json={"mode": "action_items"}).status_code == 200
    fake.outcomes[KEY_ONE] = "Second summary."
    assert client.post(f"/api/jobs/{job_id}/ai", json={"mode": "summary"}).status_code == 200
    del client

    restarted = _api()
    results = restarted.get(f"/api/jobs/{job_id}").json()["ai_results"]
    assert sorted((item["mode"], item["text"]) for item in results) == [
        ("action_items", "Action: Priya writes the release notes."),
        ("summary", "Second summary."),
    ]
    listed = restarted.get("/api/jobs").json()["jobs"][0]
    assert "ai_results" not in listed and "segments" not in listed


def test_running_or_unknown_jobs_are_refused_before_any_call(monkeypatch, keys):
    mock_audio_tools(monkeypatch, duration=3.0)
    fake = FakeNim(monkeypatch, {})
    model = PausingModel()
    client = _api(model)
    job_id = client.post("/api/jobs", files={"file": ("clip.wav", io.BytesIO(b"audio"), "audio/wav")},
                         data={"model": "tiny", "compute_type": "int8"}).json()["id"]
    assert _wait_for(lambda: model.paused.is_set())
    refused = client.post(f"/api/jobs/{job_id}/ai", json={"mode": "summary"})
    assert refused.status_code == 409 and "running" in refused.json()["detail"]
    assert client.post("/api/jobs/missing/ai", json={"mode": "summary"}).status_code == 404
    assert client.post(f"/api/jobs/{job_id}/ai", json={"mode": "translate"}).status_code == 422
    model.resume.set()
    assert _wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["status"] == "completed")
    assert fake.calls == []


def test_empty_and_oversized_transcripts_are_refused_with_a_reason(monkeypatch, keys):
    fake = FakeNim(monkeypatch, {KEY_ONE: "unused"})
    client, job_id = _completed_job(monkeypatch, SilentModel())
    empty = client.post(f"/api/jobs/{job_id}/ai", json={"mode": "summary"})
    assert empty.status_code == 422 and "empty" in empty.json()["detail"]

    client, job_id = _completed_job(monkeypatch)
    assert client.post(f"/api/jobs/{job_id}/ai", json={"mode": "summary", "segments": []}).status_code == 422
    too_long = client.post(f"/api/jobs/{job_id}/ai",
                           json={"mode": "summary", "segments": [{"text": "x" * 150_001}]})
    assert too_long.status_code == 422 and "150,000" in too_long.json()["detail"]
    at_limit = client.post(f"/api/jobs/{job_id}/ai", json={"mode": "summary", "segments": [{"text": "x" * 150_000}]})
    assert at_limit.status_code == 200
    assert len(fake.calls) == 1


def test_no_key_is_a_503_with_the_reason_and_capabilities_say_so(monkeypatch, no_keys):
    fake = FakeNim(monkeypatch, {})
    client, job_id = _completed_job(monkeypatch)
    response = client.post(f"/api/jobs/{job_id}/ai", json={"mode": "action_items"})
    assert response.status_code == 503
    assert "NVIDIA_NIM_API_KEY_1" in response.json()["detail"]
    assert client.get("/api/capabilities").json()["ai"] == {
        "available": False, "model": nim.MODEL_ID, "modes": ["summary", "action_items"], "reason": nim.NO_KEY_REASON}
    assert fake.calls == []


def test_capabilities_report_ai_available_when_a_key_is_set(monkeypatch, keys):
    body = _api().get("/api/capabilities").json()
    assert body["ai"] == {"available": True, "model": nim.MODEL_ID, "modes": ["summary", "action_items"], "reason": None}
    assert KEY_ONE not in json.dumps(body) and KEY_TWO not in json.dumps(body)


@pytest.mark.parametrize("refusal", [401, 403, 429, 500, 503])
def test_a_refused_key_rotates_to_the_next_and_no_key_reaches_a_response_or_log(monkeypatch, keys, caplog, refusal):
    fake = FakeNim(monkeypatch, {KEY_ONE: refusal, KEY_TWO: "Action items: none were stated."})
    client, job_id = _completed_job(monkeypatch)
    with caplog.at_level(logging.DEBUG):
        response = client.post(f"/api/jobs/{job_id}/ai", json={"mode": "action_items"})
    assert response.status_code == 200
    assert [key for key, _ in fake.calls] == [KEY_ONE, KEY_TWO]
    assert response.json()["text"] == "Action items: none were stated."
    assert "NVIDIA_NIM_API_KEY_1" in caplog.text
    _assert_no_secrets(response.text, caplog)


def test_a_refusal_another_key_cannot_fix_is_not_retried(monkeypatch, keys, caplog):
    fake = FakeNim(monkeypatch, {KEY_ONE: 400, KEY_TWO: "unused"})
    client, job_id = _completed_job(monkeypatch)
    response = client.post(f"/api/jobs/{job_id}/ai", json={"mode": "summary"})
    assert response.status_code == 502
    assert "HTTP 400" in response.json()["detail"]
    assert len(fake.calls) == 1
    _assert_no_secrets(response.text, caplog)


def test_unreachable_network_is_not_retried_with_other_keys(monkeypatch, keys, caplog):
    fake = FakeNim(monkeypatch, {KEY_ONE: nim.NimRequestError(None, "Could not reach NVIDIA NIM. Check the connection and try again."),
                                 KEY_TWO: "unused"})
    client, job_id = _completed_job(monkeypatch)
    response = client.post(f"/api/jobs/{job_id}/ai", json={"mode": "summary"})
    assert response.status_code == 502 and "Could not reach NVIDIA NIM" in response.json()["detail"]
    assert len(fake.calls) == 1
    _assert_no_secrets(response.text, caplog)


def test_every_key_refused_returns_an_error_that_names_no_key(monkeypatch, keys, caplog):
    fake = FakeNim(monkeypatch, {KEY_ONE: 429, KEY_TWO: 429})
    client, job_id = _completed_job(monkeypatch)
    with caplog.at_level(logging.DEBUG):
        response = client.post(f"/api/jobs/{job_id}/ai", json={"mode": "summary"})
    assert response.status_code == 502
    assert "did not accept any configured key (last HTTP 429)" in response.json()["detail"]
    assert len(fake.calls) == 2
    _assert_no_secrets(response.text, caplog)


def test_send_chat_uses_a_bearer_header_and_maps_failures_without_the_key_or_url(monkeypatch):
    seen = {}

    class Reply:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size=-1):
            return self.body

    def answer(request, timeout):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return Reply(json.dumps({"choices": [{"message": {"content": " Overview. "}}]}).encode())

    monkeypatch.setattr(nim.urllib.request, "urlopen", answer)
    messages = nim.build_messages("summary", "Priya: hello")
    assert nim.send_chat(KEY_ONE, messages) == "Overview."
    assert seen["url"] == nim.ENDPOINT == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert seen["auth"] == f"Bearer {KEY_ONE}"
    assert seen["payload"]["model"] == nim.MODEL_ID and seen["payload"]["stream"] is False
    assert seen["payload"]["messages"] == messages and seen["timeout"] == 120

    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(nim.urllib.request, "urlopen", refuse)
    with pytest.raises(nim.NimRequestError) as refused:
        nim.send_chat(KEY_ONE, messages)
    assert refused.value.status == 429
    assert KEY_ONE not in str(refused.value) and nim.ENDPOINT not in str(refused.value)

    def unreachable(request, timeout):
        raise urllib.error.URLError(f"certificate verify failed for {request.full_url}")

    monkeypatch.setattr(nim.urllib.request, "urlopen", unreachable)
    with pytest.raises(nim.NimRequestError) as offline:
        nim.send_chat(KEY_ONE, messages)
    assert offline.value.status is None
    assert nim.ENDPOINT not in str(offline.value) and "certificate" not in str(offline.value)


def test_a_version_one_database_gains_the_ai_results_column_on_open(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(data_dir / "stt-studio.sqlite")
    db.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for statement in app_module.SCHEMA_STATEMENTS:
        db.execute(statement)
    db.execute("INSERT INTO schema_version (version) VALUES (1)")
    segments = json.dumps([{"start": 0.0, "end": 1.0, "text": "hello", "avg_logprob": None, "no_speech_prob": None}])
    db.execute("INSERT INTO jobs (id, status, created_at, model, runtime, compute_type, language, beam_size, vad_filter, "
               "segments, info, error, progress, cancel_requested, filename, batch_id, diarize) "
               "VALUES (?, 'completed', 1.0, 'tiny', 'faster-whisper', 'int8', NULL, 5, 1, ?, '{}', NULL, 100.0, 0, "
               "'old.wav', NULL, 0)", ("old-job", segments))
    db.commit()
    db.close()

    client = _api()
    job = client.get("/api/jobs/old-job").json()
    assert job["segments"][0]["text"] == "hello" and job["ai_results"] == []
    db = sqlite3.connect(data_dir / "stt-studio.sqlite")
    try:
        assert db.execute("SELECT version FROM schema_version").fetchall() == [(2,)]
    finally:
        db.close()


@pytest.mark.skipif(not REAL_NIM, reason="Set STT_STUDIO_REAL_NIM=1 to call NVIDIA NIM with a synthetic transcript")
def test_real_nim_call_returns_a_summary_for_a_synthetic_meeting():
    available = nim.configured_keys()
    if not available:
        pytest.fail("No NVIDIA_NIM_API_KEY_1 to _9 in the environment or .env")
    transcript = "\n".join([
        "Maya: The harbour project review is on Thursday, and we need the budget figures first.",
        "Tom: I can send the vendor quotes by Tuesday.",
        "Maya: Good. Let's keep the scope to the north pier for now.",
    ])
    text = nim.generate("summary", transcript, available)
    assert text.strip()
    leaked = any(key in text for _, key in available)
    assert not leaked
