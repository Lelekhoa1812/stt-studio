"""Speaker labels: attachment, exports, availability, the subprocess boundary and job behaviour."""
import io
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend import diarization, exporters
from backend.app import ModelManager, create_app
from test_api import FakeModel, mock_audio_tools

AVAILABLE = {"available": True, "model": diarization.MODEL_ID, "license": "MIT", "reason": None}


def _client():
    manager = ModelManager(lambda model, compute_type: FakeModel())
    return TestClient(create_app(manager), base_url="http://127.0.0.1", client=("127.0.0.1", 12345))


def _submit(client, **fields):
    data = {"model": "tiny", "compute_type": "int8", **{k: str(v).lower() if isinstance(v, bool) else v
                                                       for k, v in fields.items()}}
    return client.post("/api/jobs", files={"file": ("talk.wav", io.BytesIO(b"audio bytes"), "audio/wav")}, data=data)


def _finished(client, job_id):
    return client.get(f"/api/jobs/{job_id}", params={"wait": 10}).json()


def test_speakers_are_numbered_by_first_turn_and_take_the_most_talk():
    segments = [{"start": 0.0, "end": 2.0, "text": "a"}, {"start": 2.0, "end": 4.0, "text": "b"}]
    turns = [(2.1, 4.0, "spk_1"), (0.0, 1.5, "spk_0"), (1.5, 2.0, "spk_0")]
    assert diarization.attach_speakers(segments, turns) == 2
    assert [row["speaker"] for row in segments] == ["Speaker 1", "Speaker 2"]


def test_numbering_follows_time_not_the_raw_label_order():
    segments = [{"start": 0.0, "end": 1.0, "text": "late"}, {"start": 3.0, "end": 4.0, "text": "early"}]
    turns = [(0.0, 1.0, "spk_9"), (3.0, 4.0, "spk_2")]
    diarization.attach_speakers(segments, turns)
    assert [row["speaker"] for row in segments] == ["Speaker 1", "Speaker 2"]
    assert segments[0]["speaker"] == "Speaker 1" and segments[1]["speaker"] == "Speaker 2"


def test_segments_without_overlap_get_no_speaker_and_no_turns_change_nothing():
    segments = [{"start": 5.0, "end": 6.0, "text": "silence"}]
    assert diarization.attach_speakers(segments, [(0.0, 1.0, "spk_0")]) == 1
    assert "speaker" not in segments[0]
    untouched = [{"start": 0.0, "end": 1.0, "text": "x"}]
    assert diarization.attach_speakers(untouched, []) == 0
    assert untouched == [{"start": 0.0, "end": 1.0, "text": "x"}]


def test_exports_carry_speaker_labels_only_when_present():
    rows = [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "Speaker 1"},
            {"start": 1.0, "end": 2.0, "text": "plain"}]
    assert exporters.to_txt(rows) == "Speaker 1: hello\nplain\n"
    assert exporters.to_srt(rows).startswith("1\n00:00:00,000 --> 00:00:01,000\nSpeaker 1: hello\n")
    assert "<v Speaker 1>hello</v>" in exporters.to_vtt(rows)
    parsed = json.loads(exporters.to_json(rows))
    assert parsed[0] == {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "Speaker 1"}
    assert "speaker" not in parsed[1]
    batch = json.loads(exporters.batch_json("b", "now", [{"filename": "a.wav", "status": "completed", "segments": rows}]))
    assert batch["files"][0]["segments"][0]["speaker"] == "Speaker 1"


def test_token_comes_from_the_environment_or_the_project_env_file(monkeypatch, tmp_path):
    monkeypatch.setattr(diarization, "ROOT_DIR", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert diarization.hf_token() is None
    (tmp_path / ".env").write_text('GEMINI_API_KEY=unused\nHF_TOKEN="hf_example"\n', encoding="utf-8")
    assert diarization.hf_token() == "hf_example"
    monkeypatch.setenv("HF_TOKEN", "from_environment")
    assert diarization.hf_token() == "from_environment"


def test_availability_says_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(diarization, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(diarization, "diarizer_python", lambda: None)
    assert "scripts/setup-diarizer.sh" in diarization.availability()["reason"]
    monkeypatch.setattr(diarization, "diarizer_python", lambda: tmp_path / "python")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert "HF_TOKEN" in diarization.availability()["reason"]
    monkeypatch.setenv("HF_TOKEN", "t")
    assert diarization.availability() == {"available": True, "model": diarization.MODEL_ID, "license": "MIT",
                                          "reason": None}


def test_diarize_is_refused_up_front_when_unavailable(monkeypatch):
    mock_audio_tools(monkeypatch)
    monkeypatch.setattr(diarization, "availability", lambda: {"available": False, "model": diarization.MODEL_ID,
                                                              "license": "MIT", "reason": "HF_TOKEN is not set"})
    response = _submit(_client(), diarize=True)
    assert response.status_code == 422
    assert response.json()["detail"] == "HF_TOKEN is not set"


def test_diarized_job_labels_segments_and_records_the_model(monkeypatch):
    mock_audio_tools(monkeypatch, duration=1.5)
    monkeypatch.setattr(diarization, "availability", lambda: AVAILABLE)
    seen = {}

    def fake_diarize(source, *, seconds=None, should_stop=None, cache_dir=None):
        seen.update(seconds=seconds, cache_dir=cache_dir)
        return [(0.0, 1.5, "spk_0")]

    monkeypatch.setattr(diarization, "diarize_file", fake_diarize)
    client = _client()
    job_id = _submit(client, diarize=True).json()["id"]
    job = _finished(client, job_id)
    assert job["status"] == "completed" and job["diarize"] is True
    assert job["segments"][0]["speaker"] == "Speaker 1"
    assert job["info"]["diarization"] == {"model": diarization.MODEL_ID, "license": "MIT",
                                          "status": "completed", "speakers": 1}
    assert seen == {"seconds": 1.5, "cache_dir": app_module.MODEL_CACHE}


def test_a_diarizer_failure_keeps_the_transcript_and_reports_why(monkeypatch):
    mock_audio_tools(monkeypatch, duration=1.5)
    monkeypatch.setattr(diarization, "availability", lambda: AVAILABLE)

    def fails(source, **kwargs):
        raise diarization.DiarizationError("Speaker labelling failed (exit 1): ValueError: bad <path>")

    monkeypatch.setattr(diarization, "diarize_file", fails)
    client = _client()
    job = _finished(client, _submit(client, diarize=True).json()["id"])
    assert job["status"] == "completed"
    assert job["segments"][0]["text"] == "hello" and "speaker" not in job["segments"][0]
    assert job["info"]["diarization"]["status"] == "failed"
    assert "bad <path>" in job["info"]["diarization"]["reason"]


def test_plain_jobs_never_touch_the_diarizer(monkeypatch):
    mock_audio_tools(monkeypatch, duration=1.5)

    def must_not_run(*args, **kwargs):
        raise AssertionError("diarizer called for a job without diarize=true")

    monkeypatch.setattr(diarization, "diarize_file", must_not_run)
    client = _client()
    job = _finished(client, _submit(client).json()["id"])
    assert job["status"] == "completed" and "diarization" not in (job["info"] or {})


@pytest.fixture()
def fake_diarizer(monkeypatch, tmp_path):
    """Make diarize_file run a small shell script in place of the pyannote interpreter."""
    monkeypatch.setattr(diarization, "hf_token", lambda: "test-token")

    def install(body):
        script = tmp_path / "fake-diarizer"
        script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        script.chmod(0o755)
        monkeypatch.setattr(diarization, "diarizer_python", lambda: script)
        return script

    return install


@pytest.fixture()
def sample_wav(tmp_path):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg and ffprobe are required")
    path = tmp_path / "sample.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-y", str(path)], check=True)
    return path


def test_subprocess_boundary_returns_turns_from_the_interpreter(fake_diarizer, sample_wav):
    # The script receives: $1 = diarize_cli.py, $2 = wav, $3 = output json.
    fake_diarizer('printf \'{"turns": [[0, 1000, "spk_0"], [1000, 2000, "spk_1"]], "device": "cpu"}\' > "$3"\n')
    turns = diarization.diarize_file(sample_wav, seconds=2.0)
    assert turns == [(0.0, 1.0, "spk_0"), (1.0, 2.0, "spk_1")]


def test_subprocess_failure_is_reported_without_local_paths(fake_diarizer, sample_wav):
    fake_diarizer('echo "Traceback (most recent call last):" >&2\n'
                  'echo "OSError: cannot read /Users/someone/private/model.bin" >&2\nexit 1\n')
    with pytest.raises(diarization.DiarizationError) as caught:
        diarization.diarize_file(sample_wav, seconds=2.0)
    message = str(caught.value)
    assert "exit 1" in message and "OSError" in message
    assert "/Users/someone" not in message and "<path>" in message


def test_cancel_stops_the_whole_diarizer_process_group(fake_diarizer, sample_wav):
    fake_diarizer("sleep 30\n")
    started = time.monotonic()
    with pytest.raises(diarization.DiarizationError, match="cancelled"):
        diarization.diarize_file(sample_wav, seconds=2.0, should_stop=lambda: time.monotonic() - started > 1.0)
    assert time.monotonic() - started < 10
