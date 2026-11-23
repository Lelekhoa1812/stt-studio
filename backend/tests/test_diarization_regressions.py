"""Regression tests for speaker labelling: numbering, worker environment, redaction, gated models, idling."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from backend import diarization


def test_numbering_skips_speakers_whose_turns_label_no_segment():
    # spk_0 talks only in silence before the first segment, so only spk_1 may be "Speaker 1".
    segments = [{"start": 3.0, "end": 4.0, "text": "hello"}]
    turns = [(0.0, 0.5, "spk_0"), (3.0, 4.0, "spk_1")]
    assert diarization.attach_speakers(segments, turns) == 1
    assert segments[0]["speaker"] == "Speaker 1"


def test_numbering_counts_only_labelled_speakers_in_order_of_first_labelled_segment():
    segments = [{"start": 1.0, "end": 2.0, "text": "a"}, {"start": 5.0, "end": 6.0, "text": "b"}]
    turns = [(0.0, 0.5, "spk_0"), (1.0, 2.0, "spk_2"), (5.0, 6.0, "spk_1"), (7.0, 8.0, "spk_3")]
    assert diarization.attach_speakers(segments, turns) == 2
    assert [row["speaker"] for row in segments] == ["Speaker 1", "Speaker 2"]


WORKER_ENV_DUMP = '''
import json
import os
import sys
from pathlib import Path

Path({dump!r}).write_text(json.dumps(dict(os.environ)), encoding="utf-8")
print(json.dumps({{"ready": True}}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    Path(request["out"]).write_text(json.dumps({{"turns": [[0, 1000, "spk_0"]], "device": "cpu"}}), encoding="utf-8")
    print(json.dumps({{"id": request["id"], "ok": True}}), flush=True)
'''


@pytest.fixture()
def env_worker(monkeypatch, tmp_path):
    """A fake worker that records the environment it was started with, so the child env can be inspected."""
    monkeypatch.setattr(diarization, "hf_token", lambda: "hf_test_token")
    monkeypatch.setattr(diarization, "diarizer_python", lambda: Path(sys.executable))
    dump = tmp_path / "worker-env.json"
    script = tmp_path / "env_worker.py"
    script.write_text(WORKER_ENV_DUMP.format(dump=str(dump)), encoding="utf-8")
    monkeypatch.setattr(diarization, "CLI_PATH", script)
    monkeypatch.setattr(diarization, "_to_wav", lambda source, target: target.write_bytes(b"wav"))
    yield dump
    diarization.stop_diarizer()


@pytest.fixture()
def sample_audio(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"not decoded: _to_wav is replaced in these tests")
    return path


def test_worker_environment_is_an_allowlist_and_drops_provider_keys(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY_1", "nvapi-secret")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-secret-2")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("HF_HOME", "/tmp/hf-home")
    monkeypatch.setenv("STT_STUDIO_DIARIZER_THRESHOLD", "0.5")
    monkeypatch.setenv("PYTHONPATH", "/tmp/extra-path")
    env = diarization._worker_environment("hf_token_value", Path("/tmp/hub-cache"))
    assert "NVIDIA_NIM_API_KEY_1" not in env and "NVIDIA_NIM_API_KEY" not in env
    assert "GEMINI_API_KEY" not in env and "AWS_SECRET_ACCESS_KEY" not in env
    assert env["HF_TOKEN"] == "hf_token_value"
    assert env["HF_HUB_CACHE"] == "/tmp/hub-cache"
    assert env["HF_HOME"] == "/tmp/hf-home"
    assert env["STT_STUDIO_DIARIZER_THRESHOLD"] == "0.5"
    assert env["PYTHONPATH"] == "/tmp/extra-path"
    assert set(env) <= diarization._WORKER_ENV_NAMES | {"HF_TOKEN", "HF_HUB_CACHE"}


def test_worker_process_does_not_inherit_nim_keys(monkeypatch, env_worker, sample_audio):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY_1", "nvapi-secret")
    monkeypatch.setattr(diarization, "_to_wav", lambda source, target: target.write_bytes(b"wav"))
    assert diarization.diarize_file(sample_audio, seconds=1.0) == [(0.0, 1.0, "spk_0")]
    seen = json.loads(env_worker.read_text(encoding="utf-8"))
    assert "NVIDIA_NIM_API_KEY_1" not in seen
    assert "nvapi-secret" not in env_worker.read_text(encoding="utf-8")
    assert seen["HF_TOKEN"] == "hf_test_token"


GATED_MESSAGE_PARTS = ("pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0", "Hugging Face")


def _worker_refusing_at_startup(tmp_path, monkeypatch, error):
    script = tmp_path / "gated_worker.py"
    script.write_text(
        "import json, sys\n"
        f"print(json.dumps({{'ready': False, 'error': {error!r}}}), flush=True)\n"
        "sys.exit(1)\n", encoding="utf-8")
    monkeypatch.setattr(diarization, "hf_token", lambda: "hf_test_token")
    monkeypatch.setattr(diarization, "diarizer_python", lambda: Path(sys.executable))
    monkeypatch.setattr(diarization, "CLI_PATH", script)
    monkeypatch.setattr(diarization, "_to_wav", lambda source, target: target.write_bytes(b"wav"))


@pytest.mark.parametrize("error", [
    "GatedRepoError: 401 Client Error. Cannot access gated repo for url https://huggingface.co/pyannote/x",
    "HfHubHTTPError: 403 Client Error: Forbidden for url https://huggingface.co/api/models/pyannote/x",
    "HTTPError: 401 Unauthorized",
    "RuntimeError: this model is restricted to accepted users",
])
def test_gated_model_refusals_name_the_models_and_the_fix(tmp_path, monkeypatch, error):
    _worker_refusing_at_startup(tmp_path, monkeypatch, error)
    with pytest.raises(diarization.DiarizationError) as caught:
        diarization.diarize_file(tmp_path / "x.wav", seconds=1.0)
    message = str(caught.value)
    assert message.startswith("Speaker labelling failed")
    for part in GATED_MESSAGE_PARTS:
        assert part in message
    assert "accept" in message.lower() and "terms" in message.lower()
    assert "/" not in message.replace("pyannote/speaker-diarization-3.1", "").replace("pyannote/segmentation-3.0", "")
    assert "hf_test_token" not in message


def test_gated_model_failure_during_a_request_gets_the_same_message(tmp_path, monkeypatch):
    script = tmp_path / "late_gated_worker.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    open(request['log'], 'w').write('GatedRepoError: Access to this model is gated\\n')\n"
        "    print(json.dumps({'id': request['id'], 'ok': False}), flush=True)\n", encoding="utf-8")
    monkeypatch.setattr(diarization, "hf_token", lambda: "hf_test_token")
    monkeypatch.setattr(diarization, "diarizer_python", lambda: Path(sys.executable))
    monkeypatch.setattr(diarization, "CLI_PATH", script)
    monkeypatch.setattr(diarization, "_to_wav", lambda source, target: target.write_bytes(b"wav"))
    try:
        with pytest.raises(diarization.DiarizationError) as caught:
            diarization.diarize_file(tmp_path / "x.wav", seconds=1.0)
        assert str(caught.value) == diarization.GATED_MESSAGE
    finally:
        diarization.stop_diarizer()


def test_ordinary_failures_keep_their_own_message(tmp_path, monkeypatch):
    _worker_refusing_at_startup(tmp_path, monkeypatch, "ValueError: bad checkpoint in model.bin")
    with pytest.raises(diarization.DiarizationError) as caught:
        diarization.diarize_file(tmp_path / "x.wav", seconds=1.0)
    assert str(caught.value) == "Speaker labelling failed: ValueError: bad checkpoint in model.bin"


def _wait_for_exit(process, seconds=5.0):
    deadline = time.monotonic() + seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    return process.poll() is not None


def test_the_resident_diarizer_idles_out(monkeypatch, env_worker, sample_audio):
    monkeypatch.setenv("STT_STUDIO_DIARIZER_IDLE_SECONDS", "0.2")
    assert diarization.diarize_file(sample_audio, seconds=1.0) == [(0.0, 1.0, "spk_0")]
    worker = diarization._worker
    assert worker is not None and worker.process.poll() is None
    assert _wait_for_exit(worker.process), "the idle diarizer was never stopped"
    assert diarization._worker is None


def test_each_request_resets_the_idle_timer(monkeypatch, env_worker, sample_audio):
    monkeypatch.setenv("STT_STUDIO_DIARIZER_IDLE_SECONDS", "0.5")
    for _ in range(4):
        diarization.diarize_file(sample_audio, seconds=1.0)
        time.sleep(0.2)
    worker = diarization._worker
    assert worker is not None and worker.process.poll() is None
    assert _wait_for_exit(worker.process)


def test_idle_limit_defaults_to_five_minutes_and_ignores_bad_values(monkeypatch):
    monkeypatch.delenv("STT_STUDIO_DIARIZER_IDLE_SECONDS", raising=False)
    assert diarization._idle_seconds() == 300.0
    monkeypatch.setenv("STT_STUDIO_DIARIZER_IDLE_SECONDS", "45")
    assert diarization._idle_seconds() == 45.0
    monkeypatch.setenv("STT_STUDIO_DIARIZER_IDLE_SECONDS", "soon")
    assert diarization._idle_seconds() == 300.0
    monkeypatch.setenv("STT_STUDIO_DIARIZER_IDLE_SECONDS", "-1")
    assert diarization._idle_seconds() == 300.0
