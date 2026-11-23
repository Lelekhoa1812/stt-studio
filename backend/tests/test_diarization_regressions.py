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
