"""Speaker labels: attachment, exports, availability, the subprocess boundary and job behaviour."""
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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
    assert diarization.attach_speakers(segments, [(0.0, 1.0, "spk_0")]) == 0
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


FAKE_WORKER = '''
import json
import os
import subprocess
import sys
import time
from pathlib import Path

events = Path(os.environ["FAKE_LOG"])
control = Path(os.environ["FAKE_CONTROL"])


def note(*fields):
    with events.open("a", encoding="utf-8") as handle:
        handle.write(" ".join(str(field) for field in (os.getpid(),) + fields) + "\\n")


mode = control.read_text(encoding="utf-8").strip()
if mode == "refuse":
    print(json.dumps({"ready": False, "error": "RuntimeError: rejected test-token for /Users/someone/model"}), flush=True)
    sys.exit(1)
note("start")
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    mode = control.read_text(encoding="utf-8").strip()
    note("request", request["id"], mode)
    if mode == "crash":
        os._exit(3)
    if mode == "hang":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path(os.environ["FAKE_CHILD"]).write_text(str(child.pid), encoding="utf-8")
        time.sleep(60)
    if mode == "fail":
        Path(request["log"]).write_text(
            "Traceback (most recent call last):\\nOSError: cannot read /Users/someone/private/model.bin with test-token\\n",
            encoding="utf-8")
        print(json.dumps({"id": request["id"], "ok": False}), flush=True)
        continue
    Path(request["out"]).write_text(json.dumps({"turns": [[0, 1000, "spk_0"], [1000, 2000, "spk_1"]], "device": "cpu"}),
                                    encoding="utf-8")
    note("out", request["out"])
    print(json.dumps({"id": request["id"], "ok": True}), flush=True)
'''

TURNS = [(0.0, 1.0, "spk_0"), (1.0, 2.0, "spk_1")]


@pytest.fixture()
def fake_worker(monkeypatch, tmp_path):
    """Run the diarizer worker protocol against a fake script in place of the pyannote interpreter."""
    monkeypatch.setattr(diarization, "hf_token", lambda: "test-token")
    monkeypatch.setattr(diarization, "diarizer_python", lambda: Path(sys.executable))
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(diarization, "CLI_PATH", script)
    control = tmp_path / "mode"
    control.write_text("ok", encoding="utf-8")
    log = tmp_path / "events.log"
    child = tmp_path / "child.pid"
    monkeypatch.setenv("FAKE_CONTROL", str(control))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_CHILD", str(child))

    def events():
        if not log.exists():
            return []
        return [line.split(" ", 2) for line in log.read_text(encoding="utf-8").splitlines()]

    yield SimpleNamespace(mode=lambda value: control.write_text(value, encoding="utf-8"), events=events, child=child)
    diarization.stop_diarizer()


def _wait_until_gone(pid):
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail(f"process {pid} survived")


@pytest.fixture()
def sample_wav(tmp_path):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg and ffprobe are required")
    path = tmp_path / "sample.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-y", str(path)], check=True)
    return path


def test_a_worker_request_returns_the_turns(fake_worker, sample_wav):
    assert diarization.diarize_file(sample_wav, seconds=2.0) == TURNS


def test_sequential_requests_reuse_one_worker_process(fake_worker, sample_wav):
    for _ in range(3):
        assert diarization.diarize_file(sample_wav, seconds=2.0) == TURNS
    events = fake_worker.events()
    starts = [event for event in events if event[1] == "start"]
    requests = [event for event in events if event[1] == "request"]
    assert len(starts) == 1 and len(requests) == 3
    assert {event[0] for event in events} == {starts[0][0]}


def test_cancel_kills_the_worker_group_and_the_next_request_starts_a_new_worker(fake_worker, sample_wav):
    fake_worker.mode("hang")
    started = time.monotonic()
    with pytest.raises(diarization.DiarizationError, match="cancelled"):
        diarization.diarize_file(sample_wav, seconds=2.0,
                                 should_stop=lambda: fake_worker.child.exists()
                                 and fake_worker.child.read_text(encoding="utf-8").strip() != "")
    assert time.monotonic() - started < 10
    _wait_until_gone(int(fake_worker.child.read_text(encoding="utf-8").strip()))
    assert diarization._worker is None
    fake_worker.mode("ok")
    assert diarization.diarize_file(sample_wav, seconds=2.0) == TURNS
    starts = [event for event in fake_worker.events() if event[1] == "start"]
    assert len(starts) == 2 and starts[0][0] != starts[1][0]


def test_a_crashed_worker_fails_the_job_and_is_replaced(fake_worker, sample_wav):
    fake_worker.mode("crash")
    with pytest.raises(diarization.DiarizationError, match="exited with code 3"):
        diarization.diarize_file(sample_wav, seconds=2.0)
    assert diarization._worker is None
    fake_worker.mode("ok")
    assert diarization.diarize_file(sample_wav, seconds=2.0) == TURNS
    assert len([event for event in fake_worker.events() if event[1] == "start"]) == 2


def test_worker_errors_carry_no_token_and_no_local_path(fake_worker, sample_wav):
    fake_worker.mode("fail")
    with pytest.raises(diarization.DiarizationError) as caught:
        diarization.diarize_file(sample_wav, seconds=2.0)
    message = str(caught.value)
    assert "OSError" in message and "cannot read <path> with <token>" in message
    assert "test-token" not in message and "/Users/someone" not in message


def test_startup_errors_carry_no_token_and_no_local_path(fake_worker, sample_wav):
    fake_worker.mode("refuse")
    with pytest.raises(diarization.DiarizationError) as caught:
        diarization.diarize_file(sample_wav, seconds=2.0)
    message = str(caught.value)
    assert "RuntimeError" in message and "<token>" in message and "<path>" in message
    assert "test-token" not in message and "/Users/someone" not in message


def test_each_request_removes_its_temporary_directory(fake_worker, sample_wav):
    for _ in range(2):
        diarization.diarize_file(sample_wav, seconds=2.0)
    outs = [Path(event[2]) for event in fake_worker.events() if event[1] == "out"]
    assert len(outs) == 2 and not any(path.parent.exists() for path in outs)


def test_unload_stops_the_resident_diarizer(fake_worker, sample_wav):
    diarization.diarize_file(sample_wav, seconds=2.0)
    worker = diarization._worker
    assert worker is not None and worker.process.poll() is None
    response = _client().post("/api/models/unload")
    assert response.status_code == 200 and response.json() == {"unloaded": [], "loaded_models": []}
    assert diarization._worker is None and worker.process.poll() is not None


def test_env_file_accepts_export_quotes_and_comments(monkeypatch, tmp_path):
    monkeypatch.setattr(diarization, "ROOT_DIR", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN_OLD=stale\nexport HF_TOKEN='hf_quoted' # rotated\n", encoding="utf-8")
    assert diarization.hf_token() == "hf_quoted"
    (tmp_path / ".env").write_text("HF_TOKEN=hf_plain # note\n", encoding="utf-8")
    assert diarization.hf_token() == "hf_plain"


def test_an_undecodable_env_file_counts_as_no_token_and_capabilities_still_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(diarization, "ROOT_DIR", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_bytes(b"GEMINI_API_KEY=x\nHF_TOKEN=\xff\xfe\n")
    assert diarization.hf_token() is None
    client = TestClient(create_app(ModelManager(lambda *a: FakeModel())), base_url="http://127.0.0.1",
                        client=("127.0.0.1", 12345))
    response = client.get("/api/capabilities")
    assert response.status_code == 200
    assert response.json()["diarization"]["available"] is False


def test_labelling_failures_never_show_a_local_path(monkeypatch, tmp_path):
    from backend import app as app_module

    def fails(source, **kwargs):
        raise OSError(f"cannot read {tmp_path}/secret/model.bin")

    monkeypatch.setattr(diarization, "diarize_file", fails)
    report = app_module._run_diarization(app_module.Job(id="j"), tmp_path / "x.wav", 1.0)
    assert report["status"] == "failed" and str(tmp_path) not in report["reason"]


def test_the_diarizer_stops_when_its_backend_dies(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("diarize_cli", Path(diarization.CLI_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exited = threading.Event()
    codes = []
    monkeypatch.setattr(module.os, "_exit", lambda code: (codes.append(code), exited.set()))
    monkeypatch.setattr(module.os, "getppid", lambda: 1)
    stop = module._exit_when_orphaned(parent_pid=99, poll_seconds=0.01)
    try:
        assert exited.wait(2) and codes == [4]
    finally:
        stop.set()


def test_stale_upload_and_diarization_copies_are_swept_but_fresh_ones_are_kept(monkeypatch, tmp_path):
    from backend import app as app_module

    monkeypatch.setattr(app_module.tempfile, "gettempdir", lambda: str(tmp_path))
    old_file = tmp_path / "whisper-upload-old.wav"
    old_dir = tmp_path / "stt-diarize-old"
    fresh_file = tmp_path / "whisper-upload-fresh.wav"
    old_file.write_bytes(b"x")
    old_dir.mkdir()
    (old_dir / "audio.wav").write_bytes(b"x")
    fresh_file.write_bytes(b"x")
    stale = time.time() - 7200
    os.utime(old_file, (stale, stale))
    os.utime(old_dir, (stale, stale))
    app_module._sweep_stale_temp_files(max_age_seconds=3600)
    assert not old_file.exists() and not old_dir.exists() and fresh_file.exists()


def test_untimed_runtimes_refuse_speaker_labels_with_the_reason():
    from fastapi import HTTPException

    from backend import app as app_module

    with pytest.raises(HTTPException) as caught:
        app_module._require_diarization(True, "mlx-audio")
    assert caught.value.status_code == 422 and "untimed segment" in caught.value.detail
    app_module._require_diarization(False, "mlx-audio")  # not requested: nothing to check
    assert app_module.RUNTIME_CAPABILITIES["faster-whisper"]["timedSegments"] is True
    assert app_module.RUNTIME_CAPABILITIES["mlx-audio"]["timedSegments"] is False


def test_cancel_during_labelling_reports_cancelled_not_failed(monkeypatch, tmp_path):
    from backend import app as app_module

    def cancelled(source, **kwargs):
        raise diarization.DiarizationError("Speaker labelling was cancelled")

    monkeypatch.setattr(diarization, "diarize_file", cancelled)
    job = app_module.Job(id="j", cancel_requested=True)
    report = app_module._run_diarization(job, tmp_path / "x.wav", 1.0)
    assert report["status"] == "cancelled" and "reason" not in report
