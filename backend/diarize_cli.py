"""Speaker diarization worker in an isolated interpreter (see scripts/setup-diarizer.sh).

Usage: <diarizer-python> diarize_cli.py
Reads HF_TOKEN from the environment and loads the pyannote pipeline once. Then it reads one JSON request per line on
stdin, {"id": ..., "audio": "<wav>", "out": "<turns.json>", "log": "<log file>"}, and writes one JSON reply per line
on stdout: {"ready": true} after loading, then {"id": ..., "ok": true|false} per request. Turns are written to out as
{"turns": [[start_ms, end_ms, "spk_0"], ...], "device": "..."}; a failure traceback goes to log.
This file must not import backend modules: it runs under the diarizer's own Python.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
import warnings


def _exit_when_orphaned(parent_pid: int, poll_seconds: float = 1.0) -> threading.Event:
    """Stop if the backend that started this process dies, so no orphaned diarizer keeps a copy of the recording.

    Returns an event that ends the watcher, for callers that are not the diarizer process itself.
    """
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(poll_seconds):
            if os.getppid() != parent_pid:
                os._exit(4)
                return

    threading.Thread(target=watch, daemon=True).start()
    return stop


def load(token: str):
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    # Keep torch.load's weights-only safety; allow only the classes pyannote checkpoints contain.
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion, Specifications, Problem, Resolution])
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if pipeline is None:
        raise RuntimeError("model access was refused")
    # Clustering threshold. On a two-person recording, 0.7046 (pyannote's default) gave two main voices holding
    # 95% of the talk, while 0.5 split the rest into about twelve small clusters. The Axion meeting project
    # needed 0.5 for a nine-person hearing, where 0.7046 merged everyone into two voices. Use
    # STT_STUDIO_DIARIZER_THRESHOLD=0.5 for large groups. Forcing a speaker count was tested and made the
    # two-person split worse, so it is not offered.
    threshold = float(os.environ.get("STT_STUDIO_DIARIZER_THRESHOLD", "0.7046"))
    pipeline.instantiate({"clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": threshold},
                          "segmentation": {"min_duration_off": 0.0}})
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(torch.device(device))
    return pipeline, device


def diarize(pipeline, device: str, audio: str, out: str) -> None:
    result = pipeline(audio)
    names: dict[str, str] = {}
    turns = []
    for segment, _track, label in result.itertracks(yield_label=True):
        name = names.setdefault(label, f"spk_{len(names)}")
        turns.append([int(segment.start * 1000), int(segment.end * 1000), name])
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"turns": turns, "device": device}, handle)


def _reply(protocol, message: dict) -> None:
    protocol.write(json.dumps(message) + "\n")
    protocol.flush()


def serve() -> int:
    _exit_when_orphaned(os.getppid())
    warnings.filterwarnings("ignore")
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)  # library output must not reach the reply stream
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        _reply(protocol, {"ready": False, "error": "HF_TOKEN is not set"})
        return 2
    try:
        pipeline, device = load(token)
    except Exception as exc:
        _reply(protocol, {"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _reply(protocol, {"ready": True})
    for line in sys.stdin:
        request = json.loads(line)
        try:
            diarize(pipeline, device, request["audio"], request["out"])
            _reply(protocol, {"id": request["id"], "ok": True})
        except Exception:
            with open(request["log"], "w", encoding="utf-8") as log:
                log.write(traceback.format_exc())
            _reply(protocol, {"id": request["id"], "ok": False})
    return 0


if __name__ == "__main__":
    sys.exit(serve())
