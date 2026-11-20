"""Speaker diarization in an isolated interpreter (see scripts/setup-diarizer.sh).

Usage: <diarizer-python> diarize_cli.py <audio.wav> <out.json>
Reads HF_TOKEN from the environment. Writes {"turns": [[start_ms, end_ms, "spk_0"], ...], "device": "..."}.
This file must not import backend modules: it runs under the diarizer's own Python.
"""

from __future__ import annotations

import json
import os
import sys
import warnings


def main(audio: str, out: str) -> int:
    warnings.filterwarnings("ignore")
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    # Keep torch.load's weights-only safety; allow only the classes pyannote checkpoints contain.
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion, Specifications, Problem, Resolution])
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("diarizer: HF_TOKEN is not set", file=sys.stderr)
        return 2
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if pipeline is None:
        print("diarizer: model access was refused", file=sys.stderr)
        return 3
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
    result = pipeline(audio)
    names: dict[str, str] = {}
    turns = []
    for segment, _track, label in result.itertracks(yield_label=True):
        name = names.setdefault(label, f"spk_{len(names)}")
        turns.append([int(segment.start * 1000), int(segment.end * 1000), name])
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"turns": turns, "device": device}, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
