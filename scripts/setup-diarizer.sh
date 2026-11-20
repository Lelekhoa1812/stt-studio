#!/usr/bin/env bash
# Builds the isolated speaker-labelling environment used by backend/diarization.py.
# pyannote 3.4 needs torch 2.8, which does not belong in the backend's own venv. Model weights are downloaded on
# first use with HF_TOKEN from .env; the account must have accepted the terms of pyannote/speaker-diarization-3.1
# and pyannote/segmentation-3.0 on Hugging Face.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${STT_STUDIO_DIARIZER_BASE_PYTHON:-$(command -v python3.12 || true)}"
[[ -n "$PY" ]] || { echo "python3.12 is required (brew install python@3.12)" >&2; exit 1; }
VENV="${ROOT}/.local/diarizer-venv"
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --disable-pip-version-check -q \
  "pyannote.audio==3.4.0" "torch==2.8.0" "torchaudio==2.8.0" "huggingface_hub<1.0" numpy
"$VENV/bin/python" -c "import pyannote.audio, torch; print('diarizer ready:', pyannote.audio.__version__, 'torch', torch.__version__)"
