"""Real pyannote run on synthetic two-voice speech.

Opt-in: set STT_STUDIO_REAL_DIARIZATION=1. It needs the diarizer environment, HF_TOKEN, macOS say and ffmpeg, and it
downloads model weights into models/ on first use, so a plain test run never does it.
"""
import os
import shutil
import subprocess

import pytest

from backend import app as app_module
from backend import diarization

pytestmark = pytest.mark.skipif(
    os.environ.get("STT_STUDIO_REAL_DIARIZATION") != "1"
    or diarization.diarizer_python() is None or diarization.hf_token() is None or shutil.which("say") is None
    or shutil.which("ffmpeg") is None,
    reason="opt-in: set STT_STUDIO_REAL_DIARIZATION=1 with the diarizer environment, HF_TOKEN, say and ffmpeg",
)

TURNS = [
    ("Samantha", "Good morning everyone. Let us start with the budget review for the quarter."),
    ("Daniel", "Thanks. I have the revised numbers, and the travel line came in under plan."),
    ("Samantha", "That is good news. Can you walk us through the software renewals as well?"),
    ("Daniel", "Yes. Two licences are up in March, and we can renegotiate both of them."),
    ("Samantha", "Perfect. Please send me the draft before Thursday afternoon."),
    ("Daniel", "Agreed. I will also review the interview schedule with the hiring managers."),
]


def _two_voice_clip(directory):
    parts, truth, cursor, gap = [], [], 0.0, 0.6
    for index, (voice, text) in enumerate(TURNS):
        aiff, wav, silence = (directory / f"{name}{index:02d}.{ext}" for name, ext in
                              (("turn", "aiff"), ("turn", "wav"), ("gap", "wav")))
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(aiff), "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", "-y", str(wav)], check=True)
        duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", str(wav)], capture_output=True, text=True,
                                        check=True).stdout)
        truth.append((cursor, cursor + duration, voice))
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                        "-t", str(gap), "-c:a", "pcm_s16le", "-y", str(silence)], check=True)
        parts += [wav, silence]
        cursor += duration + gap
    listing = directory / "list.txt"
    listing.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
    clip = directory / "two-voices.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c:a", "pcm_s16le", "-y", str(clip)], check=True)
    return clip, truth


def test_two_voices_are_separated_and_consistent(tmp_path):
    clip, truth = _two_voice_clip(tmp_path)
    turns = diarization.diarize_file(clip, seconds=truth[-1][1] + 1.0, cache_dir=app_module.MODEL_CACHE)
    labels = {raw for _, _, raw in turns}
    assert len(labels) == 2, f"expected two voices, got {len(labels)}"

    # Each true voice should map to one diarized label, and the mapping should cover most of the talk time.
    talk = {}
    for start, end, voice in truth:
        for s, e, raw in turns:
            overlap = min(end, e) - max(start, s)
            if overlap > 0:
                talk[(voice, raw)] = talk.get((voice, raw), 0.0) + overlap
    mapping = {}
    for voice in {v for v, _ in talk}:
        mapping[voice] = max((raw for (v, raw) in talk if v == voice), key=lambda raw: talk[(voice, raw)])
    assert len(set(mapping.values())) == 2
    correct = sum(value for (voice, raw), value in talk.items() if mapping[voice] == raw)
    total = sum(talk.values())
    assert correct / total >= 0.9
