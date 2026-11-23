"""Regression tests for speaker labelling: numbering, worker environment, redaction, gated models, idling."""
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
