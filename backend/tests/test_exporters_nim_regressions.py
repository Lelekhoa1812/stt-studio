"""Regression tests for transcript exporters and the NVIDIA NIM client.

Every NIM test stubs the HTTP call and the sleep function. Nothing here reaches the real NVIDIA API.
"""
import json
import re

import pytest

from backend import exporters, nim

KEY_ONE = "nvapi-regression-key-one-0001"
KEY_TWO = "nvapi-regression-key-two-0002"

SRT_TIMING = re.compile(r"^\d{2,}:\d{2}:\d{2},\d{3} --> \d{2,}:\d{2}:\d{2},\d{3}$")
VTT_TIMING = re.compile(r"^\d{2,}:\d{2}:\d{2}\.\d{3} --> \d{2,}:\d{2}:\d{2}\.\d{3}$")


def _srt_cues(text):
    """Strict parse: each cue is an index, one timing line and exactly one text line, separated by a blank line."""
    assert text.endswith("\n")
    cues = []
    for block in text[:-1].split("\n\n"):
        lines = block.split("\n")
        assert len(lines) == 3, lines
        assert lines[0] == str(len(cues) + 1)
        assert SRT_TIMING.match(lines[1]), lines[1]
        cues.append((lines[1], lines[2]))
    return cues


def _vtt_cues(text):
    assert text.startswith("WEBVTT\n") and text.endswith("\n")
    blocks = text[len("WEBVTT\n"):].split("\n\n")
    cues = []
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip("\n").split("\n")
        assert len(lines) == 2, lines
        assert VTT_TIMING.match(lines[0]), lines[0]
        cues.append((lines[0], lines[1]))
    return cues


# ---------------------------------------------------------------- item 1: cue text sanitising

@pytest.mark.parametrize("text, expected_text", [
    ("a\n\nb", "a b"),
    ("a\r\nb", "a b"),
    ("a \n\n  b", "a b"),
    ("x --> y", "x -> y"),
    ("a ---> b", "a -> b"),
])
def test_srt_cue_text_stays_on_one_line_and_keeps_arrows_out(text, expected_text):
    rendered = exporters.to_srt([{"start": 1.0, "end": 2.0, "text": text}])
    cues = _srt_cues(rendered)
    assert cues == [("00:00:01,000 --> 00:00:02,000", expected_text)]


def test_srt_speaker_name_with_a_newline_or_arrow_stays_in_one_cue():
    rendered = exporters.to_srt([{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "A\n\nB --> C"}])
    assert _srt_cues(rendered) == [("00:00:00,000 --> 00:00:01,000", "A B -> C: hi")]


@pytest.mark.parametrize("text, expected_text", [
    ("a\n\nb", "a b"),
    ("x --> y", "x -&gt; y"),
    ("<b>bold</b> & more", "&lt;b&gt;bold&lt;/b&gt; &amp; more"),
])
def test_vtt_cue_text_is_one_line_and_escaped(text, expected_text):
    rendered = exporters.to_vtt([{"start": 1.0, "end": 2.0, "text": text}])
    assert _vtt_cues(rendered) == [("00:00:01.000 --> 00:00:02.000", expected_text)]


def test_vtt_speaker_names_are_escaped_inside_the_voice_tag():
    rendered = exporters.to_vtt([{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "A>B"}])
    assert _vtt_cues(rendered) == [("00:00:00.000 --> 00:00:01.000", "<v A&gt;B>hello</v>")]


def test_vtt_speaker_with_ampersand_and_newline_is_one_cue():
    rendered = exporters.to_vtt([{"start": 0.0, "end": 1.0, "text": "a\n\nb", "speaker": "R&D\n\nTeam"}])
    assert _vtt_cues(rendered) == [("00:00:00.000 --> 00:00:01.000", "<v R&amp;D Team>a b</v>")]


def test_each_hostile_segment_yields_exactly_one_parseable_cue_in_srt_and_vtt():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "line one\n\nline two"},
        {"start": 1.0, "end": 2.0, "text": "gone --> later", "speaker": "A>B"},
        {"start": 2.0, "end": 3.0, "text": "<i>tag</i> & done"},
    ]
    assert len(_srt_cues(exporters.to_srt(segments))) == 3
    assert len(_vtt_cues(exporters.to_vtt(segments))) == 3


def test_txt_and_json_keep_the_raw_text_unchanged():
    segments = [{"start": 0.0, "end": 1.0, "text": "a\n\nb --> c", "speaker": "A>B"}]
    assert exporters.to_txt(segments) == "A>B: a\n\nb --> c\n"
    assert json.loads(exporters.to_json(segments)) == [
        {"start": 0.0, "end": 1.0, "text": "a\n\nb --> c", "speaker": "A>B"}]


# ---------------------------------------------------------------- item 2: empty and backwards cues

@pytest.mark.parametrize("segments", [[], [{"start": 0.0, "end": 1.0, "text": " "}],
                                      [{"start": 0.0, "end": 1.0, "text": ""}, {"start": 1.0, "end": 2.0, "text": "\n"}]])
def test_all_empty_input_renders_an_empty_txt_and_srt(segments):
    assert exporters.to_txt(segments) == ""
    assert exporters.to_srt(segments) == ""
    assert exporters.render(segments, "txt") == "" and exporters.render(segments, "srt") == ""


def test_all_empty_vtt_and_json_keep_their_valid_empty_documents():
    # A WebVTT file must start with its signature and JSON must stay parseable, so these two keep their shape.
    assert exporters.to_vtt([{"start": 0.0, "end": 1.0, "text": " "}]) == "WEBVTT\n"
    assert json.loads(exporters.to_json([{"start": 0.0, "end": 1.0, "text": " "}])) == []


def test_a_backwards_segment_is_clamped_to_zero_length_in_every_format():
    segments = [{"start": 5.0, "end": 3.0, "text": "backwards"}]
    assert exporters.to_srt(segments) == "1\n00:00:05,000 --> 00:00:05,000\nbackwards\n"
    assert exporters.to_vtt(segments) == "WEBVTT\n\n00:00:05.000 --> 00:00:05.000\nbackwards\n"
    assert json.loads(exporters.to_json(segments)) == [{"start": 5.0, "end": 5.0, "text": "backwards"}]
    assert exporters.to_txt(segments) == "backwards\n"


def test_a_forward_segment_keeps_its_end_time():
    segments = [{"start": 3.0, "end": 5.0, "text": "forward"}]
    assert exporters.to_srt(segments) == "1\n00:00:03,000 --> 00:00:05,000\nforward\n"
