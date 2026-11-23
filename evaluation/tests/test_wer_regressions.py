import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from evaluation.wer import compare, read_transcript, tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]


class NumberFormattingTests(unittest.TestCase):
    def test_thousands_separators_match_plain_digits(self):
        self.assertEqual(compare("1,000 people", "1000 people").errors, 0)
        self.assertEqual(tokenize("1,000,000 votes"), ["1000000", "votes"])

    def test_commas_that_are_not_thousands_separators_still_split(self):
        self.assertEqual(tokenize("1,25 or 3,4"), ["1", "25", "or", "3", "4"])


class CombiningMarkTests(unittest.TestCase):
    def test_dotted_capital_i_matches_its_plain_spelling(self):
        self.assertEqual(tokenize("İstanbul"), ["istanbul"])
        self.assertEqual(compare("İstanbul", "istanbul").errors, 0)

    def test_combining_marks_do_not_split_a_word(self):
        # q + combining tilde has no precomposed form, so NFC keeps the mark and it must stay in the word.
        self.assertEqual(tokenize("q\u0303uick"), ["q\u0303uick"])
        self.assertEqual(compare("q\u0303uick fox", "q\u0303uick fox").errors, 0)


class FileEncodingTests(unittest.TestCase):
    def test_json_with_a_utf8_byte_order_mark_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"text": "hello world"}).encode("utf-8"))
            self.assertEqual(read_transcript(path), "hello world")

    def test_plain_text_with_a_byte_order_mark_has_no_marker_in_the_words(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.txt"
            path.write_bytes(b"\xef\xbb\xbfhello world")
            self.assertEqual(tokenize(read_transcript(path)), ["hello", "world"])


class SpeakerLabelTests(unittest.TestCase):
    def test_labelled_txt_export_drops_the_speaker_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hypothesis.txt"
            path.write_text("Speaker 1: hello\n", encoding="utf-8")
            self.assertEqual(read_transcript(path), "hello\n")

    def test_every_line_loses_its_own_prefix_and_labels_do_not_count_as_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hypothesis.txt"
            path.write_text("Speaker 1: hello world\nSpeaker 2: good morning\n", encoding="utf-8")
            text = read_transcript(path)
            self.assertEqual(text, "hello world\ngood morning\n")
            self.assertEqual(compare("hello world good morning", text).errors, 0)

    def test_a_speaker_word_in_the_middle_of_a_line_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.txt"
            path.write_text("The Speaker 1: ideas came later", encoding="utf-8")
            self.assertEqual(read_transcript(path), "The Speaker 1: ideas came later")


if __name__ == "__main__":
    unittest.main()

