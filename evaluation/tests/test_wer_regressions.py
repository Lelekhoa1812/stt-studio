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


class UndefinedWerExitCodeTests(unittest.TestCase):
    def _run(self, reference_text, hypothesis_text, *extra):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.txt"
            hypothesis = Path(directory) / "hypothesis.txt"
            reference.write_text(reference_text, encoding="utf-8")
            hypothesis.write_text(hypothesis_text, encoding="utf-8")
            return subprocess.run(
                [sys.executable, "-m", "evaluation.wer", str(reference), str(hypothesis), *extra],
                cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)

    def test_empty_reference_exits_three_with_a_reason(self):
        result = self._run(" ... ", "hello world")
        self.assertEqual(result.returncode, 3)
        self.assertIn("WER: undefined", result.stdout)
        self.assertIn("undefined", result.stderr)

    def test_empty_reference_in_json_output_also_exits_three(self):
        result = self._run("", "hello", "--format", "json")
        self.assertEqual(result.returncode, 3)
        self.assertIsNone(json.loads(result.stdout)["wer"])

    def test_defined_wer_exits_zero(self):
        result = self._run("hello world", "hello there world")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WER: 50.00%", result.stdout)


class AlignmentScaleTests(unittest.TestCase):
    def test_a_thousand_word_transcript_aligns_within_five_seconds(self):
        words = [f"word{index % 97}" for index in range(1000)]
        hypothesis = [w if index % 10 else "other" for index, w in enumerate(words)]
        started = time.monotonic()
        result = compare(" ".join(words), " ".join(hypothesis))
        elapsed = time.monotonic() - started
        self.assertEqual(result.reference_words, 1000)
        self.assertEqual(result.substitutions, 100)
        self.assertLess(elapsed, 5.0, f"alignment took {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
