import json
import tempfile
import unittest
from pathlib import Path

from evaluation.wer import compare, read_transcript, tokenize


class TokenizationTests(unittest.TestCase):
    def test_case_punctuation_and_unicode_normalization(self):
        # Decomposed and precomposed accents normalize to the same token.
        self.assertEqual(tokenize("CAFÉ, don’t stop!"), tokenize("cafe\u0301 don't stop"))
        self.assertEqual(tokenize("Hello, world!"), ["hello", "world"])


class CompareTests(unittest.TestCase):
    def test_counts_substitutions_insertions_and_deletions(self):
        result = compare("we transcribe audio today", "we describe audio")
        self.assertEqual(result.reference_words, 4)
        self.assertEqual(result.hypothesis_words, 3)
        self.assertEqual(result.substitutions, 1)
        self.assertEqual(result.insertions, 0)
        self.assertEqual(result.deletions, 1)
        self.assertEqual(result.errors, 2)
        self.assertEqual(result.wer, 0.5)

    def test_insertions(self):
        result = compare("we transcribe", "we can transcribe it")
        self.assertEqual((result.substitutions, result.insertions, result.deletions), (0, 2, 0))
        self.assertEqual(result.wer, 1.0)

    def test_perfect_transcript_after_normalization(self):
        result = compare("Café, here.", "CAFE\u0301 here")
        self.assertEqual(result.errors, 0)
        self.assertEqual(result.wer, 0.0)

    def test_empty_reference_is_undefined_but_counts_insertions(self):
        result = compare(" ... ", "hello, world")
        self.assertEqual(result.reference_words, 0)
        self.assertEqual(result.insertions, 2)
        self.assertEqual(result.errors, 2)
        self.assertIsNone(result.wer)

    def test_both_empty(self):
        result = compare("", "")
        self.assertEqual(result.errors, 0)
        self.assertIsNone(result.wer)


class FileLoadingTests(unittest.TestCase):
    def test_reads_plain_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.txt"
            path.write_text("café", encoding="utf-8")
            self.assertEqual(read_transcript(path), "café")

    def test_reads_segment_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hypothesis.json"
            path.write_text(json.dumps([
                {"start": 0, "text": "hello"},
                {"start": 1, "text": "world"},
            ]), encoding="utf-8")
            self.assertEqual(read_transcript(path), "hello world")


if __name__ == "__main__":
    unittest.main()
