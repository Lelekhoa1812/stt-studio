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


if __name__ == "__main__":
    unittest.main()

