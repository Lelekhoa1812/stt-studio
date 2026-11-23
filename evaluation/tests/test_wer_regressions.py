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


if __name__ == "__main__":
    unittest.main()

