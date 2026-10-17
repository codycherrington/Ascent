"""Tokenizer tests (stdlib unittest). Run from the repo root:
    python -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tokenizer as T  # noqa: E402

TOK_PATH = os.path.join(ROOT, T.TOKENIZER_PATH)
VAL_PATH = os.path.join(ROOT, T.VAL_JSONL)

SAMPLES = [
    "Once upon a time, there was a little girl.",
    'She said, "Hello!" and he replied, \'Hi, how are you?\'',
    "Line one.\nLine two.\n\nLine four after a blank line.",
    "There were 3 cats, 12 dogs and 2024 stars.",
    "  leading and trailing spaces  ",
    "Café — naïve “quotes”",
]


@unittest.skipUnless(os.path.exists(TOK_PATH), "data/tokenizer.json missing; run `tokenizer.py train`")
class TokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = T.load_tokenizer(TOK_PATH)

    def test_special_token_ids(self):
        ids = [self.tok.token_to_id(t) for t in ["<pad>", "<unk>", "<bos>", "<eos>"]]
        self.assertEqual(ids, [0, 1, 2, 3])

    def test_vocab_size(self):
        self.assertEqual(self.tok.get_vocab_size(), 8192)
        self.assertLess(self.tok.get_vocab_size(), 65536)

    def test_round_trip_strings(self):
        for s in SAMPLES:
            with self.subTest(s=s):
                self.assertEqual(T.decode(self.tok, T.encode(self.tok, s)), s)

    def test_no_unk_for_ascii(self):
        ascii_text = "".join(chr(i) for i in range(32, 127)) + "\n\t"
        self.assertNotIn(1, T.encode(self.tok, ascii_text))

    @unittest.skipUnless(os.path.exists(VAL_PATH), "validation file missing")
    def test_round_trip_sample_stories(self):
        with open(VAL_PATH, encoding="utf-8") as f:
            for i, line in zip(range(50), f):
                text = json.loads(line)["text"]
                with self.subTest(story=i):
                    self.assertEqual(T.decode(self.tok, T.encode(self.tok, text)), text)


if __name__ == "__main__":
    unittest.main()
