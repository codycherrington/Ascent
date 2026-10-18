"""Unit tests for chat.py (CPU, tiny random model, no real checkpoint needed)."""
import io
import os
import sys
import tempfile
import unittest
from dataclasses import asdict

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chat  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=16, block_size=16)


def byte_tokenizer():
    """Byte-level tokenizer: ids 0-3 are <pad> <unk> <bos> <eos>, then 256 bytes."""
    vocab = {t: i for i, t in enumerate(["<pad>", "<unk>", "<bos>", "<eos>"])}
    for ch in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[ch] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def save_ckpt(path, seed=0, step=7):
    torch.manual_seed(seed)
    m = GPT(CFG)
    torch.save({"model": m.state_dict(), "optimizer": {}, "step": step,
                "config": {"model": asdict(CFG), "args": {}}}, path)
    return m


class TestFindCheckpoint(unittest.TestCase):
    def test_newest_run_with_best(self):
        with tempfile.TemporaryDirectory() as d:
            for run, files in [("20260101-000000", ["ckpt_best.pt"]),
                               ("20260102-000000", ["ckpt_best.pt", "ckpt.pt"]),
                               ("20260103-000000", ["ckpt.pt"])]:
                os.makedirs(os.path.join(d, run))
                for f in files:
                    open(os.path.join(d, run, f), "w").close()
            got = chat.find_checkpoint(d)
            self.assertEqual(got, os.path.join(d, "20260102-000000", "ckpt_best.pt"))

    def test_falls_back_to_ckpt(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "a"))
            open(os.path.join(d, "a", "ckpt.pt"), "w").close()
            self.assertEqual(chat.find_checkpoint(d), os.path.join(d, "a", "ckpt.pt"))

    def test_run_and_ckpt_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "r"))
            p = os.path.join(d, "r", "ckpt.pt")
            open(p, "w").close()
            self.assertEqual(chat.find_checkpoint(d, run=os.path.join(d, "r")), p)
            self.assertEqual(chat.find_checkpoint(d, ckpt=p), p)

    def test_errors_are_chat_errors(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(chat.ChatError) as cm:
                chat.find_checkpoint(d)
            self.assertIn("train.py", str(cm.exception))
            with self.assertRaises(chat.ChatError):
                chat.find_checkpoint(d, run=os.path.join(d, "missing"))
            with self.assertRaises(chat.ChatError):
                chat.find_checkpoint(d, ckpt=os.path.join(d, "nope.pt"))

    def test_load_model_roundtrip_and_bad_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ckpt.pt")
            ref = save_ckpt(p)
            model, step, params = chat.load_model(p)
            self.assertEqual(step, 7)
            self.assertEqual(params, sum(x.numel() for x in ref.parameters()))
            torch.save({"nope": 1}, p)
            with self.assertRaises(chat.ChatError):
                chat.load_model(p)


class TestIncrementalDecoder(unittest.TestCase):
    def setUp(self):
        self.tok = byte_tokenizer()

    def ids(self, text):
        return self.tok.encode(text).ids

    def test_concatenation_equals_full_decode(self):
        text = "Hi there\nnew line, ok"
        dec = chat.IncrementalDecoder(self.tok)
        out = "".join(dec.feed(i) for i in self.ids(text)) + dec.flush()
        self.assertEqual(out, text)

    def test_multibyte_held_back(self):
        text = "café ☃!"
        dec = chat.IncrementalDecoder(self.tok)
        chunks = [dec.feed(i) for i in self.ids(text)]
        self.assertTrue(all(chat.REPLACEMENT not in c for c in chunks))
        self.assertIn("", chunks)  # some step had only a partial character
        self.assertEqual("".join(chunks) + dec.flush(), text)

    def test_flush_emits_dangling_partial(self):
        dec = chat.IncrementalDecoder(self.tok)
        self.assertEqual(dec.feed(self.ids("☃")[0]), "")
        self.assertEqual(dec.flush(), chat.REPLACEMENT)


class TestCommands(unittest.TestCase):
    def test_parse(self):
        self.assertIsNone(chat.parse_command("Once upon a time"))
        self.assertEqual(chat.parse_command("/temp 0.7"), ("temp", "0.7"))
        self.assertEqual(chat.parse_command("  /QUIT "), ("quit", ""))
        self.assertEqual(chat.parse_command("/"), ("", ""))

    def test_settings(self):
        s = chat.Settings()
        chat.apply_setting(s, "temp", "0.5")
        chat.apply_setting(s, "topk", "10")
        chat.apply_setting(s, "len", "33")
        chat.apply_setting(s, "seed", "42")
        self.assertEqual((s.temperature, s.top_k, s.max_new_tokens, s.seed), (0.5, 10, 33, 42))
        chat.apply_setting(s, "greedy", "")
        self.assertEqual(s.temperature, 0)
        chat.apply_setting(s, "greedy", "")
        self.assertEqual(s.temperature, 0.5)
        chat.apply_setting(s, "seed", "random")
        self.assertIsNone(s.seed)
        for name, arg in [("temp", "abc"), ("temp", "-1"), ("len", "0"), ("topk", "x")]:
            with self.assertRaises(ValueError):
                chat.apply_setting(s, name, arg)


class TestSession(unittest.TestCase):
    def make(self, **kw):
        tok = byte_tokenizer()
        torch.manual_seed(0)
        model = GPT(CFG).eval()
        out = io.StringIO()
        settings = chat.Settings(max_new_tokens=6, **kw)
        return chat.Chat(model, tok, settings, "x.pt", 3, 123, out=out), out

    def fake_stream(self, ids_to_yield):
        """Replace the model with a scripted stream of token ids."""
        def stream(model, ids, max_new, **kw):
            for t in ids_to_yield[:max_new]:
                yield t
                if kw.get("eos_id") is not None and t == kw["eos_id"]:
                    break
        original = chat.generate_stream
        chat.generate_stream = stream
        self.addCleanup(setattr, chat, "generate_stream", original)

    def test_reply_reproducible_with_seed(self):
        c1, o1 = self.make(seed=5)
        c2, o2 = self.make(seed=5)
        c1.handle("hello")
        c2.handle("hello")
        self.assertEqual(o1.getvalue().splitlines()[0], o2.getvalue().splitlines()[0])
        self.assertIn("seed 5", o1.getvalue())
        self.assertIn("tok/s", o1.getvalue())

    def test_eos_stops_and_is_hidden(self):
        c, out = self.make(temperature=0.0)
        a, eos = c.tok.encode("a").ids[0], c.tok.token_to_id("<eos>")
        self.fake_stream([a, a, eos, a])
        c.handle("abc")
        text = out.getvalue()
        self.assertTrue(text.startswith("abcaa"))
        self.assertNotIn("<eos>", text)
        self.assertIn(chat.END_MARKER, text)
        self.assertEqual(c.last.output_ids, [a, a])

    def test_no_stop_runs_on_and_hides_eos(self):
        c, out = self.make(temperature=0.0, stop_at_eos=False)
        a, eos = c.tok.encode("a").ids[0], c.tok.token_to_id("<eos>")
        self.fake_stream([a, eos, a])
        c.handle("abc")
        self.assertIn("abcaa", out.getvalue())
        self.assertNotIn("<eos>", out.getvalue())
        self.assertNotIn(chat.END_MARKER, out.getvalue())

    def test_long_prompt_truncated_with_notice(self):
        c, out = self.make()
        c.handle("x" * 100)
        self.assertIn("prompt too long", out.getvalue())
        self.assertEqual(len(c.last.prompt_ids), CFG.block_size)

    def test_empty_unknown_and_quit(self):
        c, out = self.make()
        self.assertTrue(c.handle("   "))
        self.assertEqual(out.getvalue(), "")
        self.assertTrue(c.handle("/bogus"))
        self.assertIn("unknown command", out.getvalue())
        self.assertTrue(c.handle("/again"))
        self.assertTrue(c.handle("/continue"))
        self.assertFalse(c.handle("/quit"))

    def test_continue_extends_and_again_reruns(self):
        c, out = self.make(temperature=0.0, stop_at_eos=False)
        c.handle("hello")
        n = len(c.last.output_ids)
        c.handle("/continue")
        self.assertGreater(len(c.last.output_ids), n)
        c.handle("/again")
        self.assertEqual(len(c.last.output_ids), 6)

    def test_info(self):
        c, out = self.make()
        c.handle("/info")
        self.assertIn("x.pt", out.getvalue())
        self.assertIn("123", out.getvalue())

    def test_no_color_codes_when_disabled(self):
        c, out = self.make()
        c.handle("hi")
        self.assertNotIn("\033[", out.getvalue())


if __name__ == "__main__":
    unittest.main()
