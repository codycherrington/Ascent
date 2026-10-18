"""Tests for the chat data pipeline (scripts/prepare_chat_data.py) and train.load_init_weights.
No network and no real data needed.
"""
import os
import sys
import tempfile
import unittest
from dataclasses import asdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import GPT, GPTConfig  # noqa: E402
from scripts import prepare_chat_data as P  # noqa: E402
from train import load_init_weights  # noqa: E402

# Mimics the real files: literal backslash-n separators, bare "**Name**:" markup, trailing tag.
LINE = ('**Child**: Look, I have raisins for my snack! \\n\\n **Classmate**: "Yes, please! I have a sandwich." '
        '\\n\\n **Child**: Yummy, yes! \\n\\n **Classmate**: Thank you! <|endoftext|>')
THREE = ('**Babysitter**: Who wants a game? \\n\\n **Child**: Me! \\n\\n **Friend**: Me too! '
         '\\n\\n **Babysitter**: Okay, let\'s go. \\n\\n **Child** (jumping): Yay! <|endoftext|>')


class TestParser(unittest.TestCase):
    def test_basic(self):
        turns = P.parse_conversation(LINE)
        self.assertEqual([s for s, _ in turns], ["Child", "Classmate", "Child", "Classmate"])
        self.assertEqual(turns[1][1], "Yes, please! I have a sandwich.")   # wrapping quotes removed
        self.assertNotIn("<|endoftext|>", turns[-1][1])
        self.assertNotIn("\\n", "".join(t for _, t in turns))

    def test_stage_direction_and_normalisation(self):
        turns = P.parse_conversation(THREE)
        self.assertEqual(turns[-1], ("Child", "Yay!"))
        t = P.parse_conversation("**A**: It’s “fine” \\n\\n  too   \\n\\n **B**: ok --- sure")
        self.assertEqual(t[0][1], 'It\'s "fine" too')
        self.assertEqual(t[1][1], "ok sure")

    def test_unusable(self):
        self.assertIsNone(P.parse_conversation("**A**: only one turn <|endoftext|>"))
        self.assertIsNone(P.parse_conversation("**A**: café ☃ \\n\\n **B**: hi"))
        self.assertIsNone(P.parse_conversation("no markup at all"))


class TestRoles(unittest.TestCase):
    def test_collapse_three_speakers(self):
        sides = P.to_sides(P.parse_conversation(THREE))
        # Babysitter | Child + Friend merged | Babysitter | Child
        self.assertEqual([s for s, _ in sides], [0, 1, 0, 1])
        self.assertEqual(sides[1][1], "Me! Me too!")

    def test_single_side_rejected(self):
        self.assertIsNone(P.to_sides([("A", "hi"), ("a", "again")]))

    def test_assign_roles_alternate_and_deterministic(self):
        sides = P.to_sides(P.parse_conversation(LINE))
        r1 = P.assign_roles(sides, "5:train:1", 7)
        self.assertEqual(r1, P.assign_roles(sides, "5:train:1", 7))
        roles = [r for r, _ in r1]
        self.assertIn(roles, (["You", "Ascent"] * 2, ["Ascent", "You"] * 2))

    def test_both_assignments_occur(self):
        sides = [(0, "a"), (1, "b")]
        firsts = {P.assign_roles(sides, f"k{i}", 1)[0][0] for i in range(40)}
        self.assertEqual(firsts, {"You", "Ascent"})


class TestSplitter(unittest.TestCase):
    def check(self, roles, lens, budget):
        wins = P.split_windows(roles, lens, budget)
        for s, e in wins:
            self.assertEqual(roles[s], "You")
            self.assertEqual(roles[e - 1], "Ascent")
            self.assertGreaterEqual(e - s, 2)
            self.assertLessEqual(sum(lens[s:e]), budget)
        return wins

    def test_fits_in_one_window(self):
        roles = ["You", "Ascent"] * 3
        self.assertEqual(self.check(roles, [10] * 6, 255), [(0, 6)])

    def test_long_conversation_is_cut_on_turn_boundaries(self):
        roles = ["You", "Ascent"] * 10
        wins = self.check(roles, [30] * 20, 255)
        self.assertGreater(len(wins), 1)
        self.assertEqual(wins, [(0, 8), (8, 16), (16, 20)])

    def test_leading_ascent_skipped(self):
        roles = ["Ascent", "You", "Ascent"]
        self.assertEqual(self.check(roles, [5, 5, 5], 255), [(1, 3)])

    def test_trailing_you_not_included(self):
        roles = ["You", "Ascent", "You"]
        self.assertEqual(self.check(roles, [5, 5, 5], 255), [(0, 2)])

    def test_too_few_or_too_long_dropped(self):
        self.assertEqual(P.split_windows(["You"], [5], 255), [])
        self.assertEqual(P.split_windows(["You", "Ascent"], [200, 100], 255), [])
        # an oversized first pair is skipped, a later pair still fits
        self.assertEqual(self.check(["You", "Ascent", "You", "Ascent"], [200, 100, 10, 10], 255), [(2, 4)])


class TestValSplit(unittest.TestCase):
    def test_deterministic_and_fraction(self):
        keys = [f"5:train:{i}" for i in range(20000)]
        a = [P.is_val(k, 1, 0.01) for k in keys]
        self.assertEqual(a, [P.is_val(k, 1, 0.01) for k in keys])
        self.assertAlmostEqual(sum(a) / len(a), 0.01, delta=0.004)
        self.assertNotEqual(a, [P.is_val(k, 2, 0.01) for k in keys])


class TestInitFrom(unittest.TestCase):
    CFG = GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=32, block_size=16)

    def make_ckpt(self, d, cfg=None):
        cfg = cfg or self.CFG
        torch.manual_seed(0)
        m = GPT(cfg)
        path = os.path.join(d, "ckpt.pt")
        torch.save({"model": m.state_dict(), "optimizer": {}, "step": 123, "config": {"model": asdict(cfg)}}, path)
        return m, path

    def test_loads_weights_and_returns_step(self):
        with tempfile.TemporaryDirectory() as d:
            src, path = self.make_ckpt(d)
            torch.manual_seed(5)
            dst = GPT(self.CFG)
            self.assertEqual(load_init_weights(dst, path, self.CFG), 123)
            for (k, a), b in zip(src.state_dict().items(), dst.state_dict().values()):
                self.assertTrue(torch.equal(a, b), k)

    def test_dropout_may_differ(self):
        with tempfile.TemporaryDirectory() as d:
            _, path = self.make_ckpt(d)
            cfg = GPTConfig(**{**asdict(self.CFG), "dropout": 0.1})
            load_init_weights(GPT(cfg), path, cfg)

    def test_shape_mismatch_errors_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            _, path = self.make_ckpt(d)
            bad = GPTConfig(vocab_size=64, n_layer=3, n_head=2, n_embd=32, block_size=16)
            with self.assertRaisesRegex(ValueError, "n_layer"):
                load_init_weights(GPT(bad), path, bad)


if __name__ == "__main__":
    unittest.main()
