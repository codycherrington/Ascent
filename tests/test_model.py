"""Unit tests for model.py (CPU, tiny config)."""
import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig, generate  # noqa: E402

TINY = GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=32, block_size=16)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return GPT(TINY).eval()


class TestModel(unittest.TestCase):
    def test_forward_shape(self):
        m = tiny_model()
        x = torch.randint(0, 64, (3, 10))
        logits, loss = m(x)
        self.assertEqual(logits.shape, (3, 10, 64))
        self.assertIsNone(loss)

    def test_causal_masking(self):
        m = tiny_model()
        x = torch.randint(0, 64, (1, 12))
        x2 = x.clone()
        x2[0, 8] = (x[0, 8] + 1) % 64
        with torch.no_grad():
            a, _ = m(x)
            b, _ = m(x2)
        self.assertTrue(torch.allclose(a[:, :8], b[:, :8], atol=1e-6))
        self.assertFalse(torch.allclose(a[:, 8:], b[:, 8:]))

    def test_tied_weights(self):
        m = tiny_model()
        self.assertIs(m.head.weight, m.tok_emb.weight)

    def test_initial_loss_near_uniform(self):
        m = tiny_model()
        x = torch.randint(0, 64, (8, 16))
        y = torch.randint(0, 64, (8, 16))
        with torch.no_grad():
            _, loss = m(x, y)
        self.assertLess(abs(loss.item() - math.log(64)), 0.3)

    def test_generate_deterministic_with_seed(self):
        m = tiny_model()
        a = generate(m, [1, 2, 3], 10, temperature=0.8, top_k=5, seed=1234)
        b = generate(m, [1, 2, 3], 10, temperature=0.8, top_k=5, seed=1234)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)

    def test_top_k_one_matches_greedy(self):
        m = tiny_model()
        greedy = generate(m, [5, 6], 8, temperature=0.0)
        top1 = generate(m, [5, 6], 8, temperature=1.0, top_k=1, seed=7)
        self.assertEqual(greedy, top1)

    def test_generate_conditions_on_prompt(self):
        m = tiny_model()
        a, b = [1, 2, 3, 4], [40, 41, 42, 43]
        with torch.no_grad():
            la, _ = m(torch.tensor([a]))
            lb, _ = m(torch.tensor([b]))
        self.assertFalse(torch.allclose(la[0, -1], lb[0, -1]))
        # The first greedy token is the argmax of the forward pass on the full prompt.
        self.assertEqual(generate(m, a, 1, temperature=0.0)[0], int(la[0, -1].argmax()))
        self.assertEqual(generate(m, b, 1, temperature=0.0)[0], int(lb[0, -1].argmax()))

    def test_generate_beyond_block_size(self):
        m = tiny_model()
        out = generate(m, list(range(10)), 20, temperature=0.0)
        self.assertEqual(len(out), 20)

    def test_overfit_one_batch(self):
        m = tiny_model()
        m.train()
        x = torch.randint(0, 64, (4, 16))
        y = torch.randint(0, 64, (4, 16))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        _, first = m(x, y)
        for _ in range(60):
            _, loss = m(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.5 * first.item())


if __name__ == "__main__":
    unittest.main()
