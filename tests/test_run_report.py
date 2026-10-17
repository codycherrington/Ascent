"""Unit tests for scripts/run_report.py (synthetic run folders, no GPU or data)."""
import json
import os
import sys
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import run_report as rr  # noqa: E402

SAMPLES = """step 20
sampled: temperature 0.8, top-k 40, seed 1234; greedy; max 120 new tokens

=== prompt 1 ===
Once upon a time
--- sampled ---
 there was a cat. It was happy.<eos>
--- greedy ---
the big the big the big the big the big the big
"""


def make_run(root, rid, n_train=30, evals=(10, 20), max_steps=30, final_eval=True, loss_fn=None,
             tok_fn=None, lr=6e-4, samples=True, trunc=False, mem=3000):
    d = os.path.join(root, rid)
    os.makedirs(d)
    cfg = {"args": {"batch_size": 16, "block_size": 256, "grad_accum": 1, "lr": lr, "dtype": "fp32",
                    "out_dir": root}, "model": {"n_layer": 2}, "git_commit": "abc1234567", "device": "mps",
           "params": 1000, "tokens_per_step": 4096, "max_steps": max_steps}
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    lines = []
    for i in range(1, n_train + 1):
        loss = loss_fn(i) if loss_fn else 6.0 - 0.05 * i
        tok = tok_fn(i) if tok_fn else 5000.0
        lines.append(json.dumps({"type": "train", "step": i, "loss": loss, "lr": lr, "grad_norm": 1.0,
                                 "tok_s": tok, "step_ms": 4096 / tok * 1000}))
        if i in evals or (final_eval and i == n_train):
            vl = 5.5 - 0.04 * i
            lines.append(json.dumps({"type": "eval", "step": i, "val_loss": vl, "val_ppl": 2.7 ** vl,
                                     "val_tokens": 100, "eval_s": 1.0, "mps_driver_mb": mem}))
    text = "\n".join(lines) + "\n"
    if trunc:
        text += '{"type": "train", "step": 99, "lo'
    with open(os.path.join(d, "metrics.jsonl"), "w") as fh:
        fh.write(text)
    if samples:
        with open(os.path.join(d, "samples_step0020.txt"), "w") as fh:
            fh.write(SAMPLES)
    return d


def flag_text(d):
    return " ".join(t for _, t in rr.summarize(rr.load_run(d))["flags"])


class TestRunReport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_complete_run_writes_outputs(self):
        d = make_run(self.root, "20261017-100000")
        path = rr.write_report(d)
        self.assertTrue(os.path.isfile(path))
        md = open(os.path.join(d, "report.md")).read()
        self.assertLess(len(md.splitlines()), 150)
        self.assertIn("complete", md)
        self.assertIn("best", md)
        page = open(path).read()
        for needle in ("data:image/png;base64", "Observations", "Manual scoring", "<details>"):
            self.assertIn(needle, page)
        self.assertNotIn("http://", page.replace("http://www.w3.org", ""))
        idx = open(os.path.join(self.root, "index.html")).read()
        self.assertIn("20261017-100000/report.html", idx)

    def test_interrupted_and_incomplete_status(self):
        d = make_run(self.root, "20261017-100001", n_train=20, max_steps=30)   # final eval at 20, stopped early
        self.assertEqual(rr.summarize(rr.load_run(d))["status"], "interrupted")
        d2 = make_run(self.root, "20261017-100002", n_train=25, evals=(10, 20), final_eval=False, max_steps=30)
        self.assertEqual(rr.summarize(rr.load_run(d2))["status"], "incomplete")
        rr.write_report(d2)

    def test_truncated_last_line(self):
        d = make_run(self.root, "20261017-100003", trunc=True)
        run = rr.load_run(d)
        self.assertEqual(run["bad_lines"], 1)
        self.assertEqual(len(run["train"]), 30)
        rr.write_report(d)

    def test_empty_and_missing_files(self):
        d = os.path.join(self.root, "20261017-100004")
        os.makedirs(d)
        open(os.path.join(d, "metrics.jsonl"), "w").close()
        rr.write_report(d)                       # no config, no metrics
        self.assertIn("no train records", open(os.path.join(d, "report.md")).read())
        d2 = os.path.join(self.root, "20261017-100005")
        os.makedirs(d2)
        with open(os.path.join(d2, "metrics.jsonl"), "w") as fh:
            fh.write('garbage\n[1,2]\n{"type":"train","step":"x"}\n{"type":"train","step":1,"loss":null}\n')
        rr.write_report(d2)
        self.assertTrue(os.path.isfile(os.path.join(d2, "report.html")))

    def test_comparison(self):
        make_run(self.root, "20261017-100010", lr=6e-4)
        d = make_run(self.root, "20261017-100011", lr=3e-4)
        rr.write_report(d)
        md = open(os.path.join(d, "report.md")).read()
        self.assertIn("Vs previous run 20261017-100010", md)
        self.assertIn("args.lr", md)
        cmp_ = rr.compare_runs(rr.load_run(d), rr.load_run(os.path.join(self.root, "20261017-100010")))
        self.assertEqual([k for k, _, _ in cmp_["diff"]], ["args.lr"])
        self.assertAlmostEqual(cmp_["d_best_val"], 0.0)

    def test_flag_loss_spike_and_nonfinite(self):
        d = make_run(self.root, "20261017-100020", loss_fn=lambda i: 9.0 if i == 15 else 3.0)
        self.assertIn("Loss spike", flag_text(d))
        d2 = make_run(self.root, "20261017-100021", loss_fn=lambda i: float("nan") if i == 5 else 3.0)
        self.assertIn("Non-finite", flag_text(d2))

    def test_flag_throughput_drop(self):
        d = make_run(self.root, "20261017-100030", n_train=40, max_steps=40, tok_fn=lambda i: 5000 if i < 20 else 3000)
        self.assertIn("thermal", flag_text(d))
        s = rr.summarize(rr.load_run(d))
        self.assertGreater(s["tput_drop"], 10)
        d2 = make_run(self.root, "20261017-100031", n_train=40, max_steps=40)
        self.assertNotIn("thermal", flag_text(d2))

    def test_flag_val_trend_and_memory(self):
        d = make_run(self.root, "20261017-100040")
        self.assertIn("undertrained", flag_text(d))
        d2 = make_run(self.root, "20261017-100041", mem=7000)
        self.assertIn("swap risk", flag_text(d2))

    def test_flag_grad_spike(self):
        d = make_run(self.root, "20261017-100050")
        with open(os.path.join(d, "metrics.jsonl")) as fh:
            lines = fh.read().splitlines()
        rec = json.loads(lines[12])
        rec["grad_norm"] = 50.0
        lines[12] = json.dumps(rec)
        with open(os.path.join(d, "metrics.jsonl"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        self.assertIn("Grad-norm spike", flag_text(d))

    def test_text_stats(self):
        sf = rr.parse_samples("samples_step0020.txt", SAMPLES)
        self.assertEqual(sf["step"], 20)
        self.assertEqual(len(sf["items"]), 1)
        self.assertEqual(sf["stats"]["sampled"]["eos_frac"], 1.0)
        self.assertEqual(sf["stats"]["greedy"]["loops"], 1)
        self.assertLess(sf["stats"]["greedy"]["distinct1"], 0.5)


if __name__ == "__main__":
    unittest.main()
