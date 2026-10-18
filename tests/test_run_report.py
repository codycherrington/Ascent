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
             tok_fn=None, lr=6e-4, samples=True, trunc=False, mem=3000, extra_fn=None, ckpt=False):
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
                                 "tok_s": tok, "step_ms": 4096 / tok * 1000, **(extra_fn(i) if extra_fn else {})}))
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
    if ckpt:
        open(os.path.join(d, "ckpt.pt"), "w").close()
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
        for needle in ("data:image/png;base64", "MODEL QUALITY", "RUN HEALTH", "Proposed fixes", "What went well", "Manual scoring", "<details>"):
            self.assertIn(needle, page)
        self.assertNotIn("<details open", page)
        self.assertIn("<details><summary>Evals", page)
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
        self.assertIn("Vs previous comparable run 20261017-100010", md)
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


def vd_of(d):
    return rr.summarize(rr.load_run(d))["verdicts"]


def fixes_of(d):
    run = rr.load_run(d)
    return " ".join(rr.findings(run, rr.summarize(run))["fixes"]).lower()


class TestVerdicts(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.n = 0

    def tearDown(self):
        self._tmp.cleanup()

    def run_(self, **kw):
        self.n += 1
        return make_run(self.root, f"20261017-2000{self.n:02d}", **kw)

    def test_rate_boundaries(self):
        lim = {"good": 2.0, "medium": 2.6}
        self.assertEqual([rr.rate_low(x, lim) for x in (2.0, 2.01, 2.6, 2.61, None)],
                         ["GOOD", "MEDIUM", "MEDIUM", "FAIL", "N/A"])
        lim = {"good": 5000, "medium": 2500}
        self.assertEqual([rr.rate_high(x, lim) for x in (5000, 4999, 2500, 2499)], ["GOOD", "MEDIUM", "MEDIUM", "FAIL"])

    def test_learning_boundaries(self):
        def learning(val):
            s = {"best_val": val, "rising": False, "falling": False, "flags": [], "n_train": 0, "n_eval": 1,
                 "status": "complete", "samples_last": None, "review": None}
            return rr.verdicts({"dir": self.root, "sample_files": []}, s)["Learning"]["v"]
        self.assertEqual([learning(x) for x in (1.9, 2.0, 2.43, 2.6, 2.7, None)],
                         ["GOOD", "GOOD", "MEDIUM", "MEDIUM", "FAIL", "N/A"])

    def test_generalization(self):
        # val loss ends at 4.3, so the gap is 4.3 minus the constant train loss
        for loss, want in ((4.2, "GOOD"), (3.9, "MEDIUM"), (3.5, "FAIL")):
            d = self.run_(loss_fn=lambda i, l=loss: l)
            self.assertEqual(vd_of(d)["Generalization"]["v"], want, loss)

    def test_stability(self):
        self.assertEqual(vd_of(self.run_(loss_fn=lambda i: 3.0))["Stability"]["v"], "GOOD")
        self.assertEqual(vd_of(self.run_(loss_fn=lambda i: 9.0 if i == 15 else 3.0))["Stability"]["v"], "MEDIUM")
        self.assertEqual(vd_of(self.run_(loss_fn=lambda i: 9.0 if i in (12, 16, 20, 24) else 3.0))["Stability"]["v"], "FAIL")
        self.assertEqual(vd_of(self.run_(loss_fn=lambda i: float("nan") if i == 5 else 3.0))["Stability"]["v"], "FAIL")

    def test_speed_boundaries(self):
        def speed(tok, n=40):
            return vd_of(self.run_(n_train=n, max_steps=n, tok_fn=tok))["Speed & thermals"]["v"]
        self.assertEqual(speed(lambda i: 5000), "GOOD")
        self.assertEqual(speed(lambda i: 4000), "MEDIUM")                       # below 5000 but above 2500
        self.assertEqual(speed(lambda i: 5000 if i < 20 else 4000), "MEDIUM")   # 20% drop with a good median
        self.assertEqual(speed(lambda i: 2000), "MEDIUM")                       # flat, so the drop is under 30%
        self.assertEqual(speed(lambda i: 2000 if i < 20 else 1000), "FAIL")     # low and a 50% drop
        self.assertEqual(speed(lambda i: 5000, n=5), "GOOD")                    # too few records to judge the drop

    def test_optional_telemetry(self):
        base = vd_of(self.run_())["Speed & thermals"]
        self.assertEqual((base["v"], base["sub"]), ("GOOD", ""))
        self.assertNotIn("swap", fixes_of(self.run_()))
        grow = lambda i: {"swap_used_mb": 100 + i * 30, "temp_c": 60.0}  # noqa: E731  (+870 MB over the run)
        t = vd_of(self.run_(extra_fn=grow))["Speed & thermals"]
        self.assertEqual(t["v"], "MEDIUM")
        self.assertIn("swap", t["sub"])
        self.assertIn("close other apps", fixes_of(self.run_(extra_fn=grow)))
        self.assertEqual(vd_of(self.run_(extra_fn=lambda i: {"swap_used_mb": i * 100}))["Speed & thermals"]["v"], "FAIL")
        t = vd_of(self.run_(extra_fn=lambda i: {"temp_c": 90.0}))["Speed & thermals"]
        self.assertEqual(t["v"], "MEDIUM")
        self.assertIn("90 C", t["sub"])

    def test_sample_quality_and_review(self):
        d = self.run_()   # SAMPLES fixture: sampled text fine, the one greedy output loops (capped at MEDIUM)
        q = vd_of(d)["Sample quality"]
        self.assertEqual(q["v"], "MEDIUM")
        self.assertIn("not scored", q["sub"])
        self.assertEqual(vd_of(self.run_(samples=False))["Sample quality"]["v"], "N/A")
        with open(os.path.join(d, "review.json"), "w") as fh:
            json.dump({"step": 20, "scores": {"1": {"grammar": 5, "coherence": 4, "repetition": 4, "on-prompt": 5},
                                              "2": {"grammar": 4, "coherence": 4, "repetition": 4, "on-prompt": 4}}}, fh)
        q = vd_of(d)["Sample quality"]
        self.assertEqual(q["v"], "GOOD")          # mean 4.25 overrides the automatic MEDIUM
        self.assertIn("4.2", q["key"])
        for mean, want in ((3.0, "MEDIUM"), (2.9, "FAIL")):
            with open(os.path.join(d, "review.json"), "w") as fh:
                json.dump({"scores": {"1": {"grammar": mean}}}, fh)
            self.assertEqual(vd_of(d)["Sample quality"]["v"], want)
        with open(os.path.join(d, "review.json"), "w") as fh:
            fh.write("not json")
        self.assertIn("not scored", vd_of(d)["Sample quality"]["sub"])

    def test_completeness(self):
        self.assertEqual(vd_of(self.run_(ckpt=True))["Completeness"]["v"], "GOOD")
        self.assertEqual(vd_of(self.run_())["Completeness"]["v"], "MEDIUM")                       # no checkpoint
        self.assertEqual(vd_of(self.run_(ckpt=True, n_train=20, max_steps=30))["Completeness"]["v"], "MEDIUM")  # interrupted
        self.assertEqual(vd_of(self.run_(ckpt=True, n_train=25, final_eval=False, max_steps=30))["Completeness"]["v"], "FAIL")

    def test_group_badges(self):
        def badges(**kw):
            vd = {c: {"v": kw.get(c.split()[0], "GOOD")} for c in rr.CATEGORIES}
            return (rr.group_verdict(vd, rr.GROUPS["Model quality"]), rr.group_verdict(vd, rr.GROUPS["Run health"]))
        self.assertEqual(badges(), ("GOOD", "GOOD"))
        self.assertEqual(badges(Speed="FAIL"), ("GOOD", "FAIL"))            # model GOOD + run FAIL
        self.assertEqual(badges(Learning="FAIL"), ("FAIL", "GOOD"))
        self.assertEqual(badges(Generalization="MEDIUM", Stability="MEDIUM"), ("MEDIUM", "MEDIUM"))
        self.assertEqual(badges(Sample="MEDIUM", Completeness="FAIL"), ("MEDIUM", "FAIL"))
        self.assertEqual(badges(Sample="N/A"), ("GOOD", "GOOD"))            # N/A ignored
        na = {c: {"v": "N/A"} for c in rr.CATEGORIES}
        self.assertEqual(rr.group_verdict(na, rr.GROUPS["Run health"]), "N/A")

    def test_group_badges_in_outputs(self):
        d = self.run_(n_train=40, max_steps=40, ckpt=True, tok_fn=lambda i: 2000 if i < 20 else 1000)
        rr.write_report(d)
        md = open(os.path.join(d, "report.md")).read()
        self.assertIn("**RUN HEALTH: FAIL**", md)
        self.assertIn("**MODEL QUALITY:", md)
        page = open(os.path.join(d, "report.html")).read()
        self.assertIn("MODEL QUALITY:", page)
        self.assertIn("RUN HEALTH: FAIL", page)
        idx = open(os.path.join(self.root, "index.html")).read()
        self.assertIn("<th>model quality</th><th>run health</th>", idx)

    def test_findings_limits_and_throttle_fix(self):
        d = self.run_(n_train=40, max_steps=40, tok_fn=lambda i: 3000 if i < 20 else 1000)
        run = rr.load_run(d)
        s = rr.summarize(run)
        fd = rr.findings(run, s)
        for k in ("good", "bad", "fixes"):
            self.assertLessEqual(len(fd[k]), 5)
        self.assertIn("cool", fd["fixes"][0].lower())
        self.assertIn("train.py", fd["next_run"])
        self.assertIn("Speed & thermals", "\n".join(rr.render_md(run, s, None).splitlines()[:12]))

    def test_comparable_run_selection(self):
        make_run(self.root, "20261017-300000", n_train=30, max_steps=30)     # comparable
        make_run(self.root, "20261017-300001", n_train=10, max_steps=10)     # smoke run
        d = make_run(self.root, "20261017-300002", n_train=30, max_steps=30)
        run = rr.load_run(d)
        self.assertEqual(rr.pick_comparable(self.root, run, rr.summarize(run))["id"], "20261017-300000")  # skips the smoke run
        rr.write_report(d)
        self.assertIn("Vs previous comparable run 20261017-300000", open(os.path.join(d, "report.md")).read())
        only = tempfile.mkdtemp(dir=self.root)
        make_run(only, "20261017-400000", n_train=10, max_steps=10)
        d2 = make_run(only, "20261017-400001", n_train=30, max_steps=30)
        rr.write_report(d2)
        self.assertIn("no comparable previous run", open(os.path.join(d2, "report.md")).read())
        self.assertIn("No comparable previous run", open(os.path.join(d2, "report.html")).read())

    def test_index_has_verdict_column(self):
        make_run(self.root, "20261017-500000")
        rr.write_index(self.root)
        idx = open(os.path.join(self.root, "index.html")).read()
        self.assertIn("<th>run health</th>", idx)
        self.assertRegex(idx, r"v-(good|medium|fail)")


if __name__ == "__main__":
    unittest.main()
