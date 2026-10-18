"""Per-run training report for Ascent.

    .venv/bin/python scripts/run_report.py runs/<id>     # report.html + report.md in that folder, then the index
    .venv/bin/python scripts/run_report.py --index       # only rebuild runs/index.html

Opens with GOOD / MEDIUM / FAIL verdicts per category and two group badges (Model quality, Run health) (thresholds in THRESHOLDS below, rules and
no AI), then went-well / went-wrong / proposed-fixes, charts and collapsed details.
Reads config.json, metrics.jsonl, samples_stepNNNN.txt and optional review.json from a run folder. Tolerates incomplete
runs (crash, Ctrl-C, truncated last line, no evals, no samples); problems become notes, not errors.
Run folders are gitignored, so reports are always regenerable from them.
"""
import base64
import html
import io
import json
import math
import os
import re
import statistics
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

DATA_TOKENS = 10.69e6      # TinyStories train tokens, for "epochs over the data"
RAM_MB = 8 * 1024          # unified memory on the target machine
CRITERIA = ["grammar", "coherence", "repetition", "on-prompt"]  # eval/rubric.md
NGRAM_LOOP = 3             # a greedy 4-gram repeated this many times counts as a loop

# Every verdict threshold lives here (mirrored in docs/run-reports.md). PROVISIONAL: placeholders until
# the milestone thresholds in docs/goal.md are filled in after the baseline run.
# "good"/"medium" are the limits of GOOD and MEDIUM; anything beyond "medium" is FAIL.
THRESHOLDS = {
    "val_loss": {"good": 2.0, "medium": 2.6},            # best val loss, lower is better
    "val_rising_rel": 0.05,                                # final val above best by this much: GOOD capped at MEDIUM
    "falling_rel": 0.01,                                   # last eval interval improved by >= this: "still falling"
    "gap": {"good": 0.3, "medium": 0.6},                   # final val minus recent train loss, lower is better
    "loss_spikes": {"good": 0, "medium": 2},               # count of >1.5x spikes; more than medium is FAIL
    "grad_spike_frac": {"good": 0.02, "medium": 0.10},     # share of logged records above 3x median grad norm
    "tok_s": {"good": 5000, "medium": 2500},               # median tok/s, higher is better
    "tput_drop_pct": {"good": 10, "medium": 30},           # early-to-late drop, lower is better
    "swap_growth_mb": {"medium": 500, "fail": 2000},       # swap growth over the run (needs swap_used_mb)
    "temp_c": {"medium": 85},                              # peak temperature (needs temp_c)
    "distinct2": {"good": 0.85, "medium": 0.70},           # sampled distinct-2, higher is better
    "rep4": {"good": 0.05, "medium": 0.15},                # sampled repeated 4-gram share, lower is better
    "eos_frac": {"good": 0.30, "medium": 0.10},            # sampled outputs that ended with <eos>, higher is better
    "greedy_loop_frac": {"good": 0.20, "medium": 0.50},    # greedy outputs looping; capped at MEDIUM (expected early)
    "review": {"good": 4.0, "medium": 3.0},                # mean manual rubric score 1-5, higher is better
    "compare_min_steps_frac": 0.5,                         # previous run needs max_steps >= this x this run's
}
CATEGORIES = ["Learning", "Generalization", "Stability", "Speed & thermals", "Sample quality", "Completeness"]


# ---------------------------------------------------------------- loading

def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _fin(x):
    x = _num(x)
    return x if x is not None and math.isfinite(x) else None


def load_run(run_dir):
    """Return a dict of everything readable; never raises."""
    run_dir = os.path.abspath(run_dir)
    run = {"dir": run_dir, "id": os.path.basename(run_dir), "config": {}, "train": [], "eval": [],
           "samples_recs": [], "sample_files": [], "notes": [], "bad_lines": 0}
    try:
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        run["config"] = cfg if isinstance(cfg, dict) else {}
    except Exception as e:  # noqa: BLE001
        run["notes"].append(f"config.json unreadable ({type(e).__name__})")
    try:
        with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    run["bad_lines"] += 1
                    continue
                if not isinstance(rec, dict) or _fin(rec.get("step")) is None:
                    run["bad_lines"] += 1
                    continue
                kind = rec.get("type")
                key = {"train": "train", "eval": "eval", "samples": "samples_recs"}.get(kind)
                if key:
                    run[key].append(rec)
    except Exception as e:  # noqa: BLE001
        run["notes"].append(f"metrics.jsonl unreadable ({type(e).__name__})")
    if run["bad_lines"]:
        run["notes"].append(f"{run['bad_lines']} unparseable metrics line(s) skipped (likely truncated)")
    try:
        names = sorted(n for n in os.listdir(run_dir) if re.fullmatch(r"samples_step\d+\.txt", n))
        for n in names:
            with open(os.path.join(run_dir, n), encoding="utf-8", errors="replace") as f:
                run["sample_files"].append(parse_samples(n, f.read()))
    except Exception as e:  # noqa: BLE001
        run["notes"].append(f"samples unreadable ({type(e).__name__})")
    if not run["train"]:
        run["notes"].append("no train records in metrics.jsonl")
    if not run["eval"]:
        run["notes"].append("no eval records (no val loss)")
    if not run["sample_files"]:
        run["notes"].append("no samples files")
    return run


def parse_samples(name, text):
    m = re.search(r"^step (\d+)", text)
    step = int(m.group(1)) if m else None
    if step is None:
        m = re.search(r"step(\d+)", name)
        step = int(m.group(1)) if m else 0
    items = []
    pieces = re.split(r"^=== prompt (\d+) ===\n", text, flags=re.M)
    for i in range(1, len(pieces) - 1, 2):
        body = pieces[i + 1]
        sm = re.search(r"^--- sampled ---\n(.*?)^--- greedy ---\n(.*)$", body, flags=re.M | re.S)
        if sm:
            prompt = body[:sm.start()].strip("\n")
            items.append({"id": int(pieces[i]), "prompt": prompt, "sampled": sm.group(1).strip("\n"),
                          "greedy": sm.group(2).strip("\n")})
        else:
            items.append({"id": int(pieces[i]), "prompt": body.strip("\n"), "sampled": "", "greedy": ""})
    return {"file": name, "step": step, "items": items, "stats": text_stats(items)}


# ---------------------------------------------------------------- text stats

def _ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def _text_stats_for(texts):
    """texts: list of generated strings (may contain a trailing <eos>)."""
    d = {1: [], 2: [], 3: []}
    rep4, lengths, eos, loops = [], [], 0, 0
    for t in texts:
        hit = "<eos>" in t
        eos += hit
        words = t.replace("<eos>", " ").lower().split()
        lengths.append(len(words))
        for n in d:
            g = _ngrams(words, n)
            if g:
                d[n].append(len(set(g)) / len(g))
        g4 = _ngrams(words, 4)
        if g4:
            counts = {}
            for g in g4:
                counts[g] = counts.get(g, 0) + 1
            rep4.append(sum(c for c in counts.values() if c > 1) / len(g4))
            loops += max(counts.values()) >= NGRAM_LOOP
    mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
    return {"n": len(texts), "distinct1": mean(d[1]), "distinct2": mean(d[2]), "distinct3": mean(d[3]),
            "rep4": mean(rep4), "mean_words": mean(lengths),
            "eos_frac": eos / len(texts) if texts else None, "loops": loops}


def text_stats(items):
    return {"sampled": _text_stats_for([i["sampled"] for i in items]),
            "greedy": _text_stats_for([i["greedy"] for i in items])}


# ---------------------------------------------------------------- analysis

def _median(v):
    return statistics.median(v) if v else None


def _rolling(v, w):
    out = []
    for i in range(len(v)):
        seg = v[max(0, i - w + 1):i + 1]
        out.append(sum(seg) / len(seg))
    return out


def summarize(run):
    """Key numbers, derived values and automatic flags. Never raises on odd records."""
    cfg = run["config"]
    args = cfg.get("args") if isinstance(cfg.get("args"), dict) else {}
    tr = [r for r in run["train"] if _num(r.get("step")) is not None]
    ev = [r for r in run["eval"] if _fin(r.get("val_loss")) is not None]
    tps = _fin(cfg.get("tokens_per_step")) or (
        (_fin(args.get("batch_size")) or 0) * (_fin(args.get("block_size")) or 0) * (_fin(args.get("grad_accum")) or 1)) or None
    s = {"id": run["id"], "params": _fin(cfg.get("params")), "git": cfg.get("git_commit"),
         "device": cfg.get("device"), "dtype": args.get("dtype"), "tokens_per_step": tps,
         "max_steps": _fin(cfg.get("max_steps")), "n_train": len(tr), "n_eval": len(ev), "flags": []}
    flags = s["flags"]

    steps = [r["step"] for r in tr]
    losses = [_num(r.get("loss")) for r in tr]
    fin_losses = [x for x in losses if x is not None and math.isfinite(x)]
    s["last_step"] = max(steps + [r["step"] for r in run["eval"]], default=None)
    s["first_step"] = min(steps, default=None)
    s["train_first"] = fin_losses[0] if fin_losses else None
    s["train_min"] = min(fin_losses) if fin_losses else None
    s["train_final"] = fin_losses[-1] if fin_losses else None
    tail = fin_losses[-5:]
    s["train_tail"] = sum(tail) / len(tail) if tail else None

    best = min(ev, key=lambda r: r["val_loss"]) if ev else None
    s["best_val"] = best["val_loss"] if best else None
    s["best_step"] = best["step"] if best else None
    s["best_ppl"] = math.exp(best["val_loss"]) if best and best["val_loss"] < 50 else None
    s["final_val"] = ev[-1]["val_loss"] if ev else None
    s["final_ppl"] = math.exp(s["final_val"]) if ev and s["final_val"] < 50 else None
    s["gap"] = (s["final_val"] - s["train_tail"]) if ev and s["train_tail"] is not None else None

    # tokens / wall time. A resumed run's first train record is start_step + 1.
    if tps and s["last_step"] is not None and s["first_step"] is not None:
        s["tokens"] = (s["last_step"] - (s["first_step"] - 1)) * tps
        s["epochs"] = s["tokens"] / DATA_TOKENS
    else:
        s["tokens"] = s["epochs"] = None
    wall, prev = 0.0, None
    for r in tr:
        ms = _fin(r.get("step_ms"))
        if ms is not None:
            wall += ms * (r["step"] - (prev if prev is not None else r["step"] - 1)) / 1000
        prev = r["step"]
    wall += sum(_fin(r.get("eval_s")) or 0 for r in run["eval"])
    wall += sum(_fin(r.get("samples_s")) or 0 for r in run["samples_recs"])
    s["wall_s"] = wall if tr else None
    rates = [_fin(r.get("tok_s")) for r in tr[1:]]
    rates = [x for x in rates if x]
    s["avg_tok_s"] = _median(rates) if rates else (_fin(tr[0].get("tok_s")) if tr else None)

    # optional telemetry (absent in older runs)
    swap = [x for x in (_fin(r.get("swap_used_mb")) for r in tr) if x is not None]
    s["swap_max"] = max(swap) if swap else None
    s["swap_growth"] = (max(swap) - swap[0]) if swap else None
    temps = [x for x in (_fin(r.get("temp_c")) for r in tr) if x is not None]
    s["temp_max"] = max(temps) if temps else None

    # status
    last_eval_step = run["eval"][-1]["step"] if run["eval"] else None
    if not tr:
        s["status"] = "incomplete"
    elif s["max_steps"] and s["last_step"] >= s["max_steps"] and last_eval_step == s["last_step"]:
        s["status"] = "complete"
    elif last_eval_step is not None and last_eval_step >= max(steps):
        s["status"] = "interrupted"   # stopped early but the final eval ran (Ctrl-C or non-finite stop)
    else:
        s["status"] = "incomplete"    # no final eval: crashed or killed

    # ---- flags (level, text)
    if any(x is not None and not math.isfinite(x) for x in losses):
        flags.append(("bad", "Non-finite train loss logged (NaN/inf); training diverged."))
    spikes = []
    for i, x in enumerate(losses):
        if i >= 5 and x is not None and math.isfinite(x):
            base = _median([y for y in losses[i - 5:i] if y is not None and math.isfinite(y)])
            if base and x > 1.5 * base:
                spikes.append(steps[i])
    s["loss_spikes"] = spikes
    if spikes:
        flags.append(("warn", f"Loss spike(s) (>1.5x median of previous 5 logs) at step(s): {', '.join(map(str, spikes[:8]))}."))
    gn = [_fin(r.get("grad_norm")) for r in tr]
    gmed = _median([g for g in gn[1:] if g is not None])
    gspikes = [steps[i] for i, g in enumerate(gn) if i > 0 and g is not None and gmed and g > 3 * gmed]
    s["grad_median"], s["grad_spikes"] = gmed, gspikes
    if gspikes:
        flags.append(("warn", f"Grad-norm spike(s) >3x median ({gmed:.2f}) at step(s): {', '.join(map(str, gspikes[:8]))}."))

    # throughput: median of first 10% vs last 10% of train records (skip record 0: startup)
    s["tput_drop"] = None
    s["rising"] = s["falling"] = False
    body = [(_fin(r.get("tok_s"))) for r in tr[1:]]
    body = [x for x in body if x]
    if len(body) >= 10:
        k = max(1, len(body) // 10)
        early, late = _median(body[:k]), _median(body[-k:])
        s["tput_early"], s["tput_late"] = early, late
        s["tput_drop"] = 100 * (early - late) / early
        if s["tput_drop"] > 10:
            flags.append(("warn", f"Throughput fell {s['tput_drop']:.0f}% from the first 10% of steps "
                                  f"({early:.0f} to {late:.0f} tok/s): likely thermal throttling."))
    else:
        flags.append(("info", "Too few train records (<11) to judge throughput drop."))

    if len(ev) >= 2:
        a, b = ev[-2]["val_loss"], ev[-1]["val_loss"]
        rel = (a - b) / a if a else 0
        s["last_improve"] = rel
        s["rising"] = b > s["best_val"] * (1 + THRESHOLDS["val_rising_rel"]) and b > a
        s["falling"] = rel >= THRESHOLDS["falling_rel"]
        if b > s["best_val"] * 1.01 and b > a:
            flags.append(("warn", f"Val loss rising at the end ({a:.3f} to {b:.3f}; best {s['best_val']:.3f} at step "
                                  f"{s['best_step']}): possible overfitting."))
        elif rel < 0.01:
            flags.append(("info", f"Plateau: val loss improved {100 * rel:.2f}% over the last eval interval (<1%)."))
        else:
            flags.append(("info", f"Val loss still decreasing at the end ({100 * rel:.1f}% over the last interval): "
                                  "likely undertrained; more tokens should help."))
    elif len(ev) == 1:
        flags.append(("info", "Only one eval; cannot judge trend."))
    if s["gap"] is not None:
        if s["gap"] > 0.5:
            flags.append(("warn", f"Large val-minus-train gap {s['gap']:.2f}: overfitting / memorising."))
        elif s["gap"] < -0.2:
            flags.append(("info", f"Val loss is {-s['gap']:.2f} below recent train loss (train is a noisy minibatch mean)."))
        else:
            flags.append(("info", f"Val-minus-train gap {s['gap']:.2f} (small)."))
    mem = [_fin(r.get("mps_driver_mb")) for r in ev]
    mem = [m for m in mem if m]
    s["mps_mb"] = max(mem) if mem else None
    if s["mps_mb"] and s["mps_mb"] > 0.75 * RAM_MB:
        flags.append(("warn", f"MPS driver memory peaked at {s['mps_mb']:.0f} MB ({100 * s['mps_mb'] / RAM_MB:.0f}% of 8 GB): swap risk."))
    if s["swap_growth"] is not None and s["swap_growth"] >= THRESHOLDS["swap_growth_mb"]["medium"]:
        flags.append(("warn", f"Swap grew by {s['swap_growth']:.0f} MB during the run (peak {s['swap_max']:.0f} MB)."))
    if s["temp_max"] is not None and s["temp_max"] >= THRESHOLDS["temp_c"]["medium"]:
        flags.append(("warn", f"Temperature reached {s['temp_max']:.0f} C."))
    if s["status"] != "complete":
        flags.append(("warn", f"Run status is {s['status']}: stopped at step {s['last_step']} of {s['max_steps']:.0f}."
                      if s["max_steps"] and s["last_step"] is not None else f"Run status is {s['status']}."))
    s["samples_last"] = run["sample_files"][-1] if run["sample_files"] else None
    s["review"] = load_review(run["dir"])
    s["verdicts"] = verdicts(run, s)
    s["model"] = group_verdict(s["verdicts"], GROUPS["Model quality"])
    s["health"] = group_verdict(s["verdicts"], GROUPS["Run health"])
    return s


# ---------------------------------------------------------------- verdicts

GOOD, MEDIUM, FAIL, NA = "GOOD", "MEDIUM", "FAIL", "N/A"
_ORDER = {GOOD: 0, MEDIUM: 1, FAIL: 2}


def _worst(*vs):
    vs = [v for v in vs if v in _ORDER]
    return max(vs, key=_ORDER.get) if vs else NA


def rate_low(x, lim):
    """Lower is better: GOOD <= lim['good'], MEDIUM <= lim['medium'], else FAIL."""
    return NA if x is None else GOOD if x <= lim["good"] else MEDIUM if x <= lim["medium"] else FAIL


def rate_high(x, lim):
    """Higher is better: GOOD >= lim['good'], MEDIUM >= lim['medium'], else FAIL."""
    return NA if x is None else GOOD if x >= lim["good"] else MEDIUM if x >= lim["medium"] else FAIL


def load_review(run_dir):
    """Optional manual review: review.json = {"step": N, "scores": {"1": {"grammar": 4, ...}, ...}}.
    Returns {"mean", "n", "step"} over all numeric 1-5 scores, or None when absent/unusable."""
    try:
        with open(os.path.join(run_dir, "review.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        vals = [float(v) for sc in data["scores"].values() if isinstance(sc, dict) for v in sc.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 1 <= v <= 5]
        step = data.get("step") if isinstance(data.get("step"), (int, float)) else None
        return {"mean": sum(vals) / len(vals), "n": len(vals), "step": step} if vals else None
    except Exception:  # noqa: BLE001
        return None


def sample_quality(s):
    """Automatic sample-quality rating from the latest samples file's text stats.
    Returns (verdict, key, per-metric detail dict)."""
    sf = s.get("samples_last")
    if not sf:
        return NA, "no samples", {}
    T = THRESHOLDS
    sm, gr = sf["stats"]["sampled"], sf["stats"]["greedy"]
    gl = gr["loops"] / gr["n"] if gr["n"] else None
    parts = {"distinct2": (sm["distinct2"], rate_high(sm["distinct2"], T["distinct2"])),
             "rep4": (sm["rep4"], rate_low(sm["rep4"], T["rep4"])),
             "eos_frac": (sm["eos_frac"], rate_high(sm["eos_frac"], T["eos_frac"])),
             "greedy_loop_frac": (gl, rate_low(gl, T["greedy_loop_frac"]))}
    if parts["greedy_loop_frac"][1] == FAIL:      # greedy loops are expected early: never worse than MEDIUM
        parts["greedy_loop_frac"] = (gl, MEDIUM)
    v = _worst(*(p[1] for p in parts.values()))
    key = f"distinct-2 {f(sm['distinct2'], 2)}" if sm["distinct2"] is not None else "no stats"
    return v, key, parts


def verdicts(run, s):
    """{category: {"v": GOOD|MEDIUM|FAIL|N/A, "key": the one key number, "sub": optional small line}}."""
    T = THRESHOLDS
    out = {}

    b = s.get("best_val")
    v = rate_low(b, T["val_loss"])
    if s.get("rising") and v == GOOD:
        v = MEDIUM
    trend = ", rising" if s.get("rising") else ", still falling" if s.get("falling") else ""
    out["Learning"] = {"v": v, "key": f"val loss {f(b, 2)}{trend}" if b is not None else "no val loss"}

    g = s.get("gap")
    out["Generalization"] = {"v": rate_low(g, T["gap"]), "key": f"gap {signed(g, 2)}" if g is not None else "no gap"}

    nf = any("Non-finite" in t for _, t in s["flags"])
    n_tr = max(1, s["n_train"])
    ls, gs = len(s.get("loss_spikes", [])), len(s.get("grad_spikes", []))
    v = FAIL if nf else _worst(rate_low(ls, T["loss_spikes"]), rate_low(gs / n_tr, T["grad_spike_frac"]))
    out["Stability"] = {"v": v if s["n_train"] else NA,
                        "key": "non-finite loss" if nf else f"{ls} loss / {gs} grad spikes"}

    tps, drop = s.get("avg_tok_s"), s.get("tput_drop")
    if tps is None:
        v = NA
    else:
        if tps >= T["tok_s"]["good"] and (drop is None or drop < T["tput_drop_pct"]["good"]):
            v = GOOD
        elif tps >= T["tok_s"]["medium"] or (drop is not None and drop < T["tput_drop_pct"]["medium"]):
            v = MEDIUM
        else:
            v = FAIL
        sg, tm = s.get("swap_growth"), s.get("temp_max")
        if sg is not None:
            v = _worst(v, FAIL if sg >= T["swap_growth_mb"]["fail"] else MEDIUM if sg >= T["swap_growth_mb"]["medium"] else GOOD)
        if tm is not None and tm >= T["temp_c"]["medium"]:
            v = _worst(v, MEDIUM)
    sub = []
    if s.get("swap_growth") is not None:
        sub.append(f"swap {s['swap_growth']:+.0f} MB")
    if s.get("temp_max") is not None:
        sub.append(f"peak {s['temp_max']:.0f} C")
    out["Speed & thermals"] = {"v": v, "sub": ", ".join(sub),
                               "key": "no throughput data" if tps is None else
                               f"{tps:,.0f} tok/s" + (f", {round(-drop):+d}%" if drop is not None and round(drop) else "")}

    v, key, parts = sample_quality(s)
    rv = s.get("review")
    if rv:
        v, key = rate_high(rv["mean"], T["review"]), f"review {rv['mean']:.1f}/5"
        sub = "manual review overrides the automatic rating"
    else:
        gl = parts.get("greedy_loop_frac", (None, GOOD))
        sf = s.get("samples_last")
        loops = f"greedy loops {sf['stats']['greedy']['loops']}/{sf['stats']['greedy']['n']}; " if sf and gl[1] != GOOD else ""
        sub = loops + "manual review: not scored"
    out["Sample quality"] = {"v": v, "key": key, "sub": sub, "parts": parts}

    d = run["dir"]
    checks = {"finished": s["status"] == "complete", "final eval": s["n_eval"] > 0 and s["status"] != "incomplete",
              "checkpoints": any(os.path.isfile(os.path.join(d, n)) for n in ("ckpt.pt", "ckpt_best.pt")),
              "samples": bool(run["sample_files"])}
    missing = [k for k, ok in checks.items() if not ok]
    v = GOOD if not missing else FAIL if (s["status"] == "incomplete" or not checks["final eval"]) else MEDIUM
    out["Completeness"] = {"v": v, "key": f"{4 - len(missing)}/4 checks", "sub": ("missing: " + ", ".join(missing)) if missing else ""}
    return out


GROUPS = {"Model quality": ["Learning", "Generalization", "Sample quality"],
          "Run health": ["Stability", "Speed & thermals", "Completeness"]}


def group_verdict(vd, cats):
    """Worst of the categories' verdicts; N/A tiles are ignored, all N/A gives N/A."""
    return _worst(*(vd[c]["v"] for c in cats))


def group_line(vd, cats):
    bad = sorted((c for c in cats if vd[c]["v"] in (MEDIUM, FAIL)), key=lambda c: -_ORDER[vd[c]["v"]])
    if not bad:
        return "All GOOD." if group_verdict(vd, cats) == GOOD else "No data."
    return "; ".join(f"{c} {vd[c]['v']} ({vd[c]['key']})" for c in bad) + "."


def findings(run, s):
    """Rule-based (no AI) lists of (importance, text) for good / bad / fixes, plus a next-run command.
    Each list is returned sorted by importance and cut to 5."""
    vd, T = s["verdicts"], THRESHOLDS
    args = (run["config"].get("args") or {}) if isinstance(run["config"].get("args"), dict) else {}
    good, bad, fixes = [], [], []
    flags = {"cool": False}

    def fix(w, text):
        if text not in [t for _, t in fixes]:
            fixes.append((w, text))

    sev = {FAIL: 3, MEDIUM: 2, GOOD: 0, NA: 0}
    L, G, S, P, C, Q = (vd[c] for c in ("Learning", "Generalization", "Stability", "Speed & thermals", "Completeness", "Sample quality"))
    n = lambda c: sev[vd[c]["v"]]  # noqa: E731

    # ---- bad + fixes
    if any("Non-finite" in t for _, t in s["flags"]):
        bad.append((10, "Loss went NaN/inf: training diverged."))
        fix(10, "Lower --lr, keep --dtype fp32 and check the data; resume from the last good checkpoint.")
    if C["v"] != GOOD and s["status"] != "complete":
        bad.append((9, f"Run {s['status']} at step {s['last_step']}" + (f" of {s['max_steps']:.0f}." if s["max_steps"] else ".")))
        fix(9, f"Continue with --resume runs/{run['id']}/ckpt.pt (if the checkpoint exists).")
    elif C["v"] != GOOD:
        bad.append((4, f"Run output incomplete ({C['sub']})."))
    if L["v"] in (MEDIUM, FAIL):
        bad.append((6 + n("Learning"), f"Best val loss {f(s['best_val'], 2)} is above the {T['val_loss']['good']:.1f} target (ppl {f(s['best_ppl'], 1)})."))
    if s.get("rising"):
        bad.append((6, f"Val loss rose at the end (best {f(s['best_val'], 2)} at step {s['best_step']}, final {f(s['final_val'], 2)})."))
    if s.get("falling") and L["v"] != GOOD:
        ep = s.get("epochs") or 0
        extra = " Data is only ~10.7M tokens, so beyond ~1 epoch more data is needed." if ep >= 0.8 else ""
        fix(6.5, f"Val loss still falling: train more tokens (now {f(s['tokens'] / 1e6 if s['tokens'] else None, 1)}M, "
                 f"{f(ep, 2)} epochs).{extra}")
    if P["v"] in (MEDIUM, FAIL):
        d = s.get("tput_drop")
        txt = f"Throughput {s['avg_tok_s']:,.0f} tok/s median" + (f", fell {d:.0f}% over the run (likely thermal throttling)." if d is not None and d > T["tput_drop_pct"]["good"] else ".")
        bad.append((5 + n("Speed & thermals"), txt))
        if d is not None and d > T["tput_drop_pct"]["good"]:
            flags["cool"] = True
            fix(7, "Let the Mac cool: use a stand or cooler room, close background apps, add cooldown pauses or run in bursts with --resume.")
        else:
            fix(5, "Throughput is low from the start: close background apps and check scripts/benchmark_throughput.py settings.")
    if s.get("swap_growth") is not None and s["swap_growth"] >= T["swap_growth_mb"]["medium"]:
        bad.append((6, f"Swap grew {s['swap_growth']:.0f} MB (peak {s['swap_max']:.0f} MB): memory pressure."))
        fix(7, "Close other apps; lower --batch-size and raise --grad-accum to keep the effective batch.")
    elif s.get("mps_mb") and s["mps_mb"] > 0.75 * RAM_MB:
        bad.append((4, f"MPS driver memory peaked at {s['mps_mb']:.0f} MB: swap risk."))
        fix(4, "Close other apps; lower --batch-size and raise --grad-accum.")
    if s.get("temp_max") is not None and s["temp_max"] >= T["temp_c"]["medium"]:
        bad.append((5, f"Temperature peaked at {s['temp_max']:.0f} C."))
        flags["cool"] = True
        fix(6, "Cool the machine (stand, cooler room, cooldown pauses) before the next long run.")
    if G["v"] in (MEDIUM, FAIL):
        bad.append((4 + n("Generalization"), f"Train/val gap {signed(s['gap'], 2)}: starting to memorise."))
        fix(5, "Add data or dropout (--dropout 0.1).")
    if S["v"] in (MEDIUM, FAIL) and not any("Non-finite" in t for _, t in s["flags"]):
        bad.append((5 + n("Stability"), f"Instability: {S['key']}."))
        fix(5.5, "Lower --lr or lengthen --warmup-steps to calm the spikes.")
    parts = Q.get("parts") or {}
    if parts:
        d2, r4, gl = parts["distinct2"], parts["rep4"], parts["greedy_loop_frac"]
        sm_bad = d2[1] in (MEDIUM, FAIL) or r4[1] in (MEDIUM, FAIL)
        if sm_bad:
            bad.append((4 + max(sev[d2[1]], sev[r4[1]]), f"Sampled text repetitive (distinct-2 {f(d2[0], 2)}, repeated 4-grams {f(r4[0], 2)})."))
            fix(4, "Low variety: review temperature/top-k (T=0.8, k=40) and train longer.")
        if gl[1] in (MEDIUM, FAIL) and gl[0] is not None:
            sf = s["samples_last"]["stats"]["greedy"]
            bad.append((3, f"Greedy outputs loop in {sf['loops']}/{sf['n']} prompts."))
            if not sm_bad:
                fix(3, "Greedy loops are expected early: check the sampled outputs and train longer.")
        if parts["eos_frac"][1] in (MEDIUM, FAIL):
            bad.append((2, f"Few stories end with <eos> within the limit ({f(parts['eos_frac'][0], 2)})."))
    if not s.get("review"):
        fix(1, "Score ~10 samples by hand (eval/rubric.md) and save review.json in the run folder.")

    # ---- good
    if L["v"] == GOOD:
        good.append((8, f"Val loss {f(s['best_val'], 2)} meets the {T['val_loss']['good']:.1f} target."))
    elif s.get("train_first") is not None and s.get("best_val") is not None and L["v"] != NA:
        good.append((7, f"Learning works: train loss {f(s['train_first'], 1)} to {f(s['train_final'], 2)}, val loss {f(s['best_val'], 2)}."))
    if s.get("falling"):
        good.append((5, "Val loss still falling: more training should help."))
    if S["v"] == GOOD:
        good.append((6, "Stable: no NaNs or loss spikes."))
    if G["v"] == GOOD:
        good.append((5, f"No overfitting: train/val gap {signed(s['gap'], 2)}."))
    if P["v"] == GOOD:
        good.append((4, f"Fast and steady: {P['key']}."))
    if Q["v"] == GOOD:
        good.append((6, f"Sample text looks varied ({Q['key']}); no loops."))
    elif parts and parts["distinct2"][1] == GOOD:
        good.append((3, f"Sampled text is varied (distinct-2 {f(parts['distinct2'][0], 2)})."))
    if C["v"] == GOOD:
        good.append((4, "Finished with final eval, checkpoints and samples."))
    if s.get("review"):
        good.append((2, f"Manual review {s['review']['mean']:.1f}/5."))

    top = lambda xs: [t for _, t in sorted(xs, key=lambda x: -x[0])[:5]]  # noqa: E731
    return {"good": top(good), "bad": top(bad), "fixes": top(fixes), "next_run": next_run(run, s, args, flags["cool"])}


def next_run(run, s, args, cool):
    """A suggested command line as text; never executed."""
    flags_ = []
    tokens = int(args.get("max_tokens") or 0)
    if s["status"] in ("incomplete", "interrupted"):
        return f".venv/bin/python train.py --resume runs/{run['id']}/ckpt.pt"
    if s.get("falling") and tokens:
        flags_.append(f"--max-tokens {tokens * 2 // 1_000_000 * 1_000_000}")
    if s["verdicts"]["Generalization"]["v"] in (MEDIUM, FAIL):
        flags_.append("--dropout 0.1")
    if s["verdicts"]["Stability"]["v"] in (MEDIUM, FAIL) and args.get("lr"):
        flags_ += [f"--lr {args['lr'] / 2:g}", f"--warmup-steps {int((args.get('warmup_steps') or 200) * 2)}"]
    bs, ga = args.get("batch_size"), args.get("grad_accum") or 1
    mem = (s.get("swap_growth") or 0) >= THRESHOLDS["swap_growth_mb"]["medium"]
    if mem and bs and bs > 1:
        flags_ += [f"--batch-size {bs // 2}", f"--grad-accum {ga * 2}"]
    cmd = ".venv/bin/python train.py" + ("".join(" " + x for x in flags_))
    if cool:
        cmd += "   # first let the Mac cool; consider bursts with --resume"
    return cmd


def pick_comparable(runs_dir, run, s):
    """Most recent earlier run whose max_steps >= compare_min_steps_frac x this run's; None if there is none."""
    mine = s.get("max_steps") or s.get("last_step") or 0
    for rid in reversed([r for r in list_runs(runs_dir) if r < run["id"]]):
        prev = load_run(os.path.join(runs_dir, rid))
        ms = _fin(prev["config"].get("max_steps")) or 0
        if prev["train"] and ms >= THRESHOLDS["compare_min_steps_frac"] * mine:
            return prev
    return None


def _flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


def compare_runs(a_run, b_run):
    """Config diff and deltas of run a (current) against run b (previous)."""
    sa, sb = summarize(a_run), summarize(b_run)
    fa, fb = _flat(a_run["config"]), _flat(b_run["config"])
    diff = [(k, fb.get(k), fa.get(k)) for k in sorted(set(fa) | set(fb))
            if fa.get(k) != fb.get(k) and k not in ("args.out_dir",)]

    def d(key):
        x, y = sa.get(key), sb.get(key)
        return (x - y) if x is not None and y is not None else None
    return {"prev": b_run["id"], "prev_summary": sb, "diff": diff, "d_best_val": d("best_val"),
            "d_tok_s": d("avg_tok_s"), "d_params": d("params")}


def list_runs(runs_dir):
    try:
        return sorted(n for n in os.listdir(runs_dir)
                      if os.path.isfile(os.path.join(runs_dir, n, "metrics.jsonl"))
                      or os.path.isfile(os.path.join(runs_dir, n, "config.json")))
    except OSError:
        return []


# ---------------------------------------------------------------- formatting

def f(x, nd=3, none="n/a"):
    return none if x is None else f"{x:.{nd}f}"


def dur(sec):
    if sec is None:
        return "n/a"
    sec = int(sec)
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec % 60:02d}s" if m else f"{sec}s")


def run_date(run_id):
    m = re.match(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", run_id)
    return f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}" if m else run_id


def signed(x, nd=3):
    return "n/a" if x is None else f"{x:+.{nd}f}"


# ---------------------------------------------------------------- charts

BLUE, ORANGE, RED, GREEN, GRAY = "#3b82f6", "#f59e0b", "#ef4444", "#10b981", "#8b8f98"


def _fig(w=6.2, h=3.0):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ax.spines.values():
        sp.set_color(GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.grid(alpha=0.25, color=GRAY)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    ax.title.set_color(GRAY)
    return fig, ax


def _png(fig):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _legend(ax):
    leg = ax.legend(fontsize=8, frameon=False)
    if leg:
        for t in leg.get_texts():
            t.set_color(GRAY)


def make_charts(run, s):
    """Return list of (title, base64 png). Each chart is independent; failures are skipped."""
    tr = [r for r in run["train"] if _fin(r.get("step")) is not None]
    ev = [r for r in run["eval"] if _fin(r.get("val_loss")) is not None]
    out = []

    def add(title, fn):
        try:
            fig = fn()
            if fig is not None:
                out.append((title, _png(fig)))
        except Exception as e:  # noqa: BLE001
            run["notes"].append(f"chart '{title}' failed ({type(e).__name__})")

    def series(key):
        xs, ys = [], []
        for r in tr:
            v = _fin(r.get(key))
            if v is not None:
                xs.append(r["step"])
                ys.append(v)
        return xs, ys

    def loss_chart():
        xs, ys = series("loss")
        if not xs and not ev:
            return None
        fig, ax = _fig()
        if xs:
            ax.plot(xs, ys, color=BLUE, lw=1.3, label="train loss")
        if ev:
            ax.plot([r["step"] for r in ev], [r["val_loss"] for r in ev], color=ORANGE, lw=1.6,
                    marker="o", ms=3.5, label="val loss")
            ax.scatter([s["best_step"]], [s["best_val"]], color=GREEN, zorder=5, s=60, marker="*",
                       label=f"best val {s['best_val']:.3f} @ {s['best_step']}")
        pos = [y for y in ys + [r["val_loss"] for r in ev] if y > 0]
        if pos and max(pos) / min(pos) > 4:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Train and validation loss")
        _legend(ax)
        return fig

    def lr_chart():
        xs, ys = series("lr")
        if not xs:
            return None
        fig, ax = _fig(6.2, 2.4)
        ax.plot(xs, ys, color=BLUE)
        ax.set_xlabel("step")
        ax.set_ylabel("lr")
        ax.set_title("Learning rate schedule")
        return fig

    def gn_chart():
        xs, ys = series("grad_norm")
        if not xs:
            return None
        fig, ax = _fig(6.2, 2.4)
        ax.plot(xs, ys, color=BLUE, lw=1.2)
        if s.get("grad_median"):
            ax.axhline(3 * s["grad_median"], color=RED, ls="--", lw=1, label="3x median")
        sp = [(x, y) for x, y in zip(xs, ys) if x in set(s["grad_spikes"])]
        if sp:
            ax.scatter(*zip(*sp), color=RED, zorder=5, s=25, label="spike")
        if ys and max(ys) / max(min(ys), 1e-9) > 20:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("grad norm (pre-clip)")
        ax.set_title("Gradient norm")
        _legend(ax)
        return fig

    def tput_chart():
        xs, ys = series("tok_s")
        if len(xs) < 2:
            return None
        fig, ax = _fig()
        ax.plot(xs, ys, color=BLUE, lw=0.8, alpha=0.6, label="tok/s")
        ax.plot(xs, _rolling(ys, 5), color=ORANGE, lw=1.8, label="rolling mean (5)")
        if s.get("tput_drop") is not None:
            ax.axhline(s["tput_early"], color=GREEN, ls=":", lw=1, label=f"early median {s['tput_early']:.0f}")
            ax.axhline(s["tput_late"], color=RED, ls=":", lw=1,
                       label=f"late median {s['tput_late']:.0f} ({-s['tput_drop']:+.0f}%)")
        for r in run["eval"]:
            ax.axvline(r["step"], color=GRAY, alpha=0.3, lw=0.8)
        for r in run["samples_recs"]:
            ax.axvline(r["step"], color=GREEN, alpha=0.3, lw=0.8, ls="--")
        ax.set_xlabel("step (grey lines: evals, green dashes: sample pauses)")
        ax.set_ylabel("tokens / s")
        ax.set_title("Throughput (thermal-throttle signal)")
        _legend(ax)
        return fig

    def step_chart():
        xs, ys = series("step_ms")
        if not xs:
            return None
        fig, ax = _fig(6.2, 2.4)
        ax.plot(xs, ys, color=BLUE)
        if len(ys) > 3 and max(ys) / max(min(ys), 1e-9) > 20:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("ms / step")
        ax.set_title("Step time")
        return fig

    add("loss", loss_chart)
    add("lr", lr_chart)
    add("grad", gn_chart)
    add("tput", tput_chart)
    add("step", step_chart)
    return out


def overlay_chart(runs_dir, ids):
    """Val loss of up to 5 runs on one chart; returns base64 png or None."""
    try:
        fig, ax = _fig()
        cols = [BLUE, ORANGE, GREEN, RED, "#a855f7"]
        n = 0
        for i, rid in enumerate(ids):
            r = load_run(os.path.join(runs_dir, rid))
            ev = [e for e in r["eval"] if _fin(e.get("val_loss")) is not None]
            if ev:
                ax.plot([e["step"] for e in ev], [e["val_loss"] for e in ev], color=cols[i % 5], marker="o",
                        ms=3, lw=1.5, label=rid)
                n += 1
        if not n:
            import matplotlib.pyplot as plt
            plt.close(fig)
            return None
        ax.set_xlabel("step")
        ax.set_ylabel("val loss")
        ax.set_title("Val loss, recent runs")
        _legend(ax)
        return _png(fig)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- shared report data

def build(run, runs_dir):
    s = summarize(run)
    cmp_ = None
    try:
        prev = pick_comparable(runs_dir, run, s)
        if prev:
            cmp_ = compare_runs(run, prev)
    except Exception as e:  # noqa: BLE001
        run["notes"].append(f"comparison failed ({type(e).__name__})")
    return s, cmp_


def highlights(run):
    a = (run["config"].get("args") or {}) if isinstance(run["config"].get("args"), dict) else {}
    keys = ["n_layer", "n_head", "n_embd", "block_size", "batch_size", "grad_accum", "lr", "warmup_steps",
            "weight_decay", "grad_clip", "dropout", "max_tokens", "eval_interval", "seed", "resume"]
    return {k: a[k] for k in keys if k in a and a[k] is not None}


# ---------------------------------------------------------------- markdown

def render_md(run, s, cmp_):
    vd, fd = s["verdicts"], findings(run, s)
    L = []
    L.append(f"# Run {run['id']} - {s['status']}")
    L.append(f"**MODEL QUALITY: {s['model']}** - {group_line(vd, GROUPS['Model quality'])}")
    L.append(f"**RUN HEALTH: {s['health']}** - {group_line(vd, GROUPS['Run health'])}")
    L.append("## Scorecard")
    for g, cats in GROUPS.items():
        L.append(f"{g}:")
        for c in cats:
            L.append(f"- {vd[c]['v']} - {c} - {vd[c]['key']}" + (f" ({vd[c]['sub']})" if vd[c].get("sub") else ""))
    for title, key in (("Went well", "good"), ("Went wrong", "bad"), ("Proposed fixes", "fixes")):
        L.append(f"## {title}")
        L.extend(f"- {t}" for t in fd[key] or ["(nothing to report)"])
    L.append(f"Suggested next run: `{fd['next_run']}`")
    L.append("")
    L.append("## Details")
    L.append(f"- git `{(s['git'] or 'n/a')[:10]}` | device {s['device'] or 'n/a'} {s['dtype'] or ''} | "
             f"params {f(s['params'] / 1e6 if s['params'] else None, 2)}M | wall ~{dur(s['wall_s'])}")
    L.append("- config: " + ", ".join(f"{k}={v}" for k, v in highlights(run).items()))
    L.append(f"- tokens trained {f(s['tokens'] / 1e6 if s['tokens'] else None, 2)}M "
             f"(epochs over data {f(s['epochs'], 2)}), steps {s['first_step']}..{s['last_step']} of {f(s['max_steps'], 0)}")
    L.append(f"- train loss: first {f(s['train_first'])} | min {f(s['train_min'])} | final {f(s['train_final'])} | last-5 mean {f(s['train_tail'])}")
    L.append(f"- val loss: best {f(s['best_val'])} (ppl {f(s['best_ppl'], 1)}) at step {s['best_step']} | final {f(s['final_val'])} (ppl {f(s['final_ppl'], 1)})")
    L.append(f"- val minus train gap: {f(s['gap'], 2)}")
    L.append(f"- throughput: median {f(s['avg_tok_s'], 0)} tok/s | early->late drop {f(s['tput_drop'], 1) + '%' if s['tput_drop'] is not None else 'n/a'} | MPS driver peak {f(s['mps_mb'], 0)} MB"
             + (f" | swap growth {f(s['swap_growth'], 0)} MB" if s["swap_growth"] is not None else "")
             + (f" | peak temp {f(s['temp_max'], 0)} C" if s["temp_max"] is not None else ""))
    L.append("")
    L.append("## Flags")
    for lvl, t in s["flags"]:
        L.append(f"- [{lvl.upper()}] {t}")
    for n in run["notes"]:
        L.append(f"- [NOTE] {n}")
    ev = run["eval"]
    if ev:
        L.append("")
        L.append("## Evals")
        L.append("| step | val loss | ppl | eval s | mps MB |")
        L.append("|---|---|---|---|---|")
        rows = ev if len(ev) <= 14 else ev[:3] + ev[3:-8:max(1, (len(ev) - 11) // 3)] + ev[-8:]
        for r in rows:
            L.append(f"| {r['step']} | {f(_fin(r.get('val_loss')), 4)} | {f(_fin(r.get('val_ppl')), 1)} | "
                     f"{f(_fin(r.get('eval_s')), 1)} | {f(_fin(r.get('mps_driver_mb')), 0)} |")
    if run["sample_files"]:
        L.append("")
        L.append("## Sample text stats (20 prompts; sampled T=0.8 / greedy)")
        L.append("| step | mode | distinct1/2/3 | rep 4-gram | words | eos | loops |")
        L.append("|---|---|---|---|---|---|---|")
        for sf in run["sample_files"]:
            for mode in ("sampled", "greedy"):
                t = sf["stats"][mode]
                L.append(f"| {sf['step']} | {mode} | {f(t['distinct1'], 2)}/{f(t['distinct2'], 2)}/{f(t['distinct3'], 2)} | "
                         f"{f(t['rep4'], 2)} | {f(t['mean_words'], 0)} | {f(t['eos_frac'], 2)} | {t['loops']}/{t['n']} |")
        last = run["sample_files"][-1]
        L.append("")
        L.append(f"Excerpts from step {last['step']} (sampled, first 200 chars):")
        for it in last["items"][:3]:
            L.append(f"- p{it['id']} {it['prompt'][:60]!r} -> {it['sampled'][:200].strip()!r}")
    L.append("")
    if cmp_:
        L.append(f"## Vs previous comparable run {cmp_['prev']}")
        L.append(f"- best val delta {signed(cmp_['d_best_val'])} (prev {f(cmp_['prev_summary']['best_val'])}) | "
                 f"tok/s delta {signed(cmp_['d_tok_s'], 0)} | params delta {signed(cmp_['d_params'], 0)}")
        for k, old, new in cmp_["diff"][:20]:
            L.append(f"- {k}: {old!r} -> {new!r}")
    else:
        L.append("## Vs previous run")
        L.append("- no comparable previous run")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- html

CSS = """
:root{--bg:#fff;--fg:#1f2328;--mute:#6b7280;--card:#f6f8fa;--line:#d0d7de;--warn:#b45309;--bad:#b91c1c;--info:#1d4ed8;--hl:#dcfce7;
--good:#166534;--goodbg:#dcfce7;--med:#92400e;--medbg:#fef3c7;--fail:#991b1b;--failbg:#fee2e2;--na:#4b5563;--nabg:#e5e7eb}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mute:#9aa4b2;--card:#161b22;--line:#30363d;--warn:#fbbf24;--bad:#f87171;--info:#79b8ff;--hl:#0f3d24;
--good:#86efac;--goodbg:#0f3d24;--med:#fcd34d;--medbg:#46340b;--fail:#fca5a5;--failbg:#4c1515;--na:#cbd5e1;--nabg:#2a3039}}
body{overflow-wrap:break-word;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0 auto;padding:16px 20px;max-width:1100px}
h1{font-size:1.5rem;margin:.2em 0}h2{font-size:1.1rem;margin:1.4em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:4px}
.mute{color:var(--mute)}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}.card b{display:block;font-size:1.25rem}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));gap:10px}.charts img{width:100%;height:auto}
table{border-collapse:collapse;font-size:.9rem;width:100%}th,td{border:1px solid var(--line);padding:4px 8px;text-align:left}
th{background:var(--card)}tr.best td{background:var(--hl)}td.num{text-align:right;font-variant-numeric:tabular-nums}
.warn{color:var(--warn)}.bad{color:var(--bad)}.info{color:var(--info)}
pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;margin:4px 0}
details{margin:6px 0;border:1px solid var(--line);border-radius:8px;padding:6px 12px;background:var(--card)}
details details{background:var(--bg)}summary{cursor:pointer;font-weight:600}a{color:var(--info)}code{background:var(--card);padding:1px 4px;border-radius:4px}
.sc td{height:1.6em;min-width:3.5em}
.badge{display:flex;flex-direction:column;gap:2px;border-radius:12px;padding:14px 20px;margin:10px 0;border:2px solid currentColor}
.badge .w{font-size:1.6rem;font-weight:800;letter-spacing:.04em}.badge .t{font-size:1.05rem}
.groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(440px,100%),1fr));gap:16px;margin:10px 0}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin:8px 0}
.tile{border-radius:10px;padding:10px 12px;border:2px solid currentColor}.tile .w{display:block;font-weight:800;letter-spacing:.05em}
.tile .c{display:block;color:var(--fg);font-weight:600}.tile .k{display:block;color:var(--fg);font-size:1.05rem}.tile .s{display:block;color:var(--mute);font-size:.8rem}
.v-good{color:var(--good);background:var(--goodbg)}.v-medium{color:var(--med);background:var(--medbg)}
.v-fail{color:var(--fail);background:var(--failbg)}.v-na{color:var(--na);background:var(--nabg)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr));gap:12px;margin:12px 0}
.col{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 14px}.col h2{margin:.2em 0 .3em;border:0;font-size:1rem}
.col ul{margin:0;padding-left:1.1em}.col li{margin:.25em 0}
.next{margin:8px 0}.next code{display:block;white-space:pre-wrap;padding:6px 10px}
"""


def e(x):
    return html.escape(str(x))


def vclass(v):
    return "v-" + {GOOD: "good", MEDIUM: "medium", FAIL: "fail"}.get(v, "na")


def _table(head, rows, cls=None):
    h = "".join(f"<th>{e(c)}</th>" for c in head)
    body = ""
    for i, r in enumerate(rows):
        c = f' class="{cls[i]}"' if cls and cls[i] else ""
        body += f"<tr{c}>" + "".join(f"<td>{e(v)}</td>" for v in r) + "</tr>"
    return f"<table><tr>{h}</tr>{body}</table>"


def render_html(run, s, cmp_, charts, overlay):
    vd, fd = s["verdicts"], findings(run, s)
    P = []
    P.append(f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Ascent run {e(run['id'])}</title><style>{CSS}</style></head><body>")
    # (a) title
    P.append(f"<p class=mute><a href='../index.html'>all runs</a></p><h1>Run {e(run['id'])} <span class=mute>({e(s['status'])})</span></h1>")
    # (b)+(c) two badges, each with its three tiles
    def tile(c):
        return (f"<div class='tile {vclass(vd[c]['v'])}'><span class=w>{e(vd[c]['v'])}</span><span class=c>{e(c)}</span>"
                f"<span class=k>{e(vd[c]['key'])}</span>" + (f"<span class=s>{e(vd[c]['sub'])}</span>" if vd[c].get("sub") else "")
                + "</div>")
    groups = ""
    for g, cats in GROUPS.items():
        gv = s["model"] if g == "Model quality" else s["health"]
        groups += (f"<section class=group><div class='badge {vclass(gv)}'><span class=w>{e(g.upper())}: {e(gv)}</span>"
                   f"<span class=t>{e(group_line(vd, cats))}</span></div>"
                   f"<div class=tiles>{''.join(tile(c) for c in cats)}</div></section>")
    P.append(f"<div class=groups>{groups}</div>")
    # (d) good / bad / fixes
    P.append("<div class=cols>" + "".join(
        f"<div class=col><h2>{t}</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in (fd[k] or ["(nothing to report)"])) + "</ul></div>"
        for t, k in (("What went well", "good"), ("What went wrong", "bad"), ("Proposed fixes", "fixes"))) + "</div>")
    P.append(f"<div class=next><b>Suggested next run</b> (text only, nothing is executed)<code>{e(fd['next_run'])}</code></div>")
    # (e) charts
    if charts:
        P.append("<div class=charts>" + "".join(f"<div><img alt='{e(t)}' src='data:image/png;base64,{b}'></div>" for t, b in charts) + "</div>")
    else:
        P.append("<p class=mute>No chartable data.</p>")
    # (f) collapsed details
    cards = [("Tokens trained", f"{f(s['tokens'] / 1e6 if s['tokens'] else None, 2)}M"),
             ("Epochs over data", f(s["epochs"], 2)),
             ("Best val loss", f"{f(s['best_val'])} (ppl {f(s['best_ppl'], 1)})"), ("Best at step", str(s["best_step"])),
             ("Final val loss", f(s["final_val"])), ("Train loss first / min / final",
                                                     f"{f(s['train_first'], 2)} / {f(s['train_min'], 2)} / {f(s['train_final'], 2)}"),
             ("Val - train gap", f(s["gap"], 2)), ("Median tok/s", f(s["avg_tok_s"], 0)),
             ("Throughput drop", f"{f(s['tput_drop'], 1)}%" if s["tput_drop"] is not None else "n/a"), ("MPS driver peak", f"{f(s['mps_mb'], 0)} MB")]
    if s["swap_growth"] is not None:
        cards.append(("Swap growth", f"{s['swap_growth']:+.0f} MB (peak {s['swap_max']:.0f})"))
    if s["temp_max"] is not None:
        cards.append(("Peak temperature", f"{s['temp_max']:.0f} C"))
    P.append("<details><summary>Key numbers</summary><div class=cards>"
             + "".join(f"<div class=card><span class=mute>{e(k)}</span><b>{e(v)}</b></div>" for k, v in cards) + "</div>"
             f"<p class=mute>git <code>{e((s['git'] or 'n/a')[:10])}</code> | {e(s['device'] or 'n/a')} {e(s['dtype'] or '')} | "
             f"{f(s['params'] / 1e6 if s['params'] else None, 2)}M params | wall ~{dur(s['wall_s'])} (estimated from step times)</p></details>")
    P.append("<details><summary>Automatic flags and notes</summary><ul>" + "".join(f"<li class={lvl}><b>{lvl.upper()}</b> {e(t)}</li>" for lvl, t in s["flags"])
             + "".join(f"<li class=mute>Note: {e(n)}</li>" for n in run["notes"]) + "</ul></details>")
    if run["eval"]:
        rows = [(r["step"], f(_fin(r.get("val_loss")), 4), f(_fin(r.get("val_ppl")), 1), f(_fin(r.get("eval_s")), 1),
                 f(_fin(r.get("mps_driver_mb")), 0)) for r in run["eval"]]
        cls = ["best" if r["step"] == s["best_step"] else "" for r in run["eval"]]
        P.append("<details><summary>Evals</summary>" + _table(["step", "val loss", "ppl", "eval s", "mps MB"], rows, cls) + "</details>")
    if not run["sample_files"]:
        P.append("<details><summary>Samples</summary><p class=mute>No samples files in this run.</p></details>")
    for sf in run["sample_files"]:
        rows = []
        for mode in ("sampled", "greedy"):
            t = sf["stats"][mode]
            rows.append((mode, f"{f(t['distinct1'], 2)} / {f(t['distinct2'], 2)} / {f(t['distinct3'], 2)}", f(t["rep4"], 2),
                         f(t["mean_words"], 0), f(t["eos_frac"], 2), f"{t['loops']} / {t['n']}"))
        body = _table(["mode", "distinct-1/2/3", "repeated 4-gram frac", "mean words", "hit eos <120 tok", "looping outputs"], rows)
        body += f"<details><summary>Prompts and outputs ({len(sf['items'])})</summary>"
        for it in sf["items"]:
            body += (f"<details><summary>Prompt {it['id']}: {e(it['prompt'][:90])}</summary><b>prompt</b><pre>{e(it['prompt'])}</pre>"
                     f"<b>sampled</b><pre>{e(it['sampled'])}</pre><b>greedy</b><pre>{e(it['greedy'])}</pre></details>")
        body += "</details>"
        body += ("<details><summary>Manual scoring sheet (1-5, see eval/rubric.md; save as review.json to drive the verdict)</summary>"
                 "<table class=sc><tr><th>prompt</th>" + "".join(f"<th>{c}</th>" for c in CRITERIA) + "<th>notes</th></tr>"
                 + "".join(f"<tr><td>{i}</td>" + "<td></td>" * (len(CRITERIA) + 1) + "</tr>" for i in range(1, 21)) + "</table></details>")
        P.append(f"<details><summary>Samples, step {sf['step']} ({e(sf['file'])})</summary>{body}</details>")
    P.append("<details><summary>Compare with previous run</summary>")
    if cmp_:
        P.append(f"<p>Previous comparable run: <a href='../{e(cmp_['prev'])}/report.html'>{e(cmp_['prev'])}</a></p>")
        P.append(_table(["metric", "previous", "this run", "delta"], [
            ("best val loss", f(cmp_["prev_summary"]["best_val"]), f(s["best_val"]), signed(cmp_["d_best_val"])),
            ("median tok/s", f(cmp_["prev_summary"]["avg_tok_s"], 0), f(s["avg_tok_s"], 0), signed(cmp_["d_tok_s"], 0)),
            ("params", f(cmp_["prev_summary"]["params"], 0), f(s["params"], 0), signed(cmp_["d_params"], 0))]))
        if cmp_["diff"]:
            P.append("<p>Config differences:</p>" + _table(["key", "previous", "this run"], [(k, o, n) for k, o, n in cmp_["diff"]]))
        else:
            P.append("<p class=mute>No config differences.</p>")
    else:
        P.append("<p class=mute>No comparable previous run.</p>")
    if overlay:
        P.append(f"<div class=charts><div><img alt='val loss overlay' src='data:image/png;base64,{overlay}'></div></div>")
    P.append("</details>")
    P.append("<details><summary>Config</summary>" + _table(["key", "value"], [(k, v) for k, v in sorted(_flat(run["config"]).items())]) + "</details>")
    P.append(f"<p class=mute>Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/run_report.py from this folder; regenerate any time. "
             "Verdict thresholds are provisional (see docs/run-reports.md).</p></body></html>")
    return "".join(P)


# ---------------------------------------------------------------- index

def write_index(runs_dir):
    rows, sums = [], []
    for rid in list_runs(runs_dir):
        try:
            sums.append(summarize(load_run(os.path.join(runs_dir, rid))))
        except Exception:  # noqa: BLE001
            sums.append({"id": rid, "status": "unreadable"})
    sums.sort(key=lambda x: x["id"], reverse=True)
    bests = [x["best_val"] for x in sums if x.get("best_val") is not None]
    best = min(bests) if bests else None
    cls = []
    for x in sums:
        link = f"<a href='{e(x['id'])}/report.html'>{e(x['id'])}</a>" if os.path.isfile(os.path.join(runs_dir, x["id"], "report.html")) else e(x["id"])
        pill = lambda v: f"<b class='{vclass(v)}' style='padding:1px 6px;border-radius:4px'>{e(v)}</b>"  # noqa: E731
        rows.append([pill(x.get("model", NA)), pill(x.get("health", NA)), link, run_date(x["id"]),
                     f(x.get("params") / 1e6 if x.get("params") else None, 2) + "M",
                     f(x.get("tokens") / 1e6 if x.get("tokens") else None, 2) + "M",
                     f(x.get("final_val")) + f" ({f(x.get('final_ppl'), 1)})", f(x.get("best_val")) + f" ({f(x.get('best_ppl'), 1)})",
                     f(x.get("avg_tok_s"), 0), dur(x.get("wall_s")), e((x.get("git") or "n/a")[:8]), e(x.get("status", ""))])
        cls.append("best" if best is not None and x.get("best_val") == best else "")
    head = ["model quality", "run health", "run", "date", "params", "tokens", "final val (ppl)", "best val (ppl)", "tok/s", "wall", "git", "status"]
    body = "".join(f"<tr{' class=best' if c else ''}>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r, c in zip(rows, cls))
    doc = (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Ascent runs</title><style>{CSS}</style></head><body><h1>Ascent runs</h1>"
           f"<p class=mute>Newest first; best val loss highlighted; model quality and run health are each GOOD / MEDIUM / FAIL. "
           f"Regenerate: <code>.venv/bin/python scripts/run_report.py --index</code></p>"
           f"<table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{body}</table></body></html>")
    path = os.path.join(runs_dir, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


# ---------------------------------------------------------------- entry points

def write_report(run_dir, index=True):
    """Write report.html and report.md into run_dir (and rebuild runs/index.html). Returns the html path."""
    run_dir = os.path.abspath(run_dir)
    runs_dir = os.path.dirname(run_dir)
    run = load_run(run_dir)
    s, cmp_ = build(run, runs_dir)
    charts = make_charts(run, s)
    older = [r for r in list_runs(runs_dir) if r <= run["id"]][-5:]
    overlay = overlay_chart(runs_dir, older) if len(older) > 1 else None
    md = render_md(run, s, cmp_)          # after charts so chart failures show up in notes
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    path = os.path.join(run_dir, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(run, s, cmp_, charts, overlay))
    if index:
        write_index(runs_dir)
    return path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--index":
        print(write_index(argv[1] if len(argv) > 1 else "runs"))
        return 0
    status = 0
    for d in argv:
        try:
            print(write_report(d))
        except Exception as ex:  # noqa: BLE001
            print(f"report failed for {d}: {type(ex).__name__}: {ex}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
