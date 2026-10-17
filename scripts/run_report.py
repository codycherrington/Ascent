"""Per-run training report for Ascent.

    .venv/bin/python scripts/run_report.py runs/<id>     # report.html + report.md in that folder, then the index
    .venv/bin/python scripts/run_report.py --index       # only rebuild runs/index.html

Reads config.json, metrics.jsonl and samples_stepNNNN.txt from a run folder. Tolerates incomplete
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
    if s["status"] != "complete":
        flags.append(("warn", f"Run status is {s['status']}: stopped at step {s['last_step']} of {s['max_steps']:.0f}."
                      if s["max_steps"] and s["last_step"] is not None else f"Run status is {s['status']}."))
    return s


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
    older = [r for r in list_runs(runs_dir) if r < run["id"]]
    if older:
        try:
            cmp_ = compare_runs(run, load_run(os.path.join(runs_dir, older[-1])))
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
    L = []
    L.append(f"# Run {run['id']} - {s['status']}")
    L.append(f"- git `{(s['git'] or 'n/a')[:10]}` | device {s['device'] or 'n/a'} {s['dtype'] or ''} | "
             f"params {f(s['params'] / 1e6 if s['params'] else None, 2)}M | wall ~{dur(s['wall_s'])}")
    L.append("- config: " + ", ".join(f"{k}={v}" for k, v in highlights(run).items()))
    L.append(f"- tokens trained {f(s['tokens'] / 1e6 if s['tokens'] else None, 2)}M "
             f"(epochs over data {f(s['epochs'], 2)}), steps {s['first_step']}..{s['last_step']} of {f(s['max_steps'], 0)}")
    L.append("")
    L.append("## Key numbers")
    L.append(f"- train loss: first {f(s['train_first'])} | min {f(s['train_min'])} | final {f(s['train_final'])} | last-5 mean {f(s['train_tail'])}")
    L.append(f"- val loss: best {f(s['best_val'])} (ppl {f(s['best_ppl'], 1)}) at step {s['best_step']} | final {f(s['final_val'])} (ppl {f(s['final_ppl'], 1)})")
    L.append(f"- val minus train gap: {f(s['gap'], 2)}")
    L.append(f"- throughput: median {f(s['avg_tok_s'], 0)} tok/s | early->late drop {f(s['tput_drop'], 1) + '%' if s['tput_drop'] is not None else 'n/a'} | MPS driver peak {f(s['mps_mb'], 0)} MB")
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
    if cmp_:
        L.append("")
        L.append(f"## Vs previous run {cmp_['prev']}")
        L.append(f"- best val delta {signed(cmp_['d_best_val'])} (prev {f(cmp_['prev_summary']['best_val'])}) | "
                 f"tok/s delta {signed(cmp_['d_tok_s'], 0)} | params delta {signed(cmp_['d_params'], 0)}")
        for k, old, new in cmp_["diff"][:20]:
            L.append(f"- {k}: {old!r} -> {new!r}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- html

CSS = """
:root{--bg:#fff;--fg:#1f2328;--mute:#6b7280;--card:#f6f8fa;--line:#d0d7de;--warn:#b45309;--bad:#b91c1c;--info:#1d4ed8;--hl:#dcfce7}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mute:#9aa4b2;--card:#161b22;--line:#30363d;--warn:#fbbf24;--bad:#f87171;--info:#79b8ff;--hl:#0f3d24}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0 auto;padding:16px 20px;max-width:1100px}
h1{font-size:1.5rem;margin:.2em 0}h2{font-size:1.15rem;margin:1.6em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:4px}
.mute{color:var(--mute)}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}.card b{display:block;font-size:1.25rem}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}.charts img{width:100%;height:auto}
table{border-collapse:collapse;font-size:.9rem;width:100%}th,td{border:1px solid var(--line);padding:4px 8px;text-align:left}
th{background:var(--card)}tr.best td{background:var(--hl)}td.num{text-align:right;font-variant-numeric:tabular-nums}
.warn{color:var(--warn)}.bad{color:var(--bad)}.info{color:var(--info)}
pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;margin:4px 0}
details{margin:4px 0}summary{cursor:pointer}a{color:var(--info)}code{background:var(--card);padding:1px 4px;border-radius:4px}
.sc td{height:1.6em;min-width:3.5em}@media (max-width:600px){.charts{grid-template-columns:1fr}}
"""


def e(x):
    return html.escape(str(x))


def _table(head, rows, cls=None):
    h = "".join(f"<th>{e(c)}</th>" for c in head)
    body = ""
    for i, r in enumerate(rows):
        c = f' class="{cls[i]}"' if cls and cls[i] else ""
        body += f"<tr{c}>" + "".join(f"<td>{e(v)}</td>" for v in r) + "</tr>"
    return f"<table><tr>{h}</tr>{body}</table>"


def render_html(run, s, cmp_, charts, overlay):
    P = []
    P.append(f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Ascent run {e(run['id'])}</title><style>{CSS}</style></head><body>")
    P.append(f"<p class=mute><a href='../index.html'>all runs</a></p><h1>Run {e(run['id'])} <span class=mute>({e(s['status'])})</span></h1>")
    P.append(f"<p class=mute>git <code>{e((s['git'] or 'n/a')[:10])}</code> | {e(s['device'] or 'n/a')} {e(s['dtype'] or '')} | "
             f"{f(s['params'] / 1e6 if s['params'] else None, 2)}M params | wall ~{dur(s['wall_s'])} (estimated from step times)<br>"
             + e(", ".join(f"{k}={v}" for k, v in highlights(run).items())) + "</p>")
    cards = [("Tokens trained", f"{f(s['tokens'] / 1e6 if s['tokens'] else None, 2)}M"),
             ("Epochs over data", f(s["epochs"], 2)),
             ("Best val loss", f"{f(s['best_val'])} (ppl {f(s['best_ppl'], 1)})"), ("Best at step", str(s["best_step"])),
             ("Final val loss", f(s["final_val"])), ("Train loss first / min / final",
                                                     f"{f(s['train_first'], 2)} / {f(s['train_min'], 2)} / {f(s['train_final'], 2)}"),
             ("Val - train gap", f(s["gap"], 2)), ("Median tok/s", f(s["avg_tok_s"], 0)),
             ("Throughput drop", f"{f(s['tput_drop'], 1)}%" if s["tput_drop"] is not None else "n/a"), ("MPS driver peak", f"{f(s['mps_mb'], 0)} MB")]
    P.append("<h2>Key numbers</h2><div class=cards>" + "".join(f"<div class=card><span class=mute>{e(k)}</span><b>{e(v)}</b></div>" for k, v in cards) + "</div>")
    P.append("<h2>Charts</h2><div class=charts>" + "".join(
        f"<div><img alt='{e(t)}' src='data:image/png;base64,{b}'></div>" for t, b in charts) + "</div>")
    if not charts:
        P.append("<p class=mute>No chartable data.</p>")
    P.append("<h2>Observations</h2><ul>" + "".join(f"<li class={lvl}><b>{lvl.upper()}</b> {e(t)}</li>" for lvl, t in s["flags"])
             + "".join(f"<li class=mute>Note: {e(n)}</li>" for n in run["notes"]) + "</ul>")
    if run["eval"]:
        rows = [(r["step"], f(_fin(r.get("val_loss")), 4), f(_fin(r.get("val_ppl")), 1), f(_fin(r.get("eval_s")), 1),
                 f(_fin(r.get("mps_driver_mb")), 0)) for r in run["eval"]]
        cls = ["best" if r["step"] == s["best_step"] else "" for r in run["eval"]]
        P.append("<h2>Evals</h2>" + _table(["step", "val loss", "ppl", "eval s", "mps MB"], rows, cls))
    P.append("<h2>Samples</h2>")
    if not run["sample_files"]:
        P.append("<p class=mute>No samples files in this run.</p>")
    for sf in run["sample_files"]:
        P.append(f"<h3>Step {sf['step']} <span class=mute>({e(sf['file'])})</span></h3>")
        rows = []
        for mode in ("sampled", "greedy"):
            t = sf["stats"][mode]
            rows.append((mode, f"{f(t['distinct1'], 2)} / {f(t['distinct2'], 2)} / {f(t['distinct3'], 2)}", f(t["rep4"], 2),
                         f(t["mean_words"], 0), f(t["eos_frac"], 2), f"{t['loops']} / {t['n']}"))
        P.append(_table(["mode", "distinct-1/2/3", "repeated 4-gram frac", "mean words", "hit eos <120 tok", "looping outputs"], rows))
        P.append(f"<details><summary>Prompts and outputs ({len(sf['items'])})</summary>")
        for it in sf["items"]:
            P.append(f"<details><summary>Prompt {it['id']}: {e(it['prompt'][:90])}</summary><b>prompt</b><pre>{e(it['prompt'])}</pre>"
                     f"<b>sampled</b><pre>{e(it['sampled'])}</pre><b>greedy</b><pre>{e(it['greedy'])}</pre></details>")
        P.append("</details>")
        P.append("<details><summary>Manual scoring sheet (1-5, see eval/rubric.md)</summary><table class=sc><tr><th>prompt</th>"
                 + "".join(f"<th>{c}</th>" for c in CRITERIA) + "<th>notes</th></tr>"
                 + "".join(f"<tr><td>{i}</td>" + "<td></td>" * (len(CRITERIA) + 1) + "</tr>" for i in range(1, 21)) + "</table></details>")
    if cmp_:
        P.append(f"<h2>Compare with previous run</h2><p>Previous: <a href='../{e(cmp_['prev'])}/report.html'>{e(cmp_['prev'])}</a></p>")
        P.append(_table(["metric", "previous", "this run", "delta"], [
            ("best val loss", f(cmp_["prev_summary"]["best_val"]), f(s["best_val"]), signed(cmp_["d_best_val"])),
            ("median tok/s", f(cmp_["prev_summary"]["avg_tok_s"], 0), f(s["avg_tok_s"], 0), signed(cmp_["d_tok_s"], 0)),
            ("params", f(cmp_["prev_summary"]["params"], 0), f(s["params"], 0), signed(cmp_["d_params"], 0))]))
        if cmp_["diff"]:
            P.append("<p>Config differences:</p>" + _table(["key", "previous", "this run"], [(k, o, n) for k, o, n in cmp_["diff"]]))
        else:
            P.append("<p class=mute>No config differences.</p>")
    if overlay:
        P.append(f"<div class=charts><div><img alt='val loss overlay' src='data:image/png;base64,{overlay}'></div></div>")
    P.append(f"<p class=mute>Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/run_report.py from this folder; regenerate any time.</p></body></html>")
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
        rows.append([link, run_date(x["id"]), f(x.get("params") / 1e6 if x.get("params") else None, 2) + "M",
                     f(x.get("tokens") / 1e6 if x.get("tokens") else None, 2) + "M",
                     f(x.get("final_val")) + f" ({f(x.get('final_ppl'), 1)})", f(x.get("best_val")) + f" ({f(x.get('best_ppl'), 1)})",
                     f(x.get("avg_tok_s"), 0), dur(x.get("wall_s")), e((x.get("git") or "n/a")[:8]), e(x.get("status", ""))])
        cls.append("best" if best is not None and x.get("best_val") == best else "")
    head = ["run", "date", "params", "tokens", "final val (ppl)", "best val (ppl)", "tok/s", "wall", "git", "status"]
    body = "".join(f"<tr{' class=best' if c else ''}>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r, c in zip(rows, cls))
    doc = (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Ascent runs</title><style>{CSS}</style></head><body><h1>Ascent runs</h1>"
           f"<p class=mute>Newest first; best val loss highlighted. Regenerate: <code>.venv/bin/python scripts/run_report.py --index</code></p>"
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
