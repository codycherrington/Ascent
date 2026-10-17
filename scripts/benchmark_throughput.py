"""Throughput benchmark for the Ascent GPT on MPS.

Sweeps batch size x block size x dtype, timing full train steps
(forward + backward + AdamW step) on random tokens. No files are written.

    .venv/bin/python scripts/benchmark_throughput.py
    .venv/bin/python scripts/benchmark_throughput.py --soak-minutes 10 --batch-size 16 --block-size 256 --dtype bf16
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig  # noqa: E402

DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def mem_mb(device):
    if device != "mps":
        return 0.0, 0.0
    return torch.mps.current_allocated_memory() / 2**20, torch.mps.driver_allocated_memory() / 2**20


def make_step(device, batch_size, block_size, dtype_name):
    """Build a fresh model/optimizer and return a function running one train step."""
    model = GPT(GPTConfig(block_size=block_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    dtype = DTYPES[dtype_name]
    vocab = model.cfg.vocab_size

    def step():
        x = torch.randint(0, vocab, (batch_size, block_size), device=device)
        y = torch.randint(0, vocab, (batch_size, block_size), device=device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.detach()

    return model, opt, step


def therm():
    try:
        return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(pmset failed: {e})"


def soak(device, batch_size, block_size, dtype_name, minutes, window_s=30):
    print(f"soak: batch {batch_size}, block {block_size}, {dtype_name}, {minutes} min")
    print("pmset -g therm (start):\n" + therm())
    model, opt, step = make_step(device, batch_size, block_size, dtype_name)
    for _ in range(3):
        step()
    sync(device)
    total_s = minutes * 60
    start = time.perf_counter()
    windows, mid_done = [], False
    while time.perf_counter() - start < total_s:
        w0, n = time.perf_counter(), 0
        while time.perf_counter() - w0 < window_s:
            step()
            n += 1
            if n % 5 == 0:
                sync(device)
        sync(device)
        dt = time.perf_counter() - w0
        tps = n * batch_size * block_size / dt
        windows.append(tps)
        print(f"  window {len(windows)} @ {time.perf_counter() - start:6.0f}s: {tps:8.0f} tok/s")
        if not mid_done and time.perf_counter() - start >= total_s / 2:
            print("pmset -g therm (middle):\n" + therm())
            mid_done = True
    print("pmset -g therm (end):\n" + therm())
    drop = 100 * (windows[0] - windows[-1]) / windows[0]
    print(f"soak summary: first window {windows[0]:.0f} tok/s, last {windows[-1]:.0f} tok/s, "
          f"drop {drop:.1f}%, mean {sum(windows) / len(windows):.0f} tok/s over {len(windows)} windows")


STEP_LIMIT_S = 30.0


def run_setting_in_child(bs, bl, dt, warmup, steps, max_seconds):
    """Run one setting in a subprocess so a hang (e.g. swapping) can be killed.

    Returns (result dict or None, status string).
    """
    cmd = [sys.executable, os.path.abspath(__file__), "--child", f"{bs},{bl},{dt}",
           "--warmup", str(warmup), "--steps", str(steps), "--max-seconds", str(max_seconds)]
    timeout = max_seconds + 4 * STEP_LIMIT_S
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"aborted (no result after {timeout:.0f}s)"
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            return (r, "ok") if "error" not in r else (None, r["error"])
    return None, f"error: {(out.stderr.strip().splitlines() or ['no output'])[-1][:80]}"


def child(spec, warmup, steps, max_seconds):
    """Benchmark one setting and print a single RESULT line (used by the sweep)."""
    bs, bl, dt = spec.split(",")
    bs, bl = int(bs), int(bl)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    try:
        model, opt, step = make_step(device, bs, bl, dt)
        for _ in range(warmup):
            t = time.perf_counter()
            step()
            sync(device)
            if time.perf_counter() - t > STEP_LIMIT_S:
                raise TimeoutError(f"aborted (warmup step > {STEP_LIMIT_S:.0f}s)")
        t0, n = time.perf_counter(), 0
        while n < steps and time.perf_counter() - t0 < max_seconds:
            t = time.perf_counter()
            loss = step()
            sync(device)
            n += 1
            if time.perf_counter() - t > STEP_LIMIT_S:
                raise TimeoutError(f"aborted (step > {STEP_LIMIT_S:.0f}s)")
        dur = time.perf_counter() - t0
        loss = float(loss)
        cur, drv = mem_mb(device)
        r = {"tok_s": bs * bl * n / dur, "step_ms": 1000 * dur / n, "steps": n, "loss": loss,
             "finite": math.isfinite(loss), "cur_mb": cur, "drv_mb": drv}
    except TimeoutError as e:
        r = {"error": str(e)}
    except RuntimeError as e:
        r = {"error": "OOM" if "out of memory" in str(e).lower() else f"error: {str(e)[:60]}"}
    print("RESULT " + json.dumps(r), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-sizes", default="8,16,32")
    p.add_argument("--block-sizes", default="128,256")
    p.add_argument("--dtypes", default="fp32,fp16", help="any of fp32,fp16,bf16 (bf16 was slowest on M2)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--max-seconds", type=float, default=15.0, help="timed-section cap per setting")
    p.add_argument("--soak-minutes", type=float, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--dtype", default=None, choices=list(DTYPES))
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        child(args.child, args.warmup, args.steps, args.max_seconds)
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.soak_minutes and args.batch_size and args.block_size and args.dtype:
        soak(device, args.batch_size, args.block_size, args.dtype, args.soak_minutes)
        return

    results = []
    print(f"device {device}, torch {torch.__version__}; each setting runs in its own process, "
          f"aborted if a step takes > {STEP_LIMIT_S:.0f}s")
    print(f"{'batch':>5} {'block':>5} {'dtype':>5} {'tok/s':>9} {'step ms':>8} {'loss':>7} {'cur MB':>7} {'drv MB':>7}")
    for bl in [int(x) for x in args.block_sizes.split(",")]:
        for bs in [int(x) for x in args.batch_sizes.split(",")]:
            for dt in args.dtypes.split(","):
                r, status = run_setting_in_child(bs, bl, dt, args.warmup, args.steps, args.max_seconds)
                if r is None:
                    print(f"{bs:>5} {bl:>5} {dt:>5}  {status}", flush=True)
                    continue
                r.update(batch=bs, block=bl, dtype=dt)
                results.append(r)
                flag = "" if r["finite"] else "  NON-FINITE"
                print(f"{bs:>5} {bl:>5} {dt:>5} {r['tok_s']:>9.0f} {r['step_ms']:>8.0f} {r['loss']:>7.3f} "
                      f"{r['cur_mb']:>7.0f} {r['drv_mb']:>7.0f}{flag}", flush=True)
    ok = [r for r in results if r["finite"]]
    if not ok:
        print("no successful settings")
        return
    best = max(ok, key=lambda r: r["tok_s"])
    print(f"best: batch {best['batch']}, block {best['block']}, {best['dtype']} -> {best['tok_s']:.0f} tok/s, "
          f"{best['drv_mb']:.0f} MB driver memory")
    if args.soak_minutes:
        soak(device, args.batch_size or best["batch"], args.block_size or best["block"],
             args.dtype or best["dtype"], args.soak_minutes)


if __name__ == "__main__":
    main()
