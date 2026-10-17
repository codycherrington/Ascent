"""Baseline trainer for Ascent: train the GPT in model.py on data/train.bin.

    .venv/bin/python train.py                         # 10M-token baseline with defaults
    .venv/bin/python train.py --max-tokens 820000 --eval-interval 100 --sample-interval 0
    .venv/bin/python train.py --resume runs/<id>/ckpt.pt

Each run writes runs/<YYYYMMDD-HHMMSS>/: config.json, metrics.jsonl,
samples_stepNNNN.txt, ckpt.pt and ckpt_best.pt.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig, generate
from tokenizer import decode, encode, load_tokenizer

DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description="Train the Ascent baseline GPT.")
    p.add_argument("--train-bin", default="data/train.bin")
    p.add_argument("--val-bin", default="data/val.bin")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--prompts", default="eval/prompts.txt")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=10_000_000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    p.add_argument("--eval-interval", type=int, default=500, help="steps between val-loss evals")
    p.add_argument("--sample-interval", type=int, default=1000,
                   help="steps between sample generations (0 = only at the end; always at the end)")
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--no-samples", action="store_true", help="never generate samples")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", default=None, help="checkpoint to load weights, optimizer and step from")
    return p.parse_args()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def lr_at(step, args, max_steps):
    """Linear warmup to args.lr, then cosine decay to min_lr_frac * lr at max_steps."""
    min_lr = args.lr * args.min_lr_frac
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = min(1.0, (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps))
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - min_lr)


def get_batch(data, batch_size, block_size, rng, device):
    """Random windows (nanoGPT style) drawn with a seeded numpy generator."""
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model, val, block_size, batch_size, device, dtype):
    """Mean loss over the whole val set in fixed, non-overlapping windows."""
    model.eval()
    n_win = (len(val) - 1) // block_size
    total_loss, total_tok = 0.0, 0
    for start in range(0, n_win, batch_size):
        idx = range(start, min(start + batch_size, n_win))
        x = np.stack([val[i * block_size:(i + 1) * block_size] for i in idx]).astype(np.int64)
        y = np.stack([val[i * block_size + 1:(i + 1) * block_size + 1] for i in idx]).astype(np.int64)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
            _, loss = model(x, y)
        total_loss += float(loss) * y.numel()
        total_tok += y.numel()
    model.train()
    return total_loss / total_tok, total_tok


def write_samples(model, tok, prompts, path, step):
    """Sampled (T=0.8, top-k 40, seed 1234 per prompt) and greedy, 120 new tokens each."""
    eos = tok.token_to_id("<eos>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"step {step}\nsampled: temperature 0.8, top-k 40, seed 1234; greedy; max 120 new tokens\n\n")
        for i, prompt in enumerate(prompts, 1):
            ids = encode(tok, prompt)
            sampled = generate(model, ids, 120, temperature=0.8, top_k=40, seed=1234, eos_id=eos)
            greedy = generate(model, ids, 120, temperature=0.0, eos_id=eos)
            f.write(f"=== prompt {i} ===\n{prompt}\n--- sampled ---\n{decode(tok, sampled)}\n"
                    f"--- greedy ---\n{decode(tok, greedy)}\n\n")


def save_ckpt(path, model, opt, step, config):
    """Write a CPU copy atomically (temp file, then rename)."""
    state = {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": opt.state_dict(),
        "step": step,
        "config": config,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = DTYPES[args.dtype] if device != "cpu" else None
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    val = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    tok = load_tokenizer(args.tokenizer)
    with open(args.prompts, encoding="utf-8") as f:
        prompts = [line.rstrip("\n") for line in f if line.strip()]

    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=args.n_layer, n_head=args.n_head,
                    n_embd=args.n_embd, block_size=args.block_size, dropout=args.dropout)
    model = GPT(cfg).to(device)
    total, emb, tr = model.param_counts()
    print(f"params: {total:,} total | {emb:,} embedding | {tr:,} transformer")

    # Weight decay on matrices only (not norms/biases); tied weight appears once.
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    max_steps = args.max_tokens // tokens_per_step
    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    run_dir = os.path.join(args.out_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    config = {"args": vars(args), "model": asdict(cfg), "git_commit": git_commit(),
              "torch": torch.__version__, "device": device, "params": total,
              "params_embedding": emb, "params_transformer": tr,
              "tokens_per_step": tokens_per_step, "max_steps": max_steps}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    metrics = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    print(f"run dir {run_dir} | device {device} {args.dtype} | {tokens_per_step} tok/step | {max_steps} steps")

    def log(rec):
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()

    stop = {"flag": False}

    def on_sigint(signum, frame):
        print("\nSIGINT: finishing current step, saving checkpoint, exiting")
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_sigint)

    best_val = float("inf")

    def do_samples(step):
        t0 = time.perf_counter()
        write_samples(model, tok, prompts, os.path.join(run_dir, f"samples_step{step:04d}.txt"), step)
        sec = round(time.perf_counter() - t0, 2)
        log({"type": "samples", "step": step, "samples_s": sec})
        print(f"samples step {step}: {len(prompts)} prompts in {sec}s")

    def do_eval(step):
        nonlocal best_val
        t0 = time.perf_counter()
        vloss, ntok = evaluate(model, val, args.block_size, args.eval_batch_size, device, dtype)
        rec = {"type": "eval", "step": step, "val_loss": vloss, "val_ppl": math.exp(vloss),
               "val_tokens": ntok, "eval_s": round(time.perf_counter() - t0, 2)}
        if device == "mps":
            rec["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2**20)
        log(rec)
        print(f"eval step {step}: val loss {vloss:.4f} ppl {math.exp(vloss):.2f} ({rec['eval_s']}s eval)")
        save_ckpt(os.path.join(run_dir, "ckpt.pt"), model, opt, step, config)
        if vloss < best_val:
            best_val = vloss
            save_ckpt(os.path.join(run_dir, "ckpt_best.pt"), model, opt, step, config)

    model.train()
    step = start_step
    sync(device)
    t_last, last_step = time.perf_counter(), start_step
    while step < max_steps and not stop["flag"]:
        lr = lr_at(step, args, max_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train, args.batch_size, args.block_size, rng, device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            loss_sum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_interval == 0 or step == start_step + 1:
            sync(device)
            now = time.perf_counter()
            dt = (now - t_last) / (step - last_step)
            t_last, last_step = now, step
            lval = float(loss_sum) / args.grad_accum
            rec = {"type": "train", "step": step, "loss": lval, "lr": lr, "grad_norm": float(grad_norm),
                   "tok_s": tokens_per_step / dt, "step_ms": 1000 * dt}
            log(rec)
            print(f"step {step:5d} | loss {lval:.4f} | lr {lr:.2e} | {rec['tok_s']:7.0f} tok/s | {rec['step_ms']:.0f} ms")
            if not math.isfinite(lval):
                print("non-finite loss, stopping")
                break

        evaled = step % args.eval_interval == 0
        sampled = not args.no_samples and args.sample_interval > 0 and step % args.sample_interval == 0
        if evaled:
            do_eval(step)
        if sampled and step < max_steps:
            do_samples(step)
        if evaled or sampled:
            sync(device)
            t_last, last_step = time.perf_counter(), step

    if step % args.eval_interval != 0:
        do_eval(step)
    if not args.no_samples:
        do_samples(step)
    metrics.close()
    print(f"done at step {step}; best val loss {best_val:.4f}; run dir {run_dir}")


if __name__ == "__main__":
    main()
