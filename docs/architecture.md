# Architecture

Technical reference for how the Ascent code base is put together. Everything here was checked against the code at the time of writing; if code and doc disagree, the code wins. For the goal see `docs/goal.md`.

Ascent is a from-scratch ~29.5M-parameter GPT-style decoder-only language model, trained locally on an M2 MacBook Air (8GB, fanless) with PyTorch on MPS (falls back to CPU if MPS is unavailable).

## 1. Pipeline

```
HF roneneldan/TinyStories (streamed)
        |
        v  scripts/get_tinystories.py
data/tinystories_train_seed.jsonl   data/tinystories_val_2000.jsonl
        |
        v  tokenizer.py train        (train seed only)
data/tokenizer.json
        |
        v  tokenizer.py encode       (both jsonl files)
data/train.bin   data/val.bin        (flat uint16 token streams)
        |
        v  train.py  (imports model.py and the tokenizer helpers)
runs/<YYYYMMDD-HHMMSS>/              (config, metrics, samples, checkpoints, report)
        |
        v  scripts/run_report.py     (called by train.py at the end; also standalone)
runs/<id>/report.html, report.md     runs/index.html
```

Each arrow is a separate command; nothing runs automatically. All commands run from the repo root with `.venv/bin/python`.

## 2. Repository layout

Tracked files (`git ls-files`):

```
.claude/CLAUDE.md               project instructions for Claude Code (only tracked file in .claude/)
.gitignore
LICENSE
README.md                       short project summary
requirements.txt                torch, torchvision, torchaudio, matplotlib, datasets, tqdm, tokenizers, numpy
dashboard.py                    intentionally empty; reserved for the future training portal
model.py                        GPT model, param counts, generate()
tokenizer.py                    BPE train/encode CLI and load_tokenizer/encode/decode helpers
train.py                        baseline trainer (single file)
data/.gitkeep                   keeps the otherwise-ignored data/ dir in the repo
docs/README.md                  index of the docs folder
docs/architecture.md            this file
docs/goal.md                    goal, non-goals, what success looks like
docs/run-reports.md             how to read a run report (sections, flags, thresholds)
docs/tokenizer.md               tokenizer rationale, usage, measured stats
docs/datasets/tinystories.md    dataset card (source, license, filtering, counts)
docs/datasets/tinydialogues.md  dataset card and chat data format
eval/README.md                  evaluation protocol
eval/prompts.txt                20 fixed continuation prompts
eval/rubric.md                  1-5 fluency rubric
scripts/get_tinystories.py      data download and cleaning
scripts/get_tinydialogues.py    TinyDialogues download and per-age stats
scripts/prepare_chat_data.py    build data/chat_train.bin and data/chat_val.bin
scripts/benchmark_throughput.py MPS throughput sweep and thermal soak
scripts/run_report.py           per-run report.html/report.md and runs/index.html
tests/test_chat_data.py         chat data pipeline and --init-from tests
tests/test_model.py             model unit tests
tests/test_run_report.py        run report unit tests
tests/test_tokenizer.py         tokenizer unit tests
```

Local-only, gitignored (never committed):

| Path | What |
|------|------|
| `data/*` (except `.gitkeep`) | jsonl files, `tokenizer.json`, `train.bin`, `val.bin` |
| `runs/` | training run folders, including `report.html`, `report.md` and `runs/index.html` |
| `.venv/` | Python environment |
| `*.pt`, `*.pth`, `*.safetensors`, `*.log`, `checkpoints/` | weights and logs |
| `.claude/*` except `CLAUDE.md` | local Claude files |
| `docs/solutions-log.md`, `docs/legacy-issues-export.md` | local journal and backup |

## 3. Data

Script: `scripts/get_tinystories.py`. Full details and counts are in `docs/datasets/tinystories.md`.

| Item | Value |
|------|-------|
| Source | Hugging Face `roneneldan/TinyStories`, streamed (full split never written to disk) |
| License | CDLA-Sharing-1.0 |
| Cleaning | strip; curly quotes to straight; collapse spaces/tabs, drop blank lines (paragraphs separated by one `\n`); drop empty, under 200 chars, non-ASCII; drop exact duplicates (SHA-1 of cleaned text) |
| Train seed | first 50,000 cleaned stories of the train split -> `data/tinystories_train_seed.jsonl` (10,693,222 tokens after encoding) |
| Val set | first 2,000 cleaned stories of the validation split -> `data/tinystories_val_2000.jsonl` (373,351 tokens), fixed across runs |
| Format | JSONL, one `{"text": "<story>"}` per line |
| Options | `--out-dir` (data), `--num-train-stories` (50000), `--num-val-stories` (2000) |

The script is deterministic (dataset order, no randomness) and overwrites its outputs.

## 4. Tokenizer

`tokenizer.py`, using Hugging Face `tokenizers`. See `docs/tokenizer.md` for rationale and measured stats.

- Byte-level BPE, vocab size 8192 (`train --vocab-size`, default 8192).
- Special tokens take the first ids: `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3 (`train` asserts this). `<bos>` is reserved and never inserted; `<pad>` is not used by the trainer.
- Fit on `data/tinystories_train_seed.jsonl` only, never on validation.
- `encode` writes `data/train.bin` and `data/val.bin` as flat `uint16` arrays: each story's ids followed by one `<eos>`. Prints stories, tokens and tokens/word.
- Public helpers: `load_tokenizer(path="data/tokenizer.json")`, `encode(tok, text) -> list[int]`, `decode(tok, ids) -> str` (does not skip special tokens).
- Module constants: `TRAIN_JSONL`, `VAL_JSONL`, `TOKENIZER_PATH`, `SPECIAL_TOKENS`.

## 5. Model

`model.py`. Classes: `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT`, plus the function `generate`.

`GPTConfig` defaults:

| Field | Default |
|-------|---------|
| `vocab_size` | 8192 (the trainer overrides it with the tokenizer's vocab size) |
| `n_layer` | 8 |
| `n_head` | 8 (head dim 64) |
| `n_embd` | 512 |
| `block_size` | 256 |
| `dropout` | 0.0 |
| `bias` | False |

Structure:

- Token embedding + learned position embedding (`nn.Embedding(block_size, n_embd)`), summed, then dropout.
- 8 pre-norm blocks: `x = x + attn(ln1(x))`, then `x = x + mlp(ln2(x))`. LayerNorm without bias.
- Attention: one fused `qkv` linear, split into heads, `F.scaled_dot_product_attention(..., is_causal=True)`, output `proj`. No separate mask tensor.
- MLP: `fc` (n_embd -> 4*n_embd), GELU, `proj` (back to n_embd).
- Final `ln_f`, then `head` linear (no bias) whose weight is tied to the token embedding.
- Loss: cross-entropy on `logits.float()`; `forward(idx, targets)` returns `(logits, loss)`, loss is `None` without targets. Sequences longer than `block_size` hit an assert.
- Init (GPT-2 style): Linear and Embedding weights ~ N(0, 0.02); the two residual projections per block (attention `proj`, MLP `proj`, flagged `is_residual`) use std `0.02 / sqrt(2 * n_layer)`; biases zero.

Parameter breakdown (default config, from `.venv/bin/python model.py`; tied weight counted once):

| Part | Params |
|------|--------|
| Token embedding (8192 x 512, shared with head) | 4,194,304 |
| Position embedding (256 x 512) | 131,072 |
| Per block: qkv 786,432 + attn proj 262,144 + mlp fc 1,048,576 + mlp proj 1,048,576 + 2 LayerNorms 1,024 | 3,146,752 |
| 8 blocks | 25,174,016 |
| Final LayerNorm | 512 |
| Total | 29,499,904 |
| Embedding (tok + pos) / transformer | 4,325,376 / 25,174,528 |

`GPT.param_counts()` returns `(total, embedding, transformer)`.

`generate(model, ids, max_new_tokens, temperature=0.8, top_k=40, seed=None, eos_id=None)`:

- `ids` is a list of ints or a `(1, T)` LongTensor; returns only the new token ids as a list.
- `temperature == 0` is greedy (argmax). Otherwise logits are divided by temperature, then top-k filtered (`top_k` of None or 0 disables; k is clamped to at least 1), then sampled with `torch.multinomial`.
- `seed` creates a CPU `torch.Generator`, so sampling is reproducible.
- Stops early after emitting `eos_id` (the eos token is included in the output).
- The context is cropped to the last `block_size` tokens each step. There is no KV cache: every new token reruns the full forward pass, so generation is slow.
- Puts the model in eval mode during generation and restores the previous mode.

## 6. Trainer

`train.py` is a single-file baseline trainer. Device is MPS if available, else CPU (on CPU, `--dtype` is ignored and fp32 is used).

### Training loop

- **Batching:** `get_batch` draws `batch_size` random start offsets (seeded `numpy` generator, `--seed`) from the memory-mapped `train.bin`; x is `block_size` tokens, y is the same window shifted by one. Windows may cross story boundaries.
- **Tokens per step** = `batch_size * block_size * grad_accum`. Default 16 * 256 * 1 = 4,096.
- **max_steps** = `max_tokens // tokens_per_step`. Default 10,000,000 // 4,096 = 2,441 steps.
- **Optimizer:** AdamW, betas (0.9, 0.95). Two groups: weight decay (`--weight-decay`) on parameters with 2 or more dims (linear weights and both embeddings; the tied weight appears once), zero decay on 1-D parameters (LayerNorm weights).
- **LR schedule** (`lr_at`): linear warmup to `--lr` over `--warmup-steps` (lr = lr * (step+1)/warmup), then cosine decay to `--lr * --min-lr-frac` at `max_steps`.
- **Gradient accumulation:** loss divided by `grad_accum` per micro-batch.
- **Grad clip:** `clip_grad_norm_` at `--grad-clip`; the pre-clip norm is logged.
- **dtype:** `--dtype fp32` (default, no autocast), `fp16` or `bf16` (autocast). In our MPS benchmark fp16 gave no gain over fp32 and bf16 was the slowest.
- **Stop on non-finite loss:** checked at log steps; the loop breaks, then the final eval/samples still run.

### Eval

`evaluate` computes the mean loss over the whole val set in fixed, non-overlapping `block_size` windows (`(len(val) - 1) // block_size` windows, batched by `--eval-batch-size`), so every eval and every run measures the same tokens. Perplexity is `exp(loss)`. A final eval runs at the end unless one just ran at the last step.

### Checkpointing

Every eval writes `ckpt.pt` (latest) and, if val loss improved, `ckpt_best.pt`. A checkpoint holds `model` (CPU copy of the state dict), `optimizer`, `step` and `config`. Writes are atomic: saved to `<name>.tmp`, then `os.replace`d.

### Sampling

`write_samples` reads `eval/prompts.txt` (non-blank lines, 20 prompts) and for each prompt generates 120 new tokens twice: sampled (temperature 0.8, top-k 40, seed 1234 reset per prompt) and greedy. Output goes to `samples_stepNNNN.txt`. Samples are written every `--sample-interval` steps (if the step is below `max_steps`) and always once at the end, unless `--no-samples`.

### SIGINT

Ctrl-C sets a flag instead of raising. The loop finishes the current step and exits; the normal end-of-run path then does a final eval (with checkpoint) and samples. A second Ctrl-C also just sets the flag again.

### Resume (`--resume runs/<id>/ckpt.pt`)

Loads model weights, optimizer state and step from the checkpoint, then continues up to `max_steps` computed from the current CLI flags, with the LR schedule continuing from that step. Known quirks:

- The RNG is re-seeded from `--seed`, so the batch sequence restarts and early batches repeat those of the original run.
- A new run folder (new timestamp) is created; the old one is not appended to.
- `best_val` restarts at infinity, so the first eval after resume always writes `ckpt_best.pt` in the new folder.
- Model-shape flags must match the checkpoint, otherwise `load_state_dict` fails.

### CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--train-bin` | `data/train.bin` | training tokens |
| `--val-bin` | `data/val.bin` | validation tokens |
| `--tokenizer` | `data/tokenizer.json` | tokenizer file (sets vocab size) |
| `--prompts` | `eval/prompts.txt` | sample prompts |
| `--out-dir` | `runs` | parent of run folders |
| `--n-layer` | 8 | transformer blocks |
| `--n-head` | 8 | attention heads |
| `--n-embd` | 512 | embedding width |
| `--block-size` | 256 | context length |
| `--dropout` | 0.0 | dropout |
| `--batch-size` | 16 | sequences per micro-batch |
| `--grad-accum` | 1 | micro-batches per step |
| `--max-tokens` | 10,000,000 | token budget; sets max_steps |
| `--lr` | 6e-4 | peak learning rate |
| `--min-lr-frac` | 0.1 | final lr as a fraction of peak |
| `--warmup-steps` | 200 | linear warmup steps |
| `--weight-decay` | 0.1 | AdamW decay (matrices only) |
| `--grad-clip` | 1.0 | max grad norm |
| `--dtype` | `fp32` | `fp32`, `fp16` or `bf16` |
| `--eval-interval` | 500 | steps between val evals |
| `--sample-interval` | 1000 | steps between samples (0 = only at the end) |
| `--eval-batch-size` | 16 | batch size for eval |
| `--log-interval` | 10 | steps between train records |
| `--no-samples` | off | never generate samples |
| `--no-report` | off | skip `report.html`/`report.md` at the end |
| `--seed` | 1337 | seeds torch and the batch sampler |
| `--resume` | none | checkpoint to continue from |

## 7. Run records

Each run writes `runs/<YYYYMMDD-HHMMSS>/` (local time, second resolution; the folder is gitignored):

| File | Contents |
|------|----------|
| `config.json` | `args` (all CLI values), `model` (GPTConfig fields), `git_commit`, `torch` version, `device`, `params`, `params_embedding`, `params_transformer`, `tokens_per_step`, `max_steps` |
| `metrics.jsonl` | one JSON object per line, flushed on write; see below |
| `samples_stepNNNN.txt` | header (step and decoding settings), then per prompt: prompt, sampled output, greedy output |
| `ckpt.pt` | latest checkpoint (written at each eval) |
| `ckpt_best.pt` | checkpoint with the lowest val loss so far |
| `report.html` | self-contained report (charts, flags, samples, scoring sheet); written at the end of the run |
| `report.md` | compact plain-text summary of the same, for quick scanning by a human or an LLM |

`runs/index.html` (one level up, written by `scripts/run_report.py`) lists every run, newest first. See `docs/run-reports.md`.

`metrics.jsonl` record types:

| `type` | Fields | When |
|--------|--------|------|
| `train` | `step`, `loss`, `lr`, `grad_norm`, `tok_s`, `step_ms` | first step of the run and every `--log-interval` steps. `lr` is the value used for that step; `loss` is the mean over micro-batches |
| `eval` | `step`, `val_loss`, `val_ppl`, `val_tokens`, `eval_s`, and `mps_driver_mb` (MPS only) | every `--eval-interval` steps and at the end |
| `samples` | `step`, `samples_s` | each time samples are written |

The tokenizer is not yet copied into the run folder (`docs/tokenizer.md` lists this as future run-records work).

## 7a. Run reports

`scripts/run_report.py` turns a run folder into `report.html` and `report.md` and rebuilds `runs/index.html`. `train.py` calls it at the very end (also after Ctrl-C) inside try/except, so a report failure only prints a warning and never changes the exit code; `--no-report` skips it. Run folders are local and gitignored, so reports are derived files: regenerate any time with

    .venv/bin/python scripts/run_report.py runs/<id>     # one run (also refreshes the index)
    .venv/bin/python scripts/run_report.py --index       # index only

Sections, the exact text-stat definitions, the status values and every automatic flag with its threshold are described in `docs/run-reports.md`. Incomplete runs (crash, truncated last line, no eval or samples) degrade to notes rather than errors.

## 8. Eval

Documented in `eval/README.md`; the pieces:

- `eval/README.md`: the protocol. Held-out loss/perplexity on the fixed 2,000-story val set; fixed prompt set with fixed decoding settings; rubric use; automatic text-quality stats (which are implemented and which are not); how results are recorded.
- `eval/prompts.txt`: 20 prompts, one per line, no header or comments. Prompt id = line number. `train.py` reads this file directly.
- `eval/rubric.md`: 1-5 fluency rubric (grammar, coherence, repetition, stays on prompt), scored by eye at milestones.

Implemented in `train.py`: val loss/perplexity and the sample generation. Implemented in `scripts/run_report.py`, per samples file and per decoding mode: distinct-1/2/3, repeated 4-gram fraction, mean length in words, fraction of outputs that hit `<eos>`, and greedy loop count (definitions in `eval/README.md` section 4 and `docs/run-reports.md`). Not implemented: sentence-final punctuation fraction, and mean length in tokens. Rubric scoring is manual (a blank sheet is in `report.html`).

## 9. Tests and tooling

Run all tests:

```
.venv/bin/python -m unittest discover -s tests -v
```

| File | Covers |
|------|--------|
| `tests/test_model.py` | tiny CPU config: forward shape, causal masking, tied weights, initial loss near uniform, `generate` determinism with seed, top-k 1 equals greedy, conditioning on the prompt, generation beyond `block_size`, overfitting one batch |
| `tests/test_tokenizer.py` | special token ids, vocab size, string round trips, no `<unk>` on ASCII, round trip of sample stories from the val jsonl. Skipped if `data/tokenizer.json` is missing |

| `tests/test_run_report.py` | run report generation and its helpers |

There are no tests for `train.py`, `scripts/get_tinystories.py` or `scripts/benchmark_throughput.py`.

`scripts/benchmark_throughput.py` times full train steps (forward, backward, clip, AdamW) on random tokens and writes no files.

- Sweep (default): `--batch-sizes 8,16,32`, `--block-sizes 128,256`, `--dtypes fp32,fp16`, `--warmup 3`, `--steps 20`, `--max-seconds 15`. Each setting runs in a child process that is killed if a step takes over 30 s. Prints tok/s, step ms, loss and MPS memory, then the best setting.
- Soak: `--soak-minutes M` with `--batch-size`, `--block-size` and `--dtype` runs a thermal soak directly; with `--soak-minutes` alone it sweeps first and soaks the best setting. It reports tok/s per 30 s window plus `pmset -g therm` at start, middle and end, and the first-vs-last window drop.

## 9a. Chat fine-tuning data

Stage 1 of conversation ability: fine-tune the TinyStories checkpoint on short back-and-forth dialogues, with the same tokenizer (the checkpoint is tied to it).

- `scripts/get_tinydialogues.py` downloads `individual_age_data.zip` (exact per-age files) into `data/tinydialogues/` and prints per-age stats. Details: [datasets/tinydialogues.md](datasets/tinydialogues.md).
- `scripts/prepare_chat_data.py` streams the zip in chunks (peak RSS about 225 MB) and writes `data/chat_train.bin` / `data/chat_val.bin` (uint16, `<eos>` after each window), mixing in TinyStories at 12% of tokens.
- Format: one turn per line, `You: ...` / `Ascent: ...`. Windows are at most 256 tokens including `<eos>`, start at a `You:` turn and end on an `Ascent:` turn.
- `train.py --init-from CKPT` loads model weights only (shape checked via `load_init_weights`; error on mismatch), with a fresh optimizer, step 0 and a new LR schedule; it is exclusive with `--resume`, and `init_from` is recorded in `config.json`. Pass `--train-bin data/chat_train.bin --val-bin data/chat_val.bin`.
- Tests: `tests/test_chat_data.py` (parser, role mapping, window splitter, val split, `--init-from` helper).

## 10. Known limitations

- `generate()` has no KV cache, so sampling is slow (20 prompts x 2 modes x 120 tokens per sample round).
- The live SIGINT path and `--sample-interval` have not been tested in a live run.
- `--resume` reseeds the RNG (batches repeat) and starts a new run folder; see section 6.
- `dashboard.py` is empty; the training portal is a roadmap item.
- The modular trainer, run-record tooling (such as copying the tokenizer into runs) and the generation CLI are roadmap items (see GitHub Issues); `train.py` is a single baseline file.
- Two automatic text stats from `eval/README.md` are not implemented (sentence-final punctuation fraction, mean length in tokens); the rest are in `scripts/run_report.py`. The report flags are rough pointers, and the stats are word-based, not token-based.
- Rubric scores are not stored anywhere: the scoring sheet in `report.html` is blank and not saved.
- The report has only been exercised on short smoke runs; no baseline run exists yet.
- Pretraining data is only TinyStories. The chat data pipeline (section 9a) exists but no chat fine-tune has been run yet; no persona or factual training.

## 11. Development workflow

- Three long-lived branches: `dev`, `test`, `main`. No direct commits or pushes to them.
- Every change is on a feature branch `YYYYMMDD-short-description` (kebab-case) cut from `dev`, and opened as a PR into `dev` (squash-merged).
- Promotion is `dev` -> `test` -> `main`, each by PR with a regular merge commit.
- PRs are merged only on an explicit request from the owner.
