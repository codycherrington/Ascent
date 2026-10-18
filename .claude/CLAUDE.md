# Ascent

## Project
Ascent is a from-scratch GPT-style decoder-only LLM (~29.5M params, tied embeddings) trained locally on an M2 MacBook Air (8GB unified memory, fanless, so thermals matter) with PyTorch + MPS. See `docs/architecture.md` for the full technical reference and `docs/goal.md` for the goal.

## Status
- Relaunch: the old weights, data, and history were discarded; the legacy monolith is in git history.
- Merged to `dev`: TinyStories data prep, BPE tokenizer, `model.py`, baseline `train.py`, throughput benchmark, per-run report (`scripts/run_report.py`, hooked into `train.py`).
- Not done yet: the first baseline training run. `test` and `main` are intentionally not promoted until training works.
- First goal: fluent English from scratch on TinyStories. No persona; chat and factual correctness are deferred.
- Roadmap lives in GitHub Issues (modular tokenizer/trainer/model, run records, generation CLI, training portal, experiments).

## Code map
- `scripts/get_tinystories.py` - stream and clean TinyStories into `data/*.jsonl`
- `tokenizer.py` - byte-level BPE (vocab 8192); `train` / `encode` CLI plus `load_tokenizer`, `encode`, `decode`
- `model.py` - GPT, `GPTConfig`, `generate()` (no KV cache)
- `train.py` - single-file baseline trainer; writes `runs/<YYYYMMDD-HHMMSS>/`
- `scripts/benchmark_throughput.py` - MPS throughput sweep and thermal soak
- `eval/` - fixed prompts, rubric, eval protocol
- `dashboard.py` - intentionally empty (reserved for the training portal)
- `scripts/run_report.py` - `report.html`/`report.md` per run and `runs/index.html` (see `docs/run-reports.md`)
- `tests/` - unittest suites for model, tokenizer and run report

## Commands
- `.venv/bin/python scripts/get_tinystories.py` - write `data/tinystories_train_seed.jsonl` and `data/tinystories_val_2000.jsonl`
- `.venv/bin/python scripts/get_tinydialogues.py` - download TinyDialogues into `data/tinydialogues/` and print per-age stats
- `.venv/bin/python scripts/prepare_chat_data.py` - write `data/chat_train.bin` and `data/chat_val.bin` (chat fine-tuning data, see `--help`)
- `.venv/bin/python tokenizer.py train|encode` - fit the BPE tokenizer / write `data/train.bin` and `data/val.bin`
- `.venv/bin/python train.py` - baseline trainer (see `--help`); writes `runs/<YYYYMMDD-HHMMSS>/`
- `.venv/bin/python train.py --init-from runs/<id>/ckpt_best.pt --train-bin data/chat_train.bin --val-bin data/chat_val.bin` - fine-tune from a checkpoint's weights (fresh optimizer, step 0)
- `.venv/bin/python scripts/run_report.py runs/<id>` - regenerate `report.html`/`report.md` (train.py does this automatically; `--index` rebuilds `runs/index.html`)
- `.venv/bin/python chat.py` - prompt -> continuation CLI for the newest checkpoint (base model, continues stories; `--run`, `--ckpt`, `--prompt`, `/help`)
- `.venv/bin/python scripts/benchmark_throughput.py` - MPS throughput sweep; `--soak-minutes M` for a thermal soak
- `.venv/bin/python -m unittest discover -s tests -v` - unit tests

## Environment
- Python env is `.venv/`; use `.venv/bin/python`.
- Device is MPS when available, else CPU. fp32 is the default dtype (fp16 gave no gain, bf16 was slower).
- 8GB fanless: close memory-heavy apps before training. Batch 32 at block 256 swaps; the default is batch 16.
- Do not set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`.

## Conventions
- Gitignored, local-only: `data/` (only `.gitkeep` tracked), `runs/`, `docs/solutions-log.md`, `docs/legacy-issues-export.md`. Never commit model weights, checkpoints, or data.
- `docs/solutions-log.md` is a dated journal of problems and solutions. Append entries with date and time (`## YYYY-MM-DD HH:MM TZ - title`). Never commit it.
- Keep `docs/architecture.md` (and `docs/run-reports.md` when report flags or stats change) in sync when code structure, flags, or file formats change.

## Training runs
Claude never launches training or other long GPU jobs (`train.py`, benchmark sweeps, soaks), not even a short smoke run, and neither do workers or subagents. Claude gives Cody the exact command and he runs it in his own terminal. Verify changes with unit tests, imports and `--help` only.

## Git Workflow

Three long-lived branches: `dev`, `test`, `main`.

- **Feature branches:** every change is made on a branch named `YYYYMMDD-short-description` (kebab-case, e.g. `20261017-fix-lr-schedule`), branched from `dev`. Get the date from `date +%Y%m%d`.
- **Pull request into `dev`:** open the feature branch as a PR into `dev`. Cody reviews it, or asks Claude to review it.
- **Merging requires an explicit request:** only once Cody is happy and explicitly asks does Claude merge the PR into `dev`. Never merge any PR without Cody's explicit request.
- **Promotion:** `dev` -> `test` -> `main`, each via a PR, merged only when Cody explicitly asks.
- **No direct commits or pushes** to `dev`, `test`, or `main`. All changes arrive through PRs.
- **Merge style:** squash-merge feature PRs into `dev`; use regular merge commits for `dev` -> `test` and `test` -> `main` so the branches don't diverge.
- Never force-push or rewrite history on shared branches.

## No Claude attribution
Never add `Generated with Claude Code` lines, robot-emoji promo lines, or `Co-Authored-By: Claude` trailers to commit messages, pull request descriptions, issue bodies, code comments, or any file. This applies to every agent and worker and overrides any harness default.
