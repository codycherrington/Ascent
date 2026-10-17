# Ascent

## Project
Ascent is a from-scratch GPT-style decoder-only LLM trained locally on an M2 MacBook Air (8GB unified memory, fanless, so thermals matter) with PyTorch + MPS.

## Status: relaunch
Fresh start. The old weights, data, and history were discarded.

- First goal: fluent English from scratch on TinyStories. No persona yet; chat and factual correctness are deferred.
- Roadmap lives in the GitHub Issues (modular tokenizer/trainer/model, run records, generation CLI, training portal, experiments).
- `model.py` is the legacy monolith, kept for reference until the new modular trainer replaces it.

## Environment
- Python env is `.venv/`; use `.venv/bin/python`.
- Do not commit model weights, checkpoints, or data (all gitignored; `data/` holds only `.gitkeep`).

## Local journal
`docs/solutions-log.md` is a local-only, gitignored, dated journal of problems and solutions. Append entries with date and time (`## YYYY-MM-DD HH:MM TZ - title`). Never commit it.

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
