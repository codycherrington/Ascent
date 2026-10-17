# Goal: fluent English from scratch

## Goal

Ascent's first model is a ~30M-parameter GPT-style decoder-only language model, trained from scratch on a MacBook Air (M2, 8GB, fanless) with PyTorch/MPS, that produces fluent-looking English.

That is the only goal at this stage. The first corpus is TinyStories (short synthetic children's stories, see issue #2), so the model is a text-continuation story model, not a chatbot.

## Non-goals (for now)

- Persona or personality.
- Chat format or instruction following.
- Factual correctness.
- Benchmarks against other models or public leaderboards.
- Speed targets for training or inference.

## What a model at this stage looks like

It tells simple, grammatical short stories. It may be wrong, silly or inconsistent about details, but the text is readable: sentences are well formed, they follow from each other over a few sentences, and the model does not loop or collapse into repetition.

## Success criteria

### Quantitative

Validation loss and perplexity on the held-out split (see `eval/README.md`), measured at every eval and read from the run report (`report.html` / `report.md` in the run folder, see `docs/run-reports.md`). Milestone thresholds are not set yet.

| Milestone | Val loss | Val perplexity | Notes |
|-----------|----------|----------------|-------|
| Baseline run (first full training run) | TBD | TBD | Recorded, not judged |
| Milestone 1 | TBD | TBD | TBD |
| Milestone 2 | TBD | TBD | TBD |
| Milestone 3 | TBD | TBD | TBD |

How thresholds will be chosen: run the baseline first and record its result. Set Milestone 1 as a modest improvement over the baseline, then raise later milestones after each round of improvements (more data, longer training, tuning), using what the baseline curve shows is realistic. Thresholds are filled in after the baseline run, not guessed beforehand.

### Qualitative

Fluency rubric scores (grammar, coherence, repetition, stays-on-prompt, each 1-5) on samples from the fixed prompt set in `eval/prompts.txt`. See `eval/rubric.md`; the report has a blank scoring sheet and the automatic text stats (distinct-n, repetition, loops) to read alongside. Scores are by eye and subjective; they are a sanity check on the loss numbers, not a replacement for them.

## Roadmap

The work is staged as GitHub issues #2 through #9 in codycherrington/Ascent, starting with the TinyStories data (#2) and the run-records setup (#6). This document is issue #1; later issues depend on the eval defined here.

## Parked for later

- Curious-explainer persona.
- Chat format and instruction following.
- Synthetic teacher-written data.
- Distillation from a larger model.
