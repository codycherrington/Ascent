# Eval

How Ascent's first model is evaluated. The goal and success criteria live in `docs/goal.md`. Everything here is for a text-continuation story model trained on TinyStories, not a chatbot. Results are read from each run's report (`docs/run-reports.md`).

## 1. Held-out validation (loss and perplexity)

- Data: the TinyStories validation split as provided by the dataset. It is split by story, never by sentence, so no story is partly in training and partly in validation.
- The validation split is never trained on, including for tokenizer fitting or data-mixing decisions.
- Each eval uses a fixed subset: `data/tinystories_val_2000.jsonl`, the first 2,000 cleaned stories of the validation split, produced by `scripts/get_tinystories.py`, so every run measures the same text.
- Metrics: mean cross-entropy loss per token, and perplexity = exp(loss). Computed at every eval (`train.py` evaluates the whole `data/val.bin`, the encoded form of that subset, in fixed non-overlapping 256-token windows).

## 2. Fixed prompt set and sample generation

`eval/prompts.txt` holds 20 fixed continuation prompts, one per line. Convention: no comments and no header; every line is a prompt, and the prompt id is its 1-based line number. The set includes 4 short prompts (3-5 words) and 4 that begin mid-sentence.

After each eval, generate from every prompt with these fixed settings so samples are comparable across runs:

| Setting | Value |
|---------|-------|
| Sampled decoding | temperature 0.8, top-k 40 |
| Max new tokens | 120 |
| Greedy decoding | also generated and saved |
| Seed | 1234 (reset before each prompt) |

Do not change these settings between runs. If they must change, record it and treat earlier samples as not comparable.

## 3. Rubric

`eval/rubric.md` defines a 1-5 fluency rubric (grammar, coherence, repetition, stays-on-prompt). Samples are scored by eye at milestones only: about 10 samples per milestone, not every eval.

## 4. Automatic text-quality stats

Computed on the generated samples for every samples file, sampled and greedy reported separately, by `scripts/run_report.py` (see `docs/run-reports.md` for the report layout). Words are lowercased whitespace-split words of the generated text with `<eos>` removed, not model tokens.

Implemented:

- distinct-1, distinct-2 and distinct-3: unique n-grams / total n-grams in an output, averaged over outputs. (The protocol originally named distinct-1 and distinct-2; distinct-3 is an extra.)
- Repeated 4-gram fraction: for each output, the share of its 4-gram positions whose 4-gram occurs more than once, averaged over outputs. This is not the same as "fraction of samples with any repeated 4-gram".
- Mean length: mean generated length in words (not tokens).
- Hit-eos fraction: fraction of outputs that contain `<eos>`, i.e. ended before the 120-token limit. This is the closest implemented proxy for truncation.
- Looping outputs: count of outputs in which one 4-gram occurs 3 or more times.

Not implemented:

- Sentence-final punctuation fraction (outputs ending in `.`, `!` or `?`).
- Mean length in tokens.

These are cheap warning signs for looping and truncation, not quality scores.

## 5. Recording results

`train.py` saves samples as plain text (`samples_stepNNNN.txt`: prompt id, prompt, sampled output, greedy output) inside the run folder, and each eval's loss and perplexity in `metrics.jsonl`. The text stats above, the evals table and a blank rubric scoring sheet are in the run's `report.html` and `report.md`. The folder layout is in `docs/architecture.md` section 7. To record rubric scores, save a `review.json` in the same run folder (format in `docs/run-reports.md`); the report then shows the mean score and uses it for the Sample quality verdict.
