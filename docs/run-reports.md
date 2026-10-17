# Reading a run report

Every training run gets a report, written by `scripts/run_report.py`. `train.py` calls it at the very end of a run (also after Ctrl-C; a report failure only prints a warning). Pass `--no-report` to skip it.

Run folders and reports are local and gitignored (`runs/`). Reports are derived files: regenerate any time.

```
.venv/bin/python scripts/run_report.py runs/<id>     # report.html + report.md for one run, then the index
.venv/bin/python scripts/run_report.py --index       # only rebuild runs/index.html
```

Files: `runs/<id>/report.html` (self-contained, charts embedded), `runs/<id>/report.md` (compact text version for scanning by a human or an LLM), `runs/index.html` (all runs, newest first, best val loss highlighted). This is where the results for the success criteria in `docs/goal.md` are read from.

## Sections

1. **Header**: status, git commit, device and dtype, parameter count, config highlights, estimated wall time (summed from step times, eval and sample times).
2. **Key numbers**: tokens trained and epochs over the 10.69M-token train set; train loss first / min / final; best and final val loss with perplexity and the step of the best; val-minus-train gap (final val minus the mean of the last 5 logged train losses); median tok/s and the early-to-late throughput drop; peak MPS driver memory.
3. **Charts**: train and val loss, learning rate, grad norm (with the 3x median line), throughput with a rolling mean, step time.
4. **Observations**: the automatic flags below, plus notes about missing or unreadable data.
5. **Evals**: val loss and perplexity per eval, best row highlighted.
6. **Samples**: per samples file, text stats for sampled and greedy outputs, every prompt with its outputs (collapsed), and a blank 1-5 scoring sheet (grammar, coherence, repetition, on-prompt, matching `eval/rubric.md`). The sheet is for filling in by hand and is not saved.
7. **Comparison with the previous run**: best val, tok/s and parameter deltas, config differences, and a val-loss overlay of the last 5 runs.

Incomplete runs (crash, truncated last line, no evals, no samples) still produce a report, with notes in place of the missing parts.

## Status

| Status | Meaning |
|--------|---------|
| `complete` | last step reached `max_steps` and a final eval ran |
| `interrupted` | stopped early (Ctrl-C or non-finite loss) but a final eval ran |
| `incomplete` | no final eval: crashed or killed |

## Automatic flags

Simple rules, meant as pointers and not verdicts. Levels: BAD, WARN, INFO.

| Flag | Level | Rule |
|------|-------|------|
| non-finite loss | BAD | any NaN or inf train loss |
| loss spike | WARN | logged train loss above 1.5x the median of the previous 5 logged values |
| grad-norm spike | WARN | grad norm above 3x the run median |
| throughput drop | WARN | median tok/s of the last 10% of train records is more than 10% below the first 10% (first record skipped), likely thermal throttling. Needs at least 10 records after the first, otherwise an INFO note |
| val loss rising | WARN | last eval above 1.01x the best val loss and above the previous eval: possible overfitting |
| plateau | INFO | last eval improved by less than 1% over the previous eval |
| still decreasing | INFO | last eval improved by 1% or more: likely undertrained |
| only one eval | INFO | no trend can be judged |
| train/val gap | WARN above 0.5 | val minus recent train loss; INFO when small or when val is more than 0.2 below train |
| memory | WARN | peak `mps_driver_mb` above 75% of 8 GB |
| run status | WARN | status is not `complete` |

## Sample text stats

Computed by `scripts/run_report.py` from `samples_stepNNNN.txt`, separately for sampled and greedy outputs, over the 20 prompts. Words are lowercased whitespace-split tokens of the generated text (not model tokens), with `<eos>` removed. Each stat is averaged over outputs.

| Stat | Definition |
|------|------------|
| distinct-1/2/3 | unique n-grams divided by total n-grams in an output |
| repeated 4-gram frac | share of an output's 4-gram positions whose 4-gram occurs more than once |
| mean words | mean generated length in words |
| hit eos | fraction of outputs containing `<eos>` (the story ended before the 120-token limit) |
| looping outputs | outputs in which some 4-gram occurs 3 or more times |

See `eval/README.md` section 4 for which protocol stats are implemented.
