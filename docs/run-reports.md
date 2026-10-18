# Reading a run report

Every training run gets a report, written by `scripts/run_report.py`. `train.py` calls it at the very end of a run (also after Ctrl-C; a report failure only prints a warning). Pass `--no-report` to skip it.

Run folders and reports are local and gitignored (`runs/`). Reports are derived files: regenerate any time. They are produced by plain code (rules and thresholds), not by AI.

```
.venv/bin/python scripts/run_report.py runs/<id>     # report.html + report.md for one run, then the index
.venv/bin/python scripts/run_report.py --index       # only rebuild runs/index.html
```

Files: `runs/<id>/report.html` (self-contained, charts embedded), `runs/<id>/report.md` (compact text version for scanning by a human or an LLM), `runs/index.html` (all runs, newest first, model quality and run health verdict columns, best val loss highlighted). This is where the results for the success criteria in `docs/goal.md` are read from.

## Layout

`report.html`, top to bottom:

1. Title, run id, status.
2. Two badges side by side (stacked on narrow screens): Model quality and Run health, each GOOD / MEDIUM / FAIL with a one-line summary of what is not GOOD.
3. Scorecard: 6 tiles grouped under their badge (Model quality: Learning, Generalization, Sample quality; Run health: Stability, Speed & thermals, Completeness), each coloured with the verdict word, the category and one key number.
4. Three short blocks: What went well, What went wrong, Proposed fixes (at most 5 one-line bullets each, most important first), then the suggested next run (text only, never executed).
5. Charts: train and val loss, learning rate, grad norm, throughput, step time.
6. Collapsed (closed by default): key numbers, automatic flags and notes, evals, samples per step (stats, every prompt with outputs, blank scoring sheet), comparison with the previous comparable run, config.

The verdict is always written as a word, never colour only. `report.md` opens with the same verdict, scorecard (one line per category), good/bad/fixes and next run, then the details.

## Verdicts

There are two badges, each the worst of its three categories (FAIL beats MEDIUM beats GOOD):

- **Model quality** = worst of Learning, Generalization, Sample quality (is the model any good?).
- **Run health** = worst of Stability, Speed & thermals, Completeness (did the run go well?).

`N/A` tiles (missing data) are ignored; if all three are N/A the badge is N/A. Examples: a run with val loss 2.43 (Learning MEDIUM) that throttled to 1,599 tok/s (Speed FAIL) is Model quality MEDIUM, Run health FAIL. A model with GOOD scores on a run that crashed (Completeness FAIL) is Model quality GOOD, Run health FAIL.

All thresholds live in one dict, `THRESHOLDS` at the top of `scripts/run_report.py`, and are **provisional**: they are placeholders until the milestone thresholds in `docs/goal.md` are filled in after the baseline run. Change them in the dict and in this table together.

| Category | Key number | GOOD | MEDIUM | FAIL | Why |
|----------|------------|------|--------|------|-----|
| Learning | best val loss | <= 2.0 | <= 2.6 | > 2.6 | fluent TinyStories text needs roughly this range; final val more than 5% above best caps GOOD at MEDIUM. "Still falling" = last eval interval improved >= 1% |
| Generalization | final val minus mean of last 5 train losses | <= 0.3 | <= 0.6 | > 0.6 | a growing gap means memorising; negative gaps are normal (train is a noisy minibatch mean) |
| Stability | loss spikes / grad spikes | no non-finite loss, 0 loss spikes (>1.5x median of previous 5), grad spikes <= 2% of logs (>3x median) | <= 2 loss spikes, grad spikes <= 10% | any NaN/inf loss, > 2 loss spikes or grad spikes > 10% | |
| Speed & thermals | median tok/s, early-to-late drop | >= 5000 and drop < 10% | >= 2500 or drop < 30% | below 2500 and drop >= 30% | the M2 Air is fanless: a big drop is thermal throttling. With `swap_used_mb`: swap growth >= 500 MB caps at MEDIUM, >= 2000 MB is FAIL. With `temp_c`: peak >= 85 C caps at MEDIUM |
| Sample quality | sampled distinct-2 (plus greedy loops in the small print) | worst of the metrics below is GOOD | worst is MEDIUM | worst is FAIL | automatic, from the latest samples file; a manual review overrides it (below) |
| Completeness | checks passed of 4 | 4/4: finished, final eval, checkpoints (`ckpt.pt` or `ckpt_best.pt`), samples written | interrupted, or a checkpoint/samples missing | no final eval or crashed (`incomplete`) | |

Sample quality metrics (sampled outputs unless noted): distinct-2 GOOD >= 0.85, MEDIUM >= 0.70; repeated 4-gram share GOOD <= 0.05, MEDIUM <= 0.15; hit-`<eos>` fraction GOOD >= 0.30, MEDIUM >= 0.10; greedy looping outputs GOOD <= 20%, MEDIUM <= 50%, but capped at MEDIUM because greedy loops are expected early in training.

### Manual review (optional)

Put `review.json` in the run folder to score the samples by hand with the rubric in `eval/rubric.md` (1-5; the report has a blank scoring sheet to work from):

```json
{"step": 2441, "scores": {"1": {"grammar": 4, "coherence": 3, "repetition": 5, "on-prompt": 4},
                          "7": {"grammar": 3, "coherence": 3, "repetition": 4, "on-prompt": 4}}}
```

Keys of `scores` are prompt ids; each holds any of the four criteria. The mean of all numeric scores in 1-5 sets the Sample quality tile: GOOD >= 4.0, MEDIUM >= 3.0, else FAIL, replacing the automatic rating. Without the file (or if it cannot be read) the tile shows "manual review: not scored". Regenerate the report after saving it.

### How fixes are chosen

Rules map findings to one-line actions, ordered by importance (a NaN or crashed run first, then thermal throttling and memory, then learning, then the rest). A finding is only listed when its rule fires.

| Finding | Proposed fix |
|---------|--------------|
| non-finite loss | lower `--lr`, keep fp32, resume from the last good checkpoint |
| not complete | `--resume runs/<id>/ckpt.pt` |
| throughput drop > 10% (throttling) | let the Mac cool: stand, cooler room, close apps, cooldown pauses, run in bursts with `--resume` |
| low tok/s from the start | close background apps, check `scripts/benchmark_throughput.py` |
| swap growth (needs `swap_used_mb`) or MPS memory > 75% of 8 GB | close apps, lower `--batch-size` and raise `--grad-accum` |
| temperature high (needs `temp_c`) | cool the machine before the next long run |
| val loss still falling | train more tokens; the data is ~10.7M tokens, so beyond ~1 epoch more data is needed |
| gap too large or val rising | more data or `--dropout 0.1` |
| loss or grad spikes | lower `--lr` or lengthen `--warmup-steps` |
| low distinct-n or repeated 4-grams | review temperature/top-k, train longer |
| greedy loops only | expected early; check the sampled outputs, train longer |
| no review.json | score ~10 samples by hand |

The suggested next run is a command line assembled from the same findings (for example `--max-tokens` doubled when val loss is still falling, `--dropout 0.1`, a halved `--lr`, a smaller batch with more accumulation). It is text only; nothing is executed.

## Optional telemetry

`train.py` may add `swap_used_mb`, `temp_c` and `eta_s` to each `train` record in `metrics.jsonl`. When present, swap growth and peak temperature appear on the Speed & thermals tile and feed the verdict and fixes; when absent they are ignored.

## Status

| Status | Meaning |
|--------|---------|
| `complete` | last step reached `max_steps` and a final eval ran |
| `interrupted` | stopped early (Ctrl-C or non-finite loss) but a final eval ran |
| `incomplete` | no final eval: crashed or killed |

Incomplete runs (crash, truncated last line, no evals, no samples) still produce a report, with notes in place of the missing parts.

## Comparison with a previous run

The report compares against the most recent earlier run that is comparable: its `max_steps` is at least 50% of this run's (so a 10-step smoke run is never diffed against a full run). With none, it says "no comparable previous run". The comparison shows best val, tok/s and parameter deltas, config differences and a val-loss overlay of the last 5 runs.

## Automatic flags

The flags are the raw findings behind the verdicts, in the collapsed "Automatic flags and notes" section. Levels: BAD, WARN, INFO.

| Flag | Level | Rule |
|------|-------|------|
| non-finite loss | BAD | any NaN or inf train loss |
| loss spike | WARN | logged train loss above 1.5x the median of the previous 5 logged values |
| grad-norm spike | WARN | grad norm above 3x the run median |
| throughput drop | WARN | median tok/s of the last 10% of train records is more than 10% below the first 10% (first record skipped), likely thermal throttling. Needs at least 10 records after the first, otherwise an INFO note |
| val loss rising | WARN | last eval more than 1% above the best val loss and above the previous eval: possible overfitting |
| plateau | INFO | last eval improved by less than 1% over the previous eval |
| still decreasing | INFO | last eval improved by 1% or more: likely undertrained |
| only one eval | INFO | no trend can be judged |
| train/val gap | WARN above 0.5 | val minus recent train loss; INFO when small or when val is more than 0.2 below train |
| memory | WARN | peak `mps_driver_mb` above 75% of 8 GB |
| swap / temperature | WARN | swap growth >= 500 MB, peak temperature >= 85 C (only with the optional fields) |
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
