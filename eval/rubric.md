# Fluency rubric

Score each sample 1-5 on four criteria. Scores are for human eyeballing and are subjective; use them to compare runs, not as absolute measurements. Score the sampled output (temperature 0.8, top-k 40) unless noted.

## Grammar

- 1: Mostly broken; word salad or fragments, little recognizable syntax.
- 2: Frequent errors; many sentences are malformed or hard to parse.
- 3: Readable but with regular errors (agreement, tense, missing words).
- 4: Mostly correct; only an occasional slip.
- 5: Clean throughout, like a children's book.

## Coherence over 2-3 sentences

- 1: Sentences have no relation to each other.
- 2: Same topic words, but events or subjects jump around.
- 3: Neighboring sentences connect, but characters or details drift.
- 4: A clear thread across 2-3 sentences, with minor inconsistencies.
- 5: A consistent, sensible mini-story across several sentences.

## Repetition and looping

Higher is better (less repetition).

- 1: Stuck in a loop or repeating one phrase or token.
- 2: Obvious repeated sentences or phrases dominate.
- 3: Some noticeable repeated phrases, but the text moves forward.
- 4: Rare, minor repeats (a repeated opener, a reused word).
- 5: No distracting repetition.

## Stays on prompt

- 1: Ignores the prompt entirely.
- 2: Loosely related; characters or setting from the prompt are dropped.
- 3: Uses the prompt's characters or setting but wanders off the situation.
- 4: Continues naturally, with a small drift.
- 5: A natural continuation that keeps the prompt's characters, setting and tone (for mid-sentence prompts, the first words complete the sentence correctly).

## Scoring sheet template

One line per sample:

```
prompt id | grammar | coherence | repetition | on-prompt | notes
```

Example: `7 | 4 | 3 | 5 | 4 | drifts to a new character in sentence 3`

## Guidance

- Score about 10 samples by eye at each milestone, not at every eval. Use the same prompt ids across milestones where possible (include some short and some mid-sentence prompts).
- Read the sample first, then score all four criteria without looking at earlier scores.
- Record the run, step and decoding mode on the sheet.
- Do not treat a one-point difference on a few samples as meaningful; look at averages and the validation loss alongside.
