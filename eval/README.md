# Eval

How Ascent's first model is evaluated. The goal and success criteria live in `docs/goal.md`. Everything here is for a text-continuation story model trained on TinyStories, not a chatbot.

## 1. Held-out validation (loss and perplexity)

- Data: the TinyStories validation split as provided by the dataset. It is split by story, never by sentence, so no story is partly in training and partly in validation.
- The validation split is never trained on, including for tokenizer fitting or data-mixing decisions.
- Each eval uses a fixed subset: the first 2,000 stories of the validation split. The data script from issue #2 fixes the exact subset so every run measures the same text.
- Metrics: mean cross-entropy loss per token, and perplexity = exp(loss). Computed at every eval.

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

Computed on the generated samples at each eval (sampled and greedy reported separately):

- distinct-1 and distinct-2: unique unigrams / total unigrams, and unique bigrams / total bigrams, per sample, averaged.
- Repeated 4-gram: fraction of samples that contain any 4-gram more than once.
- Mean length: mean generated length in tokens.
- Sentence-final punctuation: fraction of samples whose generated text ends in `.`, `!` or `?`. Samples cut off by the token limit count against this, so read it alongside mean length.

These are cheap warning signs for looping and truncation, not quality scores.

## 5. Recording results

At each eval step, save the samples as plain text (prompt id, prompt, sampled output, greedy output) inside the run folder, next to that eval's loss, perplexity and the stats above. The run folder layout and file names are defined by the run-records issue (#6). Rubric score sheets for milestones go in the same run folder.
