# TinyDialogues dataset card

## Source

- Hugging Face dataset: `styfeng/TinyDialogues` (https://huggingface.co/datasets/styfeng/TinyDialogues)
- License: MIT. The conversations were written by GPT-4, so the OpenAI terms of use may apply to uses of the output (for example training competing models); check them before redistributing weights or data.
- Size: 129,732 conversations. Files on the Hub: `individual_age_data.zip`, `tinydialogue_train_ordered.txt`, `tinydialogue_val_ordered.txt`.
- Each conversation is one line: `**Mom**: text \n\n **Child**: text ... <|endoftext|>`. The `\n\n` are the literal characters backslash and `n`, not newlines. Some turns carry a stage direction (`**Child** (jumping):`), a few contain `---` or a single literal `\n`, and about 1% have non-ASCII characters.

## Age slices

`individual_age_data.zip` holds exact per-age files (`tinydialogue_age-{2,5,10,15}_{train,val}.txt`), so ages are read from there rather than inferred from the age-ordered combined file. The slice counts add up to the full 129,732.

| Age | Conversations | Unusable (non-ASCII or <2 turns) | Turns/conversation (median) | Words/turn (mean) | Main speakers |
|-----|---------------|----------------------------------|-----------------------------|-------------------|---------------|
| 2 | 43,697 | 96 | 16 | 8.4 | Toddler, Mom, Dad, Babysitter, Older Sibling |
| 5 | 33,248 | 25 | 14 | 13.5 | Child, Mom, Teacher, Babysitter, Dad, Grandparent, Neighbor |
| 10 | 27,198 | 54 | 14 | 17.4 | Child, Teacher, Neighbor, Dad, Grandparent, Mom |
| 15 | 25,589 | 139 | 13 | 19.9 | Teenager, Teacher, Tutor, Coach, Neighbor, Mom, Dad |

Age 2 is telegraphic baby-talk ("No! Me do it!") and is excluded by default. Roughly half of all conversations have 3 speakers (age 5: 16,535 with 2, 16,455 with 3, 233 with 4+), so skipping multi-speaker conversations would lose half the data.

## Chat format

`scripts/prepare_chat_data.py` turns each conversation into plain text, one turn per line, ending with `<eos>`:

```
You: Look, I made a circle with my Cheerios!
Ascent: That's cool! Can I make one too?
<eos>
```

Cleaning: `**Name**:` markup, stage directions and `<|endoftext|>` are removed; literal `\n` and `---` become spaces; curly quotes and dashes become ASCII; whitespace is collapsed; punctuation is kept; a quote pair wrapping a whole turn is removed. Conversations with non-ASCII characters left or fewer than 2 turns are dropped (218 of 86,035 for ages 5/10/15).

## Role assignment

Speakers are collapsed into two sides: side A is the first speaker, side B is everyone else. Consecutive turns on the same side are merged into one turn (155,828 merges in the default build, mostly from 3-speaker conversations). A seeded per-conversation coin flip (seed plus the conversation key) decides whether side A is `You` or `Ascent`, so Ascent plays either role. When side A is Ascent, its opening turn is skipped because windows start at a `You:` turn (42,913 skipped).

## Length handling

The model context is 256 tokens and conversations average about 300 tokens. Each conversation is cut into non-overlapping windows of at most 256 tokens including `<eos>`; each window starts at a `You:` turn and ends on an `Ascent:` turn, with at least 2 turns. A You/Ascent pair that alone does not fit is skipped. In the default build 81,578 of 85,708 used conversations were cut into several windows and 109 produced no window. Every window gets its own `<eos>`. Windows are built from per-turn token ids; the script checks a 1% sample against whole-text encoding (0 mismatches of 1,547).

## Mixing and validation split

TinyStories (`data/tinystories_train_seed.jsonl`, plain stories separated by `<eos>`) is mixed in at `--story-frac` (default 12%) of tokens, interleaved as the conversations stream. Ages are read round-robin so a `--max-tokens` cut does not remove one age. The validation split is by document (a conversation or a story), deterministic from a hash of the seed and the document key, `--val-frac` default 1%. The official val files are pooled into the same pool before this split.

## Measured stats (default build: ages 5, 10, 15, seed 1337)

| Item | Value |
|------|-------|
| Conversations used | 85,708 of 86,035 read |
| Chat windows | 154,764 (5.9 turns each) |
| TinyDialogues tokens | 27,589,515 |
| TinyStories tokens | 3,762,411 (12.0%) |
| Total tokens | 31,351,926 (below the 40M cap, so the cap did not bind) |
| Train / val tokens | 31,034,349 / 317,577 (1.01%) |
| Tokens per word | chat 1.513 (words counted including the `You:`/`Ascent:` labels), stories 1.227 |
| Longest chat window | 256 tokens |
| Peak memory / wall time | about 225 MB / 23 s |

## Regenerate

```
.venv/bin/python scripts/get_tinydialogues.py        # about 150 MB download, idempotent
.venv/bin/python scripts/prepare_chat_data.py        # writes data/chat_train.bin, data/chat_val.bin
```

Options of `prepare_chat_data.py`: `--ages` (default `5,10,15`; add `2` for baby-talk), `--max-tokens`, `--story-frac`, `--val-frac`, `--seed`, `--block-size`, `--train-out`, `--val-out`, `--zip`, `--stories`, `--tokenizer`. The tokenizer is the existing `data/tokenizer.json` and is never retrained.

## Limits

- Dialogues are GPT-4 written and idealised: friendly, on-topic and mostly about children's everyday life. They are not factual Q&A and not an assistant persona.
- The `You`/`Ascent` roles are arbitrary labels over parent/child, teacher/child and similar pairs, so Ascent will sometimes answer as a child.
- Windows cut mid-conversation lose the opening context; about half of conversations lose their first turn.

## Data files are never committed

`data/*` is gitignored. Only the scripts and this card are tracked.
