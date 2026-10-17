# TinyStories dataset card

## Source

- Hugging Face dataset: `roneneldan/TinyStories` (https://huggingface.co/datasets/roneneldan/TinyStories)
- Paper: "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" (Eldan and Li, arXiv:2305.07759)
- License: CDLA-Sharing-1.0 (Community Data License Agreement, Sharing). Verified from the `license:` field in the dataset card YAML header on Hugging Face (README fetched 2026-10-17). The card body contains no further license text. Read the CDLA-Sharing-1.0 terms itself before redistributing the data or derivatives.
- Size and splits (from the Hugging Face dataset page and `load_dataset_builder` metadata): train 2,119,719 stories (about 2.0 GB of text), validation 21,990 stories (about 17.5 MB). The page lists about 7.62 GB total because it includes auto-converted Parquet copies.
- Column: a single `text` string per story.

## What the data is

Synthetic short stories written by GPT-3.5 and GPT-4, prompted to use a small vocabulary a young child would understand. Stories are a few paragraphs of simple past-tense narrative.

## Filtering applied

Done by `scripts/get_tinystories.py`, streaming the dataset in its original order (nothing random):

1. Strip leading/trailing whitespace.
2. Replace curly quotes and apostrophes with straight ones; collapse runs of spaces/tabs; drop blank lines so paragraphs are separated by a single newline.
3. Drop stories that are empty, shorter than 200 characters, or contain any non-ASCII character after normalization.
4. Drop exact duplicate stories (hash of the cleaned text, within each output file).

Counts from the run with default settings:

| File | Source split | Stories kept | Dropped | Words | Est. tokens (words x 1.3) | Words per story (min / median / max) |
|------|--------------|--------------|---------|-------|---------------------------|--------------------------------------|
| `data/tinystories_train_seed.jsonl` | train | 50,000 | empty 12, short 0, non-ASCII 3,061, duplicate 100 | 8,735,330 | about 11.4M | 43 / 152 / 878 |
| `data/tinystories_val_2000.jsonl` | validation | 2,000 | empty 0, short 0, non-ASCII 116, duplicate 0 | 309,025 | about 0.40M | 41 / 146 / 715 |

## Files and intended use

Each line is a JSON object `{"text": "<story>"}`.

- `data/tinystories_train_seed.jsonl`: seed training subset for fast iteration (issue #2 targets 10-20M tokens; this is about 11.4M estimated tokens). Scale up later by passing a larger `--num-train-stories`.
- `data/tinystories_val_2000.jsonl`: the fixed validation subset: the first 2,000 cleaned stories of the dataset's own validation split. This is the subset `eval/README.md` refers to. It is never trained on and never used for tokenizer fitting.

## Regenerate

```
.venv/bin/python scripts/get_tinystories.py
```

Options: `--out-dir` (default `data`), `--num-train-stories` (default 50000), `--num-val-stories` (default 2000). The script streams, so the full train split is not downloaded to disk. Re-running overwrites the same outputs with identical contents.

## Limits

- Children's-story style with a deliberately simple vocabulary and repetitive story structure.
- Not conversational and not factual: no dialogue-with-a-user format, no world knowledge to speak of. Models trained on it learn fluent short-story English, not question answering.
- Synthetic text from other LLMs, so it inherits their quirks and any biases.
- About 6% of stories were dropped for non-ASCII characters, so rare characters and some story types are underrepresented.

## Data files are never committed

`data/*` is gitignored (except `data/.gitkeep`). Only the script and this card are tracked.
