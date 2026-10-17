# Tokenizer

Ascent uses a byte-level BPE tokenizer (Hugging Face `tokenizers`), implemented in `tokenizer.py` at the repo root.

## Why byte-level BPE, 8192 vocab

- Byte-level means every string can be encoded; there are no out-of-vocabulary words (the old word-level tokenizer's main flaw).
- The corpus is TinyStories: simple English with a small working vocabulary. A large vocab is not needed.
- For a ~30M model the embedding table is a real cost: vocab x embedding dim, e.g. 8192 x 384 = 3.1M params (a 29k word vocab was about 11M).
- Ids fit in uint16 (vocab < 65536), so tokenized data is stored as flat uint16 arrays.

## Special tokens

Fixed ids, the first four in the vocab: `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3. `train` asserts this. In the encoded `.bin` files each story is followed by `<eos>`.

## Usage

```
.venv/bin/python tokenizer.py train     # fit on data/tinystories_train_seed.jsonl only -> data/tokenizer.json
.venv/bin/python tokenizer.py encode    # -> data/train.bin, data/val.bin (uint16)
.venv/bin/python -m unittest discover -s tests -v
```

`train` takes `--vocab-size` (default 8192). The tokenizer is never fit on validation data (see `eval/README.md`). Importable helpers: `load_tokenizer(path)`, `encode(tok, text)`, `decode(tok, ids)`.

`data/tokenizer.json`, `train.bin` and `val.bin` are gitignored data artifacts, regenerated from the seed file. The run-records work will copy `tokenizer.json` into each run folder.

## Measured stats (default run)

| Item | Value |
|------|-------|
| Vocab size | 8192 |
| train.bin | 50,000 stories, 10,693,222 tokens (20 MB) |
| val.bin | 2,000 stories, 373,351 tokens (729 KB) |
| Tokens per word (whitespace words) | 1.224 train, 1.208 val |
| Validation round-trip | 2,000 / 2,000 stories exact, 0 `<unk>` |
| Time | train about 2.3 s, encode about 3.6 s |
