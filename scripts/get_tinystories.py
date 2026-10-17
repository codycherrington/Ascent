"""Stream TinyStories from Hugging Face and write cleaned JSONL files.

Outputs (one {"text": story} object per line, under --out-dir):
  tinystories_train_seed.jsonl  first N cleaned stories of the train split
  tinystories_val_2000.jsonl    first M cleaned stories of the validation split
                                (the fixed validation subset used by eval/README.md)

Cleaning: strip, normalize curly quotes, collapse whitespace (paragraphs kept
as single newlines), drop empty / short / non-ASCII stories and exact duplicates.
The script is deterministic and idempotent: it streams in dataset order and
overwrites its outputs. The full train split is never downloaded to disk.

Usage: .venv/bin/python scripts/get_tinystories.py
"""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from datasets import load_dataset

DATASET = "roneneldan/TinyStories"
MIN_CHARS = 200
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def clean(text):
    """Return the normalized story, or the reason string prefixed by '!' if dropped."""
    text = text.translate(QUOTES).strip()
    lines = (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n"))
    text = "\n".join(ln for ln in lines if ln)
    if not text:
        return "!empty"
    if len(text) < MIN_CHARS:
        return "!short"
    if not text.isascii():
        return "!non_ascii"
    return text


def prepare(split, limit, path):
    """Stream `split`, write the first `limit` kept stories to `path`, print a summary."""
    drops = {"empty": 0, "short": 0, "non_ascii": 0, "duplicate": 0}
    seen, lengths = set(), []
    stream = load_dataset(DATASET, split=split, streaming=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in stream:
            if len(lengths) >= limit:
                break
            text = clean(row["text"])
            if text.startswith("!"):
                drops[text[1:]] += 1
                continue
            digest = hashlib.sha1(text.encode()).digest()
            if digest in seen:
                drops["duplicate"] += 1
                continue
            seen.add(digest)
            f.write(json.dumps({"text": text}) + "\n")
            lengths.append(len(text.split()))
    words = sum(lengths)
    print(f"[{split}] wrote {path}")
    print(f"  stories kept: {len(lengths)}")
    print(f"  dropped: {drops} (total {sum(drops.values())})")
    print(f"  words: {words} (rough tokens, words*1.3: {int(words * 1.3)})")
    print(f"  story length in words: min {min(lengths)}, "
          f"median {int(statistics.median(lengths))}, max {max(lengths)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default="data")
    p.add_argument("--num-train-stories", type=int, default=50000)
    p.add_argument("--num-val-stories", type=int, default=2000)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepare("train", args.num_train_stories, out / "tinystories_train_seed.jsonl")
    prepare("validation", args.num_val_stories, out / "tinystories_val_2000.jsonl")


if __name__ == "__main__":
    main()
