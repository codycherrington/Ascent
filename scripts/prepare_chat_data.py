"""Build the Stage-1 chat fine-tuning data: data/chat_train.bin and data/chat_val.bin.

Sources: TinyDialogues (ages 5, 10, 15 by default) read straight from
data/tinydialogues/individual_age_data.zip, plus TinyStories mixed in by token fraction.

Chat format (plain text, one turn per line; a conversation or window ends with <eos>):

    You: Look, I have raisins for my snack!
    Ascent: Yummy, can I have some?
    <eos>

Role rule: the speakers of a conversation are collapsed into two sides. Side A is the
first speaker; side B is everyone else (so 3+ speaker conversations are kept, and
consecutive turns on the same side are merged into one turn). A seeded per-conversation
coin flip decides whether side A is "You" or "Ascent". Windows start at a "You:" turn and
end on an "Ascent:" turn, so a leading "Ascent:" turn is skipped when side A is Ascent.

Length: windows hold at most --block-size tokens including the <eos>; conversations that
do not fit are cut at turn boundaries into several windows. Windows under 2 turns are dropped.

Usage: .venv/bin/python scripts/prepare_chat_data.py   (see --help)
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import re
import resource
import sys
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizer import decode, load_tokenizer  # noqa: E402

ZIP_PATH = "data/tinydialogues/individual_age_data.zip"
STORIES_PATH = "data/tinystories_train_seed.jsonl"
EOT = "<|endoftext|>"
TURN_RE = re.compile(r"(?:^|\\n|\s)\*\*([^*]{1,40}?)\*\*(?:\s*\([^)]*\))?\s*:")
CHARS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", "…": "..."})


def clean_turn(text):
    """Normalise one turn: literal \\n and '---' become spaces, curly quotes straight, whitespace
    collapsed, a quote pair wrapping the whole turn removed."""
    text = text.replace("\\n", " ").replace("---", " ").translate(CHARS)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 1 and text[0] == '"' and text[-1] == '"' and text.count('"') == 2:
        text = text[1:-1].strip()
    return text


def parse_conversation(line):
    """Parse one raw line into [(speaker, text), ...]; None if unusable (non-ASCII, <2 turns)."""
    line = line.replace(EOT, "").strip()
    marks = list(TURN_RE.finditer(line))
    turns = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(line)
        text = clean_turn(line[m.end():end])
        if text:
            turns.append((" ".join(m.group(1).split()), text))
    if len(turns) < 2 or not all(t.isascii() for _, t in turns):
        return None
    return turns


def to_sides(turns):
    """Collapse speakers to two sides (0 = first speaker, 1 = everyone else), merging
    consecutive same-side turns. Returns [(side, text)] or None if only one side talks."""
    first = turns[0][0].lower()
    out = []
    for spk, text in turns:
        side = 0 if spk.lower() == first else 1
        if out and out[-1][0] == side:
            out[-1] = (side, out[-1][1] + " " + text)
        else:
            out.append((side, text))
    return out if len(out) >= 2 else None


def assign_roles(sides, key, seed):
    """Seeded coin flip per conversation: which side is 'You'. Returns [(role, text)]."""
    a_is_you = random.Random(f"{seed}:{key}").random() < 0.5
    return [("You" if (side == 0) == a_is_you else "Ascent", text) for side, text in sides]


def format_turn(role, text):
    return f"{role}: {text}\n"


def split_windows(roles, lens, budget):
    """Cut a role-alternating conversation into non-overlapping windows of turn indices
    (start, end), end exclusive. Each window starts at a You turn, ends on an Ascent turn,
    has >= 2 turns and sum(lens) <= budget. A You/Ascent pair that cannot fit is skipped."""
    windows, i, n = [], 0, len(roles)
    while i < n:
        if roles[i] != "You":
            i += 1
            continue
        j, total, last_ok = i, 0, None
        while j < n and total + lens[j] <= budget:
            total += lens[j]
            j += 1
            if roles[j - 1] == "Ascent":
                last_ok = j
        if last_ok is None:      # even the first You/Ascent pair is too long
            i += 1
            continue
        windows.append((i, last_ok))
        i = last_ok
    return windows


def is_val(key, seed, val_frac):
    h = hashlib.sha1(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < val_frac


def read_zip_lines(zip_path, age):
    """Yield (key, raw_line) for one age, official train then val file, in file order."""
    with zipfile.ZipFile(zip_path) as z:
        for part in ("train", "val"):
            with z.open(f"tinydialogue_age-{age}_{part}.txt") as f:
                for n, raw in enumerate(f):
                    line = raw.decode("utf-8").strip()
                    if line:
                        yield f"{age}:{part}:{n}", line


def interleave(zip_path, ages):
    """Round-robin over ages so a --max-tokens cut does not drop a whole age."""
    gens = [read_zip_lines(zip_path, a) for a in ages]
    for group in itertools.zip_longest(*gens):
        for item in group:
            if item is not None:
                yield item


def read_stories(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            yield f"story:{n}", json.loads(line)["text"]


class Writer:
    """Append uint16 ids to train/val bins, tracking token counts and max length per source."""

    def __init__(self, train_path, val_path):
        self.f = {"train": open(train_path, "wb"), "val": open(val_path, "wb")}
        self.tokens = {}
        self.max_len = {}

    def write(self, split, source, ids):
        np.array(ids, dtype=np.uint16).tofile(self.f[split])
        self.tokens[(split, source)] = self.tokens.get((split, source), 0) + len(ids)
        self.max_len[source] = max(self.max_len.get(source, 0), len(ids))

    def total(self):
        return sum(self.tokens.values())

    def by_source(self, source):
        return sum(v for (_, s), v in self.tokens.items() if s == source)

    def close(self):
        for f in self.f.values():
            f.close()


def chunks(it, size):
    while True:
        batch = list(itertools.islice(it, size))
        if not batch:
            return
        yield batch


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--zip", default=ZIP_PATH)
    p.add_argument("--stories", default=STORIES_PATH)
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--train-out", default="data/chat_train.bin")
    p.add_argument("--val-out", default="data/chat_val.bin")
    p.add_argument("--ages", default="5,10,15", help="comma list; add 2 for the telegraphic age-2 data")
    p.add_argument("--max-tokens", type=int, default=40_000_000, help="cap on total tokens written")
    p.add_argument("--story-frac", type=float, default=0.12, help="fraction of tokens from TinyStories")
    p.add_argument("--val-frac", type=float, default=0.01, help="fraction of documents held out, by hash")
    p.add_argument("--block-size", type=int, default=256, help="max tokens per window incl. <eos>")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--chunk", type=int, default=500, help="conversations encoded per batch")
    args = p.parse_args()

    t0 = time.time()
    ages = [int(a) for a in args.ages.split(",")]
    tok = load_tokenizer(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    budget = args.block_size - 1
    w = Writer(args.train_out, args.val_out)
    stories = read_stories(args.stories)
    debt = 0.0                       # story tokens still owed to hit --story-frac
    ratio = args.story_frac / (1 - args.story_frac)
    st = dict(convs=0, bad_parse=0, one_side=0, merged_turns=0, split=0, dropped=0, windows=0,
              lead_skipped=0, turns=0, words=0, story_words=0, mismatch=0, checked=0)
    kept_per_age = {a: 0 for a in ages}
    samples = []
    rng = random.Random(args.seed)

    def emit_stories():
        nonlocal debt
        while debt > 0:
            try:
                key, text = next(stories)
            except StopIteration:
                debt = 0
                return
            ids = tok.encode(text).ids + [eos]
            w.write("val" if is_val(key, args.seed, args.val_frac) else "train", "tinystories", ids)
            st["story_words"] += len(text.split())
            debt -= len(ids)

    done = False
    for batch in chunks(interleave(args.zip, ages), args.chunk):
        convs = []                   # (key, age, [(role, text)])
        for key, line in batch:
            turns = parse_conversation(line)
            if turns is None:
                st["bad_parse"] += 1
                continue
            sides = to_sides(turns)
            if sides is None:
                st["one_side"] += 1
                continue
            st["merged_turns"] += len(turns) - len(sides)
            convs.append((key, int(key.split(":")[0]), assign_roles(sides, key, args.seed)))
        flat = [format_turn(r, t) for _, _, rt in convs for r, t in rt]
        enc = iter(tok.encode_batch(flat))
        for key, age, rt in convs:
            ids_per_turn = [next(enc).ids for _ in rt]
            roles = [r for r, _ in rt]
            wins = split_windows(roles, [len(x) for x in ids_per_turn], budget)
            st["convs"] += 1
            if not wins:
                st["dropped"] += 1
                continue
            st["split"] += len(wins) > 1 or wins[0] != (0, len(rt))
            st["lead_skipped"] += wins[0][0]
            split = "val" if is_val(key, args.seed, args.val_frac) else "train"
            for s, e in wins:
                ids = [i for x in ids_per_turn[s:e] for i in x] + [eos]
                text = "".join(format_turn(r, t) for r, t in rt[s:e])
                if rng.random() < 0.01:   # per-turn encoding must equal whole-text encoding
                    st["checked"] += 1
                    st["mismatch"] += tok.encode(text).ids + [eos] != ids
                w.write(split, "tinydialogues", ids)
                st["windows"] += 1
                st["turns"] += e - s
                st["words"] += len(text.split())
                debt += len(ids) * ratio
                if len(samples) < 3 and split == "train" and rng.random() < 0.0005:
                    samples.append(ids)
            kept_per_age[age] += 1
            emit_stories()
            if w.total() >= args.max_tokens:
                done = True
                break
        if done:
            break
    w.close()

    chat_tok, story_tok = w.by_source("tinydialogues"), w.by_source("tinystories")
    val_tok = sum(v for (s, _), v in w.tokens.items() if s == "val")
    total = chat_tok + story_tok
    print(f"ages {ages} | seed {args.seed} | max-tokens {args.max_tokens:,} | story-frac {args.story_frac}")
    print(f"conversations read: {st['convs'] + st['bad_parse'] + st['one_side']:,} | unusable (non-ASCII/<2 turns): "
          f"{st['bad_parse']:,} | single-side: {st['one_side']:,}")
    print(f"conversations used {st['convs'] - st['dropped']:,} | cut into several windows: {st['split']:,} | "
          f"dropped (no window fits): {st['dropped']:,} | used per age: {kept_per_age}")
    print(f"turns merged by side collapsing: {st['merged_turns']:,} | leading Ascent turns skipped: {st['lead_skipped']:,}")
    print(f"chat windows: {st['windows']:,} ({st['turns']:,} turns, {st['turns'] / max(1, st['windows']):.1f}/window)")
    print(f"tokens: tinydialogues {chat_tok:,} | tinystories {story_tok:,} ({story_tok / max(1, total):.1%}) | total {total:,}")
    print(f"tokens/word: chat {chat_tok / max(1, st['words']):.3f} | stories {story_tok / max(1, st['story_words']):.3f}")
    print(f"val: {val_tok:,} tokens ({val_tok / max(1, total):.2%}) -> {args.val_out}; "
          f"train {total - val_tok:,} -> {args.train_out}")
    print(f"per-turn vs whole-text encoding mismatches: {st['mismatch']} of {st['checked']} sampled windows")
    longest = w.max_len.get("tinydialogues", 0)
    print(f"longest chat window incl. <eos>: {longest} tokens (limit {args.block_size}) -> "
          f"{'OK' if longest <= args.block_size else 'FAIL'}")
    for path in (args.train_out, args.val_out):
        data = np.fromfile(path, dtype=np.uint16)
        print(f"{path}: {len(data):,} tokens, max id {int(data.max())} (< vocab {tok.get_vocab_size()}), "
              f"{int((data == eos).sum()):,} <eos>")
        del data
    for n, ids in enumerate(samples, 1):
        print(f"\n--- sample {n} ({len(ids)} tokens) ---\n{decode(tok, ids)}")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20   # bytes on macOS
    print(f"\npeak RSS {peak:.0f} MB | wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
