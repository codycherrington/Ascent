"""Download TinyDialogues (styfeng/TinyDialogues) into data/tinydialogues/ and print a summary.

Fetches individual_age_data.zip, which holds exact per-age files
(tinydialogue_age-{2,5,10,15}_{train,val}.txt). Those files, not the age-ordered
combined file, are the reliable way to know a conversation's age. Idempotent: files
already present are not downloaded again. --all also fetches the two age-ordered text files.

Usage: .venv/bin/python scripts/get_tinydialogues.py [--out-dir data/tinydialogues] [--all]
"""
import argparse
import collections
import os
import statistics
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.prepare_chat_data import parse_conversation, to_sides  # noqa: E402

REPO = "styfeng/TinyDialogues"
ZIP_NAME = "individual_age_data.zip"
ORDERED = ["tinydialogue_train_ordered.txt", "tinydialogue_val_ordered.txt"]
AGES = [2, 5, 10, 15]


def download(out_dir, names):
    from huggingface_hub import hf_hub_download
    for name in names:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            print(f"present: {path}")
        else:
            hf_hub_download(REPO, name, repo_type="dataset", local_dir=out_dir)
            print(f"downloaded: {path}")


def summarize(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        for age in AGES:
            speakers, n_speakers = collections.Counter(), collections.Counter()
            lines = bad = one_side = merged = 0
            turn_words, turns_per_conv = [], []
            for part in ("train", "val"):
                with z.open(f"tinydialogue_age-{age}_{part}.txt") as f:
                    for raw in f:
                        line = raw.decode("utf-8").strip()
                        if not line:
                            continue
                        lines += 1
                        turns = parse_conversation(line)
                        if turns is None:
                            bad += 1
                            continue
                        names = {s.lower() for s, _ in turns}
                        n_speakers[min(len(names), 4)] += 1
                        speakers.update(s for s, _ in turns)
                        turn_words.extend(len(t.split()) for _, t in turns)
                        turns_per_conv.append(len(turns))
                        sides = to_sides(turns)
                        one_side += sides is None
            q = statistics.quantiles(turn_words, n=20)
            print(f"\nage {age}: {lines:,} conversations, {bad:,} unusable (non-ASCII or <2 turns), "
                  f"{one_side:,} single-side")
            print(f"  speakers per conversation (4 = 4+): {dict(sorted(n_speakers.items()))}")
            print(f"  turns/conversation: median {statistics.median(turns_per_conv):.0f}, "
                  f"mean {statistics.mean(turns_per_conv):.1f}")
            print(f"  words/turn: mean {statistics.mean(turn_words):.1f}, median {statistics.median(turn_words):.0f}, "
                  f"p95 {q[18]:.0f}, max {max(turn_words)}")
            print(f"  top speakers: {speakers.most_common(8)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default="data/tinydialogues")
    p.add_argument("--all", action="store_true", help="also download the age-ordered combined txt files")
    p.add_argument("--no-summary", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    download(args.out_dir, [ZIP_NAME] + (ORDERED if args.all else []))
    if not args.no_summary:
        summarize(os.path.join(args.out_dir, ZIP_NAME))


if __name__ == "__main__":
    main()
