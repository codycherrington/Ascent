"""Byte-level BPE tokenizer for Ascent (Hugging Face `tokenizers`).

Special tokens take the first ids: <pad>=0, <unk>=1, <bos>=2, <eos>=3.

    python tokenizer.py train    # fit on data/tinystories_train_seed.jsonl only
    python tokenizer.py encode   # write data/train.bin and data/val.bin (uint16)

Importable helpers for the trainer: load_tokenizer, encode, decode.
"""
import argparse
import json

import numpy as np
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
TRAIN_JSONL = "data/tinystories_train_seed.jsonl"
VAL_JSONL = "data/tinystories_val_2000.jsonl"
TOKENIZER_PATH = "data/tokenizer.json"


def read_texts(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def load_tokenizer(path=TOKENIZER_PATH):
    return Tokenizer.from_file(path)


def encode(tok, text):
    return tok.encode(text).ids


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=False)


def train(vocab_size, out=TOKENIZER_PATH):
    # Training data is the train seed only; never fit on validation.
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(read_texts(TRAIN_JSONL), trainer)
    assert [tok.token_to_id(t) for t in SPECIAL_TOKENS] == [0, 1, 2, 3]
    assert tok.get_vocab_size() < 65536, "ids must fit in uint16"
    tok.save(out)
    print(f"vocab size {tok.get_vocab_size()} -> {out}")


def encode_file(tok, src, dst):
    eos = tok.token_to_id("<eos>")
    texts = read_texts(src)
    ids = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(eos)
    np.array(ids, dtype=np.uint16).tofile(dst)
    words = sum(len(t.split()) for t in texts)
    print(f"{dst}: {len(texts)} stories, {len(ids)} tokens, "
          f"{len(ids) / words:.3f} tokens/word")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train", help="train BPE on the training seed")
    p.add_argument("--vocab-size", type=int, default=8192)
    sub.add_parser("encode", help="encode train/val files to uint16 .bin")
    args = parser.parse_args()
    if args.cmd == "train":
        train(args.vocab_size)
    else:
        tok = load_tokenizer()
        encode_file(tok, TRAIN_JSONL, "data/train.bin")
        encode_file(tok, VAL_JSONL, "data/val.bin")


if __name__ == "__main__":
    main()
