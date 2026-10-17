# Ascent

A from-scratch GPT-style language model (~29.5M parameters, decoder-only) trained locally on a MacBook Air (M2, 8GB, fanless) with PyTorch and MPS.

## Goal

Fluent English from scratch on TinyStories: a text-continuation story model, not a chatbot. No persona, chat format or factual correctness yet. See [docs/goal.md](docs/goal.md).

## Status

Relaunch in progress. The data pipeline, BPE tokenizer, model, baseline trainer and throughput benchmark are written and tested, but **no model has been trained yet**: the first baseline run is still to do, so there are no weights or sample outputs. The roadmap (modular trainer, run records, generation CLI, training portal, experiments) is tracked in the GitHub Issues.

## Quickstart

Run from the repo root on an Apple Silicon Mac (CPU fallback works but is slow).

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/get_tinystories.py   # download and clean TinyStories into data/
.venv/bin/python tokenizer.py train           # fit the BPE tokenizer -> data/tokenizer.json
.venv/bin/python tokenizer.py encode          # -> data/train.bin, data/val.bin
.venv/bin/python train.py                     # baseline run (10M tokens); see --help
.venv/bin/python -m unittest discover -s tests -v
```

Each training run writes `runs/<YYYYMMDD-HHMMSS>/` with config, metrics, samples and checkpoints. Data and runs are gitignored.

## Layout

| Path | What |
|------|------|
| `model.py` | GPT model and `generate()` |
| `tokenizer.py` | BPE tokenizer CLI and helpers |
| `train.py` | baseline trainer |
| `scripts/` | data download, MPS throughput benchmark |
| `eval/` | fixed prompts, rubric, eval protocol |
| `tests/` | unit tests |
| `docs/` | documentation |
| `dashboard.py` | empty; reserved for the future training portal |

## Docs

- [docs/architecture.md](docs/architecture.md) - how the code base is structured
- [docs/goal.md](docs/goal.md) - goal and success criteria
- [docs/tokenizer.md](docs/tokenizer.md) - tokenizer design and stats
- [docs/datasets/tinystories.md](docs/datasets/tinystories.md) - dataset card
- [eval/](eval/README.md) - evaluation protocol, prompts and rubric

## Workflow

Long-lived branches are `dev`, `test` and `main`. Changes go on `YYYYMMDD-short-description` branches cut from `dev` and arrive via pull request; promotion is `dev` -> `test` -> `main` by PR.

## License

MIT, see [LICENSE](LICENSE).
