"""Prompt -> continuation CLI for a trained Ascent checkpoint.

Ascent is a base language model trained only on TinyStories. It continues your
text as a children's story; it does not answer questions or follow instructions.

    .venv/bin/python chat.py                       # newest run's best checkpoint
    .venv/bin/python chat.py --run runs/<id>       # a specific run
    .venv/bin/python chat.py --prompt "Once upon a time"   # one-shot, then exit
"""
import argparse
import glob
import os
import random
import sys
import time
from dataclasses import dataclass, field

import torch

from model import GPT, GPTConfig, generate_stream
from tokenizer import TOKENIZER_PATH, decode, encode, load_tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
REPLACEMENT = "�"
END_MARKER = "[end of story]"

HELP_TEXT = """\
Type a story opening and the model continues it. Commands:
  /help          show this help
  /temp 0.7      set temperature (0 = greedy)
  /topk 20       set top-k (0 = off)
  /len 200       set max new tokens
  /seed 42       fix the seed (/seed random goes back to a fresh seed each time)
  /greedy        toggle greedy decoding (temperature 0)
  /again         re-run the last prompt with a new random seed
  /continue      extend the last output further
  /info          checkpoint path, step, params, settings
  /quit          exit (also Ctrl-D)"""


class ChatError(Exception):
    """An expected, user-facing problem (printed without a stack trace)."""


@dataclass
class Settings:
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 150
    seed: "int | None" = None
    stop_at_eos: bool = True
    saved_temperature: float = 0.8  # restored when /greedy is toggled off

    def describe(self):
        seed = "random" if self.seed is None else self.seed
        stop = "on" if self.stop_at_eos else "off"
        return (f"temperature={self.temperature} top_k={self.top_k} "
                f"max_new_tokens={self.max_new_tokens} seed={seed} stop_at_eos={stop}")


@dataclass
class Last:
    """State for /again and /continue."""
    prompt_ids: list = field(default_factory=list)  # ids fed to the model as the prompt
    prompt_text: str = ""
    output_ids: list = field(default_factory=list)  # generated ids, eos excluded


# ---------- helpers (importable, tested) ----------

def style(text, code, enabled):
    return f"\033[{code}m{text}\033[0m" if enabled and text else text


def use_color(stream=None):
    stream = stream or sys.stdout
    return stream.isatty() and not os.environ.get("NO_COLOR")


def find_checkpoint(runs_dir="runs", run=None, ckpt=None):
    """Resolve the checkpoint to load. --ckpt beats --run beats the newest run."""
    if ckpt:
        if not os.path.isfile(ckpt):
            raise ChatError(f"checkpoint not found: {ckpt}")
        return ckpt
    if run:
        if not os.path.isdir(run):
            raise ChatError(f"run folder not found: {run}")
        for name in ("ckpt_best.pt", "ckpt.pt"):
            path = os.path.join(run, name)
            if os.path.isfile(path):
                return path
        raise ChatError(f"no ckpt_best.pt or ckpt.pt in {run}")
    candidates = sorted(d for d in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(d))
    for pick in ("ckpt_best.pt", "ckpt.pt"):  # prefer best checkpoints, newest folder first
        for d in reversed(candidates):
            path = os.path.join(d, pick)
            if os.path.isfile(path):
                return path
    raise ChatError(f"no checkpoint found under {runs_dir}/. Train one with "
                    f".venv/bin/python train.py, or pass --run runs/<id> or --ckpt <path>.")


def pick_device(name="auto"):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise ChatError("--device mps requested but MPS is not available; use --device cpu")
    return name


def load_model(path, device="cpu"):
    """Return (model, step, params) from a train.py checkpoint."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = GPTConfig(**ck["config"]["model"])
        model = GPT(cfg)
        model.load_state_dict(ck["model"])
    except (KeyError, TypeError, RuntimeError, EOFError, OSError, ValueError) as e:
        raise ChatError(f"could not load checkpoint {path}: {e}") from e
    model.to(device).eval()
    return model, ck.get("step"), sum(p.numel() for p in model.parameters())


class IncrementalDecoder:
    """Turn a growing id list into printable text chunks.

    Decodes the whole id list each step and returns only the new suffix. A
    trailing replacement char means a multibyte character is still incomplete,
    so it is held back until later tokens finish it (or flush() at the end).
    """

    def __init__(self, tok):
        self.tok = tok
        self.ids = []
        self.emitted = ""

    def feed(self, token_id):
        self.ids.append(token_id)
        text = decode(self.tok, self.ids)
        while text.endswith(REPLACEMENT):
            text = text[:-1]
        return self._advance(text)

    def flush(self):
        return self._advance(decode(self.tok, self.ids))

    def _advance(self, text):
        if not text.startswith(self.emitted):
            return ""  # earlier text was revised by a later token; wait for it to settle
        new = text[len(self.emitted):]
        self.emitted = text
        return new


def parse_command(line):
    """Return (name, arg) for a slash command line, or None for a normal prompt."""
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped[1:].split(None, 1)
    if not parts:
        return ("", "")
    return parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")


def apply_setting(settings, name, arg):
    """Apply /temp /topk /len /seed /greedy. Returns a status message; raises ValueError."""
    if name == "temp":
        value = float(arg)
        if value < 0:
            raise ValueError("temperature must be >= 0")
        settings.temperature = settings.saved_temperature = value
        return f"temperature = {value}"
    if name == "topk":
        value = int(arg)
        if value < 0:
            raise ValueError("top-k must be >= 0 (0 = off)")
        settings.top_k = value
        return f"top_k = {value}"
    if name == "len":
        value = int(arg)
        if value < 1:
            raise ValueError("length must be >= 1")
        settings.max_new_tokens = value
        return f"max_new_tokens = {value}"
    if name == "seed":
        if arg.lower() in ("random", "none", "off"):
            settings.seed = None
            return "seed = random (new each time)"
        settings.seed = int(arg)
        return f"seed = {settings.seed}"
    if name == "greedy":
        if settings.temperature == 0:
            settings.temperature = settings.saved_temperature or 0.8
            return f"greedy off, temperature = {settings.temperature}"
        settings.saved_temperature = settings.temperature
        settings.temperature = 0.0
        return "greedy on (temperature = 0)"
    raise ValueError(f"unknown setting {name}")


def build_prompt_ids(tok, text, block_size):
    """Encode text; keep only the last block_size ids. Returns (ids, truncated)."""
    ids = encode(tok, text)
    if len(ids) > block_size:
        return ids[-block_size:], True
    return ids, False


def stream_reply(model, tok, context_ids, settings, seed, out, color):
    """Generate from context_ids, streaming text to `out`.

    Returns (new_ids excluding eos, hit_eos, interrupted, seconds). Ctrl-C stops
    generation but keeps what was produced so far.
    """
    eos = tok.token_to_id("<eos>")
    dec = IncrementalDecoder(tok)
    new_ids, hit_eos, interrupted = [], False, False
    gen = generate_stream(model, context_ids, settings.max_new_tokens,
                          temperature=settings.temperature, top_k=settings.top_k,
                          seed=seed, eos_id=eos if settings.stop_at_eos else None)
    start = time.time()
    try:
        for tid in gen:
            if tid == eos:
                if settings.stop_at_eos:
                    hit_eos = True
                    break
                continue  # --no-stop: hide the literal <eos> text but keep going
            new_ids.append(tid)
            out.write(dec.feed(tid))
            out.flush()
        out.write(dec.flush())
    except KeyboardInterrupt:
        interrupted = True
    finally:
        gen.close()
    seconds = time.time() - start
    if hit_eos:
        out.write(" " + style(END_MARKER, "2", color))
    if interrupted:
        out.write(" " + style("[stopped]", "2", color))
    out.write("\n")
    out.flush()
    return new_ids, hit_eos, interrupted, seconds


class Chat:
    def __init__(self, model, tok, settings, ckpt_path, step, params, out=None, color=False):
        self.model, self.tok, self.settings = model, tok, settings
        self.ckpt_path, self.step, self.params = ckpt_path, step, params
        self.out = out or sys.stdout
        self.color = color
        self.last = None
        self.block_size = model.cfg.block_size

    def say(self, text):
        self.out.write(text + "\n")
        self.out.flush()

    def dim(self, text):
        return style(text, "2", self.color)

    def new_seed(self):
        return self.settings.seed if self.settings.seed is not None else random.randrange(2**31)

    def _run(self, context_ids, seed, echo_prompt=None):
        if echo_prompt is not None:
            self.out.write(self.dim(echo_prompt))
        new_ids, _, _, secs = stream_reply(self.model, self.tok, context_ids, self.settings,
                                           seed, self.out, self.color)
        n = len(new_ids)
        rate = n / secs if secs > 0 else 0.0
        self.say(self.dim(f"[{n} tokens, {secs:.1f}s, {rate:.1f} tok/s, seed {seed}]"))
        return new_ids

    def reply(self, text, seed=None):
        ids, truncated = build_prompt_ids(self.tok, text, self.block_size)
        if not ids:
            self.say(self.dim("(that prompt encodes to nothing; type some text)"))
            return
        if truncated:
            self.say(self.dim(f"(prompt too long; using the last {self.block_size} tokens)"))
        seed = self.new_seed() if seed is None else seed
        new_ids = self._run(ids, seed, echo_prompt=text)
        self.last = Last(ids, text, new_ids)

    def handle(self, line):
        """Process one input line. Returns False when the session should end."""
        if not line.strip():
            return True
        cmd = parse_command(line)
        if cmd is None:
            self.reply(line)
            return True
        name, arg = cmd
        if name in ("quit", "exit", "q"):
            return False
        if name in ("help", "h", "?"):
            self.say(HELP_TEXT)
        elif name in ("temp", "topk", "len", "seed", "greedy"):
            try:
                if name != "greedy" and not arg:
                    raise ValueError(f"usage: /{name} <value>")
                self.say(self.dim(apply_setting(self.settings, name, arg)))
            except ValueError as e:
                self.say(f"could not apply /{name} {arg}: {e}")
        elif name == "again":
            if not self.last:
                self.say("nothing to repeat yet; enter a prompt first")
            else:
                self.reply(self.last.prompt_text, seed=random.randrange(2**31))
        elif name == "continue":
            if not self.last:
                self.say("nothing to continue yet; enter a prompt first")
            else:
                context = self.last.prompt_ids + self.last.output_ids
                new_ids = self._run(context, self.new_seed())
                self.last.output_ids = self.last.output_ids + new_ids
        elif name == "info":
            self.say(f"checkpoint: {self.ckpt_path}\nstep: {self.step}\n"
                     f"params: {self.params:,}  block_size: {self.block_size}\n"
                     f"settings: {self.settings.describe()}")
        else:
            self.say(f"unknown command /{name}; try /help")
        return True


def banner(ckpt_path, step, params, device, settings, color):
    lines = [
        "Ascent prompt -> continuation session",
        "This is a base model trained only on TinyStories. It continues your text as a",
        "children's story; it does not answer questions or follow instructions.",
        f"checkpoint: {ckpt_path} (step {step}, {params / 1e6:.1f}M params, {device})",
        f"settings: {settings.describe()}",
        "Type a story opening, /help for commands, /quit to exit.",
    ]
    return "\n".join([style(lines[0], "1", color)] + [style(l, "2", color) for l in lines[1:]])


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Prompt -> continuation CLI for a trained Ascent checkpoint. "
                    "Ascent is a base TinyStories model: it continues text as a children's "
                    "story, it does not answer questions.")
    p.add_argument("--run", help="run folder, e.g. runs/20261017-164238 (default: newest run)")
    p.add_argument("--ckpt", help="explicit checkpoint file (overrides --run)")
    p.add_argument("--tokenizer", default=os.path.join(ROOT, TOKENIZER_PATH),
                   help="tokenizer.json (default: data/tokenizer.json)")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy (default 0.8)")
    p.add_argument("--top-k", type=int, default=40, help="0 = off (default 40)")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--seed", type=int, default=None, help="default: random; the seed used is printed")
    p.add_argument("--no-stop", action="store_true", help="do not stop at <eos>")
    p.add_argument("--prompt", help="one-shot: print the continuation of this text and exit")
    return p.parse_args(argv)


def run(args):
    ckpt_path = find_checkpoint(os.path.join(ROOT, "runs"), args.run, args.ckpt)
    if not os.path.isfile(args.tokenizer):
        raise ChatError(f"tokenizer not found: {args.tokenizer}. "
                        f"Fit it with .venv/bin/python tokenizer.py train, or pass --tokenizer.")
    device = pick_device(args.device)
    tok = load_tokenizer(args.tokenizer)
    model, step, params = load_model(ckpt_path, device)
    settings = Settings(temperature=args.temperature, top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens, seed=args.seed,
                        stop_at_eos=not args.no_stop,
                        saved_temperature=args.temperature or 0.8)
    color = use_color(sys.stdout)
    chat = Chat(model, tok, settings, ckpt_path, step, params, out=sys.stdout, color=color)

    if args.prompt is not None:
        ids, truncated = build_prompt_ids(tok, args.prompt, model.cfg.block_size)
        if not ids:
            raise ChatError("the prompt encodes to nothing")
        if truncated:
            print(f"(prompt too long; using the last {model.cfg.block_size} tokens)", file=sys.stderr)
        seed = chat.new_seed()
        new_ids, _, _, secs = stream_reply(model, tok, ids, settings, seed, sys.stdout, color)
        rate = len(new_ids) / secs if secs > 0 else 0.0
        print(f"[{len(new_ids)} tokens, {secs:.1f}s, {rate:.1f} tok/s, seed {seed}]", file=sys.stderr)
        return

    print(banner(ckpt_path, step, params, device, settings, color))
    try:
        import readline  # noqa: F401  (line editing and history for input())
    except ImportError:
        pass
    while True:
        try:
            line = input("you> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n(use /quit or Ctrl-D to exit)")
            continue
        try:
            if not chat.handle(line):
                break
        except KeyboardInterrupt:
            print()


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except ChatError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
