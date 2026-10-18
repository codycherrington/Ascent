"""Live-console and machine-telemetry helpers for train.py (stdlib only, no torch).

Pure functions (parsers, ETA, throttle detection, formatting) are unit tested in
tests/test_console.py. Machine sampling shells out to sysctl / pmset / ioreg / vm_stat
without sudo, with a short timeout, and every failure degrades to None.
"""
import re
import statistics
import subprocess
import time
from collections import deque

HEALTHY_TOK_S = 6500.0     # speed assumed for a cool machine (expected-duration estimate)
THROTTLED_TOK_S = 1500.0   # speed seen when throttled
THROTTLE_FRAC = 0.60       # alert when rolling tok/s < 60% of the early-run median
RECOVER_FRAC = 0.80        # recovery line when it returns above 80%
SWAP_WARN_MB = 3 * 1024    # pre-flight warning: swap already used
SWAP_GROWTH_MB = 1024      # alert when swap grew this much since start
TEMP_WARN_C = 45.0         # battery-sensor temperature alert (a proxy, not die temperature)
ROLL_N = 10                # log intervals in the rolling window


# ---------- parsers (canned-output friendly) ----------

def parse_swapusage(text):
    """`sysctl vm.swapusage` -> dict(total_mb, used_mb, free_mb) or None."""
    try:
        vals = {k: float(v) for k, v in re.findall(r"(total|used|free)\s*=\s*([\d.]+)M", text)}
        if "used" not in vals:
            return None
        return {"total_mb": vals.get("total"), "used_mb": vals["used"], "free_mb": vals.get("free")}
    except Exception:  # noqa: BLE001
        return None


def parse_pmset_batt(text):
    """`pmset -g batt` -> dict(on_ac, percent, state) or None."""
    try:
        first = text.splitlines()[0]
        on_ac = "AC Power" in first
        if not on_ac and "Battery Power" not in first:
            return None
        m = re.search(r"(\d+)%;\s*([^;]+)", text)
        return {"on_ac": on_ac, "percent": int(m.group(1)) if m else None,
                "state": m.group(2).strip() if m else None}
    except Exception:  # noqa: BLE001
        return None


def parse_battery_temp(text):
    """`ioreg -rc AppleSmartBatteryPack -w0` -> battery temperature in C, or None.

    The "Temperature" key is in hundredths of a degree C. Only the exact quoted key is
    matched (not VirtualTemperature / MaximumTemperature). Implausible values give None.
    """
    try:
        m = re.search(r'"Temperature"\s*=\s*(\d+)', text)
        if not m:
            return None
        c = int(m.group(1)) / 100.0
        return round(c, 1) if 0.0 < c < 120.0 else None
    except Exception:  # noqa: BLE001
        return None


def parse_vm_stat_free_mb(text):
    """`vm_stat` -> reclaimable memory in MB (free + inactive + speculative + purgeable), or None."""
    try:
        page = int(re.search(r"page size of (\d+) bytes", text).group(1))
        pages = 0
        for key in ("free", "inactive", "speculative", "purgeable"):
            pages += int(re.search(rf"Pages {key}:\s+(\d+)", text).group(1))
        return int(pages * page / 2**20)
    except Exception:  # noqa: BLE001
        return None


# ---------- machine sampling ----------

def run_cmd(cmd, timeout=2.0):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _parsed(cmd, parser):
    out = run_cmd(cmd)
    return parser(out) if out else None


def sample_machine(runner=_parsed):
    """One telemetry snapshot. Keys are always present; unavailable values are None."""
    swap = runner(["sysctl", "vm.swapusage"], parse_swapusage)
    batt = runner(["pmset", "-g", "batt"], parse_pmset_batt)
    return {
        "swap_used_mb": int(round(swap["used_mb"])) if swap else None,
        "swap_total_mb": int(round(swap["total_mb"])) if swap and swap.get("total_mb") else None,
        "free_mb": runner(["vm_stat"], parse_vm_stat_free_mb),
        "on_ac": batt["on_ac"] if batt else None,
        "batt_percent": batt["percent"] if batt else None,
        "temp_c": runner(["ioreg", "-rc", "AppleSmartBatteryPack", "-w0"], parse_battery_temp),
    }


class Sampler:
    """Caches sample_machine() so it runs at most once per min_interval seconds."""

    def __init__(self, min_interval=5.0, sample_fn=sample_machine, clock=time.monotonic):
        self.min_interval, self.sample_fn, self.clock = min_interval, sample_fn, clock
        self.last, self.t = None, None

    def get(self):
        now = self.clock()
        if self.last is None or now - self.t >= self.min_interval:
            try:
                self.last = self.sample_fn()
            except Exception:  # noqa: BLE001
                self.last = self.last or {}
            self.t = now
        return self.last


# ---------- rolling rate, ETA, throttle detection ----------

class RollingRate:
    """Tokens/sec over the last n log intervals (training time only; callers skip pauses)."""

    def __init__(self, n=ROLL_N):
        self.items = deque(maxlen=n)

    def add(self, tokens, seconds):
        if seconds > 0 and tokens > 0:
            self.items.append((tokens, seconds))

    def rate(self, last=None):
        items = list(self.items)[-last:] if last else list(self.items)
        secs = sum(s for _, s in items)
        return sum(t for t, _ in items) / secs if secs > 0 else None


def eta_seconds(remaining_tokens, rate):
    if not rate or rate <= 0:
        return None
    return max(0.0, remaining_tokens / rate)


class Latch:
    """Hysteresis alert: 'on' once when value >= hi (or <= lo if below=True), 'off' once when it clears."""

    def __init__(self, on_at, off_at, below=False):
        self.on_at, self.off_at, self.below, self.active = on_at, off_at, below, False

    def update(self, v):
        if v is None:
            return None
        trip = v < self.on_at if self.below else v >= self.on_at
        clear = v > self.off_at if self.below else v < self.off_at
        if not self.active and trip:
            self.active = True
            return "on"
        if self.active and clear:
            self.active = False
            return "off"
        return None


class ThrottleDetector:
    """Baseline = median tok/s of log records in [first_step, first_step + window_steps),
    skipping the first record there. Afterwards, update(rate) returns 'throttle' once per
    episode when rate < 60% of baseline and 'recover' when it is back above 80%."""

    def __init__(self, first_step, window_steps=50, min_records=2):
        self.first_step, self.window_steps, self.min_records = first_step, window_steps, min_records
        self.samples, self.skipped, self.baseline = [], False, None
        self.latch = None

    def add_record(self, step, tok_s):
        """Feed every log record; builds the baseline. Returns the baseline once known."""
        if self.baseline is not None or step < self.first_step:
            return self.baseline
        if step >= self.first_step + self.window_steps:
            if len(self.samples) >= self.min_records:
                self._finish()
            return self.baseline
        if not self.skipped:
            self.skipped = True
            return None
        self.samples.append(tok_s)
        return None

    def _finish(self):
        self.baseline = statistics.median(self.samples)
        self.latch = Latch(THROTTLE_FRAC * self.baseline, RECOVER_FRAC * self.baseline, below=True)

    def update(self, rate):
        if self.baseline is None or rate is None:
            return None
        ev = self.latch.update(rate)
        return {"on": "throttle", "off": "recover"}.get(ev)


# ---------- formatting ----------

def fmt_duration(s):
    if s is None:
        return "--"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s" if m else f"{sec}s"


def fmt_mb(mb):
    return "--" if mb is None else f"{mb / 1024:.1f}G"


def fmt_lr(lr):
    m, e = f"{lr:.1e}".split("e")
    return f"{m}e{int(e)}"


def paint(text, color, enabled):
    codes = {"red": "1;31", "yellow": "1;33", "green": "32", "cyan": "36", "dim": "2", "bold": "1"}
    return f"\033[{codes[color]}m{text}\033[0m" if enabled else text


def fmt_live_line(step, max_steps, loss, avg_loss, lr, tok_s, avg_tok_s, elapsed_s, eta_s, swap_mb, temp_c):
    w = len(f"{max_steps:,}")
    pct = int(100 * step / max_steps) if max_steps else 0
    parts = [f"step {step:>{w},}/{max_steps:,} {pct:3d}%",
             f"loss {loss:.2f} (avg {avg_loss:.2f})",
             f"lr {fmt_lr(lr)}",
             f"{tok_s:6,.0f} tok/s (avg {avg_tok_s:,.0f})",
             f"elapsed {fmt_duration(elapsed_s)}",
             f"ETA {fmt_duration(eta_s)}"]
    if swap_mb is not None:
        parts.append(f"swap {fmt_mb(swap_mb)}")
    if temp_c is not None:
        parts.append(f"temp {temp_c:.0f}C")
    return "  ".join(parts)


def fmt_eval_line(step, val_loss, ppl, prev_val, is_best, eval_s):
    delta = "" if prev_val is None else f" ({val_loss - prev_val:+.4f} vs prev)"
    best = "  ** new best **" if is_best else ""
    return f"eval  step {step}: val loss {val_loss:.4f}{delta}  ppl {ppl:.2f}  {eval_s}s{best}"


def fmt_samples_line(step, n_prompts, sec):
    return f"samples step {step}: {n_prompts} prompts, {sec}s"


def expected_duration_line(total_tokens, healthy=HEALTHY_TOK_S, throttled=THROTTLED_TOK_S):
    return (f"expected duration: ~{fmt_duration(total_tokens / healthy)} healthy ({healthy:,.0f} tok/s) "
            f"to ~{fmt_duration(total_tokens / throttled)} if throttled ({throttled:,.0f} tok/s)")


def preflight_warnings(snap):
    """Warning strings for a telemetry snapshot (dict from sample_machine)."""
    out = []
    swap = snap.get("swap_used_mb")
    if swap is not None and swap > SWAP_WARN_MB:
        out.append(f"swap already {fmt_mb(swap)} used (> {SWAP_WARN_MB // 1024}G): close apps / reboot before a long run")
    if snap.get("on_ac") is False:
        out.append("running on battery power: plug in, or the machine may throttle")
    return out


def preflight_lines(snap):
    swap, tot = snap.get("swap_used_mb"), snap.get("swap_total_mb")
    swap_s = "unavailable" if swap is None else f"{fmt_mb(swap)} used" + (f" of {fmt_mb(tot)}" if tot else "")
    free = snap.get("free_mb")
    ac = snap.get("on_ac")
    pwr = "unknown" if ac is None else ("AC power" if ac else "BATTERY")
    if snap.get("batt_percent") is not None:
        pwr += f" ({snap['batt_percent']}%)"
    temp = snap.get("temp_c")
    return [f"swap {swap_s} | free memory ~{'unavailable' if free is None else fmt_mb(free)} | power {pwr}"
            f" | temp proxy {'unavailable' if temp is None else f'{temp:.1f}C (battery sensor)'}"]


# ---------- stateful live monitor used by train.py ----------

class LiveMonitor:
    """Turns each train log record into the live line plus any alert lines, and the
    optional telemetry fields for metrics.jsonl (swap_used_mb, temp_c, eta_s)."""

    def __init__(self, tokens_per_step, max_steps, start_step, warmup_steps, sampler, color=False):
        self.tps, self.max_steps, self.color, self.sampler = tokens_per_step, max_steps, color, sampler
        self.roll = RollingRate()
        self.losses = deque(maxlen=ROLL_N)
        self.detector = ThrottleDetector(first_step=max(warmup_steps, start_step))
        self.swap_latch = None   # created on first sample (needs the starting swap)
        self.swap0 = None
        self.temp_latch = Latch(TEMP_WARN_C, TEMP_WARN_C - 3.0)
        self.first = True
        self.tok_total, self.sec_total = 0, 0.0
        self.start_step = start_step

    def on_train(self, step, loss, lr, tok_s, interval_tokens, interval_s, elapsed_s):
        snap = self.sampler.get() or {}
        swap, temp = snap.get("swap_used_mb"), snap.get("temp_c")
        if self.swap0 is None and swap is not None:
            self.swap0 = swap
            self.swap_latch = Latch(SWAP_GROWTH_MB, SWAP_GROWTH_MB / 2)
        if not self.first:  # the first record carries start-up cost
            self.roll.add(interval_tokens, interval_s)
        self.first = False
        self.tok_total += interval_tokens
        self.sec_total += interval_s
        self.losses.append(loss)
        self.detector.add_record(step, tok_s)

        rate = self.roll.rate() or tok_s
        eta = eta_seconds((self.max_steps - step) * self.tps, rate)
        avg_tok = self.tok_total / self.sec_total if self.sec_total else tok_s
        lines = [fmt_live_line(step, self.max_steps, loss, sum(self.losses) / len(self.losses), lr, tok_s,
                               avg_tok, elapsed_s, eta, swap, temp)]

        ev = self.detector.update(self.roll.rate(last=3))
        if ev == "throttle":
            lines.append(paint(f"!! THROTTLE: step {step}: recent {self.roll.rate(last=3):,.0f} tok/s is below "
                               f"{THROTTLE_FRAC:.0%} of the early-run median {self.detector.baseline:,.0f} tok/s"
                               f" (temp {temp if temp is not None else '?'}C, swap {fmt_mb(swap)})", "red", self.color))
        elif ev == "recover":
            lines.append(paint(f"-- recovered: step {step}: {self.roll.rate(last=3):,.0f} tok/s is back above "
                               f"{RECOVER_FRAC:.0%} of baseline", "green", self.color))
        if self.swap_latch and swap is not None:
            ev = self.swap_latch.update(swap - self.swap0)
            if ev == "on":
                lines.append(paint(f"!! SWAP: grew {fmt_mb(swap - self.swap0)} since start (now {fmt_mb(swap)}); "
                                   "memory pressure slows training", "yellow", self.color))
            elif ev == "off":
                lines.append(paint("-- swap growth back under control", "green", self.color))
        ev = self.temp_latch.update(temp)
        if ev == "on":
            lines.append(paint(f"!! HOT: temperature proxy {temp:.1f}C >= {TEMP_WARN_C:.0f}C (battery sensor)",
                               "yellow", self.color))
        elif ev == "off":
            lines.append(paint(f"-- temperature proxy back to {temp:.1f}C", "green", self.color))

        extra = {}
        if swap is not None:
            extra["swap_used_mb"] = int(swap)
        if temp is not None:
            extra["temp_c"] = float(temp)
        if eta is not None:
            extra["eta_s"] = round(eta, 1)
        return lines, extra
