"""Unit tests for scripts/telemetry.py (live console helpers; no GPU, no subprocesses)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import telemetry as t  # noqa: E402

SWAP = "vm.swapusage: total = 3072.00M  used = 1538.38M  free = 1533.62M  (encrypted)\n"
BATT_BAT = ("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=38338659)\t100%; "
            "discharging; 11:46 remaining present: true\n")
BATT_AC = ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t80%; charging; "
           "0:40 remaining present: true\n")
IOREG = ('  | |   "VirtualTemperature"=2909,"MaximumTemperature"=43}\n'
         '  | |   "BatteryData" = {"CurrentCapacity"=100,"Temperature"=2909,"VirtualTemperature"=2909}\n')
VM_STAT = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      1000.\n"
           "Pages active: 5.\nPages inactive: 2000.\nPages speculative: 100.\nPages purgeable: 400.\n")


class TestParsers(unittest.TestCase):
    def test_swap(self):
        s = t.parse_swapusage(SWAP)
        self.assertAlmostEqual(s["used_mb"], 1538.38)
        self.assertAlmostEqual(s["total_mb"], 3072.0)
        self.assertIsNone(t.parse_swapusage("garbage"))

    def test_pmset(self):
        b = t.parse_pmset_batt(BATT_BAT)
        self.assertFalse(b["on_ac"])
        self.assertEqual((b["percent"], b["state"]), (100, "discharging"))
        self.assertTrue(t.parse_pmset_batt(BATT_AC)["on_ac"])
        self.assertIsNone(t.parse_pmset_batt(""))

    def test_battery_temp(self):
        self.assertEqual(t.parse_battery_temp(IOREG), 29.1)
        self.assertIsNone(t.parse_battery_temp('"VirtualTemperature"=2909'))
        self.assertIsNone(t.parse_battery_temp('"Temperature"=0'))
        self.assertIsNone(t.parse_battery_temp(""))

    def test_vm_stat(self):
        self.assertEqual(t.parse_vm_stat_free_mb(VM_STAT), int(3500 * 16384 / 2**20))
        self.assertIsNone(t.parse_vm_stat_free_mb("nope"))

    def test_sample_never_raises(self):
        snap = t.sample_machine(runner=lambda cmd, parser: None)
        self.assertIsNone(snap["swap_used_mb"])
        self.assertIsNone(snap["temp_c"])

    def test_sampler_caches(self):
        calls, now = [], [0.0]
        s = t.Sampler(5.0, sample_fn=lambda: calls.append(1) or {"a": 1}, clock=lambda: now[0])
        s.get()
        s.get()
        now[0] = 6.0
        s.get()
        self.assertEqual(len(calls), 2)


class TestRates(unittest.TestCase):
    def test_rolling_window(self):
        r = t.RollingRate(n=2)
        for tok, sec in [(100, 1), (100, 1), (100, 4)]:
            r.add(tok, sec)
        self.assertAlmostEqual(r.rate(), 200 / 5)
        self.assertAlmostEqual(r.rate(last=1), 25)
        self.assertIsNone(t.RollingRate().rate())

    def test_eta(self):
        self.assertEqual(t.eta_seconds(6000, 100), 60)
        self.assertIsNone(t.eta_seconds(6000, None))
        self.assertEqual(t.eta_seconds(-5, 100), 0)


class TestThrottle(unittest.TestCase):
    def make(self):
        d = t.ThrottleDetector(first_step=200)
        d.add_record(150, 1.0)  # before the window: ignored
        for step, v in [(210, 1.0), (220, 6400), (230, 6500), (240, 6600), (260, 6500)]:
            d.add_record(step, v)
        return d

    def test_baseline_skips_first_record(self):
        self.assertEqual(self.make().baseline, 6500)

    def test_trigger_once_and_recover(self):
        d = self.make()
        events = [d.update(r) for r in (6400, 3800, 2000, 1500, 4000, 5300, 6000, 3000)]
        self.assertEqual(events, [None, "throttle", None, None, None, "recover", None, "throttle"])

    def test_no_baseline_no_alerts(self):
        d = t.ThrottleDetector(first_step=200)
        self.assertIsNone(d.update(10))


class TestFormat(unittest.TestCase):
    def test_duration(self):
        self.assertEqual([t.fmt_duration(x) for x in (None, 5, 372, 3900)], ["--", "5s", "6m12s", "1h05m"])

    def test_lr_and_mb(self):
        self.assertEqual(t.fmt_lr(5.1e-4), "5.1e-4")
        self.assertEqual(t.fmt_mb(2458), "2.4G")
        self.assertEqual(t.fmt_mb(None), "--")

    def test_live_line(self):
        line = t.fmt_live_line(1200, 2441, 2.71, 2.74, 5.1e-4, 6480, 6300, 372, 390, 2458, 38.2)
        self.assertEqual(line, "step 1,200/2,441  49%  loss 2.71 (avg 2.74)  lr 5.1e-4   6,480 tok/s (avg 6,300)  "
                               "elapsed 6m12s  ETA 6m30s  swap 2.4G  temp 38C")
        self.assertNotIn("swap", t.fmt_live_line(1, 10, 1, 1, 1e-4, 100, 100, 1, 1, None, None))

    def test_paint_plain_when_disabled(self):
        self.assertEqual(t.paint("x", "red", False), "x")
        self.assertIn("\033[", t.paint("x", "red", True))

    def test_eval_line(self):
        a = t.fmt_eval_line(500, 3.0, 20.08, None, True, 4.2)
        self.assertIn("new best", a)
        self.assertNotIn("vs prev", a)
        b = t.fmt_eval_line(1000, 2.5, 12.18, 3.0, False, 4.2)
        self.assertIn("-0.5000 vs prev", b)
        self.assertNotIn("new best", b)

    def test_preflight(self):
        self.assertEqual(t.preflight_warnings({"swap_used_mb": 1000, "on_ac": True}), [])
        self.assertEqual(len(t.preflight_warnings({"swap_used_mb": 4000, "on_ac": False})), 2)
        self.assertEqual(t.preflight_warnings({}), [])
        self.assertIn("healthy", t.expected_duration_line(10_000_000))


class TestMonitor(unittest.TestCase):
    def test_telemetry_fields_optional(self):
        m = t.LiveMonitor(4096, 100, 0, 0, t.Sampler(sample_fn=lambda: {}))
        lines, extra = m.on_train(1, 5.0, 1e-4, 5000, 4096, 0.8, 1.0)
        self.assertEqual(list(extra), ["eta_s"])
        self.assertEqual(len(lines), 1)

    def test_swap_growth_alert(self):
        swaps = iter([1000, 1100, 2500])
        m = t.LiveMonitor(4096, 100, 0, 0, t.Sampler(0, sample_fn=lambda: {"swap_used_mb": next(swaps)}))
        out = [m.on_train(s, 3.0, 1e-4, 5000, 4096, 0.8, s)[0] for s in (1, 2, 3)]
        self.assertEqual([len(x) for x in out], [1, 1, 2])
        self.assertIn("SWAP", out[2][1])


if __name__ == "__main__":
    unittest.main()
