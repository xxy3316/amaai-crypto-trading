"""Risk-adjusted metrics and significance testing.

Phase 3 of the publication plan. Two defects motivated this module:

  * `summary['sharpe_ratio']` and `summary['max_drawdown']` were READ by the
    past-results panel and never WRITTEN anywhere, so both were always 0 and
    the panel guarding on `!= 0` was dead code. The app judged strategies on
    total return alone, which doubling the position size also doubles.
  * The ablation printed a table of returns with a footnote admitting that
    "significance still needs a bootstrap"; there was no way to tell a real
    effect from the best of eight coin flips.

The tests below are mostly CALIBRATION tests rather than golden values: a
statistic is only useful if its error rate is what it claims. In particular
`test_null_rejection_rate_is_near_nominal` is the one that matters -- a
bootstrap that rejects far more often than alpha manufactures findings, and
that failure is invisible in any single run.

Run directly (the package name `test` shadows the stdlib `test` module):

    python test/test_metrics.py
"""

from __future__ import annotations

import datetime as dt
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.metrics import (  # noqa: E402
    MIN_OBSERVATIONS,
    bootstrap_ci,
    bootstrap_difference,
    calmar_ratio,
    deflated_sharpe_ratio,
    describe_run,
    equity_returns,
    holm_correction,
    max_drawdown_pct,
    periods_per_year,
    sharpe_ratio,
    sortino_ratio,
)


def curve(values, start=None, step_hours=6):
    """An equity curve shaped like the app's `daily_values`."""
    t0 = start or dt.datetime(2026, 1, 1)
    return [{"timestamp": t0 + dt.timedelta(hours=step_hours * i),
             "portfolio_value": float(v)}
            for i, v in enumerate(values)]


class TestEquityCurve(unittest.TestCase):

    def test_returns_match_hand_computation(self):
        r = equity_returns(curve([100, 110, 99]))
        self.assertAlmostEqual(r[0], 0.10)
        self.assertAlmostEqual(r[1], -0.10)

    def test_zero_value_does_not_produce_infinity(self):
        """A wiped-out portfolio must not poison every downstream ratio."""
        r = equity_returns(curve([100, 0, 50]))
        self.assertTrue(np.all(np.isfinite(r)))

    def test_periods_per_year_from_spacing(self):
        """Inferred from the curve, since it is sampled at DECISIONS not bars."""
        self.assertAlmostEqual(periods_per_year(
            [row["timestamp"] for row in curve([1, 2, 3], step_hours=24)]),
            365.0, places=6)
        self.assertAlmostEqual(periods_per_year(
            [row["timestamp"] for row in curve([1, 2, 3], step_hours=1)]),
            365.0 * 24, places=4)

    def test_too_short_curve_is_empty_not_an_error(self):
        self.assertEqual(equity_returns([]).size, 0)
        self.assertEqual(equity_returns(curve([100])).size, 0)


class TestDescriptiveRatios(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_sharpe_matches_its_definition(self):
        r = self.rng.normal(0.001, 0.01, 500)
        expected = (np.mean(r) / np.std(r, ddof=1)) * math.sqrt(365.0)
        self.assertAlmostEqual(sharpe_ratio(r, 365.0), expected, places=9)

    def test_max_drawdown_is_peak_to_trough(self):
        # 100 -> 120 -> 60 -> 90: worst fall is 120 -> 60, i.e. 50%.
        self.assertAlmostEqual(max_drawdown_pct(curve([100, 120, 60, 90])), 50.0)

    def test_monotonic_curve_has_no_drawdown(self):
        self.assertAlmostEqual(max_drawdown_pct(curve([100, 110, 120])), 0.0)

    def test_sortino_exceeds_sharpe_when_upside_is_fat(self):
        """Upside volatility is not risk; Sharpe wrongly penalises it."""
        r = np.concatenate([self.rng.normal(0.0005, 0.005, 400),
                            self.rng.normal(0.05, 0.005, 30)])
        self.assertGreater(sortino_ratio(r, 365.0), sharpe_ratio(r, 365.0))

    def test_ratios_refuse_to_report_on_tiny_samples(self):
        """The defect this guards: a confident number from almost no data."""
        tiny = self.rng.normal(0.001, 0.01, MIN_OBSERVATIONS - 1)
        self.assertIsNone(sharpe_ratio(tiny, 365.0))
        self.assertIsNone(sortino_ratio(tiny, 365.0))

    def test_drawdown_is_reported_even_on_tiny_samples(self):
        """A drawdown is an observed fact about the path, not an estimate."""
        self.assertIsNotNone(max_drawdown_pct(curve([100, 50])))

    def test_zero_variance_returns_none_not_infinity(self):
        flat = np.zeros(100)
        self.assertIsNone(sharpe_ratio(flat, 365.0))
        self.assertIsNone(sortino_ratio(flat, 365.0))

    def test_calmar_handles_a_drawdown_free_run(self):
        self.assertIsNone(calmar_ratio(10.0, 0.0))
        self.assertAlmostEqual(calmar_ratio(10.0, 5.0), 2.0)


class TestDescribeRun(unittest.TestCase):

    def test_short_run_warns_rather_than_reporting_zero(self):
        """The exact regression: the UI displayed a confident 0."""
        out = describe_run(curve([100, 101, 102]), total_return_pct=2.0)
        self.assertIsNone(out["sharpe_ratio"])
        self.assertIsNotNone(out["sample_warning"])
        self.assertNotEqual(out["sharpe_ratio"], 0)

    def test_long_run_reports_everything(self):
        rng = np.random.default_rng(1)
        values, v = [10000.0], 10000.0
        for _ in range(200):
            v *= 1 + rng.normal(0.001, 0.01)
            values.append(v)
        out = describe_run(curve(values), total_return_pct=(v / 10000 - 1) * 100)
        self.assertIsNone(out["sample_warning"])
        for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
                    "volatility_annual_pct"):
            self.assertIsInstance(out[key], float, key)


class TestBootstrap(unittest.TestCase):

    def test_ci_contains_the_true_mean(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0.01, 0.02, 400)
        res = bootstrap_ci(x, n_resamples=800)
        self.assertLess(res["lo"], 0.01)
        self.assertGreater(res["hi"], 0.01)

    def test_null_rejection_rate_is_near_nominal(self):
        """The calibration that makes every other number trustworthy.

        Two arms drawn from the SAME distribution must be called different at
        roughly alpha, not far more often. A bootstrap that ignores serial
        correlation fails here by rejecting far too often.
        """
        rng = np.random.default_rng(5)
        trials, rejects = 120, 0
        for _ in range(trials):
            a = rng.normal(0.0, 0.01, 100)
            b = rng.normal(0.0, 0.01, 100)
            p = bootstrap_difference(a, b, n_resamples=400,
                                     seed=int(rng.integers(1e9)))["p_value"]
            if p is not None and p <= 0.05:
                rejects += 1
        rate = rejects / trials
        self.assertLess(rate, 0.15, f"rejection rate {rate:.1%} far above 5%")

    def test_detects_a_real_difference(self):
        rng = np.random.default_rng(6)
        a = rng.normal(0.005, 0.01, 200)
        b = rng.normal(0.0, 0.01, 200)
        res = bootstrap_difference(a, b, n_resamples=800)
        self.assertLess(res["p_value"], 0.05)
        self.assertTrue(res["excludes_zero"])

    def test_p_value_is_never_exactly_zero(self):
        """A finite resample cannot honestly report p = 0."""
        rng = np.random.default_rng(8)
        a = rng.normal(1.0, 0.01, 100)
        b = rng.normal(0.0, 0.01, 100)
        res = bootstrap_difference(a, b, n_resamples=200)
        self.assertGreater(res["p_value"], 0.0)

    def test_equal_length_arms_are_paired(self):
        """Pairing removes the shared market move that dominates variance."""
        rng = np.random.default_rng(9)
        a, b = rng.normal(0, 0.01, 50), rng.normal(0, 0.01, 50)
        self.assertTrue(bootstrap_difference(a, b, n_resamples=200)["paired"])
        self.assertFalse(
            bootstrap_difference(a, b[:40], n_resamples=200)["paired"])

    def test_is_reproducible_by_default(self):
        rng = np.random.default_rng(10)
        x = rng.normal(0, 1, 80)
        self.assertEqual(bootstrap_ci(x, n_resamples=300)["lo"],
                         bootstrap_ci(x, n_resamples=300)["lo"])

    def test_degenerate_input_does_not_raise(self):
        for bad in ([], [1.0], [float("nan"), float("inf")]):
            with self.subTest(sample=bad):
                self.assertIsNone(bootstrap_ci(bad)["point"])


class TestHolmCorrection(unittest.TestCase):

    def test_matches_the_worked_example(self):
        out = holm_correction({"a": 0.01, "b": 0.02, "c": 0.04, "d": 0.60})
        self.assertAlmostEqual(out["a"]["p_adjusted"], 0.04)   # 4 * 0.01
        self.assertAlmostEqual(out["b"]["p_adjusted"], 0.06)   # 3 * 0.02
        self.assertAlmostEqual(out["c"]["p_adjusted"], 0.08)   # 2 * 0.04
        self.assertAlmostEqual(out["d"]["p_adjusted"], 0.60)   # 1 * 0.60

    def test_only_the_smallest_survives_the_family(self):
        out = holm_correction({"a": 0.01, "b": 0.02, "c": 0.04, "d": 0.60})
        self.assertTrue(out["a"]["significant"])
        for key in ("b", "c", "d"):
            self.assertFalse(out[key]["significant"], key)

    def test_adjusted_p_is_monotonic(self):
        """Step-down: an adjusted p can never fall below an earlier one."""
        out = holm_correction({"a": 0.02, "b": 0.021, "c": 0.022})
        adj = [out[k]["p_adjusted"] for k in ("a", "b", "c")]
        self.assertEqual(adj, sorted(adj))

    def test_correction_is_never_weaker_than_the_raw_p(self):
        out = holm_correction({"a": 0.01, "b": 0.2, "c": 0.5})
        for key, res in out.items():
            self.assertGreaterEqual(res["p_adjusted"], res["p_value"], key)

    def test_unevaluated_arms_do_not_shrink_the_family(self):
        """A failed run must not make surviving comparisons look better."""
        with_fail = holm_correction({"a": 0.01, "b": 0.02, "c": None})
        self.assertEqual(with_fail["a"]["family_size"], 2)
        self.assertIsNone(with_fail["c"]["significant"])


class TestDeflatedSharpe(unittest.TestCase):

    def test_best_of_many_worthless_strategies_fails(self):
        """The selection bias this exists to catch.

        The maximum of N noisy Sharpe estimates is biased upward even when
        every strategy is worthless, so an eight-arm ablation's winner needs
        deflating before its Sharpe can be quoted.
        """
        rng = np.random.default_rng(11)
        failures = 0
        for _ in range(6):
            best = max((rng.normal(0.0, 0.01, 250) for _ in range(8)),
                       key=lambda r: r.mean() / r.std())
            if not deflated_sharpe_ratio(best, 8, 365.0)["passes_at_95"]:
                failures += 1
        self.assertGreaterEqual(failures, 5,
                                "worthless best-of-8 should almost never pass")

    def test_a_genuinely_good_strategy_passes(self):
        """The complement: deflation must not reject everything."""
        rng = np.random.default_rng(12)
        good = rng.normal(0.004, 0.01, 250)
        self.assertTrue(deflated_sharpe_ratio(good, 8, 365.0)["passes_at_95"])

    def test_more_trials_makes_passing_harder(self):
        rng = np.random.default_rng(13)
        r = rng.normal(0.003, 0.01, 300)
        scores = [deflated_sharpe_ratio(r, n, 365.0)["deflated_sharpe"]
                  for n in (1, 10, 1000, 100000)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_tiny_sample_is_refused(self):
        rng = np.random.default_rng(14)
        out = deflated_sharpe_ratio(rng.normal(0, 1, 5), 8, 365.0)
        self.assertIsNone(out["deflated_sharpe"])
        self.assertIn("note", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
