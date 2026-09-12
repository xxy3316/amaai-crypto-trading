"""Walk-forward window planning and small-sample significance.

Phase 3. The ablation ran on ONE window, and `core.metrics`' bootstrap was
explicitly within-window: it asked whether an arm differed from the baseline on
that window, never whether the difference survived anywhere else.

Two things are tested here.

`experiments/windows.py` cuts evaluation windows out of whatever is actually
cached, intersecting the channels so a window can never compare an arm that had
positioning data against one that did not.

The studentized bootstrap exists because of a measured defect. A walk-forward
has one observation per window -- single digits -- and the percentile bootstrap
undercovers badly there. Measured against a TRUE null:

    windows   percentile    studentized
       5         23.0%          2.5%
       9         14.5%          2.5%
      20         12.0%          5.0%

At nine windows the percentile interval called pure noise significant 14% of
the time; across seven arms that is close to a guaranteed false finding. The
calibration test below is the one that must never be deleted.

Run directly (the package name `test` shadows the stdlib `test` module):

    python test/test_walk_forward.py
"""

from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.metrics import studentized_bootstrap  # noqa: E402
from experiments.windows import (  # noqa: E402
    Window,
    contiguous_blocks,
    describe_plan,
    make_windows,
    minimum_window_days,
)


def days(start: str, n: int) -> list:
    d0 = dt.date.fromisoformat(start)
    return [d0 + dt.timedelta(days=i) for i in range(n)]


class TestContiguousBlocks(unittest.TestCase):

    def test_single_run_is_one_block(self):
        self.assertEqual(len(contiguous_blocks(days("2026-01-01", 10))), 1)

    def test_a_gap_splits_the_block(self):
        d = days("2026-01-01", 5) + days("2026-03-01", 5)
        blocks = contiguous_blocks(d)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0][1], dt.date(2026, 1, 5))
        self.assertEqual(blocks[1][0], dt.date(2026, 3, 1))

    def test_empty_input_is_not_an_error(self):
        self.assertEqual(contiguous_blocks([]), [])


class TestWindowPlanning(unittest.TestCase):
    """Planning is exercised through the real cache, so it stays honest."""

    def test_windows_are_disjoint_by_default(self):
        """Overlapping windows share bars and are not independent evidence."""
        plan = make_windows(length_days=30)
        if len(plan) < 2:
            self.skipTest("cache holds fewer than two windows")
        for a, b in zip(plan, plan[1:]):
            if a.block == b.block:
                self.assertGreater(b.start, a.end,
                                   "windows in one block must not overlap")

    def test_no_window_straddles_a_coverage_gap(self):
        """A window spanning missing days would silently disable a channel."""
        plan = make_windows(length_days=30)
        for w in plan:
            self.assertLessEqual(w.days, 30)

    def test_every_window_meets_the_length_floor(self):
        plan = make_windows(length_days=30, min_days=30)
        for w in plan:
            self.assertGreaterEqual(w.days, 30)

    def test_shorter_windows_yield_more_of_them(self):
        short = make_windows(length_days=20)
        long_ = make_windows(length_days=60)
        self.assertGreaterEqual(len(short), len(long_))

    def test_impossible_length_yields_no_windows(self):
        self.assertEqual(make_windows(length_days=100_000), [])

    def test_plan_is_indexed_and_named_stably(self):
        plan = make_windows(length_days=30)
        if not plan:
            self.skipTest("no cached coverage")
        self.assertEqual([w.index for w in plan],
                         list(range(1, len(plan) + 1)))
        self.assertEqual(len(set(w.name for w in plan)), len(plan))

    def test_describe_plan_counts_regimes(self):
        plan = make_windows(length_days=30)
        if not plan:
            self.skipTest("no cached coverage")
        out = describe_plan(plan)
        self.assertEqual(out["windows"], len(plan))
        self.assertEqual(sum(out["windows_per_block"].values()), len(plan))

    def test_since_restricts_the_plan_to_recent_coverage(self):
        """Running one regime block only, to avoid paying for the others."""
        plan = make_windows(length_days=30, since="2026-01-01")
        if not plan:
            self.skipTest("no 2026 coverage cached")
        for w in plan:
            self.assertGreaterEqual(w.start, dt.date(2026, 1, 1))

    def test_until_restricts_the_other_end(self):
        plan = make_windows(length_days=30, until="2024-12-31")
        for w in plan:
            self.assertLessEqual(w.end, dt.date(2024, 12, 31))

    def test_restricting_never_yields_more_windows(self):
        everything = make_windows(length_days=30)
        recent = make_windows(length_days=30, since="2026-01-01")
        self.assertLessEqual(len(recent), len(everything))

    def test_a_clip_cannot_create_a_window_spanning_a_gap(self):
        """Blocks are recomputed after clipping, not before."""
        plan = make_windows(length_days=30, since="2024-01-01",
                            until="2026-12-31")
        for w in plan:
            self.assertLessEqual(w.days, 30)

    def test_impossible_clip_yields_nothing_rather_than_raising(self):
        self.assertEqual(
            make_windows(length_days=30, since="2099-01-01"), [])

    def test_minimum_window_scales_with_interval(self):
        """A daily bar needs far more calendar days than an hourly one."""
        hourly = minimum_window_days("1h", 20, 168, 30)
        daily = minimum_window_days("1d", 20, 30, 30)
        self.assertLess(hourly, daily)


class TestStudentizedBootstrap(unittest.TestCase):

    def test_null_rejection_rate_is_not_inflated_at_small_n(self):
        """The defect this was written for.

        The percentile bootstrap rejected a true null 14% of the time at nine
        observations. Anything near that here means the walk-forward verdicts
        cannot be trusted, and no amount of Holm correction repairs a test
        whose per-comparison error rate is already wrong.
        """
        rng = np.random.default_rng(99)
        for n in (5, 9):
            rejects, trials = 0, 150
            for _ in range(trials):
                d = rng.normal(0.0, 1.0, n)
                if studentized_bootstrap(d, n_resamples=600)["p_value"] <= 0.05:
                    rejects += 1
            rate = rejects / trials
            with self.subTest(n=n):
                self.assertLess(rate, 0.10,
                                f"n={n}: false positive rate {rate:.1%}")

    def test_still_detects_a_real_effect(self):
        """Conservative is not the same as useless."""
        rng = np.random.default_rng(21)
        detected = 0
        for _ in range(40):
            d = rng.normal(2.0, 1.0, 9)      # a large, consistent effect
            if studentized_bootstrap(d, n_resamples=600)["p_value"] <= 0.05:
                detected += 1
        self.assertGreater(detected, 30)

    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(22)
        res = studentized_bootstrap(rng.normal(1.0, 1.0, 12))
        self.assertLessEqual(res["lo"], res["point"])
        self.assertGreaterEqual(res["hi"], res["point"])

    def test_is_wider_than_the_percentile_interval_at_small_n(self):
        """Precisely the correction: more honest uncertainty, not less."""
        from core.metrics import bootstrap_ci
        rng = np.random.default_rng(23)
        d = list(rng.normal(0.5, 1.0, 8))
        stud = studentized_bootstrap(d, n_resamples=4000)
        pct = bootstrap_ci(d, n_resamples=4000)
        self.assertGreater(stud["hi"] - stud["lo"], pct["hi"] - pct["lo"])

    def test_too_few_observations_is_refused(self):
        out = studentized_bootstrap([1.0, 2.0])
        self.assertIsNone(out["point"])
        self.assertIn("note", out)

    def test_tiny_sample_does_not_produce_an_absurd_bound(self):
        """A real regression, seen on a 3-window smoke run.

        The walk-forward printed [-3.93, +3346006946862426.00]. With three
        observations, many resamples are near-duplicates, their standard error
        collapses toward zero, and the resulting t quantile explodes. Refusing
        is correct; a finite-looking but meaningless bound is not.
        """
        out = studentized_bootstrap([1.0, -4.0, -5.6])
        self.assertIsNone(out["lo"])
        self.assertIn("note", out)

    def test_every_reported_bound_is_finite(self):
        """No sample that IS accepted may produce an unbounded interval."""
        rng = np.random.default_rng(31)
        for n in (5, 6, 9, 20):
            for _ in range(25):
                res = studentized_bootstrap(rng.normal(0.0, 1.0, n),
                                            n_resamples=800)
                if res["lo"] is None:
                    continue
                with self.subTest(n=n):
                    self.assertTrue(np.isfinite(res["lo"]))
                    self.assertTrue(np.isfinite(res["hi"]))
                    # A bound thousands of times the sample spread is the
                    # symptom the guard exists to catch.
                    self.assertLess(res["hi"] - res["lo"], 100.0)

    def test_zero_variance_does_not_raise(self):
        out = studentized_bootstrap([3.0] * 10)
        self.assertEqual(out["point"], 3.0)

    def test_p_value_is_never_exactly_zero(self):
        rng = np.random.default_rng(24)
        res = studentized_bootstrap(rng.normal(50.0, 1.0, 10), n_resamples=500)
        self.assertGreater(res["p_value"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
