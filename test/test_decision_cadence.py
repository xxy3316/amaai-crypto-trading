"""Decision cadence is a duration, so it must convert per bar interval.

Regression suite for a defect found on 2026-09-12: `simulation_step = 6` was a
bar COUNT written for hourly bars -- the comment in test_date_filtering.py says
"every 6 hours" outright -- and nothing converted it for other intervals. On
daily bars it therefore meant every six DAYS. A 30-day daily backtest produced

    31 bars - 20 warmup = 11 tradeable, range(20, 31, 6) = 2 decision points

two decisions, one completed round trip, and an executive summary reporting
"Win Rate 50.0%" and "BEAT the market" off a single trade. Nothing in the UI
disclosed the sample size; the mode blurb promised "10+ trades per week" while
the engine could not exceed one decision per six days, since no preset changes
simulation_step at all.

These tests are about the INVARIANT rather than about the number 6: the gap
between decisions must stay close to the configured cadence whatever the
interval, and must never round to zero.

Run directly (the package name `test` shadows the stdlib `test` module):

    python test/test_decision_cadence.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import (  # noqa: E402
    DECISION_CADENCE_HOURS,
    INDICATOR_WARMUP_BARS,
    decision_step_bars,
    interval_hours,
)

#: Intervals the app actually offers.
UI_INTERVALS = ("1h", "4h", "1d")


class TestIntervalHours(unittest.TestCase):

    def test_minute_and_month_are_not_confused(self):
        """Binance spells minutes '1m' and months '1M'. Folding case breaks it."""
        self.assertEqual(interval_hours("1m"), 1 / 60)
        self.assertEqual(interval_hours("1M"), 720.0)

    def test_case_drift_is_tolerated_where_it_is_unambiguous(self):
        self.assertEqual(interval_hours("1D"), 24.0)
        self.assertEqual(interval_hours("4H"), 4.0)

    def test_unknown_interval_is_reported_as_unknown(self):
        for bad in (None, "", "fortnight"):
            with self.subTest(interval=bad):
                self.assertIsNone(interval_hours(bad))


class TestDecisionStep(unittest.TestCase):

    def test_hourly_is_unchanged(self):
        """The historical behaviour this was tuned for must not move."""
        self.assertEqual(decision_step_bars("1h"), 6)

    def test_daily_decides_every_bar(self):
        """The bug: six DAYS between decisions on a daily run."""
        self.assertEqual(decision_step_bars("1d"), 1)

    def test_step_is_never_zero(self):
        """A cadence finer than one bar must clamp, not divide by zero.

        Without the floor, a cadence shorter than the bar rounds to 0 and
        range(start, stop, 0) raises ValueError.
        """
        for interval in ("1d", "1w", "1M"):
            with self.subTest(interval=interval):
                self.assertGreaterEqual(
                    decision_step_bars(interval, cadence_hours=1.0), 1)

    def test_realised_cadence_tracks_the_requested_one(self):
        """The invariant that makes this interval-independent.

        Where the bar is no coarser than the cadence, the gap between decisions
        must land within one bar of what was asked for.
        """
        for interval in UI_INTERVALS:
            with self.subTest(interval=interval):
                bar = interval_hours(interval)
                if bar > DECISION_CADENCE_HOURS:
                    continue  # cannot decide more often than the data updates
                realised = decision_step_bars(interval) * bar
                self.assertLessEqual(abs(realised - DECISION_CADENCE_HOURS), bar)

    def test_coarse_bars_decide_every_bar_rather_than_skipping(self):
        """A bar coarser than the cadence should decide as often as it can."""
        for interval in ("1d", "1w", "1M"):
            with self.subTest(interval=interval):
                self.assertEqual(decision_step_bars(interval), 1)

    def test_unknown_interval_keeps_the_historical_bar_count(self):
        """Unmeasurable bar size: fall back, do not invent a cadence."""
        for bad in (None, "", "fortnight"):
            with self.subTest(interval=bad):
                self.assertEqual(decision_step_bars(bad),
                                 round(DECISION_CADENCE_HOURS))


class TestTheReportedRun(unittest.TestCase):
    """The exact 30-day daily run that exposed this."""

    BARS = 31  # 2026-08-13 -> 2026-09-12 inclusive

    def points(self, step: int) -> list:
        return list(range(INDICATOR_WARMUP_BARS, self.BARS, step))

    def test_old_behaviour_gave_two_decisions(self):
        self.assertEqual(len(self.points(6)), 2)

    def test_fixed_behaviour_uses_every_tradeable_bar(self):
        step = decision_step_bars("1d")
        self.assertEqual(len(self.points(step)),
                         self.BARS - INDICATOR_WARMUP_BARS)
        self.assertEqual(len(self.points(step)), 11)

    def test_the_run_still_warrants_a_small_sample_warning(self):
        """Fixing the cadence does not make 11 decisions publishable."""
        self.assertLess(len(self.points(decision_step_bars("1d"))), 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
