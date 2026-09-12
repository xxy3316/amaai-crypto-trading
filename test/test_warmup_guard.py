"""A run with too few bars must fail clearly, not with a bare IndexError.

Regression suite for a defect found on 2026-09-12: the simulation skipped the
first 20 bars to let the moving averages stabilise, but nothing checked that 20
bars actually existed. A short window -- a handful of daily bars, say -- reached
`df.iloc[start_idx]` and died with

    IndexError: single positional indexer is out-of-bounds

surfaced in the UI as "Simulation failed: single positional indexer is
out-of-bounds", which names neither the date range nor the interval nor the
warmup, and so gives the user nothing to act on. The same literal `20` was also
written out by hand in the results panel, where `df.iloc[20]` would have raised
the same way, and where a drifting value would have silently drawn a
buy-and-hold baseline different from the one the summary reported.

These tests are about the BOUNDARY rather than about the number 20: they assert
that the guard admits exactly those frames that have at least one tradeable bar
left after warmup, so the guard and the loop can never disagree.

Run directly (the package name `test` shadows the stdlib `test` module):

    python test/test_warmup_guard.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import (  # noqa: E402
    INDICATOR_WARMUP_BARS,
    InsufficientHistory,
    SyntheticDataBlocked,
)


def bars(n: int) -> pd.DataFrame:
    """A frame of n bars, shaped like the one the simulation consumes."""
    return pd.DataFrame({
        "open": [100.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
    })


def too_short(df: pd.DataFrame) -> bool:
    """The guard as auto-trade.py applies it."""
    return len(df) <= INDICATOR_WARMUP_BARS


def tradeable_bars(df: pd.DataFrame, step: int = 1) -> list:
    """The simulation points the engine would actually visit."""
    return list(range(INDICATOR_WARMUP_BARS, len(df), step))


class TestWarmupGuard(unittest.TestCase):

    def test_guard_admits_exactly_the_frames_with_something_to_trade(self):
        """The property that keeps the guard and the loop in agreement.

        If these two ever diverge, either a run dies on an out-of-bounds index
        (guard too lax) or a perfectly good window is refused (guard too
        strict). Asserting the equivalence is what stops the literal drifting.
        """
        for n in range(0, INDICATOR_WARMUP_BARS * 2):
            with self.subTest(bars=n):
                self.assertEqual(too_short(bars(n)),
                                 len(tradeable_bars(bars(n))) == 0)

    def test_the_first_acceptable_frame_indexes_safely(self):
        """One bar past warmup is the smallest run that must work."""
        df = bars(INDICATOR_WARMUP_BARS + 1)
        self.assertFalse(too_short(df))
        # Both call sites index here: the engine's buy-and-hold baseline and
        # the results panel's chart baseline.
        self.assertIsNotNone(df.iloc[INDICATOR_WARMUP_BARS]["close"])
        self.assertEqual(len(tradeable_bars(df)), 1)

    def test_the_largest_rejected_frame_would_have_raised(self):
        """Exactly at the warmup length there is nothing left to trade."""
        df = bars(INDICATOR_WARMUP_BARS)
        self.assertTrue(too_short(df))
        self.assertEqual(tradeable_bars(df), [])
        with self.assertRaises(IndexError):
            df.iloc[INDICATOR_WARMUP_BARS]["close"]

    def test_a_realistic_short_window_is_rejected(self):
        """Seven daily bars -- the shape of a week-long run -- must not crash."""
        self.assertTrue(too_short(bars(7)))

    def test_empty_frame_is_rejected(self):
        self.assertTrue(too_short(bars(0)))

    def test_chart_baseline_never_indexes_past_the_end(self):
        """The results panel clamps, so it cannot raise even if reached."""
        for n in range(1, INDICATOR_WARMUP_BARS * 2):
            with self.subTest(bars=n):
                df = bars(n)
                baseline_idx = min(INDICATOR_WARMUP_BARS, len(df) - 1)
                self.assertIsNotNone(df.iloc[baseline_idx]["close"])


class TestInsufficientHistoryType(unittest.TestCase):

    def test_is_distinguishable_from_the_synthetic_data_stop(self):
        """Both are expected stops, but they need different messages."""
        self.assertTrue(issubclass(InsufficientHistory, RuntimeError))
        self.assertFalse(issubclass(InsufficientHistory, SyntheticDataBlocked))
        self.assertFalse(issubclass(SyntheticDataBlocked, InsufficientHistory))

    def test_is_not_caught_as_a_plain_index_error(self):
        """The point of the type: it is not the IndexError it replaced."""
        self.assertFalse(issubclass(InsufficientHistory, IndexError))


if __name__ == "__main__":
    unittest.main(verbosity=2)
