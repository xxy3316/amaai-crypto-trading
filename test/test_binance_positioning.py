"""Offline tests for the Binance positioning signal.

Runs with no network access. The CSV rows below are hand-built TEST FIXTURES
whose only purpose is to exercise the parser and the alignment arithmetic; they
are unreachable from the backtest and are never presented as measurements.

The important test here is `test_no_lookahead_*`: it is the property a reviewer
will ask about, so it is asserted rather than asserted-in-prose.

    ./venv/Scripts/python.exe -m unittest test.test_binance_positioning -v
"""

from __future__ import annotations

import datetime as dt
import io
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signals.binance_positioning import (  # noqa: E402
    DIRECTIONAL_FEATURES,
    RAW_COLUMNS,
    PositioningSignalAgent,
    _parse_zip_bytes,
    _to_utc_naive,
    align_to_bars,
    daterange,
    fetch_range,
    known_missing,
    load_cached_range,
    missing_days,
    normalise_symbol,
    score_row,
    warmup_start_date,
)
from core.config import (  # noqa: E402
    POSITIONING_ZSCORE_WINDOW,
    POSITIONING_ZSCORE_WINDOW_DAILY,
    positioning_zscore_window,
)


# ── fixture helpers ─────────────────────────────────────────────────────────

def make_rows(n: int, start: str = "2024-01-02 00:00:00", step_min: int = 5,
              crowd: float = 2.0, top: float = 1.5, taker: float = 1.0,
              oi: float = 100000.0, jitter: float = 0.0):
    """n metrics rows on the 5-minute grid.

    `jitter` adds per-row variation. It matters for any test that asserts on
    z-scores: a series held constant across a whole rolling window has zero
    variance and therefore NO z-score, which is correct behaviour but makes a
    constant fixture look like a coverage bug.
    """
    t0 = pd.Timestamp(start)
    rng = np.random.default_rng(abs(hash(start)) % (2 ** 32))
    rows = []
    for i in range(n):
        wobble = float(rng.normal(0.0, jitter)) if jitter else 0.0
        rows.append({
            "create_time": (t0 + pd.Timedelta(minutes=step_min * i)).strftime(
                "%Y-%m-%d %H:%M:%S"),
            "symbol": "BTCUSDT",
            "sum_open_interest": oi + i,
            "sum_open_interest_value": (oi + i) * 42000.0,
            "count_toptrader_long_short_ratio": 2.1,
            "sum_toptrader_long_short_ratio": top + wobble,
            "count_long_short_ratio": crowd + wobble,
            "sum_taker_long_short_vol_ratio": taker + wobble,
        })
    return rows


def make_zip(rows, header: bool = True, name: str = "BTCUSDT-metrics-2024-01-02.csv") -> bytes:
    frame = pd.DataFrame(rows)[RAW_COLUMNS]
    csv_text = frame.to_csv(index=False, header=header)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, csv_text)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text


class FakeSession:
    """Minimal stand-in for requests.Session, recording every URL requested."""

    def __init__(self, routes: dict, default=404):
        self.routes = routes
        self.default = default
        self.headers: dict = {}
        self.calls: list = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        hit = self.routes.get(url)
        if hit is None:
            return FakeResponse(self.default)
        return hit

    def close(self):
        pass


# ── symbol + date helpers ───────────────────────────────────────────────────

class TestHelpers(unittest.TestCase):

    def test_normalise_symbol(self):
        for raw, want in [
            ("BTC/USDT", "BTCUSDT"),
            ("BTC/USDT:USDT", "BTCUSDT"),
            ("btcusdt", "BTCUSDT"),
            ("ETH-USDT", "ETHUSDT"),
            ("  sol/usdt ", "SOLUSDT"),
        ]:
            self.assertEqual(normalise_symbol(raw), want, msg=raw)

    def test_normalise_symbol_rejects_empty(self):
        with self.assertRaises(ValueError):
            normalise_symbol("")

    def test_daterange_inclusive(self):
        days = daterange("2024-01-01", "2024-01-03")
        self.assertEqual(days, [dt.date(2024, 1, 1), dt.date(2024, 1, 2),
                                dt.date(2024, 1, 3)])

    def test_daterange_rejects_reversed(self):
        with self.assertRaises(ValueError):
            daterange("2024-01-05", "2024-01-01")


# ── parsing ─────────────────────────────────────────────────────────────────

class TestParsing(unittest.TestCase):

    def test_parses_with_header(self):
        frame = _parse_zip_bytes(make_zip(make_rows(12)))
        self.assertEqual(len(frame), 12)
        self.assertEqual(list(frame.columns), RAW_COLUMNS)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["create_time"]))

    def test_parses_without_header(self):
        frame = _parse_zip_bytes(make_zip(make_rows(6), header=False))
        self.assertEqual(len(frame), 6)
        self.assertEqual(list(frame.columns), RAW_COLUMNS)

    def test_rejects_missing_column(self):
        frame = pd.DataFrame(make_rows(4)).drop(columns=["count_long_short_ratio"])
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("x.csv", frame.to_csv(index=False))
        with self.assertRaises(ValueError) as ctx:
            _parse_zip_bytes(buf.getvalue())
        self.assertIn("count_long_short_ratio", str(ctx.exception))

    def test_rejects_non_zip(self):
        with self.assertRaises(ValueError):
            _parse_zip_bytes(b"this is not a zip file")

    def test_deduplicates_and_sorts(self):
        rows = make_rows(4) + make_rows(2)  # first two timestamps repeat
        frame = _parse_zip_bytes(make_zip(rows))
        self.assertEqual(len(frame), 4)
        self.assertTrue(frame["create_time"].is_monotonic_increasing)

    def test_timestamps_are_tz_naive_utc(self):
        """Must match the tz-naive price index or merge_asof refuses to join."""
        frame = _parse_zip_bytes(make_zip(make_rows(3)))
        self.assertIsNone(frame["create_time"].dt.tz)

    def test_to_utc_naive_accepts_epoch_ms(self):
        epoch = pd.Series([1704153600000, 1704153900000])
        out = _to_utc_naive(epoch)
        self.assertIsNone(out.dt.tz)
        self.assertEqual(out.iloc[0], pd.Timestamp("2024-01-02 00:00:00"))

    def test_to_utc_naive_accepts_strings(self):
        out = _to_utc_naive(pd.Series(["2024-01-02 00:00:00", "2024-01-02 00:05:00"]))
        self.assertIsNone(out.dt.tz)
        self.assertEqual(out.iloc[1], pd.Timestamp("2024-01-02 00:05:00"))


# ── point-in-time alignment: the properties reviewers ask about ─────────────

class TestNoLookahead(unittest.TestCase):

    def setUp(self):
        # A step change at 06:00 lets us prove the bar at 06:00 cannot see it.
        early = make_rows(72, start="2024-01-02 00:00:00", crowd=2.0)
        late = make_rows(72, start="2024-01-02 06:00:00", crowd=9.0)
        self.raw = _parse_zip_bytes(make_zip(early + late))
        self.bars = pd.date_range("2024-01-02 00:00:00", periods=12, freq="h")

    def test_bar_cannot_see_its_own_timestamp(self):
        """With lag_bars=1 the 06:00 bar sees 05:00 data, not the 06:00 jump."""
        aligned = align_to_bars(self.raw, self.bars, lag_bars=1, zscore_window=8)
        at_jump = aligned.loc[pd.Timestamp("2024-01-02 06:00:00")]
        self.assertEqual(at_jump["create_time"], pd.Timestamp("2024-01-02 05:00:00"))
        self.assertAlmostEqual(at_jump["count_long_short_ratio"], 2.0, places=6)

    def test_step_change_arrives_one_bar_later(self):
        aligned = align_to_bars(self.raw, self.bars, lag_bars=1, zscore_window=8)
        after = aligned.loc[pd.Timestamp("2024-01-02 07:00:00")]
        self.assertAlmostEqual(after["count_long_short_ratio"], 9.0, places=6)

    def test_zero_lag_still_excludes_future(self):
        """Even at lag_bars=0 the match is backward-looking: <= t, never > t."""
        aligned = align_to_bars(self.raw, self.bars, lag_bars=0, zscore_window=8)
        for ts in self.bars:
            matched = aligned.loc[ts, "create_time"]
            if pd.notna(matched):
                self.assertLessEqual(matched, ts, msg=f"future data reached bar {ts}")

    def test_zscores_are_causal(self):
        """Truncating the future must not change any past z-score.

        This is the strongest available check that the rolling statistics are
        right-aligned: if any future observation leaked into an earlier window,
        the truncated run would disagree with the full run.
        """
        full = align_to_bars(self.raw, self.bars, lag_bars=1, zscore_window=6)
        half_bars = self.bars[:8]
        truncated_raw = self.raw[self.raw["create_time"] <= half_bars[-1]]
        half = align_to_bars(truncated_raw, half_bars, lag_bars=1, zscore_window=6)

        for col in DIRECTIONAL_FEATURES:
            left = full[f"z_{col}"].loc[half.index]
            right = half[f"z_{col}"]
            pd.testing.assert_series_equal(
                left, right, check_names=False, rtol=1e-9, atol=1e-12,
            )

    def test_staleness_is_reported(self):
        aligned = align_to_bars(self.raw, self.bars, lag_bars=1, zscore_window=8)
        staleness = aligned["staleness_bars"].dropna()
        self.assertTrue((staleness >= 0).all())
        # 5-minute data against hourly bars is at most a fraction of a bar stale.
        self.assertLess(staleness.max(), 1.0)

    def test_stale_readings_are_masked_not_carried_forward(self):
        """A long gap must become NaN rather than a silently reused old value."""
        raw = _parse_zip_bytes(make_zip(make_rows(12, start="2024-01-02 00:00:00")))
        bars = pd.date_range("2024-01-02 00:00:00", periods=12, freq="h")
        aligned = align_to_bars(raw, bars, lag_bars=1, zscore_window=6,
                                max_staleness_bars=3.0)
        # Raw data stops at 00:55, so bars from ~05:00 onward are >3 bars stale.
        self.assertTrue(pd.isna(aligned.loc[pd.Timestamp("2024-01-02 09:00:00"),
                                            "count_long_short_ratio"]))

    def test_empty_inputs_are_safe(self):
        self.assertTrue(align_to_bars(pd.DataFrame(), self.bars).empty)
        out = align_to_bars(self.raw, pd.DatetimeIndex([]))
        self.assertEqual(len(out), 0)

    def test_output_is_indexed_like_the_bars(self):
        aligned = align_to_bars(self.raw, self.bars, lag_bars=1, zscore_window=8)
        pd.testing.assert_index_equal(aligned.index, self.bars)


# ── scoring: sign conventions must be the documented ones ───────────────────

class TestScoring(unittest.TestCase):

    def _row(self, **z):
        base = {f"z_{c}": float("nan") for c in DIRECTIONAL_FEATURES}
        base.update({c: 1.0 for c in DIRECTIONAL_FEATURES})
        base["staleness_bars"] = 0.5
        base.update({f"z_{k}": v for k, v in z.items()})
        return pd.Series(base)

    def test_crowd_long_reads_bearish(self):
        """Retail crowding long is faded, so a high crowd z must score negative."""
        reading = score_row(self._row(count_long_short_ratio=3.0))
        self.assertLess(reading.score, 0)
        self.assertGreater(reading.bearish_points, 0)
        self.assertEqual(reading.bullish_points, 0)

    def test_top_trader_long_reads_bullish(self):
        reading = score_row(self._row(sum_toptrader_long_short_ratio=3.0))
        self.assertGreater(reading.score, 0)
        self.assertGreater(reading.bullish_points, 0)

    def test_taker_buying_reads_bullish(self):
        reading = score_row(self._row(sum_taker_long_short_vol_ratio=3.0))
        self.assertGreater(reading.score, 0)

    def test_opposing_features_cancel(self):
        reading = score_row(self._row(
            count_long_short_ratio=3.0,            # bearish contribution
            sum_toptrader_long_short_ratio=3.0,    # bullish contribution
        ))
        self.assertAlmostEqual(reading.score, 0.0, places=6)
        self.assertEqual(reading.bullish_points, 0)
        self.assertEqual(reading.bearish_points, 0)

    def test_score_is_bounded(self):
        reading = score_row(self._row(**{c: 99.0 for c in DIRECTIONAL_FEATURES}))
        self.assertLessEqual(abs(reading.score), 1.0)

    def test_points_respect_max_points(self):
        reading = score_row(
            self._row(**{c: 3.0 for c in DIRECTIONAL_FEATURES}), max_points=1)
        self.assertLessEqual(max(reading.bullish_points, reading.bearish_points), 1)

    def test_no_zscores_means_unavailable(self):
        reading = score_row(self._row())
        self.assertFalse(reading.available)
        self.assertEqual(reading.score, 0.0)
        self.assertEqual(reading.bullish_points, 0)

    def test_none_row_is_unavailable(self):
        reading = score_row(None)
        self.assertFalse(reading.available)
        self.assertEqual(reading.source, "none")

    def test_confidence_falls_with_staleness(self):
        fresh = self._row(count_long_short_ratio=2.0)
        fresh["staleness_bars"] = 0.0
        stale = self._row(count_long_short_ratio=2.0)
        stale["staleness_bars"] = 2.5
        self.assertGreater(score_row(fresh).confidence, score_row(stale).confidence)

    def test_reading_serialises(self):
        payload = score_row(self._row(count_long_short_ratio=2.0)).to_dict()
        self.assertIn("score", payload)
        self.assertIn("features", payload)
        self.assertIsInstance(payload["features"], dict)


# ── cache + fetch behaviour ─────────────────────────────────────────────────

class TestCacheAndFetch(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _url(self, date: str) -> str:
        return ("https://data.binance.vision/data/futures/um/daily/metrics/"
                f"BTCUSDT/BTCUSDT-metrics-{date}.zip")

    def test_missing_cache_returns_empty_not_error(self):
        frame = load_cached_range("BTCUSDT", "2024-01-01", "2024-01-03",
                                  cache_root=self.root)
        self.assertTrue(frame.empty)

    def test_fetch_writes_cache_and_manifest(self):
        payload = make_zip(make_rows(288))
        session = FakeSession({
            self._url("2024-01-02"): FakeResponse(200, payload),
        })
        summary = fetch_range("BTC/USDT", "2024-01-02", "2024-01-02",
                              cache_root=self.root, verify_checksums=False,
                              pause_s=0, session=session)
        self.assertEqual(summary["downloaded"], 1)
        self.assertEqual(summary["symbol"], "BTCUSDT")

        cached = self.root / "BTCUSDT" / "BTCUSDT-metrics-2024-01-02.zip"
        self.assertTrue(cached.exists())
        self.assertEqual(cached.read_bytes(), payload, "cache must be byte-exact")

        manifest = self.root / "BTCUSDT" / "manifest.jsonl"
        self.assertTrue(manifest.exists())
        self.assertIn("sha256", manifest.read_text(encoding="utf-8"))

    def test_fetch_is_idempotent(self):
        session = FakeSession({
            self._url("2024-01-02"): FakeResponse(200, make_zip(make_rows(10))),
        })
        kwargs = dict(cache_root=self.root, verify_checksums=False, pause_s=0)
        fetch_range("BTCUSDT", "2024-01-02", "2024-01-02", session=session, **kwargs)
        again = fetch_range("BTCUSDT", "2024-01-02", "2024-01-02",
                            session=session, **kwargs)
        self.assertEqual(again["downloaded"], 0)
        self.assertEqual(again["cached"], 1)

    def test_missing_day_is_a_gap_not_a_crash(self):
        session = FakeSession({
            self._url("2024-01-02"): FakeResponse(200, make_zip(make_rows(10))),
        })  # 01-03 falls through to the 404 default
        summary = fetch_range("BTCUSDT", "2024-01-02", "2024-01-03",
                              cache_root=self.root, verify_checksums=False,
                              pause_s=0, session=session)
        self.assertEqual(summary["downloaded"], 1)
        self.assertEqual(summary["missing"], ["2024-01-03"])
        self.assertEqual(summary["failed"], [])

    def test_checksum_mismatch_aborts(self):
        payload = make_zip(make_rows(10))
        session = FakeSession({
            self._url("2024-01-02"): FakeResponse(200, payload),
            self._url("2024-01-02") + ".CHECKSUM": FakeResponse(
                200, text="0" * 64 + "  BTCUSDT-metrics-2024-01-02.zip"),
        })
        with self.assertRaises(RuntimeError) as ctx:
            fetch_range("BTCUSDT", "2024-01-02", "2024-01-02",
                        cache_root=self.root, verify_checksums=True,
                        pause_s=0, session=session)
        self.assertIn("Checksum mismatch", str(ctx.exception))
        self.assertFalse((self.root / "BTCUSDT" /
                          "BTCUSDT-metrics-2024-01-02.zip").exists(),
                         "corrupt payload must not be cached")

    def test_corrupt_payload_is_not_cached(self):
        session = FakeSession({
            self._url("2024-01-02"): FakeResponse(200, b"not a zip"),
        })
        with self.assertRaises(ValueError):
            fetch_range("BTCUSDT", "2024-01-02", "2024-01-02",
                        cache_root=self.root, verify_checksums=False,
                        pause_s=0, session=session)
        self.assertFalse((self.root / "BTCUSDT" /
                          "BTCUSDT-metrics-2024-01-02.zip").exists())

    def test_load_cached_range_concatenates_days(self):
        for date, start in [("2024-01-02", "2024-01-02 00:00:00"),
                            ("2024-01-03", "2024-01-03 00:00:00")]:
            session = FakeSession({
                self._url(date): FakeResponse(200, make_zip(make_rows(288, start=start))),
            })
            fetch_range("BTCUSDT", date, date, cache_root=self.root,
                        verify_checksums=False, pause_s=0, session=session)
        frame = load_cached_range("BTCUSDT", "2024-01-02", "2024-01-03",
                                  cache_root=self.root)
        self.assertEqual(len(frame), 576)
        self.assertTrue(frame["create_time"].is_monotonic_increasing)


# ── agent behaviour ─────────────────────────────────────────────────────────

class TestPrematureNotFound(unittest.TestCase):
    """A 404 for a day that had not been published yet must not be permanent.

    The regression: every recent day was fetched on the day itself, 404'd
    because Binance had not published it, and was then remembered as absent
    forever. The cache's newest edge stopped advancing and the gap grew by one
    day per day, silently -- the agent simply reported no positioning data.
    """

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dir = self.root / "BTCUSDT"
        self.dir.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _manifest(self, *rows):
        import json
        (self.dir / "manifest.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    @staticmethod
    def _utc(*args):
        return dt.datetime(*args, tzinfo=dt.timezone.utc)

    def test_404_asked_during_the_day_is_retried(self):
        self._manifest({"date": "2026-09-09", "http_status": 404,
                        "pulled_at_utc": "2026-09-09T04:00:00+00:00"})
        skip = known_missing("BTCUSDT", cache_root=self.root,
                             now=self._utc(2026, 9, 12))
        self.assertNotIn("2026-09-09", skip)

    def test_404_asked_long_after_the_day_closed_is_permanent(self):
        # Binance had five days to publish it and did not: it never will.
        self._manifest({"date": "2020-01-01", "http_status": 404,
                        "pulled_at_utc": "2020-01-06T00:00:00+00:00"})
        skip = known_missing("BTCUSDT", cache_root=self.root,
                             now=self._utc(2026, 9, 12))
        self.assertIn("2020-01-01", skip)

    def test_a_day_still_in_progress_is_not_requested(self):
        # Retrying costs a request and cannot possibly succeed.
        self._manifest({"date": "2026-09-12", "http_status": 404,
                        "pulled_at_utc": "2026-09-12T04:00:00+00:00"})
        skip = known_missing("BTCUSDT", cache_root=self.root,
                             now=self._utc(2026, 9, 12, 12))
        self.assertIn("2026-09-12", skip)

    def test_unparseable_pull_time_stays_conservative(self):
        self._manifest({"date": "2021-05-05", "http_status": 404,
                        "pulled_at_utc": "not-a-timestamp"})
        skip = known_missing("BTCUSDT", cache_root=self.root,
                             now=self._utc(2026, 9, 12))
        self.assertIn("2021-05-05", skip)

    def test_premature_404_reaches_missing_days(self):
        """The end-to-end property: a premature gap is actually re-fetched."""
        self._manifest({"date": "2026-09-09", "http_status": 404,
                        "pulled_at_utc": "2026-09-09T04:00:00+00:00"})
        pending = missing_days("BTCUSDT", "2026-09-09", "2026-09-09",
                               cache_root=self.root)
        self.assertEqual([d.isoformat() for d in pending], ["2026-09-09"])


class TestZScoreWindowPerInterval(unittest.TestCase):
    """The z-score baseline is a bar COUNT, so it must depend on bar size.

    The regression: one window of 168 bars was used for every interval. That is
    a week of 1h bars but 168 DAYS of 1d bars, so a daily run needed most of a
    year of warmup history, found a month, produced an all-NaN z-score and
    switched the channel off -- reporting "usable on 0.0% of bars", which reads
    like corrupt data rather than an unsatisfiable warmup.
    """

    def test_daily_gets_the_shorter_window(self):
        self.assertEqual(positioning_zscore_window("1d"),
                         POSITIONING_ZSCORE_WINDOW_DAILY)

    def test_sub_daily_keeps_the_long_window(self):
        for interval in ("1h", "4h", "15m"):
            with self.subTest(interval=interval):
                self.assertEqual(positioning_zscore_window(interval),
                                 POSITIONING_ZSCORE_WINDOW)

    def test_minute_and_month_are_not_confused(self):
        """Binance spells minutes '1m' and months '1M'; case is significant."""
        self.assertEqual(positioning_zscore_window("1m"),
                         POSITIONING_ZSCORE_WINDOW)
        self.assertEqual(positioning_zscore_window("1M"),
                         POSITIONING_ZSCORE_WINDOW_DAILY)

    def test_daily_window_is_satisfiable_from_a_months_cache(self):
        """The point of the change: a daily warmup must be reachable.

        168 daily bars reaches back roughly half a year; the shortened window
        must stay inside the kind of history a run can actually hold.
        """
        bars = pd.date_range("2026-08-13", "2026-09-10", freq="D")
        need = warmup_start_date(bars, positioning_zscore_window("1d"))
        span_days = (bars.min().date() - need).days
        self.assertLessEqual(span_days, 45,
                             "daily warmup must fit in a month or so of cache")

    def test_unknown_interval_falls_back_to_the_long_window(self):
        for interval in (None, "", "nonsense"):
            with self.subTest(interval=interval):
                self.assertEqual(positioning_zscore_window(interval),
                                 POSITIONING_ZSCORE_WINDOW)


class TestAgent(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bars = pd.date_range("2024-01-02 00:00:00", periods=24, freq="h")

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_cache_reports_unavailable_and_never_fakes_zero(self):
        agent = PositioningSignalAgent("BTCUSDT", self.bars, cache_root=self.root)
        self.assertFalse(agent.available)
        self.assertIn("no cached positioning data", agent.status)
        reading = agent.reading_for(self.bars[5])
        self.assertFalse(reading.available)
        self.assertEqual(reading.bullish_points, 0)
        self.assertEqual(reading.bearish_points, 0)
        self.assertIn("unavailable", reading.reasoning.lower())

    def test_disabled_agent_is_inert(self):
        agent = PositioningSignalAgent("BTCUSDT", self.bars, enabled=False,
                                       cache_root=self.root)
        self.assertFalse(agent.available)
        self.assertEqual(agent.status, "disabled")
        self.assertFalse(agent.reading_for(self.bars[0]).available)

    def test_unknown_timestamp_returns_no_data(self):
        agent = PositioningSignalAgent("BTCUSDT", self.bars, cache_root=self.root)
        reading = agent.reading_for(pd.Timestamp("1999-01-01"))
        self.assertFalse(reading.available)

    def test_summary_is_serialisable_when_unavailable(self):
        agent = PositioningSignalAgent("BTCUSDT", self.bars, cache_root=self.root)
        summary = agent.summary()
        self.assertIn("status", summary)
        self.assertFalse(summary["available"])


# ── warm-up: the requested window must be usable from its FIRST bar ─────────

class TestWarmup(unittest.TestCase):
    """The z-scores run on the bar-aligned frame, not the raw 5-minute frame.

    Loading extra raw days is therefore not enough on its own: the aligned
    index must also be extended backwards, or the first `zscore_window` bars of
    every backtest lose their signal for a purely mechanical reason, which
    would weaken the signal-on arm of the ablation.
    """

    ZWIN = 24

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bars = pd.date_range("2024-01-10 00:00:00", periods=48, freq="h")

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_cache(self, first: str, last: str, crowd: float = 2.0):
        """Write one fixture zip per day into the cache, as fetch would."""
        directory = self.root / "BTCUSDT"
        directory.mkdir(parents=True, exist_ok=True)
        for day in daterange(first, last):
            rows = make_rows(288, start=f"{day.isoformat()} 00:00:00",
                             crowd=crowd + np.random.uniform(-0.4, 0.4),
                             jitter=0.15)
            name = f"BTCUSDT-metrics-{day.isoformat()}.zip"
            (directory / name).write_bytes(make_zip(rows, name=name[:-4] + ".csv"))

    def test_warmup_start_scales_with_bar_duration(self):
        """168 bars is a week of hourly bars but 168 days of daily bars."""
        hourly = pd.date_range("2024-06-01", periods=200, freq="h")
        daily = pd.date_range("2024-06-01", periods=200, freq="D")
        self.assertGreater(
            (pd.Timestamp(daily.min()).date() - warmup_start_date(daily, 168)).days,
            (pd.Timestamp(hourly.min()).date() - warmup_start_date(hourly, 168)).days,
        )

    def test_warmup_start_precedes_the_first_bar(self):
        start = warmup_start_date(self.bars, self.ZWIN)
        self.assertLess(start, self.bars.min().date())

    def test_full_coverage_when_warmup_is_cached(self):
        np.random.seed(7)
        self._seed_cache(warmup_start_date(self.bars, self.ZWIN).isoformat(),
                         self.bars.max().date().isoformat())
        agent = PositioningSignalAgent("BTCUSDT", self.bars, zscore_window=self.ZWIN,
                                       cache_root=self.root)
        self.assertTrue(agent.available)
        self.assertEqual(agent.coverage, 1.0,
                         f"expected every bar usable, got {agent.status}")
        self.assertTrue(agent.reading_for(self.bars[0]).available,
                        "the FIRST bar must already carry a reading")

    def test_readings_are_indexed_to_the_requested_bars_only(self):
        np.random.seed(7)
        self._seed_cache(warmup_start_date(self.bars, self.ZWIN).isoformat(),
                         self.bars.max().date().isoformat())
        agent = PositioningSignalAgent("BTCUSDT", self.bars, zscore_window=self.ZWIN,
                                       cache_root=self.root)
        pd.testing.assert_index_equal(agent.aligned.index, self.bars)

    def test_extension_does_not_leak_the_future(self):
        """Extending the index BACKWARDS must not let later data reach a bar.

        Cache days beyond the window with a violent step change, then confirm
        every reading matches an agent whose cache stops at the window's end.
        """
        np.random.seed(11)
        warm = warmup_start_date(self.bars, self.ZWIN).isoformat()
        end = self.bars.max().date().isoformat()

        self._seed_cache(warm, end)
        truncated = PositioningSignalAgent(
            "BTCUSDT", self.bars, zscore_window=self.ZWIN, cache_root=self.root)

        # Same cache plus wildly different days AFTER the window.
        self._seed_cache("2024-01-12", "2024-01-20", crowd=40.0)
        with_future = PositioningSignalAgent(
            "BTCUSDT", self.bars, zscore_window=self.ZWIN, cache_root=self.root)

        for ts in self.bars:
            a = truncated.reading_for(ts)
            b = with_future.reading_for(ts)
            self.assertAlmostEqual(
                a.score, b.score, places=9,
                msg=f"future data changed the reading at {ts}: {a.score} vs {b.score}",
            )

    def test_missing_days_reports_the_warmup_gap(self):
        self._seed_cache(self.bars.min().date().isoformat(),
                         self.bars.max().date().isoformat())
        pending = missing_days("BTCUSDT", warmup_start_date(self.bars, self.ZWIN),
                               self.bars.max().date(), cache_root=self.root)
        self.assertTrue(pending, "the un-cached warm-up days must be reported")
        self.assertTrue(all(d < self.bars.min().date() for d in pending))


if __name__ == "__main__":
    unittest.main(verbosity=2)
