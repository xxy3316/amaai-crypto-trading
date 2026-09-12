"""Offline tests for the Hacker News text sentiment signal.

Runs with no network access. The documents below are hand-built TEST FIXTURES
that exercise the parser, the scorers and the windowing arithmetic; they are
unreachable from the backtest and are never presented as measurements.

The tests that matter for the paper are `TestNoLookahead`: a document must be
invisible to any bar at or before its publication, and truncating the future
must not change a past z-score.

    ./venv/Scripts/python.exe test/test_text_sentiment.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signals.text_sentiment import (  # noqa: E402
    CRYPTO_LEXICON,
    DEFAULT_QUERIES,
    EXAMPLES_PER_DIRECTION,
    POINT_THRESHOLDS,
    TextSentimentAgent,
    VaderScorer,
    Z_CLIP,
    _document_text,
    align_to_bars,
    clean_text,
    daterange,
    get_scorer,
    incomplete_days,
    load_cached_range,
    missing_days,
    score_documents,
    score_row,
    warmup_start_date,
)


# ── fixtures ────────────────────────────────────────────────────────────────

def hit(created: str, text: str, kind: str = "comment", obj: str = None) -> dict:
    ts = int(pd.Timestamp(created).tz_localize("UTC").timestamp())
    base = {"created_at_i": ts, "objectID": obj or f"{kind}-{ts}-{abs(hash(text)) % 9999}",
            "author": "tester"}
    if kind == "story":
        base["title"] = text
    else:
        base["comment_text"] = text
    return base


def write_day(root: Path, query: str, date: str, hits: list) -> Path:
    safe = query.strip().lower()
    directory = root / "hackernews" / safe
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{date}.json"
    path.write_text(json.dumps({"hits": hits, "nbHits": len(hits)}), encoding="utf-8")
    return path


def make_documents(n: int, start: str, sentiment_text: str = "bitcoin is fine",
                   step_minutes: int = 30) -> pd.DataFrame:
    t0 = pd.Timestamp(start)
    return pd.DataFrame({
        "created_at": [t0 + pd.Timedelta(minutes=step_minutes * i) for i in range(n)],
        "text": [sentiment_text] * n,
        "kind": ["comment"] * n,
        "author": ["tester"] * n,
        "object_id": [f"o{i}" for i in range(n)],
        "query": ["bitcoin"] * n,
    })


# ── text cleaning ───────────────────────────────────────────────────────────

class TestCleaning(unittest.TestCase):

    def test_strips_html_tags(self):
        self.assertEqual(clean_text("<p>hello <i>there</i></p>"), "hello there")

    def test_unescapes_entities(self):
        # HN really does return this; the sample during probing was "It&#x27;s"
        self.assertEqual(clean_text("It&#x27;s &amp; more"), "It's & more")

    def test_collapses_whitespace(self):
        self.assertEqual(clean_text("a\n\n  b\t c"), "a b c")

    def test_empty_and_none_are_safe(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")

    def test_document_text_prefers_title_then_comment(self):
        self.assertEqual(_document_text({"title": "T", "comment_text": "C"}), "T")
        self.assertEqual(_document_text({"comment_text": "C"}), "C")
        self.assertEqual(_document_text({"story_title": "S"}), "S")
        self.assertEqual(_document_text({}), "")


# ── scorers ─────────────────────────────────────────────────────────────────

class TestScorers(unittest.TestCase):

    def setUp(self):
        self.scorer = VaderScorer(use_crypto_lexicon=True)

    def test_bounded_output(self):
        for s in self.scorer.score(["great", "terrible", "", "a b c"]):
            self.assertGreaterEqual(s, -1.0)
            self.assertLessEqual(s, 1.0)

    def test_crypto_lexicon_changes_crypto_register(self):
        """VADER alone does not know 'rekt' or 'hodl'. The patch should."""
        plain = VaderScorer(use_crypto_lexicon=False)
        patched = VaderScorer(use_crypto_lexicon=True)
        self.assertEqual(plain.score(["rekt"])[0], 0.0)
        self.assertLess(patched.score(["rekt"])[0], 0.0)
        self.assertGreater(patched.score(["hodl"])[0], 0.0)

    def test_lexicon_signs_are_coherent(self):
        for term in ("scam", "ponzi", "crash", "rugpull"):
            self.assertLess(CRYPTO_LEXICON[term], 0, msg=term)
        for term in ("moon", "bullish", "rally", "adoption"):
            self.assertGreater(CRYPTO_LEXICON[term], 0, msg=term)

    def test_describe_is_serialisable(self):
        json.dumps(self.scorer.describe())

    def test_get_scorer_defaults_to_vader(self):
        self.assertEqual(get_scorer("vader").name, "vader")

    def test_cryptobert_falls_back_loudly_when_absent(self):
        """A missing model must degrade to VADER, never crash the backtest."""
        scorer = get_scorer("cryptobert")
        self.assertIn(scorer.name, ("cryptobert", "vader"))

    def test_score_documents_adds_column(self):
        docs = make_documents(4, "2024-01-02 00:00:00")
        scored = score_documents(docs, self.scorer)
        self.assertIn("sentiment", scored.columns)
        self.assertEqual(len(scored), 4)
        self.assertNotIn("sentiment", docs.columns, "input must not be mutated")

    def test_score_documents_handles_empty(self):
        self.assertTrue(score_documents(pd.DataFrame(), self.scorer).empty)


# ── point-in-time windowing ─────────────────────────────────────────────────

class TestNoLookahead(unittest.TestCase):

    def setUp(self):
        self.scorer = VaderScorer()
        self.bars = pd.date_range("2024-01-03 00:00:00", periods=24, freq="h")

    def _scored(self, docs):
        return score_documents(docs, self.scorer)

    def test_document_is_invisible_to_its_own_bar(self):
        """A post created at 12:00 must not reach the 12:00 bar."""
        docs = make_documents(20, "2024-01-02 00:00:00")
        marker = pd.DataFrame([{
            "created_at": pd.Timestamp("2024-01-03 12:00:00"),
            "text": "moon rally bullish adoption breakout",
            "kind": "comment", "author": "x", "object_id": "marker",
            "query": "bitcoin",
        }])
        scored = self._scored(pd.concat([docs, marker], ignore_index=True))
        aligned = align_to_bars(scored, self.bars, lag_bars=1, window_hours=24,
                                zscore_window=8)
        at = aligned.loc[pd.Timestamp("2024-01-03 12:00:00")]
        later = aligned.loc[pd.Timestamp("2024-01-03 14:00:00")]
        self.assertLess(at["doc_count"], later["doc_count"],
                        "the marker document must arrive only after its timestamp")

    def test_window_is_half_open_below_the_cutoff(self):
        """Every counted document must predate the bar's cutoff strictly."""
        docs = make_documents(48, "2024-01-02 00:00:00", step_minutes=30)
        scored = self._scored(docs)
        aligned = align_to_bars(scored, self.bars, lag_bars=1, window_hours=24,
                                zscore_window=8)
        delta = pd.Timedelta(hours=1)
        for ts in self.bars:
            cutoff = ts - delta
            n = aligned.loc[ts, "doc_count"]
            expected = int(((scored["created_at"] >= cutoff - pd.Timedelta(hours=24))
                            & (scored["created_at"] < cutoff)).sum())
            self.assertEqual(int(n), expected, msg=f"bar {ts}")

    def test_zero_lag_still_excludes_the_present(self):
        docs = make_documents(48, "2024-01-02 00:00:00")
        scored = self._scored(docs)
        aligned = align_to_bars(scored, self.bars, lag_bars=0, window_hours=24,
                                zscore_window=8)
        for ts in self.bars:
            counted = int(((scored["created_at"] >= ts - pd.Timedelta(hours=24))
                           & (scored["created_at"] < ts)).sum())
            self.assertEqual(int(aligned.loc[ts, "doc_count"]), counted)

    def test_zscores_are_causal(self):
        """Truncating the future must not change any past z-score."""
        rng = np.random.default_rng(3)
        texts = rng.choice(["bitcoin rally moon", "bitcoin crash scam",
                            "bitcoin news today"], size=200)
        docs = pd.DataFrame({
            "created_at": pd.date_range("2024-01-01 00:00:00", periods=200, freq="30min"),
            "text": texts, "kind": "comment", "author": "t",
            "object_id": [f"o{i}" for i in range(200)], "query": "bitcoin",
        })
        scored = self._scored(docs)
        full = align_to_bars(scored, self.bars, lag_bars=1, window_hours=6,
                             zscore_window=6)
        half_bars = self.bars[:14]
        truncated = scored[scored["created_at"] < half_bars[-1]]
        half = align_to_bars(truncated, half_bars, lag_bars=1, window_hours=6,
                             zscore_window=6)
        pd.testing.assert_series_equal(
            full["z_mean_sentiment"].loc[half.index], half["z_mean_sentiment"],
            check_names=False, rtol=1e-9, atol=1e-12,
        )

    def test_empty_corpus_is_safe(self):
        aligned = align_to_bars(pd.DataFrame(), self.bars, zscore_window=8)
        self.assertEqual(len(aligned), len(self.bars))
        self.assertTrue(aligned["doc_count"].isna().all()
                        or (aligned["doc_count"].fillna(0) == 0).all())

    def test_output_is_indexed_like_the_bars(self):
        scored = self._scored(make_documents(20, "2024-01-02 00:00:00"))
        aligned = align_to_bars(scored, self.bars, zscore_window=8)
        pd.testing.assert_index_equal(aligned.index, self.bars)

    def test_warmup_start_precedes_first_bar_and_scales(self):
        hourly = pd.date_range("2024-06-01", periods=100, freq="h")
        daily = pd.date_range("2024-06-01", periods=100, freq="D")
        self.assertLess(warmup_start_date(hourly, 168, 24), hourly.min().date())
        self.assertLess(warmup_start_date(daily, 168, 24),
                        warmup_start_date(hourly, 168, 24))


# ── scoring one bar ─────────────────────────────────────────────────────────

class TestScoreRow(unittest.TestCase):

    def _row(self, docs=50, mean=0.2, z=0.0):
        return pd.Series({"doc_count": docs, "mean_sentiment": mean,
                          "pos_share": 0.5, "neg_share": 0.2,
                          "z_mean_sentiment": z})

    def test_positive_z_reads_bullish(self):
        r = score_row(self._row(z=2.5), max_points=2)
        self.assertGreater(r.score, 0)
        self.assertGreater(r.bullish_points, 0)
        self.assertEqual(r.bearish_points, 0)

    def test_negative_z_reads_bearish(self):
        r = score_row(self._row(z=-2.5), max_points=2)
        self.assertLess(r.score, 0)
        self.assertGreater(r.bearish_points, 0)

    def test_below_document_floor_is_unavailable(self):
        """Two comments is noise, not sentiment."""
        r = score_row(self._row(docs=2, z=3.0), min_documents=5)
        self.assertFalse(r.available)
        self.assertEqual(r.bullish_points, 0)
        self.assertEqual(r.bearish_points, 0)
        self.assertIn("floor", r.reasoning)

    def test_missing_zscore_is_unavailable(self):
        r = score_row(self._row(z=float("nan")))
        self.assertFalse(r.available)
        self.assertEqual(r.score, 0.0)

    def test_none_row_is_unavailable(self):
        r = score_row(None)
        self.assertFalse(r.available)
        self.assertEqual(r.source, "none")

    def test_score_is_bounded(self):
        self.assertLessEqual(abs(score_row(self._row(z=99.0)).score), 1.0)

    def test_max_points_is_respected(self):
        r = score_row(self._row(z=3.0), max_points=1)
        self.assertLessEqual(max(r.bullish_points, r.bearish_points), 1)

    def test_confidence_rises_with_corpus_size(self):
        small = score_row(self._row(docs=5, z=2.0), min_documents=5)
        large = score_row(self._row(docs=100, z=2.0), min_documents=5)
        self.assertGreater(large.confidence, small.confidence)

    def test_reading_serialises(self):
        json.dumps(score_row(self._row(z=1.5)).to_dict())


# ── cache + agent ───────────────────────────────────────────────────────────

class TestPartialDayCapture(unittest.TestCase):
    """A day cached while it was still in progress must be re-fetched.

    The regression: `fetch_range` skipped any (query, day) that already had a
    file. A day pulled an hour after midnight UTC therefore froze at one hour's
    worth of posts -- in one observed case a single document for the whole day
    -- and was never refreshed. That does not fail loudly: the day either drops
    under the min_documents floor and vanishes, or squeaks over it and scores
    noise, which is worse.
    """

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _manifest(self, *rows):
        (self.root / "manifest.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    @staticmethod
    def _utc(*args):
        return dt.datetime(*args, tzinfo=dt.timezone.utc)

    def test_day_pulled_mid_day_is_incomplete(self):
        self._manifest({"query": "crypto", "date": "2026-09-10",
                        "pulled_at_utc": "2026-09-10T01:20:00+00:00"})
        stale = incomplete_days(cache_root=self.root, now=self._utc(2026, 9, 12))
        self.assertIn(("crypto", "2026-09-10"), stale)

    def test_day_pulled_after_it_closed_is_left_alone(self):
        self._manifest({"query": "crypto", "date": "2026-09-08",
                        "pulled_at_utc": "2026-09-09T04:56:00+00:00"})
        stale = incomplete_days(cache_root=self.root, now=self._utc(2026, 9, 12))
        self.assertEqual(stale, set())

    def test_a_later_complete_pull_supersedes_the_partial_one(self):
        # read_manifest keeps the newest pull per (query, date); the day is done.
        self._manifest(
            {"query": "crypto", "date": "2026-09-10",
             "pulled_at_utc": "2026-09-10T01:20:00+00:00"},
            {"query": "crypto", "date": "2026-09-10",
             "pulled_at_utc": "2026-09-11T06:00:00+00:00"},
        )
        stale = incomplete_days(cache_root=self.root, now=self._utc(2026, 9, 12))
        self.assertEqual(stale, set())

    def test_today_is_always_refreshable(self):
        self._manifest({"query": "crypto", "date": "2026-09-12",
                        "pulled_at_utc": "2026-09-12T04:18:00+00:00"})
        stale = incomplete_days(cache_root=self.root,
                                now=self._utc(2026, 9, 12, 12))
        self.assertIn(("crypto", "2026-09-12"), stale)

    def test_no_manifest_means_nothing_to_refresh(self):
        self.assertEqual(incomplete_days(cache_root=self.root), set())


class TestCacheAndAgent(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bars = pd.date_range("2024-01-03 00:00:00", periods=24, freq="h")

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_cache_returns_empty(self):
        self.assertTrue(load_cached_range("2024-01-01", "2024-01-03",
                                          cache_root=self.root).empty)

    def test_missing_days_lists_every_query_day_pair(self):
        pending = missing_days("2024-01-01", "2024-01-02",
                               queries=("bitcoin", "crypto"), cache_root=self.root)
        self.assertEqual(len(pending), 4)

    def test_load_deduplicates_across_queries(self):
        """The same HN post matches several queries; it must be counted once."""
        shared = hit("2024-01-02 10:00:00", "bitcoin and crypto news", obj="dup-1")
        write_day(self.root, "bitcoin", "2024-01-02", [shared])
        write_day(self.root, "crypto", "2024-01-02", [shared])
        docs = load_cached_range("2024-01-02", "2024-01-02",
                                 queries=("bitcoin", "crypto"), cache_root=self.root)
        self.assertEqual(len(docs), 1)

    def test_load_skips_documents_with_no_text(self):
        write_day(self.root, "bitcoin", "2024-01-02", [
            hit("2024-01-02 10:00:00", "real text", obj="a"),
            {"created_at_i": 1704189600, "objectID": "b"},  # no text at all
        ])
        docs = load_cached_range("2024-01-02", "2024-01-02",
                                 queries=("bitcoin",), cache_root=self.root)
        self.assertEqual(len(docs), 1)

    def test_load_timestamps_are_tz_naive_utc(self):
        write_day(self.root, "bitcoin", "2024-01-02",
                  [hit("2024-01-02 10:00:00", "text", obj="a")])
        docs = load_cached_range("2024-01-02", "2024-01-02",
                                 queries=("bitcoin",), cache_root=self.root)
        self.assertIsNone(docs["created_at"].dt.tz)
        self.assertEqual(docs["created_at"].iloc[0], pd.Timestamp("2024-01-02 10:00:00"))

    def test_corrupt_cache_file_is_skipped_not_fatal(self):
        directory = self.root / "hackernews" / "bitcoin"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "2024-01-02.json").write_text("{not json", encoding="utf-8")
        docs = load_cached_range("2024-01-02", "2024-01-02",
                                 queries=("bitcoin",), cache_root=self.root)
        self.assertTrue(docs.empty)

    def test_agent_with_empty_cache_never_fakes_a_zero(self):
        agent = TextSentimentAgent(self.bars, cache_root=self.root)
        self.assertFalse(agent.available)
        reading = agent.reading_for(self.bars[5])
        self.assertFalse(reading.available)
        self.assertEqual(reading.bullish_points, 0)
        self.assertEqual(reading.bearish_points, 0)

    def test_disabled_agent_is_inert(self):
        agent = TextSentimentAgent(self.bars, enabled=False, cache_root=self.root)
        self.assertEqual(agent.status, "disabled")
        self.assertFalse(agent.reading_for(self.bars[0]).available)

    def test_agent_serves_readings_from_a_seeded_cache(self):
        rng = np.random.default_rng(5)
        pool = ["bitcoin rally moon bullish", "bitcoin crash scam rekt",
                "bitcoin protocol discussion"]
        for day in daterange("2023-12-28", "2024-01-04"):
            hits = [
                hit(f"{day.isoformat()} {h:02d}:{m:02d}:00", str(rng.choice(pool)),
                    obj=f"{day}-{h}-{m}")
                for h in range(0, 24, 2) for m in (0, 30)
            ]
            write_day(self.root, "bitcoin", day.isoformat(), hits)

        agent = TextSentimentAgent(self.bars, queries=("bitcoin",), enabled=True,
                                   zscore_window=12, window_hours=24,
                                   min_documents=5, cache_root=self.root)
        self.assertTrue(agent.available, agent.status)

        # 8 days were seeded (192 documents) but only the warm-up window is
        # needed, so the agent must load a subset, not the whole cache. Loading
        # everything would make startup scale with the cache rather than with
        # the backtest window.
        self.assertGreater(agent.document_count, 50)
        self.assertLess(agent.document_count, 192,
                        "agent should load only the warm-up window, not the "
                        "entire cache")
        pd.testing.assert_index_equal(agent.aligned.index, self.bars)
        readings = [agent.reading_for(t) for t in self.bars]
        self.assertTrue(any(r.available for r in readings))
        for r in readings:
            self.assertLessEqual(abs(r.score), 1.0)

    def test_agent_summary_is_serialisable(self):
        agent = TextSentimentAgent(self.bars, cache_root=self.root)
        json.dumps(agent.summary(), default=str)

    def test_default_queries_exclude_btc(self):
        """'btc' matched 1185 documents in one day during probing: noise."""
        self.assertNotIn("btc", DEFAULT_QUERIES)

    # ── displayed example documents ─────────────────────────────────────────

    def _seeded_agent(self, **kwargs):
        rng = np.random.default_rng(11)
        pool = ["bitcoin rally moon bullish", "bitcoin crash scam rekt",
                "bitcoin protocol discussion"]
        for day in daterange("2023-12-28", "2024-01-04"):
            hits = [
                hit(f"{day.isoformat()} {h:02d}:{m:02d}:00", str(rng.choice(pool)),
                    obj=f"{day}-{h}-{m}")
                for h in range(0, 24, 2) for m in (0, 30)
            ]
            write_day(self.root, "bitcoin", day.isoformat(), hits)
        opts = dict(queries=("bitcoin",), enabled=True, zscore_window=12,
                    window_hours=24, min_documents=5, cache_root=self.root)
        opts.update(kwargs)
        return TextSentimentAgent(self.bars, **opts)

    def test_examples_never_include_a_document_at_or_after_the_cutoff(self):
        """The displayed evidence must obey the same point-in-time rule.

        Showing a post the bar could not have seen would be a lookahead leak in
        the UI even though it never touches the score.
        """
        agent = self._seeded_agent()
        delta = self.bars[1] - self.bars[0]
        for bar in self.bars:
            reading = agent.reading_for(bar)
            cutoff = bar - delta * agent.lag_bars
            start = cutoff - pd.Timedelta(hours=agent.window_hours)
            for doc in reading.examples:
                created = pd.Timestamp(doc["created_at"])
                self.assertLess(created, cutoff,
                                f"example from {created} leaked into bar {bar} "
                                f"(cutoff {cutoff})")
                self.assertGreaterEqual(created, start)

    def test_examples_are_present_and_span_both_directions(self):
        agent = self._seeded_agent()
        with_examples = [r for r in (agent.reading_for(b) for b in self.bars)
                         if r.available and r.examples]
        self.assertTrue(with_examples, "no reading carried example documents")
        sample = with_examples[0]
        self.assertLessEqual(len(sample.examples), 2 * EXAMPLES_PER_DIRECTION)
        scores = [d["sentiment"] for d in sample.examples]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         "examples should read most bullish first")
        self.assertGreater(max(scores), 0.0)
        self.assertLess(min(scores), 0.0)

    def test_examples_are_json_serialisable_for_the_decision_log(self):
        agent = self._seeded_agent()
        for bar in self.bars:
            json.dumps(agent.reading_for(bar).to_dict(), default=str)

    def test_unavailable_reading_carries_no_examples(self):
        """An unusable reading must not look evidenced."""
        agent = TextSentimentAgent(self.bars, cache_root=self.root)
        reading = agent.reading_for(self.bars[5])
        self.assertFalse(reading.available)
        self.assertEqual(reading.examples, [])

    def test_point_thresholds_match_score_row(self):
        """The UI renders POINT_THRESHOLDS; it must be what score_row applies."""
        for threshold, awarded in POINT_THRESHOLDS:
            z = (threshold + 1e-6) * Z_CLIP
            row = pd.Series({"doc_count": 50, "mean_sentiment": 0.2,
                             "pos_share": 0.5, "neg_share": 0.2,
                             "z_mean_sentiment": z})
            self.assertEqual(score_row(row, max_points=9).bullish_points, awarded)
            below = pd.Series(dict(row, z_mean_sentiment=(threshold - 1e-6) * Z_CLIP))
            self.assertLess(score_row(below, max_points=9).bullish_points, awarded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
