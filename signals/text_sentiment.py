"""Text sentiment from real public posts: free, keyless, byte-reproducible.

WHAT THIS IS
------------
The channel this replaces was decorative: the backtest asked for synthetic
tweets, Phase 2 correctly switched synthetic data off, and the sentiment score
was a hard 0.0 on every bar. This restores it with *actual* text.

Source: the Hacker News Search API, operated by Algolia.

    https://hn.algolia.com/api/v1/search_by_date
        ?query=bitcoin&tags=(story,comment)
        &numericFilters=created_at_i>...,created_at_i<...

Free, no API key, no account, and it indexes the COMPLETE HN archive back to
2007 with exact UTC timestamps, so a backtest window of any age can be served.

WHY NOT THE OBVIOUS SOURCES
---------------------------
Measured from this machine's network on 2026-09-09:

    Reddit (.json + search)   HTTP 403  blocked by the corporate proxy
    StockTwits                HTTP 403  blocked by the corporate proxy
    CryptoPanic               HTTP 403  blocked by the corporate proxy
    huggingface.co            HTTP 403  blocked ("Generative AI Tools"), which
                                        is why CryptoBERT cannot be fetched
    GDELT                     HTTP 429  rate-limited even after 20s idle,
                                        because the proxy's egress IP is shared
    CryptoCompare news        HTTP 401  now requires an API key
    X / Twitter               paid tiers cannot serve historical posts anyway

Hacker News returned HTTP 200 throughout. It is a smaller crowd than Reddit and
skews technical rather than retail-speculative -- that is a real limitation and
belongs in any write-up -- but it is genuine, timestamped, public text that a
reviewer can re-fetch.

VOLUME (measured, "bitcoin", 2024-01-01..2024-02-01)
    1,159 documents over 31 days, roughly 37/day, mean length 425 characters.
Adding the "crypto" query roughly doubles that. This is thin for hourly bars,
which is exactly why readings aggregate over a trailing window rather than
per-bar.

SCORING
-------
Two backends, selected by TEXT_SENTIMENT_SCORER:

  vader      (default) VADER compound score, extended with a small crypto
             lexicon. Installed already, no setup, runs in milliseconds.
  cryptobert `ElKulako/cryptobert`, a RoBERTa fine-tuned on crypto social
             posts. Needs `pip install transformers torch`. Falls back to
             VADER, loudly, if unavailable -- it never silently degrades.

             NOTE (measured 2026-09-11): huggingface.co is blocked on this
             network by Cato with "Corporate Internet policy violation",
             category "Generative AI Tools", and cdn-lfs.huggingface.co does
             not resolve, so the model CANNOT be downloaded here. transformers
             and torch install fine; only the weights are unreachable. To use
             CryptoBERT, obtain the model directory another way and set
             CRYPTOBERT_MODEL_PATH to it, or have huggingface.co allowlisted.

VADER is a 2014 general-purpose lexicon and mis-reads crypto register: "this is
going to zero", "rekt", "diamond hands", "rug pull" are invisible to it. The
crypto lexicon extension below is a documented, fixed patch for the worst of
that, and the CryptoBERT option is the proper fix. Reporting both is a free
ablation row.

POINT-IN-TIME DISCIPLINE
------------------------
A post is knowable the moment it is published, so the bar whose OPEN time is t
aggregates documents with created_at in

    [t - lag - window,  t - lag)

with lag = lag_bars * bar_duration. Nothing published at or after the cutoff
can reach the bar. Rolling z-scores use right-aligned `pandas.rolling`, which
is causal. Both properties are asserted in the test suite.

NO SYNTHETIC FALLBACK
---------------------
A window with no documents yields "no reading". This module never invents text
and never reports 0.0 as if it were a measurement.

CLI
---
    python -m signals.text_sentiment probe
    python -m signals.text_sentiment fetch  --start 2024-01-01 --end 2024-03-31
    python -m signals.text_sentiment report
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── CONSTANTS ───────────────────────────────────────────────────────────────

HN_ENDPOINT = "https://hn.algolia.com/api/v1/search_by_date"

#: Algolia caps a result set at 1000 records regardless of paging, so documents
#: are fetched ONE UTC DAY AT A TIME. At the measured ~37/day per query that is
#: far below the cap; a day that returns exactly 1000 is flagged as truncated
#: rather than silently losing posts.
HITS_PER_PAGE = 1000
ALGOLIA_RESULT_CAP = 1000

#: Default query terms. "btc" is deliberately excluded: it matched 1,185
#: documents in a single day during probing, i.e. it is matching substrings and
#: unrelated tokens rather than the asset.
DEFAULT_QUERIES = ("bitcoin", "crypto")

DEFAULT_CACHE_ROOT = Path(
    os.getenv("EXOGENOUS_CACHE_ROOT", "data/exogenous")
) / "text_sentiment"

MANIFEST_NAME = "manifest.jsonl"

USER_AGENT = os.getenv(
    "EXOGENOUS_USER_AGENT",
    "amaai-crypto-trading/1.0 (academic backtest; contact via repository)",
)

#: Crypto register VADER does not know. Values follow VADER's own scale, which
#: runs -4 (most negative) to +4 (most positive). This list is FIXED and was
#: not tuned against returns; it patches obvious vocabulary gaps only.
CRYPTO_LEXICON: Dict[str, float] = {
    # bearish
    "rekt": -3.0, "rugpull": -3.5, "rug": -2.0, "scam": -3.0, "ponzi": -3.5,
    "dump": -2.0, "dumping": -2.2, "capitulation": -2.5, "liquidated": -2.8,
    "liquidation": -2.3, "bearish": -2.5, "crash": -3.0, "collapse": -3.2,
    "bagholder": -2.5, "shitcoin": -2.5, "worthless": -3.0, "bubble": -1.8,
    "insolvent": -3.2, "hack": -2.8, "hacked": -3.0, "exploit": -2.5,
    # bullish
    "hodl": 1.5, "hodling": 1.5, "moon": 2.5, "mooning": 3.0, "bullish": 2.5,
    "pump": 1.8, "pumping": 2.0, "rally": 2.2, "breakout": 2.0, "ath": 2.5,
    "adoption": 1.8, "accumulate": 1.5, "accumulating": 1.5, "undervalued": 2.0,
    "halving": 1.2, "institutional": 1.0, "etf": 1.0,
    # ambiguous in general English, directional here
    "short": -1.0, "shorts": -1.0, "long": 1.0, "longs": 1.0,
}

Z_CLIP = 3.0

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# ── DATE HELPERS ────────────────────────────────────────────────────────────

def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.to_datetime(value).date()


def daterange(start, end) -> List[dt.date]:
    d0, d1 = _as_date(start), _as_date(end)
    if d1 < d0:
        raise ValueError(f"end {d1} precedes start {d0}")
    return [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]


def _day_bounds(date: dt.date) -> tuple:
    """UTC epoch seconds for [00:00:00, next 00:00:00) of `date`."""
    start = dt.datetime.combine(date, dt.time.min, tzinfo=dt.timezone.utc)
    return int(start.timestamp()), int((start + dt.timedelta(days=1)).timestamp())


# ── CACHE + MANIFEST ────────────────────────────────────────────────────────

def _cache_dir(query: str, cache_root: Optional[Path] = None) -> Path:
    root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(query).strip().lower()) or "query"
    return root / "hackernews" / safe


def _local_path(query: str, date: dt.date, cache_root: Optional[Path] = None) -> Path:
    return _cache_dir(query, cache_root) / f"{date.isoformat()}.json"


def _manifest_path(cache_root: Optional[Path] = None) -> Path:
    root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
    return root / MANIFEST_NAME


def _append_manifest(record: dict, cache_root: Optional[Path] = None) -> None:
    path = _manifest_path(cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def read_manifest(cache_root: Optional[Path] = None) -> pd.DataFrame:
    path = _manifest_path(cache_root)
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed manifest line in %s", path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if {"query", "date", "pulled_at_utc"}.issubset(df.columns):
        df = (df.sort_values("pulled_at_utc")
                .drop_duplicates(["query", "date"], keep="last"))
    return df.reset_index(drop=True)


# ── TEXT CLEANING ───────────────────────────────────────────────────────────

def clean_text(raw: Optional[str]) -> str:
    """HN comments arrive as HTML with entities; strip to plain text.

    Both scorers read words, so leaving `<p>` and `&#x27;` in place would add
    noise to VADER's tokeniser and waste CryptoBERT's context window.
    """
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", str(raw))
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _document_text(hit: dict) -> str:
    """Best available text for one HN hit: story title, or comment body."""
    for field in ("title", "comment_text", "story_title"):
        text = clean_text(hit.get(field))
        if text:
            return text
    return ""


# ── DOWNLOAD ────────────────────────────────────────────────────────────────

def _apply_corporate_ca(session: requests.Session) -> bool:
    """Retry through a merged certifi + Windows-trust-store bundle.

    Only called after an SSLError, so machines with no TLS-inspecting proxy are
    unaffected. Scoped to `session.verify`, never the global environment.
    """
    try:
        from signals.corporate_ca import ensure_bundle
    except ImportError:
        return False
    bundle = ensure_bundle()
    if not bundle:
        return False
    session.verify = str(bundle)
    logger.info("Retrying with corporate CA bundle at %s", bundle)
    return True


def fetch_day(
    query: str,
    date: dt.date,
    session: requests.Session,
    timeout: int = 45,
) -> Optional[dict]:
    """One UTC day of hits for one query. Returns the raw Algolia payload."""
    since, until = _day_bounds(date)
    params = {
        "query": query,
        "tags": "(story,comment)",
        "numericFilters": f"created_at_i>={since},created_at_i<{until}",
        "hitsPerPage": HITS_PER_PAGE,
    }
    response = session.get(HN_ENDPOINT, params=params, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} for {query} {date}")
    return response.json()


#: How long after a day closes its corpus is considered final. Hacker News
#: keeps accumulating posts until 24:00 UTC, and Algolia needs a moment to
#: index the last of them, so anything pulled before this captured only part of
#: the day.
DAY_SETTLES_AFTER = dt.timedelta(hours=1)


def incomplete_days(cache_root: Optional[Path] = None,
                    now: Optional[dt.datetime] = None) -> set:
    """(query, date) pairs whose cached file was captured before the day ended.

    `fetch_range` skips any pair that already has a file on disk. That is right
    for a day that is over and wrong for one that was still in progress: the
    partial capture is then kept forever, freezing that day at however many
    posts happened to exist at the moment of the pull -- which, for a day
    fetched an hour after midnight, can be a single document. A day frozen at
    one document does not fail loudly; it quietly drops under the min_documents
    floor, or worse, squeaks over it and scores noise.

    Pairs returned here are re-fetched. Today is always among them, so a run
    also refreshes the current day rather than inheriting an early snapshot.
    """
    manifest = read_manifest(cache_root)
    if manifest.empty:
        return set()
    if not {"query", "date", "pulled_at_utc"}.issubset(manifest.columns):
        return set()

    now = pd.Timestamp(now or dt.datetime.now(dt.timezone.utc))
    now = now.tz_localize("UTC") if now.tz is None else now.tz_convert("UTC")

    dates = manifest["date"].astype(str)
    day_end = pd.to_datetime(dates, utc=True, errors="coerce") + pd.Timedelta(days=1)
    pulled = pd.to_datetime(manifest["pulled_at_utc"], utc=True, errors="coerce")
    settled_at = day_end + pd.Timedelta(DAY_SETTLES_AFTER)

    # read_manifest keeps only the newest pull per (query, date), so this is the
    # state of the file currently on disk, not of some superseded attempt.
    captured_mid_day = (pulled < settled_at) & pulled.notna() & day_end.notna()

    return {
        (str(q), str(d))
        for q, d in zip(manifest["query"][captured_mid_day],
                        dates[captured_mid_day])
    }


def fetch_range(
    start,
    end,
    queries: Iterable[str] = DEFAULT_QUERIES,
    cache_root: Optional[Path] = None,
    force: bool = False,
    pause_s: float = 0.25,
    session: Optional[requests.Session] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Cache every (query, day) in [start, end]. Idempotent.

    A failed day is recorded and skipped rather than aborting the range: a
    partial corpus with a known gap is more useful than no corpus, provided the
    gap is visible, which is what the manifest is for.

    A day cached while it was still in progress is re-fetched rather than
    treated as done -- see `incomplete_days`.
    """
    queries = list(queries)
    dates = daterange(start, end)
    owns_session = session is None
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)

    summary: Dict[str, Any] = {
        "queries": queries, "requested": len(queries) * len(dates),
        "downloaded": 0, "cached": 0, "refreshed": 0, "failed": [],
        "documents": 0, "truncated_days": [],
    }
    # Computed once: the manifest does not change under us mid-run.
    stale = set() if force else incomplete_days(cache_root)
    ca_healed = False
    total = len(queries) * len(dates)
    position = 0

    try:
        for query in queries:
            for date in dates:
                position += 1
                if progress is not None:
                    try:
                        progress(position, total, f"{query} {date}")
                    except Exception:
                        pass

                target = _local_path(query, date, cache_root)
                captured_mid_day = (query, date.isoformat()) in stale
                if target.exists() and not force and not captured_mid_day:
                    summary["cached"] += 1
                    continue
                if target.exists():
                    summary["refreshed"] += 1

                pulled_at = dt.datetime.now(dt.timezone.utc).isoformat()
                try:
                    payload = fetch_day(query, date, session)
                except requests.exceptions.SSLError as e:
                    if not ca_healed and _apply_corporate_ca(session):
                        ca_healed = True
                        summary["ca_bundle_applied"] = True
                        try:
                            payload = fetch_day(query, date, session)
                        except Exception as retry_error:
                            summary["failed"].append(
                                {"query": query, "date": date.isoformat(),
                                 "error": str(retry_error)})
                            continue
                    else:
                        summary["failed"].append(
                            {"query": query, "date": date.isoformat(),
                             "error": f"SSLError: {e}"})
                        continue
                except Exception as e:
                    logger.warning("Fetch failed for %s %s: %s", query, date, e)
                    summary["failed"].append(
                        {"query": query, "date": date.isoformat(), "error": str(e)})
                    _append_manifest({
                        "query": query, "date": date.isoformat(),
                        "pulled_at_utc": pulled_at, "error": f"{type(e).__name__}: {e}",
                    }, cache_root)
                    continue

                hits = payload.get("hits", []) or []
                nb_hits = payload.get("nbHits")
                blob = json.dumps(payload, sort_keys=True).encode("utf-8")

                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob)

                truncated = len(hits) >= ALGOLIA_RESULT_CAP
                if truncated:
                    summary["truncated_days"].append(f"{query} {date.isoformat()}")
                    logger.warning(
                        "Algolia returned the %d-record cap for %s on %s; that "
                        "day's corpus is incomplete.", ALGOLIA_RESULT_CAP, query, date)

                summary["downloaded"] += 1
                summary["documents"] += len(hits)
                _append_manifest({
                    "query": query,
                    "date": date.isoformat(),
                    "endpoint": HN_ENDPOINT,
                    "pulled_at_utc": pulled_at,
                    "hits": len(hits),
                    "nb_hits_reported": nb_hits,
                    "truncated": truncated,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "bytes": len(blob),
                    "local_path": str(target),
                }, cache_root)

                if pause_s:
                    time.sleep(pause_s)
    finally:
        if owns_session:
            session.close()

    return summary


def missing_days(
    start,
    end,
    queries: Iterable[str] = DEFAULT_QUERIES,
    cache_root: Optional[Path] = None,
) -> List[tuple]:
    """(query, date) pairs in [start, end] that are not cached."""
    out = []
    for query in queries:
        for date in daterange(start, end):
            if not _local_path(query, date, cache_root).exists():
                out.append((query, date))
    return out


def ensure_cached(
    start,
    end,
    queries: Iterable[str] = DEFAULT_QUERIES,
    cache_root: Optional[Path] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Download only what is absent. Lets the app run with no manual step."""
    queries = list(queries)
    pending = missing_days(start, end, queries, cache_root)
    if not pending:
        return {"already_complete": True, "downloaded": 0, "failed": [],
                "queries": queries, "documents": 0}

    summary = fetch_range(start, end, queries=queries, cache_root=cache_root,
                          progress=progress)
    summary["already_complete"] = False
    return summary


# ── LOAD ────────────────────────────────────────────────────────────────────

def load_cached_range(
    start,
    end,
    queries: Iterable[str] = DEFAULT_QUERIES,
    cache_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Every cached document in [start, end], deduplicated. Never hits network.

    Columns: created_at (tz-naive UTC), text, kind, author, object_id, query.
    """
    rows = []
    missing = 0
    for query in queries:
        for date in daterange(start, end):
            path = _local_path(query, date, cache_root)
            if not path.exists():
                missing += 1
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Unreadable cache file %s: %s", path, e)
                missing += 1
                continue
            for hit in payload.get("hits", []) or []:
                created = hit.get("created_at_i")
                text = _document_text(hit)
                if created is None or not text:
                    continue
                rows.append({
                    "created_at": created,
                    "text": text,
                    "kind": "story" if hit.get("title") else "comment",
                    "author": hit.get("author"),
                    "object_id": hit.get("objectID"),
                    "query": query,
                })

    if missing:
        logger.warning(
            "Text sentiment cache is missing %d (query, day) file(s). Run "
            "`python -m signals.text_sentiment fetch` to fill them.", missing)

    if not rows:
        return pd.DataFrame(
            columns=["created_at", "text", "kind", "author", "object_id", "query"])

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], unit="s", utc=True)
    df["created_at"] = df["created_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    # The same document matches several queries, so dedupe on the HN id.
    df = df.drop_duplicates("object_id", keep="first")
    return df.sort_values("created_at").reset_index(drop=True)


# ── SCORING BACKENDS ────────────────────────────────────────────────────────

class VaderScorer:
    """VADER compound score, extended with the fixed crypto lexicon above."""

    name = "vader"

    def __init__(self, use_crypto_lexicon: bool = True):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()
        self.crypto_lexicon = bool(use_crypto_lexicon)
        if self.crypto_lexicon:
            self.analyzer.lexicon.update(CRYPTO_LEXICON)

    def score(self, texts: List[str]) -> List[float]:
        return [self.analyzer.polarity_scores(t)["compound"] for t in texts]

    def describe(self) -> dict:
        return {"scorer": self.name, "crypto_lexicon": self.crypto_lexicon,
                "lexicon_terms": len(CRYPTO_LEXICON) if self.crypto_lexicon else 0}


class CryptoBertScorer:
    """`ElKulako/cryptobert`: RoBERTa fine-tuned on crypto social posts.

    Requires `pip install transformers torch`. Construction raises if either is
    missing, and the caller falls back to VADER with a warning -- the failure is
    never silent, because a run scored by a different model is a different
    experiment.
    """

    name = "cryptobert"
    MODEL_ID = "ElKulako/cryptobert"
    #: The model emits Bearish / Neutral / Bullish, but checkpoints vary in
    #: whether labels are words or LABEL_n, so both spellings are mapped.
    LABEL_SIGN = {
        "bearish": -1.0, "neutral": 0.0, "bullish": 1.0,
        "label_0": -1.0, "label_1": 0.0, "label_2": 1.0,
        "negative": -1.0, "positive": 1.0,
    }

    def __init__(self, batch_size: int = 32, max_length: int = 256,
                 model_path: Optional[str] = None):
        from transformers import pipeline  # noqa: F401  (raises if absent)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        # CRYPTOBERT_MODEL_PATH points at a local copy of the model directory.
        # It exists because some networks block huggingface.co outright (this
        # one returns a Cato "Corporate Internet policy violation" 403, and
        # cdn-lfs.huggingface.co does not resolve), so the only way to use
        # CryptoBERT here is to bring the files in by hand. A local path also
        # makes a run reproducible without depending on the Hub staying up.
        self.model_ref = str(
            model_path or os.getenv("CRYPTOBERT_MODEL_PATH", "") or self.MODEL_ID
        )

        def build():
            return pipeline(
                "text-classification", model=self.model_ref,
                tokenizer=self.model_ref, truncation=True,
                max_length=self.max_length, top_k=None,
            )

        # A local directory needs no network at all. Anything else is a Hub id,
        # so check the Hub is actually reachable BEFORE handing the id to
        # transformers.
        #
        # The check is not defensive padding: HF_HUB_OFFLINE is ignored by
        # transformers 5.17 / huggingface_hub 1.31 (measured -- it still issued
        # 12 HEAD requests), and every one of them retries, so a blocked network
        # costs ~47s before failing. The probe below costs 0.3s. Without it,
        # merely setting TEXT_SENTIMENT_SCORER=cryptobert stalls app startup for
        # over a minute and then falls back to VADER anyway.
        self.offline = Path(self.model_ref).expanduser().is_dir()
        if not self.offline:
            self._require_hub_reachable()
        self.pipe = build()

    @staticmethod
    def _require_hub_reachable(timeout: float = 5.0) -> None:
        """Fail fast, and distinguish a TLS problem from a policy block."""
        session = requests.Session()
        url = "https://huggingface.co"
        try:
            response = session.head(url, timeout=timeout, allow_redirects=True)
        except requests.exceptions.SSLError:
            # Same corporate-TLS interception the data fetchers handle. Retry
            # once through the merged bundle so the real status code is seen
            # rather than reporting a certificate error as unreachable.
            if not _apply_corporate_ca(session):
                raise RuntimeError(
                    "huggingface.co failed TLS verification and no corporate CA "
                    "bundle could be built"
                )
            response = session.head(url, timeout=timeout, allow_redirects=True)
        except Exception as e:
            raise RuntimeError(
                f"huggingface.co unreachable ({type(e).__name__}: {e}), and the "
                f"model is not available as a local directory"
            ) from e

        if response.status_code >= 400:
            raise RuntimeError(
                f"huggingface.co returned HTTP {response.status_code}; on this "
                f"network that is a corporate policy block (Cato, category "
                f"'Generative AI Tools'), so the weights cannot be downloaded. "
                f"Set CRYPTOBERT_MODEL_PATH to a local copy of the model "
                f"directory instead."
            )

    def _row_score(self, output) -> float:
        """Expected value of the sign under the predicted distribution."""
        if isinstance(output, dict):
            output = [output]
        total = 0.0
        for entry in output:
            label = str(entry.get("label", "")).strip().lower()
            sign = self.LABEL_SIGN.get(label)
            if sign is None:
                continue
            total += sign * float(entry.get("score", 0.0))
        return max(-1.0, min(1.0, total))

    def score(self, texts: List[str]) -> List[float]:
        out: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t[: self.max_length * 6] for t in texts[i: i + self.batch_size]]
            out.extend(self._row_score(r) for r in self.pipe(chunk))
        return out

    def describe(self) -> dict:
        return {"scorer": self.name, "model": self.model_ref,
                "max_length": self.max_length}


def get_scorer(name: str = "vader", use_crypto_lexicon: bool = True):
    """Build the requested scorer, falling back to VADER loudly on failure."""
    requested = str(name or "vader").strip().lower()
    if requested in ("cryptobert", "crypto-bert", "bert"):
        try:
            scorer = CryptoBertScorer()
            logger.info("Text sentiment scorer: CryptoBERT (%s)",
                        CryptoBertScorer.MODEL_ID)
            return scorer
        except Exception as e:
            logger.warning(
                "CryptoBERT unavailable (%s: %s); falling back to VADER. Needs "
                "`pip install transformers torch`; if those are already present "
                "the weights themselves are unreachable -- huggingface.co is "
                "blocked on this network, so point CRYPTOBERT_MODEL_PATH at a "
                "local copy of the model directory. NOTE: results scored by a "
                "different model are not comparable.",
                type(e).__name__, e)
    return VaderScorer(use_crypto_lexicon=use_crypto_lexicon)


def score_documents(documents: pd.DataFrame, scorer) -> pd.DataFrame:
    """Add a `sentiment` column in [-1, 1]. Returns a copy."""
    if documents is None or documents.empty:
        out = documents.copy() if documents is not None else pd.DataFrame()
        if out is not None and not out.empty:
            out["sentiment"] = []
        return out
    out = documents.copy()
    out["sentiment"] = scorer.score(out["text"].astype(str).tolist())
    return out


# ── POINT-IN-TIME ALIGNMENT ─────────────────────────────────────────────────

def _infer_bar_delta(bar_index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(bar_index) < 2:
        return pd.Timedelta(hours=1)
    deltas = pd.Series(bar_index[1:]) - pd.Series(bar_index[:-1])
    median = deltas.median()
    if pd.isna(median) or median <= pd.Timedelta(0):
        return pd.Timedelta(hours=1)
    return median


def align_to_bars(
    scored: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    lag_bars: int = 1,
    window_hours: int = 24,
    zscore_window: int = 168,
) -> pd.DataFrame:
    """Aggregate documents onto bars, strictly point-in-time.

    The bar whose OPEN time is t averages documents created in

        [t - lag - window,  t - lag)

    so nothing published at or after the cutoff can reach it. HN yields only
    tens of documents a day, so a trailing window (default 24h) is what makes
    the reading stable enough to be worth anything at hourly resolution.

    Returns one row per bar: doc_count, mean sentiment, its causal z-score, and
    the share of positive/negative documents.
    """
    bar_index = pd.DatetimeIndex(bar_index)
    if bar_index.tz is not None:
        bar_index = bar_index.tz_convert("UTC").tz_localize(None)

    out = pd.DataFrame(index=bar_index)
    if scored is None or scored.empty or len(bar_index) == 0:
        for col in ("doc_count", "mean_sentiment", "pos_share", "neg_share",
                    "z_mean_sentiment"):
            out[col] = pd.Series(dtype="float64", index=bar_index)
        return out

    delta = _infer_bar_delta(bar_index)
    cutoffs = bar_index - delta * int(lag_bars)
    window = pd.Timedelta(hours=int(window_hours))

    docs = scored.sort_values("created_at")
    times = docs["created_at"].to_numpy()
    values = docs["sentiment"].to_numpy(dtype="float64")

    # searchsorted over a sorted document list: O(bars log docs), and the
    # half-open interval is what enforces the point-in-time rule.
    starts = pd.DatetimeIndex(cutoffs - window).to_numpy()
    ends = pd.DatetimeIndex(cutoffs).to_numpy()
    lo = times.searchsorted(starts, side="left")
    hi = times.searchsorted(ends, side="left")

    counts, means, pos, neg = [], [], [], []
    for i, j in zip(lo, hi):
        window_vals = values[i:j]
        n = len(window_vals)
        counts.append(n)
        if n == 0:
            means.append(float("nan"))
            pos.append(float("nan"))
            neg.append(float("nan"))
            continue
        means.append(float(window_vals.mean()))
        pos.append(float((window_vals > 0.05).mean()))
        neg.append(float((window_vals < -0.05).mean()))

    out["doc_count"] = counts
    out["mean_sentiment"] = means
    out["pos_share"] = pos
    out["neg_share"] = neg

    # Causal z-score: "unusually bullish versus its own recent baseline",
    # which removes the persistent positive drift of general news text and
    # makes this comparable with the positioning score.
    win = max(2, int(zscore_window))
    min_periods = min(win, max(8, win // 4))
    series = out["mean_sentiment"]
    mean = series.rolling(win, min_periods=min_periods).mean()
    std = series.rolling(win, min_periods=min_periods).std(ddof=0)
    z = (series - mean) / std.replace(0.0, pd.NA)
    out["z_mean_sentiment"] = pd.to_numeric(z, errors="coerce").clip(-Z_CLIP, Z_CLIP)

    return out


def warmup_start_date(bar_index: pd.DatetimeIndex, zscore_window: int = 168,
                      window_hours: int = 24) -> dt.date:
    """First UTC date whose documents are needed to serve `bar_index` in full.

    Covers the z-score warm-up, the aggregation window and the lag, so the
    requested period is usable from its FIRST bar instead of losing its first
    `zscore_window` bars for a purely mechanical reason.
    """
    index = pd.DatetimeIndex(bar_index)
    delta = _infer_bar_delta(index)
    span = delta * (int(zscore_window) + 1) + pd.Timedelta(hours=int(window_hours))
    return (index.min() - span - pd.Timedelta(days=1)).date()


# ── READING + AGENT ─────────────────────────────────────────────────────────

#: |score| -> points ladder used by `score_row`, most demanding first.
#: Exposed so the UI can explain a reading without restating the thresholds
#: and silently drifting out of sync with the logic that awards the points.
POINT_THRESHOLDS: tuple = ((0.66, 2), (0.33, 1))

#: How many example documents per direction to attach to a reading. Kept small:
#: these travel in the decision log, and their job is face validity ("is the
#: model reading crypto discussion or unrelated noise?"), not completeness.
EXAMPLES_PER_DIRECTION = 2


@dataclasses.dataclass
class TextSentimentReading:
    """One bar's reading, in the shape the decision agent already consumes."""

    score: float = 0.0
    confidence: float = 0.0
    bullish_points: int = 0
    bearish_points: int = 0
    reasoning: str = "No text sentiment data."
    source: str = "none"
    available: bool = False
    doc_count: int = 0
    mean_sentiment: Optional[float] = None
    features: Dict[str, float] = dataclasses.field(default_factory=dict)
    #: Sample documents from THIS bar's window, for display only. They never
    #: touch the score. See `TextSentimentAgent._examples_between`, which reuses
    #: the same half-open interval as the aggregation, so nothing shown here was
    #: published at or after the bar's cutoff.
    examples: List[dict] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["features"] = {k: (None if v is None or pd.isna(v) else float(v))
                         for k, v in self.features.items()}
        return d


def score_row(row: pd.Series, max_points: int = 2,
              min_documents: int = 5) -> TextSentimentReading:
    """Turn one aligned bar into a bounded score, points and a rationale.

    `min_documents` is a data-quality floor: HN can be quiet for hours, and a
    window holding two comments is noise, not sentiment. Below the floor the
    reading is reported as unavailable rather than acted on.
    """
    if row is None:
        return TextSentimentReading()

    doc_count = int(row.get("doc_count") or 0)
    mean_sentiment = row.get("mean_sentiment")
    z = row.get("z_mean_sentiment")

    features = {}
    for key in ("doc_count", "mean_sentiment", "pos_share", "neg_share",
                "z_mean_sentiment"):
        value = row.get(key)
        if value is not None and not pd.isna(value):
            features[key] = float(value)

    if doc_count < int(min_documents):
        return TextSentimentReading(
            reasoning=(f"Only {doc_count} document(s) in the window; below the "
                       f"{min_documents}-document floor, so no reading."),
            source="hackernews", available=False, doc_count=doc_count,
            mean_sentiment=(None if mean_sentiment is None or pd.isna(mean_sentiment)
                            else float(mean_sentiment)),
            features=features,
        )

    if z is None or pd.isna(z):
        return TextSentimentReading(
            reasoning="Text sentiment present but its z-score is still warming up.",
            source="hackernews", available=False, doc_count=doc_count,
            mean_sentiment=(None if mean_sentiment is None or pd.isna(mean_sentiment)
                            else float(mean_sentiment)),
            features=features,
        )

    z = float(z)
    score = max(-1.0, min(1.0, z / Z_CLIP))

    magnitude = abs(score)
    points = 0
    for threshold, awarded in POINT_THRESHOLDS:
        if magnitude >= threshold:
            points = awarded
            break
    points = min(points, int(max_points))
    bullish = points if score > 0 else 0
    bearish = points if score < 0 else 0

    # Confidence is a DATA-QUALITY statement, not a return forecast: it rises
    # with corpus size and saturates at four times the floor.
    confidence = round(min(1.0, doc_count / float(max(1, min_documents) * 4)), 3)

    direction = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")
    reasoning = (
        f"Hacker News sentiment {direction}: {doc_count} documents, mean "
        f"{float(mean_sentiment):+.3f}, z={z:+.2f} versus its own trailing "
        f"baseline. Rule contribution: +{bullish} bullish / +{bearish} bearish."
    )

    return TextSentimentReading(
        score=round(score, 4), confidence=confidence,
        bullish_points=bullish, bearish_points=bearish,
        reasoning=reasoning, source="hackernews", available=True,
        doc_count=doc_count, mean_sentiment=float(mean_sentiment),
        features=features,
    )


class TextSentimentAgent:
    """Serves a point-in-time text sentiment reading for any bar.

    Documents are loaded, scored and aligned ONCE at construction, so the
    per-bar lookup inside the simulation loop is a dictionary hit. Scoring is
    the expensive part (especially under CryptoBERT), and doing it per bar would
    re-score the same documents once per bar of the trailing window.
    """

    def __init__(
        self,
        bar_index: pd.DatetimeIndex,
        queries: Iterable[str] = DEFAULT_QUERIES,
        enabled: bool = True,
        scorer_name: str = "vader",
        lag_bars: int = 1,
        window_hours: int = 24,
        zscore_window: int = 168,
        max_points: int = 2,
        min_documents: int = 5,
        use_crypto_lexicon: bool = True,
        cache_root: Optional[Path] = None,
    ):
        self.queries = list(queries)
        self.enabled = bool(enabled)
        self.lag_bars = int(lag_bars)
        self.window_hours = int(window_hours)
        self.max_points = int(max_points)
        self.min_documents = int(min_documents)
        self.available = False
        self.status = "disabled" if not enabled else "not initialised"
        self.aligned = pd.DataFrame()
        self.coverage = 0.0
        self.document_count = 0
        self.scorer_info: Dict[str, Any] = {}
        # Display-only: the scored corpus, retained so a reading can show which
        # documents were in its window. Never consulted when scoring.
        self._docs = pd.DataFrame()
        self._bar_delta = None

        if not self.enabled or len(bar_index) == 0:
            return

        bar_index = pd.DatetimeIndex(bar_index)
        documents = load_cached_range(
            warmup_start_date(bar_index, zscore_window, window_hours),
            bar_index.max().date(),
            queries=self.queries,
            cache_root=cache_root,
        )

        if documents.empty:
            self.status = (
                "no cached text sentiment documents; run "
                f"`python -m signals.text_sentiment fetch --start "
                f"{bar_index.min().date()} --end {bar_index.max().date()}`"
            )
            logger.warning(self.status)
            return

        self.document_count = len(documents)
        scorer = get_scorer(scorer_name, use_crypto_lexicon=use_crypto_lexicon)
        self.scorer_info = scorer.describe()
        scored = score_documents(documents, scorer)

        # Extend the bar index backwards so the z-score warms up on prior bars
        # rather than eating the first `zscore_window` bars of the backtest.
        delta = _infer_bar_delta(bar_index)
        warm = pd.DatetimeIndex([
            bar_index.min() - delta * i for i in range(int(zscore_window) + 1, 0, -1)
        ])
        extended = warm.append(bar_index).drop_duplicates().sort_values()

        self.aligned = align_to_bars(
            scored, extended, lag_bars=self.lag_bars,
            window_hours=self.window_hours, zscore_window=zscore_window,
        ).reindex(bar_index)

        self._docs = scored.sort_values("created_at").reset_index(drop=True)
        self._bar_delta = delta

        usable = (
            self.aligned["z_mean_sentiment"].notna()
            & (self.aligned["doc_count"] >= self.min_documents)
        )
        self.coverage = float(usable.mean()) if len(self.aligned) else 0.0
        self.available = self.coverage > 0.0
        self.status = (
            f"{self.document_count} documents; usable on {self.coverage:.1%} of "
            f"bars ({self.scorer_info.get('scorer')}, {self.window_hours}h window, "
            f"lag {self.lag_bars} bar(s))"
        )
        logger.info("Text sentiment agent ready: %s", self.status)

    def _examples_between(self, start, end,
                          per_direction: int = EXAMPLES_PER_DIRECTION) -> List[dict]:
        """Strongest bullish and bearish documents in the half-open [start, end).

        Deliberately reuses the SAME interval as `align_to_bars`, so a document
        shown next to a bar is by construction one the bar was allowed to see.
        Building this from the timestamp alone would risk the displayed
        evidence and the scored evidence drifting apart.
        """
        if self._docs is None or self._docs.empty:
            return []
        times = self._docs["created_at"].to_numpy()
        lo = times.searchsorted(pd.Timestamp(start).to_datetime64(), side="left")
        hi = times.searchsorted(pd.Timestamp(end).to_datetime64(), side="left")
        window = self._docs.iloc[lo:hi]
        if window.empty:
            return []

        ordered = window.sort_values("sentiment")
        picks = pd.concat([ordered.head(per_direction),
                           ordered.tail(per_direction)]).drop_duplicates("object_id")

        out = []
        for _, row in ordered.loc[picks.index].sort_values(
                "sentiment", ascending=False).iterrows():
            text = str(row.get("text", "")).replace("\n", " ").strip()
            out.append({
                "text": text[:200],
                "sentiment": round(float(row.get("sentiment", 0.0)), 4),
                "kind": row.get("kind", ""),
                "created_at": pd.Timestamp(row["created_at"]).isoformat(),
                # VADER's compound score saturates with length, so the most
                # extreme documents tend to be the longest rather than the most
                # opinionated. Surfacing the length makes that visible instead
                # of letting it silently shape the tails.
                "chars": len(text),
            })
        return out

    def reading_for(self, timestamp) -> TextSentimentReading:
        if not self.enabled:
            return TextSentimentReading(reasoning="Text sentiment disabled.")
        if self.aligned.empty:
            return TextSentimentReading(
                reasoning=f"Text sentiment unavailable: {self.status}")
        try:
            ts = pd.to_datetime(timestamp)
            if ts.tz is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            if ts not in self.aligned.index:
                return TextSentimentReading(
                    reasoning=f"No text sentiment row aligned to bar {ts}.")
            reading = score_row(self.aligned.loc[ts], max_points=self.max_points,
                                min_documents=self.min_documents)
            if reading.available and self._bar_delta is not None:
                cutoff = ts - self._bar_delta * int(self.lag_bars)
                reading.examples = self._examples_between(
                    cutoff - pd.Timedelta(hours=int(self.window_hours)), cutoff)
            return reading
        except Exception as e:
            logger.warning("Text sentiment lookup failed at %s: %s", timestamp, e)
            return TextSentimentReading(reasoning=f"Lookup error: {e}")

    def summary(self) -> dict:
        base = {
            "enabled": self.enabled, "available": self.available,
            "status": self.status, "queries": self.queries,
            "source": "hackernews (Algolia HN Search API)",
            "scorer": self.scorer_info,
        }
        if self.aligned.empty:
            return base
        base.update({
            "bars": int(len(self.aligned)),
            "coverage_pct": round(self.coverage * 100, 2),
            "documents": self.document_count,
            "window_hours": self.window_hours,
            "lag_bars": self.lag_bars,
            "min_documents": self.min_documents,
            "median_docs_per_bar": (
                None if self.aligned["doc_count"].dropna().empty
                else float(self.aligned["doc_count"].median())
            ),
        })
        return base


# ── CLI ─────────────────────────────────────────────────────────────────────

def _split_queries(value: Optional[str]) -> List[str]:
    if not value:
        return list(DEFAULT_QUERIES)
    return [q.strip() for q in value.split(",") if q.strip()]


def _cmd_probe(args: argparse.Namespace) -> int:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    date = _as_date(args.date) if args.date else (dt.date.today() - dt.timedelta(days=7))
    print(f"GET {HN_ENDPOINT}  query={args.query!r}  date={date}")
    try:
        payload = fetch_day(args.query, date, session)
    except requests.exceptions.SSLError:
        if not _apply_corporate_ca(session):
            print("TLS verification failed and no CA bundle could be built.",
                  file=sys.stderr)
            return 1
        payload = fetch_day(args.query, date, session)
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    hits = payload.get("hits", []) or []
    print(f"nbHits reported   : {payload.get('nbHits')}")
    print(f"hits returned     : {len(hits)}")
    texts = [_document_text(h) for h in hits]
    texts = [t for t in texts if t]
    print(f"non-empty texts   : {len(texts)}")
    if texts:
        print(f"mean length       : {sum(map(len, texts)) / len(texts):.0f} chars")
        print(f"sample            : {texts[0][:100]}")
        scorer = get_scorer(args.scorer)
        scores = scorer.score(texts[:20])
        print(f"scorer            : {scorer.describe()}")
        print(f"sample scores     : {[round(s, 3) for s in scores[:5]]}")
        print(f"mean of 20        : {sum(scores) / len(scores):+.4f}")
    session.close()
    print("\nReachable and parseable. Safe to fetch a range.")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    queries = _split_queries(args.queries)
    total = {"n": 0}

    def report(done, count, label):
        if done == 1 or done % 25 == 0 or done == count:
            print(f"  {done}/{count}  {label}")
        total["n"] = count

    summary = fetch_range(
        args.start, args.end, queries=queries,
        cache_root=Path(args.cache_root) if args.cache_root else None,
        force=args.force, progress=report,
    )
    print(json.dumps(summary, indent=2, default=str))
    if summary["failed"]:
        print(f"\n{len(summary['failed'])} request(s) failed; re-run to retry.",
              file=sys.stderr)
        return 1
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    queries = _split_queries(args.queries)
    cache_root = Path(args.cache_root) if args.cache_root else None
    manifest = read_manifest(cache_root)

    print(f"queries           : {queries}")
    print(f"cache root        : {cache_root or DEFAULT_CACHE_ROOT}")
    if manifest.empty:
        print("\nCache is empty. Fetch first:")
        print("  python -m signals.text_sentiment fetch "
              "--start 2024-01-01 --end 2024-03-31")
        return 1

    print(f"manifest rows     : {len(manifest)}")
    if "hits" in manifest.columns:
        print(f"documents fetched : {int(manifest['hits'].fillna(0).sum())}")
    if "truncated" in manifest.columns:
        print(f"truncated days    : {int(manifest['truncated'].fillna(False).sum())}")

    dates = sorted(manifest["date"].dropna().astype(str)) if "date" in manifest else []
    if not dates:
        return 1
    first, last = dates[0], dates[-1]
    print(f"date range        : {first} .. {last}")

    documents = load_cached_range(first, last, queries=queries, cache_root=cache_root)
    print(f"unique documents  : {len(documents)}")
    if documents.empty:
        return 1
    span_days = max(1, (pd.to_datetime(last) - pd.to_datetime(first)).days + 1)
    print(f"documents per day : {len(documents) / span_days:.1f}")
    print(f"stories/comments  : {documents['kind'].value_counts().to_dict()}")

    scorer = get_scorer(args.scorer)
    print(f"scorer            : {scorer.describe()}")
    scored = score_documents(documents, scorer)
    print(f"sentiment mean/std: {scored['sentiment'].mean():+.4f} / "
          f"{scored['sentiment'].std():.4f}")

    bars = pd.date_range(pd.to_datetime(first).ceil("h"),
                         pd.to_datetime(last) + pd.Timedelta(hours=23), freq="h")
    aligned = align_to_bars(scored, bars, lag_bars=args.lag_bars,
                            window_hours=args.window_hours)
    print(f"\naligned onto {len(bars)} hourly bars "
          f"({args.window_hours}h window, lag {args.lag_bars}):")
    print(aligned[["doc_count", "mean_sentiment", "z_mean_sentiment"]]
          .describe().to_string())

    readings = [score_row(aligned.loc[t], min_documents=args.min_documents)
                for t in aligned.index]
    usable = [r for r in readings if r.available]
    moved = [r for r in usable if r.bullish_points or r.bearish_points]
    print(f"\nusable bars       : {len(usable)}/{len(readings)}")
    print(f"bars with points  : {len(moved)}"
          f"{f' ({len(moved) / len(usable):.1%} of usable)' if usable else ''}")
    if usable:
        strongest = max(usable, key=lambda r: abs(r.score))
        print("\nstrongest reading :")
        print("  " + strongest.reasoning)
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m signals.text_sentiment",
        description="Fetch and inspect free Hacker News text sentiment.")
    parser.add_argument("--cache-root", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="check reachability and scoring")
    p_probe.add_argument("--query", default="bitcoin")
    p_probe.add_argument("--date", default=None, help="YYYY-MM-DD")
    p_probe.add_argument("--scorer", default="vader", choices=["vader", "cryptobert"])
    p_probe.set_defaults(func=_cmd_probe)

    p_fetch = sub.add_parser("fetch", help="download documents into the cache")
    p_fetch.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive, UTC)")
    p_fetch.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive, UTC)")
    p_fetch.add_argument("--queries", default=None,
                         help=f"comma separated (default: {','.join(DEFAULT_QUERIES)})")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_report = sub.add_parser("report", help="summarise the corpus and features")
    p_report.add_argument("--queries", default=None)
    p_report.add_argument("--scorer", default="vader", choices=["vader", "cryptobert"])
    p_report.add_argument("--lag-bars", type=int, default=1)
    p_report.add_argument("--window-hours", type=int, default=24)
    p_report.add_argument("--min-documents", type=int, default=5)
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
