"""Binance futures positioning signal: free, no API key, byte-exact reproducible.

WHAT THIS IS
------------
Binance publishes static daily files of aggregate derivatives positioning at
5-minute resolution, downloadable without an account, an API key or a rate
limit:

    https://data.binance.vision/data/futures/um/daily/metrics/
        BTCUSDT/BTCUSDT-metrics-2024-01-02.zip

Each file carries these columns:

    create_time                        timestamp, UTC, 5-minute grid
    symbol                             e.g. BTCUSDT
    sum_open_interest                  open interest, base units
    sum_open_interest_value            open interest, quote units
    count_toptrader_long_short_ratio   top traders, ratio of long ACCOUNTS
    sum_toptrader_long_short_ratio     top traders, ratio of long POSITIONS
    count_long_short_ratio             all traders, ratio of long ACCOUNTS
    sum_taker_long_short_vol_ratio     aggressive (taker) buy/sell volume ratio

This is *revealed-preference* sentiment: what traders did with money, rather
than what they posted. It replaces the dead Twitter-scraping path, which could
never work for a backtest (see the module notes at the bottom of this file).

WHY IT SURVIVES PEER REVIEW
---------------------------
  * Free and keyless, so a reviewer can reproduce the data pull.
  * The files are STATIC. A reviewer re-downloading gets identical bytes, which
    Google Trends (a resampled index) can never offer.
  * Binance publishes a .CHECKSUM beside each file, so integrity is verifiable
    rather than asserted.
  * 5-minute native resolution, so aligning onto 1h bars is downsampling and
    never interpolation.

TWO CAVEATS THAT BELONG IN THE PAPER
------------------------------------
  1. CROSS-MARKET. These metrics describe the USD-M PERPETUAL FUTURES market,
     while the backtest trades SPOT. Using futures positioning as a signal for
     spot is standard practice, but it is a modelling choice and must be
     disclosed, not buried.
  2. COVERAGE. History depends on when Binance began publishing per symbol
     (roughly 2020 for BTCUSDT) and the newest day appears with a lag. Run
     `python -m signals.binance_positioning report` to print the coverage
     actually obtained rather than the coverage hoped for.

POINT-IN-TIME DISCIPLINE
------------------------
`align_to_bars` maps 5-minute metrics onto price bars so that the bar whose
OPEN time is t sees only rows with create_time <= t - lag_bars * bar_duration.
With the default lag_bars=1 the positioning feature is strictly one full bar
behind the bar's own open, while the price path already uses that bar's CLOSE.
Positioning is therefore always more heavily lagged than price, which is the
safe direction: it cannot manufacture a look-ahead advantage.

Rolling z-scores use pandas' right-aligned `rolling`, which is causal.

NO SYNTHETIC FALLBACK
---------------------
Consistent with the Phase 2 integrity rules, a missing day yields a gap and a
recorded warning. This module never invents a reading.

CLI
---
    python -m signals.binance_positioning fetch  --symbol BTCUSDT \
        --start 2024-01-01 --end 2024-03-31
    python -m signals.binance_positioning report --symbol BTCUSDT
    python -m signals.binance_positioning probe
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import io
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── CONSTANTS ───────────────────────────────────────────────────────────────

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
FILE_TEMPLATE = "{symbol}-metrics-{date}.zip"

#: Columns as published by Binance. Kept explicit so a silent upstream schema
#: change surfaces as a validation error instead of as quietly wrong features.
RAW_COLUMNS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

NUMERIC_COLUMNS = [c for c in RAW_COLUMNS if c not in ("create_time", "symbol")]

#: Repository-relative cache root. Raw zips are kept exactly as downloaded so
#: the cache is itself the reproducibility artefact.
DEFAULT_CACHE_ROOT = Path(
    os.getenv("EXOGENOUS_CACHE_ROOT", "data/exogenous")
) / "binance_positioning"

MANIFEST_NAME = "manifest.jsonl"

USER_AGENT = os.getenv(
    "EXOGENOUS_USER_AGENT",
    "amaai-crypto-trading/1.0 (academic backtest; contact via repository)",
)

#: The three directional features and their sign convention. This mapping is
#: FIXED and was not tuned on returns; that matters, because a reviewer will
#: otherwise assume the thresholds were fitted to the test window.
#:
#:   count_long_short_ratio        crowd account skew -> CONTRARIAN (-1)
#:     When the retail crowd is overwhelmingly long, forward returns have
#:     historically been weaker, so extreme long crowding reads bearish.
#:   sum_toptrader_long_short_ratio top-trader position skew -> MOMENTUM (+1)
#:     Binance's "top traders" cohort is the closest free proxy for informed
#:     positioning, so it is followed rather than faded.
#:   sum_taker_long_short_vol_ratio aggressive flow -> MOMENTUM (+1)
#:     Taker buy dominance is contemporaneous buying pressure.
DIRECTIONAL_FEATURES: Dict[str, int] = {
    "count_long_short_ratio": -1,
    "sum_toptrader_long_short_ratio": +1,
    "sum_taker_long_short_vol_ratio": +1,
}

#: z-score clip. Beyond 3 sigma the reading is treated as saturated, not as
#: proportionally more informative.
Z_CLIP = 3.0


# ── SYMBOL / DATE HELPERS ───────────────────────────────────────────────────

def normalise_symbol(symbol: str) -> str:
    """Map an exchange symbol onto the Binance futures file naming.

    'BTC/USDT' -> 'BTCUSDT';  'BTC/USDT:USDT' -> 'BTCUSDT';  'btcusdt' -> 'BTCUSDT'
    """
    if not symbol:
        raise ValueError("symbol is required")
    base = str(symbol).split(":")[0]
    return base.replace("/", "").replace("-", "").replace("_", "").strip().upper()


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.to_datetime(value).date()


def daterange(start, end) -> List[dt.date]:
    """Inclusive list of UTC dates from start to end."""
    d0, d1 = _as_date(start), _as_date(end)
    if d1 < d0:
        raise ValueError(f"end {d1} precedes start {d0}")
    return [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]


# ── CACHE + MANIFEST ────────────────────────────────────────────────────────

def _cache_dir(symbol: str, cache_root: Optional[Path] = None) -> Path:
    root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
    return root / normalise_symbol(symbol)


def _local_path(symbol: str, date: dt.date, cache_root: Optional[Path] = None) -> Path:
    sym = normalise_symbol(symbol)
    return _cache_dir(sym, cache_root) / FILE_TEMPLATE.format(
        symbol=sym, date=date.isoformat()
    )


def _append_manifest(symbol: str, record: dict, cache_root: Optional[Path] = None) -> None:
    """Record one provenance row. Append-only, so the pull history is auditable."""
    path = _cache_dir(symbol, cache_root) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def read_manifest(symbol: str, cache_root: Optional[Path] = None) -> pd.DataFrame:
    """Return the provenance manifest, newest pull per file kept."""
    path = _cache_dir(symbol, cache_root) / MANIFEST_NAME
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
    if "date" in df.columns and "pulled_at_utc" in df.columns:
        df = df.sort_values("pulled_at_utc").drop_duplicates("date", keep="last")
    return df.reset_index(drop=True)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ── DOWNLOAD ────────────────────────────────────────────────────────────────

def _remote_urls(symbol: str, date: dt.date) -> Tuple[str, str]:
    sym = normalise_symbol(symbol)
    name = FILE_TEMPLATE.format(symbol=sym, date=date.isoformat())
    url = f"{BASE_URL}/{sym}/{name}"
    return url, url + ".CHECKSUM"


def _apply_corporate_ca(session: requests.Session) -> bool:
    """Point this session at a merged certifi + Windows-trust-store CA bundle.

    Called only after an SSLError, so machines without a TLS-inspecting proxy
    never pay for it and never change behaviour. Scoped to `session.verify`
    rather than the REQUESTS_CA_BUNDLE environment variable, so components that
    already work (ccxt price downloads, the Azure OpenAI client) are untouched.
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


def _published_checksum(session: requests.Session, url: str, timeout: int) -> Optional[str]:
    """Fetch Binance's published SHA256 for a file, or None when unavailable."""
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        # Format is "<sha256>  <filename>"
        token = resp.text.strip().split()
        return token[0].lower() if token else None
    except requests.RequestException as e:
        logger.debug("Checksum fetch failed for %s: %s", url, e)
        return None


def fetch_range(
    symbol: str,
    start,
    end,
    cache_root: Optional[Path] = None,
    verify_checksums: bool = True,
    force: bool = False,
    timeout: int = 60,
    pause_s: float = 0.15,
    session: Optional[requests.Session] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Download every daily metrics file in [start, end] into the local cache.

    Idempotent: a file already cached is left untouched unless force=True, so
    re-running after a partial failure only fetches what is missing.

    A missing day (HTTP 404) is recorded as a gap and does NOT abort the run:
    Binance publishes the newest day with a lag, and early history simply does
    not exist for some symbols.

    A checksum MISMATCH does abort, because silently accepting corrupt bytes is
    worse than failing loudly.
    """
    sym = normalise_symbol(symbol)
    dates = daterange(start, end)
    owns_session = session is None
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)

    summary: Dict[str, Any] = {
        "symbol": sym,
        "requested_days": len(dates),
        "downloaded": 0,
        "cached": 0,
        "missing": [],
        "failed": [],
        "checksum_verified": 0,
        "checksum_unavailable": 0,
        "bytes": 0,
    }

    ca_healed = False
    try:
        for position, date in enumerate(dates, start=1):
            if progress is not None:
                try:
                    progress(position, len(dates), date)
                except Exception:  # a UI callback must never break the fetch
                    pass

            target = _local_path(sym, date, cache_root)
            if target.exists() and not force:
                summary["cached"] += 1
                continue

            url, checksum_url = _remote_urls(sym, date)
            pulled_at = dt.datetime.now(dt.timezone.utc).isoformat()
            try:
                resp = session.get(url, timeout=timeout)
            except requests.exceptions.SSLError as e:
                # Corporate TLS interception. Heal once, then retry this date.
                if not ca_healed and _apply_corporate_ca(session):
                    ca_healed = True
                    summary["ca_bundle_applied"] = True
                    try:
                        resp = session.get(url, timeout=timeout)
                    except requests.RequestException as retry_error:
                        summary["failed"].append({"date": date.isoformat(),
                                                  "error": str(retry_error)})
                        continue
                else:
                    logger.warning("TLS verification failed for %s: %s", url, e)
                    summary["failed"].append({"date": date.isoformat(),
                                              "error": f"SSLError: {e}"})
                    continue
            except requests.RequestException as e:
                logger.warning("Request failed for %s: %s", url, e)
                summary["failed"].append({"date": date.isoformat(), "error": str(e)})
                _append_manifest(sym, {
                    "date": date.isoformat(), "url": url, "pulled_at_utc": pulled_at,
                    "http_status": None, "error": f"{type(e).__name__}: {e}",
                }, cache_root)
                continue

            if resp.status_code == 404:
                summary["missing"].append(date.isoformat())
                _append_manifest(sym, {
                    "date": date.isoformat(), "url": url, "pulled_at_utc": pulled_at,
                    "http_status": 404, "note": "not published for this date",
                }, cache_root)
                continue

            if resp.status_code != 200:
                summary["failed"].append({
                    "date": date.isoformat(), "error": f"HTTP {resp.status_code}",
                })
                _append_manifest(sym, {
                    "date": date.isoformat(), "url": url, "pulled_at_utc": pulled_at,
                    "http_status": resp.status_code, "error": "unexpected status",
                }, cache_root)
                continue

            payload = resp.content
            digest = _sha256(payload)
            checksum_state = "not_checked"
            if verify_checksums:
                published = _published_checksum(session, checksum_url, timeout)
                if published is None:
                    checksum_state = "unavailable"
                    summary["checksum_unavailable"] += 1
                elif published == digest:
                    checksum_state = "verified"
                    summary["checksum_verified"] += 1
                else:
                    raise RuntimeError(
                        f"Checksum mismatch for {url}: Binance published {published}, "
                        f"downloaded bytes hash to {digest}. Refusing to cache "
                        f"corrupt data."
                    )

            # Validate before caching, so the cache never holds unparseable files.
            frame = _parse_zip_bytes(payload, source=url)

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

            summary["downloaded"] += 1
            summary["bytes"] += len(payload)
            _append_manifest(sym, {
                "date": date.isoformat(),
                "url": url,
                "local_path": str(target),
                "pulled_at_utc": pulled_at,
                "http_status": 200,
                "bytes": len(payload),
                "sha256": digest,
                "binance_checksum": checksum_state,
                "rows": int(len(frame)),
                "first_create_time": str(frame["create_time"].min()) if len(frame) else None,
                "last_create_time": str(frame["create_time"].max()) if len(frame) else None,
            }, cache_root)

            if pause_s:
                time.sleep(pause_s)
    finally:
        if owns_session:
            session.close()

    return summary


def warmup_span(bar_index: pd.DatetimeIndex, zscore_window: int = 168) -> pd.Timedelta:
    """How far before the first bar the z-scores need data.

    The rolling window is measured in BARS, so the span depends on the bar
    duration: 168 bars is one week of 1h bars but 168 days of daily bars.
    Deriving it from the actual bar spacing keeps this correct at any timeframe.

    One extra bar covers the point-in-time lag, plus a day of slack for the
    gaps Binance occasionally leaves in the 5-minute series.
    """
    delta = _infer_bar_delta(pd.DatetimeIndex(bar_index))
    return delta * (int(zscore_window) + 1) + pd.Timedelta(days=1)


def warmup_start_date(bar_index: pd.DatetimeIndex, zscore_window: int = 168) -> dt.date:
    """First UTC date whose raw data is needed to serve `bar_index` in full.

    Both the agent and the Streamlit auto-fetch call this, so the range that
    gets downloaded and the range the z-scores need can never drift apart.
    """
    index = pd.DatetimeIndex(bar_index)
    return (index.min() - warmup_span(index, zscore_window)).date()


#: How long after a day closes Binance is assumed to have published its metrics
#: file. A 404 recorded sooner than this proves nothing -- the file may simply
#: not exist yet. Observed lag runs from a few hours to about a day and a half,
#: so this sits well beyond it: the cost of waiting is one extra request, the
#: cost of deciding too early is a permanent hole in the cache.
PUBLICATION_GRACE = dt.timedelta(hours=48)


def known_missing(symbol: str, cache_root: Optional[Path] = None,
                  now: Optional[dt.datetime] = None) -> set:
    """Dates the manifest records as HTTP 404 *and* that are genuinely absent.

    A 404 means either "never published" or "not published yet", and the status
    code alone cannot tell the two apart. The pull time can: Binance publishes a
    day only after that day closes, so a 404 recorded within PUBLICATION_GRACE
    of the day's end is provisional, and is retried on every later run once the
    day itself has closed. A 404 recorded after the grace elapsed is taken as
    final -- early history that genuinely never existed.

    Retrying from the moment the day closes, rather than waiting out the full
    grace, costs at most one redundant request per run for the one or two days
    still inside the publication lag, and recovers a day as soon as it lands.

    Treating every 404 as permanent is what froze the cache before: each recent
    day was fetched on the day itself, 404'd because it did not exist yet, and
    was then remembered as absent forever, so the newest edge of the cache
    silently stopped advancing while the gap grew by a day per day.
    """
    manifest = read_manifest(symbol, cache_root)
    if manifest.empty or "http_status" not in manifest.columns:
        return set()
    if "date" not in manifest.columns:
        return set()

    absent = manifest[manifest["http_status"] == 404]
    if absent.empty:
        return set()

    dates = absent["date"].astype(str)
    if "pulled_at_utc" not in absent.columns:
        # A manifest written before pull times were recorded: nothing to reason
        # from, so keep the old conservative behaviour.
        return set(dates)

    now = pd.Timestamp(now or dt.datetime.now(dt.timezone.utc))
    now = now.tz_localize("UTC") if now.tz is None else now.tz_convert("UTC")

    day_end = pd.to_datetime(dates, utc=True, errors="coerce") + pd.Timedelta(days=1)
    pulled = pd.to_datetime(absent["pulled_at_utc"], utc=True, errors="coerce")
    settled_at = day_end + pd.Timedelta(PUBLICATION_GRACE)

    asked_late_enough = pulled >= settled_at   # a real, permanent absence
    day_still_open = now < day_end             # cannot exist yet, do not ask
    unreadable = pulled.isna() | day_end.isna()

    skip = asked_late_enough | day_still_open | unreadable
    return set(dates[skip])


def missing_days(
    symbol: str,
    start,
    end,
    cache_root: Optional[Path] = None,
    retry_known_missing: bool = False,
) -> List[dt.date]:
    """Dates in [start, end] with no cached file, excluding known-absent days."""
    sym = normalise_symbol(symbol)
    skip = set() if retry_known_missing else known_missing(sym, cache_root)
    return [
        d for d in daterange(start, end)
        if not _local_path(sym, d, cache_root).exists()
        and d.isoformat() not in skip
    ]


def ensure_cached(
    symbol: str,
    start,
    end,
    cache_root: Optional[Path] = None,
    progress: Optional[Any] = None,
    verify_checksums: bool = True,
    retry_known_missing: bool = False,
) -> Dict[str, Any]:
    """Make sure [start, end] is cached, downloading only what is absent.

    This is what lets the Streamlit app work with no manual fetch step: the
    first run over a new window downloads it, every later run is a cache hit.

    Returns a summary with `already_complete` when nothing had to be fetched,
    so the caller can stay silent instead of flashing a progress bar.
    """
    sym = normalise_symbol(symbol)
    wanted = missing_days(sym, start, end, cache_root, retry_known_missing)
    if not wanted:
        return {"symbol": sym, "already_complete": True, "downloaded": 0,
                "requested_days": 0, "missing": [], "failed": []}

    # The absent days need not be contiguous, so fetch each rather than the
    # whole span, or a single stale day would re-download months of files.
    summary: Dict[str, Any] = {
        "symbol": sym, "already_complete": False, "downloaded": 0,
        "cached": 0, "missing": [], "failed": [], "checksum_verified": 0,
        "checksum_unavailable": 0, "bytes": 0, "requested_days": len(wanted),
    }
    session = requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    try:
        for position, day in enumerate(wanted, start=1):
            if progress is not None:
                try:
                    progress(position, len(wanted), day)
                except Exception:
                    pass
            one = fetch_range(
                sym, day, day, cache_root=cache_root,
                verify_checksums=verify_checksums, pause_s=0.0,
                session=session,
            )
            for key in ("downloaded", "cached", "checksum_verified",
                        "checksum_unavailable", "bytes"):
                summary[key] += one.get(key, 0)
            summary["missing"].extend(one.get("missing", []))
            summary["failed"].extend(one.get("failed", []))
            if one.get("ca_bundle_applied"):
                summary["ca_bundle_applied"] = True
    finally:
        session.close()

    return summary


# ── PARSE ───────────────────────────────────────────────────────────────────

def _parse_zip_bytes(payload: bytes, source: str = "<bytes>") -> pd.DataFrame:
    """Parse one daily metrics zip into a validated, typed DataFrame."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"no CSV inside archive from {source}")
            with zf.open(names[0]) as fh:
                blob = fh.read()
    except zipfile.BadZipFile as e:
        raise ValueError(f"{source} is not a valid zip archive: {e}") from e

    text = blob.decode("utf-8", errors="replace")
    has_header = text.lstrip().lower().startswith("create_time")
    frame = pd.read_csv(
        io.StringIO(text),
        header=0 if has_header else None,
        names=None if has_header else RAW_COLUMNS,
    )

    missing = [c for c in RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} is missing expected column(s) {missing}. Upstream schema "
            f"may have changed; got {list(frame.columns)}"
        )

    frame["create_time"] = _to_utc_naive(frame["create_time"])
    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["create_time"])
    frame = frame.drop_duplicates("create_time", keep="last")
    return frame.sort_values("create_time").reset_index(drop=True)


def _to_utc_naive(series: pd.Series) -> pd.Series:
    """Coerce Binance timestamps to tz-naive UTC.

    The price frame built by `_fetch_from_binance` is tz-naive UTC
    (`pd.to_datetime(ms, unit='ms')`), so positioning must match it exactly or
    `merge_asof` will refuse to join.

    Binance has shipped both an ISO-ish string and epoch milliseconds in this
    column across eras, so both are handled.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    # Epoch ms for any plausible trading era is >= ~1e12.
    if numeric.notna().all() and (numeric.dropna() > 1e11).all():
        out = pd.to_datetime(numeric, unit="ms", utc=True)
    else:
        out = pd.to_datetime(series, errors="coerce", utc=True)
    return out.dt.tz_convert("UTC").dt.tz_localize(None)


def load_cached_range(
    symbol: str,
    start,
    end,
    cache_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Concatenate cached daily files for [start, end]. Never touches the network.

    The backtest reads only this, which is what makes a run deterministic and
    replayable offline.
    """
    sym = normalise_symbol(symbol)
    frames, missing = [], []
    for date in daterange(start, end):
        path = _local_path(sym, date, cache_root)
        if not path.exists():
            missing.append(date.isoformat())
            continue
        try:
            frames.append(_parse_zip_bytes(path.read_bytes(), source=str(path)))
        except ValueError as e:
            logger.warning("Skipping unreadable cache file %s: %s", path, e)
            missing.append(date.isoformat())

    if missing:
        logger.warning(
            "Positioning cache has %d missing day(s) for %s (first: %s). Run "
            "`python -m signals.binance_positioning fetch` to fill them.",
            len(missing), sym, missing[0],
        )

    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("create_time", keep="last")
    return out.sort_values("create_time").reset_index(drop=True)


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
    raw: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    lag_bars: int = 1,
    zscore_window: int = 168,
    max_staleness_bars: float = 3.0,
) -> pd.DataFrame:
    """Project 5-minute positioning onto price bars, strictly point-in-time.

    The bar whose OPEN time is t is matched to the newest metrics row with
    create_time <= t - lag_bars * bar_duration. Nothing published at or after
    that cutoff can reach the bar, so no look-ahead is possible.

    Returns one row per bar, indexed identically to `bar_index`, carrying the
    raw levels, causal rolling z-scores, open-interest change, and the
    staleness of the reading in bars.

    `zscore_window` is in BARS (default 168 = one week of hourly bars).
    """
    empty = pd.DataFrame(index=bar_index)
    if raw is None or raw.empty or len(bar_index) == 0:
        return empty

    bar_index = pd.DatetimeIndex(bar_index)
    if bar_index.tz is not None:
        bar_index = bar_index.tz_convert("UTC").tz_localize(None)

    bar_delta = _infer_bar_delta(bar_index)
    cutoffs = bar_index - (bar_delta * int(lag_bars))

    left = pd.DataFrame({"bar_open": bar_index, "cutoff": cutoffs}).sort_values("cutoff")
    right = raw.sort_values("create_time")

    merged = pd.merge_asof(
        left,
        right,
        left_on="cutoff",
        right_on="create_time",
        direction="backward",
    )
    merged = merged.set_index("bar_open").reindex(bar_index)

    out = pd.DataFrame(index=bar_index)
    out["create_time"] = merged["create_time"]
    out["staleness_bars"] = (
        (merged["cutoff"] - merged["create_time"]) / bar_delta
    ).astype(float)

    # A reading older than max_staleness_bars is a data gap, not a signal.
    stale = out["staleness_bars"] > float(max_staleness_bars)

    for col in NUMERIC_COLUMNS:
        series = pd.to_numeric(merged[col], errors="coerce")
        series = series.mask(stale)
        out[col] = series

    # Open interest: the LEVEL is not comparable across regimes, the CHANGE is.
    # fill_method=None matters: pandas' default pads NaNs forward, which would
    # quietly reinstate the stale values that were just masked out.
    oi = out["sum_open_interest"]
    out["oi_pct_change"] = oi.pct_change(periods=1, fill_method=None) * 100.0

    # Causal z-scores. pandas' rolling is right-aligned, so window t covers
    # [t-w+1, t] and never reads the future.
    window = max(2, int(zscore_window))
    min_periods = min(window, max(8, window // 4))
    for col in list(DIRECTIONAL_FEATURES) + ["oi_pct_change"]:
        series = out[col]
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std(ddof=0)
        z = (series - mean) / std.replace(0.0, pd.NA)
        out[f"z_{col}"] = pd.to_numeric(z, errors="coerce").clip(-Z_CLIP, Z_CLIP)

    return out


# ── SCORING ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PositioningReading:
    """One bar's positioning reading, in the shape the decision agent consumes."""

    score: float = 0.0              # [-1, +1]; positive = bullish
    confidence: float = 0.0         # [0, 1]; coverage-driven, not a return forecast
    bullish_points: int = 0         # contribution to the rule engine
    bearish_points: int = 0
    reasoning: str = "No positioning data."
    source: str = "none"            # 'binance_futures_metrics' | 'none'
    available: bool = False
    features: Dict[str, float] = dataclasses.field(default_factory=dict)
    staleness_bars: Optional[float] = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["features"] = {k: (None if pd.isna(v) else float(v))
                         for k, v in self.features.items()}
        return d


def score_row(row: pd.Series, max_points: int = 2) -> PositioningReading:
    """Turn one aligned bar into a bounded score, points and a rationale.

    The mapping is fixed and symmetric: the mean of the available directional
    z-scores (sign-corrected per DIRECTIONAL_FEATURES) divided by Z_CLIP. No
    threshold here was fitted to returns, which is what lets the paper describe
    it as a pre-registered rule rather than an optimised one.
    """
    if row is None:
        return PositioningReading()

    contributions: Dict[str, float] = {}
    features: Dict[str, float] = {}
    notes: List[str] = []

    for col, sign in DIRECTIONAL_FEATURES.items():
        raw_level = row.get(col)
        z = row.get(f"z_{col}")
        if raw_level is not None and not pd.isna(raw_level):
            features[col] = float(raw_level)
        if z is None or pd.isna(z):
            continue
        z = float(z)
        features[f"z_{col}"] = z
        contributions[col] = sign * z
        if abs(z) >= 1.0:
            direction = "bullish" if sign * z > 0 else "bearish"
            label = {
                "count_long_short_ratio": "crowd long/short (faded)",
                "sum_toptrader_long_short_ratio": "top-trader long/short (followed)",
                "sum_taker_long_short_vol_ratio": "taker buy/sell flow (followed)",
            }[col]
            notes.append(f"{label} z={z:+.2f} -> {direction}")

    oi_z = row.get("z_oi_pct_change")
    if oi_z is not None and not pd.isna(oi_z):
        features["z_oi_pct_change"] = float(oi_z)

    if not contributions:
        return PositioningReading(
            reasoning=(
                "Positioning data present but no z-score available yet "
                "(rolling window still warming up)."
            ),
            source="binance_futures_metrics",
            available=False,
            features=features,
            staleness_bars=(None if pd.isna(row.get("staleness_bars", float("nan")))
                            else float(row.get("staleness_bars"))),
        )

    mean_z = sum(contributions.values()) / len(contributions)
    score = max(-1.0, min(1.0, mean_z / Z_CLIP))

    # Confidence reflects DATA QUALITY only: how many of the three features were
    # available, discounted by how stale the reading is. It is deliberately not
    # a claim about predictive accuracy.
    coverage = len(contributions) / len(DIRECTIONAL_FEATURES)
    staleness = row.get("staleness_bars")
    freshness = 1.0
    if staleness is not None and not pd.isna(staleness):
        freshness = max(0.3, 1.0 - 0.2 * float(staleness))
    confidence = round(max(0.0, min(1.0, coverage * freshness)), 3)

    # Fixed magnitude ladder, mirroring the granularity of the existing RSI rule.
    magnitude = abs(score)
    if magnitude >= 0.66:
        points = 2
    elif magnitude >= 0.33:
        points = 1
    else:
        points = 0
    points = min(points, int(max_points))

    bullish = points if score > 0 else 0
    bearish = points if score < 0 else 0

    if not notes:
        notes.append("all directional z-scores within 1 sigma")
    reasoning = (
        f"Futures positioning score {score:+.3f} from {len(contributions)}/"
        f"{len(DIRECTIONAL_FEATURES)} features ({'; '.join(notes)}). "
        f"Rule contribution: +{bullish} bullish / +{bearish} bearish."
    )

    return PositioningReading(
        score=round(score, 4),
        confidence=confidence,
        bullish_points=bullish,
        bearish_points=bearish,
        reasoning=reasoning,
        source="binance_futures_metrics",
        available=True,
        features=features,
        staleness_bars=(None if staleness is None or pd.isna(staleness)
                        else float(staleness)),
    )


# ── AGENT ───────────────────────────────────────────────────────────────────

class PositioningSignalAgent:
    """Serves a point-in-time positioning reading for any bar in the backtest.

    All alignment happens once at construction, so the per-bar lookup inside
    the simulation loop is a dictionary hit rather than a recomputation.

    Reads only the local cache. If the cache is empty the agent reports
    unavailable and every reading is 'no data' -- it will not silently
    substitute a zero and pretend that is a measurement.
    """

    def __init__(
        self,
        symbol: str,
        bar_index: pd.DatetimeIndex,
        enabled: bool = True,
        lag_bars: int = 1,
        max_points: int = 2,
        zscore_window: int = 168,
        cache_root: Optional[Path] = None,
    ):
        self.symbol = normalise_symbol(symbol)
        self.enabled = bool(enabled)
        self.lag_bars = int(lag_bars)
        self.max_points = int(max_points)
        self.available = False
        self.status = "disabled" if not enabled else "not initialised"
        self.aligned = pd.DataFrame()
        self.coverage = 0.0

        if not self.enabled or len(bar_index) == 0:
            return

        bar_index = pd.DatetimeIndex(bar_index)
        raw = load_cached_range(
            self.symbol,
            warmup_start_date(bar_index, zscore_window),
            bar_index.max().date(),
            cache_root=cache_root,
        )

        if raw.empty:
            self.status = (
                f"no cached positioning data for {self.symbol}; run "
                f"`python -m signals.binance_positioning fetch --symbol {self.symbol} "
                f"--start {bar_index.min().date()} --end {bar_index.max().date()}`"
            )
            logger.warning(self.status)
            return

        # Align over a bar index EXTENDED backwards by the warm-up window, then
        # slice back to the requested bars. Rolling z-scores run on the aligned
        # per-bar frame, so without this extension the window would start at the
        # first requested bar and the backtest would silently lose its first
        # `zscore_window` bars of signal -- weakening the signal-on arm for a
        # purely mechanical reason.
        delta = _infer_bar_delta(bar_index)
        warmup_bars = pd.DatetimeIndex([
            bar_index.min() - delta * i
            for i in range(int(zscore_window) + 1, 0, -1)
        ])
        extended = warmup_bars.append(bar_index).drop_duplicates().sort_values()

        self.aligned = align_to_bars(
            raw, extended, lag_bars=self.lag_bars, zscore_window=zscore_window
        ).reindex(bar_index)
        usable = self.aligned["z_count_long_short_ratio"].notna()
        self.coverage = float(usable.mean()) if len(self.aligned) else 0.0
        self.available = self.coverage > 0.0
        self.status = (
            f"{len(raw)} raw 5-min rows; usable on {self.coverage:.1%} of bars "
            f"(lag {self.lag_bars} bar(s))"
        )
        logger.info("Positioning agent ready for %s: %s", self.symbol, self.status)

    def reading_for(self, timestamp) -> PositioningReading:
        """Point-in-time reading for one bar. Unknown bars return 'no data'."""
        if not self.enabled:
            return PositioningReading(reasoning="Positioning signal disabled.")
        if self.aligned.empty:
            return PositioningReading(reasoning=f"Positioning unavailable: {self.status}")
        try:
            ts = pd.to_datetime(timestamp)
            if ts.tz is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            if ts not in self.aligned.index:
                return PositioningReading(
                    reasoning=f"No positioning row aligned to bar {ts}."
                )
            return score_row(self.aligned.loc[ts], max_points=self.max_points)
        except Exception as e:  # a data problem must not kill the backtest
            logger.warning("Positioning lookup failed at %s: %s", timestamp, e)
            return PositioningReading(reasoning=f"Positioning lookup error: {e}")

    def summary(self) -> dict:
        """Data-quality block for the results panel and the paper's data table."""
        if self.aligned.empty:
            return {
                "enabled": self.enabled, "available": False, "status": self.status,
                "symbol": self.symbol,
            }
        staleness = self.aligned["staleness_bars"].dropna()
        return {
            "enabled": self.enabled,
            "available": self.available,
            "status": self.status,
            "symbol": self.symbol,
            "bars": int(len(self.aligned)),
            "coverage_pct": round(self.coverage * 100, 2),
            "lag_bars": self.lag_bars,
            "max_points": self.max_points,
            "median_staleness_bars": (round(float(staleness.median()), 3)
                                      if len(staleness) else None),
            "source": "binance_futures_metrics (USD-M perpetual, data.binance.vision)",
        }


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cmd_fetch(args: argparse.Namespace) -> int:
    summary = fetch_range(
        args.symbol, args.start, args.end,
        cache_root=Path(args.cache_root) if args.cache_root else None,
        verify_checksums=not args.no_checksum,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, default=str))
    if summary["failed"]:
        print(f"\n{len(summary['failed'])} day(s) failed; re-run to retry them.",
              file=sys.stderr)
        return 1
    if summary["downloaded"] == 0 and summary["cached"] == 0:
        print("\nNothing cached. Every requested day was missing upstream: check "
              "the symbol and that the dates are inside Binance's published range.",
              file=sys.stderr)
        return 1
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    cache_root = Path(args.cache_root) if args.cache_root else None
    sym = normalise_symbol(args.symbol)
    manifest = read_manifest(sym, cache_root)
    files = sorted(_cache_dir(sym, cache_root).glob("*-metrics-*.zip"))

    print(f"symbol            : {sym}")
    print(f"cache directory   : {_cache_dir(sym, cache_root)}")
    print(f"cached files      : {len(files)}")
    if not files:
        print("\nCache is empty. Fetch first:")
        print(f"  python -m signals.binance_positioning fetch --symbol {sym} "
              f"--start 2024-01-01 --end 2024-03-31")
        return 1

    if not manifest.empty:
        verified = int((manifest.get("binance_checksum") == "verified").sum())
        missing = int((manifest.get("http_status") == 404).sum())
        print(f"manifest rows     : {len(manifest)}")
        print(f"checksum verified : {verified}")
        print(f"missing upstream  : {missing}")

    first = files[0].stem.split("-metrics-")[-1]
    last = files[-1].stem.split("-metrics-")[-1]
    raw = load_cached_range(sym, first, last, cache_root=cache_root)
    print(f"date range        : {first} .. {last}")
    print(f"raw 5-min rows    : {len(raw)}")
    if raw.empty:
        return 1

    print(f"create_time span  : {raw['create_time'].min()} .. {raw['create_time'].max()}")

    bars = pd.date_range(raw["create_time"].min().ceil("h"),
                         raw["create_time"].max().floor("h"), freq="h")
    aligned = align_to_bars(raw, bars, lag_bars=args.lag_bars)
    print(f"\naligned onto {len(bars)} hourly bars (lag {args.lag_bars} bar):")
    cols = [f"z_{c}" for c in DIRECTIONAL_FEATURES] + ["staleness_bars"]
    print(aligned[cols].describe().to_string())

    scored = [score_row(aligned.loc[t]) for t in aligned.index]
    usable = [s for s in scored if s.available]
    print(f"\nscored bars       : {len(usable)}/{len(scored)} usable")
    if usable:
        scores = pd.Series([s.score for s in usable])
        acted = sum(1 for s in usable if s.bullish_points or s.bearish_points)
        print(f"score mean/std    : {scores.mean():+.4f} / {scores.std():.4f}")
        print(f"score min/max     : {scores.min():+.4f} / {scores.max():+.4f}")
        print(f"bars with points  : {acted} ({acted / len(usable):.1%})")
        print("\nexample reading   :")
        print("  " + usable[len(usable) // 2].reasoning)
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    """Confirm data.binance.vision is reachable and the schema still matches."""
    sym = normalise_symbol(args.symbol)
    date = _as_date(args.date) if args.date else (
        dt.date.today() - dt.timedelta(days=5)
    )
    url, checksum_url = _remote_urls(sym, date)
    print(f"GET {url}")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    try:
        resp = session.get(url, timeout=60)
    except requests.RequestException as e:
        print(f"FAIL {type(e).__name__}: {e}", file=sys.stderr)
        print("\nIf this is an SSLError, it is almost certainly corporate TLS "
              "interception rather than a dead endpoint. Point REQUESTS_CA_BUNDLE "
              "at your company root CA and retry.", file=sys.stderr)
        return 1

    print(f"HTTP {resp.status_code}  bytes={len(resp.content)}")
    if resp.status_code != 200:
        print("Endpoint reachable but that date is not published. Try --date "
              "a few days earlier.", file=sys.stderr)
        return 1

    frame = _parse_zip_bytes(resp.content, source=url)
    print(f"parsed rows       : {len(frame)}")
    print(f"columns           : {list(frame.columns)}")
    print(f"create_time span  : {frame['create_time'].min()} .. "
          f"{frame['create_time'].max()}")
    published = _published_checksum(session, checksum_url, 60)
    digest = _sha256(resp.content)
    print(f"sha256            : {digest}")
    print(f"binance checksum  : {published or 'unavailable'} "
          f"({'match' if published == digest else 'no match / absent'})")
    print("\nSchema and reachability both OK. Safe to fetch a full range.")
    session.close()
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="python -m signals.binance_positioning",
        description="Fetch and inspect free Binance futures positioning data.",
    )
    parser.add_argument("--cache-root", default=None,
                        help="override the cache root (default data/exogenous)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download daily metrics into the cache")
    p_fetch.add_argument("--symbol", default="BTCUSDT")
    p_fetch.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive, UTC)")
    p_fetch.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive, UTC)")
    p_fetch.add_argument("--force", action="store_true",
                         help="re-download days already cached")
    p_fetch.add_argument("--no-checksum", action="store_true",
                         help="skip Binance checksum verification (halves requests)")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_report = sub.add_parser("report", help="summarise the cache and the features")
    p_report.add_argument("--symbol", default="BTCUSDT")
    p_report.add_argument("--lag-bars", type=int, default=1)
    p_report.set_defaults(func=_cmd_report)

    p_probe = sub.add_parser("probe", help="check reachability and upstream schema")
    p_probe.add_argument("--symbol", default="BTCUSDT")
    p_probe.add_argument("--date", default=None, help="YYYY-MM-DD to test")
    p_probe.set_defaults(func=_cmd_probe)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
