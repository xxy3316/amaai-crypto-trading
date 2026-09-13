"""Every runtime flag and domain type in one place.

WHY THIS MODULE EXISTS
----------------------
These settings used to be scattered through `auto-trade.py` at lines 341, 531,
623, 630, 1193, 1194, 3308 and beyond, each with its own ad-hoc parsing. To
answer "what configuration produced this result?" you had to grep the whole
file, and every new flag added another bespoke `os.getenv(...).strip().lower()
in ("1","true",...)` incantation.

Collecting them here gives the paper a single place to describe the
experimental configuration, and `describe_configuration()` dumps the whole set
as one dict for a run's provenance record.

This module imports nothing from the project, so it can be imported from
anywhere without a cycle. It also imports no Streamlit, which is what allows a
headless backtest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# ── ENV PARSING HELPERS ─────────────────────────────────────────────────────
# One implementation each, so a malformed value fails the same way everywhere
# instead of crashing in one place and silently defaulting in another.

_TRUTHY = ("1", "true", "yes", "on")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUTHY


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def env_tuple(name: str, default: tuple) -> tuple:
    raw = env_str(name)
    if not raw:
        return tuple(default)
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    return parts or tuple(default)


# ── DOMAIN TYPES ────────────────────────────────────────────────────────────

class TradingAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MarketData:
    timestamp: datetime
    price: float
    volume: float
    ma20: float
    upper_bb: float
    lower_bb: float
    rsi: float
    macd_hist: float


@dataclass
class TradingDecision:
    action: TradingAction
    confidence: float
    reasoning: str
    price: float
    timestamp: datetime
    # Phase 1: what the rule engine alone would have done, and what the LLM
    # changed. Defaults keep every existing constructor call valid.
    rule_action: Optional[str] = None
    llm: Optional[dict] = None


class SyntheticDataBlocked(RuntimeError):
    """Raised when live data is unavailable and synthetic fallback is disabled."""


#: Bars consumed by the technical indicators before any signal is meaningful.
#: MA20 needs 20 prior closes, so the simulation starts here rather than at 0.
#: Both the engine and the results panel read this, so the buy-and-hold
#: baseline drawn in the chart is the same one the summary reports.
INDICATOR_WARMUP_BARS = 20


class InsufficientHistory(RuntimeError):
    """Raised when a run has too few bars to survive indicator warmup.

    Its own type, rather than a bare IndexError from somewhere deep in the
    loop: this is a user-correctable input problem (date range too short for
    the chosen interval), and the message should say so.
    """


class TradingConfig(BaseModel):
    initial_capital: float = 1000.0
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    max_position_size: float = 1.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    # Read from BUY_FEE_PCT / SELL_FEE_PCT via default_factory rather than a
    # literal, because those constants are defined further down with the other
    # execution-realism settings. Resolved at construction, so a test that
    # patches the module constant gets the patched value.
    buy_fee_pct: float = Field(default_factory=lambda: BUY_FEE_PCT)
    sell_fee_pct: float = Field(default_factory=lambda: SELL_FEE_PCT)
    # `enable_deep_learning` was removed on 2026-09-13 along with the agent it
    # gated. The agent had no training code, so it never produced a prediction;
    # see the note at the top of auto-trade.py.
    enable_vector_db: bool = True
    show_reasoning: bool = True
    target_win_rate: float = 0.75
    min_confidence: float = 0.65
    # Decide every N bars. 6 is the value for HOURLY bars; the run overwrites it
    # via decision_step_bars(interval) so the cadence stays ~DECISION_CADENCE_HOURS
    # whatever the interval. Left as a plain default so a config built without
    # an interval still behaves exactly as it always did.
    simulation_step: int = 6

    # Strategy presets
    trading_mode: str = "moderate"  # conservative | moderate | aggressive
    position_size_pct: float = 50.0
    signal_threshold: int = 1

    @classmethod
    def get_conservative_config(cls, initial_capital: float = 1000.0):
        return cls(
            initial_capital=initial_capital,
            rsi_oversold=25.0, rsi_overbought=75.0,
            max_position_size=0.3, stop_loss_pct=0.03, take_profit_pct=0.06,
            min_confidence=0.80, signal_threshold=3, position_size_pct=25.0,
            trading_mode="conservative",
        )

    @classmethod
    def get_moderate_config(cls, initial_capital: float = 1000.0):
        return cls(
            initial_capital=initial_capital,
            rsi_oversold=30.0, rsi_overbought=70.0,
            max_position_size=0.6, stop_loss_pct=0.05, take_profit_pct=0.10,
            min_confidence=0.65, signal_threshold=2, position_size_pct=50.0,
            trading_mode="moderate",
        )

    @classmethod
    def get_aggressive_config(cls, initial_capital: float = 1000.0):
        return cls(
            initial_capital=initial_capital,
            rsi_oversold=35.0, rsi_overbought=65.0,
            max_position_size=0.9, stop_loss_pct=0.08, take_profit_pct=0.15,
            min_confidence=0.55, signal_threshold=1, position_size_pct=75.0,
            trading_mode="aggressive",
        )

    @classmethod
    def for_mode(cls, mode: str, initial_capital: float = 1000.0):
        """Preset lookup by name, so callers stop repeating the if/elif chain."""
        builders = {
            "conservative": cls.get_conservative_config,
            "moderate": cls.get_moderate_config,
            "aggressive": cls.get_aggressive_config,
        }
        return builders.get(str(mode).strip().lower(),
                            cls.get_moderate_config)(initial_capital)


# ── PHASE 1: LLM PARTICIPATION ──────────────────────────────────────────────
# The LLM participates in the decision instead of narrating it. Setting
# USE_LLM_DECISIONS=false runs the rules-only ablation arm, which reproduces
# the pre-Phase-1 behaviour exactly and is the paper's baseline.
USE_LLM_DECISIONS = env_bool("USE_LLM_DECISIONS", True)

# Largest absolute adjustment the LLM may apply to the rule signal score. The
# bound is what makes the model's contribution both safe and measurable.
LLM_MAX_ADJUSTMENT = env_int("LLM_MAX_ADJUSTMENT", 3)

LLM_TEMPERATURE = env_float("LLM_TEMPERATURE", 0.0)


# ── PHASE 2: RESEARCH INTEGRITY GUARDS ──────────────────────────────────────
# Synthetic price data must never masquerade as market data. Previously, if
# Binance failed, the code generated a random walk and every downstream number
# looked real. Synthetic data is now opt-in, labelled, and stamped onto the
# DataFrame itself.
ALLOW_SYNTHETIC_DATA = env_bool("ALLOW_SYNTHETIC_DATA", False)

# ALLOW_SYNTHETIC_SENTIMENT used to live here. The channel it guarded invented
# quotes and attributed them to real, named people; that agent has now been
# deleted outright rather than left switchable, so the flag has no referent.
# Text sentiment comes from a cached, hashed Hacker News corpus (see
# signals/text_sentiment.py) and has no synthetic mode to enable.

# Realistic fills. A decision made from a bar's close used to be filled at that
# same close, which assumes trading at a price you only learn once the bar has
# ended. next_open fills at the following bar's open plus slippage; same_close
# keeps the optimistic convention available as an explicit ablation.
EXECUTION_MODE = env_str("EXECUTION_MODE", "next_open").lower()
SLIPPAGE_PCT = env_float("SLIPPAGE_PCT", 0.05)  # per side, percent

# Trading fees, percent per side. These were previously hard-coded on
# TradingConfig as buy=0.10 / sell=0.0, and BUY_FEE_PCT / SELL_FEE_PCT in .env
# were read by nothing at all, so every backtest paid to enter and exited free.
# On the golden window that was ~$125 of unmodelled cost against $281 of
# reported profit, i.e. roughly 44% of the result.
#
# The defaults now match Binance spot taker fees, which are symmetric at 0.1%
# per side. An exit costs the same as an entry; a strategy that trades more
# pays more, which is the whole point of charging it.
BUY_FEE_PCT = env_float("BUY_FEE_PCT", 0.10)
SELL_FEE_PCT = env_float("SELL_FEE_PCT", 0.10)


# ── PHASE 3: EXOGENOUS POSITIONING SIGNAL ───────────────────────────────────
# Free, keyless Binance futures positioning: revealed preference (what traders
# did with money) rather than stated preference (what they posted). Defaults to
# false so leaving it alone reproduces the previous behaviour, which is what
# keeps the ablation arms comparable.
USE_POSITIONING_SIGNAL = env_bool("USE_POSITIONING_SIGNAL", False)

# Cap on the signal points positioning may add, so one exogenous feed can never
# dominate the technical rule engine. 2 matches the weight of the RSI rule.
POSITIONING_MAX_POINTS = env_int("POSITIONING_MAX_POINTS", 2)

# Bars of extra lag on top of the point-in-time cutoff. 1 means the bar opening
# at t sees only data published at or before t - 1 bar, while the price path
# already uses that bar's close, so positioning is lagged MORE than price.
POSITIONING_LAG_BARS = env_int("POSITIONING_LAG_BARS", 1)

# Rolling window, in bars, for the causal z-scores. 168 = one week of 1h bars.
# Applies to sub-daily intervals; see POSITIONING_ZSCORE_WINDOW_DAILY below.
POSITIONING_ZSCORE_WINDOW = env_int("POSITIONING_ZSCORE_WINDOW", 168)

# The same bar COUNT means wildly different things per interval: 168 bars is one
# week of 1h bars but 168 DAYS of 1d bars. A daily run therefore demanded most
# of a year of warmup history before a single bar could be scored, found only a
# month of it in the cache, produced an all-NaN z-score, and switched the whole
# channel off -- reporting "usable on 0.0% of bars", which reads like broken
# data rather than an impossible warmup.
#
# 30 daily bars is roughly a month of baseline: long enough for a stable mean
# and standard deviation, short enough to be satisfiable. This is a stated
# prior, not a fitted one, and it is reported in the run metadata so the
# baseline length of any run is recoverable from its own record.
POSITIONING_ZSCORE_WINDOW_DAILY = env_int("POSITIONING_ZSCORE_WINDOW_DAILY", 30)

#: Bar intervals scored against the daily window. Binance spells the monthly
#: interval "1M" and the minute intervals "1m", so case is significant here and
#: these are matched exactly rather than case-folded.
_DAILY_OR_COARSER_INTERVALS = frozenset({"1d", "3d", "1w", "1M"})


#: Duration of one bar, in hours, for each interval the app can request.
#: Binance spells minutes "1m" and months "1M", so case is significant.
_INTERVAL_HOURS = {
    "1m": 1 / 60, "3m": 3 / 60, "5m": 5 / 60, "15m": 15 / 60, "30m": 30 / 60,
    "1h": 1.0, "2h": 2.0, "4h": 4.0, "6h": 6.0, "8h": 8.0, "12h": 12.0,
    "1d": 24.0, "3d": 72.0, "1w": 168.0, "1M": 720.0,
}


def interval_hours(interval: Optional[str] = None) -> Optional[float]:
    """Hours covered by one bar of `interval`, or None if unrecognised."""
    key = str(interval or "").strip()
    if key in _INTERVAL_HOURS:
        return _INTERVAL_HOURS[key]
    # Tolerate case drift ("1D", "4H") without ever folding "1M" into "1m".
    lowered = key.lower()
    if lowered != "1m" and lowered in _INTERVAL_HOURS:
        return _INTERVAL_HOURS[lowered]
    return None


# How often the system stops to make a decision, expressed as a DURATION rather
# than a bar count. The old `simulation_step = 6` was a bar count written for
# hourly bars ("every 6 hours"), and nothing converted it for other intervals,
# so a daily run decided every 6 DAYS: a 30-day backtest produced two decisions
# and one round trip, and the headline win rate was one trade out of two.
#
# Hours are the honest unit here -- the intent was always a cadence in time --
# and the conversion to bars happens once, per interval, below.
DECISION_CADENCE_HOURS = env_float("DECISION_CADENCE_HOURS", 6.0)


def decision_step_bars(interval: Optional[str] = None,
                       cadence_hours: Optional[float] = None) -> int:
    """Bars between decisions at this interval, never less than 1.

    A cadence finer than one bar is not expressible -- the system cannot decide
    more often than the data updates -- so it clamps to every bar rather than
    silently rounding to zero and dividing by it.
    """
    hours = cadence_hours if cadence_hours is not None else DECISION_CADENCE_HOURS
    bar = interval_hours(interval)
    if bar is None or bar <= 0:
        # Unrecognised interval: keep the historical bar count rather than
        # inventing a cadence for a bar size we cannot measure.
        return max(1, int(round(float(hours))))
    return max(1, int(round(float(hours) / bar)))


def positioning_zscore_window(interval: Optional[str] = None) -> int:
    """Bars of baseline for the positioning z-scores at this bar interval.

    Both the agent and the auto-fetch warmup range read this, so the history
    that gets downloaded and the history the z-scores need cannot drift apart.
    """
    key = str(interval or "").strip()
    if key in _DAILY_OR_COARSER_INTERVALS or key.lower() in {"1d", "3d", "1w"}:
        return POSITIONING_ZSCORE_WINDOW_DAILY
    return POSITIONING_ZSCORE_WINDOW

# Download missing days during a run. False = frozen offline run, which is the
# mode to use for the final numbers in a paper.
POSITIONING_AUTO_FETCH = env_bool("POSITIONING_AUTO_FETCH", True)


# ── PHASE 3: EXOGENOUS TEXT SENTIMENT ───────────────────────────────────────
# Real public posts with real timestamps, scored by a real model. Source is the
# Hacker News Search API, because Reddit, StockTwits and CryptoPanic are all
# blocked by this network's proxy and GDELT rate-limits its shared egress IP.
USE_TEXT_SENTIMENT = env_bool("USE_TEXT_SENTIMENT", False)

# 'vader' (installed, instant) or 'cryptobert' (needs transformers + torch).
# An unavailable CryptoBERT falls back to VADER loudly, never silently.
TEXT_SENTIMENT_SCORER = env_str("TEXT_SENTIMENT_SCORER", "vader").lower()

TEXT_SENTIMENT_QUERIES = env_tuple("TEXT_SENTIMENT_QUERIES", ("bitcoin", "crypto"))

# Capped at 1 by default, HALF the positioning cap. A stated prior, not a
# fitted parameter: text sentiment is a single noisy feature crossing its
# 1-sigma threshold on roughly a third of bars, whereas positioning averages
# three features that often disagree and fires on roughly a seventh.
TEXT_SENTIMENT_MAX_POINTS = env_int("TEXT_SENTIMENT_MAX_POINTS", 1)

TEXT_SENTIMENT_LAG_BARS = env_int("TEXT_SENTIMENT_LAG_BARS", 1)

# Trailing aggregation window. Hacker News yields tens of documents a day, so a
# per-bar reading would be noise; 24h is what makes the mean stable at 1h bars.
TEXT_SENTIMENT_WINDOW_HOURS = env_int("TEXT_SENTIMENT_WINDOW_HOURS", 24)

TEXT_SENTIMENT_ZSCORE_WINDOW = env_int("TEXT_SENTIMENT_ZSCORE_WINDOW", 168)

# Data-quality floor: a window holding two comments is noise, not sentiment.
TEXT_SENTIMENT_MIN_DOCS = env_int("TEXT_SENTIMENT_MIN_DOCS", 5)

TEXT_SENTIMENT_AUTO_FETCH = env_bool("TEXT_SENTIMENT_AUTO_FETCH", True)


# ── DATABASE (OPTIONAL) ─────────────────────────────────────────────────────
ENABLE_DATABASE = env_bool("ENABLE_DATABASE", False)


def describe_configuration() -> dict:
    """The whole experimental configuration as one dict, for provenance.

    A run's reported numbers are only interpretable alongside the flags that
    produced them, so this travels with the result rather than being
    reconstructed by hand from the environment afterwards.
    """
    return {
        "phase1": {
            "use_llm_decisions": USE_LLM_DECISIONS,
            "llm_max_adjustment": LLM_MAX_ADJUSTMENT,
            "llm_temperature": LLM_TEMPERATURE,
            # Read from the environment rather than imported: core.config must
            # not import core.llm (llm depends on config, not the reverse).
            "llm_prompt_stance": os.getenv("LLM_PROMPT_STANCE",
                                           "conservative").strip().lower(),
        },
        "phase2": {
            "allow_synthetic_data": ALLOW_SYNTHETIC_DATA,
            "execution_mode": EXECUTION_MODE,
            "slippage_pct": SLIPPAGE_PCT,
            "buy_fee_pct": BUY_FEE_PCT,
            "sell_fee_pct": SELL_FEE_PCT,
        },
        "phase3_positioning": {
            "enabled": USE_POSITIONING_SIGNAL,
            "max_points": POSITIONING_MAX_POINTS,
            "lag_bars": POSITIONING_LAG_BARS,
            "zscore_window": POSITIONING_ZSCORE_WINDOW,
            "zscore_window_daily": POSITIONING_ZSCORE_WINDOW_DAILY,
            "auto_fetch": POSITIONING_AUTO_FETCH,
        },
        "phase3_text_sentiment": {
            "enabled": USE_TEXT_SENTIMENT,
            "scorer": TEXT_SENTIMENT_SCORER,
            "queries": list(TEXT_SENTIMENT_QUERIES),
            "max_points": TEXT_SENTIMENT_MAX_POINTS,
            "lag_bars": TEXT_SENTIMENT_LAG_BARS,
            "window_hours": TEXT_SENTIMENT_WINDOW_HOURS,
            "zscore_window": TEXT_SENTIMENT_ZSCORE_WINDOW,
            "min_documents": TEXT_SENTIMENT_MIN_DOCS,
            "auto_fetch": TEXT_SENTIMENT_AUTO_FETCH,
        },
    }
