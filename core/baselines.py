"""Naive baselines, so "the strategy beat X" has a meaningful X.

Every result so far is measured against `rules_only` or against buy and hold.
Neither settles the question a reviewer asks first: does the signal carry
information, or would trading this often in this market have done as well?

A random trader answers it. Matched to the rule engine's trade COUNT and
constrained to the same decision points, it differs from the rule engine in
exactly one respect -- when it chooses to act -- so the gap between them is
attributable to the signal and nothing else.

The important part is that it is run MANY times. A single random run says
nothing; a distribution of a few hundred says where the rule engine falls
within the range of outcomes that pure chance produces. A rule engine at the
50th percentile of random is not a baseline worth measuring anything against,
however plausible its indicators look.

Both baselines run through `core.execution.execute_trade`, so slippage, both
fees and position sizing are identical to the real engine. A baseline with
cheaper execution than the strategy it benchmarks is not a baseline.

Layer: depends on core.config and core.execution. Imports no UI.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.config import TradingAction, TradingConfig, TradingDecision
from core.execution import execute_trade

logger = logging.getLogger(__name__)

#: Random draws used to build the distribution. Large enough that the reported
#: percentile is stable to about a point, small enough to stay instant.
DEFAULT_DRAWS = 400


def _fresh_portfolio(config: TradingConfig) -> dict:
    return {"cash": float(config.initial_capital), "holdings": 0.0,
            "holding": False, "entry_price": 0.0}


def _decision(action: TradingAction, price: float, timestamp) -> TradingDecision:
    return TradingDecision(
        action=action, confidence=0.5, reasoning="baseline",
        price=float(price), timestamp=timestamp,
    )


def _final_value(portfolio: dict, last_price: float) -> float:
    """Mark any open position to the final close, as the engine does."""
    return portfolio["cash"] + portfolio["holdings"] * float(last_price)


def _run_schedule(df: pd.DataFrame, schedule: Sequence[tuple],
                  config: TradingConfig) -> Dict[str, float]:
    """Execute a list of (bar_index, action) through the real execution model.

    Fills use the NEXT bar's open, matching EXECUTION_MODE="next_open": an
    order decided from a bar's close cannot be filled at that same close. A
    decision on the final bar has no successor and is therefore dropped, which
    is what the engine does too.
    """
    portfolio = _fresh_portfolio(config)
    trades: List[dict] = []
    for idx, action in schedule:
        if idx + 1 >= len(df):
            continue
        row = df.iloc[idx]
        nxt = df.iloc[idx + 1]
        result = execute_trade(
            _decision(action, float(row["close"]), df.index[idx]),
            portfolio, float(row["close"]), config, df.index[idx],
            fill_price=float(nxt["open"]), fill_timestamp=df.index[idx + 1],
        )
        if result:
            trades.append(result)

    final = _final_value(portfolio, df.iloc[-1]["close"])
    return {
        "final_value": final,
        "total_return_pct": (final - config.initial_capital)
                            / config.initial_capital * 100.0,
        "total_trades": len(trades),
    }


def buy_and_hold(df: pd.DataFrame, config: TradingConfig,
                 start_idx: int = 0) -> Dict[str, float]:
    """Buy at the first tradeable bar, hold to the end, paying real costs.

    The buy-and-hold figure quoted elsewhere is frictionless. This one pays the
    same entry fee and slippage as the strategy, which is the honest
    comparison: some of the strategy's shortfall is simply the cost of trading,
    and that part should not be counted against the signal.

    Note it buys `position_size_pct` of capital, not all of it, so it is
    comparable to a strategy that never goes fully invested either.
    """
    return _run_schedule(df, [(start_idx, TradingAction.BUY)], config)


def random_trader(df: pd.DataFrame, decision_points: Sequence[int],
                  config: TradingConfig, n_trades: int,
                  seed: Optional[int] = None) -> Dict[str, float]:
    """One random round-trip schedule with `n_trades` executions.

    Constrained exactly as the engine is: it can only act at a decision point,
    it cannot sell what it does not hold, and it alternates entries and exits.
    What it does NOT have is any reason for choosing one bar over another.
    """
    rng = np.random.default_rng(seed)
    points = list(decision_points)
    if len(points) < 2 or n_trades < 1:
        return _run_schedule(df, [], config)

    # Choose n_trades distinct decision points, in time order, and alternate
    # BUY/SELL from a flat book. An odd count leaves a position open at the
    # end, which is marked to the final close -- same as the engine.
    count = min(int(n_trades), len(points))
    chosen = sorted(rng.choice(len(points), size=count, replace=False))
    schedule = [(points[i], TradingAction.BUY if k % 2 == 0 else TradingAction.SELL)
                for k, i in enumerate(chosen)]
    return _run_schedule(df, schedule, config)


def random_distribution(df: pd.DataFrame, decision_points: Sequence[int],
                        config: TradingConfig, n_trades: int,
                        draws: int = DEFAULT_DRAWS,
                        seed: int = 12345) -> Dict[str, object]:
    """The distribution of outcomes from trading at random, same frequency.

    This is the object worth reporting. `percentile_of(x)` then answers the
    question that matters: how much of the random distribution does the real
    strategy beat? Around 50 means the signal added nothing detectable.
    """
    rng = np.random.default_rng(seed)
    returns = np.array([
        random_trader(df, decision_points, config, n_trades,
                      seed=int(rng.integers(0, 2**31 - 1)))["total_return_pct"]
        for _ in range(int(draws))
    ], dtype=float)

    return {
        "draws": int(draws),
        "n_trades": int(n_trades),
        "mean": float(np.mean(returns)),
        "median": float(np.median(returns)),
        "std": float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
        "p05": float(np.quantile(returns, 0.05)),
        "p95": float(np.quantile(returns, 0.95)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "returns": returns,
    }


def percentile_of(value: float, distribution: Dict[str, object]) -> float:
    """Share of the random distribution the given return beats, as a percent.

    50 means indistinguishable from chance. Below 50 means the strategy did
    WORSE than a coin flip trading at the same rate, which is a finding rather
    than a rounding error.
    """
    returns = np.asarray(distribution["returns"], dtype=float)
    if returns.size == 0:
        return float("nan")
    return float(100.0 * np.mean(returns < value))
