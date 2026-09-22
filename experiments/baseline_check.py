"""Where does the rule engine sit against trading at random?

Everything in the ablation is measured relative to `rules_only`, so if that
baseline is itself no better than chance, every comparison above it inherits
the weakness. This asks the question directly, and costs nothing: no model
calls, no re-running of backtests.

The strategy returns are READ FROM AN EXISTING RESULTS FILE rather than
recomputed. That is not just a speed choice. An earlier version of this script
called `run_headless` for all nine windows inside one process and produced
-4.14% for a window the ablation had recorded as -0.40%; the same window in a
fresh process reproduced -0.40% exactly. State leaks between runs in a shared
interpreter -- which is precisely why `run_ablation.py` spends a subprocess per
arm, and why its docstring says so. Reusing its output sidesteps the problem
instead of re-creating it.

Only the baselines are computed here, and neither touches the app: the random
schedules and buy-and-hold both run through `core.execution.execute_trade` on
raw price data.

    python -m experiments.baseline_check experiments/results/<file>.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics as st
import sys
from pathlib import Path

# No model, no exogenous channels: the arm under test is the rule engine.
# Set before any core import, since core.config reads these at import time.
os.environ.setdefault("USE_LLM_DECISIONS", "false")
os.environ.setdefault("POSITIONING_AUTO_FETCH", "false")
os.environ.setdefault("TEXT_SENTIMENT_AUTO_FETCH", "false")

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.baselines import buy_and_hold, percentile_of, random_distribution
from core.config import (INDICATOR_WARMUP_BARS, TradingConfig,
                         decision_step_bars)
from core.data import fetch_binance_ta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="ablation JSONL to read strategy "
                                        "returns from")
    parser.add_argument("--arm", default="rules_only",
                        help="which arm to test against chance")
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--strategy", default="moderate")
    args = parser.parse_args()

    logging.disable(logging.INFO)
    rows = [json.loads(line) for line
            in Path(args.results).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    runs = sorted([r for r in rows if r.get("arm") == args.arm
                   and r.get("total_return_pct") is not None],
                  key=lambda r: r.get("window_name", ""))
    if not runs:
        print(f"no rows for arm {args.arm!r} in {args.results}", file=sys.stderr)
        return 1

    config = getattr(TradingConfig, f"get_{args.strategy}_config")(args.capital)
    interval = runs[0].get("window", {}).get("interval", "1h")
    symbol = runs[0].get("window", {}).get("symbol", "BTC/USDT")
    step = decision_step_bars(interval)

    print(f"arm `{args.arm}` from {Path(args.results).name}")
    print(f"{len(runs)} window(s) · {symbol} · {interval} · "
          f"{args.draws} random draws per window\n")
    print(f"{'window':<6} {'strategy':>9} {'trades':>7} {'rand p50':>9} "
          f"{'rand p05':>9} {'rand p95':>9} {'B&H+fees':>9} {'pctile':>7}")
    print("-" * 72)

    percentiles, beat_bh = [], 0
    for run in runs:
        ret = run["total_return_pct"]
        trades = run["total_trades"]
        df = fetch_binance_ta(symbol, interval,
                              pd.Timestamp(run["window_start"]),
                              pd.Timestamp(run["window_end"]))
        points = list(range(INDICATOR_WARMUP_BARS, len(df), step))
        dist = random_distribution(df, points, config, n_trades=trades,
                                   draws=args.draws)
        bh = buy_and_hold(df, config, start_idx=INDICATOR_WARMUP_BARS)
        pct = percentile_of(ret, dist)
        percentiles.append(pct)
        beat_bh += ret > bh["total_return_pct"]

        print(f"{run['window_name']:<6} {ret:>+9.2f} {trades:>7} "
              f"{dist['median']:>+9.2f} {dist['p05']:>+9.2f} "
              f"{dist['p95']:>+9.2f} {bh['total_return_pct']:>+9.2f} "
              f"{pct:>6.0f}%")

    mean_pct = st.mean(percentiles)
    print(f"\n`{args.arm}` sits at the {mean_pct:.0f}th percentile of random on "
          f"average (50 = indistinguishable from chance)")
    print(f"beat buy-and-hold-with-fees in {beat_bh}/{len(runs)} windows")
    if 40 <= mean_pct <= 60:
        print(f"\nREAD: `{args.arm}` is not distinguishable from trading at the "
              f"same frequency at random. Every comparison made against it "
              f"inherits that, and should say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
