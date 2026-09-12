"""Golden-output harness: proves a refactor changed no numbers.

Runs a fully deterministic backtest (rules-only, so no LLM sampling) over a
fixed window that both exogenous caches already cover, then writes a canonical
fingerprint of every number that matters.

    # before refactoring
    ./venv/Scripts/python.exe test/golden_backtest.py --save

    # after refactoring
    ./venv/Scripts/python.exe test/golden_backtest.py --check

`--check` exits non-zero and prints a field-by-field diff if anything moved.

Determinism notes:
  * USE_LLM_DECISIONS is forced false, and since 2026-09-11 that alone is
    sufficient: the market, pattern and risk agents are gated on the same flag
    and make no model call. This harness therefore exercises the REAL code
    path rather than a stubbed one. (It previously had to stub those three
    agents by hand, because the flag gated only the decision agent and a run
    otherwise made ~570 LLM calls and took ~47 minutes.) Pass --live-llm to run
    against the real model instead.
  * Binance klines for a closed historical window are immutable.
  * Both exogenous caches are read from disk, never re-fetched.
  * Wall-clock fields (run_at, latencies) are excluded from the fingerprint.

What this therefore covers: data fetching, indicator computation, exogenous
signal alignment and scoring, rule-engine arithmetic, the confidence gate, fill
pricing, fees, slippage and the summary aggregation. What it does not cover:
the LLM's own judgement, which is not a refactor concern.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Imported after the path setup: this file is run directly, so the repo root is
# not on sys.path until the line above.
from experiments.harness import load_app, run_headless  # noqa: E402

# Must be set BEFORE the app module is imported: these are read at import time.
os.environ["USE_LLM_DECISIONS"] = "false"
os.environ["USE_POSITIONING_SIGNAL"] = "true"
os.environ["USE_TEXT_SENTIMENT"] = "true"
os.environ["POSITIONING_AUTO_FETCH"] = "false"   # cache only, no network
os.environ["TEXT_SENTIMENT_AUTO_FETCH"] = "false"
os.environ["ENABLE_DATABASE"] = ""

logging.getLogger("streamlit").setLevel(logging.ERROR)

GOLDEN_PATH = REPO / "test" / "golden_backtest.json"

# A window both caches cover (positioning 2024-01-01.., text 2023-12-20..).
START = "2024-02-01"
END = "2024-03-20"
SYMBOL = "BTC/USDT"
INTERVAL = "1h"
CAPITAL = 10000.0
STRATEGY = "moderate"


def fingerprint(summary: dict) -> dict:
    """Canonical, wall-clock-free view of a run."""

    def r(value, places=6):
        return None if value is None else round(float(value), places)

    trades = []
    for t in summary.get("trades", []):
        trades.append({
            "action": str(t.get("action")),
            "timestamp": str(t.get("timestamp")),
            "price": r(t.get("price"), 8),
            "amount": r(t.get("amount"), 8),
            "value": r(t.get("value"), 8),
            "profit": r(t.get("profit"), 8),
            "profit_pct": r(t.get("profit_pct"), 8),
        })

    meta = summary.get("run_metadata", {}) or {}
    pos = summary.get("positioning_report", {}) or {}
    txt = summary.get("text_sentiment_report", {}) or {}
    llm = summary.get("llm_report", {}) or {}

    return {
        "window": {"symbol": SYMBOL, "interval": INTERVAL,
                   "start": START, "end": END, "strategy": STRATEGY},
        "performance": {
            "initial_capital": r(summary.get("initial_capital")),
            "final_value": r(summary.get("final_value")),
            "total_return_pct": r(summary.get("total_return_pct")),
            "buy_hold_return_pct": r(summary.get("buy_hold_return_pct")),
            "total_trades": summary.get("total_trades"),
            "winning_trades": summary.get("winning_trades"),
            "win_rate_pct": r(summary.get("win_rate_pct")),
            "decisions": summary.get("decisions"),
        },
        "trades": trades,
        "positioning": {
            "enabled": pos.get("enabled"),
            "decisions_with_reading": pos.get("decisions_with_reading"),
            "decisions_where_points_added": pos.get("decisions_where_points_added"),
            "mean_score": r(pos.get("mean_score")),
            "coverage_pct": r((pos.get("agent") or {}).get("coverage_pct")),
        },
        "text_sentiment": {
            "enabled": txt.get("enabled"),
            "decisions_with_reading": txt.get("decisions_with_reading"),
            "decisions_where_points_added": txt.get("decisions_where_points_added"),
            "mean_score": r(txt.get("mean_score")),
            "mean_docs_per_reading": r(txt.get("mean_docs_per_reading")),
            "coverage_pct": r((txt.get("agent") or {}).get("coverage_pct")),
        },
        "llm": {
            "enabled": llm.get("llm_enabled"),
            "decisions": llm.get("decisions"),
            "changed": llm.get("decisions_changed_by_llm"),
        },
        "provenance": {
            "bars": meta.get("bars"),
            "data_source": meta.get("data_source"),
            "execution_mode": meta.get("execution_mode"),
            # The full cost model, not just slippage. A recorded return is
            # only comparable against another run that charged the same fees,
            # and an exit fee of 0.0 silently inflated this baseline until
            # 2026-09-11.
            "slippage_pct": r(meta.get("slippage_pct")),
            "buy_fee_pct": r(meta.get("buy_fee_pct")),
            "sell_fee_pct": r(meta.get("sell_fee_pct")),
            "signal_threshold": meta.get("signal_threshold"),
            "positioning_enabled": meta.get("positioning_enabled"),
            "text_sentiment_enabled": meta.get("text_sentiment_enabled"),
        },
    }


def run(live_llm: bool = False) -> dict:
    # No stubbing: USE_LLM_DECISIONS=false (set at import, above) now disables
    # the support agents too, so the real code path is already deterministic.
    # --live-llm flips the flag on and accepts a non-reproducible run.
    if live_llm:
        os.environ["USE_LLM_DECISIONS"] = "true"
    app = load_app(stub_support=False)
    summary, _elapsed = run_headless(
        app, start=START, end=END, symbol=SYMBOL, interval=INTERVAL,
        capital=CAPITAL, strategy=STRATEGY,
    )
    return fingerprint(summary)


def diff(old, new, path=""):
    out = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            out += diff(old.get(key), new.get(key), f"{path}.{key}" if path else key)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append(f"{path}: length {len(old)} -> {len(new)}")
        for i, (a, b) in enumerate(zip(old, new)):
            out += diff(a, b, f"{path}[{i}]")
    elif old != new:
        out.append(f"{path}: {old!r} -> {new!r}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true",
                        help="record the current output as the golden baseline")
    parser.add_argument("--check", action="store_true",
                        help="compare the current output against the baseline")
    parser.add_argument("--live-llm", action="store_true",
                        help="enable the LLM instead of running rules-only "
                             "(slow, and the result is not bit-reproducible)")
    args = parser.parse_args()
    if not (args.save or args.check):
        parser.error("pass --save or --check")

    current = run(live_llm=args.live_llm)

    if args.save:
        GOLDEN_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"baseline written to {GOLDEN_PATH}")
        print(json.dumps(current["performance"], indent=2))
        print(json.dumps({"positioning": current["positioning"],
                          "text_sentiment": current["text_sentiment"]}, indent=2))
        return 0

    if not GOLDEN_PATH.exists():
        print(f"no baseline at {GOLDEN_PATH}; run --save first", file=sys.stderr)
        return 1

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    differences = diff(golden, current)
    if differences:
        print("GOLDEN MISMATCH -- the refactor changed the result:\n", file=sys.stderr)
        for line in differences:
            print(f"  {line}", file=sys.stderr)
        return 1

    print("GOLDEN MATCH: every recorded number is identical.")
    print(f"  trades={current['performance']['total_trades']} "
          f"return={current['performance']['total_return_pct']}% "
          f"decisions={current['performance']['decisions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
