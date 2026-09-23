"""Does the model respond to a harmful channel as strongly as a useful one?

The correlational version of this question has already been answered across 530
logged decisions: the adjustment tracks the text-sentiment score (r=+0.40) at
least as strongly as the positioning score (r=+0.37), although text is the worst
arm measured in the ablation and positioning is roughly neutral. Correlation
cannot rule out a confound, though -- both channels move with the market, and so
does the rule state the model also sees.

This removes that objection by manipulating the channels directly.

Design
------
For each of a set of real bars, the rule state and the price are held exactly as
they were, and ONE channel reading is replaced with a synthetic value swept
across a grid. Everything else in the prompt is identical between conditions, so
a difference in the adjustment is caused by the injected value and nothing else.

Two details make the comparison clean:

  * The injected readings carry ZERO rule points. The rule engine's net signal
    is therefore identical across every condition, and the rule-based action is
    unchanged. Any movement in the adjustment is the model's response to the
    NUMBER IT WAS SHOWN, not a downstream effect of the rules moving.
  * The two channels are swept over the same grid with the same shape of
    reading, so the slopes are directly comparable. The quantity of interest is
    not whether the model responds -- it is whether it responds to the harmful
    channel as much as to the useful one.

A model that assesses input quality should show a materially flatter slope for
text than for positioning. Equal slopes mean it is reacting to the presence of a
number rather than to whether that number has ever been informative.

    python -m experiments.counterfactual_probe --bars 12 --stance neutral
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The probe needs the model, and needs the channels to reach the prompt.
os.environ.setdefault("USE_LLM_DECISIONS", "true")
os.environ.setdefault("USE_POSITIONING_SIGNAL", "true")
os.environ.setdefault("USE_TEXT_SENTIMENT", "true")
os.environ.setdefault("POSITIONING_AUTO_FETCH", "false")
os.environ.setdefault("TEXT_SENTIMENT_AUTO_FETCH", "false")

#: The sweep. Symmetric and spanning the channels' natural [-1, +1] range, so a
#: slope is estimated from both directions rather than extrapolated from one.
GRID = [-1.0, -0.5, 0.0, +0.5, +1.0]


def build_readings(score: float):
    """A positioning and a text reading at `score`, both worth zero rule points.

    Zero points is what keeps the rule engine still. `available=True` and a
    populated `features` dict matter because the prompt renders the z-scores
    from them; a reading that looks empty would be shown to the model as
    "not available" and the sweep would test nothing.
    """
    from signals.binance_positioning import PositioningReading
    from signals.text_sentiment import TextSentimentReading

    pos = PositioningReading(
        score=score, confidence=0.9, bullish_points=0, bearish_points=0,
        reasoning=f"counterfactual probe: injected score {score:+.2f}",
        source="binance_futures_metrics", available=True,
        features={"z_count_long_short_ratio": score * 2.0,
                  "z_sum_toptrader_long_short_ratio": score * 2.0,
                  "z_sum_taker_long_short_vol_ratio": score * 2.0},
        staleness_bars=0.5,
    )
    txt = TextSentimentReading(
        score=score, confidence=0.9, bullish_points=0, bearish_points=0,
        reasoning=f"counterfactual probe: injected score {score:+.2f}",
        source="hackernews", available=True, doc_count=60,
        mean_sentiment=score, features={"z_mean_sentiment": score * 2.0,
                                        "pos_share": 0.4, "neg_share": 0.4},
    )
    return pos, txt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=12,
                        help="distinct market states to sweep over")
    parser.add_argument("--stance", default="neutral",
                        choices=["conservative", "neutral", "assertive"])
    parser.add_argument("--start", default="2026-02-24")
    parser.add_argument("--end", default="2026-03-10")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    os.environ["LLM_PROMPT_STANCE"] = args.stance
    logging.disable(logging.INFO)

    import importlib.util
    spec = importlib.util.spec_from_file_location("autotrade", REPO / "auto-trade.py")
    APP = importlib.util.module_from_spec(spec)
    sys.modules["autotrade"] = APP
    spec.loader.exec_module(APP)

    from core.config import (INDICATOR_WARMUP_BARS, TradingConfig,
                             decision_step_bars)
    from core.data import fetch_binance_ta

    df = fetch_binance_ta(args.symbol, args.interval,
                          pd.Timestamp(args.start), pd.Timestamp(args.end))
    config = TradingConfig.get_moderate_config(10000.0)
    agent = APP.TradingDecisionAgent(df, config)
    if not agent.use_llm or agent.structured_llm is None:
        print("LLM not configured; nothing to probe.", file=sys.stderr)
        return 1

    points = list(range(INDICATOR_WARMUP_BARS, len(df),
                        decision_step_bars(args.interval)))
    step = max(1, len(points) // args.bars)
    bars = points[::step][:args.bars]
    portfolio = {"cash": 10000.0, "holdings": 0.0, "holding": False,
                 "entry_price": 0.0}

    total = len(bars) * len(GRID) * 2
    print(f"counterfactual probe · stance={args.stance} · {len(bars)} bars "
          f"× {len(GRID)} levels × 2 channels = {total} calls\n")

    records = []
    for n, idx in enumerate(bars, start=1):
        ts = df.index[idx]
        for channel in ("positioning", "text"):
            for level in GRID:
                pos, txt = build_readings(level)
                # Only the channel under test carries the injected value; the
                # other is held at a fixed neutral reading so it cannot drift.
                neutral_pos, neutral_txt = build_readings(0.0)
                decision = agent.make_decision(
                    ts, "market analysis held constant for this probe",
                    "pattern analysis held constant for this probe",
                    {"risk_level": "medium", "reasoning": "probe"},
                    dict(portfolio),
                    positioning=pos if channel == "positioning" else neutral_pos,
                    text_sentiment=txt if channel == "text" else neutral_txt,
                )
                llm = decision.llm or {}
                records.append({
                    "bar": int(idx), "timestamp": str(ts), "channel": channel,
                    "injected": level, "adjustment": llm.get("adjustment"),
                    "stance_said": llm.get("stance"),
                    "rule_signal": llm.get("rule_signal"),
                    "rule_action": llm.get("rule_action"),
                    "final_action": llm.get("final_action"),
                })
        print(f"  bar {n}/{len(bars)} ({ts}) done", flush=True)

    out = Path(args.out) if args.out else (
        REPO / "experiments" / "results" /
        f"counterfactual-{args.stance}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print(f"\n{len(records)} records -> {out}\n")
    report(records)
    return 0


def report(records: list) -> None:
    """Slope of adjustment on injected value, per channel."""
    from scipy import stats

    print(f"{'channel':<14} {'n':>5} {'slope':>9} {'r':>8} {'p':>10}  "
          f"mean adjustment by injected level")
    print("-" * 86)
    slopes = {}
    for channel in ("positioning", "text"):
        sub = [r for r in records if r["channel"] == channel
               and r["adjustment"] is not None]
        if len(sub) < 3:
            continue
        x = np.array([r["injected"] for r in sub], dtype=float)
        y = np.array([r["adjustment"] for r in sub], dtype=float)
        res = stats.linregress(x, y)
        slopes[channel] = res.slope
        by_level = {lv: np.mean([r["adjustment"] for r in sub
                                 if r["injected"] == lv]) for lv in GRID}
        cells = "  ".join(f"{lv:+.1f}:{by_level[lv]:+.2f}" for lv in GRID)
        print(f"{channel:<14} {len(sub):>5} {res.slope:>+9.3f} "
              f"{res.rvalue:>+8.3f} {res.pvalue:>10.2e}  {cells}")

    if len(slopes) == 2:
        p, t = slopes["positioning"], slopes["text"]
        print(f"\n  positioning slope {p:+.3f}   text slope {t:+.3f}")
        if abs(p) < 1e-9:
            return
        print(f"  the model moves {abs(t / p):.2f}x as much for the HARMFUL "
              f"channel as for the roughly-neutral one")
        print("\n  A model that assessed input quality would show a materially "
              "flatter\n  slope for text. Equal slopes mean it is reacting to a "
              "number, not to\n  whether that number has ever been informative.")


if __name__ == "__main__":
    raise SystemExit(main())
