"""Run the factorial ablation headlessly and write one JSONL row per arm.

This is what the restructure was for. Previously a run required a click in the
Streamlit UI, so an eight-arm ablation meant eight manual runs with no record
beyond a screenshot.

    python -m experiments.run_ablation --start 2024-02-01 --end 2024-03-20
    python -m experiments.run_ablation --arms rules_only,pos_only --repeats 3

Design
------
Each arm is a combination of the three switchable channels:

    positioning   USE_POSITIONING_SIGNAL
    text          USE_TEXT_SENTIMENT
    llm           USE_LLM_DECISIONS

Those are read at IMPORT time by `core.config`, so an arm cannot be configured
by mutating a module attribute after the fact. Each arm therefore runs in a
FRESH SUBPROCESS with its own environment, which also guarantees no state
leaks between arms (cached LLM clients, agent instances, warmed z-scores).

`llm=false` is genuinely LLM-free (since 2026-09-11)
----------------------------------------------------
It gates all four model-backed agents -- decision, market, pattern and risk --
so an `llm=false` arm makes zero model calls, needs no API key, and is
bit-reproducible. Verified by running the golden harness against the real
unstubbed path and reproducing the baseline exactly.

Before that fix the flag gated only the decision agent, so every "LLM-free"
number was contaminated: the risk agent's output feeds a confidence multiplier
that can flip a trade on its own (risk_level "high" multiplies confidence by
0.8, dropping a net_signal of 2 from 0.80 to 0.64, below the 0.65 moderate-mode
gate, turning a BUY into a HOLD). Any ablation table produced before
2026-09-11 should be re-run rather than quoted.

--stub-support-agents is now only meaningful for `llm=true` arms, where it pins
the market/pattern/risk narrators so the measured effect is the DECISION
agent's alone. It changes nothing for `llm=false` arms.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Named arms, as (positioning, text, llm). Names are what appear in the table.
ARMS: dict = {
    "rules_only":      (False, False, False),
    "pos_only":        (True,  False, False),
    "text_only":       (False, True,  False),
    "pos_text":        (True,  True,  False),
    "llm_only":        (False, False, True),
    "pos_llm":         (True,  False, True),
    "text_llm":        (False, True,  True),
    "all_on":          (True,  True,  True),
}


# ── the child process: one arm, one run ─────────────────────────────────────

def _run_one_arm(args) -> dict:
    """Executed in the child. Imports the app fresh under this arm's env."""
    sys.path.insert(0, str(REPO))
    from experiments.harness import load_app, run_headless

    app = load_app(
        stub_support=os.getenv("ABLATION_STUB_SUPPORT_AGENTS") == "1")
    try:
        summary, elapsed = run_headless(
            app, start=args.start, end=args.end, symbol=args.symbol,
            interval=args.interval, capital=args.capital,
            strategy=args.strategy,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    pos = summary.get("positioning_report", {}) or {}
    txt = summary.get("text_sentiment_report", {}) or {}
    llm = summary.get("llm_report", {}) or {}
    trades = summary.get("trades", []) or []
    returns = [t.get("profit_pct") for t in trades
               if t.get("profit_pct") is not None]

    # The equity curve, not just the headline return: significance testing needs
    # the per-period series, and two arms run over the SAME window produce
    # aligned series that can be compared pairwise. Without this every arm is a
    # single number and no interval can be put around the difference.
    curve = summary.get("daily_values", []) or []
    equity = [
        {"timestamp": str(row.get("timestamp")),
         "portfolio_value": row.get("portfolio_value")}
        for row in curve if isinstance(row, dict)
    ]

    from core.config import describe_configuration
    return {
        "total_return_pct": summary.get("total_return_pct"),
        "sharpe_ratio": summary.get("sharpe_ratio"),
        "sortino_ratio": summary.get("sortino_ratio"),
        "max_drawdown_pct": summary.get("max_drawdown_pct"),
        "calmar_ratio": summary.get("calmar_ratio"),
        "observations": summary.get("observations"),
        "equity_curve": equity,
        "buy_hold_return_pct": summary.get("buy_hold_return_pct"),
        "excess_return_pct": (
            None if summary.get("total_return_pct") is None
            else round(summary["total_return_pct"]
                       - (summary.get("buy_hold_return_pct") or 0.0), 6)
        ),
        "final_value": summary.get("final_value"),
        "total_trades": summary.get("total_trades"),
        "winning_trades": summary.get("winning_trades"),
        "win_rate_pct": summary.get("win_rate_pct"),
        "decisions": summary.get("decisions"),
        "closed_trade_returns": returns,
        "positioning_moved": pos.get("decisions_where_points_added"),
        "positioning_coverage_pct": (pos.get("agent") or {}).get("coverage_pct"),
        "text_moved": txt.get("decisions_where_points_added"),
        "text_coverage_pct": (txt.get("agent") or {}).get("coverage_pct"),
        "llm_changed": llm.get("decisions_changed_by_llm"),
        "llm_calls": llm.get("llm_calls"),
        "llm_failures": llm.get("llm_failures"),
        "elapsed_s": round(elapsed, 2),
        "configuration": describe_configuration(),
    }


# ── the parent process: fan out over arms ───────────────────────────────────

def _child_env(positioning: bool, text: bool, llm: bool,
               stub_support: bool) -> dict:
    env = dict(os.environ)
    env.update({
        "USE_POSITIONING_SIGNAL": "true" if positioning else "false",
        "USE_TEXT_SENTIMENT": "true" if text else "false",
        "USE_LLM_DECISIONS": "true" if llm else "false",
        # Frozen caches: an ablation must not have one arm silently fetch data
        # another arm did not see.
        "POSITIONING_AUTO_FETCH": "false",
        "TEXT_SENTIMENT_AUTO_FETCH": "false",
        "ABLATION_STUB_SUPPORT_AGENTS": "1" if stub_support else "0",
        "PYTHONIOENCODING": "utf-8",
    })
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2024-02-01")
    parser.add_argument("--end", default="2024-03-20")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--baseline", default="rules_only",
                        help="arm every other arm is tested against "
                             "(default: rules_only, the model-free control)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="familywise error rate for the Holm correction")
    parser.add_argument("--window-days", type=int, default=None,
                        help="walk-forward: cut windows of this many days out "
                             "of the cached coverage instead of using "
                             "--start/--end. Windows are disjoint by default.")
    parser.add_argument("--window-step-days", type=int, default=None,
                        help="stride between window starts (default: "
                             "--window-days, i.e. disjoint). Overlapping "
                             "windows share bars and are NOT independent "
                             "evidence; the bootstrap will overstate them.")
    parser.add_argument("--min-window-days", type=int, default=None,
                        help="discard trailing windows shorter than this "
                             "(default: --window-days)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap the plan, for a cheap smoke test. Takes the "
                             "EARLIEST windows; use --window-from to choose "
                             "which stretch of history you want instead.")
    parser.add_argument("--window-from", default=None,
                        help="ignore cached days before this date (ISO) when "
                             "building the window plan, e.g. 2026-01-01 to run "
                             "only the recent regime")
    parser.add_argument("--window-to", default=None,
                        help="ignore cached days after this date (ISO)")
    parser.add_argument("--strategy", default="moderate",
                        choices=["conservative", "moderate", "aggressive"])
    parser.add_argument("--arms", default="all",
                        help=f"comma separated, or 'all'. Available: "
                             f"{','.join(ARMS)}")
    parser.add_argument("--repeats", type=int, default=1,
                        help="runs per arm; >1 only makes sense without "
                             "--stub-support-agents, since a stubbed run is "
                             "deterministic and every repeat is identical")
    parser.add_argument("--stub-support-agents", action="store_true",
                        help="replace the market/pattern/risk agents with fixed "
                             "values. The ONLY genuinely model-free and "
                             "reproducible configuration here")
    parser.add_argument("--out", default=None,
                        help="output JSONL (default experiments/results/<ts>.jsonl)")
    parser.add_argument("--child-arm", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Child mode
    if args.child_arm:
        print("<<<RESULT>>>" + json.dumps(_run_one_arm(args), default=str))
        return 0

    names = list(ARMS) if args.arms == "all" else [
        a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        parser.error(f"unknown arm(s): {unknown}. Available: {list(ARMS)}")

    out_path = Path(args.out) if args.out else (
        REPO / "experiments" / "results" /
        f"ablation-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Walk-forward: one (arm x window) run each, instead of one arm per run.
    # Disjoint windows by default -- overlapping ones share bars, so treating
    # their results as independent evidence overstates the sample.
    plan = _resolve_windows(args)

    total = len(names) * args.repeats * len(plan)
    print(f"{total} run(s): {len(names)} arm(s) x {args.repeats} repeat(s) "
          f"x {len(plan)} window(s)")
    if len(plan) == 1:
        print(f"window {plan[0][0]} .. {plan[0][1]}  symbol {args.symbol}  "
              f"strategy {args.strategy}")
    else:
        summary = describe_plan_for(plan)
        print(f"walk-forward: {summary}  symbol {args.symbol}  "
              f"strategy {args.strategy}")
        for wname, (ws, we, blk) in zip(_window_names(plan), plan):
            print(f"    {wname}  {ws} .. {we}  (regime block {blk})")
    print(f"support agents: {'STUBBED (reproducible)' if args.stub_support_agents else 'LIVE LLM (slow, not reproducible)'}")
    print(f"writing {out_path}\n")

    rows, failures = [], 0
    wnames = _window_names(plan)
    combos = list(itertools.product(range(len(plan)), names, range(args.repeats)))
    for index, (widx, name, repeat) in enumerate(combos, start=1):
        positioning, text, llm = ARMS[name]
        win_start, win_end, win_block = plan[widx]
        wname = wnames[widx]
        label = f"[{index}/{total}] {wname} {name}" + (
            f" run {repeat + 1}" if args.repeats > 1 else "")
        print(f"{label} ... ", end="", flush=True)

        command = [sys.executable, "-m", "experiments.run_ablation",
                   "--child-arm", name,
                   "--start", win_start, "--end", win_end,
                   "--symbol", args.symbol, "--interval", args.interval,
                   "--capital", str(args.capital), "--strategy", args.strategy]
        completed = subprocess.run(
            command, cwd=str(REPO),
            env=_child_env(positioning, text, llm, args.stub_support_agents),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )

        payload = None
        for line in (completed.stdout or "").splitlines():
            if line.startswith("<<<RESULT>>>"):
                payload = json.loads(line[len("<<<RESULT>>>"):])
        if payload is None or "error" in payload:
            failures += 1
            print("FAILED")
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()
            for line in tail[-6:]:
                print(f"      {line}")
            continue

        row = {
            "arm": name, "repeat": repeat,
            "positioning": positioning, "text": text, "llm": llm,
            "support_agents_stubbed": args.stub_support_agents,
            # Flat keys as well as the nested block: the walk-forward analysis
            # groups on these, and a regime tag is what lets the two coverage
            # blocks be reported separately instead of silently pooled across a
            # two-year gap.
            "window_name": wname,
            "window_start": win_start,
            "window_end": win_end,
            "regime_block": win_block,
            "window": {"start": win_start, "end": win_end,
                       "symbol": args.symbol, "interval": args.interval,
                       "strategy": args.strategy, "capital": args.capital},
            "ran_at_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(f"return {payload['total_return_pct']:+.2f}%  "
              f"trades {payload['total_trades']}  "
              f"pos_moved {payload['positioning_moved']}  "
              f"text_moved {payload['text_moved']}  "
              f"llm_changed {payload['llm_changed']}  "
              f"({payload['elapsed_s']}s)")

    if not rows:
        print("\nno arm completed", file=sys.stderr)
        return 1

    print(f"\n{'arm':<12} {'pos':>4} {'text':>5} {'llm':>4} "
          f"{'return%':>9} {'excess%':>9} {'trades':>7} {'win%':>6} "
          f"{'posMv':>6} {'txtMv':>6} {'llmCh':>6}")
    print("-" * 92)
    for row in rows:
        print(f"{row['arm']:<12} {str(row['positioning'])[:1]:>4} "
              f"{str(row['text'])[:1]:>5} {str(row['llm'])[:1]:>4} "
              f"{row['total_return_pct']:>9.2f} {row['excess_return_pct']:>9.2f} "
              f"{row['total_trades']:>7} {row['win_rate_pct']:>6.1f} "
              f"{str(row['positioning_moved']):>6} {str(row['text_moved']):>6} "
              f"{str(row['llm_changed']):>6}")

    bh = rows[0].get("buy_hold_return_pct")
    print(f"\nbuy and hold over the same window: {bh:+.2f}%")

    _print_walk_forward(rows, baseline=args.baseline, alpha=args.alpha)
    _print_significance(rows, baseline=args.baseline, alpha=args.alpha)

    print(f"\n{len(rows)} row(s) written to {out_path}")
    if failures:
        print(f"{failures} run(s) failed", file=sys.stderr)
    return 1 if failures else 0


def _resolve_windows(args) -> list:
    """The evaluation windows for this run, as (start, end, block) triples.

    Without --window-days this returns the single explicit window, so every
    existing invocation behaves exactly as before.
    """
    if not getattr(args, "window_days", None):
        return [(args.start, args.end, 0)]

    from experiments.windows import make_windows

    plan = make_windows(
        length_days=args.window_days,
        step_days=args.window_step_days,
        symbol=args.symbol.replace("/", ""),
        min_days=args.min_window_days,
        since=args.window_from,
        until=args.window_to,
    )
    if args.max_windows:
        plan = plan[:args.max_windows]
    if not plan:
        bounds = ""
        if args.window_from or args.window_to:
            bounds = (f" within {args.window_from or 'start'} .. "
                      f"{args.window_to or 'end'}")
        raise SystemExit(
            f"no window of {args.window_days} day(s) fits inside the cached "
            f"coverage for {args.symbol}{bounds}. Fetch more history, widen "
            f"--window-from/--window-to, or lower --window-days.")

    from core.metrics import MIN_STUDENTIZED_OBSERVATIONS
    if len(plan) < MIN_STUDENTIZED_OBSERVATIONS:
        print(f"WARNING: {len(plan)} window(s) is below the "
              f"{MIN_STUDENTIZED_OBSERVATIONS}-window floor for the "
              f"walk-forward test. The runs will complete and the per-window "
              f"table will be printed, but no across-window interval can be "
              f"estimated from this few.\n", file=sys.stderr)
    return [(w.start.isoformat(), w.end.isoformat(), w.block) for w in plan]


def _window_names(plan: list) -> list:
    return [f"w{i:02d}" for i in range(1, len(plan) + 1)]


def describe_plan_for(plan: list) -> str:
    blocks = {}
    for _, _, blk in plan:
        blocks[blk] = blocks.get(blk, 0) + 1
    return (f"{len(plan)} disjoint window(s) across {len(blocks)} regime "
            f"block(s) {blocks}")


def _print_walk_forward(rows: list, baseline: str = "rules_only",
                        alpha: float = 0.05) -> None:
    """Test each arm against the baseline ACROSS windows, not within one.

    This is the question a single window cannot answer. The unit of observation
    becomes the window: for each window, take the arm's return minus the
    baseline's return on that same window, then bootstrap that series of
    per-window differences.

    Pairing by window is what makes it powerful. Two arms run over the same
    window share the whole market movement, which dwarfs any signal effect;
    differencing removes it, leaving only what the arm actually changed.

    The result still describes the regimes sampled, not all regimes. With a
    handful of windows the honest reading of a wide interval is "not enough
    windows yet", not "no effect".
    """
    from core.metrics import holm_correction, studentized_bootstrap

    # (arm, window) -> return. Repeats on the same window are averaged, since
    # they measure LLM nondeterminism rather than independent evidence.
    grid: dict = {}
    regimes: dict = {}
    for row in rows:
        if row.get("error") or row.get("total_return_pct") is None:
            continue
        key = (row["arm"], row.get("window_name", "w01"))
        grid.setdefault(key, []).append(row["total_return_pct"])
        regimes[row.get("window_name", "w01")] = row.get("regime_block", 0)

    windows = sorted({w for _, w in grid})
    if len(windows) < 2:
        return  # single window: the within-window report already covered it

    def series(arm: str) -> dict:
        return {w: sum(v) / len(v)
                for (a, w), v in grid.items() if a == arm}

    base = series(baseline)
    if not base:
        print(f"\n[walk-forward] baseline '{baseline}' missing; skipping.",
              file=sys.stderr)
        return

    arms = sorted({a for a, _ in grid} - {baseline})
    diffs, p_values = {}, {}
    for arm in arms:
        got = series(arm)
        shared = [w for w in windows if w in got and w in base]
        if len(shared) < 2:
            p_values[arm] = None
            continue
        d = [got[w] - base[w] for w in shared]
        # Studentized, NOT percentile: with one observation per window the
        # sample is small, and the percentile bootstrap undercovers badly there
        # (measured: 14% false positives at 9 windows against a nominal 5%).
        res = studentized_bootstrap(d, n_resamples=10_000)
        # Fraction of windows where the arm beat the baseline: a sign test read,
        # robust to one window dominating the mean.
        wins = sum(1 for x in d if x > 0)
        diffs[arm] = {"res": res, "n": len(shared), "wins": wins,
                      "mean": sum(d) / len(d)}
        p_values[arm] = res.get("p_value")

    corrected = holm_correction(p_values, alpha=alpha)

    print(f"\nWALK-FORWARD vs '{baseline}' across {len(windows)} windows "
          f"(paired by window, Holm-corrected, alpha={alpha})")
    print(f"{'arm':<12} {'mean excess%':>13} {'windows won':>12} "
          f"{'95% CI':>22} {'p_adj':>8} {'verdict':>20}")
    print("-" * 92)
    for arm in sorted(diffs, key=lambda a: p_values.get(a) if
                      p_values.get(a) is not None else 1.0):
        d, adj = diffs[arm], corrected.get(arm, {})
        if d["res"].get("lo") is None:
            # Too few windows to estimate an interval. Print the descriptive
            # numbers and say so, rather than a fabricated range.
            print(f"{arm:<12} {d['mean']:>+13.2f} {d['wins']:>7}/{d['n']:<4} "
                  f"{'—':>22} {'—':>8} {'too few windows':>20}")
            continue
        ci = f"[{d['res']['lo']:+.2f}, {d['res']['hi']:+.2f}]"
        verdict = ("SIGNIFICANT" if adj.get("significant")
                   else "not distinguishable")
        print(f"{arm:<12} {d['mean']:>+13.2f} {d['wins']:>7}/{d['n']:<4} "
              f"{ci:>22} {(adj.get('p_adjusted') or float('nan')):>8.3f} "
              f"{verdict:>20}")

    if any(d["res"].get("lo") is None for d in diffs.values()):
        from core.metrics import MIN_STUDENTIZED_OBSERVATIONS
        print(f"\n  At least {MIN_STUDENTIZED_OBSERVATIONS} windows are needed "
              f"before a walk-forward interval means anything. Re-run with a "
              f"shorter --window-days, or fetch more history.")

    # Per-regime, because pooling two blocks two years apart hides exactly the
    # regime dependence that makes or breaks a claim like this.
    blocks = sorted(set(regimes.values()))
    if len(blocks) > 1:
        print(f"\nPER-REGIME mean excess return vs '{baseline}' (%)")
        header = "  ".join(f"block{b}" for b in blocks)
        print(f"{'arm':<12} {header}")
        for arm in arms:
            got, cells = series(arm), []
            for b in blocks:
                ws = [w for w in windows
                      if regimes.get(w) == b and w in got and w in base]
                cells.append(f"{sum(got[w]-base[w] for w in ws)/len(ws):+7.2f}"
                             if ws else "      —")
            print(f"{arm:<12} " + "  ".join(cells))


def _print_significance(rows: list, baseline: str = "rules_only",
                        alpha: float = 0.05) -> None:
    """Bootstrap every arm against the baseline, corrected for the family size.

    This is the section that decides whether the table above means anything.
    Three separate things are being controlled for, and dropping any one of
    them is a standard way for a result like this to fail review:

      * serial correlation -- a stationary bootstrap, not an i.i.d. one, or the
        intervals come out far too narrow;
      * multiple comparisons -- seven arms against a baseline is seven chances
        to find a spurious winner, so Holm caps the familywise error;
      * selection bias in the Sharpe -- the best of N arms is biased upward even
        when every arm is worthless, which the deflated Sharpe corrects.
    """
    from core.metrics import (MIN_OBSERVATIONS, bootstrap_difference,
                              deflated_sharpe_ratio, equity_returns,
                              holm_correction, periods_per_year)

    by_arm = {r["arm"]: r for r in rows if not r.get("error")}
    if baseline not in by_arm:
        print(f"\n[significance] baseline arm '{baseline}' not among the runs; "
              f"skipping. Available: {list(by_arm)}", file=sys.stderr)
        return

    base_returns = equity_returns(by_arm[baseline].get("equity_curve") or [])
    if base_returns.size < 2:
        print(f"\n[significance] baseline '{baseline}' has no usable equity "
              f"curve; skipping.", file=sys.stderr)
        return

    comparisons, p_values = {}, {}
    for name, row in by_arm.items():
        if name == baseline:
            continue
        rets = equity_returns(row.get("equity_curve") or [])
        if rets.size < 2:
            p_values[name] = None
            continue
        res = bootstrap_difference(rets, base_returns)
        comparisons[name] = res
        p_values[name] = res.get("p_value")

    corrected = holm_correction(p_values, alpha=alpha)

    print(f"\nSIGNIFICANCE vs '{baseline}' "
          f"(stationary bootstrap, Holm-corrected, alpha={alpha})")
    print(f"{'arm':<12} {'mean diff/period':>17} {'95% CI':>24} "
          f"{'p':>8} {'p_adj':>8} {'verdict':>12}")
    print("-" * 86)
    for name in sorted(comparisons, key=lambda k: p_values.get(k) or 1.0):
        res, adj = comparisons[name], corrected.get(name, {})
        ci = f"[{res['lo']:+.5f}, {res['hi']:+.5f}]"
        verdict = ("SIGNIFICANT" if adj.get("significant")
                   else "not distinguishable")
        print(f"{name:<12} {res['point']:>+17.6f} {ci:>24} "
              f"{(res['p_value'] or float('nan')):>8.3f} "
              f"{(adj.get('p_adjusted') or float('nan')):>8.3f} {verdict:>12}")

    for name, p in p_values.items():
        if p is None:
            print(f"{name:<12} {'—':>17} {'not evaluated (no curve)':>24}")

    # Deflated Sharpe on the best arm by return: the one a reader would quote.
    scored = [r for r in rows if not r.get("error")
              and r.get("total_return_pct") is not None]
    if scored:
        best = max(scored, key=lambda r: r["total_return_pct"])
        rets = equity_returns(best.get("equity_curve") or [])
        ppy = periods_per_year([c.get("timestamp")
                                for c in (best.get("equity_curve") or [])])
        dsr = deflated_sharpe_ratio(rets, n_trials=len(scored), ppy=ppy)
        print(f"\nDEFLATED SHARPE on the best arm by return ('{best['arm']}', "
              f"{len(scored)} trials)")
        if dsr.get("deflated_sharpe") is None:
            print(f"   not computed: {dsr.get('note')}")
        else:
            print(f"   annualised Sharpe {dsr['sharpe']:+.2f}  ->  "
                  f"DSR {dsr['deflated_sharpe']:.3f}  "
                  f"({'passes' if dsr['passes_at_95'] else 'FAILS'} at 0.95)")
            if not dsr["passes_at_95"]:
                print("   The best arm's Sharpe is not distinguishable from the "
                      "best of that many worthless strategies.")

    print(f"\nNOTE: 'excess%' is return minus buy-and-hold. The intervals above "
          f"are within-window: they test whether an arm differs from the "
          f"baseline on THIS window, not whether it generalises to others. "
          f"Runs with fewer than {MIN_OBSERVATIONS} observations report no "
          f"ratios at all, by design.")


if __name__ == "__main__":
    raise SystemExit(main())
