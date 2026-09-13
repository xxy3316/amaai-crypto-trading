"""Turn ablation result files into the tables a paper reports.

Reads the JSONL written by `run_ablation.py` and emits Markdown. Kept separate
from the runner so tables can be regenerated, reformatted or re-checked without
re-running an experiment -- which matters when the LLM arms cost money.

Every number here is computed from the result files. Nothing is typed in by
hand, so a table can always be traced back to the run that produced it.

    python -m experiments.make_tables <results.jsonl> [more.jsonl ...]
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.metrics import (MIN_STUDENTIZED_OBSERVATIONS, holm_correction,
                          studentized_bootstrap)


def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def by_arm(rows: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = collections.defaultdict(list)
    for row in rows:
        if not row.get("error") and row.get("total_return_pct") is not None:
            out[row["arm"]].append(row)
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def _fmt(value, spec="{:+.2f}"):
    return spec.format(value) if isinstance(value, (int, float)) else "—"


def descriptive_table(rows: List[dict], order: List[str]) -> str:
    """Per-arm behaviour. Descriptive only -- proves nothing on its own."""
    groups = by_arm(rows)
    lines = ["| Arm | Windows | Mean return | Median | Mean Sharpe | "
             "Mean max DD | Trades/window | Win rate |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in order:
        runs = groups.get(arm)
        if not runs:
            continue
        rets = [r["total_return_pct"] for r in runs]
        lines.append(
            f"| `{arm}` | {len(runs)} | {_fmt(st.mean(rets))}% | "
            f"{_fmt(st.median(rets))}% | "
            f"{_fmt(_mean([r.get('sharpe_ratio') for r in runs]))} | "
            f"{_fmt(_mean([r.get('max_drawdown_pct') for r in runs]), '{:.2f}')}% | "
            f"{_fmt(_mean([r.get('total_trades') for r in runs]), '{:.1f}')} | "
            f"{_fmt(_mean([r.get('win_rate_pct') for r in runs]), '{:.1f}')}% |")
    return "\n".join(lines)


def walk_forward_table(rows: List[dict], order: List[str],
                       baseline: str = "rules_only", alpha: float = 0.05) -> str:
    """Paired across-window test. This is the table that carries a claim.

    Pairing by window removes the market movement both arms saw, which
    otherwise dwarfs any difference between them.
    """
    groups = by_arm(rows)
    series = {arm: {r["window_name"]: r["total_return_pct"] for r in runs}
              for arm, runs in groups.items()}
    base = series.get(baseline, {})
    if not base:
        return f"_baseline `{baseline}` not present in this file._"

    diffs, p_values = {}, {}
    for arm in order:
        if arm == baseline or arm not in series:
            continue
        shared = sorted(set(series[arm]) & set(base))
        if len(shared) < 2:
            p_values[arm] = None
            continue
        d = [series[arm][w] - base[w] for w in shared]
        res = studentized_bootstrap(d)
        diffs[arm] = {"res": res, "n": len(shared), "mean": st.mean(d),
                      "wins": sum(1 for x in d if x > 0)}
        p_values[arm] = res.get("p_value")

    corrected = holm_correction(p_values, alpha=alpha)
    lines = [f"| Arm vs `{baseline}` | Windows | Mean excess | Won | "
             f"95% CI | p (Holm) | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for arm in sorted(diffs, key=lambda a: p_values.get(a) if
                      p_values.get(a) is not None else 1.0):
        d, adj = diffs[arm], corrected.get(arm, {})
        if d["res"].get("lo") is None:
            lines.append(f"| `{arm}` | {d['n']} | {_fmt(d['mean'])}% | "
                         f"{d['wins']}/{d['n']} | — | — | too few windows |")
            continue
        ci = f"[{d['res']['lo']:+.2f}, {d['res']['hi']:+.2f}]"
        verdict = ("**significant**" if adj.get("significant")
                   else "not distinguishable")
        lines.append(
            f"| `{arm}` | {d['n']} | {_fmt(d['mean'])}% | {d['wins']}/{d['n']} | "
            f"{ci} | {adj.get('p_adjusted', float('nan')):.3f} | {verdict} |")
    return "\n".join(lines)


def intervention_table(rows: List[dict], order: List[str]) -> str:
    """How often each arm's LLM actually altered a trade.

    Reported separately from returns because it is the one quantity in these
    experiments measured precisely enough to carry a claim on its own.
    """
    groups = by_arm(rows)
    if not any(r.get("llm_calls") for runs in groups.values() for r in runs):
        return ""
    lines = ["| Arm | Decisions | LLM calls | Failures | Changed | Rate |",
             "|---|---|---|---|---|---|"]
    for arm in order:
        runs = groups.get(arm)
        if not runs:
            continue
        dec = sum(r.get("decisions") or 0 for r in runs)
        calls = sum(r.get("llm_calls") or 0 for r in runs)
        fails = sum(r.get("llm_failures") or 0 for r in runs)
        changed = sum(r.get("llm_changed") or 0 for r in runs)
        rate = f"{100 * changed / dec:.1f}%" if dec else "—"
        lines.append(f"| `{arm}` | {dec} | {calls} | {fails} | {changed} | "
                     f"**{rate}** |")
    return "\n".join(lines)


def per_window_table(rows: List[dict], order: List[str]) -> str:
    """Every window, so a reader can see the spread behind the means."""
    groups = by_arm(rows)
    arms = [a for a in order if a in groups]
    windows = sorted({r["window_name"] for runs in groups.values() for r in runs})
    header = "| Window | Dates | " + " | ".join(f"`{a}`" for a in arms) + " |"
    lines = [header, "|" + "---|" * (len(arms) + 2)]
    for w in windows:
        dates, cells = "", []
        for arm in arms:
            match = [r for r in groups[arm] if r["window_name"] == w]
            dates = dates or (f"{match[0]['window_start']} → "
                              f"{match[0]['window_end']}" if match else "")
            cells.append(_fmt(match[0]["total_return_pct"]) + "%" if match else "—")
        lines.append(f"| {w} | {dates} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def describe_file(path: Path, order: List[str] = None) -> str:
    rows = load(path)
    groups = by_arm(rows)
    order = order or sorted(groups)
    windows = {r["window_name"] for r in rows if r.get("window_name")}
    failed = sum(1 for r in rows if r.get("error"))
    bh = _mean([r.get("buy_hold_return_pct") for r in rows])

    out = [f"### `{path.name}`", "",
           f"{len(rows)} runs · {len(groups)} arms · {len(windows)} windows"
           + (f" · **{failed} failed**" if failed else ""),
           f"Buy and hold over the same windows: **{_fmt(bh)}%**", "",
           "**Descriptive** (one run per arm per window)", "",
           descriptive_table(rows, order), ""]
    interv = intervention_table(rows, order)
    if interv:
        out += ["**LLM intervention**", "", interv, ""]
    out += ["**Walk-forward, paired by window** "
            f"(studentized bootstrap, Holm-corrected, α=0.05; "
            f"needs ≥{MIN_STUDENTIZED_OBSERVATIONS} windows)", "",
            walk_forward_table(rows, order), "",
            "**Per-window returns**", "", per_window_table(rows, order), ""]
    return "\n".join(out)


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        results = REPO / "experiments" / "results"
        paths = sorted(results.glob("*.jsonl"),
                       key=lambda p: p.stat().st_mtime)[-2:]
    for path in paths:
        print(describe_file(path))
        print()
