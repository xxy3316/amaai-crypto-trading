"""Walk-forward window generation for the ablation.

Until now the ablation ran on ONE window, and the bootstrap in `core.metrics`
was explicitly within-window: it tested whether an arm differed from the
baseline on that window, never whether the difference survived elsewhere. A
single window cannot separate skill from luck no matter how careful the
statistics on it are, and "does it hold out of sample?" is the first question
any reviewer asks.

This module answers a narrower question so the ablation can answer the wider
one: given what is actually cached on disk, which evaluation windows are
legitimately available?

Two rules shape every window it returns:

  * A window is only usable where BOTH exogenous channels have data. A window
    covering days the positioning cache is missing would quietly compare an arm
    that had the signal against one that did not, which is the confound the
    ablation exists to avoid.
  * Warmup is charged against the window, not ignored. Each window must be long
    enough to survive indicator warmup AND the positioning z-score baseline and
    still leave decisions behind, or it is discarded rather than silently
    producing an empty run.

Coverage is read from the caches themselves rather than configured, so a window
plan can never claim data that was not downloaded.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import glob
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

#: Default cache locations, relative to the repo root.
POSITIONING_GLOB = "data/exogenous/binance_positioning/{symbol}/*.zip"
TEXT_GLOB = "data/exogenous/text_sentiment/hackernews/*/*.json"


@dataclasses.dataclass(frozen=True)
class Window:
    """One evaluation window, tagged with the coverage block it came from."""

    start: dt.date
    end: dt.date
    block: int          #: index of the contiguous coverage run it sits in
    index: int          #: position within the whole plan, for stable naming

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def name(self) -> str:
        return f"w{self.index:02d}"

    def as_dict(self) -> dict:
        return {"window": self.name, "window_start": self.start.isoformat(),
                "window_end": self.end.isoformat(), "block": self.block,
                "window_days": self.days}


def _dates_from_glob(pattern: str) -> set:
    out = set()
    for path in glob.glob(str(REPO / pattern)):
        match = _DATE_RE.search(os.path.basename(path))
        if match:
            try:
                out.add(dt.date.fromisoformat(match.group(1)))
            except ValueError:
                continue
    return out


def cached_days(symbol: str = "BTCUSDT",
                require_positioning: bool = True,
                require_text: bool = True) -> List[dt.date]:
    """Days where every REQUIRED channel has a cached file, sorted.

    Intersected rather than unioned on purpose: an arm is only comparable with
    another if both saw the same days.
    """
    sets = []
    if require_positioning:
        sets.append(_dates_from_glob(POSITIONING_GLOB.format(symbol=symbol)))
    if require_text:
        sets.append(_dates_from_glob(TEXT_GLOB))
    if not sets:
        return []
    usable = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(usable)


def contiguous_blocks(days: Sequence[dt.date],
                      max_gap_days: int = 1) -> List[Tuple[dt.date, dt.date]]:
    """Runs of consecutive days, as (first, last) pairs.

    Blocks matter beyond bookkeeping: the cache here holds two stretches nearly
    two years apart, which are different market regimes. Pooling across the gap
    would hide exactly the regime dependence worth reporting.
    """
    if not days:
        return []
    blocks, start, prev = [], days[0], days[0]
    for day in days[1:]:
        if (day - prev).days > max_gap_days:
            blocks.append((start, prev))
            start = day
        prev = day
    blocks.append((start, prev))
    return blocks


def minimum_window_days(interval: str, warmup_bars: int,
                        zscore_window: int, min_decisions: int) -> int:
    """Days a window must span to yield `min_decisions` usable decisions.

    Charges warmup honestly. The positioning z-score baseline is the binding
    constraint on daily bars, where it reaches back a further 30 days; ignoring
    it is how a window ends up running with the channel silently dead.
    """
    from core.config import decision_step_bars, interval_hours

    bar_h = interval_hours(interval) or 1.0
    step = decision_step_bars(interval)
    bars_needed = warmup_bars + zscore_window + min_decisions * step
    return max(1, int((bars_needed * bar_h + 23) // 24))


def _as_date(value) -> Optional[dt.date]:
    if value is None or isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def make_windows(length_days: int, step_days: Optional[int] = None,
                 symbol: str = "BTCUSDT",
                 require_positioning: bool = True,
                 require_text: bool = True,
                 min_days: Optional[int] = None,
                 since=None, until=None) -> List[Window]:
    """Cut evenly spaced windows out of whatever coverage exists.

    `step_days` defaults to `length_days`, giving DISJOINT windows. Disjoint is
    the honest default: overlapping windows share bars, so their results are
    correlated, and treating them as independent observations in a bootstrap
    overstates the evidence -- the same error as an i.i.d. bootstrap on serially
    correlated returns, one level up.

    `since` / `until` clip the coverage before cutting, which is how a run is
    restricted to one regime block without paying for the others. Clipping is
    applied to the DAYS, not to the finished plan, so blocks are recomputed
    afterwards and no window can straddle a gap introduced by the clip.

    Block numbering stays tied to the clipped coverage, so a restricted run
    reports `block 0` for its own first block. The window dates in every row
    are what identify a regime across runs, not the block index.
    """
    step = step_days or length_days
    floor = min_days if min_days is not None else length_days
    days = cached_days(symbol, require_positioning, require_text)

    lo, hi = _as_date(since), _as_date(until)
    if lo is not None:
        days = [d for d in days if d >= lo]
    if hi is not None:
        days = [d for d in days if d <= hi]

    plan: List[Window] = []

    for block_id, (first, last) in enumerate(contiguous_blocks(days)):
        cursor = first
        while cursor <= last:
            end = min(cursor + dt.timedelta(days=length_days - 1), last)
            span = (end - cursor).days + 1
            if span >= floor:
                plan.append(Window(start=cursor, end=end, block=block_id,
                                   index=len(plan) + 1))
            if end >= last:
                break
            cursor = cursor + dt.timedelta(days=step)

    return plan


def describe_plan(plan: Iterable[Window]) -> Dict[str, object]:
    """Summary a run can print and a paper can quote."""
    windows = list(plan)
    by_block: Dict[int, int] = {}
    for w in windows:
        by_block[w.block] = by_block.get(w.block, 0) + 1
    return {
        "windows": len(windows),
        "blocks": len(by_block),
        "windows_per_block": by_block,
        "total_days": sum(w.days for w in windows),
        "span": (f"{min(w.start for w in windows)} .. "
                 f"{max(w.end for w in windows)}") if windows else None,
    }
