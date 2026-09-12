"""Risk-adjusted performance metrics and significance testing.

Phase 3 of the publication plan. Until now the app reported total return, win
rate and "beat the market", and nothing else: `summary['sharpe_ratio']` and
`summary['max_drawdown']` were READ by the UI but never WRITTEN anywhere, so
both displayed as a literal 0 and the panel that shows them was dead code.
A strategy cannot be judged on return alone -- doubling the position size
doubles the return and changes nothing about whether the signal is real.

Two separate jobs live here, and they answer different questions:

  1. Descriptive   -- how did this run behave? Sharpe, Sortino, max drawdown,
                      Calmar. These describe ONE run and prove nothing.
  2. Inferential   -- is the difference between two arms distinguishable from
                      luck? Bootstrap confidence intervals, Holm correction
                      across the arm family, and the deflated Sharpe ratio.

The second exists because the first invites over-reading. A single window with
eleven decisions can show a Sharpe of 2 purely by chance; the honest response is
an interval that contains zero, not a number in bold.

Everything here is deliberately dependency-light (numpy only) and pure: no
Streamlit, no config imports, no I/O. That keeps it usable from the app, from
`experiments/run_ablation.py`, and from tests, without any of them dragging in
the others.

References
----------
Politis & Romano (1994), "The Stationary Bootstrap" -- block resampling that
preserves serial correlation, which crypto returns have and an i.i.d. bootstrap
would silently destroy (producing intervals far too narrow).

Bailey & López de Prado (2014), "The Deflated Sharpe Ratio" -- corrects an
observed Sharpe for the number of configurations tried, and for the skew and
kurtosis of the return distribution. Running eight ablation arms IS a multiple
trial, so the best arm's Sharpe is upward-biased by selection alone.

Holm (1979) -- step-down familywise error control. Chosen over Benjamini-
Hochberg because the arm family is small and pre-specified, and because
controlling the familywise error rate is the more conservative claim to defend
in review.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Crypto trades every day of the year, unlike equities (252 sessions).
DAYS_PER_YEAR = 365.0
SECONDS_PER_YEAR = DAYS_PER_YEAR * 24 * 3600

#: Below this many return observations, the descriptive ratios are reported as
#: None rather than as a number. A Sharpe computed from a handful of points is
#: not a small-sample estimate of anything -- its sampling distribution is so
#: wide that quoting it is worse than staying silent. The threshold is a stated
#: convention, not a derived one.
MIN_OBSERVATIONS = 20

#: Fewest observations the studentized bootstrap will accept. A walk-forward
#: supplies one per window, so this is really "fewest windows worth testing".
#: Below it the resampled standard error collapses on near-duplicate draws and
#: the interval explodes; refusing is the only honest option.
MIN_STUDENTIZED_OBSERVATIONS = 5

#: Resamples whose standard error falls below this fraction of the observed one
#: are treated as degenerate and discarded. They are duplicate-heavy draws that
#: contribute nothing but enormous t values.
DEGENERATE_SE_FRACTION = 0.10


# ── equity curve -> returns ─────────────────────────────────────────────────

def _to_timestamp(value: Any) -> Optional[float]:
    """Best-effort epoch seconds from whatever the app put in the curve."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dt.datetime):
        return value.timestamp()
    try:  # pandas.Timestamp, numpy.datetime64, ISO strings
        import pandas as pd
        ts = pd.Timestamp(value)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.to_pydatetime().timestamp()
    except Exception:
        return None


def periods_per_year(timestamps: Sequence[Any]) -> Optional[float]:
    """Annualisation factor inferred from the curve's own spacing.

    Inferred rather than passed in: the equity curve is sampled at DECISION
    points, not at bar boundaries, so the bar interval is the wrong scale and
    using it would misannualise every ratio by the decision step. The median
    gap is used because a single missing sample must not rescale everything.
    """
    stamps = [t for t in (_to_timestamp(t) for t in timestamps) if t is not None]
    if len(stamps) < 2:
        return None
    gaps = np.diff(np.asarray(sorted(stamps), dtype=float))
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return None
    return SECONDS_PER_YEAR / float(np.median(gaps))


def equity_returns(curve: Iterable[Dict[str, Any]],
                   value_key: str = "portfolio_value") -> np.ndarray:
    """Simple per-period returns from the app's `daily_values` structure.

    Non-positive or missing values end the usable series rather than producing
    an infinity: a portfolio that reached zero has no further return to report,
    and letting a division by zero through would poison every downstream ratio
    with NaN.
    """
    values: List[float] = []
    for row in curve or []:
        v = row.get(value_key) if isinstance(row, dict) else row
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        values.append(fv)

    if len(values) < 2:
        return np.asarray([], dtype=float)

    arr = np.asarray(values, dtype=float)
    prev, curr = arr[:-1], arr[1:]
    usable = prev > 0
    return np.where(usable, (curr - prev) / np.where(usable, prev, 1.0), np.nan)[usable]


# ── descriptive ratios ──────────────────────────────────────────────────────

def sharpe_ratio(returns: Sequence[float], ppy: Optional[float] = None,
                 risk_free_rate: float = 0.0) -> Optional[float]:
    """Annualised Sharpe. None when the sample is too small to mean anything."""
    r = np.asarray([x for x in returns if x is not None and math.isfinite(x)],
                   dtype=float)
    if r.size < MIN_OBSERVATIONS:
        return None
    excess = r - (risk_free_rate / (ppy or 1.0))
    sd = float(np.std(excess, ddof=1))
    if sd <= 0:
        return None
    ratio = float(np.mean(excess)) / sd
    return float(ratio * math.sqrt(ppy)) if ppy else float(ratio)


def sortino_ratio(returns: Sequence[float], ppy: Optional[float] = None,
                  target: float = 0.0) -> Optional[float]:
    """Annualised Sortino: like Sharpe but penalising only downside deviation.

    Upside volatility is not risk, and a strategy with occasional large gains is
    punished by Sharpe for exactly the behaviour one wants.
    """
    r = np.asarray([x for x in returns if x is not None and math.isfinite(x)],
                   dtype=float)
    if r.size < MIN_OBSERVATIONS:
        return None
    downside = np.minimum(r - target, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd <= 0:
        return None  # no losing period at all: the ratio is undefined, not huge
    ratio = float(np.mean(r - target)) / dd
    return float(ratio * math.sqrt(ppy)) if ppy else float(ratio)


def max_drawdown_pct(curve: Iterable[Dict[str, Any]],
                     value_key: str = "portfolio_value") -> Optional[float]:
    """Largest peak-to-trough fall in the equity curve, as a positive percent.

    Unlike the ratios this needs no minimum sample: a drawdown is an observed
    fact about the path, not an estimate of a parameter.
    """
    values = []
    for row in curve or []:
        v = row.get(value_key) if isinstance(row, dict) else row
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            values.append(fv)
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    running_peak = np.maximum.accumulate(arr)
    safe = running_peak > 0
    if not safe.any():
        return None
    drawdowns = np.zeros_like(arr)
    drawdowns[safe] = (arr[safe] - running_peak[safe]) / running_peak[safe]
    return float(-drawdowns.min() * 100.0)


def calmar_ratio(total_return_pct: Optional[float],
                 max_dd_pct: Optional[float]) -> Optional[float]:
    """Return per unit of worst drawdown. None when there was no drawdown."""
    if total_return_pct is None or not max_dd_pct:
        return None
    return float(total_return_pct / max_dd_pct)


def describe_run(curve: Iterable[Dict[str, Any]],
                 total_return_pct: Optional[float] = None,
                 risk_free_rate: float = 0.0) -> Dict[str, Any]:
    """Every descriptive metric for one run, in one call.

    Returns None for any ratio the sample cannot support, and says so in
    `sample_warning`, so a caller can render "not enough data" rather than a
    confident-looking zero -- which is exactly what the old code did.
    """
    rows = list(curve or [])
    stamps = [r.get("timestamp") for r in rows if isinstance(r, dict)]
    ppy = periods_per_year(stamps)
    rets = equity_returns(rows)
    dd = max_drawdown_pct(rows)

    enough = rets.size >= MIN_OBSERVATIONS
    return {
        "observations": int(rets.size),
        "periods_per_year": round(ppy, 2) if ppy else None,
        "sharpe_ratio": sharpe_ratio(rets, ppy, risk_free_rate),
        "sortino_ratio": sortino_ratio(rets, ppy),
        "max_drawdown_pct": dd,
        "calmar_ratio": calmar_ratio(total_return_pct, dd),
        "volatility_annual_pct": (
            float(np.std(rets, ddof=1) * math.sqrt(ppy) * 100.0)
            if enough and ppy and rets.size > 1 else None
        ),
        "sample_warning": (
            None if enough else
            f"{rets.size} return observations is below the {MIN_OBSERVATIONS}"
            f"-observation floor; risk-adjusted ratios are not reported because"
            f" their sampling error would exceed any plausible signal."
        ),
    }


# ── inferential: the stationary bootstrap ───────────────────────────────────

def _stationary_bootstrap_indices(n: int, mean_block: float,
                                  rng: np.random.Generator) -> np.ndarray:
    """One resample's worth of indices, wrapping at the end of the series.

    Block lengths are geometric with mean `mean_block`, which is what makes the
    resample stationary: a fixed block length would impose a periodicity the
    original series does not have.
    """
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=np.int64)
    current = rng.integers(0, n)
    for i in range(n):
        idx[i] = current
        if rng.random() < p:
            current = int(rng.integers(0, n))
        else:
            current = (current + 1) % n
    return idx


def _default_block(n: int) -> float:
    """Politis-White style n^(1/3) rule of thumb, floored at 1."""
    return max(1.0, float(n) ** (1.0 / 3.0))


def bootstrap_ci(sample: Sequence[float],
                 statistic=np.mean,
                 n_resamples: int = 10_000,
                 confidence: float = 0.95,
                 mean_block: Optional[float] = None,
                 seed: Optional[int] = 12345) -> Dict[str, Any]:
    """Percentile CI for `statistic` under the stationary bootstrap.

    The i.i.d. bootstrap is deliberately NOT the default. Strategy returns are
    serially correlated (trends, momentum, position persistence), and resampling
    them independently destroys that structure, yielding intervals that are too
    narrow and p-values that are too small -- the exact direction that
    manufactures false findings.

    `seed` is fixed by default so a reported interval is reproducible; pass None
    for a genuinely random resample.
    """
    x = np.asarray([v for v in sample if v is not None and math.isfinite(v)],
                   dtype=float)
    out: Dict[str, Any] = {
        "n": int(x.size), "point": None, "lo": None, "hi": None,
        "confidence": confidence, "n_resamples": int(n_resamples),
    }
    if x.size < 2:
        out["note"] = "fewer than 2 usable observations"
        return out

    block = mean_block if mean_block is not None else _default_block(x.size)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        stats[i] = statistic(x[_stationary_bootstrap_indices(x.size, block, rng)])

    alpha = (1.0 - confidence) / 2.0
    out.update({
        "point": float(statistic(x)),
        "lo": float(np.quantile(stats, alpha)),
        "hi": float(np.quantile(stats, 1.0 - alpha)),
        "mean_block": round(block, 3),
    })
    # The headline question in one field: does the interval exclude zero?
    out["excludes_zero"] = bool(out["lo"] > 0 or out["hi"] < 0)
    return out


def studentized_bootstrap(sample: Sequence[float],
                          n_resamples: int = 10_000,
                          confidence: float = 0.95,
                          seed: Optional[int] = 12345) -> Dict[str, Any]:
    """Bootstrap-t interval and p-value for the mean. Use this when n is small.

    The plain percentile bootstrap is badly anti-conservative on small samples:
    measured against a true null it rejected at 23% with 5 observations and 14%
    with 9, against a nominal 5%. A walk-forward has exactly that many
    observations -- one per window -- so using the percentile interval there
    would call noise significant several times more often than advertised, and
    across seven arms that is close to a guaranteed false finding.

    The fix is to studentise: resample the t-STATISTIC rather than the mean, so
    the interval widens when the resampled spread is itself uncertain. That is
    the one correction that restores coverage without assuming normality.

    Returns the same shape as `bootstrap_ci` so callers can swap between them.
    """
    x = np.asarray([v for v in sample if v is not None and math.isfinite(v)],
                   dtype=float)
    out: Dict[str, Any] = {
        "n": int(x.size), "point": None, "lo": None, "hi": None,
        "confidence": confidence, "n_resamples": int(n_resamples),
        "method": "studentized",
    }
    if x.size < MIN_STUDENTIZED_OBSERVATIONS:
        out["note"] = (
            f"{x.size} observation(s); the studentized bootstrap needs at "
            f"least {MIN_STUDENTIZED_OBSERVATIONS} to be meaningful. Below "
            f"that, resamples frequently contain near-duplicate values, the "
            f"resampled standard error collapses toward zero, and the t "
            f"quantiles explode -- yielding intervals like [-4, +3e15]. The "
            f"honest answer at this sample size is 'not enough observations', "
            f"not a number.")
        return out

    n = x.size
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1)) / math.sqrt(n)
    if se <= 0:
        out.update({"point": mean, "lo": mean, "hi": mean,
                    "excludes_zero": bool(mean != 0.0), "p_value": None,
                    "note": "zero variance"})
        return out

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    draws = x[idx]
    means = draws.mean(axis=1)
    ses = draws.std(axis=1, ddof=1) / math.sqrt(n)
    # Resamples with no spread would divide by zero; they carry no information
    # about the t distribution, so they are dropped rather than clamped.
    # A resample whose spread has all but collapsed carries no information
    # about the t distribution, but it does produce an enormous t and drags the
    # quantile with it. Dropping on `> 0` is not enough: near-duplicates give a
    # tiny-but-positive standard error and the same blow-up. Requiring a
    # reasonable fraction of the observed spread removes them as a class.
    ok = ses > (DEGENERATE_SE_FRACTION * se)
    if ok.sum() < 100:
        out["note"] = ("too many degenerate resamples; the sample is close to "
                       "constant and no interval can be estimated")
        return out
    t_star = (means[ok] - mean) / ses[ok]

    alpha = (1.0 - confidence) / 2.0
    t_hi = float(np.quantile(t_star, 1.0 - alpha))
    t_lo = float(np.quantile(t_star, alpha))
    # Note the crossed subtraction: the upper bound uses the LOWER t quantile.
    out.update({
        "point": mean,
        "lo": mean - t_hi * se,
        "hi": mean - t_lo * se,
    })
    out["excludes_zero"] = bool(out["lo"] > 0 or out["hi"] < 0)
    observed_t = mean / se
    out["p_value"] = float((np.sum(np.abs(t_star) >= abs(observed_t)) + 1)
                           / (t_star.size + 1))
    return out


def bootstrap_difference(treatment: Sequence[float],
                         control: Sequence[float],
                         n_resamples: int = 10_000,
                         confidence: float = 0.95,
                         seed: Optional[int] = 12345) -> Dict[str, Any]:
    """CI and two-sided p-value for mean(treatment) - mean(control).

    Paired element-wise when the two series are the same length, which is the
    case that matters here: two ablation arms run over the SAME window produce
    aligned per-period returns, and pairing removes the shared market movement
    that would otherwise dominate the variance. Falls back to an unpaired
    difference of independent resamples when the lengths differ.
    """
    a = np.asarray([v for v in treatment if v is not None and math.isfinite(v)],
                   dtype=float)
    b = np.asarray([v for v in control if v is not None and math.isfinite(v)],
                   dtype=float)
    if a.size < 2 or b.size < 2:
        return {"n_treatment": int(a.size), "n_control": int(b.size),
                "point": None, "lo": None, "hi": None, "p_value": None,
                "note": "fewer than 2 usable observations in an arm"}

    paired = a.size == b.size
    if paired:
        result = bootstrap_ci(a - b, np.mean, n_resamples, confidence, seed=seed)
        diffs_point = float(np.mean(a - b))
    else:
        rng = np.random.default_rng(seed)
        ba, bb = _default_block(a.size), _default_block(b.size)
        stats = np.empty(n_resamples, dtype=float)
        for i in range(n_resamples):
            stats[i] = (np.mean(a[_stationary_bootstrap_indices(a.size, ba, rng)])
                        - np.mean(b[_stationary_bootstrap_indices(b.size, bb, rng)]))
        alpha = (1.0 - confidence) / 2.0
        diffs_point = float(np.mean(a) - np.mean(b))
        result = {
            "n": int(min(a.size, b.size)), "point": diffs_point,
            "lo": float(np.quantile(stats, alpha)),
            "hi": float(np.quantile(stats, 1.0 - alpha)),
            "confidence": confidence, "n_resamples": int(n_resamples),
        }
        result["excludes_zero"] = bool(result["lo"] > 0 or result["hi"] < 0)

    # Two-sided bootstrap p-value: the proportion of resampled differences at
    # least as extreme as zero, centred on the observed effect. The +1s are
    # Davison & Hinkley's correction, which keeps p strictly positive -- a
    # reported p of exactly 0 is never an honest summary of a finite resample.
    centred_extreme = None
    if result.get("lo") is not None:
        rng = np.random.default_rng(seed)
        if paired:
            d = a - b
            block = _default_block(d.size)
            stats = np.empty(n_resamples, dtype=float)
            for i in range(n_resamples):
                stats[i] = np.mean(d[_stationary_bootstrap_indices(d.size, block, rng)])
        centred = stats - np.mean(stats)
        centred_extreme = float(
            (np.sum(np.abs(centred) >= abs(diffs_point)) + 1) / (n_resamples + 1))

    return {
        "n_treatment": int(a.size), "n_control": int(b.size),
        "paired": paired,
        "point": result.get("point"),
        "lo": result.get("lo"), "hi": result.get("hi"),
        "confidence": confidence,
        "excludes_zero": result.get("excludes_zero"),
        "p_value": centred_extreme,
    }


# ── inferential: multiple comparisons ───────────────────────────────────────

def holm_correction(p_values: Dict[str, Optional[float]],
                    alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Holm step-down familywise correction over a family of comparisons.

    Eight ablation arms compared against one baseline is eight chances to find
    a spurious winner; at alpha=0.05 the probability of at least one false
    positive is about 1 - 0.95^8 = 34%. Reporting uncorrected p-values from a
    factorial ablation is the single most common way a result like this fails
    review.

    Comparisons whose p-value is None (an arm that could not be evaluated) are
    carried through as None and excluded from the family size, so a failed run
    cannot make the surviving comparisons look more significant.
    """
    usable = {k: v for k, v in p_values.items()
              if v is not None and math.isfinite(v)}
    m = len(usable)
    out: Dict[str, Dict[str, Any]] = {
        k: {"p_value": None, "p_adjusted": None, "significant": None,
            "note": "not evaluated"}
        for k, v in p_values.items() if k not in usable
    }
    if m == 0:
        return out

    ordered = sorted(usable.items(), key=lambda kv: kv[1])
    running_max = 0.0
    for rank, (key, p) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * p)
        # Step-down enforces monotonicity: an adjusted p can never fall below
        # one computed for a smaller raw p.
        running_max = max(running_max, adjusted)
        out[key] = {
            "p_value": float(p),
            "p_adjusted": float(running_max),
            "significant": bool(running_max <= alpha),
            "rank": rank + 1,
            "family_size": m,
        }
    return out


# ── inferential: deflated Sharpe ────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 accurate)."""
    if not 0.0 < p < 1.0:
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def deflated_sharpe_ratio(returns: Sequence[float], n_trials: int,
                          ppy: Optional[float] = None,
                          trial_sharpe_std: Optional[float] = None) -> Dict[str, Any]:
    """Probability the observed Sharpe exceeds what selection alone would give.

    Bailey & López de Prado (2014). Running an eight-arm ablation and reporting
    the best arm's Sharpe is a selection procedure: the maximum of eight noisy
    estimates is biased upward even when every arm is worthless. This computes
    the Sharpe that the BEST of `n_trials` worthless strategies would be
    expected to show, and asks whether the observed one beats it.

    A DSR below 0.95 means the observed Sharpe is not distinguishable from the
    best of that many coin flips -- regardless of how large it looks.
    """
    r = np.asarray([x for x in returns if x is not None and math.isfinite(x)],
                   dtype=float)
    out: Dict[str, Any] = {"n": int(r.size), "n_trials": int(n_trials),
                           "sharpe": None, "expected_max_sharpe": None,
                           "deflated_sharpe": None}
    if r.size < MIN_OBSERVATIONS or n_trials < 1:
        out["note"] = (f"needs at least {MIN_OBSERVATIONS} observations and one "
                       f"trial; got {r.size} and {n_trials}")
        return out

    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        out["note"] = "zero variance"
        return out

    sr = float(np.mean(r)) / sd              # per-period, NOT annualised
    n = r.size
    centred = r - np.mean(r)
    skew = float(np.mean(centred ** 3) / sd ** 3)
    kurt = float(np.mean(centred ** 4) / sd ** 4)

    # Expected maximum Sharpe across n_trials independent worthless strategies.
    #
    # The bracketed term is in units of STANDARD DEVIATIONS OF THE SHARPE
    # ESTIMATOR, not in Sharpe units, so it must be scaled by that standard
    # deviation to be comparable with the observed Sharpe. Omitting the scale
    # makes the benchmark absurdly high (~1.46 per period for eight trials,
    # when a per-period Sharpe of 0.4 is already excellent), and then every
    # strategy fails no matter how good -- which is exactly as useless as every
    # strategy passing. Absent the actual spread of Sharpes across the trials,
    # the estimator's asymptotic standard deviation 1/sqrt(n-1) is the standard
    # substitute; pass `trial_sharpe_std` when the real spread is known.
    euler = 0.5772156649015329
    sr_std = (trial_sharpe_std if trial_sharpe_std is not None
              else 1.0 / math.sqrt(max(1, n - 1)))
    if n_trials == 1:
        expected_max = 0.0
    else:
        z1 = _norm_ppf(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        expected_max = sr_std * ((1 - euler) * z1 + euler * z2)

    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    dsr = _norm_cdf(((sr - expected_max) * math.sqrt(n - 1)) / denom)

    out.update({
        "sharpe": float(sr * math.sqrt(ppy)) if ppy else sr,
        "sharpe_per_period": sr,
        "expected_max_sharpe_per_period": float(expected_max),
        "skew": skew, "kurtosis": kurt,
        "deflated_sharpe": float(dsr),
        "passes_at_95": bool(dsr >= 0.95),
    })
    return out
