"""Per-decision record of what the LLM was shown and what it returned.

`LLMContribution` already records what the model's answer DID -- the adjustment,
the stance, whether the action changed. What it has never recorded is what the
model was asked, what it replied, and what inputs were on the table at the time.
Without those, a whole class of question about the model cannot be answered at
all:

  * Does the adjustment carry information beyond the rule features, or is it
    predictable from them? (If predictable, the model is an expensive re-encoding
    of the rule engine.)
  * Does the stated stance match the action taken? A `bearish` stance beside an
    adjustment of zero has already been observed.
  * Is the reported confidence calibrated against outcomes?
  * Does the model respond to the exogenous channels it is given, and does it
    distinguish the useful one from the harmful one?
  * Is the model deterministic at temperature 0 across reruns?

Each of those is measured precisely, unlike excess return, and none of them
requires the strategy to be profitable. That matters here, because the rule
baseline sits at the 56th percentile of random -- return-based questions are
measuring noise, while these are not.

Design constraints, in order of importance:

1. It must not change any decision. The logger only observes; if it raises, the
   backtest continues and the failure is counted. The golden backtest is the
   check on that.
2. It is OFF unless DECISION_LOG_PATH is set, so existing runs and the ablation
   are untouched and pay nothing.
3. One JSON object per line, appended, flushed per decision -- a run that dies
   halfway still leaves usable data.

Layer: depends on nothing in this project. Imports no UI.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Set to a file path to switch logging on. Absent means no logging at all.
DECISION_LOG_PATH_ENV = "DECISION_LOG_PATH"

#: A label stamped on every record, so runs that share one log file can be told
#: apart afterwards. The ablation sets it to "<window>/<arm>"; without it, a
#: multi-arm run produces one interleaved file with no way to separate the arms,
#: and every per-arm statistic computed from it is silently pooled.
DECISION_LOG_RUN_ENV = "DECISION_LOG_RUN"

#: Prompts are ~1.5k tokens each and repeat a large fixed scaffold. Storing the
#: full text every time makes the file mostly duplicate. The full text is kept
#: for the FIRST occurrence of each distinct prompt and a hash thereafter, so
#: the file stays small while every prompt remains reconstructible.
_seen_prompts: set = set()
_lock = threading.Lock()


def _jsonable(value: Any) -> Any:
    """Best-effort conversion, since readings arrive as dataclasses or dicts."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


class DecisionLogger:
    """Appends one JSON object per decision. Silent and inert when disabled."""

    def __init__(self, path: Optional[str] = None):
        raw = path if path is not None else os.getenv(DECISION_LOG_PATH_ENV, "")
        self.path = Path(raw).expanduser() if raw else None
        self.enabled = self.path is not None
        self.run_tag = os.getenv(DECISION_LOG_RUN_ENV, "") or None
        self.written = 0
        self.failures = 0
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Decision logging enabled: %s", self.path)

    def log(self, **fields: Any) -> None:
        """Record one decision. Never raises into the caller.

        A logging failure must not change a trading result, so the exception is
        counted and swallowed. `failures` being non-zero is the signal that the
        resulting file is incomplete and should not be analysed as if it were
        the whole run.
        """
        if not self.enabled:
            return
        try:
            record = {k: _jsonable(v) for k, v in fields.items()}
            prompt = record.get("prompt")
            if isinstance(prompt, str):
                digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
                record["prompt_sha"] = digest
                with _lock:
                    first_time = digest not in _seen_prompts
                    _seen_prompts.add(digest)
                if not first_time:
                    # Reconstructible from the first occurrence of this hash.
                    record["prompt"] = None
                    record["prompt_elided"] = True
            if self.run_tag:
                record.setdefault("run", self.run_tag)
            record.setdefault("logged_at_utc",
                              dt.datetime.now(dt.timezone.utc).isoformat())
            line = json.dumps(record, default=str, sort_keys=True)
            with _lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
            self.written += 1
        except Exception as e:           # observation must never break a run
            self.failures += 1
            if self.failures <= 3:
                logger.warning("Decision logging failed (%d): %s",
                               self.failures, e)

    def summary(self) -> Dict[str, Any]:
        return {"enabled": self.enabled,
                "path": str(self.path) if self.path else None,
                "records": self.written, "failures": self.failures}


def rule_features(sig: Dict[str, Any]) -> Dict[str, Any]:
    """The rule-engine state the model was shown, as flat numbers.

    Flat and numeric on purpose: the first analysis this file exists for is
    "can the LLM's adjustment be predicted from what the rules already said?",
    which wants a feature row, not nested prose.
    """
    if not isinstance(sig, dict):
        return {}
    return {
        "rsi": sig.get("rsi"),
        "ma20": sig.get("ma20"),
        "ma50": sig.get("ma50"),
        "macd_hist": sig.get("macd_hist"),
        "momentum": sig.get("momentum"),
        "bullish_points": sig.get("bullish"),
        "bearish_points": sig.get("bearish"),
        "net_signal": sig.get("net_signal"),
        "positioning_points": sig.get("positioning_points"),
        "positioning_score": sig.get("positioning_score"),
        # The signal dict spells these `text_sentiment_*`, not `text_*`. The
        # first version of this function guessed the shorter names, logged None
        # for all 53 rows, and the analysis reported the text channel as a
        # constant -- which reads exactly like "the model ignores text" rather
        # than "the field was never populated". Read the keys from the source,
        # not from the pattern the neighbouring field happens to follow.
        "text_points": sig.get("text_sentiment_points"),
        "text_score": sig.get("text_sentiment_score"),
        "triggered_rules": sig.get("details"),
    }


#: One logger per process. Built on first use so importing this module costs
#: nothing and touches no filesystem.
_logger_instance: Optional[DecisionLogger] = None


def get_decision_logger() -> DecisionLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DecisionLogger()
    return _logger_instance


def reset_decision_logger() -> None:
    """Drop the cached logger and prompt-dedup state. For tests."""
    global _logger_instance
    _logger_instance = None
    with _lock:
        _seen_prompts.clear()
