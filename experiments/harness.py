"""Shared scaffolding for running the app headlessly.

Used by both `test/golden_backtest.py` and `experiments/run_ablation.py`, which
otherwise kept two copies of the same Streamlit stub.

It lives here rather than under `test/` because that directory shadows Python's
stdlib `test` package, so `from test.x import y` fails from a subprocess.

None of this belongs in `core`: `core` is the production path and imports no
Streamlit at all. This module exists precisely to fake the UI layer that sits
above it.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]


# ── Streamlit stub ──────────────────────────────────────────────────────────

class _Ctx:
    """Stands in for anything used as a context manager or a placeholder."""

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return lambda *a, **k: _Ctx()


class _SessionState(dict):
    """Streamlit's session_state supports both attribute and item access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value


class StreamlitStub:
    """Swallows every st.* call and returns usable placeholders."""

    def __init__(self):
        self.__dict__["session_state"] = _SessionState()

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]

    def tabs(self, labels):
        return [_Ctx() for _ in labels]

    def progress(self, *a, **k): return _Ctx()
    def empty(self, *a, **k): return _Ctx()
    def expander(self, *a, **k): return _Ctx()
    def container(self, *a, **k): return _Ctx()
    def spinner(self, *a, **k): return _Ctx()
    def form(self, *a, **k): return _Ctx()
    def __getattr__(self, name): return lambda *a, **k: _Ctx()


# ── deterministic stand-ins for the LLM-backed support agents ───────────────
# NOTE (2026-09-11): this is no longer needed to obtain an LLM-free arm.
# USE_LLM_DECISIONS=false now gates the market, pattern and risk agents as well
# as the decision agent, so those three make no model call and return the
# neutral values in auto-trade.py (LLM_DISABLED_* / llm_disabled_risk). The
# golden harness was switched to the real, unstubbed path and reproduced the
# baseline exactly, which is the evidence that the gate is equivalent to these
# stubs.
#
# It is kept for one remaining use: pinning the support agents while the
# DECISION agent runs live, which isolates "what does the decision LLM add"
# from "what do the narrator agents add". That is a different experiment from
# the rules-only arm, not a substitute for it.
#
# Historical context for why the risk stub matters: risk_level drives a
# confidence multiplier ({"low": 1.2, "high": 0.8}), and with moderate-mode
# min_confidence=0.65 a "high" reading drops a net_signal of 2 from 0.80 to
# 0.64 and converts a BUY into a HOLD.
STUB_MARKET_ANALYSIS = "STUB market analysis (headless run)."
STUB_PATTERN_ANALYSIS = "STUB pattern analysis (headless run)."
STUB_RISK = {
    "risk_level": "medium",          # multiplier 1.0, so no confidence tilt
    "position_size_pct": 25.0,
    "stop_loss_pct": 5.0,
    "take_profit_pct": 10.0,
    "reasoning": "STUB risk assessment (headless run).",
    "source": "stub",
}


def stub_support_agents(app) -> None:
    """Pin the market, pattern and risk agents to fixed values."""
    app.MarketAnalystAgent.analyze = (
        lambda self, timestamp: STUB_MARKET_ANALYSIS)
    app.PatternRecognitionAgent.identify_patterns = (
        lambda self, timestamp: STUB_PATTERN_ANALYSIS)
    app.RiskManagementAgent.assess_risk = (
        lambda self, timestamp, action=None, portfolio=None, last_decision=None:
        dict(STUB_RISK))


# ── app loading and invocation ──────────────────────────────────────────────

def load_app(stub_support: bool = False):
    """Import auto-trade.py under a legal module name, with the UI stubbed.

    The filename contains a hyphen, so it cannot be imported normally.
    """
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    logging.getLogger("streamlit").setLevel(logging.ERROR)

    spec = importlib.util.spec_from_file_location("autotrade", REPO / "auto-trade.py")
    app = importlib.util.module_from_spec(spec)
    sys.modules["autotrade"] = app
    spec.loader.exec_module(app)

    app.st = StreamlitStub()
    if stub_support:
        stub_support_agents(app)
    return app


def run_headless(
    app,
    start: str,
    end: str,
    symbol: str = "BTC/USDT",
    interval: str = "1h",
    capital: float = 10000.0,
    strategy: str = "moderate",
    enable_vector_db: bool = False,
) -> tuple:
    """Run one backtest and return (summary, elapsed_seconds).

    The vector agent defaults off: its store starts empty in every run and is
    discarded at the end, so including it adds variance without adding anything
    comparable between runs. (The deep-learning agent it used to sit beside was
    removed entirely on 2026-09-13 -- it had no training code.)

    The UI and persistence tails are replaced, so only the numbers come back.
    """
    captured = {}
    app.display_simulation_results = lambda summary, *a, **k: captured.update(
        summary=summary)
    app.save_simulation_result = lambda *a, **k: None
    app.save_results_to_db = lambda *a, **k: None
    app.get_database_available = lambda: False

    import pandas as pd
    started = time.time()
    app.run_trading_simulation(
        symbol_input=symbol,
        interval=interval,
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(end),
        initial_capital=capital,
        enable_vector_db=enable_vector_db,
        show_reasoning=False,
        strategy_mode=strategy,
        # None means "keep the strategy preset", which the None guards in
        # run_trading_simulation now honour.
        custom_position_size=None,
        custom_confidence=None,
        custom_signal_threshold=None,
    )
    elapsed = time.time() - started

    summary: Optional[dict] = captured.get("summary")
    if summary is None:
        raise RuntimeError("the backtest produced no summary; check the log")
    return summary, elapsed
