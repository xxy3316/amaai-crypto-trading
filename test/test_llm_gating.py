"""USE_LLM_DECISIONS=false must mean NO model call, from ANY agent.

Regression suite for the confound found on 2026-09-09 and fixed on 2026-09-11.
The flag used to gate only `TradingDecisionAgent`; `MarketAnalystAgent`,
`PatternRecognitionAgent` and `RiskManagementAgent` each called `get_llm()` in
`__init__` and invoked the model on every bar regardless.

That was not merely wasteful. The risk agent's `risk_level` drives
`confidence_multiplier = {"low": 1.2, "high": 0.8}`, so with moderate-mode
`min_confidence = 0.65` a "high" reading drops a `net_signal` of 2 from 0.80 to
0.64 and converts a BUY into a HOLD. A single LLM word could therefore flip a
trade inside the arm that was supposed to contain no LLM at all, which made
every "the LLM adds X" claim built on that baseline unsound.

Run directly (the package name `test` shadows the stdlib `test` module):

    python test/test_llm_gating.py
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Must be set BEFORE the app module is imported: core.config reads it at import.
os.environ["USE_LLM_DECISIONS"] = "false"
logging.getLogger("streamlit").setLevel(logging.ERROR)


def load_app():
    spec = importlib.util.spec_from_file_location("autotrade", REPO / "auto-trade.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["autotrade"] = module
    spec.loader.exec_module(module)
    return module


APP = load_app()


def make_price_frame(n: int = 60) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 00:00:00", periods=n, freq="h")
    close = np.full(n, 42000.0)
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": np.full(n, 100.0),
            "RSI": np.full(n, 50.0),
            "MA20": close, "MA50": close,
            "MACD_line": np.zeros(n), "MACD_signal": np.zeros(n),
            "MACD_hist": np.zeros(n),
            "UpperBB": close * 1.02, "MidBB": close, "LowerBB": close * 0.98,
        },
        index=index,
    )


class TestFlagIsOff(unittest.TestCase):

    def test_the_app_really_loaded_with_the_llm_disabled(self):
        """Everything below is meaningless if this is not true."""
        self.assertFalse(APP.USE_LLM_DECISIONS)


class TestSupportAgentsAreGated(unittest.TestCase):
    """The three agents that used to call the model unconditionally."""

    def setUp(self):
        self.df = make_price_frame()
        self.config = APP.TradingConfig.get_moderate_config(10_000.0)
        self.timestamp = self.df.index[30]

    def test_market_agent_builds_no_client(self):
        agent = APP.MarketAnalystAgent(self.df)
        self.assertFalse(agent.use_llm)
        self.assertIsNone(agent.llm, "a provider client was constructed anyway")
        self.assertIsNone(agent.agent_executor)

    def test_pattern_agent_builds_no_client(self):
        agent = APP.PatternRecognitionAgent(self.df)
        self.assertFalse(agent.use_llm)
        self.assertIsNone(agent.llm, "a provider client was constructed anyway")

    def test_risk_agent_builds_no_client(self):
        agent = APP.RiskManagementAgent(self.df, self.config)
        self.assertFalse(agent.use_llm)
        self.assertIsNone(agent.llm, "a provider client was constructed anyway")
        self.assertIsNone(agent.structured_llm)

    def test_decision_agent_builds_no_client(self):
        agent = APP.TradingDecisionAgent(self.df, self.config)
        self.assertFalse(agent.use_llm)
        self.assertIsNone(agent.llm)

    def test_market_agent_returns_the_disabled_marker(self):
        out = APP.MarketAnalystAgent(self.df).analyze(self.timestamp)
        self.assertEqual(out, APP.LLM_DISABLED_MARKET_ANALYSIS)
        self.assertIn("USE_LLM_DECISIONS=false", out)

    def test_pattern_agent_returns_the_disabled_marker(self):
        out = APP.PatternRecognitionAgent(self.df).identify_patterns(self.timestamp)
        self.assertEqual(out, APP.LLM_DISABLED_PATTERN_ANALYSIS)

    def test_risk_agent_returns_a_neutral_assessment(self):
        risk = APP.RiskManagementAgent(self.df, self.config).assess_risk(
            self.timestamp, portfolio=None, last_decision=None)
        self.assertEqual(risk["source"], "llm_disabled")
        self.assertEqual(risk["risk_level"], "medium")

    def test_risk_agent_still_reports_volatility(self):
        """Volatility comes from price data, so switching off the LLM must not
        silently drop it — that would look like missing data rather than a
        disabled agent."""
        risk = APP.RiskManagementAgent(self.df, self.config).assess_risk(self.timestamp)
        self.assertIn("volatility_pct", risk)


class TestNeutralRiskAppliesNoTilt(unittest.TestCase):
    """"medium" is load-bearing: it is the only level with multiplier 1.0."""

    def test_medium_is_the_identity_multiplier(self):
        multipliers = {"low": 1.2, "high": 0.8}
        self.assertEqual(multipliers.get("medium", 1.0), 1.0)

    def test_disabled_risk_helper_uses_medium(self):
        self.assertEqual(APP.llm_disabled_risk()["risk_level"], "medium")

    @staticmethod
    def net_signal_2_frame() -> pd.DataFrame:
        """A frame scoring exactly net_signal = 2.

        That is the value that matters: confidence is
        `min(0.9, 0.6 + net * 0.1) * multiplier`, so net=2 gives a base of 0.80,
        and 0.80 * 0.8 = 0.64 falls just under the 0.65 moderate-mode gate.
        At net >= 3 the base is 0.90 and even a "high" reading clears the gate,
        so the flip would not be visible.

        Two bullish points, no bearish ones:
          * price above MA20 but below MA50  -> +1
          * MACD line above signal, hist > 0 -> +1
        RSI is neutral at 50 and the close is flat, so nothing else scores.
        """
        df = make_price_frame()
        df.loc[:, "RSI"] = 50.0
        df.loc[:, "MA20"] = df["close"] * 0.999
        df.loc[:, "MA50"] = df["close"] * 1.001
        df.loc[:, "MACD_line"] = 1.0
        df.loc[:, "MACD_signal"] = 0.0
        df.loc[:, "MACD_hist"] = 1.0
        return df

    def test_confidence_scales_by_the_risk_multiplier(self):
        """The mechanism itself: risk_level multiplies confidence."""
        df = self.net_signal_2_frame()
        agent = APP.TradingDecisionAgent(df, self.config())
        portfolio = {"cash": 10_000.0, "holdings": 0.0,
                     "entry_price": 0.0, "holding": False}
        sig = agent._compute_rule_signals(30, df.iloc[30], float(df["close"].iloc[30]))
        self.assertEqual(sig["net_signal"], 2, sig["details"])

        confidences = {}
        for level in ("low", "medium", "high"):
            d = agent.make_decision(df.index[30], "m", "p",
                                    {"risk_level": level, "reasoning": "r"},
                                    dict(portfolio))
            confidences[level] = round(d.confidence, 4)
        self.assertAlmostEqual(confidences["medium"], 0.80, places=4)
        self.assertAlmostEqual(confidences["low"], 0.96, places=4)

    def test_a_high_reading_would_have_flipped_the_trade(self):
        """Demonstrates the bug this suite exists to prevent.

        Same inputs, only risk_level differs: "high" suppresses the BUY. If the
        support agents were ungated, the model chose this value on every bar.
        """
        df = self.net_signal_2_frame()
        agent = APP.TradingDecisionAgent(df, self.config())
        portfolio = {"cash": 10_000.0, "holdings": 0.0,
                     "entry_price": 0.0, "holding": False}
        actions = {}
        for level in ("low", "medium", "high"):
            decision = agent.make_decision(
                df.index[30], "m", "p",
                {"risk_level": level, "reasoning": "r"}, dict(portfolio))
            actions[level] = (decision.action, round(decision.confidence, 3))

        self.assertNotEqual(
            actions["high"][0], actions["medium"][0],
            f"expected risk_level to change the action, got {actions}")
        self.assertEqual(actions["medium"][0], APP.TradingAction.BUY, actions)
        self.assertEqual(actions["high"][0], APP.TradingAction.HOLD, actions)

    @staticmethod
    def config():
        return APP.TradingConfig.get_moderate_config(10_000.0)


class TestProvenanceDoesNotNameAnUnusedModel(unittest.TestCase):

    def test_rules_only_runs_report_no_provider(self):
        """A run that made zero model calls must not name a deployment.

        Recording one implies the result depended on it, and probing the
        deployment costs a network round trip the rules-only arm should not
        need.
        """
        source = (REPO / "auto-trade.py").read_text(encoding="utf-8")
        self.assertIn("describe_active_model() if USE_LLM_DECISIONS else", source,
                      "run metadata should only describe a model when one was used")


if __name__ == "__main__":
    unittest.main(verbosity=2)
