"""End-to-end wiring tests: does positioning actually reach the decision?

The Phase 1 audit found six of seven agents computing outputs that never
reached a trade. These tests exist so that cannot happen again silently: they
assert that a positioning reading moves the rule signal, that it appears in the
LLM prompt, and -- the case that broke sentiment -- that an UNAVAILABLE reading
changes nothing rather than being counted as a neutral measurement.

No network access. `auto-trade.py` is loaded by path because its filename is
not a valid module name.

    ./venv/Scripts/python.exe test/test_positioning_integration.py
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

# Keep the LLM out of the loop before the app module is imported.
os.environ["USE_LLM_DECISIONS"] = "false"
logging.getLogger("streamlit").setLevel(logging.ERROR)

from signals.binance_positioning import PositioningReading  # noqa: E402
from signals.text_sentiment import TextSentimentReading  # noqa: E402


def load_app():
    """Import auto-trade.py under a legal module name."""
    spec = importlib.util.spec_from_file_location("autotrade", REPO / "auto-trade.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["autotrade"] = module
    spec.loader.exec_module(module)
    return module


APP = load_app()


def make_price_frame(n: int = 60) -> pd.DataFrame:
    """A price frame carrying exactly the indicator columns the rules read.

    Values are chosen to sit in the rules' NEUTRAL band, so any change in the
    net signal during these tests is attributable to positioning alone.
    """
    index = pd.date_range("2024-01-02 00:00:00", periods=n, freq="h")
    close = np.full(n, 42000.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 100.0),
            "RSI": np.full(n, 50.0),        # neutral: no RSI points
            "MA20": close,                  # price == MA20: no MA points
            "MA50": close,
            "MACD_line": np.zeros(n),       # flat: no MACD points
            "MACD_signal": np.zeros(n),
            "MACD_hist": np.zeros(n),
        },
        index=index,
    )


def reading(score: float, bull: int = 0, bear: int = 0,
            available: bool = True) -> PositioningReading:
    return PositioningReading(
        score=score, confidence=0.9, bullish_points=bull, bearish_points=bear,
        reasoning="test fixture reading", source="binance_futures_metrics",
        available=available, features={"z_count_long_short_ratio": -2.0},
        staleness_bars=0.5,
    )


class TestRuleEngineWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = make_price_frame()
        APP.USE_LLM_DECISIONS = False
        cls.config = APP.TradingConfig.get_moderate_config(initial_capital=10000.0)
        cls.agent = APP.TradingDecisionAgent(cls.df, cls.config)
        cls.agent.use_llm = False
        cls.agent.structured_llm = None
        cls.idx = 30
        cls.row = cls.df.iloc[cls.idx]
        cls.price = float(cls.row["close"])

    def _signals(self, positioning=None):
        return self.agent._compute_rule_signals(
            self.idx, self.row, self.price, positioning=positioning
        )

    def test_baseline_is_neutral(self):
        """The fixture must start at zero, or the deltas below prove nothing."""
        self.assertEqual(self._signals()["net_signal"], 0)

    def test_bullish_positioning_raises_net_signal(self):
        base = self._signals()["net_signal"]
        with_pos = self._signals(reading(+0.8, bull=2))["net_signal"]
        self.assertEqual(with_pos - base, 2)

    def test_bearish_positioning_lowers_net_signal(self):
        base = self._signals()["net_signal"]
        with_pos = self._signals(reading(-0.8, bear=2))["net_signal"]
        self.assertEqual(with_pos - base, -2)

    def test_unavailable_reading_changes_nothing(self):
        """The exact failure mode that made sentiment decorative.

        An unavailable channel must be inert, NOT a neutral vote. If this test
        ever passes only because the reading happened to score 0.0, the channel
        has silently become decorative again.
        """
        base = self._signals()
        inert = self._signals(reading(0.0, available=False))
        self.assertEqual(inert["net_signal"], base["net_signal"])
        self.assertEqual(inert["details"], base["details"])
        self.assertIsNone(inert["positioning_score"])
        self.assertEqual(inert["positioning_points"], 0)

    def test_none_positioning_changes_nothing(self):
        self.assertEqual(self._signals(None)["net_signal"],
                         self._signals()["net_signal"])

    def test_positioning_is_reported_in_signal_details(self):
        sig = self._signals(reading(+0.8, bull=2))
        self.assertEqual(sig["positioning_points"], 2)
        self.assertAlmostEqual(sig["positioning_score"], 0.8)
        self.assertTrue(any("positioning" in d.lower() for d in sig["details"]),
                        f"expected a positioning entry in {sig['details']}")

    def test_zero_point_reading_is_recorded_but_moves_nothing(self):
        """A real but weak reading is logged, yet must not add points."""
        sig = self._signals(reading(+0.1, bull=0))
        self.assertEqual(sig["net_signal"], 0)
        self.assertAlmostEqual(sig["positioning_score"], 0.1)


class TestDecisionAndPromptWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = make_price_frame()
        APP.USE_LLM_DECISIONS = False
        cls.config = APP.TradingConfig.get_aggressive_config(initial_capital=10000.0)
        cls.agent = APP.TradingDecisionAgent(cls.df, cls.config)
        cls.agent.use_llm = False
        cls.agent.structured_llm = None
        cls.timestamp = cls.df.index[30]
        cls.portfolio = {"cash": 10000.0, "holdings": 0.0,
                         "holding": False, "entry_price": 0.0}

    def _decide(self, positioning=None):
        return self.agent.make_decision(
            self.timestamp, "market analysis text", "pattern analysis text",
            {"risk_level": "medium", "reasoning": "test"}, dict(self.portfolio),
            positioning=positioning,
        )

    def test_make_decision_accepts_positioning(self):
        decision = self._decide(reading(+0.8, bull=2))
        self.assertIsNotNone(decision)
        self.assertIn(decision.action, list(APP.TradingAction))

    def test_positioning_can_flip_the_action_in_rules_only_arm(self):
        """Rules-only arm: positioning alone must be able to change the trade.

        The aggressive config has signal_threshold=1, so +2 points from
        positioning crosses it from a neutral baseline. This is what makes the
        exogenous channel identifiable WITHOUT the LLM.
        """
        flat = self._decide(None)
        bullish = self._decide(reading(+0.8, bull=2))
        self.assertEqual(flat.action, APP.TradingAction.HOLD)
        self.assertEqual(bullish.action, APP.TradingAction.BUY)

    def test_unavailable_positioning_does_not_flip_the_action(self):
        flat = self._decide(None)
        dead = self._decide(reading(0.0, available=False))
        self.assertEqual(dead.action, flat.action)

    def test_prompt_contains_positioning_block(self):
        sig = self.agent._compute_rule_signals(
            30, self.df.iloc[30], 42000.0, positioning=reading(+0.8, bull=2))
        prompt = self.agent._build_llm_context(
            self.timestamp, sig, 42000.0, APP.TradingAction.BUY,
            "market", "pattern", {"risk_level": "medium", "reasoning": "r"},
            dict(self.portfolio), dl_prediction=None, vector_insights=None,
            positioning=reading(+0.8, bull=2),
        )
        self.assertIn("FUTURES POSITIONING AGENT", prompt)
        self.assertIn("+0.800", prompt)
        # The prompt must warn against double counting, or the model adds the
        # same evidence twice: once as baseline, once as new information.
        self.assertIn("already worth", prompt)

    def test_prompt_marks_missing_positioning_as_unavailable(self):
        sig = self.agent._compute_rule_signals(30, self.df.iloc[30], 42000.0)
        prompt = self.agent._build_llm_context(
            self.timestamp, sig, 42000.0, APP.TradingAction.HOLD,
            "market", "pattern", {"risk_level": "medium", "reasoning": "r"},
            dict(self.portfolio), dl_prediction=None, vector_insights=None,
            positioning=None,
        )
        self.assertIn("FUTURES POSITIONING AGENT", prompt)
        self.assertIn("not available", prompt)


def text_reading(score: float, bull: int = 0, bear: int = 0,
                 available: bool = True, docs: int = 42) -> TextSentimentReading:
    return TextSentimentReading(
        score=score, confidence=0.8, bullish_points=bull, bearish_points=bear,
        reasoning="test fixture text reading", source="hackernews",
        available=available, doc_count=docs, mean_sentiment=0.21,
        features={"z_mean_sentiment": 2.0},
    )


class TestTextSentimentWiring(unittest.TestCase):
    """Same contract as positioning: reaches the rules, inert when unavailable."""

    @classmethod
    def setUpClass(cls):
        cls.df = make_price_frame()
        APP.USE_LLM_DECISIONS = False
        cls.config = APP.TradingConfig.get_aggressive_config(initial_capital=10000.0)
        cls.agent = APP.TradingDecisionAgent(cls.df, cls.config)
        cls.agent.use_llm = False
        cls.agent.structured_llm = None
        cls.idx = 30
        cls.row = cls.df.iloc[cls.idx]
        cls.timestamp = cls.df.index[cls.idx]
        cls.portfolio = {"cash": 10000.0, "holdings": 0.0,
                         "holding": False, "entry_price": 0.0}

    def _signals(self, text=None, positioning=None):
        return self.agent._compute_rule_signals(
            self.idx, self.row, float(self.row["close"]),
            positioning=positioning, text_sentiment=text,
        )

    def test_bullish_text_raises_net_signal(self):
        base = self._signals()["net_signal"]
        self.assertEqual(self._signals(text_reading(+0.8, bull=1))["net_signal"] - base, 1)

    def test_bearish_text_lowers_net_signal(self):
        base = self._signals()["net_signal"]
        self.assertEqual(self._signals(text_reading(-0.8, bear=1))["net_signal"] - base, -1)

    def test_unavailable_text_changes_nothing(self):
        """The exact failure mode that made the original sentiment decorative."""
        base = self._signals()
        inert = self._signals(text_reading(0.0, available=False))
        self.assertEqual(inert["net_signal"], base["net_signal"])
        self.assertEqual(inert["details"], base["details"])
        self.assertIsNone(inert["text_sentiment_score"])
        self.assertEqual(inert["text_sentiment_points"], 0)

    def test_none_text_changes_nothing(self):
        self.assertEqual(self._signals(None)["net_signal"], self._signals()["net_signal"])

    def test_text_appears_in_signal_details(self):
        sig = self._signals(text_reading(+0.8, bull=1))
        self.assertEqual(sig["text_sentiment_points"], 1)
        self.assertTrue(any("text sentiment" in d.lower() for d in sig["details"]),
                        f"expected a text sentiment entry in {sig['details']}")

    def test_both_channels_combine_additively(self):
        both = self._signals(text=text_reading(+0.8, bull=1),
                             positioning=reading(+0.8, bull=2))
        self.assertEqual(both["net_signal"], 3)
        self.assertEqual(both["text_sentiment_points"], 1)
        self.assertEqual(both["positioning_points"], 2)

    def test_channels_can_oppose_each_other(self):
        mixed = self._signals(text=text_reading(-0.8, bear=1),
                              positioning=reading(+0.8, bull=2))
        self.assertEqual(mixed["net_signal"], 1)

    def test_text_alone_can_flip_the_action_without_the_llm(self):
        """Aggressive config has signal_threshold=1, so +1 point crosses it."""
        flat = self.agent.make_decision(
            self.timestamp, "m", "p", {"risk_level": "medium", "reasoning": "r"},
            dict(self.portfolio))
        bullish = self.agent.make_decision(
            self.timestamp, "m", "p", {"risk_level": "medium", "reasoning": "r"},
            dict(self.portfolio), text_sentiment=text_reading(+0.8, bull=1))
        self.assertEqual(flat.action, APP.TradingAction.HOLD)
        self.assertEqual(bullish.action, APP.TradingAction.BUY)

    def test_prompt_contains_text_sentiment_block(self):
        sig = self._signals(text_reading(+0.8, bull=1))
        prompt = self.agent._build_llm_context(
            self.timestamp, sig, 42000.0, APP.TradingAction.BUY, "m", "p",
            {"risk_level": "medium", "reasoning": "r"}, dict(self.portfolio),
            dl_prediction=None, vector_insights=None, positioning=None,
            text_sentiment=text_reading(+0.8, bull=1),
        )
        self.assertIn("TEXT SENTIMENT AGENT", prompt)
        self.assertIn("already worth", prompt)
        self.assertIn("42 documents", prompt)

    def test_prompt_has_no_legacy_sentiment_channel(self):
        """The generated-post channel is deleted, not disabled.

        It previously reached the prompt as a "LEGACY SENTIMENT AGENT" row.
        Asserting its absence stops it being reintroduced as a neutral 0.0,
        which is what let it stay decorative for so long.
        """
        sig = self._signals()
        prompt = self.agent._build_llm_context(
            self.timestamp, sig, 42000.0, APP.TradingAction.HOLD, "m", "p",
            {"risk_level": "medium", "reasoning": "r"}, dict(self.portfolio),
            dl_prediction=None, vector_insights=None, positioning=None,
            text_sentiment=None,
        )
        self.assertNotIn("LEGACY SENTIMENT", prompt)
        self.assertNotIn("social-post", prompt)
        self.assertNotIn("score +0.000, confidence 0.00", prompt)
        # The real text channel must still have its own row.
        self.assertIn("TEXT SENTIMENT AGENT", prompt)


class TestAgentContract(unittest.TestCase):
    """The attributes auto-trade.py reads off a reading must exist."""

    def test_reading_exposes_every_field_the_app_uses(self):
        r = reading(+0.5, bull=1)
        for attr in ("score", "confidence", "bullish_points", "bearish_points",
                     "reasoning", "source", "available"):
            self.assertTrue(hasattr(r, attr), f"missing {attr}")
        payload = r.to_dict()
        for key in ("score", "available", "bullish_points", "bearish_points"):
            self.assertIn(key, payload)

    def test_app_exposes_the_phase3_flags(self):
        for flag in ("USE_POSITIONING_SIGNAL", "POSITIONING_MAX_POINTS",
                     "POSITIONING_LAG_BARS", "POSITIONING_ZSCORE_WINDOW",
                     "positioning_zscore_window",
                     "POSITIONING_AVAILABLE",
                     "USE_TEXT_SENTIMENT", "TEXT_SENTIMENT_SCORER",
                     "TEXT_SENTIMENT_MAX_POINTS", "TEXT_SENTIMENT_LAG_BARS",
                     "TEXT_SENTIMENT_WINDOW_HOURS", "TEXT_SENTIMENT_MIN_DOCS",
                     "TEXT_SENTIMENT_AVAILABLE"):
            self.assertTrue(hasattr(APP, flag), f"missing {flag}")

    def test_text_reading_exposes_every_field_the_app_uses(self):
        r = text_reading(+0.5, bull=1)
        for attr in ("score", "confidence", "bullish_points", "bearish_points",
                     "reasoning", "source", "available", "doc_count"):
            self.assertTrue(hasattr(r, attr), f"missing {attr}")
        payload = r.to_dict()
        for key in ("score", "available", "bullish_points", "doc_count"):
            self.assertIn(key, payload)

    def test_text_sentiment_max_points_is_below_positioning(self):
        """A noisier single-feature channel must not outweigh positioning."""
        self.assertLessEqual(APP.TEXT_SENTIMENT_MAX_POINTS,
                             APP.POSITIONING_MAX_POINTS)

    def test_positioning_defaults_to_off(self):
        """Leaving the flag alone must reproduce the pre-Phase-3 behaviour."""
        os.environ.pop("USE_POSITIONING_SIGNAL", None)
        self.assertIn(
            os.getenv("USE_POSITIONING_SIGNAL", "false").strip().lower(),
            ("false", "0", "no", "off"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
