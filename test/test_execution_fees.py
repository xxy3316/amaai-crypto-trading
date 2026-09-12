"""Fees must actually be charged, on BOTH sides, and must come from the config.

Regression suite for a defect found on 2026-09-09: `TradingConfig` hard-coded
`buy_fee_pct = 0.10` and `sell_fee_pct = 0.0`, and nothing anywhere read
BUY_FEE_PCT / SELL_FEE_PCT from the environment. Every backtest therefore paid
to enter a position and exited free. On the golden window that was $127.88 of
unmodelled cost against $281.11 of reported profit -- 45% of the result, and
larger than any signal effect measured so far.

These tests are deliberately about ARITHMETIC rather than about a particular
fee level: they assert that a round trip costs both fees, that the fees are
symmetric by default, and that the environment overrides reach the config.

Run directly (the package name `test` shadows the stdlib `test` module):

    python test/test_execution_fees.py
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.config as config_module  # noqa: E402
import core.execution as execution_module  # noqa: E402
from core.config import TradingAction, TradingConfig, TradingDecision  # noqa: E402
from core.execution import execute_trade  # noqa: E402


def a_decision(action: TradingAction) -> TradingDecision:
    return TradingDecision(
        action=action, confidence=0.9, reasoning="test",
        price=100.0, timestamp=datetime(2024, 1, 1),
    )


def a_portfolio(cash: float = 10_000.0) -> dict:
    return {"cash": cash, "holdings": 0.0, "entry_price": 0.0, "holding": False}


class TestFeesAreCharged(unittest.TestCase):
    """A fee that is configured must show up in the cash."""

    def setUp(self):
        # 100% position sizing and zero slippage make the arithmetic exact, so
        # a failure points at the fee rather than at sizing or fill costs.
        # Slippage is charged per side too, so leaving it on would mean these
        # assertions silently tested fees + slippage together.
        self._slippage = execution_module.SLIPPAGE_PCT
        execution_module.SLIPPAGE_PCT = 0.0
        self.config = TradingConfig(initial_capital=10_000.0,
                                    position_size_pct=100.0,
                                    buy_fee_pct=1.0, sell_fee_pct=2.0)

    def tearDown(self):
        execution_module.SLIPPAGE_PCT = self._slippage

    def _round_trip(self, buy_at=100.0, sell_at=100.0):
        portfolio = a_portfolio()
        buy = execute_trade(a_decision(TradingAction.BUY), portfolio, buy_at,
                            self.config, datetime(2024, 1, 1), fill_price=buy_at)
        sell = execute_trade(a_decision(TradingAction.SELL), portfolio, sell_at,
                             self.config, datetime(2024, 1, 2), fill_price=sell_at)
        return buy, sell, portfolio

    def test_buy_fee_is_deducted(self):
        buy, _, _ = self._round_trip()
        self.assertAlmostEqual(buy["fees"], 10_000.0 * 0.01, places=6)

    def test_sell_fee_is_deducted(self):
        """The bug: this was always 0.0 because the default was 0.0."""
        _, sell, _ = self._round_trip()
        self.assertGreater(sell["fees"], 0.0,
                           "sell fee was not charged at all")
        gross = sell["shares"] * sell["price"]
        self.assertAlmostEqual(sell["fees"], gross * 0.02, places=6)

    def test_flat_round_trip_loses_exactly_both_fees(self):
        """Buy and sell at the same price: the loss IS the two fees."""
        buy, sell, portfolio = self._round_trip(100.0, 100.0)
        self.assertFalse(portfolio["holding"])
        lost = 10_000.0 - portfolio["cash"]
        self.assertAlmostEqual(lost, buy["fees"] + sell["fees"], places=6)
        self.assertGreater(lost, 0.0)

    def test_zero_fee_config_costs_nothing(self):
        self.config = TradingConfig(initial_capital=10_000.0,
                                    position_size_pct=100.0,
                                    buy_fee_pct=0.0, sell_fee_pct=0.0)
        _, _, portfolio = self._round_trip(100.0, 100.0)
        self.assertAlmostEqual(portfolio["cash"], 10_000.0, places=6)

    def test_more_trades_cost_more(self):
        """Turnover has to be penalised or every ablation favours overtrading."""
        one = a_portfolio()
        for _ in range(1):
            execute_trade(a_decision(TradingAction.BUY), one, 100.0,
                          self.config, datetime(2024, 1, 1), fill_price=100.0)
            execute_trade(a_decision(TradingAction.SELL), one, 100.0,
                          self.config, datetime(2024, 1, 2), fill_price=100.0)

        many = a_portfolio()
        for _ in range(5):
            execute_trade(a_decision(TradingAction.BUY), many, 100.0,
                          self.config, datetime(2024, 1, 1), fill_price=100.0)
            execute_trade(a_decision(TradingAction.SELL), many, 100.0,
                          self.config, datetime(2024, 1, 2), fill_price=100.0)

        self.assertLess(many["cash"], one["cash"])


class TestFeeDefaults(unittest.TestCase):

    def test_defaults_are_symmetric(self):
        """Binance spot taker fees are 0.1% per side. An exit is not free."""
        cfg = TradingConfig()
        self.assertGreater(cfg.sell_fee_pct, 0.0,
                           "a zero default sell fee is the original bug")
        self.assertAlmostEqual(cfg.buy_fee_pct, cfg.sell_fee_pct, places=9)

    def test_every_preset_carries_the_fees(self):
        presets = [TradingConfig.get_conservative_config(10_000.0),
                   TradingConfig.get_moderate_config(10_000.0),
                   TradingConfig.get_aggressive_config(10_000.0)]
        for cfg in presets:
            self.assertGreater(cfg.buy_fee_pct, 0.0, cfg.trading_mode)
            self.assertGreater(cfg.sell_fee_pct, 0.0, cfg.trading_mode)

    def test_module_constants_reach_the_config(self):
        """Patching the constant must change new configs.

        This is the property that was missing: BUY_FEE_PCT / SELL_FEE_PCT
        existed in .env but were wired to nothing.
        """
        original = (config_module.BUY_FEE_PCT, config_module.SELL_FEE_PCT)
        try:
            config_module.BUY_FEE_PCT = 0.42
            config_module.SELL_FEE_PCT = 0.77
            cfg = TradingConfig()
            self.assertAlmostEqual(cfg.buy_fee_pct, 0.42, places=9)
            self.assertAlmostEqual(cfg.sell_fee_pct, 0.77, places=9)
        finally:
            config_module.BUY_FEE_PCT, config_module.SELL_FEE_PCT = original

    def test_environment_variables_are_read(self):
        env = dict(os.environ)
        try:
            os.environ["BUY_FEE_PCT"] = "0.33"
            os.environ["SELL_FEE_PCT"] = "0.44"
            reloaded = importlib.reload(config_module)
            self.assertAlmostEqual(reloaded.BUY_FEE_PCT, 0.33, places=9)
            self.assertAlmostEqual(reloaded.SELL_FEE_PCT, 0.44, places=9)
        finally:
            os.environ.clear()
            os.environ.update(env)
            importlib.reload(config_module)

    def test_fees_are_recorded_in_the_run_provenance(self):
        """A reported return is only interpretable next to its cost model."""
        from core.config import describe_configuration
        text = repr(describe_configuration())
        self.assertIn("fee", text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
