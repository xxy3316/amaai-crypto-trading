"""Order execution: fill pricing, fees and slippage.

Moved out of `auto-trade.py` verbatim.

Layer: depends on `core.config` only.
"""

from __future__ import annotations

import logging
from datetime import datetime

from core.config import (
    SLIPPAGE_PCT,
    TradingAction,
    TradingConfig,
    TradingDecision,
)

logger = logging.getLogger(__name__)

# ── SIMULATION FUNCTIONS ───────────────────────────────────────────────────
# PHASE 2 INTEGRITY RULE: realistic fills.
# Previously a decision made from a bar's close was also FILLED at that same
# close, which assumes you can trade at a price you only know once the bar has
# ended. Standard practice, and what a reviewer will insist on, is to fill at
# the NEXT bar's open, plus slippage. Both are configurable so the optimistic
# same-bar convention remains available as an explicit ablation.
# SLIPPAGE_PCT and EXECUTION_MODE are imported from core.config above.

def apply_slippage(price: float, side: str) -> float:
    """Move the fill price against the trader by SLIPPAGE_PCT."""
    factor = SLIPPAGE_PCT / 100.0
    return price * (1 + factor) if side == "BUY" else price * (1 - factor)

def execute_trade(decision: TradingDecision, portfolio: dict, current_price: float,
                  config: TradingConfig, timestamp: datetime,
                  fill_price: float = None, fill_timestamp: datetime = None) -> dict:
    """Execute a trading decision and update the portfolio.

    `current_price` is the decision-time price (bar close). `fill_price` is the
    price the order actually executes at, normally the next bar's open. When
    `fill_price` is omitted the function falls back to same-bar execution so
    existing callers keep working.
    """
    raw_fill = current_price if fill_price is None else fill_price
    fill_ts = timestamp if fill_timestamp is None else fill_timestamp

    if decision.action == TradingAction.BUY and not portfolio['holding']:
        exec_price = apply_slippage(raw_fill, "BUY")

        # Calculate position size based on config
        position_pct = config.position_size_pct / 100.0  # Convert percentage to decimal
        max_spend = portfolio['cash'] * position_pct
        fees = max_spend * config.buy_fee_pct / 100
        net_spend = max_spend - fees
        shares = net_spend / exec_price

        portfolio['holdings'] = shares
        portfolio['cash'] -= max_spend
        portfolio['entry_price'] = exec_price
        portfolio['holding'] = True

        return {
            'timestamp': fill_ts,
            'decision_timestamp': timestamp,
            'action': 'BUY',
            'price': exec_price,
            'decision_price': current_price,
            'slippage_pct': SLIPPAGE_PCT,
            'shares': shares,
            'cost': max_spend,
            'fees': fees,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'position_pct': config.position_size_pct
        }

    elif decision.action == TradingAction.SELL and portfolio['holding']:
        exec_price = apply_slippage(raw_fill, "SELL")

        # Sell all holdings
        gross_proceeds = portfolio['holdings'] * exec_price
        fees = gross_proceeds * config.sell_fee_pct / 100
        net_proceeds = gross_proceeds - fees

        profit = net_proceeds - (portfolio['holdings'] * portfolio['entry_price'])

        portfolio['cash'] += net_proceeds
        sold_shares = portfolio['holdings']
        portfolio['holdings'] = 0.0
        portfolio['entry_price'] = 0.0
        portfolio['holding'] = False

        return {
            'timestamp': fill_ts,
            'decision_timestamp': timestamp,
            'action': 'SELL',
            'price': exec_price,
            'decision_price': current_price,
            'slippage_pct': SLIPPAGE_PCT,
            'shares': sold_shares,
            'proceeds': net_proceeds,
            'fees': fees,
            'profit': profit,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning
        }

    return None

