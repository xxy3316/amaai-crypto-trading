"""Core trading system: configuration, data, execution and the backtest engine.

Layering, strictly one-directional so there are no import cycles:

    core.config      env flags and domain types; depends on nothing in-project
    core.llm         provider setup and the LLM contract; depends on config
    core.data        market data and technical indicators; depends on config
    core.execution   fills, fees and slippage; depends on config
    signals.*        exogenous data sources; independent of core
    auto-trade.py    agents, the simulation loop and the Streamlit UI

Nothing in `core` imports Streamlit. That is the property that makes a headless
run possible, which the ablation table in the paper needs.
"""

__all__: list = []
