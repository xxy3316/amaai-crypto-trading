"""Exogenous (non-price) signal sources for the trading system.

Each module here is a self-contained, reproducible data source: it downloads
from a public endpoint into a byte-exact local cache, records provenance in a
manifest, and exposes features aligned to the price bars on a strict
point-in-time basis (a bar may only see data published before it).

Sources are deliberately kept out of `auto-trade.py` so that the data pipeline
can be re-run, audited and cited independently of the Streamlit application.

Available sources
-----------------
    signals.binance_positioning   Binance USD-M futures positioning (free, keyless)

Nothing is re-exported here on purpose. Importing a submodule at package level
would make `python -m signals.binance_positioning` import the module twice and
emit a RuntimeWarning, so callers import the submodule directly:

    from signals.binance_positioning import PositioningSignalAgent
"""

__all__: list = []
