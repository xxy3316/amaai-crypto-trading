"""Market data acquisition and technical indicators.

Moved out of `auto-trade.py` verbatim.

Layer: depends on `core.config` only. Imports no Streamlit, so a headless
backtest can fetch and prepare data without a UI session.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

from core.config import ALLOW_SYNTHETIC_DATA, SyntheticDataBlocked

logger = logging.getLogger(__name__)

def resolve_bar_index(index: pd.Index, timestamp) -> int:
    """Return the position of the last bar at or BEFORE `timestamp`.

    PHASE 2 INTEGRITY RULE: no lookahead. The previous code used
    `get_indexer(method='nearest')`, which can snap forward to a bar that had
    not closed yet at decision time. With exactly-aligned timestamps the two
    agree, but 'nearest' silently leaks future data the moment a timestamp is
    off-grid. Backward-only matching makes the guarantee structural instead of
    accidental, so the paper can state it plainly.
    """
    ts = pd.to_datetime(timestamp)
    pos = index.get_indexer([ts], method='pad')[0]
    if pos == -1:
        # Timestamp precedes the first bar; clamp forward but never past it.
        return 0
    return int(pos)

def fetch_binance_ta(symbol, timeframe, start, end, original_start=None, original_end=None):
    """Fetch OHLCV data from Binance and calculate technical indicators.

    Returns a DataFrame carrying `df.attrs['data_source']`, which is either
    'binance_live' or 'synthetic'. Callers must surface that provenance.
    """
    # Store original user-requested date range on first call
    if original_start is None:
        original_start = start
    if original_end is None:
        original_end = end

    try:
        logger.info("Attempting to fetch live data from Binance...")
        df = _fetch_from_binance(symbol, timeframe, start, end, original_start, original_end)
        if df is not None and not df.empty:
            df.attrs['data_source'] = 'binance_live'
            df.attrs['data_source_label'] = 'Binance live market data'
            df.attrs['symbol'] = symbol
            df.attrs['timeframe'] = timeframe
            df.attrs['fetched_at'] = datetime.now().isoformat()
            logger.info(f"Successfully fetched LIVE data from Binance: {len(df)} rows")
            return df
        raise ValueError("Binance returned an empty dataset")

    except Exception as live_error:
        logger.warning(f"Binance live fetch failed: {live_error}")

        if not ALLOW_SYNTHETIC_DATA:
            # Fail loudly. A blocked run is recoverable; a silently fabricated
            # result that reaches a paper is not.
            raise SyntheticDataBlocked(
                f"Live market data unavailable for {symbol} ({live_error}). "
                "Synthetic fallback is DISABLED, so no results were produced. "
                "Set ALLOW_SYNTHETIC_DATA=true in .env only for code testing - "
                "never for results you intend to report."
            ) from live_error

        logger.warning("ALLOW_SYNTHETIC_DATA is on - generating SYNTHETIC data. NOT valid for publication.")
        df = _generate_simulated_data(symbol, timeframe, start, end, original_start, original_end)
        if df is None or df.empty:
            raise ValueError(f"Synthetic data generation also failed for {symbol}")
        df.attrs['data_source'] = 'synthetic'
        df.attrs['data_source_label'] = 'SYNTHETIC random-walk data (NOT real market data)'
        df.attrs['symbol'] = symbol
        df.attrs['timeframe'] = timeframe
        df.attrs['fetched_at'] = datetime.now().isoformat()
        return df

def _fetch_from_binance(symbol, timeframe, start, end, original_start, original_end):
    """Attempt to fetch real data from Binance"""
    try:
        ex = ccxt.binance()
        since = ex.parse8601(start.isoformat())
        end_ts = ex.parse8601(end.isoformat())
        
        # Validate date range
        if since >= end_ts:
            raise ValueError("Start date must be before end date")
        
        rows = []
        logger.info(f"Fetching data for {symbol} from {start} to {end} (original request: {original_start} to {original_end})")
        
        # Calculate expected data points based on timeframe and date range
        timeframe_minutes = ex.parse_timeframe(timeframe) / 60  # Convert to minutes
        date_range_days = (end - start).total_seconds() / (24 * 3600)  # Convert to days
        expected_points = int((date_range_days * 24 * 60) / timeframe_minutes)
        
        logger.info(f"Expected data points for {timeframe} over {date_range_days:.1f} days: ~{expected_points}")
        
        # Use appropriate batch size (max 1000 for CCXT)
        batch_size = min(1000, expected_points + 100)  # Add buffer for safety
        
        while since < end_ts:
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
                if not batch:
                    break
                rows += batch
                since = batch[-1][0] + ex.parse_timeframe(timeframe) * 1000
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
                # Break if we have enough data to avoid over-fetching
                if len(rows) >= expected_points * 1.5:  # 50% buffer
                    logger.info(f"Fetched sufficient data: {len(rows)} points")
                    break
                
            except Exception as api_error:
                logger.error(f"API error fetching batch: {api_error}")
                break
        
        if not rows:
            raise ValueError(f"No data returned for {symbol} from Binance API.")
        
        df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df.ts, unit='ms')
        df.set_index('ts', inplace=True)
        
        # Remove any duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        # Ensure numeric data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN prices (critical data)
        df = df.dropna(subset=['close'])
        
        return _process_technical_indicators(df, original_start, original_end)
        
    except Exception as e:
        logger.error(f"Binance API error: {e}")
        raise

def _generate_simulated_data(symbol, timeframe, start, end, original_start, original_end):
    """Generate simulated market data for testing when real data is unavailable"""
    try:
        logger.info(f"Generating simulated data for {symbol} from {start} to {end}")
        
        # Determine timeframe in minutes
        timeframe_minutes = 1
        if timeframe == '5m':
            timeframe_minutes = 5
        elif timeframe == '15m':
            timeframe_minutes = 15
        elif timeframe == '1h':
            timeframe_minutes = 60
        elif timeframe == '4h':
            timeframe_minutes = 240
        elif timeframe == '1d':
            timeframe_minutes = 1440
        
        # Create time series
        time_range = pd.date_range(start=start, end=end, freq=f'{timeframe_minutes}min')
        
        if len(time_range) < 10:
            # Ensure we have at least 10 data points
            time_range = pd.date_range(start=start, periods=100, freq=f'{timeframe_minutes}min')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        
        # Starting price based on symbol
        if 'BTC' in symbol.upper():
            base_price = 45000  # Bitcoin around $45k
        elif 'ETH' in symbol.upper():
            base_price = 2500   # Ethereum around $2.5k
        else:
            base_price = 100    # Generic price
        
        # Generate random walk with trend and volatility
        n_points = len(time_range)
        returns = np.random.normal(0.0001, 0.02, n_points)  # Small positive drift with volatility
        
        # Add some market structure (trend changes)
        for i in range(0, n_points, max(1, n_points // 10)):
            trend_change = np.random.choice([-1, 1]) * np.random.uniform(0.001, 0.005)
            end_idx = min(i + n_points // 10, n_points)
            returns[i:end_idx] += trend_change
        
        # Calculate prices
        log_prices = np.log(base_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, close) in enumerate(zip(time_range, prices)):
            # Generate realistic OHLC from close price
            volatility = abs(returns[i]) * close
            
            high = close + np.random.uniform(0, volatility * 2)
            low = close - np.random.uniform(0, volatility * 2)
            
            # Ensure logical OHLC relationships
            high = max(high, close)
            low = min(low, close)
            
            # Generate open price close to previous close
            if i == 0:
                open_price = close * np.random.uniform(0.995, 1.005)
            else:
                open_price = prices[i-1] * np.random.uniform(0.998, 1.002)
            
            # Adjust high/low to include open
            high = max(high, open_price)
            low = min(low, open_price)
            
            # Generate volume
            volume = np.random.uniform(1000, 10000) * (1 + abs(returns[i]) * 10)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data, index=time_range)
        
        logger.info(f"Generated {len(df)} simulated data points")
        
        return _process_technical_indicators(df, original_start, original_end)
        
    except Exception as e:
        logger.error(f"Error generating simulated data: {e}")
        raise

def _process_technical_indicators(df, original_start, original_end):
    """Process technical indicators for the given dataframe"""
    try:
        # Ensure we have enough data for indicators
        min_required = 60  # Need enough for MA50 + some buffer
        if len(df) < min_required:
            # Try to generate more data points if needed
            if len(df) < 10:
                raise ValueError(f"Insufficient data: only {len(df)} rows. Need at least 10 for trading simulation.")
            else:
                logger.warning(f"Limited data ({len(df)} rows), some indicators may be less reliable")
        
        logger.info(f"Processing technical indicators for {len(df)} rows of data")
        
        # Debug: Check data quality before indicators
        logger.info(f"Data sample before indicators - Close price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")
        
        # Technical indicators with error handling
        try:
            # Simple Moving Averages
            sma20 = SMAIndicator(close=df['close'], window=20)
            df['MA20'] = sma20.sma_indicator()
            
            # Add MA50 for trend analysis (only if we have enough data)
            if len(df) >= 50:
                sma50 = SMAIndicator(close=df['close'], window=50)
                df['MA50'] = sma50.sma_indicator()
            else:
                # Fallback to MA20 for MA50 if insufficient data
                df['MA50'] = df['MA20']
                logger.warning("Using MA20 as fallback for MA50 due to insufficient data")
            
            # Debug: Check MA calculation
            logger.info(f"MA20 calculated - valid values: {df['MA20'].notna().sum()}/{len(df)}")
            
            # Bollinger Bands
            bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['UpperBB'] = bb_indicator.bollinger_hband()
            df['MidBB'] = bb_indicator.bollinger_mavg()
            df['LowerBB'] = bb_indicator.bollinger_lband()
            
            # Debug: Check BB calculation
            logger.info(f"Bollinger Bands calculated - Upper valid: {df['UpperBB'].notna().sum()}, Lower valid: {df['LowerBB'].notna().sum()}")
            
            # RSI
            rsi_indicator = RSIIndicator(close=df['close'], window=14)
            df['RSI'] = rsi_indicator.rsi()
            
            # Debug: Check RSI calculation
            logger.info(f"RSI calculated - valid values: {df['RSI'].notna().sum()}/{len(df)}, range: {df['RSI'].min():.1f} to {df['RSI'].max():.1f}")
            
            # MACD - Enhanced with all components
            macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['MACD_line'] = macd_indicator.macd()  # MACD line
            df['MACD_signal'] = macd_indicator.macd_signal()  # Signal line
            df['MACD_hist'] = macd_indicator.macd_diff()  # Histogram (MACD - Signal)
            
            # Debug: Check MACD calculation
            logger.info(f"MACD calculated - Line valid: {df['MACD_line'].notna().sum()}, Signal valid: {df['MACD_signal'].notna().sum()}, Hist valid: {df['MACD_hist'].notna().sum()}")
            
        except Exception as indicator_error:
            logger.error(f"Error calculating technical indicators: {indicator_error}")
            raise ValueError(f"Failed to calculate technical indicators: {indicator_error}")
        
        # More robust NaN checking and cleaning
        critical_indicators = ['MA20', 'RSI', 'UpperBB', 'LowerBB']
        
        # First, fill NaN values with forward/backward fill for technical indicators
        # This handles the initial periods where indicators can't be calculated
        for indicator in critical_indicators:
            if indicator in df.columns:
                df[indicator] = df[indicator].ffill().bfill()
        
        # Check if we still have significant NaN values after filling
        nan_counts = {}
        for indicator in critical_indicators:
            if indicator in df.columns:
                nan_count = df[indicator].isna().sum()
                nan_pct = (nan_count / len(df)) * 100
                nan_counts[indicator] = {'count': nan_count, 'percentage': nan_pct}
                
                # Only raise error if more than 50% of values are NaN after filling
                if nan_pct > 50:
                    logger.error(f"High NaN percentage for {indicator}: {nan_pct:.1f}%")
                    raise ValueError(f"Too many NaN values for {indicator}: {nan_pct:.1f}% of data")
        
        logger.info(f"NaN counts after filling: {nan_counts}")
        
        # Drop rows where ANY critical indicator is still NaN (but be less aggressive)
        before_drop = len(df)
        df = df.dropna(subset=critical_indicators, how='any')
        after_drop = len(df)
        
        dropped_rows = before_drop - after_drop
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with remaining NaN values")
        
        logger.info(f"After cleaning data: {after_drop} rows")
        
        # Final validation - be more lenient
        if df.empty or len(df) < 10:
            raise ValueError(f"Insufficient clean data after processing: only {len(df)} rows. Need at least 10 for trading simulation.")
        
        # Final fill of any remaining NaN values
        df = df.ffill().bfill()
        
        # Filter dataframe to ORIGINAL user-requested date range
        # Convert original dates to timezone-aware if the dataframe index is timezone-aware
        filter_start = original_start
        filter_end = original_end
        
        if df.index.tz is not None:
            if filter_start.tzinfo is None:
                filter_start = filter_start.replace(tzinfo=df.index.tz)
            if filter_end.tzinfo is None:
                filter_end = filter_end.replace(tzinfo=df.index.tz)
        
        # Filter to ORIGINAL requested date range (not the extended range)
        original_len = len(df)
        df = df[filter_start:filter_end]
        filtered_len = len(df)
        
        if filtered_len < original_len:
            logger.info(f"Filtered data to ORIGINAL requested date range ({original_start} to {original_end}): {original_len} -> {filtered_len} rows")
        
        # For very short periods (like 1 day), ensure we have at least some data
        if len(df) < 5:
            logger.warning(f"Very limited data after filtering ({len(df)} rows). Results may be less reliable for short time periods.")
        
        logger.info(f"Successfully processed {len(df)} rows with technical indicators for original date range {original_start} to {original_end}")
        return df
        
    except Exception as e:
        logger.error(f"Error processing technical indicators: {e}")
        raise

