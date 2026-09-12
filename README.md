# 🤖 AI-Powered Multi-Agent Trading System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)](https://openai.com/)

An AI trading research system that combines **multi-agent coordination**, **exogenous non-price signals** (text sentiment and futures positioning) and **technical analysis** into auditable cryptocurrency trading decisions. Built with LangChain and Streamlit, on live Binance market data, with point-in-time discipline and a factorial ablation harness.

![Trading System Demo](https://img.shields.io/badge/Demo-Live-green) 

## ✨ Key Features

### 🚀 **Advanced AI Trading**
- **Multi-Agent Architecture**: Coordinated agents for technical analysis, text sentiment, futures positioning and risk management
- **Text Sentiment**: Real Hacker News posts and comments (keyless Algolia API), scored with VADER plus a crypto lexicon, or optionally CryptoBERT
- **Futures Positioning**: Free Binance USD-M perpetual positioning metrics — crowd long/short faded, top-trader and taker flow followed
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages with professional TradingView-style charts
- **Smart Decision Making**: LangChain-powered decision engine with reasoning transparency

> **Removed:** an earlier "real-time sentiment analysis of influential accounts
> (Elon Musk, Donald Trump)" feature. It generated its own posts, attributed
> them to real named people, and — once synthetic data was switched off —
> returned a hard `0.0` on every bar while still occupying two large UI panels.
> It has been deleted outright rather than left switchable. See
> [Exogenous Signal: Text Sentiment](#-exogenous-signal-text-sentiment-hacker-news)
> for what replaced it.

### 📊 **Professional Trading Interface**
- **Interactive Dashboard**: Streamlit-based web interface with real-time updates
- **Trade History & P&L**: Comprehensive trade tracking with profit/loss analysis
- **Per-Channel Attribution**: How many decisions each exogenous channel actually moved, and how many the LLM changed
- **Performance Metrics**: Win rate, drawdown analysis, portfolio performance vs buy-and-hold
- **Technical Charts**: Professional candlestick charts with trading signals and indicators

### 🎯 **Smart Features**
- **Point-in-Time Signals**: Every exogenous channel is lagged and z-scored against its own trailing baseline, so no bar can see its own future
- **Risk Management**: Configurable position sizing and stop-loss mechanisms
- **Historical Analysis**: Review past simulation results and trading performance
- **Real-Time Reasoning**: Transparent agent decision-making process

### 🛠 **Technical Architecture**
- **LangChain Framework**: Structured agent orchestration and tool management
- **Real-Time Data**: Live cryptocurrency data via CCXT and Binance API
- **Advanced Analytics**: Technical analysis with TA-Lib integration
- **Secure Configuration**: Environment-based API key management


## 📋 Prerequisites

- **Python 3.8+** (Recommended: Python 3.9 or 3.10)
- **OpenAI API Key** (for AI decision-making agent)
- **Internet Connection** (for real-time market data)
- **Optional**: PostgreSQL database (for persistent result storage)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-trading-system.git
cd ai-trading-system
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv trading-env
source trading-env/bin/activate  # On Windows: trading-env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Create environment file
cp .env.example .env

# Edit with your settings
nano .env
```

**Required environment variables:**
```env
OPENAI_API_KEY=your_openai_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

### 4. Run the Application

```bash
streamlit run auto-trade.py
```

Navigate to `http://localhost:8501` in your browser to access the trading interface.

### 5. Run Tests (Optional)

```bash
# Quick validation test
python test/test_quick_validation.py

# Comprehensive test suite
python test/test_runner.py

# Exogenous positioning signal (41 offline unit tests)
python test/test_binance_positioning.py

# Exogenous text sentiment (40 offline unit tests)
python test/test_text_sentiment.py

# Signal wiring: do the signals actually reach a trade? (27 tests)
python test/test_positioning_integration.py
```

## 📡 Exogenous Signal: Binance Futures Positioning

The social-sentiment channel this project shipped with was **decorative**: the
backtest loop requested synthetic posts, the Phase 2 integrity guards correctly
switched synthetic data off, and the sentiment score was therefore a hard `0.0`
on every bar. A constant cannot change a trade.

Buying an X/Twitter API tier does not fix that, because a backtest needs posts
at *specific past timestamps* and the affordable tiers only serve a 7-day
recent-search window. Instead, the channel is replaced with **Binance USD-M
futures positioning** — free, keyless, and reproducible.

### Why this source

| Property | Detail |
|---|---|
| Cost | Free, no account, no API key, no rate limit |
| Endpoint | `data.binance.vision/data/futures/um/daily/metrics/` |
| Resolution | 5-minute, so mapping onto 1h bars is downsampling, never interpolation |
| Reproducibility | **Static files.** A reviewer re-downloading gets identical bytes |
| Integrity | Binance publishes a `.CHECKSUM` per file; the fetcher verifies it and refuses to cache a mismatch |
| Construct | *Revealed* preference (what traders did with money) rather than *stated* preference (what they posted) |

Features used, with a fixed and deliberately un-tuned sign convention:

| Column | Reading | Direction |
|---|---|---|
| `count_long_short_ratio` | crowd account skew | **contrarian** (faded) |
| `sum_toptrader_long_short_ratio` | top-trader position skew | **momentum** (followed) |
| `sum_taker_long_short_vol_ratio` | aggressive taker flow | **momentum** (followed) |
| `sum_open_interest` | crowding intensity | non-directional, diagnostic only |

### Point-in-time discipline

The bar whose **open** time is `t` sees only rows with
`create_time <= t - POSITIONING_LAG_BARS * bar_duration`. With the default lag
of 1 bar, positioning is lagged strictly more than price (the price path already
uses that bar's *close*), so the signal cannot manufacture a look-ahead
advantage. Rolling z-scores use right-aligned `pandas.rolling`, which is causal.
Both properties are asserted in `test/test_binance_positioning.py`, including a
truncation test that fails if any future observation leaks into an earlier
window.

### Running it — nothing extra to do

```bash
streamlit run auto-trade.py
```

That is the whole workflow. On the first run over a new date range the app
downloads the positioning days it needs (showing a progress bar), caches them,
and every later run over that range is a pure cache hit. It also fetches the
z-score warm-up window, so the requested period is usable from its **first**
bar rather than losing its first `POSITIONING_ZSCORE_WINDOW` bars to warm-up.

Behind a TLS-inspecting corporate proxy, the fetcher detects the `SSLError`,
builds a merged certifi + Windows-trust-store CA bundle once, and retries. This
is applied per-request rather than by setting `REQUESTS_CA_BUNDLE` globally, so
components that already work — ccxt price downloads, the Azure OpenAI client —
are untouched. `verify=False` is never used: it would make the data-provenance
claim unverifiable.

### Optional CLI (for inspection and frozen runs)

```bash
# Confirm reachability and that the upstream schema still matches
python -m signals.binance_positioning probe --symbol BTCUSDT --date 2024-01-02

# Pre-populate the cache (idempotent; re-run to fill gaps)
python -m signals.binance_positioning fetch --symbol BTCUSDT \
    --start 2024-01-01 --end 2024-03-31

# Coverage, feature distributions and how often the signal would fire
python -m signals.binance_positioning report --symbol BTCUSDT

# Rebuild the corporate CA bundle by hand, if the auto-heal ever fails
python -m signals.corporate_ca
```

Raw zips are gitignored because `fetch` reconstructs them byte-for-byte;
`manifest.jsonl` is committed and carries the URL, pull timestamp and SHA-256
of every file.

### Configuration

```bash
USE_POSITIONING_SIGNAL=true      # false = exogenous-signal-off ablation arm
POSITIONING_AUTO_FETCH=true      # false = frozen offline run (use for the paper)
POSITIONING_MAX_POINTS=2         # cap; 2 matches the weight of the RSI rule
POSITIONING_LAG_BARS=1           # extra lag beyond the point-in-time cutoff
POSITIONING_ZSCORE_WINDOW=168    # causal z-score window in bars (1 week of 1h)
```

Set `USE_POSITIONING_SIGNAL=false` to get the exogenous-signal-off arm, which
reproduces the pre-Phase-3 behaviour exactly — that is what keeps the ablation
arms comparable. For the final numbers in a write-up, pre-populate the cache
with `fetch` and set `POSITIONING_AUTO_FETCH=false`, so the run touches the
network zero times and is replayable offline.

Positioning enters the **rule engine**, not only the LLM prompt. That
distinction is what makes the ablation identifiable: a signal that reached the
decision solely through the prompt would contribute nothing in the
`USE_LLM_DECISIONS=false` arm, and "positioning adds information" could not be
separated from "the LLM adds information". Every run reports
`positioning_report`, including the share of decisions the channel actually
moved, so a channel that fired on zero bars is visibly distinguishable from one
that fired and simply did not help.

### Two caveats that belong in any write-up

1. **Cross-market.** These metrics describe the USD-M *perpetual futures*
   market while the backtest trades *spot*. Using futures positioning as a
   signal for spot is standard practice, but it is a modelling choice and
   should be disclosed.
2. **Coverage.** History depends on when Binance began publishing per symbol
   (roughly 2020 for BTCUSDT) and the newest day arrives with a lag. Trust the
   `report` output over any assumed range: on Q1 2024 BTCUSDT it yields 26,083
   raw 5-minute rows, 96–98% usable hourly bars after the z-score warm-up, and
   points added on roughly 15% of bars.

## 💬 Exogenous Signal: Text Sentiment (Hacker News)

The second exogenous channel, and the one that finally makes "sentiment" mean
something here: **real public posts, with real timestamps, scored by a real
model**. It replaces the synthetic-tweet path entirely.

### Why Hacker News

Every obvious source was measured from this project's network on 2026-09-09:

| Source | Result |
|---|---|
| Reddit (`.json`, search) | **HTTP 403** — blocked by the corporate proxy |
| StockTwits | **HTTP 403** — blocked by the corporate proxy |
| CryptoPanic | **HTTP 403** — blocked by the corporate proxy |
| GDELT | **HTTP 429** — rate-limits the proxy's shared egress IP, even after 20s idle |
| CryptoCompare news | **HTTP 401** — now requires an API key |
| X / Twitter | Paid tiers cannot serve historical posts for a backtest anyway |
| **Hacker News (Algolia)** | **HTTP 200** — free, keyless, complete archive back to 2007 |

Hacker News is a smaller and more technical crowd than Reddit. That is a real
limitation and belongs in any write-up — but it is genuine, timestamped, public
text that a reviewer can re-fetch, which none of the blocked options are.

**Measured on the cached corpus (2023-12-20 → 2024-03-31):** 9,036 unique
documents, 87.7/day, 8,186 comments and 850 stories, mean length 425 characters.

### Scoring: VADER now, CryptoBERT optionally

| Backend | Setup | Notes |
|---|---|---|
| `vader` (default) | none — already installed | VADER compound score, plus a fixed 42-term crypto lexicon |
| `cryptobert` | `pip install transformers torch` **+ model files** | `ElKulako/cryptobert`, RoBERTa fine-tuned on crypto social posts |

> **CryptoBERT cannot be fetched on the development network.** Measured
> 2026-09-11: `huggingface.co` returns HTTP 403 from Cato with
> `error: "Corporate Internet policy violation", categories: "… Generative AI
> Tools"`, and `cdn-lfs.huggingface.co` does not resolve. `transformers` and
> `torch` install normally — only the weights are unreachable. To use it,
> either have `huggingface.co` allowlisted, or copy the model directory in by
> hand and set `CRYPTOBERT_MODEL_PATH` to it. The fallback to VADER is loud.

VADER is a 2014 general-purpose lexicon that cannot read crypto register —
`rekt`, `hodl`, `rug pull`, `diamond hands` are all invisible to it. The
included lexicon patch fixes the worst of that and is fixed rather than tuned
against returns. CryptoBERT is the proper fix, and running both gives a free
ablation row. **An unavailable CryptoBERT falls back to VADER with a warning,
never silently** — a run scored by a different model is a different experiment.

### Point-in-time discipline

The bar whose **open** time is `t` averages documents created in
`[t - lag - window, t - lag)`. Nothing published at or after the cutoff can
reach the bar. Because Hacker News yields only tens of documents a day, readings
aggregate over a trailing window (default 24h) rather than per bar, and a window
holding fewer than `TEXT_SENTIMENT_MIN_DOCS` documents reports *no reading*
instead of acting on noise.

The score is the **causal z-score** of the window mean against its own trailing
baseline, not the raw mean. General text carries a persistent positive drift
(the measured corpus mean is `+0.176`); the z-score removes it and makes this
comparable with the positioning score.

### Configuration

```bash
USE_TEXT_SENTIMENT=true          # false = text-sentiment-off ablation arm
TEXT_SENTIMENT_SCORER=vader      # or cryptobert
TEXT_SENTIMENT_QUERIES=bitcoin,crypto
TEXT_SENTIMENT_MAX_POINTS=1      # half the positioning cap; see below
TEXT_SENTIMENT_LAG_BARS=1
TEXT_SENTIMENT_WINDOW_HOURS=24
TEXT_SENTIMENT_ZSCORE_WINDOW=168
TEXT_SENTIMENT_MIN_DOCS=5        # below this, report "no reading"
TEXT_SENTIMENT_AUTO_FETCH=true   # false = frozen offline run
```

`btc` is deliberately **not** a default query: it matched 1,185 documents in a
single day during probing, i.e. it matches substrings and unrelated tokens
rather than the asset.

`TEXT_SENTIMENT_MAX_POINTS` defaults to **1, half the positioning cap**. This is
a stated prior, not a fitted parameter: text sentiment is a single noisy feature
that crosses its 1-sigma threshold on ~36% of bars, whereas positioning averages
three features that often disagree and fires on ~15%. Giving the noisier channel
equal weight would let it dominate the technical rule engine.

### Measured behaviour (BTC, 2024-01-05 → 2024-03-25, 1,944 hourly bars)

```
7,946 documents; usable on 100.0% of bars (vader, 24h window, lag 1 bar)
median 72 documents per bar
706 bars would add points  (324 bullish / 382 bearish)
```

Like positioning, it enters the **rule engine** as well as the LLM prompt, so
it is identifiable in the `USE_LLM_DECISIONS=false` arm. Every run reports
`text_sentiment_report`.

### CLI

```bash
python -m signals.text_sentiment probe --query bitcoin --date 2024-01-15
python -m signals.text_sentiment fetch --start 2024-01-01 --end 2024-03-31
python -m signals.text_sentiment report --scorer vader
```

## 📁 Project Structure

```
core/                    production logic, imports NO Streamlit
  config.py              every runtime flag + domain types, one place
  llm.py                 provider setup + the structured LLM contract
  data.py                market data + technical indicators
  execution.py           fill pricing, fees, slippage
signals/                 exogenous data sources, independent of core
  binance_positioning.py futures positioning (free, keyless)
  text_sentiment.py      Hacker News text sentiment (free, keyless)
  corporate_ca.py        CA bundle builder for TLS-inspecting proxies
experiments/
  harness.py             headless app loader + Streamlit stub
  run_ablation.py        factorial ablation runner, one JSONL row per arm
test/
  golden_backtest.py     proves a refactor changed no numbers
auto-trade.py            agents, the simulation loop and the Streamlit UI
```

The layering is strictly one-directional: `core` → nothing in-project,
`signals` → nothing in-project, `auto-trade.py` → both.

**`core` imports no Streamlit.** That single property is what makes a headless
run possible, which the ablation table needs. Provider errors still reach the
user: `core.llm.set_error_reporter(st.error)` is injected by the app at startup,
so the red banners appear in the UI while a CLI run gets the message in the log.

### Verifying a change moved no numbers

```bash
python test/golden_backtest.py --save    # before
python test/golden_backtest.py --check   # after
```

A deterministic rules-only backtest over a fixed cached window, fingerprinted
down to individual trade prices. `--check` prints a field-by-field diff and
exits non-zero if anything moved. It runs in ~50 seconds because the three
LLM-backed support agents are stubbed; see the caveat below for why that is
necessary rather than merely convenient.

### Running the ablation

```bash
python -m experiments.run_ablation --arms rules_only,pos_only,text_only,pos_text
```

Each arm runs in a **fresh subprocess** with its own environment, because the
feature flags are read at import time by `core.config` and cannot be changed by
mutating a module attribute afterwards. That also guarantees no state leaks
between arms (cached LLM clients, agent instances, warmed z-scores).

`USE_LLM_DECISIONS=false` gates **all four** model-backed agents — decision,
market, pattern and risk — so an `llm=false` arm makes zero model calls, needs
no API key, runs in seconds and is bit-reproducible. `--stub-support-agents` is
now only meaningful for `llm=true` arms, where it pins the narrator agents so
the measured effect is the *decision* agent's alone.

> ### ⚠️ Any ablation table produced before 2026-09-11 is contaminated
>
> Until then the flag gated only `TradingDecisionAgent`, and the risk agent's
> output was not inert: `risk_level` drives a confidence multiplier
> (`{"low": 1.2, "high": 0.8}`) applied to
> `min(0.9, 0.6 + net_signal * 0.1)`. At `net_signal = 2` that is `0.80`, so a
> `"high"` reading gives `0.64` — just below the `0.65` moderate-mode gate —
> turning a **BUY into a HOLD**. A single LLM word could flip a trade inside
> the arm that was meant to contain no LLM.
>
> Verified after the fix: the golden harness now runs the **real, unstubbed**
> path and still reports `GOLDEN MATCH`, and the four model-free arms reproduce
> their previously stubbed numbers exactly — including with the Azure
> credentials blanked. Regression suite: `python test/test_llm_gating.py`.

## 🏗 Architecture Overview

### Multi-Agent System Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                     AMAAI Trading System Architecture                │
├──────────────────────────────────────────────────────────────────────┤
│  SIGNAL SOURCES (all point-in-time, all lagged and z-scored)         │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────────────┐    │
│  │  Technical    │  │ Text          │  │ Futures Positioning    │    │
│  │  Indicators   │  │ Sentiment     │  │ (Binance USD-M)        │    │
│  │  (from price) │  │ (Hacker News) │  │ crowd / top / taker    │    │
│  │               │  │ VADER|CryptoBERT│ │ ~1 day publish lag    │    │
│  └───────┬───────┘  └───────┬───────┘  └───────────┬────────────┘    │
│          │                  │                      │                 │
│          └──────────────────┼──────────────────────┘                 │
│                             ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  RULE ENGINE  →  net_signal (integer, auditable)               │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  TradingDecisionAgent (LLM)                                    │  │
│  │  returns a CLAMPED integer adjustment + optional risk veto,    │  │
│  │  so the model's contribution is measured, not assumed.         │  │
│  │  USE_LLM_DECISIONS=false skips it -> rules-only baseline       │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               ▼                                      │
│  EXECUTION: fill at NEXT bar's open + slippage (never this close)    │
├──────────────────────────────────────────────────────────────────────┤
│  Support agents (Market / Pattern / Risk) also call the LLM, and are │
│  gated by the SAME flag, so llm=false makes zero model calls.        │
└──────────────────────────────────────────────────────────────────────┘
```

### Code layout

| Path | Imports Streamlit? | Contents |
|---|---|---|
| `core/` | **no** | `config` (every flag + domain type), `llm`, `data`, `execution` |
| `signals/` | **no** | `binance_positioning`, `text_sentiment`, `corporate_ca` |
| `experiments/` | no | `harness` (headless scaffolding), `run_ablation` |
| `test/` | no | `golden_backtest` + unit suites |
| `auto-trade.py` | yes | agents, simulation loop, Streamlit UI |

`core` and `signals` importing no Streamlit is what makes headless, scripted
runs possible; it is asserted by the ablation harness rather than trusted.

## 📱 Usage Guide

### Starting a Trading Simulation

1. **Configure Parameters**
   - Set initial capital (default: $10,000)
   - Choose time period for backtesting
   - Select cryptocurrency pair (e.g., BTC/USDT)
   - Fees and slippage come from `.env`, not the sidebar

2. **Run Simulation**
   - Click "Start Trading Simulation"
   - Monitor real-time agent decisions
   - View sentiment analysis and technical indicators
   - Track portfolio performance

3. **Analyze Results**
   - Review trade history with P&L analysis
   - Compare performance vs buy-and-hold
   - Examine per-channel attribution (which signals moved which decisions)
   - Export results for further analysis

### Key Interface Features

#### 🔮 **Next Action Recommendation**
- **Multi-agent voting system** for trading decisions
- **Technical Analysis Agent**: RSI, MACD, moving averages
- **Text Sentiment Agent**: Hacker News score, document count, and the exact rule points it contributed
- **Futures Positioning Agent**: per-feature z-scores against each feature's own trailing baseline
- **Risk Management Agent**: Position sizing and risk assessment
- **Final recommendation** with confidence scores

Each agent panel states whether it had a usable reading, and an unavailable
channel contributes nothing rather than being counted as a neutral vote.

#### 📊 **Technical Analysis Dashboard**
- **Professional TradingView-style charts**
- **Candlestick patterns** with volume analysis
- **Technical indicators**: Bollinger Bands, RSI, MACD
- **Buy/sell signals** overlaid on price charts

#### 💬 **Text Sentiment**
- **Real posts and comments** from Hacker News via the keyless Algolia API
- **VADER + 42-term crypto lexicon**, or CryptoBERT when `TEXT_SENTIMENT_SCORER=cryptobert`
- **Causal z-scores** over a half-open `[cutoff - window, cutoff)` interval
- **Byte-reproducible cache** with a `manifest.jsonl` recording URL, pull time, sha256 and document counts

> Futures positioning is published by Binance as one file per **completed**
> day, so the most recent 24-48 hours are normally absent. That channel can
> inform a backtest but never a live next-bar decision.

#### 📈 **Performance Analytics**
- **Portfolio value tracking** over time
- **Win rate and profit metrics**
- **Drawdown analysis**
- **Comparison with market benchmarks**

## ⚙️ Configuration Options

### Trading Parameters

```bash
# Set in .env (capital and position sizing are also on the Streamlit sidebar)
INITIAL_CAPITAL=10000     # Starting portfolio value
BUY_FEE_PCT=0.1           # Buy fee, percent per side
SELL_FEE_PCT=0.1          # Sell fee, percent per side
SLIPPAGE_PCT=0.05         # Slippage, percent per side, charged on top
MAX_POSITION_SIZE=0.3     # Maximum 30% of portfolio per trade
STOP_LOSS_PCT=0.05        # 5% stop-loss threshold
```

> **Fees are charged on both sides.** Until 2026-09-11 `sell_fee_pct` defaulted
> to `0.0` and neither fee variable was read from `.env` at all, so every
> backtest paid to enter and exited free. On the golden window that was $127.88
> of unmodelled cost against $281.11 of reported profit — the reported return
> fell from **2.81% to 1.53%** once it was fixed. Any figure produced by this
> repository before that date is inflated by roughly that much.
> Regression suite: `python test/test_execution_fees.py`.

### Environment Variables

```env
# API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini  # or gpt-4, gpt-3.5-turbo

# Database Configuration (Optional)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_results
DB_USER=postgres
DB_PASSWORD=your_password

# Trading Configuration
DEFAULT_SYMBOL=BTC/USDT
DEFAULT_TIMEFRAME=1h
SENTIMENT_THRESHOLD=0.2
RISK_TOLERANCE=medium
```

## 🧪 Testing

### Quick Validation

Run the quick validation test to ensure everything is working:

```bash
python test/test_quick_validation.py
```

### Comprehensive Test Suite

For thorough testing of all components:

```bash
python test/test_runner.py
```

### Test Coverage

- ✅ Package imports and dependencies
- ✅ Main module and agent initialization  
- ✅ Data fetching and technical analysis
- ✅ Text sentiment scoring (direction, not just "returns a dict")
- ✅ Trading simulation logic
- ✅ P&L calculation accuracy
- ✅ Display functions and UI components

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Key Performance Indicators
- **Total Return**: Overall strategy performance vs initial capital
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Typical holding period
- **Profit Factor**: Ratio of gross profit to gross loss

### Comparative Analysis
- **Strategy vs Buy & Hold**: Performance comparison
- **Risk Metrics**: Volatility and risk analysis
- **Trade Distribution**: Win/loss breakdown
- **Channel Attribution**: How many decisions each exogenous channel moved, and how many the LLM changed

## 🚀 Deployment

### Local Development

```bash
# Development mode with hot reload
streamlit run auto-trade.py --server.runOnSave true
```

### Production Deployment

#### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "auto-trade.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t ai-trading-system .
docker run -p 8501:8501 --env-file .env ai-trading-system
```

#### Cloud Deployment (Streamlit Cloud)

1. Push code to GitHub repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Configure environment variables in Streamlit Cloud dashboard
4. Deploy with one click

### Environment Security

- **Never commit API keys** to version control
- **Use environment variables** for all sensitive data
- **Enable rate limiting** for production deployments
- **Monitor API usage** and costs

## 🛠 Development

### Project Structure

```
ai-trading-system/
├── auto-trade.py              # Main application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── .env.example             # Environment template
├── test/                    # Test suite
│   ├── test_runner.py       # Comprehensive tests
│   ├── test_quick_validation.py  # Quick validation
│   └── *.py                 # Individual test files
├── docs/                    # Documentation
└── .gitignore              # Git ignore rules
```

### Adding New Features

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Follow coding standards**: Use type hints and docstrings
3. **Add tests**: Update test suite for new functionality
4. **Update documentation**: Modify README and code comments
5. **Submit pull request**: Include description and test results

### Code Style

- **PEP 8 compliance** for Python code formatting
- **Type hints** for all function parameters and returns
- **Comprehensive docstrings** for classes and methods
- **Error handling** with try/catch blocks
- **Logging** for debugging and monitoring

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Add features, fix bugs, improve documentation
4. **Add tests**: Ensure your changes are tested
5. **Run test suite**: `python test/test_runner.py`
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open Pull Request**: Describe your changes and provide test results

### Contribution Areas

- 🔧 **Core Features**: Trading algorithms, agent improvements
- 📊 **Analytics**: New performance metrics, visualization enhancements
- 🎨 **UI/UX**: Streamlit interface improvements
- 📚 **Documentation**: README, code comments, tutorials
- 🧪 **Testing**: Additional test cases, performance testing
- 🔒 **Security**: Security improvements, code reviews

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ai-trading-system.git
cd ai-trading-system

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests before making changes
python test/test_runner.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This is educational software for learning about algorithmic trading and AI systems.**

- **Not Financial Advice**: This system is for educational and research purposes only
- **No Investment Recommendations**: Do not use for actual trading without proper due diligence
- **Risk Warning**: Cryptocurrency trading involves substantial risk of loss
- **Use at Your Own Risk**: Authors are not responsible for any financial losses

## 📞 Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/yourusername/ai-trading-system/issues)
- **Discussions**: Join the conversation in [GitHub Discussions](https://github.com/yourusername/ai-trading-system/discussions)
- **Documentation**: Check the [Wiki](https://github.com/yourusername/ai-trading-system/wiki) for detailed guides

## 🙏 Acknowledgments

- **OpenAI** for GPT API and language models
- **LangChain** for agent framework and tools
- **Streamlit** for the excellent web framework
- **CCXT** for cryptocurrency exchange connectivity
- **TA-Lib** for technical analysis indicators
- **Plotly** for interactive charting capabilities

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
│           └─────────────────────┼───────────────────┘       │
│                                 │                           │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │          Trading Decision Agent (OpenAI)                │  │
│  │         - Synthesizes all agent inputs                 │  │
│  │         - Makes final BUY/SELL/HOLD decisions          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                 │                           │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ Deep Learning   │  │ Vector Database │                   │
│  │ Agent           │  │ Agent           │                   │
│  │ (TensorFlow)    │  │ (FAISS)         │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Fast Local Agents** (No OpenAI calls)
   - Market Analyst: Technical indicator analysis
   - Pattern Recognition: Rule-based chart patterns
   - Risk Management: Position sizing and risk metrics

2. **AI Decision Maker** (OpenAI-powered)
   - Synthesizes all agent inputs
   - Makes final trading decisions
   - Provides detailed reasoning

3. **Machine Learning Components**
   - Deep Learning Agent: Neural network for pattern prediction
   - Vector Database: Historical pattern storage and retrieval

## 🎛 Configuration Options

### Trading Parameters

```python
config = TradingConfig(
    initial_capital=1000.0,      # Starting capital
    buy_fee_pct=0.10,            # 0.1% fee per buy (percent, not fraction)
    sell_fee_pct=0.10,           # 0.1% fee per sell — symmetric
    enable_deep_learning=True,    # Enable ML predictions
    enable_vector_db=True,        # Enable pattern storage
    show_reasoning=True           # Show agent reasoning panel
)
```

### Advanced Settings

- **RSI Thresholds**: Oversold/overbought levels
- **Stop Loss/Take Profit**: Risk management parameters
- **Model Training**: Deep learning configuration
- **Vector Database**: Pattern similarity settings

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

### Key Metrics
- **Total Return**: Overall strategy performance
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Trade Statistics**: Count, frequency, and success rate

### Real-time Monitoring
- Live portfolio value tracking
- Agent reasoning display
- Trade execution visualization
- Performance comparison charts

## 🔍 Agent Reasoning Panel

The right-side panel shows real-time agent decision-making:

- **📈 Market Analysis**: Technical indicator insights
- **🔍 Pattern Recognition**: Identified chart patterns
- **⚠️ Risk Assessment**: Current risk exposure
- **🤖 ML Prediction**: Deep learning recommendations
- **🔍 Historical Insights**: Vector database suggestions
- **🎯 Final Decision**: Synthesized trading decision

## 📚 Historical Results

View and analyze past trading results:

- **Results Table**: Summary of all past simulations
- **Detailed Analysis**: Deep dive into specific runs
- **Performance Comparison**: Compare different strategies
- **Trade Log Review**: Examine individual trade decisions

## 🛡 Security Features

- **Environment Variables**: Secure API key storage
- **Database Encryption**: Protected data storage
- **Error Handling**: Comprehensive error management
- **Rate Limiting**: API call optimization

## 🔧 Development

### Project Structure

```
Neural-Symbolic/
├── auto-trade.py           # Main application
├── requirements.txt        # Dependencies
├── .env.template          # Environment template
├── README.md             # Documentation
└── .gitignore           # Git ignore rules
```

### Adding New Agents

1. Create agent class inheriting from base structure
2. Implement required methods (`analyze`, `step`, etc.)
3. Add to `EnhancedMultiAgentTradingSystem`
4. Update UI components for new agent reasoning

### Extending ML Components

1. **Deep Learning**: Modify `DeepLearningAgent` class
2. **Vector Database**: Extend `VectorDBAgent` capabilities
3. **New Models**: Add additional prediction models
4. **Feature Engineering**: Enhance input features

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
- Check the GitHub issues page
- Review the troubleshooting section
- Contact the development team

## 🚀 Roadmap

### Planned Features
- [ ] Additional technical indicators
- [ ] More sophisticated ML models
- [ ] Real-time market data integration
- [ ] Portfolio optimization algorithms
- [ ] Advanced risk management strategies
- [ ] Multi-asset trading support
- [ ] Cloud deployment options

### Performance Improvements
- [ ] Async processing for better speed
- [ ] Caching layer for repeated calculations
- [ ] Optimized database queries
- [ ] Real-time streaming updates

---

**⚠️ Disclaimer**: This system is for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough testing before using with real funds.
- **Prompt engineering**: Specialized prompts for each agent type

### Technical Analysis
- **Multiple indicators**: SMA, Bollinger Bands, RSI, MACD
- **Pattern recognition**: Automated chart pattern detection
- **Risk metrics**: Real-time P&L and risk assessment
- **Portfolio tracking**: Complete portfolio state management

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
```bash
cp .env.template .env
# Edit .env with your actual values
```

3. **Set up PostgreSQL database**:
- Install PostgreSQL
- Create database `amaai_trading`
- Update database credentials in `.env`

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI API Key for LangChain agents
OPENAI_API_KEY=your_openai_api_key_here

# Database configuration
DB_HOST=localhost
DB_PORT=5433
DB_NAME=amaai_trading
DB_USER=postgres
DB_PASSWORD=your_db_password_here

# Trading configuration
INITIAL_CAPITAL=1000.0
RSI_OVERSOLD=30.0
RSI_OVERBOUGHT=70.0
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.10
```

### Trading Parameters

Adjust trading parameters in the `TradingConfig` class:

```python
class TradingConfig(BaseModel):
    initial_capital: float = 1000.0
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    max_position_size: float = 1.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
```

## Usage

### Running the Application

```bash
streamlit run auto-trade.py
```

### Using the Interface

1. **Set parameters** in the sidebar:
   - Trading symbol (e.g., BTC/USDT)
   - Time interval (15m, 1h, 4h, 1d)
   - Start and end dates
   - Initial capital

2. **Run simulation**: Click "Run Simulation" to start the multi-agent trading system

3. **View results**:
   - Performance metrics
   - Interactive charts
   - Trade log with agent reasoning
   - Portfolio value tracking

## Multi-Agent Workflow

The system follows this workflow for each trading decision:

1. **Data Collection**: Fetch market data and calculate technical indicators
2. **Market Analysis**: MarketAnalystAgent analyzes current market conditions
3. **Pattern Recognition**: PatternRecognitionAgent identifies trading patterns
4. **Risk Assessment**: RiskManagementAgent evaluates current risk exposure
5. **Decision Making**: TradingDecisionAgent synthesizes all inputs
6. **Execution**: MultiAgentTradingSystem executes the trading decision
7. **Tracking**: Results are logged and portfolio is updated

## Database Schema

The system stores results in PostgreSQL with the following schema:

```sql
CREATE TABLE backtest_runs (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    interval TEXT NOT NULL,
    initial_capital NUMERIC NOT NULL,
    strategy_return FLOAT,
    buyhold_return FLOAT,
    projection TEXT,
    trade_log JSONB,
    price_history JSONB,
    knowledge_graph JSONB
);
```

## Customization

### Adding New Agents

To add a new agent:

1. Create a new tool class inheriting from `BaseTool`
2. Implement the agent class with LangChain integration
3. Add the agent to `MultiAgentTradingSystem`
4. Update the decision-making process

### Modifying Trading Logic

The trading logic can be customized in:
- `TradingDecisionAgent.decide()`: Final decision logic
- `MultiAgentTradingSystem._execute_decision()`: Execution logic
- Individual agent analysis methods

### Adding New Indicators

To add new technical indicators:

1. Add indicator calculation in `fetch_binance_ta()`
2. Update `TechnicalAnalysisTool` to include new indicators
3. Modify agent prompts to consider new indicators

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **Database**: Use strong passwords and proper network security
- **Input Validation**: All user inputs are validated
- **Error Handling**: Sensitive information is not exposed in error messages

## Performance Optimization

- **Caching**: Consider implementing caching for repeated calculations
- **Batch Processing**: API calls are batched for efficiency
- **Memory Management**: Large datasets are processed in chunks
- **Database Indexing**: Proper indexing for query performance

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Database connection**: Check database credentials and connectivity
3. **API limits**: Monitor OpenAI API usage and rate limits
4. **Memory issues**: Reduce data range for large backtests

### Debugging

Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with proper testing
4. Submit a pull request

## License

This project is licensed under the MIT License.
