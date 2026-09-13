# ══════════════════════════════════════════════════════════════════════════════
# AMAAI MULTI-AGENT TRADING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

# ── STANDARD LIBRARY IMPORTS ────────────────────────────────────────────────
import datetime as dt
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union

# ── THIRD-PARTY IMPORTS ─────────────────────────────────────────────────────
# Technical-indicator (ta.*) and LLM-construction (ChatOpenAI/AzureChatOpenAI)
# imports moved to core/data.py and core/llm.py respectively; ccxt, requests
# and BeautifulSoup went with them and with the deleted social-post agent.
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from pydantic import Field

# ── LANGCHAIN IMPORTS ───────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
# NOTE: langchain_core.pydantic_v1 used to be imported here and shadowed the
# pydantic v2 `Field` imported above. LangChain 0.3 is pydantic v2 native, so
# the v1 shim is both deprecated and actively harmful. Removed deliberately.

# ── CONDITIONAL IMPORTS (with error handling) ──────────────────────────────
# Setup logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit's auto-reload watcher walks every entry in sys.modules and reads
# each one's __path__ to decide which files to watch. `transformers` -- pulled
# in by langchain.agents and langchain_openai, not by anything here -- resolves
# its submodules lazily, so merely READING that attribute executes a real
# import of every transformers.models.* image processor. Dozens of those import
# torchvision, which is not installed, and each failure prints a full traceback
# before the watcher shrugs and moves on. Hundreds of lines of alarming output,
# no consequence: the app is already running by then, which is why the UI loads
# fine.
#
# Silencing this specific logger loses nothing. Streamlit blacklists
# `**/site-packages` and `**/venv` from watching anyway, so every path those
# failed imports were being asked for would have been discarded a moment later
# -- the blacklist is applied to the RESULT of the probe, too late to prevent
# it. Only this one logger is touched; genuine Streamlit warnings elsewhere are
# untouched.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# `streamlit run auto-trade.py` puts this directory on sys.path, but running the
# file by absolute path from elsewhere does not, and then `import signals`
# fails and the positioning channel goes quietly dead. Make the repo root
# importable regardless of how the file was launched.
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── LOG NOISE CONTROL ──────────────────────────────────────────────────────
# Three requests were logged per decision, which buried the two lines that
# actually matter (the LLM contribution report and the positioning report).
for _noisy in ("httpx", "httpcore", "openai._base_client", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# LangChain's structured-output path hands pydantic a parsed model where the
# schema declares `parsed: None`, so pydantic emits a serializer warning on
# EVERY successful call. The parse itself succeeds; the authoritative signal
# for genuine failures is `llm_failures` in the run's llm_report, which is
# incremented from the real exception handler in make_decision. Suppressed
# narrowly by message and module so unrelated pydantic warnings still surface.
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings",
    category=UserWarning, module=r"pydantic\.main",
)

# Exogenous positioning signal (Binance futures metrics; free, keyless, cached)
try:
    from signals.binance_positioning import (
        PositioningReading, PositioningSignalAgent, ensure_cached, missing_days,
        warmup_start_date,
    )
    POSITIONING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Positioning signal module not available: {e}")
    POSITIONING_AVAILABLE = False
    PositioningReading = None
    PositioningSignalAgent = None
    ensure_cached = None
    missing_days = None
    warmup_start_date = None

# Exogenous text sentiment (Hacker News via Algolia; free, keyless, cached)
try:
    from signals.text_sentiment import (
        DEFAULT_QUERIES as TEXT_DEFAULT_QUERIES,
        POINT_THRESHOLDS as TEXT_POINT_THRESHOLDS,
        TextSentimentAgent,
        TextSentimentReading,
        Z_CLIP as TEXT_Z_CLIP,
        ensure_cached as ensure_text_cached,
        missing_days as text_missing_days,
        warmup_start_date as text_warmup_start_date,
    )
    TEXT_SENTIMENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Text sentiment module not available: {e}")
    TEXT_SENTIMENT_AVAILABLE = False
    TextSentimentAgent = None
    TextSentimentReading = None
    ensure_text_cached = None
    text_missing_days = None
    text_warmup_start_date = None
    TEXT_DEFAULT_QUERIES = ("bitcoin", "crypto")
    TEXT_POINT_THRESHOLDS = ((0.66, 2), (0.33, 1))
    TEXT_Z_CLIP = 3.0

# ── CORPORATE TLS TRUST ────────────────────────────────────────────────────
# This network re-signs HTTPS with a private root CA that Windows trusts but
# certifi does not, so any library verifying against certifi alone fails with
# CERTIFICATE_VERIFY_FAILED. That was silently killing the vector-DB agent:
# `tiktoken` downloads its BPE encoding from openaipublic.blob.core.windows.net
# on first use, the download failed, and every store_pattern call errored out.
# Same class of bug as the decorative sentiment agent -- an agent that runs,
# fails, and contributes nothing.
#
# The generated bundle is certifi's roots PLUS the Windows trust store, i.e. a
# strict superset of the default, so anything that verified before still
# verifies (checked against ccxt price fetches and the Azure client). An
# existing REQUESTS_CA_BUNDLE is never overridden.
#
# verify=False is deliberately NOT used anywhere: disabling verification would
# make the data-provenance claim in any write-up unverifiable.
def _configure_corporate_tls() -> None:
    if os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("DISABLE_CORPORATE_CA"):
        return
    try:
        from signals.corporate_ca import ensure_bundle
        bundle = ensure_bundle()
    except Exception as e:  # never let trust-store setup break startup
        logger.debug(f"Corporate CA bundle unavailable: {e}")
        return
    if bundle and Path(bundle).exists():
        os.environ["REQUESTS_CA_BUNDLE"] = str(bundle)
        logger.info(f"Using merged CA bundle for HTTPS verification: {bundle}")


_configure_corporate_tls()

# SQLAlchemy
try:
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    create_engine = None
    Engine = None

# Deep learning was REMOVED on 2026-09-13, along with the TensorFlow, sklearn
# and joblib imports that existed only to serve it.
#
# The `DeepLearningAgent` was never a working component. It had no training
# code at all -- a function that BUILT an empty network, and nothing that ever
# fitted one -- so `load_model()` always failed, `predict()` always
# short-circuited, and the only thing it ever contributed was a fixed string in
# the LLM prompt reading "signal n/a, confidence 0.00". It was excluded from
# every experiment (see experiments/harness.py), so no reported number depended
# on it, while the UI advertised "Deep Learning: Enabled" and promised neural
# pattern recognition that did not exist.
#
# Nothing reusable was lost: a real deep-learning arm needs labels and
# time-respecting train/validation splits, neither of which this design had,
# so it would have to be written from scratch regardless.

# Vector DB
try:
    # faiss itself is unused here: it is imported so that a missing native
    # wheel is detected now and sets VECTOR_DB_AVAILABLE=False, rather than
    # blowing up later inside FAISS. Do not "clean up" this import.
    import faiss  # noqa: F401
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    VECTOR_DB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vector DB libraries not available: {e}")
    VECTOR_DB_AVAILABLE = False
    class FAISS:
        @staticmethod
        def from_documents(docs, embeddings): return None
    class Document:
        def __init__(self, page_content="", metadata=None): pass

# Load environment variables
load_dotenv()

# ── STREAMLIT PAGE CONFIG ───────────────────────────────────────────────────
# MUST be the first Streamlit command executed by the script. Keeping it here,
# at module scope, guarantees it runs before any st.* call that a later import
# or startup check might emit (e.g. a warning banner).
st.set_page_config(
    page_title="AMAAI Multi-Agent Trading System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CONFIGURATION & MODELS ──────────────────────────────────────────────────
# ── DOMAIN TYPES ───────────────────────────────────────────────────────────
# Moved to core/config.py.
from core.config import (  # noqa: E402
    TradingAction,
    TradingConfig,
    TradingDecision,
)

# ── LLM PROVIDER + DECISION CONTRACT ───────────────────────────────────────
# Moved to core/llm.py.
from core.llm import (  # noqa: E402
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    USE_AZURE_OPENAI,
    LLMContribution,
    LLMRiskAssessment,
    LLMTradeAdjustment,
    describe_active_model,
    get_llm,
    get_llm_provider_status,
    set_error_reporter,
)

# `core` deliberately does not import Streamlit, so provider errors reach the
# user through this hook instead of a direct st.error call. Without it a
# missing API key would only appear in the terminal log, which is where the
# person running the app is least likely to look.
set_error_reporter(st.error)

# ── CONFIGURATION ──────────────────────────────────────────────────────────
# Every runtime flag and domain type now lives in core/config.py, which had
# been scattered across this file at lines 341, 531, 623, 630, 1193, 1194 and
# 3308, each with its own ad-hoc parsing.
from core.config import (  # noqa: E402
    ALLOW_SYNTHETIC_DATA,
    ENABLE_DATABASE,
    EXECUTION_MODE,
    LLM_MAX_ADJUSTMENT,
    POSITIONING_AUTO_FETCH,
    POSITIONING_LAG_BARS,
    POSITIONING_MAX_POINTS,
    POSITIONING_ZSCORE_WINDOW,
    positioning_zscore_window,
    DECISION_CADENCE_HOURS,
    decision_step_bars,
    interval_hours,
    SLIPPAGE_PCT,
    TEXT_SENTIMENT_AUTO_FETCH,
    TEXT_SENTIMENT_LAG_BARS,
    TEXT_SENTIMENT_MAX_POINTS,
    TEXT_SENTIMENT_MIN_DOCS,
    TEXT_SENTIMENT_QUERIES,
    TEXT_SENTIMENT_SCORER,
    TEXT_SENTIMENT_WINDOW_HOURS,
    TEXT_SENTIMENT_ZSCORE_WINDOW,
    USE_LLM_DECISIONS,
    USE_POSITIONING_SIGNAL,
    USE_TEXT_SENTIMENT,
    INDICATOR_WARMUP_BARS,
    InsufficientHistory,
    SyntheticDataBlocked,
)

# ── DATA HANDLING ──────────────────────────────────────────────────────────
# Moved to core/data.py. Its private helpers (_fetch_from_binance,
# _generate_simulated_data, _process_technical_indicators) are reached through
# fetch_binance_ta and are no longer re-exported here.
from core.data import (  # noqa: E402
    fetch_binance_ta,
    resolve_bar_index,
)

# ── POSTGRES CONNECTION (OPTIONAL) ──────────────────────────────────────────
# Result persistence is OFF by default: the app runs fully without PostgreSQL
# and keeps past runs in the Streamlit session instead. To turn persistence
# back on, set ENABLE_DATABASE=true in .env and point DB_* at a live server.
# ENABLE_DATABASE itself is imported from core.config above; it used to be
# re-parsed here as well, which left the import dead and gave two names for
# one flag.

def get_db_config():
    """Get database configuration from environment variables"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5433')),
        'database': os.getenv('DB_NAME', 'amaai_trading'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'P@ssw0rd')
    }

def get_db_url():
    """Get database URL for SQLAlchemy"""
    config = get_db_config()
    return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

def pg_conn():
    """Create PostgreSQL connection"""
    try:
        return psycopg2.connect(**get_db_config())
    except Exception as e:
        # Log only - never render UI from a connection helper, or the banner
        # becomes the page's first Streamlit command and breaks set_page_config.
        logger.error(f"Database connection failed: {e}")
        raise

def get_sqlalchemy_engine():
    """Create SQLAlchemy engine for pandas operations"""
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy is required for pandas database operations")
    
    try:
        return create_engine(get_db_url())
    except Exception as e:
        logger.error(f"SQLAlchemy engine creation failed: {e}")
        raise

# Initialize database
def init_database():
    """Initialize database with error handling"""
    try:
        with pg_conn() as c:
            cur = c.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
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
            """)
            c.commit()
            logger.info("Database initialized successfully")
            
            # Warn if SQLAlchemy is not available
            if not SQLALCHEMY_AVAILABLE:
                logger.warning("SQLAlchemy not available. Installing: pip install sqlalchemy psycopg2-binary")

            return True
    except Exception as e:
        logger.error(f"Database initialization error: {e}. Results won't be saved.")
        return False

@st.cache_resource(show_spinner=False)
def get_database_available() -> bool:
    """Resolve database availability lazily, once per app session.

    Returns False immediately when persistence is disabled, so no connection
    is attempted and no error surfaces at import time.
    """
    if not ENABLE_DATABASE:
        logger.info("Database persistence disabled (ENABLE_DATABASE is not set). Results are kept in-session only.")
        return False
    return init_database()

# ── EXECUTION ──────────────────────────────────────────────────────────────
# Moved to core/execution.py so a headless run can execute fills without the UI.
from core.execution import execute_trade  # noqa: E402
from core.metrics import describe_run  # noqa: E402

def display_simulation_results(summary: dict, df: pd.DataFrame, decisions_log: list, symbol: str = None, show_reasoning: bool = True):
    """Display comprehensive simulation results with enhanced UI and trend recommendations"""
    
    # Main header with status indicator
    if summary['outperformed_market']:
        st.success("🎉 **TRADING SIMULATION COMPLETED - STRATEGY OUTPERFORMED MARKET**")
    else:
        st.warning("📊 **TRADING SIMULATION COMPLETED - STRATEGY UNDERPERFORMED MARKET**")
    
    # === RUN PROVENANCE & LLM CONTRIBUTION (PHASE 1 + 2) ===
    meta = summary.get('run_metadata', {})
    llm_report = summary.get('llm_report', {})

    if meta:
        if meta.get('publication_safe'):
            st.success(
                f"🔬 **Publication-safe run.** Data: {meta.get('data_source_label')}. "
                f"Model: {meta.get('model', {}).get('deployment', meta.get('model', {}).get('model', 'n/a'))}. "
                f"Execution: {meta.get('execution_mode')} with {meta.get('slippage_pct')}% slippage."
            )
        else:
            # Since the legacy synthetic-post channel was removed, the price
            # series is the only thing that can make a run unpublishable: both
            # exogenous channels read from cached, hashed real corpora.
            st.error(
                "🚫 **NOT publication-safe: price data is synthetic.** "
                "These numbers are valid for code testing only."
            )

        with st.expander("🔍 Run provenance (for the methods section)", expanded=False):
            st.json(meta)

    if llm_report:
        st.subheader("🧠 LLM Contribution")
        if not llm_report.get('llm_enabled'):
            st.info("Rules-only ablation arm: the LLM was disabled for this run (USE_LLM_DECISIONS=false).")
        else:
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                st.metric("Decisions", llm_report.get('decisions', 0))
            with lc2:
                st.metric(
                    "Changed by LLM",
                    llm_report.get('decisions_changed_by_llm', 0),
                    f"{llm_report.get('change_rate_pct', 0):.1f}% of decisions",
                )
            with lc3:
                st.metric("Risk vetoes", llm_report.get('risk_vetoes', 0))
            with lc4:
                st.metric(
                    "LLM call success",
                    f"{llm_report.get('llm_success_rate_pct', 0):.0f}%",
                    f"{llm_report.get('avg_llm_latency_s', 0):.1f}s avg",
                )
            if llm_report.get('llm_failures'):
                st.warning(
                    f"{llm_report['llm_failures']} LLM calls failed and fell back to the rule engine. "
                    "Report this coverage figure alongside any result from this run."
                )

    # === EXOGENOUS SIGNAL CONTRIBUTION ===
    # The counterpart to the LLM panel above, and the number that answers the
    # only question that matters about a signal channel: did it change anything?
    # A channel that ran on every bar and moved 0% of decisions is decorative,
    # and that has to be visible rather than buried in a log line.
    positioning_report = summary.get('positioning_report', {})
    text_report = summary.get('text_sentiment_report', {})

    if positioning_report or text_report:
        st.subheader("📡 Exogenous Signal Contribution")

        for report, title, off_hint in (
            (positioning_report, "Futures positioning (Binance USD-M)",
             "USE_POSITIONING_SIGNAL=false"),
            (text_report, "Text sentiment (Hacker News)",
             "USE_TEXT_SENTIMENT=false"),
        ):
            if not report:
                continue
            st.markdown(f"**{title}**")
            if not report.get('enabled'):
                st.info(f"Switched off for this run ({off_hint}). This is the "
                        f"signal-off ablation arm.")
                continue

            agent_info = report.get('agent') or {}
            if not agent_info.get('available'):
                st.warning(
                    f"Enabled but produced no usable readings: "
                    f"{agent_info.get('status', 'unknown reason')}"
                )
                continue

            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                st.metric("Decisions with a reading",
                          report.get('decisions_with_reading', 0),
                          f"of {report.get('decisions_total', 0)}")
            with ec2:
                st.metric("Decisions it moved",
                          report.get('decisions_where_points_added', 0),
                          f"{report.get('pct_decisions_moved', 0):.1f}% of decisions")
            with ec3:
                mean_score = report.get('mean_score')
                st.metric("Mean score",
                          "n/a" if mean_score is None else f"{mean_score:+.3f}")
            with ec4:
                st.metric("Bar coverage",
                          f"{agent_info.get('coverage_pct', 0):.0f}%")

            if report.get('decisions_where_points_added', 0) == 0:
                st.warning(
                    "This channel produced readings but never crossed its "
                    "threshold, so it changed no trade in this run. Treat it as "
                    "inactive when interpreting the result."
                )
            st.caption(agent_info.get('status', ''))

    # === EXECUTIVE SUMMARY SECTION ===
    st.header("📊 Executive Summary")
    
    # Key performance indicators in a visually appealing layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🏦 Portfolio Value",
            f"${summary['final_value']:,.2f}",
            f"${summary['final_value'] - summary['initial_capital']:,.2f}"
        )
    
    with col2:
        return_color = "normal" if summary['total_return_pct'] >= 0 else "inverse"
        st.metric(
            "📊 Strategy Return",
            f"{summary['total_return_pct']:.2f}%",
            f"{summary['total_return_pct'] - summary['buy_hold_return_pct']:.2f}% vs Market",
            delta_color=return_color
        )
    
    with col3:
        st.metric(
            "🎯 Win Rate",
            f"{summary['win_rate_pct']:.1f}%",
            f"{summary['winning_trades']} wins of {summary['total_trades']} trades"
        )
    
    with col4:
        if summary['outperformed_market']:
            st.metric("🚀 vs Market", "BEAT", f"+{summary['total_return_pct'] - summary['buy_hold_return_pct']:.2f}%")
        else:
            st.metric("📉 vs Market", "TRAIL", f"{summary['total_return_pct'] - summary['buy_hold_return_pct']:.2f}%")
    
    with col5:
        if summary['total_trades'] > 0:
            avg_profit = (summary['final_value'] - summary['initial_capital']) / summary['total_trades']
            st.metric(
                "💰 Avg/Trade",
                f"${avg_profit:,.2f}",
                f"From {summary['total_trades']} executions"
            )
        else:
            st.metric("💰 Avg/Trade", "N/A", "No trades executed")

    # ── Risk-adjusted view ──────────────────────────────────────────────────
    # Return alone says nothing about how the return was earned: doubling the
    # position size doubles it and changes nothing about whether the signal is
    # real. These were computed but never displayed before Phase 3.
    st.subheader("📉 Risk-Adjusted Performance")
    rcol1, rcol2, rcol3, rcol4 = st.columns(4)

    def _ratio(value, fmt="{:.2f}"):
        return fmt.format(value) if isinstance(value, (int, float)) else "—"

    rcol1.metric("Sharpe (annualised)", _ratio(summary.get('sharpe_ratio')),
                 "return per unit of total volatility")
    rcol2.metric("Sortino (annualised)", _ratio(summary.get('sortino_ratio')),
                 "penalises only downside")
    rcol3.metric("Max Drawdown",
                 _ratio(summary.get('max_drawdown_pct'), "{:.2f}%"),
                 "worst peak-to-trough fall")
    rcol4.metric("Calmar", _ratio(summary.get('calmar_ratio')),
                 "return per unit of drawdown")

    if summary.get('sample_warning'):
        st.warning(
            f"⚠️ {summary['sample_warning']} A drawdown is still shown because "
            f"it is an observed fact about the path rather than an estimate."
        )
    else:
        st.caption(
            f"From {summary.get('observations', 0)} return observations. "
            f"These describe THIS run only — they are not evidence the strategy "
            f"generalises. Use the ablation's bootstrap intervals for that."
        )

    # === NEXT TREND RECOMMENDATION SECTION ===
    st.header("🔮 Next Action Recommendation")
    st.markdown("*Multi-Agent Analysis for Next Trading Decision*")
    
    # Get current market state for recommendations
    current_data = df.iloc[-1]
    current_price = current_data['close']
    current_rsi = current_data['RSI']
    current_ma20 = current_data['MA20']
    current_ma50 = current_data['MA50']
    macd_hist = current_data['MACD_hist']
    macd_line = current_data['MACD_line']
    macd_signal = current_data['MACD_signal']
    
    # Calculate recent price momentum
    if len(df) >= 5:
        price_5_ago = df.iloc[-6]['close']
        momentum_5d = (current_price - price_5_ago) / price_5_ago * 100
    else:
        momentum_5d = 0
    
    # Multi-agent recommendation analysis
    recommendation_col1, recommendation_col2 = st.columns([2, 1])
    
    with recommendation_col1:
        st.subheader("🤖 Agent Recommendations")
        
        # Technical Analysis Agent
        with st.expander("📊 Technical Analysis Agent", expanded=True):
            # Calculate technical signals
            ta_signals = []
            ta_score = 0
            
            # RSI Analysis
            if current_rsi < 30:
                ta_signals.append("🟢 RSI Oversold (Bullish)")
                ta_score += 2
            elif current_rsi > 70:
                ta_signals.append("🔴 RSI Overbought (Bearish)")
                ta_score -= 2
            elif current_rsi < 40:
                ta_signals.append("🟡 RSI Approaching Oversold")
                ta_score += 1
            elif current_rsi > 60:
                ta_signals.append("🟡 RSI Approaching Overbought")
                ta_score -= 1
            else:
                ta_signals.append("⚪ RSI Neutral")
            
            # Moving Average Analysis
            if current_price > current_ma20 > current_ma50:
                ta_signals.append("🟢 Strong Uptrend (Price > MA20 > MA50)")
                ta_score += 2
            elif current_price > current_ma20:
                ta_signals.append("🟡 Weak Uptrend (Price > MA20)")
                ta_score += 1
            elif current_price < current_ma20 < current_ma50:
                ta_signals.append("🔴 Strong Downtrend (Price < MA20 < MA50)")
                ta_score -= 2
            elif current_price < current_ma20:
                ta_signals.append("🟡 Weak Downtrend (Price < MA20)")
                ta_score -= 1
            else:
                ta_signals.append("⚪ Neutral Trend")
            
            # MACD Analysis
            if macd_line > macd_signal and macd_hist > 0:
                ta_signals.append("🟢 MACD Bullish Crossover")
                ta_score += 1
            elif macd_line < macd_signal and macd_hist < 0:
                ta_signals.append("🔴 MACD Bearish Crossover")
                ta_score -= 1
            else:
                ta_signals.append("⚪ MACD Neutral")
            
            # Momentum Analysis
            if momentum_5d > 2:
                ta_signals.append(f"🟢 Strong Momentum (+{momentum_5d:.1f}%)")
                ta_score += 1
            elif momentum_5d < -2:
                ta_signals.append(f"🔴 Negative Momentum ({momentum_5d:.1f}%)")
                ta_score -= 1
            else:
                ta_signals.append(f"⚪ Neutral Momentum ({momentum_5d:.1f}%)")
            
            # Display TA signals
            for signal in ta_signals:
                st.write(f"• {signal}")
            
            # TA Recommendation
            if ta_score >= 3:
                ta_recommendation = "STRONG BUY"
                ta_color = "🟢"
            elif ta_score >= 1:
                ta_recommendation = "BUY"
                ta_color = "🟡"
            elif ta_score <= -3:
                ta_recommendation = "STRONG SELL"
                ta_color = "🔴"
            elif ta_score <= -1:
                ta_recommendation = "SELL"
                ta_color = "🟡"
            else:
                ta_recommendation = "HOLD"
                ta_color = "⚪"
            
            st.markdown(f"**TA Recommendation: {ta_color} {ta_recommendation}** (Score: {ta_score:+d})")
        
        # ── Text Sentiment Agent (Hacker News) ──────────────────────────────
        # This panel used to read decisions_log['sentiment'], the synthetic-post
        # channel that Phase 2 switched off, so it always printed "No recent
        # sentiment data available". It now reads the real text sentiment
        # channel, and reports the exact points it contributed to the trade
        # rather than inventing a separate BUY/SELL vote from the same number.
        latest_text = None
        if decisions_log:
            for decision in reversed(decisions_log):
                if decision.get('text_sentiment'):
                    latest_text = decision['text_sentiment']
                    break

        text_available = bool(latest_text and latest_text.get('available'))
        header = "💬 Text Sentiment Agent (Hacker News)"
        with st.expander(header, expanded=True):
            if text_available:
                t_score = latest_text.get('score', 0.0)
                t_points = (latest_text.get('bullish_points', 0)
                            - latest_text.get('bearish_points', 0))
                if t_score > 0:
                    text_signal, sentiment_recommendation = "🟢 Bullish", "BUY"
                elif t_score < 0:
                    text_signal, sentiment_recommendation = "🔴 Bearish", "SELL"
                else:
                    text_signal, sentiment_recommendation = "⚪ Neutral", "HOLD"

                t_feat = latest_text.get('features') or {}
                t_z = t_feat.get('z_mean_sentiment')
                pos_share = t_feat.get('pos_share')
                neg_share = t_feat.get('neg_share')

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Sentiment score", f"{t_score:+.3f}")
                col_b.metric("Documents", f"{latest_text.get('doc_count', 0)}")
                col_c.metric("Signal points", f"{t_points:+d}")

                st.write(f"• **Signal:** {text_signal}")
                if t_z is not None:
                    st.write(f"• **Trend strength:** z = {t_z:+.2f} "
                             f"versus its own trailing baseline")
                if pos_share is not None and neg_share is not None:
                    neutral = max(0.0, 1.0 - pos_share - neg_share)
                    st.write(f"• **Document mix:** {pos_share:.0%} positive · "
                             f"{neg_share:.0%} negative · {neutral:.0%} neutral")
                st.write(f"• **Mean raw sentiment:** "
                         f"{latest_text.get('mean_sentiment') or 0:+.3f} "
                         f"(before de-trending)")
                st.write(f"• **Confidence (data quality):** "
                         f"{latest_text.get('confidence', 0):.2f}")

                # The arithmetic that turned a z-score into rule points. Without
                # it a reading like "+0.221 -> 0 points" looks arbitrary, which
                # is the single most common question this panel gets.
                if t_z is not None:
                    cap = int(TEXT_SENTIMENT_MAX_POINTS)
                    ladder = []
                    for thr, awarded in TEXT_POINT_THRESHOLDS:
                        if awarded > cap:
                            continue
                        ladder.append(f"|z| ≥ {thr * TEXT_Z_CLIP:.2f} → ±{awarded}")
                    st.markdown(
                        f"**How this became {t_points:+d} point(s):** "
                        f"`z = {t_z:+.2f}` → `score = z / {TEXT_Z_CLIP:.1f} = "
                        f"{t_score:+.3f}` → **{t_points:+d}**  \n"
                        f"Thresholds: {', '.join(ladder)} "
                        f"(capped at ±{cap} this run via TEXT_SENTIMENT_MAX_POINTS)."
                    )

                examples = latest_text.get('examples') or []
                if examples:
                    # Rendered inline, not in a nested expander: this block is
                    # already inside the agent's own expander and Streamlit
                    # forbids one expander inside another. A widget-based
                    # disclosure (checkbox/toggle) is no good either — clicking
                    # it reruns the script, and this whole results panel only
                    # renders during a simulation run, so the page would go
                    # blank. At most EXAMPLES_PER_DIRECTION * 2 documents of
                    # 200 chars each land here, so inline costs a few lines.
                    st.markdown(
                        f"**📄 What it actually read** ({len(examples)} of "
                        f"{latest_text.get('doc_count', 0)} documents in the "
                        f"{TEXT_SENTIMENT_WINDOW_HOURS}h window)"
                    )
                    st.caption(
                        "Strongest bullish and bearish documents in this "
                        "bar's window, from the same half-open interval "
                        "used to compute the score — nothing here was "
                        "published at or after the bar's cutoff."
                    )
                    for doc in examples:
                        s = doc.get('sentiment', 0.0)
                        icon = "🟢" if s > 0.05 else ("🔴" if s < -0.05 else "⚪")
                        when = str(doc.get('created_at', ''))[:16].replace('T', ' ')
                        st.markdown(
                            f"{icon} `{s:+.3f}`  *{doc.get('kind', '')}*  "
                            f"{when}  ·  {doc.get('chars', 0)} chars  \n"
                            f"{doc.get('text', '')}"
                        )
                else:
                    st.caption(latest_text.get('reasoning', ''))

                st.markdown(f"**Text Sentiment Recommendation: "
                            f"{sentiment_recommendation}**")
            elif latest_text:
                sentiment_recommendation = "HOLD"
                st.info(f"No usable reading: {latest_text.get('reasoning', '')}")
                st.caption("An unavailable channel contributes nothing to the "
                           "trade. It is not counted as a neutral vote.")
            else:
                sentiment_recommendation = "HOLD"
                st.info("Text sentiment is switched off for this run "
                        "(USE_TEXT_SENTIMENT=false).")
                st.caption("Turn it on in .env to add Hacker News sentiment to "
                           "the signal score.")

        # ── Futures Positioning Agent (Binance USD-M) ───────────────────────
        latest_pos = None
        if decisions_log:
            for decision in reversed(decisions_log):
                if decision.get('positioning'):
                    latest_pos = decision['positioning']
                    break

        pos_available = bool(latest_pos and latest_pos.get('available'))
        # Expanded even when unavailable. A collapsed panel next to a populated
        # one reads as "this channel is missing" rather than "this channel has
        # no data for the most recent bar", which is a different claim.
        with st.expander("📡 Futures Positioning Agent", expanded=True):
            if pos_available:
                p_score = latest_pos.get('score', 0.0)
                p_points = (latest_pos.get('bullish_points', 0)
                            - latest_pos.get('bearish_points', 0))
                if p_score > 0:
                    pos_signal, positioning_recommendation = "🟢 Bullish", "BUY"
                elif p_score < 0:
                    pos_signal, positioning_recommendation = "🔴 Bearish", "SELL"
                else:
                    pos_signal, positioning_recommendation = "⚪ Neutral", "HOLD"

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Positioning score", f"{p_score:+.3f}")
                col_b.metric("Signal points", f"{p_points:+d}")
                col_c.metric("Confidence", f"{latest_pos.get('confidence', 0):.2f}")

                features = latest_pos.get('features') or {}
                labels = {
                    'z_count_long_short_ratio': 'Crowd long/short (faded)',
                    'z_sum_toptrader_long_short_ratio': 'Top traders (followed)',
                    'z_sum_taker_long_short_vol_ratio': 'Taker flow (followed)',
                }
                st.write(f"• **Signal:** {pos_signal}")
                for key, label in labels.items():
                    z = features.get(key)
                    if z is None:
                        continue
                    st.write(f"• **{label}:** z = {z:+.2f}")
                st.caption(latest_pos.get('reasoning', ''))
                st.markdown(f"**Positioning Recommendation: "
                            f"{positioning_recommendation}**")
            elif latest_pos:
                positioning_recommendation = "HOLD"
                st.info(f"No usable reading for the latest bar: "
                        f"{latest_pos.get('reasoning', '')}")
                st.caption(
                    "Binance publishes futures positioning as one file per "
                    "**completed** day, so the most recent 24-48 hours are "
                    "normally absent. That is a property of the free feed, not "
                    "a fault: this channel can inform a backtest but never a "
                    "live next-bar decision. Earlier bars in the run above may "
                    "still have had readings — see the coverage figure in the "
                    "Exogenous Signal Contribution panel."
                )
            else:
                positioning_recommendation = "HOLD"
                st.info("Futures positioning is switched off for this run "
                        "(USE_POSITIONING_SIGNAL=false).")

        # Risk Management Agent
        with st.expander("⚠️ Risk Management Agent"):
            # Get latest risk assessment
            latest_risk = None
            if decisions_log:
                for decision in reversed(decisions_log):
                    if 'risk_assessment' in decision and decision['risk_assessment']:
                        latest_risk = decision['risk_assessment']
                        break
            
            if latest_risk and isinstance(latest_risk, dict):
                risk_level = latest_risk.get('risk_level', 'medium')
                position_size = latest_risk.get('position_size_pct', 5.0)
                
                # Calculate recent volatility
                recent_prices = df.tail(20)['close']
                volatility = recent_prices.pct_change().std() * 100
                
                if risk_level == 'low' and volatility < 2:
                    risk_signal = "🟢 Low Risk Environment"
                    risk_recommendation = "BUY"
                elif risk_level == 'high' or volatility > 5:
                    risk_signal = "🔴 High Risk Environment"
                    risk_recommendation = "SELL/HOLD"
                else:
                    risk_signal = "🟡 Medium Risk Environment"
                    risk_recommendation = "MODERATE"
                
                st.write(f"• **Risk Level:** {risk_level.title()}")
                st.write(f"• **Volatility:** {volatility:.2f}%")
                st.write(f"• **Position Size:** {position_size:.1f}%")
                st.write(f"• **Signal:** {risk_signal}")
                st.markdown(f"**Risk Recommendation: {risk_recommendation}**")
            else:
                st.write("• Risk Level: Medium (Default)")
                st.write("• Volatility: Calculating...")
                st.markdown("**Risk Recommendation: MODERATE**")
                risk_recommendation = "MODERATE"
    
    with recommendation_col2:
        st.subheader("🎯 Final Decision")
        
        # Aggregate all recommendations
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        
        # Count TA votes
        if ta_recommendation in ["STRONG BUY", "BUY"]:
            buy_votes += 2 if ta_recommendation == "STRONG BUY" else 1
        elif ta_recommendation in ["STRONG SELL", "SELL"]:
            sell_votes += 2 if ta_recommendation == "STRONG SELL" else 1
        else:
            hold_votes += 1
        
        # Count Text Sentiment votes. Only a channel that actually produced a
        # reading gets a vote: an unavailable channel voting HOLD would let a
        # dead feed tilt the tally, which is how the old sentiment agent looked
        # like it was participating while contributing nothing.
        if text_available and 'sentiment_recommendation' in locals():
            if sentiment_recommendation == "BUY":
                buy_votes += 1
            elif sentiment_recommendation == "SELL":
                sell_votes += 1
            else:
                hold_votes += 1

        # Count Positioning votes, same rule
        if pos_available and 'positioning_recommendation' in locals():
            if positioning_recommendation == "BUY":
                buy_votes += 1
            elif positioning_recommendation == "SELL":
                sell_votes += 1
            else:
                hold_votes += 1

        # Count Risk votes
        if 'risk_recommendation' in locals():
            if risk_recommendation == "BUY":
                buy_votes += 1
            elif "SELL" in risk_recommendation:
                sell_votes += 1
            else:
                hold_votes += 1
        
        # Determine final recommendation
        if buy_votes > sell_votes and buy_votes > hold_votes:
            final_action = "BUY"
            final_color = "#00FF88"
            final_emoji = "🟢"
            confidence_score = buy_votes / (buy_votes + sell_votes + hold_votes)
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            final_action = "SELL"
            final_color = "#FF4444"
            final_emoji = "🔴"
            confidence_score = sell_votes / (buy_votes + sell_votes + hold_votes)
        else:
            final_action = "HOLD"
            final_color = "#FFC107"
            final_emoji = "🟡"
            confidence_score = hold_votes / (buy_votes + sell_votes + hold_votes)
        
        # Display final recommendation in a prominent box
        st.markdown(f"""
        <div style="
            background-color: {final_color}20;
            border: 2px solid {final_color};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
        ">
            <h2 style="color: {final_color}; margin: 0;">
                {final_emoji} {final_action}
            </h2>
            <p style="color: white; margin: 5px 0;">
                <strong>Confidence: {confidence_score:.0%}</strong>
            </p>
            <p style="color: #CCCCCC; margin: 0; font-size: 14px;">
                Price: ${current_price:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Vote breakdown
        st.markdown("**📊 Agent Votes:**")
        st.write(f"🟢 BUY: {buy_votes} votes")
        st.write(f"🔴 SELL: {sell_votes} votes")
        st.write(f"🟡 HOLD: {hold_votes} votes")
        
        # Key metrics summary
        st.markdown("**📈 Key Metrics:**")
        st.write(f"• RSI: {current_rsi:.1f}")
        st.write(f"• Price vs MA20: {((current_price/current_ma20-1)*100):+.1f}%")
        st.write(f"• 5-Day Momentum: {momentum_5d:+.1f}%")
        
        # Next steps
        st.markdown("**🔄 Next Steps:**")
        if final_action == "BUY":
            st.write("• Consider opening long position")
            st.write("• Monitor for entry confirmation")
            st.write("• Set stop-loss orders")
        elif final_action == "SELL":
            st.write("• Consider closing long positions")
            st.write("• Monitor for exit confirmation")
            st.write("• Preserve capital")
        else:
            st.write("• Wait for clearer signals")
            st.write("• Monitor market conditions")
            st.write("• Prepare for next opportunity")
    
    # === CURRENT TRADING SUMMARY ===
    st.subheader("📈 Current Trading Position Summary")
    
    # Get final portfolio state
    final_cash = summary.get('final_cash', summary['final_value'])
    final_holdings = summary.get('final_holdings', 0.0)
    current_price = df.iloc[-1]['close']
    is_holding = final_holdings > 0
    
    # Calculate current position details
    if is_holding:
        position_value = final_holdings * current_price
        total_invested = summary['final_value'] - final_cash
        position_pct = (position_value / summary['final_value']) * 100 if summary['final_value'] > 0 else 0
    else:
        position_value = 0
        total_invested = 0
        position_pct = 0
    
    # Display current position
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if is_holding:
            st.metric(
                "Current Position",
                "🟢 LONG",
                f"{final_holdings:.6f} {symbol.split('/')[0] if symbol else 'units'}"
            )
        else:
            st.metric(
                "Current Position", 
                "🔵 CASH",
                f"${final_cash:,.2f}"
            )
    
    with col2:
        st.metric(
            "Position Value",
            f"${position_value:,.2f}",
            f"{position_pct:.1f}% of portfolio"
        )
    
    with col3:
        st.metric(
            "Current Price",
            f"${current_price:,.2f}",
            f"{symbol.split('/')[0] if symbol else 'Asset'}"
        )
    
    with col4:
        if summary['total_trades'] > 0:
            avg_profit_per_trade = (summary['final_value'] - summary['initial_capital']) / summary['total_trades']
            st.metric(
                "Avg Profit/Trade",
                f"${avg_profit_per_trade:,.2f}",
                f"From {summary['total_trades']} trades"
            )
        else:
            st.metric(
                "Total Trades",
                "0",
                "No trades executed"
            )
    
    # Trading activity breakdown
    if summary['total_trades'] > 0:
        st.markdown("#### 📊 Trading Activity Breakdown")
        
        activity_col1, activity_col2, activity_col3 = st.columns(3)
        
        with activity_col1:
            # Calculate buy vs sell trades
            buy_trades = sum(1 for trade in summary['trades'] if trade['action'] == 'BUY')
            sell_trades = sum(1 for trade in summary['trades'] if trade['action'] == 'SELL')
            
            st.markdown(f"""
            **Trade Distribution:**
            - 🟢 Buy Orders: {buy_trades}
            - 🔴 Sell Orders: {sell_trades}
            - 📈 Win Rate: {summary['win_rate_pct']:.1f}%
            """)
        
        with activity_col2:
            # Calculate average holding period
            if len(summary['trades']) >= 2:
                buy_times = [pd.to_datetime(trade['timestamp']) for trade in summary['trades'] if trade['action'] == 'BUY']
                sell_times = [pd.to_datetime(trade['timestamp']) for trade in summary['trades'] if trade['action'] == 'SELL']
                
                if buy_times and sell_times:
                    holding_periods = []
                    for i, sell_time in enumerate(sell_times):
                        if i < len(buy_times):
                            period = (sell_time - buy_times[i]).total_seconds() / 3600  # hours
                            holding_periods.append(period)
                    
                    if holding_periods:
                        avg_holding = sum(holding_periods) / len(holding_periods)
                        st.markdown(f"""
                        **Timing Analysis:**
                        - ⏱️ Avg Hold: {avg_holding:.1f} hours
                        - 📅 Period: {(df.index[-1] - df.index[0]).days} days
                        - 🔄 Frequency: {summary['total_trades'] / max(1, (df.index[-1] - df.index[0]).days):.1f} trades/day
                        """)
                    else:
                        st.markdown("**Timing Analysis:**\n- No completed trades")
                else:
                    st.markdown("**Timing Analysis:**\n- Incomplete trade data")
            else:
                st.markdown("**Timing Analysis:**\n- Insufficient trade data")
        
        with activity_col3:
            # Calculate profit/loss breakdown
            profitable_amount = sum(trade.get('profit', 0) for trade in summary['trades'] if trade.get('profit', 0) > 0)
            loss_amount = sum(trade.get('profit', 0) for trade in summary['trades'] if trade.get('profit', 0) < 0)
            
            st.markdown(f"""
            **P&L Breakdown:**
            - 💰 Total Gains: ${profitable_amount:,.2f}
            - 💸 Total Losses: ${abs(loss_amount):,.2f}
            - 📊 Net P&L: ${profitable_amount + loss_amount:,.2f}
            """)
    else:
        st.info("ℹ️ No trades were executed during this simulation. Consider adjusting strategy parameters for more active trading.")
    
    # Portfolio value chart
    st.subheader("📈 Portfolio Performance")
    
    if summary['daily_values']:
        chart_df = pd.DataFrame(summary['daily_values'])
        chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
        chart_df.set_index('timestamp', inplace=True)
        
        # Add buy & hold comparison. This must use the same first traded bar as
        # the engine's buy_hold_return (see the summary calculation), or the
        # chart and the reported number quietly disagree -- which is why the
        # index is the shared constant and not a literal repeated here.
        baseline_idx = min(INDICATOR_WARMUP_BARS, len(df) - 1)
        initial_price = df.iloc[baseline_idx]['close']
        chart_df['buy_hold_value'] = summary['initial_capital'] * (chart_df['price'] / initial_price)
        
        st.line_chart(chart_df[['portfolio_value', 'buy_hold_value']])

    # Professional TradingView-Style Technical Analysis Chart
    st.subheader("📊 Technical Analysis Dashboard")
    st.info("📈 **Charts Display:** Technical indicators with BUY/SELL trading signals")
     # Chart generation progress indicator
    with st.spinner("🔄 Generating technical analysis charts..."):
        # Check for required data
        required_columns = ['open', 'high', 'low', 'close', 'MA20', 'MA50', 'RSI', 'MACD_line', 'MACD_signal', 'MACD_hist', 'UpperBB', 'LowerBB']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"Missing technical indicator columns: {missing_columns}")
            st.info("Charts may have limited functionality. Ensure technical indicators are calculated.")
        
        if df.empty:
            st.error("No data available for charting.")
            return
        
        # Create professional TradingView-style chart
        chart_created = False
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots: Price + Volume + RSI + MACD
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Price Action & Technical Indicators', 'Volume', 'RSI (14)', 'MACD'),
                vertical_spacing=0.05,
                row_heights=[0.5, 0.2, 0.15, 0.15],
                shared_xaxes=True
            )
            
            # 1. Main Price Chart with Candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price",
                    increasing_line_color='#00FF88',  # Green for up candles
                    decreasing_line_color='#FF4444',  # Red for down candles
                    increasing_fillcolor='#00FF88',
                    decreasing_fillcolor='#FF4444'
                ),
                row=1, col=1
            )
            
            # 2. Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['UpperBB'],
                    mode='lines',
                    name='Upper BB',
                    line=dict(color='#9C27B0', width=1, dash='dot'),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['LowerBB'],
                    mode='lines',
                    name='Lower BB',
                    line=dict(color='#9C27B0', width=1, dash='dot'),
                    fill='tonexty',  # Fill between upper and lower BB
                    fillcolor='rgba(156, 39, 176, 0.1)',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # 3. Moving Averages
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['MA20'],
                    mode='lines',
                    name='MA20',
                    line=dict(color='#FFC107', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['MA50'],
                    mode='lines',
                    name='MA50',
                    line=dict(color='#FF9800', width=2)
                ),
                row=1, col=1
            )
            
            # 4. Add Buy/Sell Trading Signals from decisions_log
            if decisions_log:
                buy_signals_x = []
                buy_signals_y = []
                sell_signals_x = []
                sell_signals_y = []
                
                for decision_entry in decisions_log:
                    if 'decision' in decision_entry and hasattr(decision_entry['decision'], 'action'):
                        timestamp = decision_entry['timestamp']
                        action = decision_entry['decision'].action
                        price = decision_entry['decision'].price
                        
                        # Only show actual BUY/SELL decisions, not HOLD
                        if action.value == 'BUY':
                            buy_signals_x.append(timestamp)
                            buy_signals_y.append(price)
                        elif action.value == 'SELL':
                            sell_signals_x.append(timestamp)
                            sell_signals_y.append(price)
                
                # Add buy signals (green arrows pointing up)
                if buy_signals_x:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals_x,
                            y=buy_signals_y,
                            mode='markers',
                            name='BUY Signals',
                            marker=dict(
                                symbol='triangle-up',
                                size=15,
                                color='#00FF88',
                                line=dict(color='#FFFFFF', width=2)
                            )
                        ),
                        row=1, col=1
                    )
                
                # Add sell signals (red arrows pointing down)
                if sell_signals_x:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals_x,
                            y=sell_signals_y,
                            mode='markers',
                            name='SELL Signals',
                            marker=dict(
                                symbol='triangle-down',
                                size=15,
                                color='#FF4444',
                                line=dict(color='#FFFFFF', width=2)
                            )
                        ),
                        row=1, col=1
                    )
            
            # 5. Volume Chart
            if 'volume' in df.columns:
                colors = ['#00FF88' if close >= open else '#FF4444' 
                         for close, open in zip(df['close'], df['open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # 6. RSI Chart
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='#2196F3', width=2)
                ),
                row=3, col=1
            )
            
            # RSI Overbought/Oversold levels
            fig.add_hline(y=70, line=dict(color='#FF4444', width=1, dash='dash'), row=3, col=1)
            fig.add_hline(y=30, line=dict(color='#00FF88', width=1, dash='dash'), row=3, col=1)
            fig.add_hline(y=50, line=dict(color='#666666', width=1, dash='dot'), row=3, col=1)
            
            # 7. MACD Chart
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['MACD_line'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='#2196F3', width=2)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['MACD_signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='#FF9800', width=2)
                ),
                row=4, col=1
            )
            
            # MACD Histogram
            colors = ['#00FF88' if val >= 0 else '#FF4444' for val in df['MACD_hist']]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_hist'],
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=4, col=1
            )
            
            # Professional styling
            fig.update_layout(
                title={
                    'text': f"Technical Analysis Dashboard - {symbol or 'Asset'} with Trading Signals",
                    'x': 0.5,
                    'font': {'size': 20, 'color': '#FFFFFF'}
                },
                paper_bgcolor='#1E1E1E',  # Dark background
                plot_bgcolor='#1E1E1E',
                font=dict(color='#FFFFFF'),
                height=800,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color='#FFFFFF')
                )
            )
            
            # Update all axes for professional look
            for i in range(1, 5):
                fig.update_xaxes(
                    gridcolor='#333333',
                    gridwidth=1,
                    tickcolor='#FFFFFF',
                    linecolor='#333333',
                    row=i, col=1
                )
                fig.update_yaxes(
                    gridcolor='#333333',
                    gridwidth=1,
                    tickcolor='#FFFFFF',
                    linecolor='#333333',
                    row=i, col=1
                )
            
            # Remove range slider (professional charts don't typically have them)
            fig.update_layout(xaxis_rangeslider_visible=False)
            
            # Display the professional chart
            st.plotly_chart(fig, use_container_width=True)
            chart_created = True
            
        except ImportError:
            st.warning("Install plotly for professional charts: `pip install plotly`")
            # Fallback to simple charts
            st.write("**📊 Fallback Charts:**")
            
            # Simple price chart
            st.write("**Price Chart:**")
            price_df = df[['close', 'MA20', 'MA50']].copy()
            price_df.columns = ['Close Price', 'MA20', 'MA50']
            st.line_chart(price_df)
            
            # Simple RSI chart
            if 'RSI' in df.columns:
                st.write("**RSI Chart:**")
                st.line_chart(df[['RSI']])
            
            # Simple MACD chart
            if 'MACD_line' in df.columns and 'MACD_signal' in df.columns:
                st.write("**MACD Chart:**")
                macd_df = df[['MACD_line', 'MACD_signal']].copy()
                macd_df.columns = ['MACD Line', 'Signal Line']
                st.line_chart(macd_df)
                
        except Exception as e:
            st.error(f"Error creating professional chart: {str(e)}")
            st.write("**📊 Fallback Charts:**")
            
            try:
                # Simple price chart
                st.write("**Price Chart:**")
                price_df = df[['close', 'MA20', 'MA50']].copy()
                price_df.columns = ['Close Price', 'MA20', 'MA50']
                st.line_chart(price_df)
                
                # Simple RSI chart
                if 'RSI' in df.columns:
                    st.write("**RSI Chart:**")
                    st.line_chart(df[['RSI']])
                
                # Simple MACD chart
                if 'MACD_line' in df.columns and 'MACD_signal' in df.columns:
                    st.write("**MACD Chart:**")
                    macd_df = df[['MACD_line', 'MACD_signal']].copy()
                    macd_df.columns = ['MACD Line', 'Signal Line']
                    st.line_chart(macd_df)
                    
            except Exception as fallback_error:
                st.error(f"Error creating fallback charts: {str(fallback_error)}")
                st.write("**Chart data summary:**")
                st.write(f"DataFrame shape: {df.shape}")
                st.write(f"Available columns: {list(df.columns)}")
                if not df.empty:
                    st.write("**Sample data:**")
                    st.dataframe(df.head())
    
    # Chart completion message (shown after spinner completes)
    st.success("✅ Technical analysis charts generated successfully!")

    # Trade log with P&L and P&L%. The "Sentiment Context" column was removed
    # with the legacy channel: it only ever showed 0.000 for every trade.
    if summary['trades']:
        st.subheader("📋 Trade History with P&L Analysis")
        trades_df = pd.DataFrame(summary['trades'])
        
        # Format the trades DataFrame for better display
        if not trades_df.empty:
            # Format timestamp column
            if 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Format price columns with proper handling of NaN values
            if 'price' in trades_df.columns:
                trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "$0.00")
            if 'cost' in trades_df.columns:
                trades_df['cost'] = trades_df['cost'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "$0.00")
            if 'proceeds' in trades_df.columns:
                trades_df['proceeds'] = trades_df['proceeds'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "$0.00")
            
            # Calculate P&L and P&L% for each trade with proper percentage calculation
            if 'profit' not in trades_df.columns:
                trades_df['profit'] = 0.0
            if 'profit_pct' not in trades_df.columns:
                # Calculate profit percentage based on investment amount
                trades_df['profit_pct'] = 0.0
                
                # Track paired trades for proper P&L calculation
                buy_trades = {}  # Store buy trades by index
                
                for idx, row in trades_df.iterrows():
                    action = row.get('action', '')
                    profit = row.get('profit', 0) if pd.notna(row.get('profit', 0)) else 0
                    
                    if action == 'BUY':
                        # Store BUY trade details for later P&L calculation
                        cost = row.get('cost', 0)
                        if isinstance(cost, str):
                            try:
                                cost = float(cost.replace('$', '').replace(',', ''))
                            except:
                                cost = 0
                        buy_trades[idx] = cost
                        
                        # BUY trades: Always start with 0% and only update if there's actual profit
                        trades_df.at[idx, 'profit_pct'] = 0.0
                        
                    elif action == 'SELL':
                        # For SELL trades, always check if profit is actually non-zero
                        if abs(profit) > 0.01:  # Only calculate percentage if there's real profit
                            # Find the most recent BUY trade before this SELL
                            investment_amount = 0
                            corresponding_buy_idx = None
                            for buy_idx in reversed(range(idx)):
                                if buy_idx in buy_trades:
                                    investment_amount = buy_trades[buy_idx]
                                    corresponding_buy_idx = buy_idx
                                    break
                            
                            if investment_amount > 0:
                                profit_pct = (profit / investment_amount) * 100
                                trades_df.at[idx, 'profit_pct'] = profit_pct
                                
                                # Only update the BUY trade if there's actual profit
                                if corresponding_buy_idx is not None:
                                    trades_df.at[corresponding_buy_idx, 'profit_pct'] = profit_pct
                            else:
                                trades_df.at[idx, 'profit_pct'] = 0.0
                        else:
                            # For SELL trades with zero or negligible profit
                            trades_df.at[idx, 'profit_pct'] = 0.0
                            
                            # Make sure the corresponding BUY trade also shows 0%
                            for buy_idx in reversed(range(idx)):
                                if buy_idx in buy_trades:
                                    trades_df.at[buy_idx, 'profit_pct'] = 0.0
                                    break
                    else:
                        # For any other case
                        trades_df.at[idx, 'profit_pct'] = 0.0
            
            # Format profit columns with proper display names
            trades_df['profit_display'] = trades_df['profit'].apply(
                lambda x: f"${x:+.2f}" if pd.notna(x) else "$0.00"
            )
            
            # Enhanced P&L% formatting: If profit is $0.00, then P&L% must be 0.00%
            def format_profit_pct(row):
                profit = row.get('profit', 0) if pd.notna(row.get('profit', 0)) else 0
                profit_pct = row.get('profit_pct', 0) if pd.notna(row.get('profit_pct', 0)) else 0
                
                # If profit is exactly zero, force percentage to be 0.00%
                if abs(profit) < 0.01:  # Profit is essentially zero
                    return "0.00%"
                # Otherwise, format the actual percentage
                elif abs(profit_pct) > 0.001:
                    return f"{profit_pct:+.2f}%"
                else:
                    return "0.00%"
            
            trades_df['profit_pct_display'] = trades_df.apply(format_profit_pct, axis=1)
            
            # Reorder columns for better presentation (remove fees column)
            display_columns = ['timestamp', 'action', 'price', 'shares']
            
            # Add cost or proceeds based on action
            if 'cost' in trades_df.columns:
                display_columns.append('cost')
            if 'proceeds' in trades_df.columns:
                display_columns.append('proceeds')
            
            # Add P&L columns
            display_columns.extend(['profit_display', 'profit_pct_display'])
            
            # Add other relevant columns (excluding fees)
            other_columns = ['confidence', 'reasoning']
            for col in other_columns:
                if col in trades_df.columns:
                    display_columns.append(col)
            
            # Filter to display columns that exist
            final_columns = [col for col in display_columns if col in trades_df.columns]
            display_df = trades_df[final_columns].copy()
            
            # Rename display columns for better presentation
            display_df = display_df.rename(columns={
                'profit_display': 'P&L',
                'profit_pct_display': 'P&L %',
            })
            
            # Apply color formatting for profit columns
            def color_profit_cell(val):
                if isinstance(val, str) and ('$' in val or '%' in val):
                    try:
                        # Extract numeric value
                        numeric_val = float(val.replace('$', '').replace('%', '').replace('+', ''))
                        if numeric_val > 0:
                            return 'color: #00FF88; font-weight: bold'  # Green for profit
                        elif numeric_val < 0:
                            return 'color: #FF4444; font-weight: bold'  # Red for loss
                    except (ValueError, AttributeError):
                        pass
                return ''
            
            # Style the dataframe
            try:
                if 'P&L' in display_df.columns or 'P&L %' in display_df.columns:
                    styled_df = display_df.style
                    
                    # Apply styling to profit columns
                    if 'P&L' in display_df.columns:
                        styled_df = styled_df.map(color_profit_cell, subset=['P&L'])
                    if 'P&L %' in display_df.columns:
                        styled_df = styled_df.map(color_profit_cell, subset=['P&L %'])
                    
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(display_df, use_container_width=True)
            except Exception:
                # Fallback: display without styling
                st.dataframe(display_df, use_container_width=True)
            
        else:
            st.info("No trades executed during the simulation.")
    else:
        st.info("No trades executed during the simulation.")
    

def save_results_to_db(summary, df, symbol, start_date, end_date, interval):
    """Save simulation results to database"""
    try:
        # This would normally save to a database
        # For now, just log the results
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'interval': interval,
            'summary': summary
        }
        
        # In a real implementation, this would save to SQLite or other database
        logger.info(f"Simulation results: {json.dumps(result_data, default=str)}")
        st.success("✅ Results logged successfully")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        st.warning("Could not save results to database")

# ── LANGCHAIN AGENTS AND TOOLS ────────────────────────────────────────────────────
# Define the TechnicalAnalysisTool first (used by MarketAnalystAgent)
class TechnicalAnalysisTool(BaseTool):
    """Tool for accessing technical analysis data"""
    name: str = "technical_analysis"
    description: str = "Get technical indicators for a specific time point"
    df: pd.DataFrame = Field(description="DataFrame containing market data")
    
    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(df=df, **kwargs)
    
    def _run(self, timestamp_str: str) -> str:
        try:
            # Convert input string to datetime
            timestamp = pd.to_datetime(timestamp_str)
            
            # Find the closest timestamp in the dataframe
            closest_idx = resolve_bar_index(self.df.index, timestamp)
            data = self.df.iloc[closest_idx]
            
            # Format response
            response = {
                "timestamp": str(self.df.index[closest_idx]),
                "price": round(data['close'], 2),
                "ma20": round(data['MA20'], 2),
                "ma50": round(data['MA50'], 2),
                "rsi": round(data['RSI'], 2),
                "upper_bb": round(data['UpperBB'], 2),
                "lower_bb": round(data['LowerBB'], 2),
                "macd_line": round(data['MACD_line'], 4),
                "macd_signal": round(data['MACD_signal'], 4),
                "macd_histogram": round(data['MACD_hist'], 4)
            }
            
            return json.dumps(response, indent=2)
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

# ── LLM-FREE SUPPORT AGENT OUTPUTS ──────────────────────────────────────────
# USE_LLM_DECISIONS=false has to mean "no model call anywhere", not "no model
# call in the decision agent". Until 2026-09-11 it gated only
# TradingDecisionAgent while the market, pattern and risk agents still invoked
# the model on every bar -- about 570 calls and ~47 minutes per run -- so the
# paper's LLM-free baseline was neither LLM-free nor reproducible.
#
# The risk agent was the one that actually mattered: its risk_level drives
# confidence_multiplier = {"low": 1.2, "high": 0.8}, and with moderate-mode
# min_confidence=0.65 a "high" reading drops a net_signal of 2 from 0.80 to
# 0.64, turning a BUY into a HOLD. One LLM word could therefore flip a trade in
# the arm that was supposed to contain no LLM at all.
#
# When the LLM is off these agents are ABSENT, not silent, so the neutral
# values below are the honest representation: "medium" is the only risk level
# whose multiplier is 1.0, i.e. it applies no tilt. Nothing here is tuned.
LLM_DISABLED_MARKET_ANALYSIS = (
    "Market analyst agent disabled for this run (USE_LLM_DECISIONS=false)."
)
LLM_DISABLED_PATTERN_ANALYSIS = (
    "Pattern recognition agent disabled for this run (USE_LLM_DECISIONS=false)."
)


def llm_disabled_risk(volatility_pct: float = None) -> dict:
    """Neutral risk assessment used when the LLM is switched off.

    `risk_level="medium"` is deliberate: it is the only value that leaves the
    confidence multiplier at 1.0, so disabling the LLM removes its influence
    instead of replacing it with a different constant tilt. Volatility is still
    reported because it is computed from price data and needs no model.
    """
    return {
        "risk_level": "medium",
        "position_size_pct": 25.0,
        "stop_loss_price": None,
        "take_profit_price": None,
        "risk_reward_ratio": None,
        "volatility_pct": volatility_pct,
        "reasoning": "Risk agent disabled (USE_LLM_DECISIONS=false); "
                     "neutral assessment applied, no confidence tilt.",
        "source": "llm_disabled",
    }


class MarketAnalystAgent:
    """LangChain-based market analyst agent"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.use_llm = USE_LLM_DECISIONS
        if not self.use_llm:
            # Deliberately do not call get_llm(): the rules-only arm must run
            # with no provider configured at all, which is also what makes it
            # cheap enough to repeat across windows.
            self.llm = None
            self.technical_tool = None
            self.agent_executor = None
            return
        self.llm = get_llm()
        self.technical_tool = TechnicalAnalysisTool(df)
        
        # Create agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a professional market analyst with expertise in technical analysis.
            Your role is to analyze market data and provide insights based on technical indicators.
            
            Available tools:
            - technical_analysis: Get technical indicators for a specific time point
            
            Always provide:
            1. Current market conditions
            2. Technical indicator analysis
            3. Support and resistance levels
            4. Market sentiment assessment
            
            Be objective and data-driven in your analysis.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent with tools
        self.tools = [
            Tool(
                name="technical_analysis",
                func=self.technical_tool._run,
                description="Get technical indicators for a specific time point"
            )
        ]
        
        # Create the agent with the LangChain agent executor
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False)
    
    def analyze(self, timestamp) -> str:
        """Run the market analyst agent"""
        if not self.use_llm:
            return LLM_DISABLED_MARKET_ANALYSIS
        response = self.agent_executor.invoke({
            "input": f"Analyze the market at timestamp {timestamp}. Focus on key indicators and technical signals.",
            "chat_history": []
        })

        return response["output"]

class PatternRecognitionAgent:
    """Specialized agent for identifying chart patterns"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.use_llm = USE_LLM_DECISIONS
        if not self.use_llm:
            self.llm = None
            self.agent_executor = None
            return
        self.llm = get_llm()

        # Create agent prompt for pattern recognition
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a pattern recognition specialist in trading.
            Your role is to identify chart patterns and potential market setups.
            
            Focus on these patterns:
            - Trend patterns (uptrends, downtrends, consolidations)
            - Reversal patterns (head & shoulders, double tops/bottoms)
            - Continuation patterns (flags, pennants)
            - Support/resistance levels & breakouts
            - Candlestick patterns (engulfing, doji, hammers)
            
            Always provide:
            1. Key patterns identified
            2. Strength of the pattern (weak/moderate/strong)
            3. Potential price targets
            4. Confirmation signals to watch for
            
            Be thorough in your analysis but focus on actionable insights.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent executor (without tools for now)
        self.agent_executor = self.llm
    
    def identify_patterns(self, timestamp) -> str:
        """Identify patterns around the given timestamp"""
        if not self.use_llm:
            return LLM_DISABLED_PATTERN_ANALYSIS
        try:
            # Get a relevant window of data (20 periods before timestamp)
            timestamp_dt = pd.to_datetime(timestamp)
            closest_idx = resolve_bar_index(self.df.index, timestamp_dt)
            
            window_start = max(0, closest_idx - 20)
            window_end = closest_idx + 1
                
            window_df = self.df.iloc[window_start:window_end].copy()
            
            # Prepare data summary for the LLM
            price_data = []
            for idx, row in window_df.iterrows():
                price_data.append({
                    "date": str(idx.date()),
                    "close": round(row['close'], 2),
                    "open": round(row['open'], 2),
                    "high": round(row['high'], 2),
                    "low": round(row['low'], 2),
                    "ma20": round(row['MA20'], 2),
                    "rsi": round(row['RSI'], 2)
                })
            
            # Describe recent price action
            recent_change = ((window_df['close'].iloc[-1] - window_df['close'].iloc[0]) / 
                             window_df['close'].iloc[0]) * 100
            
            # Create prompt with data
            messages = [
                SystemMessage(content="""You are a pattern recognition specialist in trading.
                Your role is to identify chart patterns and market setups from price data."""),
                HumanMessage(content=f"""
                I need pattern analysis for a trading session. Recent price change: {recent_change:.2f}%
                
                Here's recent price data (most recent last):
                {json.dumps(price_data[-5:], indent=2)}
                
                Identify any technical patterns forming or completing.
                Focus on actionable insights and pattern reliability.
                """)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error identifying patterns: {str(e)}"

class RiskManagementAgent:
    """Agent for risk assessment and management"""
    
    def __init__(self, df: pd.DataFrame, config: TradingConfig):
        self.df = df
        self.config = config
        self.use_llm = USE_LLM_DECISIONS
        if not self.use_llm:
            self.llm = None
            self.agent_executor = None
            self.structured_llm = None
            return
        self.llm = get_llm()

        # Create risk management prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            You are a risk management specialist for trading operations.
            Your role is to assess risk levels and provide risk management advice.
            
            Configuration parameters:
            - Stop loss percentage: {config.stop_loss_pct:.1f}%
            - Take profit percentage: {config.take_profit_pct:.1f}%
            - Max position size: {config.max_position_size:.2f} of capital
            - RSI oversold level: {config.rsi_oversold}
            - RSI overbought level: {config.rsi_overbought}
            
            Always provide:
            1. Current risk assessment (low/medium/high)
            2. Volatility analysis
            3. Suggested position size
            4. Stop loss and take profit recommendations
            5. Risk-reward ratio calculation
            
            Be conservative in your risk assessment - capital preservation comes first.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent executor (without tools for now)
        self.agent_executor = self.llm

        # Structured channel so the model's risk_level actually reaches the
        # decision agent instead of being flattened to a hardcoded "medium".
        try:
            self.structured_llm = self.llm.with_structured_output(LLMRiskAssessment)
        except Exception as e:
            logger.error(f"Risk agent structured output unavailable: {e}")
            self.structured_llm = None

    def assess_risk(self, timestamp, action: TradingAction = None, portfolio: dict = None, last_decision=None) -> dict:
        """Assess risk for a potential trade"""
        try:
            # Get relevant window of data (20 periods before timestamp)
            timestamp_dt = pd.to_datetime(timestamp)
            idx = resolve_bar_index(self.df.index, timestamp_dt)
            
            window_start = max(0, idx - 20)
            window_end = idx + 1
            window_df = self.df.iloc[window_start:window_end].copy()
            
            # Calculate recent volatility
            volatility = window_df['close'].pct_change().std() * 100

            # Volatility is computed above because it needs no model; the rest
            # of this method is entirely LLM work, so stop here when it is off.
            if not self.use_llm:
                return llm_disabled_risk(float(volatility))

            # Current price and indicators
            current_data = window_df.iloc[-1]
            current_price = current_data['close']
            current_rsi = current_data['RSI']
            
            # Determine if market is in oversold/overbought condition
            market_condition = "neutral"
            if current_rsi <= self.config.rsi_oversold:
                market_condition = "oversold"
            elif current_rsi >= self.config.rsi_overbought:
                market_condition = "overbought"
                
            # Portfolio context
            portfolio_context = ""
            if portfolio:
                portfolio_context = f"""
                Current portfolio:
                - Cash: ${portfolio['cash']:.2f}
                - Holdings: {portfolio['holdings']}
                - Currently holding: {portfolio['holding']}
                - Entry price (if holding): ${portfolio['entry_price']:.2f}
                """
            
            # Create contextual prompt
            action_str = str(action.value) if action else "ANALYSIS"
            prompt = f"""
            Risk assessment requested for {action_str} at {timestamp}.
            
            Market context:
            - Current price: ${current_price:.2f}
            - Recent volatility: {volatility:.2f}%
            - RSI: {current_rsi:.2f} ({market_condition})
            - BB Width: {(current_data['UpperBB'] - current_data['LowerBB']) / current_data['MidBB']:.4f}
            {portfolio_context}
            
            Provide a risk assessment with:
            1. Risk level (low/medium/high)
            2. Recommended position size (% of capital)
            3. Suggested stop loss price 
            4. Suggested take profit price
            5. Risk-to-reward ratio
            """
            
            # PHASE 1 FIX: the LLM's answer is now PARSED, not discarded.
            # Previously this method called the model and then returned a dict of
            # hardcoded defaults with risk_level pinned to "medium", so the only
            # channel from the LLM to a trade was permanently dead.
            messages = [
                SystemMessage(content=(
                    "You are a risk management specialist for trading operations. "
                    "Assess risk conservatively; capital preservation comes first."
                )),
                HumanMessage(content=prompt)
            ]

            structured = self.structured_llm.invoke(messages) if self.structured_llm else None

            if structured is None:
                raise RuntimeError("Structured risk output unavailable")

            level = str(structured.risk_level).strip().lower()
            if level not in ("low", "medium", "high"):
                level = "medium"

            # Position size is advisory, so clamp it into a sane band rather
            # than trusting the model with unbounded leverage.
            size = float(structured.position_size_pct or 5.0)
            size = max(1.0, min(100.0, size))

            return {
                "risk_level": level,
                "position_size_pct": size,
                "stop_loss_price": float(structured.stop_loss_price) if structured.stop_loss_price
                                   else current_price * (1 - self.config.stop_loss_pct),
                "take_profit_price": float(structured.take_profit_price) if structured.take_profit_price
                                     else current_price * (1 + self.config.take_profit_pct),
                "risk_reward_ratio": float(structured.risk_reward_ratio or 2.0),
                "volatility_pct": float(volatility),
                "reasoning": str(structured.reasoning or ""),
                "source": "llm",
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {
                "risk_level": "high",
                "position_size_pct": 1.0,  # Conservative default
                "stop_loss_price": None,
                "take_profit_price": None,
                "risk_reward_ratio": None,
                "reasoning": f"Error in risk assessment: {str(e)}",
                # Marked so a run can report how many risk calls actually
                # reached the model rather than silently using the fallback.
                "source": "fallback",
            }


class TradingDecisionAgent:
    """Final decision maker: rule engine proposes, LLM adjusts within bounds.

    Design rationale for the paper. A rule engine alone is deterministic but
    blind to context; an LLM alone is unbounded and unauditable. Here the rules
    produce a signal score, the LLM returns a clamped integer adjustment plus an
    optional risk veto, and the same threshold logic maps both the pre- and
    post-adjustment score to an action. That yields a per-decision measurement
    of exactly what the model contributed.
    """

    def __init__(self, df: pd.DataFrame, config: TradingConfig):
        self.df = df
        self.config = config
        self.use_llm = USE_LLM_DECISIONS
        # Only build a provider client when one will actually be used, so the
        # rules-only arm runs with no API key configured at all.
        self.llm = get_llm() if self.use_llm else None
        self.structured_llm = None
        if self.use_llm:
            try:
                self.structured_llm = self.llm.with_structured_output(LLMTradeAdjustment)
            except Exception as e:
                logger.error(f"Structured output unavailable, falling back to rules only: {e}")
                self.use_llm = False
        # Aggregate telemetry across the whole run
        self.stats = {
            "decisions": 0, "llm_calls": 0, "llm_failures": 0,
            "changed": 0, "vetoes": 0, "total_latency_s": 0.0,
        }

    # ── Rule engine (unchanged logic, now isolated as the baseline arm) ──
    def _compute_rule_signals(self, idx: int, current_data, current_price: float,
                              positioning=None, text_sentiment=None) -> dict:
        """Technical rule score, optionally plus the exogenous positioning score.

        PHASE 3: positioning enters the RULE engine, not only the LLM prompt.
        That distinction is what makes the ablation identifiable: if the signal
        reached the decision solely through the prompt, then in the
        USE_LLM_DECISIONS=false arm it would contribute nothing, and "positioning
        adds information" could not be separated from "the LLM adds information".
        """
        rsi = current_data['RSI']
        ma20 = current_data['MA20']
        ma50 = current_data['MA50']
        macd_hist = current_data['MACD_hist']
        macd_line = current_data['MACD_line']
        macd_signal = current_data['MACD_signal']

        bullish_signals = 0
        bearish_signals = 0
        signal_details = []

        # RSI Analysis
        if rsi < 30:
            bullish_signals += 2
            signal_details.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            bearish_signals += 2
            signal_details.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 40:
            bullish_signals += 1
            signal_details.append(f"RSI bullish ({rsi:.1f})")
        elif rsi > 60:
            bearish_signals += 1
            signal_details.append(f"RSI bearish ({rsi:.1f})")

        # Moving Average Analysis
        if current_price > ma20 > ma50:
            bullish_signals += 2
            signal_details.append("Price above MA20 & MA50 (uptrend)")
        elif current_price > ma20:
            bullish_signals += 1
            signal_details.append("Price above MA20")
        elif current_price < ma20 < ma50:
            bearish_signals += 2
            signal_details.append("Price below MA20 & MA50 (downtrend)")
        elif current_price < ma20:
            bearish_signals += 1
            signal_details.append("Price below MA20")

        # MACD Analysis
        if macd_line > macd_signal and macd_hist > 0:
            bullish_signals += 1
            signal_details.append("MACD bullish crossover")
        elif macd_line < macd_signal and macd_hist < 0:
            bearish_signals += 1
            signal_details.append("MACD bearish crossover")

        # Momentum
        momentum = 0.0
        if idx >= 5:
            price_5_ago = self.df.iloc[idx - 5]['close']
            momentum = (current_price - price_5_ago) / price_5_ago
            if momentum > 0.02:
                bullish_signals += 1
                signal_details.append(f"Strong momentum (+{momentum*100:.1f}%)")
            elif momentum < -0.02:
                bearish_signals += 1
                signal_details.append(f"Negative momentum ({momentum*100:.1f}%)")

        # Positioning (exogenous, non-price). Contributes only when the flag is
        # on AND a real point-in-time reading exists. A missing reading adds
        # nothing; it is never treated as a neutral measurement.
        pos_points = 0
        pos_score = None
        if positioning is not None and getattr(positioning, "available", False):
            pos_score = float(positioning.score)
            bullish_signals += int(positioning.bullish_points)
            bearish_signals += int(positioning.bearish_points)
            pos_points = int(positioning.bullish_points) - int(positioning.bearish_points)
            if pos_points:
                direction = "bullish" if pos_points > 0 else "bearish"
                signal_details.append(
                    f"Futures positioning {direction} ({pos_score:+.2f})"
                )

        # Text sentiment (exogenous, non-price). Same contract as positioning:
        # contributes only when a real point-in-time reading exists, and a
        # missing reading adds nothing rather than voting neutral.
        text_points = 0
        text_score = None
        if text_sentiment is not None and getattr(text_sentiment, "available", False):
            text_score = float(text_sentiment.score)
            bullish_signals += int(text_sentiment.bullish_points)
            bearish_signals += int(text_sentiment.bearish_points)
            text_points = (int(text_sentiment.bullish_points)
                           - int(text_sentiment.bearish_points))
            if text_points:
                direction = "bullish" if text_points > 0 else "bearish"
                signal_details.append(
                    f"Text sentiment {direction} ({text_score:+.2f}, "
                    f"{text_sentiment.doc_count} docs)"
                )

        return {
            "bullish": bullish_signals,
            "bearish": bearish_signals,
            "net_signal": bullish_signals - bearish_signals,
            "details": signal_details,
            "rsi": rsi, "ma20": ma20, "ma50": ma50,
            "macd_hist": macd_hist, "momentum": momentum,
            "positioning_score": pos_score,
            "positioning_points": pos_points,
            "text_sentiment_score": text_score,
            "text_sentiment_points": text_points,
        }

    def _signal_to_action(self, net_signal: int, sig: dict, portfolio: dict,
                          current_price: float, confidence_multiplier: float) -> tuple:
        """Map a signal score to (action, confidence, reasoning).

        Called twice per decision: once on the rule score to establish the
        baseline, once on the adjusted score to get the final action.
        """
        rsi = sig["rsi"]
        ma20 = sig["ma20"]
        bullish_signals = sig["bullish"]
        bearish_signals = sig["bearish"]
        signal_details = sig["details"]

        action = TradingAction.HOLD
        confidence = 0.6
        reasoning = "Default HOLD position"

        if net_signal >= self.config.signal_threshold and not portfolio['holding']:
            action = TradingAction.BUY
            confidence = min(0.9, 0.6 + (net_signal * 0.1)) * confidence_multiplier
            reasoning = f"BUY Signal: {bullish_signals} bullish vs {bearish_signals} bearish signals. Details: {', '.join(signal_details)}"

        elif net_signal <= -self.config.signal_threshold and portfolio['holding']:
            action = TradingAction.SELL
            confidence = min(0.9, 0.6 + (abs(net_signal) * 0.1)) * confidence_multiplier
            reasoning = f"SELL Signal: {bearish_signals} bearish vs {bullish_signals} bullish signals. Details: {', '.join(signal_details)}"

        elif portfolio['holding']:
            if rsi > 75:
                action = TradingAction.SELL
                confidence = 0.8 * confidence_multiplier
                reasoning = f"SELL: RSI extremely overbought ({rsi:.1f}). Risk management."
            elif portfolio.get('entry_price', 0) > 0:
                loss_pct = (current_price - portfolio['entry_price']) / portfolio['entry_price']
                if loss_pct < -self.config.stop_loss_pct:
                    action = TradingAction.SELL
                    confidence = 0.9
                    reasoning = f"SELL: Stop loss triggered. Loss: {loss_pct*100:.1f}%"
                elif loss_pct > self.config.take_profit_pct:
                    action = TradingAction.SELL
                    confidence = 0.8
                    reasoning = f"SELL: Take profit triggered. Gain: {loss_pct*100:.1f}%"

        elif not portfolio['holding']:
            if rsi < self.config.rsi_oversold + 5 and current_price > ma20:
                action = TradingAction.BUY
                confidence = 0.8 * confidence_multiplier
                reasoning = f"BUY: Oversold in uptrend. RSI: {rsi:.1f}, Price > MA20"
            elif self.config.trading_mode == "aggressive" and bullish_signals >= 1:
                action = TradingAction.BUY
                confidence = 0.7 * confidence_multiplier
                reasoning = f"BUY: Aggressive mode - bullish momentum. Signals: {bullish_signals}"

        # Minimum-confidence gate. (Bug fix: the original formatted the message
        # after overwriting `confidence`, so it always printed 0.70.)
        if action != TradingAction.HOLD and confidence < self.config.min_confidence:
            rejected_confidence = confidence
            action = TradingAction.HOLD
            confidence = 0.7
            reasoning = (f"HOLD: Trade signal present but confidence {rejected_confidence:.2f} "
                         f"below minimum {self.config.min_confidence}")

        if action == TradingAction.HOLD and signal_details:
            reasoning = (f"HOLD: Net signal {net_signal} (bullish: {bullish_signals}, "
                         f"bearish: {bearish_signals}). Details: {', '.join(signal_details[:3])}")

        return action, confidence, reasoning

    def _build_llm_context(self, timestamp, sig: dict, current_price: float,
                           rule_action: TradingAction, market_analysis: str,
                           pattern_analysis: str, risk_assessment: dict,
                           portfolio: dict,
                           vector_insights: dict, positioning=None,
                           text_sentiment=None) -> str:
        """Assemble every agent's output into one decision prompt."""
        def clip(text, n=700):
            if not text:
                return "not available"
            text = str(text).strip()
            return text if len(text) <= n else text[:n] + "..."

        # Already counted in the rule score, so say so: otherwise the model
        # double-counts it as both the baseline and fresh evidence.
        text_line = "not available"
        if text_sentiment is not None and getattr(text_sentiment, "available", False):
            text_line = (
                f"score {text_sentiment.score:+.3f} (positive = bullish), "
                f"confidence {text_sentiment.confidence:.2f}, "
                f"{text_sentiment.doc_count} documents, "
                f"already worth {sig.get('text_sentiment_points', 0):+d} point(s) in "
                f"the rule score above. {text_sentiment.reasoning}"
            )
        elif text_sentiment is not None:
            text_line = f"not available ({text_sentiment.reasoning})"

        # Positioning is already counted in the rule score above, so the prompt
        # says so explicitly. Otherwise the model double-counts it: once as the
        # baseline it is adjusting, and again as fresh evidence.
        pos_line = "not available"
        if positioning is not None and getattr(positioning, "available", False):
            pos_line = (
                f"score {positioning.score:+.3f} (positive = bullish), "
                f"confidence {positioning.confidence:.2f}, "
                f"already worth {sig.get('positioning_points', 0):+d} point(s) in the "
                f"rule score above. {positioning.reasoning}"
            )
        elif positioning is not None:
            pos_line = f"not available ({positioning.reasoning})"

        vec_line = "not available"
        if vector_insights:
            vec_line = clip(json.dumps(vector_insights, default=str), 300)

        position = "FLAT (no open position)"
        if portfolio.get('holding'):
            entry = portfolio.get('entry_price', 0)
            pnl = ((current_price - entry) / entry * 100) if entry else 0.0
            position = f"LONG from ${entry:.2f}, unrealised P&L {pnl:+.2f}%"

        return f"""Trading decision review for {timestamp}.

CURRENT MARKET STATE
- Price: ${current_price:.2f}
- RSI: {sig['rsi']:.1f}
- MA20: ${sig['ma20']:.2f} | MA50: ${sig['ma50']:.2f}
- MACD histogram: {sig['macd_hist']:.4f}
- 5-bar momentum: {sig['momentum']*100:+.2f}%

RULE ENGINE OUTPUT (the baseline you are adjusting)
- Bullish points: {sig['bullish']} | Bearish points: {sig['bearish']}
- Net signal score: {sig['net_signal']}
- Signal threshold for action: {self.config.signal_threshold}
- Rule engine would: {rule_action.value}
- Triggered rules: {', '.join(sig['details']) if sig['details'] else 'none'}

PORTFOLIO
- {position}
- Cash: ${portfolio.get('cash', 0):.2f}
- Strategy mode: {self.config.trading_mode}

MARKET ANALYST AGENT
{clip(market_analysis)}

PATTERN RECOGNITION AGENT
{clip(pattern_analysis)}

RISK AGENT
- Assessed level: {risk_assessment.get('risk_level', 'unknown')}
- Notes: {clip(risk_assessment.get('reasoning'), 400)}

FUTURES POSITIONING AGENT (Binance USD-M perpetuals, point-in-time)
- {pos_line}

TEXT SENTIMENT AGENT (Hacker News posts and comments, point-in-time)
- {text_line}

SIMILAR HISTORICAL PATTERNS: {vec_line}

YOUR TASK
Return a bounded adjustment to the net signal score of {sig['net_signal']}.
- signal_adjustment must be an integer in [-{LLM_MAX_ADJUSTMENT}, +{LLM_MAX_ADJUSTMENT}].
- Use 0 when the agent inputs add nothing beyond what the rules already capture.
- Set veto=true ONLY to block a trade on clear risk grounds.
- Every entry in key_factors must cite a specific input above, not generic advice.
Be conservative: the rules are a reasonable baseline, so only move the score when
the qualitative agent inputs genuinely justify it."""

    def _query_llm(self, prompt: str) -> tuple:
        """Return (LLMTradeAdjustment or None, error string, latency seconds)."""
        started = time.time()
        try:
            system = SystemMessage(content=(
                "You are the final arbiter in a multi-agent crypto trading system. "
                "You receive a rule-based signal score and qualitative analysis from "
                "specialist agents, and you return a small, bounded correction to that "
                "score. You are not a cheerleader: returning 0 is the correct answer "
                "whenever the qualitative inputs do not add information."
            ))
            result = self.structured_llm.invoke([system, HumanMessage(content=prompt)])
            return result, "", time.time() - started
        except Exception as e:
            return None, f"{type(e).__name__}: {e}", time.time() - started

    def make_decision(self, timestamp, market_analysis: str, pattern_analysis: str,
                      risk_assessment: dict, portfolio: dict, last_decision=None,
                      vector_insights: dict = None, positioning=None,
                      text_sentiment=None) -> TradingDecision:
        """Integrate all analyses and make a trading decision."""
        current_price = 0.0
        try:
            timestamp_dt = pd.to_datetime(timestamp)
            idx = resolve_bar_index(self.df.index, timestamp_dt)
            current_data = self.df.iloc[idx]
            current_price = current_data['close']

            sig = self._compute_rule_signals(
                idx, current_data, current_price, positioning=positioning,
                text_sentiment=text_sentiment,
            )

            # Risk level now genuinely comes from the risk agent's parsed output
            risk_level = risk_assessment.get('risk_level', 'medium')
            confidence_multiplier = {"low": 1.2, "high": 0.8}.get(risk_level, 1.0)

            # 1. Baseline: what the rules alone would do
            rule_action, rule_conf, rule_reason = self._signal_to_action(
                sig["net_signal"], sig, portfolio, current_price, confidence_multiplier
            )

            contrib = LLMContribution(
                rule_action=rule_action.value,
                final_action=rule_action.value,
                rule_signal=sig["net_signal"],
                key_factors=[],
            )
            self.stats["decisions"] += 1

            # 2. Rules-only ablation arm stops here
            if not self.use_llm or self.structured_llm is None:
                return TradingDecision(
                    action=rule_action, confidence=rule_conf,
                    reasoning=rule_reason, price=current_price, timestamp=timestamp,
                    rule_action=rule_action.value, llm=contrib.to_dict(),
                )

            # 3. Ask the LLM for a bounded adjustment
            prompt = self._build_llm_context(
                timestamp, sig, current_price, rule_action, market_analysis,
                pattern_analysis, risk_assessment, portfolio,
                vector_insights, positioning,
                text_sentiment,
            )
            adjustment, error, latency = self._query_llm(prompt)
            contrib.invoked = True
            contrib.latency_s = latency
            self.stats["llm_calls"] += 1
            self.stats["total_latency_s"] += latency

            if adjustment is None:
                # Fail safe: the rule decision stands, and the failure is recorded
                # rather than hidden, so runs with degraded LLM coverage are visible.
                contrib.error = error
                self.stats["llm_failures"] += 1
                logger.warning(f"LLM decision call failed at {timestamp}: {error}")
                return TradingDecision(
                    action=rule_action, confidence=rule_conf,
                    reasoning=f"{rule_reason} [LLM unavailable: {error}]",
                    price=current_price, timestamp=timestamp,
                    rule_action=rule_action.value, llm=contrib.to_dict(),
                )

            contrib.succeeded = True
            raw_adj = int(adjustment.signal_adjustment or 0)
            clamped = max(-LLM_MAX_ADJUSTMENT, min(LLM_MAX_ADJUSTMENT, raw_adj))
            contrib.adjustment = clamped
            contrib.stance = str(adjustment.stance)
            contrib.veto = bool(adjustment.veto)
            contrib.confidence = float(adjustment.confidence or 0.0)
            contrib.key_factors = list(adjustment.key_factors or [])
            contrib.rationale = str(adjustment.rationale or "")

            # 4. Recompute the action from the adjusted score
            adjusted_signal = sig["net_signal"] + clamped
            final_action, final_conf, final_reason = self._signal_to_action(
                adjusted_signal, sig, portfolio, current_price, confidence_multiplier
            )

            # 5. Veto can only block a trade, never create one
            if adjustment.veto and final_action != TradingAction.HOLD:
                final_action = TradingAction.HOLD
                final_conf = 0.7
                final_reason = f"HOLD: LLM risk veto. {contrib.rationale}"
                self.stats["vetoes"] += 1

            # Blend confidence: rule confidence weighted with the model's own
            final_conf = min(0.95, 0.7 * final_conf + 0.3 * contrib.confidence)

            contrib.final_action = final_action.value
            if contrib.changed_decision:
                self.stats["changed"] += 1

            reasoning = (
                f"{final_reason} | LLM {contrib.stance} adj {clamped:+d} "
                f"(score {sig['net_signal']} -> {adjusted_signal}): {contrib.rationale}"
            )

            return TradingDecision(
                action=final_action, confidence=final_conf, reasoning=reasoning,
                price=current_price, timestamp=timestamp,
                rule_action=rule_action.value, llm=contrib.to_dict(),
            )

        except Exception as e:
            logger.error(f"Decision agent error: {e}")
            return TradingDecision(
                action=TradingAction.HOLD,
                confidence=0.9,
                reasoning=f"Error occurred: {str(e)}. Defaulting to HOLD for safety.",
                price=current_price,
                timestamp=timestamp,
            )

# ── VECTOR DATABASE AGENT ──────────────────────────────────────────────────
class VectorDBAgent:
    """Vector database agent for storing and retrieving trading patterns"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the vector database"""
        try:
            if USE_AZURE_OPENAI:
                # Requires an embedding deployment in the same Foundry resource.
                self.embeddings = AzureOpenAIEmbeddings(
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_key=AZURE_OPENAI_API_KEY,
                    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                    api_version=AZURE_OPENAI_API_VERSION,
                )
            else:
                openai_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not openai_key:
                    return False

                # Use a more accessible embedding model
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=openai_key,
                    model="text-embedding-3-small"  # More accessible model
                )
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing vector DB: {e}")
            logger.warning("Vector DB disabled due to initialization error")
            return False
    
    def store_trading_pattern(self, market_data: Dict[str, Any], decision: TradingDecision, 
                            performance: float) -> bool:
        """Store a trading pattern in the vector database"""
        try:
            if not self.initialized:
                return False
            
            # Create document content
            content = f"""
            Market Conditions:
            Price: {market_data.get('price', 0)}
            RSI: {market_data.get('RSI', 0)}
            MA20: {market_data.get('MA20', 0)}
            MACD: {market_data.get('MACD_hist', 0)}
            
            Decision: {decision.action.value}
            Confidence: {decision.confidence}
            Reasoning: {decision.reasoning}
            
            Performance: {performance}%
            """
            
            metadata = {
                "action": decision.action.value,
                "confidence": decision.confidence,
                "performance": performance,
                "timestamp": decision.timestamp.isoformat(),
                "price": market_data.get('price', 0),
                "rsi": market_data.get('RSI', 0)
            }
            
            document = Document(page_content=content, metadata=metadata)
            
            # Store in vector database
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents([document], self.embeddings)
            else:
                self.vectorstore.add_documents([document])
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            return False
    
    def retrieve_similar_patterns(self, current_conditions: Dict[str, Any], 
                                 k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar trading patterns from the vector database"""
        try:
            if not self.initialized or self.vectorstore is None:
                return []
            
            # Create query from current conditions
            query = f"""
            Price: {current_conditions.get('price', 0)}
            RSI: {current_conditions.get('RSI', 0)}
            MA20: {current_conditions.get('MA20', 0)}
            MACD: {current_conditions.get('MACD_hist', 0)}
            """
            
            # Search for similar patterns
            similar_docs = self.vectorstore.similarity_search_with_score(query, k=k)
            
            patterns = []
            for doc, score in similar_docs:
                patterns.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": 1 - score  # Convert distance to similarity
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error retrieving patterns: {e}")
            return []
    
    def get_performance_insights(self, current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance insights based on historical patterns"""
        try:
            patterns = self.retrieve_similar_patterns(current_conditions)
            
            if not patterns:
                return {"recommendation": "HOLD", "confidence": 0.0, "reasoning": "No similar patterns found"}
            
            # Analyze patterns
            buy_performance = []
            sell_performance = []
            hold_performance = []
            
            for pattern in patterns:
                metadata = pattern["metadata"]
                action = metadata.get("action", "HOLD")
                performance = metadata.get("performance", 0)
                
                if action == "BUY":
                    buy_performance.append(performance)
                elif action == "SELL":
                    sell_performance.append(performance)
                else:
                    hold_performance.append(performance)
            
            # Calculate average performance for each action
            avg_buy = np.mean(buy_performance) if buy_performance else 0
            avg_sell = np.mean(sell_performance) if sell_performance else 0
            avg_hold = np.mean(hold_performance) if hold_performance else 0
            
            # Recommend best action
            performances = {"BUY": avg_buy, "SELL": avg_sell, "HOLD": avg_hold}
            best_action = max(performances, key=performances.get)
            
            return {
                "recommendation": best_action,
                "confidence": min(len(patterns) / 10.0, 1.0),  # Confidence based on pattern count
                "reasoning": f"Based on {len(patterns)} similar patterns, {best_action} had average performance of {performances[best_action]:.2f}%",
                "pattern_count": len(patterns),
                "performance_breakdown": performances
            }
            
        except Exception as e:
            logger.error(f"Error getting performance insights: {e}")
            return {"recommendation": "HOLD", "confidence": 0.0, "reasoning": "Error analyzing patterns"}

# ── SENTIMENT ANALYSIS AGENT ──────────────────────────────────────────────
    

# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def show_performance_summary(st_module, summary):
    """Display simulation performance summary"""
    st_module.success("✅ Simulation Complete!")
    
    col1, col2, col3 = st_module.columns(3)
    
    with col1:
        st_module.metric(
            "Total Return",
            f"{summary['total_return']:.2f}%",
            delta=f"${summary['final_value'] - summary['initial_capital']:.2f}"
        )
    
    with col2:
        st_module.metric(
            "Final Portfolio Value",
            f"${summary['final_value']:.2f}",
            delta_color="off"
        )
    
    with col3:
        st_module.metric(
            "Win Rate",
            f"{summary['win_rate']:.1f}%",
            delta_color="off"
        )
    
    # Additional metrics
    st_module.markdown("### 📊 Detailed Performance")
    
    performance_col1, performance_col2 = st_module.columns(2)
    
    with performance_col1:
        st_module.write(f"**Initial Capital:** ${summary['initial_capital']:.2f}")
        st_module.write(f"**Total Trades:** {summary['total_trades']}")
        st_module.write(f"**Total Decisions:** {summary['decisions']}")
    
    with performance_col2:
        st_module.write(f"**Sentiment Influence:** {summary['sentiment_influence']}")
        
        # Performance color coding
        if summary['total_return'] > 0:
            st_module.markdown("🟢 **Profitable Strategy**")
        else:
            st_module.markdown("🔴 **Loss-Making Strategy**")

# ── DATABASE FUNCTIONS ─────────────────────────────────────────────────────
def initialize_session_state():
    """Initialize session state for storing past results"""
    if 'past_simulation_results' not in st.session_state:
        st.session_state.past_simulation_results = []
    if 'show_past_results' not in st.session_state:
        st.session_state.show_past_results = False
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False

def save_simulation_result(summary: dict, symbol: str, strategy_mode: str, start_date, end_date):
    """Save simulation result to session state"""
    try:
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'strategy_mode': strategy_mode,
            'start_date': start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
            'end_date': end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
            'initial_capital': summary.get('initial_capital', 0),
            'final_value': summary.get('final_value', 0),
            'total_return_pct': summary.get('total_return_pct', 0),
            'buy_hold_return_pct': summary.get('buy_hold_return_pct', 0),
            'total_trades': summary.get('total_trades', 0),
            'winning_trades': summary.get('winning_trades', 0),
            'win_rate_pct': summary.get('win_rate_pct', 0),
            'outperformed_market': summary.get('outperformed_market', False),
            'max_drawdown': summary.get('max_drawdown', 0),
            'sharpe_ratio': summary.get('sharpe_ratio', 0),
            'profit_factor': summary.get('profit_factor', 0),
            'period_days': (end_date - start_date).days if hasattr(start_date, 'strftime') and hasattr(end_date, 'strftime') else 0
        }
        
        # Add to session state (keep last 20 results)
        if 'past_simulation_results' not in st.session_state:
            st.session_state.past_simulation_results = []
        
        st.session_state.past_simulation_results.append(result)
        
        # Keep only last 20 results to avoid memory issues
        if len(st.session_state.past_simulation_results) > 20:
            st.session_state.past_simulation_results = st.session_state.past_simulation_results[-20:]
            
        logger.info(f"Saved simulation result: {result['total_return_pct']:.2f}% return")
        
    except Exception as e:
        logger.error(f"Error saving simulation result: {e}")

def fetch_past_results():
    """Fetch past simulation results from session state"""
    try:
        initialize_session_state()
        return st.session_state.past_simulation_results
        
    except Exception as e:
        logger.error(f"Error fetching past results: {e}")
        return []

def display_past_results():
    """Display past simulation results with proper expand/collapse functionality"""
    past_results = fetch_past_results()
    
    if not past_results:
        st.info("📭 No past simulation results found. Run a simulation first to see results here.")
        st.markdown("""
        **How to generate results:**
        1. 🔧 Configure your trading parameters above
        2. 📅 Select a date range (e.g., last 30 days)
        3. ▶️ Click **"Run Simulation"** button
        4. ✅ Wait for simulation to complete
        5. 📊 Return here to view your results!
        
        **Note:** Results are stored temporarily in your browser session.
        """)
        
        # Debug information for troubleshooting
        with st.expander("🔧 Debug Information", expanded=False):
            st.write(f"Session state keys: {list(st.session_state.keys())}")
            st.write(f"Past results list exists: {'past_simulation_results' in st.session_state}")
            if 'past_simulation_results' in st.session_state:
                st.write(f"Past results count: {len(st.session_state.past_simulation_results)}")
            st.write(f"Show past results flag: {st.session_state.get('show_past_results', False)}")
        return
    
    st.markdown("### 📊 Past Simulation Results")
    st.markdown(f"*Showing {len(past_results)} most recent simulation runs*")
    
    # Summary statistics of all past results
    if len(past_results) > 1:
        avg_return = sum(r['total_return_pct'] for r in past_results) / len(past_results)
        best_return = max(r['total_return_pct'] for r in past_results)
        worst_return = min(r['total_return_pct'] for r in past_results)
        win_rate = sum(1 for r in past_results if r['total_return_pct'] > 0) / len(past_results) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Avg Return", f"{avg_return:.2f}%", f"{len(past_results)} simulations")
        with col2:
            st.metric("🚀 Best Return", f"{best_return:.2f}%")
        with col3:
            st.metric("📉 Worst Return", f"{worst_return:.2f}%")
        with col4:
            st.metric("🎯 Success Rate", f"{win_rate:.1f}%", f"{sum(1 for r in past_results if r['total_return_pct'] > 0)} wins")
    
    # Display each result with expand/collapse functionality
    for i, result in enumerate(reversed(past_results)):  # Show newest first
        result_index = len(past_results) - i
        
        # Color-code the summary based on performance
        if result['total_return_pct'] > 0:
            performance_color = "🟢"
            performance_text = "PROFIT"
        else:
            performance_color = "🔴"
            performance_text = "LOSS"
        
        # Market comparison
        vs_market = result['total_return_pct'] - result['buy_hold_return_pct']
        market_status = "📈 BEAT MARKET" if vs_market > 0 else "📉 TRAIL MARKET"
        
        # Create expander with summary info
        expander_title = f"{performance_color} **Run #{result_index}** - {result['symbol']} ({result['strategy_mode'].title()}) | {performance_text}: {result['total_return_pct']:+.2f}% | {market_status}: {vs_market:+.2f}%"
        
        with st.expander(expander_title, expanded=False):
            # Detailed results inside the expander
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            with detail_col1:
                st.markdown("**📈 Performance Metrics**")
                st.write(f"• **Date Range:** {result['start_date']} to {result['end_date']}")
                st.write(f"• **Period:** {result['period_days']} days")
                st.write(f"• **Initial Capital:** ${result['initial_capital']:,.2f}")
                st.write(f"• **Final Value:** ${result['final_value']:,.2f}")
                st.write(f"• **Strategy Return:** {result['total_return_pct']:+.2f}%")
                st.write(f"• **Buy & Hold Return:** {result['buy_hold_return_pct']:+.2f}%")
                st.write(f"• **Alpha (vs Market):** {vs_market:+.2f}%")
            
            with detail_col2:
                st.markdown("**🎯 Trading Statistics**")
                st.write(f"• **Total Trades:** {result['total_trades']}")
                st.write(f"• **Winning Trades:** {result['winning_trades']}")
                st.write(f"• **Win Rate:** {result['win_rate_pct']:.1f}%")
                
                if result['total_trades'] > 0:
                    profit_per_trade = (result['final_value'] - result['initial_capital']) / result['total_trades']
                    st.write(f"• **Avg Profit/Trade:** ${profit_per_trade:,.2f}")
                    
                    # Trading frequency
                    trades_per_day = result['total_trades'] / max(1, result['period_days'])
                    st.write(f"• **Trading Frequency:** {trades_per_day:.2f} trades/day")
                else:
                    st.write("• **Avg Profit/Trade:** N/A")
                    st.write("• **Trading Frequency:** No trades")
                
                if result.get('max_drawdown', 0) != 0:
                    st.write(f"• **Max Drawdown:** {result['max_drawdown']:.2f}%")
                if result.get('sharpe_ratio', 0) != 0:
                    st.write(f"• **Sharpe Ratio:** {result['sharpe_ratio']:.2f}")
            
            with detail_col3:
                st.markdown("**⚙️ Configuration**")
                st.write(f"• **Symbol:** {result['symbol']}")
                st.write(f"• **Strategy:** {result['strategy_mode'].title()}")
                st.write(f"• **Timestamp:** {result['timestamp']}")
                
                # Performance assessment
                st.markdown("**📊 Assessment**")
                if result['outperformed_market']:
                    st.success("✅ Strategy outperformed buy & hold")
                else:
                    st.warning("⚠️ Strategy underperformed buy & hold")
                
                if result['win_rate_pct'] > 60:
                    st.success(f"✅ High win rate: {result['win_rate_pct']:.1f}%")
                elif result['win_rate_pct'] > 40:
                    st.info(f"ℹ️ Moderate win rate: {result['win_rate_pct']:.1f}%")
                else:
                    st.warning(f"⚠️ Low win rate: {result['win_rate_pct']:.1f}%")
                
                # ROI assessment
                annualized_return = (result['total_return_pct'] / max(1, result['period_days'])) * 365
                if annualized_return > 15:
                    st.success(f"🚀 Strong ROI: {annualized_return:.1f}% annualized")
                elif annualized_return > 5:
                    st.info(f"📈 Good ROI: {annualized_return:.1f}% annualized")
                else:
                    st.warning(f"📉 Weak ROI: {annualized_return:.1f}% annualized")
    
    # Add clear results button
    if st.button("🗑️ Clear All Past Results", type="secondary"):
        st.session_state.past_simulation_results = []
        st.success("✅ All past results cleared!")
        st.rerun()

# ── MAIN SIMULATION FUNCTION ──────────────────────────────────────────────
def run_trading_simulation(symbol_input, interval, start_date, end_date, initial_capital, 
                          enable_vector_db, show_reasoning, 
                          strategy_mode, custom_position_size, custom_confidence, custom_signal_threshold):
    """Main function to run the multi-agent trading simulation"""
    
    # Set simulation running state
    st.session_state.simulation_running = True
    
    # Create configuration based on selected strategy
    if strategy_mode == "conservative":
        config = TradingConfig.get_conservative_config(initial_capital)
    elif strategy_mode == "moderate":
        config = TradingConfig.get_moderate_config(initial_capital)
    else:  # aggressive
        config = TradingConfig.get_aggressive_config(initial_capital)
    
    # Override with custom settings if provided.
    # The None guards make this honour its own comment. Previously the three
    # assignments were unconditional, so any programmatic caller that passed
    # None (an ablation runner, a headless script) silently set
    # signal_threshold to None and every decision then raised
    # "'>=' not supported between instances of 'int' and 'NoneType'".
    # In the UI this was harmless because the slider defaults mirror the
    # presets exactly, but it made the function unusable from anywhere else,
    # and it would have masked any future drift between sliders and presets.
    if custom_position_size is not None:
        config.position_size_pct = custom_position_size
    if custom_confidence is not None:
        config.min_confidence = custom_confidence
    if custom_signal_threshold is not None:
        config.signal_threshold = custom_signal_threshold
    config.enable_vector_db = enable_vector_db
    config.show_reasoning = show_reasoning

    # Decision cadence is a DURATION, not a bar count. The preset's `6` means
    # "every 6 hours", which is only true on hourly bars: on daily bars it meant
    # every 6 days, so a 30-day backtest made two decisions and one round trip.
    # Convert once, here, so the whole run shares one resolved value.
    config.simulation_step = decision_step_bars(interval)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch market data
        status_text.text("📊 Fetching market data...")
        progress_bar.progress(10)
        
        df = fetch_binance_ta(symbol_input, interval, start_date, end_date)

        # PHASE 2: data provenance is surfaced, never assumed.
        data_source = df.attrs.get('data_source', 'unknown')
        if data_source == 'binance_live':
            st.success(f"✅ Fetched {len(df)} data points from Binance (live market data)")
        else:
            st.error(
                f"⚠️ **{len(df)} data points are SYNTHETIC, not real market data.** "
                "Any result below is a code test only and must NOT be reported as a finding."
            )

        # The first INDICATOR_WARMUP_BARS rows are consumed by MA20 and friends,
        # so a run needs strictly more bars than that to have anything to trade.
        # Without this check the shortfall surfaced far downstream as a bare
        # "single positional indexer is out-of-bounds" from df.iloc[start_idx],
        # which says nothing about the actual problem: the window is too short
        # for the interval. Fail here instead, while the numbers are still in
        # hand to explain it.
        if len(df) <= INDICATOR_WARMUP_BARS:
            raise InsufficientHistory(
                f"{len(df)} {interval} bars were returned for "
                f"{start_date} to {end_date}, but the technical indicators "
                f"consume the first {INDICATOR_WARMUP_BARS} and at least one "
                f"bar must remain to trade. Widen the date range, or pick a "
                f"finer interval so the same range yields more bars."
            )

        # Step 2: Initialize agents
        status_text.text("🤖 Initializing AI agents...")
        progress_bar.progress(20)
        
        market_agent = MarketAnalystAgent(df)
        pattern_agent = PatternRecognitionAgent(df)
        risk_agent = RiskManagementAgent(df, config)
        decision_agent = TradingDecisionAgent(df, config)
        
        # Optional agents
        vector_agent = None

        if config.enable_vector_db and VECTOR_DB_AVAILABLE:
            vector_agent = VectorDBAgent(config)
            vector_agent.initialize()
        
        # PHASE 3: exogenous positioning agent. All alignment happens once here
        # against the price index, so the per-bar lookup in the loop below is a
        # dictionary hit and cannot accidentally re-derive features from data
        # the bar should not see.
        # The z-score baseline is interval-dependent: 168 bars is a week of 1h
        # bars but 168 days of 1d bars, which no realistic cache can warm up.
        # Resolved once here so the auto-fetch range, the agent and the run
        # metadata all quote the same number.
        pos_zscore_window = positioning_zscore_window(interval)

        positioning_agent = None
        if USE_POSITIONING_SIGNAL and POSITIONING_AVAILABLE:
            # Download whatever the chosen date range needs and the cache does
            # not have. First run over a new window pays for it once; every
            # later run over the same window is a pure cache hit.
            if POSITIONING_AUTO_FETCH:
                try:
                    # Fetch the z-score warm-up window too, or the first
                    # POSITIONING_ZSCORE_WINDOW bars of the backtest silently
                    # run with no positioning signal. Same helper the agent
                    # uses, so the two ranges cannot drift apart.
                    pos_start = warmup_start_date(df.index, pos_zscore_window)
                    pos_end = df.index.max().date()
                    pending = missing_days(symbol_input, pos_start, pos_end)
                    if pending:
                        fetch_bar = st.progress(0)
                        fetch_text = st.empty()
                        fetch_text.text(
                            f"📥 Downloading {len(pending)} day(s) of positioning "
                            f"data (free, no API key)..."
                        )

                        def _report(done, total, day):
                            fetch_bar.progress(min(1.0, done / max(total, 1)))
                            fetch_text.text(
                                f"📥 Positioning data {done}/{total} ({day})"
                            )

                        fetch_summary = ensure_cached(
                            symbol_input, pos_start, pos_end, progress=_report,
                        )
                        fetch_bar.empty()
                        fetch_text.empty()
                        logger.info(
                            f"Positioning auto-fetch: {json.dumps(fetch_summary, default=str)}"
                        )
                        if fetch_summary.get("failed"):
                            st.warning(
                                f"⚠️ {len(fetch_summary['failed'])} positioning day(s) "
                                f"could not be downloaded; the signal will run on "
                                f"the days that succeeded."
                            )
                except Exception as e:
                    # A data-download problem must not take down the backtest.
                    logger.warning(f"Positioning auto-fetch failed: {e}")
                    st.warning(f"⚠️ Positioning auto-fetch failed: {e}")

            positioning_agent = PositioningSignalAgent(
                symbol=symbol_input,
                bar_index=df.index,
                enabled=True,
                lag_bars=POSITIONING_LAG_BARS,
                max_points=POSITIONING_MAX_POINTS,
                zscore_window=pos_zscore_window,
            )
            if positioning_agent.available:
                st.success(f"✅ Positioning signal: {positioning_agent.status}")
            else:
                # Loud, not silent. A run with a dead exogenous channel is a
                # different experiment from one with a live channel, and the
                # difference must be visible in the UI, not just the log.
                st.warning(
                    f"⚠️ Positioning signal enabled but unusable: "
                    f"{positioning_agent.status}"
                )
        elif USE_POSITIONING_SIGNAL and not POSITIONING_AVAILABLE:
            st.warning("⚠️ USE_POSITIONING_SIGNAL is on but the signals module "
                       "failed to import; running without it.")

        # PHASE 3: exogenous text sentiment. Documents are fetched, scored and
        # aligned once here rather than per bar, because scoring is the
        # expensive step and a 24h trailing window would otherwise re-score the
        # same documents once for every bar that window covers.
        text_agent = None
        if USE_TEXT_SENTIMENT and TEXT_SENTIMENT_AVAILABLE:
            if TEXT_SENTIMENT_AUTO_FETCH:
                try:
                    txt_start = text_warmup_start_date(
                        df.index, TEXT_SENTIMENT_ZSCORE_WINDOW,
                        TEXT_SENTIMENT_WINDOW_HOURS)
                    txt_end = df.index.max().date()
                    pending = text_missing_days(
                        txt_start, txt_end, queries=TEXT_SENTIMENT_QUERIES)
                    if pending:
                        txt_bar = st.progress(0)
                        txt_text = st.empty()
                        txt_text.text(
                            f"📥 Downloading {len(pending)} day(s) of text sentiment "
                            f"documents (free, no API key)..."
                        )

                        def _txt_report(done, total, label):
                            txt_bar.progress(min(1.0, done / max(total, 1)))
                            txt_text.text(f"📥 Text sentiment {done}/{total} ({label})")

                        txt_summary = ensure_text_cached(
                            txt_start, txt_end, queries=TEXT_SENTIMENT_QUERIES,
                            progress=_txt_report,
                        )
                        txt_bar.empty()
                        txt_text.empty()
                        logger.info(
                            f"Text sentiment auto-fetch: "
                            f"{json.dumps(txt_summary, default=str)}"
                        )
                        if txt_summary.get("failed"):
                            st.warning(
                                f"⚠️ {len(txt_summary['failed'])} text sentiment "
                                f"request(s) failed; the signal will run on what "
                                f"succeeded."
                            )
                except Exception as e:
                    logger.warning(f"Text sentiment auto-fetch failed: {e}")
                    st.warning(f"⚠️ Text sentiment auto-fetch failed: {e}")

            with st.spinner("🧠 Scoring text sentiment corpus..."):
                text_agent = TextSentimentAgent(
                    bar_index=df.index,
                    queries=TEXT_SENTIMENT_QUERIES,
                    enabled=True,
                    scorer_name=TEXT_SENTIMENT_SCORER,
                    lag_bars=TEXT_SENTIMENT_LAG_BARS,
                    window_hours=TEXT_SENTIMENT_WINDOW_HOURS,
                    zscore_window=TEXT_SENTIMENT_ZSCORE_WINDOW,
                    max_points=TEXT_SENTIMENT_MAX_POINTS,
                    min_documents=TEXT_SENTIMENT_MIN_DOCS,
                )
            if text_agent.available:
                st.success(f"✅ Text sentiment: {text_agent.status}")
            else:
                st.warning(f"⚠️ Text sentiment enabled but unusable: "
                           f"{text_agent.status}")
        elif USE_TEXT_SENTIMENT and not TEXT_SENTIMENT_AVAILABLE:
            st.warning("⚠️ USE_TEXT_SENTIMENT is on but the text sentiment module "
                       "failed to import; running without it.")

        st.success("✅ All agents initialized")
        
        # Step 3: Run simulation
        status_text.text("🔄 Running trading simulation...")
        progress_bar.progress(30)
        
        # Initialize portfolio
        portfolio = {
            'cash': config.initial_capital,
            'holdings': 0.0,
            'holding': False,
            'entry_price': 0.0
        }
        
        trades = []
        decisions_log = []
        portfolio_values = []
        daily_values = []
        
        # Skip initial rows for technical indicators to stabilize
        start_idx = INDICATOR_WARMUP_BARS
        simulation_points = list(range(start_idx, len(df), config.simulation_step))

        # A run's headline numbers rest on the number of DECISIONS, not the
        # number of bars or days. A 30-day daily window looks substantial and
        # yields eleven decisions; at the old six-day cadence it yielded two,
        # and "50% win rate" then meant one winning trade out of two. Say so
        # here rather than letting the executive summary imply more than the
        # sample can support.
        st.info(
            f"🧮 {len(df)} bars · {INDICATOR_WARMUP_BARS} consumed by indicator "
            f"warmup · deciding every {config.simulation_step} bar(s) "
            f"(~{DECISION_CADENCE_HOURS:g}h) → **{len(simulation_points)} decision "
            f"points**."
        )
        if len(simulation_points) < 30:
            st.warning(
                f"⚠️ Only {len(simulation_points)} decision points. Win rate, "
                f"average trade and 'beat the market' are dominated by noise at "
                f"this sample size and should not be read as evidence. Widen the "
                f"date range, or use a finer interval, before drawing conclusions."
            )

        # Track sentiment for the entire period

        for i, current_idx in enumerate(simulation_points):
            try:
                current_data = df.iloc[current_idx]
                current_price = current_data['close']
                timestamp = df.index[current_idx]
                
                # Update progress
                progress = 30 + int((i / len(simulation_points)) * 60)
                progress_bar.progress(progress)
                status_text.text(f"🔄 Processing {timestamp.strftime('%Y-%m-%d %H:%M')} ({i+1}/{len(simulation_points)})")
                
                # Step 3a: Market Analysis
                market_analysis = market_agent.analyze(timestamp)
                
                # Step 3b: Pattern Recognition
                pattern_analysis = pattern_agent.identify_patterns(timestamp)
                
                # Step 3c: Risk Assessment
                last_decision = decisions_log[-1] if decisions_log else None
                risk_assessment = risk_agent.assess_risk(timestamp, portfolio=portfolio, last_decision=last_decision)
                
                # Step 3d: Exogenous futures positioning (point-in-time)
                positioning_reading = None
                if positioning_agent is not None:
                    positioning_reading = positioning_agent.reading_for(timestamp)

                # Step 3e: Exogenous text sentiment (point-in-time)
                text_reading = None
                if text_agent is not None:
                    text_reading = text_agent.reading_for(timestamp)

                # Step 3e: Vector DB Insights
                vector_insights = None
                if vector_agent and vector_agent.initialized:
                    current_conditions = {
                        'price': current_price,
                        'RSI': current_data['RSI'],
                        'MA20': current_data['MA20'],
                        'MACD_hist': current_data['MACD_hist']
                    }
                    vector_insights = vector_agent.get_performance_insights(current_conditions)
                
                # Step 3g: Final Decision
                # PHASE 1: every agent's output is now passed to the decision
                # maker. Sentiment, deep learning and retrieval used to be
                # computed here and then dropped into the log without ever
                # reaching make_decision.
                decision = decision_agent.make_decision(
                    timestamp, market_analysis, pattern_analysis,
                    risk_assessment, portfolio, last_decision,
                    vector_insights=vector_insights,
                    positioning=positioning_reading,
                    text_sentiment=text_reading,
                )

                # Step 3h: Execute Trade
                # PHASE 2: fill at the NEXT bar's open, since the decision was
                # made from this bar's close. The final bar has no successor, so
                # no order placed on it can be filled.
                fill_price, fill_timestamp = None, None
                if EXECUTION_MODE == "next_open" and current_idx + 1 < len(df):
                    next_bar = df.iloc[current_idx + 1]
                    fill_price = next_bar['open']
                    fill_timestamp = df.index[current_idx + 1]
                elif EXECUTION_MODE == "next_open":
                    fill_price = None  # no next bar: order cannot be filled
                    trade_result = None

                if EXECUTION_MODE != "next_open" or current_idx + 1 < len(df):
                    trade_result = execute_trade(
                        decision, portfolio, current_price, config, timestamp,
                        fill_price=fill_price, fill_timestamp=fill_timestamp,
                    )
                else:
                    trade_result = None

                if trade_result:
                    trades.append(trade_result)

                # Log decision
                decisions_log.append({
                    'timestamp': timestamp,
                    'decision': decision,
                    'market_analysis': market_analysis[:200] + "..." if len(market_analysis) > 200 else market_analysis,
                    'pattern_analysis': pattern_analysis[:200] + "..." if len(pattern_analysis) > 200 else pattern_analysis,
                    'risk_assessment': risk_assessment,
                    'positioning': (positioning_reading.to_dict()
                                    if positioning_reading is not None else None),
                    'text_sentiment': (text_reading.to_dict()
                                       if text_reading is not None else None),
                    'vector_insights': vector_insights,
                    # Phase 1 telemetry: what the LLM changed on this decision
                    'llm_contribution': decision.llm,
                    'rule_action': decision.rule_action,
                })
                
                # Calculate portfolio value
                current_portfolio_value = portfolio['cash']
                if portfolio['holding']:
                    current_portfolio_value += portfolio['holdings'] * current_price
                
                portfolio_values.append(current_portfolio_value)
                daily_values.append({
                    'timestamp': timestamp,
                    'portfolio_value': current_portfolio_value,
                    'price': current_price
                })
                
                # Store pattern in vector DB if available
                if vector_agent and vector_agent.initialized and trade_result:
                    performance = (trade_result.get('profit', 0) / config.initial_capital) * 100
                    market_data = {
                        'price': current_price,
                        'RSI': current_data['RSI'],
                        'MA20': current_data['MA20'],
                        'MACD_hist': current_data['MACD_hist']
                    }
                    vector_agent.store_trading_pattern(market_data, decision, performance)
                
                # Brief pause for UI updates
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error at step {i}: {e}")
                continue
        
        # Step 4: Calculate final results
        status_text.text("📊 Calculating results...")
        progress_bar.progress(90)
        
        final_portfolio_value = portfolio['cash']
        if portfolio['holding']:
            final_portfolio_value += portfolio['holdings'] * df.iloc[-1]['close']
        
        total_return = ((final_portfolio_value - config.initial_capital) / config.initial_capital) * 100
        
        # Calculate buy and hold return for comparison
        initial_price = df.iloc[start_idx]['close']
        final_price = df.iloc[-1]['close']
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        
        # Calculate win rate
        profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        win_rate = (profitable_trades / len(trades) * 100) if trades else 0
        
        # PHASE 1 MEASUREMENT: how much did the LLM actually contribute?
        # This is the number that answers the reviewer's central question, so it
        # travels with the result rather than being recomputed by hand.
        agent_stats = decision_agent.stats
        llm_calls = agent_stats["llm_calls"]
        llm_report = {
            'llm_enabled': decision_agent.use_llm,
            'decisions': agent_stats["decisions"],
            'llm_calls': llm_calls,
            'llm_failures': agent_stats["llm_failures"],
            'llm_success_rate_pct': ((llm_calls - agent_stats["llm_failures"]) / llm_calls * 100) if llm_calls else 0.0,
            'decisions_changed_by_llm': agent_stats["changed"],
            'change_rate_pct': (agent_stats["changed"] / agent_stats["decisions"] * 100) if agent_stats["decisions"] else 0.0,
            'risk_vetoes': agent_stats["vetoes"],
            'avg_llm_latency_s': (agent_stats["total_latency_s"] / llm_calls) if llm_calls else 0.0,
            'total_llm_latency_s': agent_stats["total_latency_s"],
        }

        # PHASE 3 MEASUREMENT: what did the exogenous positioning channel do?
        # Reported per run for the same reason as llm_report: the ablation table
        # in the paper needs the marginal contribution of each channel, and a
        # channel that fired on zero bars must be visibly distinguishable from
        # one that fired and simply did not help.
        pos_readings = [d.get('positioning') for d in decisions_log
                        if d.get('positioning')]
        pos_usable = [p for p in pos_readings if p.get('available')]
        pos_acted = [p for p in pos_usable
                     if p.get('bullish_points') or p.get('bearish_points')]
        positioning_report = {
            'enabled': USE_POSITIONING_SIGNAL,
            'module_available': POSITIONING_AVAILABLE,
            'agent': positioning_agent.summary() if positioning_agent else None,
            'decisions_with_reading': len(pos_usable),
            'decisions_total': len(decisions_log),
            'decisions_where_points_added': len(pos_acted),
            'pct_decisions_moved': (
                round(100.0 * len(pos_acted) / len(decisions_log), 2)
                if decisions_log else 0.0
            ),
            'mean_score': (
                round(float(np.mean([p['score'] for p in pos_usable])), 4)
                if pos_usable else None
            ),
            'max_points': POSITIONING_MAX_POINTS,
            'lag_bars': POSITIONING_LAG_BARS,
            # The RESOLVED window, not the raw config value: on a daily run
            # these differ, and the number that describes the run is this one.
            'zscore_window_bars': pos_zscore_window,
        }

        # PHASE 3 MEASUREMENT: what did the text sentiment channel do? Same
        # rationale as positioning_report -- a channel that fired on zero bars
        # must stay visibly distinct from one that fired and did not help.
        txt_readings = [d.get('text_sentiment') for d in decisions_log
                        if d.get('text_sentiment')]
        txt_usable = [t for t in txt_readings if t.get('available')]
        txt_acted = [t for t in txt_usable
                     if t.get('bullish_points') or t.get('bearish_points')]
        text_sentiment_report = {
            'enabled': USE_TEXT_SENTIMENT,
            'module_available': TEXT_SENTIMENT_AVAILABLE,
            'agent': text_agent.summary() if text_agent else None,
            'decisions_with_reading': len(txt_usable),
            'decisions_total': len(decisions_log),
            'decisions_where_points_added': len(txt_acted),
            'pct_decisions_moved': (
                round(100.0 * len(txt_acted) / len(decisions_log), 2)
                if decisions_log else 0.0
            ),
            'mean_score': (
                round(float(np.mean([t['score'] for t in txt_usable])), 4)
                if txt_usable else None
            ),
            'mean_docs_per_reading': (
                round(float(np.mean([t.get('doc_count', 0) for t in txt_usable])), 1)
                if txt_usable else None
            ),
            'scorer': TEXT_SENTIMENT_SCORER,
            'queries': list(TEXT_SENTIMENT_QUERIES),
            'max_points': TEXT_SENTIMENT_MAX_POINTS,
            'lag_bars': TEXT_SENTIMENT_LAG_BARS,
            'window_hours': TEXT_SENTIMENT_WINDOW_HOURS,
        }

        # PHASE 2: every run carries its own provenance so a reported number can
        # always be traced back to its data source, model and execution rules.
        run_metadata = {
            'run_at': datetime.now().isoformat(),
            'symbol': symbol_input,
            'interval': interval,
            'start_date': str(start_date),
            'end_date': str(end_date),
            'data_source': df.attrs.get('data_source', 'unknown'),
            'data_source_label': df.attrs.get('data_source_label', 'unknown'),
            'bars': len(df),
            # How often the system stopped to decide, and how many of the bars
            # it could actually act on. A run's headline numbers rest on the
            # DECISION count, not the bar count, so both belong in the record.
            'decision_cadence_hours': DECISION_CADENCE_HOURS,
            'decision_step_bars': config.simulation_step,
            'warmup_bars': INDICATOR_WARMUP_BARS,
            'decision_points': len(simulation_points),
            # Naming a deployment that made zero calls would misdescribe the
            # run, and probing it costs a network round trip the rules-only arm
            # should not need.
            'model': (describe_active_model() if USE_LLM_DECISIONS else {
                "provider": "none",
                "note": "rules-only arm (USE_LLM_DECISIONS=false): no model "
                        "was called by any agent",
            }),
            'execution_mode': EXECUTION_MODE,
            'slippage_pct': SLIPPAGE_PCT,
            'buy_fee_pct': config.buy_fee_pct,
            'sell_fee_pct': config.sell_fee_pct,
            'strategy_mode': strategy_mode,
            'signal_threshold': config.signal_threshold,
            'min_confidence': config.min_confidence,
            'llm_max_adjustment': LLM_MAX_ADJUSTMENT,
            'positioning_enabled': USE_POSITIONING_SIGNAL,
            'positioning_source': (
                positioning_agent.summary().get('source')
                if positioning_agent and positioning_agent.available else None
            ),
            'text_sentiment_enabled': USE_TEXT_SENTIMENT,
            'text_sentiment_scorer': (
                TEXT_SENTIMENT_SCORER if text_agent and text_agent.available else None
            ),
            'text_sentiment_source': (
                text_agent.summary().get('source')
                if text_agent and text_agent.available else None
            ),
            'synthetic_data_allowed': ALLOW_SYNTHETIC_DATA,
            # Every text channel in the system now reads from a cached, hashed
            # real corpus (see the manifests under data/exogenous/), so the only
            # remaining publication risk is the price series itself.
            'publication_safe': df.attrs.get('data_source') == 'binance_live',
        }

        # Prepare summary
        summary = {
            'initial_capital': config.initial_capital,
            'final_value': final_portfolio_value,
            'total_return_pct': total_return,
            'buy_hold_return_pct': buy_hold_return,
            'trades': trades,
            'total_trades': len(trades),
            'winning_trades': profitable_trades,
            'win_rate_pct': win_rate,
            'daily_values': daily_values,
            # PHASE 3: risk-adjusted metrics. These keys were READ by the past-
            # results panel and never WRITTEN, so Sharpe and max drawdown always
            # displayed as 0 and the panel guarding on `!= 0` never rendered.
            # Return alone cannot separate a good strategy from a leveraged one.
            **describe_run(daily_values, total_return_pct=total_return),
            'outperformed_market': total_return > buy_hold_return,
            'decisions': len(decisions_log),
            'llm_report': llm_report,
            'positioning_report': positioning_report,
            'text_sentiment_report': text_sentiment_report,
            'run_metadata': run_metadata,
        }

        logger.info(f"Run metadata: {json.dumps(run_metadata, default=str)}")
        logger.info(f"LLM contribution: {json.dumps(llm_report, default=str)}")
        logger.info(f"Positioning contribution: {json.dumps(positioning_report, default=str)}")
        logger.info(f"Text sentiment contribution: {json.dumps(text_sentiment_report, default=str)}")

        progress_bar.progress(100)
        status_text.text("✅ Simulation complete!")
        
        # Add completion confirmation before showing results
        st.success("🎉 **Simulation Processing Complete!** Results are now ready for display.")
        
        # Step 5: Display results (only after processing is complete)
        display_simulation_results(summary, df, decisions_log, symbol_input, show_reasoning)
        
        # Step 5a: Save simulation result to session state for past results
        save_simulation_result(summary, symbol_input, strategy_mode, start_date, end_date)
        
        # Step 6: Save results (skipped unless database persistence is enabled)
        if get_database_available():
            save_results_to_db(summary, df, symbol_input, start_date, end_date, interval)
        
        # Reset simulation state
        st.session_state.simulation_running = False
        
    except InsufficientHistory as e:
        # Expected, correctable stop: the requested window is too short for the
        # chosen interval. Not a bug, and not worth a traceback.
        st.session_state.simulation_running = False
        st.error(f"📏 Not enough history: {e}")
        st.info(
            f"The first {INDICATOR_WARMUP_BARS} bars are spent warming up the "
            "moving averages, so they can never be traded. A daily interval "
            f"needs more than {INDICATOR_WARMUP_BARS} calendar days; an hourly "
            f"interval needs more than {INDICATOR_WARMUP_BARS} hours."
        )

    except SyntheticDataBlocked as e:
        # Expected, deliberate stop: live data was unavailable and we refuse to
        # silently substitute fabricated prices.
        st.session_state.simulation_running = False
        st.error(f"🛑 Simulation stopped: {e}")
        st.info(
            "This guard exists so a synthetic run can never be mistaken for a real result. "
            "Check your network or Binance availability and retry."
        )

    except Exception as e:
        # Reset simulation state on error
        st.session_state.simulation_running = False
        st.error(f"Simulation failed: {str(e)}")
        st.exception(e)

# ── STREAMLIT UI ───────────────────────────────────────────────────────────
def main():
    """Main Streamlit application"""
    # NOTE: st.set_page_config() is called once at module scope (top of file),
    # because Streamlit requires it to be the very first st.* command.

    # Initialize session state for past results
    initialize_session_state()
    
    # Header
    st.title("🤖 AMAAI Multi-Agent Trading System")
    st.markdown("*Advanced Multi-Agent AI for Intelligent Trading Decisions*")
    
    # Sidebar configuration
    st.sidebar.header("📋 Trading Configuration")
    
    # Trading pair selection (always visible)
    symbol_input = st.sidebar.selectbox(
        "🎯 Select Trading Pair",
        ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT", "DOT/USDT"],
        index=0
    )
    
    # Time configuration (collapsible)
    with st.sidebar.expander("⏰ Time Configuration", expanded=True):
        # Date range selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - dt.timedelta(days=30),
                max_value=datetime.now().date()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
        
        # Time interval
        interval = st.selectbox(
            "Time Interval",
            ["1h", "4h", "1d"],
            index=0
        )
    
    # Convert to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())
    
    # Trading configuration (collapsible)
    with st.sidebar.expander("💰 Trading Configuration", expanded=True):
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100
        )
    
    # Trading Strategy Selection (collapsible)
    with st.sidebar.expander("⚡ Trading Strategy", expanded=True):
        strategy_mode = st.selectbox(
            "Trading Mode",
            ["conservative", "moderate", "aggressive"],
            index=1,  # Default to moderate
            help="Select your trading strategy preference"
        )
        
        # Display strategy characteristics
        if strategy_mode == "conservative":
            st.info("""
            **Conservative Strategy:**
            • 25% position size
            • High confidence required (0.80)
            • Strong signals needed (3+)
            • 3% stop loss, 6% take profit
            • RSI: 25/75 levels
            """)
        elif strategy_mode == "moderate":
            st.info("""
            **Moderate Strategy:**
            • 50% position size
            • Moderate confidence (0.65)
            • Balanced signals (2+)
            • 5% stop loss, 10% take profit
            • RSI: 30/70 levels
            """)
        else:  # aggressive
            st.info("""
            **Aggressive Strategy:**
            • 75% position size
            • Lower confidence (0.55)
            • Quick signals (1+)
            • 8% stop loss, 15% take profit
            • RSI: 35/65 levels
            """)
    
    # Advanced settings (collapsible)
    with st.sidebar.expander("🔧 Advanced Settings"):
        custom_position_size = st.number_input(
            "Custom Position Size (%)",
            min_value=5,
            max_value=100,
            value=50 if strategy_mode == "moderate" else (25 if strategy_mode == "conservative" else 75),
            step=5,
            help="Percentage of capital to use per trade"
        )
        
        custom_confidence = st.number_input(
            "Minimum Confidence",
            min_value=0.1,
            max_value=0.95,
            value=0.65 if strategy_mode == "moderate" else (0.80 if strategy_mode == "conservative" else 0.55),
            step=0.05,
            help="Minimum confidence required for trades"
        )
        
        custom_signal_threshold = st.number_input(
            "Signal Threshold",
            min_value=1,
            max_value=5,
            value=2 if strategy_mode == "moderate" else (3 if strategy_mode == "conservative" else 1),
            help="Minimum signal strength for trades"
        )
    
    # AI Configuration (collapsible)
    with st.sidebar.expander("🧠 AI Configuration", expanded=False):
        # Defaults to OFF, and says plainly what it does. It used to default ON
        # with the help text "Use vector database for historical pattern
        # matching", which oversold it: the store is built up during a single
        # run and discarded at the end, so it recalls only trades from the run
        # in progress and never anything from history. It also reaches the
        # decision only as text inside the LLM prompt, so it does nothing at all
        # in a rules-only run, and it has no telemetry -- unlike positioning and
        # text sentiment, there is no measurement of whether it changed anything.
        # Every ablation already runs with it off (experiments/harness.py), so
        # no reported number depends on it.
        enable_vector_db = st.checkbox(
            "Enable Vector Database (experimental)",
            value=False,
            disabled=not VECTOR_DB_AVAILABLE,
            help=("Feeds the LLM a summary of earlier trades from THIS run that "
                  "had similar conditions. The memory starts empty and is "
                  "discarded when the run ends, so it is recall within a run, "
                  "not learning across runs. Costs one embedding call per "
                  "trade. Unmeasured: excluded from all experiments."
                  if VECTOR_DB_AVAILABLE else "Install FAISS to enable")
        )
        
        show_reasoning = st.checkbox(
            "Show AI Reasoning",
            value=True,
            help="Display detailed AI decision reasoning"
        )
    
    # System Status (collapsible)
    with st.sidebar.expander("ℹ️ System Status", expanded=False):
        # LLM provider status
        llm_ok, llm_detail = get_llm_provider_status()
        if llm_ok:
            st.success(f"✅ LLM configured - {llm_detail}")
        else:
            st.error(f"❌ LLM not configured - {llm_detail}")
            st.info("Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY (Foundry) or OPENAI_API_KEY in your .env file")
        
        # Result persistence status
        if get_database_available():
            st.success("✅ Database connected - results are saved")
        elif ENABLE_DATABASE:
            st.warning("⚠️ Database enabled but unreachable - results kept in-session only")
        else:
            st.info("💾 Result saving disabled - past runs are kept for this session only")
        
        # Feature availability
        st.markdown("**Available Features:**")
        st.markdown(f"🗄️ Vector DB: {'✅' if VECTOR_DB_AVAILABLE else '❌'}")
        st.markdown(f"📊 Text Sentiment: {'✅' if TEXT_SENTIMENT_AVAILABLE else '❌'}")
    
    # Main content area
    if start_date >= end_date:
        st.error("❌ Start date must be before end date")
        return
    
    if (end_date - start_date).days > 90:
        st.warning("⚠️ Large date ranges may take longer to process and use more API calls")
    
    # Quick configuration summary (always visible)
    st.markdown(f"**Quick Setup:** {symbol_input} | {strategy_mode.title()} Mode | ${initial_capital:,} | {(end_date - start_date).days} days")
    
    # Display current configuration (collapsible)
    with st.expander("🎛️ Current Configuration", expanded=False):
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            st.markdown(f"""
            **Trading Setup:**
            - Symbol: {symbol_input}
            - Capital: ${initial_capital:,}
            - Interval: {interval}
            """)
        
        with config_col2:
            st.markdown(f"""
            **Date Range:**
            - Start: {start_date.strftime('%Y-%m-%d')}
            - End: {end_date.strftime('%Y-%m-%d')}
            - Days: {(end_date - start_date).days}
            """)
        
        with config_col3:
            st.markdown(f"""
            **Strategy Settings:**
            - Mode: {strategy_mode.title()}
            - Position Size: {custom_position_size}%
            - Min Confidence: {custom_confidence:.2f}
            - Signal Threshold: {custom_signal_threshold}
            """)
        
        # Display strategy summary inside the expander
        st.markdown("#### 📋 Strategy Summary")
        
        strategy_info_col1, strategy_info_col2, strategy_info_col3 = st.columns(3)
        
        with strategy_info_col1:
            st.markdown(f"""
            **Risk Profile:**
            - Trading Mode: **{strategy_mode.upper()}**
            - Position Size: **{custom_position_size}%** of capital
            - Confidence Required: **{custom_confidence:.0%}**
            """)
        
        with strategy_info_col2:
            # Calculate strategy characteristics. The trade ceiling is set by
            # the DECISION CADENCE and the interval, not by the mode: all three
            # presets decide equally often and differ only in how readily a
            # decision becomes a trade (signal_threshold, min_confidence).
            # These numbers used to be hardcoded fictions -- "10+ per week" for
            # aggressive, while a daily run could not exceed one decision every
            # six days -- so they are computed from the real cadence now.
            if strategy_mode == "conservative":
                risk_level, selectivity = "Low", "Very selective (3 signals, 80% confidence)"
            elif strategy_mode == "moderate":
                risk_level, selectivity = "Medium", "Selective (2 signals, 65% confidence)"
            else:  # aggressive
                risk_level, selectivity = "High", "Permissive (1 signal, 55% confidence)"

            step_bars = decision_step_bars(interval)
            bar_hours = interval_hours(interval) or 1.0
            hours_between = step_bars * bar_hours
            per_week = 168.0 / hours_between if hours_between else 0.0

            st.markdown(f"""
            **Trading Characteristics:**
            - Risk Level: **{risk_level}**
            - Decides: **every {hours_between:g}h** ({step_bars} × {interval} bar)
            - Max decisions: **{per_week:.0f} per week** (trades ≤ this)
            - Signal filter: **{selectivity}**
            """)
        
        with strategy_info_col3:
            st.markdown(f"""
            **AI Features:**
            - LLM decisions: {'On' if USE_LLM_DECISIONS else 'Off (rules-only arm)'}
            - Vector DB: {'On (experimental)' if enable_vector_db else 'Off'}
            - Reasoning: {'Shown' if show_reasoning else 'Hidden'}
            """)
        
        # Warning for API usage
        if (end_date - start_date).days > 7:
            st.info("💡 **Note:** Longer simulations will make more API calls to OpenAI. Monitor your usage if you have API limits.")
    
    # Run simulation button
    st.markdown("---")
    
    # Check if simulation is running
    simulation_running = st.session_state.get('simulation_running', False)
    
    if simulation_running:
        st.info("🔄 **Simulation in progress...** Please wait for completion before starting a new simulation.")
    
    run_button = st.button(
        "🚀 Start Trading Simulation",
        type="primary",
        help="Begin the multi-agent trading simulation" if not simulation_running else "Simulation currently running",
        use_container_width=True,
        disabled=simulation_running
    )
    
    # Quick test button for shorter runs
    col1, col2 = st.columns(2)
    
    with col1:
        quick_test = st.button(
            "⚡ Quick Test (Last 7 Days)",
            help="Run a quick test with the last 7 days of data" if not simulation_running else "Simulation currently running",
            disabled=simulation_running
        )
    
    with col2:
        # Disable button during simulation
        view_results_disabled = st.session_state.get('simulation_running', False)
        if st.button(
            "📊 View Past Results", 
            disabled=view_results_disabled,
            help="View previous simulation results" if not view_results_disabled else "Please wait for current simulation to complete"
        ):
            st.session_state.show_past_results = not st.session_state.get('show_past_results', False)
    
    # Display past results if toggled on
    if st.session_state.get('show_past_results', False):
        display_past_results()
    
    # Run simulation based on button clicked
    if run_button:
        st.info("🚀 Starting simulation...")
        try:
            run_trading_simulation(symbol_input, interval, start_date, end_date, initial_capital,
                                 enable_vector_db, show_reasoning,
                                 strategy_mode, custom_position_size, custom_confidence, custom_signal_threshold)
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.exception(e)
    
    elif quick_test:
        st.info("⚡ Starting quick test...")
        # Override dates for quick test
        quick_start_date = datetime.now() - dt.timedelta(days=7)
        quick_end_date = datetime.now()
        try:
            run_trading_simulation(symbol_input, interval, quick_start_date, quick_end_date, initial_capital,
                                 enable_vector_db, show_reasoning,
                                 strategy_mode, custom_position_size, custom_confidence, custom_signal_threshold)
        except Exception as e:
            st.error(f"Quick test failed: {str(e)}")
            st.exception(e)

# ── MAIN EXECUTION ──────────────────────────────────────────────────────────
# Call main function to run the Streamlit app
if __name__ == "__main__":
    main()
else:
    # When run with streamlit, call main() directly
    main()

