"""The trading agents: technical, pattern, risk, decision, retrieval.

Extracted from `auto-trade.py` on 2026-09-21, verbatim apart from imports.
None of these classes ever touched Streamlit -- the weld was in the simulation
loop and the display layer around them, not in the agents themselves -- so this
move is a relocation, not a rewrite.

Why it matters: `TradingDecisionAgent` is the component every claim in the
paper rests on, and it lived in a 3,900-line Streamlit script alongside chart
rendering. The experiment harness could only reach it by faking the whole
Streamlit API, which meant the experiments depended on that stub staying
faithful. Now `core` holds the engine and imports no Streamlit at all.

Layer: depends on core.config, core.llm and core.data. Imports no UI.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pydantic import Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from core.config import (
    LLM_MAX_ADJUSTMENT,
    USE_LLM_DECISIONS,
    TradingAction,
    TradingConfig,
    TradingDecision,
)
from core.data import resolve_bar_index
from core.decision_log import get_decision_logger, rule_features
from core.llm import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    USE_AZURE_OPENAI,
    LLMContribution,
    LLMRiskAssessment,
    LLMTradeAdjustment,
    get_llm,
    prompt_stance_text,
    resolved_prompt_stance,
)

logger = logging.getLogger(__name__)

# The vector store is optional: FAISS ships a native wheel that is not always
# installable. Import failure disables the retrieval agent rather than the app.
try:
    import faiss  # noqa: F401
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    VECTOR_DB_AVAILABLE = True
except ImportError as e:  # pragma: no cover - depends on the environment
    logger.warning("Vector DB libraries not available: %s", e)
    VECTOR_DB_AVAILABLE = False
    FAISS = None
    Document = None


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
{prompt_stance_text()}"""

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

            # Observation only: what was asked, what came back, and the inputs
            # that were on the table. Inert unless DECISION_LOG_PATH is set, and
            # it cannot raise into this function -- a logging failure must never
            # change a trade.
            get_decision_logger().log(
                timestamp=timestamp,
                price=current_price,
                prompt=prompt,
                raw_adjustment=raw_adj,
                adjustment=clamped,
                clamped=(raw_adj != clamped),
                stance=contrib.stance,
                veto=contrib.veto,
                llm_confidence=contrib.confidence,
                key_factors=contrib.key_factors,
                rationale=contrib.rationale,
                rule_action=rule_action.value,
                final_action=final_action.value,
                changed_decision=contrib.changed_decision,
                adjusted_signal=adjusted_signal,
                final_confidence=final_conf,
                latency_s=latency,
                prompt_stance=resolved_prompt_stance(),
                holding=bool(portfolio.get("holding")),
                positioning=positioning,
                text_sentiment=text_sentiment,
                **rule_features(sig),
            )
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
