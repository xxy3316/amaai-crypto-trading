"""LLM provider setup and the structured decision contract.

Moved out of `auto-trade.py` verbatim. Holding the provider wiring and the
LLM's input/output schemas here keeps `core` importable without Streamlit,
which is what allows a headless backtest.

Layer: depends on `core.config` only.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Optional, Union

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, Field

from core.config import LLM_MAX_ADJUSTMENT, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

# ── UI ERROR REPORTING HOOK ─────────────────────────────────────────────────
# This module must not import Streamlit: that is the property that lets a
# headless backtest import core.*. The app injects st.error here at startup, so
# the red banners users relied on still appear, while a CLI or test run simply
# gets the message in the log.
_error_reporter = None


def set_error_reporter(reporter) -> None:
    """Register a UI callback for user-facing errors (the app passes st.error)."""
    global _error_reporter
    _error_reporter = reporter


def _report_error(message: str) -> None:
    logger.error(message)
    if _error_reporter is not None:
        try:
            _error_reporter(message)
        except Exception:  # a failing UI callback must not mask the real error
            logger.debug("error reporter itself failed", exc_info=True)


# ── LLM PROVIDER SETUP (Azure AI Foundry or OpenAI) ─────────────────────────
# Two providers are supported and selected automatically:
#   * Azure AI Foundry / Azure OpenAI - set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
#   * Public OpenAI                   - set OPENAI_API_KEY (key must start with 'sk-')
# Azure wins when both are configured. Azure keys are opaque (typically 32-char
# hex), so the 'sk-' prefix check is applied ONLY to public OpenAI keys.

def _normalize_azure_endpoint(endpoint: str) -> str:
    """Reduce a pasted Foundry/Azure URL to the resource base the SDK expects.

    Accepts the forms people copy out of the portal, e.g.
        https://my-res.openai.azure.com/
        https://my-res.services.ai.azure.com/models
        https://my-res.services.ai.azure.com/openai/v1
    and returns  https://my-res.<domain>  in every case.
    """
    endpoint = endpoint.strip().rstrip('/')
    for suffix in ('/openai/v1', '/openai/deployments', '/openai', '/models'):
        if endpoint.lower().endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return endpoint.rstrip('/')

AZURE_OPENAI_ENDPOINT = _normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT", ""))
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
# The Foundry *deployment* name, which is not necessarily the model name.
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview").strip()
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002").strip()

USE_AZURE_OPENAI = bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)

# Deterministic sampling by default: reproducibility matters more than variety
# for a backtest you intend to publish. Override with LLM_TEMPERATURE.
# LLM_TEMPERATURE is imported from core.config above.

# Some GPT-5 / o-series reasoning deployments reject `temperature` values other
# than the default. Others, including gpt-5.4-nano, accept it normally. Guessing
# from the deployment name is therefore unreliable in BOTH directions:
#   * guess "unsupported" wrongly -> LLM_TEMPERATURE is silently ignored and
#     runs stop being reproducible, which quietly invalidates a paper's results
#   * guess "supported" wrongly   -> every call fails with HTTP 400
# So the name is only used to decide WHETHER TO CHECK. When it looks like a
# reasoning model we make one tiny probe call and record what the API actually
# accepts. The answer is cached for the session and written into run metadata.
_REASONING_MODEL_MARKERS = ("gpt-5", "gpt5", "o1", "o3", "o4", "-nano", "reasoning")

def _looks_like_reasoning_deployment(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _REASONING_MODEL_MARKERS)

@lru_cache(maxsize=None)
def azure_supports_temperature(deployment: str, api_version: str) -> bool:
    """Ask the deployment itself whether it accepts a `temperature` argument.

    Returns True/False based on a single minimal completion. On any non-400
    failure (network, auth, quota) we fall back to the conservative assumption
    for the model family, because a transient error must not silently change
    sampling behaviour.
    """
    if not _looks_like_reasoning_deployment(deployment):
        return True  # ordinary chat deployments always accept it

    url = (f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{deployment}"
           f"/chat/completions?api-version={api_version}")
    payload = {
        "messages": [{"role": "user", "content": "ok"}],
        "temperature": 0,
        "max_completion_tokens": 4,
    }
    try:
        import urllib.request, urllib.error
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30):
            logger.info(f"Deployment '{deployment}' accepts temperature; using LLM_TEMPERATURE={LLM_TEMPERATURE}")
            return True
    except Exception as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode("utf-8", "replace")
            except Exception:
                body = ""
        if "temperature" in body.lower():
            logger.info(f"Deployment '{deployment}' rejects temperature; using the model default")
            return False
        logger.warning(
            f"Could not probe temperature support for '{deployment}' ({e}). "
            "Assuming unsupported, which is the safe default for this model family."
        )
        return False

def get_llm_provider_status() -> tuple[bool, str]:
    """Return (configured, human readable description) for the sidebar."""
    if USE_AZURE_OPENAI:
        return True, f"Azure AI Foundry - deployment `{AZURE_OPENAI_DEPLOYMENT}`"
    if AZURE_OPENAI_ENDPOINT and not AZURE_OPENAI_API_KEY:
        return False, "Azure endpoint set but AZURE_OPENAI_API_KEY is missing"
    if AZURE_OPENAI_API_KEY and not AZURE_OPENAI_ENDPOINT:
        return False, "Azure key set but AZURE_OPENAI_ENDPOINT is missing"
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return False, "No LLM credentials found (set AZURE_OPENAI_* or OPENAI_API_KEY)"
    if not key.startswith("sk-"):
        return False, "OPENAI_API_KEY is not a valid OpenAI key (must start with 'sk-')"
    return True, "OpenAI - model `gpt-4o-mini`"

@lru_cache(maxsize=None)
def get_llm() -> Union[ChatOpenAI, AzureChatOpenAI]:
    """Initialize the chat model for whichever provider is configured.

    Cached so all agents share one client instead of building four.
    """
    if USE_AZURE_OPENAI:
        try:
            temp_ok = azure_supports_temperature(AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION)
            logger.info(
                f"Using Azure AI Foundry endpoint {AZURE_OPENAI_ENDPOINT} "
                f"(deployment={AZURE_OPENAI_DEPLOYMENT}, api_version={AZURE_OPENAI_API_VERSION}, "
                f"temperature_supported={temp_ok})"
            )
            kwargs = dict(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION,
                max_retries=3,
                timeout=60,
            )
            if temp_ok:
                kwargs["temperature"] = LLM_TEMPERATURE
            return AzureChatOpenAI(**kwargs)
        except Exception as e:
            error_msg = f"Failed to initialize Azure AI Foundry client: {str(e)}"
            _report_error(error_msg)
            raise

    # ── Fall back to public OpenAI ──
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        error_msg = """
        No LLM credentials configured!

        For Azure AI Foundry, add to your .env file:
            AZURE_OPENAI_ENDPOINT=https://<your-resource>.services.ai.azure.com
            AZURE_OPENAI_API_KEY=<your key>
            AZURE_OPENAI_DEPLOYMENT=<your deployment name>

        For public OpenAI, add:
            OPENAI_API_KEY=sk-...

        Then restart the application.
        """
        _report_error(error_msg)
        raise ValueError("No LLM credentials configured (AZURE_OPENAI_* or OPENAI_API_KEY)")

    if not api_key.startswith('sk-'):
        error_msg = (
            "OPENAI_API_KEY does not look like a public OpenAI key (should start with 'sk-'). "
            "If you are using Azure AI Foundry, set AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_API_KEY instead."
        )
        _report_error(error_msg)
        raise ValueError("Invalid OpenAI API key format")

    try:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=LLM_TEMPERATURE,
            openai_api_key=api_key,
            max_retries=3,
            request_timeout=60
        )
    except Exception as e:
        error_msg = f"Failed to initialize OpenAI client: {str(e)}"
        logger.error(error_msg)
        _report_error(error_msg)
        raise

def describe_active_model() -> dict:
    """Machine-readable record of which model produced a run's decisions.

    Recorded alongside every result so a published number can be traced back to
    an exact deployment. Azure can update a deployment underneath you, so the
    api_version and deployment name belong in the run record.
    """
    if USE_AZURE_OPENAI:
        temp_ok = azure_supports_temperature(AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION)
        return {
            "provider": "azure_ai_foundry",
            "endpoint": AZURE_OPENAI_ENDPOINT,
            "deployment": AZURE_OPENAI_DEPLOYMENT,
            "api_version": AZURE_OPENAI_API_VERSION,
            "temperature": LLM_TEMPERATURE if temp_ok else "model_default",
            "temperature_configurable": temp_ok,
            "deterministic_sampling": bool(temp_ok and LLM_TEMPERATURE == 0),
        }
    return {
        "provider": "openai",
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": LLM_TEMPERATURE,
    }

# ── LLM DECISION CONTRACT (PHASE 1) ─────────────────────────────────────────
# The LLM returns a *bounded, structured* adjustment rather than free text.
# Bounded, because an unconstrained model overriding a rule engine is neither
# safe nor measurable; structured, because a paper needs the model's
# contribution as a number, not a paragraph.

# LLM_MAX_ADJUSTMENT is imported from core.config above.

class LLMTradeAdjustment(BaseModel):
    """Schema the decision LLM must fill in."""
    stance: str = Field(
        description="One of: bullish, bearish, neutral - the model's overall read of the setup"
    )
    signal_adjustment: int = Field(
        description=(
            "Integer adjustment to the rule-based signal score. Negative is more bearish, "
            f"positive more bullish. Must be between -{LLM_MAX_ADJUSTMENT} and +{LLM_MAX_ADJUSTMENT}."
        )
    )
    confidence: float = Field(
        description="Confidence in this assessment, between 0.0 and 1.0"
    )
    key_factors: List[str] = Field(
        description="2-4 short factors that drove the adjustment, each citing a specific input"
    )
    veto: bool = Field(
        description="True only to block a trade the rules would otherwise take, on clear risk grounds"
    )
    rationale: str = Field(
        description="One or two sentences explaining the adjustment"
    )

class LLMRiskAssessment(BaseModel):
    """Schema the risk LLM must fill in, replacing the old hardcoded defaults."""
    risk_level: str = Field(description="Exactly one of: low, medium, high")
    position_size_pct: float = Field(
        description="Recommended position size as a percent of capital, 1 to 100"
    )
    stop_loss_price: Optional[float] = Field(
        default=None, description="Suggested stop loss price in quote currency"
    )
    take_profit_price: Optional[float] = Field(
        default=None, description="Suggested take profit price in quote currency"
    )
    risk_reward_ratio: Optional[float] = Field(
        default=None, description="Expected reward divided by risk"
    )
    reasoning: str = Field(description="Two or three sentences justifying the assessment")

@dataclass
class LLMContribution:
    """Per-decision telemetry: what did the LLM actually change?

    This is the measurement that answers the reviewer question 'does the LLM
    add anything', so it is recorded for every single decision, not sampled.
    """
    invoked: bool = False
    succeeded: bool = False
    rule_action: str = "HOLD"
    final_action: str = "HOLD"
    rule_signal: int = 0
    adjustment: int = 0
    stance: str = "n/a"
    veto: bool = False
    confidence: float = 0.0
    key_factors: List[str] = None
    rationale: str = ""
    error: str = ""
    latency_s: float = 0.0

    @property
    def changed_decision(self) -> bool:
        return self.rule_action != self.final_action

    def to_dict(self) -> dict:
        return {
            "invoked": self.invoked,
            "succeeded": self.succeeded,
            "rule_action": self.rule_action,
            "final_action": self.final_action,
            "rule_signal": self.rule_signal,
            "adjustment": self.adjustment,
            "stance": self.stance,
            "veto": self.veto,
            "confidence": round(self.confidence, 3),
            "key_factors": self.key_factors or [],
            "rationale": self.rationale,
            "changed_decision": self.changed_decision,
            "error": self.error,
            "latency_s": round(self.latency_s, 2),
        }

