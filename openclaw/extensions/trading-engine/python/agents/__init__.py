"""Multi-agent trading pipeline.

Agents:
    MarketAgent   — regime + sector momentum analysis
    QuantAgent    — XGBoost cross-sectional factor ranking (lazy — requires xgboost)
    Synthesizer   — weighted signal aggregation + final decisions
"""

from agents.market_agent import MarketAgent
from agents.synthesizer import Synthesizer

__all__ = ["MarketAgent", "QuantAgent", "Synthesizer"]


def __getattr__(name):
    """Lazy import for QuantAgent (avoids xgboost dependency at package load)."""
    if name == "QuantAgent":
        from agents.quant_agent import QuantAgent
        return QuantAgent
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
