# Reference Papers

Research papers that informed the trading engine design, especially v8 architecture.

## Core Papers

### TradingAgents — Multi-Agent LLM Framework (NeurIPS 2025)
- **Xiao et al., "TradingAgents: Multi-Agents LLM Financial Trading Framework"**
- Multi-agent architecture: analyst team -> researcher -> trader -> risk manager
- Each agent specializes in one analysis domain
- Our v3 pipeline directly inspired by this: MarketAgent, QuantAgent, Synthesizer

### Cross-Sectional Momentum (Moskowitz & Grinblatt, 1999)
- **Foundation for sector rotation strategy**
- Sector-level momentum explains most of cross-sectional stock returns
- Our MarketAgent uses 14 sector ETFs with multi-window momentum

### Regime-Adaptive Trading (arXiv 2601.19504)
- **HMM-based regime detection for strategy switching**
- Sharpe 1.05, +1-4% alpha vs buy-and-hold
- Our regime_detector.py: 2-state HMM (low_vol/high_vol)
- Sector bias: defensive in high-vol, growth in low-vol

### FinBERT Sentiment (arXiv 2502.14897)
- **ProsusAI/finbert transformer for financial sentiment**
- +11% accuracy over keyword-based sentiment
- Our sentiment_finbert.py uses this model

### LLM+RL Hybrid (arXiv 2511.12120)
- **Sharpe 1.10** with LLM sentiment as RL observation feature
- 72% improvement over RL-only baseline
- Our ensemble_agent.py integrates sentiment into observation space

### RL Ensemble Methods (FinRL Literature)
- **PPO+A2C+DDPG ensemble achieves Sharpe 1.30**
- Dynamic Sharpe-based weight allocation
- Our ensemble_agent.py: PPO+A2C+SAC with Sharpe-weighted voting

### Meta-Analysis of 167 Papers
- **Algorithm selection explains only 8% of Sharpe variance**
- **Implementation quality explains 31%**
- Implication: engineering matters more than algorithm novelty
- This informed our focus on data processing, risk management, and validation

## v8-Specific Research

### Cross-Sectional Factor Models
- **Z-score normalization preserves signal magnitude** — key insight for v8
- Percentile ranking destroys information: stocks at 95th vs 99th both map to ~1.0
- Z-scoring: +1.1 sigma vs +2.3 sigma — XGBoost can learn meaningful splits

### Top-Quartile Labeling
- **Predicting "top 25% of returns" is more stable than "beat benchmark"**
- SPY-relative labels are noisy (market beta dominates)
- Top-quartile is regime-independent: always 25% positive regardless of market direction
- Class imbalance handled by scale_pos_weight in XGBoost

### Walk-Forward Validation
- **No look-ahead bias**: 18-month train, 21-day predict, 5-day purge gap
- Monthly retraining captures evolving market dynamics
- Literature consensus: expanding window often worse than rolling window

## Implementation Impact

| Finding | Source | Impact on Our Design |
|---------|--------|---------------------|
| Multi-agent specialization | NeurIPS 2025 TradingAgents | MarketAgent/QuantAgent/Synthesizer pipeline |
| Z-score > percentile ranking | Factor model literature | v8 QuantAgent normalization method |
| Top-quartile labeling | Cross-sectional studies | v8 label definition |
| HMM regime detection | arXiv 2601.19504 | Regime-dependent sector bias |
| FinBERT > keyword sentiment | arXiv 2502.14897 | Transformer-based sentiment |
| Sharpe-weighted ensemble | arXiv 2511.12120 | RL model combination |
| Quarter-Kelly sizing | Renaissance Technologies | Conservative position sizing |
| Implementation > Algorithm | Meta-analysis (167 papers) | Focus on data quality, risk mgmt |

## Backtest Validation History

| Version | Strategy | Excess vs SPY | Sharpe | Key Change |
|---------|----------|--------------|--------|------------|
| v1 | Sector scan + simple rank | +9.6% (12m) | — | Initial implementation |
| v6 | Percentile rank + XGBoost | +1.71% (3yr) | 0.95 | Cross-sectional ranking |
| v7 | Concentrated v6 (max 6) | -29.10% (3yr) | 1.15 | Fewer positions (failed) |
| **v8** | **Z-score + top-quartile** | **+38~123% (3yr)** | **1.28~1.56** | **Z-score, 13 features** |

## Full Research Notes

See [ai-trading-research.md](../../../ai-trading-research.md) for complete research survey including:
- 논문별 Sharpe ratio 비교
- 알고리즘 선택 가이드
- 데이터 품질 이슈
- 실전 배포 고려사항
