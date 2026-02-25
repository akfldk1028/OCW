# Academic Research Papers for Algorithmic Trading (2024-2026)
## Comprehensive Literature Review

**Date Compiled:** 2026-02-13
**Focus:** Sector rotation, multi-factor scoring, agent-based trading, regime detection, sentiment analysis, and RL-based portfolio construction

---

## 1. Sector Rotation with RL/ML

### 1.1 Foundational Research: Moskowitz & Grinblatt (1999)

**Paper:** "Do Industries Explain Momentum?"
**Authors:** Tobias J. Moskowitz, Mark Grinblatt
**Published:** The Journal of Finance, 1999
**Key Findings:**
- Strong and prevalent momentum effect in industry components of stock returns accounts for much of the individual stock momentum anomaly
- Industry momentum strategies (buying past winning industries, selling past losing industries) are highly profitable even after controlling for size, book-to-market equity, and individual stock momentum
- **Strategy Performance:** 0.43% monthly return (July 1963 - July 1995) using top 3/bottom 3 industries from 20 value-weighted portfolios based on 6-month lookback with 6-month holding period
- **Methodology:** Used 2-digit SIC codes to form 20 value-weighted industry portfolios

**Implications for Sector Scanner:**
- Validates industry/sector momentum as a fundamental trading signal
- 6-month lookback period is academically validated
- Cross-sectional momentum (relative performance) is more robust than time-series momentum
- Industry rotation is NOT just a momentum play—it captures distinct factors

**Academic Source:** [Do Industries Explain Momentum?](http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/Momentum/MoskowitzGrinblatt99.pdf) | [AQR Insights](https://www.aqr.com/Insights/Research/Journal-Article/Do-Industries-Explain-Momentum)

---

### 1.2 Recent ML-Based Sector Rotation

**Title:** "Python for Machine Learning-Based Sector Rotation Strategies"
**Year:** 2024-2025 practical application
**Key Approach:**
- Feature engineering: 5-day and 20-day moving averages of sector returns to capture recent trends
- Random Forest Classifier as primary ML algorithm
- Multiple decision trees averaged to improve accuracy
- Risk-on/risk-off regime detection: cyclical sectors (tech, consumer discretionary, financials) vs. defensive sectors (utilities, consumer staples, healthcare)

**Key Numbers:**
- Algorithms based on sector rotation consistently beat S&P 500 benchmark
- Works with both individual stocks and ETFs

**Implications for Sector Scanner:**
- Moving averages of sector returns are effective features
- Random Forest is a validated approach for sector classification
- Regime-aware sector selection improves performance
- Equal-weighted vs. ML-weighted: ML weighting adds value

**Source:** [Medium - Python for ML-Based Sector Rotation](https://medium.com/@deepml1818/python-for-machine-learning-based-sector-rotation-strategies-69d7f97b5e29)

---

### 1.3 Momentum Factor Performance in 2024-2025

**Research:** "What Drove Momentum's Strong 2024"
**Year:** 2024-2025
**Key Findings:**
- In 2024, Momentum ranked as best-performing factor for US, developed ex-US, and emerging markets (4th time in 20 years)
- US Momentum's rolling 12-month excess return to S&P 500 in 96th percentile over past 50 years
- **Critical Difference from Dot-Com Era:** 2024 momentum driven by firms with durable growth and strong balance sheets, NOT speculation on weak profitability firms
- Momentum, low-risk, and quality factors remain relevant in modern era

**Implications for Sector Scanner:**
- Momentum is NOT dead—it's stronger than ever when combined with quality metrics
- Combining momentum with profitability/quality improves robustness
- Momentum strategy should filter out low-quality stocks

**Source:** [State Street - Momentum 2024 Analysis](https://www.ssga.com/us/en/intermediary/insights/what-drove-momentums-strong-2024-and-what-it-could-mean-for-2025)

---

## 2. Multi-Factor Stock Scoring with Learned Weights

### 2.1 Machine Learning Enhanced Multi-Factor Trading (June 2025)

**ArXiv ID:** 2507.07107
**Title:** "Machine Learning Enhanced Multi-Factor Quantitative Trading: A Cross-Sectional Portfolio Optimization Approach with Bias Correction"
**Year:** June 2025
**Key Contributions:**
- Comprehensive ML framework integrating multi-factor alpha discovery with bias correction
- PyTorch-accelerated factor computation
- **Adaptive factor weighting:** Model adapts factor weights based on regime-dependent performance
- Advanced portfolio optimization techniques

**Key Approach:**
- Dynamic factor weighting instead of static equal-weighting
- Regime-aware factor adaptation
- Bias correction to prevent overfitting

**Implications for Sector Scanner:**
- Don't use equal-weighted factors—learn optimal weights dynamically
- Regime detection should trigger factor weight rebalancing
- PyTorch acceleration enables real-time factor computation

**Source:** [arXiv 2507.07107](https://arxiv.org/html/2507.07107) | [PDF](https://www.arxiv.org/pdf/2507.07107)

---

### 2.2 Combined Machine Learning for Stock Selection (August 2025)

**ArXiv ID:** 2508.18592
**Title:** "Combined machine learning for stock selection strategy based on dynamic weighting methods"
**Year:** August 2025
**Models Tested:** Logistic Regression, XGBoost, LightGBM, AdaBoost, CatBoost, Multilayer Perceptron
**Winner:** XGBoost selected as optimal model
**Weighting Method:** AdaBoost algorithm for dynamic weighting

**Key Findings:**
- AdaBoost adjusts data weights and weak classifier weights to improve accuracy and robustness
- Dynamic weighting outperforms static factor models
- XGBoost provides best balance of speed, accuracy, and interpretability

**Implications for Sector Scanner:**
- XGBoost is academically validated for stock selection
- AdaBoost-style dynamic weighting can improve ensemble performance
- Multiple weak classifiers combined > single strong classifier

**Source:** [arXiv 2508.18592](https://arxiv.org/html/2508.18592v1)

---

### 2.3 Deep Learning for Short-Term Equity Trend Forecasting (August 2025)

**ArXiv ID:** 2508.14656
**Title:** "Deep Learning for Short-Term Equity Trend Forecasting: A Behavior-Driven Multi-Factor Approach"
**Year:** August 2025
**Performance Metrics:**
- **MLP:** Sharpe Ratio 1.6075 (highest)
- **CNN:** Sharpe Ratio 1.1487
- **SVM:** Sharpe Ratio 0.7709 (significantly lower)

**Key Findings:**
- Deep learning models (MLP, CNN) significantly outperform traditional ML (SVM)
- Behavior-driven factors improve prediction accuracy
- Short-term trend forecasting benefits from neural network architectures

**Implications for Sector Scanner:**
- Neural networks (especially MLP) validated for short-term forecasting
- Sharpe ratios above 1.5 are achievable with proper architecture
- Deep learning worth the complexity for short-term strategies

**Source:** [arXiv 2508.14656](https://arxiv.org/html/2508.14656v1)

---

### 2.4 R&D-Agent-Quant Multi-Agent Framework (September 2025)

**ArXiv ID:** 2505.15155
**Title:** "R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization"
**Year:** September 2025
**Markets Tested:** CSI 500 (China), NASDAQ 100 (US) from 2024 to June 2025
**Framework:** Multi-agent system for factor discovery and model optimization

**Key Approach:**
- Data-centric factor engineering
- Joint optimization of factors and models
- Multi-agent collaboration for alpha discovery

**Implications for Sector Scanner:**
- Multi-agent systems can optimize both feature engineering AND model selection
- Recent validation on both Chinese and US markets
- Agent-based approach to alpha discovery is emerging trend

**Source:** [arXiv 2505.15155](https://arxiv.org/html/2505.15155v2)

---

### 2.5 Stockformer: Price-Volume Factor Model (2024-2025)

**ArXiv ID:** 2401.06139
**Title:** "Stockformer: A Price-Volume Factor Stock Selection Model Based on Wavelet Transform and Multi-Task Self-Attention Networks"
**Year:** 2024-2025
**Key Innovation:**
- Integrates wavelet transform with multi-task self-attention networks
- Exceptional stability across market conditions (rising, falling, fluctuating)
- Particularly high performance during downturns and volatile periods

**Key Findings:**
- Price-volume factors capture critical market dynamics
- Wavelet transform decomposes time series into multiple frequency components
- Self-attention mechanism identifies important temporal patterns

**Implications for Sector Scanner:**
- Volume should be incorporated as a primary factor (not just price momentum)
- Transformer architectures validated for stock selection
- Model should be tested specifically during volatile/bearish regimes

**Source:** [arXiv 2401.06139](https://arxiv.org/pdf/2401.06139) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425004257)

---

### 2.6 Hybrid CNN-LSTM-GNN for Stock Prediction (2025)

**Paper:** "Hybrid CNN-LSTM-GNN Neural Network for A-Share Stock Prediction"
**Year:** 2025
**Performance:**
- **3-month cumulative return:** 20.7%
- **Annualized return:** ~110%

**Architecture:**
- CNN: Local feature extraction
- LSTM: Temporal dependency modeling
- GNN: Graph-based relationship modeling

**Implications for Sector Scanner:**
- Combining multiple architectures (CNN + LSTM + GNN) yields superior returns
- Graph neural networks can capture inter-stock relationships
- 110% annualized return suggests significant alpha potential with proper architecture

**Source:** [MDPI Entropy](https://www.mdpi.com/1099-4300/27/8/881)

---

### 2.7 Ensemble Methods: Gradient Boosting & XGBoost

**Research Summary (2024-2025):**
- **XGBoost ROC-AUC:** 0.953 in stock price prediction tasks
- **Accuracy Range:** 96-98% when combining fundamental, macroeconomic, and technical indicators
- **Stacking Ensembles:** Integrate ARIMA, Random Forest, Transformer, LSTM, GRU with XGBoost as meta-learner
- **Performance Improvement:** 10-15% accuracy improvement vs. individual models, especially during regime transitions

**Key Findings:**
- LightGBM and XGBoost preferred for speed and efficiency
- Stacking multiple diverse models (statistical + ML + DL) with XGBoost meta-learner is optimal approach
- Ensemble methods reduce error volatility during market regime changes

**Implications for Sector Scanner:**
- Use XGBoost as final meta-learner for combining signals
- Stack diverse model types (statistical, ML, DL) for robustness
- Ensemble approach critical for regime transition periods

**Sources:**
- [Comparative Study of Ensemble Learning](https://www.sciencedirect.com/science/article/pii/S2468227624001066)
- [Stacked Heterogeneous Ensemble](https://www.mdpi.com/2227-7072/13/4/201)
- [Advancing Stock Price Prediction](https://link.springer.com/article/10.1186/s40537-025-01185-8)

---

## 3. Agent vs Agent Trading Competition

### 3.1 Trading Agent Competition (TAC)

**Organization:** Strategic Reasoning Group, University of Michigan
**Focus:** Annual competition advancing research in AI, multi-agent systems, and economic theory
**Key Characteristics:**
- Autonomous trading agents interact in simulated markets
- Strategic decision-making and resource allocation
- Evaluates performance in competitive trading scenarios

**Source:** [Strategic Reasoning Group TAC](https://strategicreasoning.org/trading-agent-competition/) | [ACM AAMAS 2025](https://dl.acm.com/doi/10.5555/3398761.3399091)

---

### 3.2 Agent Exchange: AI Agent Economics (July 2025)

**ArXiv ID:** 2507.03904
**Title:** "Agent Exchange: Shaping the Future of AI Agent Economics"
**Year:** July 2025
**Key Concept:** AI agents are no longer merely computational tools but active participants in economic systems capable of value creation through collaborative processes

**Key Findings:**
- Foundation models enable wide adoption of AI agents interacting with humans and each other
- Cooperation and safety necessities as agents gain autonomy in high-stakes markets
- Markets become "agentic" as autonomous agents participate as primary actors

**Source:** [arXiv 2507.03904](https://arxiv.org/html/2507.03904v1)

---

### 3.3 Market Growth: The Agent Economy Thesis

**Market Size Projections:**
- **2024:** Global autonomous agents market at $3.06 billion
- **2034 Projection:** $103.28 billion (CAGR: 42.19%)
- **Agentic AI Market (2025):** $7.55 billion
- **Agentic AI Market (2034):** $199.05 billion (CAGR: 43.84%)
- **GDP Impact Projection:** $2.6–4.4 trillion annually by 2030

**2025 Market Status:**
- Leading agents in autonomous trading: **200%+ annualized returns**, 65-75% win rates
- AI trading agents leverage Financial Learning Models (FLMs) for 5-min and 15-min timeframes
- Large enterprises contribute 69% of autonomous agents market revenue (2025)

**Source:** [Precedence Research - Agentic AI](https://www.precedenceresearch.com/agentic-ai-market) | [Autonomous Agents Market](https://www.precedenceresearch.com/autonomous-agents-market)

---

### 3.4 Multi-Agent Trading Systems in Practice (2025)

**Application:** Multi-agent systems coordinate autonomous AI agents that analyze markets, execute trades, and manage risk in parallel
**Key Technologies:**
- Agentic AI for real-time collaboration
- Adaptive strategy control
- Resilient decision-making in live trading environments

**Use Cases (2025):**
- Portfolio management with agent teams
- Fraud detection and compliance
- High-frequency trading with competing agents
- Complex strategy implementation through agent coordination

**Sources:**
- [Multi-Agent Trading Systems with LangChain](https://servicesground.com/blog/multi-agent-trading-systems/)
- [9 Use Cases of Agentic AI for Stock Trading 2025](https://www.ampcome.com/post/9-use-cases-of-agentic-ai-for-stock-trading-in-2025)

---

## 4. Market Regime Detection for Adaptive Trading

### 4.1 Generating Alpha: Hybrid AI-Driven Trading System (January 2025)

**ArXiv ID:** 2601.19504
**Title:** "Generating Alpha: A Hybrid AI-Driven Trading System Integrating Technical Analysis, Machine Learning and Financial Sentiment for Regime-Adaptive Equity Strategies"
**Conference:** International Conference on Computing Systems and Intelligent Applications (ComSIA 2026)
**Publisher:** Springer Lecture Notes in Networks and Systems (LNNS)
**Backtest Period:** January 2023 - January 2025

**Key Components:**
- Classical technical analysis
- Statistical machine learning
- Sentiment filtering
- Market regime adaptation

**Regime Types Handled:**
- Bull markets
- Bear markets
- Sideways markets
- High volatility phases

**Key Finding:** System demonstrated flexible adaptability across ALL regime types, with regime-specific strategy adaptation

**Implications for Sector Scanner:**
- Regime detection is CRITICAL—different regimes require different strategies
- Hybrid approach (technical + ML + sentiment) outperforms single-method approaches
- Must test performance across all regime types separately

**Source:** [arXiv 2601.19504](https://arxiv.org/html/2601.19504v1) | [PDF](https://arxiv.org/pdf/2601.19504)

---

### 4.2 Hidden Markov Model for Regime Detection (2024-2025)

**Title:** "HMM-Based Market Regime Detection with RL for Portfolio Management"
**Year:** 2024-2025
**Methodology:**
- Train HMM on daily returns to find hidden "states"
- Typically 2 states: low-volatility and high-volatility
- Can extend to 3+ states: bull, bear, sideways

**Recent Performance (NIFTY 50, Jan 2018 - Dec 2024):**
- **Sharpe Ratio:** 1.0461
- **Sortino Ratio:** 1.5119
- **Cumulative Return:** 44.83%

**Bitcoin Trading Strategy (2024-2025):**
- HMM framework backtested from January 2024
- Historical data from 2008-2025
- Monte Carlo simulation for price forecasting

**Key Benefits:**
- Improves portfolio performance with higher returns
- Lower maximum drawdown
- Manages volatility clustering effectively

**Implications for Sector Scanner:**
- HMM is validated approach for regime detection
- 2-state models (high/low volatility) are sufficient
- Can be extended to 3-state (bull/bear/sideways) for sector rotation
- Sharpe >1.0 achievable with regime-adaptive strategies

**Sources:**
- [MDPI - Regime-Switching Factor Investing](https://www.mdpi.com/1911-8074/13/12/311)
- [Cloud-Conf HMM-RL Portfolio Management](https://www.cloud-conf.net/datasec/2025/proceedings/pdfs/IDS2025-3SVVEmiJ6JbFRviTl4Otnv/966100a067/966100a067.pdf)
- [Regime-Aware Short-Term Trading](https://internationalpubls.com/index.php/cana/article/view/6029)

---

### 4.3 Multi-Model Ensemble-HMM Voting Framework

**Title:** "A forest of opinions: A multi-model ensemble-HMM voting framework for market regime shift detection and trading"
**Year:** 2024-2025
**Approach:**
- Homogeneous ensemble methods (bagging and boosting)
- Hidden Markov Model for regime transitions
- Identifies bull, bear, and neutral market states
- Voting mechanism across multiple models

**Implications for Sector Scanner:**
- Ensemble of HMMs more robust than single HMM
- Voting mechanism reduces false regime switches
- Can detect neutral/sideways regime (not just bull/bear)

**Source:** [AIMS Press](https://www.aimspress.com/article/id/69045d2fba35de34708adb5d)

---

### 4.4 Statistical Jump Models for Regime-Aware Allocation

**ArXiv ID:** 2402.05272
**Title:** "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach"
**Year:** 2024 (revised 2025)
**Key Results:**
- Regime-switching strategy reduces market exposure during unfavorable regimes
- **Annualized return improvement:** 1-4% across different regions
- Focus on downside risk mitigation

**Implications for Sector Scanner:**
- Defensive positioning during unfavorable regimes adds 1-4% annual alpha
- Jump detection models can identify regime transitions earlier than HMM
- Risk management through regime-aware exposure adjustment is validated

**Source:** [arXiv 2402.05272](https://arxiv.org/html/2402.05272v2)

---

### 4.5 LLM Strategies Across Market Regimes

**ArXiv ID:** 2505.07078
**Title:** "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?"
**Year:** 2025
**Key Finding:** LLM strategies show regime-dependent performance issues:
- **Bull markets:** Overly conservative, underperform passive benchmarks
- **Bear markets:** Overly aggressive, incur heavy losses

**Implications for Sector Scanner:**
- LLM-based strategies need regime-aware calibration
- Risk tolerance parameters should adjust by regime
- Pure LLM approaches underperform regime-adaptive ML strategies

**Source:** [arXiv 2505.07078](https://arxiv.org/html/2505.07078v4)

---

## 5. Market-Derived Sentiment Labels

### 5.1 Market-Derived Financial Sentiment Analysis (February 2025)

**ArXiv ID:** 2502.14897 (CRITICAL PAPER)
**Title:** "Market-Derived Financial Sentiment Analysis: Context-Aware Language Models for Crypto Forecasting"
**Submitted:** February 17, 2025 | Revised: March 2, 2025
**Focus:** Cryptocurrency forecasting (generalizable to equities)

**Core Hypothesis:** Historical market reactions to words offer MORE reliable indicators than subjective human sentiment labels

**Methodology:**
- Market-derived labeling: Assign tweet labels based on ensuing short-term price trends
- Enables language model to capture direct relationship between textual signals and market dynamics
- Domain-specific language model fine-tuned on market-derived labels

**Key Results:**
- **11% improvement** in short-term trend prediction accuracy over traditional sentiment-based benchmarks
- Outperforms human-labeled sentiment models
- More predictive than generic sentiment analysis (e.g., VADER, TextBlob)

**Implications for Sector Scanner:**
- Don't use pre-labeled sentiment data (FinancialPhraseBank, etc.)
- Train sentiment models on actual price reactions (next-day, next-week returns)
- Market-derived labels > human labels for trading
- 11% accuracy improvement is substantial for alpha generation

**Source:** [arXiv 2502.14897](https://arxiv.org/abs/2502.14897) | [HTML](https://arxiv.org/html/2502.14897) | [PDF](https://arxiv.org/pdf/2502.14897)

---

### 5.2 FinBERT Financial Sentiment Analysis (2024-2025)

**Model:** FinBERT (pre-trained BERT fine-tuned on financial corpus)
**Architecture:** BERT-based transformer pre-trained on financial text, fine-tuned for sentiment classification

**Recent Applications (2024-2025):**

#### FinBERT-LSTM Framework
- Multimodal prediction: FinBERT sentiment + technical indicators + statistical features
- **Performance:** FinBERT-LSTM > LSTM > DNN
- Sentiment analysis significantly enhances market fluctuation anticipation

#### FinBERT with SHAP Explainability (2025)
**Title:** "Stock Price Prediction Using FinBERT-Enhanced Sentiment with SHAP Explainability and Differential Privacy"
**Year:** 2025
**Key Innovation:** Combines FinBERT sentiment with SHAP (SHapley Additive exPlanations) for interpretability

#### FinBERT vs. GPT-4 Comparison (2024)
- Comparative study of FinBERT, GPT-4, and Logistic Regression
- FinBERT optimized for financial domain > general-purpose GPT-4 for financial sentiment
- Domain-specific pre-training provides significant advantage

**Implications for Sector Scanner:**
- FinBERT is validated tool for financial sentiment extraction
- Combining FinBERT + LSTM outperforms standalone models
- FinBERT > GPT-4 for financial sentiment (domain specialization matters)
- Interpretability (SHAP) important for production systems

**Sources:**
- [MDPI - FinBERT with SHAP](https://www.mdpi.com/2227-7390/13/17/2747)
- [ACM - FinBERT-LSTM Stock Prediction](https://dl.acm.org/doi/10.1145/3694860.3694870)
- [arXiv 2306.02136](https://arxiv.org/abs/2306.02136)
- [MDPI - FinBERT vs GPT-4](https://www.mdpi.com/2504-2289/8/11/143)

---

## 6. RL-Based Stock Selection / Portfolio Construction

### 6.1 Portfolio Double Deep Q-Network (PDQN) (2024)

**Paper:** "Deep reinforcement learning for portfolio selection"
**Year:** 2024
**Architecture:** Novel PDQN (Portfolio Double Deep Q-Network)

**Key Innovations:**
- Integrates Double Q-Learning to reduce overestimation
- Leaky ReLU activation
- Xavier initialization
- Huber loss function
- Dropout regularization for generalization

**Implications for Sector Scanner:**
- Double Q-Learning addresses overestimation bias in standard DQN
- Specific architectural choices (Leaky ReLU, Huber loss) validated for portfolio management
- Regularization critical for preventing overfitting

**Source:** [ScienceDirect - Deep RL for Portfolio Selection](https://www.sciencedirect.com/science/article/pii/S1044028324000887)

---

### 6.2 Smart Tangency Portfolio: Dynamic Rebalancing (2024-2025)

**Paper:** "Smart Tangency Portfolio: Deep Reinforcement Learning for Dynamic Rebalancing and Risk–Return Trade-Off"
**Year:** 2024-2025
**Algorithms:** PPO (Proximal Policy Optimization), A2C (Advantage Actor-Critic)

**Key Approach:**
- Integrates DRL with classical portfolio optimization
- Finds optimal trade-off between risk and return
- Determines optimal rebalancing frequency dynamically

**Implications for Sector Scanner:**
- PPO and A2C are validated algorithms for portfolio management
- Rebalancing frequency should be learned, not fixed
- Can integrate with Markowitz mean-variance framework

**Source:** [MDPI - Smart Tangency Portfolio](https://www.mdpi.com/2227-7072/13/4/227)

---

### 6.3 Multi-Agent RL for Dynamic Portfolio Optimization (December 2024)

**ArXiv ID:** 2412.18563
**Title:** "A Deep Reinforcement Learning Framework for Dynamic Portfolio Optimization: Evidence from China's Stock Market"
**Year:** December 2024
**Market:** China's stock market

**Framework:** Multi-agent RL with dynamic reward functions, group decision-making, and continual learning

**Performance Metrics:**
- **Annualized Return:** 68.83%
- **Sharpe Ratio:** 6.83
- Strong adaptability in volatile markets

**Key Innovations:**
- Multiple agents with different reward functions collaborate
- Group decision-making aggregates diverse strategies
- Continual learning adapts to market evolution

**Implications for Sector Scanner:**
- Multi-agent approach significantly outperforms single-agent (Sharpe 6.83 is exceptional)
- Diverse reward functions (returns, Sharpe, drawdown) should be combined
- Continual learning critical for non-stationary markets

**Source:** [arXiv 2412.18563](https://arxiv.org/abs/2412.18563)

---

### 6.4 Risk-Adjusted Deep RL: Multi-Reward Approach (2025)

**Paper:** "Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization: A Multi-reward Approach"
**Year:** 2025
**Framework:** Multiple DRL agents trained with distinct reward functions

**Reward Functions:**
1. Log returns (maximize returns)
2. Differential Sharpe ratio (maximize risk-adjusted returns)
3. Maximum drawdown (minimize downside risk)

**Unified Policy:** Combines all three agents into single policy incorporating multiple investment objectives

**Implications for Sector Scanner:**
- Single reward function insufficient—need multi-objective optimization
- Sharpe ratio and maximum drawdown should be explicit objectives
- Ensemble of agents with different objectives > single agent

**Source:** [Springer - Risk-Adjusted Deep RL](https://link.springer.com/article/10.1007/s44196-025-00875-8)

---

### 6.5 Behaviorally Informed Deep RL (2026)

**Paper:** "Behaviorally informed deep reinforcement learning for portfolio optimization with loss aversion and overconfidence"
**Year:** 2026 (early release)
**Key Innovation:** Integrates behavioral biases into RL architecture

**Behavioral Factors:**
- Loss aversion (investors feel losses more than equivalent gains)
- Overconfidence (overestimate prediction accuracy)
- Regime-dependent bias thresholds adjust position sizing
- RL policy determines trading direction

**Implications for Sector Scanner:**
- Human behavioral biases can improve RL performance (counterintuitive)
- Loss aversion useful for risk management
- Regime-dependent bias adjustment improves adaptability

**Source:** [Nature Scientific Reports](https://www.nature.com/articles/s41598-026-35902-x)

---

### 6.6 Volatility-Guided Asset Pre-Selection (2025)

**ArXiv ID:** 2505.03760
**Title:** "Deep Reinforcement Learning for Investor-Specific Portfolio Optimization: A Volatility-Guided Asset Selection Approach"
**Year:** 2025
**Key Innovation:** DRL with investor-specific risk profiles and volatility-guided pre-selection

**Critical Insight:** "The choice of optimal assets is crucial for portfolio construction as it determines overall performance. Effective pre-selection reduces computational efficiency, avoids overfitting risk, and ensures proper diversification, though very few studies focus on this crucial component."

**Implications for Sector Scanner:**
- Asset/sector pre-selection is CRITICAL but understudied
- Volatility-based filtering improves RL training efficiency
- Risk profile should guide asset universe selection
- Pre-selection prevents overfitting and improves generalization

**Source:** [arXiv 2505.03760](https://arxiv.org/html/2505.03760)

---

### 6.7 Stock Ranking and Matching: RL Approach (2025)

**Paper:** "Optimizing portfolio selection through stock ranking and matching: A reinforcement learning approach"
**Year:** 2025
**Key Approach:**
- RL used for stock ranking (which stocks to trade)
- Matching algorithm for portfolio construction (how much to allocate)
- Two-stage process: selection + allocation

**Implications for Sector Scanner:**
- Separate stock selection from position sizing
- RL for ranking/selection, optimization for allocation
- Two-stage approach improves interpretability and performance

**Source:** [ScienceDirect - Stock Ranking and Matching](https://www.sciencedirect.com/science/article/abs/pii/S0957417425000521)

---

### 6.8 Graph Attention-Based Heterogeneous MARL (2025)

**Paper:** "Graph attention-based heterogeneous multi-agent deep reinforcement learning for adaptive portfolio optimization"
**Year:** 2025
**Key Innovation:** Graph neural networks + multi-agent RL

**Architecture:**
- Graph attention mechanism captures inter-stock relationships
- Heterogeneous agents (different objectives/strategies)
- Adaptive portfolio construction

**Implications for Sector Scanner:**
- Graph neural networks can model sector correlations
- Stock relationships (within/across sectors) are valuable features
- Heterogeneous agents improve robustness

**Source:** [Nature Scientific Reports](https://www.nature.com/articles/s41598-025-32408-w)

---

### 6.9 Multi-Agent RL for Market Making (2025)

**ArXiv ID:** 2510.25929
**Title:** "Multi-Agent Reinforcement Learning for Market Making: Competition without Collusion"
**Conference:** 6th ACM International Conference on AI in Finance (November 2025, Singapore)
**Year:** 2025

**Key Findings:**
- MARL enables simulation of strategic interactions in markets
- Competition dynamics without collusive behavior
- Extends classical RL to strategic interdependence
- Facilitates simulation of execution competition and behavioral feedback

**Source:** [arXiv 2510.25929](https://arxiv.org/html/2510.25929v1) | [PDF](https://arxiv.org/pdf/2510.25929)

---

### 6.10 JaxMARL-HFT: GPU-Accelerated MARL for High-Frequency Trading (2025)

**ArXiv ID:** 2511.02136
**Title:** "JaxMARL-HFT: GPU-Accelerated Large-Scale Multi-Agent Reinforcement Learning for High-Frequency Trading"
**Year:** November 2025

**Key Innovation:**
- First GPU-accelerated open-source MARL environment for HFT
- Market-by-order data support
- **240x reduction in end-to-end training time**
- Enables exploitation of large, granular datasets

**Implications for Sector Scanner:**
- GPU acceleration critical for MARL scalability
- JAX framework enables massive parallelization
- Large-scale MARL now computationally feasible
- Open-source framework available

**Source:** [arXiv 2511.02136](https://arxiv.org/abs/2511.02136)

---

### 6.11 Comprehensive RL Meta-Analysis (2025)

**ArXiv ID:** 2512.10913
**Title:** "Reinforcement Learning in Financial Decision Making: A Systematic Review of Performance, Challenges, and Implementation Strategies"
**Year:** 2025
**Scope:** Meta-analysis of 167 publications from 2020-2025

**Focus Areas:**
- Market making
- Portfolio optimization
- Algorithmic trading

**Critical Finding:** "Successful RL in finance relies more on quality implementation, domain expertise, and data quality than complex algorithms"

**Implications for Sector Scanner:**
- Simpler, well-implemented RL > complex poorly-implemented RL
- Domain expertise (finance knowledge) more important than algorithm novelty
- Data quality is bottleneck, not model complexity
- Focus on engineering and data pipeline, not just algorithms

**Source:** [arXiv 2512.10913](https://arxiv.org/html/2512.10913v1)

---

## 7. Additional Relevant Research

### 7.1 Transformer Attention Mechanisms for Stock Prediction

#### TLOB: Dual Attention Transformer (February 2025)
**ArXiv ID:** 2502.15757
**Title:** "TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data"
**Year:** February 2025

**Key Innovation:**
- Dual attention mechanism: spatial + temporal dependencies
- Outperforms SOTA methods across all datasets and horizons
- **Critical Finding:** Stock price predictability declined over time by -6.68 F1-score

**Source:** [arXiv 2502.15757](https://arxiv.org/abs/2502.15757) | [HTML](https://arxiv.org/html/2502.15757v1)

---

#### Transformer Encoder with Multi-Feature Time2Vec (2025)
**ArXiv ID:** 2504.13801
**Title:** "Transformer Encoder and Multi-features Time2Vec for Financial Prediction"
**Year:** 2025

**Key Innovations:**
- Time2Vec encoding outperforms standard positional encoding
- Correlation feature selection across different markets
- Multi-stock prediction with cross-market features

**Source:** [arXiv 2504.13801](https://arxiv.org/abs/2504.13801)

---

#### MCI-GRU: Multi-Head Cross-Attention (October 2024)
**ArXiv ID:** 2410.20679
**Title:** "MCI-GRU: Stock Prediction Model Based on Multi-Head Cross-Attention and Improved GRU"
**Year:** October 2024

**Architecture:** Combines multi-head cross-attention with improved GRU for stock prediction

**Source:** [arXiv 2410.20679](https://arxiv.org/html/2410.20679v3) | [PDF](https://arxiv.org/pdf/2410.20679)

---

#### MASTER: Market-Guided Stock Transformer (December 2023 - relevant to 2024)
**ArXiv ID:** 2312.15235
**Title:** "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting"
**Year:** December 2023

**Key Innovation:** Market-guided attention mechanism for stock price forecasting

**Source:** [arXiv 2312.15235](https://arxiv.org/html/2312.15235v1)

---

### 7.2 Additional Multi-Agent Papers

#### ABIDES-MARL Framework (November 2025)
**ArXiv ID:** 2511.02016
**Title:** "ABIDES-MARL: A Multi-Agent Reinforcement Learning Framework"
**Year:** November 2025

**Key Application:** Multi-period trading games with heterogeneous agents:
- Informed trader
- Liquidity trader
- Noise traders
- Competing market makers

**Source:** [arXiv 2511.02016](https://arxiv.org/pdf/2511.02016)

---

#### StockMARL: Multi-Agent RL for Stock Trading
**Paper:** "StockMARL: A Novel Multi-Agent Reinforcement Learning..."
**Year:** 2025

**Source:** [University of Nottingham Research](https://people.cs.nott.ac.uk/pszps/resources/zou-siebers-emss2025-corrected.pdf)

---

## Summary: Key Numbers and Actionable Insights

### Sector Rotation
- **Moskowitz-Grinblatt (1999):** 0.43% monthly return (5.16% annually) with industry momentum
- **2024 Momentum:** 96th percentile performance vs. S&P 500 over 50 years
- **Feature Engineering:** 5-day and 20-day moving averages validated
- **Algorithm:** Random Forest validated; XGBoost optimal for classification

### Multi-Factor Scoring
- **Sharpe Ratios:** MLP 1.61, CNN 1.15, SVM 0.77 (deep learning wins)
- **XGBoost Accuracy:** 96-98% with combined factors
- **Ensemble Improvement:** 10-15% accuracy gain vs. individual models
- **Annualized Returns:** 110% achievable with CNN-LSTM-GNN hybrid (2025)

### Regime Detection
- **HMM Sharpe Ratio:** 1.05 (NIFTY 50, 2018-2024)
- **Regime-Switching Alpha:** +1-4% annualized returns
- **States:** 2-state (high/low volatility) or 3-state (bull/bear/sideways)

### Sentiment Analysis
- **Market-Derived Labels:** +11% accuracy vs. human labels
- **FinBERT-LSTM:** Best architecture for financial sentiment
- **Critical:** Train on actual price reactions, not pre-labeled sentiment

### Reinforcement Learning
- **Multi-Agent Sharpe:** 6.83 (exceptional)
- **Multi-Agent Return:** 68.83% annualized
- **Training Speedup:** 240x with GPU-accelerated MARL (JAX)
- **Key Insight:** Implementation quality > algorithm complexity

### Agent Economy
- **Market Growth (2025-2034):** 42-44% CAGR
- **Leading Agents (2025):** 200%+ returns, 65-75% win rates
- **Enterprise Adoption:** 69% of market revenue from large enterprises

---

## Recommendations for Sector Scanner Implementation

### 1. Core Strategy
- Implement Moskowitz-Grinblatt industry momentum with 6-month lookback
- Use cross-sectional ranking (relative performance) not time-series
- Combine momentum with quality/profitability filters (2024 insight)

### 2. Factor Engineering
- Dynamic factor weighting (NOT equal-weighted)
- Include volume factors, not just price momentum
- Use 5-day and 20-day moving averages
- Market-derived sentiment labels (train on price reactions)

### 3. Regime Detection
- HMM with 2-3 states minimum
- Adjust factor weights by regime
- Defensive positioning during unfavorable regimes
- Use ensemble-HMM voting for robustness

### 4. Model Architecture
- XGBoost for factor combination (meta-learner)
- Stack diverse models: statistical + ML + DL
- Consider MLP or transformer for non-linear patterns
- GPU-accelerated JAX for scalability

### 5. Reinforcement Learning (Advanced)
- Multi-agent RL with diverse reward functions (returns, Sharpe, drawdown)
- PPO or A2C for continuous action spaces
- Separate stock selection from position sizing
- Asset pre-selection critical for computational efficiency

### 6. Sentiment Integration
- Use FinBERT for financial text sentiment
- Train on market-derived labels (next-day/week returns)
- Combine FinBERT + LSTM architecture
- Avoid generic sentiment tools (VADER, TextBlob)

### 7. Risk Management
- Regime-aware position sizing
- Maximum drawdown as explicit objective
- Volatility-guided sector selection
- Adaptive rebalancing frequency

### 8. Validation Requirements
- Test across bull, bear, sideways regimes separately
- Target Sharpe ratio >1.5 (achievable based on literature)
- Monitor predictability degradation (TLOB: -6.68 F1 over time)
- Continual learning for market evolution

---

## Sources

All sources are embedded as hyperlinks throughout this document. Key source categories:

- **arXiv papers:** Preprints of academic research (2024-2026)
- **Journal articles:** Peer-reviewed publications from Nature, Springer, MDPI, ScienceDirect, ACM
- **Industry reports:** Market analysis from Precedence Research, State Street, AQR
- **Conference papers:** AAMAS, ACM AI in Finance, ComSIA 2026

**Last Updated:** 2026-02-13

---

## Invalidated Approaches

Based on the research:

1. **Equal-weighted factors:** Consistently underperform learned weights
2. **Human-labeled sentiment:** Market-derived labels provide +11% accuracy improvement
3. **Single-objective RL:** Multi-objective (returns + Sharpe + drawdown) significantly better
4. **Regime-agnostic strategies:** LLMs fail without regime adaptation (overly conservative in bull, overly aggressive in bear)
5. **Pure LLM strategies:** Underperform ML approaches without proper calibration
6. **Static rebalancing:** Dynamic rebalancing frequency learned by RL outperforms fixed schedules
7. **Individual stock momentum alone:** Industry/sector momentum explains much of individual momentum (Moskowitz-Grinblatt)

---

**End of Research Document**
