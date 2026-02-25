# Agent-vs-Agent Competition in Financial Markets: Annotated Bibliography

> **Thesis**: "Human trading will be replaced by agent-vs-agent competition."
>
> This bibliography compiles 25+ sources from academic papers, institutional reports, industry data, and live competitions that support or challenge the thesis that financial markets are transitioning from human-dominated trading to an ecosystem where AI agents primarily compete against each other.

---

## I. Multi-Agent Reinforcement Learning in Markets

### 1. Dou, Goldstein & Ji -- "AI-Powered Trading, Algorithmic Collusion, and Price Efficiency" (2024)
- **Citation**: Dou, W.W., Goldstein, I., & Ji, Y. (2024). *AI-Powered Trading, Algorithmic Collusion, and Price Efficiency*. NBER Working Paper No. 34054. [https://www.nber.org/papers/w34054](https://www.nber.org/papers/w34054)
- **Key Finding**: RL-based AI speculators autonomously sustain collusive supra-competitive profits without agreement, communication, or intent. Collusion emerges through two mechanisms: price-trigger strategies ("artificial intelligence") and learning bias that over-prunes aggressive strategies ("artificial stupidity").
- **Relevance**: **Core evidence for the thesis.** When agents compete, they may paradoxically learn to collude, fundamentally altering market dynamics in ways impossible with human traders. This is the Wharton/NBER paper that directly models agent-vs-agent interaction.
- **Year**: 2024

### 2. Fish, Gonczarowski & Shorrer -- "Algorithmic Collusion by Large Language Models" (2024)
- **Citation**: Fish, S., Gonczarowski, Y.A., & Shorrer, R.I. (2024). *Algorithmic Collusion by Large Language Models*. arXiv:2404.00806. [https://arxiv.org/abs/2404.00806](https://arxiv.org/abs/2404.00806)
- **Key Finding**: LLM-based pricing agents quickly and autonomously reach supracompetitive prices in oligopoly settings. Variation in seemingly innocuous phrases in LLM instructions substantially influence the degree of supracompetitive pricing. Results extend to auction settings.
- **Relevance**: Demonstrates that even general-purpose LLMs (not just RL agents) spontaneously develop collusive behavior when placed in competitive market settings. The "agent economy" naturally produces emergent coordination.
- **Year**: 2024 (revised 2025)

### 3. Wang et al. -- "Multi-Agent Reinforcement Learning for Market Making: Competition without Collusion" (2025)
- **Citation**: Wang, Z. et al. (2025). *Multi-Agent Reinforcement Learning for Market Making: Competition without Collusion*. ICAIF '25, ACM. arXiv:2510.25929. [https://arxiv.org/abs/2510.25929](https://arxiv.org/abs/2510.25929)
- **Key Finding**: Proposes interaction-level metrics that quantify behavioral asymmetry and system-level dynamics in multi-agent market making. Tests self-interested, competitive, and hybrid agent strategies to understand when competition leads to collusion vs. efficient markets.
- **Relevance**: **Directly models agent-vs-agent competition** in market making. Shows the spectrum from pure competition to emergent coordination when multiple RL agents are deployed simultaneously.
- **Year**: 2025

### 4. Lussange et al. -- "Mesoscale Effects of Trader Learning Behaviors in Financial Markets" (2024)
- **Citation**: Lussange, J. et al. (2024). *Mesoscale effects of trader learning behaviors in financial markets: A multi-agent reinforcement learning study*. PLoS ONE 19(4): e0301141. [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0301141](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0301141)
- **Key Finding**: Using a stock market simulator with autonomous RL-based investors, larger learning rates significantly increase crashes, herding behavior undermines market stability, while random (non-AI) trading preserves it.
- **Relevance**: When agents learn aggressively, market dynamics become fundamentally different from human-dominated markets. Implies that an all-agent market would exhibit qualitatively different crash dynamics.
- **Year**: 2024

### 5. Zou & Siebers -- "StockMARL: A Multi-Agent RL Stock Market Simulation" (2025)
- **Citation**: Zou, Z. & Siebers, P. (2025). *StockMARL: A Novel Multi-Agent Reinforcement Learning Framework*. EMSS 2025. [https://people.cs.nott.ac.uk/pszps/resources/zou-siebers-emss2025-corrected.pdf](https://people.cs.nott.ac.uk/pszps/resources/zou-siebers-emss2025-corrected.pdf)
- **Key Finding**: RL agents learn autonomously in a market populated by heterogeneous rule-based agents emulating real-world investor behaviors (day trading, momentum chasing, risk aversion), calibrated to a decade of real market data.
- **Relevance**: Provides a testbed for studying the transition period where AI agents coexist with rule-based (human-proxy) agents, modeling the replacement trajectory.
- **Year**: 2025

---

## II. Market Microstructure with AI

### 6. Wang et al. -- "When AI Trading Agents Compete: Adverse Selection of Meta-Orders by RL-Based Market Making" (2025)
- **Citation**: Wang, Z. et al. (2025). *When AI Trading Agents Compete: Adverse Selection of Meta-Orders by Reinforcement Learning-Based Market Making*. arXiv:2510.27334. [https://arxiv.org/abs/2510.27334](https://arxiv.org/abs/2510.27334)
- **Key Finding**: Investigates how medium-frequency trading agents are adversely selected by opportunistic high-frequency RL-based market makers within a Hawkes Limit Order Book model.
- **Relevance**: Models the predator-prey dynamics between AI agents operating at different frequencies -- a direct depiction of agent-vs-agent competition in market microstructure.
- **Year**: 2025

### 7. Sadighian -- "Gray-box Adversarial Attack of Deep RL-based Trading Agents" (2023)
- **Citation**: Sadighian, J. (2023). *Gray-box Adversarial Attack of Deep Reinforcement Learning-based Trading Agents*. arXiv:2309.14615. [https://arxiv.org/abs/2309.14615](https://arxiv.org/abs/2309.14615)
- **Key Finding**: Demonstrates a "gray-box" approach for attacking a Deep RL-based trading agent by trading in the same stock market, with no extra access required. Proves automated agents are vulnerable to adversarial manipulation by other agents.
- **Relevance**: In an agent-vs-agent world, adversarial exploitation becomes a primary competitive strategy. This paper formalizes the attack surface.
- **Year**: 2023

### 8. Guo et al. -- "Market Making with Deep Reinforcement Learning from Limit Order Books" (2023)
- **Citation**: Guo, H. et al. (2023). *Market Making with Deep Reinforcement Learning from Limit Order Books*. IEEE IJCNN 2023. arXiv:2305.15821. [https://arxiv.org/abs/2305.15821](https://arxiv.org/abs/2305.15821)
- **Key Finding**: Proposes an RL agent for market making that leverages neural networks with convolutional filters and attention mechanisms for feature extraction from limit order books, outperforming traditional market-making strategies.
- **Relevance**: AI market makers replace human market makers -- a concrete example of the transition from human to agent dominance in a specific market function.
- **Year**: 2023

---

## III. The "Agent Economy" Thesis

### 9. IMF -- "Global Financial Stability Report, Chapter 3: AI and Capital Markets" (October 2024)
- **Citation**: International Monetary Fund. (2024). *Global Financial Stability Report, October 2024 -- Steadying the Course: Uncertainty, Artificial Intelligence, and Financial Stability*, Chapter 3. [https://www.imf.org/en/publications/gfsr/issues/2024/10/22/global-financial-stability-report-october-2024](https://www.imf.org/en/publications/gfsr/issues/2024/10/22/global-financial-stability-report-october-2024)
- **Key Finding**: AI adoption in capital markets is likely to increase significantly. Warns that AI-based participants may take increasingly correlated positions due to convergence on similar model designs and datasets. The potential for increased market speed and volatility under stress, and greater opacity, creates new systemic risks.
- **Relevance**: The IMF explicitly recognizes the structural shift toward AI-dominated markets and warns about agent-vs-agent dynamics (correlated positions, herding, volatility amplification).
- **Year**: 2024

### 10. Citi GPS -- "Agentic AI: Finance & the 'Do It For Me' Economy" (2025)
- **Citation**: Citi GPS. (2025). *Agentic AI: Finance & the 'Do It For Me' Economy*. [https://www.citigroup.com/global/insights/agentic-ai](https://www.citigroup.com/global/insights/agentic-ai)
- **Key Finding**: Agentic AI could have a bigger impact on the economy and finance than the internet era. Users will have their own AI agents choosing products, executing transactions, and managing portfolios. 37% of VC funding in 2024 went to AI startups, with autonomous agents seeing the biggest growth.
- **Relevance**: Major Wall Street institution (Citi) explicitly articulates the vision of an agent-mediated economy where bots transact with bots on behalf of humans.
- **Year**: 2025

### 11. World Economic Forum -- "How Agentic AI Will Transform Financial Services" (2024)
- **Citation**: World Economic Forum. (2024). *How Agentic AI will transform financial services*. [https://www.weforum.org/stories/2024/12/agentic-ai-financial-services-autonomy-efficiency-and-inclusion/](https://www.weforum.org/stories/2024/12/agentic-ai-financial-services-autonomy-efficiency-and-inclusion/)
- **Key Finding**: AI agents are transitioning from tools that suggest to systems that act autonomously. Direct AI-to-AI communication is emerging, where agents coordinate tasks, request data from one another, and dynamically adjust strategies in real time.
- **Relevance**: Explicitly describes the future as agent-to-agent interaction, not human-to-human or even human-to-agent.
- **Year**: 2024

---

## IV. Emergent Behavior & Systemic Risk

### 12. IMF Blog -- "AI Can Make Markets More Efficient -- and More Volatile" (2024)
- **Citation**: Obstfeld, M. et al. (2024). *Artificial Intelligence Can Make Markets More Efficient -- and More Volatile*. IMF Blog. [https://www.imf.org/en/blogs/articles/2024/10/15/artificial-intelligence-can-make-markets-more-efficient-and-more-volatile](https://www.imf.org/en/blogs/articles/2024/10/15/artificial-intelligence-can-make-markets-more-efficient-and-more-volatile)
- **Key Finding**: AI may reduce some financial stability risks via better risk management and deeper liquidity, but new risks arise from increased market speed, volatility under stress, opacity, and reliance on few AI-service providers.
- **Relevance**: Documents the double-edged nature of agent-dominated markets -- efficiency gains but also new categories of systemic risk that only exist when agents interact with agents.
- **Year**: 2024

### 13. "AI as a Systemic Risk Amplifier in High-Frequency Trading" (2024)
- **Citation**: (2024). *AI as a Systemic Risk Amplifier in High-Frequency Trading*. BCT Journal. [https://bctjournal.com/article_492_981a2fa9523f01153d5b5e1c25337126.pdf](https://bctjournal.com/article_492_981a2fa9523f01153d5b5e1c25337126.pdf)
- **Key Finding**: AI-driven trading systems using deep/reinforcement learning may converge on similar trading strategies when exposed to the same price signals. In stress scenarios, they act in unison, amplifying volatility and undermining liquidity.
- **Relevance**: Algorithmic herding -- multiple AI agents converging on identical strategies -- is a unique emergent property of agent-vs-agent markets that cannot occur in human-dominated markets at the same scale.
- **Year**: 2024

### 14. Lawfare -- "Selling Spirals: Avoiding an AI Flash Crash" (2024)
- **Citation**: Lawfare. (2024). *Selling Spirals: Avoiding an AI Flash Crash*. [https://www.lawfaremedia.org/article/selling-spirals--avoiding-an-ai-flash-crash](https://www.lawfaremedia.org/article/selling-spirals--avoiding-an-ai-flash-crash)
- **Key Finding**: Liquidity crunches or flash crashes might be more frequent and harder to unravel when algorithms are interacting with each other at high speed, without human intervention or understanding.
- **Relevance**: Articulates the nightmare scenario of agent-vs-agent markets: cascading failures where no human is in the loop.
- **Year**: 2024

### 15. Chen et al. -- "Exploring Sentiment Manipulation by LLM-Enabled Intelligent Trading Agents" (2025)
- **Citation**: Chen, X. et al. (2025). *Exploring Sentiment Manipulation by LLM-Enabled Intelligent Trading Agents*. arXiv:2502.16343. [https://arxiv.org/abs/2502.16343](https://arxiv.org/abs/2502.16343)
- **Key Finding**: First investigation of an intelligent trading agent (continuous deep RL) that controls an LLM to post on social media, manipulating the sentiment observed by other traders. The agent learns to augment its profit through a pump-and-dump scheme.
- **Relevance**: **Critical for the thesis.** In an agent-vs-agent world, manipulation shifts from human deception to automated sentiment warfare -- agents generating fake signals to exploit other agents.
- **Year**: 2025

---

## V. Real-World Agent Competitions & Benchmarks

### 16. nof1.ai -- "Alpha Arena: Six Frontier LLMs' Trading Competition" (2024-2025)
- **Citation**: nof1.ai. (2024-2025). *Alpha Arena: LLM Crypto Trading Competition*. [https://nof1.ai/](https://nof1.ai/) | Coverage: [https://www.euclideanai.com/blog/llm-crypto-trading](https://www.euclideanai.com/blog/llm-crypto-trading)
- **Key Finding**: Six frontier LLMs (GPT-5, Claude 4.5, Gemini 2.5 Pro, Grok 4, Qwen 3 Max, DeepSeek V3.1) traded crypto perpetuals on Hyperliquid with $10,000 real capital each. Qwen 3 Max won Season 1. The competition proved LLMs can autonomously trade with real money.
- **Relevance**: **The most literal instantiation of agent-vs-agent competition.** Real money, real markets, no humans -- just LLMs competing against each other.
- **Year**: 2024-2025

### 17. ICAIF -- "FinRL Contest 2024" (2024)
- **Citation**: ACM ICAIF. (2024). *FinRL Contest 2024*. [https://open-finance-lab.github.io/finrl-contest-2024.github.io/](https://open-finance-lab.github.io/finrl-contest-2024.github.io/)
- **Key Finding**: Competition challenges participants to develop innovative financial trading strategies using ensemble methods for cryptocurrency and LLM-generated signals for stock trading, with second-level LOB data for Bitcoin.
- **Relevance**: Academic competitions accelerating the development of trading agents, building the ecosystem of agent capabilities.
- **Year**: 2024

### 18. ICAIF -- "Secure FinAI Contest 2025" (2025)
- **Citation**: ACM ICAIF. (2025). *Secure FinAI Contest 2025*. [https://icaif25.org/competitions/](https://icaif25.org/competitions/)
- **Key Finding**: Advances secure and reliable FinAI agents across three tasks: RL-transformer for crypto trading, LLM agents for SEC filings analysis, and LLM agents for regulation/compliance.
- **Relevance**: Competition structure now explicitly focuses on agent security and robustness -- recognition that agents will operate in adversarial (agent-vs-agent) environments.
- **Year**: 2025

### 19. AI-Trader Benchmark -- "Benchmarking Autonomous Agents in Real-Time Financial Markets" (2024)
- **Citation**: HKUDS. (2024). *AI-Trader: Benchmarking Autonomous Agents in Real-Time Financial Markets*. arXiv:2512.10971. [https://arxiv.org/abs/2512.10971](https://arxiv.org/abs/2512.10971)
- **Key Finding**: First fully-automated, live, data-uncontaminated evaluation benchmark. Six mainstream LLMs tested across U.S. stocks, A-shares, and crypto. **General intelligence does not automatically translate to effective trading capability.** Risk control determines cross-market robustness.
- **Relevance**: Establishes standardized benchmarks for measuring agent trading capability -- the infrastructure needed for systematic agent-vs-agent competition.
- **Year**: 2024

### 20. "When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets" (2025)
- **Citation**: (2025). *When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets*. arXiv:2510.00332. [https://arxiv.org/abs/2510.00332](https://arxiv.org/abs/2510.00332)
- **Key Finding**: Even frontier models achieve only 28% accuracy (vs. 80% human baseline) on adversarial financial tasks. Tool augmentation plateaus at 67.4%. Models are easily misled by manipulation and confidently hallucinate critical data. $30B was lost to crypto exploits in 2024.
- **Relevance**: **Counter-evidence to the thesis.** Current AI agents still significantly underperform humans in adversarial, high-stakes financial reasoning. The transition to agent-vs-agent may be slower than assumed for complex decision-making.
- **Year**: 2025

---

## VI. LLM Agents as Market Participants

### 21. Havrilla et al. -- "LLM Agents Do Not Replicate Human Market Traders: Evidence from Experimental Finance" (2025)
- **Citation**: Havrilla, A. et al. (2025). *LLM Agents Do Not Replicate Human Market Traders: Evidence From Experimental Finance*. arXiv:2502.15800. [https://arxiv.org/abs/2502.15800](https://arxiv.org/abs/2502.15800)
- **Key Finding**: LLMs exhibit "textbook-rational" behavior, pricing assets near fundamental value with muted bubble formation. In "battle royale" settings where multiple LLMs compete, they show less trading strategy variance than humans. Key behavioral features like large emergent bubbles are not reproduced.
- **Relevance**: **Nuanced evidence for the thesis.** Agent-dominated markets would be qualitatively different -- potentially more rational and less bubble-prone, but also more homogeneous and potentially fragile in novel ways.
- **Year**: 2025

### 22. Mariani et al. -- "Can Large Language Models Trade? Testing Financial Theories with LLM Agents" (2025)
- **Citation**: Mariani, G. et al. (2025). *Can Large Language Models Trade? Testing Financial Theories with LLM Agents in Market Simulations*. arXiv:2504.10789. [https://arxiv.org/abs/2504.10789](https://arxiv.org/abs/2504.10789)
- **Key Finding**: LLMs demonstrate consistent strategy adherence and can function as value investors, momentum traders, or market makers. Market dynamics exhibit features of real financial markets including price discovery, bubbles, underreaction, and strategic liquidity provision.
- **Relevance**: Demonstrates LLM agents can replicate the full spectrum of human trading roles, making complete human replacement technically feasible.
- **Year**: 2025

### 23. "Agent-Based Simulation of a Financial Market with Large Language Models" (2025)
- **Citation**: (2025). *Agent-Based Simulation of a Financial Market with Large Language Models*. arXiv:2510.12189. [https://arxiv.org/abs/2510.12189](https://arxiv.org/abs/2510.12189)
- **Key Finding**: LLM-based FCLAgents reproduce path-dependent patterns that conventional agents fail to capture. Reference points guiding loss aversion vary with market trajectories -- LLMs exhibit nuanced behavioral economics-like behavior.
- **Relevance**: LLM agents are not just rational optimizers; they exhibit behavioral complexity comparable to human traders, enabling richer agent-vs-agent dynamics.
- **Year**: 2025

### 24. TradingAgents -- "Multi-Agents LLM Financial Trading Framework" (2024-2025)
- **Citation**: Li, Y. et al. (2024). *TradingAgents: Multi-Agents LLM Financial Trading Framework*. arXiv:2412.20138. [https://arxiv.org/abs/2412.20138](https://arxiv.org/abs/2412.20138)
- **Key Finding**: LLM agents in specialized roles (fundamental/sentiment/technical analysts, risk managers, traders) outperform baselines in cumulative returns, Sharpe ratio, and max drawdown. Framework mimics institutional trading firm structure.
- **Relevance**: Models the internal structure of AI-native trading firms -- not just individual agents but organized teams of agents replacing entire firms' worth of human traders.
- **Year**: 2024

---

## VII. Industry Adoption Data

### 25. Quantified Strategies -- "What Percentage of Trading Is Algorithmic?" (2025)
- **Citation**: QuantifiedStrategies.com. (2025). *What Percentage of Trading Is Algorithmic?* [https://www.quantifiedstrategies.com/what-percentage-of-trading-is-algorithmic/](https://www.quantifiedstrategies.com/what-percentage-of-trading-is-algorithmic/)
- **Key Finding**: 60-80% of U.S. equity trading volume is algorithmic. 40-45% of options volume is algorithmic. The global algorithmic trading market reached ~$21B in 2024, projected to hit $43B by 2030 (12.9% CAGR).
- **Relevance**: **The transition is already well underway.** Humans are already the minority in equity trading. The remaining question is when the human fraction approaches zero.
- **Year**: 2025

### 26. JPMorgan, Goldman Sachs, and Wall Street AI Adoption (2024-2025)
- **Citation**: Training the Street. (2025). *The State of AI in Finance: 2025 Global Outlook*. [https://trainingthestreet.com/the-state-of-ai-in-finance-2025-global-outlook/](https://trainingthestreet.com/the-state-of-ai-in-finance-2025-global-outlook/) | Evident AI Index: [https://evidentinsights.com/bankingbrief/heres-the-2025-evident-ai-index/](https://evidentinsights.com/bankingbrief/heres-the-2025-evident-ai-index/)
- **Key Finding**: Wall Street's AI budget was $17B in 2024, set to double. JPMorgan ranked #1 in AI capabilities for 3 consecutive years, with $1.3B dedicated to AI in 2024. Goldman Sachs pioneering RL for derivatives trading and developing "agentic behavior" for autonomous task completion. Two Sigma's algorithm-driven strategies returned 10.9-14.3% in 2024. Renaissance Technologies' Medallion Fund returned 30%.
- **Relevance**: The largest financial institutions are all-in on AI trading. Institutional adoption ensures the competitive arms race dynamics that will push human traders out.
- **Year**: 2024-2025

### 27. Sidley Austin -- "Artificial Intelligence in Financial Markets: Systemic Risk and Market Abuse Concerns" (2024)
- **Citation**: Sidley Austin LLP. (2024). *Artificial Intelligence in Financial Markets: Systemic Risk and Market Abuse Concerns*. [https://www.sidley.com/en/insights/newsupdates/2024/12/artificial-intelligence-in-financial-markets-systemic-risk-and-market-abuse-concerns](https://www.sidley.com/en/insights/newsupdates/2024/12/artificial-intelligence-in-financial-markets-systemic-risk-and-market-abuse-concerns)
- **Key Finding**: The share of AI content in patent applications related to algorithmic trading rose from 19% in 2017 to over 50% annually since 2020. Most institutions currently favor simpler supervised learning models, but the patent pipeline signals a coming wave of advanced AI trading capabilities.
- **Relevance**: Leading indicator data showing the trajectory. The patent pipeline confirms that far more sophisticated agent capabilities are coming.
- **Year**: 2024

---

## Synthesis: Evidence Assessment

### Strong Support for the Thesis
| Evidence | Source |
|----------|--------|
| 60-80% of equity volume already algorithmic | #25 |
| Wall Street AI budgets doubling year-over-year | #26 |
| LLMs can autonomously trade with real money | #16, #19 |
| AI agents learn to collude without communication | #1, #2 |
| Agent-vs-agent predator-prey dynamics documented | #6, #7 |
| Multi-agent LLM frameworks replicate full trading firms | #24 |
| LLMs can execute value investing, momentum, and market making | #22 |
| AI patent share in algo trading exceeded 50% since 2020 | #27 |
| Major institutions (IMF, WEF, Citi) predict agent-dominated finance | #9, #10, #11 |

### Challenges / Counter-Evidence
| Evidence | Source |
|----------|--------|
| AI agents achieve only 28% accuracy in adversarial scenarios (vs. 80% human) | #20 |
| LLM agents do not replicate human behavioral dynamics (bubbles, irrationality) | #21 |
| Most institutions still use simple supervised learning, not RL | #27 |
| General intelligence does not translate to effective trading | #19 |
| Hallucination risk in high-stakes financial decisions | #20 |
| Agent herding creates new categories of systemic risk | #4, #13, #14 |

### Timeline Assessment

The transition is not binary but a spectrum:
1. **Already happened** (2010s-now): Execution, market making, high-frequency trading -- already agent-dominated.
2. **In progress** (2024-2027): Signal generation, portfolio construction, risk management -- rapidly being automated with LLM agents.
3. **Emerging** (2025-2030+): Fully autonomous investment decision-making, agent-to-agent negotiation, self-improving trading strategies.
4. **Uncertain**: Whether agents can handle truly adversarial, novel, crisis scenarios without human oversight.

---

*Compiled: 2026-02-21*
*Total sources: 27*
