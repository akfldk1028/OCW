"""Step-by-step market analysis with real data.

Sequential thinking:
  Step 1: Raw data collection (14 sector ETFs + SPY)
  Step 2: Momentum indicators (4w/12w/26w returns)
  Step 3: Volume analysis (current vs 20d SMA)
  Step 4: RSI calculation (14-day)
  Step 5: Regime detection (HMM)
  Step 6: Sector ranking (relative strength vs SPY)
  Step 7: Stock-level analysis (top sectors)
  Step 8: FinBERT sentiment (news headlines)
  Step 9: Signal aggregation → decision
  Step 10: Transaction cost impact analysis
"""

import sys
import warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def calc_rsi(prices, period=14):
    """Calculate RSI from price series."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def main():
    print("=" * 70)
    print("STEP-BY-STEP MARKET ANALYSIS WITH REAL DATA")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # ============================================================
    # STEP 1: Data Collection
    # ============================================================
    print("\n[STEP 1] Downloading market data...")

    sector_map = {
        "Technology":     {"etf": "XLK",  "stocks": ["NVDA","AAPL","MSFT","AVGO","AMD","CRM","ORCL"]},
        "Semis":          {"etf": "SMH",  "stocks": ["NVDA","TSM","AVGO","MU","AMD","ASML","QCOM"]},
        "Financials":     {"etf": "XLF",  "stocks": ["JPM","V","MA","BAC","GS","WFC"]},
        "Healthcare":     {"etf": "XLV",  "stocks": ["LLY","UNH","JNJ","ABBV","TMO","MRK"]},
        "ConsDisc":       {"etf": "XLY",  "stocks": ["AMZN","TSLA","HD","MCD","NKE"]},
        "Communication":  {"etf": "XLC",  "stocks": ["META","GOOGL","NFLX","DIS","TMUS"]},
        "Industrials":    {"etf": "XLI",  "stocks": ["CAT","GE","RTX","BA","UNP","HON"]},
        "Energy":         {"etf": "XLE",  "stocks": ["XOM","CVX","COP","SLB"]},
        "Staples":        {"etf": "XLP",  "stocks": ["WMT","COST","PG","KO","PEP"]},
        "Biotech":        {"etf": "IBB",  "stocks": ["VRTX","GILD","AMGN","REGN"]},
        "Innovation":     {"etf": "ARKK", "stocks": ["TSLA","COIN","SHOP","ROKU"]},
        "Materials":      {"etf": "XLB",  "stocks": ["LIN","APD","SHW","ECL"]},
        "Utilities":      {"etf": "XLU",  "stocks": ["NEE","DUK","SO","D"]},
        "RealEstate":     {"etf": "XLRE", "stocks": ["PLD","AMT","EQIX","PSA"]},
    }

    etf_tickers = [s["etf"] for s in sector_map.values()] + ["SPY"]
    end = datetime.now()
    start = end - timedelta(days=400)  # ~26 weeks + buffer

    etf_data = yf.download(etf_tickers, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    etf_close = etf_data["Close"] if isinstance(etf_data.columns, pd.MultiIndex) else etf_data
    etf_volume = etf_data["Volume"] if isinstance(etf_data.columns, pd.MultiIndex) else etf_data

    print(f"  ETFs loaded: {len(etf_tickers)} tickers, {len(etf_close)} trading days")
    print(f"  Period: {etf_close.index[0].strftime('%Y-%m-%d')} ~ {etf_close.index[-1].strftime('%Y-%m-%d')}")

    # ============================================================
    # STEP 2: Momentum (4w / 12w / 26w relative to SPY)
    # ============================================================
    print("\n[STEP 2] Calculating momentum indicators...")

    spy = etf_close["SPY"]
    windows = {"4w": 20, "12w": 60, "26w": 130}
    weights = {"4w": 0.3, "12w": 0.4, "26w": 0.3}

    spy_returns = {}
    for label, days in windows.items():
        if len(spy) > days:
            spy_returns[label] = float(spy.iloc[-1] / spy.iloc[-days] - 1)
        else:
            spy_returns[label] = 0.0

    print(f"  SPY returns: 4w={spy_returns['4w']:+.2%}, 12w={spy_returns['12w']:+.2%}, 26w={spy_returns['26w']:+.2%}")

    sector_scores = []
    for name, info in sector_map.items():
        etf = info["etf"]
        if etf not in etf_close.columns:
            continue
        prices = etf_close[etf].dropna()
        if len(prices) < 130:
            continue

        rets = {}
        composite = 0.0
        for label, days in windows.items():
            if len(prices) > days:
                ret = float(prices.iloc[-1] / prices.iloc[-days] - 1)
            else:
                ret = 0.0
            rets[label] = ret
            relative = ret - spy_returns[label]
            composite += weights[label] * relative

        sector_scores.append({
            "sector": name,
            "etf": etf,
            "ret_4w": rets["4w"],
            "ret_12w": rets["12w"],
            "ret_26w": rets["26w"],
            "composite": composite,
        })

    sector_scores.sort(key=lambda x: x["composite"], reverse=True)

    print(f"\n  {'Sector':<16} {'ETF':<6} {'4w':>7} {'12w':>7} {'26w':>7} {'Score':>8}")
    print(f"  {'-'*16} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for s in sector_scores:
        marker = " <--" if s == sector_scores[0] else ""
        print(f"  {s['sector']:<16} {s['etf']:<6} {s['ret_4w']:+.2%} {s['ret_12w']:+.2%} {s['ret_26w']:+.2%} {s['composite']:+.4f}{marker}")

    # ============================================================
    # STEP 3 & 4: Volume + RSI for sector ETFs
    # ============================================================
    print("\n[STEP 3-4] Volume ratio & RSI for sector ETFs...")

    print(f"\n  {'Sector':<16} {'Price':>8} {'Vol Ratio':>10} {'RSI':>6} {'Signal':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*6} {'-'*8}")
    for s in sector_scores[:5]:
        etf = s["etf"]
        prices = etf_close[etf].dropna()
        vol = etf_volume[etf].dropna() if etf in etf_volume.columns else pd.Series([0])

        # Volume ratio
        vol_sma20 = vol.rolling(20).mean()
        vol_ratio = float(vol.iloc[-1] / vol_sma20.iloc[-1]) if vol_sma20.iloc[-1] > 0 else 1.0

        # RSI
        rsi_series = calc_rsi(prices)
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

        # Signal
        if s["composite"] > 0.03 and rsi < 70:
            signal = "BULLISH"
        elif s["composite"] < -0.03 or rsi > 70:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        s["vol_ratio"] = vol_ratio
        s["rsi"] = rsi
        s["signal"] = signal
        s["price"] = float(prices.iloc[-1])

        print(f"  {s['sector']:<16} ${s['price']:>7.2f} {vol_ratio:>9.2f}x {rsi:>5.1f} {signal:>8}")

    # ============================================================
    # STEP 5: Regime Detection
    # ============================================================
    print("\n[STEP 5] HMM Regime Detection...")

    from regime_detector import RegimeDetector
    detector = RegimeDetector()
    regime = detector.detect()
    adj = detector.get_adjustments(regime)

    print(f"  Regime:       {regime['regime_label']}")
    print(f"  Confidence:   {regime['confidence']:.0%}")
    print(f"  Volatility:   {regime['volatility']:.1%}")
    print(f"  Exposure:     {adj['exposure_scale']:.0%}")
    print(f"  Transition:   {regime['transition_probability']['to_high_volatility']:.2%} chance of regime switch")

    # ============================================================
    # STEP 6: Top 3 Sectors - Stock-Level Deep Dive
    # ============================================================
    print("\n[STEP 6] Stock-level analysis for top 3 sectors...")

    top3 = sector_scores[:3]
    all_stock_tickers = []
    for s in top3:
        all_stock_tickers.extend(sector_map[s["sector"]]["stocks"])
    all_stock_tickers = list(set(all_stock_tickers))

    stock_data = yf.download(all_stock_tickers, start=(end - timedelta(days=120)).strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_close = stock_data["Close"]
        stock_vol = stock_data["Volume"]
    else:
        stock_close = stock_data
        stock_vol = stock_data

    all_stocks = []
    for s in top3:
        sector_name = s["sector"]
        tickers = sector_map[sector_name]["stocks"]

        print(f"\n  --- {sector_name} (score: {s['composite']:+.4f}) ---")
        print(f"  {'Ticker':<7} {'Price':>8} {'Mom30d':>8} {'VolRatio':>9} {'RSI':>6} {'Score':>7}")

        for tic in tickers:
            if tic not in stock_close.columns:
                continue
            prices = stock_close[tic].dropna()
            vols = stock_vol[tic].dropna() if tic in stock_vol.columns else pd.Series([0])
            if len(prices) < 30:
                continue

            # 30-day momentum
            momentum = float(prices.iloc[-1] / prices.iloc[-30] - 1) if len(prices) >= 30 else 0
            # Volume ratio
            vol_sma = vols.rolling(20).mean()
            vol_ratio = float(vols.iloc[-1] / vol_sma.iloc[-1]) if len(vol_sma) > 20 and vol_sma.iloc[-1] > 0 else 1.0
            # RSI
            rsi_s = calc_rsi(prices)
            rsi = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50
            # Composite score
            mom_norm = min(max(momentum / 0.3, -1), 1)
            vol_norm = min(max((vol_ratio - 1) / 2, -0.5), 0.5)
            rsi_norm = 1 - abs(rsi - 50) / 50
            stock_score = 0.4 * mom_norm + 0.2 * vol_norm + 0.4 * rsi_norm

            stock_info = {
                "ticker": tic,
                "sector": sector_name,
                "sector_score": s["composite"],
                "price": float(prices.iloc[-1]),
                "momentum": momentum,
                "volume_ratio": vol_ratio,
                "rsi": rsi,
                "score": stock_score,
            }
            all_stocks.append(stock_info)
            print(f"  {tic:<7} ${stock_info['price']:>7.2f} {momentum:>+7.1%} {vol_ratio:>8.2f}x {rsi:>5.1f} {stock_score:>+6.3f}")

    # Sort all stocks by score
    all_stocks.sort(key=lambda x: x["score"], reverse=True)

    # ============================================================
    # STEP 7: FinBERT Sentiment for Top Stocks
    # ============================================================
    print("\n[STEP 7] FinBERT sentiment analysis for top candidates...")

    top_candidates = all_stocks[:8]

    try:
        from sentiment_finbert import FinBERTScorer
        finbert = FinBERTScorer()

        for stock in top_candidates:
            tic = stock["ticker"]
            try:
                ticker_obj = yf.Ticker(tic)
                news = ticker_obj.news or []
                headlines = [n.get("title", "") for n in news[:5] if n.get("title")]
                if headlines:
                    scores = finbert.score_batch(headlines)
                    stock["sentiment"] = float(np.mean(scores))
                    stock["sent_headlines"] = len(headlines)
                    best_idx = np.argmax(np.abs(scores))
                    stock["strongest_headline"] = headlines[best_idx][:60]
                    stock["strongest_score"] = scores[best_idx]
                else:
                    stock["sentiment"] = 0.0
                    stock["sent_headlines"] = 0
            except Exception:
                stock["sentiment"] = 0.0
                stock["sent_headlines"] = 0

        print(f"\n  {'Ticker':<7} {'Sentiment':>10} {'N':>3} {'Strongest Headline':<60}")
        for stock in top_candidates:
            sent = stock.get("sentiment", 0)
            n = stock.get("sent_headlines", 0)
            hl = stock.get("strongest_headline", "N/A")[:58]
            label = "POS" if sent > 0.3 else "NEG" if sent < -0.3 else "NEU"
            print(f"  {stock['ticker']:<7} {sent:>+8.3f} {label} {n:>2}  {hl}")

    except ImportError:
        print("  FinBERT not available, skipping sentiment")
        for stock in top_candidates:
            stock["sentiment"] = 0.0

    # ============================================================
    # STEP 8: Signal Aggregation -> Decision
    # ============================================================
    print("\n[STEP 8] Signal aggregation -> Trading decisions...")

    # Weights from auto_trader.py
    W = {"sector": 0.25, "momentum": 0.25, "volume": 0.10,
         "rsi": 0.10, "sentiment": 0.15, "regime": 0.15}

    regime_signal = -0.2 if regime["regime"] == 1 else 0.0
    portfolio_value = 100_000

    decisions = []
    print(f"\n  {'Ticker':<7} {'Sector':>6} {'Mom':>6} {'Vol':>6} {'RSI':>6} {'Sent':>6} {'Reg':>6} | {'Conf':>7} {'Action':<12} {'Size':>7}")
    print(f"  {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}   {'-'*7} {'-'*12} {'-'*7}")

    for stock in top_candidates:
        # Individual signals
        s_sector = min(stock["sector_score"] * 5, 1.0) if stock["sector_score"] > 0 else max(stock["sector_score"] * 3, -1.0)
        s_mom = min(stock["momentum"] * 3, 1.0) if stock["momentum"] > 0.1 else (stock["momentum"] * 2 if stock["momentum"] > 0 else max(stock["momentum"] * 3, -1.0))
        s_vol = 0.3 if stock["volume_ratio"] > 1.5 else (-0.2 if stock["volume_ratio"] < 0.5 else 0.0)
        s_rsi = -0.3 if stock["rsi"] > 70 else (0.1 if 40 <= stock["rsi"] <= 60 else 0.0)
        s_sent = stock.get("sentiment", 0) * 0.5 if abs(stock.get("sentiment", 0)) > 0.3 else 0.0
        s_reg = regime_signal

        # Weighted confidence
        confidence = (W["sector"] * s_sector + W["momentum"] * s_mom +
                      W["volume"] * s_vol + W["rsi"] * s_rsi +
                      W["sentiment"] * s_sent + W["regime"] * s_reg)
        confidence = max(-1, min(1, confidence))

        # Action
        if confidence >= 0.6:
            action = "STRONG_BUY"
        elif confidence >= 0.35:
            action = "BUY"
        elif confidence >= 0.15:
            action = "HOLD"
        elif confidence <= -0.4:
            action = "STRONG_SELL"
        elif confidence <= -0.2:
            action = "SELL"
        else:
            action = "HOLD"

        # Position size (quarter-Kelly)
        if confidence > 0:
            edge = confidence * 0.05
            size_pct = edge * 0.25 * adj["exposure_scale"]
            size_pct = min(size_pct, 0.15)
        else:
            size_pct = 0.0

        decisions.append({
            "ticker": stock["ticker"],
            "action": action,
            "confidence": confidence,
            "size_pct": size_pct,
            "size_usd": round(portfolio_value * size_pct, 0),
            "price": stock["price"],
            "signals": {"sector": s_sector, "mom": s_mom, "vol": s_vol,
                        "rsi": s_rsi, "sent": s_sent, "regime": s_reg},
        })

        print(f"  {stock['ticker']:<7} {s_sector:>+5.2f} {s_mom:>+5.2f} {s_vol:>+5.2f} {s_rsi:>+5.2f} {s_sent:>+5.2f} {s_reg:>+5.2f} | {confidence:>+6.3f} {action:<12} {size_pct:>6.1%}")

    # ============================================================
    # STEP 9: Transaction Cost Impact
    # ============================================================
    print("\n[STEP 9] Transaction cost impact analysis...")

    buy_decisions = [d for d in decisions if "BUY" in d["action"]]
    total_invested = sum(d["size_usd"] for d in buy_decisions)

    costs_scenarios = [0, 5, 10, 20]  # basis points
    print(f"\n  Invested: ${total_invested:,.0f} ({total_invested/portfolio_value:.1%} of portfolio)")
    print(f"  Trades: {len(buy_decisions)}")
    print(f"\n  {'Cost (bps)':>10} {'Round-trip':>12} {'Annual (12x)':>14} {'% of Alpha':>12}")
    for bps in costs_scenarios:
        # Round-trip cost (buy + sell)
        rt_cost = total_invested * (bps / 10000) * 2
        annual_cost = rt_cost * 12  # monthly rebalance
        alpha_annual = portfolio_value * 0.0089 * 12  # 0.89%/month excess
        pct_of_alpha = annual_cost / alpha_annual * 100 if alpha_annual > 0 else 0
        print(f"  {bps:>10} ${rt_cost:>11,.0f} ${annual_cost:>13,.0f} {pct_of_alpha:>11.1f}%")

    # ============================================================
    # STEP 10: Final Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Regime:          {regime['regime_label']} ({regime['confidence']:.0%} confidence)")
    print(f"  Top Sectors:     {', '.join(s['sector'] for s in top3)}")
    print(f"  Candidates:      {len(all_stocks)} stocks analyzed")
    print(f"  Decisions:       {sum(1 for d in decisions if 'BUY' in d['action'])} BUY, "
          f"{sum(1 for d in decisions if 'SELL' in d['action'])} SELL, "
          f"{sum(1 for d in decisions if d['action'] == 'HOLD')} HOLD")
    print(f"  Total Invested:  ${total_invested:,.0f} ({total_invested/portfolio_value:.1%})")
    print(f"  Cash Remaining:  ${portfolio_value - total_invested:,.0f} ({(portfolio_value-total_invested)/portfolio_value:.1%})")

    if buy_decisions:
        print(f"\n  --- BUY Decisions ---")
        for d in sorted(buy_decisions, key=lambda x: x["confidence"], reverse=True):
            qty = int(d["size_usd"] / d["price"]) if d["price"] > 0 else 0
            print(f"  {d['action']:<12} {d['ticker']:<6} conf={d['confidence']:+.3f} "
                  f"${d['size_usd']:>6,.0f} ({qty} shares @ ${d['price']:.2f})")

    # Honest assessment
    print(f"\n  --- Honest Assessment ---")
    max_conf = max(d["confidence"] for d in decisions) if decisions else 0
    if max_conf >= 0.6:
        print(f"  Signal strength: STRONG (max confidence {max_conf:+.3f})")
    elif max_conf >= 0.35:
        print(f"  Signal strength: MODERATE (max confidence {max_conf:+.3f})")
        print(f"  Warning: Moderate signals - position sizes kept conservative")
    else:
        print(f"  Signal strength: WEAK (max confidence {max_conf:+.3f})")
        print(f"  Warning: No strong conviction - consider waiting")

    cost_5bps = total_invested * (5/10000) * 2 * 12
    excess_annual = portfolio_value * 0.0089 * 12
    net_alpha = excess_annual - cost_5bps
    print(f"\n  Gross alpha (backtest): ${excess_annual:,.0f}/year")
    print(f"  Transaction costs @5bps: -${cost_5bps:,.0f}/year")
    print(f"  Net alpha estimate:      ${net_alpha:,.0f}/year ({net_alpha/portfolio_value:.1%})")


if __name__ == "__main__":
    main()
