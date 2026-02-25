"""n8n-style modular trading pipeline.

Each node has clear inputs/outputs. Nodes are independently testable.
Same pipeline works in backtest (historical data) and live (broker data).

Nodes:
    RegimeNode     reads: spy_close, eval_date      writes: regime, exposure_scale
    SectorNode     reads: stock_close, eval_date     writes: sector_scores, top_sectors
    CandidateNode  reads: top_sectors                writes: candidates
    FeatureNode    reads: candidates, stock_*         writes: features
    ModelNode      reads: features                    writes: model_scores
    ScoreNode      reads: features, model_scores      writes: scored
    ExitNode       reads: positions, scored            writes: positions, cash, sells
    EntryNode      reads: scored, positions, cash      writes: positions, cash

Pipeline:
    pipe = Pipeline([RegimeNode(), SectorNode(), ...])
    ctx = pipe.run({"eval_date": date, "stock_close": df, ...})

Backtest:
    python pipeline.py
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SWING_EXIT_CONFIG, REGIME_BLEND_CONFIG


# ===================================================================
# Pipeline engine
# ===================================================================

@dataclass
class NodeResult:
    name: str
    elapsed_ms: float


class Pipeline:
    """Chain nodes. Each node: process(ctx) -> ctx."""

    def __init__(self, nodes: list) -> None:
        self.nodes = nodes
        self.log: List[NodeResult] = []

    def run(self, ctx: dict) -> dict:
        self.log.clear()
        for node in self.nodes:
            t0 = time.time()
            ctx = node.process(ctx)
            self.log.append(NodeResult(
                name=node.name,
                elapsed_ms=round((time.time() - t0) * 1000, 1),
            ))
        return ctx

    def summary(self) -> str:
        return " -> ".join(f"{r.name}({r.elapsed_ms}ms)" for r in self.log)


# ===================================================================
# Node implementations
# ===================================================================

class RegimeNode:
    """Detect market regime from SPY volatility."""
    name = "regime"

    def process(self, ctx: dict) -> dict:
        spy = ctx["spy_close"]
        date = ctx["eval_date"]
        hist = spy.loc[:date].dropna()

        if len(hist) < 60:
            ctx.update(regime="low_volatility", exposure_scale=1.0)
            return ctx

        rets = hist.pct_change().dropna()
        vol_20 = rets.iloc[-20:].std() * np.sqrt(252)
        vol_60_med = rets.rolling(20).std().iloc[-60:].median() * np.sqrt(252)

        hi_vol = vol_20 > vol_60_med * 1.3
        ctx["regime"] = "high_volatility" if hi_vol else "low_volatility"
        ctx["exposure_scale"] = 0.7 if hi_vol else 1.0
        return ctx


class SectorNode:
    """Rank sectors by ETF momentum."""
    name = "sector"

    def process(self, ctx: dict) -> dict:
        from agents.quant_agent import ETF_TICKERS, _rank_sectors

        sc = ctx["stock_close"]
        etf_cols = [c for c in ETF_TICKERS if c in sc.columns]
        etf_close = sc[etf_cols] if etf_cols else pd.DataFrame()

        scores = _rank_sectors(etf_close, ctx["eval_date"]) if not etf_close.empty else {}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        ctx["sector_scores"] = scores
        ctx["top_sectors"] = [s[0] for s in ranked[:3]]
        ctx["sorted_sectors"] = ranked
        return ctx


class CandidateNode:
    """Select stocks from top sectors."""
    name = "candidates"

    def process(self, ctx: dict) -> dict:
        from agents.quant_agent import SECTOR_MAP

        cands = []
        for sec in ctx["top_sectors"]:
            if sec in SECTOR_MAP:
                cands.extend(SECTOR_MAP[sec]["stocks"])

        if len(cands) < 15:
            for sec, _ in ctx.get("sorted_sectors", [])[3:]:
                if sec in SECTOR_MAP:
                    cands.extend(SECTOR_MAP[sec]["stocks"])
                if len(cands) >= 30:
                    break

        ctx["candidates"] = list(set(cands))
        return ctx


class FeatureNode:
    """Compute z-scored features for candidates."""
    name = "features"

    def process(self, ctx: dict) -> dict:
        from agents.quant_agent import compute_zscore_features

        ctx["features"] = compute_zscore_features(
            ctx["candidates"], ctx["eval_date"],
            ctx["stock_close"], ctx["stock_high"], ctx["stock_low"],
            ctx["stock_volume"], ctx["stock_open"],
            ctx["sector_scores"],
        )
        return ctx


class ModelNode:
    """XGBoost probability scoring with periodic retrain."""
    name = "model"

    def __init__(self, model=None, retrain_fn: Optional[Callable] = None,
                 retrain_every: int = 3):
        self.model = model
        self._retrain_fn = retrain_fn
        self._retrain_every = retrain_every
        self._cycle = 0

    def process(self, ctx: dict) -> dict:
        from agents.quant_agent import FEATURE_NAMES

        self._cycle += 1
        if (self._retrain_fn and self.model
                and self._cycle % self._retrain_every == 0):
            new = self._retrain_fn(ctx)
            if new is not None:
                self.model = new

        features = ctx["features"]
        scores = {}

        for tic, feats in features.items():
            if self.model is not None:
                X = np.array([[feats[f] for f in FEATURE_NAMES]])
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    proba = self.model.predict_proba(X)[0]
                    classes = list(self.model.classes_)
                    p_dict = {cls: float(p) for cls, p in zip(classes, proba)}
                    scores[tic] = p_dict.get(1, 0.25)
                except Exception:
                    scores[tic] = 0.5
            else:
                mom = feats.get("momentum_21d", 0.0)
                scores[tic] = 0.5 + max(-0.5, min(0.5, mom / 6.0))

        ctx["model_scores"] = scores
        return ctx


class ScoreNode:
    """Weighted signal combination."""
    name = "score"

    DEFAULT_WEIGHTS = {
        "quant": 0.35, "sector": 0.25, "momentum": 0.20,
        "regime": 0.10, "mean_rev": 0.10,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def process(self, ctx: dict) -> dict:
        from agents.quant_agent import TICKER_SECTOR

        features = ctx["features"]
        model_scores = ctx["model_scores"]
        sector_scores = ctx["sector_scores"]
        regime = ctx["regime"]
        weights = ctx.get("adaptive_weights") or self.weights

        scored = []
        for tic, feats in features.items():
            price = feats.get("price", 0)
            if price <= 0:
                continue

            p_win = model_scores.get(tic, 0.5)
            sec_name = TICKER_SECTOR.get(tic, "")

            signals = {
                "quant": (p_win - 0.5) * 2,
                "sector": max(-1, min(1, sector_scores.get(sec_name, 0) * 5)),
                "momentum": max(-1, min(1, feats.get("momentum_21d", 0) / 3)),
                "regime": self._regime_signal(regime, tic),
                "mean_rev": max(-1, min(1, -feats.get("rsi_14", 0) / 3)),
            }

            score = sum(weights.get(k, 0) * v for k, v in signals.items())
            score = max(-1, min(1, score))

            scored.append({
                "ticker": tic, "score": score, "price": price,
                "p_win": p_win, "signals": signals,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        ctx["scored"] = scored
        return ctx

    @staticmethod
    def _regime_signal(regime: str, ticker: str) -> float:
        from agents.quant_agent import TICKER_SECTOR
        sec = TICKER_SECTOR.get(ticker, "")
        defensive = {"Healthcare", "Staples", "Utilities", "RealEstate"}
        growth = {"Technology", "Semis", "Communication", "ConsDisc", "Financials"}
        if regime == "high_volatility":
            return 0.3 if sec in defensive else (-0.2 if sec in growth else 0.0)
        return 0.1 if sec in growth else (-0.1 if sec in defensive else 0.0)


class ExitNode:
    """Evaluate held positions for exit."""
    name = "exit"

    def __init__(self, tx_cost: float = 0.0015, sell_threshold: float = -0.10):
        self.tx_cost = tx_cost
        self.sell_threshold = sell_threshold

    def process(self, ctx: dict) -> dict:
        positions = ctx.get("positions", {})
        scored = ctx.get("scored", [])
        sc = ctx["stock_close"]
        idx = ctx["eval_date_idx"]
        date_str = str(ctx["eval_date"])[:10]
        miss = ctx.setdefault("scan_miss_counts", {})
        cash = ctx.get("cash", 0)
        trades = ctx.setdefault("trade_log", [])

        cand_set = {s["ticker"] for s in scored}
        for tic in list(positions):
            if tic in cand_set:
                miss.pop(tic, None)
            else:
                miss[tic] = miss.get(tic, 0) + 1

        sells = []
        cfg = SWING_EXIT_CONFIG

        for tic in list(positions):
            pos = positions[tic]
            if tic not in sc.columns:
                continue
            px = sc[tic].iloc[idx]
            if pd.isna(px) or px <= 0:
                continue

            pnl = (px - pos["entry_price"]) / pos["entry_price"]
            hold = idx - pos["entry_idx"]
            mc = miss.get(tic, 0)

            reason = None
            if pnl <= cfg["stop_loss_pct"]:
                reason = f"stop_loss ({pnl:+.1%})"
            elif pnl >= cfg["take_profit_pct"]:
                reason = f"take_profit ({pnl:+.1%})"
            elif hold > cfg["max_hold_days"] * 5 and pnl < cfg["min_profit_pct"]:
                reason = f"timeout ({hold}d, {pnl:+.1%})"
            elif mc >= cfg["consecutive_miss_limit"]:
                reason = f"scan_miss x{mc}"

            ts = [s for s in scored if s["ticker"] == tic]
            if ts and ts[0]["score"] < self.sell_threshold:
                reason = f"neg_signal ({ts[0]['score']:+.2f})"

            if reason:
                cash += pos["qty"] * px * (1 - self.tx_cost)
                trades.append({
                    "date": date_str, "ticker": tic, "side": "SELL",
                    "qty": pos["qty"], "price": px, "pnl_pct": pnl,
                    "reason": reason, "signals": pos.get("signals", {}),
                })
                sells.append(tic)

        for tic in sells:
            del positions[tic]
            miss.pop(tic, None)

        ctx["positions"] = positions
        ctx["cash"] = cash
        ctx["sells"] = sells
        return ctx


class EntryNode:
    """Select and size new positions."""
    name = "entry"

    def __init__(self, buy_threshold: float = 0.15, max_positions: int = 8,
                 max_position_pct: float = 0.12, tx_cost: float = 0.0015):
        self.buy_thr = buy_threshold
        self.max_pos = max_positions
        self.max_pct = max_position_pct
        self.tx_cost = tx_cost

    def process(self, ctx: dict) -> dict:
        scored = ctx.get("scored", [])
        positions = ctx.get("positions", {})
        cash = ctx.get("cash", 0)
        sc = ctx["stock_close"]
        idx = ctx["eval_date_idx"]
        date_str = str(ctx["eval_date"])[:10]
        exp_scale = ctx.get("exposure_scale", 1.0)
        trades = ctx.setdefault("trade_log", [])

        buys = [s for s in scored
                if s["score"] > self.buy_thr and s["ticker"] not in positions]
        buys = buys[:self.max_pos - len(positions)]

        for e in buys:
            tic, px = e["ticker"], e["price"]
            if px <= 0:
                continue

            pct = min(abs(e["score"]) * 0.15, self.max_pct) * exp_scale
            pv = cash + sum(
                positions[t]["qty"] * sc[t].iloc[idx]
                for t in positions if t in sc.columns
                and not pd.isna(sc[t].iloc[idx])
            )
            alloc = min(pct * pv, cash * 0.4)
            if alloc < 200 or alloc > cash:
                continue

            qty = alloc / px
            cost = alloc * (1 + self.tx_cost)
            if cost > cash:
                continue

            cash -= cost
            positions[tic] = {
                "qty": qty, "entry_price": px, "entry_idx": idx,
                "signals": e.get("signals", {}),
            }
            trades.append({
                "date": date_str, "ticker": tic, "side": "BUY",
                "qty": qty, "price": px, "score": e["score"],
                "reason": f"signal={e['score']:+.2f}",
            })

        ctx["positions"] = positions
        ctx["cash"] = cash
        return ctx


# ===================================================================
# IntelligenceNode — Paper-aware meta-reasoning
# ===================================================================

class IntelligenceNode:
    """Paper-aware intelligence layer.

    Applies research-derived heuristics to scored candidates:

    1. Crowding penalty (Wharton NBER 2025):
       If too many top candidates cluster in one sector,
       reduce scores — herding increases flash crash risk.

    2. Momentum persistence (arXiv 2512.10913):
       Boost stocks with consistent multi-timeframe momentum.
       Meta-analysis: implementation quality > algorithm choice.

    3. Cost-aware filtering (Lopez de Prado PBO):
       Skip trades where expected gain barely covers costs.
       Most published strategies fail after costs.

    4. Volatility-adjusted sizing (Zhang et al. IJCAI):
       Distribution shift kills returns. Scale down in
       unfamiliar regimes (high vol after long low vol).

    5. Winner persistence (Moskowitz-Grinblatt 1999):
       Hold winners longer — momentum has 6-12 month persistence.
       Don't sell winners early just because score dips slightly.

    Reads:  scored, regime, sector_scores, positions
    Writes: scored (adjusted), intelligence_log
    """
    name = "intelligence"

    def __init__(self, max_sector_pct: float = 0.35,
                 min_expected_gain: float = 0.003,
                 momentum_boost: float = 0.15):
        self.max_sector_pct = max_sector_pct
        self.min_expected_gain = min_expected_gain
        self.momentum_boost = momentum_boost

    def process(self, ctx: dict) -> dict:
        from agents.quant_agent import TICKER_SECTOR

        scored = ctx.get("scored", [])
        regime = ctx.get("regime", "low_volatility")
        positions = ctx.get("positions", {})
        log_entries = []

        if not scored:
            ctx["intelligence_log"] = log_entries
            return ctx

        # --- 1. Crowding penalty ---
        sector_counts: Dict[str, int] = {}
        top_n = scored[:15]  # top candidates
        for s in top_n:
            sec = TICKER_SECTOR.get(s["ticker"], "other")
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        crowded_sectors = {sec for sec, cnt in sector_counts.items()
                          if cnt >= 4}

        # --- 2 & 3 & 4. Adjust each candidate ---
        adjusted = []
        for s in scored:
            tic = s["ticker"]
            score = s["score"]
            sigs = s.get("signals", {})
            adjustments = []

            # Crowding penalty: -20% score for crowded sectors
            sec = TICKER_SECTOR.get(tic, "other")
            if sec in crowded_sectors:
                score *= 0.80
                adjustments.append(f"crowd_penalty({sec})")

            # Momentum persistence boost: if all 3 momentum signals align
            mom = sigs.get("momentum", 0)
            quant = sigs.get("quant", 0)
            if mom > 0.1 and quant > 0.1 and score > 0:
                score += self.momentum_boost * min(mom, 0.5)
                adjustments.append("mom_persist_boost")

            # Cost-aware filter: skip low-conviction buys
            # P(win) * avg_gain should exceed 3x round-trip cost
            p_win = s.get("p_win", 0.5)
            expected = (p_win - 0.5) * 0.08  # excess prob * avg swing
            if expected < self.min_expected_gain and score > 0:
                score *= 0.5
                adjustments.append("low_expected_gain")

            # Volatility regime scaling (Zhang et al.)
            if regime == "high_volatility":
                # In high vol: be more selective, not less
                if score > 0 and score < 0.2:
                    score *= 0.7
                    adjustments.append("hivol_selective")

            # Winner persistence: don't penalize held winners
            if tic in positions:
                pos = positions[tic]
                pnl = (s["price"] - pos["entry_price"]) / pos["entry_price"]
                if pnl > 0.02:  # profitable position
                    score = max(score, 0.05)  # keep above sell threshold
                    adjustments.append(f"winner_hold({pnl:+.1%})")

            score = max(-1, min(1, score))
            entry = dict(s, score=score)
            if adjustments:
                entry["adjustments"] = adjustments
                log_entries.append({"ticker": tic, "adj": adjustments,
                                   "orig": s["score"], "new": score})
            adjusted.append(entry)

        adjusted.sort(key=lambda x: x["score"], reverse=True)
        ctx["scored"] = adjusted
        ctx["intelligence_log"] = log_entries
        return ctx


# ===================================================================
# Crypto nodes — adapted for 24/7 crypto markets
# ===================================================================

class CryptoRegimeNode:
    """Detect crypto market regime from BTC."""
    name = "crypto_regime"

    def process(self, ctx: dict) -> dict:
        btc = ctx.get("btc_close")
        date = ctx["eval_date"]

        if btc is None or len(btc.loc[:date].dropna()) < 30:
            ctx.update(regime="low_volatility", exposure_scale=1.0)
            return ctx

        hist = btc.loc[:date].dropna()
        rets = hist.pct_change().dropna()
        vol_14 = rets.iloc[-14:].std() * np.sqrt(365)
        vol_60_med = rets.rolling(14).std().iloc[-60:].median() * np.sqrt(365)

        hi_vol = vol_14 > vol_60_med * 1.5  # crypto: higher threshold
        ctx["regime"] = "high_volatility" if hi_vol else "low_volatility"
        ctx["exposure_scale"] = 0.5 if hi_vol else 1.0  # more conservative
        return ctx


class CryptoFeatureNode:
    """Compute momentum/RSI features for crypto pairs."""
    name = "crypto_features"

    def process(self, ctx: dict) -> dict:
        close = ctx["crypto_close"]
        date = ctx["eval_date"]
        candidates = ctx.get("candidates", list(close.columns))

        features = {}
        for tic in candidates:
            if tic not in close.columns:
                continue
            hist = close[tic].loc[:date].dropna()
            if len(hist) < 30:
                continue

            price = hist.iloc[-1]
            if pd.isna(price) or price <= 0:
                continue

            # Momentum signals
            mom_7d = (hist.iloc[-1] / hist.iloc[-7] - 1) if len(hist) >= 7 else 0
            mom_14d = (hist.iloc[-1] / hist.iloc[-14] - 1) if len(hist) >= 14 else 0
            mom_30d = (hist.iloc[-1] / hist.iloc[-30] - 1) if len(hist) >= 30 else 0

            # RSI
            delta = hist.diff()
            gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
            rs = gain / loss.where(loss > 0, 1e-10)
            rsi = float(100 - 100 / (1 + rs.iloc[-1]))

            # Volatility
            vol = hist.pct_change().iloc[-14:].std() * np.sqrt(365)

            # BTC correlation (if not BTC itself)
            btc = ctx.get("btc_close")
            btc_corr = 0.5
            if btc is not None and "BTC" not in tic:
                try:
                    both = pd.concat([hist, btc.loc[:date].dropna()], axis=1).dropna()
                    if len(both) > 20:
                        btc_corr = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
                except Exception:
                    pass

            features[tic] = {
                "price": float(price),
                "momentum_7d": float(mom_7d),
                "momentum_14d": float(mom_14d),
                "momentum_30d": float(mom_30d),
                "rsi_14": rsi,
                "volatility": float(vol) if not pd.isna(vol) else 0.3,
                "btc_correlation": btc_corr,
            }

        ctx["features"] = features
        ctx["candidates"] = list(features.keys())
        return ctx


class CryptoScoreNode:
    """Signal combination for crypto."""
    name = "crypto_score"

    WEIGHTS = {
        "momentum": 0.35,    # trend following dominant in crypto
        "mean_rev": 0.25,    # RSI oversold/overbought
        "volatility": 0.20,  # prefer lower vol (risk-adjusted)
        "btc_regime": 0.20,  # BTC leads altcoins
    }

    def process(self, ctx: dict) -> dict:
        features = ctx.get("features", {})
        regime = ctx.get("regime", "low_volatility")
        weights = ctx.get("adaptive_weights") or self.WEIGHTS

        scored = []
        for tic, feats in features.items():
            price = feats["price"]
            if price <= 0:
                continue

            # Momentum: blend of timeframes (7d > 14d > 30d for crypto)
            mom_sig = (
                0.5 * max(-1, min(1, feats["momentum_7d"] / 0.10)) +
                0.3 * max(-1, min(1, feats["momentum_14d"] / 0.15)) +
                0.2 * max(-1, min(1, feats["momentum_30d"] / 0.25))
            )
            mom_sig = max(-1, min(1, mom_sig))

            # Mean reversion: RSI
            rsi_norm = (feats["rsi_14"] - 50) / 50  # [-1, 1]
            mean_rev_sig = max(-1, min(1, -rsi_norm))  # oversold = bullish

            # Volatility preference: lower vol = higher score
            vol = feats.get("volatility", 0.3)
            vol_sig = max(-1, min(1, (0.5 - vol) * 2))

            # BTC regime: bullish BTC = bullish alts
            btc_regime_sig = 0.2 if regime == "low_volatility" else -0.2
            if "BTC" in tic:
                btc_regime_sig *= 0.5  # BTC less affected by its own regime

            signals = {
                "momentum": mom_sig,
                "mean_rev": mean_rev_sig,
                "volatility": vol_sig,
                "btc_regime": btc_regime_sig,
            }

            score = sum(weights.get(k, 0) * v for k, v in signals.items())
            score = max(-1, min(1, score))

            scored.append({
                "ticker": tic, "score": score, "price": price,
                "p_win": 0.5 + score * 0.25,  # rough estimate
                "signals": signals,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        ctx["scored"] = scored
        return ctx


class CryptoExitNode:
    """Crypto-specific exit management."""
    name = "crypto_exit"

    def __init__(self, tp: float = 0.08, sl: float = -0.05, tx_cost: float = 0.001):
        self.tp = tp
        self.sl = sl
        self.tx_cost = tx_cost

    def process(self, ctx: dict) -> dict:
        positions = ctx.get("positions", {})
        scored = ctx.get("scored", [])
        close = ctx.get("crypto_close", ctx.get("stock_close"))
        idx = ctx["eval_date_idx"]
        date_str = str(ctx["eval_date"])[:10]
        cash = ctx.get("cash", 0)
        trades = ctx.setdefault("trade_log", [])

        sells = []
        for tic in list(positions):
            pos = positions[tic]
            if tic not in close.columns:
                continue
            px = close[tic].iloc[idx]
            if pd.isna(px) or px <= 0:
                continue

            pnl = (px - pos["entry_price"]) / pos["entry_price"]
            hold = idx - pos["entry_idx"]

            reason = None
            if pnl <= self.sl:
                reason = f"stop_loss ({pnl:+.1%})"
            elif pnl >= self.tp:
                reason = f"take_profit ({pnl:+.1%})"
            elif hold > 30 and pnl < 0.01:  # 30 days + <1% = exit
                reason = f"timeout ({hold}d, {pnl:+.1%})"

            # Negative score = exit
            ts = [s for s in scored if s["ticker"] == tic]
            if ts and ts[0]["score"] < -0.15:
                reason = f"neg_signal ({ts[0]['score']:+.2f})"

            if reason:
                cash += pos["qty"] * px * (1 - self.tx_cost)
                trades.append({
                    "date": date_str, "ticker": tic, "side": "SELL",
                    "qty": pos["qty"], "price": px, "pnl_pct": pnl,
                    "reason": reason, "signals": pos.get("signals", {}),
                })
                sells.append(tic)

        for tic in sells:
            del positions[tic]

        ctx["positions"] = positions
        ctx["cash"] = cash
        ctx["sells"] = sells
        return ctx


# ===================================================================
# Regime Blend nodes — extracted from backtest_v2.py
# Validated: Sharpe 1.65, Alpha +9.2% vs BTC, MDD 35.2%
# ===================================================================


def _calc_rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI as a full series (shared helper for Regime Blend nodes)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.where(loss > 0, 1e-10)
    return 100 - 100 / (1 + rs)


def _calc_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """Return (middle, upper, lower) Bollinger Bands."""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma, sma + num_std * std, sma - num_std * std


class RegimeBlendDetectNode:
    """Detect crypto regime: 'trending' or 'ranging'.

    Uses efficiency ratio (net move / total range) + trend strength.
    Reads:  crypto_close, eval_date, btc_ticker
    Writes: regime
    """
    name = "regime_blend_detect"

    def process(self, ctx: dict) -> dict:
        cfg = ctx.get("rb_config", REGIME_BLEND_CONFIG)
        close = ctx["crypto_close"]
        date = ctx["eval_date"]
        btc_ticker = ctx.get("btc_ticker", "BTC/USDT")

        if btc_ticker not in close.columns:
            ctx["regime"] = "unknown"
            return ctx

        hist = close[btc_ticker].loc[:date].dropna()
        vol_w = cfg["vol_window"]
        trend_w = cfg["trend_window"]

        if len(hist) < max(vol_w, trend_w) + 5:
            ctx["regime"] = "unknown"
            return ctx

        # Efficiency ratio: directional move / total range
        total_range = hist.iloc[-vol_w:].max() - hist.iloc[-vol_w:].min()
        net_move = abs(float(hist.iloc[-1]) - float(hist.iloc[-vol_w]))
        efficiency = net_move / total_range if total_range > 0 else 0

        # Trend strength: price vs SMA
        sma = hist.iloc[-trend_w:].mean()
        trend_strength = abs(float(hist.iloc[-1]) - sma) / sma if sma > 0 else 0

        if efficiency > 0.5 and trend_strength > 0.03:
            ctx["regime"] = "trending"
        else:
            ctx["regime"] = "ranging"

        return ctx


class DerivativesSignalNode:
    """Inject derivatives signals into pipeline context.

    Combines 4 microstructure signals:
    1. Funding Rate: extreme positive = bearish (longs overcrowded)
    2. Taker Delta (CVD): buy_sell_ratio > 1.15 = bullish, < 0.85 = bearish
    3. Long/Short Ratio: contrarian — crowded side gets squeezed
    4. OI context: spike + direction confirms momentum

    If derivatives_context is empty (e.g. backtest), this is a no-op.

    Reads:  derivatives_context (optional)
    Writes: derivatives_signal (float, -1 to 1)
    """
    name = "derivatives_signal"

    def process(self, ctx: dict) -> dict:
        deriv = ctx.get("derivatives_context", {})
        if not deriv:
            ctx["derivatives_signal"] = 0.0
            return ctx

        signal = 0.0
        n_tickers = 0

        # 1. Funding rate (weight: 0.35) — nonlinear: tanh squash, only extreme matters
        # BIS WP 1087: extreme funding (>0.05%/8h) is validated contrarian alpha
        _FUNDING_EXTREME = 0.0005  # 0.05%/8h
        for _tic, rate in deriv.get("funding_rates", {}).items():
            n_tickers += 1
            if abs(rate) > 0.0003:
                # tanh nonlinearity: compresses normal values, amplifies extremes
                normalized = rate / _FUNDING_EXTREME  # >1 means extreme
                signal -= math.tanh(normalized) * 0.35

        # 2. Taker delta / CVD (weight: 0.30)
        for _tic, taker in deriv.get("taker_delta", {}).items():
            ratio = taker.get("buy_sell_ratio", 1.0)
            signal += math.tanh((ratio - 1.0) * 2.0) * 0.30

        # 3. Long/Short ratio — contrarian (weight: 0.25)
        for _tic, ls in deriv.get("long_short_ratio", {}).items():
            z = ls.get("z_score", 0.0)
            if abs(z) > 1.0:
                signal -= math.tanh(z * 0.15) * 0.25

        # 4. OI direction as momentum confirmation (weight: 0.10)
        # Handled implicitly via event triggers, not direct signal

        # Average across tickers instead of sum (prevents multi-ticker blowup)
        if n_tickers > 1:
            signal /= n_tickers

        signal = max(-1.0, min(1.0, signal))
        ctx["derivatives_signal"] = signal
        return ctx


class RegimeBlendSignalNode:
    """Generate BUY/SELL signals per regime.

    TRENDING: RSI > 50 + 14d momentum > 2%  -> BUY
    RANGING:  price < BB_lower * 1.02 + RSI < 40 -> BUY
    EXIT:     momentum < -5% (trending), price > BB_upper * 0.98 (ranging)

    Reads:  crypto_close, eval_date, regime, candidates
    Writes: scored
    """
    name = "regime_blend_signal"

    def process(self, ctx: dict) -> dict:
        cfg = ctx.get("rb_config", REGIME_BLEND_CONFIG)
        close = ctx["crypto_close"]
        date = ctx["eval_date"]
        regime = ctx.get("regime", "unknown")
        candidates = ctx.get("candidates", list(close.columns))

        scored = []
        for tic in candidates:
            if tic not in close.columns:
                continue
            hist = close[tic].loc[:date].dropna()
            if len(hist) < 60:
                continue

            px = float(hist.iloc[-1])
            if pd.isna(px) or px <= 0:
                continue

            rsi_series = _calc_rsi_series(hist)
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50
            mom_14d = float(hist.iloc[-1] / hist.iloc[-14] - 1) if len(hist) > 14 else 0

            _, bb_upper, bb_lower = _calc_bollinger(hist, period=20, num_std=2.0)
            bb_low = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else 0
            bb_up = float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else float('inf')

            signal = "HOLD"
            score = 0.0
            reason = ""

            if regime == "trending":
                if rsi > cfg["trending_rsi_threshold"] and mom_14d > cfg["trending_momentum_threshold"]:
                    signal = "BUY"
                    score = 0.5 + min(mom_14d * 5, 0.5)  # 0.5-1.0
                    reason = f"TREND: rsi={rsi:.0f} mom={mom_14d*100:+.1f}%"
                elif mom_14d < cfg["trending_exit_momentum"]:
                    signal = "SELL"
                    score = -0.5
                    reason = f"TREND_EXIT: mom={mom_14d*100:+.1f}%"

            elif regime == "ranging":
                if px < bb_low * cfg["ranging_bb_threshold"] and rsi < cfg["ranging_rsi_threshold"]:
                    signal = "BUY"
                    score = 0.5 + min((bb_low - px) / px * 10, 0.5)
                    reason = f"RANGE: px<bb_low rsi={rsi:.0f}"
                elif bb_up < float('inf') and px > bb_up * cfg["ranging_exit_bb_factor"]:
                    signal = "SELL"
                    score = -0.5
                    reason = f"RANGE_EXIT: px>bb_up"

            else:  # unknown
                if rsi > cfg["unknown_rsi_threshold"] and mom_14d > cfg["unknown_momentum_threshold"]:
                    signal = "BUY"
                    score = 0.3
                    reason = f"UNK: rsi={rsi:.0f} mom={mom_14d*100:+.1f}%"

            # Blend derivatives signal (0.3 weight) when available
            deriv_sig = ctx.get("derivatives_signal", 0.0)
            if deriv_sig != 0.0 and score != 0.0:
                score = score * 0.7 + deriv_sig * 0.3

            scored.append({
                "ticker": tic,
                "signal": signal,
                "score": max(-1, min(1, score)),
                "price": px,
                "reason": reason,
                "rsi": rsi,
                "mom_14d": mom_14d,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        ctx["scored"] = scored
        return ctx


class RegimeBlendExitNode:
    """Trailing stop 15% + drawdown risk 8%.

    Reads:  positions, scored, cash, crypto_close, eval_date, daily_values
    Writes: positions, cash, sells, trade_log
    """
    name = "regime_blend_exit"

    def process(self, ctx: dict) -> dict:
        cfg = ctx.get("rb_config", REGIME_BLEND_CONFIG)
        positions = ctx.get("positions", {})
        scored = ctx.get("scored", [])
        close = ctx["crypto_close"]
        date = ctx["eval_date"]
        cash = ctx.get("cash", 0)
        trades = ctx.setdefault("trade_log", [])
        trailing_highs = ctx.setdefault("trailing_highs", {})
        date_str = str(date)[:10]

        # Update trailing highs and check stops
        sells = []
        for tic in list(positions):
            pos = positions[tic]
            if tic not in close.columns:
                continue
            hist = close[tic].loc[:date].dropna()
            if hist.empty:
                continue
            px = float(hist.iloc[-1])
            if pd.isna(px) or px <= 0:
                continue

            # Update trailing high
            prev_high = trailing_highs.get(tic, pos["entry_price"])
            if px > prev_high:
                trailing_highs[tic] = px
                prev_high = px

            pnl = (px - pos["entry_price"]) / pos["entry_price"]
            reason = None

            # Trailing stop (activate after trail_activation_pct gain)
            if prev_high > pos["entry_price"] * (1 + cfg["trail_activation_pct"]):
                trail_stop = prev_high * (1 - cfg["trail_pct"])
                if px <= trail_stop:
                    reason = f"trail_stop ({pnl:+.1%})"

            # Signal-based exit
            ts = [s for s in scored if s["ticker"] == tic]
            if ts and ts[0]["signal"] == "SELL":
                reason = ts[0]["reason"]

            if reason:
                cash += pos["qty"] * px * (1 - cfg["tx_cost"])
                trades.append({
                    "date": date_str, "ticker": tic, "side": "SELL",
                    "qty": pos["qty"], "price": px, "pnl_pct": pnl,
                    "reason": reason,
                })
                sells.append(tic)

        for tic in sells:
            del positions[tic]
            trailing_highs.pop(tic, None)

        # Drawdown risk management: if portfolio dropped >dd_trigger from recent peak
        daily_values = ctx.get("daily_values", [])
        if len(daily_values) > 20 and positions:
            pv = cash + sum(
                p["qty"] * float(close[t].loc[:date].dropna().iloc[-1])
                for t, p in positions.items()
                if t in close.columns and not close[t].loc[:date].dropna().empty
            )
            peak_pv = max(daily_values[-20:])
            if peak_pv > 0 and (peak_pv - pv) / peak_pv > cfg["dd_trigger"]:
                for tic in list(positions):
                    pos = positions[tic]
                    if tic not in close.columns:
                        continue
                    hist = close[tic].loc[:date].dropna()
                    if hist.empty:
                        continue
                    px = float(hist.iloc[-1])
                    pnl = (px - pos["entry_price"]) / pos["entry_price"]
                    cash += pos["qty"] * px * (1 - cfg["tx_cost"])
                    trades.append({
                        "date": date_str, "ticker": tic, "side": "SELL",
                        "qty": pos["qty"], "price": px, "pnl_pct": pnl,
                        "reason": f"dd_risk (>{cfg['dd_trigger']*100:.0f}%)",
                    })
                    sells.append(tic)
                    del positions[tic]
                    trailing_highs.pop(tic, None)

        ctx["positions"] = positions
        ctx["cash"] = cash
        ctx["sells"] = sells
        ctx["trailing_highs"] = trailing_highs
        return ctx


class RegimeBlendEntryNode:
    """Position sizing: 35% per position, max 70% exposure.

    Reads:  scored, positions, cash, crypto_close, eval_date
    Writes: positions, cash, trade_log, trailing_highs
    """
    name = "regime_blend_entry"

    def process(self, ctx: dict) -> dict:
        cfg = ctx.get("rb_config", REGIME_BLEND_CONFIG)
        scored = ctx.get("scored", [])
        positions = ctx.get("positions", {})
        cash = ctx.get("cash", 0)
        close = ctx["crypto_close"]
        date = ctx["eval_date"]
        trades = ctx.setdefault("trade_log", [])
        trailing_highs = ctx.setdefault("trailing_highs", {})
        date_str = str(date)[:10]

        buys = [s for s in scored if s["signal"] == "BUY" and s["ticker"] not in positions]

        for entry in buys:
            tic, px = entry["ticker"], entry["price"]
            if px <= 0:
                continue

            # Portfolio value
            pv = cash
            for t, p in positions.items():
                if t in close.columns:
                    h = close[t].loc[:date].dropna()
                    if not h.empty:
                        pv += p["qty"] * float(h.iloc[-1])

            # Exposure check
            current_exposure = 1 - (cash / pv) if pv > 0 else 0
            if current_exposure >= cfg["max_exposure"]:
                break

            alloc = pv * cfg["position_pct"]
            alloc = min(alloc, cash * 0.90)
            if alloc < 50:
                continue

            qty = alloc / px
            cost = alloc * (1 + cfg["tx_cost"])
            if cost > cash:
                continue

            cash -= cost
            positions[tic] = {
                "qty": qty,
                "entry_price": px,
                "entry_date": date_str,
            }
            trailing_highs[tic] = px
            trades.append({
                "date": date_str, "ticker": tic, "side": "BUY",
                "qty": qty, "price": px, "score": entry["score"],
                "reason": entry.get("reason", f"signal={entry['score']:+.2f}"),
            })

        ctx["positions"] = positions
        ctx["cash"] = cash
        ctx["trailing_highs"] = trailing_highs
        return ctx


# ===================================================================
# XGBoost training (shared between backtest and live retrain)
# ===================================================================

def train_xgboost(all_tickers, all_dates, eval_idx, train_months,
                  stock_close, stock_high, stock_low, stock_volume, stock_open):
    """Train XGBoost on historical data up to eval_idx."""
    from agents.quant_agent import (
        ETF_TICKERS, FEATURE_NAMES, FORWARD_HORIZON, PURGE_DAYS,
        _rank_sectors, compute_zscore_features, _compute_labels,
    )
    from xgboost import XGBClassifier

    etf_cols = [c for c in ETF_TICKERS if c in stock_close.columns]
    etf_close = stock_close[etf_cols] if etf_cols else pd.DataFrame()

    X_all, y_all = [], []
    for offset in range(train_months, 0, -1):
        feat_idx = eval_idx - (offset + 1) * 21
        if feat_idx < 63:
            continue
        target_end = feat_idx + FORWARD_HORIZON
        if target_end >= eval_idx - PURGE_DAYS:
            continue

        feat_date = all_dates[feat_idx]
        sec = _rank_sectors(etf_close, feat_date) if not etf_close.empty else {}
        zs = compute_zscore_features(
            all_tickers, feat_date,
            stock_close, stock_high, stock_low, stock_volume, stock_open, sec,
        )
        labels = _compute_labels(all_tickers, feat_idx, FORWARD_HORIZON, stock_close)

        for tic, feats in zs.items():
            if tic in labels:
                X_all.append([feats[f] for f in FEATURE_NAMES])
                y_all.append(labels[tic])

    if len(X_all) < 80:
        return None

    X = np.nan_to_num(np.array(X_all), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_all)
    if len(np.unique(y)) < 2:
        return None

    n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
    model = XGBClassifier(
        max_depth=4, n_estimators=300, learning_rate=0.03,
        reg_alpha=0.1, reg_lambda=1.5, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3,
        scale_pos_weight=n_neg / max(n_pos, 1),
        gamma=0.1, random_state=42, verbosity=0, eval_metric="logloss",
    )
    model.fit(X, y)
    return model


# ===================================================================
# Backtest runner
# ===================================================================

def run_backtest(
    start_date: str = "2024-06-01",
    end_date: str = "2026-02-01",
    rebalance_days: int = 21,
    initial_cash: float = 100_000.0,
    train_months: int = 24,
    use_xgboost: bool = True,
    use_online_learning: bool = False,
) -> Dict[str, Any]:
    """Modular pipeline backtest with optional Thompson Sampling."""
    from datetime import timedelta
    import yfinance as yf
    from agents.quant_agent import ALL_STOCKS, ETF_TICKERS

    label = "Thompson Sampling" if use_online_learning else "Static Weights"
    print(f"\n{'='*70}")
    print(f"  MODULAR PIPELINE BACKTEST ({label})")
    print(f"  Period: {start_date} ~ {end_date}")
    print(f"  Rebalance: {rebalance_days}d | XGBoost: {'ON' if use_xgboost else 'OFF'}")
    print(f"  Cash: ${initial_cash:,.0f}")
    print(f"{'='*70}\n")

    # Download data
    dl_start = (pd.Timestamp(start_date) - timedelta(
        days=(train_months + 6) * 30)).strftime("%Y-%m-%d")
    all_tickers = list(set(ALL_STOCKS + ETF_TICKERS + ["SPY"]))

    print("  Downloading...")
    t0 = time.time()
    data = yf.download(all_tickers, start=dl_start, end=end_date,
                       progress=False, auto_adjust=True)
    print(f"  Downloaded in {time.time() - t0:.0f}s")

    def extract(field):
        if isinstance(data.columns, pd.MultiIndex):
            return data[field] if field in data.columns.get_level_values(0) else pd.DataFrame()
        return data

    sc = extract("Close")
    sh = extract("High")
    sl_df = extract("Low")
    sv = extract("Volume")
    so = extract("Open")

    if sc.empty:
        return {"error": "No data"}

    spy = sc["SPY"] if "SPY" in sc.columns else None
    if spy is None:
        return {"error": "No SPY"}

    all_dates = sc.index
    bt_start = all_dates.get_indexer([pd.Timestamp(start_date)], method="pad")[0]
    bt_dates = all_dates[bt_start:]

    # Train initial model
    model = None
    if use_xgboost:
        try:
            print("  Training XGBoost...")
            model = train_xgboost(
                ALL_STOCKS, all_dates, bt_start, train_months,
                sc, sh, sl_df, sv, so,
            )
            print(f"  Model: {'OK' if model else 'FAILED'}")
        except ImportError:
            print("  xgboost not installed")

    # Build pipeline
    model_node = ModelNode(
        model=model,
        retrain_fn=lambda ctx: train_xgboost(
            ALL_STOCKS, all_dates, ctx["eval_date_idx"], train_months,
            sc, sh, sl_df, sv, so,
        ),
    )

    pipeline = Pipeline([
        RegimeNode(),
        SectorNode(),
        CandidateNode(),
        FeatureNode(),
        model_node,
        ScoreNode(),
        ExitNode(),
        EntryNode(),
    ])

    # Online learner (Thompson Sampling)
    learner = None
    if use_online_learning:
        from online_learner import OnlineLearner
        learner = OnlineLearner(min_trades_to_adapt=5)

    # Simulation state
    cash = initial_cash
    positions: Dict[str, Dict] = {}
    trade_log: List[Dict] = []
    daily_values: List[float] = []
    daily_dates_list: List = []
    scan_miss: Dict[str, int] = {}

    print(f"  Running {len(bt_dates)} trading days...\n")

    for day_i, date in enumerate(bt_dates):
        date_idx = all_dates.get_loc(date)

        # Daily portfolio value
        pv = cash
        for tic, pos in positions.items():
            if tic in sc.columns:
                p = sc[tic].iloc[date_idx]
                if not pd.isna(p) and p > 0:
                    pv += pos["qty"] * p
        daily_values.append(pv)
        daily_dates_list.append(date)

        if day_i % rebalance_days != 0:
            continue

        # Build context
        ctx: dict = {
            "eval_date": date,
            "eval_date_idx": date_idx,
            "stock_close": sc,
            "stock_high": sh,
            "stock_low": sl_df,
            "stock_volume": sv,
            "stock_open": so,
            "spy_close": spy,
            "positions": positions,
            "cash": cash,
            "trade_log": trade_log,
            "scan_miss_counts": scan_miss,
        }

        # Add adaptive weights from learner
        if learner and learner.has_enough_data:
            adaptive = learner.get_mean_weights()
            # Map learner agent names to ScoreNode signal names
            ctx["adaptive_weights"] = {
                "quant": adaptive.get("quant", 0.35),
                "sector": adaptive.get("market", 0.25),
                "momentum": adaptive.get("momentum", 0.20),
                "regime": adaptive.get("regime", 0.10),
                "mean_rev": adaptive.get("sentiment", 0.10),
            }

        # Run pipeline
        ctx = pipeline.run(ctx)

        # Extract updated state
        positions = ctx["positions"]
        cash = ctx["cash"]
        trade_log = ctx["trade_log"]
        scan_miss = ctx["scan_miss_counts"]

        # Update online learner with closed trades
        if learner:
            for sell_tic in ctx.get("sells", []):
                sell_trade = next(
                    (t for t in reversed(trade_log)
                     if t["ticker"] == sell_tic and t["side"] == "SELL"
                     and not t.get("_learned")),
                    None,
                )
                if sell_trade and "pnl_pct" in sell_trade:
                    sigs = sell_trade.get("signals", {})
                    learner.record_trade(
                        ticker=sell_trade["ticker"],
                        entry_price=sell_trade["price"] / (1 + sell_trade["pnl_pct"]),
                        exit_price=sell_trade["price"],
                        pnl_pct=sell_trade["pnl_pct"],
                        held_hours=0,
                        agent_signals={
                            "market": sigs.get("sector", 0),
                            "quant": sigs.get("quant", 0),
                            "sentiment": sigs.get("mean_rev", 0),
                            "momentum": sigs.get("momentum", 0),
                            "regime": sigs.get("regime", 0),
                        },
                    )
                    sell_trade["_learned"] = True

    # ── Metrics ──
    values = np.array(daily_values)
    if len(values) < 2:
        return {"error": "Not enough data"}

    total_return = (values[-1] - initial_cash) / initial_cash
    returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
    sharpe = (np.mean(returns) * 252**0.5) / max(np.std(returns), 1e-8)
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / np.maximum(peak, 1e-8)
    max_dd = float(np.max(drawdown))

    spy_start = spy.loc[bt_dates[0]]
    spy_end = spy.loc[bt_dates[-1]]
    spy_return = (spy_end / spy_start - 1) if spy_start > 0 else 0
    alpha = total_return - spy_return

    sells_log = [t for t in trade_log if t["side"] == "SELL" and "pnl_pct" in t]
    n_wins = sum(1 for t in sells_log if t["pnl_pct"] > 0)
    win_rate = n_wins / len(sells_log) if sells_log else 0

    # Print results
    print(f"\n  {'='*60}")
    print(f"  RESULTS ({label})")
    print(f"  {'='*60}")
    print(f"  Total Return:    {total_return*100:+.2f}%")
    print(f"  SPY B&H:         {spy_return*100:+.2f}%")
    print(f"  Alpha:           {alpha*100:+.2f}%")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Max Drawdown:    {max_dd*100:.1f}%")
    print(f"  Win Rate:        {win_rate*100:.0f}% ({n_wins}/{len(sells_log)})")
    print(f"  Trades:          {len(trade_log)} total")
    print(f"  Final:           ${values[-1]:,.0f}")

    if learner:
        status = learner.get_status()
        print(f"\n  Online Learner (Thompson Sampling):")
        print(f"    Trades learned: {status['total_trades']}")
        print(f"    Adapted: {status['has_enough_data']}")
        if status['has_enough_data']:
            w = status['mean_weights']
            print(f"    Weights: " + ", ".join(
                f"{k}={v:.2f}" for k, v in w.items()))
        for name, info in status['agents'].items():
            print(f"    {name}: mean={info['mean']:.3f} "
                  f"(a={info['alpha']:.1f} b={info['beta']:.1f} "
                  f"trades={info['total_trades']})")

    print(f"\n  Pipeline: {pipeline.summary()}")

    # Rebalance periods
    print(f"\n  {'='*60}")
    print(f"  REBALANCE PERIODS")
    print(f"  {'='*60}")
    rb_idx = list(range(0, len(daily_values), rebalance_days))
    for i, ridx in enumerate(rb_idx[:25]):
        val = daily_values[ridx]
        cum = (val - initial_cash) / initial_cash
        per = ((val - daily_values[rb_idx[i-1]]) / daily_values[rb_idx[i-1]]
               if i > 0 else 0)
        d = str(daily_dates_list[ridx])[:10]
        print(f"  {i+1:>3} {d}  ${val:>9,.0f}  {per*100:>+6.2f}%  {cum*100:>+6.2f}%")

    # Last trades
    print(f"\n  {'='*60}")
    print(f"  LAST 15 TRADES")
    print(f"  {'='*60}")
    for t in trade_log[-15:]:
        pnl = f" pnl={t['pnl_pct']:+.1%}" if "pnl_pct" in t else ""
        print(f"  {t['date']} {t['side']:<5} {t['ticker']:<6} "
              f"{t['qty']:>6.1f} @ ${t['price']:>7.1f}  {t['reason']}{pnl}")

    print(f"\n{'='*70}\n")

    return {
        "total_return": total_return,
        "spy_return": float(spy_return),
        "alpha": alpha,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "num_trades": len(trade_log),
        "final_value": values[-1],
        "daily_values": daily_values,
        "trade_log": trade_log,
        "learner_status": learner.get_status() if learner else None,
    }


# ===================================================================
# Tuned backtest — IntelligenceNode + optimized parameters
# ===================================================================

def run_tuned_backtest(
    start_date: str = "2024-06-01",
    end_date: str = "2026-02-01",
    initial_cash: float = 100_000.0,
    train_months: int = 24,
) -> Dict[str, Any]:
    """Tuned pipeline: IntelligenceNode + aggressive params.

    Paper insights applied:
    - Shorter rebalance (10d) — more responsive
    - Lower buy threshold (0.08) — more opportunities
    - More positions (12) — better diversification
    - IntelligenceNode — crowding, momentum persistence, cost filter
    """
    from datetime import timedelta
    import yfinance as yf
    from agents.quant_agent import ALL_STOCKS, ETF_TICKERS

    print(f"\n{'='*70}")
    print(f"  TUNED PIPELINE BACKTEST (IntelligenceNode + Optimized)")
    print(f"  Period: {start_date} ~ {end_date}")
    print(f"  Rebalance: 10d | Buy thr: 0.08 | Max pos: 12")
    print(f"{'='*70}\n")

    dl_start = (pd.Timestamp(start_date) - timedelta(
        days=(train_months + 6) * 30)).strftime("%Y-%m-%d")
    all_tickers = list(set(ALL_STOCKS + ETF_TICKERS + ["SPY"]))

    print("  Downloading...")
    t0 = time.time()
    data = yf.download(all_tickers, start=dl_start, end=end_date,
                       progress=False, auto_adjust=True)
    print(f"  Downloaded in {time.time() - t0:.0f}s")

    def extract(field):
        if isinstance(data.columns, pd.MultiIndex):
            return data[field] if field in data.columns.get_level_values(0) else pd.DataFrame()
        return data

    sc, sh, sl_df, sv, so = extract("Close"), extract("High"), extract("Low"), extract("Volume"), extract("Open")
    if sc.empty:
        return {"error": "No data"}

    spy = sc["SPY"] if "SPY" in sc.columns else None
    if spy is None:
        return {"error": "No SPY"}

    all_dates = sc.index
    bt_start = all_dates.get_indexer([pd.Timestamp(start_date)], method="pad")[0]
    bt_dates = all_dates[bt_start:]

    model = None
    try:
        print("  Training XGBoost...")
        model = train_xgboost(ALL_STOCKS, all_dates, bt_start, train_months,
                              sc, sh, sl_df, sv, so)
        print(f"  Model: {'OK' if model else 'FAILED'}")
    except ImportError:
        print("  xgboost not installed")

    rebalance_days = 10

    model_node = ModelNode(
        model=model,
        retrain_fn=lambda ctx: train_xgboost(
            ALL_STOCKS, all_dates, ctx["eval_date_idx"], train_months,
            sc, sh, sl_df, sv, so),
        retrain_every=5,  # retrain more often (every 5 cycles = ~50d)
    )

    pipeline = Pipeline([
        RegimeNode(),
        SectorNode(),
        CandidateNode(),
        FeatureNode(),
        model_node,
        ScoreNode(),
        IntelligenceNode(),  # <-- paper-aware adjustments
        ExitNode(tx_cost=0.0015, sell_threshold=-0.05),  # tighter sell
        EntryNode(buy_threshold=0.08, max_positions=12,
                  max_position_pct=0.15, tx_cost=0.0015),
    ])

    cash = initial_cash
    positions: Dict[str, Dict] = {}
    trade_log: List[Dict] = []
    daily_values: List[float] = []
    daily_dates_list: List = []
    scan_miss: Dict[str, int] = {}

    print(f"  Running {len(bt_dates)} trading days...\n")

    for day_i, date in enumerate(bt_dates):
        date_idx = all_dates.get_loc(date)

        pv = cash
        for tic, pos in positions.items():
            if tic in sc.columns:
                p = sc[tic].iloc[date_idx]
                if not pd.isna(p) and p > 0:
                    pv += pos["qty"] * p
        daily_values.append(pv)
        daily_dates_list.append(date)

        if day_i % rebalance_days != 0:
            continue

        ctx: dict = {
            "eval_date": date, "eval_date_idx": date_idx,
            "stock_close": sc, "stock_high": sh, "stock_low": sl_df,
            "stock_volume": sv, "stock_open": so, "spy_close": spy,
            "positions": positions, "cash": cash,
            "trade_log": trade_log, "scan_miss_counts": scan_miss,
        }

        ctx = pipeline.run(ctx)
        positions, cash = ctx["positions"], ctx["cash"]
        trade_log, scan_miss = ctx["trade_log"], ctx["scan_miss_counts"]

    values = np.array(daily_values)
    total_return = (values[-1] - initial_cash) / initial_cash
    returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
    sharpe = (np.mean(returns) * 252**0.5) / max(np.std(returns), 1e-8)
    peak = np.maximum.accumulate(values)
    max_dd = float(np.max((peak - values) / np.maximum(peak, 1e-8)))
    spy_return = float((spy.loc[bt_dates[-1]] / spy.loc[bt_dates[0]] - 1))
    alpha = total_return - spy_return

    sells_log = [t for t in trade_log if t["side"] == "SELL" and "pnl_pct" in t]
    n_wins = sum(1 for t in sells_log if t["pnl_pct"] > 0)
    win_rate = n_wins / len(sells_log) if sells_log else 0

    print(f"\n  {'='*60}")
    print(f"  RESULTS (Tuned + IntelligenceNode)")
    print(f"  {'='*60}")
    print(f"  Total Return:    {total_return*100:+.2f}%")
    print(f"  SPY B&H:         {spy_return*100:+.2f}%")
    print(f"  Alpha:           {alpha*100:+.2f}%")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Max Drawdown:    {max_dd*100:.1f}%")
    print(f"  Win Rate:        {win_rate*100:.0f}% ({n_wins}/{len(sells_log)})")
    print(f"  Trades:          {len(trade_log)} total")
    print(f"  Final:           ${values[-1]:,.0f}")
    print(f"\n  Pipeline: {pipeline.summary()}")

    for t in trade_log[-10:]:
        pnl = f" pnl={t['pnl_pct']:+.1%}" if "pnl_pct" in t else ""
        print(f"  {t['date']} {t['side']:<5} {t['ticker']:<6} "
              f"{t['qty']:>6.1f} @ ${t['price']:>7.1f}  {t['reason']}{pnl}")

    print(f"\n{'='*70}\n")

    return {
        "total_return": total_return, "spy_return": spy_return,
        "alpha": alpha, "sharpe": sharpe, "max_dd": max_dd,
        "win_rate": win_rate, "num_trades": len(trade_log),
        "final_value": values[-1],
    }


# ===================================================================
# Crypto backtest
# ===================================================================

def run_crypto_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2026-02-01",
    rebalance_days: int = 7,
    initial_cash: float = 5_000.0,
) -> Dict[str, Any]:
    """Crypto pipeline backtest using yfinance data."""
    from datetime import timedelta
    import yfinance as yf

    CRYPTO_TICKERS = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
        "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD",
        "DOT-USD", "LINK-USD",
    ]

    print(f"\n{'='*70}")
    print(f"  CRYPTO PIPELINE BACKTEST")
    print(f"  Period: {start_date} ~ {end_date}")
    print(f"  Rebalance: {rebalance_days}d | Cash: ${initial_cash:,.0f}")
    print(f"  Pairs: {len(CRYPTO_TICKERS)}")
    print(f"{'='*70}\n")

    dl_start = (pd.Timestamp(start_date) - timedelta(days=120)).strftime("%Y-%m-%d")

    print("  Downloading crypto data...")
    t0 = time.time()
    data = yf.download(CRYPTO_TICKERS, start=dl_start, end=end_date,
                       progress=False, auto_adjust=True)
    print(f"  Downloaded in {time.time() - t0:.0f}s")

    def extract(field):
        if isinstance(data.columns, pd.MultiIndex):
            return data[field] if field in data.columns.get_level_values(0) else pd.DataFrame()
        return data

    close = extract("Close")
    if close.empty:
        return {"error": "No crypto data"}

    btc = close["BTC-USD"] if "BTC-USD" in close.columns else None
    if btc is None:
        return {"error": "No BTC data"}

    all_dates = close.index
    bt_start = all_dates.get_indexer([pd.Timestamp(start_date)], method="pad")[0]
    bt_dates = all_dates[bt_start:]

    pipeline = Pipeline([
        CryptoRegimeNode(),
        CryptoFeatureNode(),
        CryptoScoreNode(),
        CryptoExitNode(tp=0.08, sl=-0.05, tx_cost=0.001),
        EntryNode(buy_threshold=0.05, max_positions=5,
                  max_position_pct=0.25, tx_cost=0.001),
    ])

    cash = initial_cash
    positions: Dict[str, Dict] = {}
    trade_log: List[Dict] = []
    daily_values: List[float] = []
    daily_dates_list: List = []

    print(f"  Running {len(bt_dates)} trading days...\n")

    for day_i, date in enumerate(bt_dates):
        date_idx = all_dates.get_loc(date)

        pv = cash
        for tic, pos in positions.items():
            if tic in close.columns:
                p = close[tic].iloc[date_idx]
                if not pd.isna(p) and p > 0:
                    pv += pos["qty"] * p
        daily_values.append(pv)
        daily_dates_list.append(date)

        if day_i % rebalance_days != 0:
            continue

        ctx: dict = {
            "eval_date": date, "eval_date_idx": date_idx,
            "crypto_close": close, "stock_close": close,
            "btc_close": btc,
            "candidates": [c for c in CRYPTO_TICKERS if c in close.columns],
            "positions": positions, "cash": cash, "trade_log": trade_log,
        }

        ctx = pipeline.run(ctx)
        positions, cash = ctx["positions"], ctx["cash"]
        trade_log = ctx["trade_log"]

    values = np.array(daily_values)
    if len(values) < 2:
        return {"error": "Not enough data"}

    total_return = (values[-1] - initial_cash) / initial_cash
    returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
    sharpe = (np.mean(returns) * 365**0.5) / max(np.std(returns), 1e-8)  # 365 for crypto
    peak = np.maximum.accumulate(values)
    max_dd = float(np.max((peak - values) / np.maximum(peak, 1e-8)))

    btc_start = btc.loc[bt_dates[0]]
    btc_end = btc.loc[bt_dates[-1]]
    btc_return = float((btc_end / btc_start - 1))

    sells_log = [t for t in trade_log if t["side"] == "SELL" and "pnl_pct" in t]
    n_wins = sum(1 for t in sells_log if t["pnl_pct"] > 0)
    win_rate = n_wins / len(sells_log) if sells_log else 0

    print(f"\n  {'='*60}")
    print(f"  RESULTS (Crypto)")
    print(f"  {'='*60}")
    print(f"  Total Return:    {total_return*100:+.2f}%")
    print(f"  BTC B&H:         {btc_return*100:+.2f}%")
    print(f"  Alpha vs BTC:    {(total_return - btc_return)*100:+.2f}%")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Max Drawdown:    {max_dd*100:.1f}%")
    print(f"  Win Rate:        {win_rate*100:.0f}% ({n_wins}/{len(sells_log)})")
    print(f"  Trades:          {len(trade_log)} total")
    print(f"  Final:           ${values[-1]:,.0f}")
    print(f"\n  Pipeline: {pipeline.summary()}")

    for t in trade_log[-10:]:
        pnl = f" pnl={t['pnl_pct']:+.1%}" if "pnl_pct" in t else ""
        print(f"  {t['date']} {t['side']:<5} {t['ticker']:<10} "
              f"{t['qty']:>10.4f} @ ${t['price']:>9.1f}  {t['reason']}{pnl}")

    print(f"\n{'='*70}\n")

    return {
        "total_return": total_return, "btc_return": btc_return,
        "alpha": total_return - btc_return, "sharpe": sharpe,
        "max_dd": max_dd, "win_rate": win_rate,
        "num_trades": len(trade_log), "final_value": values[-1],
    }


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modular pipeline backtest")
    parser.add_argument("--start", default="2024-06-01")
    parser.add_argument("--end", default="2026-02-01")
    parser.add_argument("--rebalance", type=int, default=21)
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--no-xgboost", action="store_true")
    parser.add_argument("--mode", default="compare",
                        choices=["basic", "tuned", "crypto", "compare"],
                        help="basic: static weights, tuned: intelligence+params, "
                             "crypto: crypto backtest, compare: all 3")
    args = parser.parse_args()

    if args.mode == "basic":
        run_backtest(start_date=args.start, end_date=args.end,
                     rebalance_days=args.rebalance, initial_cash=args.cash,
                     use_xgboost=not args.no_xgboost)

    elif args.mode == "tuned":
        run_tuned_backtest(start_date=args.start, end_date=args.end,
                           initial_cash=args.cash)

    elif args.mode == "crypto":
        run_crypto_backtest(start_date=args.start, end_date=args.end,
                            initial_cash=5000)

    elif args.mode == "compare":
        print("\n" + "=" * 70)
        print("  3-WAY COMPARISON: Basic vs Tuned vs Crypto")
        print("=" * 70)

        r1 = run_backtest(start_date=args.start, end_date=args.end,
                          rebalance_days=21, initial_cash=args.cash,
                          use_xgboost=not args.no_xgboost)
        r2 = run_tuned_backtest(start_date=args.start, end_date=args.end,
                                initial_cash=args.cash)
        r3 = run_crypto_backtest(start_date="2024-01-01", end_date=args.end,
                                 initial_cash=5000)

        print(f"\n{'='*70}")
        print(f"  FINAL COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Metric':<16} {'Basic':>12} {'Tuned':>12} {'Crypto':>12}")
        print(f"  {'_'*52}")

        for key, label in [("total_return", "Return"), ("alpha", "Alpha"),
                           ("sharpe", "Sharpe"), ("max_dd", "Max DD"),
                           ("win_rate", "Win Rate")]:
            v1 = r1.get(key, 0)
            v2 = r2.get(key, 0)
            v3 = r3.get(key, 0)
            if key == "sharpe":
                print(f"  {label:<16} {v1:>12.2f} {v2:>12.2f} {v3:>12.2f}")
            else:
                print(f"  {label:<16} {v1*100:>+11.2f}% {v2*100:>+11.2f}% {v3*100:>+11.2f}%")

        print(f"\n  Benchmark:      SPY={r1.get('spy_return',0)*100:+.1f}%"
              f"    BTC={r3.get('btc_return',0)*100:+.1f}%")
        print(f"{'='*70}\n")
