"""Trading Diary — 3-Layer Self-Reflection Memory System.

Papers applied:
  1. Reflexion (Shinn 2023, NeurIPS) — per-trade structured self-reflection
  2. TradingGroup (Tian 2025)        — similarity-based lesson retrieval
  3. FinMem (Yu 2024)                — 3-layer hierarchy (working/episodic/semantic)
  4. ECHO (Hu 2025)                  — hindsight rewriting for losses
  5. Chain of Hindsight (Liu 2023)   — win/loss contrastive signal analysis
  6. SAGE (Wei 2024)                 — Ebbinghaus forgetting curve decay
  7. LLM Regret (Park 2024, MIT)     — explicit regret scoring

Layer 1: Per-trade Reflection (JSONL, template-based, no LLM)
Layer 2: Daily Digest (MD + JSON, Claude 1x/day)
Layer 3: Weekly Semantic Distillation (JSON, Ebbinghaus decay, no LLM)
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trading-engine.diary")

# Papers reference for MD output
PAPERS = [
    ("Reflexion", "Shinn 2023", "Per-trade self-reflection"),
    ("Chain of Hindsight", "Liu 2023", "Win/loss contrastive analysis"),
    ("FinMem", "Yu 2024", "3-layer memory hierarchy"),
    ("ECHO", "Hu 2025", "Hindsight rewriting for losses"),
    ("SAGE", "Wei 2024", "Ebbinghaus forgetting curve"),
    ("TradingGroup", "Tian 2025", "Similarity-based lesson retrieval"),
    ("LLM Regret", "Park 2024", "Explicit regret minimization"),
]

# Digest system prompt (structured output, catches structural issues)
DIGEST_SYSTEM = (
    "You are a quantitative trading journal AI. Analyze today's trades with focus on:\n"
    "1. Win/Loss Size Asymmetry: avg win size vs avg loss size. Need R:R > 1:1.\n"
    "2. Tiny Wins: wins <0.1% are meaningless after fees — flag them.\n"
    "3. Profit Protection: if exit price is far below entry peak, profit_protect leaked gains.\n"
    "4. Per-Ticker Breakdown: which tickers are profitable vs draining capital?\n"
    "5. Hold Duration: are losers held too long vs winners cut too early?\n"
    "6. Exit Reason Patterns: which exit reasons (trailing_stop, hard_sl, profit_protect, claude_agent) correlate with best outcomes?\n"
    "\nOutput ONLY valid JSON:\n"
    "{\"narrative\": \"2-3 sentence summary focusing on the BIGGEST problem today\",\n"
    " \"patterns\": [\"list of observed patterns with numbers\"],\n"
    " \"lessons\": [\"actionable lessons, each starting with regime context\"]}\n"
    "Be specific with numbers. Do NOT be generic."
)

MAX_SEMANTIC_LESSONS = 20
DECAY_RATE = 0.1           # Ebbinghaus: exp(-0.1 * days)
DECAY_THRESHOLD = 0.15     # strength * decay < 0.15 → forget
PROMPT_STRENGTH_THRESHOLD = 0.25  # minimum for prompt injection
MAX_PROMPT_CHARS = 1200    # ~300 tokens budget


class TradingDiary:
    """3-layer self-reflection memory for trading decisions."""

    def __init__(
        self,
        data_dir: Path,
        md_dir: Path,
        claude_agent=None,
        online_learner=None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._md_dir = Path(md_dir)
        self._claude_agent = claude_agent
        self._online_learner = online_learner  # H-TS for diary→prior bridge

        # Ensure directories
        self._data_dir.mkdir(parents=True, exist_ok=True)
        (self._data_dir / "daily").mkdir(exist_ok=True)
        self._md_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self._reflections_path = self._data_dir / "reflections.jsonl"
        self._state_path = self._data_dir / "diary_state.json"
        self._semantic_path = self._data_dir / "semantic_lessons.json"

        # Load schedule state
        self._state = self._load_json(self._state_path, {
            "last_digest_date": None,
            "last_weekly_date": None,
        })

    # ------------------------------------------------------------------
    # Layer 1: Per-trade Reflection (template-based, no LLM)
    # ------------------------------------------------------------------

    def record_reflection(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        held_hours: float,
        regime: str,
        exit_reason: str,
        exit_source: str,
        agent_signals: Dict[str, float],
        entry_context: Optional[Dict[str, Any]] = None,
        position_side: str = "long",
        mfe: float = 0.0,
        mae: float = 0.0,
        capture_ratio: float = 0.0,
    ) -> None:
        """Layer 1: Record structured reflection for a single trade.

        Papers: Reflexion (Shinn 2023), Chain of Hindsight (Liu 2023), ECHO (Hu 2025).
        No LLM call — pure template.
        """
        now = datetime.now(timezone.utc)
        trade_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{ticker.split('/')[0]}"

        is_short = position_side == "short"
        side_label = "SHORT" if is_short else "LONG"

        # Chain of Hindsight: classify signals as worked/failed (side-aware)
        # Signal weight > 0 = "supported my decision", < 0 = "contradicted my decision"
        what_worked = []
        what_failed = []
        for sig, weight in (agent_signals or {}).items():
            if weight > 0 and pnl_pct > 0:
                what_worked.append(f"{sig} correctly supported winning {side_label}")
            elif weight > 0 and pnl_pct < -0.001:
                what_failed.append(f"{sig} supported {side_label} but trade lost")
            elif weight < 0 and pnl_pct > 0:
                what_failed.append(f"{sig} contradicted {side_label} but trade still profited (signal wrong)")
            elif weight < 0 and pnl_pct < -0.001:
                what_worked.append(f"{sig} correctly warned against {side_label} ({side_label} lost as predicted)")

        # Determine outcome
        outcome = "win" if pnl_pct > 0 else ("loss" if pnl_pct < -0.001 else "breakeven")

        # ECHO: hindsight rewriting for losses
        hindsight = ""
        if outcome == "loss" and entry_price > 0 and exit_price > 0:
            reverse_pnl = -pnl_pct
            opposite = "bought" if is_short else "shorted"
            hindsight = f"If {opposite} instead, PnL would be ~{reverse_pnl:+.2%}"
        elif outcome == "win" and entry_price > 0:
            hindsight = f"{side_label} entry at ${entry_price:,.2f}, held {held_hours:.1f}h"

        # Build lesson string (include MFE/MAE for exit quality feedback)
        top_signals = sorted(agent_signals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        sig_names = " + ".join(s[0] for s in top_signals) if top_signals else "unknown"
        mfe_str = f", MFE={mfe:+.2%}" if mfe else ""
        mae_str = f", MAE={mae:+.2%}" if mae else ""
        cap_str = f", capture={capture_ratio*100:.0f}%" if mfe > 0 else ""
        lesson = (
            f"In {regime.split('_')[0] if '_' in regime else regime}, "
            f"{sig_names} → {side_label} {outcome} (PnL {pnl_pct:+.2%}{mfe_str}{mae_str}{cap_str})"
        )

        # LLM Regret (Park 2024): regret score for losses
        papers_applied = ["Reflexion (Shinn 2023)", "Chain of Hindsight (Liu 2023)"]
        if outcome == "loss":
            papers_applied.append("ECHO (Hu 2025)")
            papers_applied.append("LLM Regret (Park 2024)")

        side = side_label

        entry_ctx = entry_context or {}

        record = {
            "trade_id": trade_id,
            "timestamp": now.isoformat(),
            "ticker": ticker,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 6),
            "held_hours": round(held_hours, 4),
            "mfe": round(mfe, 6),
            "mae": round(mae, 6),
            "capture_ratio": round(capture_ratio, 4),
            "regime": regime,
            "exit_reason": exit_reason,
            "exit_source": exit_source,
            "entry_context": {
                "reasoning": entry_ctx.get("reasoning", ""),
                "confidence": entry_ctx.get("confidence", 0),
                "signal_weights": entry_ctx.get("signal_weights", {}),
                "learning_note": entry_ctx.get("learning_note", ""),
            },
            "reflection": {
                "outcome": outcome,
                "what_worked": what_worked,
                "what_failed": what_failed,
                "hindsight": hindsight,
                "lesson": lesson,
                "paper_applied": " + ".join(papers_applied),
            },
        }

        # Atomic append
        self._append_jsonl(self._reflections_path, record)
        logger.info("[diary] L1 reflection: %s %s %s PnL=%+.2f%%",
                    ticker, outcome, exit_reason, pnl_pct * 100)

    def record_missed_opportunity(
        self,
        ticker: str,
        hold_price: float,
        current_price: float,
        raw_pnl_pct: float,
        horizon: str,
        regime: str,
        ta_signals: Dict[str, float],
        direction: Optional[str] = None,
    ) -> None:
        """Record a missed opportunity (counterfactual) in diary.

        Only records significant misses (> 0.5%) so the diary doesn't flood
        with noise. Claude sees these in prompt context as lessons.
        """
        if abs(raw_pnl_pct) < 0.005:  # < 0.5% = not worth noting
            return

        now = datetime.now(timezone.utc)
        if direction is None:
            direction = "BUY" if raw_pnl_pct > 0 else "SHORT"
        top_signals = sorted(ta_signals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        sig_names = " + ".join(s[0] for s in top_signals) if top_signals else "unknown"

        record = {
            "trade_id": f"{now.strftime('%Y%m%d_%H%M%S')}_{ticker.split('/')[0]}_missed",
            "timestamp": now.isoformat(),
            "ticker": ticker,
            "side": direction,
            "entry_price": hold_price,
            "exit_price": current_price,
            "pnl_pct": round(raw_pnl_pct, 6),
            "held_hours": 0,
            "regime": regime,
            "exit_reason": "counterfactual",
            "exit_source": f"missed_{horizon}",
            "entry_context": {
                "reasoning": f"HOLD decision — price moved {raw_pnl_pct:+.2%} over {horizon} window",
                "confidence": 0,
                "signal_weights": ta_signals,
                "learning_note": "",
            },
            "reflection": {
                "outcome": "missed",
                "what_worked": [],
                "what_failed": [f"Missed {direction} opportunity: {sig_names} signaled but SIT OUT overrode"],
                "hindsight": f"If {direction.lower()} at ${hold_price:,.2f}, would be {raw_pnl_pct:+.2%} after {horizon}",
                "lesson": f"In {regime.split('_')[0] if '_' in regime else regime}, {sig_names} → missed {raw_pnl_pct:+.2%} ({horizon})",
                "paper_applied": "LLM Regret (Park 2024) + Chain of Hindsight (Liu 2023)",
            },
        }

        self._append_jsonl(self._reflections_path, record)
        logger.info("[diary] L1 missed: %s %s %+.2f%% (%s)",
                    ticker, direction, raw_pnl_pct * 100, horizon)

    def record_correct_hold(
        self,
        ticker: str,
        hold_price: float,
        current_price: float,
        avoided_pct: float,
        horizon: str,
        regime: str,
        ta_signals: Dict[str, float],
    ) -> None:
        """Record a validated correct HOLD — price dropped, we were right not to buy."""
        if abs(avoided_pct) < 0.005:
            return

        now = datetime.now(timezone.utc)
        top_signals = sorted(ta_signals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        bearish_sigs = [s for s, v in top_signals if v < 0]
        sig_names = " + ".join(bearish_sigs) if bearish_sigs else " + ".join(s for s, _ in top_signals)

        record = {
            "trade_id": f"{now.strftime('%Y%m%d_%H%M%S')}_{ticker.split('/')[0]}_hold_ok",
            "timestamp": now.isoformat(),
            "ticker": ticker,
            "side": "HOLD",
            "entry_price": hold_price,
            "exit_price": current_price,
            "pnl_pct": round(avoided_pct, 6),
            "held_hours": 0,
            "regime": regime,
            "exit_reason": "correct_hold",
            "exit_source": f"validated_{horizon}",
            "entry_context": {
                "reasoning": f"Correct HOLD — avoided {avoided_pct:+.2%} loss over {horizon}",
                "confidence": 0,
                "signal_weights": ta_signals,
                "learning_note": "",
            },
            "reflection": {
                "outcome": "correct_hold",
                "what_worked": [f"HOLD decision validated: {sig_names} correctly warned bearish"],
                "what_failed": [],
                "hindsight": f"Avoided buying at ${hold_price:,.2f}, now ${current_price:,.2f} ({avoided_pct:+.2%})",
                "lesson": f"In {regime.split('_')[0] if '_' in regime else regime}, {sig_names} → correct HOLD, avoided {avoided_pct:+.2%} ({horizon})",
                "paper_applied": "Reflexion (Shinn 2023)",
            },
        }

        self._append_jsonl(self._reflections_path, record)
        logger.info("[diary] L1 correct_hold: %s avoided %+.2f%% (%s)",
                    ticker, avoided_pct * 100, horizon)

    def record_exit_regret(
        self,
        ticker: str,
        exit_price: float,
        current_price: float,
        pnl_at_exit: float,
        additional_pnl: float,
        horizon: str,
        regime: str,
        agent_signals: Dict[str, float],
        held_hours: float,
        position_side: str = "long",
        was_premature: bool = True,
    ) -> None:
        """Record post-exit counterfactual — premature exit or good exit."""
        if abs(additional_pnl) < 0.005:  # < 0.5% = not worth noting
            return

        now = datetime.now(timezone.utc)
        top_signals = sorted(agent_signals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        sig_names = " + ".join(s[0] for s in top_signals) if top_signals else "unknown"

        if was_premature:
            outcome = "premature_exit"
            suffix = "early"
            what_failed = [f"Exited too early: held {held_hours:.1f}h, missed {additional_pnl:+.2%} more"]
            what_worked = []
            hindsight = (f"Sold {position_side.upper()} at ${exit_price:,.2f} ({pnl_at_exit:+.2%}), "
                         f"now ${current_price:,.2f} ({additional_pnl:+.2%} additional)")
            lesson = f"In {regime.split('_')[0] if '_' in regime else regime}, {sig_names} → premature {position_side} exit, missed {additional_pnl:+.2%} ({horizon})"
        else:
            outcome = "good_exit"
            suffix = "good"
            what_worked = [f"Good exit timing: {sig_names} correctly signaled reversal"]
            what_failed = []
            hindsight = (f"Sold {position_side.upper()} at ${exit_price:,.2f} ({pnl_at_exit:+.2%}), "
                         f"now ${current_price:,.2f} (avoided {abs(additional_pnl):+.2%} loss)")
            lesson = f"In {regime.split('_')[0] if '_' in regime else regime}, {sig_names} → correct {position_side} exit, avoided {abs(additional_pnl):+.2%} ({horizon})"

        record = {
            "trade_id": f"{now.strftime('%Y%m%d_%H%M%S')}_{ticker.split('/')[0]}_exit_{suffix}",
            "timestamp": now.isoformat(),
            "ticker": ticker,
            "side": position_side.upper(),
            "entry_price": exit_price,  # "entry" for this CF = the exit price
            "exit_price": current_price,
            "pnl_pct": round(additional_pnl, 6),
            "held_hours": round(held_hours, 4),
            "regime": regime,
            "exit_reason": "exit_regret",
            "exit_source": f"{outcome}_{horizon}",
            "entry_context": {
                "reasoning": f"Post-exit tracking: price moved {additional_pnl:+.2%} after {horizon}",
                "confidence": 0,
                "signal_weights": agent_signals,
                "learning_note": "",
            },
            "reflection": {
                "outcome": outcome,
                "what_worked": what_worked,
                "what_failed": what_failed,
                "hindsight": hindsight,
                "lesson": lesson,
                "paper_applied": "LLM Regret (Park 2024) + Exit Regret Learning",
            },
        }

        self._append_jsonl(self._reflections_path, record)
        logger.info("[diary] L1 %s: %s %+.2f%% (%s)",
                    outcome, ticker, additional_pnl * 100, horizon)

    # ------------------------------------------------------------------
    # Layer 2: Daily Digest (Claude 1x/day)
    # ------------------------------------------------------------------

    async def maybe_generate_digest(self) -> bool:
        """Layer 2: Generate daily digest — updates throughout the day.

        Papers: FinMem (Yu 2024) episodic layer.
        Returns True if digest was generated/updated.

        Logic:
        - Always re-render MD when new trades exist (template, no LLM cost)
        - Call Claude for AI Analysis only when 3+ new trades since last analysis
        - Tracks last_digest_trade_count to detect new trades
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Collect today's reflections
        reflections = self._read_reflections_for_date(today)
        if not reflections:
            return False

        # Check if new trades since last digest
        last_count = self._state.get("last_digest_trade_count", 0)
        last_date = self._state.get("last_digest_date")
        if last_date != today:
            last_count = 0  # new day, reset

        if len(reflections) <= last_count:
            return False  # no new trades

        new_trade_count = len(reflections) - last_count
        logger.info("[diary] L2 updating daily digest for %s (%d trades, +%d new)",
                    today, len(reflections), new_trade_count)

        # Stats
        wins = sum(1 for r in reflections if r["reflection"]["outcome"] == "win")
        losses = sum(1 for r in reflections if r["reflection"]["outcome"] == "loss")
        total_pnl = sum(r["pnl_pct"] for r in reflections)
        win_rate = wins / len(reflections) if reflections else 0

        # Claude AI Analysis: only when 3+ new trades since last analysis
        narrative = ""
        patterns = []
        lessons = []
        last_claude_count = self._state.get("last_claude_digest_count", 0)
        if last_date != today:
            last_claude_count = 0

        need_claude = (len(reflections) - last_claude_count) >= 3

        if need_claude and self._claude_agent and self._claude_agent.is_available:
            try:
                narrative, patterns, lessons = await self._call_claude_digest(reflections)
                self._state["last_claude_digest_count"] = len(reflections)
            except Exception as exc:
                logger.warning("[diary] Claude digest failed: %s", exc)
        elif not need_claude and last_date == today:
            # Reuse previous Claude analysis if available
            prev_digest = self._load_json(
                self._data_dir / "daily" / f"{today}.json", {})
            narrative = prev_digest.get("narrative", "")
            patterns = prev_digest.get("patterns", [])
            lessons = prev_digest.get("lessons", [])

        if not narrative:
            # Template fallback — still catch structural issues
            win_trades = [r for r in reflections if r["pnl_pct"] > 0]
            loss_trades = [r for r in reflections if r["pnl_pct"] < -0.001]
            avg_win = sum(r["pnl_pct"] for r in win_trades) / len(win_trades) if win_trades else 0
            avg_loss = sum(r["pnl_pct"] for r in loss_trades) / len(loss_trades) if loss_trades else 0
            tiny_count = sum(1 for r in win_trades if r["pnl_pct"] < 0.001)

            narrative = (
                f"{len(reflections)} trades today ({wins}W {losses}L). "
                f"Net PnL: {total_pnl:+.2%}. Win rate: {win_rate:.0%}. "
                f"Avg win: {avg_win:+.2%}, Avg loss: {avg_loss:+.2%}."
            )

            # Pattern detection
            all_worked = []
            all_failed = []
            for r in reflections:
                all_worked.extend(r["reflection"]["what_worked"])
                all_failed.extend(r["reflection"]["what_failed"])
            if all_worked:
                patterns.append(f"Working signals: {', '.join(list(set(all_worked))[:3])}")
            if all_failed:
                patterns.append(f"Failing signals: {', '.join(list(set(all_failed))[:3])}")

            # Win/loss asymmetry warning
            if avg_loss != 0 and abs(avg_win / avg_loss) < 0.5:
                patterns.append(f"CRITICAL: Win/loss asymmetry — avg win {avg_win:+.2%} vs avg loss {avg_loss:+.2%} (R:R {abs(avg_win/avg_loss):.2f}:1)")
            if tiny_count > 0:
                patterns.append(f"WARNING: {tiny_count}/{len(win_trades)} wins are <0.1% (meaningless after fees)")

            # Per-ticker breakdown
            ticker_pnl: Dict[str, float] = {}
            for r in reflections:
                t = r["ticker"]
                ticker_pnl[t] = ticker_pnl.get(t, 0) + r["pnl_pct"]
            worst_ticker = min(ticker_pnl.items(), key=lambda x: x[1]) if ticker_pnl else None
            if worst_ticker and worst_ticker[1] < -0.005:
                patterns.append(f"Worst ticker: {worst_ticker[0]} at {worst_ticker[1]:+.2%}")

            lessons = [r["reflection"]["lesson"] for r in reflections[:3]]

        # Rule-based fallback detection
        fallback_trades = [r for r in reflections if r.get("exit_source", "").startswith("rule_based")]
        if fallback_trades:
            fb_pnl = sum(r["pnl_pct"] for r in fallback_trades)
            patterns.insert(0, (
                f"FALLBACK WARNING: {len(fallback_trades)} trades made by rule-based pipeline "
                f"(Claude unavailable). PnL: {fb_pnl:+.2%}. These trades lack agent reasoning."
            ))

        # Build digest data
        digest = {
            "date": today,
            "trade_count": len(reflections),
            "wins": wins,
            "losses": losses,
            "breakeven": len(reflections) - wins - losses,
            "net_pnl_pct": round(total_pnl, 6),
            "win_rate": round(win_rate, 4),
            "narrative": narrative,
            "patterns": patterns,
            "lessons": lessons,
            "trades": [
                {
                    "ticker": r["ticker"],
                    "side": r["side"],
                    "pnl_pct": r["pnl_pct"],
                    "held_hours": r.get("held_hours", 0),
                    "entry_price": r["entry_price"],
                    "exit_price": r["exit_price"],
                    "exit_reason": r["exit_reason"],
                    "exit_source": r.get("exit_source", "unknown"),
                    "lesson": r["reflection"]["lesson"],
                    "what_worked": r["reflection"]["what_worked"],
                    "what_failed": r["reflection"]["what_failed"],
                    "paper_applied": r["reflection"]["paper_applied"],
                }
                for r in reflections
            ],
        }

        # Save JSON (machine-readable)
        daily_json = self._data_dir / "daily" / f"{today}.json"
        self._write_json_atomic(daily_json, digest)

        # Save MD (human-readable)
        md_content = self._render_daily_md(digest)
        md_path = self._md_dir / f"{today}.md"
        self._write_text_atomic(md_path, md_content)

        # Update state
        self._state["last_digest_date"] = today
        self._state["last_digest_trade_count"] = len(reflections)
        self._save_state()

        # Trigger weekly distillation check (only on first digest of the day)
        if last_count == 0:
            await self.maybe_distill_weekly()

        logger.info("[diary] L2 digest saved: %s + %s (%d trades)", daily_json, md_path, len(reflections))
        return True

    async def _call_claude_digest(self, reflections: List[Dict]) -> tuple:
        """Call Claude for narrative digest (~300 tokens)."""
        # Compute aggregate stats for the prompt
        wins = [r for r in reflections if r["pnl_pct"] > 0]
        losses = [r for r in reflections if r["pnl_pct"] < -0.001]
        avg_win = sum(r["pnl_pct"] for r in wins) / len(wins) if wins else 0
        avg_loss = sum(r["pnl_pct"] for r in losses) / len(losses) if losses else 0
        tiny_wins = sum(1 for r in wins if r["pnl_pct"] < 0.001)  # <0.1%

        # Per-ticker PnL
        ticker_pnl: Dict[str, float] = {}
        for r in reflections:
            t = r["ticker"]
            ticker_pnl[t] = ticker_pnl.get(t, 0) + r["pnl_pct"]

        # Build compact per-trade input
        trades_text = []
        for r in reflections:
            trades_text.append(
                f"- {r['ticker']} {r['side']} PnL={r['pnl_pct']:+.2%} "
                f"held={r.get('held_hours', 0):.1f}h "
                f"entry=${r.get('entry_price', 0):,.0f} exit=${r.get('exit_price', 0):,.0f} "
                f"regime={r['regime']} exit_reason={r['exit_reason']} "
                f"worked={r['reflection']['what_worked']} "
                f"failed={r['reflection']['what_failed']}"
            )

        # Aggregate summary at top
        ticker_summary = ", ".join(f"{t}: {p:+.2%}" for t, p in sorted(ticker_pnl.items(), key=lambda x: x[1]))
        agg_block = (
            f"AGGREGATE: {len(reflections)} trades ({len(wins)}W {len(losses)}L), "
            f"avg_win={avg_win:+.2%}, avg_loss={avg_loss:+.2%}, "
            f"R:R={abs(avg_win/avg_loss) if avg_loss else 0:.2f}:1, "
            f"tiny_wins(<0.1%)={tiny_wins}\n"
            f"PER-TICKER: {ticker_summary}\n"
        )

        prompt = (
            agg_block
            + f"\nTRADES:\n"
            + "\n".join(trades_text)
            + "\n\nOutput JSON: {\"narrative\": \"...\", \"patterns\": [...], \"lessons\": [...]}"
        )

        response = await self._claude_agent._call_cli(DIGEST_SYSTEM, prompt)
        if not response:
            return ("", [], [])

        result = self._claude_agent._parse_json(response)
        if not result:
            return ("", [], [])

        return (
            result.get("narrative", ""),
            result.get("patterns", []),
            result.get("lessons", []),
        )

    def _render_daily_md(self, digest: Dict) -> str:
        """Render daily digest as Markdown for user reading."""
        # Compute aggregate stats for MD
        trades = digest.get("trades", [])
        win_pnls = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
        loss_pnls = [t["pnl_pct"] for t in trades if t["pnl_pct"] < -0.001]
        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
        rr = abs(avg_win / avg_loss) if avg_loss else 0

        lines = [
            f"# Trading Diary — {digest['date']}",
            "",
            "## Summary",
            f"- Trades: {digest['trade_count']} ({digest['wins']}W {digest['losses']}L)",
            f"- Net PnL: {digest['net_pnl_pct']:+.2%}",
            f"- Win Rate: {digest['win_rate']:.0%}",
            f"- Avg Win: {avg_win:+.2%} | Avg Loss: {avg_loss:+.2%} | R:R: {rr:.2f}:1",
            "",
            "## Trades",
        ]

        # Count rule-based fallback trades
        fallback_trades = [t for t in trades if t.get("exit_source", "").startswith("rule_based")]
        if fallback_trades:
            fb_pnl = sum(t["pnl_pct"] for t in fallback_trades)
            lines.append(f"- **Rule-based fallback**: {len(fallback_trades)} trades ({fb_pnl:+.2%} PnL) — Claude was unavailable")
            lines.append("")

        for i, t in enumerate(digest["trades"], 1):
            source_tag = ""
            src = t.get("exit_source", "")
            if "rule_based" in src or "safety" in src:
                source_tag = f" [{'FALLBACK' if 'rule_based' in src else 'SAFETY'}]"
            lines.append(f"### {i}. {t['ticker']} {t['side']} → {t['pnl_pct']:+.2%}{source_tag}")
            held = t.get("held_hours", 0)
            held_str = f"{held*60:.0f}min" if held < 1 else f"{held:.1f}h"
            lines.append(f"- Entry: ${t['entry_price']:,.2f} → Exit: ${t['exit_price']:,.2f} ({t['exit_reason']}, held {held_str})")
            if t["what_worked"]:
                lines.append(f"- Worked: {', '.join(t['what_worked'])}")
            if t["what_failed"]:
                lines.append(f"- Failed: {', '.join(t['what_failed'])}")
            lines.append(f"- Lesson: {t['lesson']}")
            lines.append(f"- Paper: {t['paper_applied']}")
            lines.append("")

        if digest.get("narrative"):
            lines.append("## AI Analysis")
            lines.append(digest["narrative"])
            lines.append("")

        if digest.get("patterns"):
            lines.append("## Patterns")
            for p in digest["patterns"]:
                lines.append(f"- {p}")
            lines.append("")

        if digest.get("lessons"):
            lines.append("## Key Lessons")
            for le in digest["lessons"]:
                lines.append(f"- {le}")
            lines.append("")

        lines.append("## Papers Applied")
        for name, author, desc in PAPERS:
            lines.append(f"- **{name}** ({author}): {desc}")

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Layer 3: Weekly Semantic Distillation (no LLM)
    # ------------------------------------------------------------------

    async def maybe_distill_weekly(self) -> bool:
        """Layer 3: Distill semantic lessons weekly.

        Papers: SAGE (Wei 2024) Ebbinghaus decay, TradingGroup (Tian 2025) similarity merge.
        Returns True if distillation ran.
        """
        today = datetime.now(timezone.utc).date()
        last_weekly = self._state.get("last_weekly_date")
        if last_weekly:
            try:
                last_date = datetime.strptime(last_weekly, "%Y-%m-%d").date()
                if (today - last_date).days < 7:
                    return False
            except ValueError:
                pass

        logger.info("[diary] L3 weekly semantic distillation")

        # Load existing lessons
        semantic = self._load_json(self._semantic_path, {"lessons": []})
        lessons = semantic.get("lessons", [])

        # Apply Ebbinghaus decay to existing lessons
        for lesson in lessons:
            last_reinforced = lesson.get("last_reinforced", "")
            try:
                lr_date = datetime.strptime(last_reinforced, "%Y-%m-%d").date()
                days_since = (today - lr_date).days
            except (ValueError, TypeError):
                days_since = 30  # default high decay

            lesson["decay_factor"] = math.exp(-DECAY_RATE * days_since)

        # Collect new lessons from last 7 days of reflections
        cutoff = (today - timedelta(days=7)).isoformat()
        new_lessons = self._extract_lessons_since(cutoff)

        # Merge new lessons into existing (TradingGroup similarity)
        for nl in new_lessons:
            merged = False
            for existing in lessons:
                if self._lessons_similar(existing, nl):
                    # Reinforce existing lesson
                    existing["occurrences"] = existing.get("occurrences", 1) + 1
                    existing["decay_factor"] = 1.0  # reset decay
                    existing["last_reinforced"] = today.isoformat()
                    # Update strength: use max of existing or new
                    existing["strength"] = max(
                        existing.get("strength", 0.5),
                        nl.get("strength", 0.5),
                    )
                    merged = True
                    break
            if not merged:
                lessons.append({
                    "text": nl["text"],
                    "regime_scope": nl.get("regime_scope", "*"),
                    "signals": nl.get("signals", []),
                    "outcome": nl.get("outcome", ""),
                    "occurrences": 1,
                    "strength": nl.get("strength", 0.5),
                    "decay_factor": 1.0,
                    "first_observed": today.isoformat(),
                    "last_reinforced": today.isoformat(),
                })

        # Prune: remove decayed lessons
        lessons = [
            l for l in lessons
            if l.get("strength", 0.5) * l.get("decay_factor", 1.0) >= DECAY_THRESHOLD
        ]

        # Keep top N by effective strength
        lessons.sort(
            key=lambda l: l.get("strength", 0.5) * l.get("decay_factor", 1.0),
            reverse=True,
        )
        lessons = lessons[:MAX_SEMANTIC_LESSONS]

        # Save
        semantic["lessons"] = lessons
        self._write_json_atomic(self._semantic_path, semantic)

        # Update state
        self._state["last_weekly_date"] = today.isoformat()
        self._save_state()

        logger.info("[diary] L3 distilled: %d semantic lessons", len(lessons))

        # Diary L3 → H-TS prior bridge (TradingGroup + LLM Regret papers)
        if self._online_learner and lessons:
            try:
                n_adj = self._online_learner.apply_diary_lessons(lessons)
                logger.info("[diary] H-TS prior injection: %d adjustments", n_adj)
            except Exception as exc:
                logger.debug("[diary] H-TS injection failed: %s", exc)

        return True

    def _extract_lessons_since(self, cutoff_iso: str) -> List[Dict]:
        """Extract lesson entries from reflections since cutoff date."""
        lessons = []
        if not self._reflections_path.exists():
            return lessons

        with open(self._reflections_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = r.get("timestamp", "")
                if ts < cutoff_iso:
                    continue

                refl = r.get("reflection", {})
                pnl = r.get("pnl_pct", 0)
                regime = r.get("regime", "")

                # LLM Regret: bigger loss → higher base strength
                if pnl < -0.01:
                    strength = min(0.9, 0.5 + abs(pnl) * 10)
                elif pnl > 0.005:
                    strength = min(0.8, 0.4 + pnl * 10)
                else:
                    strength = 0.3  # breakeven — low signal

                # Extract regime scope (wildcard on macro part)
                regime_parts = regime.split("_", 1)
                regime_scope = f"{regime_parts[0]}_*" if len(regime_parts) > 1 else regime

                # Extract signal names from entry_context
                sig_names = list(r.get("entry_context", {}).get("signal_weights", {}).keys())

                outcome = refl.get("outcome", "win" if pnl > 0 else ("loss" if pnl < -0.001 else "breakeven"))
                lessons.append({
                    "text": refl.get("lesson", ""),
                    "regime_scope": regime_scope,
                    "signals": sig_names,
                    "strength": strength,
                    "outcome": outcome,
                })

        return lessons

    def _lessons_similar(self, existing: Dict, new: Dict) -> bool:
        """TradingGroup heuristic: same regime + outcome + high signal overlap → merge."""
        # Outcome must match (win↔win, loss↔loss)
        e_outcome = existing.get("outcome", "")
        n_outcome = new.get("outcome", "")
        if e_outcome and n_outcome and e_outcome != n_outcome:
            return False

        # Regime match (wildcard)
        e_scope = existing.get("regime_scope", "*")
        n_scope = new.get("regime_scope", "*")
        if e_scope != "*" and n_scope != "*":
            e_base = e_scope.split("_")[0]
            n_base = n_scope.split("_")[0]
            if e_base != n_base:
                return False

        # Signal overlap: Jaccard >= 0.5 AND min 3 shared
        e_sigs = set(existing.get("signals", []))
        n_sigs = set(new.get("signals", []))
        if not e_sigs or not n_sigs:
            return False
        shared = e_sigs & n_sigs
        union = e_sigs | n_sigs
        jaccard = len(shared) / len(union) if union else 0
        return len(shared) >= 3 and jaccard >= 0.5

    # ------------------------------------------------------------------
    # Prompt injection: get_prompt_context
    # ------------------------------------------------------------------

    def get_prompt_context(
        self,
        regime: str = "",
        tickers: List[str] = None,
    ) -> str:
        """Build diary context string for Claude prompt (~300 tokens max).

        Papers: TradingGroup (Tian 2025) for regime-based retrieval,
                SAGE (Wei 2024) for Ebbinghaus-filtered lessons.
        """
        parts = []

        # Recent reflections (last 3)
        recent = self._read_recent_reflections(3)
        if recent:
            lines = ["### Recent Trades (last {})".format(len(recent))]
            for i, r in enumerate(recent, 1):
                refl = r.get("reflection", {})
                outcome = refl.get("outcome", "")
                outcome_sym = "+" if outcome == "win" else ("~" if outcome == "missed" else ("v" if outcome == "correct_hold" else "-"))
                short_lesson = refl.get("lesson", "")[:80]
                lines.append(
                    f"{i}. {r['ticker']} {r['side']} {outcome_sym}{abs(r['pnl_pct'])*100:.2f}%: {short_lesson}"
                )
            parts.append("\n".join(lines))

        # Semantic lessons (Ebbinghaus-filtered, regime-matched)
        semantic = self._load_json(self._semantic_path, {"lessons": []})
        lessons = semantic.get("lessons", [])

        # Apply current decay
        today = datetime.now(timezone.utc).date()
        filtered = []
        for l in lessons:
            try:
                lr_date = datetime.strptime(l.get("last_reinforced", ""), "%Y-%m-%d").date()
                days_since = (today - lr_date).days
            except (ValueError, TypeError):
                days_since = 30
            decay = math.exp(-DECAY_RATE * days_since)
            effective = l.get("strength", 0.5) * decay
            if effective < PROMPT_STRENGTH_THRESHOLD:
                continue

            # Regime match
            if regime:
                scope = l.get("regime_scope", "*")
                if scope != "*":
                    scope_base = scope.split("_")[0]
                    regime_base = regime.split("_")[0]
                    if scope_base != regime_base:
                        continue

            label = "HIGH" if effective >= 0.6 else ("MED" if effective >= 0.4 else "LOW")
            occ = l.get("occurrences", 1)
            filtered.append((effective, f"- [{label}] {l['text']} ({occ}x)"))

        if filtered:
            filtered.sort(key=lambda x: -x[0])
            lesson_lines = ["### Lessons (Ebbinghaus-filtered)"]
            for _, line in filtered[:5]:  # max 5 lessons
                lesson_lines.append(line)
            parts.append("\n".join(lesson_lines))

        result = "\n\n".join(parts)

        # Enforce token budget
        if len(result) > MAX_PROMPT_CHARS:
            result = result[:MAX_PROMPT_CHARS] + "..."

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_recent_reflections(self, n: int) -> List[Dict]:
        """Read last N reflections from JSONL (tail read)."""
        if not self._reflections_path.exists():
            return []
        lines = []
        try:
            with open(self._reflections_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
        except Exception:
            return []
        result = []
        for line in lines[-n:]:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def _read_reflections_for_date(self, date_str: str) -> List[Dict]:
        """Read all reflections for a specific date (YYYY-MM-DD)."""
        if not self._reflections_path.exists():
            return []
        result = []
        try:
            with open(self._reflections_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("timestamp", "").startswith(date_str):
                        result.append(r)
        except Exception:
            pass
        return result

    def _append_jsonl(self, path: Path, record: Dict) -> None:
        """Atomic append: write to tmp then rename (append mode)."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            logger.warning("[diary] JSONL append failed: %s", exc)

    def _write_json_atomic(self, path: Path, data: Any) -> None:
        """Atomic write: tmp + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except Exception as exc:
            logger.warning("[diary] Atomic write failed: %s", exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _write_text_atomic(self, path: Path, text: str) -> None:
        """Atomic write for text files."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except Exception as exc:
            logger.warning("[diary] Text write failed: %s", exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _load_json(self, path: Path, default: Any = None) -> Any:
        """Load JSON file, return default if missing/corrupt."""
        if not path.exists():
            return default if default is not None else {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return default if default is not None else {}

    def _save_state(self) -> None:
        """Persist diary schedule state."""
        self._write_json_atomic(self._state_path, self._state)
