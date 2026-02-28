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
    ) -> None:
        """Layer 1: Record structured reflection for a single trade.

        Papers: Reflexion (Shinn 2023), Chain of Hindsight (Liu 2023), ECHO (Hu 2025).
        No LLM call — pure template.
        """
        now = datetime.now(timezone.utc)
        trade_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{ticker.split('/')[0]}"

        # Chain of Hindsight: classify signals as worked/failed (long-only system)
        what_worked = []
        what_failed = []
        for sig, weight in (agent_signals or {}).items():
            if weight > 0 and pnl_pct > 0:
                what_worked.append(f"{sig} correctly bullish (aligned with winning long)")
            elif weight > 0 and pnl_pct < -0.001:
                what_failed.append(f"{sig} was bullish but price fell (long lost)")
            elif weight < 0 and pnl_pct > 0:
                what_failed.append(f"{sig} was bearish but long still profited (signal wrong)")
            elif weight < 0 and pnl_pct < -0.001:
                what_worked.append(f"{sig} correctly warned bearish (long lost as predicted)")

        # Determine outcome
        outcome = "win" if pnl_pct > 0 else ("loss" if pnl_pct < -0.001 else "breakeven")

        # ECHO: hindsight rewriting for losses (long-only system)
        hindsight = ""
        if outcome == "loss" and entry_price > 0 and exit_price > 0:
            reverse_pnl = -pnl_pct
            hindsight = f"If shorted instead, PnL would be ~{reverse_pnl:+.2%}"
        elif outcome == "win" and entry_price > 0:
            hindsight = f"Entry at ${entry_price:,.2f}, held {held_hours:.1f}h"

        # Build lesson string
        top_signals = sorted(agent_signals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        sig_names = " + ".join(s[0] for s in top_signals) if top_signals else "unknown"
        lesson = (
            f"In {regime.split('_')[0] if '_' in regime else regime}, "
            f"{sig_names} → {outcome} (PnL {pnl_pct:+.2%})"
        )

        # LLM Regret (Park 2024): regret score for losses
        papers_applied = ["Reflexion (Shinn 2023)", "Chain of Hindsight (Liu 2023)"]
        if outcome == "loss":
            papers_applied.append("ECHO (Hu 2025)")
            papers_applied.append("LLM Regret (Park 2024)")

        # Determine side from signals
        net_signal = sum(agent_signals.values()) if agent_signals else 0
        side = "LONG" if net_signal >= 0 else "SHORT"

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

        response = await self._claude_agent._call_sdk(DIGEST_SYSTEM, prompt, use_tools=False)
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
        """TradingGroup heuristic: same regime + 2+ shared signals → merge."""
        # Regime match (wildcard)
        e_scope = existing.get("regime_scope", "*")
        n_scope = new.get("regime_scope", "*")
        if e_scope != "*" and n_scope != "*":
            e_base = e_scope.split("_")[0]
            n_base = n_scope.split("_")[0]
            if e_base != n_base:
                return False

        # Signal overlap
        e_sigs = set(existing.get("signals", []))
        n_sigs = set(new.get("signals", []))
        shared = e_sigs & n_sigs
        return len(shared) >= 2

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
                outcome_sym = "+" if refl.get("outcome") == "win" else "-"
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
