"""Analyze Claude's decision history for quality, risk management, and profitability."""
import json
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

def main():
    decisions = []
    with open(LOG_DIR / "decisions.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))

    print(f"=== CLAUDE DECISION ANALYSIS ({len(decisions)} decisions) ===\n")

    # 1. Timeline
    print("--- DECISION TIMELINE ---")
    trade_decisions = []
    hold_decisions = []
    for i, d in enumerate(decisions):
        ts = d["timestamp"][:19]
        trigger = d.get("trigger", "?")
        btc = d.get("btc_price", 0)
        fg = d.get("fear_greed", "?")
        regime = d.get("regime", "?")
        positions = d.get("positions", [])
        trades = d.get("trades", [])
        next_s = d.get("next_check_s", 0)
        wake_conds = d.get("wake_conditions", [])
        assessment = d.get("claude_assessment", "")[:150]

        trade_str = ""
        if trades:
            for t in trades:
                trade_str += f" -> {t.get('action','?')} {t.get('ticker','?')} @${t.get('price',0):.0f}"
            trade_decisions.append(d)
        else:
            hold_decisions.append(d)

        pos_str = ",".join(positions) if positions else "CASH"
        print(f"#{i+1} [{ts}] trig={trigger} BTC=${btc:.0f} F&G={fg}")
        print(f"   pos=[{pos_str}] next={next_s}s wakes={len(wake_conds)}{trade_str}")
        print(f"   > {assessment}")
        print()

    # 2. Decision quality metrics
    print("\n--- DECISION QUALITY METRICS ---")
    print(f"Total decisions: {len(decisions)}")
    print(f"Trade decisions: {len(trade_decisions)} ({len(trade_decisions)/len(decisions)*100:.0f}%)")
    print(f"Hold decisions:  {len(hold_decisions)} ({len(hold_decisions)/len(decisions)*100:.0f}%)")

    # 3. Wake condition analysis
    print("\n--- WAKE CONDITION ANALYSIS ---")
    trigger_counts = {}
    for d in decisions:
        t = d.get("trigger", "unknown")
        trigger_counts[t] = trigger_counts.get(t, 0) + 1
    for t, c in sorted(trigger_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} times")

    # 4. Self-scheduling analysis
    print("\n--- SELF-SCHEDULING ANALYSIS ---")
    check_intervals = [d.get("next_check_s", 0) for d in decisions if d.get("next_check_s")]
    if check_intervals:
        avg_interval = sum(check_intervals) / len(check_intervals)
        min_interval = min(check_intervals)
        max_interval = max(check_intervals)
        print(f"  Avg next_check: {avg_interval:.0f}s ({avg_interval/60:.1f}m)")
        print(f"  Min: {min_interval}s, Max: {max_interval}s")

    wake_types = {}
    for d in decisions:
        for w in d.get("wake_conditions", []):
            op = w.get("operator", "?")
            metric = w.get("metric", "?")
            key = f"{metric}:{op}"
            wake_types[key] = wake_types.get(key, 0) + 1
    if wake_types:
        print("  Wake condition types:")
        for k, c in sorted(wake_types.items(), key=lambda x: -x[1]):
            print(f"    {k}: {c}")

    # 5. Risk management assessment
    print("\n--- RISK MANAGEMENT ASSESSMENT ---")
    if trade_decisions:
        for td in trade_decisions:
            for trade in td.get("trades", []):
                action = trade.get("action", "?")
                ticker = trade.get("ticker", "?")
                price = trade.get("price", 0)
                qty = trade.get("qty", 0)
                value = price * qty if price and qty else 0
                portfolio = td.get("portfolio_value", 327000)
                pct = (value / portfolio * 100) if portfolio else 0
                confidence = trade.get("confidence", 0)
                reason = trade.get("reason", "")[:200]
                print(f"  {action} {ticker}: ${value:.0f} ({pct:.1f}% of portfolio), conf={confidence}")
                # Check if stop loss is mentioned
                if "stop" in reason.lower() or "$61" in reason or "1,750" in reason:
                    print(f"    SL mentioned in reasoning: YES")
                else:
                    print(f"    SL mentioned in reasoning: NO")

    # 6. Memory evolution
    print("\n--- AGENT MEMORY EVOLUTION ---")
    memories = [d.get("memory", "") for d in decisions if d.get("memory")]
    if memories:
        print(f"  Memory entries: {len(memories)}")
        print(f"  First memory: {memories[0][:200]}")
        print(f"  Last memory:  {memories[-1][:200]}")
        # Check for learning signals
        learning_keywords = ["learned", "mistake", "previous", "note", "ignore", "never", "always"]
        learning_count = 0
        for m in memories:
            ml = m.lower()
            if any(k in ml for k in learning_keywords):
                learning_count += 1
        print(f"  Memories with learning signals: {learning_count}/{len(memories)} ({learning_count/len(memories)*100:.0f}%)")

    # 7. Position awareness
    print("\n--- POSITION AWARENESS ---")
    pos_counts = {"with_positions": 0, "no_positions": 0}
    for d in decisions:
        if d.get("positions"):
            pos_counts["with_positions"] += 1
        else:
            pos_counts["no_positions"] += 1
    print(f"  Decisions with positions: {pos_counts['with_positions']}")
    print(f"  Decisions without positions: {pos_counts['no_positions']}")

    # 8. Profitability signals
    print("\n--- PROFITABILITY SIGNALS ---")
    # Check regime awareness
    regimes = set(d.get("regime", "unknown") for d in decisions)
    print(f"  Regimes encountered: {regimes}")

    # Check F&G distribution
    fg_values = [d.get("fear_greed", 50) for d in decisions if d.get("fear_greed") is not None]
    if fg_values:
        avg_fg = sum(fg_values) / len(fg_values)
        print(f"  Avg Fear & Greed: {avg_fg:.0f}")
        print(f"  Traded in extreme fear (<10): {any(d.get('fear_greed',50) < 10 for d in trade_decisions)}")

    # BTC price range during session
    btc_prices = [d.get("btc_price", 0) for d in decisions if d.get("btc_price")]
    if btc_prices:
        print(f"  BTC range: ${min(btc_prices):.0f} - ${max(btc_prices):.0f} ({(max(btc_prices)/min(btc_prices)-1)*100:.1f}%)")


if __name__ == "__main__":
    main()
