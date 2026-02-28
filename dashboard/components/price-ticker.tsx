"use client";

import { useEffect, useRef, useState } from "react";

const SYMBOLS = [
  { label: "BTC", symbol: "btcusdt" },
  { label: "ETH", symbol: "ethusdt" },
  { label: "SOL", symbol: "solusdt" },
  { label: "PAXG", symbol: "paxgusdt" },
];

interface TickerPrice {
  price: number;
  prevPrice: number;
}

export default function PriceTicker() {
  const [prices, setPrices] = useState<Record<string, TickerPrice>>({});
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const streams = SYMBOLS.map((s) => `${s.symbol}@miniTicker`).join("/");
    const ws = new WebSocket(
      `wss://stream.binance.com:9443/stream?streams=${streams}`
    );

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const data = msg.data;
      if (!data || !data.s) return;

      const symbol = data.s.toLowerCase();
      const newPrice = parseFloat(data.c); // close price

      setPrices((prev) => ({
        ...prev,
        [symbol]: {
          price: newPrice,
          prevPrice: prev[symbol]?.price ?? newPrice,
        },
      }));
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div className="flex items-center gap-4">
      {SYMBOLS.map((s) => {
        const tick = prices[s.symbol];
        const price = tick?.price ?? 0;
        const isUp = tick ? tick.price >= tick.prevPrice : true;

        return (
          <div key={s.symbol} className="flex items-center gap-1.5">
            <span className="text-xs text-[var(--muted)] font-mono">
              {s.label}
            </span>
            <span
              className={`text-sm font-mono font-bold transition-colors duration-300 ${
                !tick
                  ? "text-[var(--text)]"
                  : isUp
                  ? "text-[var(--green)]"
                  : "text-[var(--red)]"
              }`}
            >
              {price > 0
                ? `$${price.toLocaleString(undefined, {
                    minimumFractionDigits: 0,
                    maximumFractionDigits: price < 10 ? 4 : 0,
                  })}`
                : "..."}
            </span>
          </div>
        );
      })}
      <span className="w-1.5 h-1.5 rounded-full bg-[var(--green)] pulse-dot" />
    </div>
  );
}
