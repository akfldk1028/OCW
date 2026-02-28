"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const TICKERS = [
  { label: "BTC", symbol: "btcusdt", pair: "BTC/USDT" },
  { label: "ETH", symbol: "ethusdt", pair: "ETH/USDT" },
  { label: "SOL", symbol: "solusdt", pair: "SOL/USDT" },
];

interface ChartPoint {
  time: string;
  price: number;
  high: number;
  low: number;
  volume: number;
  ts: number;
}

export default function PriceChart() {
  const [selected, setSelected] = useState(0);
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [livePrice, setLivePrice] = useState(0);
  const [loading, setLoading] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);

  const ticker = TICKERS[selected];

  // Fetch initial klines from Binance REST API + connect WebSocket
  const connect = useCallback(() => {
    setLoading(true);

    // 1. Fetch historical 1m klines (last 200 candles)
    fetch(
      `https://api.binance.com/api/v3/klines?symbol=${ticker.symbol.toUpperCase()}&interval=1m&limit=200`
    )
      .then((r) => r.json())
      .then((klines: number[][]) => {
        const points: ChartPoint[] = klines.map((k) => ({
          ts: k[0] as number,
          time: new Date(k[0] as number).toLocaleTimeString("ko-KR", {
            hour: "2-digit",
            minute: "2-digit",
          }),
          price: parseFloat(k[4] as unknown as string), // close
          high: parseFloat(k[2] as unknown as string),
          low: parseFloat(k[3] as unknown as string),
          volume: parseFloat(k[5] as unknown as string),
        }));
        setChartData(points);
        setLivePrice(points[points.length - 1]?.price ?? 0);
        setLoading(false);
      })
      .catch(() => setLoading(false));

    // 2. Connect WebSocket for real-time kline updates
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(
      `wss://stream.binance.com:9443/ws/${ticker.symbol}@kline_1m`
    );

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const k = msg.k;
      if (!k) return;

      const point: ChartPoint = {
        ts: k.t,
        time: new Date(k.t).toLocaleTimeString("ko-KR", {
          hour: "2-digit",
          minute: "2-digit",
        }),
        price: parseFloat(k.c), // close
        high: parseFloat(k.h),
        low: parseFloat(k.l),
        volume: parseFloat(k.v),
      };

      setLivePrice(point.price);

      setChartData((prev) => {
        if (prev.length === 0) return prev;
        const last = prev[prev.length - 1];
        // Same candle — update in place
        if (last.ts === point.ts) {
          return [...prev.slice(0, -1), point];
        }
        // New candle — append and trim
        const next = [...prev, point];
        if (next.length > 200) next.shift();
        return next;
      });
    };

    ws.onerror = () => {
      // Reconnect after 3s
      setTimeout(connect, 3000);
    };

    wsRef.current = ws;
  }, [ticker.symbol]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  if (loading && chartData.length === 0) {
    return <div className="card animate-pulse h-80" />;
  }

  const prices = chartData.map((d) => d.price);
  const minPrice = Math.min(...prices) * 0.9995;
  const maxPrice = Math.max(...prices) * 1.0005;
  const firstPrice = prices[0] ?? livePrice;
  const change =
    firstPrice > 0 ? ((livePrice - firstPrice) / firstPrice) * 100 : 0;
  const isUp = change >= 0;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-medium text-[var(--muted)]">Price</h2>
            <span className="w-2 h-2 rounded-full bg-[var(--green)] pulse-dot" />
            <span className="text-[10px] text-[var(--green)]">LIVE</span>
          </div>
          <div className="flex gap-1">
            {TICKERS.map((t, i) => (
              <button
                key={t.symbol}
                onClick={() => setSelected(i)}
                className={`text-xs px-2 py-1 rounded font-mono transition-colors ${
                  selected === i
                    ? "bg-[var(--blue)] text-white"
                    : "bg-[var(--border)] text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <div className="text-right">
          <p className="text-xl font-mono font-bold tracking-tight">
            $
            {livePrice.toLocaleString(undefined, {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}
          </p>
          <span
            className={`text-sm font-mono font-bold px-2 py-0.5 rounded-full ${
              isUp
                ? "bg-[var(--green)]/10 text-[var(--green)]"
                : "bg-[var(--red)]/10 text-[var(--red)]"
            }`}
          >
            {isUp ? "▲" : "▼"} {Math.abs(change).toFixed(2)}%
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient
              id={`priceGrad-${ticker.symbol}`}
              x1="0"
              y1="0"
              x2="0"
              y2="1"
            >
              <stop
                offset="0%"
                stopColor={isUp ? "#22c55e" : "#ef4444"}
                stopOpacity={0.3}
              />
              <stop
                offset="100%"
                stopColor={isUp ? "#22c55e" : "#ef4444"}
                stopOpacity={0}
              />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: "#71717a" }}
            interval="preserveStartEnd"
            tickCount={6}
          />
          <YAxis
            domain={[minPrice, maxPrice]}
            tick={{ fontSize: 10, fill: "#71717a" }}
            tickFormatter={(v: number) =>
              v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(2)
            }
            width={60}
          />
          <Tooltip
            contentStyle={{
              background: "#12121a",
              border: "1px solid #1e1e2e",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#71717a" }}
            formatter={(value: number) => [
              `$${value.toLocaleString(undefined, {
                minimumFractionDigits: 2,
              })}`,
              "Price",
            ]}
          />
          <Area
            type="monotone"
            dataKey="price"
            stroke={isUp ? "#22c55e" : "#ef4444"}
            strokeWidth={2}
            fill={`url(#priceGrad-${ticker.symbol})`}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
