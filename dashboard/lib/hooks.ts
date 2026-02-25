"use client";

import useSWR from "swr";
import {
  fetchAPI,
  type StatusData,
  type Decision,
  type Trade,
  type TradeSummary,
  type TSWeights,
} from "./api";

const fetcher = <T,>(path: string) => fetchAPI<T>(path);

export function useStatus() {
  return useSWR<StatusData>("/api/status", fetcher, {
    refreshInterval: 30_000,
  });
}

export function useDecisions(limit = 50) {
  return useSWR<{ decisions: Decision[]; total: number }>(
    `/api/decisions?limit=${limit}`,
    fetcher,
    { refreshInterval: 60_000 }
  );
}

export function useTrades(limit = 100) {
  return useSWR<{ trades: Trade[]; total: number; summary: TradeSummary }>(
    `/api/trades?limit=${limit}`,
    fetcher,
    { refreshInterval: 60_000 }
  );
}

export function useTSWeights() {
  return useSWR<TSWeights>("/api/ts-weights", fetcher, {
    refreshInterval: 300_000,
  });
}
