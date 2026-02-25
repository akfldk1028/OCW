"""OHLCV data fetching and technical indicator computation.

Uses yfinance for market data and the ``ta`` library (pure Python) for
computing standard technical indicators.  The output is a combined,
normalised DataFrame suitable for consumption by the RL trading
environment.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import yfinance as yf

from config import DATA_DIR, TECHNICAL_INDICATORS, TRAIN_CONFIG

logger = logging.getLogger(__name__)


class DataProcessor:
    """Fetches OHLCV data, computes technical indicators and prepares
    feature arrays for reinforcement-learning training and inference."""

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def fetch_data(
        self,
        tickers: List[str],
        start: str,
        end: str,
        *,
        save_raw: bool = True,
    ) -> pd.DataFrame:
        """Download OHLCV data for *tickers* between *start* and *end*.

        Parameters
        ----------
        tickers : list[str]
            Yahoo Finance ticker symbols.
        start, end : str
            Date strings accepted by ``yfinance`` (``YYYY-MM-DD``).
        save_raw : bool
            Persist the raw download as a Parquet file.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns
            ``[date, tic, open, high, low, close, volume]``.
        """
        logger.info("Fetching data for %s from %s to %s", tickers, start, end)

        frames: List[pd.DataFrame] = []
        for tic in tickers:
            try:
                df = yf.download(tic, start=start, end=end, progress=False, auto_adjust=True)
                if df.empty:
                    logger.warning("No data returned for %s", tic)
                    continue

                # yfinance may return MultiIndex columns when downloading a
                # single ticker -- flatten if necessary.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                df = df.rename(columns={"adj close": "close"})  # already auto-adjusted
                df["tic"] = tic
                df = df[["date", "tic", "open", "high", "low", "close", "volume"]]
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to fetch %s: %s", tic, exc)

        if not frames:
            raise ValueError("No data could be fetched for any of the requested tickers.")

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["date", "tic"]).reset_index(drop=True)

        if save_raw:
            path = DATA_DIR / "raw" / f"ohlcv_{start}_{end}.csv"
            combined.to_csv(path, index=False)
            logger.info("Raw data saved to %s", path)

        return combined

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators per ticker and append them as new
        columns.

        Uses the ``ta`` library (pure Python, no C dependency).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``[date, tic, open, high, low, close, volume]``.

        Returns
        -------
        pd.DataFrame
            Input frame augmented with indicator columns listed in
            ``config.TECHNICAL_INDICATORS``.
        """
        logger.info("Computing technical indicators")
        result_frames: List[pd.DataFrame] = []

        for tic, group in df.groupby("tic"):
            g = group.sort_values("date").copy()
            close = g["close"]
            high = g["high"]
            low = g["low"]
            volume = g["volume"].astype(float)

            # --- Trend ---
            g["ema_12"] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
            g["ema_26"] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()
            macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            g["macd"] = macd_ind.macd()
            g["macd_signal"] = macd_ind.macd_signal()
            g["macd_diff"] = macd_ind.macd_diff()

            # --- Momentum ---
            g["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

            # --- Volatility ---
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            g["bb_upper"] = bb.bollinger_hband()
            g["bb_middle"] = bb.bollinger_mavg()
            g["bb_lower"] = bb.bollinger_lband()
            g["atr_14"] = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()

            # --- Volume ---
            g["obv"] = ta.volume.OnBalanceVolumeIndicator(
                close=close, volume=volume
            ).on_balance_volume()
            g["volume_sma_20"] = volume.rolling(window=20).mean()

            result_frames.append(g)

        combined = pd.concat(result_frames, ignore_index=True)
        combined = combined.sort_values(["date", "tic"]).reset_index(drop=True)

        # Drop rows where any indicator is NaN (warm-up period)
        indicator_cols = [c for c in TECHNICAL_INDICATORS if c in combined.columns]
        combined = combined.dropna(subset=indicator_cols).reset_index(drop=True)

        return combined

    def create_feature_array(
        self,
        df: pd.DataFrame,
        *,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Convert the enriched DataFrame into a 3-D numpy array suitable
        for the RL environment.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``add_technical_indicators``.
        normalize : bool
            Apply z-score normalisation to the feature columns.

        Returns
        -------
        feature_array : np.ndarray
            Shape ``(num_dates, num_tickers, num_features)``.
        meta_df : pd.DataFrame
            A tidy reference frame with date / ticker info aligned to
            *feature_array*.
        """
        feature_cols = ["close", "open", "high", "low", "volume"] + [
            c for c in TECHNICAL_INDICATORS if c in df.columns
        ]

        tickers = sorted(df["tic"].unique())
        dates = sorted(df["date"].unique())

        num_dates = len(dates)
        num_tickers = len(tickers)
        num_features = len(feature_cols)

        # Pivot into a nested structure: date -> ticker -> features
        feature_array = np.zeros((num_dates, num_tickers, num_features), dtype=np.float32)

        tic_to_idx = {t: i for i, t in enumerate(tickers)}
        date_to_idx = {d: i for i, d in enumerate(dates)}

        for _, row in df.iterrows():
            di = date_to_idx[row["date"]]
            ti = tic_to_idx[row["tic"]]
            feature_array[di, ti, :] = [float(row[c]) for c in feature_cols]

        if normalize:
            feature_array = self._zscore_normalize(feature_array)

        # Build a compact meta DataFrame for reference
        meta_rows = []
        for d_idx, d in enumerate(dates):
            for t_idx, t in enumerate(tickers):
                meta_rows.append({"date_idx": d_idx, "tic_idx": t_idx, "date": d, "tic": t})
        meta_df = pd.DataFrame(meta_rows)

        return feature_array, meta_df

    def prepare_train_test(
        self,
        tickers: Optional[List[str]] = None,
        lookback_days: Optional[int] = None,
        train_split: Optional[float] = None,
    ) -> Dict[str, object]:
        """End-to-end convenience method: fetch, enrich and split data.

        Returns
        -------
        dict with keys:
            ``train_array``, ``test_array`` (np.ndarray),
            ``train_df``, ``test_df`` (pd.DataFrame),
            ``tickers`` (list), ``feature_cols`` (list).
        """
        tickers = tickers or TRAIN_CONFIG["tickers"]
        lookback_days = lookback_days or TRAIN_CONFIG["lookback_days"]
        train_split = train_split or TRAIN_CONFIG["train_split"]

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        raw_df = self.fetch_data(tickers, start_date, end_date)
        enriched_df = self.add_technical_indicators(raw_df)

        dates = sorted(enriched_df["date"].unique())
        split_idx = int(len(dates) * train_split)
        train_dates = set(dates[:split_idx])
        test_dates = set(dates[split_idx:])

        train_df = enriched_df[enriched_df["date"].isin(train_dates)].reset_index(drop=True)
        test_df = enriched_df[enriched_df["date"].isin(test_dates)].reset_index(drop=True)

        train_array, _ = self.create_feature_array(train_df)
        test_array, _ = self.create_feature_array(test_df)

        feature_cols = ["close", "open", "high", "low", "volume"] + [
            c for c in TECHNICAL_INDICATORS if c in enriched_df.columns
        ]

        # Persist processed data
        train_df.to_csv(DATA_DIR / "processed" / "train.csv", index=False)
        test_df.to_csv(DATA_DIR / "processed" / "test.csv", index=False)

        logger.info(
            "Data prepared -- train: %d days, test: %d days, tickers: %d, features: %d",
            train_array.shape[0],
            test_array.shape[0],
            train_array.shape[1],
            train_array.shape[2],
        )

        return {
            "train_array": train_array,
            "test_array": test_array,
            "train_df": train_df,
            "test_df": test_df,
            "tickers": sorted(enriched_df["tic"].unique().tolist()),
            "feature_cols": feature_cols,
        }

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    @staticmethod
    def _zscore_normalize(arr: np.ndarray) -> np.ndarray:
        """Z-score normalise along the time axis (axis 0).

        Each feature for each ticker is independently normalised so that
        the time-series has zero mean and unit variance.
        """
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        # Avoid division by zero for constant features
        std = np.where(std < 1e-8, 1.0, std)
        return (arr - mean) / std
