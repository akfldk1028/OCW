"""Trading engine configuration."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for d in [MODELS_DIR, DATA_DIR / "raw", DATA_DIR / "processed", LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Training defaults
TRAIN_CONFIG = {
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"],
    "lookback_days": 365,
    "train_split": 0.8,
    "total_timesteps": 50_000,
    "learning_rate": 3e-4,
}

# Technical indicators
TECHNICAL_INDICATORS = [
    "ema_12", "ema_26", "macd", "macd_signal", "macd_diff",
    "rsi_14", "bb_upper", "bb_middle", "bb_lower",
    "atr_14", "obv", "volume_sma_20",
]

# Ensemble config
ENSEMBLE_CONFIG = {
    "models": ["ppo", "a2c", "sac"],  # SAC replaces DDPG: entropy regularization optimal for crypto volatility
    "sharpe_window": 63,  # ~3 months trading days
    "min_weight": 0.1,
    "rebalance_freq": 5,  # trading days
}

# Risk management defaults
RISK_CONFIG = {
    "take_profit_pct": 0.04,       # 4% take profit (crypto: daily vol 5-10%)
    "stop_loss_pct": -0.025,       # -2.5% stop loss
    "max_daily_trades": 20,
    "max_position_pct": 0.15,      # 15% max single position
    "max_exposure_pct": 0.80,      # 80% max total exposure
    "kelly_fraction": 0.25,        # quarter-Kelly for safety
    # ATR-based dynamic adjustment
    "atr_tp_multiplier": 2.5,     # TP = ATR * multiplier
    "atr_sl_multiplier": 1.5,     # SL = ATR * multiplier
    "use_atr_dynamic": True,      # enable ATR-based TP/SL
    # Trailing stop
    "trailing_stop_enabled": True,
    "trailing_stop_atr_multiplier": 2.0,  # trail distance = ATR * multiplier
}

# Sector → ETF → stocks mapping (for sector rotation scanner)
SECTOR_MAP = {
    "Technology":             {"etf": "XLK",  "stocks": ["NVDA","AAPL","MSFT","AVGO","AMD","CRM","ORCL"]},
    "Semiconductors":         {"etf": "SMH",  "stocks": ["NVDA","TSM","AVGO","MU","AMD","ASML","QCOM"]},
    "Financials":             {"etf": "XLF",  "stocks": ["JPM","V","MA","BAC","GS","WFC"]},
    "Healthcare":             {"etf": "XLV",  "stocks": ["LLY","UNH","JNJ","ABBV","TMO","MRK"]},
    "Consumer Discretionary": {"etf": "XLY",  "stocks": ["AMZN","TSLA","HD","MCD","NKE"]},
    "Communication":          {"etf": "XLC",  "stocks": ["META","GOOGL","NFLX","DIS","TMUS"]},
    "Industrials":            {"etf": "XLI",  "stocks": ["CAT","GE","RTX","BA","UNP","HON"]},
    "Energy":                 {"etf": "XLE",  "stocks": ["XOM","CVX","COP","SLB"]},
    "Consumer Staples":       {"etf": "XLP",  "stocks": ["WMT","COST","PG","KO","PEP"]},
    "Biotech":                {"etf": "IBB",  "stocks": ["VRTX","GILD","AMGN","REGN"]},
    "Innovation":             {"etf": "ARKK", "stocks": ["TSLA","COIN","SHOP","ROKU"]},
    "Materials":              {"etf": "XLB",  "stocks": ["LIN","APD","SHW","ECL"]},
    "Utilities":              {"etf": "XLU",  "stocks": ["NEE","DUK","SO","D"]},
    "Real Estate":            {"etf": "XLRE", "stocks": ["PLD","AMT","EQIX","PSA"]},
}

SECTOR_SCAN_CONFIG = {
    # Moskowitz-Grinblatt (1999): 6-month lookback validated for industry momentum
    # ~1month / ~3months / ~6months in trading weeks
    "lookback_weeks": [4, 12, 26],
    # Weights: start equal, will be replaced by backtest-calibrated values
    "momentum_weights": [0.333, 0.334, 0.333],
    "benchmark": "SPY",
    "top_sectors": 3,
    "stocks_per_sector": 5,
    "min_volume": 1_000_000,
    # Stock scoring momentum window (days)
    "stock_momentum_days": 60,  # 2-month for individual stocks (vs 6m sector)
}

# Swing trading EXIT rules (for position-aware cron)
SWING_EXIT_CONFIG = {
    "max_hold_days": 10,           # SELL if held > N days and profit < min_profit_pct
    "min_profit_pct": 0.01,        # 1% minimum profit after max_hold_days
    "stop_loss_pct": -0.03,        # -3% hard stop loss
    "take_profit_pct": 0.05,       # +5% take profit
    "consecutive_miss_limit": 3,   # SELL after N consecutive scan misses
}

# Transaction costs by platform (bps → decimal)
TRANSACTION_COSTS = {
    "alpaca": 0.0000,           # 0 bps (commission-free, but ~1bp spread)
    "kis": 0.0025,              # 25 bps (0.25%)
    "binance_spot_taker": 0.0010,   # 10 bps
    "binance_spot_maker": 0.0010,   # 10 bps (0.075% with BNB)
    "binance_futures_taker": 0.0005,  # 5 bps
    "binance_futures_maker": 0.0002,  # 2 bps
    "default": 0.0015,          # 15 bps conservative default
}

# Crypto trading config
CRYPTO_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT",
    "DOT/USDT", "MATIC/USDT", "LINK/USDT", "NEAR/USDT",
    "PAXG/USDT",  # Gold RWA token (1 PAXG = 1 troy oz gold)
]

CRYPTO_RISK_CONFIG = {
    "take_profit_pct": 0.06,       # 6% TP (higher vol than equities)
    "stop_loss_pct": -0.05,        # -5% hard SL (was -10%, too wide for swing)
    "max_position_pct": 0.15,      # 15% max single position
    "max_exposure_pct": 0.60,      # 60% max total (crypto is riskier)
    "kelly_fraction": 0.15,        # conservative Kelly for crypto
    "trailing_stop_atr_multiplier": 2.5,
    "max_hold_hours": 48,          # time stop: 48h max for swing trades
}

# Regime Blend config — validated via backtest_v2.py (daily bars)
# Sharpe 1.65, Alpha +9.2% vs BTC, MDD 35.2% (< BTC 37.0%)
REGIME_BLEND_CONFIG = {
    "tickers": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PAXG/USDT"],
    "ohlcv_timeframe": "1d",
    "trail_pct": 0.12,
    "max_exposure": 0.70,
    "position_pct": 0.30,
    "rebalance_days": 3,
    "dd_trigger": 0.15,
    "portfolio_dd_trigger": 0.12,
    "tx_cost": 0.001,
    "trail_activation_pct": 0.08,
    "vol_window": 20,
    "trend_window": 50,
    "trending_rsi_threshold": 50,
    "trending_momentum_threshold": 0.02,
    "ranging_bb_threshold": 1.02,
    "ranging_rsi_threshold": 40,
    "trending_exit_momentum": -0.05,
    "ranging_exit_bb_factor": 0.98,
    "unknown_rsi_threshold": 55,
    "unknown_momentum_threshold": 0.05,
    "ohlcv_lookback_days": 100,
}

# Swing Regime Blend — multi-TF enabled (1h primary, 15m/4h auxiliary)
# Tighter stops + faster cycle for active swing trading (target: 1-3 trades/day)
SWING_BLEND_CONFIG = {
    **REGIME_BLEND_CONFIG,
    "ohlcv_timeframe": "1h",           # 4h → 1h (more frequent data)
    "position_pct": 0.20,              # smaller per-trade (higher frequency)
    "trail_pct": 0.06,                 # tighter trail
    "trail_activation_pct": 0.04,      # faster trail activation
    "dd_trigger": 0.10,
    "portfolio_dd_trigger": 0.10,
    # Windows adjusted for 1h bars (24 bars/day)
    "vol_window": 48,           # 48 x 1h = 2 days
    "trend_window": 96,         # 96 x 1h = 4 days
    "ohlcv_lookback_days": 60,  # 60 days = 1440 bars
}

# LightGBM short-timeframe signal config
# Based on: arXiv 2511.00665 (TA+ML), 2503.18096 (Informer HF), 2309.00626 (Ensemble DRL)
LGBM_SIGNAL_CONFIG = {
    "tickers": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "train_window_days": 90,      # rolling train window
    "retrain_interval_days": 7,   # retrain weekly
    "predict_horizon_bars": 16,   # 16 x 15min = 4 hours ahead
    "bar_interval": "15m",        # 15-minute candles
    "min_train_samples": 2000,    # minimum bars for training
    "buy_threshold": 0.58,        # predict proba > threshold → BUY signal
    "sell_threshold": 0.42,       # predict proba < threshold → SELL signal
    "lgbm_params": {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 50,
        "verbose": -1,
    },
    "tx_cost": 0.001,
    # Ensemble with Regime Blend: both must agree
    "ensemble_mode": "vote",      # "vote" = 2/2 agree, "lgbm_only", "blend_only"
    # Risk params (inherit from REGIME_BLEND_CONFIG)
    "max_exposure": 0.70,
    "position_pct": 0.25,         # smaller per-position for higher frequency
    "dd_trigger": 0.12,
    "trail_pct": 0.10,
    "trail_activation_pct": 0.06,
}

# Server config
SERVER_CONFIG = {
    "host": "127.0.0.1",
    "port": 8787,
}
