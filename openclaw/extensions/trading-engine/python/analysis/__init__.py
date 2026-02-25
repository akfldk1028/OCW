# Lazy imports — avoid pulling heavy deps (gymnasium, transformers, torch)
# when only a single submodule is needed (e.g. runner.py → regime_detector_crypto).
def __getattr__(name):
    _MAP = {
        "DataProcessor": "analysis.data_processor",
        "RegimeDetector": "analysis.regime_detector",
        "CryptoRegimeDetector": "analysis.regime_detector_crypto",
        "EnsembleAgent": "analysis.ensemble_agent",
        "AutoTrader": "analysis.auto_trader",
        "SectorScanner": "analysis.sector_scanner",
        "StockRanker": "analysis.stock_ranker",
        "SentimentScorer": "analysis.sentiment_scorer",
        "FinBERTScorer": "analysis.sentiment_finbert",
    }
    if name in _MAP:
        import importlib
        mod = importlib.import_module(_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'analysis' has no attribute {name!r}")
