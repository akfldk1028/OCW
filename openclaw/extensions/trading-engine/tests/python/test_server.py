"""Tests for server.py FastAPI endpoints."""
import pytest
import sys
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from fastapi.testclient import TestClient
from server import app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "training_status" in data


class TestStatusEndpoint:
    def test_status_returns_info(self, client):
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "model_loaded" in data
        assert "ensemble_weights" in data
        assert "risk_manager_status" in data


class TestTrainEndpoint:
    def test_train_accepts_request(self, client):
        response = client.post("/train", json={
            "tickers": ["AAPL"],
            "lookback_days": 30,
            "total_timesteps": 1000,
        })
        # Training starts in background, should return 202
        assert response.status_code in [200, 202]
        data = response.json()
        assert "training_status" in data

    def test_train_with_defaults(self, client):
        response = client.post("/train", json={})
        # 409 = training already in progress (from previous test), also valid
        assert response.status_code in [200, 202, 409]

    def test_train_invalid_ticker(self, client):
        response = client.post("/train", json={
            "tickers": ["INVALID_TICKER_XYZ123"],
            "lookback_days": 30,
            "total_timesteps": 100,
        })
        # Should handle gracefully
        assert response.status_code in [200, 202, 400, 422]


class TestPredictEndpoint:
    def test_predict_without_model(self, client):
        response = client.post("/predict", json={
            "tickers": ["AAPL"],
        })
        # Without trained model, should still return (zeros or error)
        assert response.status_code in [200, 400, 503]

    def test_predict_accepts_tickers(self, client):
        response = client.post("/predict", json={
            "tickers": ["AAPL", "MSFT"],
        })
        assert response.status_code in [200, 400, 503]


class TestBacktestEndpoint:
    def test_backtest_accepts_request(self, client):
        response = client.post("/backtest", json={
            "tickers": ["AAPL"],
            "start_date": "2025-01-01",
            "end_date": "2025-06-01",
        })
        # May take time or fail without trained model
        assert response.status_code in [200, 400, 503]

    def test_backtest_with_capital(self, client):
        response = client.post("/backtest", json={
            "tickers": ["AAPL"],
            "start_date": "2025-01-01",
            "end_date": "2025-06-01",
            "initial_capital": 50000,
        })
        assert response.status_code in [200, 400, 503]
