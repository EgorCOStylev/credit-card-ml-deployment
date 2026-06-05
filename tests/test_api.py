"""
API tests. Run with: pytest tests/test_api.py
Requires the Flask service to be running at http://localhost:5000,
OR use the Flask test client (no server needed) via the client fixture.
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from api import app

# Minimal valid feature set (23 features from UCI dataset)
SAMPLE_FEATURES = {
    "limit_bal": 20000, "sex": 2, "education": 2, "marriage": 1, "age": 24,
    "pay_0": 2, "pay_2": 2, "pay_3": -1, "pay_4": -1, "pay_5": -1, "pay_6": -1,
    "bill_amt1": 3913, "bill_amt2": 3102, "bill_amt3": 689,
    "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
    "pay_amt1": 0, "pay_amt2": 689, "pay_amt3": 0,
    "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0,
}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_healthy(self, client):
        data = resp = client.get("/health").get_json()
        assert data["status"] == "healthy"


class TestPredict:
    def test_predict_returns_200(self, client):
        resp = client.post(
            "/predict?model=v1",
            data=json.dumps(SAMPLE_FEATURES),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_predict_response_structure(self, client):
        resp = client.post(
            "/predict?model=v1",
            data=json.dumps(SAMPLE_FEATURES),
            content_type="application/json",
        )
        data = resp.get_json()
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data

    def test_predict_binary_output(self, client):
        resp = client.post(
            "/predict?model=v1",
            data=json.dumps(SAMPLE_FEATURES),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["prediction"] in (0, 1)
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_v2_model(self, client):
        resp = client.post(
            "/predict?model=v2",
            data=json.dumps(SAMPLE_FEATURES),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["model_version"] == "v2"

    def test_predict_missing_features(self, client):
        resp = client.post(
            "/predict?model=v1",
            data=json.dumps({"limit_bal": 10000}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_predict_empty_body(self, client):
        resp = client.post("/predict", content_type="application/json")
        assert resp.status_code == 400

    def test_predict_invalid_json(self, client):
        resp = client.post(
            "/predict",
            data="not json",
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestFeatures:
    def test_features_endpoint(self, client):
        resp = client.get("/features?model=v1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "features" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) == 23
