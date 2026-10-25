"""API tests: run with `uv run pytest` from the repo root (app.py loads model.joblib from the cwd)."""

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

VALID = {
    "machine_type": "L",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500,
    "torque": 40.0,
    "tool_wear": 100,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_reading():
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["failure"] == (body["failure_probability"] >= 0.5)


def test_predict_high_risk_reading_is_flagged():
    # High torque + heavy tool wear: the overstrain region where AI4I failures cluster.
    risky = {**VALID, "rotational_speed": 1400, "torque": 60.0, "tool_wear": 220}
    assert client.post("/predict", json=risky).json()["failure"] is True


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("machine_type", "X"),          # not one of L/M/H
        ("air_temperature", 27.0),      # Celsius instead of Kelvin
        ("rotational_speed", 0),        # stopped machine is out of scope
        ("torque", -5.0),               # negative torque
        ("tool_wear", "lots"),          # not a number
    ],
)
def test_predict_rejects_bad_input(field, bad_value):
    response = client.post("/predict", json={**VALID, field: bad_value})
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incomplete = {k: v for k, v in VALID.items() if k != "torque"}
    assert client.post("/predict", json=incomplete).status_code == 422
