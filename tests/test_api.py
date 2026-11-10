"""Minimal API contract: one happy path, two validation failures."""

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


def test_valid_request_returns_200():
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200


def test_unknown_machine_type_returns_422():
    response = client.post("/predict", json={**VALID, "machine_type": "X"})
    assert response.status_code == 422


def test_negative_torque_returns_422():
    response = client.post("/predict", json={**VALID, "torque": -5.0})
    assert response.status_code == 422
