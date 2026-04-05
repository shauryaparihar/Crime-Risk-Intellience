import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_metrics_endpoint():
    """Verify metrics endpoint returns 200 and expected keys."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "training_date" in data
    assert "regression" in data
    assert "classification" in data
    assert "risk" in data

def test_forecast_regression():
    """Verify forecast endpoint returns floats for regression predictions."""
    response = client.get("/api/forecast?state=Bihar")
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "ensemble" in data["predictions"]
    assert isinstance(data["predictions"]["ensemble"], float)
    assert isinstance(data["predictions"]["linear"], float)

def test_crime_pattern_classification():
    """Verify crime pattern endpoint returns a valid classification string."""
    response = client.get("/api/crime-pattern?state=Bihar")
    assert response.status_code == 200
    data = response.json()
    assert "trend" in data
    if "prediction" in data:
        assert "label" in data["prediction"]
        assert isinstance(data["prediction"]["label"], str)

def test_invalid_forecast_input():
    """Verify FastAPI automatically returns 422 Unprocessable Entity when required params are missing."""
    response = client.get("/api/forecast")
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
