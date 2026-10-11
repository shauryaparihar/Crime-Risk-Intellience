import pytest

STATE_ENDPOINTS = ["/api/forecast", "/api/crime-pattern", "/api/state-drilldown"]


def test_status_reports_ready(client):
    data = client.get("/api/status").json()
    assert data["models_ready"] is True
    assert data["data_loaded"] is True
    assert data["error"] is None


def test_states_lists_twelve(client):
    states = client.get("/api/states").json()["states"]
    assert len(states) == 12
    assert "Bihar" in states


def test_metrics_endpoint(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    for key in ("training_date", "regression", "classification", "risk", "limitations"):
        assert key in data
    assert data["evaluation"] == "out_of_fold"
    assert data["n_states"] == 12 and data["n_folds"] == 4
    assert "oof_mae_rate_per_100k" in data["regression"]["rf"]
    assert "oof_accuracy" in data["risk"]["rf"]
    assert "train" not in str(data).lower().replace("training_date", "")


def test_forecast_schema(client):
    response = client.get("/api/forecast?state=Bihar")
    assert response.status_code == 200
    data = response.json()
    for key in ("predictions", "uncertainty", "metrics", "shap", "feature_importance"):
        assert key in data
    assert isinstance(data["predictions"]["ensemble"], float)
    assert isinstance(data["predictions"]["linear"], float)
    assert data["evaluation"] == "out_of_fold"
    assert data["uncertainty"]["calibrated"] is False
    assert "tree spread" in data["uncertainty"]["label"]


def test_crime_pattern_schema(client):
    response = client.get("/api/crime-pattern?state=Bihar")
    assert response.status_code == 200
    data = response.json()
    assert "trend" in data and "metrics" in data
    assert isinstance(data["prediction"]["label"], str)


def test_state_drilldown_schema(client):
    data = client.get("/api/state-drilldown?state=Bihar").json()
    assert set(data) == {"districts_2023", "types_2023", "trend"}
    assert len(data["districts_2023"]) > 0


def test_risk_analysis_and_geojson(client):
    risk = client.get("/api/risk-analysis").json()
    assert len(risk["rankings"]) == 12
    geo = client.get("/api/geojson").json()
    assert set(geo) == {"geojson", "rates"}
    assert len(geo["rates"]) == 12


@pytest.mark.parametrize("path", STATE_ENDPOINTS)
def test_unknown_state_is_404(client, path):
    assert client.get(f"{path}?state=Atlantis").status_code == 404


@pytest.mark.parametrize("path", STATE_ENDPOINTS)
def test_missing_state_is_422(client, path):
    response = client.get(path)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_forecast_serves_stored_oof_values(client):
    from api import app
    stored = app.state.runtime.models["backtest_results"]["regression"]["per_state"]["Bihar"]
    pred = client.get("/api/forecast?state=Bihar").json()["predictions"]
    assert pred["ensemble"] == round(stored["ensemble"], 2)
    assert pred["actual"] == round(stored["observed"], 2)


def test_crime_pattern_and_risk_serve_stored_oof_values(client):
    from api import app
    bt = app.state.runtime.models["backtest_results"]
    cp = client.get("/api/crime-pattern?state=Bihar").json()
    assert cp["prediction"]["label"] == bt["crime_type"]["per_state"]["Bihar"]["rf"]["pred"]
    total = sum(p["probability"] for p in cp["prediction"]["probabilities"])
    assert total == pytest.approx(1.0)
    risk = client.get("/api/risk-analysis").json()
    row = next(r for r in risk["rankings"] if r["state"] == "Bihar")
    assert row["Predicted_Risk_Class"] == bt["risk"]["per_state"]["Bihar"]["rf"]["pred"]
