"""Baselines, data fingerprint staleness check, readiness endpoint, evaluation script."""
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import models
from api import create_app

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("trained_impr")
    models.train_all(models.DATA_PATH, d)
    return d


@pytest.fixture(scope="module")
def datasets():
    df = models.load()
    return models.build_datasets(df, models.detect(df))


# ------------------------------------------------------------------ baselines
def test_regression_baselines_match_a_direct_calculation(trained_dir, datasets):
    bt = joblib.load(trained_dir / "backtest_results.pkl")
    base = bt["regression"]["metrics"]["baselines"]
    X, y = datasets["X_reg"].loc[sorted(datasets["X_reg"].index)], datasets["y_reg"]
    y = y.loc[X.index]
    last_year = np.abs(X["rate_2022"].to_numpy() - y.to_numpy()).mean()
    assert base["last_year_rate"]["mae_rate_per_100k"] == pytest.approx(last_year)
    others = (y.sum() - y.to_numpy()) / (len(y) - 1)
    assert base["mean_of_other_states"]["mae_rate_per_100k"] == pytest.approx(np.abs(others - y.to_numpy()).mean())


def test_beats_flags_agree_with_the_numbers(trained_dir):
    bt = joblib.load(trained_dir / "backtest_results.pkl")
    reg = bt["regression"]["metrics"]
    best = min(v["mae_rate_per_100k"] for v in reg["baselines"].values())
    assert reg["vs_baseline"]["best_value"] == pytest.approx(best)
    for name, flag in reg["vs_baseline"]["beats"].items():
        assert flag == (reg[name]["oof_mae_rate_per_100k"] < best)
    for task in ("risk", "crime_type"):
        m = bt[task]["metrics"]
        top = max(v["accuracy"] for v in m["baselines"].values())
        for name, flag in m["vs_baseline"]["beats"].items():
            assert flag == (m[name]["oof_accuracy"] > top)


def test_classification_baselines_are_valid_accuracies(trained_dir):
    bt = joblib.load(trained_dir / "backtest_results.pkl")
    for task in ("risk", "crime_type"):
        for v in bt[task]["metrics"]["baselines"].values():
            assert 0.0 <= v["accuracy"] <= 1.0


def test_api_exposes_baselines(client):
    metrics = client.get("/api/metrics").json()
    for key in ("regression", "classification", "risk"):
        assert "baselines" in metrics[key] and "vs_baseline" in metrics[key]
    assert len(metrics["data_fingerprint"]) == 12
    assert "scikit-learn" in metrics["library_versions"]
    assert "baselines" in client.get("/api/forecast?state=Bihar").json()["metrics"]


# --------------------------------------------------- data fingerprint / staleness
def _start(models_dir, data_path):
    with TestClient(create_app(data_path=data_path, models_dir=models_dir)) as c:
        return c.get("/api/status").json()


def test_unchanged_data_does_not_retrain(trained_dir, tmp_path):
    d = tmp_path / "same"
    shutil.copytree(trained_dir, d)
    before = (d / "backtest_results.pkl").stat().st_mtime_ns
    assert _start(d, models.DATA_PATH)["models_ready"] is True
    assert (d / "backtest_results.pkl").stat().st_mtime_ns == before


def test_changed_data_triggers_a_retrain(trained_dir, tmp_path):
    d = tmp_path / "changed"
    shutil.copytree(trained_dir, d)
    csv = tmp_path / "crime_data.csv"
    shutil.copyfile(models.DATA_PATH, csv)
    assert models.data_fingerprint(csv) == models.data_fingerprint(models.DATA_PATH)  # a copy has the same fingerprint
    lines = csv.read_text().splitlines()
    parts = lines[1].split(",")
    parts[4] = str(int(parts[4]) + 1)  # Cases_Reported of the first row
    lines[1] = ",".join(parts)
    csv.write_text("\n".join(lines) + "\n")
    new_hash = models.data_fingerprint(csv)
    assert new_hash != models.data_fingerprint(models.DATA_PATH)
    before = (d / "backtest_results.pkl").stat().st_mtime_ns
    assert _start(d, csv)["models_ready"] is True
    assert (d / "backtest_results.pkl").stat().st_mtime_ns != before
    assert joblib.load(d / "backtest_results.pkl")["data_sha256"] == new_hash


def test_artifacts_without_a_fingerprint_are_rebuilt_once(trained_dir, tmp_path):
    d = tmp_path / "old"
    shutil.copytree(trained_dir, d)
    bt = joblib.load(d / "backtest_results.pkl")
    del bt["data_sha256"]
    joblib.dump(bt, d / "backtest_results.pkl")
    assert _start(d, models.DATA_PATH)["models_ready"] is True
    assert joblib.load(d / "backtest_results.pkl")["data_sha256"] == models.data_fingerprint(models.DATA_PATH)


def test_library_version_change_is_detected(trained_dir):
    loaded = models.load_models(trained_dir)
    assert models.stale_reason(loaded, models.DATA_PATH) is None
    loaded["backtest_results"]["library_versions"] = {**loaded["backtest_results"]["library_versions"], "scikit-learn": "0.0.0"}
    assert "scikit-learn" in models.stale_reason(loaded, models.DATA_PATH)


# ------------------------------------------------------------------ readiness
def test_ready_is_200_when_serving(client):
    r = client.get("/api/ready")
    assert r.status_code == 200 and r.json() == {"ready": True}


def test_ready_is_503_but_status_stays_200_after_failed_startup(tmp_path):
    app = create_app(data_path=tmp_path / "missing.csv", models_dir=tmp_path / "m")
    with TestClient(app) as c:
        assert c.get("/api/status").status_code == 200
        assert c.get("/api/status").json()["models_ready"] is False
        r = c.get("/api/ready")
        assert r.status_code == 503
        assert "Traceback" not in r.text


# --------------------------------------------------------- evaluation script
def test_evaluation_report_runs_and_is_consistent():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import evaluation_report as er
    finally:
        sys.path.remove(str(SCRIPTS))
    report = er.compute_report(n_seeds=2, n_boot=50)
    assert report["published_seed"] == 42 and report["seeds"]["n_seeds"] == 2
    lo, hi = report["bootstrap"]["rf_mae_95"]
    assert 0 <= lo <= hi
    assert len(report["rf_error_by_state"]) == 12
    text = er.format_report(report)
    assert "Baselines" in text and ("beats" in text or "does NOT beat" in text)
