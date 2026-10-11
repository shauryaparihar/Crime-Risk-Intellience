import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import models
from api import create_app

BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory):
    """A complete artifact set, trained once and copied per test."""
    d = tmp_path_factory.mktemp("trained")
    models.train_all(models.DATA_PATH, d)
    return d


def test_paths_do_not_depend_on_cwd():
    code = (
        "import models; "
        "assert models.DATA_PATH.is_file(), models.DATA_PATH; "
        "assert models.GEOJSON_PATH.is_file(), models.GEOJSON_PATH; "
        "assert models.DATA_PATH.is_absolute() and models.MODELS_DIR.is_absolute()"
    )
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR)}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd="/", env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_missing_artifacts_detects_partial_set(trained_dir, tmp_path):
    assert models.missing_artifacts(tmp_path) == models.ARTIFACT_NAMES
    assert models.missing_artifacts(trained_dir) == []
    partial = tmp_path / "partial"
    shutil.copytree(trained_dir, partial)
    (partial / "type_enc.pkl").unlink()
    (partial / "risk_feat.pkl").unlink()
    assert set(models.missing_artifacts(partial)) == {"type_enc", "risk_feat"}


def test_required_list_matches_what_training_writes(trained_dir):
    written = {p.stem for p in trained_dir.glob("*.pkl")}
    assert written == set(models.ARTIFACT_NAMES)


def test_lifespan_trains_then_becomes_ready_without_restart(tmp_path):
    app = create_app(models_dir=tmp_path / "fresh")
    with TestClient(app) as c:
        status = c.get("/api/status").json()
        assert status["models_ready"] is True
        assert c.get("/api/forecast?state=Bihar").status_code == 200
    assert models.missing_artifacts(tmp_path / "fresh") == []


def test_partial_artifacts_trigger_full_rebuild(trained_dir, tmp_path):
    d = tmp_path / "partial"
    shutil.copytree(trained_dir, d)
    (d / "type_enc.pkl").unlink()
    with TestClient(create_app(models_dir=d)) as c:
        assert c.get("/api/status").json()["models_ready"] is True
        assert c.get("/api/crime-pattern?state=Bihar").status_code == 200
    assert models.missing_artifacts(d) == []


def test_corrupt_artifact_is_retrained(trained_dir, tmp_path):
    d = tmp_path / "corrupt"
    shutil.copytree(trained_dir, d)
    (d / "reg_rf.pkl").write_bytes(b"not a pickle")
    with TestClient(create_app(models_dir=d)) as c:
        assert c.get("/api/status").json()["models_ready"] is True


def test_init_failure_gives_safe_503(tmp_path):
    app = create_app(data_path=tmp_path / "missing.csv", models_dir=tmp_path / "m")
    with TestClient(app) as c:
        status = c.get("/api/status").json()
        assert status["models_ready"] is False
        assert status["error"] == "Initialization failed (FileNotFoundError)"
        for path in ["/api/states", "/api/forecast?state=Bihar", "/api/crime-pattern?state=Bihar",
                     "/api/state-drilldown?state=Bihar", "/api/risk-analysis", "/api/geojson", "/api/metrics"]:
            r = c.get(path)
            assert r.status_code == 503, path
            assert "Traceback" not in r.text and str(tmp_path) not in r.text


def test_bad_csv_schema_fails_safely(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("State,Year\nBihar,2020\n")
    with TestClient(create_app(data_path=bad, models_dir=tmp_path / "m")) as c:
        status = c.get("/api/status").json()
        assert status["models_ready"] is False
        assert status["error"] == "Initialization failed (ValueError)"
        assert c.get("/api/states").status_code == 503
