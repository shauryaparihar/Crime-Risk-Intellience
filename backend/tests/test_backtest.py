import warnings

import joblib
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

import models


@pytest.fixture(scope="module")
def datasets():
    df = models.load()
    cols = models.detect(df)
    state_year = models.build_state_year_table(df, cols)
    X_reg, y_reg, state_feat = models.build_state_features(state_year, state_col=cols["state"])
    X_type, y_type, _ = models.build_crime_type_dataset(df, cols)
    X_risk, y_risk = models.build_risk_dataset(state_feat)
    return X_reg, y_reg, X_type, y_type, X_risk, y_risk


@pytest.fixture(scope="module")
def backtest(datasets):
    # Convergence problems must fail the test, not be hidden as warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return models.run_backtest(*datasets)


def test_folds_are_deterministic(datasets):
    states = sorted(datasets[0].index)
    assert models.make_folds(states)[0] == models.make_folds(states)[0]


def test_two_runs_are_identical(datasets, backtest):
    again = models.run_backtest(*datasets)
    assert again["fold_of"] == backtest["fold_of"]
    assert again["regression"] == backtest["regression"]
    assert again["crime_type"] == backtest["crime_type"]
    assert again["risk"] == backtest["risk"]


def test_each_state_held_out_exactly_once_and_never_in_its_training(backtest):
    fold_of = backtest["fold_of"]
    states = sorted(datasets_states(backtest))
    assert len(states) == 12
    valid_all = [s for f in backtest["fold_valid_states"].values() for s in f]
    assert sorted(valid_all) == states and len(valid_all) == len(set(valid_all))
    for state, fold in fold_of.items():
        assert state not in backtest["fold_train_states"][fold]
        assert state in backtest["fold_valid_states"][fold]
    for task in ("regression", "crime_type", "risk"):
        per_state = backtest[task]["per_state"]
        assert sorted(per_state) == states
        assert all(per_state[s]["fold"] == fold_of[s] for s in states)


def datasets_states(backtest):
    return backtest["fold_of"].keys()


def test_regression_metrics_are_original_scale_and_match_stored_values(backtest):
    per_state = backtest["regression"]["per_state"]
    states = sorted(per_state)
    obs = np.array([per_state[s]["observed"] for s in states])
    # Observed values are real crime rates per 100k, not log1p values (which would be < ~7).
    assert obs.min() > 10
    for name in ["lr", "ridge", "rf", "ensemble"]:
        pred = np.array([per_state[s][name] for s in states])
        m = backtest["regression"]["metrics"][name]
        assert m["oof_mae_rate_per_100k"] == pytest.approx(mean_absolute_error(obs, pred))
        assert m["oof_r2"] == pytest.approx(r2_score(obs, pred))
        assert np.isfinite(m["oof_mae_rate_per_100k"])


def test_ensemble_is_mean_of_three_models(backtest):
    for rec in backtest["regression"]["per_state"].values():
        assert rec["ensemble"] == pytest.approx(np.mean([rec["lr"], rec["ridge"], rec["rf"]]))


def test_no_training_metrics_are_reported(backtest):
    for task in ("regression", "crime_type", "risk"):
        text = repr(backtest[task]["metrics"]).lower()
        assert "train" not in text and "cv_" not in text


@pytest.mark.parametrize("task", ["crime_type", "risk"])
def test_probabilities_aligned_and_metrics_recomputed(backtest, task):
    classes = backtest[task]["metrics"]["meta"]["classes"]
    per_state = backtest[task]["per_state"]
    states = sorted(per_state)
    actual = [per_state[s]["actual"] for s in states]
    for s in states:
        for name in ("rf", "logit"):
            probs = per_state[s][name]["probabilities"]
            assert list(probs) == classes
            assert sum(probs.values()) == pytest.approx(1.0)
    for name in ("rf", "logit"):
        pred = [per_state[s][name]["pred"] for s in states]
        m = backtest[task]["metrics"][name]
        assert m["oof_accuracy"] == pytest.approx(accuracy_score(actual, pred))
        assert m["oof_macro_f1"] == pytest.approx(
            f1_score(actual, pred, labels=classes, average="macro", zero_division=0)
        )


def test_align_proba_fills_missing_class_with_zero():
    # A fold whose training portion never saw global class 1 and 3.
    proba = np.array([[0.7, 0.3], [0.2, 0.8]])
    out = models.align_proba(np.array([0, 2]), proba, n_classes=4)
    assert out.shape == (2, 4)
    assert out[:, 1].tolist() == [0.0, 0.0] and out[:, 3].tolist() == [0.0, 0.0]
    assert out.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_singleton_class_state_is_scored_without_error(backtest):
    meta = backtest["crime_type"]["metrics"]["meta"]
    singleton = [c for c, n in meta["class_counts"].items() if n == 1]
    assert singleton, "expected the documented singleton dominant-crime class"
    assert meta["stratified"] is False
    holder = [s for s, r in backtest["crime_type"]["per_state"].items() if r["actual"] == singleton[0]]
    # Held out, its class was absent from training: probability for it must be exactly 0.
    for s in holder:
        assert backtest["crime_type"]["per_state"][s]["rf"]["probabilities"][singleton[0]] == 0.0


def test_shap_matches_feature_list(datasets, backtest):
    feats = list(datasets[0].columns)
    for rec in backtest["regression"]["per_state"].values():
        assert [x["feature"] for x in rec["shap"]] == feats
        assert len(rec["feature_importance"]) == len(feats)
        assert all(np.isfinite(x["value"]) for x in rec["shap"])


def test_train_all_writes_complete_set_without_warnings(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        models.train_all(models.DATA_PATH, tmp_path)
    assert models.missing_artifacts(tmp_path) == []
    stored = joblib.load(tmp_path / "backtest_results.pkl")
    assert stored["n_states"] == 12 and stored["n_folds"] == 4
