from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# All paths hang off this file's location, never the shell's current directory.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crime_data.csv"
MODELS_DIR = BASE_DIR / "models"
GEOJSON_PATH = BASE_DIR / "india_state.geojson"

# Single source of truth for the artifacts the app needs before it can be "ready".
# Each name is both the key returned by load_models() and the .pkl file stem.
ARTIFACT_NAMES = [
    "reg_lr", "reg_ridge", "reg_rf", "reg_feat", "reg_metrics",
    "type_rf", "type_logit", "type_enc", "type_feat", "type_metrics",
    "risk_rf", "risk_logit", "risk_enc", "risk_feat", "risk_metrics",
    "derived_data", "backtest_results",
]

REQUIRED_COLUMNS = [
    "State", "District", "Year", "Crime_Type", "Cases_Reported", "Population",
]
REQUIRED_YEARS = [2019, 2020, 2021, 2022, 2023]


def artifact_path(name, models_dir=MODELS_DIR):
    return Path(models_dir) / f"{name}.pkl"


def missing_artifacts(models_dir=MODELS_DIR):
    return [n for n in ARTIFACT_NAMES if not artifact_path(n, models_dir).is_file()]


def validate_schema(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    absent_years = [y for y in REQUIRED_YEARS if y not in set(df["Year"].unique())]
    if absent_years:
        raise ValueError(f"CSV has no rows for required years: {absent_years}")


def load(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    validate_schema(df)
    df = df[df["Year"].between(2019, 2023)].copy()
    return df


def detect(df):
    cols = {}
    for c in df.columns:
        s = c.lower().strip()
        if "state" in s and "district" not in s:
            cols["state"] = c
        elif "district" in s:
            cols["district"] = c
        elif "year" in s:
            cols["year"] = c
        elif "crime_type" in s:
            cols["crime_type"] = c
        elif "cases_reported" in s:
            cols["total"] = c
        elif "population" in s:
            cols["population"] = c
        elif "crime_rate_per_100k" in s:
            cols["rate"] = c
    return cols


def build_state_year_table(df, cols):
    pop_df = (
        df[[cols["state"], cols["district"], cols["year"], cols["population"]]]
        .drop_duplicates(subset=[cols["state"], cols["district"], cols["year"]])
        .groupby([cols["state"], cols["year"]], as_index=False)[cols["population"]]
        .sum()
    )

    cases_df = (
        df.groupby([cols["state"], cols["year"]], as_index=False)[cols["total"]]
        .sum()
    )

    out = cases_df.merge(pop_df, on=[cols["state"], cols["year"]], how="left")
    out["State_Crime_Rate_per_100k"] = (out[cols["total"]] / out[cols["population"]]) * 100000
    return out


def build_state_features(state_year, state_col="State"):
    rate = state_year.pivot_table(
        index=state_col,
        columns="Year",
        values="State_Crime_Rate_per_100k",
        aggfunc="first"
    )

    rate.columns = [f"rate_{int(c)}" for c in rate.columns]
    rate = rate.sort_index(axis=1)
    rate = rate.dropna(subset=["rate_2019", "rate_2020", "rate_2021", "rate_2022", "rate_2023"], how="any")

    feat = rate.copy()
    feat["avg_19_22"] = feat[["rate_2019", "rate_2020", "rate_2021", "rate_2022"]].mean(axis=1)
    feat["trend_19_22"] = feat["rate_2022"] - feat["rate_2019"]
    feat["momentum_21_22"] = feat["rate_2022"] - feat["rate_2021"]
    feat["volatility_19_22"] = feat[["rate_2019", "rate_2020", "rate_2021", "rate_2022"]].std(axis=1)
    feat["growth_21_22"] = (feat["rate_2022"] - feat["rate_2021"]) / (feat["rate_2021"] + 1e-9)

    X = feat.drop(columns=["rate_2023"])
    y = feat["rate_2023"]
    return X, y, feat


def build_crime_type_dataset(df, cols):
    grp = (
        df.groupby([cols["state"], cols["year"], cols["crime_type"]], as_index=False)[cols["total"]]
        .sum()
    )

    wide = grp.pivot_table(
        index=[cols["state"], cols["year"]],
        columns=cols["crime_type"],
        values=cols["total"],
        aggfunc="sum",
        fill_value=0
    )

    wide.columns = [str(c) for c in wide.columns]
    wide = wide.sort_index(axis=1)
    share = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    share = share.reset_index()

    train = share[share[cols["year"]].between(2019, 2022)].copy()
    test = share[share[cols["year"]] == 2023].copy()

    crime_cols = [c for c in share.columns if c not in [cols["state"], cols["year"]]]

    train_avg = train.groupby(cols["state"])[crime_cols].mean().add_prefix("avg_")
    train_last = train[train[cols["year"]] == 2022].set_index(cols["state"])[crime_cols].add_prefix("last_")
    train_prev = train[train[cols["year"]] == 2021].set_index(cols["state"])[crime_cols].add_prefix("prev_")

    X = train_avg.join(train_last, how="inner").join(train_prev, how="inner")
    for c in crime_cols:
        X[f"delta_{c}"] = X[f"last_{c}"] - X[f"prev_{c}"]

    test_last = test.set_index(cols["state"])[crime_cols]
    y = test_last.idxmax(axis=1)

    X = X.loc[X.index.intersection(y.index)]
    y = y.loc[X.index]

    return X, y, crime_cols


def build_risk_dataset(state_features):
    feat = state_features.copy()
    q_low = feat["rate_2023"].quantile(0.33)
    q_high = feat["rate_2023"].quantile(0.66)

    def make_label(v):
        if v <= q_low:
            return "low"
        if v >= q_high:
            return "high"
        return "medium"

    y = feat["rate_2023"].apply(make_label)
    X = feat.drop(columns=["rate_2023"])
    return X, y


BACKTEST_SEED = 42
BACKTEST_FOLDS = 4

LIMITATIONS = [
    "Only 12 states: every metric is based on 12 held-out predictions.",
    "Dominant-crime classes are imbalanced and one class has a single state, "
    "so stratified folds are impossible and that class cannot be learned when held out.",
    "Dataset is an unverified NCRB-style sample, not official data.",
    "RF tree spread is the 10th-90th percentile of individual trees, not a calibrated prediction interval.",
    "Risk classes are terciles of the 2023 rate across all 12 states (labels use every state's 2023 value).",
]


def _numeric_pipeline(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def regression_models():
    return {
        "lr": _numeric_pipeline(LinearRegression()),
        "ridge": _numeric_pipeline(Ridge(alpha=1.0), scale=True),
        "rf": _numeric_pipeline(RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=1,
            random_state=42
        )),
    }


def classification_models():
    return {
        "logit": _numeric_pipeline(LogisticRegression(max_iter=1000, random_state=42), scale=True),
        "rf": _numeric_pipeline(RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=1,
            random_state=42
        )),
    }


def align_proba(fold_classes, proba, n_classes):
    """Place a fold model's predict_proba columns into the full global class list.
    A fold can lack a class entirely (e.g. the singleton one); that column is 0."""
    out = np.zeros((proba.shape[0], n_classes))
    for col, cls in enumerate(fold_classes):
        out[:, int(cls)] = proba[:, col]
    return out


def make_folds(states, n_splits=BACKTEST_FOLDS, seed=BACKTEST_SEED):
    """Deterministic state-level folds: {state: fold_id}, plus per-fold train/valid states."""
    states = list(states)
    if len(states) < n_splits:
        raise ValueError(f"Need at least {n_splits} states for {n_splits}-fold backtest, got {len(states)}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of, train_states, valid_states = {}, {}, {}
    for fold, (tr, va) in enumerate(kf.split(states)):
        train_states[fold] = [states[i] for i in tr]
        valid_states[fold] = [states[i] for i in va]
        for i in va:
            fold_of[states[i]] = fold
    return fold_of, train_states, valid_states


def _regression_backtest(X, y, fold_of, train_states, valid_states):
    feats = list(X.columns)
    per_state = {}
    for fold in sorted(train_states):
        tr, va = train_states[fold], valid_states[fold]
        y_log = np.log1p(y.loc[tr])
        fitted = {}
        for name, model in regression_models().items():
            model.fit(X.loc[tr], y_log)
            fitted[name] = model
        rf = fitted["rf"]
        rf_core = rf.named_steps["model"]
        X_va_imp = pd.DataFrame(rf.named_steps["imputer"].transform(X.loc[va]), index=va, columns=feats)
        shap_vals = shap.TreeExplainer(rf_core).shap_values(X_va_imp.to_numpy())
        for k, state in enumerate(va):
            preds = {n: float(np.expm1(m.predict(X.loc[[state]]))[0]) for n, m in fitted.items()}
            preds["ensemble"] = float(np.mean([preds["lr"], preds["ridge"], preds["rf"]]))
            tree_preds = np.array([np.expm1(t.predict(X_va_imp.loc[[state]].to_numpy()))[0] for t in rf_core.estimators_])
            observed = float(y.loc[state])
            per_state[state] = {
                "fold": fold,
                "observed": observed,
                **preds,
                "abs_error": {n: abs(v - observed) for n, v in preds.items()},
                "rf_tree_spread": {
                    "p10": float(np.percentile(tree_preds, 10)),
                    "p90": float(np.percentile(tree_preds, 90)),
                    "std": float(np.std(tree_preds)),
                },
                "shap": [{"feature": f, "value": float(v)} for f, v in zip(feats, shap_vals[k])],
                "feature_importance": [
                    {"feature": f, "importance": float(v)} for f, v in zip(feats, rf_core.feature_importances_)
                ],
            }
    states = sorted(per_state)
    obs = np.array([per_state[s]["observed"] for s in states])
    metrics = {}
    for name in ["lr", "ridge", "rf", "ensemble"]:
        pred = np.array([per_state[s][name] for s in states])
        metrics[name] = {
            "oof_mae_rate_per_100k": float(mean_absolute_error(obs, pred)),
            "oof_r2": float(r2_score(obs, pred)),
        }
    return per_state, metrics


def _classification_backtest(X, y, fold_of, train_states, valid_states):
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index)
    n_classes = len(le.classes_)
    labels = list(range(n_classes))
    per_state = {}
    for fold in sorted(train_states):
        tr, va = train_states[fold], valid_states[fold]
        for name, model in classification_models().items():
            model.fit(X.loc[tr], y_enc.loc[tr])
            proba = align_proba(model.classes_, model.predict_proba(X.loc[va]), n_classes)
            for k, state in enumerate(va):
                rec = per_state.setdefault(state, {
                    "fold": fold,
                    "actual": str(y.loc[state]),
                })
                rec[name] = {
                    "pred": str(le.classes_[int(np.argmax(proba[k]))]),
                    "probabilities": {str(c): float(p) for c, p in zip(le.classes_, proba[k])},
                }
    states = sorted(per_state)
    actual = [per_state[s]["actual"] for s in states]
    metrics = {}
    for name in ["rf", "logit"]:
        pred = [per_state[s][name]["pred"] for s in states]
        metrics[name] = {
            "oof_accuracy": float(accuracy_score(actual, pred)),
            "oof_macro_f1": float(f1_score(
                actual, pred, labels=list(le.classes_), average="macro", zero_division=0
            )),
        }
    meta = {
        "classes": [str(c) for c in le.classes_],
        "class_counts": {str(c): int((y == c).sum()) for c in le.classes_},
        "stratified": False,
    }
    return per_state, metrics, meta


def run_backtest(X_reg, y_reg, X_type, y_type, X_risk, y_risk):
    """Four-fold, state-level out-of-fold backtest shared by all three tasks.
    Every stored prediction for a state comes from a model that never saw that state."""
    states = sorted(set(X_reg.index) & set(X_type.index) & set(X_risk.index))
    fold_of, train_states, valid_states = make_folds(states)

    reg, reg_metrics = _regression_backtest(X_reg.loc[states], y_reg.loc[states], fold_of, train_states, valid_states)
    typ, type_metrics, type_meta = _classification_backtest(X_type.loc[states], y_type.loc[states], fold_of, train_states, valid_states)
    risk, risk_metrics, risk_meta = _classification_backtest(X_risk.loc[states], y_risk.loc[states], fold_of, train_states, valid_states)

    meta = {
        "evaluation": "out_of_fold",
        "n_states": len(states),
        "n_folds": BACKTEST_FOLDS,
        "seed": BACKTEST_SEED,
        "limitations": LIMITATIONS,
    }
    return {
        **meta,
        "fold_of": fold_of,
        "fold_train_states": train_states,
        "fold_valid_states": valid_states,
        "regression": {"per_state": reg, "metrics": {"meta": {**meta, "unit": "crime_rate_per_100k"}, **reg_metrics}},
        "crime_type": {"per_state": typ, "metrics": {"meta": {**meta, **type_meta}, **type_metrics}},
        "risk": {"per_state": risk, "metrics": {"meta": {**meta, **risk_meta}, **risk_metrics}},
    }


def fit_final_models(X_reg, y_reg, X_type, y_type, X_risk, y_risk, models_dir=MODELS_DIR):
    """Full-data models and their feature lists / encoders. These are NOT used as
    evaluation evidence; all reported quality comes from run_backtest()."""
    models_dir = Path(models_dir)
    y_log = np.log1p(y_reg)
    for name, model in regression_models().items():
        model.fit(X_reg, y_log)
        joblib.dump(model, models_dir / f"reg_{name}.pkl")
    joblib.dump(list(X_reg.columns), models_dir / "reg_feat.pkl")

    for prefix, X, y in [("type", X_type, y_type), ("risk", X_risk, y_risk)]:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        for name, model in classification_models().items():
            model.fit(X, y_enc)
            joblib.dump(model, models_dir / f"{prefix}_{name}.pkl")
        joblib.dump(le, models_dir / f"{prefix}_enc.pkl")
        joblib.dump(list(X.columns), models_dir / f"{prefix}_feat.pkl")


def train_all(data_path=DATA_PATH, models_dir=MODELS_DIR):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    df = load(data_path)
    cols = detect(df)

    state_year = build_state_year_table(df, cols)
    X_reg, y_reg, state_feat = build_state_features(state_year, state_col=cols["state"])
    X_type, y_type, crime_cols = build_crime_type_dataset(df, cols)
    X_risk, y_risk = build_risk_dataset(state_feat)

    backtest = run_backtest(X_reg, y_reg, X_type, y_type, X_risk, y_risk)
    fit_final_models(X_reg, y_reg, X_type, y_type, X_risk, y_risk, models_dir)

    joblib.dump(backtest["regression"]["metrics"], models_dir / "reg_metrics.pkl")
    joblib.dump(backtest["crime_type"]["metrics"], models_dir / "type_metrics.pkl")
    joblib.dump(backtest["risk"]["metrics"], models_dir / "risk_metrics.pkl")
    joblib.dump({
        "state_year": state_year,
        "state_feature_table": state_feat,
        "crime_type_labels": crime_cols,
    }, models_dir / "derived_data.pkl")
    # Written last: its presence marks a complete, consistent artifact set.
    joblib.dump(backtest, models_dir / "backtest_results.pkl")

    return {
        "regression": backtest["regression"]["metrics"],
        "crime_type": backtest["crime_type"]["metrics"],
        "risk": backtest["risk"]["metrics"],
    }


def load_models(models_dir=MODELS_DIR):
    return {name: joblib.load(artifact_path(name, models_dir)) for name in ARTIFACT_NAMES}
