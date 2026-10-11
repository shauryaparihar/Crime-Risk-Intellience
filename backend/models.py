from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

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
    "derived_data",
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


def regression_models():
    return {
        "lr": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())]),
        "ridge": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", Ridge(alpha=1.0))]),
        "rf": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=1,
            random_state=42
        ))]),
    }


def classification_models():
    return {
        "logit": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))]),
        "rf": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=1,
            random_state=42
        ))]),
    }


def safe_cv_regression(model, X, y):
    n = len(X)
    if n < 4:
        return {"cv_mae_mean": None, "cv_mae_std": None, "cv_r2_mean": None, "cv_r2_std": None}

    folds = min(5, n)
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    mae_scores = -cross_val_score(clone(model), X, y, cv=cv, scoring="neg_mean_absolute_error")
    r2_scores = cross_val_score(clone(model), X, y, cv=cv, scoring="r2")
    return {
        "cv_mae_mean": float(mae_scores.mean()),
        "cv_mae_std": float(mae_scores.std()),
        "cv_r2_mean": float(r2_scores.mean()),
        "cv_r2_std": float(r2_scores.std()),
    }


def train_regression(X, y, models_dir=MODELS_DIR):
    models = regression_models()
    y_log = np.log1p(y)

    metrics = {}
    fitted = {}

    for name, model in models.items():
        cvm = safe_cv_regression(model, X, y_log)
        model.fit(X, y_log)
        pred = np.expm1(model.predict(X))
        metrics[name] = {
            **cvm,
            "train_mae": float(mean_absolute_error(y, pred)),
            "train_r2": float(r2_score(y, pred)),
        }
        fitted[name] = model

    joblib.dump(fitted["lr"], Path(models_dir) / "reg_lr.pkl")
    joblib.dump(fitted["ridge"], Path(models_dir) / "reg_ridge.pkl")
    joblib.dump(fitted["rf"], Path(models_dir) / "reg_rf.pkl")
    joblib.dump(list(X.columns), Path(models_dir) / "reg_feat.pkl")
    joblib.dump(metrics, Path(models_dir) / "reg_metrics.pkl")

    return metrics


def train_crime_type(X, y, models_dir=MODELS_DIR):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    models = classification_models()
    rf = models["rf"]
    logit = models["logit"]

    rf.fit(X, y_enc)
    logit.fit(X, y_enc)

    rf_pred = rf.predict(X)
    logit_pred = logit.predict(X)

    metrics = {
        "rf_acc_train": float(accuracy_score(y_enc, rf_pred)),
        "rf_f1_train": float(f1_score(y_enc, rf_pred, average="macro")),
        "logit_acc_train": float(accuracy_score(y_enc, logit_pred)),
        "logit_f1_train": float(f1_score(y_enc, logit_pred, average="macro")),
        "classes": list(le.classes_),
    }

    joblib.dump(rf, Path(models_dir) / "type_rf.pkl")
    joblib.dump(logit, Path(models_dir) / "type_logit.pkl")
    joblib.dump(le, Path(models_dir) / "type_enc.pkl")
    joblib.dump(list(X.columns), Path(models_dir) / "type_feat.pkl")
    joblib.dump(metrics, Path(models_dir) / "type_metrics.pkl")

    return metrics


def train_risk(X, y, models_dir=MODELS_DIR):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    models = classification_models()
    rf = models["rf"]
    logit = models["logit"]

    rf.fit(X, y_enc)
    logit.fit(X, y_enc)

    rf_pred = rf.predict(X)
    logit_pred = logit.predict(X)

    metrics = {
        "rf_acc_train": float(accuracy_score(y_enc, rf_pred)),
        "rf_f1_train": float(f1_score(y_enc, rf_pred, average="macro")),
        "logit_acc_train": float(accuracy_score(y_enc, logit_pred)),
        "logit_f1_train": float(f1_score(y_enc, logit_pred, average="macro")),
        "classes": list(le.classes_),
    }

    joblib.dump(rf, Path(models_dir) / "risk_rf.pkl")
    joblib.dump(logit, Path(models_dir) / "risk_logit.pkl")
    joblib.dump(le, Path(models_dir) / "risk_enc.pkl")
    joblib.dump(list(X.columns), Path(models_dir) / "risk_feat.pkl")
    joblib.dump(metrics, Path(models_dir) / "risk_metrics.pkl")

    return metrics


def train_all(data_path=DATA_PATH, models_dir=MODELS_DIR):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    df = load(data_path)
    cols = detect(df)

    state_year = build_state_year_table(df, cols)
    X_reg, y_reg, state_feat = build_state_features(state_year, state_col=cols["state"])
    X_type, y_type, crime_cols = build_crime_type_dataset(df, cols)

    state_feat_with_target = state_feat.copy()
    X_risk, y_risk = build_risk_dataset(state_feat_with_target)

    reg_metrics = train_regression(X_reg, y_reg, models_dir)
    type_metrics = train_crime_type(X_type, y_type, models_dir)
    risk_metrics = train_risk(X_risk, y_risk, models_dir)

    artifacts = {
        "state_year": state_year,
        "state_feature_table": state_feat_with_target,
        "crime_type_labels": crime_cols,
    }

    joblib.dump(artifacts, Path(models_dir) / "derived_data.pkl")

    return {
        "regression": reg_metrics,
        "crime_type": type_metrics,
        "risk": risk_metrics,
    }


def load_models(models_dir=MODELS_DIR):
    return {name: joblib.load(artifact_path(name, models_dir)) for name in ARTIFACT_NAMES}
