from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import json
import os
import shap

from models import (
    load,
    detect,
    build_state_year_table,
    build_state_features,
    load_models,
    train_all,
    MODELS_DIR,
)

MODEL_FILES = ["reg_lr.pkl", "reg_ridge.pkl", "reg_rf.pkl", "type_rf.pkl", "risk_rf.pkl"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train models on startup if not present (e.g. fresh Render deploy)."""
    missing = [f for f in MODEL_FILES if not os.path.isfile(os.path.join(MODELS_DIR, f))]
    if missing:
        print(f"[startup] Missing models: {missing}. Training now...")
        train_all()
        print("[startup] Training complete.")
    yield


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Crime Risk API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

allowed_origin = os.getenv("ALLOWED_ORIGIN", "http://localhost:5174")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Initialization
MODELS_READY = all(
    os.path.isfile(os.path.join(MODELS_DIR, f))
    for f in [
        "reg_lr.pkl",
        "reg_ridge.pkl",
        "reg_rf.pkl",
        "type_rf.pkl",
        "risk_rf.pkl",
    ]
)

try:
    df = load()
    cols = detect(df)
    state_year = build_state_year_table(df, cols)
    X_reg_live, y_reg_live, state_feat_live = build_state_features(state_year, state_col=cols["state"])
    models = load_models() if MODELS_READY else {}
except Exception as e:
    print(f"Error initializing data/models: {e}")
    df, cols, state_year, X_reg_live, y_reg_live, state_feat_live, models = None, None, None, None, None, None, {}

GEOJSON_PATH = "india_state.geojson"
geojson_data = None
if os.path.exists(GEOJSON_PATH):
    with open(GEOJSON_PATH, "r") as f:
        geojson_data = json.load(f)

@app.get("/api/status")
@limiter.limit("60/minute")
def status(request: Request):
    return {"models_ready": MODELS_READY, "data_loaded": df is not None}

@app.get("/api/states")
@limiter.limit("60/minute")
def get_states(request: Request):
    if df is None:
        return {"states": []}
    return {"states": sorted(df[cols["state"]].unique().tolist())}

@app.get("/api/forecast")
@limiter.limit("60/minute")
def get_forecast(request: Request, state: str = Query(..., description="The state to forecast for")):
    if not MODELS_READY:
        raise HTTPException(status_code=400, detail="Models not trained")
    
    if state not in X_reg_live.index:
        raise HTTPException(status_code=404, detail="State not found")
        
    X_one = X_reg_live.loc[[state]].reindex(columns=models["reg_feat"], fill_value=0)
    
    lr_pred = float(np.expm1(models["reg_lr"].predict(X_one))[0])
    ridge_pred = float(np.expm1(models["reg_ridge"].predict(X_one))[0])
    rf_pred = float(np.expm1(models["reg_rf"].predict(X_one))[0])
    ensemble_pred = float(np.mean([lr_pred, ridge_pred, rf_pred]))
    
    rf_core = models["reg_rf"].named_steps["model"]
    rf_imputer = models["reg_rf"].named_steps["imputer"]
    
    X_one_imp = rf_imputer.transform(X_one)
    rf_tree_preds = np.array([
        np.expm1(tree.predict(X_one_imp))[0]
        for tree in rf_core.estimators_
    ])
    
    pred_low = float(np.percentile(rf_tree_preds, 10))
    pred_high = float(np.percentile(rf_tree_preds, 90))
    pred_std = float(np.std(rf_tree_preds))
    actual = float(y_reg_live.loc[state]) if state in y_reg_live.index else 0.0
    
    # SHAP
    X_one_df_imp = pd.DataFrame(X_one_imp, index=X_one.index, columns=models["reg_feat"])
    explainer = shap.TreeExplainer(rf_core)
    shap_values = explainer.shap_values(X_one_df_imp)
    shap_list = [{"feature": msg, "value": val} for msg, val in zip(models["reg_feat"], shap_values[0])]
    shap_list = sorted(shap_list, key=lambda x: abs(x["value"]), reverse=True)
    
    feat_imp = [{"feature": f, "importance": i} for f, i in zip(models["reg_feat"], rf_core.feature_importances_)]
    feat_imp = sorted(feat_imp, key=lambda x: x["importance"], reverse=True)
    
    return {
        "predictions": {
            "linear": round(lr_pred, 2),
            "ridge": round(ridge_pred, 2),
            "random_forest": round(rf_pred, 2),
            "ensemble": round(ensemble_pred, 2),
            "actual": round(actual, 2),
            "absolute_error": round(abs(ensemble_pred - actual), 2)
        },
        "uncertainty": {
            "low": round(pred_low, 2),
            "high": round(pred_high, 2),
            "std": round(pred_std, 2)
        },
        "metrics": models["reg_metrics"],
        "shap": shap_list,
        "feature_importance": feat_imp
    }

@app.get("/api/crime-pattern")
@limiter.limit("60/minute")
def get_crime_pattern(request: Request, state: str = Query(...)):
    if not MODELS_READY:
        raise HTTPException(status_code=400, detail="Models not trained")
    
    grp = df.groupby([cols["state"], cols["year"], cols["crime_type"]], as_index=False)[cols["total"]].sum()
    state_sub = grp[grp[cols["state"]] == state].copy()
    pivot = state_sub.pivot_table(index=cols["year"], columns=cols["crime_type"], values=cols["total"], aggfunc="sum", fill_value=0)
    share = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    
    result = {}
    if 2022 in share.index and 2021 in share.index:
        X_type_one = pd.DataFrame(index=[state])
        base_years = [y for y in [2019, 2020, 2021, 2022] if y in share.index]
        for c in share.columns:
            X_type_one[f"avg_{c}"] = float(share.loc[base_years, c].mean())
            X_type_one[f"last_{c}"] = float(share.loc[2022, c])
            X_type_one[f"prev_{c}"] = float(share.loc[2021, c])
            X_type_one[f"delta_{c}"] = float(share.loc[2022, c] - share.loc[2021, c])
            
        X_type_one = X_type_one.reindex(columns=models["type_feat"], fill_value=0)
        
        pred_idx = models["type_rf"].predict(X_type_one)[0]
        pred_label = models["type_enc"].inverse_transform([pred_idx])[0]
        probs = models["type_rf"].predict_proba(X_type_one)[0]
        
        prob_list = [{"type": cls, "probability": p} for cls, p in zip(models["type_enc"].classes_, probs)]
        prob_list.sort(key=lambda x: x["probability"], reverse=True)
        
        result["prediction"] = {
            "label": pred_label,
            "probabilities": prob_list
        }
        
    actual_2023 = df[(df[cols["state"]] == state) & (df[cols["year"]] == 2023)].groupby(cols["crime_type"])[cols["total"]].sum().sort_values(ascending=False)
    if not actual_2023.empty:
        result["actual_2023"] = [{"type": k, "count": int(v)} for k, v in actual_2023.items()]
        
    trend_data = []
    for year in share.index:
        item = {"year": str(year)}
        for c in share.columns:
            item[c] = float(share.loc[year, c])
        trend_data.append(item)
        
    result["trend"] = trend_data
    result["metrics"] = models["type_metrics"]
    
    return result

@app.get("/api/risk-analysis")
@limiter.limit("60/minute")
def get_risk_analysis(request: Request):
    if not MODELS_READY:
        raise HTTPException(status_code=400, detail="Models not trained")
        
    risk_X = state_feat_live.drop(columns=["rate_2023"], errors="ignore").reindex(columns=models["risk_feat"], fill_value=0)
    risk_probs = models["risk_rf"].predict_proba(risk_X)
    risk_pred = models["risk_rf"].predict(risk_X)
    risk_labels = models["risk_enc"].inverse_transform(risk_pred)
    
    risk_df = state_feat_live.copy()
    risk_df["Predicted_Risk_Class"] = risk_labels
    
    classes = list(models["risk_enc"].classes_)
    for i, c in enumerate(classes):
        risk_df[f"prob_{c}"] = risk_probs[:, i]
        
    if {"low", "medium", "high"}.issubset(set(classes)):
        risk_df["Risk_Score"] = risk_df["prob_low"] * 1 + risk_df["prob_medium"] * 2 + risk_df["prob_high"] * 3
    else:
        risk_df["Risk_Score"] = risk_probs.max(axis=1)
        
    risk_df = risk_df.sort_values("Risk_Score", ascending=False)
    
    analysis_results = []
    for state, row in risk_df.iterrows():
        item = {"state": state}
        for k in ["rate_2019", "rate_2020", "rate_2021", "rate_2022", "rate_2023", "trend_19_22", "momentum_21_22", "volatility_19_22", "Predicted_Risk_Class", "Risk_Score"]:
            if k in row:
                v = row[k]
                item[k] = float(v) if pd.notna(v) and isinstance(v, (int, float, np.number)) else v
        analysis_results.append(item)
        
    return {
        "rankings": analysis_results,
        "metrics": models["risk_metrics"]
    }

@app.get("/api/state-drilldown")
@limiter.limit("60/minute")
def get_state_drilldown(request: Request, state: str = Query(...)):
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
        
    state_df = df[df[cols["state"]] == state].copy()
    
    district_2023 = state_df[state_df[cols["year"]] == 2023].groupby(cols["district"])[cols["total"]].sum().sort_values(ascending=False).head(10)
    district_data = [{"district": k, "count": int(v)} for k, v in district_2023.items()]
    
    type_2023 = state_df[state_df[cols["year"]] == 2023].groupby(cols["crime_type"])[cols["total"]].sum().sort_values(ascending=False)
    type_data = [{"type": k, "count": int(v)} for k, v in type_2023.items()]
    
    state_rate = state_year[state_year[cols["state"]] == state].sort_values("Year")
    trend_data = [{"year": int(row["Year"]), "rate": float(row["State_Crime_Rate_per_100k"])} for _, row in state_rate.iterrows()]
    
    return {
        "districts_2023": district_data,
        "types_2023": type_data,
        "trend": trend_data
    }

@app.get("/api/geojson")
@limiter.limit("60/minute")
def get_geojson(request: Request):
    if not geojson_data:
        raise HTTPException(status_code=404, detail="GeoJSON not found")
        
    by_state = state_year[state_year["Year"] == 2023][[cols["state"], "State_Crime_Rate_per_100k"]]
    rates = {row[cols["state"]]: float(row["State_Crime_Rate_per_100k"]) for _, row in by_state.iterrows()}
    
    return {
        "geojson": geojson_data,
        "rates": rates
    }

@app.get("/api/metrics")
@limiter.limit("60/minute")
def get_metrics(request: Request):
    if not MODELS_READY:
        raise HTTPException(status_code=400, detail="Models not trained")
        
    import datetime
    try:
        mtime = os.path.getmtime(os.path.join(MODELS_DIR, "reg_metrics.pkl"))
        train_date = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    except Exception:
        train_date = "Unknown"

    return {
        "training_date": train_date,
        "regression": models.get("reg_metrics", {}),
        "classification": models.get("type_metrics", {}),
        "risk": models.get("risk_metrics", {})
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
