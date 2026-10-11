import datetime
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from models import (
    DATA_PATH,
    GEOJSON_PATH,
    MODELS_DIR,
    build_state_features,
    build_state_year_table,
    detect,
    load,
    load_models,
    missing_artifacts,
    train_all,
)

log = logging.getLogger("crime_risk_api")


@dataclass
class Runtime:
    """Everything the endpoints need; only exists once startup fully succeeded."""
    df: pd.DataFrame
    cols: dict
    states: set
    state_year: pd.DataFrame
    state_feat: pd.DataFrame
    models: dict
    geojson: Any
    models_dir: Path


def initialize_runtime(data_path=DATA_PATH, models_dir=MODELS_DIR, geojson_path=GEOJSON_PATH):
    """One coherent startup: validate data, train if artifacts are missing or
    unloadable, load them, build runtime tables. Raises if anything fails."""
    df = load(data_path)
    cols = detect(df)

    if missing_artifacts(models_dir):
        log.warning("[startup] Missing artifacts %s. Training full set...", missing_artifacts(models_dir))
        train_all(data_path, models_dir)
    try:
        models = load_models(models_dir)
    except Exception:
        log.warning("[startup] Artifacts unreadable or incompatible. Retraining once...", exc_info=True)
        train_all(data_path, models_dir)
        models = load_models(models_dir)

    state_year = build_state_year_table(df, cols)
    _, _, state_feat = build_state_features(state_year, state_col=cols["state"])

    geojson = None
    if Path(geojson_path).is_file():
        with open(geojson_path, "r") as f:
            geojson = json.load(f)

    return Runtime(
        df=df, cols=cols, states=set(df[cols["state"]].unique().tolist()),
        state_year=state_year, state_feat=state_feat,
        models=models, geojson=geojson, models_dir=Path(models_dir),
    )


def create_app(data_path=DATA_PATH, models_dir=MODELS_DIR, geojson_path=GEOJSON_PATH):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = None
        app.state.startup_error = None
        try:
            app.state.runtime = initialize_runtime(data_path, models_dir, geojson_path)
            log.info("[startup] Ready.")
        except Exception as e:
            # Full detail goes to the server log only; clients see just the error type.
            log.exception("[startup] Initialization failed")
            app.state.startup_error = f"Initialization failed ({type(e).__name__})"
        yield

    limiter = Limiter(key_func=get_remote_address)
    app = FastAPI(title="Crime Risk API", lifespan=lifespan)
    app.state.limiter = limiter
    app.state.runtime = None
    app.state.startup_error = None
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[os.getenv("ALLOWED_ORIGIN", "http://localhost:5174")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_ready(request: Request) -> Runtime:
        rt = request.app.state.runtime
        if rt is None:
            raise HTTPException(status_code=503, detail="Service unavailable: models or data not initialized")
        return rt

    def require_state(rt: Runtime, state: str):
        if state not in rt.states:
            raise HTTPException(status_code=404, detail="State not found")

    @app.get("/api/status")
    @limiter.limit("60/minute")
    def status(request: Request):
        ready = request.app.state.runtime is not None
        return {
            "models_ready": ready,
            "data_loaded": ready,
            "error": request.app.state.startup_error,
        }

    @app.get("/api/states")
    @limiter.limit("60/minute")
    def get_states(request: Request):
        rt = require_ready(request)
        return {"states": sorted(rt.states)}

    @app.get("/api/forecast")
    @limiter.limit("60/minute")
    def get_forecast(request: Request, state: str = Query(..., description="The state to look up")):
        """Stored out-of-fold backtest of the 2023 rate: the fold model never saw this state."""
        rt = require_ready(request)
        require_state(rt, state)
        bt = rt.models["backtest_results"]["regression"]
        rec = bt["per_state"].get(state)
        if rec is None:
            raise HTTPException(status_code=404, detail="State not found")
        spread = rec["rf_tree_spread"]
        shap_list = sorted(rec["shap"], key=lambda x: abs(x["value"]), reverse=True)
        feat_imp = sorted(rec["feature_importance"], key=lambda x: x["importance"], reverse=True)

        return {
            "evaluation": "out_of_fold",
            "fold": rec["fold"],
            "predictions": {
                "linear": round(rec["lr"], 2),
                "ridge": round(rec["ridge"], 2),
                "random_forest": round(rec["rf"], 2),
                "ensemble": round(rec["ensemble"], 2),
                "actual": round(rec["observed"], 2),
                "absolute_error": round(rec["abs_error"]["ensemble"], 2)
            },
            "uncertainty": {
                "label": "RF tree spread (10th-90th percentile)",
                "calibrated": False,
                "low": round(spread["p10"], 2),
                "high": round(spread["p90"], 2),
                "std": round(spread["std"], 2)
            },
            "metrics": bt["metrics"],
            "shap": shap_list,
            "feature_importance": feat_imp
        }

    @app.get("/api/crime-pattern")
    @limiter.limit("60/minute")
    def get_crime_pattern(request: Request, state: str = Query(...)):
        rt = require_ready(request)
        require_state(rt, state)
        df, cols = rt.df, rt.cols
        bt = rt.models["backtest_results"]["crime_type"]

        grp = df.groupby([cols["state"], cols["year"], cols["crime_type"]], as_index=False)[cols["total"]].sum()
        state_sub = grp[grp[cols["state"]] == state].copy()
        pivot = state_sub.pivot_table(index=cols["year"], columns=cols["crime_type"], values=cols["total"], aggfunc="sum", fill_value=0)
        share = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

        result = {"evaluation": "out_of_fold"}
        rec = bt["per_state"].get(state)
        if rec is not None:
            prob_list = [{"type": cls, "probability": p} for cls, p in rec["rf"]["probabilities"].items()]
            prob_list.sort(key=lambda x: x["probability"], reverse=True)
            result["prediction"] = {
                "label": rec["rf"]["pred"],
                "actual_dominant_2023": rec["actual"],
                "fold": rec["fold"],
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
        result["metrics"] = bt["metrics"]

        return result

    @app.get("/api/risk-analysis")
    @limiter.limit("60/minute")
    def get_risk_analysis(request: Request):
        """Out-of-fold risk class per state (each from a model that excluded that state)."""
        rt = require_ready(request)
        bt = rt.models["backtest_results"]["risk"]
        w = {"low": 1, "medium": 2, "high": 3}

        analysis_results = []
        for state, rec in bt["per_state"].items():
            probs = rec["rf"]["probabilities"]
            if set(w) == set(probs):
                score = sum(probs[c] * w[c] for c in w)
            else:
                score = max(probs.values())
            item = {"state": state}
            row = rt.state_feat.loc[state]
            for k in ["rate_2019", "rate_2020", "rate_2021", "rate_2022", "rate_2023", "trend_19_22", "momentum_21_22", "volatility_19_22"]:
                if k in row and pd.notna(row[k]):
                    item[k] = float(row[k])
            item["Predicted_Risk_Class"] = rec["rf"]["pred"]
            item["Actual_Risk_Class"] = rec["actual"]
            item["Risk_Score"] = float(score)
            analysis_results.append(item)
        analysis_results.sort(key=lambda r: r["Risk_Score"], reverse=True)

        return {
            "evaluation": "out_of_fold",
            "rankings": analysis_results,
            "metrics": bt["metrics"]
        }

    @app.get("/api/state-drilldown")
    @limiter.limit("60/minute")
    def get_state_drilldown(request: Request, state: str = Query(...)):
        rt = require_ready(request)
        require_state(rt, state)
        df, cols = rt.df, rt.cols

        state_df = df[df[cols["state"]] == state].copy()

        district_2023 = state_df[state_df[cols["year"]] == 2023].groupby(cols["district"])[cols["total"]].sum().sort_values(ascending=False).head(10)
        district_data = [{"district": k, "count": int(v)} for k, v in district_2023.items()]

        type_2023 = state_df[state_df[cols["year"]] == 2023].groupby(cols["crime_type"])[cols["total"]].sum().sort_values(ascending=False)
        type_data = [{"type": k, "count": int(v)} for k, v in type_2023.items()]

        state_rate = rt.state_year[rt.state_year[cols["state"]] == state].sort_values("Year")
        trend_data = [{"year": int(row["Year"]), "rate": float(row["State_Crime_Rate_per_100k"])} for _, row in state_rate.iterrows()]

        return {
            "districts_2023": district_data,
            "types_2023": type_data,
            "trend": trend_data
        }

    @app.get("/api/geojson")
    @limiter.limit("60/minute")
    def get_geojson(request: Request):
        rt = require_ready(request)
        if not rt.geojson:
            raise HTTPException(status_code=404, detail="GeoJSON not found")

        cols = rt.cols
        by_state = rt.state_year[rt.state_year["Year"] == 2023][[cols["state"], "State_Crime_Rate_per_100k"]]
        rates = {row[cols["state"]]: float(row["State_Crime_Rate_per_100k"]) for _, row in by_state.iterrows()}

        return {
            "geojson": rt.geojson,
            "rates": rates
        }

    @app.get("/api/metrics")
    @limiter.limit("60/minute")
    def get_metrics(request: Request):
        rt = require_ready(request)
        bt = rt.models["backtest_results"]
        try:
            mtime = (rt.models_dir / "backtest_results.pkl").stat().st_mtime
            train_date = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        except OSError:
            train_date = "Unknown"

        return {
            "training_date": train_date,
            "evaluation": bt["evaluation"],
            "n_states": bt["n_states"],
            "n_folds": bt["n_folds"],
            "limitations": bt["limitations"],
            "regression": bt["regression"]["metrics"],
            "classification": bt["crime_type"]["metrics"],
            "risk": bt["risk"]["metrics"]
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001)
