"""Re-run the evaluation checks that show how much to trust the headline numbers.

    python backend/scripts/evaluation_report.py              # 20 fold seeds, 2000 bootstrap draws
    python backend/scripts/evaluation_report.py --seeds 5    # quicker
    python backend/scripts/evaluation_report.py --json out.json

It answers four questions the published backtest (seed 42) cannot answer alone:
  1. Do the models beat simple baselines such as "same as last year"?
  2. How much do the results move when the states are shuffled into different folds?
  3. How wide is the uncertainty with only 12 states (bootstrap)?
  4. Which states are predicted worst?
Nothing here is written to backend/models and nothing is served by the API.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as m  # noqa: E402


def _summary(values):
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def compute_report(n_seeds=20, n_boot=2000, data_path=m.DATA_PATH):
    df = m.load(data_path)
    d = m.build_datasets(df, m.detect(df))
    args = (d["X_reg"], d["y_reg"], d["X_type"], d["y_type"], d["X_risk"], d["y_risk"])

    published = m.run_backtest(*args)  # seed 42, the run the app serves
    reg_m = published["regression"]["metrics"]
    type_m = published["crime_type"]["metrics"]
    risk_m = published["risk"]["metrics"]

    runs = [m.run_backtest(*args, seed=s) for s in range(n_seeds)]
    names = ["lr", "ridge", "rf", "ensemble"]
    best_mae = reg_m["vs_baseline"]["best_value"]
    seeds = {
        "regression": {
            n: {
                "mae": _summary([r["regression"]["metrics"][n]["oof_mae_rate_per_100k"] for r in runs]),
                "r2": _summary([r["regression"]["metrics"][n]["oof_r2"] for r in runs]),
            } for n in names
        },
        "risk": {n: _summary([r["risk"]["metrics"][n]["oof_accuracy"] for r in runs]) for n in ("rf", "logit")},
        "crime_type": {n: _summary([r["crime_type"]["metrics"][n]["oof_accuracy"] for r in runs]) for n in ("rf", "logit")},
        "rf_beats_best_baseline_in_seeds": int(sum(
            r["regression"]["metrics"]["rf"]["oof_mae_rate_per_100k"] < best_mae for r in runs)),
        "n_seeds": n_seeds,
    }

    per_state = published["regression"]["per_state"]
    states = sorted(per_state)
    obs = np.array([per_state[s]["observed"] for s in states])
    pred = np.array([per_state[s]["rf"] for s in states])
    rng = np.random.default_rng(0)
    maes, r2s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(states), len(states))
        maes.append(mean_absolute_error(obs[idx], pred[idx]))
        if obs[idx].std() > 0:
            r2s.append(r2_score(obs[idx], pred[idx]))
    bootstrap = {
        "rf_mae_95": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
        "rf_r2_95": [float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))],
        "n_boot": n_boot,
    }
    worst = sorted(((abs(per_state[s]["rf"] - per_state[s]["observed"]), s) for s in states), reverse=True)

    return {
        "published_seed": published["seed"],
        "published": {"regression": reg_m, "risk": risk_m, "crime_type": type_m},
        "seeds": seeds,
        "bootstrap": bootstrap,
        "rf_error_by_state": [{"state": s, "abs_error": float(e)} for e, s in worst],
    }


def format_report(r):
    reg, risk, typ = (r["published"][k] for k in ("regression", "risk", "crime_type"))
    L = []
    L.append("EVALUATION REPORT (12-state sample; read every number as weak evidence)")
    L.append("")
    L.append("1. Baselines vs the published run (seed %d)" % r["published_seed"])
    for n, v in reg["baselines"].items():
        L.append(f"   rate  {n:<22} MAE {v['mae_rate_per_100k']:6.2f}   R2 {v['r2']:6.2f}")
    for n in ("lr", "ridge", "rf", "ensemble"):
        v = reg[n]
        flag = "beats" if reg["vs_baseline"]["beats"][n] else "does NOT beat"
        L.append(f"   rate  model {n:<16} MAE {v['oof_mae_rate_per_100k']:6.2f}   R2 {v['oof_r2']:6.2f}   {flag} best baseline ({reg['vs_baseline']['best_baseline']})")
    for label, mm in (("risk", risk), ("crime", typ)):
        for n, v in mm["baselines"].items():
            L.append(f"   {label:<5} {n:<22} accuracy {v['accuracy']:.2f}")
        for n in ("rf", "logit"):
            flag = "beats" if mm["vs_baseline"]["beats"][n] else "does NOT beat"
            L.append(f"   {label:<5} model {n:<16} accuracy {mm[n]['oof_accuracy']:.2f}   {flag} best baseline ({mm['vs_baseline']['best_baseline']})")
    s = r["seeds"]
    L.append("")
    L.append(f"2. Same backtest over {s['n_seeds']} different fold seeds (mean, min-max)")
    for n, v in s["regression"].items():
        L.append(f"   rate  {n:<9} MAE {v['mae']['mean']:5.2f} ({v['mae']['min']:.2f}-{v['mae']['max']:.2f})   R2 {v['r2']['mean']:5.2f} ({v['r2']['min']:.2f}-{v['r2']['max']:.2f})")
    for label in ("risk", "crime_type"):
        for n, v in s[label].items():
            L.append(f"   {label:<10} {n:<6} accuracy {v['mean']:.2f} ({v['min']:.2f}-{v['max']:.2f})")
    L.append(f"   Random Forest beat the best rate baseline in {s['rf_beats_best_baseline_in_seeds']} of {s['n_seeds']} seeds")
    b = r["bootstrap"]
    L.append("")
    L.append(f"3. Bootstrap over the 12 held-out Random Forest predictions ({b['n_boot']} draws, crude)")
    L.append(f"   MAE 95% interval {b['rf_mae_95'][0]:.1f} to {b['rf_mae_95'][1]:.1f}   R2 95% interval {b['rf_r2_95'][0]:.2f} to {b['rf_r2_95'][1]:.2f}")
    L.append("")
    L.append("4. Random Forest absolute error by state (published run), largest first")
    L.append("   " + ", ".join(f"{e['state']} {e['abs_error']:.1f}" for e in r["rf_error_by_state"]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--json", help="also write the full report to this file")
    a = ap.parse_args()
    report = compute_report(a.seeds, a.bootstrap)
    print(format_report(report))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))
        print(f"\nWrote {a.json}")


if __name__ == "__main__":
    main()
