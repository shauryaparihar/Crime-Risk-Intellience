# Risk Intelligence Dashboard

A React and FastAPI dashboard that performs a small **historical 2023 backtest** over an **unverified NCRB-style sample dataset**. It uses 2019–2022 state-level features to compare regression and classification models, visualizes observed 2023 crime rates and crime patterns, and shows model explanations and limitations.

**Hosted demo:** [crime-risk-intellience-ashy.vercel.app](https://crime-risk-intellience-ashy.vercel.app)

> **This is an educational full-stack ML demonstration.** It is not a live crime feed, not an official NCRB product, not a public-safety decision system, and not a production forecasting service.

## What the app does

Four views, all over the same 12-state sample:

| View | What it shows |
|---|---|
| **Backtest & Analysis** | For a chosen state: the 2023 crime rate (per 100k) estimated by Linear, Ridge, Random Forest and their mean, next to the observed 2023 rate, plus SHAP feature impact for the Random Forest. |
| **Crime Patterns** | The model's guess for the state's dominant 2023 crime type, the observed one, and the share trend of the five crime types with the highest average share. |
| **Risk Classification** | A low / medium / high risk class per state (terciles of the 2023 rate), predicted vs observed, with a 1–3 risk score. |
| **State Map** | Observed 2023 crime rate per 100k on a state map, plus top districts and crime-type counts for the selected state. |

## How the evaluation works (and why it is weak evidence)

- **Inputs:** features built only from 2019–2022 (rates, average, trend, momentum, volatility, growth; crime-type shares for the pattern task).
- **Targets:** observed 2023 crime rate, dominant 2023 crime type, and relative 2023 risk class.
- **Out-of-fold (OOF) predictions:** the 12 states are split once into 4 folds (`KFold`, shuffled, `random_state=42`). Every displayed prediction comes from a model trained on the *other* folds, so it never saw that state. Training-set accuracy or R² is never shown as model quality.
- **Metrics** are computed on the 12 held-out predictions: MAE and R² in crime-rate-per-100k units for regression; accuracy and macro-F1 for the two classification tasks.

### Results of the current backtest

| Task | Model | OOF result |
|---|---|---|
| 2023 rate (per 100k, range 136–181) | Random Forest | MAE 6.7, R² 0.61 |
| | Ridge | MAE 8.4, R² 0.46 |
| | Linear | MAE 12.3, R² −0.25 |
| | Mean of the three | MAE 8.1, R² 0.45 |
| Dominant 2023 crime type (5 classes) | Random Forest | accuracy 8% |
| | Logistic | accuracy 50% |
| Risk class (low / medium / high) | Random Forest | accuracy 92% |
| | Logistic | accuracy 67% |

These come from `train_all()` on the bundled CSV (shown by `GET /api/metrics`). They can change if the data, library versions or code change.

### Do the models beat a simple baseline? Not on the rate task.

Baselines need no fitting, so they show what a model must beat. They are computed in the same backtest and returned by `GET /api/metrics` under `baselines` and `vs_baseline`.

| Task | Baseline | Result | Best model |
|---|---|---|---|
| 2023 rate | Last year's rate (2023 = 2022) | MAE 3.7, R² 0.86 | RF MAE 6.7, R² 0.61 |
| | Average of 2019–2022 | MAE 3.5, R² 0.87 | |
| | Average of the other states | MAE 13.2, R² −0.19 | |
| Risk class | Rank states by last year's rate | accuracy 83% | RF 92% on this split |
| Dominant crime | Always the most common class | accuracy 33% | Logistic 50%, RF 8% |

On the rate task no model beats the best baseline, because rates barely change year to year (2022 and 2023 correlate at 0.95). The classification results are from a single split (seed 42); repeating the backtest with other fold seeds gives lower averages (risk RF about 63%, dominant-crime logistic about 39%). Reproduce the checks with:

```bash
python backend/scripts/evaluation_report.py            # 20 fold seeds, bootstrap, per-state errors
python backend/scripts/evaluation_report.py --seeds 5  # quicker
```

### Limitations

- **Only 12 states**, so every metric rests on 12 held-out predictions and can swing a lot.
- **The models do not beat a no-model baseline** on the rate task (see above), and the published classification numbers come from one split that is at the favourable end of the range.
- **Dominant-crime classes are imbalanced** (4/3/2/2/1 states per class; one class has a single state). Stratified folds are impossible, and a held-out singleton class can never be predicted. Treat this task as unreliable.
- **The risk task is easier than it looks:** labels are terciles of the 2023 rate computed over all 12 states, and the features are the earlier-year rates of the same states.
- **"RF tree spread (10th–90th percentile)"** is the spread of individual trees. It is *not* a calibrated confidence or prediction interval.
- **The data is not official** (see below). Nothing here supports conclusions about real regions.

## Data

`backend/data/crime_data.csv` follows the structure of National Crime Records Bureau (NCRB) statistics. Its origin is not documented and it has **not been verified against the official NCRB tables**. Treat it as sample data.

| Attribute | Value |
|---|---|
| Rows | 30,000 |
| Coverage | 12 states, 300 districts, 10 crime types |
| Years in file | 2014–2023 (the pipeline uses 2019–2023) |
| Granularity | State → District → Crime type → Year |

`backend/india_state.geojson` supplies state outlines for the map (35 features; the 12 dataset states match `properties.NAME_1`). Its origin and licence are also undocumented.

## Tech stack

- **Backend:** Python (verified with 3.11), FastAPI, Uvicorn, pandas, NumPy, scikit-learn (Linear, Ridge, Logistic, Random Forest), SHAP, Joblib, SlowAPI (60 requests/minute per client), pytest.
- **Frontend:** React 19, Vite 8, Recharts, Plotly (`react-plotly.js`), Lucide React, plain CSS.
- **Delivery files:** Docker Compose, backend and frontend Dockerfiles (frontend served by Nginx), `render.yaml`, `frontend/vercel.json`. Their presence does not mean the app is deployed anywhere.

## Run it locally

Backend (port **8001**):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt     # runtime + pytest

cd backend
uvicorn api:app --port 8001
```

On startup the API checks all 17 saved model files in `backend/models/`. If any is missing or unreadable, or if the CSV or the scikit-learn, NumPy, pandas or SHAP versions differ from what produced the saved results (a SHA-256 of the data and the library versions are stored with them), it trains the full set once (a few seconds) and becomes ready without a restart. `GET /api/status` reports the real state; if startup fails, model endpoints return `503`. Paths are resolved from the code's location, so the command also works as `uvicorn --app-dir backend api:app --port 8001` from the repository root.

Frontend (port **5174**), in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5174>. Select a state in the sidebar.

### Configuration

| Variable | Used by | Meaning | Default |
|---|---|---|---|
| `ALLOWED_ORIGIN` | backend | The single browser origin allowed by CORS | `http://localhost:5174` |
| `VITE_API_URL` | frontend (build time) | Backend base URL, without `/api` | `http://localhost:8001` |

### Tests, lint, build

```bash
python -m pytest backend/tests -q     # backend (from the repo root, venv active)

cd frontend
npm run lint
npm run build
```

There are no frontend unit tests.

### Docker Compose

```bash
docker compose build
docker compose up -d      # backend on :8001, frontend (Nginx) on :5174
docker compose ps
docker compose down
```

The backend image trains its models at build time and the build fails if training fails. The frontend image uses Node 22 and `npm ci`.

## API

All endpoints are `GET` under `/api`: `status`, `ready`, `states`, `forecast?state=`, `crime-pattern?state=`, `risk-analysis`, `state-drilldown?state=`, `geojson`, `metrics`.
`status` always answers 200 with a readiness flag, for people. `ready` is the platform health probe: 200 when the app can serve, 503 after a failed startup (not rate limited).
Unknown states return `404`, a missing `state` returns `422`, and `503` means data or models failed to initialize. `forecast`, `crime-pattern`, `risk-analysis` and `metrics` return the stored out-of-fold results (`"evaluation": "out_of_fold"`); nothing is refit per request.

## Possible future work (not implemented)

Verify or replace the dataset with documented official figures, add more states/years, true time-based validation, and calibrated uncertainty.

## License

No license file is provided in this repository, so no license is granted.
