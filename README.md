# Risk Intelligence Dashboard

**Risk Intelligence Dashboard** is a modern, enterprise-grade, full-stack risk intelligence terminal. It predicts regional crime anomalies, categorizes structural risks, and visualizes geographic threat distributions across India using advanced machine learning ensembles.

## 🚀 Recent Architectural Overhaul
This project was recently migrated from a basic monolithic Streamlit script into a high-performance, decoupled **Full-Stack Application**:

1. **Frontend (`/frontend`)**: A Lightning-fast React SPA built with Vite.
    - Features a bespoke, premium **Obsidian Dark Mode** design system.
    - Highly disciplined UI employing strict 8-point typographic and spatial grids.
    - Rich visualizations using customized *Recharts* (featuring glassmorphism tooltips and linear gradients) and *Plotly.js* (Carto-Darkmatter mapped to state geojson data).
    - Advanced UI/UX elements: Global search pallets, live UTC clocks, blinking intel feeds, and agent clearance avatars.
2. **Backend (`/backend`)**: A highly concurrent RESTful API built on **FastAPI** (Python).
    - Exposes strict JSON endpoints allowing the frontend to fetch regression projections, classification rankings, tabular drilldowns, and SHAP feature importance.
    - Seamlessly loads robust serialized `.pkl` machine learning models (Random Forest, Ridge, Logit).

---

## 💻 Tech Stack

### Frontend Hub
- **Core Framework**: React 18 & Vite
- **Styling**: Vanilla CSS (CSS3 Variables & Strict 8-point Grid System)
- **Data Visualization**: Recharts (SVG gradients) & Plotly.js (Carto-Darkmatter GeoJSON polygons)
- **Icons**: Lucide-React

### Intelligence Backend
- **Web Framework**: FastAPI (running on Uvicorn)
- **Predictive Engines**: Scikit-Learn (Random Forest, Ridge Regression, Logistic Regression)
- **Data Engineering**: Pandas & NumPy
- **Model Explainability**: SHAP (SHapley Additive exPlanations)
- **Serialization**: Joblib (`.pkl` model persistence)

---

## 📸 Key Features

* **Live Intel Feed Dashboard**: Displays national statistics, dynamically pulsing statuses, and command-center-like data integrations.
* **Ensemble Forecasts**: Blends multiple ML models (Linear, Ridge, Random Forest) to project crime rates with extremely high accuracy, accompanied by absolute error calculations.
* **SHAP Impact Analysis**: Transparent metric attribution. Instantly see whether recent momentum (e.g. `momentum_21_22`) or historical baseline rates are driving a particular state's risk classification.
* **Geospatial Mapping**: Interactive heatmaps routing structural threat scores directly to regional polygons.

---

## 🛠️ Getting Started (Local Deployment)

Because the architecture has been decoupled, you will need to run the backend and the frontend simultaneously in two separate terminal windows.

### 1. Initialize the Intelligence Backend
Open your first terminal and navigate to the backend directory:
```bash
cd backend

# Install the Python dependencies
pip install fastapi uvicorn pandas scikit-learn shap

# If you haven't generated the models yet, run the training engine:
python -c "import models; models.train_all()"

# Boot up the FastAPI server on port 8001
uvicorn api:app --reload --port 8001
```

### 2. Initialize the Obsidian Frontend
Open your second terminal and navigate to the frontend directory:
```bash
cd frontend

# Install the Node dependencies (React, Recharts, Plotly, Lucide)
npm install

# Launch the Vite development server
npm run dev
```

### 3. Access the Terminal
Once both servers are running, the application should automatically open in your browser, or you can manually navigate to:
**[http://localhost:5174](http://localhost:5174)**

---

## 🎨 Design Philosophy
The frontend enforces a non-negotiable **"Intentional Premium UI"** mandate:
* **Zero generic frameworks**: Engineered entirely with a custom Vanilla CSS variable architecture (`index.css`) rather than relying on bloated Tailwind or Bootstrap scaffolding.
* **Glassmorphism**: Leverages subtle translucent backgrounds (`rgba(255,255,255,0.02)`) and background blurring to create immersive depth.
* **High-Clarity Typography**: Strictly uses the *Inter* typeface, leaning entirely into weights and tracking to establish clear visual hierarchies. 

## 🔒 License
Proprietary Analysis Engine. All rights reserved.
