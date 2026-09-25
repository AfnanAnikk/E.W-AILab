"""
model_server.py
================
Loads the two trained models (regressor_demand.json, classifier_risk_tier.json
from train_models.py) and serves them over HTTP so the SimCity dashboard
(browser JS, can't run XGBoost directly) can get real predictions instead of
a heuristic stand-in.

Run alongside GridWise, not instead of it:
    dashboard -> POST /predict (this server)  -> demand + risk tier per building
    dashboard -> POST /optimize-energy (GridWise) -> actual grid/battery plan

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
1. pip install fastapi uvicorn xgboost pandas
2. Put regressor_demand.json and classifier_risk_tier.json (from running
   train_models.py) in the same folder as this file, or set MODEL_DIR below.
3. python model_server.py
4. Server runs on http://localhost:8001 (GridWise typically runs on 8000,
   kept these on different ports so you can run both at once)

--------------------------------------------------------------------------
API
--------------------------------------------------------------------------
GET  /health
    -> {"status": "ok", "models_loaded": true}

POST /predict
    Request body: one building's current feature snapshot. Field names and
    meaning MUST match train_models.py's feature_cols exactly -- if you
    change feature engineering there, update FEATURE_COLS here too.

    {
      "hour": 14, "dayofweek": 2, "is_weekend": 0, "month": 7,
      "lag_1h": 120.5, "lag_24h": 118.2, "rolling_mean_24h": 115.0,
      "airTemperature": 32.1, "dewTemperature": 24.0, "windSpeed": 3.2,
      "sqft": 12000,
      "chilledwater_lag_1h": 45.0, "chilledwater_rolling_std_6h": 2.1,
      "steam_lag_1h": 0.0, "steam_rolling_std_6h": 0.0,
      "hotwater_lag_1h": 10.0, "hotwater_rolling_std_6h": 1.0,
      "gas_lag_1h": 0.0, "gas_rolling_std_6h": 0.0,
      "water_lag_1h": 5.0, "water_rolling_std_6h": 0.5,
      "irrigation_lag_1h": 0.0, "irrigation_rolling_std_6h": 0.0
    }

    Response:
    {
      "predicted_demand_kwh": 122.4,
      "risk_tier": "Watch",
      "risk_tier_int": 1,
      "risk_tier_probabilities": {"Normal": 0.1, "Watch": 0.7,
                                    "High-Risk": 0.15, "Shed-Now": 0.05}
    }

POST /predict/batch
    Same, but body is {"buildings": {"<building_id>": {...features...}, ...}}
    Response: {"<building_id>": {...same shape as /predict...}, ...}
    This is the one the dashboard should actually use each tick -- one call
    for the whole campus instead of one HTTP round-trip per building.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import xgboost as xgb
import pandas as pd

MODEL_DIR = os.environ.get("MODEL_DIR", ".")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "regressor_demand.json")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier_risk_tier.json")

# MUST match train_models.py's feature_cols exactly, same order doesn't
# matter (we pass a dict / DataFrame with named columns) but the SET of
# names must match, or predictions will silently use the wrong values.
FEATURE_COLS = [
    "hour", "dayofweek", "is_weekend", "month",
    "lag_1h", "lag_24h", "rolling_mean_24h",
    "airTemperature", "dewTemperature", "windSpeed", "sqft",
    "chilledwater_lag_1h", "steam_lag_1h", "hotwater_lag_1h",
    "gas_lag_1h", "water_lag_1h", "irrigation_lag_1h",
    "chilledwater_rolling_std_6h", "steam_rolling_std_6h",
    "hotwater_rolling_std_6h", "gas_rolling_std_6h",
    "water_rolling_std_6h", "irrigation_rolling_std_6h",
]

# Must match TIER_ORDER / TIER_TO_INT in train_models.py exactly.
TIER_ORDER = ["Normal", "Watch", "High-Risk", "Shed-Now"]

_regressor = None
_classifier = None


def load_models():
    if not os.path.exists(REGRESSOR_PATH):
        raise FileNotFoundError(
            f"{REGRESSOR_PATH} not found. Run train_models.py first and "
            f"copy regressor_demand.json here, or set MODEL_DIR."
        )
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"{CLASSIFIER_PATH} not found. Run train_models.py first and "
            f"copy classifier_risk_tier.json here, or set MODEL_DIR."
        )
    reg = xgb.XGBRegressor()
    reg.load_model(REGRESSOR_PATH)
    clf = xgb.XGBClassifier()
    clf.load_model(CLASSIFIER_PATH)
    return reg, clf


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _regressor, _classifier
    _regressor, _classifier = load_models()
    yield  # server runs here
    # no teardown needed -- models are just in-memory objects


app = FastAPI(title="Campus Energy Model Server", lifespan=lifespan)

# CORS: allow the dashboard (running from a local file or localhost) to
# call this server from the browser. Tighten allow_origins for anything
# beyond local dev/demo use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class BuildingFeatures(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerate extra fields, just ignore them
    hour: float
    dayofweek: float
    is_weekend: float
    month: float
    lag_1h: float
    lag_24h: float
    rolling_mean_24h: float
    airTemperature: float
    dewTemperature: float
    windSpeed: float
    sqft: float
    chilledwater_lag_1h: float
    steam_lag_1h: float
    hotwater_lag_1h: float
    gas_lag_1h: float
    water_lag_1h: float
    irrigation_lag_1h: float
    chilledwater_rolling_std_6h: float
    steam_rolling_std_6h: float
    hotwater_rolling_std_6h: float
    gas_rolling_std_6h: float
    water_rolling_std_6h: float
    irrigation_rolling_std_6h: float


class BatchRequest(BaseModel):
    buildings: Dict[str, BuildingFeatures]


def _predict_one(features: BuildingFeatures) -> dict:
    row = pd.DataFrame([features.model_dump()])[FEATURE_COLS]

    demand = float(_regressor.predict(row)[0])

    proba = _classifier.predict_proba(row)[0]
    tier_int = int(proba.argmax())
    tier_name = TIER_ORDER[tier_int]

    return {
        "predicted_demand_kwh": round(demand, 3),
        "risk_tier": tier_name,
        "risk_tier_int": tier_int,
        "risk_tier_probabilities": {
            TIER_ORDER[i]: round(float(p), 4) for i, p in enumerate(proba)
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _regressor is not None and _classifier is not None}


@app.post("/predict")
def predict(features: BuildingFeatures):
    if _regressor is None or _classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        return _predict_one(features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if _regressor is None or _classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    results = {}
    for building_id, features in req.buildings.items():
        try:
            results[building_id] = _predict_one(features)
        except Exception as e:
            results[building_id] = {"error": str(e)}
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
