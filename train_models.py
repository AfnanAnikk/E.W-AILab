"""
train_models.py
================
Trains the two models for the campus energy project, on the Building Data
Genome 2.0 (BDG2) dataset, using ALL meter types as signal (electricity,
chilled water, steam, hot water, gas, water, irrigation, solar) - not
electricity in isolation.

  MODEL 1 - XGBoost Regressor
      Predicts next-hour electricity demand (kWh) per building.
      This is the forecasting engine: feeds GridWise's `demand_kwh` field.

  MODEL 2 - XGBoost Classifier
      Predicts a load-shed risk tier for that hour, one of:
          Normal < Watch < High-Risk < Shed-Now
      This is the decision-support signal: which buildings need attention
      before an outage notice lands, tying back to the "critical grid
      state" framing in the original project proposal.

Both models share the same feature set (built once in build_dataset(), see
below) - the same hour/weather/lag/HVAC-health inputs, two different jobs
(forecast the number vs. classify urgency). Electricity is the target
meter for both; chilled water / steam / hot water / gas become lag +
rolling-std features per building, since sustained high flow with near-
zero variation is the actual "condenser never cycling down" signal the
proposal calls out - not just weather + time. Water/irrigation are
included as auxiliary building-activity features. Solar is pulled out
separately into solar_for_gridwise.csv (real generation data, not a model
feature) so it can later feed GridWise's `solar_kwh` field directly.

--------------------------------------------------------------------------
HOW TO RUN (Google Colab) - 4 CELLS, COPY DIRECTLY FROM THIS FILE
--------------------------------------------------------------------------
1. Mount Drive and find your data folder (see chat for the `!ls` steps).
2. This file is split into 4 blocks, each marked "===== CELL N =====" below.
   Copy everything between one ===== CELL N ===== marker and the next into
   its own Colab cell, in order, and run them 1 -> 2 -> 3 -> 4.
     CELL 1: imports + config          (edit DATA_DIR here)
     CELL 2: load data + build dataset (run once; re-run only if you
              change DATA_DIR, N_BUILDINGS, or the *_PCTL thresholds)
     CELL 3: train the regressor       (re-run anytime on its own)
     CELL 4: train the classifier      (re-run anytime on its own)
   Cells 3 and 4 both depend on `data` from cell 2, but not on each other -
   re-running one doesn't require re-running the other.

Expected files in DATA_DIR (same wide shape: one column per building,
index = timestamp): electricity_cleaned.csv, chilledwater_cleaned.csv,
steam_cleaned.csv, hotwater_cleaned.csv, gas_cleaned.csv,
water_cleaned.csv, irrigation_cleaned.csv, solar_cleaned.csv,
weather.csv, metadata.csv.

--------------------------------------------------------------------------
CHANGELOG (most recent first)
--------------------------------------------------------------------------
- Fixed: train/test split was random across time (shuffle=True), not
  chronological. For a next-hour forecaster, that lets train and test hours
  sit right next to each other in the same week rather than test being
  genuinely future-relative-to-train - an easier task than real deployment,
  and a likely contributor to the very high regressor R2. Now splits each
  building's timeline at the 80% mark (train = earlier hours, test = later
  hours) instead of randomly sampling rows.
- Fixed: risk-tier labels were scrambled. `LabelEncoder().fit(tier_order)`
  ignores the order you pass it and always sorts classes alphabetically
  (High-Risk, Normal, Shed-Now, Watch -> 0,1,2,3), but the code assumed
  Normal=0, Watch=1, High-Risk=2, Shed-Now=3. The model trained fine
  internally but every printed/predicted label was wrong relative to what
  it claimed to mean (e.g. "Shed-Now" rows were actually "Watch"). Fixed
  by mapping tiers to integers explicitly instead of via LabelEncoder.
- Fixed: metadata.csv has columns literally named "electricity",
  "chilledwater", "gas", etc. (Yes/blank meter-installed flags) that
  collide with the melted meter value columns of the same name. Renamed
  to has_<meter> before merging.
- Fixed: mean_squared_error(..., squared=False) was removed in recent
  scikit-learn; now uses root_mean_squared_error with a fallback.
- Fixed: XGBClassifier needs integer labels, not raw strings.
- Verified end-to-end against real BDG2 headers (Sept 2026): confirmed
  column names (airTemperature, dewTemperature, windSpeed, sqft, site_id,
  building_id) match this script with no changes needed.
"""

import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error  # sklearn >= 1.4
except ImportError:
    root_mean_squared_error = None
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ===== CELL 1 ===== imports + config - edit DATA_DIR to your Google Drive folder
DATA_DIR = "/content/drive/MyDrive/Electricity/"

METER_CSVS = {
    "electricity": DATA_DIR + "electricity_cleaned.csv",   # target meter
    "chilledwater": DATA_DIR + "chilledwater_cleaned.csv",  # HVAC signal
    "steam": DATA_DIR + "steam_cleaned.csv",                # HVAC signal
    "hotwater": DATA_DIR + "hotwater_cleaned.csv",          # HVAC signal
    "gas": DATA_DIR + "gas_cleaned.csv",                    # HVAC signal
    "water": DATA_DIR + "water_cleaned.csv",                # auxiliary
    "irrigation": DATA_DIR + "irrigation_cleaned.csv",      # auxiliary
    "solar": DATA_DIR + "solar_cleaned.csv",                # generation -> GridWise, not a feature
}
WEATHER_CSV = DATA_DIR + "weather.csv"
METADATA_CSV = DATA_DIR + "metadata.csv"

N_BUILDINGS = 30        # already validated at this size (421,195 rows) - raise
                         # this once you're ready to train on more buildings
RANDOM_STATE = 42

# Risk-tier thresholds, as a percentile of each building's OWN historical
# demand. Tune these once you've looked at real distributions.
WATCH_PCTL = 0.75
HIGH_RISK_PCTL = 0.90
SHED_NOW_PCTL = 0.97

# Canonical tier order -> integer mapping. Defined ONCE, here, and used
# everywhere below (training, evaluation, printing) so int labels can never
# drift out of sync with their names. Do not use sklearn's LabelEncoder for
# this - it silently re-sorts classes alphabetically and breaks the mapping.
TIER_ORDER = ["Normal", "Watch", "High-Risk", "Shed-Now"]
TIER_TO_INT = {t: i for i, t in enumerate(TIER_ORDER)}

HVAC_METERS = ["chilledwater", "steam", "hotwater", "gas"]
AUX_METERS = ["water", "irrigation"]


# ===== CELL 2 ===== load data + feature engineering + build dataset (run once)
def melt_meter(path, value_name, building_cols=None):
    """Each meter CSV is wide (one column per building_id, index=timestamp).
    Melt to long format: timestamp, building_id, <value_name>."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    cols = building_cols or [c for c in df.columns if c != "timestamp"]
    cols = [c for c in cols if c in df.columns]  # not every building has every meter
    return df.melt(id_vars="timestamp", value_vars=cols,
                    var_name="building_id", value_name=value_name)


def load_data():
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["timestamp"])
    meta = pd.read_csv(METADATA_CSV)

    # metadata.csv has columns literally named "electricity", "chilledwater",
    # "gas", etc. -- Yes/blank meter-installed flags, NOT consumption values.
    # Rename before merging or they collide with the melted meter columns.
    meter_flag_cols = ["electricity", "hotwater", "chilledwater", "steam",
                        "water", "irrigation", "solar", "gas"]
    meta = meta.rename(columns={c: f"has_{c}" for c in meter_flag_cols if c in meta.columns})

    elec_wide = pd.read_csv(METER_CSVS["electricity"], parse_dates=["timestamp"])
    building_cols = [c for c in elec_wide.columns if c != "timestamp"][:N_BUILDINGS]

    long = melt_meter(METER_CSVS["electricity"], "electricity", building_cols)
    long = long.dropna(subset=["electricity"])

    hvac_and_aux = HVAC_METERS + AUX_METERS
    for meter in hvac_and_aux:
        m = melt_meter(METER_CSVS[meter], meter, building_cols)
        long = long.merge(m, on=["timestamp", "building_id"], how="left")

    # solar is generation, not a model feature -- keep it separate for
    # GridWise, don't merge it into the training feature set
    solar = melt_meter(METER_CSVS["solar"], "solar", building_cols)
    solar.to_csv("solar_for_gridwise.csv", index=False)

    long[hvac_and_aux] = long[hvac_and_aux].fillna(0.0)  # meter not installed -> treat as 0 load

    long = long.merge(meta, on="building_id", how="left")
    long = long.merge(weather, on=["site_id", "timestamp"], how="left")
    return long.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def engineer_features(df):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month

    # electricity lag features per building
    df["lag_1h"] = df.groupby("building_id")["electricity"].shift(1)
    df["lag_24h"] = df.groupby("building_id")["electricity"].shift(24)
    df["rolling_mean_24h"] = (
        df.groupby("building_id")["electricity"]
        .shift(1).rolling(24, min_periods=6).mean()
        .reset_index(level=0, drop=True)
    )

    # HVAC meter lag + "is this meter stuck on" features -- high sustained
    # chilledwater/steam flow with little variation over the last few hours
    # suggests a stuck valve or overload, not just normal cooling.
    for meter in HVAC_METERS + AUX_METERS:
        df[f"{meter}_lag_1h"] = df.groupby("building_id")[meter].shift(1)
        df[f"{meter}_rolling_std_6h"] = (
            df.groupby("building_id")[meter]
            .shift(1).rolling(6, min_periods=3).std()
            .reset_index(level=0, drop=True)
        )

    # target: next hour's electricity demand
    df["target_next_hour"] = df.groupby("building_id")["electricity"].shift(-1)

    # per-building risk thresholds computed on historical (non-leaking) demand
    thresh = df.groupby("building_id")["electricity"].quantile(
        [WATCH_PCTL, HIGH_RISK_PCTL, SHED_NOW_PCTL]
    ).unstack()
    thresh.columns = ["watch_thresh", "high_thresh", "shed_thresh"]
    df = df.merge(thresh, on="building_id", how="left")

    def tier(row):
        if row["target_next_hour"] >= row["shed_thresh"]:
            return "Shed-Now"
        elif row["target_next_hour"] >= row["high_thresh"]:
            return "High-Risk"
        elif row["target_next_hour"] >= row["watch_thresh"]:
            return "Watch"
        return "Normal"

    df["risk_tier"] = df.apply(tier, axis=1)
    df["risk_tier_int"] = df["risk_tier"].map(TIER_TO_INT)

    feature_cols = (
        ["hour", "dayofweek", "is_weekend", "month",
         "lag_1h", "lag_24h", "rolling_mean_24h",
         "airTemperature", "dewTemperature", "windSpeed",  # from weather.csv
         "sqft"]  # from metadata.csv
        + [f"{m}_lag_1h" for m in HVAC_METERS + AUX_METERS]
        + [f"{m}_rolling_std_6h" for m in HVAC_METERS + AUX_METERS]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.dropna(subset=feature_cols + ["target_next_hour", "risk_tier"])
    return df, feature_cols


def build_dataset():
    """Runs load + feature engineering once, returns everything both models
    need. Call this once per session; re-run only if DATA_DIR, N_BUILDINGS,
    or the threshold percentiles change."""
    print("Loading data...")
    raw = load_data()
    df, feature_cols = engineer_features(raw)
    print(f"{len(df):,} rows, {len(feature_cols)} features: {feature_cols}")
    print(f"Risk tier distribution (expected ~{WATCH_PCTL:.0%} Normal, "
          f"~{HIGH_RISK_PCTL-WATCH_PCTL:.0%} Watch, "
          f"~{SHED_NOW_PCTL-HIGH_RISK_PCTL:.0%} High-Risk, "
          f"~{1-SHED_NOW_PCTL:.0%} Shed-Now):")
    print(df["risk_tier"].value_counts(normalize=True).reindex(TIER_ORDER))

    # Chronological split, per building: train on each building's earlier
    # 80% of hours, test on its later 20%. NOT train_test_split(shuffle=True)
    # -- a random split would interleave train/test hours from the same
    # weeks, which is an easier task than the real deployment scenario
    # (forecasting hours the model has never seen any neighbors of) and
    # would optimistically inflate the regressor's R2.
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    split_idx = df.groupby("building_id").cumcount() < (
        df.groupby("building_id")["building_id"].transform("count") * 0.8
    )
    train_mask = split_idx
    test_mask = ~split_idx

    X = df[feature_cols]
    y_reg = df["target_next_hour"]
    y_clf = df["risk_tier_int"].values  # int labels, mapped via TIER_TO_INT above

    X_train, X_test = X[train_mask], X[test_mask]
    yreg_train, yreg_test = y_reg[train_mask], y_reg[test_mask]
    yclf_train, yclf_test = y_clf[train_mask.values], y_clf[test_mask.values]

    print(f"Chronological split: {train_mask.sum():,} train rows, "
          f"{test_mask.sum():,} test rows (last ~20% of each building's timeline)")

    return dict(df=df, feature_cols=feature_cols, X_train=X_train, X_test=X_test,
                yreg_train=yreg_train, yreg_test=yreg_test,
                yclf_train=yclf_train, yclf_test=yclf_test)


data = build_dataset()   # <-- end of CELL 2: this line actually runs it


# ===== CELL 3 ===== train the regressor (Model 1) - re-run this cell alone anytime
def train_regressor(data):
    print("\nTraining XGBoost Regressor...")
    reg = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
    )
    reg.fit(data["X_train"], data["yreg_train"])
    pred = reg.predict(data["X_test"])

    rmse = (root_mean_squared_error(data["yreg_test"], pred) if root_mean_squared_error
            else mean_squared_error(data["yreg_test"], pred) ** 0.5)
    print(f"  MAE:  {mean_absolute_error(data['yreg_test'], pred):.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R2:   {r2_score(data['yreg_test'], pred):.3f}")

    print("\n  Feature importances (top 10):")
    importances = sorted(zip(data["feature_cols"], reg.feature_importances_), key=lambda x: -x[1])
    for name, score in importances[:10]:
        print(f"    {name:25s} {score:.4f}")

    reg.save_model("regressor_demand.json")
    print("\nSaved regressor_demand.json")
    return reg


reg = train_regressor(data)   # <-- end of CELL 3: this line actually runs it


# ===== CELL 4 ===== train the classifier (Model 2) - re-run this cell alone anytime
def train_classifier(data):
    print("\nTraining XGBoost Classifier...")
    clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        eval_metric="mlogloss"
    )
    clf.fit(data["X_train"], data["yclf_train"])
    pred_c = clf.predict(data["X_test"])

    print(f"  Accuracy: {accuracy_score(data['yclf_test'], pred_c):.3f}")
    print(f"  F1 (macro): {f1_score(data['yclf_test'], pred_c, average='macro'):.3f}")
    print(classification_report(
        data["yclf_test"], pred_c, labels=range(len(TIER_ORDER)), target_names=TIER_ORDER
    ))

    clf.save_model("classifier_risk_tier.json")
    print("Saved classifier_risk_tier.json")
    print(f"Risk tier label order (int -> name), fixed mapping: {list(TIER_TO_INT.items())}")
    return clf


clf = train_classifier(data)   # <-- end of CELL 4: this line actually runs it

# Note: if running this whole file as a single plain Python script (not
# split into Colab cells), the lines above already execute everything -
# no separate __main__ block needed.
