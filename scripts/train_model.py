import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
import os
import warnings
from datetime import datetime

from feature_config import MODEL_FEATURE_COLUMNS as FEATURES
from feature_config import SESSION_COLUMNS, TARGET_GROUP_COLUMNS, build_next_lap_target
from env_config import env_str, env_int, env_float

warnings.filterwarnings("ignore")

# ======================================================
# SETUP RESULTS DIRECTORY
# ======================================================
SCRIPT_NAME = "train_model"
RESULTS_DIR = f"{env_str('RESULTS_ROOT')}/{SCRIPT_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)

log_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_log.txt"
log_file = open(log_path, "w", encoding="utf-8", errors="replace")

generated_files = []

def log(msg: str):
    # Replace characters the console/redirect encoding can't show (e.g. cp1252 when output is
    # redirected on Windows); the log file is UTF-8 and keeps the original text
    enc = sys.stdout.encoding or "utf-8"
    safe_msg = msg.encode(enc, errors="replace").decode(enc, "replace")
    print(safe_msg)
    log_file.write(msg + "\n")

# ======================================================
# CONFIG
# ======================================================
DATASET_PATH = os.path.join(env_str("DATA_DIR"), env_str("PROCESSED_DIR"), env_str("ENGINEERED_FILE"))
MODEL_BUNDLE_PATH = env_str("MODEL_BUNDLE_PATH")
RANDOM_STATE = env_int("RANDOM_STATE")
TEST_SIZE = env_float("TEST_SIZE")

# ======================================================
# LOAD DATA
# ======================================================
log("Loading engineered dataset...")

df = pd.read_csv(DATASET_PATH)

log(f"Dataset rows: {len(df)}")

# ======================================================
# BUILD TARGET
# ======================================================
rows_before = len(df)
n_stints = df.groupby(TARGET_GROUP_COLUMNS).ngroups
df = build_next_lap_target(df)

log(f"Rows after target shift drop: {len(df)}")
log(f"  → Dropped as last lap of a stint: {n_stints}")
log(f"  → Dropped for a gap to the next lap: {rows_before - len(df) - n_stints}")


# ======================================================
# FEATURES / TARGET
# ======================================================
X = df[FEATURES].copy()
y = df["target_next_fcp"]

# Clean
X.replace([np.inf, -np.inf], np.nan, inplace=True)
medians = X.median(numeric_only=True)
X = X.fillna(medians)

# Share of training rows per practice session; the simulator averages its
# predictions over the session one-hots with these weights (the race is no session)
session_weights = df[SESSION_COLUMNS].mean()
log(f"Session weights: {session_weights.round(3).to_dict()}")

# Save snapshot
snapshot_name = f"{SCRIPT_NAME}_feature_matrix_snapshot.csv"
X.assign(target=y).head(200).to_csv(f"{RESULTS_DIR}/{snapshot_name}", index=False)
generated_files.append(snapshot_name)

log("Saved feature snapshot.")


# ======================================================
# TRAIN/TEST SPLIT
# ======================================================
# Split by stint, not by lap: laps of one stint are highly correlated, and the
# simulator predicts stints the model hasn't seen. TEST_SIZE is a share of stints.
groups = df.groupby(TARGET_GROUP_COLUMNS).ngroup()
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])
log(
    f"Completed train/test split by stint: {len(train_groups)} stints / {len(train_idx)} rows train, "
    f"{len(test_groups)} stints / {len(test_idx)} rows test, {len(train_groups & test_groups)} shared."
)


# ======================================================
# MODELS
# ======================================================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=env_int("RF_N_ESTIMATORS"), random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=env_int("XGB_N_ESTIMATORS"),
        learning_rate=env_float("XGB_LEARNING_RATE"),
        max_depth=env_int("XGB_MAX_DEPTH"),
        subsample=env_float("XGB_SUBSAMPLE"),
        colsample_bytree=env_float("XGB_COLSAMPLE_BYTREE"),
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
    )
}

preds = {}
fitted_models = {}

log("\nTraining models...")
for name, model in models.items():
    log(f"Training {name}...")
    model.fit(X_train, y_train)
    fitted_models[name] = model
    preds[name] = model.predict(X_test)


# ======================================================
# METRICS
# ======================================================
def compute_metrics(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    rmse = np.sqrt(mse)
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, pred),
    }


metrics_df = pd.DataFrame({
    name: compute_metrics(y_test, preds[name])
    for name in models
}).T

metrics_file = f"{SCRIPT_NAME}_model_metrics.csv"
metrics_df.to_csv(f"{RESULTS_DIR}/{metrics_file}")
generated_files.append(metrics_file)

log("\nTest metrics (MAE/RMSE in seconds):")
log(metrics_df.round(4).to_string())
log("Saved model metrics table.")


# ======================================================
# FEATURE IMPORTANCES
# ======================================================
if "RandomForest" in fitted_models:
    rf_imp = pd.Series(
        fitted_models["RandomForest"].feature_importances_,
        index=FEATURES
    )

    rf_file = f"{SCRIPT_NAME}_feature_importance_rf.csv"
    rf_imp.to_csv(f"{RESULTS_DIR}/{rf_file}")
    generated_files.append(rf_file)

if "XGBoost" in fitted_models:
    xgb_imp = pd.Series(
        fitted_models["XGBoost"].feature_importances_,
        index=FEATURES
    )

    xgb_file = f"{SCRIPT_NAME}_feature_importance_xgb.csv"
    xgb_imp.to_csv(f"{RESULTS_DIR}/{xgb_file}")
    generated_files.append(xgb_file)

log("Saved feature importance CSVs.")


# ======================================================
# SAVE MODEL + FEATURE METADATA
# ======================================================
os.makedirs(os.path.dirname(MODEL_BUNDLE_PATH), exist_ok=True)
model_file = os.path.basename(MODEL_BUNDLE_PATH)
joblib.dump(
    {
        "model": fitted_models["XGBoost"],
        "features": FEATURES,
        "medians": medians.to_dict(),
        "session_weights": session_weights.to_dict(),
    },
    MODEL_BUNDLE_PATH
)
generated_files.append(model_file)

log("Saved trained XGB model.")


# ======================================================
# PLOTS
# ======================================================
# 1 — Prediction overlay (first 300 samples)
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:300], label="True", linewidth=2)
for name in models:
    plt.plot(preds[name][:300], label=name)
plt.legend()
plt.title("Model Predictions (First 300 Samples)")
plt.xlabel("Index")
plt.ylabel("Fuel Corrected Pace (s)")

overlay_file = f"{SCRIPT_NAME}_prediction_overlay.png"
plt.savefig(f"{RESULTS_DIR}/{overlay_file}", dpi=300)
plt.close()
generated_files.append(overlay_file)


# 2 — Metric bar chart
fig = plt.figure(figsize=(10, 6))
bar_width = 0.25
idx = np.arange(len(metrics_df))

plt.bar(idx, metrics_df["MAE"], bar_width, label="MAE")
plt.bar(idx + bar_width, metrics_df["RMSE"], bar_width, label="RMSE")
plt.bar(idx + 2 * bar_width, metrics_df["R2"], bar_width, label="R²")

plt.xticks(idx + bar_width, metrics_df.index)
plt.title("Model Metrics")
plt.ylabel("Value (MAE/RMSE in s)")
plt.legend()

metric_plot = f"{SCRIPT_NAME}_metric_bars.png"
plt.savefig(f"{RESULTS_DIR}/{metric_plot}", dpi=300)
plt.close()
generated_files.append(metric_plot)


# 3 — XGB vs true
fig = plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:300], label="True", linewidth=2)
plt.plot(preds["XGBoost"][:300], label="XGBoost")
plt.legend()
plt.title("XGB vs True (First 300 Samples)")
plt.ylabel("Fuel Corrected Pace (s)")

xgb_plot = f"{SCRIPT_NAME}_xgb_vs_true.png"
plt.savefig(f"{RESULTS_DIR}/{xgb_plot}", dpi=300)
plt.close()
generated_files.append(xgb_plot)

log("Saved all plots.")


# ======================================================
# MANIFEST
# ======================================================
manifest_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt"
with open(manifest_path, "w") as mf:
    for f in generated_files:
        mf.write(f + "\n")

log("\nGenerated files:")
for f in generated_files:
    log("  - " + f)

log_file.close()