import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

warnings.filterwarnings("ignore")

# ======================================================
# SETUP RESULTS DIRECTORY
# ======================================================
SCRIPT_NAME = "train_model"
RESULTS_DIR = f"./results/{SCRIPT_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)

log_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_log.txt"
log_file = open(log_path, "w", encoding="utf-8", errors="replace")

generated_files = []

def log(msg: str):
    # Remove unsupported characters for Windows console/logging
    safe_msg = msg.encode("utf-8", errors="replace").decode("utf-8", "replace")
    print(safe_msg)
    log_file.write(safe_msg + "\n")

# ======================================================
# LOAD DATA
# ======================================================
log("Loading engineered dataset...")

df = pd.read_csv("../data/final_engineered_dataset.csv")
df = df.sort_values(["stint_number", "lap_number"]).reset_index(drop=True)

log(f"Dataset rows: {len(df)}")

# ======================================================
# BUILD TARGET
# ======================================================
df["target_next_fcp"] = df.groupby("stint_number")["fuel_corrected_pace"].shift(-1)
df = df.dropna(subset=["target_next_fcp"]).reset_index(drop=True)

log(f"Rows after target shift drop: {len(df)}")


# ======================================================
# FEATURES / TARGET
# ======================================================
X = df[FEATURES].copy()
y = df["target_next_fcp"]

# Clean
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.median(numeric_only=True))

# Save snapshot
snapshot_name = f"{SCRIPT_NAME}_feature_matrix_snapshot.csv"
X.assign(target=y).head(200).to_csv(f"{RESULTS_DIR}/{snapshot_name}", index=False)
generated_files.append(snapshot_name)

log("Saved feature snapshot.")


# ======================================================
# TRAIN/TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, random_state=42
)
log("Completed train/test split.")


# ======================================================
# MODELS
# ======================================================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
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

log("\nSaved model metrics table.")


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
model_file = f"{SCRIPT_NAME}_xgb_model.pkl"
joblib.dump(
    {"model": fitted_models["XGBoost"], "features": FEATURES},
    f"{RESULTS_DIR}/{model_file}"
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
plt.ylabel("Fuel Corrected Pace")

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
plt.ylabel("Value")
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