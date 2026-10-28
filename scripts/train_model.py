import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Clean (the median fill comes after the split, so held-out rows don't feed it)
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Share of training rows per practice session; the simulator averages its
# predictions over the session one-hots with these weights (the race is no session)
session_weights = df[SESSION_COLUMNS].mean()
log(f"Session weights: {session_weights.round(3).to_dict()}")


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

# Inputs still missing after feature_engineer.py's stint fill get the median of the training
# rows only; the simulator starts from these medians too (#33)
medians = X_train.median(numeric_only=True)
log(f"Inputs median-filled: {int(X_train.isna().any(axis=1).sum())} train rows, "
    f"{int(X_test.isna().any(axis=1).sum())} test rows; per column: {X.isna().sum()[X.isna().sum() > 0].to_dict()}")
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# Save snapshot
snapshot_name = f"{SCRIPT_NAME}_feature_matrix_snapshot.csv"
X_train.assign(target=y_train).head(200).to_csv(f"{RESULTS_DIR}/{snapshot_name}", index=False)
generated_files.append(snapshot_name)

log("Saved feature snapshot.")


# ======================================================
# MODEL (chosen in ../explore/explore_master.py)
# ======================================================
# The pipeline trains one model. Candidates, tuning and feature sets are
# compared in explore/; the one picked there goes here, with its settings in .env
MODEL_NAME = "XGBoost"
model = XGBRegressor(
    n_estimators=env_int("XGB_N_ESTIMATORS"),
    learning_rate=env_float("XGB_LEARNING_RATE"),
    max_depth=env_int("XGB_MAX_DEPTH"),
    subsample=env_float("XGB_SUBSAMPLE"),
    colsample_bytree=env_float("XGB_COLSAMPLE_BYTREE"),
    random_state=RANDOM_STATE,
    objective="reg:squarederror",
)

log(f"\nTraining {MODEL_NAME}...")
model.fit(X_train, y_train)
pred = model.predict(X_test)


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


metrics_df = pd.DataFrame({MODEL_NAME: compute_metrics(y_test, pred)}).T

metrics_file = f"{SCRIPT_NAME}_model_metrics.csv"
metrics_df.to_csv(f"{RESULTS_DIR}/{metrics_file}")
generated_files.append(metrics_file)

log("\nTest metrics (MAE/RMSE in seconds):")
log(metrics_df.round(4).to_string())
log("Saved model metrics table.")


# ======================================================
# FEATURE IMPORTANCES
# ======================================================
if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_, index=FEATURES)

    imp_file = f"{SCRIPT_NAME}_feature_importance.csv"
    imp.to_csv(f"{RESULTS_DIR}/{imp_file}")
    generated_files.append(imp_file)

    log("Saved feature importance CSV.")


# ======================================================
# SAVE MODEL + FEATURE METADATA
# ======================================================
os.makedirs(os.path.dirname(MODEL_BUNDLE_PATH), exist_ok=True)
model_file = os.path.basename(MODEL_BUNDLE_PATH)
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "medians": medians.to_dict(),
        "session_weights": session_weights.to_dict(),
    },
    MODEL_BUNDLE_PATH
)
generated_files.append(model_file)

log(f"Saved trained {MODEL_NAME} model.")


# ======================================================
# PLOTS
# ======================================================
# Prediction vs true on the held-out stints (first 300 samples)
fig = plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:300], label="True", linewidth=2)
plt.plot(pred[:300], label=MODEL_NAME)
plt.legend()
plt.title(f"{MODEL_NAME} vs True (First 300 Samples)")
plt.ylabel("Fuel Corrected Pace (s)")

pred_plot = f"{SCRIPT_NAME}_pred_vs_true.png"
plt.savefig(f"{RESULTS_DIR}/{pred_plot}", dpi=300)
plt.close()
generated_files.append(pred_plot)

log("Saved prediction plot.")


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