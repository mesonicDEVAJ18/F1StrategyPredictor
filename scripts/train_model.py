import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
import os
import warnings
from datetime import datetime

from feature_config import MODEL_FEATURE_COLUMNS as FEATURES
from feature_config import (
    COMPOUNDS, FALL_OFF_ANCHOR_AGE, STINT_GROUP_COLUMNS, fall_off_curve, fall_off_rows, fall_off_target,
    fit_compound_gap, fit_stint_levels, linear_fall_off, multi_lap_rows, stint_shape_errors,
)
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
# TRAIN/TEST SPLIT
# ======================================================
# Split by stint, not by lap: laps of one stint are highly correlated, and the
# simulator prices stints the model hasn't seen. TEST_SIZE is a share of stints.
groups = df.groupby(STINT_GROUP_COLUMNS).ngroup()
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(df, groups=groups))
train_all, test_all = df.iloc[train_idx], df.iloc[test_idx]
log(
    f"Completed train/test split by stint: {train_all.groupby(STINT_GROUP_COLUMNS).ngroups} stints / "
    f"{len(train_all)} laps train, {test_all.groupby(STINT_GROUP_COLUMNS).ngroups} stints / {len(test_all)} laps test, "
    f"{len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))} shared."
)


# ======================================================
# TARGET: FALL-OFF WITHIN THE STINT (#31)
# ======================================================
# fcp = level[stint] + fall_off(compound, tyre_age, conditions). Stage 1 fits the stint levels
# (and one linear slope per compound) on the training stints; the model learns fcp - level.
# Only stints with two or more clean laps carry fall-off.
train = multi_lap_rows(train_all)
test = multi_lap_rows(test_all)
log(f"Fall-off rows (stints with >= 2 clean laps): {len(train)} laps / "
    f"{train.groupby(STINT_GROUP_COLUMNS).ngroups} stints train, {len(test)} laps / "
    f"{test.groupby(STINT_GROUP_COLUMNS).ngroups} stints test")

levels, linear_slopes = fit_stint_levels(train)
log("Stage 1, linear fall-off per compound (s per lap of tyre age): "
    + ", ".join(f"{c} {linear_slopes[c]:+.3f}" for c in COMPOUNDS))

X_train = train[FEATURES].replace([np.inf, -np.inf], np.nan)
X_test = test[FEATURES].replace([np.inf, -np.inf], np.nan)
y_train = fall_off_target(train, levels)

# Inputs still missing get the median of the training rows only; the simulator holds the
# conditions at these medians
medians = X_train.median(numeric_only=True)
log(f"Inputs median-filled: {int(X_train.isna().any(axis=1).sum())} train rows, "
    f"{int(X_test.isna().any(axis=1).sum())} test rows")
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
# METRICS (shape of the curve within each held-out stint)
# ======================================================
# A held-out stint's level is unknown, so each stint's mean residual is removed and only the
# change with tyre age is scored. R2_within: share of the within-stint variation explained.
test_stints = test.groupby(STINT_GROUP_COLUMNS).ngroup().to_numpy()
fcp_test = test["fuel_corrected_pace"].to_numpy()
shape = {
    MODEL_NAME: stint_shape_errors(fcp_test, pred, test_stints),
    "baseline_flat": stint_shape_errors(fcp_test, np.zeros(len(test)), test_stints),
    "baseline_linear": stint_shape_errors(fcp_test, linear_fall_off(test, linear_slopes), test_stints),
}
flat_sse = np.sum(shape["baseline_flat"] ** 2)
metrics_df = pd.DataFrame({
    name: {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)), "R2_within": 1 - np.sum(e ** 2) / flat_sse}
    for name, e in shape.items()
}).T

metrics_file = f"{SCRIPT_NAME}_model_metrics.csv"
metrics_df.to_csv(f"{RESULTS_DIR}/{metrics_file}")
generated_files.append(metrics_file)

log("\nHeld-out within-stint shape metrics (MAE/RMSE in seconds):")
log(metrics_df.round(4).to_string())
log("Saved model metrics table.")


# ======================================================
# FALL-OFF CURVES AND COMPOUND GAP
# ======================================================
conditions = medians.to_dict()
curve_ages = np.arange(1, int(df["tyre_age"].max()) + 2)
curves = {c: fall_off_curve(model, FEATURES, c, curve_ages, conditions) for c in COMPOUNDS}
table = pd.DataFrame({
    "tyre_age": curve_ages,
    **{f"{c}_model": curves[c] for c in COMPOUNDS},
    **{f"{c}_linear": linear_slopes[c] * (curve_ages - FALL_OFF_ANCHOR_AGE) for c in COMPOUNDS},
})
curves_file = f"{SCRIPT_NAME}_fall_off_curves.csv"
table.to_csv(f"{RESULTS_DIR}/{curves_file}", index=False)
generated_files.append(curves_file)
log(f"\nFall-off from a new set's first lap (s), training-median conditions; tyre_age in training "
    f"{int(train['tyre_age'].min())}-{int(train['tyre_age'].max())}:")
for c in COMPOUNDS:
    at = ", ".join(f"age {a}: {curves[c][a - 1]:+.2f} (linear {linear_slopes[c] * (a - FALL_OFF_ANCHOR_AGE):+.2f})"
                   for a in (5, 10, 15) if a <= len(curve_ages))
    log(f"  {c}: {at}")

# Stage 3: the gap between compounds on a new set, net of the model's fall-off, from every
# training lap (one-lap stints too), with run_laps as a control for the run type
gap_fall_off = fall_off_rows(model, FEATURES, train_all, conditions)
compound_gap, pace_levels, run_coef = fit_compound_gap(train_all, gap_fall_off)
log(f"Stage 3, compound gap on a new set (s, vs SOFT): MEDIUM {compound_gap['MEDIUM']:+.3f}, "
    f"HARD {compound_gap['HARD']:+.3f}; run_laps {run_coef:+.3f} s per lap of run; "
    f"{len(pace_levels)} driver-session levels from {len(train_all)} laps")


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
        "anchor_age": FALL_OFF_ANCHOR_AGE,
        "compound_gap": compound_gap,
        "levels": pace_levels.to_dict(),
        "linear_slopes": linear_slopes,
    },
    MODEL_BUNDLE_PATH
)
generated_files.append(model_file)

log(f"Saved trained {MODEL_NAME} model.")


# ======================================================
# PLOTS
# ======================================================
# Fall-off curve of each compound: the model against the stage-1 linear slope
fig = plt.figure(figsize=(10, 6))
colours = {"SOFT": "tab:red", "MEDIUM": "goldenrod", "HARD": "dimgray"}
for c in COMPOUNDS:
    plt.plot(curve_ages, curves[c], color=colours[c], label=f"{c} {MODEL_NAME}")
    plt.plot(curve_ages, table[f"{c}_linear"], color=colours[c], linestyle="--", label=f"{c} linear")
plt.axvspan(train["tyre_age"].max(), curve_ages.max(), color="0.9", label="beyond training ages")
plt.legend()
plt.title("Fall-off from a new set's first lap (training-median conditions)")
plt.xlabel("Tyre age (laps)")
plt.ylabel("Fall-off (s)")

pred_plot = f"{SCRIPT_NAME}_fall_off_curves.png"
plt.savefig(f"{RESULTS_DIR}/{pred_plot}", dpi=300)
plt.close()
generated_files.append(pred_plot)

log("Saved fall-off plot.")


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