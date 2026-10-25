import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
from datetime import datetime

from feature_config import TARGET_GROUP_COLUMNS, build_next_lap_target
from env_config import env_str, env_int, env_float

warnings.filterwarnings("ignore")

# ============================================================
# SETUP RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "validate_strategy"
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


# ============================================================
# CONFIG
# ============================================================
DATASET_PATH = os.path.join(env_str("DATA_DIR"), env_str("PROCESSED_DIR"), env_str("ENGINEERED_FILE"))
MODEL_BUNDLE_PATH = env_str("MODEL_BUNDLE_PATH")
CV_FOLDS = env_int("CV_FOLDS")
RANDOM_STATE = env_int("RANDOM_STATE")
SIGNIFICANCE_LEVEL = env_float("SIGNIFICANCE_LEVEL")


# ============================================================
# LOAD MODEL + FEATURES
# ============================================================
bundle = joblib.load(MODEL_BUNDLE_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

log("Loaded model and feature metadata.")


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATASET_PATH)

log(f"Loaded dataset with {len(df)} rows.")


# ============================================================
# TARGET (one-step-ahead, same driver and stint, lap + 1)
# ============================================================
df = build_next_lap_target(df)

# Fill missing inputs (e.g. stint_avg_pace on a stint's first lap) with the training medians, as train_model.py does
X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(bundle["medians"]).values
y = df["target_next_fcp"].values
# One group per stint, so a stint's laps never sit in both the train and test folds
groups = df.groupby(TARGET_GROUP_COLUMNS).ngroup().values
# Persistence baseline: lap t+1's fcp = lap t's fcp. Nothing is fitted, so it is out-of-sample for every row
baseline_pred = df["fuel_corrected_pace"].values

log(f"Rows after target shift drop: {len(df)} ({len(np.unique(groups))} stints)")


# ============================================================
# CROSS-VALIDATION
# ============================================================
kf = GroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_mae = []
cv_rmse = []

fold_results = []

# Out-of-fold predictions: each row is predicted by the fold that held it out
oof_pred = np.full(len(y), np.nan)
oof_fold = np.zeros(len(y), dtype=int)

log(f"\nRunning {CV_FOLDS}-fold cross-validation grouped by stint...")

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, groups), 1):
    test_stints = len(np.unique(groups[test_idx]))
    shared = len(np.intersect1d(groups[train_idx], groups[test_idx]))
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    oof_pred[test_idx] = pred
    oof_fold[test_idx] = fold

    mae =mean_absolute_error(y[test_idx], pred)
    rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
    base_mae = mean_absolute_error(y[test_idx], baseline_pred[test_idx])

    cv_mae.append(mae)
    cv_rmse.append(rmse)

    fold_results.append({
        "fold": fold,
        "test_stints": test_stints,
        "test_rows": len(test_idx),
        "MAE": mae,
        "RMSE": rmse,
        "baseline_MAE": base_mae
    })

    log(
        f" Fold {fold} → {test_stints} stints / {len(test_idx)} rows, {shared} shared, "
        f"MAE={mae:.4f} s, RMSE={rmse:.4f} s, baseline MAE={base_mae:.4f} s"
    )

mean_mae = np.mean(cv_mae)
mean_rmse = np.mean(cv_rmse)

log(f"\nCV MAE = {mean_mae} s")
log(f"CV RMSE = {mean_rmse} s")

# GroupKFold puts every row in exactly one test fold
if np.isnan(oof_pred).any():
    raise RuntimeError(f"{int(np.isnan(oof_pred).sum())} rows got no out-of-fold prediction")
log(f"Out-of-fold predictions: {len(oof_pred)} rows, each predicted once")


# Save fold-by-fold results
fold_df = pd.DataFrame(fold_results)
fold_file = f"{SCRIPT_NAME}_cv_fold_metrics.csv"
fold_df.to_csv(f"{RESULTS_DIR}/{fold_file}", index=False)
generated_files.append(fold_file)


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================
log("\nGenerating residual plot...")

# Out-of-fold, so no row is scored by a model that trained on it
residuals = y - oof_pred

oof_mae = mean_absolute_error(y, oof_pred)
oof_rmse = np.sqrt(mean_squared_error(y, oof_pred))
oof_bias = residuals.mean()

log(f"OOF MAE = {oof_mae:.4f} s, OOF RMSE = {oof_rmse:.4f} s, mean residual = {oof_bias:+.4f} s")

oof_df = df[["session_key", "driver_number", "stint_number", "lap_number", "target_next_fcp"]].copy()
oof_df["oof_pred"] = oof_pred
oof_df["residual"] = residuals
oof_df["fold"] = oof_fold
oof_df["baseline_pred"] = baseline_pred
oof_file = f"{SCRIPT_NAME}_oof_predictions.csv"
oof_df.to_csv(f"{RESULTS_DIR}/{oof_file}", index=False)
generated_files.append(oof_file)

plt.figure(figsize=(9,4))
plt.scatter(oof_pred, residuals, s=4, alpha=0.4)
plt.axhline(0, color="red", linewidth=1)
plt.title("Residual Plot (out-of-fold, grouped by stint)")
plt.xlabel("Predicted Fuel-Corrected Pace (s)")
plt.ylabel("Residual (s)")

residual_file = f"{SCRIPT_NAME}_residual_plot.png"
plt.savefig(f"{RESULTS_DIR}/{residual_file}", dpi=300)
plt.close()
generated_files.append(residual_file)

log("Residual plot saved.")


# ============================================================
# BASELINE SIGNIFICANCE TEST
# ============================================================
log("\nTesting the model against the persistence baseline (lap t+1 = lap t)...")

base_mae = mean_absolute_error(y, baseline_pred)
base_rmse = np.sqrt(mean_squared_error(y, baseline_pred))

log(f"Baseline MAE = {base_mae:.4f} s, baseline RMSE = {base_rmse:.4f} s")

# Laps of one stint aren't independent, so each stint is one paired observation
stint_df = oof_df.assign(
    model_abs_err=np.abs(residuals),
    baseline_abs_err=np.abs(y - baseline_pred)
)
stint_df = stint_df.groupby(TARGET_GROUP_COLUMNS).agg(
    session_key=("session_key", "first"),
    rows=("lap_number", "size"),
    fold=("fold", "first"),
    model_MAE=("model_abs_err", "mean"),
    baseline_MAE=("baseline_abs_err", "mean")
).reset_index()
stint_df["diff"] = stint_df["model_MAE"] - stint_df["baseline_MAE"]

# One-sided: is the model's per-stint error lower than the baseline's?
t_stat, t_p = ttest_rel(stint_df["model_MAE"], stint_df["baseline_MAE"], alternative="less")
# Wilcoxon signed-rank as a check, since the residuals are right-skewed
_, w_p = wilcoxon(stint_df["model_MAE"], stint_df["baseline_MAE"], alternative="less")

model_better = int((stint_df["diff"] < 0).sum())
log(
    f"Stints: {len(stint_df)}, model better on {model_better}; "
    f"per-stint MAE difference mean {stint_df['diff'].mean():+.4f} s, median {stint_df['diff'].median():+.4f} s"
)
log(f"Paired t-test (one-sided): t = {t_stat:.4f}, p = {t_p:.6f}")
log(f"Wilcoxon signed-rank (one-sided): p = {w_p:.6f}")
if t_p < SIGNIFICANCE_LEVEL and w_p < SIGNIFICANCE_LEVEL:
    log(f"Model beats the persistence baseline on unseen stints at alpha = {SIGNIFICANCE_LEVEL}")
else:
    log(f"Model does NOT significantly beat the persistence baseline at alpha = {SIGNIFICANCE_LEVEL}")

stint_file = f"{SCRIPT_NAME}_baseline_test_stints.csv"
stint_df.to_csv(f"{RESULTS_DIR}/{stint_file}", index=False)
generated_files.append(stint_file)


# ============================================================
# SUMMARY OUTPUTS
# ============================================================
summary = pd.DataFrame({
    "CV_MAE": [mean_mae],
    "CV_RMSE": [mean_rmse],
    "OOF_MAE": [oof_mae],
    "OOF_RMSE": [oof_rmse],
    "OOF_mean_residual": [oof_bias],
    "baseline_MAE": [base_mae],
    "baseline_RMSE": [base_rmse],
    "paired_t_stat": [t_stat],
    "paired_t_p": [t_p],
    "wilcoxon_p": [w_p]
})

summary_file = f"{SCRIPT_NAME}_validation_summary.csv"
summary.to_csv(f"{RESULTS_DIR}/{summary_file}", index=False)
generated_files.append(summary_file)

log("\nValidation summary saved.")


# ============================================================
# MANIFEST
# ============================================================
manifest_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt"
with open(manifest_path, "w") as mf:
    for f in generated_files:
        mf.write(f + "\n")

log("\nFILES GENERATED:")
for f in generated_files:
    log(f"  - {f}")

log("\n--- VALIDATION COMPLETE ---\n")
log_file.close()

