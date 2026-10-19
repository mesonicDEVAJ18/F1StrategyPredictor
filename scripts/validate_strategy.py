import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import ttest_ind
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
from datetime import datetime

from feature_config import build_next_lap_target
from env_config import env_str, env_int

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
    # Remove unsupported characters for Windows console/logging
    safe_msg = msg.encode("utf-8", errors="replace").decode("utf-8", "replace")
    print(safe_msg)
    log_file.write(safe_msg + "\n")


# ============================================================
# CONFIG
# ============================================================
DATASET_PATH = os.path.join(env_str("DATA_DIR"), env_str("ENGINEERED_FILE"))
MODEL_BUNDLE_PATH = env_str("MODEL_BUNDLE_PATH")
CV_FOLDS = env_int("CV_FOLDS")
RANDOM_STATE = env_int("RANDOM_STATE")


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

X = df[FEATURES].values
y = df["target_next_fcp"].values

log(f"Rows after target shift drop: {len(df)}")


# ============================================================
# CROSS-VALIDATION
# ============================================================
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_mae = []
cv_rmse = []

fold_results = []

log(f"\nRunning {CV_FOLDS}-fold cross-validation...")

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])

    mae = mean_absolute_error(y[test_idx], pred)
    rmse = np.sqrt(mean_squared_error(y[test_idx], pred))

    cv_mae.append(mae)
    cv_rmse.append(rmse)

    fold_results.append({
        "fold": fold,
        "MAE": mae,
        "RMSE": rmse
    })

    log(f" Fold {fold} → MAE={mae:.4f} s, RMSE={rmse:.4f} s")

mean_mae = np.mean(cv_mae)
mean_rmse = np.mean(cv_rmse)

log(f"\nCV MAE = {mean_mae} s")
log(f"CV RMSE = {mean_rmse} s")


# Save fold-by-fold results
fold_df = pd.DataFrame(fold_results)
fold_file = f"{SCRIPT_NAME}_cv_fold_metrics.csv"
fold_df.to_csv(f"{RESULTS_DIR}/{fold_file}", index=False)
generated_files.append(fold_file)


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================
log("\nGenerating residual plot...")

pred_full = model.predict(X)
residuals = y - pred_full

N = min(3000, len(y))

plt.figure(figsize=(9,4))
plt.scatter(pred_full[:N], residuals[:N], s=4, alpha=0.4)
plt.axhline(0, color="red", linewidth=1)
plt.title("Residual Plot")
plt.xlabel("Predicted Fuel-Corrected Pace (s)")
plt.ylabel("Residual (s)")

residual_file = f"{SCRIPT_NAME}_residual_plot.png"
plt.savefig(f"{RESULTS_DIR}/{residual_file}", dpi=300)
plt.close()
generated_files.append(residual_file)

log("Residual plot saved.")


# ============================================================
# T-TEST EXAMPLE
# ============================================================
log("\nRunning example t-test...")

s1 = np.random.normal(4800, 10, size=500)
s2 = np.random.normal(4820, 10, size=500)

t, p = ttest_ind(s1, s2)

log(f"T-test p-value: {p}")

# Save t-test distributions
ttest_df = pd.DataFrame({"sample1": s1, "sample2": s2})
ttest_file = f"{SCRIPT_NAME}_ttest_samples.csv"
ttest_df.to_csv(f"{RESULTS_DIR}/{ttest_file}", index=False)
generated_files.append(ttest_file)


# ============================================================
# SUMMARY OUTPUTS
# ============================================================
summary = pd.DataFrame({
    "CV_MAE": [mean_mae],
    "CV_RMSE": [mean_rmse],
    "t_test_p": [p]
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

