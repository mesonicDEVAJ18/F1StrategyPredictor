import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from scipy.stats import ttest_rel, wilcoxon
import os
import warnings
from datetime import datetime

from feature_config import (
    COMPOUNDS, STINT_GROUP_COLUMNS, compound_of, fall_off_curve, fall_off_rows, fall_off_target, fit_compound_gap,
    fit_stint_levels, linear_fall_off, multi_lap_rows, stint_shape_errors,
)
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
# ROWS (fall-off within the stint, #31)
# ============================================================
# Only stints with two or more clean laps carry fall-off. Each fold fits its own stage-1 stint
# levels and input medians on its training rows, as train_model.py does with its split
rows = multi_lap_rows(df)
groups = rows.groupby(STINT_GROUP_COLUMNS).ngroup().to_numpy()
fcp = rows["fuel_corrected_pace"].to_numpy()
X_all = rows[FEATURES].replace([np.inf, -np.inf], np.nan)
CURVE_AGES = [5, 10, 15]

log(f"Fall-off rows (stints with >= 2 clean laps): {len(rows)} laps, {len(np.unique(groups))} stints")


# ============================================================
# CROSS-VALIDATION
# ============================================================
# A held-out stint's level is unknown, so the score is the shape of the curve: each stint's
# mean residual is removed (stint_shape_errors). Two baselines: no fall-off at all (flat) and
# the stage-1 linear slope per compound (linear)
kf = GroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

ERR_NAMES = ["model", "flat", "linear"]
oof = {n: np.full(len(rows), np.nan) for n in ERR_NAMES}   # out-of-fold shape errors
oof_fall_off = np.full(len(rows), np.nan)
oof_fold = np.zeros(len(rows), dtype=int)
fold_results, curve_rows = [], []

log(f"\nRunning {CV_FOLDS}-fold cross-validation grouped by stint...")

for fold, (train_idx, test_idx) in enumerate(kf.split(rows, groups=groups), 1):
    train, test = rows.iloc[train_idx], rows.iloc[test_idx]
    test_stints = len(np.unique(groups[test_idx]))
    shared = len(np.intersect1d(groups[train_idx], groups[test_idx]))

    levels, slopes = fit_stint_levels(train)
    fold_medians = X_all.iloc[train_idx].median()
    model.fit(X_all.iloc[train_idx].fillna(fold_medians), fall_off_target(train, levels))
    pred = model.predict(X_all.iloc[test_idx].fillna(fold_medians))

    errs = {
        "model": stint_shape_errors(fcp[test_idx], pred, groups[test_idx]),
        "flat": stint_shape_errors(fcp[test_idx], np.zeros(len(test_idx)), groups[test_idx]),
        "linear": stint_shape_errors(fcp[test_idx], linear_fall_off(test, slopes), groups[test_idx]),
    }
    for n in ERR_NAMES:
        oof[n][test_idx] = errs[n]
    conditions = fold_medians.to_dict()
    oof_fall_off[test_idx] = fall_off_rows(model, FEATURES, test, conditions)
    oof_fold[test_idx] = fold

    # The fold's curve, to see how much the fall-off moves with the training stints
    for c in COMPOUNDS:
        curve = fall_off_curve(model, FEATURES, c, CURVE_AGES, conditions)
        curve_rows += [{"fold": fold, "compound": c, "tyre_age": age, "fall_off": v,
                        "linear_fall_off": slopes[c] * (age - 1)} for age, v in zip(CURVE_AGES, curve)]

    # Stage-3 compound gap on the fold's training stints (one-lap stints too)
    gap_rows = df[~df.set_index(STINT_GROUP_COLUMNS).index.isin(test.set_index(STINT_GROUP_COLUMNS).index.unique())]
    gap, _, _ = fit_compound_gap(gap_rows, fall_off_rows(model, FEATURES, gap_rows, conditions))

    res = {"fold": fold, "test_stints": test_stints, "test_rows": len(test_idx)}
    for n in ERR_NAMES:
        res[f"{n}_MAE"] = np.mean(np.abs(errs[n]))
        res[f"{n}_RMSE"] = np.sqrt(np.mean(errs[n] ** 2))
    res.update({"gap_MEDIUM": gap["MEDIUM"], "gap_HARD": gap["HARD"]})
    fold_results.append(res)

    log(
        f" Fold {fold} → {test_stints} stints / {len(test_idx)} rows, {shared} shared, "
        f"shape MAE model {res['model_MAE']:.4f} s, flat {res['flat_MAE']:.4f} s, linear {res['linear_MAE']:.4f} s; "
        f"gap MEDIUM {gap['MEDIUM']:+.2f}, HARD {gap['HARD']:+.2f} s"
    )

fold_df = pd.DataFrame(fold_results)
mean_mae = fold_df["model_MAE"].mean()
mean_rmse = fold_df["model_RMSE"].mean()

log(f"\nCV shape MAE = {mean_mae} s")
log(f"CV shape RMSE = {mean_rmse} s")

# GroupKFold puts every row in exactly one test fold
if np.isnan(oof["model"]).any():
    raise RuntimeError(f"{int(np.isnan(oof['model']).sum())} rows got no out-of-fold prediction")
log(f"Out-of-fold predictions: {len(rows)} rows, each predicted once")

# Save fold-by-fold results
fold_file = f"{SCRIPT_NAME}_cv_fold_metrics.csv"
fold_df.to_csv(f"{RESULTS_DIR}/{fold_file}", index=False)
generated_files.append(fold_file)

# Fall-off curve per fold: its spread is the curve's uncertainty
curve_df = pd.DataFrame(curve_rows)
curve_file = f"{SCRIPT_NAME}_fold_curves.csv"
curve_df.to_csv(f"{RESULTS_DIR}/{curve_file}", index=False)
generated_files.append(curve_file)
spread = curve_df.groupby(["compound", "tyre_age"], sort=False)["fall_off"].agg(["mean", "std", "min", "max"])
log("\nFall-off per fold from a new set's first lap (s), over the folds:")
log(spread.round(2).to_string())


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================
log("\nGenerating residual plot...")

oof_mae = np.mean(np.abs(oof["model"]))
oof_rmse = np.sqrt(np.mean(oof["model"] ** 2))
log(f"OOF shape MAE = {oof_mae:.4f} s, OOF shape RMSE = {oof_rmse:.4f} s")

oof_df = rows[["session_key", "driver_number", "stint_number", "lap_number", "tyre_age", "fuel_corrected_pace"]].copy()
oof_df.insert(4, "compound", compound_of(rows))
oof_df["oof_fall_off"] = oof_fall_off
oof_df["shape_residual"] = oof["model"]
oof_df["flat_shape_residual"] = oof["flat"]
oof_df["linear_shape_residual"] = oof["linear"]
oof_df["fold"] = oof_fold
oof_file = f"{SCRIPT_NAME}_oof_predictions.csv"
oof_df.to_csv(f"{RESULTS_DIR}/{oof_file}", index=False)
generated_files.append(oof_file)

# Residual against tyre age: a trend left here is fall-off the model misses
plt.figure(figsize=(9,4))
colours = {"SOFT": "tab:red", "MEDIUM": "goldenrod", "HARD": "dimgray"}
for c in COMPOUNDS:
    part = oof_df[oof_df["compound"] == c]
    plt.scatter(part["tyre_age"], part["shape_residual"], s=6, alpha=0.5, color=colours[c], label=c)
plt.axhline(0, color="red", linewidth=1)
plt.title("Within-stint residual (out-of-fold, grouped by stint)")
plt.xlabel("Tyre age (laps)")
plt.ylabel("Residual minus stint mean (s)")
plt.legend()

residual_file = f"{SCRIPT_NAME}_residual_plot.png"
plt.savefig(f"{RESULTS_DIR}/{residual_file}", dpi=300)
plt.close()
generated_files.append(residual_file)

log("Residual plot saved.")


# ============================================================
# BASELINE SIGNIFICANCE TESTS
# ============================================================
# Laps of one stint aren't independent, so each stint is one paired observation
stint_df = oof_df.assign(**{f"{n}_abs_err": np.abs(oof[n]) for n in ERR_NAMES})
stint_df = stint_df.groupby(STINT_GROUP_COLUMNS).agg(
    session_key=("session_key", "first"),
    compound=("compound", "first"),
    rows=("lap_number", "size"),
    fold=("fold", "first"),
    **{f"{n}_MAE": (f"{n}_abs_err", "mean") for n in ERR_NAMES}
).reset_index()

tests = {}
for base in ["flat", "linear"]:
    log(f"\nTesting the model against the {base} baseline (shape MAE {np.mean(np.abs(oof[base])):.4f} s, "
        f"RMSE {np.sqrt(np.mean(oof[base] ** 2)):.4f} s)...")
    diff = stint_df["model_MAE"] - stint_df[f"{base}_MAE"]
    # One-sided: is the model's per-stint error lower than the baseline's?
    t_stat, t_p = ttest_rel(stint_df["model_MAE"], stint_df[f"{base}_MAE"], alternative="less")
    # Wilcoxon signed-rank as a check, since the errors are skewed
    _, w_p = wilcoxon(stint_df["model_MAE"], stint_df[f"{base}_MAE"], alternative="less")
    tests[base] = (t_stat, t_p, w_p)
    log(
        f"Stints: {len(stint_df)}, model better on {int((diff < 0).sum())}; "
        f"per-stint MAE difference mean {diff.mean():+.4f} s, median {diff.median():+.4f} s"
    )
    log(f"Paired t-test (one-sided): t = {t_stat:.4f}, p = {t_p:.6f}")
    log(f"Wilcoxon signed-rank (one-sided): p = {w_p:.6f}")
    if t_p < SIGNIFICANCE_LEVEL and w_p < SIGNIFICANCE_LEVEL:
        log(f"Model beats the {base} baseline on unseen stints at alpha = {SIGNIFICANCE_LEVEL}")
    else:
        log(f"Model does NOT significantly beat the {base} baseline at alpha = {SIGNIFICANCE_LEVEL}")

stint_file = f"{SCRIPT_NAME}_baseline_test_stints.csv"
stint_df.to_csv(f"{RESULTS_DIR}/{stint_file}", index=False)
generated_files.append(stint_file)


# ============================================================
# SUMMARY OUTPUTS
# ============================================================
summary = pd.DataFrame({
    "CV_shape_MAE": [mean_mae],
    "CV_shape_RMSE": [mean_rmse],
    "OOF_shape_MAE": [oof_mae],
    "OOF_shape_RMSE": [oof_rmse],
    **{f"{base}_shape_MAE": [np.mean(np.abs(oof[base]))] for base in ["flat", "linear"]},
    **{f"{base}_shape_RMSE": [np.sqrt(np.mean(oof[base] ** 2))] for base in ["flat", "linear"]},
    **{f"paired_t_p_vs_{base}": [tests[base][1]] for base in ["flat", "linear"]},
    **{f"wilcoxon_p_vs_{base}": [tests[base][2]] for base in ["flat", "linear"]},
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

