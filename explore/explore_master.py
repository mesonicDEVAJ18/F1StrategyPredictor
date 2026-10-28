# Alternate master for choosing the fall-off model (#27, #31). Runs the data stages of the pipeline,
# then, instead of train_model.py's one model, compares the candidates in the experiment block below
# with nested cross-validation grouped by stint, scored on the shape of the fall-off curve within each
# held-out stint, against the flat and linear baselines, and draws the best configuration of each
# candidate's curve the way strategy_simulator.py does.
#
#   cd explore
#   python explore_master.py                  # data stages + experiment
#   python explore_master.py --skip-pipeline  # reuse data/processed/
#   python explore_master.py --skip-pipeline --quick   # smoke run: 3 candidates, 2 sets, small search
#
# Writes only to explore/results/explore_master/. It writes no hand-off file and doesn't touch the
# model bundle: the winner is copied into train_model.py (and .env) by hand.
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # .env paths (../data, ./results) are relative to explore/
SCRIPTS_DIR = os.path.abspath(os.path.join("..", "scripts"))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from env_config import env_str, env_int, env_float  # noqa: E402
from feature_config import (  # noqa: E402
    COMPOUNDS, LAP_TIMING_COLUMNS, MODEL_FEATURE_COLUMNS, STINT_GROUP_COLUMNS, fall_off_curve, fall_off_rows,
    fit_compound_gap, linear_fall_off, stint_shape_errors,
)
from race_weekend import selected_weekend  # noqa: E402
import explore_lib as lib  # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_STATE = env_int("RANDOM_STATE")

# ######################################################################
# EXPERIMENT BLOCK: edit here
# ######################################################################
# --- Cross-validation -------------------------------------------------
OUTER_FOLDS = env_int("CV_FOLDS")   # outer GroupKFold by stint, as validate_strategy.py
OUTER_REPEATS = 2                   # outer CV repeated with this many shuffles (seeds RANDOM_STATE + r)
INNER_FOLDS = 4                     # inner GroupKFold for the hyperparameter search
N_ITER = 20                        # grids larger than this get a randomized search of N_ITER draws

# --- Curve check ------------------------------------------------------
ROLLOUT_SLOPE_LAPS = 20             # fall-off slope fitted over tyre ages 2..this
PLAUSIBLE_MAX_STEP_S = 1.0          # largest change from one tyre age to the next for a curve to count as stable
MIN_MONOTONE_SHARE = 0.95           # share of steps over the whole race length that mustn't get faster (an RBF
                                    # kernel, e.g. SVR, drifts back to its mean past the oldest tyre in training)
CURVE_AGES = [5, 10, 15]            # tyre ages at which each fold's curve is saved (its spread = the curve's uncertainty)

# --- Feature sets (tyre and conditions only; never the lap's own timing, #31) ---
TYRE = ["compound_HARD", "compound_MEDIUM", "compound_SOFT", "tyre_age"]
WEATHER = ["air_temperature", "humidity", "pressure", "track_temperature", "wind_direction", "wind_speed"]  # rainfall is constant

FEATURE_SETS = {
    "core": list(MODEL_FEATURE_COLUMNS),             # train_model.py's inputs
    "tyre_only": TYRE,
    "all_weather": TYRE + WEATHER,
    "plus_age_at_start": list(MODEL_FEATURE_COLUMNS) + ["tyre_age_at_start"],
    "plus_stint_lap": list(MODEL_FEATURE_COLUMNS) + ["laps_since_stint_start"],
}
for _name, _cols in FEATURE_SETS.items():
    assert not set(_cols) & set(LAP_TIMING_COLUMNS), f"{_name} uses the lap's own timing"

# --- Candidates ---------------------------------------------------------
# name: model factory, search grid (keys are the model's own parameters), scale inputs?
# Optional feature_sets limits a candidate to those (default: all of them)
# Tree models use n_jobs=1: the search already runs its fits in parallel
XGB_GRID = {"n_estimators": [100, 300, 600], "learning_rate": [0.02, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5, 7], "min_child_weight": [1, 3, 5, 10],
            "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.5, 0.8, 1.0],
            "reg_lambda": [0, 1, 5, 20], "reg_alpha": [0, 0.5, 2]}
HGB_GRID = {"learning_rate": [0.03, 0.1], "max_iter": [100, 300],
            "max_depth": [2, 3, 5], "min_samples_leaf": [5, 10, 20],
            "l2_regularization": [0.0, 1.0, 10.0]}
CANDIDATES = {
    # One shared tyre_age slope (no compound x age term): the simplest reference
    "Ridge": dict(model=lambda: Ridge(), scale=True,
                  grid={"alpha": list(np.logspace(-2, 3, 12))}),
    # Read only the compound and tyre_age, so every feature set gives the same model
    "PerCompoundFallOff": dict(model=lambda: lib.PerCompoundFallOff(), scale=False,
                               grid={"degree": [1, 2, 3], "min_rows": [10, 20]},
                               feature_sets=["tyre_only"]),
    "PerCompoundIsotonic": dict(model=lambda: lib.PerCompoundIsotonic(), scale=False,
                                grid={"min_rows": [10, 20]},
                                feature_sets=["tyre_only"]),
    "RandomForest": dict(model=lambda: RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
                         scale=False,
                         grid={"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 3, 5, 10],
                               "max_features": [0.3, 0.6, 1.0]}),
    "ExtraTrees": dict(model=lambda: ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
                       scale=False,
                       grid={"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 3, 5, 10],
                             "max_features": [0.3, 0.6, 1.0]}),
    "HistGradientBoosting": dict(model=lambda: HistGradientBoostingRegressor(early_stopping=False,
                                                                              random_state=RANDOM_STATE),
                                 scale=False, grid=HGB_GRID),
    # Fall-off never decreases with tyre age
    "HistGB_monotone": dict(model=lambda: HistGradientBoostingRegressor(early_stopping=False, random_state=RANDOM_STATE,
                                                                         monotonic_cst={"tyre_age": 1}),
                            scale=False, grid=HGB_GRID),
    # train_model.py's current model, untuned, as the reference
    "XGBoost_current": dict(model=lambda: XGBRegressor(
                                n_estimators=env_int("XGB_N_ESTIMATORS"), learning_rate=env_float("XGB_LEARNING_RATE"),
                                max_depth=env_int("XGB_MAX_DEPTH"), subsample=env_float("XGB_SUBSAMPLE"),
                                colsample_bytree=env_float("XGB_COLSAMPLE_BYTREE"), random_state=RANDOM_STATE,
                                objective="reg:squarederror", n_jobs=1),
                            scale=False, grid={}),
    "XGBoost": dict(model=lambda: XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror", n_jobs=1),
                    scale=False, grid=XGB_GRID),
    "XGBoost_monotone": dict(model=lambda: XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror",
                                                        monotone_constraints={"tyre_age": 1}, n_jobs=1),
                             scale=False, grid=XGB_GRID),
    "SVR": dict(model=lambda: SVR(kernel="rbf"), scale=True,
                grid={"C": [0.3, 1, 3, 10, 30, 100], "gamma": ["scale", 0.01, 0.03, 0.1],
                      "epsilon": [0.1, 0.3, 1.0]}),
    "KNeighbors": dict(model=lambda: KNeighborsRegressor(), scale=True,
                       grid={"n_neighbors": [3, 5, 8, 12, 20], "weights": ["uniform", "distance"]}),
    # Neural net: the target is standardised too
    "MLP": dict(model=lambda: TransformedTargetRegressor(
                    regressor=MLPRegressor(max_iter=3000, random_state=RANDOM_STATE), transformer=StandardScaler()),
                scale=True,
                grid={"regressor__hidden_layer_sizes": [(16,), (32,), (64, 32), (32, 16, 8)],
                      "regressor__alpha": [1e-3, 1e-2, 0.1, 1.0, 10.0],
                      "regressor__solver": ["adam", "lbfgs"],
                      "regressor__learning_rate_init": [1e-3, 1e-2]}),
}

# Restrict a run (None = everything above)
RUN_ONLY_CANDIDATES = None          # e.g. ["Ridge", "XGBoost"]
RUN_ONLY_FEATURE_SETS = None        # e.g. ["core", "tyre_only"]
# ######################################################################


# ======================================================
# SETUP RESULTS DIRECTORY
# ======================================================
SCRIPT_NAME = "explore_master"
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
    print(safe_msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()


# ======================================================
# PIPELINE STAGES (as master.py, up to and including eda.py)
# ======================================================
PIPELINE_STAGES = [
    "race_weekend.py",
    "process_laps.py",
    "process_stints.py",
    "tyre_sets.py",             # needs process_stints.py's hand-off
    "process_weather.py",
    "process_cardata.py",       # keeps only the laps that survive process_laps.py
    "feature_engineer.py",
    "eda.py",                   # descriptive only; nothing here reads it
]


def run_step(script_name):
    """Runs a pipeline script from scripts/ and stops on a failure, as master.py does."""
    log(f"--- [START] Running {script_name} ---")
    start_time = time.time()
    try:
        subprocess.run([sys.executable, script_name], check=True, cwd=SCRIPTS_DIR)
    except subprocess.CalledProcessError as e:
        log(f"!!! [ERROR] {script_name} failed with exit code {e.returncode}. Stopping.")
        sys.exit(1)
    except FileNotFoundError:
        log(f"!!! [ERROR] Could not find file: {script_name}. Check your directory.")
        sys.exit(1)
    log(f"--- [DONE] {script_name} finished in {time.time() - start_time:.2f}s ---\n")


def stint_abs_errors(errors, stint_ids):
    """Mean |shape error| per stint, over the rows and the repeats ([repeats, rows] -> Series)."""
    return pd.Series(np.nanmean(np.abs(errors), axis=0)).groupby(stint_ids).mean()


def one_sided_wilcoxon(a, b):
    try:
        return wilcoxon(a, b, alternative="less").pvalue if (a != b).any() else np.nan
    except ValueError:
        return np.nan


# ======================================================
# MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pipeline", action="store_true", help="reuse data/processed/")
    parser.add_argument("--quick", action="store_true", help="smoke run: few candidates and sets, small search")
    args = parser.parse_args()

    global OUTER_REPEATS, N_ITER, RUN_ONLY_CANDIDATES, RUN_ONLY_FEATURE_SETS
    if args.quick:
        OUTER_REPEATS, N_ITER = 1, 3
        RUN_ONLY_CANDIDATES = ["Ridge", "PerCompoundFallOff", "XGBoost"]
        RUN_ONLY_FEATURE_SETS = ["core", "tyre_only"]

    t0 = time.time()
    log("Starting explore master...\n")
    if args.skip_pipeline:
        log("Skipping the pipeline stages (--skip-pipeline).\n")
    else:
        for script in PIPELINE_STAGES:
            run_step(script)

    weekend = selected_weekend()
    race_laps = weekend["race_laps"]
    log(f"Race weekend: {weekend['year']} {weekend['name']}, {race_laps} laps")

    # ---------------- data ----------------
    dataset_path = os.path.join(env_str("DATA_DIR"), env_str("PROCESSED_DIR"), env_str("ENGINEERED_FILE"))
    df, groups = lib.load_fall_off_rows(dataset_path)
    all_laps = pd.read_csv(dataset_path).replace([np.inf, -np.inf], np.nan)   # stage 3 (compound gap) uses every lap
    fcp = df["fuel_corrected_pace"].to_numpy()
    log(f"Fall-off rows (stints with >= 2 clean laps): {len(df)} laps, {len(np.unique(groups))} stints "
        f"(of {len(all_laps)} laps)")

    feature_sets = {k: v for k, v in FEATURE_SETS.items()
                    if RUN_ONLY_FEATURE_SETS is None or k in RUN_ONLY_FEATURE_SETS}
    candidates = {k: v for k, v in CANDIDATES.items()
                  if RUN_ONLY_CANDIDATES is None or k in RUN_ONLY_CANDIDATES}
    for name, cols in feature_sets.items():
        log(f"Feature set {name}: {len(cols)} inputs ({', '.join(cols)})")
    log(f"Candidates: {list(candidates)}")
    log(f"Outer CV: {OUTER_FOLDS} folds x {OUTER_REPEATS} repeats by stint; inner {INNER_FOLDS} folds, "
        f"randomized search above {N_ITER} settings\n")

    # Outer splits and their stage-1 targets, shared by every candidate and baseline so they are
    # scored on the same stints
    splits = []
    for r in range(OUTER_REPEATS):
        kf = GroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE + r)
        for fold, (tr, te) in enumerate(kf.split(df, groups=groups), 1):
            shared = len(np.intersect1d(groups[tr], groups[te]))
            if shared:
                raise RuntimeError(f"Repeat {r} fold {fold} shares {shared} stints between train and test")
            y_tr, slopes = lib.fold_target(df.iloc[tr])
            splits.append((r, fold, tr, te, y_tr, slopes))

    fold_rows, curve_rows = [], []
    oof = {}   # (candidate, feature set) -> [repeats, rows] out-of-fold shape errors

    # ---------------- baselines ----------------
    for r, fold, tr, te, y_tr, slopes in splits:
        for name, pred in lib.baseline_predictions(df.iloc[te], slopes).items():
            oof.setdefault((name, "-"), np.full((OUTER_REPEATS, len(df)), np.nan))[r, te] = \
                stint_shape_errors(fcp[te], pred, groups[te])
            fold_rows.append({"candidate": name, "feature_set": "-", "repeat": r, "fold": fold, "test_rows": len(te),
                              **lib.shape_metrics(fcp[te], pred, groups[te]), "params": ""})
        for c in COMPOUNDS:
            curve_rows += [{"candidate": "baseline_linear", "feature_set": "-", "repeat": r, "fold": fold,
                            "compound": c, "tyre_age": a, "fall_off": slopes[c] * (a - 1)} for a in CURVE_AGES]

    # ---------------- candidates ----------------
    combos = [(c, fs) for c, spec in candidates.items()
              for fs in feature_sets if fs in spec.get("feature_sets", feature_sets)]
    log(f"Running {len(combos)} candidate x feature set combinations...")
    for i, (cname, fsname) in enumerate(combos, 1):
        spec = candidates[cname]
        cols = feature_sets[fsname]
        X = df[cols]
        est, prefix = lib.build_estimator(spec["model"](), spec["scale"])
        errs = np.full((OUTER_REPEATS, len(df)), np.nan)
        start = time.time()
        for r, fold, tr, te, y_tr, slopes in splits:
            fitted, params = lib.fit_tuned(est, spec["grid"], prefix, X.iloc[tr], y_tr, groups[tr],
                                           INNER_FOLDS, N_ITER, RANDOM_STATE + r)
            pred = fitted.predict(X.iloc[te])
            errs[r, te] = stint_shape_errors(fcp[te], pred, groups[te])
            fold_rows.append({"candidate": cname, "feature_set": fsname, "repeat": r, "fold": fold,
                              "test_rows": len(te), **lib.shape_metrics(fcp[te], pred, groups[te]),
                              "params": json.dumps(params, default=str)})
            curves = lib.fold_curves(fitted, cols, X.iloc[tr].median().to_dict(), CURVE_AGES)
            for c in COMPOUNDS:
                curve_rows += [{"candidate": cname, "feature_set": fsname, "repeat": r, "fold": fold,
                                "compound": c, "tyre_age": a, "fall_off": v} for a, v in zip(CURVE_AGES, curves[c])]
        oof[(cname, fsname)] = errs
        fm = [row["MAE"] for row in fold_rows[-len(splits):]]
        log(f" [{i}/{len(combos)}] {cname} / {fsname}: shape MAE {np.mean(fm):.4f} ± {np.std(fm):.4f} s "
            f"({time.time() - start:.1f}s)")

    folds_df = pd.DataFrame(fold_rows)
    folds_file = f"{SCRIPT_NAME}_cv_folds.csv"
    folds_df.to_csv(f"{RESULTS_DIR}/{folds_file}", index=False)
    generated_files.append(folds_file)

    fold_curves_df = pd.DataFrame(curve_rows)
    fold_curves_file = f"{SCRIPT_NAME}_fold_curves.csv"
    fold_curves_df.to_csv(f"{RESULTS_DIR}/{fold_curves_file}", index=False)
    generated_files.append(fold_curves_file)

    # ---------------- summary ----------------
    BASELINES = ["baseline_flat", "baseline_linear"]
    fold_mae = {b: folds_df[folds_df["candidate"] == b].set_index(["repeat", "fold"])["MAE"] for b in BASELINES}
    stint_ids = df[STINT_GROUP_COLUMNS].astype(str).agg("_".join, axis=1).to_numpy()
    base_stint = {b: stint_abs_errors(oof[(b, "-")], stint_ids) for b in BASELINES}
    # Curve uncertainty: the spread of each fold's fall-off at CURVE_AGES, the largest over the compounds
    curve_std = (fold_curves_df.groupby(["candidate", "feature_set", "compound", "tyre_age"])["fall_off"].std()
                 .groupby(["candidate", "feature_set", "tyre_age"]).max().unstack("tyre_age"))
    summary_rows = []
    for key, part in folds_df.groupby(["candidate", "feature_set"], sort=False):
        idx = list(zip(part["repeat"], part["fold"]))
        stint_mae = stint_abs_errors(oof[key], stint_ids)
        row = {
            "candidate": key[0], "feature_set": key[1],
            "n_inputs": len(feature_sets[key[1]]) if key[1] in feature_sets else 0,
            "MAE_mean": part["MAE"].mean(), "MAE_std": part["MAE"].std(ddof=1),
            "RMSE_mean": part["RMSE"].mean(), "RMSE_std": part["RMSE"].std(ddof=1),
            "R2_within_mean": part["R2_within"].mean(), "folds": len(part),
        }
        for b in BASELINES:
            short = b.replace("baseline_", "")
            gain = fold_mae[b].loc[idx].to_numpy() - part["MAE"].to_numpy()
            diff = stint_mae - base_stint[b]
            row.update({
                f"MAE_gain_vs_{short}": gain.mean(), f"folds_beating_{short}": int((gain > 0).sum()),
                f"stints_beating_{short}": int((diff < 0).sum()),
                f"wilcoxon_p_vs_{short}": one_sided_wilcoxon(stint_mae, base_stint[b]),
            })
        row["stints"] = len(stint_mae)
        for a in CURVE_AGES:
            row[f"curve_std_age{a}"] = curve_std.loc[key, a] if key in curve_std.index else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("MAE_mean").reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary_file = f"{SCRIPT_NAME}_summary.csv"
    summary.to_csv(f"{RESULTS_DIR}/{summary_file}", index=False)
    generated_files.append(summary_file)

    show = ["rank", "candidate", "feature_set", "n_inputs", "MAE_mean", "MAE_std", "RMSE_mean", "R2_within_mean",
            "MAE_gain_vs_flat", "MAE_gain_vs_linear", "folds_beating_linear", "stints_beating_linear",
            "wilcoxon_p_vs_linear", f"curve_std_age{CURVE_AGES[-1]}"]
    log("\nLeaderboard (outer-CV means over all folds and repeats; within-stint shape MAE/RMSE in s):")
    log(summary[show].head(30).round(4).to_string(index=False))
    log("\nBaselines:")
    log(summary[summary["candidate"].str.startswith("baseline")][show].round(4).to_string(index=False))

    log("\nBest shape MAE per candidate and feature set:")
    models_only = summary[~summary["candidate"].str.startswith("baseline")]
    pivot = models_only.pivot_table(index="candidate", columns="feature_set", values="MAE_mean", aggfunc="min")
    pivot = pivot.loc[pivot.min(axis=1).sort_values().index, [c for c in feature_sets if c in pivot.columns]]
    log(pivot.round(3).to_string())

    # ---------------- final fit + curve, best configuration of each candidate ----------------
    y_all, slopes_all = lib.fold_target(df)
    ages = np.arange(1, race_laps + 1)
    best_per_cand = models_only.drop_duplicates("candidate")   # summary is sorted by MAE
    best_params = {}
    curve_out, metric_rows, curves_by, gaps_by = [], [], {}, {}

    def record(cname, fsname, curves, gap, cv_row, params):
        rm = lib.curve_metrics(curves, ROLLOUT_SLOPE_LAPS)
        rm["stable"] = bool(rm["finite"] and rm["max_step"] <= PLAUSIBLE_MAX_STEP_S)
        rm["min_monotone_share"] = min(rm[f"{c}_monotone_share"] for c in COMPOUNDS)
        rm["monotone"] = bool(rm["min_monotone_share"] >= MIN_MONOTONE_SHARE)
        rm["order_lap5"] = lib.compound_order(curves, gap, 5)
        rm[f"order_lap{ROLLOUT_SLOPE_LAPS}"] = lib.compound_order(curves, gap, ROLLOUT_SLOPE_LAPS)
        curves_by[cname], gaps_by[cname] = curves, gap
        for c, fall in curves.items():
            curve_out.extend({"candidate": cname, "compound": c, "tyre_age": int(a), "fall_off": v,
                              "gap_plus_fall_off": gap[c] + v} for a, v in zip(ages, fall))
        metric_rows.append({"candidate": cname, "feature_set": fsname, "cv_MAE": cv_row["MAE_mean"],
                            "cv_MAE_std": cv_row["MAE_std"], "gap_MEDIUM": gap["MEDIUM"], "gap_HARD": gap["HARD"],
                            **rm})
        slopes = ", ".join(f"{c[0]} {rm[f'{c}_slope']:+.3f}" for c in COMPOUNDS)
        fall15 = ", ".join(f"{c[0]} {curves[c][14]:+.2f}" for c in COMPOUNDS)
        log(f"  {cname} ({fsname}): max step {rm['max_step']:.2f} s, monotone share {rm['min_monotone_share']:.2f}, "
            f"slope s/lap [{slopes}], fall-off at age 15 "
            f"[{fall15}], gap M {gap['MEDIUM']:+.2f} H {gap['HARD']:+.2f}, order lap 5 {rm['order_lap5']}, "
            f"lap {ROLLOUT_SLOPE_LAPS} {rm[f'order_lap{ROLLOUT_SLOPE_LAPS}']}, params {params}")

    log(f"\nCurves of the best configuration of each candidate (new set, tyre ages 1-{race_laps}; "
        f"compound gap from stage 3 on all {len(all_laps)} laps):")
    lin_curves = {c: slopes_all[c] * (ages - 1) for c in COMPOUNDS}
    lin_gap, _, _ = fit_compound_gap(all_laps, linear_fall_off(all_laps, slopes_all))
    record("baseline_linear", "-", lin_curves, lin_gap,
           summary[summary["candidate"] == "baseline_linear"].iloc[0], {})
    for _, row in best_per_cand.iterrows():
        cname, fsname = row["candidate"], row["feature_set"]
        spec, cols = candidates[cname], feature_sets[fsname]
        est, prefix = lib.build_estimator(spec["model"](), spec["scale"])
        fitted, params = lib.fit_tuned(est, spec["grid"], prefix, df[cols], y_all, groups,
                                       INNER_FOLDS, N_ITER, RANDOM_STATE)
        conditions = df[cols].median().to_dict()
        curves = {c: fall_off_curve(fitted, cols, c, ages, conditions) for c in COMPOUNDS}
        gap, _, _ = fit_compound_gap(all_laps, fall_off_rows(fitted, cols, all_laps, conditions))
        best_params[cname] = {"feature_set": fsname, "features": cols, "params": params,
                              "cv_MAE": row["MAE_mean"], "cv_MAE_std": row["MAE_std"], "compound_gap": gap}
        record(cname, fsname, curves, gap, row, params)

    params_file = f"{SCRIPT_NAME}_best_params.json"
    with open(f"{RESULTS_DIR}/{params_file}", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, default=str)
    generated_files.append(params_file)

    curves_file = f"{SCRIPT_NAME}_curves.csv"
    pd.DataFrame(curve_out).to_csv(f"{RESULTS_DIR}/{curves_file}", index=False)
    generated_files.append(curves_file)

    curve_metrics = pd.DataFrame(metric_rows)
    cm_file = f"{SCRIPT_NAME}_curve_metrics.csv"
    curve_metrics.to_csv(f"{RESULTS_DIR}/{cm_file}", index=False)
    generated_files.append(cm_file)

    # ---------------- recommendation ----------------
    flat = summary[summary["candidate"] == "baseline_flat"].iloc[0]
    lin = summary[summary["candidate"] == "baseline_linear"].iloc[0]
    log(f"\nFlat baseline (no fall-off): shape MAE {flat['MAE_mean']:.4f} ± {flat['MAE_std']:.4f} s; "
        f"linear baseline (stage-1 slope per compound): {lin['MAE_mean']:.4f} ± {lin['MAE_std']:.4f} s")
    # Counts only if it beats both baselines with a stable curve that rises on every compound and doesn't
    # get faster with age anywhere in the race
    models_cm = curve_metrics[curve_metrics["candidate"] != "baseline_linear"]
    ok = models_cm[models_cm["stable"] & models_cm["all_slopes_positive"] & models_cm["monotone"]
                   & (models_cm["cv_MAE"] < min(flat["MAE_mean"], lin["MAE_mean"]))].sort_values("cv_MAE")
    if ok.empty:
        log(f"No candidate beats both baselines with a stable curve (max step <= {PLAUSIBLE_MAX_STEP_S} s) "
            f"rising on every compound (monotone share >= {MIN_MONOTONE_SHARE}).")
    else:
        top = ok.iloc[0]
        log(f"Recommendation: {top['candidate']} ({top['feature_set']}), "
            f"shape MAE {top['cv_MAE']:.4f} ± {top['cv_MAE_std']:.4f} s: the lowest among candidates with a "
            f"stable curve rising on every compound (monotone share >= {MIN_MONOTONE_SHARE}) that beat both baselines")
        close = ok[ok["cv_MAE"] <= top["cv_MAE"] + top["cv_MAE_std"]]["candidate"].tolist()[1:]
        if close:
            log(f"  Within one fold std of it: {close}")
    best_any = best_per_cand.iloc[0]
    log(f"Lowest shape MAE regardless of the curve: {best_any['candidate']} ({best_any['feature_set']}), "
        f"{best_any['MAE_mean']:.4f} ± {best_any['MAE_std']:.4f} s")

    # ---------------- plots ----------------
    fig, ax = plt.subplots(figsize=(1.6 * pivot.shape[1] + 4, 0.45 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for (a, b), v in np.ndenumerate(pivot.to_numpy()):
        if np.isfinite(v):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, label="outer-CV shape MAE (s)")
    ax.set_title(f"Shape MAE by candidate and feature set (linear {lin['MAE_mean']:.2f} s, "
                 f"flat {flat['MAE_mean']:.2f} s)")
    fig.tight_layout()
    heat_file = f"{SCRIPT_NAME}_mae_by_feature_set.png"
    fig.savefig(f"{RESULTS_DIR}/{heat_file}", dpi=200)
    plt.close(fig)
    generated_files.append(heat_file)

    n = len(curves_by)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True, squeeze=False)
    colours = {"SOFT": "tab:red", "MEDIUM": "goldenrod", "HARD": "dimgray"}
    fsname_of = dict(zip(curve_metrics["candidate"], curve_metrics["feature_set"]))
    for ax, (cname, curves) in zip(axes.flat, curves_by.items()):
        band = fold_curves_df[(fold_curves_df["candidate"] == cname)
                              & (fold_curves_df["feature_set"] == fsname_of[cname])]
        for c, fall in curves.items():
            ax.plot(ages, fall, color=colours[c], label=c)
            spread = band[band["compound"] == c].groupby("tyre_age")["fall_off"].agg(["min", "max"])
            ax.vlines(spread.index, spread["min"], spread["max"], color=colours[c], alpha=0.6, linewidth=3)
        ax.set_title(cname, fontsize=9)
        ax.set_xlabel("tyre age (laps)")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes[0, 0].set_ylabel("fall-off (s)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Fall-off of each candidate's best configuration (new set; bars: fold min-max at ages "
                 f"{', '.join(map(str, CURVE_AGES))})")
    fig.tight_layout()
    curves_png = f"{SCRIPT_NAME}_curves.png"
    fig.savefig(f"{RESULTS_DIR}/{curves_png}", dpi=200)
    plt.close(fig)
    generated_files.append(curves_png)

    # ---------------- manifest ----------------
    with open(f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt", "w") as mf:
        for f in generated_files:
            mf.write(f + "\n")

    log(f"\nFinished in {time.time() - t0:.0f}s. FILES GENERATED:")
    for f in generated_files:
        log(f"  - {f}")
    log_file.close()


if __name__ == "__main__":
    main()
