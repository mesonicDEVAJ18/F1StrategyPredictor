# Alternate master for choosing the model (#27). Runs the data stages of the pipeline, then,
# instead of train_model.py's one model, compares the candidates in the experiment block below
# with nested cross-validation grouped by stint, against naive baselines, and rolls the best
# configuration of each candidate out like strategy_simulator.py does.
#
#   cd explore
#   python explore_master.py                  # data stages + experiment
#   python explore_master.py --skip-pipeline  # reuse data/processed/ and scripts/results/eda/
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
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from env_config import env_str, env_int, env_float  # noqa: E402
from feature_config import (  # noqa: E402
    CARDATA_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS, SESSION_COLUMNS, TARGET_GROUP_COLUMNS,
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

# --- Target modes -----------------------------------------------------
# level: predict next-lap fcp (as train_model.py); delta: predict next-lap fcp - lap t time
TARGET_MODES = ["level", "delta"]

# --- Rollout check ----------------------------------------------------
ROLLOUT_SLOPE_LAPS = 20             # degradation slope fitted over stint laps 2..this
PLAUSIBLE_MAX_STEP_S = 1.0          # largest lap-to-lap step a rollout may have to count as stable

# --- Feature sets (subsets of MODEL_FEATURE_COLUMNS, from diagnostic.md #25 / #27) ---
CONSTANT = ["segments_sector_1", "segments_sector_2", "rainfall"]       # constant in eda.py
TYRE = ["compound_HARD", "compound_MEDIUM", "compound_SOFT",
        "tyre_age_at_start", "laps_since_stint_start", "tyre_age"]
SECTORS = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
# What the simulator varies during a rollout, plus the session and track temperature
ROLLOUT_CORE = SECTORS + TYRE + ["stint_avg_pace", *SESSION_COLUMNS, "track_temperature"]
# Never dropped when a collinear group is cut down (the simulator sets them)
PROTECTED = set(SECTORS + TYRE + SESSION_COLUMNS + ["stint_avg_pace"])


def without(cols, drop):
    return [c for c in cols if c not in set(drop)]


def only(keep):
    return [c for c in MODEL_FEATURE_COLUMNS if c in set(keep)]  # keep MODEL_FEATURE_COLUMNS order


FEATURE_SETS = {
    "all": list(MODEL_FEATURE_COLUMNS),
    "no_constant": without(MODEL_FEATURE_COLUMNS, CONSTANT),
    "no_cardata": without(MODEL_FEATURE_COLUMNS, CONSTANT + CARDATA_FEATURE_COLUMNS),
    "no_collinear": None,   # no_constant with each eda.py collinear group cut to one member; built below
    "rollout_core": only(ROLLOUT_CORE),
    "rollout_core_speeds": only(ROLLOUT_CORE + ["i1_speed", "i2_speed", "st_speed"]),
}

# --- Candidates ---------------------------------------------------------
# name: model factory, search grid (keys are the model's own parameters), scale inputs?
# Optional feature_sets / target_modes limit a candidate to those (default: all of them)
# Tree models use n_jobs=1: the search already runs its fits in parallel
CANDIDATES = {
    "LinearRegression": dict(model=lambda: LinearRegression(), grid={}, scale=True),
    "Ridge": dict(model=lambda: Ridge(), scale=True,
                  grid={"alpha": list(np.logspace(-2, 3, 12))}),
    "Lasso": dict(model=lambda: Lasso(max_iter=50000), scale=True,
                  grid={"alpha": list(np.logspace(-3, 1, 12))}),
    "ElasticNet": dict(model=lambda: ElasticNet(max_iter=50000), scale=True,
                       grid={"alpha": list(np.logspace(-3, 1, 8)), "l1_ratio": [0.1, 0.5, 0.9]}),
    # Reads only the sector times, tyre_age and compound, and is linear in the lap t time,
    # so every feature set and target mode gives the same model
    "PerCompoundDegradation": dict(model=lambda: lib.PerCompoundDegradation(), scale=False,
                                   grid={"min_rows": [10, 20]},
                                   feature_sets=["rollout_core"], target_modes=["level"]),
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
                                 scale=False,
                                 grid={"learning_rate": [0.03, 0.1], "max_iter": [100, 300],
                                       "max_depth": [2, 3, 5], "min_samples_leaf": [5, 10, 20],
                                       "l2_regularization": [0.0, 1.0, 10.0]}),
    # train_model.py's current model, untuned, as the reference
    "XGBoost_current": dict(model=lambda: XGBRegressor(
                                n_estimators=env_int("XGB_N_ESTIMATORS"), learning_rate=env_float("XGB_LEARNING_RATE"),
                                max_depth=env_int("XGB_MAX_DEPTH"), subsample=env_float("XGB_SUBSAMPLE"),
                                colsample_bytree=env_float("XGB_COLSAMPLE_BYTREE"), random_state=RANDOM_STATE,
                                objective="reg:squarederror", n_jobs=1),
                            scale=False, grid={}),
    "XGBoost": dict(model=lambda: XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror", n_jobs=1),
                    scale=False,
                    grid={"n_estimators": [100, 300, 600], "learning_rate": [0.02, 0.05, 0.1],
                          "max_depth": [2, 3, 4, 5, 7], "min_child_weight": [1, 3, 5, 10],
                          "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.5, 0.8, 1.0],
                          "reg_lambda": [0, 1, 5, 20], "reg_alpha": [0, 0.5, 2]}),
    "SVR": dict(model=lambda: SVR(kernel="rbf"), scale=True,
                grid={"C": [0.3, 1, 3, 10, 30, 100], "gamma": ["scale", 0.01, 0.03, 0.1],
                      "epsilon": [0.1, 0.3, 1.0]}),
    "KNeighbors": dict(model=lambda: KNeighborsRegressor(), scale=True,
                       grid={"n_neighbors": [3, 5, 8, 12, 20], "weights": ["uniform", "distance"]}),
    # Neural net: the target is standardised too, so the net doesn't have to learn a ~97 s bias
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
RUN_ONLY_FEATURE_SETS = None        # e.g. ["all", "rollout_core"]
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
    "eda.py",                   # its collinear groups build the no_collinear feature set
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


def collinear_reduced(base):
    """base with each eda.py collinear group cut down: a group holding a PROTECTED column keeps only
    those, any other group keeps the member most correlated (|Pearson r|) with the target."""
    eda_dir = os.path.join(SCRIPTS_DIR, env_str("RESULTS_ROOT"), "eda")
    pairs = pd.read_csv(os.path.join(eda_dir, "eda_collinear_pairs.csv"))
    corr = pd.read_csv(os.path.join(eda_dir, "eda_target_correlations.csv")).set_index("feature")["pearson_r"].abs()
    drop = []
    for g, part in pairs.groupby("group"):
        members = sorted(set(part["feature_a"]) | set(part["feature_b"]))
        protected = [m for m in members if m in PROTECTED]
        keep = protected or [max(members, key=lambda m: corr.get(m, 0.0))]
        drop += [m for m in members if m not in keep]
        log(f"  collinear group {g}: keep {keep}, drop {[m for m in members if m not in keep]}")
    return without(base, drop)


# ======================================================
# MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pipeline", action="store_true", help="reuse data/processed/ and the eda results")
    parser.add_argument("--quick", action="store_true", help="smoke run: few candidates and sets, small search")
    args = parser.parse_args()

    global OUTER_REPEATS, N_ITER, RUN_ONLY_CANDIDATES, RUN_ONLY_FEATURE_SETS
    if args.quick:
        OUTER_REPEATS, N_ITER = 1, 3
        RUN_ONLY_CANDIDATES = ["Ridge", "PerCompoundDegradation", "XGBoost"]
        RUN_ONLY_FEATURE_SETS = ["all", "rollout_core"]

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
    df, groups = lib.load_target_rows(dataset_path)
    y = df["target_next_fcp"].to_numpy()
    log(f"Target rows: {len(df)} ({len(np.unique(groups))} stints)")

    FEATURE_SETS["no_collinear"] = collinear_reduced(FEATURE_SETS["no_constant"])
    feature_sets = {k: v for k, v in FEATURE_SETS.items()
                    if RUN_ONLY_FEATURE_SETS is None or k in RUN_ONLY_FEATURE_SETS}
    candidates = {k: v for k, v in CANDIDATES.items()
                  if RUN_ONLY_CANDIDATES is None or k in RUN_ONLY_CANDIDATES}
    for name, cols in feature_sets.items():
        log(f"Feature set {name}: {len(cols)} inputs")
    log(f"Candidates: {list(candidates)}; target modes: {TARGET_MODES}")
    log(f"Outer CV: {OUTER_FOLDS} folds x {OUTER_REPEATS} repeats by stint; inner {INNER_FOLDS} folds, "
        f"randomized search above {N_ITER} settings\n")

    # Outer splits, shared by every candidate and baseline so they are scored on the same stints
    splits = []
    for r in range(OUTER_REPEATS):
        kf = GroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE + r)
        for fold, (tr, te) in enumerate(kf.split(df, y, groups), 1):
            shared = len(np.intersect1d(groups[tr], groups[te]))
            if shared:
                raise RuntimeError(f"Repeat {r} fold {fold} shares {shared} stints between train and test")
            splits.append((r, fold, tr, te))

    fold_rows = []
    oof = {}   # (candidate, feature set, mode) -> [repeats, rows] out-of-fold predictions

    # ---------------- baselines ----------------
    for r, fold, tr, te in splits:
        base = lib.baseline_predictions(df, y[tr].mean())
        for name, pred in base.items():
            oof.setdefault((name, "-", "-"), np.full((OUTER_REPEATS, len(y)), np.nan))[r, te] = pred[te]
            fold_rows.append({"candidate": name, "feature_set": "-", "target_mode": "-", "repeat": r,
                              "fold": fold, "test_rows": len(te), **lib.metrics(y[te], pred[te]), "params": ""})
    persistence = df["fuel_corrected_pace"].to_numpy()

    # ---------------- candidates ----------------
    combos = [(c, fs, m) for c, spec in candidates.items()
              for fs in feature_sets if fs in spec.get("feature_sets", feature_sets)
              for m in TARGET_MODES if m in spec.get("target_modes", TARGET_MODES)]
    log(f"Running {len(combos)} candidate x feature set x target mode combinations...")
    for i, (cname, fsname, mode) in enumerate(combos, 1):
        spec = candidates[cname]
        cols = feature_sets[fsname]
        X = df[cols]
        est, prefix = lib.build_estimator(spec["model"](), spec["scale"], mode)
        preds = np.full((OUTER_REPEATS, len(y)), np.nan)
        start = time.time()
        for r, fold, tr, te in splits:
            fitted, params = lib.fit_tuned(est, spec["grid"], prefix, X.iloc[tr], y[tr], groups[tr],
                                           INNER_FOLDS, N_ITER, RANDOM_STATE + r)
            pred = fitted.predict(X.iloc[te])
            preds[r, te] = pred
            fold_rows.append({"candidate": cname, "feature_set": fsname, "target_mode": mode, "repeat": r,
                              "fold": fold, "test_rows": len(te), **lib.metrics(y[te], pred),
                              "params": json.dumps(params, default=str)})
        oof[(cname, fsname, mode)] = preds
        fm = [row["MAE"] for row in fold_rows[-len(splits):]]
        log(f" [{i}/{len(combos)}] {cname} / {fsname} / {mode}: MAE {np.mean(fm):.4f} ± {np.std(fm):.4f} s "
            f"({time.time() - start:.1f}s)")

    folds_df = pd.DataFrame(fold_rows)
    folds_file = f"{SCRIPT_NAME}_cv_folds.csv"
    folds_df.to_csv(f"{RESULTS_DIR}/{folds_file}", index=False)
    generated_files.append(folds_file)

    # ---------------- summary ----------------
    fold_mae = {b: folds_df[folds_df["candidate"] == b].set_index(["repeat", "fold"])["MAE"]
                for b in ["baseline_persistence", "baseline_train_mean"]}
    stint_ids = df[TARGET_GROUP_COLUMNS].astype(str).agg("_".join, axis=1)
    base_abs = pd.Series(np.abs(y - persistence)).groupby(stint_ids.values).mean()
    summary_rows = []
    for key, part in folds_df.groupby(["candidate", "feature_set", "target_mode"], sort=False):
        idx = list(zip(part["repeat"], part["fold"]))
        gain = fold_mae["baseline_persistence"].loc[idx].to_numpy() - part["MAE"].to_numpy()
        gain_mean = fold_mae["baseline_train_mean"].loc[idx].to_numpy() - part["MAE"].to_numpy()
        # Per stint, mean absolute error over the rows and repeats, against the persistence baseline
        abs_err = np.nanmean(np.abs(oof[key] - y), axis=0)
        stint_mae = pd.Series(abs_err).groupby(stint_ids.values).mean()
        diff = stint_mae - base_abs
        try:
            w_p = wilcoxon(stint_mae, base_abs, alternative="less").pvalue if (diff != 0).any() else np.nan
        except ValueError:
            w_p = np.nan
        summary_rows.append({
            "candidate": key[0], "feature_set": key[1], "target_mode": key[2],
            "n_inputs": len(feature_sets[key[1]]) if key[1] in feature_sets else 0,
            "MAE_mean": part["MAE"].mean(), "MAE_std": part["MAE"].std(ddof=1),
            "RMSE_mean": part["RMSE"].mean(), "RMSE_std": part["RMSE"].std(ddof=1),
            "R2_mean": part["R2"].mean(), "R2_std": part["R2"].std(ddof=1),
            "MAE_gain_vs_persistence": gain.mean(),
            "folds_beating_persistence": int((gain > 0).sum()),
            "MAE_gain_vs_train_mean": gain_mean.mean(),
            "folds_beating_train_mean": int((gain_mean > 0).sum()), "folds": len(part),
            "stints_beating_persistence": int((diff < 0).sum()), "stints": len(diff),
            "wilcoxon_p_vs_persistence": w_p,
        })
    summary = pd.DataFrame(summary_rows).sort_values("MAE_mean").reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary_file = f"{SCRIPT_NAME}_summary.csv"
    summary.to_csv(f"{RESULTS_DIR}/{summary_file}", index=False)
    generated_files.append(summary_file)

    show = ["rank", "candidate", "feature_set", "target_mode", "n_inputs", "MAE_mean", "MAE_std",
            "RMSE_mean", "R2_mean", "MAE_gain_vs_persistence", "MAE_gain_vs_train_mean", "folds_beating_train_mean",
            "stints_beating_persistence", "wilcoxon_p_vs_persistence"]
    log("\nLeaderboard (outer-CV means over all folds and repeats; MAE/RMSE in s):")
    log(summary[show].head(30).round(4).to_string(index=False))
    log("\nBaselines:")
    log(summary[summary["candidate"].str.startswith("baseline")][show].round(4).to_string(index=False))

    log("\nBest MAE per candidate and feature set (min over target modes):")
    models_only = summary[~summary["candidate"].str.startswith("baseline")]
    pivot = models_only.pivot_table(index="candidate", columns="feature_set", values="MAE_mean", aggfunc="min")
    pivot = pivot.loc[pivot.min(axis=1).sort_values().index, [c for c in feature_sets if c in pivot.columns]]
    log(pivot.round(3).to_string())
    log("\nMean MAE by target mode (all candidates and sets):")
    log(models_only.groupby("target_mode")["MAE_mean"].describe()[["mean", "min", "50%"]].round(4).to_string())

    # ---------------- final fit + rollout, best configuration of each candidate ----------------
    medians = df[MODEL_FEATURE_COLUMNS].median(numeric_only=True).to_dict()
    session_weights = df[SESSION_COLUMNS].mean().to_dict()
    best_per_cand = models_only.drop_duplicates("candidate")   # summary is sorted by MAE
    best_params = {}
    rollout_rows, metric_rows, curves_by = [], [], {}
    log(f"\nRolling out the best configuration of each candidate ({race_laps}-lap stint on a new set):")
    for _, row in best_per_cand.iterrows():
        cname, fsname, mode = row["candidate"], row["feature_set"], row["target_mode"]
        spec, cols = candidates[cname], feature_sets[fsname]
        est, prefix = lib.build_estimator(spec["model"](), spec["scale"], mode)
        fitted, params = lib.fit_tuned(est, spec["grid"], prefix, df[cols], y, groups,
                                       INNER_FOLDS, N_ITER, RANDOM_STATE)
        best_params[cname] = {"feature_set": fsname, "target_mode": mode, "features": cols,
                              "params": params, "cv_MAE": row["MAE_mean"], "cv_MAE_std": row["MAE_std"]}
        curves = {c: lib.rollout_stint(fitted, cols, medians, session_weights, c, 0, race_laps) for c in lib.COMPOUNDS}
        curves_by[cname] = curves
        for c, fcp in curves.items():
            rollout_rows += [{"candidate": cname, "compound": c, "stint_lap": n + 1, "fuel_corrected_pace": v}
                             for n, v in enumerate(fcp)]
        rm = lib.rollout_metrics(curves, ROLLOUT_SLOPE_LAPS)
        rm["stable"] = bool(rm["finite"] and rm["max_step"] <= PLAUSIBLE_MAX_STEP_S)
        metric_rows.append({"candidate": cname, "feature_set": fsname, "target_mode": mode,
                            "cv_MAE": row["MAE_mean"], "cv_MAE_std": row["MAE_std"], **rm})
        slopes = ", ".join(f"{c[0]} {rm[f'{c}_slope']:+.3f}" for c in lib.COMPOUNDS)
        log(f"  {cname} ({fsname}, {mode}): max step {rm['max_step']:.2f} s, slope s/lap [{slopes}], "
            f"order lap 5 {rm.get('order_lap5', '-')}, lap {ROLLOUT_SLOPE_LAPS} "
            f"{rm.get(f'order_lap{ROLLOUT_SLOPE_LAPS}', '-')}, params {params}")

    params_file = f"{SCRIPT_NAME}_best_params.json"
    with open(f"{RESULTS_DIR}/{params_file}", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, default=str)
    generated_files.append(params_file)

    rollouts_file = f"{SCRIPT_NAME}_rollouts.csv"
    pd.DataFrame(rollout_rows).to_csv(f"{RESULTS_DIR}/{rollouts_file}", index=False)
    generated_files.append(rollouts_file)

    rollout_metrics = pd.DataFrame(metric_rows)
    rm_file = f"{SCRIPT_NAME}_rollout_metrics.csv"
    rollout_metrics.to_csv(f"{RESULTS_DIR}/{rm_file}", index=False)
    generated_files.append(rm_file)

    # ---------------- recommendation ----------------
    pers = summary[summary["candidate"] == "baseline_persistence"].iloc[0]
    tmean = summary[summary["candidate"] == "baseline_train_mean"].iloc[0]
    log(f"\nPersistence baseline: MAE {pers['MAE_mean']:.4f} ± {pers['MAE_std']:.4f} s; "
        f"training-mean baseline: {tmean['MAE_mean']:.4f} ± {tmean['MAE_std']:.4f} s")
    # Counts only if it beats both naive forecasts and rolls out stably with degradation on every compound
    ok = rollout_metrics[rollout_metrics["stable"] & rollout_metrics["all_slopes_positive"]
                         & (rollout_metrics["cv_MAE"] < min(pers["MAE_mean"], tmean["MAE_mean"]))]
    if ok.empty:
        log(f"No candidate beats both baselines with a stable rollout (max step <= {PLAUSIBLE_MAX_STEP_S} s) "
            f"and degradation on every compound.")
    else:
        top = ok.iloc[0]
        log(f"Recommendation: {top['candidate']} ({top['feature_set']}, {top['target_mode']}), "
            f"CV MAE {top['cv_MAE']:.4f} ± {top['cv_MAE_std']:.4f} s: the lowest CV MAE among candidates with a "
            f"stable rollout and degradation on every compound that beat both baselines")
        close = ok[ok["cv_MAE"] <= top["cv_MAE"] + top["cv_MAE_std"]]["candidate"].tolist()[1:]
        if close:
            log(f"  Within one fold std of it: {close}")
    best_any = best_per_cand.iloc[0]
    log(f"Lowest CV MAE regardless of rollout: {best_any['candidate']} ({best_any['feature_set']}, "
        f"{best_any['target_mode']}), {best_any['MAE_mean']:.4f} ± {best_any['MAE_std']:.4f} s")

    # ---------------- plots ----------------
    fig, ax = plt.subplots(figsize=(1.6 * pivot.shape[1] + 4, 0.45 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for (a, b), v in np.ndenumerate(pivot.to_numpy()):
        if np.isfinite(v):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, label="outer-CV MAE (s)")
    ax.set_title(f"MAE by candidate and feature set (persistence {pers['MAE_mean']:.2f} s)")
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
    for ax, (cname, curves) in zip(axes.flat, curves_by.items()):
        for c, fcp in curves.items():
            ax.plot(np.arange(1, len(fcp) + 1), fcp, color=colours[c], label=c)
        ax.set_title(cname, fontsize=9)
        ax.set_xlabel("stint lap")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes[0, 0].set_ylabel("predicted fcp (s)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Rollouts of each candidate's best configuration (new set)")
    fig.tight_layout()
    roll_png = f"{SCRIPT_NAME}_rollouts.png"
    fig.savefig(f"{RESULTS_DIR}/{roll_png}", dpi=200)
    plt.close(fig)
    generated_files.append(roll_png)

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
