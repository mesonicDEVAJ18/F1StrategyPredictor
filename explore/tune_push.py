# Tunes process_laps.py's push-lap filter (#30): PUSH_LAP_MAX_BEST_RATIO (lap / driver's session best)
# and PUSH_LAP_MAX_STINT_RATIO (lap / driver's stint best), over the weekends in the experiment block below.
# A lap is kept only when it passes both.
#
#   cd explore
#   python tune_push.py            # every weekend in TUNE_WEEKENDS (downloads what data/cache/ lacks)
#   python tune_push.py --quick    # smoke run: the two weekends cached first, small grid, 1 repeat
#
# Two methods, both on the pipeline's own outputs (the stages run from scripts/ with the settings
# passed as environment variables, which override .env):
#   A. label sweep: process_laps.py runs with the filter off and saves every lap's two ratios
#      (process_laps_push_ratios_*.csv); process_cardata.py gives each lap's full-throttle share. A lap whose
#      share is >= LABEL_PUSH_MIN x its driver's level in the session (the 90th percentile) is a push lap,
#      <= LABEL_COOL_MAX a cool-down lap; the label doesn't depend on the ratios being tuned. The label pick is
#      the grid pair with the highest Youden J (push laps kept - cool-down laps kept, as shares).
#   B. CV sweep: the data stages rerun for each (r_session, r_stint); XGBoost (train_model.py's MODEL block)
#      is scored by grouped CV on the target rows present in every configuration of a weekend, with the
#      same stint folds for each configuration, so the scores are paired.
#
# While it runs it switches data/raw/ between weekends and rewrites data/processed/; at the end it
# switches back to the .env weekend and reruns the data stages at the .env values. It writes no hand-off
# file: the winner is copied into .env by hand.
import argparse
import hashlib
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
from scipy.stats import ttest_rel, wilcoxon  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from env_config import env_str, env_int, env_float, env_list  # noqa: E402
from feature_config import MODEL_FEATURE_COLUMNS, TARGET_GROUP_COLUMNS  # noqa: E402
import explore_lib as lib  # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_STATE = env_int("RANDOM_STATE")
CV_FOLDS = env_int("CV_FOLDS")

# ######################################################################
# EXPERIMENT BLOCK: edit here
# ######################################################################
# Non-sprint weekends in race_weekend.WEEKENDS, a mix of track types over both years (as tune_iqr.py)
TUNE_WEEKENDS = [
    (2025, "bahrain"), (2024, "saudi_arabia"),
    (2025, "australia"), (2025, "japan"), (2025, "spain"), (2025, "austria"),
    (2025, "britain"), (2025, "netherlands"), (2025, "italy"), (2025, "singapore"),
    (2024, "hungary"), (2024, "spain"),
]
OFF = 100.0                                              # a ratio no lap reaches: that test is off
R_SESSION = [1.10, 1.12, 1.15, 1.18, 1.20, 1.25, 1.30]   # label sweep
R_STINT = [1.04, 1.06, 1.08, 1.10, 1.12, OFF]
R_SESSION_CV = [1.10, 1.12, 1.15, 1.20, 1.25, 1.30]      # CV sweep; the label pick is added
R_STINT_CV = [1.06, 1.08, 1.12, OFF]
CURRENT = (1.15, OFF)                                    # the session test alone, before #30
LABEL_PUSH_MIN = 0.8                                     # full-throttle share / driver's level: push lap
LABEL_COOL_MAX = 0.6                                     # ... and cool-down lap; in between is ambiguous
LABEL_LEVEL_QUANTILE = 0.9                               # driver's level = this quantile in the session
CV_REPEATS = 2                                           # stint folds reshuffled with seeds RANDOM_STATE + r
ALPHA = 0.05                                             # paired t-test and Wilcoxon (one-sided), both must pass
# ######################################################################


# ======================================================
# SETUP RESULTS DIRECTORY
# ======================================================
SCRIPT_NAME = "tune_push"
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


def save_csv(df, name):
    df.to_csv(f"{RESULTS_DIR}/{name}", index=False)
    generated_files.append(name)


def fmt(cfg):
    return f"({cfg[0]:g}, {'off' if cfg[1] >= OFF else f'{cfg[1]:g}'})"


# ======================================================
# PIPELINE STAGES
# ======================================================
STAGE_RESULTS = os.path.join(SCRIPTS_DIR, env_str("RESULTS_ROOT"))
PROCESSED = os.path.join(env_str("DATA_DIR"), env_str("PROCESSED_DIR"))
LAPS_CLEAN_PATH = os.path.join(PROCESSED, env_str("LAPS_CLEAN_FILE"))
CARDATA_PATH = os.path.join(PROCESSED, env_str("CARDATA_CLEAN_FILE"))
DATASET_PATH = os.path.join(PROCESSED, env_str("ENGINEERED_FILE"))
LAPS_FILES = env_list("LAPS_FILES")
LAP_KEYS = ["session_key", "driver_number", "lap_number"]


def run_stage(script_name, overrides=None):
    """Runs a pipeline script from scripts/ with `overrides` as environment variables (they win over
    .env). Its output goes to the stage's own log, not here. Returns False when it fails."""
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (overrides or {}).items()})
    result = subprocess.run([sys.executable, script_name], cwd=SCRIPTS_DIR, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        log(f"!!! {script_name} {overrides or ''} failed with exit code {result.returncode}:")
        log("    " + result.stderr.strip().replace("\n", "\n    ")[-2000:])
        return False
    return True


def weekend_env(year, name):
    return {"RACE_YEAR": year, "RACE_WEEKEND": name}


def push_env(r_session, r_stint):
    return {"PUSH_LAP_MAX_BEST_RATIO": r_session, "PUSH_LAP_MAX_STINT_RATIO": r_stint}


def switch_weekend(year, name):
    """Puts the weekend in data/raw/ (downloading what the cache lacks) and builds the stints and
    weather hand-offs, which don't depend on the push-lap filter."""
    t0 = time.time()
    for script in ["race_weekend.py", "process_stints.py", "process_weather.py"]:
        if not run_stage(script, weekend_env(year, name)):
            return False
    log(f"  {year} {name}: in data/raw/, stints and weather processed ({time.time() - t0:.0f}s)")
    return True


# ======================================================
# A. LABEL SWEEP
# ======================================================
def labelled_laps(year, name):
    """Every lap reaching the push filter with its two ratios and a telemetry label. process_laps runs
    with the filter off and process_cardata with its z-score filter off, so cool-down laps keep their
    telemetry."""
    env = {**weekend_env(year, name), **push_env(OFF, OFF), "CARDATA_ZSCORE_THRESHOLD": 1e9}
    if not (run_stage("process_laps.py", env) and run_stage("process_cardata.py", env)):
        return None
    ratios = pd.concat(
        [pd.read_csv(os.path.join(STAGE_RESULTS, "process_laps", f"process_laps_push_ratios_{f.replace('.csv', '')}.csv"))
         .assign(session=f"FP{f.replace('.csv', '')[-1]}") for f in LAPS_FILES], ignore_index=True)
    car = pd.read_csv(CARDATA_PATH, usecols=LAP_KEYS + ["full_throttle_pct"])
    df = ratios.merge(car, on=LAP_KEYS, how="left")
    level = df.groupby(["session", "driver_number"])["full_throttle_pct"].transform(
        lambda x: x.quantile(LABEL_LEVEL_QUANTILE))
    df["throttle_rel"] = df["full_throttle_pct"] / level.where(level > 0)
    df["label"] = np.select([df["throttle_rel"] >= LABEL_PUSH_MIN, df["throttle_rel"] <= LABEL_COOL_MAX],
                            ["push", "cool"], default="ambiguous")
    df.loc[df["throttle_rel"].isna(), "label"] = "no_telemetry"
    df["weekend"] = f"{year}_{name}"
    return df


def label_sweep(laps):
    """Push and cool-down laps kept per session for every grid pair."""
    rows = []
    for (weekend, session), g in laps.groupby(["weekend", "session"]):
        push, cool = g["label"] == "push", g["label"] == "cool"
        for rs in R_SESSION:
            for rt in R_STINT:
                kept = (g["session_ratio"] <= rs) & ~(g["stint_ratio"] > rt)
                rows.append({"weekend": weekend, "session": session, "r_session": rs, "r_stint": rt,
                             "push": int(push.sum()), "cool": int(cool.sum()),
                             "push_kept": int((kept & push).sum()), "cool_kept": int((kept & cool).sum()),
                             "ambiguous_kept": int((kept & (g["label"] == "ambiguous")).sum()),
                             "laps_kept": int(kept.sum())})
    return pd.DataFrame(rows)


def youden(df):
    df = df.copy()
    df["push_recall"] = df["push_kept"] / df["push"]
    df["cool_leak"] = df["cool_kept"] / df["cool"]
    df["youden_j"] = df["push_recall"] - df["cool_leak"]
    return df


# ======================================================
# B. CV SWEEP
# ======================================================
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def removed_counts():
    s = pd.read_csv(os.path.join(STAGE_RESULTS, "process_laps", "process_laps_removed_rows_summary.csv"))
    return {stage: int(s.loc[s["stage"] == stage, "rows_removed"].sum())
            for stage in ["iqr_residual_slow", "non_push", "non_push_stint"]}


def xgb():
    # As train_model.py's MODEL block
    return XGBRegressor(
        n_estimators=env_int("XGB_N_ESTIMATORS"),
        learning_rate=env_float("XGB_LEARNING_RATE"),
        max_depth=env_int("XGB_MAX_DEPTH"),
        subsample=env_float("XGB_SUBSAMPLE"),
        colsample_bytree=env_float("XGB_COLSAMPLE_BYTREE"),
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
    )


def stint_folds(stint_keys, seed):
    """The same stint -> fold map for every configuration (ngroup() ids change with the data)."""
    keys = sorted(stint_keys)
    order = np.random.default_rng(seed).permutation(len(keys))
    return {keys[i]: pos % CV_FOLDS for pos, i in enumerate(order)}


def oof_predictions(df, folds):
    """Out-of-fold XGBoost predictions over the stint folds; train medians fill inputs still missing
    after feature_engineer.py's stint fill (as train_model.py / validate_strategy.py do)."""
    fold = df["stint_key"].map(folds).to_numpy()
    pred = np.full(len(df), np.nan)
    for f in range(CV_FOLDS):
        train, test = fold != f, fold == f
        if not test.any() or not train.any():
            continue
        X_train = df.loc[train, MODEL_FEATURE_COLUMNS]
        medians = X_train.median(numeric_only=True)
        model = xgb().fit(X_train.fillna(medians), df.loc[train, "target_next_fcp"])
        pred[test] = model.predict(df.loc[test, MODEL_FEATURE_COLUMNS].fillna(medians))
    return pred


def cv_sweep(year, name, configs, repeats):
    """Reruns the data stages for each configuration and scores XGBoost on the rows common to all.
    Returns (one row per configuration, per-stint MAE on the common rows)."""
    weekend_label = f"{year}_{name}"
    datasets, config_hash, config_info = {}, {}, {}
    for r_session, r_stint in configs:
        env = {**weekend_env(year, name), **push_env(r_session, r_stint)}
        if not run_stage("process_laps.py", env):
            return None, None
        h = file_hash(LAPS_CLEAN_PATH)
        config_hash[(r_session, r_stint)] = h
        config_info[(r_session, r_stint)] = {"cleaned_laps": len(pd.read_csv(LAPS_CLEAN_PATH)), **removed_counts()}
        if h in datasets:
            continue
        for script in ["process_cardata.py", "feature_engineer.py"]:
            if not run_stage(script, env):
                return None, None
        df, _ = lib.load_target_rows(DATASET_PATH)
        df["stint_key"] = list(zip(*[df[c] for c in TARGET_GROUP_COLUMNS]))
        df["row_key"] = list(zip(df["session_key"], df["driver_number"], df["lap_number"]))
        datasets[h] = df
    log(f"  {len(configs)} configurations, {len(datasets)} distinct cleaned-lap sets")

    common = set.intersection(*[set(df["row_key"]) for df in datasets.values()])
    stints = set().union(*[set(df["stint_key"]) for df in datasets.values()])
    fold_maps = [stint_folds(stints, RANDOM_STATE + r) for r in range(repeats)]

    scores, per_stint = {}, {}
    for h, df in datasets.items():
        on_common = df["row_key"].isin(common).to_numpy()
        preds = [oof_predictions(df, folds) for folds in fold_maps]
        y = df["target_next_fcp"].to_numpy()
        maes = [np.mean(np.abs(p[on_common] - y[on_common])) for p in preds]
        rmses = [np.sqrt(np.mean((p[on_common] - y[on_common]) ** 2)) for p in preds]
        own = [np.nanmean(np.abs(p - y)) for p in preds]
        persistence = lib.baseline_predictions(df[on_common], 0.0)["baseline_persistence"]
        scores[h] = {"target_rows": len(df), "target_stints": df["stint_key"].nunique(),
                     "common_rows": int(on_common.sum()),
                     "mae_common": np.mean(maes), "mae_common_sd": np.std(maes),
                     "rmse_common": np.mean(rmses), "rmse_common_sd": np.std(rmses),
                     "mae_own_rows": np.mean(own),
                     "persistence_mae_common": np.mean(np.abs(persistence - y[on_common]))}
        err = pd.DataFrame({"stint_key": df.loc[on_common, "stint_key"].to_numpy(),
                            "abs_err": np.mean([np.abs(p[on_common] - y[on_common]) for p in preds], axis=0)})
        per_stint[h] = err.groupby("stint_key")["abs_err"].agg(["mean", "size"])

    rows, stint_rows = [], []
    for (r_session, r_stint), h in config_hash.items():
        rows.append({"weekend": weekend_label, "r_session": r_session, "r_stint": r_stint,
                     **config_info[(r_session, r_stint)], **scores[h], "dataset": h[:8]})
        for key, r in per_stint[h].iterrows():
            stint_rows.append({"weekend": weekend_label, "r_session": r_session, "r_stint": r_stint,
                               "stint": f"{weekend_label}:{key[0]}:{key[1]}", "rows": int(r["size"]), "mae": r["mean"]})
    s = scores[config_hash[configs[0]]]
    log(f"  {s['common_rows']} target rows common to every configuration ({len(stints)} stints); "
        f"MAE on them {min(x['mae_common'] for x in scores.values()):.3f}-{max(x['mae_common'] for x in scores.values()):.3f} s")
    return rows, stint_rows


def paired_test(per_stint, config, reference):
    """One-sided: is `config` better (lower per-stint MAE) than `reference`? Pooled over weekends."""
    wide = per_stint.pivot_table(index="stint", columns=["r_session", "r_stint"], values="mae")
    a, b = wide[config].to_numpy(), wide[reference].to_numpy()
    d = a - b
    if np.allclose(d, 0):
        return {"mean_diff": 0.0, "median_diff": 0.0, "better_stints": 0, "t_p": 1.0, "wilcoxon_p": 1.0}
    _, t_p = ttest_rel(a, b, alternative="less")
    _, w_p = wilcoxon(a, b, alternative="less")
    return {"mean_diff": d.mean(), "median_diff": np.median(d), "better_stints": int((d < 0).sum()),
            "t_p": t_p, "wilcoxon_p": w_p}


# ======================================================
# PLOTS
# ======================================================
def plot_label_sweep(pooled, pick):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (vary, fixed, fixed_val) in zip(axes, [("r_session", "r_stint", pick[1]), ("r_stint", "r_session", pick[0])]):
        part = pooled[pooled[fixed] == fixed_val].sort_values(vary)
        x = [f"{v:g}" if v < OFF else "off" for v in part[vary]]
        ax.plot(x, part["push_recall"], marker="o", label="push laps kept")
        ax.plot(x, part["cool_leak"], marker="o", label="cool-down laps kept")
        ax.plot(x, part["youden_j"], marker="o", ls="--", color="black", label="Youden J")
        ax.set_title(f"{vary} varied, {fixed} = {'off' if fixed_val >= OFF else f'{fixed_val:g}'}")
        ax.set_xlabel(vary)
    axes[0].set_ylabel("share of labelled laps (all sessions)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    name = f"{SCRIPT_NAME}_label_sweep.png"
    fig.savefig(f"{RESULTS_DIR}/{name}", dpi=130)
    plt.close(fig)
    generated_files.append(name)


def plot_cv_heatmap(pooled):
    grid = pooled.pivot(index="r_session", columns="r_stint", values="mean_stint_mae")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(grid.columns)), [f"{c:g}" if c < OFF else "off" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{i:g}" for i in grid.index])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid.iat[i, j]):
                ax.text(j, i, f"{grid.iat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xlabel("r_stint")
    ax.set_ylabel("r_session")
    ax.set_title("Pooled per-stint MAE (s), common rows")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    name = f"{SCRIPT_NAME}_cv_heatmap.png"
    fig.savefig(f"{RESULTS_DIR}/{name}", dpi=130)
    plt.close(fig)
    generated_files.append(name)


# ======================================================
# MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke run: two weekends, small grid, 1 repeat")
    args = parser.parse_args()

    global TUNE_WEEKENDS, R_SESSION_CV, R_STINT_CV, CV_REPEATS
    if args.quick:
        TUNE_WEEKENDS, R_SESSION_CV, R_STINT_CV, CV_REPEATS = TUNE_WEEKENDS[:2], [1.15, 1.20], [1.08, OFF], 1

    t0 = time.time()
    log(f"Tuning the push-lap filter on {len(TUNE_WEEKENDS)} weekends: "
        f"{', '.join(f'{y} {n}' for y, n in TUNE_WEEKENDS)}\n")

    # ---------------- A. label sweep ----------------
    log(f"=== A. Label sweep (full-throttle share / driver's q{LABEL_LEVEL_QUANTILE:g}: push >= {LABEL_PUSH_MIN}, "
        f"cool-down <= {LABEL_COOL_MAX}) ===")
    laps, usable = [], []
    for year, name in TUNE_WEEKENDS:
        log(f"\n{year} {name}")
        df = labelled_laps(year, name) if switch_weekend(year, name) else None
        if df is None:
            log("  skipped: a stage failed")
            continue
        counts = df["label"].value_counts()
        log(f"  {len(df)} laps: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in ["push", "cool", "ambiguous", "no_telemetry"]))
        laps.append(df)
        usable.append((year, name))

    laps = pd.concat(laps, ignore_index=True)
    save_csv(laps, f"{SCRIPT_NAME}_labelled_laps.csv")
    sweep = youden(label_sweep(laps))
    save_csv(sweep, f"{SCRIPT_NAME}_label_sweep.csv")
    pooled_label = youden(sweep.groupby(["r_session", "r_stint"])[["push", "cool", "push_kept", "cool_kept",
                                                                   "ambiguous_kept", "laps_kept"]].sum().reset_index())
    save_csv(pooled_label, f"{SCRIPT_NAME}_label_pooled.csv")
    best = pooled_label.sort_values(["youden_j", "r_session", "r_stint"], ascending=False).iloc[0]
    pick = (float(best["r_session"]), float(best["r_stint"]))
    cur = pooled_label[(pooled_label["r_session"] == CURRENT[0]) & (pooled_label["r_stint"] == CURRENT[1])].iloc[0]

    # Per session: the pair with the highest J, to show how much the pick varies
    per_session = sweep.sort_values(["youden_j", "r_session", "r_stint"], ascending=False).groupby(["weekend", "session"]).head(1)
    log(f"\nLabelled laps over all sessions: push {int(best['push'])}, cool-down {int(best['cool'])}")
    log(f"Label pick {fmt(pick)}: push kept {best['push_recall']:.3f}, cool-down kept {best['cool_leak']:.3f}, "
        f"J {best['youden_j']:.3f}, laps kept {int(best['laps_kept'])}")
    log(f"Current {fmt(CURRENT)}: push kept {cur['push_recall']:.3f}, cool-down kept {cur['cool_leak']:.3f}, "
        f"J {cur['youden_j']:.3f}, laps kept {int(cur['laps_kept'])}")
    log("Best pair per session: r_session " + ", ".join(f"{v:g}: {n}" for v, n in per_session["r_session"].value_counts().sort_index().items())
        + "; r_stint " + ", ".join(f"{'off' if v >= OFF else f'{v:g}'}: {n}" for v, n in per_session["r_stint"].value_counts().sort_index().items()))
    log("Pooled, stint test off (r_session: push / cool-down kept):")
    for r in pooled_label[pooled_label["r_stint"] == OFF].itertuples():
        log(f"  {r.r_session:g}: {r.push_recall:.3f} / {r.cool_leak:.3f}, J {r.youden_j:.3f}")
    log(f"Pooled, r_session {pick[0]:g} (r_stint: push / cool-down kept):")
    for r in pooled_label[pooled_label["r_session"] == pick[0]].itertuples():
        log(f"  {'off' if r.r_stint >= OFF else f'{r.r_stint:g}'}: {r.push_recall:.3f} / {r.cool_leak:.3f}, J {r.youden_j:.3f}")
    plot_label_sweep(pooled_label, pick)

    # ---------------- B. CV sweep ----------------
    r_sessions = sorted(set(R_SESSION_CV) | {pick[0], CURRENT[0]})
    r_stints = sorted(set(R_STINT_CV) | {pick[1], CURRENT[1]})
    configs = [(float(a), float(b)) for a in r_sessions for b in r_stints]
    log(f"\n=== B. CV sweep: r_session {r_sessions} x r_stint {r_stints}, {CV_FOLDS} stint folds x {CV_REPEATS} ===")
    cv_rows, stint_rows = [], []
    for year, name in usable:
        log(f"\n{year} {name}")
        if not switch_weekend(year, name):
            continue
        rows, srows = cv_sweep(year, name, configs, CV_REPEATS)
        if rows is None:
            log("  skipped: a stage failed")
            continue
        cv_rows += rows
        stint_rows += srows

    cv, per_stint = pd.DataFrame(cv_rows), pd.DataFrame(stint_rows)
    save_csv(cv, f"{SCRIPT_NAME}_cv_sweep.csv")
    save_csv(per_stint, f"{SCRIPT_NAME}_cv_per_stint.csv")

    # Pooled over weekends: mean of the per-stint MAEs, and the row-weighted MAE
    pooled = (per_stint.assign(abs_sum=per_stint["mae"] * per_stint["rows"])
              .groupby(["r_session", "r_stint"])
              .agg(stints=("stint", "size"), rows=("rows", "sum"), mean_stint_mae=("mae", "mean"), abs_sum=("abs_sum", "sum"))
              .reset_index())
    pooled["row_mae"] = pooled["abs_sum"] / pooled["rows"]
    pooled = pooled.drop(columns="abs_sum")
    laps_cv = cv.groupby(["r_session", "r_stint"])[["cleaned_laps", "target_rows", "target_stints", "iqr_residual_slow",
                                                   "non_push", "non_push_stint"]].sum().reset_index()
    pooled = pooled.merge(laps_cv, on=["r_session", "r_stint"])

    reference = pick
    tests = []
    for cfg in configs:
        vs_ref = paired_test(per_stint, cfg, reference) if cfg != reference else {}
        vs_cur = paired_test(per_stint, cfg, CURRENT) if cfg != CURRENT else {}
        tests.append({"r_session": cfg[0], "r_stint": cfg[1],
                      **{f"vs_reference_{k}": v for k, v in vs_ref.items()},
                      **{f"vs_current_{k}": v for k, v in vs_cur.items()}})
    pooled = pooled.merge(pd.DataFrame(tests), on=["r_session", "r_stint"], how="left")
    save_csv(pooled, f"{SCRIPT_NAME}_cv_pooled.csv")
    plot_cv_heatmap(pooled)

    log(f"\nPooled over {cv['weekend'].nunique()} weekends ({int(pooled['stints'].iloc[0])} stints, "
        f"{int(pooled['rows'].iloc[0])} common rows):")
    for r in pooled.sort_values(["r_session", "r_stint"]).itertuples():
        log(f"  {fmt((r.r_session, r.r_stint))}: stint MAE {r.mean_stint_mae:.4f} s, row MAE {r.row_mae:.4f} s, "
            f"cleaned laps {r.cleaned_laps}, target rows {r.target_rows} ({r.target_stints} stints), "
            f"removed slow-IQR/session/stint {r.iqr_residual_slow}/{r.non_push}/{r.non_push_stint}")

    # ---------------- winner ----------------
    # The label pick, unless another configuration beats it on both one-sided paired tests; among those,
    # the lowest pooled per-stint MAE. Caution (#30): the common rows are the strictest configuration's own
    # rows, so this favours strict filters, which train on exactly the laps they are scored on. Check what
    # a stricter winner drops (labelled push laps, long runs) before copying it into .env

    significant = pooled[(pooled["vs_reference_t_p"] < ALPHA) & (pooled["vs_reference_wilcoxon_p"] < ALPHA)]
    if significant.empty:
        winner = reference
        why = (f"no configuration beats the label pick {fmt(reference)} on both paired tests "
               f"at p < {ALPHA}, so the label pick is kept")
    else:
        top = significant.sort_values("mean_stint_mae").iloc[0]
        winner = (top["r_session"], top["r_stint"])
        why = (f"{fmt(winner)} beats the label pick {fmt(reference)} on both paired "
               f"tests (t p {top['vs_reference_t_p']:.4f}, Wilcoxon p {top['vs_reference_wilcoxon_p']:.4f})")
    w = pooled[(pooled["r_session"] == winner[0]) & (pooled["r_stint"] == winner[1])].iloc[0]
    c = pooled[(pooled["r_session"] == CURRENT[0]) & (pooled["r_stint"] == CURRENT[1])].iloc[0]

    lines = [
        f"Push-lap filter tuning over {cv['weekend'].nunique()} weekends ({', '.join(sorted(cv['weekend'].unique()))})",
        "",
        f"A. Label sweep: pick {fmt(pick)}, push laps kept {best['push_recall']:.3f}, cool-down laps kept "
        f"{best['cool_leak']:.3f} (J {best['youden_j']:.3f}); current {fmt(CURRENT)} {cur['push_recall']:.3f} / "
        f"{cur['cool_leak']:.3f} (J {cur['youden_j']:.3f}).",
        f"B. CV sweep: {why}.",
        f"   Winner {fmt(winner)}: pooled per-stint MAE {w['mean_stint_mae']:.4f} s, row MAE {w['row_mae']:.4f} s, "
        f"{int(w['cleaned_laps'])} cleaned laps, {int(w['target_rows'])} target rows.",
        f"   Current {fmt(CURRENT)}: pooled per-stint MAE {c['mean_stint_mae']:.4f} s, row MAE {c['row_mae']:.4f} s, "
        f"{int(c['cleaned_laps'])} cleaned laps, {int(c['target_rows'])} target rows.",
    ]
    if winner != CURRENT:
        lines.append(f"   Winner vs current: mean per-stint diff {w['vs_current_mean_diff']:+.4f} s, better on "
                     f"{int(w['vs_current_better_stints'])} of {int(w['stints'])} stints, t p {w['vs_current_t_p']:.4f}, "
                     f"Wilcoxon p {w['vs_current_wilcoxon_p']:.4f}.")
    lines += ["", f"Copy into .env: PUSH_LAP_MAX_BEST_RATIO={winner[0]:g}  PUSH_LAP_MAX_STINT_RATIO={winner[1]:g}"]
    name = f"{SCRIPT_NAME}_recommendation.txt"
    with open(f"{RESULTS_DIR}/{name}", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    generated_files.append(name)
    log("\n" + "\n".join(lines))

    # ---------------- restore the .env weekend ----------------
    log("\nRestoring the .env weekend and data stages at the .env values...")
    for script in ["race_weekend.py", "process_laps.py", "process_stints.py", "process_weather.py",
                   "process_cardata.py", "feature_engineer.py"]:
        if not run_stage(script):
            log("!!! Restore failed: run master.py before using data/raw/ or data/processed/.")
            sys.exit(1)

    with open(f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt", "w") as f:
        for item in generated_files:
            f.write(item + "\n")
    generated_files.append(f"{SCRIPT_NAME}_manifest.txt")
    log(f"\nDONE in {time.time() - t0:.0f}s. Files generated:")
    for item in generated_files:
        log(f"  - {item}")
    log_file.close()


if __name__ == "__main__":
    main()
