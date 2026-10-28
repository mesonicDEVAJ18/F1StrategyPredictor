# Tunes process_laps.py's two IQR multipliers (#32): IQR_LOWER_MULTIPLIER (fast fence) and
# IQR_UPPER_MULTIPLIER (slow fence), over the weekends in the experiment block below.
#
#   cd explore
#   python tune_iqr.py            # every weekend in TUNE_WEEKENDS (downloads what data/cache/ lacks)
#   python tune_iqr.py --quick    # smoke run: the two weekends cached first, small grid, 1 repeat
#
# Two methods, both on the pipeline's own outputs (the stages run from scripts/ with the settings
# passed as environment variables, which override .env):
#   A. label sweep: process_laps.py saves every lap's residual and its ratio to the driver's best
#      (process_laps_iqr_residuals_*.csv). A lap <= PUSH_LAP_MAX_BEST_RATIO is a push lap. For each side
#      and session, the k at which the fence starts removing push laps; the lower-side floor is the
#      smallest K_GRID value at least LABEL_MARGIN x the largest of them.
#   B. CV sweep: the data stages rerun for each (k_lower, k_upper); XGBoost (train_model.py's MODEL block)
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
PUSH_LAP_MAX_BEST_RATIO = env_float("PUSH_LAP_MAX_BEST_RATIO")

# ######################################################################
# EXPERIMENT BLOCK: edit here
# ######################################################################
# Non-sprint weekends in race_weekend.WEEKENDS, a mix of track types over both years
TUNE_WEEKENDS = [
    (2025, "bahrain"), (2024, "saudi_arabia"),
    (2025, "australia"), (2025, "japan"), (2025, "spain"), (2025, "austria"),
    (2025, "britain"), (2025, "netherlands"), (2025, "italy"), (2025, "singapore"),
    (2024, "hungary"), (2024, "spain"),
]
K_GRID = [1, 1.5, 2, 3, 4, 6, 8, 10, 12, 15, 20]   # label sweep, both sides
K_LOWER = [3, 6, 10, 20]                           # CV sweep; the label floor is added
K_UPPER = [1.5, 2, 3, 5]
CURRENT = (3.0, 3.0)                               # the single multiplier before #32
LABEL_MARGIN = 1.2                                 # floor >= this x the largest lower k needed
CV_REPEATS = 2                                     # stint folds reshuffled with seeds RANDOM_STATE + r
ALPHA = 0.05                                       # paired t-test and Wilcoxon (one-sided), both must pass
# ######################################################################


# ======================================================
# SETUP RESULTS DIRECTORY
# ======================================================
SCRIPT_NAME = "tune_iqr"
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


# ======================================================
# PIPELINE STAGES
# ======================================================
STAGE_RESULTS = os.path.join(SCRIPTS_DIR, env_str("RESULTS_ROOT"))
PROCESSED = os.path.join(env_str("DATA_DIR"), env_str("PROCESSED_DIR"))
LAPS_CLEAN_PATH = os.path.join(PROCESSED, env_str("LAPS_CLEAN_FILE"))
DATASET_PATH = os.path.join(PROCESSED, env_str("ENGINEERED_FILE"))
LAPS_FILES = env_list("LAPS_FILES")


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


def iqr_env(k_lower, k_upper):
    return {"IQR_LOWER_MULTIPLIER": k_lower, "IQR_UPPER_MULTIPLIER": k_upper}


def switch_weekend(year, name):
    """Puts the weekend in data/raw/ (downloading what the cache lacks) and builds the stints and
    weather hand-offs, which don't depend on the IQR fences."""
    t0 = time.time()
    for script in ["race_weekend.py", "process_stints.py", "process_weather.py"]:
        if not run_stage(script, weekend_env(year, name)):
            return False
    log(f"  {year} {name}: in data/raw/, stints and weather processed ({time.time() - t0:.0f}s)")
    return True


# ======================================================
# A. LABEL SWEEP
# ======================================================
def label_sweep(weekend_label):
    """Residuals of the laps reaching the IQR filter, from process_laps.py at the current fences.
    Returns (counts per side and k, k needed per side) for each session."""
    counts, needed = [], []
    for file in LAPS_FILES:
        fname = file.replace(".csv", "")
        res = pd.read_csv(os.path.join(STAGE_RESULTS, "process_laps", f"process_laps_iqr_residuals_{fname}.csv"))
        r = res["residual"]
        q1, q3 = r.quantile(0.25), r.quantile(0.75)   # as process_laps.py
        iqr = q3 - q1
        push = res["best_ratio"] <= PUSH_LAP_MAX_BEST_RATIO
        session = f"FP{fname[-1]}"

        k_lower = max(0.0, (q1 - r[push].min()) / iqr)
        k_upper = max(0.0, (r[push].max() - q3) / iqr)
        needed.append({"weekend": weekend_label, "session": session, "laps": len(res), "push_laps": int(push.sum()),
                       "q1": q1, "q3": q3, "iqr": iqr, "min_push_residual": r[push].min(),
                       "max_push_residual": r[push].max(), "k_needed_lower": k_lower, "k_needed_upper": k_upper})
        for k in K_GRID:
            for side, out in [("lower", r < q1 - k * iqr), ("upper", r > q3 + k * iqr)]:
                counts.append({"weekend": weekend_label, "session": session, "side": side, "k": k,
                               "push_dropped": int((out & push).sum()), "non_push_dropped": int((out & ~push).sum())})
        log(f"  {session}: {len(res)} laps ({int(push.sum())} push), IQR {iqr:.2f} s; "
            f"k needed lower {k_lower:.2f}, upper {k_upper:.2f}")
    return counts, needed


def label_floor(k_needed):
    target = LABEL_MARGIN * k_needed["k_needed_lower"].max()
    above = [k for k in K_GRID if k >= target]
    return float(above[0]) if above else float(np.ceil(target))


# ======================================================
# B. CV SWEEP
# ======================================================
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def removed_counts():
    s = pd.read_csv(os.path.join(STAGE_RESULTS, "process_laps", "process_laps_removed_rows_summary.csv"))
    return {stage: int(s.loc[s["stage"] == stage, "rows_removed"].sum())
            for stage in ["iqr_residual_fast", "iqr_residual_slow", "non_push"]}


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
    for k_lower, k_upper in configs:
        env = {**weekend_env(year, name), **iqr_env(k_lower, k_upper)}
        if not run_stage("process_laps.py", env):
            return None, None
        h = file_hash(LAPS_CLEAN_PATH)
        config_hash[(k_lower, k_upper)] = h
        config_info[(k_lower, k_upper)] = {"cleaned_laps": len(pd.read_csv(LAPS_CLEAN_PATH)), **removed_counts()}
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
        scores[h] = {"target_rows": len(df), "common_rows": int(on_common.sum()),
                     "mae_common": np.mean(maes), "mae_common_sd": np.std(maes),
                     "rmse_common": np.mean(rmses), "rmse_common_sd": np.std(rmses),
                     "mae_own_rows": np.mean(own),
                     "persistence_mae_common": np.mean(np.abs(persistence - y[on_common]))}
        err = pd.DataFrame({"stint_key": df.loc[on_common, "stint_key"].to_numpy(),
                            "abs_err": np.mean([np.abs(p[on_common] - y[on_common]) for p in preds], axis=0)})
        per_stint[h] = err.groupby("stint_key")["abs_err"].agg(["mean", "size"])

    rows, stint_rows = [], []
    for (k_lower, k_upper), h in config_hash.items():
        rows.append({"weekend": weekend_label, "k_lower": k_lower, "k_upper": k_upper,
                     **config_info[(k_lower, k_upper)], **scores[h], "dataset": h[:8]})
        for key, r in per_stint[h].iterrows():
            stint_rows.append({"weekend": weekend_label, "k_lower": k_lower, "k_upper": k_upper,
                               "stint": f"{weekend_label}:{key[0]}:{key[1]}", "rows": int(r["size"]), "mae": r["mean"]})
    s = scores[config_hash[configs[0]]]
    log(f"  {s['common_rows']} target rows common to every configuration ({len(stints)} stints); "
        f"MAE on them {min(x['mae_common'] for x in scores.values()):.3f}-{max(x['mae_common'] for x in scores.values()):.3f} s")
    return rows, stint_rows


def paired_test(per_stint, config, reference):
    """One-sided: is `config` better (lower per-stint MAE) than `reference`? Pooled over weekends."""
    wide = per_stint.pivot_table(index="stint", columns=["k_lower", "k_upper"], values="mae")
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
def plot_label_sweep(counts, floor):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, side in zip(axes, ["lower", "upper"]):
        part = counts[counts["side"] == side].groupby(["weekend", "k"])["push_dropped"].sum().reset_index()
        for weekend, g in part.groupby("weekend"):
            ax.plot(g["k"], g["push_dropped"], marker="o", ms=3, lw=1, label=weekend)
        if side == "lower":
            ax.axvline(floor, color="black", ls="--", lw=1, label=f"floor k={floor:g}")
        ax.axvline(CURRENT[0], color="grey", ls=":", lw=1)
        ax.set_title(f"{side} fence: push laps removed")
        ax.set_xlabel("k (multiple of the session IQR)")
    axes[0].set_ylabel("push laps removed (FP1-3)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    name = f"{SCRIPT_NAME}_label_sweep.png"
    fig.savefig(f"{RESULTS_DIR}/{name}", dpi=130)
    plt.close(fig)
    generated_files.append(name)


def plot_cv_heatmap(pooled):
    grid = pooled.pivot(index="k_lower", columns="k_upper", values="mean_stint_mae")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(grid.columns)), [f"{c:g}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{i:g}" for i in grid.index])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.iat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xlabel("k_upper")
    ax.set_ylabel("k_lower")
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

    global TUNE_WEEKENDS, K_LOWER, K_UPPER, CV_REPEATS
    if args.quick:
        TUNE_WEEKENDS, K_LOWER, K_UPPER, CV_REPEATS = TUNE_WEEKENDS[:2], [3, 10], [2, 3], 1

    t0 = time.time()
    log(f"Tuning the IQR fences on {len(TUNE_WEEKENDS)} weekends: "
        f"{', '.join(f'{y} {n}' for y, n in TUNE_WEEKENDS)}\n")

    # ---------------- A. label sweep ----------------
    log("=== A. Label sweep (push lap = best_ratio <= "
        f"{PUSH_LAP_MAX_BEST_RATIO}) ===")
    counts, needed, usable = [], [], []
    for year, name in TUNE_WEEKENDS:
        log(f"\n{year} {name}")
        if not (switch_weekend(year, name) and run_stage("process_laps.py", {**weekend_env(year, name), **iqr_env(*CURRENT)})):
            log(f"  skipped: a stage failed")
            continue
        c, n = label_sweep(f"{year}_{name}")
        counts += c
        needed += n
        usable.append((year, name))

    counts, k_needed = pd.DataFrame(counts), pd.DataFrame(needed)
    save_csv(counts, f"{SCRIPT_NAME}_label_sweep.csv")
    save_csv(k_needed, f"{SCRIPT_NAME}_k_needed.csv")
    floor = label_floor(k_needed)
    worst = k_needed.loc[k_needed["k_needed_lower"].idxmax()]
    totals = counts.groupby(["side", "k"])[["push_dropped", "non_push_dropped"]].sum().reset_index()
    log(f"\nLargest lower k needed: {worst['k_needed_lower']:.2f} ({worst['weekend']} {worst['session']}, "
        f"IQR {worst['iqr']:.2f} s, push residual {worst['min_push_residual']:.1f} s)")
    log(f"Largest upper k needed: {k_needed['k_needed_upper'].max():.2f}")
    log(f"Lower-side floor: {floor:g} (smallest K_GRID value >= {LABEL_MARGIN} x {worst['k_needed_lower']:.2f})")
    log("Laps removed over all sessions (push / non-push):")
    for side in ["lower", "upper"]:
        part = totals[totals["side"] == side]
        log(f"  {side}: " + ", ".join(f"k={r.k:g} {r.push_dropped}/{r.non_push_dropped}" for r in part.itertuples()))
    plot_label_sweep(counts, floor)

    # ---------------- B. CV sweep ----------------
    k_lowers = sorted(set(K_LOWER) | {floor, CURRENT[0]})
    k_uppers = sorted(set(K_UPPER) | {CURRENT[1]})
    configs = [(float(kl), float(ku)) for kl in k_lowers for ku in k_uppers]
    log(f"\n=== B. CV sweep: k_lower {k_lowers} x k_upper {k_uppers}, {CV_FOLDS} stint folds x {CV_REPEATS} ===")
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
              .groupby(["k_lower", "k_upper"])
              .agg(stints=("stint", "size"), rows=("rows", "sum"), mean_stint_mae=("mae", "mean"), abs_sum=("abs_sum", "sum"))
              .reset_index())
    pooled["row_mae"] = pooled["abs_sum"] / pooled["rows"]
    pooled = pooled.drop(columns="abs_sum")
    laps = cv.groupby(["k_lower", "k_upper"])[["cleaned_laps", "target_rows", "iqr_residual_fast",
                                                "iqr_residual_slow", "non_push"]].sum().reset_index()
    pooled = pooled.merge(laps, on=["k_lower", "k_upper"])

    reference = (floor, CURRENT[1])
    tests = []
    for cfg in configs:
        vs_ref = paired_test(per_stint, cfg, reference) if cfg != reference else {}
        vs_cur = paired_test(per_stint, cfg, CURRENT) if cfg != CURRENT else {}
        tests.append({"k_lower": cfg[0], "k_upper": cfg[1],
                      **{f"vs_reference_{k}": v for k, v in vs_ref.items()},
                      **{f"vs_current_{k}": v for k, v in vs_cur.items()}})
    pooled = pooled.merge(pd.DataFrame(tests), on=["k_lower", "k_upper"], how="left")
    save_csv(pooled, f"{SCRIPT_NAME}_cv_pooled.csv")
    plot_cv_heatmap(pooled)

    log(f"\nPooled over {cv['weekend'].nunique()} weekends ({int(pooled['stints'].iloc[0])} stints, "
        f"{int(pooled['rows'].iloc[0])} common rows):")
    for r in pooled.sort_values(["k_lower", "k_upper"]).itertuples():
        log(f"  ({r.k_lower:g}, {r.k_upper:g}): stint MAE {r.mean_stint_mae:.4f} s, row MAE {r.row_mae:.4f} s, "
            f"cleaned laps {r.cleaned_laps}, target rows {r.target_rows}, fast/slow/non-push removed "
            f"{r.iqr_residual_fast}/{r.iqr_residual_slow}/{r.non_push}")

    # ---------------- winner ----------------
    # The label floor (keeps every push lap) with the current upper fence, unless another configuration
    # beats it on both one-sided paired tests; among those, the lowest pooled per-stint MAE
    significant = pooled[(pooled["vs_reference_t_p"] < ALPHA) & (pooled["vs_reference_wilcoxon_p"] < ALPHA)]
    if significant.empty:
        winner = reference
        why = (f"no configuration beats the reference ({reference[0]:g}, {reference[1]:g}) on both paired tests "
               f"at p < {ALPHA}, so the reference is kept")
    else:
        best = significant.sort_values("mean_stint_mae").iloc[0]
        winner = (best["k_lower"], best["k_upper"])
        why = (f"({winner[0]:g}, {winner[1]:g}) beats the reference ({reference[0]:g}, {reference[1]:g}) on both paired "
               f"tests (t p {best['vs_reference_t_p']:.4f}, Wilcoxon p {best['vs_reference_wilcoxon_p']:.4f})")
    w = pooled[(pooled["k_lower"] == winner[0]) & (pooled["k_upper"] == winner[1])].iloc[0]
    c = pooled[(pooled["k_lower"] == CURRENT[0]) & (pooled["k_upper"] == CURRENT[1])].iloc[0]

    at3 = totals[totals["k"] == 3].set_index("side")
    lines = [
        f"IQR fence tuning over {cv['weekend'].nunique()} weekends ({', '.join(sorted(cv['weekend'].unique()))})",
        "",
        f"A. Label sweep: largest lower k needed {worst['k_needed_lower']:.2f} ({worst['weekend']} {worst['session']}); "
        f"floor {floor:g} (>= {LABEL_MARGIN}x). Largest upper k needed {k_needed['k_needed_upper'].max():.2f}.",
        f"   At k=3 the lower fence removes {at3.at['lower', 'push_dropped']} push / {at3.at['lower', 'non_push_dropped']} "
        f"non-push laps, the upper fence {at3.at['upper', 'push_dropped']} push / {at3.at['upper', 'non_push_dropped']} non-push.",
        f"B. CV sweep: {why}.",
        f"   Winner ({winner[0]:g}, {winner[1]:g}): pooled per-stint MAE {w['mean_stint_mae']:.4f} s, row MAE {w['row_mae']:.4f} s, "
        f"{int(w['cleaned_laps'])} cleaned laps, {int(w['target_rows'])} target rows.",
        f"   Current (3, 3): pooled per-stint MAE {c['mean_stint_mae']:.4f} s, row MAE {c['row_mae']:.4f} s, "
        f"{int(c['cleaned_laps'])} cleaned laps, {int(c['target_rows'])} target rows.",
    ]
    if winner != CURRENT:
        lines.append(f"   Winner vs current: mean per-stint diff {w['vs_current_mean_diff']:+.4f} s, better on "
                     f"{int(w['vs_current_better_stints'])} of {int(w['stints'])} stints, t p {w['vs_current_t_p']:.4f}, "
                     f"Wilcoxon p {w['vs_current_wilcoxon_p']:.4f}.")
    lines += ["", f"Copy into .env: IQR_LOWER_MULTIPLIER={winner[0]:g}  IQR_UPPER_MULTIPLIER={winner[1]:g}"]
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
