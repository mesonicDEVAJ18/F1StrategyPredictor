import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from feature_config import MODEL_FEATURE_COLUMNS as FEATURES
from feature_config import CARDATA_FEATURE_COLUMNS, SESSION_COLUMNS, TARGET_GROUP_COLUMNS
from feature_config import FUEL_EFFECT_K, build_next_lap_target
from env_config import env_str, env_float

warnings.filterwarnings("ignore")

# ============================================================
# SETUP RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "eda"
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


def save_csv(frame, name, index=False):
    fname = f"{SCRIPT_NAME}_{name}.csv"
    frame.to_csv(f"{RESULTS_DIR}/{fname}", index=index)
    generated_files.append(fname)


def save_plot(name):
    fname = f"{SCRIPT_NAME}_{name}.png"
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{fname}", dpi=300)
    plt.close()
    generated_files.append(fname)


# ============================================================
# CONFIG
# ============================================================
DATASET_PATH = os.path.join(env_str("DATA_DIR"), env_str("PROCESSED_DIR"), env_str("ENGINEERED_FILE"))
EDA_COLLINEAR_THRESHOLD = env_float("EDA_COLLINEAR_THRESHOLD")

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_COLUMNS = [f"compound_{c}" for c in COMPOUNDS]
COMPOUND_COLORS = {"SOFT": "tab:red", "MEDIUM": "goldenrod", "HARD": "dimgray"}
SESSIONS = [c.replace("session_", "") for c in SESSION_COLUMNS]
SECTOR_COLUMNS = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
# Scaled by process_weather.py (StandardScaler); wind_direction is left in degrees
SCALED_WEATHER_COLUMNS = ["air_temperature", "humidity", "pressure", "rainfall", "track_temperature", "wind_speed"]
PACE_COLUMNS = ["lap_duration", "fuel_corrected_pace"]


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATASET_PATH)
log(f"Loaded dataset with {len(df)} rows and {df.shape[1]} columns.")

# Labels from the one-hots (EDA only, not written back)
df["session"] = df[SESSION_COLUMNS].idxmax(axis=1).str.replace("session_", "", regex=False)
df["compound"] = df[COMPOUND_COLUMNS].idxmax(axis=1).str.replace("compound_", "", regex=False)

# feature_engineer.py drops lap_duration; undo the fuel correction to get it back exactly
df["lap_duration"] = df["fuel_corrected_pace"] + FUEL_EFFECT_K * (df["lap_end"] - df["lap_number"] + 1)

sector_sum = df[SECTOR_COLUMNS].sum(axis=1, min_count=len(SECTOR_COLUMNS))
has_sectors = sector_sum.notna()
max_gap = (df.loc[has_sectors, "lap_duration"] - sector_sum[has_sectors]).abs().max()
log(
    f"Reconstructed lap_duration (fcp + k * laps_remaining) against the sector sum: "
    f"{int(has_sectors.sum())} laps with all three sectors, max |difference| = {max_gap:.3f} s"
)

log(
    "Units: lap and sector times in s, speeds in km/h, car-data aggregates in physical units; "
    f"weather columns {', '.join(SCALED_WEATHER_COLUMNS)} are z-scores (process_weather.py), "
    "wind_direction in degrees."
)

# Next-lap target rows, the ones the model actually trains on
target_df = build_next_lap_target(df)
log(f"Rows after the next-lap target shift: {len(target_df)} (from {len(df)})")


# ============================================================
# 1. SUMMARY STATISTICS AND MISSING VALUES
# ============================================================
log("\n=== 1. Summary statistics and missing values ===")

summary_cols = FEATURES + PACE_COLUMNS

stats = [df[summary_cols].describe().T.assign(session="ALL")]
for session in SESSIONS:
    stats.append(df.loc[df["session"] == session, summary_cols].describe().T.assign(session=session))
stats_df = pd.concat(stats).rename_axis("column").reset_index()
save_csv(stats_df, "summary_stats")

missing_rows = []
for session in ["ALL"] + SESSIONS:
    part = df if session == "ALL" else df[df["session"] == session]
    for col in summary_cols:
        n_missing = int(part[col].isna().sum())
        missing_rows.append({
            "session": session,
            "column": col,
            "rows": len(part),
            "missing": n_missing,
            "missing_share": n_missing / len(part) if len(part) else np.nan
        })
missing_df = pd.DataFrame(missing_rows)
save_csv(missing_df, "missing_values")

missing_all = missing_df[(missing_df["session"] == "ALL") & (missing_df["missing"] > 0)]
log(f"Columns with missing values ({len(missing_all)} of {len(summary_cols)}):")
for _, r in missing_all.sort_values("missing", ascending=False).iterrows():
    by_session = ", ".join(
        f"{s} {int(missing_df[(missing_df['session'] == s) & (missing_df['column'] == r['column'])]['missing'].iloc[0])}"
        for s in SESSIONS
    )
    log(f"  {r['column']}: {r['missing']} ({r['missing_share']:.1%}); {by_session}")

constant_cols = [c for c in FEATURES if df[c].nunique(dropna=True) <= 1]
log(f"Constant inputs (no information for the model): {constant_cols if constant_cols else 'none'}")


# ============================================================
# 2. DISTRIBUTIONS OF LAP TIME AND FUEL-CORRECTED PACE
# ============================================================
log("\n=== 2. Pace distributions by session and compound ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, col in enumerate(PACE_COLUMNS):
    for ax, (group_col, groups) in zip(axes[row], [("session", SESSIONS), ("compound", COMPOUNDS)]):
        data = [df.loc[df[group_col] == g, col].dropna() for g in groups]
        ax.boxplot(data, labels=[f"{g}\n(n={len(d)})" for g, d in zip(groups, data)], showfliers=True)
        ax.set_title(f"{col} by {group_col}")
        ax.set_ylabel(f"{col} (s)")
save_plot("pace_distributions")

pace_rows = []
for (session, compound), part in df.groupby(["session", "compound"]):
    for col in PACE_COLUMNS:
        q1, med, q3 = part[col].quantile([0.25, 0.5, 0.75])
        pace_rows.append({
            "session": session,
            "compound": compound,
            "column": col,
            "laps": int(part[col].notna().sum()),
            "mean": part[col].mean(),
            "median": med,
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "min": part[col].min(),
            "max": part[col].max()
        })
pace_df = pd.DataFrame(pace_rows)
save_csv(pace_df, "pace_by_session_compound")

for _, r in pace_df[pace_df["column"] == "fuel_corrected_pace"].iterrows():
    log(
        f"  {r['session']} {r['compound']:<6}: {r['laps']:>3} laps, fcp median {r['median']:.3f} s, "
        f"IQR {r['iqr']:.3f} s, range {r['min']:.3f}-{r['max']:.3f} s"
    )


# ============================================================
# 3. PACE AGAINST TYRE AGE (the degradation the model should learn)
# ============================================================
log("\n=== 3. Fuel-corrected pace against tyre age ===")

# Raw pace mixes drivers, sessions and run plans; subtracting each stint's mean leaves
# the change within a stint, which is where degradation shows
df["fcp_stint_demeaned"] = df["fuel_corrected_pace"] - df.groupby(TARGET_GROUP_COLUMNS)["fuel_corrected_pace"].transform("mean")
stint_laps = df.groupby(TARGET_GROUP_COLUMNS)["lap_number"].transform("size")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
slope_rows = []
for compound in COMPOUNDS:
    part = df[df["compound"] == compound]
    color = COMPOUND_COLORS[compound]
    axes[0].scatter(part["tyre_age"], part["fuel_corrected_pace"], s=8, alpha=0.5, color=color, label=compound)

    # Single-lap stints de-mean to exactly 0 and carry no within-stint information
    multi = part[stint_laps[part.index] > 1]
    axes[1].scatter(multi["tyre_age"], multi["fcp_stint_demeaned"], s=8, alpha=0.5, color=color, label=compound)

    slope = np.nan
    intercept = np.nan
    if multi["tyre_age"].nunique() > 1:
        slope, intercept = np.polyfit(multi["tyre_age"], multi["fcp_stint_demeaned"], 1)
        xs = np.array([multi["tyre_age"].min(), multi["tyre_age"].max()])
        axes[1].plot(xs, intercept + slope * xs, color=color, linewidth=2)

    slope_rows.append({
        "compound": compound,
        "laps": len(part),
        "stints": part.groupby(TARGET_GROUP_COLUMNS).ngroups,
        "multi_lap_stints": multi.groupby(TARGET_GROUP_COLUMNS).ngroups,
        "laps_in_multi_lap_stints": len(multi),
        "tyre_age_min": part["tyre_age"].min(),
        "tyre_age_max": part["tyre_age"].max(),
        "demeaned_slope_s_per_lap": slope,
        "demeaned_intercept_s": intercept
    })

axes[0].set_title("Fuel-corrected pace vs tyre age (raw)")
axes[0].set_xlabel("Tyre age (laps)")
axes[0].set_ylabel("Fuel-corrected pace (s)")
axes[0].legend()
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Within-stint pace vs tyre age (stint mean removed, linear fit)")
axes[1].set_xlabel("Tyre age (laps)")
axes[1].set_ylabel("fcp - stint mean fcp (s)")
axes[1].legend()
save_plot("pace_vs_tyre_age")

slope_df = pd.DataFrame(slope_rows)
save_csv(slope_df, "degradation_slopes")

log("Descriptive linear slope of within-stint fcp on tyre age (no inference, see #26):")
for _, r in slope_df.iterrows():
    log(
        f"  {r['compound']:<6}: {r['demeaned_slope_s_per_lap']:+.4f} s/lap over {r['laps_in_multi_lap_stints']} laps "
        f"in {r['multi_lap_stints']} multi-lap stints ({r['stints']} stints in all, "
        f"tyre age {r['tyre_age_min']:.0f}-{r['tyre_age_max']:.0f})"
    )


# ============================================================
# 4. CORRELATIONS AND COLLINEAR GROUPS
# ============================================================
log("\n=== 4. Correlations between model inputs and with the target ===")

corr_cols = [c for c in FEATURES if target_df[c].nunique(dropna=True) > 1]
dropped = [c for c in FEATURES if c not in corr_cols]
log(f"Correlations on the {len(target_df)} target rows; constant inputs left out: {dropped if dropped else 'none'}")

pearson = target_df[corr_cols + ["target_next_fcp"]].corr(method="pearson")
spearman = target_df[corr_cols + ["target_next_fcp"]].corr(method="spearman")
save_csv(pearson, "correlation_matrix", index=True)
save_csv(spearman, "correlation_matrix_spearman", index=True)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(pearson.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(pearson.columns)))
ax.set_yticks(range(len(pearson.columns)))
ax.set_xticklabels(pearson.columns, rotation=90, fontsize=7)
ax.set_yticklabels(pearson.columns, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
ax.set_title(f"Pearson correlation, model inputs and target ({len(target_df)} target rows)")
save_plot("correlation_heatmap")

pair_rows = []
for i, a in enumerate(corr_cols):
    for b in corr_cols[i + 1:]:
        r = pearson.loc[a, b]
        if abs(r) >= EDA_COLLINEAR_THRESHOLD:
            pair_rows.append({"feature_a": a, "feature_b": b, "pearson_r": r, "spearman_r": spearman.loc[a, b]})
pairs_df = pd.DataFrame(pair_rows, columns=["feature_a", "feature_b", "pearson_r", "spearman_r"])

# Collinear groups: connected components of the |r| >= threshold graph
parent = {c: c for c in corr_cols}

def find(c):
    while parent[c] != c:
        parent[c] = parent[parent[c]]
        c = parent[c]
    return c

for _, r in pairs_df.iterrows():
    parent[find(r["feature_a"])] = find(r["feature_b"])

groups = {}
for c in corr_cols:
    groups.setdefault(find(c), []).append(c)
collinear_groups = sorted([g for g in groups.values() if len(g) > 1], key=len, reverse=True)
group_id = {c: n for n, g in enumerate(collinear_groups, 1) for c in g}
pairs_df["group"] = pairs_df["feature_a"].map(group_id)
pairs_df["abs_r"] = pairs_df["pearson_r"].abs()
pairs_df = pairs_df.sort_values(["group", "abs_r"], ascending=[True, False]).drop(columns="abs_r")
save_csv(pairs_df, "collinear_pairs")

log(f"Input pairs with |Pearson r| >= {EDA_COLLINEAR_THRESHOLD}: {len(pairs_df)}, in {len(collinear_groups)} groups")
for n, g in enumerate(collinear_groups, 1):
    log(f"  Group {n} ({len(g)}): {', '.join(g)}")
cardata_grouped = [c for c in CARDATA_FEATURE_COLUMNS if c in group_id]
log(f"Car-data aggregates in a collinear group: {len(cardata_grouped)} of {len(CARDATA_FEATURE_COLUMNS)}")

target_corr = pd.DataFrame({
    "feature": corr_cols,
    "pearson_r": [pearson.loc[c, "target_next_fcp"] for c in corr_cols],
    "spearman_r": [spearman.loc[c, "target_next_fcp"] for c in corr_cols],
    "non_missing_rows": [int(target_df[c].notna().sum()) for c in corr_cols]
}).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
save_csv(target_corr, "target_correlations")

log("Strongest correlations with target_next_fcp (Pearson / Spearman):")
for _, r in target_corr.head(10).iterrows():
    log(f"  {r['feature']:<24} {r['pearson_r']:+.3f} / {r['spearman_r']:+.3f}")
for col in ["tyre_age", "laps_since_stint_start", *COMPOUND_COLUMNS]:
    if col in target_corr["feature"].values:
        r = target_corr[target_corr["feature"] == col].iloc[0]
        log(f"  (tyre) {col:<17} {r['pearson_r']:+.3f} / {r['spearman_r']:+.3f}")


# ============================================================
# 5. ROW COVERAGE BEFORE AND AFTER THE TARGET SHIFT
# ============================================================
log("\n=== 5. Row coverage before and after the next-lap target shift ===")

def coverage(frame, stage):
    frame = frame.assign(stint_id=frame.groupby(TARGET_GROUP_COLUMNS).ngroup())
    out = frame.groupby(["session", "compound"]).agg(
        rows=("lap_number", "size"),
        stints=("stint_id", "nunique"),
        drivers=("driver_number", "nunique")
    ).reset_index()
    out.insert(0, "stage", stage)
    return out

target_df["session"] = target_df[SESSION_COLUMNS].idxmax(axis=1).str.replace("session_", "", regex=False)
target_df["compound"] = target_df[COMPOUND_COLUMNS].idxmax(axis=1).str.replace("compound_", "", regex=False)

coverage_df = pd.concat([coverage(df, "engineered"), coverage(target_df, "after_target_shift")], ignore_index=True)
save_csv(coverage_df, "row_coverage")

for stage, frame in [("engineered", df), ("after_target_shift", target_df)]:
    per_session = ", ".join(f"{s} {int((frame['session'] == s).sum())}" for s in SESSIONS)
    per_compound = ", ".join(f"{c} {int((frame['compound'] == c).sum())}" for c in COMPOUNDS)
    n_stints = frame.groupby(TARGET_GROUP_COLUMNS).ngroups
    log(f"  {stage}: {len(frame)} rows, {n_stints} stints; by session {per_session}; by compound {per_compound}")

stint_len = df.groupby(TARGET_GROUP_COLUMNS).agg(
    laps=("lap_number", "size"),
    session=("session", "first"),
    compound=("compound", "first")
).reset_index()
log(
    f"Clean laps per stint: median {stint_len['laps'].median():.0f}, max {stint_len['laps'].max()}; "
    f"{int((stint_len['laps'] == 1).sum())} of {len(stint_len)} stints have a single clean lap (no target row)"
)

fig, ax = plt.subplots(figsize=(9, 4))
bins = np.arange(0.5, stint_len["laps"].max() + 1.5, 1)
ax.hist([stint_len.loc[stint_len["session"] == s, "laps"] for s in SESSIONS], bins=bins, stacked=True, label=SESSIONS)
ax.set_title("Clean laps per stint (engineered dataset)")
ax.set_xlabel("Clean laps in stint")
ax.set_ylabel("Stints")
ax.legend()
save_plot("stint_lengths")


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

log("\n--- EDA COMPLETE ---\n")
log_file.close()
