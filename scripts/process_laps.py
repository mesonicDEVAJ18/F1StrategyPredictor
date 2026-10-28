import sys
import pandas as pd
import numpy as np
import os
import re
import ast

from env_config import env_str, env_int, env_float, env_list


# ============================================================
# SETUP SCRIPT-SPECIFIC RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "process_laps"
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
# INPUT FILE CONFIG
# ============================================================
data_dir = env_str("DATA_DIR")
laps_dir = os.path.join(data_dir, env_str("LAPS_DIR"))
stints_dir = os.path.join(data_dir, env_str("STINTS_DIR"))
processed_dir = os.path.join(data_dir, env_str("PROCESSED_DIR"))
laps_files = env_list("LAPS_FILES")
stints_files = env_list("STINTS_FILES")   # same session order as LAPS_FILES
cleaned_dfs = []

# Cleaning thresholds (see .env)
LAP_MIN_DURATION_S = env_float("LAP_MIN_DURATION_S")
LAP_MAX_MEDIAN_MULTIPLE = env_float("LAP_MAX_MEDIAN_MULTIPLE")
MIN_SPEED_KMH = env_float("MIN_SPEED_KMH")
ROLLING_MEDIAN_WINDOW = env_int("ROLLING_MEDIAN_WINDOW")
IQR_LOWER_MULTIPLIER = env_float("IQR_LOWER_MULTIPLIER")
IQR_UPPER_MULTIPLIER = env_float("IQR_UPPER_MULTIPLIER")
PUSH_LAP_MAX_BEST_RATIO = env_float("PUSH_LAP_MAX_BEST_RATIO")
PUSH_LAP_MAX_STINT_RATIO = env_float("PUSH_LAP_MAX_STINT_RATIO")

removed_rows_summary = []
list_fix_summary = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rebuild_missing_sector(df, fname, sector_cols):
    """A lap missing exactly one sector, with lap_duration and the other two (> 0) recorded, gets the
    missing one as lap_duration minus the other two (#33; deterministic). A result <= 0 stays NaN."""
    df = df.copy()
    missing = df[sector_cols].isna()
    others_ok = df[sector_cols].fillna(1).gt(0).all(axis=1)
    candidates = missing.sum(axis=1).eq(1) & df["lap_duration"].notna() & others_ok

    rebuilt = []
    for col in sector_cols:
        rows = candidates & missing[col]
        value = df.loc[rows, "lap_duration"] - df.loc[rows, [c for c in sector_cols if c != col]].sum(axis=1)
        value = value[value > 0]
        df.loc[value.index, col] = value
        rebuilt.append(df.loc[value.index, [c for c in ["session_key", "driver_number", "lap_number", "lap_duration"]
                                             if c in df.columns]].assign(sector=col, value=value))

    rebuilt = pd.concat(rebuilt, ignore_index=True)
    log(f"[{fname}] Rebuilt {len(rebuilt)} missing sectors as lap_duration minus the other two "
        f"({int(candidates.sum()) - len(rebuilt)} gave <= 0 and stay invalid)")
    rebuilt_name = f"{SCRIPT_NAME}_rebuilt_sectors_{fname.replace('.csv', '')}.csv"
    rebuilt.to_csv(os.path.join(RESULTS_DIR, rebuilt_name), index=False)
    generated_files.append(rebuilt_name)
    return df


def apply_domain_filters(df, fname):
    sector_cols = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]

    df = rebuild_missing_sector(df, fname, sector_cols)

    # Invalid sector rows (missing sectors that could not be rebuilt, or sectors <= 0)
    before = len(df)
    df = df.dropna(subset=sector_cols)
    df = df[(df[sector_cols] > 0).all(axis=1)]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} invalid-sector laps")
    removed_rows_summary.append([fname, "invalid_sector", removed])

    # Unrealistic lap durations
    before = len(df)
    median_lap = df["lap_duration"].median()
    df = df[(df["lap_duration"] > LAP_MIN_DURATION_S) & (df["lap_duration"] < LAP_MAX_MEDIAN_MULTIPLE * median_lap)]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} extreme lap-time outliers")
    removed_rows_summary.append([fname, "unrealistic_laptime", removed])

    # Unrealistic speeds: only a recorded speed can be too slow. NaN <= MIN_SPEED_KMH is False, so a lap
    # with a missing speed trap is kept; feature_engineer.py fills the missing value from the stint's earlier laps
    # (else the training median in train_model.py)
    speed_cols = ["i1_speed", "i2_speed", "st_speed"]
    if all(col in df.columns for col in speed_cols):
        before = len(df)
        df = df[~(df[speed_cols] <= MIN_SPEED_KMH).any(axis=1)]
        removed = before - len(df)
        log(f"[{fname}] Removed {removed} unrealistic-speed laps")
        removed_rows_summary.append([fname, "unrealistic_speed", removed])
        missing = df[speed_cols].isna().any(axis=1).sum()
        log(f"[{fname}] Kept {missing} laps with a missing speed trap (filled in feature_engineer.py / train_model.py)")

    return df


def apply_iqr_residual_filter(df, fname):
    """IQR on lap_duration - rolling_median, with a separate multiplier for the fast (lower)
    and slow (upper) side. Saves every lap's residual for explore/tune_iqr.py."""
    if "driver_number" not in df.columns:
        df["driver_number"] = 1  # fallback grouping

    df["rolling_median"] = df.groupby("driver_number")["lap_duration"].transform(
        lambda x: x.rolling(ROLLING_MEDIAN_WINDOW, center=True, min_periods=1).median()
    )
    df["residual"] = df["lap_duration"] - df["rolling_median"]

    # Lap / driver's best among the laps reaching this filter: the push-lap label tune_iqr.py scores the fences against
    df["best_ratio"] = df["lap_duration"] / df.groupby("driver_number")["lap_duration"].transform("min")
    residual_cols = [c for c in ["session_key", "driver_number", "lap_number", "lap_duration",
                                 "rolling_median", "residual", "best_ratio"] if c in df.columns]
    residual_name = f"{SCRIPT_NAME}_iqr_residuals_{fname.replace('.csv', '')}.csv"
    df[residual_cols].to_csv(os.path.join(RESULTS_DIR, residual_name), index=False)
    generated_files.append(residual_name)

    Q1 = df["residual"].quantile(0.25)
    Q3 = df["residual"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - IQR_LOWER_MULTIPLIER * IQR
    upper = Q3 + IQR_UPPER_MULTIPLIER * IQR
    log(f"[{fname}] Residual Q1 {Q1:.2f} s, Q3 {Q3:.2f} s, IQR {IQR:.2f} s; bounds [{lower:.2f}, {upper:.2f}] s "
        f"(k_lower {IQR_LOWER_MULTIPLIER:g}, k_upper {IQR_UPPER_MULTIPLIER:g})")

    fast = int((df["residual"] < lower).sum())
    slow = int((df["residual"] > upper).sum())
    df = df[(df["residual"] >= lower) & (df["residual"] <= upper)]
    log(f"[{fname}] Removed {fast + slow} IQR-residual outliers ({fast} fast, {slow} slow)")
    removed_rows_summary.append([fname, "iqr_residual_fast", fast])
    removed_rows_summary.append([fname, "iqr_residual_slow", slow])

    return df.drop(columns=["rolling_median", "residual", "best_ratio"], errors="ignore")


def assign_stints(df, stints, fname):
    """Tags each lap with the driver's stint_number whose lap_start..lap_end holds it (as
    feature_engineer.assign_stint_numbers(), within one session). Unmatched laps stay NaN."""
    df["stint_number"] = np.nan
    for _, row in stints.iterrows():
        mask = (
            (df["driver_number"] == row["driver_number"]) &
            (df["lap_number"] >= row["lap_start"]) &
            (df["lap_number"] <= row["lap_end"])
        )
        df.loc[mask, "stint_number"] = row["stint_number"]
    log(f"[{fname}] Laps without a stint: {df['stint_number'].isna().sum()} (only the session test applies to them)")
    return df


def apply_push_lap_filter(df, fname, stints):
    """Drop non-push laps (#30): slower than PUSH_LAP_MAX_BEST_RATIO x the driver's best lap in
    this session, or than PUSH_LAP_MAX_STINT_RATIO x the driver's best lap in the same stint.
    Each file is one session. Saves every lap's two ratios for explore/tune_push.py."""
    df = assign_stints(df, stints, fname)
    df["session_ratio"] = df["lap_duration"] / df.groupby("driver_number")["lap_duration"].transform("min")
    df["stint_ratio"] = df["lap_duration"] / df.groupby(["driver_number", "stint_number"])["lap_duration"].transform("min")

    ratio_cols = [c for c in ["session_key", "driver_number", "lap_number", "stint_number", "lap_duration",
                              "session_ratio", "stint_ratio"] if c in df.columns]
    ratio_name = f"{SCRIPT_NAME}_push_ratios_{fname.replace('.csv', '')}.csv"
    df[ratio_cols].to_csv(os.path.join(RESULTS_DIR, ratio_name), index=False)
    generated_files.append(ratio_name)

    before = len(df)
    df = df[df["session_ratio"] <= PUSH_LAP_MAX_BEST_RATIO]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} non-push laps (> {PUSH_LAP_MAX_BEST_RATIO:g}x driver's session best)")
    removed_rows_summary.append([fname, "non_push", removed])

    # NaN stint_ratio (no stint) is kept
    before = len(df)
    df = df[~(df["stint_ratio"] > PUSH_LAP_MAX_STINT_RATIO)]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} non-push laps (> {PUSH_LAP_MAX_STINT_RATIO:g}x driver's stint best)")
    removed_rows_summary.append([fname, "non_push_stint", removed])

    return df.drop(columns=["stint_number", "session_ratio", "stint_ratio"])


def session_name_from_file(file):
    """laps_p1.csv -> FP1 (practice order, stable across weekends unlike session_key)."""
    match = re.search(r"_p(\d+)", file)
    if match is None:
        raise ValueError(f"Cannot read the practice session number from {file}")
    return f"FP{match.group(1)}"


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

for file, stints_file in zip(laps_files, stints_files):
    file_path = os.path.join(laps_dir, file)
    df = pd.read_csv(file_path)
    stints = pd.read_csv(os.path.join(stints_dir, stints_file))
    fname = file.replace(".csv", "")
    log(f"\nLoaded {file} with {len(df)} rows")

    # Fix list-like columns
    LIST_COLS = ["segments_sector_1", "segments_sector_2", "segments_sector_3"]
    for col in LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else x)
            list_fix_summary.append([file, col, "converted_list_to_len"])
            log(f"  → Fixed list column: {col}")

    # Remove pit-out laps
    if "is_pit_out_lap" in df.columns:
        before = len(df)
        df = df[~df["is_pit_out_lap"]]
        removed = before - len(df)
        log(f"  → Removed {removed} pit-out laps")
        removed_rows_summary.append([file, "pit_out", removed])
        df = df.drop(columns=["is_pit_out_lap"])

    # Drop metadata columns (session_key is kept: lap numbers restart every session,
    # so feature_engineer.py needs it to assign laps to the right session's stints)
    df = df.drop(columns=[c for c in ["meeting_key"] if c in df.columns], errors="ignore")

    # Apply cleaning pipeline
    df = apply_domain_filters(df, file)
    df = apply_iqr_residual_filter(df, file)
    df = apply_push_lap_filter(df, file, stints)

    # Lap times and speeds stay in physical units (s, km/h): the models are scale-invariant,
    # and per-session scaling would erase each session's pace level
    log(f"[{file}] Kept lap_duration, sector times and speeds unscaled (s, km/h)")

    # Practice session label, one-hot encoded as a model input in feature_engineer.py
    df["session_name"] = session_name_from_file(file)
    log(f"[{file}] Tagged session_name = {df['session_name'].iloc[0]}")

    # Save cleaned file
    cleaned_name = f"{SCRIPT_NAME}_cleaned_{fname}.csv"
    cleaned_path = os.path.join(RESULTS_DIR, cleaned_name)
    df.to_csv(cleaned_path, index=False)
    generated_files.append(cleaned_name)
    log(f"  → Saved cleaned file: {cleaned_name}")

    cleaned_dfs.append(df)

# Combine all cleaned data
combined = pd.concat(cleaned_dfs, ignore_index=True)
combined_name = f"{SCRIPT_NAME}_combined_cleaned.csv"
combined_path = os.path.join(RESULTS_DIR, combined_name)
combined.to_csv(combined_path, index=False)
generated_files.append(combined_name)
log(f"\nSaved combined dataset: {combined_name}")

# Hand-off for feature_engineer.py
handoff_path = os.path.join(processed_dir, env_str("LAPS_CLEAN_FILE"))
combined.to_csv(handoff_path, index=False)
log(f"Saved hand-off dataset: {handoff_path}")

# Save row-removal summary
summary_df = pd.DataFrame(removed_rows_summary, columns=["file", "stage", "rows_removed"])
summary_name = f"{SCRIPT_NAME}_removed_rows_summary.csv"
summary_df.to_csv(f"{RESULTS_DIR}/{summary_name}", index=False)
generated_files.append(summary_name)

# Save list column fix summary
fix_df = pd.DataFrame(list_fix_summary, columns=["file", "column", "action"])
fix_name = f"{SCRIPT_NAME}_list_column_fix.csv"
fix_df.to_csv(f"{RESULTS_DIR}/{fix_name}", index=False)
generated_files.append(fix_name)

# Create manifest
manifest_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt"
with open(manifest_path, "w") as f:
    for item in generated_files:
        f.write(item + "\n")

generated_files.append(f"{SCRIPT_NAME}_manifest.txt")

log("\nPROCESS COMPLETED. Files generated:")
for f in generated_files:
    log(f"  - {f}")

log_file.close()