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
processed_dir = os.path.join(data_dir, env_str("PROCESSED_DIR"))
laps_files = env_list("LAPS_FILES")
cleaned_dfs = []

# Cleaning thresholds (see .env)
LAP_MIN_DURATION_S = env_float("LAP_MIN_DURATION_S")
LAP_MAX_MEDIAN_MULTIPLE = env_float("LAP_MAX_MEDIAN_MULTIPLE")
MIN_SPEED_KMH = env_float("MIN_SPEED_KMH")
ROLLING_MEDIAN_WINDOW = env_int("ROLLING_MEDIAN_WINDOW")
IQR_LOWER_MULTIPLIER = env_float("IQR_LOWER_MULTIPLIER")
IQR_UPPER_MULTIPLIER = env_float("IQR_UPPER_MULTIPLIER")
PUSH_LAP_MAX_BEST_RATIO = env_float("PUSH_LAP_MAX_BEST_RATIO")

removed_rows_summary = []
list_fix_summary = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def apply_domain_filters(df, fname):
    sector_cols = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]

    # Invalid sector rows
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
    # with a missing speed trap is kept; the missing value gets the training median in train_model.py
    speed_cols = ["i1_speed", "i2_speed", "st_speed"]
    if all(col in df.columns for col in speed_cols):
        before = len(df)
        df = df[~(df[speed_cols] <= MIN_SPEED_KMH).any(axis=1)]
        removed = before - len(df)
        log(f"[{fname}] Removed {removed} unrealistic-speed laps")
        removed_rows_summary.append([fname, "unrealistic_speed", removed])
        missing = df[speed_cols].isna().any(axis=1).sum()
        log(f"[{fname}] Kept {missing} laps with a missing speed trap (filled with the training medians in train_model.py)")

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


def apply_push_lap_filter(df, fname):
    """Drop non-push laps: slower than PUSH_LAP_MAX_BEST_RATIO x the driver's best lap
    in this session. Each file is one session."""
    best_lap = df.groupby("driver_number")["lap_duration"].transform("min")

    before = len(df)
    df = df[df["lap_duration"] <= PUSH_LAP_MAX_BEST_RATIO * best_lap]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} non-push laps (> {PUSH_LAP_MAX_BEST_RATIO}x driver's session best)")
    removed_rows_summary.append([fname, "non_push", removed])

    return df


def session_name_from_file(file):
    """laps_p1.csv -> FP1 (practice order, stable across weekends unlike session_key)."""
    match = re.search(r"_p(\d+)", file)
    if match is None:
        raise ValueError(f"Cannot read the practice session number from {file}")
    return f"FP{match.group(1)}"


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

for file in laps_files:
    file_path = os.path.join(laps_dir, file)
    df = pd.read_csv(file_path)
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
    df = apply_push_lap_filter(df, file)

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