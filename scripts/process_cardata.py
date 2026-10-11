import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETUP SCRIPT-SPECIFIC RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "process_cardata"
RESULTS_DIR = f"./results/{SCRIPT_NAME}"
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
# INPUT FILE CONFIG
# ============================================================
data_dir = Path("../data")
cardata_files = sorted(data_dir.glob("car_data_p*_*.csv"))
laps_files = sorted(data_dir.glob("laps_p*.csv"))
clean_laps_file = data_dir / "laps_combined_clean.csv"

if not cardata_files:
    raise FileNotFoundError("No car_data_p*_*.csv files found at ../data")
if not clean_laps_file.exists():
    raise FileNotFoundError("laps_combined_clean.csv not found at ../data, run process_laps.py first")

removed_rows_summary = []

# brake/throttle read 104 when there is no valid input (car stationary / in the garage)
NO_INPUT_VALUE = 104
# drs codes 10, 12, 14 mean the flap is open
DRS_OPEN_CODES = [10, 12, 14]
# laps whose sample count is below this fraction of the expected count are dropped
MIN_COVERAGE = 0.5

LAP_KEYS = ["session_key", "driver_number", "lap_number"]

# Per-lap aggregates checked for outliers. drs_open_pct is left out: it is mostly 0,
# so any lap that uses DRS would look like an outlier.
ZSCORE_COLS = [
    "mean_speed", "max_speed",
    "mean_throttle", "full_throttle_pct",
    "brake_pct",
    "mean_rpm", "max_rpm",
    "mean_gear", "gear_changes",
]
SCALE_COLS = ZSCORE_COLS + ["drs_open_pct"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def apply_domain_filters(df, fname):
    # Duplicate timestamps
    before = len(df)
    df = df.drop_duplicates(subset=["date"])
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} duplicate-timestamp samples")
    removed_rows_summary.append([fname, "duplicate_timestamp", removed])

    # No-input samples (brake/throttle = 104)
    before = len(df)
    df = df[(df["brake"] != NO_INPUT_VALUE) & (df["throttle"] != NO_INPUT_VALUE)]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} no-input samples (brake/throttle = {NO_INPUT_VALUE})")
    removed_rows_summary.append([fname, "no_input", removed])

    # Physically impossible readings
    before = len(df)
    df = df[
        df["speed"].between(0, 400)
        & df["rpm"].between(0, 16000)
        & df["n_gear"].between(0, 8)
        & df["throttle"].between(0, 100)
        & df["brake"].isin([0, 100])
    ]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} out-of-range samples")
    removed_rows_summary.append([fname, "out_of_range", removed])

    return df


def encode_samples(df, fname):
    """brake 0/100 -> braking 0/1, drs status codes -> drs_open 0/1. n_gear stays ordinal."""
    df = df.copy()
    df["braking"] = (df["brake"] == 100).astype(int)
    df["drs_open"] = df["drs"].isin(DRS_OPEN_CODES).astype(int)
    df = df.drop(columns=["brake", "drs"])
    log(f"[{fname}] Encoded brake -> braking, drs -> drs_open")
    return df


def load_lap_windows():
    """Lap time windows [date_start, date_start + lap_duration) from the raw lap files.

    The cleaned laps hand-off scales lap_duration, so the raw files are used here.
    """
    laps = pd.concat(
        pd.read_csv(f, usecols=["session_key", "driver_number", "lap_number", "date_start", "lap_duration"])
        for f in laps_files
    )
    laps = laps.dropna(subset=["date_start", "lap_duration"]).copy()
    laps["date_start"] = pd.to_datetime(laps["date_start"], format="ISO8601", utc=True)
    laps["lap_end"] = laps["date_start"] + pd.to_timedelta(laps["lap_duration"], unit="s")
    return laps.sort_values("date_start").reset_index(drop=True)


def assign_samples_to_laps(df, laps, fname):
    """Match each sample to the lap whose time window contains it."""
    df = pd.merge_asof(
        df.sort_values("date"),
        laps,
        left_on="date",
        right_on="date_start",
        by=["session_key", "driver_number"],
        direction="backward",
    )

    before = len(df)
    df = df[df["lap_end"].notna() & (df["date"] < df["lap_end"])]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} samples outside any lap window")
    removed_rows_summary.append([fname, "outside_lap", removed])

    # Unmatched samples made lap_number float during the merge
    df["lap_number"] = df["lap_number"].astype(int)
    return df


def aggregate_per_lap(df):
    """Summarise the ~4 Hz samples of each lap."""
    df = df.copy()
    df["full_throttle"] = df["throttle"] >= 99
    df["gear_change"] = df.groupby(LAP_KEYS)["n_gear"].diff().fillna(0).ne(0)

    return df.groupby(LAP_KEYS).agg(
        date_start=("date_start", "first"),
        lap_duration=("lap_duration", "first"),
        n_samples=("date", "size"),
        mean_speed=("speed", "mean"),
        max_speed=("speed", "max"),
        mean_throttle=("throttle", "mean"),
        full_throttle_pct=("full_throttle", "mean"),
        brake_pct=("braking", "mean"),
        mean_rpm=("rpm", "mean"),
        max_rpm=("rpm", "max"),
        mean_gear=("n_gear", "mean"),
        gear_changes=("gear_change", "sum"),
        drs_open_pct=("drs_open", "mean"),
    ).reset_index()


def apply_coverage_filter(per_lap, sample_rate, fname):
    """Drop laps with too few samples for their duration (gappy telemetry)."""
    per_lap["coverage"] = per_lap["n_samples"] / (per_lap["lap_duration"] * sample_rate)

    before = len(per_lap)
    per_lap = per_lap[per_lap["coverage"] >= MIN_COVERAGE]
    removed = before - len(per_lap)
    log(f"[{fname}] Removed {removed} laps with coverage < {MIN_COVERAGE}")
    removed_rows_summary.append([fname, "low_coverage_laps", removed])

    return per_lap


def keep_clean_laps(per_lap, fname):
    """Keep only laps that survived process_laps.py (drops pit-out, invalid and outlier laps).

    (driver_number, date_start) is unique, so it identifies a lap across sessions.
    """
    clean = pd.read_csv(clean_laps_file, usecols=["driver_number", "date_start"])
    clean["date_start"] = pd.to_datetime(clean["date_start"], format="ISO8601", utc=True)

    before = len(per_lap)
    per_lap = per_lap.merge(clean, on=["driver_number", "date_start"], how="inner")
    removed = before - len(per_lap)
    log(f"[{fname}] Removed {removed} laps not in the cleaned lap data")
    removed_rows_summary.append([fname, "not_in_clean_laps", removed])

    return per_lap


def zscore_outlier_filter(per_lap, cols, fname):
    """Drop laps with |z| > 3 in any aggregate, z computed per session."""
    keep = pd.Series(True, index=per_lap.index)
    for col in cols:
        grouped = per_lap.groupby("session_key")[col]
        std = grouped.transform("std")
        z = (per_lap[col] - grouped.transform("mean")) / std
        # Zero-variance columns are skipped
        keep &= ~((std > 0) & (z.abs() > 3))

    before = len(per_lap)
    per_lap = per_lap[keep]
    removed = before - len(per_lap)
    log(f"[{fname}] Removed {removed} laps with |z| > 3 (per session)")
    removed_rows_summary.append([fname, "zscore_outlier", removed])

    return per_lap


def scale_per_session(per_lap, cols, fname):
    """Standard-scale the per-lap aggregates, one scaler per session (as for laps)."""
    per_lap = per_lap.copy()
    # max_speed, max_rpm and gear_changes are integers
    per_lap[cols] = per_lap[cols].astype(float)
    for session, idx in per_lap.groupby("session_key").groups.items():
        scaler = StandardScaler()
        per_lap.loc[idx, cols] = scaler.fit_transform(per_lap.loc[idx, cols])
    log(f"[{fname}] Scaled per session: {cols}")
    return per_lap


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================
laps = load_lap_windows()
log(f"Loaded {len(laps)} lap windows from {[f.name for f in laps_files]}")

samples_by_file = {}
per_lap_dfs = []

for file_path in cardata_files:
    file = file_path.name
    df = pd.read_csv(file_path)
    log(f"\nLoaded {file} with {len(df)} rows (drivers {df['driver_number'].unique().tolist()}, "
        f"session {df['session_key'].unique().tolist()})")

    df["date"] = pd.to_datetime(df["date"], format="ISO8601", utc=True)

    # Drop metadata columns (session_key is kept to match samples to laps)
    df = df.drop(columns=["meeting_key"], errors="ignore")

    # Sampling rate from the raw stream, before filtering opens gaps
    sample_rate = 1 / df["date"].sort_values().diff().dt.total_seconds().median()
    log(f"[{file}] Median sample rate: {sample_rate:.2f} Hz")

    # Apply cleaning pipeline
    df = apply_domain_filters(df, file)
    df = encode_samples(df, file)
    df = assign_samples_to_laps(df, laps, file)

    per_lap = aggregate_per_lap(df)
    per_lap = apply_coverage_filter(per_lap, sample_rate, file)
    per_lap = keep_clean_laps(per_lap, file)
    log(f"[{file}] {len(per_lap)} laps aggregated")

    samples_by_file[file] = df
    per_lap_dfs.append(per_lap)

# Combine all per-lap aggregates
combined = pd.concat(per_lap_dfs, ignore_index=True)
combined = combined.sort_values(LAP_KEYS).reset_index(drop=True)

# Outliers are judged per session across both cars, so this runs on the combined table
combined = zscore_outlier_filter(combined, ZSCORE_COLS, "all files")

# lap_duration was only needed for coverage; it is the dependent variable, so keep it out
combined = combined.drop(columns=["lap_duration"])

laps_per_driver = combined.groupby(["session_key", "driver_number"]).size()
log(f"\nLaps per session/driver:\n{laps_per_driver.to_string()}")

# Save the cleaned samples of the laps that were kept
kept_laps = combined[LAP_KEYS]
for file, df in samples_by_file.items():
    df = df.merge(kept_laps, on=LAP_KEYS, how="inner")
    df = df.drop(columns=["date_start", "lap_duration", "lap_end"])

    cleaned_name = f"{SCRIPT_NAME}_cleaned_{file.replace('.csv', '')}.csv"
    df.to_csv(os.path.join(RESULTS_DIR, cleaned_name), index=False)
    generated_files.append(cleaned_name)
    log(f"  → Saved cleaned file: {cleaned_name} ({len(df)} samples)")

# Unscaled copy, in physical units
unscaled_name = f"{SCRIPT_NAME}_per_lap_unscaled.csv"
combined.to_csv(f"{RESULTS_DIR}/{unscaled_name}", index=False)
generated_files.append(unscaled_name)
log(f"\nSaved unscaled per-lap dataset: {unscaled_name}")

# Normalization
combined = scale_per_session(combined, SCALE_COLS, "all files")

scaling_summary = pd.DataFrame({"scaled_columns": SCALE_COLS})
scaling_name = f"{SCRIPT_NAME}_scaling_summary.csv"
scaling_summary.to_csv(f"{RESULTS_DIR}/{scaling_name}", index=False)
generated_files.append(scaling_name)

combined_name = f"{SCRIPT_NAME}_per_lap.csv"
combined.to_csv(f"{RESULTS_DIR}/{combined_name}", index=False)
generated_files.append(combined_name)
log(f"Saved per-lap dataset: {combined_name} ({len(combined)} laps)")

# Hand-off (not consumed by feature_engineer.py yet, see diagnostic.md #16)
handoff_path = data_dir / "cardata_per_lap_clean.csv"
combined.to_csv(handoff_path, index=False)
log(f"Saved hand-off dataset: {handoff_path}")

# Save row-removal summary
summary_df = pd.DataFrame(removed_rows_summary, columns=["file", "stage", "rows_removed"])
summary_name = f"{SCRIPT_NAME}_removed_rows_summary.csv"
summary_df.to_csv(f"{RESULTS_DIR}/{summary_name}", index=False)
generated_files.append(summary_name)

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
