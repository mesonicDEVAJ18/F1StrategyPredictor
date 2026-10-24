import pandas as pd
import os
from pathlib import Path

from feature_config import CARDATA_FEATURE_COLUMNS
from env_config import env_str, env_int, env_float, env_list


# ============================================================
# SETUP SCRIPT-SPECIFIC RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "process_cardata"
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
# INPUT FILE CONFIG
# ============================================================
data_dir = Path(env_str("DATA_DIR"))
cardata_dir = data_dir / env_str("CARDATA_DIR")
laps_dir = data_dir / env_str("LAPS_DIR")
processed_dir = data_dir / env_str("PROCESSED_DIR")
cardata_glob = env_str("CARDATA_GLOB")
cardata_files = sorted(cardata_dir.glob(cardata_glob))
laps_files = sorted(laps_dir.glob(env_str("LAPS_GLOB")))
clean_laps_file = processed_dir / env_str("LAPS_CLEAN_FILE")

if not cardata_files:
    raise FileNotFoundError(f"No {cardata_glob} files found at {cardata_dir}")

# Per-file outputs are named by driver, which changes between weekends,
# so a previous weekend's files are cleared first
for stale in Path(RESULTS_DIR).glob(f"{SCRIPT_NAME}_cleaned_{cardata_glob}"):
    stale.unlink()
if not clean_laps_file.exists():
    raise FileNotFoundError(f"{clean_laps_file.name} not found at {processed_dir}, run process_laps.py first")

removed_rows_summary = []

# brake/throttle read this when there is no valid input (car stationary / in the garage)
NO_INPUT_VALUE = env_int("CARDATA_NO_INPUT_VALUE")
# drs codes that mean the flap is open
DRS_OPEN_CODES = env_list("CARDATA_DRS_OPEN_CODES", cast=int)
# laps whose sample count is below this fraction of the expected count are dropped
MIN_COVERAGE = env_float("CARDATA_MIN_COVERAGE")
# throttle (%) at or above this counts as full throttle
FULL_THROTTLE_MIN = env_float("CARDATA_FULL_THROTTLE_MIN")
# per-session |z| above this on any aggregate drops the lap
ZSCORE_THRESHOLD = env_float("CARDATA_ZSCORE_THRESHOLD")
# readings above these are physically impossible
MAX_SPEED_KMH = env_float("CARDATA_MAX_SPEED_KMH")
MAX_RPM = env_float("CARDATA_MAX_RPM")
MAX_GEAR = env_int("CARDATA_MAX_GEAR")

LAP_KEYS = ["session_key", "driver_number", "lap_number"]

# Per-lap aggregates checked for outliers. drs_open_pct is left out: it is mostly 0,
# so any lap that uses DRS would look like an outlier.
ZSCORE_COLS = [c for c in CARDATA_FEATURE_COLUMNS if c != "drs_open_pct"]


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

    # No-input samples (brake/throttle = NO_INPUT_VALUE)
    before = len(df)
    df = df[(df["brake"] != NO_INPUT_VALUE) & (df["throttle"] != NO_INPUT_VALUE)]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} no-input samples (brake/throttle = {NO_INPUT_VALUE})")
    removed_rows_summary.append([fname, "no_input", removed])

    # Physically impossible readings
    before = len(df)
    df = df[
        df["speed"].between(0, MAX_SPEED_KMH)
        & df["rpm"].between(0, MAX_RPM)
        & df["n_gear"].between(0, MAX_GEAR)
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

    The raw files are used so every lap has a window, including the laps process_laps.py
    drops (pit-out, invalid, outliers); samples are then matched to the lap they were driven in.
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
    df["full_throttle"] = df["throttle"] >= FULL_THROTTLE_MIN
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
    """Drop laps with |z| > ZSCORE_THRESHOLD in any aggregate, z computed per session."""
    keep = pd.Series(True, index=per_lap.index)
    for col in cols:
        grouped = per_lap.groupby("session_key")[col]
        std = grouped.transform("std")
        z = (per_lap[col] - grouped.transform("mean")) / std
        # Zero-variance columns are skipped
        keep &= ~((std > 0) & (z.abs() > ZSCORE_THRESHOLD))

    before = len(per_lap)
    per_lap = per_lap[keep]
    removed = before - len(per_lap)
    log(f"[{fname}] Removed {removed} laps with |z| > {ZSCORE_THRESHOLD:g} (per session)")
    removed_rows_summary.append([fname, "zscore_outlier", removed])

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

# Not scaled: the aggregates are model inputs and stay in physical units like the laps,
# so the simulator's median baseline is meaningful
combined_name = f"{SCRIPT_NAME}_per_lap.csv"
combined.to_csv(f"{RESULTS_DIR}/{combined_name}", index=False)
generated_files.append(combined_name)
log(f"\nSaved per-lap dataset: {combined_name} ({len(combined)} laps)")

# Hand-off for feature_engineer.py (merged on lap t as model inputs)
handoff_path = processed_dir / env_str("CARDATA_CLEAN_FILE")
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
