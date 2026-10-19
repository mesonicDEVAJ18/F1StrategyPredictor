import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

from env_config import env_str, env_float

# ============================================================
# SETUP SCRIPT-SPECIFIC RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "process_weather"
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
# INPUT FILES
# ============================================================
data_dir = Path(env_str("DATA_DIR"))
input_file = data_dir / env_str("WEATHER_FILE")

# Weather rows with |z| above this are dropped
ZSCORE_THRESHOLD = env_float("WEATHER_ZSCORE_THRESHOLD")

if not input_file.exists():
    raise FileNotFoundError(f"{input_file.name} not found at {data_dir}")

weather = pd.read_csv(input_file)
log(f"Loaded {input_file.name} with {len(weather)} rows")


# ============================================================
# KEEP PRACTICE SESSIONS ONLY
# ============================================================
# weather.csv covers the whole meeting (practice, qualifying, race); keep only
# the sessions the lap data comes from so filtering/scaling reflect practice.
practice_sessions = pd.concat(
    pd.read_csv(f, usecols=["session_key"]) for f in sorted(data_dir.glob(env_str("LAPS_GLOB")))
)["session_key"].dropna().astype(int).unique()

before = len(weather)
weather = weather[weather["session_key"].isin(practice_sessions)].reset_index(drop=True)
log(f"Kept practice sessions {sorted(practice_sessions.tolist())}: {before} -> {len(weather)} rows")


# ============================================================
# DROP UNUSED COLUMNS
# ============================================================
weather = weather.drop(columns=[c for c in ["meeting_key", "session_key"] if c in weather.columns], errors="ignore")
log("Dropped meeting/session keys if present.")


# ============================================================
# A. OUTLIER REMOVAL USING Z-SCORE
# ============================================================
def zscore_outlier_filter(df, cols, fname):
    before = len(df)
    removal_count = 0

    for col in cols:
        mean = df[col].mean()
        std = df[col].std()

        if std == 0:  # skip to avoid division by zero
            continue

        z = (df[col] - mean) / std
        removed = (z.abs() > ZSCORE_THRESHOLD).sum()
        removal_count += removed
        df = df[z.abs() <= ZSCORE_THRESHOLD]

    log(f"[{fname}] Removed {removal_count} weather outliers (Z-score)")
    return df, removal_count


zscore_cols = [
    "air_temperature",
    "track_temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "rainfall"
]
zscore_cols = [c for c in zscore_cols if c in weather.columns]

weather, removed_outliers = zscore_outlier_filter(weather, zscore_cols, "weather.csv")

# Save summary
zscore_summary = pd.DataFrame([[removed_outliers]], columns=["total_zscore_removed"])
zscore_name = f"{SCRIPT_NAME}_zscore_summary.csv"
zscore_path = f"{RESULTS_DIR}/{zscore_name}"
zscore_summary.to_csv(zscore_path, index=False)
generated_files.append(zscore_name)


# ============================================================
# B. NORMALIZATION VIA STANDARD SCALER
# ============================================================
def scale_weather_features(df, cols, fname):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    log(f"[{fname}] Scaled continuous weather features: {cols}")
    return df, cols


scale_cols = [c for c in zscore_cols if c in weather.columns and c != "wind_direction"]

weather, scaled_cols = scale_weather_features(weather, scale_cols, "weather.csv")

# Save scaling summary
scaling_summary = pd.DataFrame({"scaled_columns": scaled_cols})
scaling_name = f"{SCRIPT_NAME}_scaling_summary.csv"
scaling_path = f"{RESULTS_DIR}/{scaling_name}"
scaling_summary.to_csv(scaling_path, index=False)
generated_files.append(scaling_name)

log(" → wind_direction kept unscaled (cyclical variable)")


# ============================================================
# SAVE CLEANED WEATHER DATA
# ============================================================
cleaned_name = f"{SCRIPT_NAME}_cleaned_weather.csv"
cleaned_path = f"{RESULTS_DIR}/{cleaned_name}"

weather.to_csv(cleaned_path, index=False)
generated_files.append(cleaned_name)

log(f"Saved cleaned weather file: {cleaned_name}")

# Hand-off for feature_engineer.py
handoff_path = data_dir / env_str("WEATHER_CLEAN_FILE")
weather.to_csv(handoff_path, index=False)
log(f"Saved hand-off dataset: {handoff_path}")


# ============================================================
# MANIFEST FILE
# ============================================================
manifest_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt"
with open(manifest_path, "w") as mf:
    for item in generated_files:
        mf.write(item + "\n")
generated_files.append(f"{SCRIPT_NAME}_manifest.txt")

log("\nPROCESS COMPLETE. Files generated:")
for f in generated_files:
    log(f"  - {f}")

log_file.close()