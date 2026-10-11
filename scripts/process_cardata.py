import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import ast


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
data_dir = "../data"
cardata_files = ["cardata_p1_1.csv", "cardata_p1_2.csv", "cardata_p2_1.csv", "cardata_p2_2.csv", "cardata_p3_1.csv", "cardata_p3_2.csv"]
cleaned_dfs = []

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
    df = df[(df["lap_duration"] > 30) & (df["lap_duration"] < 2 * median_lap)]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} extreme lap-time outliers")
    removed_rows_summary.append([fname, "unrealistic_laptime", removed])

    # Unrealistic speeds
    speed_cols = ["i1_speed", "i2_speed", "st_speed"]
    if all(col in df.columns for col in speed_cols):
        before = len(df)
        df = df[(df[speed_cols] > 50).all(axis=1)]
        removed = before - len(df)
        log(f"[{fname}] Removed {removed} unrealistic-speed laps")
        removed_rows_summary.append([fname, "unrealistic_speed", removed])

    return df


def apply_iqr_residual_filter(df, fname):
    """IQR on lap_duration - rolling_median."""
    if "driver_number" not in df.columns:
        df["driver_number"] = 1  # fallback grouping

    df["rolling_median"] = df.groupby("driver_number")["lap_duration"].transform(
        lambda x: x.rolling(5, center=True, min_periods=1).median()
    )
    df["residual"] = df["lap_duration"] - df["rolling_median"]

    Q1 = df["residual"].quantile(0.25)
    Q3 = df["residual"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR

    before = len(df)
    df = df[(df["residual"] >= lower) & (df["residual"] <= upper)]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} IQR-residual outliers")
    removed_rows_summary.append([fname, "iqr_residual", removed])

    return df.drop(columns=["rolling_median", "residual"], errors="ignore")


def scale_continuous_features(df, fname):
    """Standard-scale numeric telemetry metrics."""
    scale_cols = [
        "lap_duration",
        "duration_sector_1", "duration_sector_2", "duration_sector_3",
        "i1_speed", "i2_speed", "st_speed"
    ]

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    log(f"[{fname}] Scaled continuous features: {scale_cols}")
    return df


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

for file in cardata_files:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)
    fname = file.replace(".csv", "")
    log(f"\nLoaded {file} with {len(df)} rows")

    # Drop metadata columns
    df = df.drop(columns=[c for c in ["meeting_key", "session_key"] if c in df.columns], errors="ignore")

    # Apply cleaning pipeline
    df = apply_domain_filters(df, file)
    df = apply_iqr_residual_filter(df, file)
    df = scale_continuous_features(df, file)

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