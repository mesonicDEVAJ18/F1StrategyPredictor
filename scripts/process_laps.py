import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import ast

# --- Input and output directory ---
data_dir = "../data"

# List of laps files to process
laps_files = ["laps_p1.csv", "laps_p2.csv", "laps_p3.csv"]

# List to store cleaned DataFrames
cleaned_dfs = []


# =====================================================
# HELPER FUNCTIONS FOR PREPROCESSING
# =====================================================

def apply_domain_filters(df):
    """Remove impossible or invalid lap/sector/speed values."""
    # Remove missing/invalid sector times
    sector_cols = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
    before = len(df)
    df = df.dropna(subset=sector_cols)
    df = df[(df[sector_cols] > 0).all(axis=1)]
    print(f"  → Removed {before - len(df)} invalid-sector laps")

    # Remove unrealistic lap times (telemetry glitches)
    before = len(df)
    median_lap = df["lap_duration"].median()
    df = df[(df["lap_duration"] > 30) & (df["lap_duration"] < 2 * median_lap)]
    print(f"  → Removed {before - len(df)} extreme lap-time outliers")

    # Remove unrealistic speeds
    speed_cols = ["i1_speed", "i2_speed", "st_speed"]
    if all(col in df.columns for col in speed_cols):
        before = len(df)
        df = df[(df[speed_cols] > 50).all(axis=1)]
        print(f"  → Removed {before - len(df)} unrealistic-speed laps")

    return df


def apply_iqr_residual_filter(df):
    """Apply IQR filtering on residuals (lap - rolling median)."""

    # Rolling median every 5 laps per driver
    df["rolling_median"] = df.groupby("driver_number")["lap_duration"].transform(
        lambda x: x.rolling(5, center=True, min_periods=1).median()
    )

    # Residual between lap and rolling median
    df["residual"] = df["lap_duration"] - df["rolling_median"]

    # Compute IQR boundaries
    Q1 = df["residual"].quantile(0.25)
    Q3 = df["residual"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR

    before = len(df)
    df = df[(df["residual"] >= lower) & (df["residual"] <= upper)]
    print(f"  → Removed {before - len(df)} IQR-residual outliers")

    # Remove temporary columns
    df = df.drop(columns=["rolling_median", "residual"], errors="ignore")
    return df


def scale_continuous_features(df):
    """Standard-scale the physics-based continuous telemetry features."""
    scale_cols = [
        "lap_duration",
        "duration_sector_1", "duration_sector_2", "duration_sector_3",
        "i1_speed", "i2_speed", "st_speed"
    ]

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print("  → Scaled continuous telemetry features")

    return df


# =====================================================
# MAIN PROCESSING LOOP
# =====================================================

for file in laps_files:
    # Construct full file path
    file_path = os.path.join(data_dir, file)

    # Read the CSV
    df = pd.read_csv(file_path)
    print(f"\nLoaded {file} with {len(df)} rows")

    # -------------------------------------------------
    # CLEANING 
    # -------------------------------------------------

    LIST_COLS = [
        "segments_sector_1",
        "segments_sector_2",
        "segments_sector_3"
    ]

    for col in LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else x
            )
            print(f"  → Fixed list column: {col}")

    # Remove pit-out laps AND drop the column entirely
    if "is_pit_out_lap" in df.columns:
        original_len = len(df)
        df = df[~df["is_pit_out_lap"]]
        print(f"  → Removed {original_len - len(df)} pit-out laps")

        # Drop column so it never appears in outputs
        df = df.drop(columns=["is_pit_out_lap"])


    # Drop meeting_key and session_key if they exist
    df = df.drop(columns=[c for c in ["meeting_key", "session_key"] if c in df.columns], errors="ignore")

    # -------------------------------------------------
    # OUR PREPROCESSING PIPELINE
    # -------------------------------------------------

    # 1. Domain Outlier Filtering
    df = apply_domain_filters(df)

    # 2. IQR Residual Filtering
    df = apply_iqr_residual_filter(df)

    # 3. Scale continuous telemetry variables
    df = scale_continuous_features(df)

    # -------------------------------------------------
    # SAVE CLEANED FILES (unchanged from your code)
    # -------------------------------------------------

    base_name = os.path.splitext(file)[0]  # e.g. "laps_p1"
    cleaned_file = f"cleaned_{base_name}.csv"
    cleaned_path = os.path.join(data_dir, cleaned_file)

    df.to_csv(cleaned_path, index=False)
    print(f"  → Saved cleaned file as {cleaned_path}")

    # Add cleaned DataFrame to list for combination
    cleaned_dfs.append(df)

# --- Combine all cleaned DataFrames ---
combined = pd.concat(cleaned_dfs, ignore_index=True)

# Output combined file
combined_file = os.path.join(data_dir, "laps_combined_clean.csv")
combined.to_csv(combined_file, index=False)

print(f"\n✅ All cleaned files saved in {data_dir}")
print(f"✅ Combined dataset saved as {combined_file}")