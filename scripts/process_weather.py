import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# --- Define file paths ---
data_dir = Path("../data")
input_file = data_dir / "weather.csv"
output_file = data_dir / "cleaned_weather.csv"

# --- Read the CSV ---
weather = pd.read_csv(input_file)
print(f"Loaded weather.csv with {len(weather)} rows")

# --- Drop unnecessary columns if they exist (your logic) ---
weather = weather.drop(columns=[c for c in ["meeting_key", "session_key"] if c in weather.columns], errors="ignore")


# ======================================================
# WEATHER PREPROCESSING PIPELINE
# ======================================================

# ------------------------------------------------------
# A. Outlier Removal using Z-score
# ------------------------------------------------------
def zscore_outlier_filter(df, cols):
    """Remove |z| > 3 outliers for specified continuous weather columns."""
    before = len(df)

    for col in cols:
        mean = df[col].mean()
        std = df[col].std()

        if std == 0:  # avoid div-by-zero
            continue

        z = (df[col] - mean) / std
        df = df[z.abs() <= 3]

    print(f"  → Removed {before - len(df)} weather outliers (Z-score)")
    return df


zscore_cols = [
    "air_temperature",
    "track_temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "rainfall"
]

# Only keep columns that actually exist
zscore_cols = [c for c in zscore_cols if c in weather.columns]

weather = zscore_outlier_filter(weather, zscore_cols)


# ------------------------------------------------------
# B. Normalization (StandardScaler)
# ------------------------------------------------------
def scale_weather_features(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    print("  → Scaled continuous weather features")
    return df


# wind_direction is not scaled (it's angular)
scale_cols = [c for c in zscore_cols if c != "wind_direction" and c in weather.columns]

weather = scale_weather_features(weather, scale_cols)


# ------------------------------------------------------
# C. Final cleanup (no encoding needed)
# ------------------------------------------------------
# wind_direction stays raw
print("  → wind_direction kept unscaled (cyclical variable)")


# ======================================================
# SAVE CLEANED WEATHER DATA
# ======================================================
weather.to_csv(output_file, index=False)
print(f"✅ Cleaned weather data saved as {output_file}")