# Model input order (WHAT THE MODEL EXPECTS)
MODEL_FEATURE_COLUMNS = [
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "i1_speed",
    "i2_speed",
    "st_speed",
    "segments_sector_1",
    "segments_sector_2",
    "segments_sector_3",
    "tyre_age_at_start",
    "compound_HARD",
    "compound_MEDIUM",
    "compound_SOFT",
    "air_temperature",
    "humidity",
    "pressure",
    "rainfall",
    "track_temperature",
    "wind_direction",
    "wind_speed",
    "laps_since_stint_start",
    "stint_avg_pace",
]

# FULL dataset row structure (used by simulator)
SIM_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS + ["fuel_corrected_pace"]

# One stint of one driver (stint_number is the global per-driver number,
# so it also pins the session)
TARGET_GROUP_COLUMNS = ["driver_number", "stint_number"]


def build_next_lap_target(df):
    """Add target_next_fcp = the same driver's fuel_corrected_pace on lap t+1 of the same stint.

    Rows without a next lap (end of stint, or a gap left by lap cleaning) are dropped.
    Shared by train_model.py and validate_strategy.py so the target stays identical.
    """
    df = df.sort_values(TARGET_GROUP_COLUMNS + ["lap_number"]).reset_index(drop=True)
    grp = df.groupby(TARGET_GROUP_COLUMNS)
    next_lap = grp["lap_number"].shift(-1)
    df["target_next_fcp"] = grp["fuel_corrected_pace"].shift(-1)

    consecutive = next_lap == df["lap_number"] + 1
    df = df[consecutive & df["target_next_fcp"].notna()].reset_index(drop=True)
    return df