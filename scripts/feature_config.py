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