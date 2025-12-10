import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------
# CONFIG 
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR.parent / "data"

laps_file = data_dir / "laps_combined_clean.csv"
stints_file = data_dir / "stints_combined_clean.csv"
weather_file = data_dir / "cleaned_weather.csv"

output_file = data_dir / "final_engineered_dataset.csv"


# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
print("Loading preprocessed data...")

laps = pd.read_csv(laps_file)
stints = pd.read_csv(stints_file)
weather = pd.read_csv(weather_file)

print(f"  → Laps:   {len(laps)} rows")
print(f"  → Stints: {len(stints)} rows")
print(f"  → Weather:{len(weather)} rows")


# -------------------------------------------------------
# ASSIGN STINT NUMBER TO LAPS
# -------------------------------------------------------
def assign_stint_numbers(laps, stints):
    laps["stint_number"] = np.nan
    laps["driver_number"] = laps["driver_number"].astype(int)
    stints["driver_number"] = stints["driver_number"].astype(int)

    # ----------------------------------------------------
    # MAKE STINT NUMBER SESSION-INDEPENDENT (GLOBAL ORDER)
    # ----------------------------------------------------
    stints = stints.sort_values(["driver_number", "lap_start"]).copy()

    # Assign new global stint IDs per driver
    stints["global_stint_number"] = (
        stints.groupby("driver_number").cumcount() + 1
    )

    # Now assign global stint numbers into laps
    for _, row in stints.iterrows():
        driver = row["driver_number"]
        global_stint_no = row["global_stint_number"]
        start = row["lap_start"]
        end = row["lap_end"]

        mask = (
            (laps["driver_number"] == driver) &
            (laps["lap_number"] >= start) &
            (laps["lap_number"] <= end)
        )
        laps.loc[mask, "stint_number"] = global_stint_no

    return laps

print("\nAssigning stint_number to laps...")
laps = assign_stint_numbers(laps, stints)
print(f"  → Missing stint numbers: {laps['stint_number'].isna().sum()}")


# -------------------------------------------------------
# MERGE STINTS INTO LAPS
# -------------------------------------------------------
print("\nMerging stints with laps...")

stints_merge_cols = [
    "driver_number",
    "stint_number",
    "lap_start",
    "lap_end",
    "tyre_age_at_start"
] + [col for col in stints.columns if col.startswith("compound_")]

laps = laps.merge(
    stints[stints_merge_cols],
    on=["driver_number", "stint_number"],
    how="left"
)

print("  → Stints merged into laps")


# -------------------------------------------------------
# MERGE WEATHER USING PSEUDO TIME
# -------------------------------------------------------
print("\nMerging weather using pseudo-time alignment...")

# Create pseudo timeline for laps
laps = laps.sort_values("lap_number").reset_index(drop=True)
laps["pseudo_time"] = laps["lap_number"]

# Create evenly spaced pseudo timeline for weather
weather = weather.reset_index(drop=True)
weather["pseudo_time"] = np.linspace(
    laps["pseudo_time"].min(),
    laps["pseudo_time"].max(),
    len(weather)
)

# Ensure types match
laps["pseudo_time"] = laps["pseudo_time"].astype(float)
weather["pseudo_time"] = weather["pseudo_time"].astype(float)

# Merge nearest pseudo-time
laps = pd.merge_asof(
    laps.sort_values("pseudo_time"),
    weather.sort_values("pseudo_time"),
    on="pseudo_time",
    direction="nearest"
)

print("  → Weather merged successfully")


# -------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------
print("\nComputing engineered features...")

# laps_since_stint_start
laps["laps_since_stint_start"] = laps["lap_number"] - laps["lap_start"] + 1
laps["laps_since_stint_start"] = laps["laps_since_stint_start"].clip(lower=1)

# Leave-one-out stint average pace (safe)
grp = laps.groupby(["driver_number", "stint_number"])

stint_sum = grp["lap_duration"].transform("sum")
stint_count = grp["lap_duration"].transform("count")

# Leave-one-out average:
laps["stint_avg_pace"] = (stint_sum - laps["lap_duration"]) / (stint_count - 1)

# Handle stints with only 1 lap → NaN
laps.loc[stint_count <= 1, "stint_avg_pace"] = np.nan

# fuel_corrected_pace
fuel_effect_k = 0.035
laps["laps_remaining"] = laps["lap_end"] - laps["lap_number"]
laps["fuel_corrected_pace"] = laps["lap_duration"] + fuel_effect_k * laps["laps_remaining"]


# -------------------------------------------------------
# SAVE FINAL DATASET
# -------------------------------------------------------
laps = laps.drop(columns=["driver_number", "date","date_start","lap_duration", "pseudo_time", "laps_remaining", "laps_remaining"], errors="ignore") #Removed lap duaration and laps_remaining to avoid leakage
laps.to_csv(output_file, index=False)

print(f"\n✅ Final engineered dataset saved as {output_file}")
