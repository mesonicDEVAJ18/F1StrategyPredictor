import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime

# ============================================================
# SETUP SCRIPT-SPECIFIC RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "feature_engineer"
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
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR.parent / "data"

laps_file = data_dir / "laps_combined_clean.csv"
stints_file = data_dir / "stints_combined_clean.csv"
weather_file = data_dir / "cleaned_weather.csv"

final_output_file = f"{RESULTS_DIR}/{SCRIPT_NAME}_final_engineered_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================
log("Loading preprocessed data...")

laps = pd.read_csv(laps_file)
stints = pd.read_csv(stints_file)
weather = pd.read_csv(weather_file)

log(f"  → Laps:   {len(laps)} rows")
log(f"  → Stints: {len(stints)} rows")
log(f"  → Weather:{len(weather)} rows")

laps_initial_name = f"{SCRIPT_NAME}_laps_initial_snapshot.csv"
laps.to_csv(f"{RESULTS_DIR}/{laps_initial_name}", index=False)
generated_files.append(laps_initial_name)


# ============================================================
# ASSIGN GLOBAL STINT NUMBERS
# ============================================================
def assign_stint_numbers(laps, stints):
    laps = laps.copy()
    stints = stints.copy()

    # Ensure driver_number is numeric
    if "driver_number" in laps.columns:
        laps["driver_number"] = laps["driver_number"].astype(int)
    if "driver_number" in stints.columns:
        stints["driver_number"] = stints["driver_number"].astype(int)

    stints = stints.sort_values(["driver_number", "lap_start"]).copy()

    # Create global per-driver stint numbering
    stints["global_stint_number"] = (
        stints.groupby("driver_number").cumcount() + 1
    )

    # Assign to laps
    laps["stint_number"] = np.nan
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

    return laps, stints


log("\nAssigning stint numbers to laps...")
laps, stints = assign_stint_numbers(laps, stints)
missing = laps["stint_number"].isna().sum()
log(f"  → Missing stint numbers: {missing}")

assign_snapshot = f"{SCRIPT_NAME}_laps_with_stints.csv"
laps.to_csv(f"{RESULTS_DIR}/{assign_snapshot}", index=False)
generated_files.append(assign_snapshot)


# ============================================================
# MERGE STINT METADATA INTO LAPS
# ============================================================
log("\nMerging stints into laps...")

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

merge_stints_snapshot = f"{SCRIPT_NAME}_laps_after_stint_merge.csv"
laps.to_csv(f"{RESULTS_DIR}/{merge_stints_snapshot}", index=False)
generated_files.append(merge_stints_snapshot)

log("  → Stints merged successfully")


# ============================================================
# MERGE WEATHER WITH PSEUDO-TIME ALIGNMENT
# ============================================================
log("\nMerging weather using pseudo-time...")

laps = laps.sort_values("lap_number").reset_index(drop=True)
laps["pseudo_time"] = laps["lap_number"]

weather = weather.reset_index(drop=True)
weather["pseudo_time"] = np.linspace(
    laps["pseudo_time"].min(),
    laps["pseudo_time"].max(),
    len(weather)
)

laps["pseudo_time"] = laps["pseudo_time"].astype(float)
weather["pseudo_time"] = weather["pseudo_time"].astype(float)

laps = laps.sort_values("pseudo_time").reset_index(drop=True)
weather = weather.sort_values("pseudo_time").reset_index(drop=True)

laps = pd.merge_asof(
    laps.sort_values("pseudo_time"),
    weather.sort_values("pseudo_time"),
    on="pseudo_time",
    direction="nearest"
)

merge_weather_snapshot = f"{SCRIPT_NAME}_laps_after_weather_merge.csv"
laps.to_csv(f"{RESULTS_DIR}/{merge_weather_snapshot}", index=False)
generated_files.append(merge_weather_snapshot)

log("  → Weather merged successfully")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
log("\nEngineering features...")

# laps_since_stint_start
laps["laps_since_stint_start"] = laps["lap_number"] - laps["lap_start"] + 1
laps["laps_since_stint_start"] = laps["laps_since_stint_start"].clip(lower=1)

# Leave-one-out stint average pace
grp = laps.groupby(["driver_number", "stint_number"])
stint_sum = grp["lap_duration"].transform("sum")
stint_count = grp["lap_duration"].transform("count")

laps["stint_avg_pace"] = (stint_sum - laps["lap_duration"]) / (stint_count - 1)
laps.loc[stint_count <= 1, "stint_avg_pace"] = np.nan

# fuel_corrected_pace
fuel_effect_k = 0.035
laps["laps_remaining"] = laps["lap_end"] - laps["lap_number"]
laps["fuel_corrected_pace"] = laps["lap_duration"] + fuel_effect_k * laps["laps_remaining"]

feat_snapshot = f"{SCRIPT_NAME}_feature_engineered_snapshot.csv"
laps.to_csv(f"{RESULTS_DIR}/{feat_snapshot}", index=False)
generated_files.append(feat_snapshot)


# ============================================================
# FINAL CLEANUP & SAVE FINAL DATASET
# ============================================================
log("\nFinal cleanup...")

drops = ["driver_number", "date", "date_start", "lap_duration",
         "pseudo_time", "laps_remaining"]

laps = laps.drop(columns=[c for c in drops if c in laps.columns], errors="ignore")

laps.to_csv(final_output_file, index=False)
generated_files.append(final_output_file.split("/")[-1])

log(f"\n✅ Final engineered dataset saved as {final_output_file}")


# ============================================================
# SAVE MANIFEST
# ============================================================
manifest_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt"
with open(manifest_path, "w") as m:
    for f in generated_files:
        m.write(f + "\n")
generated_files.append(f"{SCRIPT_NAME}_manifest.txt")

log("\nFILES GENERATED:")
for f in generated_files:
    log(f"  - {f}")

log_file.close()