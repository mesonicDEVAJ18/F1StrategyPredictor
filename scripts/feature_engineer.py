import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime

from feature_config import SESSION_COLUMNS, CARDATA_FEATURE_COLUMNS, FUEL_EFFECT_K
from env_config import env_str

# ============================================================
# SETUP SCRIPT-SPECIFIC RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "feature_engineer"
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
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
processed_dir = (BASE_DIR / env_str("DATA_DIR") / env_str("PROCESSED_DIR")).resolve()

laps_file = processed_dir / env_str("LAPS_CLEAN_FILE")
stints_file = processed_dir / env_str("STINTS_CLEAN_FILE")
weather_file = processed_dir / env_str("WEATHER_CLEAN_FILE")
cardata_file = processed_dir / env_str("CARDATA_CLEAN_FILE")

# Max gap between a lap's start and its nearest weather sample
WEATHER_MERGE_TOLERANCE = env_str("WEATHER_MERGE_TOLERANCE")

final_output_file = f"{RESULTS_DIR}/{SCRIPT_NAME}_final_engineered_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================
log("Loading preprocessed data...")

laps = pd.read_csv(laps_file)
stints = pd.read_csv(stints_file)
weather = pd.read_csv(weather_file)
cardata = pd.read_csv(cardata_file)

log(f"  → Laps:   {len(laps)} rows")
log(f"  → Stints: {len(stints)} rows")
log(f"  → Weather:{len(weather)} rows")
log(f"  → Car data:{len(cardata)} laps")

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

    # Session keys increase FP1 -> FP3, so this numbers stints chronologically
    stints = stints.sort_values(["driver_number", "session_key", "stint_number"]).copy()

    # Create global per-driver stint numbering
    stints["global_stint_number"] = (
        stints.groupby("driver_number").cumcount() + 1
    )

    # Assign to laps (lap numbers restart every session, so match the session too)
    laps["stint_number"] = np.nan
    for _, row in stints.iterrows():
        session = row["session_key"]
        driver = row["driver_number"]
        global_stint_no = row["global_stint_number"]
        start = row["lap_start"]
        end = row["lap_end"]

        mask = (
            (laps["session_key"] == session) &
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
    "global_stint_number",
    "lap_start",
    "lap_end",
    "tyre_age_at_start"
] + [col for col in stints.columns if col.startswith("compound_")]

# Laps carry the global number, which is unique per driver, so merge on that
# (the stints' own stint_number restarts every session)
rows_before = len(laps)
laps = laps.merge(
    stints[stints_merge_cols].rename(columns={"global_stint_number": "stint_number"}),
    on=["driver_number", "stint_number"],
    how="left",
    validate="many_to_one"
)

merge_stints_snapshot = f"{SCRIPT_NAME}_laps_after_stint_merge.csv"
laps.to_csv(f"{RESULTS_DIR}/{merge_stints_snapshot}", index=False)
generated_files.append(merge_stints_snapshot)

log(f"  → Rows before/after merge: {rows_before} / {len(laps)}")
log(f"  → Laps without stint metadata: {laps['lap_start'].isna().sum()}")
log("  → Stints merged successfully")


# ============================================================
# MERGE WEATHER ON TIMESTAMPS
# ============================================================
log("\nMerging weather on timestamps (lap date_start ~ weather date)...")

laps["date_start"] = pd.to_datetime(laps["date_start"], utc=True, format="ISO8601")
weather["date"] = pd.to_datetime(weather["date"], utc=True, format="ISO8601")

# Sessions are hours apart and weather is sampled ~1/min, so the nearest
# sample within the tolerance always comes from the lap's own session.
laps = pd.merge_asof(
    laps.sort_values("date_start"),
    weather.sort_values("date"),
    left_on="date_start",
    right_on="date",
    direction="nearest",
    tolerance=pd.Timedelta(WEATHER_MERGE_TOLERANCE)
)

unmatched = laps["date"].isna().sum()
log(f"  → Laps without weather within {WEATHER_MERGE_TOLERANCE}: {unmatched}")

merge_weather_snapshot = f"{SCRIPT_NAME}_laps_after_weather_merge.csv"
laps.to_csv(f"{RESULTS_DIR}/{merge_weather_snapshot}", index=False)
generated_files.append(merge_weather_snapshot)

log("  → Weather merged successfully")


# ============================================================
# MERGE CAR DATA ON LAP KEYS
# ============================================================
log("\nMerging per-lap car data (session_key, driver_number, lap_number)...")

# Each row gets its own lap t's telemetry; the target is lap t+1, so nothing leaks.
# Laps whose telemetry was dropped (low coverage, outliers) stay NaN and are filled
# with the training medians in train_model.py
lap_keys = ["session_key", "driver_number", "lap_number"]
rows_before = len(laps)
laps = laps.merge(
    cardata[lap_keys + CARDATA_FEATURE_COLUMNS],
    on=lap_keys,
    how="left",
    validate="one_to_one"
)

no_cardata = laps[CARDATA_FEATURE_COLUMNS[0]].isna().sum()
log(f"  → Rows before/after merge: {rows_before} / {len(laps)}")
log(f"  → Laps without car data: {no_cardata} of {len(laps)}")

merge_cardata_snapshot = f"{SCRIPT_NAME}_laps_after_cardata_merge.csv"
laps.to_csv(f"{RESULTS_DIR}/{merge_cardata_snapshot}", index=False)
generated_files.append(merge_cardata_snapshot)

log("  → Car data merged successfully")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
log("\nEngineering features...")

# Practice session one-hot (model input; session_key stays an identifier)
session_dummies = (
    pd.get_dummies(laps["session_name"], prefix="session", dtype=int)
    .reindex(columns=SESSION_COLUMNS, fill_value=0)
)
laps = pd.concat([laps.drop(columns=["session_name"]), session_dummies], axis=1)
log(f"  → Session one-hot rows: {session_dummies.sum().to_dict()}")

# laps_since_stint_start
laps["laps_since_stint_start"] = laps["lap_number"] - laps["lap_start"] + 1
laps["laps_since_stint_start"] = laps["laps_since_stint_start"].clip(lower=1)

# Real tyre age: laps on the set including lap t (1 on a new set's first lap), so a stint
# on a used set is told apart from one on a new set
laps["tyre_age"] = laps["tyre_age_at_start"] + laps["laps_since_stint_start"]
log(f"  → tyre_age range: {laps['tyre_age'].min():.0f}-{laps['tyre_age'].max():.0f} laps")

# Causal stint average pace (seconds): mean of the stint's earlier laps only, so the
# target lap t+1 can't leak in. NaN on a stint's first lap; training
# fills it with the median, which is also the simulator's seed value.
laps["stint_avg_pace"] = (
    laps.sort_values(["driver_number", "stint_number", "lap_number"])
    .groupby(["driver_number", "stint_number"])["lap_duration"]
    .transform(lambda s: s.shift(1).expanding().mean())
)
log(f"  → stint_avg_pace NaN (first lap of a stint): {laps['stint_avg_pace'].isna().sum()}")

# fuel_corrected_pace (seconds; lap_duration is unscaled, k is s/lap): the fuel effect is
# removed. Practice fuel loads aren't recorded, so the laps left in the stint, counting lap t,
# stand in for the fuel on board at the start of lap t (assumption). The
# simulator's race fuel term uses the same convention (laps left in the race, counting the lap).
laps["laps_remaining"] = laps["lap_end"] - laps["lap_number"] + 1
laps["fuel_corrected_pace"] = laps["lap_duration"] - FUEL_EFFECT_K * laps["laps_remaining"]

feat_snapshot = f"{SCRIPT_NAME}_feature_engineered_snapshot.csv"
laps.to_csv(f"{RESULTS_DIR}/{feat_snapshot}", index=False)
generated_files.append(feat_snapshot)


# ============================================================
# FINAL CLEANUP & SAVE FINAL DATASET
# ============================================================
log("\nFinal cleanup...")

# session_key and driver_number are kept as identifiers (next-lap target grouping,
# car-data merge), not model inputs: the models only read MODEL_FEATURE_COLUMNS
drops = ["date", "date_start", "lap_duration", "laps_remaining"]

laps = laps.drop(columns=[c for c in drops if c in laps.columns], errors="ignore")

laps.to_csv(final_output_file, index=False)
generated_files.append(final_output_file.split("/")[-1])

log(f"\n✅ Final engineered dataset saved as {final_output_file}")

# Hand-off for train_model.py / validate_strategy.py
handoff_path = processed_dir / env_str("ENGINEERED_FILE")
laps.to_csv(handoff_path, index=False)
log(f"Saved hand-off dataset: {handoff_path}")


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