import sys
import pandas as pd
import os
from datetime import datetime

from env_config import env_str, env_list
from feature_config import COMPOUNDS

# ============================================================
# SETUP SCRIPT-SPECIFIC RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "process_stints"
RESULTS_DIR = f"{env_str('RESULTS_ROOT')}/{SCRIPT_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)

log_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_log.txt"
log_file = open(log_path, "w", encoding="utf-8", errors="replace")

generated_files = []

def log(msg: str):
    # Replace characters the console/redirect encoding can't show (e.g. cp1252 when output is
    # redirected on Windows); the log file is UTF-8 and keeps the original text
    enc = sys.stdout.encoding or "utf-8"
    safe_msg = msg.encode(enc, errors="replace").decode(enc, "replace")
    print(safe_msg)
    log_file.write(msg + "\n")

# ============================================================
# INPUT FILES
# ============================================================
data_dir = env_str("DATA_DIR")
stints_dir = os.path.join(data_dir, env_str("STINTS_DIR"))
processed_dir = os.path.join(data_dir, env_str("PROCESSED_DIR"))
stints_files = env_list("STINTS_FILES")

cleaned_dfs = []
removed_rows_summary = []
encoding_summary = []


# ============================================================
# HELPER: Logical consistency checks
# ============================================================
def apply_logical_filters(df, fname):
    before = len(df)

    df = df[df["lap_start"] <= df["lap_end"]]
    df = df[df["tyre_age_at_start"] >= 0]
    df = df[df["compound"].notna()]

    removed = before - len(df)
    log(f"[{fname}] Removed {removed} logically invalid stint rows")
    removed_rows_summary.append([fname, "logical_checks", removed])

    # Dry compounds only: an INTERMEDIATE, WET or UNKNOWN stint has no SOFT/MEDIUM/HARD one-hot, so it
    # would be read as SOFT downstream (compound_of / idxmax). Its laps then match no stint and are dropped
    # in feature_engineer.py
    before = len(df)
    non_dry = df.loc[~df["compound"].isin(COMPOUNDS), "compound"].value_counts().to_dict()
    df = df[df["compound"].isin(COMPOUNDS)]
    removed = before - len(df)
    log(f"[{fname}] Removed {removed} non-dry stint rows" + (f" {non_dry}" if non_dry else ""))
    removed_rows_summary.append([fname, "non_dry_compound", removed])

    return df


# ============================================================
# HELPER: One-Hot Encoding for compound
# ============================================================
def encode_stints(df, fname):
    before_cols = df.columns.tolist()

    if "compound" in df.columns:
        df = pd.get_dummies(df, columns=["compound"], prefix="compound", dtype=int)
        encoding_summary.append([fname, "compound", "one_hot"])
        log(f"[{fname}] Applied OHE for tyre compound")

    return df


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================
for file in stints_files:
    file_path = os.path.join(stints_dir, file)
    df = pd.read_csv(file_path)
    fname = file.replace(".csv", "")

    log(f"\nLoaded {file} with {len(df)} rows")

    # Drop metadata columns (session_key is kept: stint_number and lap ranges restart
    # every session, so feature_engineer.py needs it to assign laps to stints)
    df = df.drop(columns=[c for c in ["meeting_key"] if c in df.columns], errors="ignore")

    # 1. Logical outlier removal
    df = apply_logical_filters(df, fname)

    # 2. Encoding
    df = encode_stints(df, fname)

    # Save cleaned file
    cleaned_name = f"{SCRIPT_NAME}_cleaned_{fname}.csv"
    cleaned_path = f"{RESULTS_DIR}/{cleaned_name}"
    df.to_csv(cleaned_path, index=False)

    log(f"  → Saved cleaned stint file: {cleaned_name}")
    generated_files.append(cleaned_name)

    cleaned_dfs.append(df)

# ============================================================
# COMBINE ALL CLEANED STINTS
# ============================================================
combined = pd.concat(cleaned_dfs, ignore_index=True)
# A session without a compound has no column for it, so the concat leaves NaN there: fill with 0
# (and add a compound no session ran), so every stint has a complete one-hot
for c in COMPOUNDS:
    if f"compound_{c}" not in combined.columns:
        combined[f"compound_{c}"] = 0
compound_cols = [c for c in combined.columns if c.startswith("compound_")]
filled = int(combined[compound_cols].isna().sum().sum())
combined[compound_cols] = combined[compound_cols].fillna(0).astype(int)
log(f"\nCompound one-hot cells filled with 0 (compound not run in that session): {filled}")
combined_name = f"{SCRIPT_NAME}_combined_cleaned.csv"
combined_path = f"{RESULTS_DIR}/{combined_name}"
combined.to_csv(combined_path, index=False)
generated_files.append(combined_name)

log(f"\nSaved combined stints dataset: {combined_name}")

# Hand-off for feature_engineer.py
handoff_path = os.path.join(processed_dir, env_str("STINTS_CLEAN_FILE"))
combined.to_csv(handoff_path, index=False)
log(f"Saved hand-off dataset: {handoff_path}")

# ============================================================
# SAVE SUMMARY TABLES
# ============================================================

# Removed-row summary
removed_df = pd.DataFrame(removed_rows_summary, columns=["file", "stage", "rows_removed"])
removed_name = f"{SCRIPT_NAME}_removed_rows_summary.csv"
removed_df.to_csv(f"{RESULTS_DIR}/{removed_name}", index=False)
generated_files.append(removed_name)

# OHE summary
encode_df = pd.DataFrame(encoding_summary, columns=["file", "column", "action"])
encode_name = f"{SCRIPT_NAME}_encoding_summary.csv"
encode_df.to_csv(f"{RESULTS_DIR}/{encode_name}", index=False)
generated_files.append(encode_name)

# ============================================================
# MANIFEST FILE
# ============================================================
manifest_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt"
with open(manifest_path, "w") as m:
    for item in generated_files:
        m.write(item + "\n")
generated_files.append(f"{SCRIPT_NAME}_manifest.txt")

# ============================================================
# FINAL LOG OUTPUT
# ============================================================
log("\nPROCESS COMPLETE. Files generated:")
for f in generated_files:
    log(f"  - {f}")

log_file.close()
