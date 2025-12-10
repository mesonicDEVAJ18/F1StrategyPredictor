import pandas as pd
import os
from datetime import datetime

# ============================================================
# SETUP SCRIPT-SPECIFIC RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "process_stints"
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
# INPUT FILES
# ============================================================
data_dir = "../data"
stints_files = ["stints_p1.csv", "stints_p2.csv", "stints_p3.csv"]

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
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)
    fname = file.replace(".csv", "")

    log(f"\nLoaded {file} with {len(df)} rows")

    # Drop metadata columns
    df = df.drop(columns=[c for c in ["meeting_key", "session_key"] if c in df.columns], errors="ignore")

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
combined_name = f"{SCRIPT_NAME}_combined_cleaned.csv"
combined_path = f"{RESULTS_DIR}/{combined_name}"
combined.to_csv(combined_path, index=False)
generated_files.append(combined_name)

log(f"\nSaved combined stints dataset: {combined_name}")

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
