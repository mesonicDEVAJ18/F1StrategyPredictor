import pandas as pd
import os

# --- Input and output directory ---
data_dir = "../data"

# List of stints files to process
stints_files = ["stints_p1.csv", "stints_p2.csv", "stints_p3.csv"]

# List to store cleaned DataFrames
cleaned_dfs = []


# =====================================================
# HELPER: Logical consistency checks (no statistical outliers)
# =====================================================
def apply_logical_filters(df):
    before = len(df)

    # lap_start <= lap_end
    df = df[df["lap_start"] <= df["lap_end"]]

    # tyre_age_at_start >= 0
    df = df[df["tyre_age_at_start"] >= 0]

    # compound should not be missing
    df = df[df["compound"].notna()]

    print(f"  → Removed {before - len(df)} logically invalid stint rows")
    return df


# =====================================================
# HELPER: Encoding (compound OHE, driver label-ready)
# =====================================================
def encode_stints(df):
    # One-Hot encode compound (Soft/Medium/Hard/Inter/Wet)
    if "compound" in df.columns:
        df = pd.get_dummies(df, columns=["compound"], prefix="compound", dtype=int)
        print("  → Applied One-Hot encoding for tyre compound")

    # driver_number remains numeric (XGBoost handles labels)
    return df


# =====================================================
# MAIN LOOP
# =====================================================
for file in stints_files:
    # Construct full file path
    file_path = os.path.join(data_dir, file)

    # Read the CSV
    df = pd.read_csv(file_path)
    print(f"\nLoaded {file} with {len(df)} rows")

    # -------------------------------------------------
    # YOUR ORIGINAL STEP — drop useless columns
    # -------------------------------------------------
    df = df.drop(columns=[c for c in ["meeting_key", "session_key"] if c in df.columns], errors="ignore")

    # -------------------------------------------------
    # APPLY STINTS PREPROCESSING PIPELINE
    # -------------------------------------------------

    # 1. Logical outlier removal
    df = apply_logical_filters(df)

    # 2. Encoding for compound and driver
    df = encode_stints(df)

    # -------------------------------------------------
    # SAVE CLEANED FILE — YOUR ORIGINAL LOGIC
    # -------------------------------------------------
    base_name = os.path.splitext(file)[0]  # e.g. "stints_p1"
    cleaned_file = f"cleaned_{base_name}.csv"
    cleaned_path = os.path.join(data_dir, cleaned_file)

    df.to_csv(cleaned_path, index=False)
    print(f"  → Saved cleaned file as {cleaned_path}")

    # Add cleaned DataFrame to list for combination
    cleaned_dfs.append(df)

# --- Combine all cleaned DataFrames ---
combined = pd.concat(cleaned_dfs, ignore_index=True)

# Output combined file
combined_file = os.path.join(data_dir, "stints_combined_clean.csv")
combined.to_csv(combined_file, index=False)

print(f"\n✅ All cleaned files saved in {data_dir}")
print(f"✅ Combined dataset saved as {combined_file}")