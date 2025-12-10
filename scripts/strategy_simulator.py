import pandas as pd
import numpy as np
import itertools
import json
import joblib
import os
import warnings
from datetime import datetime

from feature_config import MODEL_FEATURE_COLUMNS, SIM_FEATURE_COLUMNS

warnings.filterwarnings("ignore")

# ============================================================
# SETUP RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "strategy_simulator"
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


log("\n--- STRATEGY SIMULATOR START ---\n")


# ============================================================
# LOAD MODEL
# ============================================================
bundle = joblib.load("./results/train_model/train_model_xgb_model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]

log("Loaded trained XGBoost model.")


# ============================================================
# CONSTANTS
# ============================================================
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
PIT_LOSS = 22.0
FUEL_PENALTY_PER_LAP = 0.035
NUM_LAPS = 57


# ============================================================
# PRECOMPUTE LAP PREDICTIONS
# ============================================================
def precompute_lap_predictions(total_laps, base_features):
    """
    Creates predicted lap times for each compound:
        preds[c][lap] = predicted lap time
        prefix[c] = cumulative lap times

    Saves:
      - predictions CSV
      - prefix sums CSV
    """
    preds = {c: [] for c in COMPOUNDS}

    for c in COMPOUNDS:
        feats = base_features.copy()

        # Set one-hot compound
        for cc in COMPOUNDS:
            feats[f"compound_{cc}"] = 1 if cc == c else 0

        for lap in range(1, total_laps + 1):

            feats["laps_since_stint_start"] = lap
            feats["fuel_corrected_pace"] = 0.0   # dummy placeholder

            # Align vector
            vec = np.array([feats[f] for f in SIM_FEATURE_COLUMNS], dtype=float).reshape(1, -1)

            model_vec = vec[:, :len(FEATURES)]
            fcp = model.predict(model_vec)[0]

            fuel_load = total_laps - lap + 1
            lap_time = fcp + fuel_load * FUEL_PENALTY_PER_LAP
            preds[c].append(lap_time)

    prefix = {c: np.cumsum(preds[c]) for c in COMPOUNDS}

    # Save predictions
    pred_df = pd.DataFrame(preds)
    pred_file = f"{SCRIPT_NAME}_precomputed_predictions.csv"
    pred_df.to_csv(f"{RESULTS_DIR}/{pred_file}", index=False)
    generated_files.append(pred_file)

    # Save prefix sums
    prefix_df = pd.DataFrame(prefix)
    prefix_file = f"{SCRIPT_NAME}_prefix_sums.csv"
    prefix_df.to_csv(f"{RESULTS_DIR}/{prefix_file}", index=False)
    generated_files.append(prefix_file)

    log("Saved precomputed predictions and prefix sums.")

    return preds, prefix


# ============================================================
# COMPUTE STINT TIME
# ============================================================
def stint_time(prefix, compound, start_lap, end_lap):
    arr = prefix[compound]
    if start_lap == 1:
        return arr[end_lap - 1]
    return arr[end_lap - 1] - arr[start_lap - 2]


# ============================================================
# FAST STRATEGY SIMULATION
# ============================================================
def simulate_strategy_fast(prefix, total_laps, compounds, pit_laps):
    pit_laps = sorted(pit_laps)
    segments = []

    last = 1
    for p in pit_laps:
        segments.append((last, p - 1))
        last = p + 1
    segments.append((last, total_laps))

    total = 0
    for idx, (s, e) in enumerate(segments):
        comp = compounds[idx]
        total += stint_time(prefix, comp, s, e)
        if idx < len(segments) - 1:
            total += PIT_LOSS

    return total


# ============================================================
# STRATEGY GENERATOR
# ============================================================
def generate_strategies(total_laps):
    all_strats = []
    pit_range = list(range(10, total_laps - 10))

    for stops in [0, 1, 2]:
        for seq in itertools.product(COMPOUNDS, repeat=stops + 1):
            if stops == 0:
                all_strats.append((seq, []))
            else:
                for pits in itertools.combinations(pit_range, stops):
                    all_strats.append((seq, list(pits)))

    return all_strats


# ============================================================
# OPTIMIZER
# ============================================================
def optimize_strategy(total_laps, base_features):
    preds, prefix = precompute_lap_predictions(total_laps, base_features)
    strategies = generate_strategies(total_laps)

    log(f"Total strategies to evaluate: {len(strategies)}")

    results = []
    best_time = float("inf")
    best_strategy = None

    for compounds, pit_laps in strategies:
        t = simulate_strategy_fast(prefix, total_laps, compounds, pit_laps)

        results.append((compounds, pit_laps, t))

        if t < best_time:
            best_time = t
            best_strategy = (compounds, pit_laps)

    # Save all evaluated strategies
    strategies_df = pd.DataFrame([
        {
            "compound_sequence": comp,
            "pit_laps": pits,
            "total_time": t
        }
        for comp, pits, t in results
    ])

    strat_file = f"{SCRIPT_NAME}_all_strategy_results.csv"
    strategies_df.to_csv(f"{RESULTS_DIR}/{strat_file}", index=False)
    generated_files.append(strat_file)

    # Save best strategy summary
    best_summary = {
        "best_compound_sequence": best_strategy[0],
        "best_pit_laps": best_strategy[1],
        "best_total_time": float(best_time)
    }

    best_json_file = f"{SCRIPT_NAME}_best_strategy.json"
    with open(f"{RESULTS_DIR}/{best_json_file}", "w") as f:
        json.dump(best_summary, f, indent=2)
    generated_files.append(best_json_file)

    log("Best strategy:")
    log(str(best_summary))

    return best_strategy, best_time


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    start_features = {f: 0.0 for f in SIM_FEATURE_COLUMNS}

    # default initial tyres
    start_features.update({
        "compound_HARD": 1,
        "compound_MEDIUM": 0,
        "compound_SOFT": 0,
        "laps_since_stint_start": 1,
        "fuel_corrected_pace": 0.0
    })

    best, best_time = optimize_strategy(NUM_LAPS, start_features)

    log("\nOPTIMIZATION COMPLETE")
    log(f"Best strategy: {best}")
    log(f"Total time: {best_time}")

    # Save final summary
    final_txt = f"{SCRIPT_NAME}_final_summary.txt"
    with open(f"{RESULTS_DIR}/{final_txt}", "w") as f:
        f.write(f"Best: {best}\nTime: {best_time}\n")

    generated_files.append(final_txt)


    # Manifest
    manifest_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt"
    with open(manifest_path, "w") as mf:
        for f in generated_files:
            mf.write(f + "\n")

    log("\nFILES GENERATED:")
    for f in generated_files:
        log(f"  - {f}")

    log_file.close()