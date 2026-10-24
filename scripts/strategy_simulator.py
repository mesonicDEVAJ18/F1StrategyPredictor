import pandas as pd
import numpy as np
import itertools
import json
from collections import Counter
import joblib
import os
import warnings
from datetime import datetime

from feature_config import MODEL_FEATURE_COLUMNS, SIM_FEATURE_COLUMNS, SESSION_COLUMNS, FUEL_EFFECT_K
from env_config import env_str, env_int, env_float, env_list
from race_weekend import selected_weekend

warnings.filterwarnings("ignore")

# ============================================================
# SETUP RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "strategy_simulator"
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


log("\n--- STRATEGY SIMULATOR START ---\n")


# ============================================================
# LOAD MODEL
# ============================================================
bundle = joblib.load(env_str("MODEL_BUNDLE_PATH"))
model = bundle["model"]
FEATURES = bundle["features"]
MEDIANS = bundle["medians"]
SESSION_WEIGHTS = bundle["session_weights"]

log("Loaded trained XGBoost model.")
log(f"Session weights for averaging predictions: {SESSION_WEIGHTS}")


# ============================================================
# LOAD TYRE SET INVENTORY
# ============================================================
# Sets each car has left for the race, inferred from practice by tyre_sets.py (diagnostic.md #21).
# The race is simulated for one car, SIM_DRIVER; the model itself is trained on the whole field (#23)
SIM_DRIVER = env_int("SIM_DRIVER")
all_inventory = pd.read_csv(os.path.join(env_str("DATA_DIR"), env_str("PROCESSED_DIR"), env_str("TYRE_SETS_FILE")))
inventory = all_inventory[all_inventory["car_number"] == SIM_DRIVER]
if inventory.empty:
    raise ValueError(
        f"SIM_DRIVER={SIM_DRIVER} has no tyre set inventory. Cars available: "
        f"{sorted(all_inventory['car_number'].unique().tolist())} (a practice stand-in counts as its race car, see tyre_sets.py)")

log(f"Loaded tyre set inventory for car {SIM_DRIVER}: " + ", ".join(
    f"{c} {sorted(inventory.loc[inventory['compound'] == c, 'age'].tolist())}" for c in ["SOFT", "MEDIUM", "HARD"]))


# ============================================================
# CONSTANTS
# ============================================================
# Race settings come from .env and the race laps from the weekend table (diagnostic.md #24);
# COMPOUNDS and SECTOR_COLUMNS must match the model's columns
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
PIT_LOSS = env_float("PIT_LOSS_S")                         # seconds
FUEL_PENALTY_PER_LAP = env_float("FUEL_PENALTY_PER_LAP")   # seconds per lap of fuel
WEEKEND = selected_weekend()
NUM_LAPS = WEEKEND["race_laps"]
PIT_WINDOW_START_LAP = env_int("PIT_WINDOW_START_LAP")     # earliest pit lap
PIT_WINDOW_END_BUFFER = env_int("PIT_WINDOW_END_BUFFER")   # no pit in the last laps
STOP_COUNTS = env_list("STOP_COUNTS", cast=int)
MIN_COMPOUNDS_USED = env_int("MIN_COMPOUNDS_USED")         # two-compound rule (diagnostic.md #8)
SECTOR_COLUMNS = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
SMOOTHNESS_AGES = env_list("SMOOTHNESS_CHECK_AGES", cast=int) + [NUM_LAPS]  # 18 is past the longest stint age in training

log(f"Race weekend: {WEEKEND['year']} {WEEKEND['name']} ({WEEKEND['id']}), {NUM_LAPS} laps")


# ============================================================
# ONE-LAP PREDICTION
# ============================================================
def predict_fcp(state):
    """
    Predicts the next lap's fuel_corrected_pace (seconds) from lap-t state.

    The race is not a practice session, so the lap is predicted once per
    session one-hot and averaged with SESSION_WEIGHTS (share of training rows).
    """
    feats = state.copy()
    rows = []
    weights = []
    for s in SESSION_COLUMNS:
        for ss in SESSION_COLUMNS:
            feats[ss] = 1 if ss == s else 0
        rows.append([feats[f] for f in SIM_FEATURE_COLUMNS])
        weights.append(SESSION_WEIGHTS[s])

    model_vec = np.array(rows, dtype=float)[:, :len(FEATURES)]
    return float(np.average(model.predict(model_vec), weights=weights))


# ============================================================
# STINT ROLLOUT
# ============================================================
def rollout_stint(base_features, compound, set_age, stint_length):
    """
    Rolls one stint forward from stint lap 1 to stint_length, on a set that
    already has set_age laps on it (0 = new set).

    Lap 1 is the seed lap (training medians). Each later lap's fcp is predicted
    from the previous lap's state, then converted back to a raw lap time by
    adding back training's fuel term (fcp = lap_duration - k * laps left in the
    stint, counting the current lap)
    and fed in as the next state's sector times and running stint_avg_pace.
    """
    state = base_features.copy()
    for cc in COMPOUNDS:
        state[f"compound_{cc}"] = 1 if cc == compound else 0
    state["tyre_age_at_start"] = set_age
    state["laps_since_stint_start"] = 1
    state["tyre_age"] = set_age + 1

    seed_sectors = np.array([base_features[s] for s in SECTOR_COLUMNS], dtype=float)
    sector_share = seed_sectors / seed_sectors.sum()

    raw = float(seed_sectors.sum())
    fcp = raw - FUEL_EFFECT_K * stint_length
    state["fuel_corrected_pace"] = fcp

    records = []
    raw_laps = []
    for age in range(1, stint_length + 1):
        if age > 1:
            fcp = predict_fcp(state)
            raw = fcp + FUEL_EFFECT_K * (stint_length - age + 1)

            # Previous lap's raw times are the laps before this one
            state["stint_avg_pace"] = float(np.mean(raw_laps))
            for s, share in zip(SECTOR_COLUMNS, sector_share):
                state[s] = raw * share
            state["laps_since_stint_start"] = age
            state["tyre_age"] = set_age + age
            state["fuel_corrected_pace"] = fcp

        raw_laps.append(raw)
        records.append({
            "compound": compound,
            "set_age_at_start": set_age,
            "stint_length": stint_length,
            "laps_since_stint_start": age,
            "tyre_age": state["tyre_age"],
            "fuel_corrected_pace": fcp,
            "raw_lap_time": raw,
            **{s: state[s] for s in SECTOR_COLUMNS},
            "stint_avg_pace": state["stint_avg_pace"],
        })

    return records


# ============================================================
# PRECOMPUTE STINT ROLLOUTS
# ============================================================
def precompute_stint_rollouts(total_laps, base_features, set_types):
    """
    Rolls out every (compound, set age at start) in set_types for every stint
    length 1..total_laps:
        stint_totals[(c, a)][L] = sum of predicted fcp over a stint of L laps (seconds)
        fuel_prefix[lap]        = cumulative race fuel penalty up to race lap `lap`

    A stint is priced from stint lap 1 on its set wherever it starts in the race;
    only the race fuel term depends on the race laps it covers.

    Saves:
      - stint rollouts CSV (one row per compound, set age, stint length, stint lap)
      - stint totals CSV (compound x set age x stint length)
    """
    records = []
    stint_totals = {}

    for c, a in set_types:
        stint_totals[(c, a)] = np.zeros(total_laps + 1)
        for L in range(1, total_laps + 1):
            stint = rollout_stint(base_features, c, a, L)
            records.extend(stint)
            stint_totals[(c, a)][L] = sum(r["fuel_corrected_pace"] for r in stint)

    fuel_per_lap = [FUEL_PENALTY_PER_LAP * (total_laps - lap + 1) for lap in range(1, total_laps + 1)]
    fuel_prefix = np.concatenate([[0.0], np.cumsum(fuel_per_lap)])

    rollout_df = pd.DataFrame(records)
    rollout_file = f"{SCRIPT_NAME}_stint_rollouts.csv"
    rollout_df.to_csv(f"{RESULTS_DIR}/{rollout_file}", index=False)
    generated_files.append(rollout_file)

    totals_df = pd.DataFrame([
        {"compound": c, "set_age_at_start": a, "stint_length": L, "total_fcp": stint_totals[(c, a)][L]}
        for c, a in set_types for L in range(1, total_laps + 1)
    ])
    totals_file = f"{SCRIPT_NAME}_stint_totals.csv"
    totals_df.to_csv(f"{RESULTS_DIR}/{totals_file}", index=False)
    generated_files.append(totals_file)

    log("Saved stint rollouts and stint totals.")

    # Smoothness / drift check on the full-length rollout of each compound and set age
    log(f"\nRollout check (stint of {total_laps} laps, fcp in s, by stint lap):")
    for c, a in set_types:
        full = rollout_df[(rollout_df["compound"] == c) & (rollout_df["set_age_at_start"] == a)
                          & (rollout_df["stint_length"] == total_laps)]
        fcp = full["fuel_corrected_pace"].to_numpy()
        at_ages = ", ".join(f"lap {n}: {fcp[n - 1]:.2f}" for n in SMOOTHNESS_AGES if n <= len(fcp))
        max_step = np.abs(np.diff(fcp)).max() if len(fcp) > 1 else 0.0
        log(f"  {c} set age {a}: {at_ages}, max |lap-to-lap step|: {max_step:.2f}")

    return stint_totals, fuel_prefix


# ============================================================
# COMPUTE STINT TIME
# ============================================================
def stint_time(stint_totals, fuel_prefix, compound, set_age, start_lap, end_lap):
    length = end_lap - start_lap + 1
    if length <= 0:
        return 0.0
    fuel = fuel_prefix[end_lap] - fuel_prefix[start_lap - 1]
    return stint_totals[(compound, set_age)][length] + fuel


# ============================================================
# FAST STRATEGY SIMULATION
# ============================================================
def simulate_strategy_fast(stint_totals, fuel_prefix, total_laps, compounds, set_ages, pit_laps):
    pit_laps = sorted(pit_laps)
    segments = []

    # Every race lap is summed once: lap p is the in-lap (last lap of the old stint),
    # the next stint starts on lap p + 1, and PIT_LOSS is added on top
    last = 1
    for p in pit_laps:
        segments.append((last, p))
        last = p + 1
    segments.append((last, total_laps))

    total = 0
    for idx, (s, e) in enumerate(segments):
        comp = compounds[idx]
        total += stint_time(stint_totals, fuel_prefix, comp, set_ages[idx], s, e)
        if idx < len(segments) - 1:
            total += PIT_LOSS

    return total


# ============================================================
# SET CHOICES
# ============================================================
def set_age_options(ages_by_compound, seq):
    """
    Distinct ordered tuples of set ages a car can fit for compound sequence seq,
    each of its sets used at most once. Empty if the car hasn't enough sets.
    """
    left = {c: Counter(ages) for c, ages in ages_by_compound.items()}
    options = []

    def pick(i, chosen):
        if i == len(seq):
            options.append(tuple(chosen))
            return
        pool = left[seq[i]]
        for age in sorted(pool):
            if pool[age] > 0:
                pool[age] -= 1
                pick(i + 1, chosen + [age])
                pool[age] += 1

    pick(0, [])
    return options


# ============================================================
# STRATEGY GENERATOR
# ============================================================
def generate_strategies(total_laps):
    all_strats = []
    dropped = 0
    pit_range = list(range(PIT_WINDOW_START_LAP, total_laps - PIT_WINDOW_END_BUFFER))

    for stops in STOP_COUNTS:
        for seq in itertools.product(COMPOUNDS, repeat=stops + 1):
            # Illegal: fewer distinct compounds than the rule requires (this also rules out 0 stops)
            if len(set(seq)) < MIN_COMPOUNDS_USED:
                dropped += 1 if stops == 0 else len(list(itertools.combinations(pit_range, stops)))
                continue
            if stops == 0:
                all_strats.append((seq, []))
            else:
                for pits in itertools.combinations(pit_range, stops):
                    all_strats.append((seq, list(pits)))

    log(f"Dropped {dropped} strategies using fewer than {MIN_COMPOUNDS_USED} compounds.")
    return all_strats


# ============================================================
# OPTIMIZER
# ============================================================
def optimize_strategy(total_laps, base_features):
    # One rollout per (compound, set age) the car can fit
    set_types = sorted(set(zip(inventory["compound"], inventory["age"].astype(int))),
                       key=lambda t: (COMPOUNDS.index(t[0]), t[1]))
    log(f"\nRolling out {len(set_types)} (compound, set age) pairs: {set_types}")
    stint_totals, fuel_prefix = precompute_stint_rollouts(total_laps, base_features, set_types)
    strategies = generate_strategies(total_laps)

    log(f"Total strategies to evaluate: {len(strategies)}")

    rows = []
    ages_by_compound = {c: inventory.loc[inventory["compound"] == c, "age"].astype(int).tolist() for c in COMPOUNDS}
    options_by_seq = {}
    infeasible = 0
    best_time = float("inf")
    best_strategy = None

    for compounds, pit_laps in strategies:
        if compounds not in options_by_seq:
            options_by_seq[compounds] = set_age_options(ages_by_compound, compounds)
        options = options_by_seq[compounds]
        if not options:
            infeasible += 1
            continue

        # The car fits whichever of its sets makes this plan fastest
        t, set_ages = min(
            (simulate_strategy_fast(stint_totals, fuel_prefix, total_laps, compounds, ages, pit_laps), ages)
            for ages in options
        )
        rows.append({
            "car_number": SIM_DRIVER,
            "compound_sequence": compounds,
            "set_ages": set_ages,
            "pit_laps": pit_laps,
            "total_time": t
        })

        if t < best_time:
            best_time = t
            best_strategy = (compounds, set_ages, pit_laps)

    log(f"\nCar {SIM_DRIVER}: {infeasible} strategies skipped (not enough sets of a compound), "
        f"{len(strategies) - infeasible} evaluated")

    best = {
        "car_number": SIM_DRIVER,
        "best_compound_sequence": best_strategy[0],
        "best_set_ages_at_start": best_strategy[1],
        "best_pit_laps": best_strategy[2],
        "best_total_time": float(best_time),
    }
    log(f"Car {SIM_DRIVER} best strategy: {best}")

    # Save all evaluated strategies
    strategies_df = pd.DataFrame(rows)
    strat_file = f"{SCRIPT_NAME}_all_strategy_results.csv"
    strategies_df.to_csv(f"{RESULTS_DIR}/{strat_file}", index=False)
    generated_files.append(strat_file)

    # Save best strategy summary
    best_summary = {"time_units": "seconds", **best}

    best_json_file = f"{SCRIPT_NAME}_best_strategy.json"
    with open(f"{RESULTS_DIR}/{best_json_file}", "w") as f:
        json.dump(best_summary, f, indent=2)
    generated_files.append(best_json_file)

    return best


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Baseline = training medians (features are in physical units, so 0 is not "average").
    # Speed traps and car-data aggregates stay at these medians for the whole rollout.
    start_features = {f: 0.0 for f in SIM_FEATURE_COLUMNS}
    start_features.update({f: v for f, v in MEDIANS.items() if f in start_features})

    # default initial tyres (tyre age, sector times and stint_avg_pace are set by the rollout)
    start_features.update({
        "compound_HARD": 1,
        "compound_MEDIUM": 0,
        "compound_SOFT": 0,
    })

    best = optimize_strategy(NUM_LAPS, start_features)

    log("\nOPTIMIZATION COMPLETE")
    line = (f"Car {best['car_number']}: {best['best_compound_sequence']}, set ages {best['best_set_ages_at_start']}, "
            f"pits {best['best_pit_laps']}, time {best['best_total_time']:.2f} s")
    log(line)

    # Save final summary
    final_txt = f"{SCRIPT_NAME}_final_summary.txt"
    with open(f"{RESULTS_DIR}/{final_txt}", "w") as f:
        f.write(line + "\n")

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