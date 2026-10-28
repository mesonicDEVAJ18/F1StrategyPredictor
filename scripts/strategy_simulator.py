import sys
import pandas as pd
import numpy as np
import itertools
import json
from collections import Counter
import joblib
import os
import warnings
from datetime import datetime

from feature_config import COMPOUNDS, PACE_GROUP_COLUMNS, fall_off_curve
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
    # Replace characters the console/redirect encoding can't show (e.g. cp1252 when output is
    # redirected on Windows); the log file is UTF-8 and keeps the original text
    enc = sys.stdout.encoding or "utf-8"
    safe_msg = msg.encode(enc, errors="replace").decode(enc, "replace")
    print(safe_msg)
    log_file.write(msg + "\n")


log("\n--- STRATEGY SIMULATOR START ---\n")


# ============================================================
# LOAD MODEL
# ============================================================
bundle = joblib.load(env_str("MODEL_BUNDLE_PATH"))
model = bundle["model"]
FEATURES = bundle["features"]
MEDIANS = bundle["medians"]
COMPOUND_GAP = bundle["compound_gap"]
LEVELS = bundle["levels"]
LINEAR_SLOPES = bundle["linear_slopes"]
ANCHOR_AGE = bundle["anchor_age"]

log("Loaded trained XGBoost fall-off model.")
log(f"Compound gap on a new set (s, vs SOFT): {COMPOUND_GAP}")


# ============================================================
# LOAD TYRE SET INVENTORY
# ============================================================
# Sets each car has left for the race, inferred from practice by tyre_sets.py.
# The race is simulated for one car, SIM_DRIVER; the model itself is trained on the whole field
SIM_DRIVER = env_int("SIM_DRIVER")
all_inventory = pd.read_csv(os.path.join(env_str("DATA_DIR"), env_str("PROCESSED_DIR"), env_str("TYRE_SETS_FILE")))
inventory = all_inventory[all_inventory["car_number"] == SIM_DRIVER]
if inventory.empty:
    raise ValueError(
        f"SIM_DRIVER={SIM_DRIVER} has no tyre set inventory. Cars available: "
        f"{sorted(all_inventory['car_number'].unique().tolist())} (a practice stand-in counts as its race car, see tyre_sets.py)")

# Race pace level: the mean of the car's driver-session levels from train_model.py's compound-gap fit.
# It is the same for every strategy, so it only shifts the totals, not the ranking
car_levels = [v for k, v in LEVELS.items() if k[PACE_GROUP_COLUMNS.index("driver_number")] == SIM_DRIVER]
if not car_levels:
    raise ValueError(f"SIM_DRIVER={SIM_DRIVER} has no pace level in the model bundle (no training laps)")
RACE_REFERENCE = float(np.mean(car_levels))
log(f"Race pace level for car {SIM_DRIVER}: {RACE_REFERENCE:.2f} s (mean of {len(car_levels)} driver-session levels)")

log(f"Loaded tyre set inventory for car {SIM_DRIVER}: " + ", ".join(
    f"{c} {sorted(inventory.loc[inventory['compound'] == c, 'age'].tolist())}" for c in ["SOFT", "MEDIUM", "HARD"]))


# ============================================================
# CONSTANTS
# ============================================================
# Race settings come from .env and the race laps from the weekend table
PIT_LOSS = env_float("PIT_LOSS_S")                         # seconds
FUEL_PENALTY_PER_LAP = env_float("FUEL_PENALTY_PER_LAP")   # seconds per lap of fuel
WEEKEND = selected_weekend()
NUM_LAPS = WEEKEND["race_laps"]
PIT_WINDOW_START_LAP = env_int("PIT_WINDOW_START_LAP")     # earliest pit lap
PIT_WINDOW_END_BUFFER = env_int("PIT_WINDOW_END_BUFFER")   # no pit in the last laps
STOP_COUNTS = env_list("STOP_COUNTS", cast=int)
MIN_COMPOUNDS_USED = env_int("MIN_COMPOUNDS_USED")         # two-compound rule
SMOOTHNESS_AGES = env_list("SMOOTHNESS_CHECK_AGES", cast=int) + [NUM_LAPS]  # 18 is near the oldest tyre in training

log(f"Race weekend: {WEEKEND['year']} {WEEKEND['name']} ({WEEKEND['id']}), {NUM_LAPS} laps")


# ============================================================
# STINT CURVE
# ============================================================
def stint_curve(compound, set_age, n_laps):
    """
    Predicted fuel_corrected_pace (seconds) on stint laps 1..n_laps on a set that
    already has set_age laps on it (0 = new set):
        fcp = race pace level + compound gap + fall_off(compound, tyre age)
    with tyre age = set_age + stint lap, fall-off measured from a new set's first lap
    and the conditions held at the training medians. A used set starts further along
    the curve. Nothing is fed back from one lap to the next.
    """
    ages = set_age + np.arange(1, n_laps + 1)
    fall_off = fall_off_curve(model, FEATURES, compound, ages, MEDIANS, set_age)
    fcp = RACE_REFERENCE + COMPOUND_GAP[compound] + fall_off
    return ages, fall_off, fcp


# ============================================================
# PRECOMPUTE STINT TOTALS
# ============================================================
def precompute_stint_totals(total_laps, set_types):
    """
    Prices every (compound, set age at start) in set_types for every stint length
    1..total_laps:
        stint_totals[(c, a)][L] = sum of predicted fcp over the first L stint laps (seconds)
        fuel_prefix[lap]        = cumulative race fuel penalty up to race lap `lap`

    The curve doesn't depend on the stint length, so one curve per set is summed
    cumulatively. A stint is priced from stint lap 1 on its set wherever it starts
    in the race; only the race fuel term depends on the race laps it covers.

    Saves:
      - stint curves CSV (one row per compound, set age, stint lap)
      - stint totals CSV (compound x set age x stint length)
    """
    records = []
    stint_totals = {}

    for c, a in set_types:
        ages, fall_off, fcp = stint_curve(c, a, total_laps)
        stint_totals[(c, a)] = np.concatenate([[0.0], np.cumsum(fcp)])
        records += [{"compound": c, "set_age_at_start": a, "laps_since_stint_start": n + 1, "tyre_age": int(ages[n]),
                     "fall_off": fall_off[n], "fuel_corrected_pace": fcp[n]} for n in range(total_laps)]

    fuel_per_lap = [FUEL_PENALTY_PER_LAP * (total_laps - lap + 1) for lap in range(1, total_laps + 1)]
    fuel_prefix = np.concatenate([[0.0], np.cumsum(fuel_per_lap)])

    curve_df = pd.DataFrame(records)
    curve_file = f"{SCRIPT_NAME}_stint_curves.csv"
    curve_df.to_csv(f"{RESULTS_DIR}/{curve_file}", index=False)
    generated_files.append(curve_file)

    totals_df = pd.DataFrame([
        {"compound": c, "set_age_at_start": a, "stint_length": L, "total_fcp": stint_totals[(c, a)][L]}
        for c, a in set_types for L in range(1, total_laps + 1)
    ])
    totals_file = f"{SCRIPT_NAME}_stint_totals.csv"
    totals_df.to_csv(f"{RESULTS_DIR}/{totals_file}", index=False)
    generated_files.append(totals_file)

    log("Saved stint curves and stint totals.")

    # Curve check: fcp and fall-off at a few stint laps, the largest step, and the stage-1 linear fall-off
    log(f"\nCurve check (stint of {total_laps} laps; fcp and fall-off in s, by stint lap):")
    for c, a in set_types:
        part = curve_df[(curve_df["compound"] == c) & (curve_df["set_age_at_start"] == a)]
        fcp = part["fuel_corrected_pace"].to_numpy()
        fall = part["fall_off"].to_numpy()
        at_ages = ", ".join(
            f"lap {n}: {fcp[n - 1]:.2f} (fall-off {fall[n - 1]:+.2f}, linear "
            f"{LINEAR_SLOPES[c] * (a + n - ANCHOR_AGE):+.2f})" for n in SMOOTHNESS_AGES if n <= len(fcp))
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
def optimize_strategy(total_laps):
    # One curve per (compound, set age) the car can fit
    set_types = sorted(set(zip(inventory["compound"], inventory["age"].astype(int))),
                       key=lambda t: (COMPOUNDS.index(t[0]), t[1]))
    log(f"\nPricing {len(set_types)} (compound, set age) pairs: {set_types}")
    stint_totals, fuel_prefix = precompute_stint_totals(total_laps, set_types)
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
    # The conditions (weather) stay at the training medians; stint_curve() sets each
    # stint's compound and tyre age
    best = optimize_strategy(NUM_LAPS)

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