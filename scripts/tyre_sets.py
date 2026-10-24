import pandas as pd
import os

from env_config import env_str, env_list

# ============================================================
# SETUP SCRIPT-SPECIFIC RESULTS DIRECTORY
# ============================================================
SCRIPT_NAME = "tyre_sets"
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


def env_pairs(key, key_cast=str, value_cast=int):
    """KEY=a:1,b:2 -> {a: 1, b: 2}"""
    pairs = {}
    for item in env_list(key):
        k, v = item.split(":")
        pairs[key_cast(k.strip())] = value_cast(v.strip())
    return pairs


# ============================================================
# CONFIG
# ============================================================
# OpenF1 stints have no set ID, so sets are inferred by chaining: a stint that starts
# used continues the same-compound set that earlier running left at that age (diagnostic.md #21)
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

processed_dir = os.path.join(env_str("DATA_DIR"), env_str("PROCESSED_DIR"))
stints_file = os.path.join(processed_dir, env_str("STINTS_CLEAN_FILE"))
inventory_file = os.path.join(processed_dir, env_str("TYRE_SETS_FILE"))
drivers_files = [os.path.join(env_str("DATA_DIR"), env_str("DRIVERS_DIR"), f) for f in env_list("DRIVERS_FILES")]

TYRE_ALLOCATION = env_pairs("TYRE_ALLOCATION")                    # compound -> sets per weekend
TYRE_RETURNS_PER_SESSION = env_list("TYRE_RETURNS_PER_SESSION", cast=int)
TYRE_RACE_RESERVED = env_pairs("TYRE_RACE_RESERVED")              # compound -> new sets kept for the race


# ============================================================
# PRACTICE STAND-INS
# ============================================================
def derive_stand_ins():
    """Practice driver -> race car, from the FP1-FP3 entry lists (diagnostic.md #24).

    The FP3 lineup is taken as the race cars. In each session, a driver missing from the FP3
    lineup is paired with the one driver of the same team who sat that session out, e.g. an FP1
    rookie or a driver replaced before FP3 (Sainz #55 -> Bearman #38, Jeddah 2024).
    """
    sessions = [pd.read_csv(f, usecols=["driver_number", "team_name"]) for f in drivers_files]
    race_lineup = sessions[-1]
    stand_ins = {}
    for n, entries in enumerate(sessions[:-1], start=1):
        for team, team_entries in entries.groupby("team_name"):
            ran = set(team_entries["driver_number"].astype(int))
            race_cars = set(race_lineup.loc[race_lineup["team_name"] == team, "driver_number"].astype(int))
            extra, absent = sorted(ran - race_cars), sorted(race_cars - ran)
            if not extra:
                continue
            if len(extra) == 1 and len(absent) == 1:
                if stand_ins.setdefault(extra[0], absent[0]) != absent[0]:
                    log(f"WARNING: FP{n} {team}: #{extra[0]} already maps to #{stand_ins[extra[0]]}, not #{absent[0]}")
            else:
                log(f"WARNING: FP{n} {team}: can't pair {extra} with race cars {absent}; left as their own cars")
    return stand_ins


# ============================================================
# LOAD STINTS
# ============================================================
log("\n--- TYRE SETS START ---\n")

FP_STAND_INS = derive_stand_ins()                                 # practice driver -> race car

stints = pd.read_csv(stints_file)
stints["compound"] = (
    stints[[f"compound_{c}" for c in COMPOUNDS]].idxmax(axis=1).str.replace("compound_", "")
)
stints["car_number"] = stints["driver_number"].map(lambda d: FP_STAND_INS.get(d, d))
# Every car in practice gets an inventory; strategy_simulator.py picks one with SIM_DRIVER (diagnostic.md #23)
CARS = sorted(int(c) for c in stints["car_number"].unique())

# Session keys increase FP1 -> FP3, as in feature_engineer.py
session_keys = sorted(int(k) for k in stints["session_key"].unique())
session_names = {k: f"FP{i + 1}" for i, k in enumerate(session_keys)}

log(f"Loaded {len(stints)} stints from {stints_file}")
log(f"Sessions: {session_names}")
log(f"Practice stand-ins (driver: race car, from {[os.path.basename(f) for f in drivers_files]}): "
    + ",".join(f"{d}:{c}" for d, c in FP_STAND_INS.items()))
log(f"Cars ({len(CARS)}): {CARS}")
log(f"Allocation: {TYRE_ALLOCATION}, returns per session: {TYRE_RETURNS_PER_SESSION}, "
    f"race-reserved new sets: {TYRE_RACE_RESERVED}")


# ============================================================
# SET CHAINING
# ============================================================
def new_set(sets, compound, car):
    """Next unused set of the compound; adds an over-allocation set if none is left."""
    for s in sets:
        if s["compound"] == compound and not s["used"] and s["returned_after"] is None:
            return s
    log(f"  [car {car}] WARNING: no new {compound} set left in the allocation, adding one")
    s = {"set_id": f"{compound[0]}x{len(sets) + 1}", "compound": compound, "age": 0,
         "used": False, "returned_after": None, "last_used": -1, "history": []}
    sets.append(s)
    return s


def return_sets(sets, n, session, car, returned_rows):
    """Hand back n sets, most-worn first, keeping TYRE_RACE_RESERVED new sets per compound."""
    for _ in range(n):
        candidates = []
        for s in sets:
            if s["returned_after"] is not None:
                continue
            if not s["used"]:
                new_left = sum(1 for o in sets if o["compound"] == s["compound"]
                               and not o["used"] and o["returned_after"] is None)
                if new_left <= TYRE_RACE_RESERVED.get(s["compound"], 0):
                    continue
            candidates.append(s)
        if not candidates:
            log(f"  [car {car}] WARNING: no set can be returned after {session}")
            return
        s = max(candidates, key=lambda o: (o["age"], o["used"], -o["last_used"]))
        s["returned_after"] = session
        returned_rows.append({"car_number": car, "returned_after": session, "set_id": s["set_id"],
                              "compound": s["compound"], "age": s["age"]})
        log(f"  [car {car}] Returned after {session}: {s['set_id']} ({s['compound']}, age {s['age']})")


def infer_car_sets(car, car_stints, assignment_rows, returned_rows):
    sets = []
    for c, n in TYRE_ALLOCATION.items():
        for i in range(n):
            sets.append({"set_id": f"{c[0]}{i + 1}", "compound": c, "age": 0, "used": False,
                         "returned_after": None, "last_used": -1, "history": []})

    counts = {"new": 0, "matched": 0, "ambiguous": 0, "unmatched": 0}
    order = 0
    for i, session_key in enumerate(session_keys):
        session = session_names[session_key]
        for _, st in car_stints[car_stints["session_key"] == session_key].iterrows():
            compound, start_age = st["compound"], int(st["tyre_age_at_start"])

            if start_age == 0:
                s, match = new_set(sets, compound, car), "new"
            else:
                cands = [o for o in sets if o["compound"] == compound and o["used"]
                         and o["returned_after"] is None and o["age"] == start_age]
                if cands:
                    # Sets sharing compound and age are interchangeable for the inventory
                    s = max(cands, key=lambda o: o["last_used"])
                    match = "ambiguous" if len(cands) > 1 else "matched"
                else:
                    # Used before the data starts (or ages don't chain): unknown history
                    s, match = new_set(sets, compound, car), "unmatched"
                    log(f"  [car {car}] No {compound} set at age {start_age} for "
                        f"{session} stint {st['stint_number']}; using a new set at that age")
            counts[match] += 1

            end_age = start_age + int(st["lap_end"] - st["lap_start"])
            s.update(age=end_age, used=True, last_used=order)
            s["history"].append(f"{session} #{st['driver_number']} laps {st['lap_start']}-{st['lap_end']}")
            order += 1

            assignment_rows.append({
                "car_number": car, "session": session, "driver_number": st["driver_number"],
                "stint_number": st["stint_number"], "compound": compound,
                "lap_start": st["lap_start"], "lap_end": st["lap_end"],
                "tyre_age_at_start": start_age, "set_id": s["set_id"], "match": match,
                "set_age_after": end_age,
            })

        n_return = TYRE_RETURNS_PER_SESSION[i] if i < len(TYRE_RETURNS_PER_SESSION) else 0
        return_sets(sets, n_return, session, car, returned_rows)

    log(f"  [car {car}] Stints: {counts}")
    for k, n in counts.items():
        match_totals[k] += n
    return sets


# ============================================================
# MAIN
# ============================================================
assignment_rows = []
returned_rows = []
inventory_rows = []
match_totals = {"new": 0, "matched": 0, "ambiguous": 0, "unmatched": 0}

for car in CARS:
    car_stints = (
        stints[stints["car_number"] == car]
        .sort_values(["session_key", "stint_number"])
    )
    log(f"\nCar {car}: {len(car_stints)} practice stints "
        f"(drivers {sorted(car_stints['driver_number'].unique().tolist())})")

    sets = infer_car_sets(car, car_stints, assignment_rows, returned_rows)

    for s in sets:
        if s["returned_after"] is None:
            inventory_rows.append({"car_number": car, "set_id": s["set_id"], "compound": s["compound"],
                                   "age": s["age"], "history": "; ".join(s["history"]) or "new"})

    left = [r for r in inventory_rows if r["car_number"] == car]
    for c in COMPOUNDS:
        ages = sorted(r["age"] for r in left if r["compound"] == c)
        log(f"  [car {car}] Race {c} sets (age in laps): {ages}")

log(f"\nStints across all {len(CARS)} cars: {match_totals}")


# ============================================================
# SAVE OUTPUTS
# ============================================================
assign_df = pd.DataFrame(assignment_rows)
assign_name = f"{SCRIPT_NAME}_stint_set_assignment.csv"
assign_df.to_csv(f"{RESULTS_DIR}/{assign_name}", index=False)
generated_files.append(assign_name)

returned_df = pd.DataFrame(returned_rows)
returned_name = f"{SCRIPT_NAME}_returned_sets.csv"
returned_df.to_csv(f"{RESULTS_DIR}/{returned_name}", index=False)
generated_files.append(returned_name)

inventory_df = pd.DataFrame(inventory_rows)
inventory_name = f"{SCRIPT_NAME}_race_inventory.csv"
inventory_df.to_csv(f"{RESULTS_DIR}/{inventory_name}", index=False)
generated_files.append(inventory_name)

# Hand-off for strategy_simulator.py
inventory_df.to_csv(inventory_file, index=False)
log(f"\nSaved hand-off dataset: {inventory_file}")


# ============================================================
# MANIFEST FILE
# ============================================================
manifest_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt"
with open(manifest_path, "w") as m:
    for item in generated_files:
        m.write(item + "\n")
generated_files.append(f"{SCRIPT_NAME}_manifest.txt")

log("\nPROCESS COMPLETE. Files generated:")
for f in generated_files:
    log(f"  - {f}")

log_file.close()
