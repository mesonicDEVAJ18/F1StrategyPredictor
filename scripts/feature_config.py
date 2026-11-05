import numpy as np
import pandas as pd

from env_config import env_float

# Fuel effect (s per lap) used to build fuel_corrected_pace = lap_duration - k * laps_remaining,
# where laps_remaining = lap_end - lap_number + 1 (laps left in the stint, counting lap t).
FUEL_EFFECT_K = env_float("FUEL_EFFECT_K")

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Practice session one-hot (named by order, so stable across weekends; session_key is an identifier only)
SESSION_COLUMNS = ["session_FP1", "session_FP2", "session_FP3"]

# Per-lap car-data aggregates (process_cardata.py), in physical units; lap t's own telemetry
CARDATA_FEATURE_COLUMNS = [
    "mean_speed", "max_speed",
    "mean_throttle", "full_throttle_pct",
    "brake_pct",
    "mean_rpm", "max_rpm",
    "mean_gear", "gear_changes",
    "drs_open_pct",
]

# Values that can be missing on a lap (speed traps, segment counts, masked or missing car data):
# feature_engineer.py fills them with the median of the same stint's earlier laps (#33)
STINT_FILL_COLUMNS = [
    "i1_speed", "i2_speed", "st_speed",
    "segments_sector_1", "segments_sector_2", "segments_sector_3",
    *CARDATA_FEATURE_COLUMNS,
]

SECTOR_COLUMNS = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]

# The lap's own timing and telemetry. The target is that lap's pace, so these are parts or proxies
# of it and must never be model inputs (#31); they stay in the dataset for eda.py and explore/
LAP_TIMING_COLUMNS = [
    *SECTOR_COLUMNS,
    *STINT_FILL_COLUMNS,
    "stint_avg_pace", "fuel_corrected_pace", "lap_duration",
]

# Feature sets the fall-off model can use (the tyre and the conditions only; never the lap's own timing, #31).
# train_model.py picks one with FALL_OFF_FEATURE_SET in .env; explore/explore_master.py compares them all
TYRE = [
    "compound_HARD",
    "compound_MEDIUM",
    "compound_SOFT",
    "tyre_age",                 # tyre_age_at_start + laps_since_stint_start (laps on the set)
]
WEATHER = ["air_temperature", "humidity", "pressure", "track_temperature", "wind_direction", "wind_speed"]  # rainfall is constant
CORE = TYRE + ["track_temperature", "air_temperature"]

FEATURE_SETS = {
    "core": CORE,
    "tyre_only": TYRE,
    "all_weather": TYRE + WEATHER,
    "plus_age_at_start": CORE + ["tyre_age_at_start"],
    "plus_stint_lap": CORE + ["laps_since_stint_start"],
}
for _name, _cols in FEATURE_SETS.items():
    _leaks = set(_cols) & set(LAP_TIMING_COLUMNS)
    assert not _leaks, f"Feature set {_name} takes inputs from the lap's own timing: {sorted(_leaks)}"

# The default model input order (FALL_OFF_FEATURE_SET=core)
MODEL_FEATURE_COLUMNS = FEATURE_SETS["core"]

# One stint of one driver (stint_number is the global per-driver number, so it also pins the session)
STINT_GROUP_COLUMNS = ["driver_number", "stint_number"]
# One driver in one practice session (the compound-gap fit's level)
PACE_GROUP_COLUMNS = ["driver_number", "session_key"]
# Fall-off is measured from this tyre age (1 = a new set's first lap)
FALL_OFF_ANCHOR_AGE = 1


# ============================================================
# FALL-OFF HELPERS (shared by train_model, validate_strategy,
# strategy_simulator, eda and explore/)
# ============================================================
def compound_of(df):
    """Compound name of each row, from the compound one-hot."""
    cols = [f"compound_{c}" for c in COMPOUNDS]
    return np.array(COMPOUNDS)[df[cols].to_numpy().argmax(axis=1)]


def multi_lap_rows(df):
    """The laps of stints with at least two clean laps, sorted by stint and lap.

    Only these carry fall-off: a one-lap stint has no change with age once its level is removed.
    """
    size = df.groupby(STINT_GROUP_COLUMNS)["lap_number"].transform("size")
    out = df[size >= 2].sort_values(STINT_GROUP_COLUMNS + ["lap_number"])
    return out.reset_index(drop=True)


def _dummies(df, columns):
    """One indicator column per group of `columns`; returns (matrix, group keys)."""
    codes, keys = pd.factorize(pd.MultiIndex.from_frame(df[columns]))
    mat = np.zeros((len(df), len(keys)))
    mat[np.arange(len(df)), codes] = 1.0
    return mat, keys


def fit_stint_levels(df):
    """Stage 1: OLS of fuel_corrected_pace on one level per stint + one tyre_age slope per compound.

    The level absorbs everything constant over a run (driver, session, fuel plan, engine mode, run type),
    so the slopes come from the change within stints only. Returns (levels: Series indexed by
    STINT_GROUP_COLUMNS, slopes: {compound: s per lap}).
    """
    stint_mat, keys = _dummies(df, STINT_GROUP_COLUMNS)
    age_terms = np.column_stack([df[f"compound_{c}"].to_numpy() * df["tyre_age"].to_numpy() for c in COMPOUNDS])
    coef, *_ = np.linalg.lstsq(np.hstack([stint_mat, age_terms]), df["fuel_corrected_pace"].to_numpy(), rcond=None)
    levels = pd.Series(coef[:len(keys)], index=pd.MultiIndex.from_tuples(keys, names=STINT_GROUP_COLUMNS))
    return levels, dict(zip(COMPOUNDS, coef[len(keys):]))


def fall_off_target(df, levels):
    """fuel_corrected_pace minus the row's stint level (NaN for a stint without one)."""
    lvl = levels.reindex(pd.MultiIndex.from_frame(df[STINT_GROUP_COLUMNS])).to_numpy()
    return df["fuel_corrected_pace"].to_numpy() - lvl


def linear_fall_off(df, slopes):
    """Stage-1 fall-off of each row: its compound's slope x (tyre_age - FALL_OFF_ANCHOR_AGE)."""
    slope = np.array([slopes[c] for c in compound_of(df)])
    return slope * (df["tyre_age"].to_numpy() - FALL_OFF_ANCHOR_AGE)


def stint_shape_errors(fcp, pred, stint_ids):
    """Residual fcp - pred minus each stint's mean residual: the error in the shape of the curve.

    A held-out stint's level is unknown by construction, so it is fitted on the stint itself (one
    constant) and only the change with age is scored.
    """
    resid = pd.Series(np.asarray(fcp, dtype=float) - np.asarray(pred, dtype=float))
    return (resid - resid.groupby(np.asarray(stint_ids)).transform("mean")).to_numpy()


def _curve_frame(features, compound, tyre_ages, conditions, set_age):
    tyre_ages = np.asarray(tyre_ages, dtype=float)
    frame = pd.DataFrame({f: np.full(len(tyre_ages), conditions.get(f, 0.0), dtype=float) for f in features})
    for c in COMPOUNDS:
        if f"compound_{c}" in frame:
            frame[f"compound_{c}"] = 1.0 if c == compound else 0.0
    if "tyre_age" in frame:
        frame["tyre_age"] = tyre_ages
    if "tyre_age_at_start" in frame:
        frame["tyre_age_at_start"] = float(set_age)
    if "laps_since_stint_start" in frame:
        frame["laps_since_stint_start"] = tyre_ages - set_age
    return frame


def fall_off_anchor(model, features, compound, conditions):
    """The model's prediction on a new set's first lap (FALL_OFF_ANCHOR_AGE) at `conditions`."""
    frame = _curve_frame(features, compound, [FALL_OFF_ANCHOR_AGE], conditions, 0)
    return float(model.predict(frame)[0])


def fall_off_curve(model, features, compound, tyre_ages, conditions, set_age=0):
    """Fall-off (s) at each tyre age on a set that started the stint at set_age laps, measured from a
    new set's first lap; every other input is held at `conditions` (e.g. the training medians)."""
    frame = _curve_frame(features, compound, tyre_ages, conditions, set_age)
    return model.predict(frame).astype(float) - fall_off_anchor(model, features, compound, conditions)


def fall_off_rows(model, features, df, conditions):
    """Fall-off of each row at its own inputs, from its compound's anchor at `conditions`."""
    X = df[features].fillna(pd.Series(conditions))
    anchors = {c: fall_off_anchor(model, features, c, conditions) for c in COMPOUNDS}
    return model.predict(X).astype(float) - np.array([anchors[c] for c in compound_of(df)])


def stint_new_set_levels(df):
    """Each stint's pace on a new set, from stage 1 on every row of df (no fall-off model, #27).

    A one-lap stint's level is its own lap with the fall-off removed, so every clean lap counts.
    new_set_pace = stint level + the compound's slope x FALL_OFF_ANCHOR_AGE: the stint's fcp at a new
    set's first lap. Returns one row per stint (sorted by STINT_GROUP_COLUMNS) with session_key, the
    compound one-hot, compound, run_laps, laps (its clean laps) and new_set_pace.
    """
    levels, slopes = fit_stint_levels(df)
    stints = df.assign(compound=compound_of(df)).groupby(STINT_GROUP_COLUMNS).agg(
        session_key=("session_key", "first"),
        **{f"compound_{c}": (f"compound_{c}", "first") for c in COMPOUNDS},
        compound=("compound", "first"),
        run_laps=("run_laps", "first"),
        laps=("lap_number", "size"),
    ).reset_index()
    stints["new_set_pace"] = (
        levels.reindex(pd.MultiIndex.from_frame(stints[STINT_GROUP_COLUMNS])).to_numpy()
        + stints["compound"].map(slopes).to_numpy() * FALL_OFF_ANCHOR_AGE
    )
    return stints


def fit_compound_gap(stints):
    """Stage 3 (model-free, #27): OLS of each stint's new_set_pace (stint_new_set_levels) on one level
    per driver and session + a HARD and a MEDIUM gap to SOFT + run_laps, each stint weighted by its laps.
    Compound gaps need comparisons between stints, so run_laps controls for the run type (short quali
    sim vs long run).

    The regressors are constant within a stint and stage 1's residuals sum to zero over each stint, so
    this equals the per-lap fit of fcp minus the stage-1 linear fall-off.

    Returns (gap: {compound: s on a new set, SOFT = 0}, levels: Series indexed by PACE_GROUP_COLUMNS
    and evaluated at the median run_laps over the laps, run_laps coefficient in s per lap of run).
    """
    level_mat, keys = _dummies(stints, PACE_GROUP_COLUMNS)
    extra = stints[["compound_HARD", "compound_MEDIUM", "run_laps"]].to_numpy(dtype=float)
    w = np.sqrt(stints["laps"].to_numpy(dtype=float))[:, None]
    y = stints["new_set_pace"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(np.hstack([level_mat, extra]) * w, y * w[:, 0], rcond=None)
    k = len(keys)
    run_coef = float(coef[k + 2])
    lap_run_laps = np.repeat(stints["run_laps"].to_numpy(dtype=float), stints["laps"].to_numpy())
    levels = pd.Series(coef[:k] + run_coef * float(np.median(lap_run_laps)),
                       index=pd.MultiIndex.from_tuples(keys, names=PACE_GROUP_COLUMNS))
    gap = {"SOFT": 0.0, "MEDIUM": float(coef[k + 1]), "HARD": float(coef[k])}
    return gap, levels, run_coef
