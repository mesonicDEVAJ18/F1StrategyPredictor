# Helpers for explore_master.py: custom estimators, model building, nested grouped CV
# and the simulator-style rollout. No settings here: the experiment block in
# explore_master.py and the repo-root .env hold them.
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from feature_config import (  # noqa: E402
    FUEL_EFFECT_K, MODEL_FEATURE_COLUMNS, SESSION_COLUMNS, TARGET_GROUP_COLUMNS, build_next_lap_target,
)

SECTOR_COLUMNS = ["duration_sector_1", "duration_sector_2", "duration_sector_3"]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


# ============================================================
# CUSTOM ESTIMATORS
# ============================================================
class PerCompoundDegradation(BaseEstimator, RegressorMixin):
    """Simple per-compound linear degradation model (#27).

    For each compound, next-lap fcp = a + b * (lap t time) + c * tyre_age, where lap t time is
    the sum of lap t's sector times. A compound with too few rows falls back to one fit on all rows.
    Needs a DataFrame with the sector, tyre_age and compound_* columns.
    """

    def __init__(self, min_rows=10):
        self.min_rows = min_rows

    def _design(self, X):
        return np.column_stack([X[SECTOR_COLUMNS].sum(axis=1), X["tyre_age"]])

    def _compound(self, X):
        cols = [f"compound_{c}" for c in COMPOUNDS]
        return np.array(COMPOUNDS)[X[cols].to_numpy().argmax(axis=1)]

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        D, comp = self._design(X), self._compound(X)
        self.pooled_ = LinearRegression().fit(D, y)
        self.models_ = {}
        for c in COMPOUNDS:
            m = comp == c
            if m.sum() >= self.min_rows:
                self.models_[c] = LinearRegression().fit(D[m], y[m])
        return self

    def predict(self, X):
        D, comp = self._design(X), self._compound(X)
        out = self.pooled_.predict(D)
        for c, model in self.models_.items():
            m = comp == c
            if m.any():
                out[m] = model.predict(D[m])
        return out


class DeltaTarget(BaseEstimator, RegressorMixin):
    """Fits `estimator` on target - lap t time (sum of lap t's sector times) and adds it back.

    The sector times belong to lap t, so this is no leakage; it lets trees model the change from
    lap t instead of the pace level. predict(X) takes the same X, so it drops into the bundle as is.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def _offset(self, X):
        return X[SECTOR_COLUMNS].sum(axis=1).fillna(self.offset_median_).to_numpy()

    def fit(self, X, y):
        self.offset_median_ = float(X[SECTOR_COLUMNS].sum(axis=1).median())
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y, dtype=float) - self._offset(X))
        return self

    def predict(self, X):
        return self.estimator_.predict(X) + self._offset(X)


# ============================================================
# MODEL BUILDING
# ============================================================
def build_estimator(model, scale, target_mode):
    """Median imputer (fitted per fold; the fallback after feature_engineer.py's stint fill, as in
    train_model.py) -> optional scaler -> model, wrapped for the target mode.

    Steps keep pandas output so custom estimators can select columns by name.
    Returns (estimator, param prefix for the model's grid keys).
    """
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    pipe = Pipeline(steps).set_output(transform="pandas")
    if target_mode == "delta":
        return DeltaTarget(pipe), "estimator__model__"
    if target_mode == "level":
        return pipe, "model__"
    raise ValueError(f"Unknown target mode: {target_mode}")


def make_search(estimator, grid, prefix, inner_folds, n_iter, seed):
    """Grid search when the grid is small, randomized search capped at n_iter otherwise.

    Returns None for a model without a grid (fitted as is).
    """
    if not grid:
        return None
    grid = {prefix + k: v for k, v in grid.items()}
    inner = GroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    common = dict(scoring="neg_mean_absolute_error", cv=inner, n_jobs=-1, refit=True, error_score="raise")
    if len(ParameterGrid(grid)) <= n_iter:
        return GridSearchCV(estimator, grid, **common)
    return RandomizedSearchCV(estimator, grid, n_iter=n_iter, random_state=seed, **common)


def fit_tuned(estimator, grid, prefix, X, y, groups, inner_folds, n_iter, seed):
    """Fit with an inner grouped search (or plainly). Returns (fitted estimator, chosen params)."""
    search = make_search(estimator, grid, prefix, inner_folds, n_iter, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if search is None:
            return clone(estimator).fit(X, y), {}
        search.fit(X, y, groups=groups)
    params = {k[len(prefix):]: v for k, v in search.best_params_.items()}
    return search.best_estimator_, params


# ============================================================
# DATA
# ============================================================
def load_target_rows(dataset_path):
    """Engineered dataset with the shared next-lap target; returns (df, stint group ids)."""
    df = build_next_lap_target(pd.read_csv(dataset_path))
    df[MODEL_FEATURE_COLUMNS] = df[MODEL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    groups = df.groupby(TARGET_GROUP_COLUMNS).ngroup().to_numpy()
    return df, groups


def baseline_predictions(df, y_train_mean):
    """Naive forecasts every model has to beat (#27). Nothing in them is fitted except the mean."""
    persistence = df["fuel_corrected_pace"].to_numpy()
    # On a stint's first lap stint_avg_pace is lap t's own raw time (feature_engineer.py, #33)
    stint_avg = df["stint_avg_pace"].to_numpy()
    return {
        "baseline_persistence": persistence,
        "baseline_stint_avg_pace": stint_avg,
        "baseline_train_mean": np.full(len(df), y_train_mean),
    }


def metrics(y_true, pred):
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "R2": r2_score(y_true, pred),
    }


# ============================================================
# ROLLOUT (mirrors strategy_simulator.rollout_stint / predict_fcp)
# ============================================================
# strategy_simulator.py can't be imported (it loads the bundle and opens its log at import),
# so this reproduces its rollout by feature name. Keep it in step with the simulator.
def predict_fcp(model, features, state, session_weights):
    """Next lap's fcp, averaged over the session one-hots with the training session weights."""
    rows, weights = [], []
    for s in SESSION_COLUMNS:
        row = dict(state)
        for ss in SESSION_COLUMNS:
            row[ss] = 1 if ss == s else 0
        rows.append([row[f] for f in features])
        weights.append(session_weights[s])
    X = pd.DataFrame(rows, columns=features, dtype=float)
    return float(np.average(model.predict(X), weights=weights))


def rollout_stint(model, features, medians, session_weights, compound, set_age, stint_length):
    """fcp per stint lap from lap 1 (the seed lap, training medians) to stint_length."""
    state = {f: medians.get(f, 0.0) for f in MODEL_FEATURE_COLUMNS}
    for cc in COMPOUNDS:
        state[f"compound_{cc}"] = 1 if cc == compound else 0
    state["tyre_age_at_start"] = set_age
    state["laps_since_stint_start"] = 1
    state["tyre_age"] = set_age + 1

    seed = np.array([state[s] for s in SECTOR_COLUMNS], dtype=float)
    share = seed / seed.sum()
    raw = float(seed.sum())
    fcp = raw - FUEL_EFFECT_K * stint_length
    # No earlier laps on stint lap 1, so stint_avg_pace is the lap's own time (as in training)
    state["stint_avg_pace"] = raw

    fcps, raw_laps = [], []
    for age in range(1, stint_length + 1):
        if age > 1:
            fcp = predict_fcp(model, features, state, session_weights)
            raw = fcp + FUEL_EFFECT_K * (stint_length - age + 1)
            state["stint_avg_pace"] = float(np.mean(raw_laps))
            for s, sh in zip(SECTOR_COLUMNS, share):
                state[s] = raw * sh
            state["laps_since_stint_start"] = age
            state["tyre_age"] = set_age + age
        raw_laps.append(raw)
        fcps.append(fcp)
    return np.array(fcps)


def rollout_metrics(curves, slope_laps):
    """Stability and plausibility of one model's rollouts ({compound: fcp per stint lap}).

    max_step: largest lap-to-lap change after the seed lap; drift: last lap - lap 2;
    slope: fcp against stint lap over laps 2..slope_laps (should be > 0: tyres degrade).
    """
    out = {}
    for c, fcp in curves.items():
        tail = fcp[1:]
        lo, hi = 1, min(slope_laps, len(fcp))
        ages = np.arange(lo + 1, hi + 1)
        out[f"{c}_max_step"] = float(np.abs(np.diff(tail)).max()) if len(tail) > 1 else 0.0
        out[f"{c}_drift"] = float(fcp[-1] - fcp[1])
        out[f"{c}_slope"] = float(np.polyfit(ages, fcp[lo:hi], 1)[0]) if len(ages) > 1 else np.nan
        out[f"{c}_range"] = float(tail.max() - tail.min())
    out["max_step"] = max(out[f"{c}_max_step"] for c in curves)
    out["all_slopes_positive"] = all(out[f"{c}_slope"] > 0 for c in curves)
    for lap in (5, slope_laps):
        if lap <= min(len(v) for v in curves.values()):
            order = sorted(curves, key=lambda c: curves[c][lap - 1])
            out[f"order_lap{lap}"] = "<".join(c[0] for c in order)
    out["finite"] = all(np.isfinite(v).all() for v in curves.values())
    return out
