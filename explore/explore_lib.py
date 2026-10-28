# Helpers for explore_master.py: custom estimators, model building, nested grouped CV,
# the fall-off target and its scoring. No settings here: the experiment block in
# explore_master.py and the repo-root .env hold them.
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from feature_config import (  # noqa: E402
    COMPOUNDS, FALL_OFF_ANCHOR_AGE, STINT_GROUP_COLUMNS, compound_of, fall_off_curve, fall_off_target,
    fit_stint_levels, linear_fall_off, multi_lap_rows, stint_shape_errors,
)


# ============================================================
# CUSTOM ESTIMATORS
# ============================================================
class _PerCompound(BaseEstimator, RegressorMixin):
    """One small model per compound on tyre_age; a compound with fewer than min_rows rows falls back
    to one fit on all rows. Needs a DataFrame with the tyre_age and compound_* columns."""

    def _new(self):
        raise NotImplementedError

    def _fit_one(self, age, y):
        raise NotImplementedError

    def _predict_one(self, model, age):
        raise NotImplementedError

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        age, comp = X["tyre_age"].to_numpy(dtype=float), compound_of(X)
        self.pooled_ = self._fit_one(age, y)
        self.models_ = {c: self._fit_one(age[comp == c], y[comp == c])
                        for c in COMPOUNDS if (comp == c).sum() >= self.min_rows}
        return self

    def predict(self, X):
        age, comp = X["tyre_age"].to_numpy(dtype=float), compound_of(X)
        out = self._predict_one(self.pooled_, age)
        for c, model in self.models_.items():
            m = comp == c
            if m.any():
                out[m] = self._predict_one(model, age[m])
        return out


class PerCompoundFallOff(_PerCompound):
    """Fall-off per compound as a polynomial in tyre_age (degree 1 = a straight line per compound)."""

    def __init__(self, degree=1, min_rows=10):
        self.degree = degree
        self.min_rows = min_rows

    def _fit_one(self, age, y):
        return np.polyfit(age, y, self.degree)

    def _predict_one(self, model, age):
        return np.polyval(model, age)


class PerCompoundIsotonic(_PerCompound):
    """Fall-off per compound as a non-decreasing step function of tyre_age (tyres never get faster)."""

    def __init__(self, min_rows=10):
        self.min_rows = min_rows

    def _fit_one(self, age, y):
        return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(age, y)

    def _predict_one(self, model, age):
        return model.predict(age)


# ============================================================
# MODEL BUILDING
# ============================================================
def build_estimator(model, scale):
    """Median imputer (fitted per fold, as train_model.py fills from its training rows) -> optional
    scaler -> model.

    Steps keep pandas output so custom estimators (and monotone constraints) can select columns by name.
    Returns (estimator, param prefix for the model's grid keys).
    """
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps).set_output(transform="pandas"), "model__"


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
# DATA, TARGET AND SCORING (fall-off within the stint, #31)
# ============================================================
def load_fall_off_rows(dataset_path):
    """The laps of stints with two or more clean laps (the only ones that carry fall-off);
    returns (df, stint group ids)."""
    df = multi_lap_rows(pd.read_csv(dataset_path).replace([np.inf, -np.inf], np.nan))
    groups = df.groupby(STINT_GROUP_COLUMNS).ngroup().to_numpy()
    return df, groups


def fold_target(train):
    """Stage 1 on the training rows (as train_model.py): fcp - stint level, and the linear slopes.

    The inner search trains and scores on this target too. Its validation stints' levels come
    from the same fit, which saw their laps, so the inner score is slightly optimistic; it only
    ranks settings, and the outer score doesn't use any level.
    """
    levels, slopes = fit_stint_levels(train)
    return fall_off_target(train, levels), slopes


def baseline_predictions(test, slopes):
    """Fall-off forecasts every model has to beat: none at all (flat), and the stage-1 slope per compound."""
    return {
        "baseline_flat": np.zeros(len(test)),
        "baseline_linear": linear_fall_off(test, slopes),
    }


def shape_metrics(fcp, pred, stint_ids):
    """Error in the shape of the curve within each stint (the stint's mean residual removed, as
    validate_strategy.py). R2_within: share of the within-stint variation explained."""
    err = stint_shape_errors(fcp, pred, stint_ids)
    flat = stint_shape_errors(fcp, np.zeros(len(err)), stint_ids)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2_within": float(1 - np.sum(err ** 2) / np.sum(flat ** 2)),
    }


# ============================================================
# CURVES (the simulator's fall_off_curve, shared through feature_config)
# ============================================================
def curve_metrics(curves, slope_laps):
    """Stability and plausibility of one model's fall-off curves ({compound: fall-off from tyre age 1}).

    max_step: largest change from one tyre age to the next; drift: last age - age 1;
    slope: fall-off against tyre age over ages FALL_OFF_ANCHOR_AGE + 1 .. slope_laps (should be > 0);
    monotone_share: share of steps that don't get faster.
    """
    out = {}
    for c, fall in curves.items():
        hi = min(slope_laps, len(fall))
        ages = np.arange(FALL_OFF_ANCHOR_AGE + 1, hi + 1)
        steps = np.diff(fall)
        out[f"{c}_max_step"] = float(np.abs(steps).max()) if len(steps) else 0.0
        out[f"{c}_drift"] = float(fall[-1] - fall[0])
        out[f"{c}_slope"] = float(np.polyfit(ages, fall[1:hi], 1)[0]) if len(ages) > 1 else np.nan
        out[f"{c}_range"] = float(fall.max() - fall.min())
        out[f"{c}_monotone_share"] = float((steps >= -1e-9).mean()) if len(steps) else 1.0
    out["max_step"] = max(out[f"{c}_max_step"] for c in curves)
    out["all_slopes_positive"] = all(out[f"{c}_slope"] > 0 for c in curves)
    out["finite"] = all(np.isfinite(v).all() for v in curves.values())
    return out


def compound_order(curves, gap, lap):
    """Compounds from fastest to slowest at a stint lap on a new set, gap + fall-off (e.g. "S<M<H")."""
    order = sorted(curves, key=lambda c: gap[c] + curves[c][lap - 1])
    return "<".join(c[0] for c in order)


def fold_curves(model, features, conditions, ages):
    """{compound: fall-off at `ages`} of one fold's model, at that fold's training medians."""
    return {c: fall_off_curve(model, features, c, ages, conditions) for c in COMPOUNDS}
