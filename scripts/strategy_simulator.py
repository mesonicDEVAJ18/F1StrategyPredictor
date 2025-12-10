import pandas as pd
import numpy as np
import itertools
import joblib
import warnings

from feature_config import MODEL_FEATURE_COLUMNS, SIM_FEATURE_COLUMNS

# LOAD MODEL
bundle = joblib.load("./results/xgb_model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]    # MODEL_FEATURE_COLUMNS

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
PIT_LOSS = 22.0
FUEL_PENALTY_PER_LAP = 0.035
NUM_LAPS = 57

# --- PRECOMPUTE ---
def precompute_lap_predictions(total_laps, base_features):
    preds = {c: [] for c in COMPOUNDS}

    for c in COMPOUNDS:
        feats = base_features.copy()

        for cc in COMPOUNDS:
            feats[f"compound_{cc}"] = 1 if cc == c else 0

        for lap in range(1, total_laps + 1):
            feats["laps_since_stint_start"] = lap

            # Always include dummy fuel_corrected_pace
            feats["fuel_corrected_pace"] = 0.0

            # Build aligned feature vector
            vec = np.array([feats[f] for f in SIM_FEATURE_COLUMNS], dtype=float).reshape(1, -1)

            # Model only uses first MODEL_FEATURE_COLUMNS
            model_vec = vec[:, :len(FEATURES)]

            fcp = model.predict(model_vec)[0]
            lap_time = fcp + (total_laps - lap + 1) * FUEL_PENALTY_PER_LAP

            preds[c].append(lap_time)

    prefix = {c: np.cumsum(preds[c]) for c in COMPOUNDS}
    return preds, prefix