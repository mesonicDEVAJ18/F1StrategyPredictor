# Diagnostic: known bugs

The pipeline runs end to end (`python master.py` from `scripts/`), but the issues below affect how far the current output can be trusted. None of them are fixed yet.

## Data

### 1. Weather comes from a different event
- **Where:** `data/weather.csv`, download URL in `readme.txt` / `README.md`
- **What:** the weather was downloaded with `meeting_key=1253`. That is Bahrain pre-season testing (sessions 9683–9685, 26–28 Feb 2025). The laps, stints and car data are from `meeting_key=1257`, the Grand Prix weekend (11–12 Apr 2025).
- **Effect:** the weather features describe a different week. Because the timestamps can't line up, `feature_engineer.py` spreads the weather rows evenly over the lap numbers ("pseudo time"), so each lap's weather is effectively arbitrary.
- **Fix direction:** download `https://api.openf1.org/v1/weather?meeting_key=1257&csv=true`, then merge on timestamps (`date_start` against `date`) per session.

## Feature engineering (`scripts/feature_engineer.py`)

### 2. Stint merge joins on the wrong stint number
- **Where:** `assign_stint_numbers()` and the `laps.merge(stints[...], on=["driver_number", "stint_number"])` that follows it.
- **What:** laps are given `global_stint_number`, which runs 1–18 per driver across all three sessions. The merge then matches that against the stints table's own `stint_number`, which runs 1–9 and restarts in every session.
- **Effect:**
  - A lap can match the same-numbered stint from up to three sessions, so it is duplicated, sometimes with different compounds: the 839 cleaned laps become 1126 rows.
  - Laps whose global number has no per-session match get no `lap_start`, `lap_end`, compound or tyre age, so 507 rows have no `fuel_corrected_pace`.
- **Fix direction:** merge on `global_stint_number`, or keep `session_key` and merge on `(session_key, driver_number, stint_number)`.

### 3. Laps are assigned to stints without checking the session
- **Where:** the mask in `assign_stint_numbers()`.
- **What:** `session_key` is dropped during cleaning, so a lap is matched only on driver and lap-number range. Lap numbers restart every session, so FP1 lap 5 can be assigned an FP2 stint that also covers lap 5. Whichever matching stint is processed last wins.
- **Fix direction:** keep `session_key` through cleaning and include it in the mask.

## Training and validation (`scripts/train_model.py`, `scripts/validate_strategy.py`)

### 4. The "next lap" target can come from another driver
- **Where:** `df.groupby("stint_number")["fuel_corrected_pace"].shift(-1)` in both scripts.
- **What:** `driver_number` is dropped in `feature_engineer.py`, so the target is shifted within `stint_number` only. Every driver's stint 1 is one group, sorted by lap number, so a row's "next lap" is often another driver's lap.
- **Fix direction:** keep a driver/session identifier in the engineered dataset (not as a model input), group by it plus the stint, and also check that the next lap number is exactly the current one plus 1.

### 5. The t-test is a placeholder
- **Where:** the "T-TEST EXAMPLE" block in `validate_strategy.py`.
- **What:** it compares two sets of random numbers (`np.random.normal(4800, 10)` against `np.random.normal(4820, 10)`), unseeded. The p-value says nothing about the model or the strategy and changes every run.

### 6. The residual plot mixes training and test data
- **Where:** `pred_full = model.predict(X)` in `validate_strategy.py`.
- **What:** it uses the model from the last cross-validation fold to predict every row, including the 80% that fold was trained on, so the residuals look better than they really are.

## Strategy simulator (`scripts/strategy_simulator.py`)

### 7. Units are mixed
- **What:** `lap_duration` is standardised in `process_laps.py` (z-scores, one scaler per session), so the model predicts in z-score units. The fuel correction (`0.035`/lap), `FUEL_PENALTY_PER_LAP` and `PIT_LOSS = 22.0` are in seconds.
- **Effect:** the best strategy totals 95.97 over 57 laps (about 1.7 per lap), so one pit stop costs about as much as 13 laps of racing. The optimiser therefore always picks 0 stops. The reported total is not a race time in seconds.
- **Fix direction:** either model in seconds (don't scale `lap_duration`), or convert predictions back to seconds with the per-session scaler before simulating.

### 8. Illegal strategies are allowed
- **Where:** `generate_strategies()`.
- **What:** it includes 0-stop strategies and sequences that use a single compound, which the two-compound rule forbids. The current recommendation (HARD, no stop) is one of these.
- **Fix direction:** keep only sequences with `len(set(seq)) >= 2`.

### 9. The simulation does not roll predictions forward
- **Where:** `precompute_lap_predictions()`.
- **What:**
  - Each lap is predicted on its own, with every feature other than the compound and `laps_since_stint_start` fixed at 0.
  - `fuel_corrected_pace` is set to 0.0 as a placeholder. It isn't a model input anyway; the vector is cut down to `FEATURES`.
  - `readme.txt` describes feeding each prediction into the next lap, which doesn't happen.
- **Effect:** "degradation" comes only from the model's response to `laps_since_stint_start`.

### 10. Tyre age does not reset after a pit stop
- **Where:** `precompute_lap_predictions()` / `stint_time()`.
- **What:** predictions are indexed by race lap, and `laps_since_stint_start` is set to the race lap number. A stint that starts on lap 30 is priced as if its tyres were already 30 laps old.
- **Fix direction:** predict each stint from tyre age 1, i.e. index by stint length rather than race lap.

### 11. The pit lap is missing from the race-time sum
- **Where:** `simulate_strategy_fast()`.
- **What:** stints are summed over `(last, p - 1)`, then `last = p + 1`, so lap `p` is never counted. The pit loss effectively replaces a whole lap instead of being added to it.

### 12. The starting compound setting has no effect
- **Where:** `start_features` in `__main__`.
- **What:** `compound_HARD = 1` is set, but `precompute_lap_predictions()` overwrites the compound one-hots for every compound anyway.

## Model choice

### 13. The saved model is the weakest of the three on the test split
- **Where:** `train_model.py` always saves XGBoost.
- **What:** on the current data, held-out R² is 0.306 for Linear Regression, 0.232 for Random Forest and 0.102 for XGBoost. `readme.txt` justifies XGBoost on qualitative grounds, but the metrics don't currently back that up. This is partly downstream of bugs 2–4.

## Car data (`scripts/process_cardata.py`)

### 14. The car-data stage is not functional
- **What:**
  - It's a copy of `process_laps.py` and looks for `cardata_p*_*.csv`, but the files are named `car_data_p*_*.csv`.
  - It filters on lap columns (`duration_sector_*`, `i1_speed`, `lap_duration`) that car data doesn't have.
  - It isn't called from `master.py`.
- **Fix direction:** rename the inputs, clean the actual columns (`brake`, `throttle`, `speed`, `rpm`, `n_gear`, `drs`), and aggregate the ~4 Hz samples per lap by matching each sample's timestamp to the lap's time window.
