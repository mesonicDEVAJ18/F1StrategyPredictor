# Diagnostic: TODO list

The pipeline runs end to end (`python master.py` from `scripts/`), but the open items below affect how far the current output can be trusted. None of them are fixed yet.

Items keep a stable ID (`#N`). `readme.txt` marks every place an item applies with a `[TODO #N, P?, see diagnostic.md]` line. When an item is fixed, tick it here, say what changed, and remove or update its tags in `readme.txt`.

**Priorities**
- **P1, results are invalid until fixed.** The training data, the target or the strategy objective is wrong, so the recommendation can't be trusted.
- **P2, method or validation is misleading.** The code doesn't do what `readme.txt` claims, or the validation overstates the model, but the inputs aren't corrupted.
- **P3, completeness and cleanup.** Dead settings, unfinished scope, and the results write-up.

---

## P1: results are invalid until fixed

- [ ] **#1 Weather comes from a different event**
  - **Where:** `data/weather.csv`, and the download URL in `readme.txt` / `README.md`.
  - **Problem:** the weather was downloaded with `meeting_key=1253`, which is Bahrain pre-season testing (sessions 9683–9685, 26–28 Feb 2025). The laps, stints and car data are from `meeting_key=1257`, the Grand Prix weekend (11–12 Apr 2025).
  - **Effect:** the weather features describe a different week. The timestamps can't line up, so `feature_engineer.py` spreads the weather rows evenly over the lap numbers ("pseudo time"). Each lap's weather is therefore effectively arbitrary.
  - **Fix:** download `https://api.openf1.org/v1/weather?meeting_key=1257&csv=true`, then merge per session on timestamps (`date_start` against `date`).
  - **readme.txt:** Data Collection step 4; Feature Engineering, pseudo time alignment.

- [ ] **#2 Stint merge joins on the wrong stint number**
  - **Where:** `scripts/feature_engineer.py`: `assign_stint_numbers()` and the `laps.merge(stints[...], on=["driver_number", "stint_number"])` that follows it.
  - **Problem:** laps get a `global_stint_number`, which runs 1–18 per driver across all three sessions. The merge then matches it against the stints table's own `stint_number`, which runs 1–9 and restarts in every session.
  - **Effect:**
    - A lap can match the same-numbered stint from up to three sessions, so it is duplicated, sometimes with different compounds. The 839 cleaned laps become 1126 rows.
    - Laps whose global number has no per-session match get no `lap_start`, `lap_end`, compound or tyre age, so 507 rows have no `fuel_corrected_pace`.
  - **Fix:** merge on `global_stint_number`, or keep `session_key` and merge on `(session_key, driver_number, stint_number)`.
  - **Depends on:** shares a fix with #3 (keep `session_key`).
  - **readme.txt:** Feature Engineering, stint number and merge; ML Modelling, training (1126 → 613 rows).

- [ ] **#3 Laps are assigned to stints without checking the session**
  - **Where:** `scripts/feature_engineer.py`: the mask in `assign_stint_numbers()`.
  - **Problem:** `session_key` is dropped during cleaning, so a lap is matched on driver and lap-number range only. Lap numbers restart every session, so FP1 lap 5 can be given an FP2 stint that also covers lap 5. Whichever matching stint is processed last wins.
  - **Fix:** keep `session_key` through cleaning and include it in the mask.
  - **Depends on:** shares a fix with #2.
  - **readme.txt:** Data Preparation step 1 (session keys removed); Feature Engineering, stint number.

- [ ] **#4 The "next lap" target can come from another driver**
  - **Where:** `df.groupby("stint_number")["fuel_corrected_pace"].shift(-1)` in both `scripts/train_model.py` and `scripts/validate_strategy.py`.
  - **Problem:** `driver_number` is dropped in `feature_engineer.py`, so the target is shifted within `stint_number` only. Every driver's stint 1 is one group, sorted by lap number, so a row's "next lap" is often another driver's lap.
  - **Fix:** keep a driver/session identifier in the engineered dataset (not as a model input). Group by it plus the stint, and check that the next lap number is exactly the current one plus 1.
  - **readme.txt:** Data Preparation step 3, encoding (driver number removed); Feature Engineering, final drop list; ML Modelling, training.

- [ ] **#7 The simulator mixes units**
  - **Where:** `scripts/strategy_simulator.py`, with the scaling done in `scripts/process_laps.py`.
  - **Problem:** `lap_duration` is standardised in `process_laps.py` (z-scores, one scaler per session), so the model predicts in z-score units. The fuel correction (`0.035`/lap), `FUEL_PENALTY_PER_LAP` and `PIT_LOSS = 22.0` are in seconds.
  - **Effect:** the best strategy totals 95.97 over 57 laps (about 1.7 per lap), so one pit stop costs as much as roughly 13 laps of racing. The optimiser therefore always picks 0 stops. The reported total is not a race time in seconds.
  - **Fix:** either model in seconds (don't scale `lap_duration`), or convert predictions back to seconds with the per-session scaler before simulating.
  - **readme.txt:** Data Preparation step 4, normalization; Race Simulation steps 3 and 4.

- [ ] **#8 Illegal strategies are allowed**
  - **Where:** `generate_strategies()` in `scripts/strategy_simulator.py`.
  - **Problem:** it includes 0-stop strategies and sequences that use a single compound, which the two-compound rule forbids. The current recommendation (HARD, no stop) is one of these.
  - **Fix:** keep only sequences with `len(set(seq)) >= 2`.
  - **readme.txt:** Race Simulation steps 1 and 4.

- [ ] **#10 Tyre age does not reset after a pit stop**
  - **Where:** `precompute_lap_predictions()` / `stint_time()` in `scripts/strategy_simulator.py`.
  - **Problem:** predictions are indexed by race lap, and `laps_since_stint_start` is set to the race lap number. A stint that starts on lap 30 is priced as if its tyres were already 30 laps old.
  - **Fix:** predict each stint from tyre age 1, i.e. index by stint length rather than by race lap.
  - **readme.txt:** Race Simulation step 2.

- [ ] **#11 The pit lap is missing from the race-time sum**
  - **Where:** `simulate_strategy_fast()` in `scripts/strategy_simulator.py`.
  - **Problem:** stints are summed over `(last, p - 1)`, then `last = p + 1`, so lap `p` is never counted. The pit loss replaces a whole lap instead of being added to it.
  - **Effect:** once #7 is fixed, a dropped lap (~90 s) outweighs the 22 s pit loss, so every extra stop would make a strategy look faster.
  - **Fix:** sum every race lap exactly once and add the pit loss on top.
  - **readme.txt:** Race Simulation step 3.

## P2: method or validation is misleading

- [ ] **#9 The simulation does not roll predictions forward**
  - **Where:** `precompute_lap_predictions()` in `scripts/strategy_simulator.py`.
  - **Problem:**
    - Each lap is predicted on its own, with every feature other than the compound and `laps_since_stint_start` fixed at 0.
    - `fuel_corrected_pace` is set to 0.0 as a placeholder. It isn't a model input anyway, because the vector is cut down to `FEATURES`.
    - `readme.txt` says each prediction is fed into the next lap, which doesn't happen.
  - **Effect:** "degradation" comes only from the model's response to `laps_since_stint_start`.
  - **Fix:** roll the lap-*t* feature vector forward from each prediction, or change `readme.txt` to describe the per-lap approach that is actually used.
  - **readme.txt:** ML Modelling step 2 (autoregressive, emergent degradation); Race Simulation step 2; Evaluation, planned critical checks.

- [ ] **#6 The residual plot mixes training and test data**
  - **Where:** `pred_full = model.predict(X)` in `scripts/validate_strategy.py`.
  - **Problem:** the model from the last cross-validation fold predicts every row, including the 80% that fold was trained on, so the residuals look better than they really are.
  - **Fix:** plot out-of-fold predictions (collect each fold's test predictions), or use the held-out split only.
  - **readme.txt:** Evaluation step 1, residual plot.

- [ ] **#5 The t-test is a placeholder**
  - **Where:** the "T-TEST EXAMPLE" block in `scripts/validate_strategy.py`.
  - **Problem:** it compares two sets of random numbers (`np.random.normal(4800, 10)` against `np.random.normal(4820, 10)`), unseeded. The p-value says nothing about the model or the strategy, and it changes every run.
  - **Fix:** replace it with a meaningful test (e.g. per-fold errors of two models, or simulated race times of the top strategies), or remove it.
  - **readme.txt:** Evaluation step 1, t-test.

- [ ] **#13 The saved model is the weakest of the three on the test split**
  - **Where:** `scripts/train_model.py` always saves XGBoost.
  - **Problem:** on the current data, held-out R² is 0.306 for Linear Regression, 0.232 for Random Forest and 0.102 for XGBoost. `readme.txt` justifies XGBoost on qualitative grounds, but the metrics don't currently back that up.
  - **Fix:** re-run the comparison after the P1 data fixes. Then either keep XGBoost with metrics that support it, or save the best model and update the reasoning.
  - **Depends on:** partly downstream of #2, #3 and #4.
  - **readme.txt:** ML Modelling step 3, results table.

## P3: completeness and cleanup

- [ ] **#12 The starting compound setting has no effect**
  - **Where:** `start_features` in `__main__` of `scripts/strategy_simulator.py`.
  - **Problem:** `compound_HARD = 1` is set, but `precompute_lap_predictions()` overwrites the compound one-hots for every compound anyway.
  - **Fix:** remove the setting, or use it as the base feature vector once #9 is done.
  - **readme.txt:** Race Simulation step 2.

- [ ] **#14 The car-data stage is not functional**
  - **Where:** `scripts/process_cardata.py`.
  - **Problem:**
    - It's a copy of `process_laps.py` and looks for `cardata_p*_*.csv`, but the files are named `car_data_p*_*.csv`.
    - It filters on lap columns (`duration_sector_*`, `i1_speed`, `lap_duration`) that car data doesn't have.
    - It isn't called from `master.py`.
  - **Fix:** rename the inputs, clean the actual columns (`brake`, `throttle`, `speed`, `rpm`, `n_gear`, `drs`), and aggregate the ~4 Hz samples per lap by matching each sample's timestamp to the lap's time window. Then wire it into `master.py`.
  - **readme.txt:** Data Collection step 5; Running the Pipeline; Data Preparation, CarData.

- [ ] **#15 Results documentation**
  - **Problem:** results currently exist only as raw files in `scripts/results/<script>/` and a few numbers in the readmes.
  - **Depends on:** all P1 items, plus #9 and #13. The numbers will change once those are fixed, so write this up afterwards.
  - **readme.txt:** Known Issues.
  - Cover:
    - [ ] **Data cleaning:** rows removed per session and per filter (from the `*_removed_rows_summary.csv` files and the weather z-score summary), and the final row counts.
    - [ ] **Feature engineering:** final dataset shape, how many laps matched a stint, and how many rows are usable for training.
    - [ ] **Model comparison:** test MAE, RMSE and R² for all three models, in seconds rather than z-scores, with the metric chart and prediction overlay plots and what they show.
    - [ ] **Feature importance:** top features for Random Forest and XGBoost, and whether they match tyre physics (tyre age, compound, track temperature).
    - [ ] **Validation:** CV results per fold with the spread, the residual plot and what it shows about bias, and the t-test replaced with a meaningful test or removed (#5).
    - [ ] **Strategy:** recommended strategy (compound sequence and pit laps), predicted race time in seconds, the next-best alternatives and the gap to them, and predicted pace by compound across the race.
    - [ ] **Limitations:** a short note on which results depend on items still open above.
    - [ ] Decide where this lives (a results section in `readme.txt` / `README.md`, or a separate `RESULTS.md`) and link it from both readmes.
