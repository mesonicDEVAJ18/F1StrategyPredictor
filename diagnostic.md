# Diagnostic: TODO list

The pipeline runs end to end (`python master.py` from `scripts/`), but the open items below affect how far the current output can be trusted. Fixed items are ticked and say what changed.

Items keep a stable ID (`#N`). `readme.txt` marks every place an item applies with a `[TODO #N, P?, see diagnostic.md]` line. When an item is fixed, tick it here, say what changed, and remove or update its tags in `readme.txt`.

**Priorities**
- **P1, results are invalid until fixed.** The training data, the target or the strategy objective is wrong, so the recommendation can't be trusted.
- **P2, method or validation is misleading.** The code doesn't do what `readme.txt` claims, or the validation overstates the model, but the inputs aren't corrupted.
- **P3, completeness and cleanup.** Dead settings, unfinished scope, and the results write-up.

---

## P1: results are invalid until fixed

- [x] **#1 Weather comes from a different event**
  - **Where:** `data/weather.csv`, and the download URL in `readme.txt` / `README.md`.
  - **Problem:** the weather was downloaded with `meeting_key=1253`, which is Bahrain pre-season testing (sessions 9683–9685, 26–28 Feb 2025). The laps, stints and car data are from `meeting_key=1257`, the Grand Prix weekend (11–12 Apr 2025).
  - **Effect:** the weather features describe a different week. The timestamps can't line up, so `feature_engineer.py` spreads the weather rows evenly over the lap numbers ("pseudo time"). Each lap's weather is therefore effectively arbitrary.
  - **Fix:** download `https://api.openf1.org/v1/weather?meeting_key=1257&csv=true`, then merge per session on timestamps (`date_start` against `date`).
  - **readme.txt:** Data Collection step 4; Feature Engineering, pseudo time alignment.
  - **Fixed:** `data/weather.csv` re-downloaded for `meeting_key=1257`. `process_weather.py` keeps only the practice sessions (the `session_key`s in `laps_p*.csv`), dropping qualifying and race. `feature_engineer.py` replaces pseudo time with `merge_asof` on `date_start` ~ `date` (nearest, 5 min tolerance); all 839 laps match, within 30 s. Held-out R² went from 0.306/0.232/0.102 to 0.425/0.474/0.419 (LR/RF/XGB).

- [x] **#2 Stint merge joins on the wrong stint number**
  - **Where:** `scripts/feature_engineer.py`: `assign_stint_numbers()` and the `laps.merge(stints[...], on=["driver_number", "stint_number"])` that follows it.
  - **Problem:** laps get a `global_stint_number`, which runs 1–18 per driver across all three sessions. The merge then matches it against the stints table's own `stint_number`, which runs 1–9 and restarts in every session.
  - **Effect:**
    - A lap can match the same-numbered stint from up to three sessions, so it is duplicated, sometimes with different compounds. The 839 cleaned laps become 1126 rows.
    - Laps whose global number has no per-session match get no `lap_start`, `lap_end`, compound or tyre age, so 507 rows have no `fuel_corrected_pace`.
  - **Fix:** merge on `global_stint_number`, or keep `session_key` and merge on `(session_key, driver_number, stint_number)`.
  - **Depends on:** shares a fix with #3 (keep `session_key`).
  - **readme.txt:** Feature Engineering, stint number and merge; ML Modelling, training (1126 → 613 rows).
  - **Fixed** (together with #3): the merge now uses the stints' `global_stint_number`, renamed to `stint_number`, on `(driver_number, stint_number)`. The global number is unique per driver, so the merge is one-to-one (`validate="many_to_one"`). All 839 cleaned laps keep exactly one row, all of them get a stint, and none is missing `fuel_corrected_pace`. Every lap's compound and lap range were checked against the raw `stints_p*.csv` of its own session, with 0 mismatches. 821 rows remain after the target shift (was 613).
    - **Effect on metrics:** held-out R² fell from 0.425/0.474/0.419 to 0.043/0.060/−0.039 (LR/RF/XGB). The old figures were inflated by the duplicated rows. The target is still mostly cross-driver (#4): only 5.5% of target pairs are the same driver's next lap (was 2.0%). As a diagnostic only, building the target per `(session_key, driver_number, stint_number)` with lap + 1 gives 514 rows and RF R² 0.617.

- [x] **#3 Laps are assigned to stints without checking the session**
  - **Where:** `scripts/feature_engineer.py`: the mask in `assign_stint_numbers()`.
  - **Problem:** `session_key` is dropped during cleaning, so a lap is matched on driver and lap-number range only. Lap numbers restart every session, so FP1 lap 5 can be given an FP2 stint that also covers lap 5. Whichever matching stint is processed last wins.
  - **Fix:** keep `session_key` through cleaning and include it in the mask.
  - **Depends on:** shares a fix with #2.
  - **readme.txt:** Data Preparation step 1 (session keys removed); Feature Engineering, stint number.
  - **Fixed:** `process_laps.py` and `process_stints.py` now drop only `meeting_key`, so `session_key` reaches `data/laps_combined_clean.csv` and `data/stints_combined_clean.csv`. `assign_stint_numbers()` adds `session_key` to the mask. It also numbers the global stints in order of `(driver_number, session_key, stint_number)` rather than `lap_start`, which interleaved the sessions. `feature_engineer.py` drops `session_key` from the final dataset along with `driver_number`, so the engineered dataset's schema is unchanged. #4 decides which id to keep there.

- [x] **#4 The "next lap" target can come from another driver**
  - **Where:** `df.groupby("stint_number")["fuel_corrected_pace"].shift(-1)` in both `scripts/train_model.py` and `scripts/validate_strategy.py`.
  - **Problem:** `driver_number` is dropped in `feature_engineer.py`, so the target is shifted within `stint_number` only. Every driver's stint 1 is one group, sorted by lap number, so a row's "next lap" is often another driver's lap.
  - **Fix:** keep a driver/session identifier in the engineered dataset (not as a model input). Group by it plus the stint, and check that the next lap number is exactly the current one plus 1.
  - **readme.txt:** Data Preparation step 3, encoding (driver number removed); Feature Engineering, final drop list; ML Modelling, training.
  - **Fixed:** `feature_engineer.py` keeps `session_key` and `driver_number` in the engineered dataset as identifiers. They are not in `MODEL_FEATURE_COLUMNS`, so the model still has the same 22 inputs. `feature_config.py` has a shared `build_next_lap_target()`, used by both `train_model.py` and `validate_strategy.py`. It groups by `(driver_number, stint_number)` (the global stint number, so one stint in one session) and keeps a row only if the next row is `lap_number + 1`.
    - **Rows:** 839 → 514. 260 are the last lap of a stint and 65 have a gap to the next lap (laps removed in cleaning). All 514 pairs were checked: same driver, stint and session, and lap + 1.
    - **Effect on metrics:** held-out R² rose from 0.043/0.060/−0.039 to 0.534/0.625/0.652 (LR/RF/XGB), and MAE to 0.504/0.435/0.419 (z-score units). XGBoost 5-fold CV MAE is 0.409 and RMSE 0.578. Part of this may be leakage through `stint_avg_pace` (#17).

- [x] **#7 The simulator mixes units**
  - **Where:** `scripts/strategy_simulator.py`, with the scaling done in `scripts/process_laps.py`.
  - **Problem:** `lap_duration` is standardised in `process_laps.py` (z-scores, one scaler per session), so the model predicts in z-score units. The fuel correction (`0.035`/lap), `FUEL_PENALTY_PER_LAP` and `PIT_LOSS = 22.0` are in seconds.
  - **Effect:** the best strategy totals 84.89 over 57 laps (about 1.49 per lap), so one pit stop costs as much as roughly 15 laps of racing. The optimiser therefore always picks 0 stops. The reported total is not a race time in seconds.
  - **Fix:** either model in seconds (don't scale `lap_duration`), or convert predictions back to seconds with the per-session scaler before simulating.
  - **readme.txt:** Data Preparation step 4, normalization; Race Simulation steps 3 and 4.
  - **Fixed:** the model now works in seconds, so the target, the fuel term, the pit loss and the metrics share one unit.
    - **No lap scaling:** `process_laps.py` no longer scales `lap_duration`, the sector times or the speeds. Linear Regression, Random Forest and XGBoost don't change under a global rescale. The scaler's only real effect was the per-session part, which removed each session's pace level and left no race session to convert back to. Converting back with pooled session statistics was considered and rejected: the race pace level would have been a fixed convention, not something the model learns.
    - **Session one-hot:** `process_laps.py` tags each lap with `session_name` (FP1/FP2/FP3, from the `laps_p*` file name). `feature_engineer.py` one-hot encodes it to `session_FP1/FP2/FP3` (`SESSION_COLUMNS` in `feature_config.py`), a model input, so session-level pace offsets are separated from conditions and tyres. `session_key` stays an identifier. 25 model inputs (was 22).
    - **Simulator:**
      - The base feature vector is the training medians (saved in the bundle as `medians`). All-zero inputs only meant "average" while everything was a z-score.
      - The race is not a practice session, so each lap is predicted under each session one-hot and averaged, weighted by each session's share of training rows (bundle `session_weights`: FP1 0.407, FP2 0.385, FP3 0.208).
      - Totals are labelled in seconds (`"time_units": "seconds"` in `best_strategy.json`).
    - **Checks:** the sector times sum to `lap_duration` exactly. `fuel_corrected_pace − 0.035 × (lap_end − lap_number)` equals the raw `lap_duration` of `laps_p*.csv` for all 839 laps. Every row has exactly one session one-hot, matching its `session_key`.
    - **Effect on metrics** (now in seconds): held-out MAE 12.12 / 10.65 / 10.51 s, RMSE 16.61 / 15.50 / 15.13 s and R² 0.586 / 0.640 / 0.656 (LR/RF/XGB; R² was 0.534 / 0.625 / 0.652). XGBoost 5-fold CV MAE 9.81 s, RMSE 14.05 s. The errors are large in seconds because the target still includes slow non-push laps (#20). The session one-hots have XGB importance 0.020 / 0.009 / 0.045. `stint_avg_pace` is still top at 0.260 (#17).
    - **Effect on strategy:** predicted laps are 103–112 s and the best total is 6055.98 s (SOFT–SOFT–SOFT, pits on laps 10 and 11). This is still **not** a valid recommendation. Pitting on consecutive laps leaves an empty middle stint, and the two laps never summed are worth more than the 44 s of pit loss (#11). The middle compound doesn't matter, so SOFT–MEDIUM–SOFT and SOFT–HARD–SOFT tie (#8).

- [ ] **#8 Illegal strategies are allowed**
  - **Where:** `generate_strategies()` in `scripts/strategy_simulator.py`.
  - **Problem:** it includes 0-stop strategies and sequences that use a single compound, which the two-compound rule forbids. The current recommendation (HARD, no stop) is one of these.
  - **Fix:** keep only sequences with `len(set(seq)) >= 2`.
  - **readme.txt:** Race Simulation steps 1 and 4.

- [x] **#10 Tyre age does not reset after a pit stop**
  - **Where:** `precompute_lap_predictions()` / `stint_time()` in `scripts/strategy_simulator.py`.
  - **Problem:** predictions are indexed by race lap, and `laps_since_stint_start` is set to the race lap number. A stint that starts on lap 30 is priced as if its tyres were already 30 laps old.
  - **Fix:** predict each stint from tyre age 1, i.e. index by stint length rather than by race lap.
  - **readme.txt:** Race Simulation step 2.
  - **Fixed** (together with #9): `precompute_stint_rollouts()` rolls every compound out for every stint length L = 1..57, always from tyre age 1 (`laps_since_stint_start` = 1 on the stint's first lap). `stint_time()` prices a stint as the summed predicted fcp of the L-lap rollout plus the race fuel term (`FUEL_PENALTY_PER_LAP` × fuel on board) of the race laps it covers. So a stint costs the same wherever it starts, apart from the fuel. Race stints start on a new set (`RACE_TYRE_AGE_AT_START = 0`, see #21); the training median was 3.
    - **Training range:** `laps_since_stint_start` is 2–18 in training (pit-out laps are cleaned out). Age 1 is treated like 2. Beyond 18, XGBoost's response to tyre age is flat, so any later change comes from the fed-back inputs (#9).

- [ ] **#11 The pit lap is missing from the race-time sum**
  - **Where:** `simulate_strategy_fast()` in `scripts/strategy_simulator.py`.
  - **Problem:** stints are summed over `(last, p - 1)`, then `last = p + 1`, so lap `p` is never counted. The pit loss replaces a whole lap instead of being added to it.
  - **Effect:** now that #7 is fixed, a dropped lap (~100–110 s) outweighs the 22 s pit loss, so every extra stop makes a strategy look faster. The optimiser picks 2 stops on consecutive laps (12 and 13 since #9/#10; 10 and 11 before). The middle "stint" is empty and two laps go missing from the total.
  - **Fix:** sum every race lap exactly once and add the pit loss on top.
  - **readme.txt:** Race Simulation step 3.

- [ ] **#19 The fuel correction has the wrong sign and horizon**
  - **Where:** `fuel_corrected_pace` in `scripts/feature_engineer.py`.
  - **Problem:**
    - `fuel_corrected_pace = lap_duration + 0.035 × laps_remaining` adds the fuel effect. Laps with more fuel on board are already slower, so the correction should subtract it.
    - `laps_remaining = lap_end - lap_number` counts the laps left in the *stint*, not the fuel on board. In practice, the fuel load depends on the run plan, which the data doesn't record.
  - **Effect:** the target is not fuel-free pace. The simulator still adds race fuel (`0.035 ×` laps of fuel remaining) on top of it, so early-stint fuel is counted twice in training and differently in simulation.
  - **Fix:** subtract the term, and decide on a fuel proxy (e.g. laps left in the stint as an approximation of the run's fuel, stated as an assumption), or drop the correction and let the model learn it from `laps_since_stint_start`.
  - **readme.txt:** Feature Engineering, fuel_corrected_pace.

## P2: method or validation is misleading

- [x] **#17 `stint_avg_pace` contains the target lap**
  - **Where:** the leave-one-out `stint_avg_pace` in `scripts/feature_engineer.py`.
  - **Problem:** it averages `lap_duration` over every other lap of the stint, and that includes lap *t+1*. The target is lap *t+1*'s `fuel_corrected_pace`, which is that same `lap_duration` plus a fuel term, so part of the target is fed in as an input.
  - **Effect:** it is the top feature by far (importance 0.25 XGB, 0.46 RF), so the R² of ~0.65 after #4 is probably inflated. The simulator sets it to 0, so the model's main signal is missing when it predicts.
  - **Fix:** use only laps before *t* (an expanding mean of the stint so far, NaN or median on the first lap), or drop the feature.
  - **readme.txt:** Feature Engineering, stint_avg_pace.
  - **Fixed:** `feature_engineer.py` now builds `stint_avg_pace` as the mean `lap_duration` of the stint's earlier laps only (per `(driver_number, stint_number)`, ordered by `lap_number`: `shift(1).expanding().mean()`). Lap *t* itself is left out, since it is already an input through the sector times. This is the definition the simulator's rollout already used (#9), so training and simulation now agree.
    - **First lap:** NaN on the first cleaned lap of each stint (260 rows). `train_model.py` already fills it with the training median, which is also the simulator's lap-1 seed. `validate_strategy.py` passed NaN straight to XGBoost; it now fills with the bundle's `medians` too, so CV sees the same inputs as training.
    - **Checks:** on all 839 laps, the value equals the mean of the earlier laps of the same driver and stint (0 mismatches), and it is NaN on exactly the 260 first laps. Rows are unchanged: 839 engineered, 514 after the target.
    - **Effect on metrics:** held-out MAE 16.62 / 12.54 / 12.06 s, RMSE 20.44 / 17.03 / 15.91 s and R² 0.373 / 0.565 / 0.620 (LR/RF/XGB; was 0.586 / 0.640 / 0.656). XGBoost 5-fold CV MAE 12.02 s, RMSE 16.32 s (was 9.81 / 14.05 s). XGBoost is still the best of the three (#13). These still use the shuffled per-lap split (#18).
    - **Feature importance:** `stint_avg_pace` fell from 0.260 to 0.045 (XGB, now 10th) and from 0.492 to 0.037 (RF, now 7th). XGB's top five are now `track_temperature` 0.110, `i2_speed` 0.106, `st_speed` 0.073, `duration_sector_3` 0.072 and `pressure` 0.065. RF's are led by `i2_speed` 0.233 and `duration_sector_3` 0.212.
    - **Rollout behaviour** (57-lap stint, fcp in s):
      - SOFT: 100.94 at age 1, 109.30 at 18, 109.72 at 57
      - MEDIUM: 100.94 → 106.62 → 110.23
      - HARD: 100.94 → 108.72 → 109.84
      - Each compound stays near 99–102 s for a few laps, then steps up by 7–8 s over 2–4 laps (HARD from age ~5, SOFT ~11, MEDIUM ~16) and plateaus near 109–110 s. The early ±3 s zig-zag is gone; the largest lap-to-lap step is 2.8–5.2 s (was 5.7–6.4 s). All rollouts stay within 98.98–110.39 s.
      - HARD degrading first and MEDIUM last is still not tyre physics (#18, #20).
    - **Effect on strategy:** the best total is now 6023.36 s, HARD–HARD–SOFT with pits on laps 12 and 24 (was SOFT–SOFT–SOFT, 12 and 13, 5695.12 s). Neighbouring pit laps are within 1 s. The best 1-stop is HARD–SOFT, pit on lap 11, 6144.94 s, and the best 0-stop is MEDIUM at 6178.24 s. Each extra stop still drops its ~100–110 s pit lap for 22 s of pit loss, so #11 still favours more stops and this is still **not** a valid recommendation. The winner happens to use two compounds, but #8 is still open.

- [ ] **#18 Train/test splits mix laps from the same stint**
  - **Where:** `train_test_split(..., shuffle=True)` in `scripts/train_model.py` and `KFold(n_splits=5, shuffle=True)` in `scripts/validate_strategy.py`.
  - **Problem:** rows are shuffled one lap at a time, so most stints have laps in both the training and the test set. Consecutive laps of one stint are highly correlated (same driver, car, tyre, fuel and conditions), and before #17 was fixed the test laps' own lap times were spread through the training rows.
  - **Effect:** the test and CV scores measure interpolation within stints the model has already seen, not prediction on a new stint (which is what the simulator does). They overstate the model.
  - **Fix:** split by stint with `GroupShuffleSplit` / `GroupKFold`, grouped on `(driver_number, stint_number)` (or by session, or by driver, for a stricter test). Both id columns are in the engineered dataset since #4.
  - **readme.txt:** ML Modelling, training (80/20 split); Evaluation step 1, cross-validation.

- [ ] **#20 The cleaned laps still include slow non-push laps**
  - **Where:** the filters in `scripts/process_laps.py`.
  - **Problem:** the filters drop pit-out laps, invalid sectors, laps over 2× the median and IQR residual outliers. Cool-down, in- and traffic laps under 2× the median remain. Cleaned laps have a median of about 100–117 s per session, but a mean of 111–121 s and a 90th percentile of about 147–159 s.
  - **Effect:** the target mixes push laps with slow laps, so the errors in seconds are large (XGB MAE about 10.5 s). The simulator's median-feature lap (103–112 s) is slower than race pace.
  - **Fix:** flag or drop non-push laps, e.g. laps more than a set margin slower than the driver's best on that compound in the session, or train only on long-run stints.
  - **readme.txt:** Data Preparation, laps cleaning.

- [x] **#9 The simulation does not roll predictions forward**
  - **Where:** `precompute_lap_predictions()` in `scripts/strategy_simulator.py`.
  - **Problem:**
    - Each lap is predicted on its own, with every feature other than the compound, the session one-hot and `laps_since_stint_start` fixed at its training median (fixed at 0 before #7).
    - `fuel_corrected_pace` is set to 0.0 as a placeholder. It isn't a model input anyway, because the vector is cut down to `FEATURES`.
    - `readme.txt` says each prediction is fed into the next lap, which doesn't happen.
  - **Effect:** "degradation" comes only from the model's response to `laps_since_stint_start`.
  - **Fix:** roll the lap-*t* feature vector forward from each prediction, or change `readme.txt` to describe the per-lap approach that is actually used.
  - **readme.txt:** ML Modelling step 2 (autoregressive, emergent degradation); Race Simulation step 2; Evaluation, planned critical checks.
  - **Fixed** (together with #10): `rollout_stint()` in `scripts/strategy_simulator.py` rolls each stint forward lap by lap.
    - **Seed:** lap 1 of a stint is a median lap. Its state is the training medians (bundle `medians`) with the compound set, and its lap time is the sum of the median sector times (98.98 s).
    - **Step:** the model predicts lap *a+1*'s fcp from lap *a*'s state (the training relationship), still averaged over the session one-hots (`predict_fcp()`).
    - **Feedback:** the prediction is turned back into a raw lap time by undoing training's fuel term, `raw = fcp − FUEL_EFFECT_K × (L − age)` for a stint of L laps. This needs one rollout per (compound, L): 4959 laps in total, about 6 s. It then becomes the next state:
      - the sector times are the median sector shares scaled to sum to `raw`
      - `stint_avg_pace` is the mean of the stint's earlier raw laps (the causal form #17 proposed; training uses it too since #17 was fixed)
      - `laps_since_stint_start` goes up by one
      - `fuel_corrected_pace` holds the lap's fcp instead of a 0.0 dummy
    - **Held fixed:** speeds, segment counts and weather stay at the medians. The k = 0.035 used to build fcp is now `FUEL_EFFECT_K` in `feature_config.py`, shared by `feature_engineer.py` and the simulator.
    - **Outputs:** `strategy_simulator_stint_rollouts.csv` has one row per compound, stint length and tyre age, with fcp, raw lap time, sectors and `stint_avg_pace`. `strategy_simulator_stint_totals.csv` is compound × stint length. They replace `_precomputed_predictions.csv` and `_prefix_sums.csv`.
    - **Checks:**
      - sectors sum to the raw lap
      - `fcp − raw = 0.035 × (L − age)` on every row
      - every age-1 lap is the seed
      - `stint_avg_pace` is the mean of the earlier raw laps
      - stint totals equal the rollout sums
    - **Rollout behaviour** (57-lap stint, fcp in s; logged on every run):
      - SOFT: 100.94 at age 1, 99.34 at 18, 114.05 at 57
      - MEDIUM: 100.94 → 114.82 → 118.40
      - HARD: 100.94 → 115.06 → 119.42
      - The largest lap-to-lap step is 5.7–6.4 s. For the first ~8 laps the predictions zig-zag by ±3 s: a slow fed-back lap is followed by a fast prediction.
      - MEDIUM and HARD step up by about 12 s around ages 11–15. SOFT stays near 99 s until about age 48.
      - SOFT degrading least is not tyre physics. It is what the model has learnt from FP data that still has #17 and #20 open.
      - Predictions stay within 96.1–120.1 s over all rollouts, so there's no runaway drift.
    - **Effect on strategy:** the best total went from 6055.98 s to 5695.12 s, now SOFT–SOFT–SOFT with pits on laps 12 and 13. SOFT–MEDIUM–SOFT and SOFT–HARD–SOFT tie with it, and SOFT–SOFT–SOFT with pits on 45 and 46 is 2.3 s behind. The best 1-stop is SOFT–SOFT, pit on lap 13, 5775.17 s. The best 0-stop is SOFT at 5847.05 s. It is still **not** a valid recommendation, because of #8 and #11.

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
  - **Problem:** `readme.txt` justifies XGBoost on qualitative grounds. Before #4 was fixed, the held-out R² was 0.043 for Linear Regression, 0.060 for Random Forest and −0.039 for XGBoost. After #4, XGBoost is now the best of the three: 0.534 for LR, 0.625 for RF and 0.652 for XGB. After #7 (seconds, 25 inputs with the session one-hot), it is 0.586 / 0.640 / 0.656, with MAE 12.12 / 10.65 / 10.51 s. After #17 (causal `stint_avg_pace`), it is 0.373 / 0.565 / 0.620, with MAE 16.62 / 12.54 / 12.06 s: XGBoost's lead over Random Forest grew, but it still comes from one shuffled split.
  - **Fix:** re-run the comparison once #18 is fixed. Then either keep XGBoost with metrics that support it, or save the best model and update the reasoning.
  - **Depends on:** #18 (#2, #3, #4 and #17 are fixed).
  - **readme.txt:** ML Modelling step 3, results table.

## P3: completeness and cleanup

- [ ] **#12 The starting compound setting has no effect**
  - **Where:** `start_features` in `__main__` of `scripts/strategy_simulator.py`.
  - **Problem:** `compound_HARD = 1` is set, but `precompute_lap_predictions()` (now `rollout_stint()`) overwrites the compound one-hots for every compound anyway. Since #7, `start_features` starts from the training medians, not zeros, but the compound override is still dead.
  - **Fix:** remove the setting. #9 is done and seeds every stint from the medians with the stint's own compound, so the setting has no remaining use.
  - **readme.txt:** Race Simulation step 2.

- [x] **#14 The car-data stage is not functional**
  - **Where:** `scripts/process_cardata.py`.
  - **Problem:**
    - It's a copy of `process_laps.py` and looks for `cardata_p*_*.csv`, but the files are named `car_data_p*_*.csv`.
    - It filters on lap columns (`duration_sector_*`, `i1_speed`, `lap_duration`) that car data doesn't have.
    - It isn't called from `master.py`.
  - **Fix:** rename the inputs, clean the actual columns (`brake`, `throttle`, `speed`, `rpm`, `n_gear`, `drs`), and aggregate the ~4 Hz samples per lap by matching each sample's timestamp to the lap's time window. Then wire it into `master.py`.
  - **readme.txt:** Data Collection step 5; Running the Pipeline; Data Preparation, CarData.
  - **Fixed:** `process_cardata.py` rewritten to read `car_data_p*_*.csv` and keep `session_key`.
    - **Sample cleaning:** drops duplicate timestamps, no-input samples (`brake`/`throttle` = 104, about a third of all samples, all stationary) and out-of-range readings.
    - **Encoding:** `brake` → `braking` (0/1), `drs` codes → `drs_open` (0/1 for 10/12/14).
    - **Per-lap aggregation:** each sample is assigned to its lap window `[date_start, date_start + lap_duration)` from the raw `laps_p*.csv` with `merge_asof` by `(session_key, driver_number)`. Samples are then aggregated per lap: speed, throttle, full-throttle share, braking share, rpm, gear, gear changes and DRS-open share.
    - **Lap filters:**
      - laps with less than half the expected samples are dropped
      - laps not in `data/laps_combined_clean.csv` are dropped (pit-out, invalid and outlier laps), so car data covers the same laps as the lap data
      - per-session |z| > 3 outliers on the aggregates are dropped
    - **Normalization:** per-session `StandardScaler`, the same scheme as the laps.
    - **Output:** 90 laps (11–19 per driver-session) in `data/cardata_per_lap_clean.csv`. Also written: an unscaled copy, and the cleaned samples per input file in `scripts/results/process_cardata/`.
    - **Pipeline:** wired into `master.py` after `process_weather.py`. It is not a model input yet, see #16.
    - **Data fix:** `car_data_p1_1.csv` was driver #7 (Alpine). It was re-downloaded for #38, who ran Ferrari's #16 car in FP1.

- [ ] **#16 Car data is not a model input yet**
  - **Where:** `scripts/feature_engineer.py` and `scripts/feature_config.py`, reading `data/cardata_per_lap_clean.csv` (written by `scripts/process_cardata.py`).
  - **Problem:**
    - The car data is cleaned, aggregated per lap and scaled, but nothing reads it.
    - It covers only the two Ferrari cars per session (2 of ~20 drivers), and the model is trained on the whole field. Adding it as a feature as-is would leave about 90% of training rows empty.
  - **Fix:**
    - Decide how to use it: a Ferrari-only model, a second model with the car-data features, or imputation for the other drivers.
    - Merge on `(session_key, driver_number, lap_number)`, using only lap *t*'s car data to predict lap *t+1* (no leakage).
    - Add the chosen columns to `MODEL_FEATURE_COLUMNS` / `SIM_FEATURE_COLUMNS` in `feature_config.py`, and decide what the simulator feeds for them.
  - **Depends on:**
    - #4 (fixed): the engineered dataset now keeps `session_key` and `driver_number`, so the merge keys are available.
    - #7 (fixed): the laps are no longer scaled, but the car-data aggregates still are (per session). Leave them unscaled too before merging, so the inputs are physical units throughout and the simulator's median baseline stays meaningful.
    - #9 (fixed): the simulator rolls a lap-*t* feature vector forward. Car-data features would need a seed value and a rule for how they evolve, or they stay at their medians like the speeds.
  - **readme.txt:** Data Collection step 5; Data Preparation, CarData.

- [ ] **#21 Race stints always start on a new tyre set**
  - **Where:** `RACE_TYRE_AGE_AT_START` in `scripts/strategy_simulator.py`.
  - **Problem:** every simulated stint has `tyre_age_at_start = 0`. In a real race, some stints start on sets already used in practice or qualifying, and the sets available depend on what is left of the weekend allocation.
  - **What the data allows:**
    - OpenF1 `stints` gives each stint's compound, lap range and `tyre_age_at_start`, but no set ID.
    - Sets can be inferred by chaining: a stint that starts used, at the age an earlier stint of the same compound ended at, is probably the same set. For example, #44 in FP1 ran HARD for laps 1–9, then started stint 2 on HARD at age 8.
    - This is a heuristic, and it is ambiguous when two sets share a compound and age.
  - **Fix:**
    - Infer Ferrari's sets and their ages from the stints.
    - Add the qualifying stints. This broadens scope beyond FP1–FP3, so confirm first.
    - Hard-code the allocation and return rules (typically 2 HARD / 3 MEDIUM / 8 SOFT per weekend, with sets returned after sessions), which OpenF1 doesn't provide.
    - Let `generate_strategies()` choose from the sets left, and start each stint at its set's age.
  - **readme.txt:** Race Simulation step 2.

- [x] **#22 Settings are hardcoded across scripts**
  - **Where:** all 8 stage scripts and `scripts/feature_config.py`.
  - **Problem:** the race constants (`PIT_LOSS = 22.0`, `NUM_LAPS = 57`, `FUEL_PENALTY_PER_LAP`, the pit window `range(10, total_laps - 10)`, stops `[0, 1, 2]`) and `FUEL_EFFECT_K` were literals. So were the cleaning thresholds (30 s, 2× median, 50 km/h, IQR × 3, weather and car-data |z| > 3, car-data limits), the 5 min weather-merge tolerance, the model hyperparameters, the split and CV settings, `random_state=42`, and every data/results path and file name. Trying a different pit loss or another weekend's files meant editing code, and the same value (e.g. the model bundle path, the seed) was repeated across scripts.
  - **Fix:** move them to one settings file that every script reads.
  - **Fixed:** settings live in `.env` at the repo root, which is committed: there are no secrets, and a fresh clone runs as-is. The new `scripts/env_config.py` parses it, with no new dependency. Variables already set in the environment win, so `$env:PIT_LOSS_S="25"; python strategy_simulator.py` overrides one run. The getters (`env_str` / `env_int` / `env_float` / `env_list`) raise `KeyError` on a missing key instead of falling back to a hidden default. Each script reads its keys once at module level. `FUEL_EFFECT_K` stays exported from `feature_config.py`, so its importers are unchanged.
    - **Kept in code** (schema or format, not settings): the column lists and `COMPOUNDS`, the throttle 0–100 / brake {0, 100} encodings, the XGBoost objective, `n_jobs`, plot cosmetics, the t-test placeholder (#5) and the dead `compound_HARD = 1` start setting (#12).
    - **Checks:** the full pipeline was run before and after. All 69 output CSV/JSON/TXT files are identical (CSVs to rtol 1e-9; the RandomForest RMSE differs only in the last digit between runs anyway). The exceptions are the unseeded t-test samples and p-value (#5) and one log line now reading "within 5min". The best strategy is unchanged (SOFT–SOFT–SOFT, pits 12 and 13, 5695.12 s). With `PIT_LOSS_S=30` it becomes 5711.12 s, which is 2 stops × 8 s more, as expected.
  - **readme.txt:** Running the Pipeline.

- [ ] **#15 Results documentation**
  - **Problem:** results currently exist only as raw files in `scripts/results/<script>/` and a few numbers in the readmes.
  - **Depends on:** all P1 items, plus #13 (#9 is fixed). The numbers will change once those are fixed, so write this up afterwards.
  - **readme.txt:** Known Issues.
  - Cover:
    - [ ] **Data cleaning:** rows removed per session and per filter (from the `*_removed_rows_summary.csv` files and the weather z-score summary), and the final row counts. This includes car data: samples and laps removed per stage, and the laps kept per driver and session.
    - [ ] **Feature engineering:** final dataset shape, how many laps matched a stint, and how many rows are usable for training.
    - [ ] **Model comparison:** test MAE, RMSE and R² for all three models, in seconds rather than z-scores, with the metric chart and prediction overlay plots and what they show.
    - [ ] **Feature importance:** top features for Random Forest and XGBoost, and whether they match tyre physics (tyre age, compound, track temperature).
    - [ ] **Validation:** CV results per fold with the spread, the residual plot and what it shows about bias, and the t-test replaced with a meaningful test or removed (#5).
    - [ ] **Strategy:** recommended strategy (compound sequence and pit laps), predicted race time in seconds, the next-best alternatives and the gap to them, and predicted pace by compound across the race.
    - [ ] **Limitations:** a short note on which results depend on items still open above.
    - [ ] Decide where this lives (a results section in `readme.txt` / `README.md`, or a separate `RESULTS.md`) and link it from both readmes.
