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
  - **Where:** `data/raw/weather/weather.csv`, and the download URL in `readme.txt` / `README.md`.
  - **Problem:** the weather was downloaded with `meeting_key=1253`, which is Bahrain pre-season testing (sessions 9683–9685, 26–28 Feb 2025). The laps, stints and car data are from `meeting_key=1257`, the Grand Prix weekend (11–12 Apr 2025).
  - **Effect:** the weather features describe a different week. The timestamps can't line up, so `feature_engineer.py` spreads the weather rows evenly over the lap numbers ("pseudo time"). Each lap's weather is therefore effectively arbitrary.
  - **Fix:** download `https://api.openf1.org/v1/weather?meeting_key=1257&csv=true`, then merge per session on timestamps (`date_start` against `date`).
  - **readme.txt:** Data Collection step 4; Feature Engineering, pseudo time alignment.
  - **Fixed:** `data/raw/weather/weather.csv` re-downloaded for `meeting_key=1257`. `process_weather.py` keeps only the practice sessions (the `session_key`s in `laps_p*.csv`), dropping qualifying and race. `feature_engineer.py` replaces pseudo time with `merge_asof` on `date_start` ~ `date` (nearest, 5 min tolerance); all 839 laps match, within 30 s. Held-out R² went from 0.306/0.232/0.102 to 0.425/0.474/0.419 (LR/RF/XGB).

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
  - **Fixed:** `process_laps.py` and `process_stints.py` now drop only `meeting_key`, so `session_key` reaches `data/processed/laps_combined_clean.csv` and `data/processed/stints_combined_clean.csv`. `assign_stint_numbers()` adds `session_key` to the mask. It also numbers the global stints in order of `(driver_number, session_key, stint_number)` rather than `lap_start`, which interleaved the sessions. `feature_engineer.py` drops `session_key` from the final dataset along with `driver_number`, so the engineered dataset's schema is unchanged. #4 decides which id to keep there.

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

- [x] **#8 Illegal strategies are allowed**
  - **Where:** `generate_strategies()` in `scripts/strategy_simulator.py`.
  - **Problem:** it includes 0-stop strategies and sequences that use a single compound, which the two-compound rule forbids. The current recommendation (HARD, no stop) is one of these.
  - **Fix:** keep only sequences with `len(set(seq)) >= 2`.
  - **readme.txt:** Race Simulation steps 1 and 4.
  - **Fixed:** `generate_strategies()` skips every compound sequence with fewer than `MIN_COMPOUNDS_USED` (`.env`, 2) distinct compounds and logs how many strategies it dropped. A 0-stop strategy has one compound, so the filter rules it out even if `STOP_COUNTS` includes 0. `STOP_COUNTS` in `.env` is now `1,2`.
    - **Strategy space:** 18,318 → 16,206. 1-stop: 6 sequences × 37 pit laps = 222. 2-stop: 24 sequences × 666 pit pairs = 15,984. The 2,109 dropped (3 × 37 + 3 × 666) are SOFT–SOFT, HARD–HARD–HARD and the like. Checked on `strategy_simulator_all_strategy_results.csv`: every row uses ≥2 compounds, and none has 0 stops.
    - **Effect on strategy:** none on the winner, which already used two compounds. It is still HARD–HARD–SOFT, pits on laps 12 and 24, 6023.36 s. The best 1-stop is still HARD–SOFT, pit on lap 11, 6144.94 s. The previous best 0-stop (MEDIUM, 6178.24 s) is no longer a candidate. It is still **not** a valid recommendation, because of #11.

- [x] **#10 Tyre age does not reset after a pit stop**
  - **Where:** `precompute_lap_predictions()` / `stint_time()` in `scripts/strategy_simulator.py`.
  - **Problem:** predictions are indexed by race lap, and `laps_since_stint_start` is set to the race lap number. A stint that starts on lap 30 is priced as if its tyres were already 30 laps old.
  - **Fix:** predict each stint from tyre age 1, i.e. index by stint length rather than by race lap.
  - **readme.txt:** Race Simulation step 2.
  - **Fixed** (together with #9): `precompute_stint_rollouts()` rolls every compound out for every stint length L = 1..57, always from tyre age 1 (`laps_since_stint_start` = 1 on the stint's first lap). `stint_time()` prices a stint as the summed predicted fcp of the L-lap rollout plus the race fuel term (`FUEL_PENALTY_PER_LAP` × fuel on board) of the race laps it covers. So a stint costs the same wherever it starts, apart from the fuel. Race stints originally always started on a new set; since #21 they start at the age of the set the car fits.
    - **Training range:** `laps_since_stint_start` is 2–18 in training (pit-out laps are cleaned out). Age 1 is treated like 2. Beyond 18, XGBoost's response to tyre age is flat, so any later change comes from the fed-back inputs (#9).

- [x] **#11 The pit lap is missing from the race-time sum**
  - **Where:** `simulate_strategy_fast()` in `scripts/strategy_simulator.py`.
  - **Problem:** stints are summed over `(last, p - 1)`, then `last = p + 1`, so lap `p` is never counted. The pit loss replaces a whole lap instead of being added to it.
  - **Effect:** now that #7 is fixed, a dropped lap (~100–110 s) outweighs the 22 s pit loss, so every extra stop makes a strategy look faster. The optimiser picks 2 stops on consecutive laps (12 and 13 since #9/#10; 10 and 11 before). The middle "stint" is empty and two laps go missing from the total.
  - **Fix:** sum every race lap exactly once and add the pit loss on top.
  - **readme.txt:** Race Simulation step 3.
  - **Fixed:** stints are now summed over `(last, p)`, then `last = p + 1`. "Pit on lap `p`" means the car comes in at the end of lap `p`: lap `p` is the in-lap, the last lap of the old stint, and the next stint starts on lap `p + 1`. The in- and out-lap time loss is all in `PIT_LOSS_S`, which is added once per stop on top of the laps. The pit window (laps 10–46) and the strategy space are unchanged. Consecutive pit laps now give a 1-lap middle stint instead of an empty one.
    - **Checks:** on all 24,420 rows of `strategy_simulator_all_strategy_results.csv`, the stint lengths sum to 57 and are each ≥ 1. Each `total_time` equals the summed `stint_totals` of its stints + the full-race fuel term + 22 s × stops, to within 3e-12 s. The rollouts and stint totals are unchanged; only the race sum moved. Because every lap is counted once, the race fuel term is the same for every strategy, so a total depends only on which (compound, set age, stint length) stints it uses.
    - **Effect on strategy:** each total rises by about one lap per stop. 2 stops still win, but by about 100 s instead of ~210 s, and consecutive pit laps no longer do (best one is 104–113 s behind the winner).
      - **Car 16:** SOFT(0)–SOFT(0)–HARD(0), pits on laps 19 and 38 (three 19-lap stints), 6113.18 s. SOFT–HARD–SOFT and HARD–SOFT–SOFT on the same sets and laps tie with it. Swapping in the used SOFT(6) set is 3.42 s slower. Best 1-stop: SOFT(6)–HARD(0), pit on lap 38, 6214.55 s (+101.37 s). Was SOFT(6)–SOFT(0)–HARD(0), 18 and 38, 5890.48 s.
      - **Car 44:** MEDIUM(3)–SOFT(4)–SOFT(5), pits on laps 17 and 40, 6082.41 s. Any order of the same three sets with stints of 17, 17 and 23 laps ties with it, e.g. pits on 17 and 34. Best 1-stop: MEDIUM(3)–SOFT(4), pit on lap 17, 6185.01 s (+102.60 s). Was the same sets, 18 and 38, 5856.92 s.
      - 2 stops still win because each rollout steps up 6–13 s within about 14–20 stint laps and then plateaus, so three short stints keep more laps on the fast early part. That shape comes from practice data with #20 still open, and #19 is still open, so this is still **not** a valid recommendation.

- [x] **#19 The fuel correction has the wrong sign and horizon**
  - **Where:** `fuel_corrected_pace` in `scripts/feature_engineer.py`.
  - **Problem:**
    - `fuel_corrected_pace = lap_duration + 0.035 × laps_remaining` adds the fuel effect. Laps with more fuel on board are already slower, so the correction should subtract it.
    - `laps_remaining = lap_end - lap_number` counts the laps left in the *stint*, not the fuel on board. In practice, the fuel load depends on the run plan, which the data doesn't record.
  - **Effect:** the target is not fuel-free pace. The simulator still adds race fuel (`0.035 ×` laps of fuel remaining) on top of it, so early-stint fuel is counted twice in training and differently in simulation.
  - **Fix:** subtract the term, and decide on a fuel proxy (e.g. laps left in the stint as an approximation of the run's fuel, stated as an assumption), or drop the correction and let the model learn it from `laps_since_stint_start`.
  - **readme.txt:** Feature Engineering, fuel_corrected_pace.
  - **Fixed:** `feature_engineer.py` now subtracts the fuel term: `fuel_corrected_pace = lap_duration − k × laps_remaining`, with `laps_remaining = lap_end − lap_number + 1`.
    - **Fuel proxy (assumption):** practice fuel loads aren't recorded, so the laps left in the stint, counting lap *t*, stand in for the fuel on board at the start of lap *t*. This assumes each run is fuelled for its own length. `lap_end` comes from the stints, so laps removed in cleaning still count. The `+ 1` matches the race term, which counts the current lap (`FUEL_PENALTY_PER_LAP × (NUM_LAPS − lap + 1)`), so fcp is fuel-free pace in both and the race fuel is added once, in the simulator only. `laps_remaining` is 1–17 in training, so the term moves fcp by −0.035 to −0.595 s (it was +0 to +0.56 s).
    - **Simulator:** `rollout_stint()` inverts the new definition. The seed is `fcp = raw − k × L`, and each later lap is `raw = fcp + k × (L − age + 1)`. The race fuel term is unchanged.
    - **Checks:** `fuel_corrected_pace + 0.035 × (lap_end − lap_number + 1)` equals `lap_duration` on all 839 laps (514 rows after the target, unchanged). On all 11,571 rollout rows, `raw − fcp = 0.035 × (L − age + 1)`, the sectors sum to the raw lap, and every stint lap 1 is the 98.98 s seed. On all 24,420 result rows, the totals equal the summed stint totals + the race fuel term + 22 s × stops (to 3e-12 s).
    - **Effect on metrics:** almost none, since the target moves by at most ~1.2 s per lap. Held-out MAE is 16.64 / 12.42 / 12.19 s, RMSE 20.48 / 17.02 / 16.06 s and R² 0.374 / 0.568 / 0.615 (LR/RF/XGB; was 16.62 / 12.37 / 11.97 s and R² 0.373 / 0.574 / 0.620). XGBoost 5-fold CV MAE is 11.92 s, RMSE 16.36 s (was 11.77 / 16.16 s). XGB's top features are `i2_speed` 0.133, `tyre_age` 0.094, `track_temperature` 0.079 and `duration_sector_3` 0.077.
    - **Rollout behaviour** (57-lap stint, fcp in s):
      - The seed fcp is now 96.98 s (98.98 − 0.035 × 57), below the predicted laps rather than above them.
      - New sets still predict a slow second lap (111.7–112.2 s, a step of up to 15.2 s from the seed), then 102–105 s, and step up to ~110.7–110.9 s around stint lap 20.
      - Used sets have no jump: 100.7–103 s, stepping up to ~110.3 s at stint lap 14 (SOFT 6), 15 (SOFT 5), 16 (SOFT 4) and 17 (MEDIUM 3). Every set steps up at tyre age ~20, near the top of the training range (2–21), so this is the model running out of data, not tyre physics.
      - All rollouts stay within 96.98–112.19 s.
    - **Effect on strategy:** both cars now pick new sets. SOFT(0)–SOFT(0)–HARD(0), pits on laps 19 and 38 (three 19-lap stints), 6005.04 s, tied with SOFT–HARD–SOFT and HARD–SOFT–SOFT on the same sets.
      - **Car 16:** the same strategy as before, 108.14 s faster (was 6113.18 s). The best strategy with a used set is SOFT(6)–SOFT(0)–HARD(0), pits on 14 and 33, +25.23 s. The best 1-stop is SOFT(0)–HARD(0), pit on lap 19, +114.18 s.
      - **Car 44:** was MEDIUM(3)–SOFT(4)–SOFT(5) at 6082.41 s. Swapping one stint to MEDIUM(3) is +3.43 s. The best 1-stop is SOFT(4)–HARD(0), pit on lap 38, +113.37 s.
      - Used sets now lose because they reach the tyre-age ~20 step sooner, which outweighs the new-set second lap. Neighbouring pit laps are 4–7 s behind, and consecutive pit laps 124.53 s.
      - No P1 items are left, but the rollout shape still comes from practice laps that include cool-down laps (#20), and the metrics come from a shuffled split (#18). This is still **not** a valid recommendation.

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

- [x] **#18 Train/test splits mix laps from the same stint**
  - **Where:** `train_test_split(..., shuffle=True)` in `scripts/train_model.py` and `KFold(n_splits=5, shuffle=True)` in `scripts/validate_strategy.py`.
  - **Problem:** rows are shuffled one lap at a time, so most stints have laps in both the training and the test set. Consecutive laps of one stint are highly correlated (same driver, car, tyre, fuel and conditions), and before #17 was fixed the test laps' own lap times were spread through the training rows.
  - **Effect:** the test and CV scores measure interpolation within stints the model has already seen, not prediction on a new stint (which is what the simulator does). They overstate the model.
  - **Fix:** split by stint with `GroupShuffleSplit` / `GroupKFold`, grouped on `(driver_number, stint_number)` (or by session, or by driver, for a stricter test). Both id columns are in the engineered dataset since #4.
  - **readme.txt:** ML Modelling, training (80/20 split); Evaluation step 1, cross-validation.
  - **Fixed:** both splits are grouped by stint, on `TARGET_GROUP_COLUMNS` = `(driver_number, stint_number)` from `feature_config.py` (the global stint number, so one group is one stint in one session). A stint's laps are now all in train or all in test.
    - **`train_model.py`:** `GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)`. `TEST_SIZE` (0.20) is now a share of stints, so the test row count depends on which stints are drawn: 40 stints / 128 rows train, 10 stints / 25 rows test (was 122 / 31 rows). The test set is 12 FP1, 10 FP2 and 3 FP3 rows; 12 SOFT, 8 MEDIUM, 5 HARD. The medians, session weights and bundle format are unchanged.
    - **`validate_strategy.py`:** `GroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)` (shuffle needs scikit-learn ≥ 1.6; 1.7.2 is pinned). Each fold holds out 10 stints (23–41 rows). `validate_strategy_cv_fold_metrics.csv` gains `test_stints` and `test_rows`. Fold 1 happens to hold out the same 10 stints as the held-out split, so its scores equal XGBoost's held-out ones.
    - **Checks:** both scripts log the stints shared between train and test: 0 in the held-out split and in every fold. The folds' test sets cover all 50 stints and 153 rows once.
    - **Effect on metrics:** held-out MAE 2.37 / 1.32 / 1.12 s, RMSE 2.74 / 1.68 / 1.56 s and R² −1.002 / 0.250 / 0.353 (LR/RF/XGB; was 1.44 / 1.20 / 1.07 s, 1.90 / 1.68 / 1.60 s and 0.308 / 0.460 / 0.510). XGBoost grouped 5-fold CV MAE 1.23 s, RMSE 1.70 s (was 1.13 / 1.60 s), with per-fold MAE 0.75–1.68 s and RMSE 1.18–2.19 s.
      - Linear Regression collapses on unseen stints (worse than predicting the test mean); with 36 inputs on 128 rows it was fitting within-stint detail. RF and XGB lose about 0.1–0.2 in R² but keep their order, so XGBoost is still the best of the three (#13).
      - Different stints in the training set also change the saved model, so the feature importances and rollouts below move too. With only 10 test stints, the held-out scores are noisy: the per-fold spread (MAE 0.75–1.68 s) is the better guide.
    - **Feature importance:** XGB's top five are now `session_FP2` 0.210, `drs_open_pct` 0.116, `stint_avg_pace` 0.087, `full_throttle_pct` 0.085 and `i2_speed` 0.069; `tyre_age` is 7th (0.030). RF is led by `i2_speed` 0.151, `duration_sector_2` 0.139 and `full_throttle_pct` 0.115.
    - **Rollout behaviour** (57-lap stint, fcp in s): the seed is unchanged (97.15 s). Every compound and set age predicts 99.0–99.4 s on stint lap 2 and 99.9–100.5 s on lap 3, then stays within 99.74–100.87 s (about 0.7 s faster than before). The largest lap-to-lap step is 1.85–2.28 s. Still almost flat; the used SOFT(6) set is ~0.5 s slower per lap than the new sets.
    - **Effect on strategy** (car 16): MEDIUM(0)–HARD(0), pit on lap 10, 5774.26 s. SOFT(0)–HARD(0) on the same lap is +0.02 s, and the reverse orders and neighbouring pit laps are +0.07–0.54 s. The best 2-stop is MEDIUM(0)–MEDIUM(0)–HARD(0), pits on laps 10 and 20, +17.16 s. No kept strategy uses the SOFT(6) set. Was HARD(0)–SOFT(6), pit on lap 45, 5840.85 s; the ~67 s drop is the faster rollouts, not a better strategy.
      - The top strategies are still within ~0.5 s, so the compound and pit-lap choice are still close to arbitrary. The metrics are now honest about new stints, but the model has still learnt almost no compound or tyre-age effect from 153 rows (#27). This is still **not** a valid recommendation.

- [x] **#20 The cleaned laps still include slow non-push laps**
  - **Where:** the filters in `scripts/process_laps.py`.
  - **Problem:** the filters drop pit-out laps, invalid sectors, laps over 2× the median and IQR residual outliers. Cool-down, in- and traffic laps under 2× the median remain. Cleaned laps have a median of about 100–117 s per session, but a mean of 111–121 s and a 90th percentile of about 147–159 s.
  - **Effect:** the target mixes push laps with slow laps, so the errors in seconds are large (XGB MAE about 10.5 s). The simulator's median-feature lap (103–112 s) is slower than race pace.
  - **Since #21:** with the real `tyre_age` as an input, the model predicts a ~113 s second lap on every new set (push lap then cool-down lap in practice), so the optimiser prefers used sets.
  - **Fix:** flag or drop non-push laps, e.g. laps more than a set margin slower than the driver's best on that compound in the session, or train only on long-run stints.
  - **readme.txt:** Data Preparation, laps cleaning.
  - **Fixed:** `apply_push_lap_filter()` in `process_laps.py` runs after the IQR filter and drops every lap slower than `PUSH_LAP_MAX_BEST_RATIO` (`.env`, 1.15) × the driver's best lap in that session.
    - **Threshold:** after the other filters, laps divided by the driver's session best split into two groups: push laps and long runs at ≤ 1.10× (FP2's long runs sit at 1.06–1.10×), and cool-down laps from about 1.3×. Few laps fall in between, with a gap around 1.14–1.16×.
    - **Driver's best, not per compound:** the compound only reaches the laps in `feature_engineer.py` (from the stints). The gap between compounds (about 1 s, ~1%) is well inside the 15% margin.
    - **Rows:** 839 → 467 cleaned laps (145 / 109 / 118 removed in FP1 / FP2 / FP3, leaving 164 / 221 / 82). Median 98.1 / 98.8 / 94.9 s, mean 98.4 / 97.8 / 95.7 s and 90th percentile 101.8 / 101.3 / 99.9 s.
    - **Target rows:** 514 → 153 (223 are a stint's last lap, 91 have a gap to the next lap). Practice runs alternate push and cool-down laps, so dropping the cool-down lap breaks most lap + 1 pairs. 50 stints remain (SOFT 64, MEDIUM 63, HARD 26 rows), and FP3 has only 7 rows: session weights are now FP1 0.294, FP2 0.660, FP3 0.046. In training, `laps_since_stint_start` runs 2–13 and `tyre_age` 2–18.
    - **Car data:** 90 → 55 laps (5–14 per driver and session), since `process_cardata.py` keeps only the laps in `data/processed/laps_combined_clean.csv`.
    - **Checks:** every lap in `data/processed/laps_combined_clean.csv` is within 1.15× its driver's session best (max 1.1475). The removed counts add up: 839 − 372 = 467.
    - **Effect on metrics:** held-out MAE 1.67 / 1.13 / 1.03 s, RMSE 2.10 / 1.61 / 1.46 s and R² 0.157 / 0.505 / 0.590 (LR/RF/XGB; was 16.64 / 12.42 / 12.19 s and R² 0.374 / 0.568 / 0.615). XGBoost 5-fold CV MAE 1.19 s, RMSE 1.67 s (was 11.92 / 16.36 s).
      - The errors are about 10× smaller because the slow laps are gone.
      - R² fell because there is much less variance left to explain.
      - The test set is now 31 rows from a shuffled split (#18).
    - **Feature importance:** XGB's top five are `i2_speed` 0.144, `stint_avg_pace` 0.120, `duration_sector_2` 0.084, `wind_speed` 0.077 and `session_FP1` 0.074. `tyre_age` is 10th (0.042) and `track_temperature` 16th (0.020). RF is led by `duration_sector_2` 0.265, `i2_speed` 0.182, `duration_sector_1` 0.120 and `stint_avg_pace` 0.109.
    - **Rollout behaviour** (57-lap stint, fcp in s):
      - The seed is now 99.15 s raw (median sectors of the training rows; was 98.98 s), so fcp 97.15 s on stint lap 1.
      - Every compound and set age predicts about 99.4–99.7 s on stint lap 2 and about 101 s on lap 3. It then stays at 100.5–101.7 s to lap 57, with all seven (compound, set age) pairs within 0.3 s of each other.
      - The new-set second-lap jump is gone. The largest lap-to-lap step is 2.2–2.5 s (laps 1–3), and every rollout stays within 97.15–101.70 s.
    - **Effect on strategy:** both cars now pick 1 stop.
      - **Car 16:** SOFT(6)–MEDIUM(0), pit on lap 46, 5838.86 s. It ties with MEDIUM(0)–SOFT(6), pit on lap 11 (the same two stints in reverse order); SOFT(6)–HARD(0) is +0.09 s. The best 2-stop is MEDIUM(0)–SOFT(6)–SOFT(6), pits on laps 18 and 44, +9.44 s.
      - **Car 44:** SOFT(6)–MEDIUM(3), pit on lap 46, 5838.15 s (ties with the reverse, pit on lap 11). The best 2-stop is +11.21 s.
      - Neighbouring pit laps are within about 1 s. The rollouts are almost flat, so tyre age barely matters, and an extra stop can't win back its 22 s. The model has learnt almost no compound or tyre-age effect from 153 rows. Every new stint also gets a fast median seed lap, which narrows the gap to 2 stops.
      - The metrics still come from a shuffled split (#18). This is still **not** a valid recommendation.

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

- [x] **#6 The residual plot mixes training and test data**
  - **Where:** `pred_full = model.predict(X)` in `scripts/validate_strategy.py`.
  - **Problem:** the model from the last cross-validation fold predicts every row, including the 80% that fold was trained on, so the residuals look better than they really are.
  - **Fix:** plot out-of-fold predictions (collect each fold's test predictions), or use the held-out split only.
  - **readme.txt:** Evaluation step 1, residual plot.
  - **Fixed:** the CV loop in `validate_strategy.py` now stores each fold's test predictions in `oof_pred`. The residual plot uses those, so every row is scored by the one model that held its stint out. GroupKFold (#18) holds each row out exactly once, so all 153 rows are plotted, and the script raises if any row has no prediction. The `N = min(3000, len(y))` slice (first rows in sort order, not a sample) is gone.
    - **Outputs:** `validate_strategy_oof_predictions.csv` (ids, target, OOF prediction, residual, fold). `validate_strategy_validation_summary.csv` gains `OOF_MAE`, `OOF_RMSE` and `OOF_mean_residual`. The plot title says it is out-of-fold.
    - **Checks:** the fold metrics are unchanged (CV MAE 1.23 s, RMSE 1.70 s). Every fold holds 10 stints, every row has exactly one prediction, and `residual = target − prediction` to 1e-14 s.
    - **Effect:** pooled OOF MAE 1.18 s, RMSE 1.69 s, mean residual −0.20 s. The old plot (last fold's model on all rows) gave MAE 0.23 s and RMSE 0.85 s, so it understated the error about 5× (MAE) and 2× (RMSE).
    - **What the plot shows:** no clear trend against predicted pace (slope −0.09, correlation −0.07). The residuals are right-skewed: median −0.32 s, 5th–95th percentile −2.31 to +3.75 s, with a handful of laps 4–5 s slower than predicted. The predictions span 96.4–104.2 s against targets of 95.9–107.8 s, so the model doesn't reach the slowest laps. The per-fold mean residual ranges from −1.00 s (fold 5) to +0.12 s, so the small overall bias depends on which stints are held out.

- [ ] **#5 The t-test is a placeholder**
  - **Where:** the "T-TEST EXAMPLE" block in `scripts/validate_strategy.py`.
  - **Problem:** it compares two sets of random numbers (`np.random.normal(4800, 10)` against `np.random.normal(4820, 10)`), unseeded. The p-value says nothing about the model or the strategy, and it changes every run.
  - **Fix:** replace it with a meaningful test (e.g. per-fold errors of two models, or simulated race times of the top strategies), or remove it.
  - **readme.txt:** Evaluation step 1, t-test.

- [ ] **#13 The saved model is the weakest of the three on the test split**
  - **Where:** `scripts/train_model.py` always saves XGBoost.
  - **Problem:** `readme.txt` justifies XGBoost on qualitative grounds. Before #4 was fixed, the held-out R² was 0.043 for Linear Regression, 0.060 for Random Forest and −0.039 for XGBoost. After #4, XGBoost is now the best of the three: 0.534 for LR, 0.625 for RF and 0.652 for XGB. After #7 (seconds, 25 inputs with the session one-hot), it is 0.586 / 0.640 / 0.656, with MAE 12.12 / 10.65 / 10.51 s. After #17 (causal `stint_avg_pace`), it is 0.373 / 0.565 / 0.620, with MAE 16.62 / 12.54 / 12.06 s: XGBoost's lead over Random Forest grew, but it still comes from one shuffled split. After #21 and #19 (fuel term subtracted), it is 0.374 / 0.568 / 0.615, with MAE 16.64 / 12.42 / 12.19 s. After #20 (non-push laps dropped, 153 target rows), it is 0.157 / 0.505 / 0.590, with MAE 1.67 / 1.13 / 1.03 s. After #16 (10 car-data inputs, 36 in all), it is 0.308 / 0.460 / 0.510, with MAE 1.44 / 1.20 / 1.07 s. After #18 (split grouped by stint, 10 test stints), it is −1.002 / 0.250 / 0.353, with MAE 2.37 / 1.32 / 1.12 s: XGBoost is still the best on the held-out stints, but only one model's grouped CV is run.
  - **Fix:** re-run the comparison with grouped CV for all three models (#18 is fixed, so the split is by stint). Then either keep XGBoost with metrics that support it, or save the best model and update the reasoning.
  - **Depends on:** nothing left (#2, #3, #4, #17 and #18 are fixed).
  - **See also:** #27 widens the comparison beyond these three models.
  - **readme.txt:** ML Modelling step 3, results table.

- [ ] **#27 The model choice rests on three untuned models and untested reasoning**
  - **Where:** `scripts/train_model.py` (Linear Regression, Random Forest 200 trees, XGBoost with one fixed set of hyperparameters from `.env`); the "Reasoning" paragraph in `readme.txt` ML Modelling step 3.
  - **Problem:**
    - Only three models are compared, each with one configuration, on one held-out split (10 stints, 25 rows since #18; 31 rows before). XGBoost at max depth 7 with 36 inputs on 153 rows is likely to overfit, but no other setting has been tried.
    - There is no naive baseline, so it isn't known whether any model beats a trivial forecast.
    - The reasoning is asserted, not tested. "Random Forest produces noisy, stepwise predictions, unstable in autoregressive simulations" and "XGBoost: stable rollouts" were never checked: only XGBoost is ever rolled out. The "aligns with real-world motorsport pipelines" line has no source.
  - **Fix:**
    - [ ] **Baselines:** the next lap equals lap *t*'s fcp, and the next lap equals `stint_avg_pace`. Every model has to beat these to count.
    - [ ] **More candidates:** regularised linear models (Ridge / Lasso), a simple per-compound linear degradation model, ExtraTrees and `HistGradientBoostingRegressor` from scikit-learn. LightGBM or others would be a new dependency, so confirm first.
    - [ ] **Tuning:** a small hyperparameter search for each model (for XGBoost: shallower trees, `min_child_weight`, regularisation), scored with grouped CV by stint (#18). The tuned values go in `.env`.
    - [ ] **Fewer inputs:** try reduced feature sets (e.g. without the car-data aggregates or collinear groups, see #25), since 36 inputs on 153 rows is a lot.
    - [ ] **Rollout comparison:** roll each serious candidate out through the simulator's `rollout_stint()`. Compare stability (largest lap-to-lap step, drift), whether pace rises with tyre age, and whether compounds come out in a plausible order. One-step error alone doesn't show how a model behaves when rolled forward.
    - [ ] **Report:** per-fold MAE / RMSE / R² with the spread for every candidate, not a single split.
    - [ ] **Update the reasoning** in `readme.txt` ML Modelling step 3 to match the results. Keep each claim only if the experiments back it, replace the asserted ones with the measured comparison, drop or source the "real-world pipelines" line, and update the results table and the "XGBoost is kept" sentence. If another model wins, save it instead (the bundle format and the simulator would need to accept it).
  - **Depends on:** nothing left (#18, grouped CV, is fixed). Supersedes the comparison part of #13; close both together.
  - **readme.txt:** ML Modelling step 3, model choice, reasoning and results table.

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
      - laps not in `data/processed/laps_combined_clean.csv` are dropped (pit-out, invalid and outlier laps), so car data covers the same laps as the lap data
      - per-session |z| > 3 outliers on the aggregates are dropped
    - **Normalization:** per-session `StandardScaler`, the same scheme as the laps.
    - **Output:** 90 laps (11–19 per driver-session) in `data/processed/cardata_per_lap_clean.csv`. Also written: an unscaled copy, and the cleaned samples per input file in `scripts/results/process_cardata/`.
    - **Pipeline:** wired into `master.py` after `process_weather.py`. It is not a model input yet, see #16.
    - **Since #16:** the data is full-field (60 files in `data/raw/car_data/`, 448 laps), the per-session scaling is removed, and the aggregates are model inputs.
    - **Data fix:** `car_data_p1_1.csv` was driver #7 (Alpine). It was re-downloaded for #38, who ran Ferrari's #16 car in FP1.

- [x] **#16 Car data is not a model input yet**
  - **Where:** `scripts/feature_engineer.py` and `scripts/feature_config.py`, reading `data/processed/cardata_per_lap_clean.csv` (written by `scripts/process_cardata.py`).
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
  - **Fixed** (car data for the whole field, all 10 aggregates as lap-*t* inputs):
    - **Download:** car data for every driver in each practice session's `laps_p*.csv` (20 per session, 60 files, ~66 MB). The files are now in `data/raw/car_data/` as `car_data_p<session>_<driver>.csv` (new `.env` key `CARDATA_DIR`; `CARDATA_GLOB` is unchanged). The six re-downloaded Ferrari files are identical to the old `car_data_p*_{1,2}.csv`, which are removed.
    - **Unscaled:** `process_cardata.py` no longer scales the aggregates (#7), so they are in physical units, as the laps are. The `_per_lap_unscaled.csv` copy and `_scaling_summary.csv` are gone, and `_per_lap.csv` and the hand-off are the unscaled table. Medians: `mean_speed` 195 km/h, `mean_rpm` 9978, `full_throttle_pct` 0.46.
    - **Columns:** `CARDATA_FEATURE_COLUMNS` in `feature_config.py` lists `mean_speed`, `max_speed`, `mean_throttle`, `full_throttle_pct`, `brake_pct`, `mean_rpm`, `max_rpm`, `mean_gear`, `gear_changes` and `drs_open_pct`. They are spliced into `MODEL_FEATURE_COLUMNS` after the weather, which gives 36 inputs. `process_cardata.py` takes its z-score columns from the same list.
    - **Car-data laps:** 448 (all 20 drivers in every session, 1–15 per driver and session). Of the laps assigned, 129 are dropped for low coverage, 596 for not being in the cleaned lap data and 19 as z-score outliers (now judged per session across the whole field).
    - **Merge:** `feature_engineer.py` left-merges the car data on `(session_key, driver_number, lap_number)` with `validate="one_to_one"`, after the weather. Each row gets its own lap *t*'s telemetry and the target is lap *t+1*, so nothing leaks. 467 rows before and after. 19 laps have no car data (dropped as outliers). `train_model.py` fills those with the training medians, as it does for other missing inputs. Of the 153 target rows, 148 have car data (was 19). Spot-checked: engineered rows match `process_cardata_per_lap.csv` for the same key and lap.
    - **Simulator:** no logic change. The car-data inputs start at the training medians and stay there for the whole rollout, like the speed traps. That makes them a constant offset in the race, not a driving-style or tyre effect. A seed per car, or a rule for how they evolve, would be a separate change.
    - **Effect on metrics:** held-out MAE 1.44 / 1.20 / 1.07 s, RMSE 1.90 / 1.68 / 1.60 s and R² 0.308 / 0.460 / 0.510 (LR/RF/XGB; was 1.67 / 1.13 / 1.03 s and R² 0.157 / 0.505 / 0.590). XGBoost 5-fold CV MAE 1.13 s, RMSE 1.60 s (was 1.19 / 1.67 s).
      - Linear Regression gains the most. On the 31-row held-out split, RF and XGB are slightly worse, while CV (all 153 rows) is slightly better. With 36 inputs and 153 rows, both changes are within noise, and both splits still shuffle laps (#18).
    - **Feature importance:** together the car-data inputs carry 0.249 of XGB's importance and 0.241 of RF's. XGB's top five are `stint_avg_pace` 0.145, `i2_speed` 0.100, `mean_gear` 0.080, `duration_sector_2` 0.066 and `air_temperature` 0.051, with `mean_speed` 6th (0.047). `tyre_age` is 7th (0.039) and `track_temperature` 21st. RF is led by `duration_sector_2` 0.201, `i2_speed` 0.153, `duration_sector_1` 0.096, `stint_avg_pace` 0.087 and `mean_gear` 0.075.
    - **Rollout behaviour** (57-lap stint, fcp in s): the seed is unchanged (97.15 s fcp). Every compound and set age predicts 99.3–99.8 s on stint lap 2 and 101.1–101.4 s on lap 3, then stays within 100.55–101.53 s. The largest lap-to-lap step is 2.1–2.6 s. Still almost flat, and compounds and set ages are within ~0.4 s of each other.
    - **Effect on strategy:** both cars still pick 1 stop, now on HARD and SOFT.
      - **Car 16:** HARD(0)–SOFT(6), pit on lap 45, 5840.85 s (ties with SOFT(6)–HARD(0), pit on lap 12). Neighbouring pit laps are +0.24 s. The best 2-stop is SOFT(0)–SOFT(0)–HARD(0), pits on laps 16 and 34, +13.75 s. Was SOFT(6)–MEDIUM(0), pit on lap 46, 5838.86 s.
      - **Car 44:** SOFT(4)–HARD(0), pit on lap 14, 5840.39 s (ties with the reverse, pit on lap 43). The best 2-stop is HARD(0)–SOFT(4)–SOFT(0), pits on laps 24 and 39, +13.38 s. Was SOFT(6)–MEDIUM(3), pit on lap 46, 5838.15 s.
      - The top strategies are within ~0.3 s, so the compound and pit-lap choice are still close to arbitrary. This is still **not** a valid recommendation (#18).

- [x] **#21 Race stints always start on a new tyre set**
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
  - **Fixed** (practice only, both Ferraris; the real tyre age is also a model input now):
    - **New stage `scripts/tyre_sets.py`** (in `master.py` after `process_stints.py`) chains each race car's practice stints into sets. It writes the sets left for the race to `data/processed/tyre_sets_race_inventory.csv`.
      - A stint at `tyre_age_at_start` 0 takes a new set from the allocation.
      - A used-set stint continues the same-compound set left at exactly that age, preferring the most recently used. If none matches, it is logged and takes a new set at that age.
      - A set's age after a stint is `tyre_age_at_start + (lap_end − lap_start)`, which is how OpenF1 counts it.
      - After practice session *i*, the `TYRE_RETURNS_PER_SESSION` most-worn sets are returned, keeping the `TYRE_RACE_RESERVED` new sets.
      - New `.env` keys: `RACE_CARS=16,44`, `FP_STAND_INS=38:16`, `TYRE_ALLOCATION=SOFT:8,MEDIUM:3,HARD:2`, `TYRE_RETURNS_PER_SESSION=1,1,2`, `TYRE_RACE_RESERVED=HARD:1,MEDIUM:1` and `TYRE_SETS_FILE`. `RACE_TYRE_AGE_AT_START` is removed.
    - **Chaining result:** 26 Ferrari stints (13 per car). 14 new sets, 11 exact matches and 1 ambiguous match (car 16, FP3: two SOFT sets at age 6; either choice gives the same inventory). No stint is unmatched. Assignments are in `tyre_sets_stint_set_assignment.csv` and returns in `tyre_sets_returned_sets.csv`.
      - **Car 16:** returns HARD 12 (FP1), MEDIUM 17 (FP2), SOFT 15 and SOFT 7 (FP3). Races with SOFT 0, 0, 0, 0, 6, 6 / MEDIUM 0, 0 / HARD 0.
      - **Car 44:** returns HARD 16, SOFT 12, then SOFT 8 and 7. Races with SOFT 0, 0, 4, 5, 6 / MEDIUM 0, 0, 3 / HARD 0.
    - **Real tyre age feature:** `feature_engineer.py` adds `tyre_age = tyre_age_at_start + laps_since_stint_start` (laps on the set including lap *t*; 2–21 in training), which is in `MODEL_FEATURE_COLUMNS`. That gives 26 inputs; `tyre_age_at_start` and `laps_since_stint_start` are kept. Checked equal to the sum on all 839 rows; 514 rows after the target, unchanged.
    - **Simulator:**
      - `rollout_stint()` takes the set's age: `tyre_age_at_start` = set age, and `tyre_age` = set age + stint lap.
      - Rollouts run once per (compound, set age) any car has (7 pairs, shared by both cars).
      - `set_age_options()` lists the distinct set choices for a compound sequence, each set used at most once. For each sequence and set of pit laps, the fastest choice is kept.
      - Sequences a car can't cover are skipped: 3,996 per car, since both have one HARD. That leaves 12,210 per car.
      - Outputs gain `car_number`/`set_ages` (all-strategy CSV), `set_age_at_start` (rollouts, totals in long format), and a per-car `best_strategy.json`.
      - Checked: every one of the 24,420 result rows uses only sets the car has, each at most once, and every rollout lap has `tyre_age = set_age_at_start + laps_since_stint_start`.
    - **Effect on metrics:** held-out MAE 16.62 / 12.37 / 11.97 s, RMSE 20.44 / 16.85 / 15.92 s and R² 0.373 / 0.574 / 0.620 (LR/RF/XGB; was 12.54 / 12.06 MAE and R² 0.565 / 0.620 for RF/XGB). LR is unchanged, because `tyre_age` is a sum of two of its inputs. XGBoost 5-fold CV MAE 11.77 s, RMSE 16.16 s (was 12.02 / 16.32 s). `tyre_age` is XGB's 2nd feature (0.087, after `i2_speed` 0.130) and RF's 6th (0.062).
    - **Rollout behaviour:**
      - On a new set, every compound predicts its second lap at about 113 s, a 12–13 s jump, then drops back to 104–106 s before stepping up to 111–114 s around stint lap 20.
      - Used sets (age 3–6) have no jump, and step up about 6 s around stint laps 14–17.
      - Cause: in practice, new sets are mostly run as push lap then cool-down lap. The training median fcp on new-set stint lap 2 is 94.8 s and on stint lap 3 is 148.8 s. The model has learnt that pattern now that it can see the set's real age. This is #20, not tyre physics.
    - **Effect on strategy:**
      - **Car 16:** SOFT(6)–SOFT(0)–HARD(0), pits on laps 18 and 38, 5890.48 s (tied with SOFT(6)–HARD(0)–SOFT(0), since the last two stints are both 19 laps). Its best with new sets only is 5901.72 s, and its best 1-stop is HARD(0)–SOFT(6) at 6101.64 s.
      - **Car 44:** MEDIUM(3)–SOFT(4)–SOFT(5), pits on laps 18 and 38, 5856.92 s. Its best 1-stop is MEDIUM(3)–SOFT(4) at 6072.59 s.
      - The previous winner, HARD–HARD–SOFT (6023.36 s), needs two HARD sets, which neither car has left.
      - Both cars prefer used sets because of the new-set second lap above, and #11 still drops the pit laps. So this is still **not** a valid recommendation.
    - **Still open (limitations):**
      - Qualifying stints aren't used (scope kept to FP1–FP3), so sets used in qualifying are counted as available.
      - The allocation, the return counts, returning the most-worn sets and the reserved sets are assumptions in `.env`. They should be checked against the season's sporting regulations.
      - Chaining is a heuristic: it assumes OpenF1's `tyre_age_at_start` counts every lap on the set.
    - **Since #23:** `tyre_sets.py` builds an inventory for every car in practice, not just the Ferraris, and the simulator runs for the one car in `SIM_DRIVER`. `RACE_CARS` is removed.
    - **Since #24:** `FP_STAND_INS` is removed; the stand-ins are derived from OpenF1's per-session entry lists.

- [x] **#22 Settings are hardcoded across scripts**
  - **Where:** all 8 stage scripts and `scripts/feature_config.py`.
  - **Problem:** the race constants (`PIT_LOSS = 22.0`, `NUM_LAPS = 57`, `FUEL_PENALTY_PER_LAP`, the pit window `range(10, total_laps - 10)`, stops `[0, 1, 2]`) and `FUEL_EFFECT_K` were literals. So were the cleaning thresholds (30 s, 2× median, 50 km/h, IQR × 3, weather and car-data |z| > 3, car-data limits), the 5 min weather-merge tolerance, the model hyperparameters, the split and CV settings, `random_state=42`, and every data/results path and file name. Trying a different pit loss or another weekend's files meant editing code, and the same value (e.g. the model bundle path, the seed) was repeated across scripts.
  - **Fix:** move them to one settings file that every script reads.
  - **Fixed:** settings live in `.env` at the repo root, which is committed: there are no secrets, and a fresh clone runs as-is. The new `scripts/env_config.py` parses it, with no new dependency. Variables already set in the environment win, so `$env:PIT_LOSS_S="25"; python strategy_simulator.py` overrides one run. The getters (`env_str` / `env_int` / `env_float` / `env_list`) raise `KeyError` on a missing key instead of falling back to a hidden default. Each script reads its keys once at module level. `FUEL_EFFECT_K` stays exported from `feature_config.py`, so its importers are unchanged.
    - **Kept in code** (schema or format, not settings): the column lists and `COMPOUNDS`, the throttle 0–100 / brake {0, 100} encodings, the XGBoost objective, `n_jobs`, plot cosmetics, the t-test placeholder (#5) and the dead `compound_HARD = 1` start setting (#12).
    - **Checks:** the full pipeline was run before and after. All 69 output CSV/JSON/TXT files are identical (CSVs to rtol 1e-9; the RandomForest RMSE differs only in the last digit between runs anyway). The exceptions are the unseeded t-test samples and p-value (#5) and one log line now reading "within 5min". The best strategy is unchanged (SOFT–SOFT–SOFT, pits 12 and 13, 5695.12 s). With `PIT_LOSS_S=30` it becomes 5711.12 s, which is 2 stops × 8 s more, as expected.
  - **readme.txt:** Running the Pipeline.

- [x] **#23 Strategy is only produced for the two Ferraris**
  - **Where:** `RACE_CARS=16,44` in `.env`, read by `scripts/tyre_sets.py` and `scripts/strategy_simulator.py`.
  - **Problem:** the model is trained on the whole field and has no driver input, but `tyre_sets.py` built set inventories for the two Ferraris only, and the simulator looped over both. Each car could fit only its own sets, so every run declared two results, and no other car could be simulated.
  - **Fix:** run the race for one chosen car, with inventories available for every car.
  - **readme.txt:** intro; Data Collection step 5 (stand-ins); Race Simulation, tyre sets and step 4.
  - **Fixed:**
    - **`.env`:** `RACE_CARS` is replaced by `SIM_DRIVER=16`, the race car the strategy is simulated for. `FP_STAND_INS` now maps all six FP1 stand-ins to their race cars: `38:16,34:14,37:1,46:55,50:87,72:63`. This comes from OpenF1's `drivers` endpoint, FP1 (session 10007) against FP2 (10008); each team ran exactly one stand-in. OpenF1 lists #38 as Dino Beganovic, not Bearman as the old comment said.
    - **`tyre_sets.py`:** loops over every car in the practice stints after the stand-in mapping (20 cars), with no logic change, and logs the field-wide match counts. Of 275 stints, 143 are new, 108 matched, 16 ambiguous and 8 unmatched (cars 4, 5, 10, 12, 18, 22, 23, 55: no set left at that age, so they take a new set at that age). Cars 6, 10, 22 and 55 run past the MEDIUM allocation and car 23 past the HARD one, which adds a set with a warning. Car 30 has no MEDIUM left for the race.
    - **`strategy_simulator.py`:** filters the inventory to `SIM_DRIVER`. An unknown number raises `ValueError` listing the available cars. The per-car loop is gone; set choice, infeasible-sequence skipping and the rollouts are unchanged. `best_strategy.json` is flat (`time_units`, `car_number`, `best_compound_sequence`, `best_set_ages_at_start`, `best_pit_laps`, `best_total_time`), and `final_summary.txt` has one line. Only the chosen car's (compound, set age) pairs are rolled out (4 for car 16, was 7 for both cars).
    - **Checks:** the inventory rows for cars 16 and 44 are identical to before. The train and CV metrics are unchanged (the RandomForest RMSE differs only in the last digits between runs anyway). `SIM_DRIVER=16` reproduces the previous car-16 result exactly, and `SIM_DRIVER=44` the car-44 one. `SIM_DRIVER=99` raises with the list of 20 cars.
    - **Effect on strategy:** for car 16 (default), HARD(0)–SOFT(6), pit on lap 45, 5840.85 s, from 12,210 feasible strategies (3,996 skipped). Other cars, e.g.:
      - Car 1: same result as car 16.
      - Car 44: SOFT(4)–HARD(0), pit on lap 14, 5840.39 s.
      - Car 30: no MEDIUM left, so only 2,072 strategies are feasible; SOFT(3)–HARD(0), pit on lap 10, 5839.87 s.
      - Cars differ only through their sets, and the rollouts are almost flat, so the gaps are below 1 s. This is still **not** a valid recommendation (#18).
    - **Since #24:** `FP_STAND_INS` is derived from `data/raw/drivers/` instead of set in `.env`; for Bahrain 2025 it gives the same six pairs.

- [x] **#24 The race weekend is fixed to Bahrain 2025 and raw data is downloaded by hand**
  - **Where:** `data/raw/` (filled with the curl commands in README.md), and the weekend constants in `.env`: `NUM_LAPS=57` and `FP_STAND_INS`.
  - **Problem:** running another weekend meant looking up its session and meeting keys, re-downloading ~70 files by hand over the current ones, and editing the race length and stand-ins. Switching back meant downloading Bahrain again.
  - **Fix:** choose the weekend in `.env` from a hardcoded table of 2024 and 2025 session keys, and have `master.py` fetch it, keeping every weekend fetched in a local cache.
  - **readme.txt:** Data Collection; Running the Pipeline; Race Simulation, tyre sets.
  - **Fixed:**
    - **New stage `scripts/race_weekend.py`** (step 0 of `master.py`, before `process_laps.py`).
      - `WEEKENDS` hardcodes every non-sprint 2024 and 2025 weekend (36): meeting key, FP1–FP3 session keys and scheduled race laps. The keys come from OpenF1 `/v1/sessions?session_type=Practice` and `/v1/meetings`. The race laps were checked against the highest race lap in OpenF1 and all match, except 2025 Austria, which ran 70 laps, so the table uses 70.
      - The 12 sprint weekends (FP1 only) are listed in `SPRINT_WEEKENDS` and raise an error that says why. An unknown name raises with the valid names for that year.
      - Session keys rise FP1 → FP3 on every weekend (2024 Abu Dhabi too: 9461, 9656, 9657), so `tyre_sets.py`'s sorted-key session order still holds.
      - `selected_weekend()` is importable with no side effects; `strategy_simulator.py` takes its race laps from it.
    - **Cache:** `data/cache/<year>_<name>/` mirrors `data/raw/` and is gitignored (~67 MB per weekend); `data/raw/` stays the active weekend, so no stage's paths changed.
      - If raw already holds the complete selected weekend (`meeting_key` in `laps_p1.csv`), the stage does nothing.
      - Otherwise raw is moved into its own weekend's cache folder (a file already cached is kept and the raw copy dropped), the selected weekend's folder is completed from OpenF1, and it is copied into raw.
      - Downloads write to `.part` and rename, so an interrupted run resumes. They wait `OPENF1_REQUEST_DELAY_S` between requests and retry HTTP 429/5xx and network errors with backoff.
      - Laps, stints, drivers and weather with no results raise. A car-data query with no results leaves a `.nodata` marker, so it isn't requested again.
      - It uses `urllib` from the standard library: `requests` is pinned in requirements.txt but isn't installed in `.venv`.
    - **Car numbers follow the weekend:** car data is requested for the drivers in each session's downloaded laps file, after the laps are in.
    - **Stand-ins:** `tyre_sets.py` derives them from the new `data/raw/drivers/drivers_p1-3.csv` (OpenF1 `drivers` per session). The FP3 lineup is taken as the race cars. In FP1/FP2, a driver not in it is paired with the one driver of the same team who sat that session out; anything else is logged and left unmapped. This also covers drivers replaced before FP3.
    - **`process_cardata.py`** clears its per-file `process_cardata_cleaned_car_data_*.csv` outputs first, since the driver numbers in their names change between weekends.
    - **`.env`:**
      - New keys: `RACE_YEAR=2025`, `RACE_WEEKEND=bahrain`, `OPENF1_BASE_URL`, `OPENF1_REQUEST_DELAY_S=2.0`, `OPENF1_MAX_RETRIES=5`, `OPENF1_TIMEOUT_S=120`, `RAW_DIR`, `DRIVERS_DIR`, `CACHE_DIR`, `DRIVERS_FILES` and `CARDATA_FILE_FORMAT`.
      - Removed: `NUM_LAPS` and `FP_STAND_INS`.
    - **Checks:**
      - **First run on the committed Bahrain 2025 data:** it moved 67 files into the cache and downloaded only the 3 drivers files. Every other raw file is byte-identical (no git diff). A second run does nothing.
      - **Stand-ins:** the derived stand-ins for Bahrain 2025 are exactly the old `38:16,34:14,37:1,46:55,50:87,72:63`, and the tyre-set inventory is unchanged.
      - **Full pipeline, Bahrain 2025:** the same strategy as before, HARD(0)–SOFT(6), pit on lap 45, 5840.85 s. The only differences are log lines, the RandomForest's last digits and the unseeded t-test (#5).
      - **`RACE_YEAR=2024 RACE_WEEKEND=saudi_arabia`:** downloads 70 files (all 60 car-data files present, ~290 s) and runs end to end. The field includes #2, #3, #11, #20, #24 and #77, with #38 in FP3 only. Derived stand-ins: `55:38` (Bearman for Sainz). 50 race laps. Car 16: SOFT(0)–MEDIUM(3), pit on lap 16, 4827.39 s.
      - **Switching back to Bahrain 2025:** loaded from the cache with no requests, and the results match the Bahrain run above.
      - `RACE_WEEKEND=miami`, `nowhere` and `RACE_YEAR=2023` each raise the expected error.
    - **Still open:** the tyre allocation and return rules, `PIT_LOSS_S`, the fuel settings and the cleaning thresholds are the same for every weekend and were tuned on Bahrain 2025. Results on other weekends carry every open item above (#18 in particular).

- [ ] **#25 No exploratory data analysis**
  - **Where:** no script. `readme.txt` has an "EXPLORATORY DATA ANALYSIS AND STATISTICAL Inference" heading with nothing under it, and `master.py` has no EDA stage.
  - **Problem:** nothing looks at the data before it is modelled. The only plots judge the model after training: the prediction overlay, metric chart and feature importance in `train_model.py`, and the residual plot in `validate_strategy.py` (#6).
  - **Effect:** the cleaning thresholds, the feature set (36 inputs on 153 target rows) and the reading of the flat rollouts (#16, #20) aren't backed by a look at the data. For example, nothing shows whether the practice laps contain a compound or tyre-age effect for the model to learn.
  - **Fix:** a new stage `scripts/eda.py`, run after `feature_engineer.py` in `master.py`, with its own `results/eda/` directory, log and manifest. It reads `data/processed/final_engineered_dataset.csv`. Cover:
    - [ ] **Summary statistics and missing values** per column and session.
    - [ ] **Distributions** of `lap_duration` / `fuel_corrected_pace` by session and compound.
    - [ ] **Pace against tyre age:** `fuel_corrected_pace` against `tyre_age` per compound, i.e. the degradation the model is meant to learn.
    - [ ] **Correlations** between the 36 inputs and with the target, flagging collinear groups (sector times, speed traps and car-data aggregates).
    - [ ] **Row coverage:** rows per session, compound and stint before and after the next-lap target shift (467 → 153, with only 7 FP3 rows).
  - Any thresholds go in `.env`; take the column lists from `feature_config.py`.
  - **readme.txt:** Exploratory Data Analysis and Statistical Inference.

- [ ] **#26 No statistical inference on the practice data**
  - **Where:** no script. The only statistical test is the placeholder t-test in `validate_strategy.py` (#5), which runs on random numbers.
  - **Problem:** nothing tests whether the effects the strategy depends on are real in the data: differences in pace between compounds, tyre degradation, and pace changes between sessions.
  - **Fix:** add tests on the real laps, in the #25 stage or next to it:
    - [ ] **Compound pace:** do SOFT, MEDIUM and HARD differ in `fuel_corrected_pace`? Use pairwise Welch t-tests or one-way ANOVA / Kruskal–Wallis, within session so the session's pace level doesn't confound it.
    - [ ] **Degradation:** is the slope of `fuel_corrected_pace` against `tyre_age` per compound different from 0? Report the slope with a confidence interval.
    - [ ] **Session pace:** do FP1, FP2 and FP3 differ (track evolution)?
  - **Watch out:** laps within one stint aren't independent (the same issue as #18), so per-lap tests overstate significance. Test on stint-level values (e.g. per-stint slopes or means) or account for the grouping.
  - `scipy` is already pinned. `statsmodels` isn't installed, so confirm before adding a dependency.
  - **Depends on:** #25 (same data, likely the same stage). A real test here could replace the #5 placeholder.
  - **readme.txt:** Exploratory Data Analysis and Statistical Inference.

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

---

## Things to explore

Ideas that could make more data usable, not confirmed bugs. Each one changes the results, so check the effect before adopting it.

Context: the 1,257 raw practice laps (426 / 522 / 309) are cleaned down to 467 (164 / 221 / 82), and only 153 rows survive the next-lap target shift in `train_model.py`. Removed by `process_laps.py`: 275 pit-out laps, 7 invalid-sector or extreme lap times, 102 for "unrealistic speed", 34 IQR outliers and 372 non-push laps.

- [ ] **Speed filter drops laps with a missing speed trap, not just slow ones**
  - **Where:** `scripts/process_laps.py`: `df[(df[speed_cols] > MIN_SPEED_KMH).all(axis=1)]`.
  - **Problem:** `NaN > MIN_SPEED_KMH` is False, so a lap missing any of `i1_speed` / `i2_speed` / `st_speed` is dropped as "unrealistic speed". The filter removes 102 laps (25 / 70 / 7). The non-pit-out raw laps with at least one missing speed number 24 / 71 / 7, so nearly all of the 102 are missing data, not slow laps.
  - **Effect:** FP2, the session with the long runs, loses the most laps this way.
  - **Idea:** drop a lap only when a speed is present and below the threshold, and let missing speeds fall through to the training medians. Or require only the speed columns the model actually uses. That would recover up to ~100 laps.

- [ ] **Push-lap filter may be too strict for long runs**
  - **Where:** `scripts/process_laps.py`: `apply_push_lap_filter()`, `PUSH_LAP_MAX_BEST_RATIO=1.15` in `.env` (#20).
  - **Problem:** the cutoff is relative to the driver's best lap in the whole session, which is usually a low-fuel SOFT lap. Heavy-fuel long-run laps, which are what the race simulation needs, sit close to that cutoff.
  - **Effect:** 372 laps (145 / 109 / 118) are removed, the largest single filter.
  - **Idea:** try a looser ratio, or apply the ratio per stint (against the stint's best lap) so long runs are judged against themselves while cool-down laps are still removed.

- [ ] **Next-lap target loses most rows**
  - **Where:** `build_next_lap_target()` in `scripts/feature_config.py`; `train_model.py` logs "Rows after target shift drop: 153".
  - **Problem:** the target needs the same driver's lap `lap_number + 1` in the same stint, and that lap must also survive cleaning. Practice runs alternate push and cool-down laps, so dropping the cool-downs breaks most consecutive pairs.
  - **Effect:** 467 cleaned laps give only 153 training pairs, about 120 after the train/test split.
  - **Idea:** allow the next *surviving* lap in the stint (lap *t+k*) as the target, with *k* as a feature, so the model learns pace k laps ahead. This is a bigger modelling change: the simulator's rollout would need to match, and every feature must still belong to lap *t* (no leakage).
