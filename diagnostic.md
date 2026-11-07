# Diagnostic: TODO list

The pipeline runs end to end (`python master.py` from `scripts/`), but the open items below affect how far the current output can be trusted.

Open items come first, then ideas to explore, then the fixed items. Within each group, items are split by priority and sorted by ID. Fixed items are kept short: what was wrong, what changed and the effect that matters now. The full before → after numbers of each fix are in its commit and its closed GitHub issue.

Every item keeps a stable ID (`#N`), which is also its GitHub issue number (open items are open issues, fixed ones are closed). `readme.txt` marks every place an open item applies with a `[TODO #N, P?, see diagnostic.md]` line. When an item is fixed: tick it, say what changed, move it under **Fixed** at its priority, remove or update its tags in `readme.txt`, and close the issue. Open a new issue for a new item, so the numbers stay in step.

**Priorities**
- **P1, results are invalid until fixed.** The training data, the target or the strategy objective is wrong, so the recommendation can't be trusted.
- **P2, method or validation is misleading.** The code doesn't do what `readme.txt` claims, or the validation overstates the model, but the inputs aren't corrupted.
- **P3, completeness and cleanup.** Dead settings, unfinished scope, and the results write-up.

---

## Open

### P2: method or validation is misleading

- [ ] **#27 The model choice rests on three untuned models and untested reasoning** (now: the fall-off model and the compound gap are not settled)
  - **Where:**
    - `scripts/train_model.py`: its MODEL block is the one place the model is defined (an sklearn `Pipeline` and `MODEL_NAME`; still XGBoost, depth 7, from `XGB_*` in `.env`), on `FEATURE_SETS[FALL_OFF_FEATURE_SET]`. Bundle: `train_model_fall_off_model.pkl`.
    - `scripts/feature_config.py`: `FEATURE_SETS`, `stint_new_set_levels()` and the model-free `fit_compound_gap()`.
    - `scripts/validate_strategy.py` and `scripts/strategy_simulator.py`: they read `bundle["features"]` and call `predict`; validation fits a `clone()` per fold. The simulator enumerates 1- and 2-stop strategies by brute force.
    - `explore/explore_master.py` + `explore/explore_lib.py`: the model comparison.
    - `readme.txt`: ML Modelling step 3 (model choice, reasoning, results table) and Race Simulation step 4.
  - **Problem now:**
    - The pipeline's XGBoost is a placeholder from the old next-lap model and beats neither baseline: validate CV shape MAE 1.053 s against flat 1.034 s and linear 0.900 s (better than flat on 46 of 94 stints, than linear on 28).
    - explore/ (2026-11-05) can hardly separate the better candidates: the MLP (`plus_age_at_start`) 0.888 ± 0.113 s, Ridge (`tyre_only`) 0.890 and the per-compound line 0.905 are within one fold std. The MLP is the only rule-passing candidate that beats linear significantly (Wilcoxon p 0.038).
    - The compound gap is unstable (Step 2), and the strategy rests mostly on it, since the shortlisted curves have nearly the same slope on every compound.
    - The simulator's brute force can't search more stops or every set choice exactly (Step 3).
  - **Plan (2026-11-01): make the pipeline independent of the model, then plug the model in last.**
    - **Goal:** nothing up to the fall-off model depends on which model is chosen (lap cleaning and its tuned factors, the stage-1 stint levels, the compound gap, the race pace level). The model only predicts each compound's fall-off with tyre age. The simulator takes that fall-off and searches every legal strategy for the fastest. Data scope is unchanged: the selected weekend's FP1–FP3 only.
    - **Decisions (user, 2026-11-01):**
      - Make the pipeline independent first, plug the model in last. Dynamic programming in the simulator is part of the target design.
      - The compound gap needs better estimation for the data constraints, not more data ("the project is FP data only"). Pooling the other cached weekends in a hierarchical model was proposed and rejected.
      - Nothing starts without the user asking. The model choice (Step 4) and the new DP issue (Step 3) need the user's OK.
      - Branch: `model-choice-27`, off `fall-off-target-31`.
    - **Workflow:** the steps run in order. After each one, `master.py` and `explore_master.py --skip-pipeline` must run end to end. Each step is one commit with its before → after numbers.
    - **Swap test** (acceptance check from Step 1 on): run train_model.py with two different models (e.g. XGBoost and Ridge). The cleaned data, the stage-1 levels and slopes, `compound_gap` and the driver-session `levels` must come out identical; only the fall-off curves, the stage-2 metrics and the strategy may differ.
  - [x] **Step 0: check the workflow** (done 2026-11-01, no code changes). `master.py` reproduced every tracked file byte for byte, and `explore_master.py --skip-pipeline` reproduced every number (then 573 laps, 436 rows / 96 multi-lap stints; Ridge `tyre_only` 1.0478 recommended). `--quick` must run with `RESULTS_ROOT` outside the repo, or it overwrites the tracked full-run results.
  - [x] **Step 1: a model-independent pipeline** (done 2026-11-02, commit a72dd8c; full explore rerun 2026-11-05).
    - **Code:**
      - The compound gap is fitted on each stint's stage-1 new-set level (`stint_new_set_levels()`, lap-weighted `fit_compound_gap()`), not on `fcp − fall_off(model)`. It equals the old per-lap fit with the linear fall-off (checked to 1e-11).
      - `FEATURE_SETS` moved to feature_config.py (leak-asserted); train_model.py's model is a `Pipeline` picked by `FALL_OFF_FEATURE_SET`; the bundle is `train_model_fall_off_model.pkl` with `model_name` / `feature_set`; validate_strategy.py fits a `clone()` per fold; feature importances are saved only if the model has them.
      - tune_iqr.py / tune_push.py score their CV sweeps with the model-free stage-1 linear fall-off (tune_iqr.py 87 → 10 min). A full run writes its pick into `.env` itself (`env_config.write_env()`); `--quick` never writes.
    - **Decisions (user, 2026-11-02):** stage 3 keeps the train/test split (it fits on the training stints). tune_iqr.py writes its own winner; tune_push.py writes the label pick, not the CV winner (#30).
    - **Swap test** (XGBoost vs Ridge on `core`): `compound_gap`, `levels`, `linear_slopes`, `medians` and `features` identical; only the curves, the stage-2 metrics and the strategy differ.
    - **Tune reruns (12 weekends):** `IQR_LOWER_MULTIPLIER` 60 → **62** (largest k needed 51.23, 2025 Spain FP2), `IQR_UPPER_MULTIPLIER=3`; `PUSH_LAP_MAX_STINT_RATIO` off → **1.1** (label pick (1.15, 1.10), J 0.986). The stint test removes 7 Bahrain laps (573 → 566; multi-lap 436 / 96 → 427 / 94).
    - **Before → after (Bahrain 2025, car 16):**
      - Bundle gap MEDIUM +2.972 / HARD +0.848 (net of XGBoost's fall-off) → **+1.156 / +0.295** (model-free, training split). On all 566 laps: **+0.978 / +1.228**.
      - validate's per-fold gap MEDIUM 0.78–2.04 → 0.14–1.54 s, HARD 0.46–1.98 → 0.36–1.76 s. Validate CV shape MAE 1.2492 → 1.0534 (flat 1.0344, linear 0.8999).
      - eda slopes S +0.366 → +0.341, M +0.227, H +0.306 → +0.211 s/lap (HARD now not significant).
      - Race pace level 93.06 → 94.42 s. Strategy SOFT(0)–SOFT(0)–HARD(0), pits 14/28, 5649.51 s → **SOFT(0)–SOFT(0)–MEDIUM(0), pits 14/28, 5625.24 s**.
    - **explore (2026-11-05, 57 configurations, 26 min):** baselines flat 1.035, linear 0.905. SVR fits best (0.877) but its curve gets faster with age, so the rule rejects it. Recommendation moved from Ridge (`tyre_only`, the 2026-10-28 pick) to the **MLP (`plus_age_at_start`) 0.888**: slopes S 0.18 / M 0.17 / H 0.17 s/lap, largest step 0.18 s, monotone. Its lead over Ridge (0.890) is 0.002 s against a fold std of ~0.11 s. One model-free gap is shared by every candidate (before Step 1 it ranged over MEDIUM +0.15 to +2.05 s across them).
  - [ ] **Step 2: a better compound-gap estimator** (model-free).
    - **Problem:** stage 3's gap on a new set is unstable, and on the training split it is implausible. Since Step 1 these are the estimator's own numbers (Bahrain 2025, 566 laps):
      - Training split (the bundle): MEDIUM +1.156 s, HARD +0.295 s, so MEDIUM is slower than HARD. On all 566 laps: MEDIUM +0.978, HARD +1.228 (S < M < H).
      - It moves a lot with the data: across validate_strategy.py's folds MEDIUM +0.14 to +1.54 s and HARD +0.36 to +1.76 s. On the training split HARD went from −0.35 to +0.30 s when the stint test removed 7 laps.
      - Why:
        - With one fixed level per driver-session, it is identified only within the 44 driver-sessions that ran two or more compounds.
        - HARD and MEDIUM are compared directly in just 1 of them.
        - `run_laps` is nearly collinear with compound within a driver-session (median run SOFT 4, MEDIUM / HARD 6 laps).
      - eda.py 6b's paired new-set comparison gives MEDIUM +1.61 s and HARD +2.23 s, with no `run_laps` control. Stage 3 with the control gives MEDIUM +0.98 s and HARD +1.23 s on all laps, with `run_laps` at +0.36 s per lap of run on the training split. How much of the difference it accounts for hasn't been measured yet; G1 vs G0 measures it.
      - Session evolution is large (eda.py 6c: FP2 2.77 s faster than FP1, FP3 about the same as FP2), so G3's session effect must be fitted, not pooled away.
    - **Where:** `explore/explore_lib.py`; `explore/explore_master.py` (a new `--gap-only` section); then `fit_compound_gap()` in `scripts/feature_config.py`, plus `.env`.
    - [ ] **Candidates**, on Step 1's stint new-set levels, numpy / scipy / scikit-learn only:
      - **G0:** today's structure.
      - **G1:** no `run_laps` control.
      - **G2:** run-type controls (`run_laps`, used-set start, a short/long-run indicator).
      - **G3:** crossed partial pooling: a driver effect, a session effect and a driver × session deviation shrunk with a ridge penalty (≙ random effects), the penalty tuned by grouped CV. Driver-sessions on a single compound then count through the shared driver effect.
      - **G4:** G3 with S ≤ M ≤ H enforced (`scipy.optimize.lsq_linear` on non-negative increments).
      - **G5:** G3 on run-type-matched stints only.
    - [ ] **Scoring:**
      - Grouped CV by stint, predicting each held-out stint's new-set pace. Lap-weighted MAE, with the same scored stints for every candidate.
      - A stint cluster bootstrap for 95% CIs of the MEDIUM and HARD gaps and P(S < M < H).
      - The spread of the gap across folds.
      - The order of the compounds, against eda.py 6b.
      - Pick rule: the best prediction among the candidates with a plausible order, with the narrowest CI breaking ties.
    - [ ] **Apply the winner** in `fit_compound_gap()`. Its tunable values become commented `.env` keys.
    - **Verify:** validate_strategy.py's per-fold gap is narrower than now and plausibly ordered in every fold; the swap test still gives an identical gap; `master.py` exits 0.
  - [ ] **Step 3: dynamic programming over strategies in the simulator** (new issue, to open with the user's OK; readme.txt "Future work").
    - **Where:** `scripts/strategy_simulator.py`, `.env`.
    - **Input:** the per-(compound, set age) stint cost table the simulator already builds (cumulative fall-off + gap + race pace level + fuel). This is the only place the model's output enters.
    - **State:** race lap, the set on the car, a bitmask of the car's sets already used, the compounds used and the stops made.
    - **Transitions:**
      - Either continue on the current set, or pit at the end of the lap onto an unused set.
      - Pits only within the pit window, each adding `PIT_LOSS_S`.
      - `MIN_COMPOUNDS_USED` is enforced at the finish.
      - `MAX_STOPS` (`.env`) replaces `STOP_COUNTS`.
    - **Output:** the best strategy, plus the best per stop count and per compound sequence (for the alternatives table), with back-pointers for the pit laps and sets.
    - **Verify:** with `MAX_STOPS=2`, run it once side by side with today's brute force: the best strategy and total must be the same. With 3 stops allowed, it finishes in seconds. Each total equals its stint totals + fuel + pit loss.
  - [ ] **Step 4: plug in the fall-off model.**
    - **Where:** train_model.py's MODEL block; `.env`: `FALL_OFF_FEATURE_SET` and the model's hyperparameter keys (e.g. `RIDGE_ALPHA`), with the reasoning in a comment.
    - Rerun `explore_master.py --skip-pipeline` after Steps 2 and 3: its curve and order checks use the model-free gap, which Step 2 changes.
    - Choose between (2026-11-05 run):
      - The MLP (`plus_age_at_start`): the rule's pick, 0.888 s, the only rule-passing candidate that beats linear significantly (Wilcoxon p 0.038).
      - Ridge (`tyre_only`): 0.890 s; one shared slope, the most stable fold curve. eda.py 6a finds no significant slope difference between compounds.
      - The per-compound line: 0.905 s, = the linear baseline (interpretable, compound-specific, rises without limit past age 19).
    - **Verify:** the swap test passes (nothing upstream of stage 2 moves); validate_strategy.py's shape MAE matches explore's for that configuration; the curves are smooth and rise on every compound; the DP strategy is logged.
  - [ ] **Step 5: documentation, and close #27.** The docs were brought up to date after Step 1 (2026-11-05, rewritten 2026-11-08); what is left comes after Steps 2–4.
    - **`readme.txt`:** ML Modelling steps 2–3 (the plug-in model), the reasoning (keep each claim only if the experiments back it), the results table, Race Simulation (DP and the current output), Evaluation, and the `[TODO #27]` tags.
    - **`README.md`:** results, strategy and project status.
    - **`AGENTS.md`:** the model plug-in point, the tune scorer and DP.
    - **Here:** move #27 to Fixed (P2), update #15's "Depends on", and close the issues.
  - **History:** #13 (the three-model comparison) is merged into this item. The first explore runs (2026-10-26 on the next-lap target, 2026-10-28 on the fall-off target before the tuned thresholds) built the harness: grouped nested CV, 13 candidates, flat / linear baselines, the curve checks and `MIN_MONOTONE_SHARE` (added after SVR's curve was found to fall back past the training ages). Their numbers are superseded by the 2026-11-05 run.
  - **Depends on:** nothing left.

### P3: completeness and cleanup

- [ ] **#15 Results documentation**
  - **Problem:** results exist only as raw files in `scripts/results/<script>/` and a few numbers in the readmes.
  - **Depends on:** #27. The numbers will change once it is settled, so write this up afterwards.
  - **readme.txt:** Known Issues.
  - Cover:
    - [ ] **Data cleaning:** rows removed per session and per filter (from the `*_removed_rows_summary.csv` files and the weather z-score summary) and the final row counts, including car data.
    - [ ] **Feature engineering:** final dataset shape, the multi-lap stints and the rows the model learns from.
    - [ ] **Model comparison:** explore/'s leaderboard with the fold spread, the fall-off curves per compound and what they show.
    - [ ] **Validation:** CV per fold with the spread, the tests against the flat and linear baselines, the residual plot and the curve uncertainty.
    - [ ] **Strategy:** the recommended strategy (compound sequence, sets and pit laps), predicted race time, the next-best alternatives and the gap to them.
    - [ ] **Limitations:** which results depend on items still open.
    - [ ] Decide where this lives (a results section in `readme.txt` / `README.md`, or a separate `RESULTS.md`) and link it from both readmes.

---

## Things to explore

Ideas that could make more data usable, not confirmed bugs. Each one changes the results, so check the effect before adopting it.

Context (Bahrain 2025): the 1,257 raw practice laps (426 / 522 / 309) are cleaned down to 566 (186 / 295 / 85). The model learns from the 427 laps of the 94 stints with two or more clean laps; the compound-gap fit uses all 566. Removed by `process_laps.py`: 275 pit-out laps, 5 invalid-sector or extreme lap times, 1 for unrealistic speed, 67 IQR outliers (all slow), 336 non-push laps on the session test and 7 more on the stint test.

None at the moment.

---

## Fixed

### P1: results are invalid until fixed

- [x] **#1 Weather comes from a different event**
  - **Problem:** the weather was downloaded with `meeting_key=1253` (Bahrain pre-season testing), not 1257 (the Grand Prix), so `feature_engineer.py` spread it over the laps by "pseudo time" and each lap's weather was arbitrary.
  - **Fixed:** weather re-downloaded for 1257; `process_weather.py` keeps the practice sessions only; `feature_engineer.py` joins on timestamps (`merge_asof` on `date_start` ~ `date`, 5 min tolerance), every lap within 30 s.

- [x] **#2 Stint merge joins on the wrong stint number**
  - **Problem:** laps got a global stint number (1–18 per driver across sessions) but were merged against the per-session `stint_number` (1–9), so laps were duplicated with different compounds (839 laps → 1,126 rows) and 507 rows had no `fuel_corrected_pace`.
  - **Fixed** (with #3): the merge uses the global number on both sides, one-to-one (`validate="many_to_one"`). Every lap's compound and lap range checked against its own session's raw stints, 0 mismatches. The old metrics were inflated by the duplicates.

- [x] **#3 Laps are assigned to stints without checking the session**
  - **Problem:** `session_key` was dropped in cleaning, so a lap was matched on driver and lap range only, and FP1 lap 5 could get an FP2 stint.
  - **Fixed:** `process_laps.py` / `process_stints.py` drop only `meeting_key`, and `assign_stint_numbers()` matches on the session too, numbering global stints by `(driver_number, session_key, stint_number)`.

- [x] **#4 The "next lap" target can come from another driver**
  - **Problem:** the target was shifted within `stint_number` only, after `driver_number` was dropped, so a row's "next lap" was often another driver's lap.
  - **Fixed:** `session_key` and `driver_number` stay in the engineered dataset as identifiers (not inputs), and the target was built per driver and stint, only where the next lap was `lap_number + 1`. (The next-lap target itself was replaced in #31.)

- [x] **#7 The simulator mixes units**
  - **Problem:** `lap_duration` was standardised per session, so the model predicted z-scores while the fuel term and the 22 s pit loss were in seconds; one stop cost as much as ~15 laps and the optimiser always picked 0 stops.
  - **Fixed:** lap times, sectors and speeds are no longer scaled, so the target, predictions, fuel, pit loss and metrics are all in seconds. A per-session scaler would remove each session's pace level, with no race session to convert back with. Session offsets went into a session one-hot (now absorbed by the stint level, #31).

- [x] **#8 Illegal strategies are allowed**
  - **Problem:** `generate_strategies()` included 0-stop and single-compound strategies, which the two-compound rule forbids.
  - **Fixed:** sequences with fewer than `MIN_COMPOUNDS_USED` (2) distinct compounds are skipped and counted; `STOP_COUNTS=1,2`. Bahrain: 18,318 → 16,206 strategies (222 one-stop, 15,984 two-stop).

- [x] **#10 Tyre age does not reset after a pit stop**
  - **Problem:** predictions were indexed by race lap, so a stint starting on lap 30 was priced as if its tyres were 30 laps old.
  - **Fixed** (with #9): each stint is priced from its own start, so a stint costs the same wherever it starts, apart from the fuel. Since #21 it starts at the age of the set the car fits.

- [x] **#11 The pit lap is missing from the race-time sum**
  - **Problem:** stints were summed over `(last, p − 1)`, so lap `p` was never counted, and each extra stop dropped a ~100 s lap for 22 s of pit loss: the optimiser picked 2 stops on consecutive laps.
  - **Fixed:** every race lap is summed once. "Pit on lap `p`" means lap `p` is the in-lap and the next stint starts on `p + 1`; `PIT_LOSS_S` is added on top. Checked: stint lengths sum to the race, and every total equals its stints + race fuel + 22 s × stops (to 3e-12 s).

- [x] **#19 The fuel correction has the wrong sign and horizon**
  - **Problem:** `fuel_corrected_pace = lap_duration + 0.035 × laps_remaining` added the fuel effect instead of removing it, with `laps_remaining = lap_end − lap_number`.
  - **Fixed:** `fuel_corrected_pace = lap_duration − k × laps_remaining`, `laps_remaining = lap_end − lap_number + 1`. Assumption: practice fuel isn't recorded, so the laps left in the stint (counting lap *t*) stand in for the fuel on board; this matches the race term, so race fuel is added once, in the simulator only.

- [x] **#34 Other weekends' stints and laps break stage 1** (found by #27 Step 1's 12-weekend tune reruns)
  - **Problem:** three cases Bahrain 2025 doesn't have: a session with no stint on a compound left a NaN one-hot column (2025 Australia FP1, no HARD), which crashed stage 1's least squares; non-dry stints (2025 Austria `UNKNOWN`, Netherlands `INTERMEDIATE`) had all-0 one-hots and read as SOFT; and laps with no `date_start` (2025 Spain FP3: 11) broke the weather `merge_asof`.
  - **Fixed (2026-11-02):** process_stints.py keeps only SOFT / MEDIUM / HARD and fills a missing compound column with 0 after the concat; feature_engineer.py drops (and counts) laps with no stint or no `date_start`. Bahrain's `data/processed/` is byte-identical.

### P2: method or validation is misleading

- [x] **#5 The t-test is a placeholder**
  - **Problem:** `validate_strategy.py`'s "T-TEST EXAMPLE" compared two sets of unseeded random numbers, so its p-value meant nothing.
  - **Fixed:** replaced by a per-stint paired test of the model against a naive baseline (one-sided paired t-test and Wilcoxon at `SIGNIFICANCE_LEVEL`), each stint one observation since its laps aren't independent. Since #31 the baselines are flat (no fall-off) and linear (stage-1 slope per compound).

- [x] **#6 The residual plot mixes training and test data**
  - **Problem:** the last CV fold's model predicted every row, including the ones it was trained on, which understated the error ~5× (MAE).
  - **Fixed:** the plot uses out-of-fold predictions, each row scored by the model that held its stint out (`validate_strategy_oof_predictions.csv`).

- [x] **#9 The simulation does not roll predictions forward**
  - **Problem:** each lap was predicted on its own from fixed median features, while `readme.txt` claimed an autoregressive rollout.
  - **Fixed** (with #10): a lap-by-lap rollout fed each prediction back into the next lap's state. Since #31 it is replaced by a fall-off curve per (compound, set age), and nothing is fed back between laps.

- [x] **#13 The saved model is the weakest of the three on the test split**
  - **Merged into #27:** its fix, a grouped-CV comparison of the candidate models, became #27's first fix item (done in `explore/`). Closed so the ID stays in step with the GitHub issue.

- [x] **#17 `stint_avg_pace` contains the target lap**
  - **Problem:** the leave-one-out stint mean included lap *t+1*, the next-lap target, so part of the target was an input (top feature, inflated R²).
  - **Fixed:** `stint_avg_pace` is the mean of the stint's *earlier* laps only (`shift(1).expanding().mean()`), the lap's own time on its first lap (#33). Since #31 it is not a model input at all (`LAP_TIMING_COLUMNS`).

- [x] **#18 Train/test splits mix laps from the same stint**
  - **Problem:** laps were shuffled one at a time, so most stints were on both sides and the scores measured interpolation within seen stints, not prediction on new ones.
  - **Fixed:** train_model.py uses `GroupShuffleSplit` and validate_strategy.py `GroupKFold` (shuffled, scikit-learn ≥ 1.6), grouped on `(driver_number, stint_number)`; both log 0 stints shared between train and test.

- [x] **#20 The cleaned laps still include slow non-push laps**
  - **Problem:** cool-down, in- and traffic laps under 2× the median stayed in, so the target mixed push and slow laps (XGB MAE ~10.5 s) and new sets showed a fake slow second lap.
  - **Fixed:** `apply_push_lap_filter()` drops laps slower than `PUSH_LAP_MAX_BEST_RATIO` (1.15) × the driver's session best, the natural gap between push laps / long runs (≤ 1.10×) and cool-down laps (≥ ~1.3×). Errors fell ~10×. Tuned further in #30.

- [x] **#29 Speed filter drops laps with a missing speed trap, not just slow ones**
  - **Problem:** `NaN > 50` is False, so laps with a missing `i1_speed` / `st_speed` were dropped as "unrealistic speed": 102 laps, only 1 of them really slow, and mostly FP2 long-run laps.
  - **Fixed:** a lap is dropped only when a *recorded* speed is ≤ `MIN_SPEED_KMH` (1 lap now). The extra FP2 laps shrank FP2's IQR, which exposed #32.

- [x] **#30 Push-lap filter may be too strict for long runs** (was under "Things to explore")
  - **Problem:** the cutoff is relative to the driver's session best, usually a low-fuel SOFT lap, so heavy-fuel long-run laps might sit close to it.
  - **Fixed:** a second test against the driver's best in the same stint (`PUSH_LAP_MAX_STINT_RATIO`), tuned with `explore/tune_push.py` over 12 weekends. The label (full-throttle share from car data, independent of the ratios) picks (1.15, 1.10): 99.4% of push laps and 0.8% of cool-down laps kept. 1.15 already keeps 2,785 of 2,789 labelled push laps in runs of 8+ laps, so the concern didn't hold.
    - The CV sweep favours the strictest corner because it is scored on the strictest setting's own rows: a bias, so it is for information only. Kept at (1.15, off) in #30; the label pick (stint test on) was adopted in #27 Step 1 (user, 2026-11-02).
    - **Open idea:** a CV evaluation set that doesn't depend on the strictest configuration would judge filters fairly (a caution comment is in `tune_push.py`'s winner rule).

- [x] **#31 Next-lap target loses most rows** (was under "Things to explore")
  - **Problem:** the next-lap target needed lap *t+1* to survive cleaning, and practice alternates push and cool-down laps, so 573 cleaned laps gave only 273 rows. It also didn't fit the job: next-lap pace is mostly driver / session / run-type level, `tyre_age` was 15th in importance, and rollouts reverted to the training mean within ~3 laps.
  - **Fixed:** the model learns **tyre fall-off with age within a stint**: `fcp(lap) = level[stint] + fall_off(compound, tyre_age, conditions)`, in three stages (stint levels + slope per compound; the model on `fcp − level`; the compound gap from stints). Every lap of a multi-lap stint is a row (427 now). The lap's own timing and telemetry can't be inputs (`LAP_TIMING_COLUMNS`, asserted). Validation scores the within-stint curve shape against flat and linear baselines. The simulator prices each stint from a fall-off curve, with no rollout.
    - **Rejected:** imputing the missing next lap (at most 340 rows, same target); pace relative to a driver-session level (the run type confounded the slopes); backfitting stages 1 and 2 (didn't converge).
    - eda.py's slope bug was fixed at the same time: it de-meaned the pace but not tyre age, pulling the slopes towards 0.
    - The compound-gap problem found here moved to #27 (Step 2).

- [x] **#32 IQR residual filter drops fast push laps run between cool-down laps**
  - **Problem:** a push lap among cool-down laps has a cool-down lap as its rolling median, so its residual is far below 0, and in sessions full of long runs (FP2) the IQR is small, so one shared multiplier of 3 dropped those push laps (33 on Bahrain 2025).
  - **Fixed:** one multiplier per side (`IQR_LOWER_MULTIPLIER`, `IQR_UPPER_MULTIPLIER`), tuned by `explore/tune_iqr.py` over 12 weekends (label sweep + CV sweep). A medcouple-adjusted boxplot was rejected (it narrows the lower fence and removes 34–76 push laps per session). Winner (60, 3), rerun in #27 Step 1 → **(62, 3)**; the fast side is in effect off. Rerun tune_iqr.py when weekends are added, since the k needed grows with the densest FP2 seen.

- [x] **#33 Some dropped or median-filled data could be imputed instead** (was under "Things to explore")
  - **Problem:** the car-data z-score filter dropped whole laps for one outlying aggregate; laps missing one sector were dropped though it could be rebuilt; and 45% of target rows had a median-filled input, with the median pooled over the field and computed before the split.
  - **Fixed:** the z-score test masks only the flagged aggregate (`zscore_outlier_mask()`); a single missing sector is rebuilt as `lap_duration` minus the other two; missing speed traps, segment counts and car data are filled with the median of the same stint's *earlier* laps (`STINT_FILL_COLUMNS`); any fallback median is fitted on training rows only. Group-wise medians and a regression from `i2_speed` were rejected.

### P3: completeness and cleanup

- [x] **#12 The starting compound setting has no effect**
  - **Fixed:** the dead `compound_HARD = 1` block in `strategy_simulator.py`'s `__main__` was removed; every output was byte-identical.

- [x] **#14 The car-data stage is not functional**
  - **Problem:** `process_cardata.py` was a copy of `process_laps.py` with the wrong file names and columns, and wasn't called from `master.py`.
  - **Fixed:** rewritten to clean the real telemetry (duplicates, no-input 104 samples, out-of-range readings), encode brake / DRS, assign each ~4 Hz sample to its lap window and aggregate per lap, keeping only laps that survive `process_laps.py`. Wired into `master.py`. `car_data_p1_1.csv` was re-downloaded for #38, who drove Ferrari's #16 car in FP1.

- [x] **#16 Car data is not a model input yet**
  - **Fixed:** car data downloaded for the whole field (20 drivers per session), left unscaled, and merged per lap on `(session_key, driver_number, lap_number)`. It was a lap-*t* input to the next-lap model; since #31 it is the lap's own telemetry, so not a model input (EDA and tune_push.py's label only).

- [x] **#21 Race stints always start on a new tyre set**
  - **Problem:** every simulated stint started at tyre age 0, whatever sets the car had left.
  - **Fixed:** new stage `scripts/tyre_sets.py` chains each car's practice stints into sets (OpenF1 has no set ID: a used-set stint continues the same-compound set left at exactly that age), applies the allocation, returns and reserved sets from `.env`, and writes each car's race inventory. The simulator uses only the car's sets, each at most once, and `tyre_age` (set age + laps on it) is a feature.
    - **Limitations:** qualifying isn't used (scope is FP1–FP3), so sets used there count as available; the allocation and return rules are `.env` assumptions to check against the sporting regulations; chaining is a heuristic.

- [x] **#22 Settings are hardcoded across scripts**
  - **Fixed:** settings live in the committed `.env` (no secrets), read by `scripts/env_config.py`; a missing key raises, and an environment variable overrides one run. Schema definitions (column lists, `COMPOUNDS`, encodings) stay in code. All outputs were identical before and after.

- [x] **#23 Strategy is only produced for the two Ferraris**
  - **Fixed:** `tyre_sets.py` builds inventories for every car in practice (20), and the simulator runs for the one car in `SIM_DRIVER` (default 16); an unknown number raises with the list of cars.

- [x] **#24 The race weekend is fixed to Bahrain 2025 and raw data is downloaded by hand**
  - **Fixed:** new stage `scripts/race_weekend.py` (step 0 of `master.py`) with a hardcoded table of every non-sprint 2024 and 2025 weekend (meeting key, FP1–FP3 session keys, race laps; sprint weekends raise). It fetches the weekend set by `RACE_YEAR` / `RACE_WEEKEND` from OpenF1 into a gitignored cache, `data/cache/<year>_<name>/`, and copies it into `data/raw/`; resumable downloads, `.nodata` markers, `urllib` only. Stand-ins are derived from OpenF1's per-session entry lists. Checked on Jeddah 2024 and back on Bahrain 2025 (no requests).
    - **Still open:** the tyre rules, `PIT_LOSS_S` and the fuel settings are the same for every weekend.

- [x] **#25 No exploratory data analysis**
  - **Fixed:** new stage `scripts/eda.py` between `feature_engineer.py` and `train_model.py`, writing only to `results/eda/`: summary statistics and missing values, pace distributions, pace against tyre age (the within-stint slope, = stage 1), correlations with collinear groups (`EDA_COLLINEAR_THRESHOLD`) and row coverage.

- [x] **#26 No statistical inference on the practice data**
  - **Fixed:** section 6 of `eda.py` (scipy only), every test on stints or driver-session means, never single laps, with Holm within each family. Degradation: stage 1's slopes with a stint cluster bootstrap (`EDA_BOOTSTRAP_RESAMPLES`). Compound pace and session pace: new-set pace paired within driver-session / driver-compound (t, Wilcoxon, Friedman). Current results are in readme.txt (EDA step 6).

- [x] **#28 Redirected output crashes on non-ASCII log text**
  - **Problem:** redirected stdout on Windows is cp1252, which can't hold `→`, so `print` raised `UnicodeEncodeError` and `python master.py > run.txt` stopped at `process_laps.py`.
  - **Fixed:** `log()` encodes the printed copy with `sys.stdout.encoding` and `errors="replace"`; the UTF-8 log file keeps the original text.

- [x] **#35 SIM_DRIVER can't be left unset for a fresh-tyre race**
  - **Problem:** readme.txt says `SIM_DRIVER` only adds one car's set ages and set limits, and that unset means fresh tyres, but `strategy_simulator.py` read it with `env_int()`, so a blank value crashed.
  - **Fixed:** a blank `SIM_DRIVER=` gives new sets of every compound (enough for `max(STOP_COUNTS) + 1` stints of each), the field's pace level (mean of all driver-session levels) and `car_number` = `"field"`. A set value works as before (car 16's outputs byte-identical). Blank on Bahrain 2025: 0 strategies skipped (car 16: 3,996), same ranking and gaps as car 16, total +74.4 s from the field pace level.
