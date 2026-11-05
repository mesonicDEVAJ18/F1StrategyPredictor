# AGENTS.md

Guidance for AI coding agents working in the **F1 Race Strategy Optimiser** repository.

## What this project is

An end-to-end data-science pipeline that predicts the **optimal tyre strategy** for a
non-sprint Formula 1 race using only data from the three practice sessions (FP1/FP2/FP3).
The reference weekend is **Bahrain 2025**; any non-sprint 2024 or 2025 weekend can be chosen with
`RACE_YEAR` / `RACE_WEEKEND` in `.env` (names in `scripts/race_weekend.py`). The
model is trained on the whole field; the race is simulated for one car, `SIM_DRIVER` in `.env`
(default 16, Ferrari), using that car's tyre sets. All raw data comes from the free [OpenF1 API](https://openf1.org/).

The core idea (#31): the model learns **tyre fall-off with age**. Each lap's fuel-corrected pace is split into a
level for its stint (driver, session, fuel plan, run type: everything constant over a run) and a fall-off that depends
on the compound, the tyre age and the conditions. A linear stage fits the stint levels, an XGBoost model learns the
fall-off from what is left, and a second linear stage gives the gap between compounds on a new set. A simulator
enumerates legal pit/compound strategies, prices each stint from the fall-off curve of its set, and ranks them by
total race time.

**Target design (#27, in progress):**
- Everything before the fall-off model must not depend on which model is chosen: lap cleaning and its tuned
  thresholds, the stage-1 stint levels, the compound gap and the race pace level.
- The model is plugged in last. Its only job is to predict each compound's fall-off with tyre age.
- The simulator turns that fall-off into stint costs and searches every legal strategy for the fastest. This is
  planned to move from brute force to dynamic programming.
- XGBoost is a placeholder until #27 Step 4.
- Since #27 Step 1 nothing upstream uses it: the compound gap is fitted on each stint's stage-1 new-set level
  (`stint_new_set_levels()`), and the tune scripts' CV sweeps score with the stage-1 linear fall-off. See "Current
  work" below.

Read [readme.txt](readme.txt) for the full, authoritative narrative (problem statement, data
dictionary, cleaning rules, feature-engineering rationale, and modelling decisions). When the
code and this file disagree, trust the code; when the *reasoning* is unclear, trust readme.txt.

## Repository layout

```
scripts/            # ALL pipeline code lives here; run scripts FROM this directory
  master.py             # orchestrator — runs the whole pipeline in order
  race_weekend.py       # weekend table (session keys, race laps); puts the selected weekend in data/raw/ (cache or OpenF1)
  env_config.py         # loads ../.env into os.environ; env_str/env_int/env_float/env_list getters
  feature_config.py     # shared column lists (FEATURE_SETS, LAP_TIMING_COLUMNS, ...) and the fall-off / compound-gap helpers
  process_laps.py       # clean lap data (kept in s / km/h, not scaled), tag session_name FP1-3
  process_stints.py     # clean stints, one-hot encode compound
  tyre_sets.py          # infer every car's tyre sets from practice stints -> race inventory
  process_weather.py    # clean + scale weather
  process_cardata.py    # clean car telemetry (whole field), aggregate per lap, unscaled (EDA, tune_push label; not model inputs)
  feature_engineer.py   # merge laps+stints+weather+car data, build engineered features
  eda.py                # exploratory data analysis of the engineered dataset (read-only, no hand-off)
  train_model.py        # train the one fall-off model (XGBoost as a placeholder until #27 Step 4), persist the bundle
  strategy_simulator.py # enumerate + simulate strategies, pick best
  validate_strategy.py  # cross-validation of the fall-off curve, residual plot, tests vs flat/linear baselines (strategy robustness is planned, not built)
  results/<script>/     # per-script outputs (logs, intermediate CSVs, plots, model)
explore/            # model selection (#27), kept out of the pipeline; run FROM this directory
  explore_master.py     # alternate master: data stages + the experiment block (candidates, grids, feature sets)
  explore_lib.py        # custom estimators, nested grouped CV, fall-off target and shape scoring
  tune_iqr.py           # tunes process_laps.py's two IQR multipliers over many cached weekends (#32)
  tune_push.py          # tunes process_laps.py's two push-lap ratios (session best, stint best) the same way (#30)
  results/explore_master/  # leaderboard, per-fold CV and curves, best params, fall-off curves
  results/tune_iqr/        # label sweep, CV sweep, recommendation
  results/tune_push/       # labelled laps, label sweep, CV sweep, recommendation
data/
  raw/                  # the selected weekend's OpenF1 downloads (RAW_DIR), one folder per source (LAPS_DIR, STINTS_DIR, DRIVERS_DIR, WEATHER_DIR, CARDATA_DIR)
    laps/                 # laps_p1-3.csv
    stints/               # stints_p1-3.csv
    drivers/              # drivers_p1-3.csv (entry lists; tyre_sets.py derives stand-ins from them)
    weather/              # weather.csv
    car_data/             # car telemetry, car_data_p<session>_<driver>.csv (every driver; 60 files for Bahrain 2025)
  cache/                # every weekend fetched so far, <year>_<name>/ laid out like raw/ (CACHE_DIR; gitignored)
  processed/            # pipeline hand-off CSVs (PROCESSED_DIR)
.env                # pipeline settings (paths, thresholds, hyperparameters, race constants); committed, no secrets
requirements.txt    # pinned deps (UTF-8), the .venv versions — see below
README.md           # higher-level project overview
readme.txt          # detailed methodology write-up (source of truth for "why")
diagnostic.md       # known bugs, with location and fix direction
```

## Running the pipeline

The virtualenv is `.venv/` (Python 3.10, gitignored). Scripts use **relative paths**
(`../data`, `./results/...`), so they must be launched with the working directory set to
`scripts/`.

```bash
# from repo root
cd scripts
python master.py            # run the full pipeline end-to-end
python process_laps.py      # or run any single stage on its own
```

`master.py` runs, in order: `race_weekend` → `process_laps` → `process_stints` → `tyre_sets` → `process_weather` →
`process_cardata`, then `feature_engineer` → `eda` → `train_model` → `strategy_simulator` →
`validate_strategy`. It aborts the whole run if any stage exits non-zero.

The model is chosen separately, in `explore/`:

```bash
cd explore
python explore_master.py                   # data stages (race_weekend .. eda), then the experiment
python explore_master.py --skip-pipeline   # reuse data/processed/
python explore_master.py --skip-pipeline --quick   # smoke run
```

`explore_master.py` runs `master.py`'s stages up to `eda.py` from `scripts/`, then compares the candidates in its
experiment block (models and search grids: plain code at the top of the file, by design, not `.env`) over the feature
sets in `feature_config.FEATURE_SETS` (shared with train_model.py) on the
fall-off target, with nested CV grouped by stint. Stage 1 (stint levels) is fitted per outer fold, and the score is the
shape of the curve within each held-out stint (`stint_shape_errors()`), against the flat (no fall-off) and linear
(stage-1 slope per compound) baselines. Each fold's curve at a few tyre ages is saved (its spread is the curve's
uncertainty), and each candidate's best configuration is drawn with `fall_off_curve()`, next to the one model-free
compound gap (`fit_compound_gap(stint_new_set_levels(all laps))`), the same shared helpers the pipeline uses, so explore
can't drift from the simulator. A candidate
is recommended only if it beats both baselines and its curve has steps ≤ `PLAUSIBLE_MAX_STEP_S`, rises on every
compound and doesn't get faster with age over the race length (`MIN_MONOTONE_SHARE`; an RBF SVR reverts past the
oldest tyre in training). The feature
sets must never touch `LAP_TIMING_COLUMNS` (asserted). It writes only to `explore/results/explore_master/`: no hand-off
file, and the bundle is untouched. The winner is copied into `train_model.py`'s MODEL block and `.env`
(`FALL_OFF_FEATURE_SET` and its hyperparameters) by hand.
Its `--quick` run writes into the same tracked results folder; to keep the full run's results, point `RESULTS_ROOT`
elsewhere for it (an environment variable wins over `.env`).

```bash
cd explore
python tune_iqr.py           # every weekend in its TUNE_WEEKENDS (downloads what data/cache/ lacks)
python tune_iqr.py --quick   # smoke run: two weekends, small grid
```

`tune_iqr.py` picks `IQR_LOWER_MULTIPLIER` / `IQR_UPPER_MULTIPLIER` (#32) with two methods: a label sweep over
`process_laps.py`'s `process_laps_iqr_residuals_*.csv` (how many push laps, `best_ratio <= PUSH_LAP_MAX_BEST_RATIO`, each fence
removes at each k), and a CV sweep that reruns `process_laps` → `process_cardata` → `feature_engineer` for each pair of multipliers
and scores the model-free stage-1 linear fall-off (fitted per fold, #27) by within-stint shape MAE on the rows common to
every pair, with the same stint folds. Its winner is the label floor with the current upper fence unless another pair beats
it on both paired tests. It runs the stages from `scripts/` with the
settings as environment variables, so while it runs it switches `data/raw/` between weekends and rewrites `data/processed/` and
those stages' `scripts/results/`. A full run then **writes the winner into `.env` itself** (`env_config.write_env()`: the
`KEY=value` lines in place, with a dated `# tune_iqr.py …` comment line above them; `--quick` never writes), switches back
to the `.env` weekend and reruns the stages at the new values. Its outputs go to `explore/results/tune_iqr/`. The
hand-written reasoning comment above the keys in `.env` is not rewritten automatically; update it after a run.

```bash
cd explore
python tune_push.py           # same weekends as tune_iqr.py
python tune_push.py --quick   # smoke run: two weekends, small grid
```

`tune_push.py` picks `PUSH_LAP_MAX_BEST_RATIO` / `PUSH_LAP_MAX_STINT_RATIO` (#30) the same way, and is a copy of `tune_iqr.py`'s
harness (keep the two in step). Its label doesn't use the ratios it tunes. `process_laps` runs with the push filter off and
`process_cardata` with its z-score filter off, and each lap's `full_throttle_pct`, relative to the driver's 90th percentile in the
session, labels it a push lap (≥ 0.8) or a cool-down lap (≤ 0.6). The label pick is the pair with the highest Youden J over
`process_laps_push_ratios_*.csv`. The CV sweep, the switching of weekends, the `.env` write and the final restore are as in
`tune_iqr.py`, except that a full run writes the **label pick**, not the CV winner: the CV sweep favours strict filters (it is
scored on the strictest setting's rows, #30), so it is reported for information only. Outputs go to `explore/results/tune_push/`.
Run `tune_iqr.py` before `tune_push.py`, never both at once (they share `data/raw/` and `data/processed/`).

**Known gaps (don't "fix" silently — confirm intent first):**
- Open issues (the unconfirmed model choice, no statistical inference, etc.) are listed in [diagnostic.md](diagnostic.md). Fixing them
  changes results, so confirm before fixing, and update diagnostic.md when one is resolved.

## Current work: #27 (branch `model-choice-27`)

The full step list, with files, sub-tasks and checks, is in [diagnostic.md](diagnostic.md) under #27. The rules
agreed for it:

- **Steps run in order, one at a time:**
  - Step 0: check the workflow.
  - Step 1: make the pipeline independent of the model.
  - Step 2: a better compound-gap estimator, with no model involved.
  - Step 3: dynamic programming in the simulator. This needs a new issue, opened only with the user's OK.
  - Step 4: plug in the fall-off model.
  - Step 5: docs, and close #27.

  Start a step only when the user asks. Each step is one commit with its before → after numbers. After each one,
  `master.py` and `explore_master.py --skip-pipeline` must run end to end.
- **Swap test** (the acceptance check from Step 1 on): run train_model.py with two different models.
  - These must come out identical: the cleaned data, the stage-1 levels and slopes, `compound_gap` and the
    driver-session `levels`.
  - Only these may differ: the fall-off curves, the stage-2 metrics and the strategy.
- **The model choice comes last.** Don't pick or tune the fall-off model before Steps 1–3 are done. explore/ compares
  candidates, but the choice (Ridge `tyre_only`, the MLP or the per-compound line) is the user's.
- **The compound gap must not use the model.** It is fitted from the stage-1 new-set level of each stint, not from
  `fcp − fall_off(model)`. Improving it means better estimation on the data there is, not more data:
  - partial pooling of the driver and session effects
  - run-type controls or matching
  - an S ≤ M ≤ H order constraint
  - bootstrap CIs over whole stints

  Use numpy, scipy and scikit-learn only. No new dependencies without asking.
- **Data scope:** the user ruled "the project is FP data only". The gap and the model use the selected weekend's
  FP1–FP3 only.
  - Don't pool other weekends, even though `data/cache/` holds 12 of them.
  - tune_iqr.py and tune_push.py use other weekends only to set cleaning thresholds.
  - No qualifying or race data.
- **Tuned factors must not depend on the model.** The label sweeps in tune_iqr.py and tune_push.py use no model, and
  since Step 1 their CV sweeps score with the model-free stage-1 linear fall-off instead of XGBoost.
- **Evidence from eda.py #26** to build on:
  - The per-compound slopes are all significant (SOFT +0.365, MEDIUM +0.227, HARD +0.306 s/lap), but the
    differences between them aren't (Holm p ≥ 0.08).
  - Paired within driver-session, a new set is MEDIUM +1.47 s and HARD +1.72 s slower than SOFT.
  - FP2 is 2.36 s faster than FP1, and FP3 is about the same as FP2.

## Data flow / hand-offs

Scripts communicate through CSVs in `data/processed/`, not by importing each other (the exception is
`selected_weekend()` from `race_weekend.py`, which has no side effects on import). Raw downloads live in
`data/raw/<source>/`. Each stage writes its hand-off file to `data/processed/` in addition to its own `results/<name>/` copy:

- `race_weekend.py`   → `data/raw/` holding the weekend in `RACE_YEAR` / `RACE_WEEKEND`. If raw already
  holds it in full (found from `meeting_key` in `laps_p1.csv`), nothing happens. Otherwise raw is moved into
  `data/cache/<its weekend>/` (an unknown meeting goes to `meeting_<key>/`; a file the cache already has wins),
  whatever `data/cache/<selected>/` is missing is downloaded from OpenF1, and that folder is copied into raw. A weekend
  fetched before needs no requests. Downloads go to `<file>.part` and are renamed when complete, so an interrupted run
  resumes. Car data is requested for the drivers in each session's downloaded laps file, so it follows each
  weekend's field. A car-data query with no results (HTTP 404) leaves a `.nodata` marker so it isn't re-requested;
  laps, stints, drivers or weather with no results raise. To force a re-download, delete the file from
  `data/cache/<weekend>/` and, for the active weekend, from `data/raw/` too. The `WEEKENDS` table is hardcoded data
  (meeting key, FP1-3 session keys, scheduled race laps); sprint weekends are rejected. Each run's source and file
  counts are in `results/race_weekend/race_weekend_summary.csv`.
- `process_laps.py`   → `data/processed/laps_combined_clean.csv`. A lap missing exactly one sector, with `lap_duration` and the
  other two recorded, first gets that sector as `lap_duration` minus the other two (kept only if > 0; saved to
  `results/process_laps/process_laps_rebuilt_sectors_*.csv`); laps still missing a sector are dropped. Its residual filter has a separate multiplier for each side
  (`IQR_LOWER_MULTIPLIER` for fast laps, `IQR_UPPER_MULTIPLIER` for slow ones, tuned by `explore/tune_iqr.py`), and it saves
  every lap reaching that filter with its residual and `best_ratio` to `results/process_laps/process_laps_iqr_residuals_*.csv`
  (diagnostic only; `tune_iqr.py` reads it). Its push-lap filter drops a lap slower than `PUSH_LAP_MAX_BEST_RATIO` × the
  driver's session best or `PUSH_LAP_MAX_STINT_RATIO` × the driver's best in that stint. Stints come from the raw
  `data/raw/stints/`, matched by `lap_start` / `lap_end` as in `feature_engineer.py`. It saves every lap reaching that filter
  with both ratios to `results/process_laps/process_laps_push_ratios_*.csv` (diagnostic only; `tune_push.py` reads it)
- `process_stints.py` → `data/processed/stints_combined_clean.csv`
- `tyre_sets.py`      → `data/processed/tyre_sets_race_inventory.csv` (one row per set every car in practice has
  left for the race: compound and age in laps). OpenF1 has no set ID, so sets are chained by compound and
  `tyre_age_at_start`; practice stand-ins count toward their race car, derived from `data/raw/drivers/` (the FP3
  lineup is the race cars; a non-FP3 driver is paired with the same team's driver who sat that session out), and
  allocation, returns and race-reserved sets are `.env` settings. Practice only: qualifying usage isn't tracked.
- `process_weather.py`→ `data/processed/cleaned_weather.csv`
- `process_cardata.py`→ `data/processed/cardata_per_lap_clean.csv` (reads `data/raw/car_data/` (`CARDATA_DIR`); keyed by
  `session_key`, `driver_number`, `lap_number`; lap windows come from the raw `data/raw/laps/laps_p*.csv`, and only laps in
  `data/processed/laps_combined_clean.csv` are kept, so it must run after `process_laps.py`; its per-session z-score test
  (`CARDATA_ZSCORE_THRESHOLD`) masks only the flagged aggregate to NaN and keeps the lap (`process_cardata_zscore_masked_cells.csv`);
  the aggregates are left unscaled, in physical units)
- `feature_engineer.py` reads the laps, stints, weather and car data above → `data/processed/final_engineered_dataset.csv`
  (and a copy under `results/feature_engineer/`). It keeps `session_key` and `driver_number` as
  identifiers only; they are not in `MODEL_FEATURE_COLUMNS`. It builds the `session_FP1/FP2/FP3` one-hot from
  `process_laps.py`'s `session_name` and `run_laps` (`lap_end − lap_start + 1`, the run's length, a control in
  train_model.py's compound-gap fit). The car-data aggregates (`CARDATA_FEATURE_COLUMNS`) are left-merged on the lap's
  keys. Missing values in `STINT_FILL_COLUMNS` (speed traps, segment counts, car data) are filled with the median of the
  same stint's *earlier* laps, and `stint_avg_pace` on a stint's first lap is its own lap time. Since #31 none of
  these is a model input (`LAP_TIMING_COLUMNS`); they stay for eda.py and tune_push.py.
- `eda.py` reads the engineered dataset and writes only to `results/eda/` (no hand-off file, nothing in the pipeline
  reads it): summary statistics and missing values, pace distributions by session and compound, pace against
  tyre age (raw, and the within-stint slope with each stint's mean pace and age removed, = train_model.py's stage 1),
  correlations of the candidate inputs with within-stint pace, with collinear groups (`EDA_COLLINEAR_THRESHOLD`), and
  row coverage of all laps and of the multi-lap stints (the fall-off rows). `lap_duration` is
  rebuilt from `fuel_corrected_pace` with `FUEL_EFFECT_K`. Section 6 (#26) tests the data on stints or driver-session
  means, never single laps: a stint cluster bootstrap of stage 1's slopes (`EDA_BOOTSTRAP_RESAMPLES`), and compound and
  session pace on a new set, paired within driver-session / driver-compound (t, Wilcoxon, Holm; `SIGNIFICANCE_LEVEL`).
- `train_model.py` reads the engineered dataset, splits it 80/20 by stint, and on the training side fits stage 1
  (`fit_stint_levels()`: stint levels + a linear slope per compound, on the multi-lap stints), the one model in its
  MODEL block on `fcp − level` (no comparison: that is `explore/`'s job), and stage 3 (`fit_compound_gap()` on
  `stint_new_set_levels()`: compound gap on a new set + driver-session levels, from each training stint's stage-1
  new-set level weighted by its laps; model-free, #27). The MODEL block is the only place the model is defined (an
  sklearn `Pipeline` and `MODEL_NAME`); its inputs are `FEATURE_SETS[FALL_OFF_FEATURE_SET]`. It scores the held-out
  stints on curve shape against the flat and linear baselines → `MODEL_BUNDLE_PATH`,
  `results/train_model/train_model_fall_off_model.pkl` (a joblib bundle: `{"model", "model_name", "feature_set",
  "features", "medians", "anchor_age", "compound_gap", "levels", "linear_slopes"}`; `medians` are the training rows'
  medians, used for `fillna` and as the simulator's conditions). `validate_strategy.py` fits a `clone()` of the model,
  its own stage 1, fill and model-free gap per CV fold from that fold's training rows, not the bundle's
- `strategy_simulator.py` and `validate_strategy.py` load that `.pkl` bundle. The simulator prices each (compound, set
  age at start) the car has with one curve over stint laps 1..race laps: fcp = race pace level (mean of `SIM_DRIVER`'s
  driver-session levels) + compound gap + `fall_off_curve()` at tyre age = set age + stint lap, with the conditions at
  `medians`. Nothing is fed back between laps; a stint of L laps costs the cumulative sum of its first L laps.
  Strategies are optimised for the one car in `SIM_DRIVER` from its tyre-set inventory (an unknown car number
  raises): each set is fitted at most once, and the fastest choice of sets is kept for each compound sequence
  and pit-lap plan.

The **model target** is the fall-off within a stint: `fuel_corrected_pace − level[stint]`, with the levels from stage 1
fitted on the training rows only (`fit_stint_levels()` / `fall_off_target()` in `feature_config.py`, shared by
`train_model.py`, `validate_strategy.py`, `eda.py` and `explore/`, so the target can't diverge between them). Only stints
with two or more clean laps carry fall-off (`multi_lap_rows()`). A held-out stint's level is unknown by construction, so
validation scores the shape of the curve (`stint_shape_errors()`: residual minus the stint's mean residual). The target
is the lap's own pace, so nothing measured on the lap may be an input: `lap_duration`, sector times, speed traps,
segment counts, car data and `stint_avg_pace` (`LAP_TIMING_COLUMNS`) are excluded, and `feature_config.py` asserts it.
The target, every prediction, the simulator's totals and the error metrics are in **seconds**: lap times
are not scaled. Don't reintroduce a per-session scaler on lap or car-data columns.

## Conventions to follow

Every pipeline script follows the same house style — match it when adding or editing scripts:

- **Per-script results dir**: `SCRIPT_NAME = "..."`, `RESULTS_DIR = f"{env_str('RESULTS_ROOT')}/{SCRIPT_NAME}"`
  (`RESULTS_ROOT=./results` in `.env`), created with `os.makedirs(..., exist_ok=True)`.
- **Logging**: a `log(msg)` helper that both `print`s and appends to
  `results/<name>/<name>_log.txt`, opened with `encoding="utf-8", errors="replace"`. The printed
  copy is encoded with `sys.stdout.encoding` and `errors="replace"`, because redirected output on Windows
  is cp1252 and can't hold `→` (#28); the log file gets the original text. Keep this encode/replace dance.
- **Output naming**: intermediate CSVs, plots, and a `<name>_manifest.txt` are prefixed with the
  script name and written into that script's results dir. Track generated paths in the
  `generated_files` list.
- **Shared feature order**: never hard-code the model's feature list. The feature sets live in
  [feature_config.py](scripts/feature_config.py) (`FEATURE_SETS`, each checked against `LAP_TIMING_COLUMNS`;
  `MODEL_FEATURE_COLUMNS` is `FEATURE_SETS["core"]`), and train_model.py picks one with `FALL_OFF_FEATURE_SET`. The
  simulator and validation read `bundle["features"]`. Add or change a set there, so training, simulation, validation
  and explore/ stay consistent.
- **Settings live in `.env`**: never hard-code a tunable value or a path/file name in a script. Add a
  commented key to the repo-root `.env` and read it once at module level with the getters from
  [env_config.py](scripts/env_config.py). A missing key raises (there are no silent defaults), and an
  environment variable of the same name overrides `.env`. Schema definitions (column lists, `COMPOUNDS`)
  stay in code.
- **Compound one-hot columns** are `compound_HARD`, `compound_MEDIUM`, `compound_SOFT`.
- **Session one-hot columns** are `session_FP1`, `session_FP2`, `session_FP3` (`SESSION_COLUMNS` in
  feature_config.py), named by practice order so they stay the same on any weekend.

## Environment notes

- **Platform is Windows** (PowerShell primary). Prefer forward-slash relative paths in Python;
  they work fine via `pathlib`/`os.path`.
- **`requirements.txt`** (UTF-8) pins the `.venv` versions that produced the committed results, for
  Python 3.10: `pandas==2.3.3`, `numpy==2.2.6`, `scikit-learn==1.7.2`, `xgboost==3.2.0`,
  `matplotlib==3.10.9`, `seaborn==0.13.2`, `scipy==1.15.3`, `joblib==1.5.3`, plus their dependencies.
  `shap` is installed in `.venv` but unused, so it isn't pinned. OpenF1 downloads use `urllib` from the
  standard library, not `requests`.
- `.gitignore` covers `.venv/`, `__pycache__/` and `data/cache/`; the weekend cache is the only
  ignored data. `data/raw/`, `data/processed/` and the generated `results/` CSVs and logs **are**
  tracked (they show up in `git status` after every run). Don't assume they're throwaway — the repo
  commits them.

## Working guidelines for agents

- The whole pipeline runs on **one weekend at a time** (`RACE_YEAR` / `RACE_WEEKEND`); the scope is intentionally limited to
  the three practice sessions. Don't broaden scope (extra sessions, race data, other weekends' practice data in the model
  or the compound gap) unless asked.
- Keep everything before the fall-off model independent of the model (see "Current work"). A change that makes the
  cleaning, stage 1, the compound gap or the race pace level depend on the model is a regression.
- Running another weekend rewrites the tracked `data/raw/`, `data/processed/` and `scripts/results/`. Switch back to
  2025 `bahrain` (loaded from `data/cache/` with no requests) before committing, unless the change is intended.
- Guard against **target leakage** above all else — it's the central design constraint. The target is the lap's
  own pace, so no model input may come from the lap's own timing or telemetry (`LAP_TIMING_COLUMNS`); tyre and
  condition columns only.
- When adding a new pipeline stage, wire it into `master.py`'s sequence and give it its own
  `results/<name>/` directory and log, following the pattern above.
- Prefer editing the existing scripts over introducing new frameworks; this is a plain
  pandas/scikit-learn/xgboost project with no test suite or CI.
