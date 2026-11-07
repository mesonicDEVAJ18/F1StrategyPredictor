# AGENTS.md

Guidance for AI coding agents working in the **F1 Race Strategy Optimiser** repository.

## What this project is

An end-to-end data-science pipeline that predicts the **optimal tyre strategy** for a non-sprint Formula 1 race from
the three practice sessions (FP1/FP2/FP3) only. The reference weekend is **Bahrain 2025**; any non-sprint 2024 or 2025
weekend can be chosen with `RACE_YEAR` / `RACE_WEEKEND` in `.env` (names in `scripts/race_weekend.py`). The model is
trained on the whole field; the race is simulated for one car, `SIM_DRIVER` (default 16, Ferrari), using that car's
tyre sets (their ages and how many are left). Left blank (`SIM_DRIVER=`), the race runs on fresh tyres: unlimited new
sets of every compound and the field's pace level (#35). All raw data comes from the free
[OpenF1 API](https://openf1.org/).

The model learns **tyre fall-off with age** (#31). Each lap's fuel-corrected pace is a level for its stint (driver,
session, fuel plan, run type: everything constant over a run) plus a fall-off that depends on the compound, tyre age
and conditions:
- **Stage 1** (linear) fits the stint levels and a slope per compound.
- **Stage 2** (the model) learns the fall-off from what is left.
- **Stage 3** (linear, model-free) gives the gap between compounds on a new set and the driver-session levels.
- The **simulator** enumerates legal pit/compound strategies, prices each stint from the fall-off curve of its set, and
  ranks them by total race time (brute force now, dynamic programming planned in #27 Step 3).

**Target design (#27, in progress):** everything before the fall-off model (lap cleaning and its tuned thresholds,
stage 1, the compound gap, the race pace level) must not depend on which model is chosen. The model is plugged in
last; its only job is the fall-off curve. XGBoost is a placeholder until #27 Step 4, and since Step 1 nothing upstream
uses it.

[readme.txt](readme.txt) is the full methodology (cleaning rules, feature rationale, modelling decisions, current
numbers). When code and docs disagree, trust the code; when the *reasoning* is unclear, trust readme.txt.

## Repository layout

```
scripts/            # ALL pipeline code; run scripts FROM this directory
  master.py             # orchestrator: runs the whole pipeline in order
  race_weekend.py       # weekend table (session keys, race laps); puts the selected weekend in data/raw/ (cache or OpenF1)
  env_config.py         # loads ../.env into os.environ; env_str/env_int/env_float/env_list getters, write_env()
  feature_config.py     # shared column lists (FEATURE_SETS, LAP_TIMING_COLUMNS, ...) and the fall-off / compound-gap helpers
  process_laps.py       # clean lap data (seconds / km/h, not scaled), tag session_name FP1-3
  process_stints.py     # clean stints (dry compounds only), one-hot encode compound
  tyre_sets.py          # infer every car's tyre sets from practice stints -> race inventory
  process_weather.py    # clean + scale weather
  process_cardata.py    # clean car telemetry, aggregate per lap, unscaled (EDA and tune_push label; not model inputs)
  feature_engineer.py   # merge laps+stints+weather+car data, build engineered features
  eda.py                # exploratory analysis + statistical inference (read-only, no hand-off)
  train_model.py        # train the one fall-off model (MODEL block), persist the bundle
  strategy_simulator.py # enumerate + simulate strategies, pick best
  validate_strategy.py  # grouped CV of the fall-off curve, residual plot, tests vs flat/linear baselines
  results/<script>/     # per-script outputs (logs, intermediate CSVs, plots, model)
explore/            # model selection and threshold tuning, kept out of the pipeline; run FROM this directory
  explore_master.py     # data stages + the experiment block (candidates, grids, feature sets)
  explore_lib.py        # custom estimators, nested grouped CV, fall-off target and shape scoring
  tune_iqr.py           # tunes process_laps.py's two IQR multipliers over many cached weekends (#32)
  tune_push.py          # tunes process_laps.py's two push-lap ratios the same way (#30)
  results/<script>/     # leaderboards, CV sweeps, curves, recommendations
data/
  raw/                  # the selected weekend (RAW_DIR): laps/, stints/, drivers/, weather/, car_data/
  cache/                # every weekend fetched so far, <year>_<name>/ laid out like raw/ (CACHE_DIR; gitignored)
  processed/            # pipeline hand-off CSVs (PROCESSED_DIR)
.env                # pipeline settings (paths, thresholds, hyperparameters, race constants); committed, no secrets
requirements.txt    # pinned deps (UTF-8), the .venv versions
README.md           # project overview
readme.txt          # detailed methodology (source of truth for "why")
diagnostic.md       # open work and fixed issues, with location and fix direction
```

## Running

The virtualenv is `.venv/` (Python 3.10, gitignored). Scripts use relative paths (`../data`, `./results/...`), so
launch them with the working directory set to `scripts/` (or `explore/` for the explore scripts).

```bash
cd scripts
python master.py            # full pipeline
python process_laps.py      # or any single stage
```

`master.py` runs `race_weekend` → `process_laps` → `process_stints` → `tyre_sets` → `process_weather` →
`process_cardata` → `feature_engineer` → `eda` → `train_model` → `strategy_simulator` → `validate_strategy`, and aborts
if any stage exits non-zero.

```bash
cd explore
python explore_master.py                          # data stages (race_weekend .. eda), then the experiment
python explore_master.py --skip-pipeline          # reuse data/processed/
python explore_master.py --skip-pipeline --quick  # smoke run
```

**explore_master.py** compares the candidates in its experiment block (plain code at the top of the file, by design,
not `.env`) over `feature_config.FEATURE_SETS`, with nested CV grouped by stint; stage 1 is fitted per outer fold, and
the score is the within-stint curve shape (`stint_shape_errors()`) against the flat and linear baselines. Curves are
drawn with the pipeline's own `fall_off_curve()` and the one model-free gap
(`fit_compound_gap(stint_new_set_levels(all laps))`), so explore can't drift from the simulator. A candidate is
recommended only if it beats both baselines, its steps are ≤ `PLAUSIBLE_MAX_STEP_S`, it rises on every compound and it
doesn't get faster with age (`MIN_MONOTONE_SHARE`). It writes only to `explore/results/explore_master/`, never touches
the bundle; the winner is copied into `train_model.py`'s MODEL block and `.env` by hand. `--quick` writes into the same
tracked results folder, so point `RESULTS_ROOT` elsewhere for it (an environment variable wins over `.env`).

```bash
cd explore
python tune_iqr.py           # every weekend in its TUNE_WEEKENDS (downloads what data/cache/ lacks)
python tune_push.py          # same weekends; run after tune_iqr.py, never both at once
python tune_iqr.py --quick   # smoke run: two weekends, small grid (never writes .env)
```

**tune_iqr.py** picks `IQR_LOWER_MULTIPLIER` / `IQR_UPPER_MULTIPLIER` (#32); **tune_push.py** picks
`PUSH_LAP_MAX_BEST_RATIO` / `PUSH_LAP_MAX_STINT_RATIO` (#30) and is a copy of tune_iqr.py's harness (keep the two in
step). Each runs a model-free label sweep and a CV sweep scored with the stage-1 linear fall-off; the method is in
readme.txt (Data Preparation, the two "Tuning" sections). What an agent must know:
- They run the pipeline stages from `scripts/` with settings as environment variables, so while running they switch
  `data/raw/` between weekends and rewrite `data/processed/` and those stages' `scripts/results/`. They share these, so
  never run both at once.
- A full run **writes its pick into `.env` itself** (`env_config.write_env()`: the `KEY=value` lines in place, with a
  dated comment line), switches back to the `.env` weekend and reruns the stages. The hand-written reasoning comment
  above the keys is not rewritten; update it after a run.
- tune_iqr.py writes its winner (the label floor unless another pair beats it on both paired CV tests). tune_push.py
  writes the **label pick**, not the CV winner: its CV favours strict filters (a bias, #30), so it is information only.
- Outputs go to `explore/results/tune_iqr/` and `explore/results/tune_push/`.

## Current work: #27 (branch `model-choice-27`)

The full step list, with files, sub-tasks and checks, is in [diagnostic.md](diagnostic.md) under #27. Rules agreed for
it:

- **Steps run in order, one at a time:** 0 check the workflow; 1 make the pipeline independent of the model; 2 a better
  compound-gap estimator, no model involved; 3 dynamic programming in the simulator (needs a new issue, opened only
  with the user's OK); 4 plug in the fall-off model; 5 docs, and close #27. Start a step only when the user asks. Each
  step is one commit with its before → after numbers. After each one, `master.py` and
  `explore_master.py --skip-pipeline` must run end to end.
- **Status:** Steps 0 and 1 are done (Step 1 is commit a72dd8c; its full explore rerun finished 2026-11-05). Step 2 is
  next and starts only when the user asks.
- **Swap test** (the acceptance check from Step 1 on): run train_model.py with two different models. The cleaned data,
  the stage-1 levels and slopes, `compound_gap` and the driver-session `levels` must come out identical; only the
  fall-off curves, the stage-2 metrics and the strategy may differ.
- **The model choice comes last.** Don't pick or tune the fall-off model before Steps 1–3 are done. explore/ compares
  candidates, but the choice is the user's. The 2026-11-05 run recommends the MLP (`plus_age_at_start`), with Ridge
  (`tyre_only`) and the per-compound line within one fold std of it: that is the shortlist.
- **The compound gap must not use the model.** It is fitted from each stint's stage-1 new-set level
  (`stint_new_set_levels()`), not from `fcp − fall_off(model)`. Improving it means better estimation on the data there
  is, not more data: partial pooling of the driver and session effects, run-type controls or matching, an S ≤ M ≤ H
  order constraint, bootstrap CIs over whole stints. numpy, scipy and scikit-learn only; no new dependencies without
  asking.
- **Data scope:** the user ruled "the project is FP data only". The gap and the model use the selected weekend's
  FP1–FP3 only. Don't pool other weekends, even though `data/cache/` holds 12 of them (tune_iqr.py and tune_push.py use
  them only to set cleaning thresholds). No qualifying or race data.
- **Tuned factors must not depend on the model.** The tune scripts' label sweeps use no model, and their CV sweeps
  score with the model-free stage-1 linear fall-off.
- **Evidence from eda.py #26** to build on (Bahrain 2025, 566 laps):
  - Per-compound slopes SOFT +0.341, MEDIUM +0.227, HARD +0.211 s/lap. SOFT and MEDIUM are significant, HARD is not
    (95% CI −0.05 to +0.46, p 0.12). The differences between them aren't significant (Holm p ≥ 0.25).
  - Paired within driver-session, a new set is MEDIUM +1.61 s and HARD +2.23 s slower than SOFT. Only one
    driver-session ran both MEDIUM and HARD.
  - FP2 is 2.77 s faster than FP1, and FP3 is about the same as FP2.

## Data flow / hand-offs

Scripts communicate through CSVs in `data/processed/`, not by importing each other (the exception is
`selected_weekend()` from `race_weekend.py`, which has no side effects on import). Each stage writes its hand-off file to
`data/processed/` and a copy under its own `results/<name>/`. readme.txt has the details; these are the facts that
code changes can break:

- `race_weekend.py` → `data/raw/` holding the weekend in `.env`. A complete raw is left alone; otherwise raw is moved
  into `data/cache/<its weekend>/`, the selected weekend's cache folder is completed from OpenF1 and copied in.
  Downloads go to `<file>.part`; a car-data 404 leaves a `.nodata` marker; laps, stints, drivers or weather with no
  results raise. The `WEEKENDS` table is hardcoded (meeting key, FP1–3 session keys, race laps); sprint weekends are
  rejected. To force a re-download, delete the file from `data/cache/<weekend>/` and, for the active weekend, from
  `data/raw/` too.
- `process_laps.py` → `laps_combined_clean.csv`. Rebuilds a single missing sector, then domain filters, the two-sided
  IQR residual filter and the two-test push-lap filter (stints matched from the raw `data/raw/stints/`). Saves
  `process_laps_iqr_residuals_*.csv` and `process_laps_push_ratios_*.csv`, which the tune scripts read.
- `process_stints.py` → `stints_combined_clean.csv` (dry compounds only; a compound column missing in a session is
  filled with 0 after the concat).
- `tyre_sets.py` → `tyre_sets_race_inventory.csv` (one row per set each car has left: compound and age). OpenF1 has no
  set ID, so sets are chained by compound and `tyre_age_at_start`; stand-ins are derived from `data/raw/drivers/`;
  allocation, returns and reserved sets are `.env` settings. Practice only.
- `process_weather.py` → `cleaned_weather.csv` (practice sessions only, scaled).
- `process_cardata.py` → `cardata_per_lap_clean.csv`, keyed by `session_key`, `driver_number`, `lap_number`. Lap windows
  come from the raw laps files, and only laps in `laps_combined_clean.csv` are kept, so it **must run after
  process_laps.py**. Its z-score test masks only the flagged aggregate to NaN and keeps the lap. Unscaled.
- `feature_engineer.py` → `final_engineered_dataset.csv`. Keeps `session_key` and `driver_number` as identifiers only.
  Drops laps with no stint or no `date_start` (counted). Builds `session_FP1/2/3`, `tyre_age`, `run_laps` (a control in
  the compound-gap fit) and `fuel_corrected_pace`. `STINT_FILL_COLUMNS` are filled from the same stint's **earlier**
  laps only; `stint_avg_pace` on a stint's first lap is its own lap time. None of these fills is a model input
  (`LAP_TIMING_COLUMNS`); they serve eda.py and tune_push.py.
- `eda.py` writes only to `results/eda/`. Its slopes are stage 1's; section 6 (#26) tests on stints or driver-session
  means, never single laps.
- `train_model.py` splits 80/20 by stint and, on the training side, fits stage 1 (`fit_stint_levels()`, multi-lap
  stints), the one model in its MODEL block on `fcp − level` (no comparison: that is explore/'s job), and stage 3
  (`fit_compound_gap()` on `stint_new_set_levels()`, model-free). The MODEL block (an sklearn `Pipeline` and
  `MODEL_NAME`) is the only place the model is defined; its inputs are `FEATURE_SETS[FALL_OFF_FEATURE_SET]`. Bundle
  `MODEL_BUNDLE_PATH` = `results/train_model/train_model_fall_off_model.pkl`, a joblib dict `{"model", "model_name",
  "feature_set", "features", "medians", "anchor_age", "compound_gap", "levels", "linear_slopes"}` (`medians` = training
  rows' medians, used for `fillna` and as the simulator's conditions).
- `strategy_simulator.py` loads the bundle. For each (compound, set age) the car has: fcp = race pace level (mean of
  `SIM_DRIVER`'s driver-session levels) + compound gap + `fall_off_curve()` at tyre age = set age + stint lap. A stint
  of L laps costs the cumulative sum of its first L laps; nothing is fed back between laps. Each set is fitted at most
  once; an unknown car number raises. A blank `SIM_DRIVER` skips the inventory: new sets only (age 0, enough for
  `max(STOP_COUNTS) + 1` stints of each compound), race pace level = mean of all driver-session levels, and
  `car_number` = `"field"` in the outputs.
- `validate_strategy.py` fits a `clone()` of the bundle's model, its own stage 1, fill and model-free gap per CV fold
  from that fold's training rows, not the bundle's.

## Model target and leakage

The **target** is the fall-off within a stint: `fuel_corrected_pace − level[stint]`, with stage 1 fitted on training
rows only (`fit_stint_levels()` / `fall_off_target()` in `feature_config.py`, shared by train_model.py,
validate_strategy.py, eda.py and explore/, so it can't diverge). Only stints with two or more clean laps carry fall-off
(`multi_lap_rows()`). A held-out stint's level is unknown by construction, so validation scores the curve's shape
(`stint_shape_errors()`: residual minus the stint's mean residual).

Guard against **target leakage** above all else. The target is the lap's own pace, so nothing measured on the lap may
be an input: `lap_duration`, sector times, speed traps, segment counts, car data and `stint_avg_pace`
(`LAP_TIMING_COLUMNS`) are excluded, and `feature_config.py` asserts it for every feature set. Tyre and condition
columns only.

Everything (target, predictions, simulator totals, error metrics) is in **seconds**: lap times are not scaled. Don't
reintroduce a per-session scaler on lap or car-data columns.

## Conventions

Every pipeline script follows the same house style; match it when adding or editing scripts:

- **Results dir**: `SCRIPT_NAME = "..."`, `RESULTS_DIR = f"{env_str('RESULTS_ROOT')}/{SCRIPT_NAME}"`
  (`RESULTS_ROOT=./results`), created with `os.makedirs(..., exist_ok=True)`.
- **Logging**: a `log(msg)` helper that both `print`s and appends to `results/<name>/<name>_log.txt`, opened with
  `encoding="utf-8", errors="replace"`. The printed copy is encoded with `sys.stdout.encoding` and `errors="replace"`,
  because redirected output on Windows is cp1252 and can't hold `→` (#28); the log file gets the original text. Keep
  this encode/replace dance.
- **Output naming**: intermediate CSVs, plots and a `<name>_manifest.txt` are prefixed with the script name and written
  to its results dir. Track generated paths in the `generated_files` list.
- **Shared feature order**: never hard-code the model's feature list. Feature sets live in
  [feature_config.py](scripts/feature_config.py) (`FEATURE_SETS`, each checked against `LAP_TIMING_COLUMNS`;
  `MODEL_FEATURE_COLUMNS` is `FEATURE_SETS["core"]`); train_model.py picks one with `FALL_OFF_FEATURE_SET`, and the
  simulator and validation read `bundle["features"]`.
- **Settings live in `.env`**: never hard-code a tunable value or a path/file name. Add a commented key to the repo-root
  `.env` and read it once at module level with the [env_config.py](scripts/env_config.py) getters. A missing key raises
  (no silent defaults); an environment variable of the same name overrides `.env`. Schema definitions (column lists,
  `COMPOUNDS`) stay in code.
- **One-hot columns**: `compound_HARD`, `compound_MEDIUM`, `compound_SOFT`; `session_FP1`, `session_FP2`, `session_FP3`
  (`SESSION_COLUMNS`, named by practice order so they mean the same on any weekend).
- **New pipeline stage**: wire it into `master.py`'s sequence, with its own `results/<name>/` directory and log.

## Environment notes

- **Platform is Windows** (PowerShell primary). Forward-slash relative paths in Python work fine.
- **`requirements.txt`** (UTF-8) pins the `.venv` versions for Python 3.10: `pandas==2.3.3`, `numpy==2.2.6`,
  `scikit-learn==1.7.2`, `xgboost==3.2.0`, `matplotlib==3.10.9`, `seaborn==0.13.2`, `scipy==1.15.3`, `joblib==1.5.3`,
  plus their dependencies. `shap` is installed but unused, so not pinned. OpenF1 downloads use `urllib`, not `requests`.
- **Git**: `.gitignore` covers `.venv/`, `__pycache__/` and `data/cache/`. `data/raw/`, `data/processed/` and the
  generated `results/` CSVs and logs **are** tracked (they show up in `git status` after every run). Don't treat them
  as throwaway.

## Working guidelines

- The pipeline runs on **one weekend at a time**, from the three practice sessions only. Don't broaden scope (extra
  sessions, race data, other weekends' data in the model or the compound gap) unless asked.
- Running another weekend rewrites the tracked `data/raw/`, `data/processed/` and `scripts/results/`. Switch back to
  2025 `bahrain` (loaded from `data/cache/` with no requests) before committing, unless the change is intended.
- A change that makes the cleaning, stage 1, the compound gap or the race pace level depend on the model is a
  regression (see "Current work").
- **Known gaps** are listed in [diagnostic.md](diagnostic.md). Fixing one changes results, so confirm intent before
  fixing, and update diagnostic.md (and its `[TODO #N]` tags in readme.txt) when one is resolved.
- Prefer editing the existing scripts over introducing new frameworks; this is a plain pandas / scikit-learn / xgboost
  project with no test suite or CI.
