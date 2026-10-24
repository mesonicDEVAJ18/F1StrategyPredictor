# AGENTS.md

Guidance for AI coding agents working in the **F1 Race Strategy Optimiser** repository.

## What this project is

An end-to-end data-science pipeline that predicts the **optimal tyre strategy** for a
non-sprint Formula 1 race using only data from the three practice sessions (FP1/FP2/FP3).
The reference weekend is **Bahrain 2025**; any non-sprint 2024 or 2025 weekend can be chosen with
`RACE_YEAR` / `RACE_WEEKEND` in `.env` (names in `scripts/race_weekend.py`). The
model is trained on the whole field; the race is simulated for one car, `SIM_DRIVER` in `.env`
(default 16, Ferrari), using that car's tyre sets. All raw data comes from the free [OpenF1 API](https://openf1.org/).

The core idea: rather than labelling a "degradation rate" directly (circular and leak-prone),
an XGBoost model is trained to predict the **next lap's fuel-corrected pace**. Degradation then
emerges as the falling trend when the model is rolled forward autoregressively. A simulator
enumerates legal pit/compound strategies, simulates each lap-by-lap, and ranks them by total
race time.

Read [readme.txt](readme.txt) for the full, authoritative narrative (problem statement, data
dictionary, cleaning rules, feature-engineering rationale, and modelling decisions). When the
code and this file disagree, trust the code; when the *reasoning* is unclear, trust readme.txt.

## Repository layout

```
scripts/            # ALL pipeline code lives here; run scripts FROM this directory
  master.py             # orchestrator — runs the whole pipeline in order
  race_weekend.py       # weekend table (session keys, race laps); puts the selected weekend in data/raw/ (cache or OpenF1)
  env_config.py         # loads ../.env into os.environ; env_str/env_int/env_float/env_list getters
  feature_config.py     # shared column lists (MODEL_FEATURE_COLUMNS, SIM_FEATURE_COLUMNS, SESSION_COLUMNS)
  process_laps.py       # clean lap data (kept in s / km/h, not scaled), tag session_name FP1-3
  process_stints.py     # clean stints, one-hot encode compound
  tyre_sets.py          # infer every car's tyre sets from practice stints -> race inventory
  process_weather.py    # clean + scale weather
  process_cardata.py    # clean car telemetry (whole field), aggregate per lap, unscaled; model inputs
  feature_engineer.py   # merge laps+stints+weather+car data, build engineered features
  train_model.py        # train/compare models, persist XGBoost bundle
  strategy_simulator.py # enumerate + simulate strategies, pick best
  validate_strategy.py  # cross-validation, residual plot, placeholder t-test (strategy robustness is planned, not built)
  results/<script>/     # per-script outputs (logs, intermediate CSVs, plots, model)
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
`process_cardata`, then `feature_engineer` → `train_model` → `strategy_simulator` →
`validate_strategy`. It aborts the whole run if any stage exits non-zero.

**Known gaps (don't "fix" silently — confirm intent first):**
- Open issues (train/test splits that mix laps of one stint, residual-plot leakage, the placeholder
  t-test, the unconfirmed model choice, etc.) are listed in [diagnostic.md](diagnostic.md). Fixing them
  changes results, so confirm before fixing, and update diagnostic.md when one is resolved.

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
- `process_laps.py`   → `data/processed/laps_combined_clean.csv`
- `process_stints.py` → `data/processed/stints_combined_clean.csv`
- `tyre_sets.py`      → `data/processed/tyre_sets_race_inventory.csv` (one row per set every car in practice has
  left for the race: compound and age in laps). OpenF1 has no set ID, so sets are chained by compound and
  `tyre_age_at_start`; practice stand-ins count toward their race car, derived from `data/raw/drivers/` (the FP3
  lineup is the race cars; a non-FP3 driver is paired with the same team's driver who sat that session out), and
  allocation, returns and race-reserved sets are `.env` settings. Practice only: qualifying usage isn't tracked.
- `process_weather.py`→ `data/processed/cleaned_weather.csv`
- `process_cardata.py`→ `data/processed/cardata_per_lap_clean.csv` (reads `data/raw/car_data/` (`CARDATA_DIR`); keyed by
  `session_key`, `driver_number`, `lap_number`; lap windows come from the raw `data/raw/laps/laps_p*.csv`, and only laps in
  `data/processed/laps_combined_clean.csv` are kept, so it must run after `process_laps.py`; the aggregates are left
  unscaled, in physical units)
- `feature_engineer.py` reads the laps, stints, weather and car data above → `data/processed/final_engineered_dataset.csv`
  (and a copy under `results/feature_engineer/`). It keeps `session_key` and `driver_number` as
  identifiers only; they are not in `MODEL_FEATURE_COLUMNS`. The session reaches the model only as the
  `session_FP1/FP2/FP3` one-hot, built from `process_laps.py`'s `session_name`. The car-data aggregates
  (`CARDATA_FEATURE_COLUMNS`) are left-merged on lap *t*'s keys. Laps without telemetry stay NaN and get the
  training medians, and the simulator holds them at those medians.
- `train_model.py` reads the engineered dataset → `results/train_model/train_model_xgb_model.pkl`
  (a joblib bundle: `{"model": ..., "features": [...], "medians": {...}, "session_weights": {...}}`;
  `medians` are the training medians used for `fillna`, and `session_weights` is each session's share of
  training rows)
- `strategy_simulator.py` and `validate_strategy.py` load that `.pkl` bundle. The simulator starts from
  `medians` (inputs are in physical units, so 0 is not "average") and averages each lap's prediction over
  the session one-hots with `session_weights`, because the race is not a practice session. Each stint is
  rolled out autoregressively from stint lap 1, once per (compound, set age at start, stint length), with
  `tyre_age` = set age + stint lap: every predicted fcp is turned back into a raw lap time with
  `FUEL_EFFECT_K` (feature_config.py) and fed into the next lap's sector times and running `stint_avg_pace`.
  Strategies are optimised for the one car in `SIM_DRIVER` from its tyre-set inventory (an unknown car number
  raises): each set is fitted at most once, and the fastest choice of sets is kept for each compound sequence
  and pit-lap plan.

The **model target** is next-lap `fuel_corrected_pace`: the same driver's lap `lap_number + 1` in the
same stint, built by `build_next_lap_target()` in `feature_config.py` (shared by `train_model.py` and
`validate_strategy.py`, so the target can't diverge between them). `lap_duration` and `laps_remaining` are
dependent variables and are deliberately excluded from model inputs to avoid leakage.
The target, every prediction, the simulator's totals and the error metrics are in **seconds**: lap times
are not scaled. Don't reintroduce a per-session scaler on lap or car-data columns.

## Conventions to follow

Every pipeline script follows the same house style — match it when adding or editing scripts:

- **Per-script results dir**: `SCRIPT_NAME = "..."`, `RESULTS_DIR = f"{env_str('RESULTS_ROOT')}/{SCRIPT_NAME}"`
  (`RESULTS_ROOT=./results` in `.env`), created with `os.makedirs(..., exist_ok=True)`.
- **Logging**: a `log(msg)` helper that both `print`s and appends to
  `results/<name>/<name>_log.txt`, opened with `encoding="utf-8", errors="replace"`. This
  encode/replace dance exists to survive the Windows console — keep it.
- **Output naming**: intermediate CSVs, plots, and a `<name>_manifest.txt` are prefixed with the
  script name and written into that script's results dir. Track generated paths in the
  `generated_files` list.
- **Shared feature order**: never hard-code the model's feature list. Import it from
  [feature_config.py](scripts/feature_config.py) (`MODEL_FEATURE_COLUMNS` /
  `SIM_FEATURE_COLUMNS`). If you change the feature set, change it there so training,
  simulation, and validation stay consistent.
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
  the three practice sessions. Don't broaden scope (extra sessions, race data) unless asked.
- Running another weekend rewrites the tracked `data/raw/`, `data/processed/` and `scripts/results/`. Switch back to
  2025 `bahrain` (loaded from `data/cache/` with no requests) before committing, unless the change is intended.
- Guard against **target leakage** above all else — it's the central design constraint. Any new
  feature must belong to lap *t* when predicting lap *t+1*.
- When adding a new pipeline stage, wire it into `master.py`'s sequence and give it its own
  `results/<name>/` directory and log, following the pattern above.
- Prefer editing the existing scripts over introducing new frameworks; this is a plain
  pandas/scikit-learn/xgboost project with no test suite or CI.
