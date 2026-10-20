# AGENTS.md

Guidance for AI coding agents working in the **F1 Race Strategy Optimiser** repository.

## What this project is

An end-to-end data-science pipeline that predicts the **optimal tyre strategy** for a
non-sprint Formula 1 race using only data from the three practice sessions (FP1/FP2/FP3).
The reference weekend is **Ferrari, Bahrain 2025**, but the same pipeline generalises to any
weekend. All raw data comes from the free [OpenF1 API](https://openf1.org/).

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
  env_config.py         # loads ../.env into os.environ; env_str/env_int/env_float/env_list getters
  feature_config.py     # shared column lists (MODEL_FEATURE_COLUMNS, SIM_FEATURE_COLUMNS, SESSION_COLUMNS)
  process_laps.py       # clean lap data (kept in s / km/h, not scaled), tag session_name FP1-3
  process_stints.py     # clean stints, one-hot encode compound
  tyre_sets.py          # infer each race car's tyre sets from practice stints -> race inventory
  process_weather.py    # clean + scale weather
  process_cardata.py    # clean car telemetry, aggregate per lap (not a model input yet)
  feature_engineer.py   # merge laps+stints+weather, build engineered features
  train_model.py        # train/compare models, persist XGBoost bundle
  strategy_simulator.py # enumerate + simulate strategies, pick best
  validate_strategy.py  # cross-validation, residuals, strategy robustness
  results/<script>/     # per-script outputs (logs, intermediate CSVs, plots, model)
data/               # raw + cleaned CSVs (OpenF1 downloads and pipeline hand-offs)
results/            # stray: empty validate_strategy log from running a script outside scripts/
.env                # pipeline settings (paths, thresholds, hyperparameters, race constants); committed, no secrets
requirements.txt    # pinned deps (NOTE: UTF-16 encoded — see below)
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

`master.py` runs, in order: `process_laps` → `process_stints` → `tyre_sets` → `process_weather` →
`process_cardata`, then `feature_engineer` → `train_model` → `strategy_simulator` →
`validate_strategy`. It aborts the whole run if any stage exits non-zero.

**Known gaps (don't "fix" silently — confirm intent first):**
- Car data is cleaned, encoded, aggregated per lap and scaled, but it is **not** merged into
  the engineered dataset or used as a model feature (diagnostic.md #16). It only covers the two
  Ferrari cars, while the model trains on the whole field.
- Known modelling bugs (cross-driver targets, mixed units in the simulator, illegal strategies,
  etc.) are listed in [diagnostic.md](diagnostic.md). Fixing them changes
  results, so confirm before fixing, and update diagnostic.md when one is resolved.

## Data flow / hand-offs

Scripts communicate through CSVs in `data/`, not by importing each other. Each stage writes its
hand-off file to `data/` in addition to its own `results/<name>/` copy:

- `process_laps.py`   → `data/laps_combined_clean.csv`
- `process_stints.py` → `data/stints_combined_clean.csv`
- `tyre_sets.py`      → `data/tyre_sets_race_inventory.csv` (one row per set each car in `RACE_CARS` has
  left for the race: compound and age in laps). OpenF1 has no set ID, so sets are chained by compound and
  `tyre_age_at_start`; allocation, returns and race-reserved sets are `.env` settings (diagnostic.md #21).
  Practice only: qualifying usage isn't tracked.
- `process_weather.py`→ `data/cleaned_weather.csv`
- `process_cardata.py`→ `data/cardata_per_lap_clean.csv` (keyed by `session_key`,
  `driver_number`, `lap_number`; lap windows come from the raw `laps_p*.csv`, and only laps in
  `data/laps_combined_clean.csv` are kept, so it must run after `process_laps.py`; not read
  downstream yet)
- `feature_engineer.py` reads the three above → `data/final_engineered_dataset.csv`
  (and a copy under `results/feature_engineer/`). It keeps `session_key` and `driver_number` as
  identifiers only; they are not in `MODEL_FEATURE_COLUMNS`. The session reaches the model only as the
  `session_FP1/FP2/FP3` one-hot, built from `process_laps.py`'s `session_name`.
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
  Strategies are optimised per car in `RACE_CARS` from its tyre-set inventory: each set is fitted at most
  once, and the fastest choice of sets is kept for each compound sequence and pit-lap plan.

The **model target** is next-lap `fuel_corrected_pace`: the same driver's lap `lap_number + 1` in the
same stint, built by `build_next_lap_target()` in `feature_config.py` (shared by `train_model.py` and
`validate_strategy.py`, so the target can't diverge between them). `lap_duration` and `laps_remaining` are
dependent variables and are deliberately excluded from model inputs to avoid leakage.
The target, every prediction, the simulator's totals and the error metrics are in **seconds**: lap times
are not scaled (diagnostic.md #7). Don't reintroduce a per-session scaler on lap columns.

## Conventions to follow

Every pipeline script follows the same house style — match it when adding or editing scripts:

- **Per-script results dir**: `SCRIPT_NAME = "..."`, `RESULTS_DIR = f"./results/{SCRIPT_NAME}"`,
  created with `os.makedirs(..., exist_ok=True)`.
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
  stay in code (diagnostic.md #22).
- **Compound one-hot columns** are `compound_HARD`, `compound_MEDIUM`, `compound_SOFT`.
- **Session one-hot columns** are `session_FP1`, `session_FP2`, `session_FP3` (`SESSION_COLUMNS` in
  feature_config.py), named by practice order so they stay the same on any weekend.

## Environment notes

- **Platform is Windows** (PowerShell primary). Prefer forward-slash relative paths in Python;
  they work fine via `pathlib`/`os.path`.
- **`requirements.txt` is UTF-16 encoded.** It will look garbled in a plain read and
  `pip install -r` may choke. Key pins: `pandas==2.3.3`, `numpy==2.3.4`,
  `scikit-learn==1.8.0`, `xgboost==3.1.2`, `matplotlib==3.10.7`, `seaborn==0.13.2`,
  `scipy==1.16.3`, `joblib==1.5.2`, `requests==2.32.5`. Preserve the encoding if you rewrite it,
  or convert to UTF-8 deliberately and mention it.
- Only `.venv/` is gitignored; generated `results/` CSVs and logs **are** tracked (they show up
  in `git status` after every run). Don't assume they're throwaway — the repo commits them.

## Working guidelines for agents

- The whole pipeline is **for a single reference weekend**; the scope is intentionally limited to
  the three practice sessions. Don't broaden scope (extra sessions, race data) unless asked.
- Guard against **target leakage** above all else — it's the central design constraint. Any new
  feature must belong to lap *t* when predicting lap *t+1*.
- When adding a new pipeline stage, wire it into `master.py`'s sequence and give it its own
  `results/<name>/` directory and log, following the pattern above.
- Prefer editing the existing scripts over introducing new frameworks; this is a plain
  pandas/scikit-learn/xgboost project with no test suite or CI.
