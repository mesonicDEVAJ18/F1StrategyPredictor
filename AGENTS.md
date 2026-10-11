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
  feature_config.py     # shared column lists (MODEL_FEATURE_COLUMNS, SIM_FEATURE_COLUMNS)
  process_laps.py       # clean + scale lap data
  process_stints.py     # clean stints, one-hot encode compound
  process_weather.py    # clean + scale weather
  process_cardata.py    # clean car telemetry, aggregate per lap (not a model input yet)
  feature_engineer.py   # merge laps+stints+weather, build engineered features
  train_model.py        # train/compare models, persist XGBoost bundle
  strategy_simulator.py # enumerate + simulate strategies, pick best
  validate_strategy.py  # cross-validation, residuals, strategy robustness
  results/<script>/     # per-script outputs (logs, intermediate CSVs, plots, model)
data/               # raw + cleaned CSVs (OpenF1 downloads and pipeline hand-offs)
results/            # stray: empty validate_strategy log from running a script outside scripts/
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

`master.py` runs, in order: `process_laps` → `process_stints` → `process_weather` →
`process_cardata`, then `feature_engineer` → `train_model` → `strategy_simulator` →
`validate_strategy`. It aborts the whole run if any stage exits non-zero.

**Known gaps (don't "fix" silently — confirm intent first):**
- Car data is cleaned, encoded, aggregated per lap and scaled, but it is **not** merged into
  the engineered dataset or used as a model feature (diagnostic.md #16). It only covers the two
  Ferrari cars, while the model trains on the whole field.
- Known modelling bugs (stint merge, cross-driver targets, mixed units in the simulator, the
  wrong weather event, etc.) are listed in [diagnostic.md](diagnostic.md). Fixing them changes
  results, so confirm before fixing, and update diagnostic.md when one is resolved.

## Data flow / hand-offs

Scripts communicate through CSVs in `data/`, not by importing each other. Each stage writes its
hand-off file to `data/` in addition to its own `results/<name>/` copy:

- `process_laps.py`   → `data/laps_combined_clean.csv`
- `process_stints.py` → `data/stints_combined_clean.csv`
- `process_weather.py`→ `data/cleaned_weather.csv`
- `process_cardata.py`→ `data/cardata_per_lap_clean.csv` (keyed by `session_key`,
  `driver_number`, `lap_number`; lap windows come from the raw `laps_p*.csv`, and only laps in
  `data/laps_combined_clean.csv` are kept, so it must run after `process_laps.py`; not read
  downstream yet)
- `feature_engineer.py` reads the three above → `data/final_engineered_dataset.csv`
  (and a copy under `results/feature_engineer/`)
- `train_model.py` reads the engineered dataset → `results/train_model/train_model_xgb_model.pkl`
  (a joblib bundle: `{"model": ..., "features": [...]}`)
- `strategy_simulator.py` and `validate_strategy.py` load that `.pkl` bundle.

The **model target** is next-lap `fuel_corrected_pace`. `lap_duration` and `laps_remaining` are
dependent variables and are deliberately excluded from model inputs to avoid leakage.

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
- **Compound one-hot columns** are `compound_HARD`, `compound_MEDIUM`, `compound_SOFT`.

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
