# F1 Race Strategy Optimizer

> A machine learning pipeline for predicting optimal tyre strategies in Formula 1 races using practice session telemetry data.

**Data Science Project (CSE558)**  
Indraprastha Institute of Information Technology Delhi  
*Author: Devaj Rathore*

---

## 📋 Problem Statement

Predict the optimal tyre strategy for a Formula 1 race weekend using data from three practice sessions (FP1, FP2, FP3). The strategy must utilize at least two of the three available compounds: **Soft**, **Medium**, and **Hard**.

---

## 🏎️ Project Overview

This pipeline analyzes practice session data to:
- Model tyre degradation behavior across different compounds
- Simulate race conditions lap-by-lap
- Optimize pit stop timing and tyre compound selection
- Minimize total race time through strategic planning

**Case Study**: Bahrain Grand Prix 2025  
*(Pipeline is generalizable to any non-sprint F1 weekend)*

---

## 📊 Data Collection

All data is sourced from the [OpenF1 API](https://openf1.org/).

### Data Sources

**1. Session Identification**

Each weekend's meeting key, FP1-FP3 session keys and scheduled race laps are hardcoded in the `WEEKENDS` table in `scripts/race_weekend.py` (every non-sprint 2024 and 2025 weekend; sprint weekends have FP1 only and are rejected). The keys come from:
```
GET https://api.openf1.org/v1/sessions?year=<year>&session_type=Practice
```

For the reference weekend, Bahrain 2025 (meeting 1257):

| Session ID | Session Name | Date | Time (UTC) |
|------------|-------------|------|------------|
| 10007 | Practice 1 | 2025-04-11 | 11:30 - 12:30 |
| 10008 | Practice 2 | 2025-04-11 | 15:00 - 16:00 |
| 10009 | Practice 3 | 2025-04-12 | 12:30 - 13:30 |

**2. Telemetry Data Downloads**

`scripts/race_weekend.py` (step 0 of `master.py`) downloads these for the weekend in `RACE_YEAR` / `RACE_WEEKEND` (`.env`), for any non-sprint 2024 or 2025 weekend. Its `WEEKENDS` table hardcodes each weekend's meeting key, FP1-FP3 session keys and scheduled race laps. Each weekend is kept in `data/cache/<year>_<name>/` (gitignored, laid out like `data/raw/`). A weekend fetched before is loaded from there with no requests; otherwise the current `data/raw/` is moved into the cache and the missing files are downloaded, waiting `OPENF1_REQUEST_DELAY_S` between requests and retrying HTTP 429/5xx and network errors (see [Weekend cache](#weekend-cache)). The requests per weekend are:

```
GET https://api.openf1.org/v1/laps?session_key=<FP1|FP2|FP3>&csv=true        -> data/raw/laps/laps_p1-3.csv
GET https://api.openf1.org/v1/stints?session_key=<FP1|FP2|FP3>&csv=true      -> data/raw/stints/stints_p1-3.csv
GET https://api.openf1.org/v1/drivers?session_key=<FP1|FP2|FP3>&csv=true     -> data/raw/drivers/drivers_p1-3.csv
GET https://api.openf1.org/v1/weather?meeting_key=<meeting>&csv=true         -> data/raw/weather/weather.csv (process_weather.py keeps the practice sessions)
GET https://api.openf1.org/v1/car_data?session_key=<s>&driver_number=<d>&csv=true -> data/raw/car_data/car_data_p<n>_<d>.csv
```

Car data is requested for every driver in that session's downloaded laps file, so it follows each weekend's field (for Bahrain 2025: 20 per session, 60 files, ~66 MB). The drivers files are the entry lists that `tyre_sets.py` uses to find stand-ins: the FP3 lineup is taken as the race cars, and an FP1/FP2 driver outside it counts toward the same team's driver who sat that session out (in Bahrain 2025, six FP1 stand-ins, e.g. driver 38 in Ferrari's #16).

Laps and stints cover the whole field (20 drivers per session in Bahrain 2025), and the model is trained on all of them.

### Weekend cache

- **Layout:** one folder per weekend, `data/cache/<year>_<name>/` (`CACHE_DIR`, e.g. `2025_bahrain`), with the same `laps/`, `stints/`, `drivers/`, `weather/` and `car_data/` subfolders as `data/raw/`. About 70 files and 67-69 MB per weekend. Gitignored.
- **`data/raw/` is the active weekend only**, copied from its cache folder, so no other stage depends on which weekend is selected.
- **Switching weekends:** `data/raw/` is moved into the cache folder of the weekend it holds (found from the `meeting_key` in `laps_p1.csv`; an unknown meeting goes to `meeting_<key>/`). Files the cache already has are kept there. The selected weekend's folder is completed from OpenF1 and copied into `data/raw/`. A weekend fetched before needs no requests.
- **Interrupted downloads** are written to `<file>.part` and renamed when complete, so a re-run fetches only what is still missing.
- **No car data:** a car-data request with no results (HTTP 404) leaves a `<file>.nodata` marker, so it isn't requested again; delete the marker to retry. Laps, stints, drivers or weather with no results stop the run.
- **Force a re-download:** delete the file (or the weekend's folder) from `data/cache/`, and also from `data/raw/` if it is the active weekend (a complete `data/raw/` is left alone, and a file deleted from `data/raw/` only is restored from the cache).
- **Run record:** `scripts/results/race_weekend/race_weekend_summary.csv` says where the run got its data (`data/raw (unchanged)`, `cache`, or `cache, N file(s) downloaded`), with file counts and any drivers without car data.
- **Git:** `data/raw/`, `data/processed/` and `scripts/results/` are tracked, so running another weekend rewrites them. Switching back to 2025 `bahrain` restores the committed data from the cache.

### Dataset Descriptions

#### **Laps Dataset**
Key features for lap-level telemetry:
- `lap_duration`: Total lap time (the basis of fuel-corrected pace, whose fall-off with tyre age the model learns)
- `duration_sector_1/2/3`: Individual sector times
- `i1_speed`, `i2_speed`, `st_speed`: Speeds at the two intermediates and the speed trap
- `lap_number`: Sequential lap index
- `is_pit_out_lap`: Flag for pit exit laps (excluded from analysis)

#### **Stints Dataset**
Tyre compound and usage information:
- `compound`: Tyre type (Soft/Medium/Hard/Inter/Wet)
- `lap_start`, `lap_end`: Stint boundaries
- `stint_number`: Sequential stint identifier
- `tyre_age_at_start`: Previous laps on this tyre set

#### **Drivers Dataset**
Per-session entry lists:
- `driver_number`, `team_name`: used by `tyre_sets.py` to pair practice stand-ins with their team's race car

#### **Weather Dataset**
Environmental conditions:
- `air_temperature`, `track_temperature`: Critical for tyre performance
- `humidity`, `pressure`: Atmospheric conditions
- `wind_speed`, `wind_direction`: Aerodynamic factors
- `rainfall`: Rain flag (0/1)

---

## 🔧 Data Preparation

### Cleaning Scripts
- `process_laps.py`: Removes outliers and invalid laps
- `process_stints.py`: Validates stint metadata
- `tyre_sets.py`: Infers every car's tyre sets from the practice stints (chained by compound and `tyre_age_at_start`) and writes the sets left for the race
- `process_weather.py`: Filters atmospheric anomalies
- `process_cardata.py`: Drops no-input (brake/throttle = 104) and out-of-range telemetry samples, encodes brake/DRS as 0/1, aggregates the ~4 Hz samples per lap (speed, throttle, braking, rpm, gears, DRS), keeps only laps that survive `process_laps.py`, masks z-score outliers (only the flagged aggregate is set to NaN; the lap is kept), and leaves the aggregates unscaled (physical units). They are merged into the engineered dataset as model inputs

### Data Cleaning Process

**1. Redundancy Removal**
- Removed meeting keys; `session_key` is kept in laps, stints and car data (lap and stint numbers restart every session), and only weather drops it
- Excluded pit-out laps (slow in-laps are caught by the non-push filter)

**2. Outlier Detection & Removal**

*Laps (per session):*
- Missing or non-positive sector times (a single missing sector is first rebuilt as `lap_duration` minus the other two, #33)
- Unrealistic lap times (≤ 30s or ≥ 2× session median)
- Aborted laps (a recorded `i1_speed`, `i2_speed` or `st_speed` ≤ 50 km/h; laps with a missing speed are kept, and the missing value is filled from the stint's earlier laps, see Feature Engineering)
- Rolling median residual filtering: residual = lap time − 5-lap centred rolling median per driver; removed laps below `Q1 − 60×IQR` (fast) or above `Q3 + 3×IQR` (slow). The two multipliers (`IQR_LOWER_MULTIPLIER`, `IQR_UPPER_MULTIPLIER`) were tuned over 12 weekends by `explore/tune_iqr.py` (#32): a fast lap in practice is a push lap run between cool-down laps, so the fast side is in effect off
- Non-push laps (cool-down, in- and traffic laps): slower than 1.15× the driver's best lap in the session (`PUSH_LAP_MAX_BEST_RATIO`), which leaves 573 laps (Bahrain 2025). A second test against the driver's stint best (`PUSH_LAP_MAX_STINT_RATIO`) exists but is off: `explore/tune_push.py` (#30) found 1.15 already sits at the gap between push and cool-down laps (labelled from throttle telemetry over 12 weekends), and that a stricter filter mostly drops long-run laps

*Stints:*
- Logical validation: `lap_start ≤ lap_end`
- Non-negative `tyre_age_at_start`
- Non-null compound values

*Weather:*
- Practice sessions only; Z-score filtering (|z| > 3) on temperature, humidity, pressure, wind speed and rainfall (wind direction excluded, zero-variance columns skipped)

**3. Encoding**
- `driver_number`: Kept numerical as an identifier (with `session_key`) to build stints and driver-session levels; not a model input (generalized model)
- `compound`: One-hot encoded → `compound_SOFT`, `compound_MEDIUM`, `compound_HARD`
- `session_name` (FP1/FP2/FP3, from the `laps_p1/p2/p3` file): One-hot encoded → `session_FP1`, `session_FP2`, `session_FP3`, a model input that separates session-level pace offsets; `session_key` stays an identifier
- `segments_sector_1/2/3`: List of mini-sector codes replaced by its length

**4. Normalization**
- **Laps**: Not scaled. Lap times, sector times and speeds stay in seconds and km/h. The models are unaffected by a global rescale, a per-session scaler would remove each session's pace level and leave predictions in z-scores, and session offsets are handled by the session one-hot instead
- **Car data**: Not scaled, like the laps, so the per-lap aggregates stay in km/h, %, rpm and shares
- **Weather**: StandardScaler on temperature, humidity, pressure, wind speed, rainfall
- **Wind direction**: Kept raw (circular variable)

**5. Feature Engineering**

| Feature | Formula | Purpose |
|---------|---------|---------|
| `stint_number` | Stints numbered 1, 2, 3… per driver across all sessions | Track tyre degradation per stint |
| weather join | nearest weather sample to each lap's `date_start` (`merge_asof`, 5 min tolerance) | Give each lap the conditions it was driven in |
| car-data join | the lap's telemetry aggregates, merged on `session_key`, `driver_number`, `lap_number` (all 573 laps; 30 outlier values on 23 laps masked) | Describe how the lap was driven (speed, throttle, braking, rpm, gears, DRS); EDA and `tune_push.py` only, not a model input (#31) |
| missing-value fill | speed traps, segment counts and car data: median of the same stint's *earlier* laps (`STINT_FILL_COLUMNS`) | Keep real, lap-specific values (#33); not model inputs since #31 |
| `laps_since_stint_start` | `lap_number - lap_start + 1` | Measure tyre age in current stint |
| `tyre_age` | `tyre_age_at_start + laps_since_stint_start` | Real age of the set (laps on it including lap `t`), so used sets are told apart from new ones |
| `session_FP1/2/3` | One-hot of the practice session | Split EDA summaries by session (the stint level absorbs the session in the model) |
| `run_laps` | `lap_end - lap_start + 1` | Length of the run (quali sim vs long run); a control in the compound-gap fit (#31) |
| `stint_avg_pace` | Mean lap time (s) of the stint's earlier laps (the lap's own time on the first lap) | Descriptive; built from the lap times being predicted, so not a model input |
| `fuel_corrected_pace` | `lap_duration − 0.035 × laps_remaining` in seconds (`laps_remaining = lap_end - lap_number + 1`, laps left in the stint counting this one) | Remove the fuel effect; practice fuel loads aren't recorded, so the stint's remaining laps stand in for fuel on board (assumption) |

> **Note**: Track evolution was excluded to avoid deriving dependent variables.

### Exploratory Data Analysis (`eda.py`)

Runs after `feature_engineer.py` and describes the engineered dataset before it is modelled; outputs go to `scripts/results/eda/`. It covers summary statistics and missing values per session, lap-time and fuel-corrected-pace distributions by session and compound, pace against tyre age (raw, and the within-stint slope with each stint's mean pace and tyre age removed so driver, session and run offsets drop out), correlations of the candidate inputs (tyre and weather) with within-stint pace (flagging pairs at |r| ≥ `EDA_COLLINEAR_THRESHOLD`), and row coverage of the stints with two or more clean laps. On Bahrain 2025, within-stint pace rises by about 0.37 s/lap on SOFT, 0.23 on MEDIUM and 0.31 on HARD (the same slopes as `train_model.py`'s stage 1), over 436 laps in 96 multi-lap stints. A final section tests these effects on stints rather than single laps (#26): a stint bootstrap puts all three slopes above 0 (95% CIs SOFT 0.30–0.51, MEDIUM 0.13–0.34, HARD 0.04–0.54 s/lap) but can't tell the compounds' slopes apart; within the same driver and session, MEDIUM and HARD are 1.5 and 1.7 s slower than SOFT on a new set (short SOFT runs still inflate this); and FP2 and FP3 are about 2 s faster than FP1 for the same driver and compound.

---

## 🤖 Machine Learning Model

### Objective
Predict **tyre fall-off with age** (#31): each lap's fuel-corrected pace is a level for its stint plus a fall-off that depends on the tyre.

```
fuel_corrected_pace(lap) = level[stint] + fall_off(compound, tyre_age, conditions)
```

**Why this approach?**
- The stint's level absorbs everything constant over a run (driver, session, fuel plan, engine mode, quali sim or long run), which in practice data swamps the tyre effect
- What is left, the change within a run as the tyre ages, is exactly what the strategy needs
- No target leakage: the inputs are the compound, the tyre age and the weather, never the lap's own timing or telemetry (`LAP_TIMING_COLUMNS`, asserted in `feature_config.py`)
- Every lap of a stint with two or more clean laps is a row (436), not only laps followed by a clean next lap (273 under the old next-lap target)

### Model Selection

`train_model.py` trains one fall-off model and saves it; for now that is still XGBoost (depth 7), a leftover from the next-lap model that beats neither baseline on the new target. The model is chosen in [explore/explore_master.py](explore/explore_master.py) (`cd explore; python explore_master.py`): 13 candidates on up to 5 feature sets, nested CV grouped by stint, scored on the within-stint shape of the curve, against no fall-off (flat) and a straight line per compound (linear). Each candidate's best configuration is drawn as a curve over the race length, and it is recommended only if the curve is stable, rises on every compound and never gets faster with age. K-Means was ruled out without being implemented.

| Model (best feature set) | Shape MAE (s) | Curve |
|-------|-----------|----------|
| baseline: flat (no fall-off) | 1.200 ± 0.236 | – |
| baseline: linear per compound | 1.064 ± 0.230 | S 0.37, M 0.23, H 0.31 s/lap; rises without limit past age 19 |
| SVR (6 inputs + stint lap) | 1.026 ± 0.250 | best fit, but gets faster again past age 19 |
| **Ridge (tyre only)** | 1.048 ± 0.246 | recommended: one 0.265 s/lap slope for every compound, most stable across folds |
| MLP (6 inputs + set age) | 1.048 ± 0.239 | S 0.21, M 0.19, H 0.21 s/lap, keeps rising |
| ExtraTrees / Random Forest / XGBoost (tuned) | 1.053 / 1.069 / 1.076 | steps of 1–3 s, flat past age 19 |
| XGBoost as in `train_model.py` | 1.160–1.246 | worst: overfits |

*Bahrain 2025, 436 laps in 96 stints, mean ± std over 10 outer folds. The top configurations are within 0.04 s, against a standard error of ~0.08 s, so the data doesn't separate them. Every model beats no fall-off, so fall-off is real; none of the recommended ones significantly beats the straight line. The choice (Ridge, MLP or the per-compound line) and copying it into `train_model.py` are still open. See [diagnostic.md](diagnostic.md) (#27, #31).*

### Training Process

**1. Split**: 80/20 by stint (`GroupShuffleSplit` on `driver_number` + `stint_number`, `random_state=42`), so a stint's laps are never on both sides.

**2. Stage 1, stint levels** (`fit_stint_levels()` in `scripts/feature_config.py`): least squares of fcp on one level per stint plus one `tyre_age` slope per compound, on the training stints with two or more clean laps. The slopes (SOFT +0.36, MEDIUM +0.21, HARD +0.31 s/lap) are also the linear baseline.

**3. Stage 2, the model**: target `fcp − level[stint]`; 6 inputs (`MODEL_FEATURE_COLUMNS`): the compound one-hots, `tyre_age`, track and air temperature. The curve is measured from a new set's first lap (`fall_off_curve()`).

**4. Stage 3, compound gap** (`fit_compound_gap()`): fall-off gives each curve's shape, not how fast a compound is on a new set, so a second least-squares fit over every training lap gives the gap to SOFT and each driver-session's level, with `run_laps` as a control for the run type.

The saved bundle holds `{"model", "features", "medians", "anchor_age", "compound_gap", "levels", "linear_slopes"}`.

---

## 🏁 Race Simulation & Optimization

### 1. Strategy Space Generation

Enumerate strategies for the weekend's race length from `race_weekend.py` (Bahrain 2025: 57 laps, 16,206 strategies):
- **Stop configurations**: 1-stop, 2-stop (0 stops can't use two compounds)
- **Compound sequences**: Every SOFT/MEDIUM/HARD sequence that uses at least two different compounds (the dry-race rule)
- **Pit windows**: Pit laps between lap 10 and lap 46 (Bahrain, 57 laps; the end of the window follows the weekend's race laps)
- **Tyre sets**: the model is trained on the whole field, and the race is simulated for one car, `SIM_DRIVER` in `.env` (default 16). That car can only fit the sets it has left. `tyre_sets.py` infers them for every car from the practice stints, with each practice stand-in's running counted toward its race car (derived from the drivers files). OpenF1 has no set ID, so a stint that starts used is chained to the same-compound set that was left at that age. Allocation, returns after each session and the reserved race sets are `.env` assumptions. Each set is fitted at most once, and the fastest choice of sets is kept for each plan. Sequences needing more sets than the car has (e.g. two HARDs) are skipped, leaving 12,210 for car 16 in Bahrain 2025. Qualifying usage isn't tracked.

### 2. Lap Time Prediction

For each compound and set age a car has left, evaluate the curve for stint laps 1 to the race length (57 in Bahrain 2025) in one model call. Tyre age resets to the set's age at every stop, and a used set starts further along its curve.

```
fcp on stint lap n = race pace level + compound gap + fall_off(compound, set age + n)
```

- **fall_off**: the model's curve from a new set's first lap, with the weather at the training medians
- **compound gap**: stage 3 (SOFT 0, MEDIUM +2.97 s, HARD +0.85 s on the current data)
- **race pace level**: the mean of `SIM_DRIVER`'s driver-session levels (93.06 s for car 16); the same for every strategy, so it doesn't change the ranking
- **Price**: a stint of L laps costs the sum of its first L laps, plus a race fuel penalty of `0.035 s × laps of fuel remaining` for each race lap it covers. Nothing is fed back from one lap to the next.

### 3. Objective Function

```
Minimize: Total Race Time (s) = Σ(stint lap times) + 22 s × number of pit stops
```

Current output for `SIM_DRIVER=16` (set age at the start of each stint in brackets):
- SOFT(0)–SOFT(0)–HARD(0), stops on laps 14 and 28, 5649.51 s (tied with the same three stints in another order; pitting a lap later +0.17 s, best with MEDIUM +1.88 s); best 1-stop (MEDIUM(0)–HARD(0), lap 42) 28.36 s slower

Other cars differ through their sets and pace level.

Every race lap is summed once: "stop on lap p" means lap p is the in-lap and the next stint starts on lap p + 1, with the 22 s pit loss on top.

2 stops win because the curves fall off steeply (5–8 s by tyre age 15), so fresh tyres are worth a stop. This is not a valid recommendation yet: the 29-lap HARD stint runs past the oldest tyre in training (19 laps), where the tree's curve is flat; the compound gap is implausible (MEDIUM slower than HARD); and the pipeline's XGBoost doesn't beat the simple baselines in validation. See [diagnostic.md](diagnostic.md) (#27).

### 4. Optimization Engine

- **Method**: Brute force: every strategy is evaluated and the lowest total wins
- **Future enhancement**: Dynamic Programming to prune inferior strategies, Genetic Algorithm for deeper search
- **Output**: `scripts/results/strategy_simulator/` (best strategy JSON, all strategy results CSV)

---

## ✅ Evaluation & Validation

### Model Validation (`validate_strategy.py`)
- **Cross-validation**: 5-fold CV grouped by stint over the 436 laps of multi-lap stints, with stage 1 and the input medians fitted per fold. A held-out stint's level is unknown, so the score is the **shape** of the curve within each stint (residual minus the stint's mean residual). Current: shape MAE 1.25 s, against 1.19 s for no fall-off (flat) and 1.06 s for the stage-1 slope per compound (linear)
- **Baseline tests**: per-stint paired t-test and Wilcoxon test against each baseline. Current: the pipeline's XGBoost (depth 7, untuned) beats neither (better than flat on 48 of 96 stints, p = 0.81 / 0.75); the straight line per compound does best
- **Curve uncertainty**: each fold's fall-off at tyre ages 5, 10 and 15. At age 15: SOFT +5.6 to +7.9 s, MEDIUM +2.5 to +4.7 s, HARD +4.6 to +5.6 s; SOFT falls off most in every fold
- **Residual analysis**: out-of-fold within-stint residual against tyre age, by compound
- **Curve check** (logged by `strategy_simulator.py`): fcp and fall-off at stint laps 1, 18 and 57 for each compound and set age, next to the linear fall-off, plus the largest step (2.4–3.6 s: the tree's curve moves in steps, and is flat past tyre age 19)

### Strategy Validation *(planned, not implemented yet)*
- Simulate multiple race conditions with variations
- Compare alternative strategies across pit windows
- Generate heatmaps of predicted race time vs strategy grid
- **Stability check**: Ensure optimal strategy persists under small perturbations

---

## 🚀 Getting Started

### Prerequisites
Python 3.10. Dependencies are pinned in `requirements.txt` (the versions that produced the committed results):
```bash
pip install -r requirements.txt
```

### Usage

**1. Choose the weekend**: set `RACE_YEAR` and `RACE_WEEKEND` in [.env](.env) (default 2025 `bahrain`; the names are in `scripts/race_weekend.py`). `master.py` fetches its data (see [Data Sources](#data-sources)), or run `python race_weekend.py` on its own. A weekend fetched before is loaded from `data/cache/` with no requests ([Weekend cache](#weekend-cache)).

**2. Run the pipeline.** Scripts use relative paths, so run them from `scripts/`:
```bash
cd scripts
python master.py
```

`master.py` runs these in order and stops at the first failure:

| Step | Script | Hands off |
|------|--------|-----------|
| 0 | `race_weekend.py` | `data/raw/` (the selected weekend, from `data/cache/` or OpenF1) |
| 1 | `process_laps.py` | `data/processed/laps_combined_clean.csv` |
| 2 | `process_stints.py` | `data/processed/stints_combined_clean.csv` |
| 3 | `tyre_sets.py` | `data/processed/tyre_sets_race_inventory.csv` (each car's sets left for the race) |
| 4 | `process_weather.py` | `data/processed/cleaned_weather.csv` |
| 5 | `process_cardata.py` | `data/processed/cardata_per_lap_clean.csv` (merged by `feature_engineer.py`) |
| 6 | `feature_engineer.py` | `data/processed/final_engineered_dataset.csv` |
| 7 | `eda.py` | EDA tables and plots in `scripts/results/eda/` (nothing downstream reads them) |
| 8 | `train_model.py` | `scripts/results/train_model/train_model_xgb_model.pkl` |
| 9 | `strategy_simulator.py` | best strategy for `SIM_DRIVER` |
| 10 | `validate_strategy.py` | CV metrics, residual plot |

Each script can also be run on its own, and writes its log and outputs to `scripts/results/<script_name>/`.

**Settings** (race weekend, paths, file names, cleaning thresholds, model hyperparameters, seeds, pit loss, pit window, stop counts) are in [.env](.env) at the repo root, which `scripts/env_config.py` reads; the race length comes from the weekend table. An environment variable with the same name overrides the file for one run, e.g. `$env:PIT_LOSS_S="25"; python strategy_simulator.py`, or `$env:RACE_YEAR="2024"; $env:RACE_WEEKEND="saudi_arabia"; python master.py` (PowerShell).

Detailed methodology: [readme.txt](readme.txt). Known bugs: [diagnostic.md](diagnostic.md).

---

## 📈 Key Insights

### Physical Considerations the Model Is Meant to Capture
- Tyre thermal degradation over stint length
- Compound-specific grip vs durability trade-offs
- Track temperature impact on tyre performance
- Fuel load reduction over race distance

Since #31 the model learns fall-off with tyre age directly, and the curves rise on every compound (SOFT most). The compound gap on a new set and the shape past the oldest tyre in training (19 laps) are not reliable yet (see Model Validation).

### Real-World Alignment
This pipeline mirrors professional motorsport analytics:
> "Learn how a tyre falls off with age within a run, and price every stint from that curve."

---

## 🔮 Future Enhancements

- [ ] Incorporate qualifying position for overtaking probability
- [ ] Add safety car probability modeling
- [ ] Implement real-time strategy adjustment during races
- [ ] Extend to sprint race weekends
- [ ] Driver-specific modeling for personalized strategies
- [ ] Genetic Algorithm integration for larger strategy spaces

---

## 📝 License

This project is developed for academic purposes at IIIT Delhi.

---

## 👤 Author

**Devaj Rathore**  
*Newly Self-Appointed ML Expert*  
Data Science CSE558  
Indraprastha Institute of Information Technology Delhi

---

## 🙏 Acknowledgments

- [OpenF1 API](https://openf1.org/) for comprehensive F1 telemetry data
- The data science community for pandas, scikit-learn and XGBoost
- Formula 1 for being the ultimate testing ground for optimization problems

---

*"In racing, they say that your car goes where your eyes go. The driver who cannot tear his eyes away from the wall as he spins out of control will meet that wall; the driver who looks down the track as he struggles to control his car will somehow manage to find his way back to the track."* — Garth Stein
