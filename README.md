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
- `lap_duration`: Total lap time (the basis of fuel-corrected pace; the model's target is the next lap's fuel-corrected pace)
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
- `process_cardata.py`: Drops no-input (brake/throttle = 104) and out-of-range telemetry samples, encodes brake/DRS as 0/1, aggregates the ~4 Hz samples per lap (speed, throttle, braking, rpm, gears, DRS), keeps only laps that survive `process_laps.py`, removes z-score outliers, and leaves the aggregates unscaled (physical units). They are merged into the engineered dataset as model inputs

### Data Cleaning Process

**1. Redundancy Removal**
- Removed meeting keys; `session_key` is kept in laps, stints and car data (lap and stint numbers restart every session), and only weather drops it
- Excluded pit-out laps (slow in-laps are caught by the non-push filter)

**2. Outlier Detection & Removal**

*Laps (per session):*
- Missing or non-positive sector times
- Unrealistic lap times (≤ 30s or ≥ 2× session median)
- Aborted laps (`i1_speed`, `i2_speed` or `st_speed` ≤ 50 km/h)
- Rolling median residual filtering: residual = lap time − 5-lap centred rolling median per driver; removed laps outside `[Q1 - 3×IQR, Q3 + 3×IQR]`
- Non-push laps (cool-down, in- and traffic laps): slower than 1.15× the driver's best lap in the session (`PUSH_LAP_MAX_BEST_RATIO`), which leaves 467 laps (Bahrain 2025)

*Stints:*
- Logical validation: `lap_start ≤ lap_end`
- Non-negative `tyre_age_at_start`
- Non-null compound values

*Weather:*
- Practice sessions only; Z-score filtering (|z| > 3) on temperature, humidity, pressure, wind speed and rainfall (wind direction excluded, zero-variance columns skipped)

**3. Encoding**
- `driver_number`: Kept numerical as an identifier (with `session_key`) to build the per-driver next-lap target; not a model input (generalized model)
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
| car-data join | lap `t`'s per-lap telemetry aggregates, merged on `session_key`, `driver_number`, `lap_number` (448 of 467 laps; the rest filled with medians) | Describe how lap `t` was driven (speed, throttle, braking, rpm, gears, DRS) |
| `laps_since_stint_start` | `lap_number - lap_start + 1` | Measure tyre age in current stint |
| `tyre_age` | `tyre_age_at_start + laps_since_stint_start` | Real age of the set (laps on it including lap `t`), so used sets are told apart from new ones |
| `session_FP1/2/3` | One-hot of the practice session | Separate session-level pace offsets |
| `stint_avg_pace` | Mean lap time (s) of the stint's earlier laps (NaN on the first lap, filled with the median) | Capture stint-level performance without seeing lap `t+1` |
| `fuel_corrected_pace` | `lap_duration − 0.035 × laps_remaining` in seconds (`laps_remaining = lap_end - lap_number + 1`, laps left in the stint counting this one) | Remove the fuel effect; practice fuel loads aren't recorded, so the stint's remaining laps stand in for fuel on board (assumption) |

> **Note**: Track evolution was excluded to avoid deriving dependent variables.

### Exploratory Data Analysis (`eda.py`)

Runs after `feature_engineer.py` and describes the engineered dataset before it is modelled; outputs go to `scripts/results/eda/`. It covers summary statistics and missing values per session, lap-time and fuel-corrected-pace distributions by session and compound, pace against tyre age (raw, and with each stint's mean removed so driver and session offsets drop out), correlations between the inputs and with the next-lap target (flagging pairs at |r| ≥ `EDA_COLLINEAR_THRESHOLD`), and row coverage before and after the next-lap target shift. On Bahrain 2025, within-stint pace rises by about 0.17 s/lap on SOFT and 0.04–0.05 s/lap on MEDIUM/HARD, but only 153 of 467 laps (7 in FP3) survive the target shift, and the session one-hots and weather inputs form one collinear group.

---

## 🤖 Machine Learning Model

### Objective
Predict **fuel-corrected pace on the next lap** rather than explicitly computing degradation rates.

**Why this approach?**
- Fully causal: All inputs from lap `t` predict lap `t+1`
- No target leakage
- Degradation emerges naturally from repeated predictions
- Perfect for autoregressive race simulation

### Model Selection: XGBoost

`train_model.py` trains Linear Regression, Random Forest and XGBoost on the same split and saves XGBoost. K-Means was ruled out without being implemented.

| Model | Reasoning | Test MAE | Test RMSE | Test R² |
|-------|-----------|----------|-----------|---------|
| Linear Regression | Underfits non-linear tyre behavior | 2.37 s | 2.74 s | −1.002 |
| Random Forest | Noisy, stepwise predictions | 1.32 s | 1.68 s | 0.250 |
| **XGBoost** | Chosen for non-linear learning and smooth predictions | 1.12 s | 1.56 s | 0.353 |

*MAE and RMSE are in seconds, from the current run. The split is by stint, so the test set is 10 stints (25 rows) the model never saw; with so few, the gap between Random Forest and XGBoost is close to noise. See [diagnostic.md](diagnostic.md) (#27).*

**XGBoost Advantages:**
- Captures non-linear tyre physics (warm-up, thermal degradation, cliff effects)
- Smooth predictions for repeated (autoregressive) use
- Fast inference
- Interpretable via built-in feature importance

### Training Process

**1. Target Variable**
```python
y = fuel_corrected_pace_next_lap
```
Built by `build_next_lap_target()` in `scripts/feature_config.py`: the same driver's lap `lap_number + 1` in the same stint. The last lap of each stint and laps followed by a cleaning gap are dropped (467 → 153 rows: practice runs alternate push and cool-down laps, so removing the cool-down laps breaks most pairs).

**2. Removed Dependent Variables**
- `lap_duration` (sum of sector times)
- `laps_remaining` (used in fuel correction formula)

**3. Features and Split**
- 36 inputs defined in `scripts/feature_config.py` (`MODEL_FEATURE_COLUMNS`), including the `session_FP1/2/3` one-hots and 10 car-data aggregates (`CARDATA_FEATURE_COLUMNS`)
- Missing values filled with column medians; 80/20 train/test split by stint (`GroupShuffleSplit` on `driver_number` + `stint_number`, `random_state=42`), so a stint's laps are never on both sides
- The saved bundle holds the model, the feature list, the training medians and each session's share of training rows (`{"model", "features", "medians", "session_weights"}`)

**4. Validation**
- 5-fold cross-validation grouped by stint (`GroupKFold`)
- Residual analysis for systematic bias
- Random Forest and XGBoost feature importances

---

## 🏁 Race Simulation & Optimization

### 1. Strategy Space Generation

Enumerate strategies for the weekend's race length from `race_weekend.py` (Bahrain 2025: 57 laps, 16,206 strategies):
- **Stop configurations**: 1-stop, 2-stop (0 stops can't use two compounds)
- **Compound sequences**: Every SOFT/MEDIUM/HARD sequence that uses at least two different compounds (the dry-race rule)
- **Pit windows**: Pit laps between lap 10 and lap 46 (Bahrain, 57 laps; the end of the window follows the weekend's race laps)
- **Tyre sets**: the model is trained on the whole field, and the race is simulated for one car, `SIM_DRIVER` in `.env` (default 16). That car can only fit the sets it has left. `tyre_sets.py` infers them for every car from the practice stints, with each practice stand-in's running counted toward its race car (derived from the drivers files). OpenF1 has no set ID, so a stint that starts used is chained to the same-compound set that was left at that age. Allocation, returns after each session and the reserved race sets are `.env` assumptions. Each set is fitted at most once, and the fastest choice of sets is kept for each plan. Sequences needing more sets than the car has (e.g. two HARDs) are skipped, leaving 12,210 for car 16 in Bahrain 2025. Qualifying usage isn't tracked.

### 2. Lap Time Prediction

For each compound and set age a car has left, and every stint length L = 1 to the race length (57 in Bahrain 2025), roll the stint forward from stint lap 1. Tyre age resets to the set's age at every stop.
1. **Seed:** lap 1 is a median lap: the training medians, the compound one-hot set, `tyre_age_at_start` = the set's age and `tyre_age` = set age + 1.
2. **Predict:** XGBoost predicts the next lap's fuel-corrected pace from the current lap's inputs.
   - The race is not a practice session, so each lap is predicted under `session_FP1`, `session_FP2` and `session_FP3` and averaged, weighted by each session's share of training rows.
3. **Feed back:** each prediction becomes the next lap's inputs.
   - The training fuel term is added back first: `raw = fcp + 0.035 × (L − age + 1)`.
   - The sector times are rescaled to sum to `raw`.
   - `stint_avg_pace` is the mean of the earlier laps.
   - `laps_since_stint_start` and `tyre_age` go up by one.
   - Speeds, weather and the car-data aggregates stay at their medians.
4. **Price:** a stint's time is the summed rollout fcp for its length, plus a race fuel penalty of `0.035 s × laps of fuel remaining` for each race lap it covers.

### 3. Objective Function

```
Minimize: Total Race Time (s) = Σ(stint lap times) + 22 s × number of pit stops
```

Current output for `SIM_DRIVER=16` (set age at the start of each stint in brackets):
- MEDIUM(0)–HARD(0), stop on lap 10, 5774.26 s (SOFT(0)–HARD(0) on the same lap is +0.02 s); best 2-stop 17.16 s slower

Other cars give similar results from their own sets.

Every race lap is summed once: "stop on lap p" means lap p is the in-lap and the next stint starts on lap p + 1, with the 22 s pit loss on top.

None of this is tyre physics yet. 1 stop wins because every rollout is almost flat (about 100 s from stint lap 3 on, whatever the compound or set age), so an extra stop can't win back its 22 s. The top strategies are within about 0.5 s and the compounds barely differ, so the pit lap and compound choice are close to arbitrary. The model is trained on only 153 rows, so this is not a valid recommendation yet.

### 4. Optimization Engine

- **Method**: Brute force: every strategy is evaluated and the lowest total wins
- **Future enhancement**: Dynamic Programming to prune inferior strategies, Genetic Algorithm for deeper search
- **Output**: `scripts/results/strategy_simulator/` (best strategy JSON, all strategy results CSV)

---

## ✅ Evaluation & Validation

### Model Validation (`validate_strategy.py`)
- **Cross-validation**: 5-fold CV grouped by stint on next-lap predictions (current: MAE 1.23 s, RMSE 1.70 s)
- **Residual analysis**: Check for systematic bias
- **Feature importance**: Verify alignment with tyre physics
  - Expected top features: tyre age, compound, track temperature
- **Baseline test**: per-stint paired t-test and Wilcoxon test of the model's out-of-fold error against a persistence baseline (lap *t+1* = lap *t*). Current: baseline MAE 1.80 s, model better on 32 of 50 stints, p = 0.0008 (t) / 0.011 (Wilcoxon), median gain 0.14 s
- **Simulation stability** (logged by `strategy_simulator.py`): fcp at stint laps 1, 18 and 57 of a full-length rollout for each compound and set age, plus the largest lap-to-lap step.
  - No runaway drift (car 16, Bahrain 2025): every rollout stays within 97.15–101.54 s, and the largest lap-to-lap step is 2.28–2.60 s (stint laps 1–3).
  - Smooth but nearly flat: about 99.5 s on stint lap 2 and 101 s on lap 3, then 100.5–101.5 s to lap 57. All compounds and set ages are within about 0.4 s, so the model shows almost no tyre degradation yet.

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

So far only the fuel term is built in; the model has learnt almost none of the tyre effects from 153 training rows (see Simulation stability).

### Real-World Alignment
This pipeline mirrors professional motorsport analytics:
> "Predict the next lap's corrected pace and let degradation emerge naturally from repeated predictions."

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
