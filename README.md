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
```
GET https://api.openf1.org/v1/sessions?country_name=Bahrain&year=2025
```

| Session ID | Session Name | Date | Time (UTC) |
|------------|-------------|------|------------|
| 10007 | Practice 1 | 2025-04-11 | 11:30 - 12:30 |
| 10008 | Practice 2 | 2025-04-11 | 15:00 - 16:00 |
| 10009 | Practice 3 | 2025-04-12 | 12:30 - 13:30 |

**2. Telemetry Data Downloads**

```bash
# Lap data for all three sessions
curl "https://api.openf1.org/v1/laps?session_key=10007&csv=true" -o data/laps_p1.csv
curl "https://api.openf1.org/v1/laps?session_key=10008&csv=true" -o data/laps_p2.csv
curl "https://api.openf1.org/v1/laps?session_key=10009&csv=true" -o data/laps_p3.csv

# Stint data for all three sessions
curl "https://api.openf1.org/v1/stints?session_key=10007&csv=true" -o data/stints_p1.csv
curl "https://api.openf1.org/v1/stints?session_key=10008&csv=true" -o data/stints_p2.csv
curl "https://api.openf1.org/v1/stints?session_key=10009&csv=true" -o data/stints_p3.csv

# Weather data for the whole meeting (process_weather.py keeps the practice sessions)
curl "https://api.openf1.org/v1/weather?meeting_key=1257&csv=true" -o data/weather.csv

# Car data for the Ferrari cars (cleaned and aggregated per lap, not a model input yet)
# In FP1, driver 38 ran Ferrari's #16 car
curl "https://api.openf1.org/v1/car_data?session_key=10007&driver_number=38&csv=true" -o data/car_data_p1_1.csv
curl "https://api.openf1.org/v1/car_data?session_key=10007&driver_number=44&csv=true" -o data/car_data_p1_2.csv
curl "https://api.openf1.org/v1/car_data?session_key=10008&driver_number=16&csv=true" -o data/car_data_p2_1.csv
curl "https://api.openf1.org/v1/car_data?session_key=10008&driver_number=44&csv=true" -o data/car_data_p2_2.csv
curl "https://api.openf1.org/v1/car_data?session_key=10009&driver_number=16&csv=true" -o data/car_data_p3_1.csv
curl "https://api.openf1.org/v1/car_data?session_key=10009&driver_number=44&csv=true" -o data/car_data_p3_2.csv
```

Laps and stints cover the whole field (~20 drivers per session), and the model is trained on all of them.

### Dataset Descriptions

#### **Laps Dataset**
Key features for lap-level telemetry:
- `lap_duration`: Total lap time (primary target variable)
- `duration_sector_1/2/3`: Individual sector times
- `i1_speed`, `i2_speed`, `st_speed`: Speed measurements at key points
- `lap_number`: Sequential lap index
- `is_pit_out_lap`: Flag for pit exit laps (excluded from analysis)

#### **Stints Dataset**
Tyre compound and usage information:
- `compound`: Tyre type (Soft/Medium/Hard/Inter/Wet)
- `lap_start`, `lap_end`: Stint boundaries
- `stint_number`: Sequential stint identifier
- `tyre_age_at_start`: Previous laps on this tyre set

#### **Weather Dataset**
Environmental conditions:
- `air_temperature`, `track_temperature`: Critical for tyre performance
- `humidity`, `pressure`: Atmospheric conditions
- `wind_speed`, `wind_direction`: Aerodynamic factors
- `rainfall`: Wet vs dry condition indicator

---

## 🔧 Data Preparation

### Cleaning Scripts
- `process_laps.py`: Removes outliers and invalid laps
- `process_stints.py`: Validates stint metadata
- `tyre_sets.py`: Infers each Ferrari's tyre sets from the practice stints (chained by compound and `tyre_age_at_start`) and writes the sets left for the race
- `process_weather.py`: Filters atmospheric anomalies
- `process_cardata.py`: Drops no-input (brake/throttle = 104) and out-of-range telemetry samples, encodes brake/DRS as 0/1, aggregates the ~4 Hz samples per lap (speed, throttle, braking, rpm, gears, DRS), keeps only laps that survive `process_laps.py`, removes z-score outliers and scales per session. Not a model input yet (see #16 in [diagnostic.md](diagnostic.md))

### Data Cleaning Process

**1. Redundancy Removal**
- Removed session and meeting keys
- Excluded pit-out laps (slow in-laps are caught by the non-push filter)

**2. Outlier Detection & Removal**

*Laps (per session):*
- Missing or non-positive sector times
- Unrealistic lap times (≤ 30s or ≥ 2× session median)
- Aborted laps (`i1_speed`, `i2_speed` or `st_speed` ≤ 50 km/h)
- Rolling median residual filtering: residual = lap time − 5-lap centred rolling median per driver; removed laps outside `[Q1 - 3×IQR, Q3 + 3×IQR]`
- Non-push laps (cool-down, in- and traffic laps): slower than 1.15× the driver's best lap in the session (`PUSH_LAP_MAX_BEST_RATIO`), which leaves 467 laps (diagnostic.md #20)

*Stints:*
- Logical validation: `lap_start ≤ lap_end`
- Non-negative `tyre_age_at_start`
- Non-null compound values

*Weather:*
- Z-score filtering (|z| > 3) on all continuous variables

**3. Encoding**
- `driver_number`: Kept numerical as an identifier (with `session_key`) to build the per-driver next-lap target; not a model input (generalized model)
- `compound`: One-hot encoded → `compound_SOFT`, `compound_MEDIUM`, `compound_HARD`
- `session_name` (FP1/FP2/FP3, from the `laps_p1/p2/p3` file): One-hot encoded → `session_FP1`, `session_FP2`, `session_FP3`, a model input that separates session-level pace offsets; `session_key` stays an identifier
- `segments_sector_1/2/3`: List of mini-sector codes replaced by its length

**4. Normalization**
- **Laps**: Not scaled. Lap times, sector times and speeds stay in seconds and km/h. The models are unaffected by a global rescale, and per-session scaling erased each session's pace level and left predictions in z-scores
- **Weather**: StandardScaler on temperature, humidity, pressure, wind speed, rainfall
- **Wind direction**: Kept raw (circular variable)

**5. Feature Engineering**

| Feature | Formula | Purpose |
|---------|---------|---------|
| `stint_number` | Stints numbered 1, 2, 3… per driver across all sessions | Track tyre degradation per stint |
| weather join | nearest weather sample to each lap's `date_start` (`merge_asof`, 5 min tolerance) | Give each lap the conditions it was driven in |
| `laps_since_stint_start` | `lap_number - lap_start + 1` | Measure tyre age in current stint |
| `tyre_age` | `tyre_age_at_start + laps_since_stint_start` | Real age of the set (laps on it including lap `t`), so used sets are told apart from new ones |
| `session_FP1/2/3` | One-hot of the practice session | Separate session-level pace offsets |
| `stint_avg_pace` | Mean lap time (s) of the stint's earlier laps (NaN on the first lap, filled with the median) | Capture stint-level performance without seeing lap `t+1` |
| `fuel_corrected_pace` | `lap_duration − 0.035 × laps_remaining` in seconds (`laps_remaining = lap_end - lap_number + 1`, laps left in the stint counting this one) | Remove the fuel effect; practice fuel loads aren't recorded, so the stint's remaining laps stand in for fuel on board (assumption, diagnostic.md #19) |

> **Note**: Track evolution was excluded to avoid deriving dependent variables.

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
| Linear Regression | Underfits non-linear tyre behavior | 1.67 s | 2.10 s | 0.157 |
| Random Forest | Noisy, stepwise predictions | 1.13 s | 1.61 s | 0.505 |
| **XGBoost** | Chosen for non-linear learning and smooth predictions | 1.03 s | 1.46 s | 0.590 |

*MAE and RMSE are in seconds, from the current run (31 test rows). The split still shuffles single laps, so laps of one stint are in both train and test, which flatters these numbers (diagnostic.md #18); see [diagnostic.md](diagnostic.md).*

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
- 26 inputs defined in `scripts/feature_config.py` (`MODEL_FEATURE_COLUMNS`), including the `session_FP1/2/3` one-hots
- Missing values filled with column medians; 80/20 shuffled train/test split (`random_state=42`)
- The saved bundle holds the model, the feature list, the training medians and each session's share of training rows (`{"model", "features", "medians", "session_weights"}`)

**4. Validation**
- 5-fold cross-validation
- Residual analysis for systematic bias
- Random Forest and XGBoost feature importances

---

## 🏁 Race Simulation & Optimization

### 1. Strategy Space Generation

Enumerate strategies for a 57-lap race (16,206 in total):
- **Stop configurations**: 1-stop, 2-stop (0 stops can't use two compounds)
- **Compound sequences**: Every SOFT/MEDIUM/HARD sequence that uses at least two different compounds (the dry-race rule)
- **Pit windows**: Pit laps between lap 10 and lap 46
- **Tyre sets**: each Ferrari (`RACE_CARS`: 16, 44) can only fit the sets it has left. `tyre_sets.py` infers them from the practice stints. OpenF1 has no set ID, so a stint that starts used is chained to the same-compound set that was left at that age. Allocation, returns after each session and the reserved race sets are `.env` assumptions. Each set is fitted at most once, and the fastest choice of sets is kept for each plan. Sequences needing more sets than the car has (e.g. two HARDs) are skipped, leaving 12,210 per car. Qualifying usage isn't tracked (diagnostic.md #21).

### 2. Lap Time Prediction

For each compound and set age a car has left, and every stint length L = 1–57, roll the stint forward from stint lap 1. Tyre age resets to the set's age at every stop.
1. **Seed:** lap 1 is a median lap: the training medians, the compound one-hot set, `tyre_age_at_start` = the set's age and `tyre_age` = set age + 1.
2. **Predict:** XGBoost predicts the next lap's fuel-corrected pace from the current lap's inputs.
   - The race is not a practice session, so each lap is predicted under `session_FP1`, `session_FP2` and `session_FP3` and averaged, weighted by each session's share of training rows.
3. **Feed back:** each prediction becomes the next lap's inputs.
   - The training fuel term is added back first: `raw = fcp + 0.035 × (L − age + 1)`.
   - The sector times are rescaled to sum to `raw`.
   - `stint_avg_pace` is the mean of the earlier laps.
   - `laps_since_stint_start` and `tyre_age` go up by one.
4. **Price:** a stint's time is the summed rollout fcp for its length, plus a race fuel penalty of `0.035 s × laps of fuel remaining` for each race lap it covers.

### 3. Objective Function

```
Minimize: Total Race Time (s) = Σ(stint lap times) + 22 s × number of pit stops
```

Current output (set age at the start of each stint in brackets):
- car 16: SOFT(6)–MEDIUM(0), stop on lap 46, 5838.86 s (ties with MEDIUM(0)–SOFT(6), stop on lap 11); best 2-stop 9.44 s slower
- car 44: SOFT(6)–MEDIUM(3), stop on lap 46, 5838.15 s (ties with the reverse, stop on lap 11); best 2-stop 11.21 s slower

Every race lap is summed once: "stop on lap p" means lap p is the in-lap and the next stint starts on lap p + 1, with the 22 s pit loss on top (diagnostic.md #11).

None of this is tyre physics yet. 1 stop wins because every rollout is almost flat (about 101 s from stint lap 3 on, whatever the compound or set age), so an extra stop can't win back its 22 s. Neighbouring pit laps are within about 1 s and the compounds barely differ, so the pit lap and compound choice are close to arbitrary. The model is trained on only 153 rows, and its metrics still come from a shuffled split (#18), so this is not a valid recommendation yet.

### 4. Optimization Engine

- **Method**: Brute force: every strategy is evaluated and the lowest total wins
- **Future enhancement**: Dynamic Programming to prune inferior strategies, Genetic Algorithm for deeper search
- **Output**: `scripts/results/strategy_simulator/` (best strategy JSON, all strategy results CSV)

---

## ✅ Evaluation & Validation

### Model Validation (`validate_strategy.py`)
- **Cross-validation**: 5-fold CV on next-lap predictions (current: MAE 1.19 s, RMSE 1.67 s)
- **Residual analysis**: Check for systematic bias
- **Feature importance**: Verify alignment with tyre physics
  - Expected top features: tyre age, compound, track temperature
- **t-test**: Placeholder on random sample data
- **Simulation stability** (logged by `strategy_simulator.py`): fcp at stint laps 1, 18 and 57 of a full-length rollout for each compound and set age, plus the largest lap-to-lap step.
  - No runaway drift: every rollout stays within 97.15–101.70 s, and the largest lap-to-lap step is 2.2–2.5 s (stint laps 1–3).
  - Smooth but nearly flat: about 99.5 s on stint lap 2 and 101 s on lap 3, then 100.5–101.7 s to lap 57. All compounds and set ages are within 0.3 s, so the model shows almost no tyre degradation yet.

### Strategy Validation *(planned, not implemented yet)*
- Simulate multiple race conditions with variations
- Compare alternative strategies across pit windows
- Generate heatmaps of predicted race time vs strategy grid
- **Stability check**: Ensure optimal strategy persists under small perturbations

---

## 🚀 Getting Started

### Prerequisites
Python 3.10. Dependencies are pinned in `requirements.txt` (UTF-16 encoded):
```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn joblib requests
```

### Usage

**1. Data Collection**: download the CSVs into `data/` with the URLs under [Data Sources](#data-sources).

**2. Run the pipeline.** Scripts use relative paths, so run them from `scripts/`:
```bash
cd scripts
python master.py
```

`master.py` runs these in order and stops at the first failure:

| Step | Script | Hands off |
|------|--------|-----------|
| 1 | `process_laps.py` | `data/laps_combined_clean.csv` |
| 2 | `process_stints.py` | `data/stints_combined_clean.csv` |
| 3 | `tyre_sets.py` | `data/tyre_sets_race_inventory.csv` (each car's sets left for the race) |
| 4 | `process_weather.py` | `data/cleaned_weather.csv` |
| 5 | `process_cardata.py` | `data/cardata_per_lap_clean.csv` (not read by later stages yet) |
| 6 | `feature_engineer.py` | `data/final_engineered_dataset.csv` |
| 7 | `train_model.py` | `scripts/results/train_model/train_model_xgb_model.pkl` |
| 8 | `strategy_simulator.py` | best strategy per car |
| 9 | `validate_strategy.py` | CV metrics, residual plot |

Each script can also be run on its own, and writes its log and outputs to `scripts/results/<script_name>/`.

**Settings** (paths, file names, cleaning thresholds, model hyperparameters, seeds, race length, pit loss, pit window, stop counts) are in [.env](.env) at the repo root, which `scripts/env_config.py` reads. An environment variable with the same name overrides the file for one run, e.g. `$env:PIT_LOSS_S="25"; python strategy_simulator.py` (PowerShell).

Detailed methodology: [readme.txt](readme.txt). Known bugs: [diagnostic.md](diagnostic.md).

---

## 📈 Key Insights

### Physical Considerations Modeled
- Tyre thermal degradation over stint length
- Compound-specific grip vs durability trade-offs
- Track temperature impact on tyre performance
- Fuel load reduction over race distance

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
- The data science community for XGBoost and SHAP tools
- Formula 1 for being the ultimate testing ground for optimization problems

---

*"In racing, they say that your car goes where your eyes go. The driver who cannot tear his eyes away from the wall as he spins out of control will meet that wall; the driver who looks down the track as he struggles to control his car will somehow manage to find his way back to the track."* — Garth Stein
