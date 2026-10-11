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
- `process_weather.py`: Filters atmospheric anomalies
- `process_cardata.py`: Drops no-input (brake/throttle = 104) and out-of-range telemetry samples, encodes brake/DRS as 0/1, aggregates the ~4 Hz samples per lap (speed, throttle, braking, rpm, gears, DRS), keeps only laps that survive `process_laps.py`, removes z-score outliers and scales per session. Not a model input yet (see #16 in [diagnostic.md](diagnostic.md))

### Data Cleaning Process

**1. Redundancy Removal**
- Removed session and meeting keys
- Excluded pit-out laps (slow in-laps are caught by the outlier filters)

**2. Outlier Detection & Removal**

*Laps (per session):*
- Missing or non-positive sector times
- Unrealistic lap times (≤ 30s or ≥ 2× session median)
- Aborted laps (`i1_speed`, `i2_speed` or `st_speed` ≤ 50 km/h)
- Rolling median residual filtering: residual = lap time − 5-lap centred rolling median per driver; removed laps outside `[Q1 - 3×IQR, Q3 + 3×IQR]`

*Stints:*
- Logical validation: `lap_start ≤ lap_end`
- Non-negative `tyre_age_at_start`
- Non-null compound values

*Weather:*
- Z-score filtering (|z| > 3) on all continuous variables

**3. Encoding**
- `driver_number`: Kept numerical as an identifier (with `session_key`) to build the per-driver next-lap target; not a model input (generalized model)
- `compound`: One-hot encoded → `compound_SOFT`, `compound_MEDIUM`, `compound_HARD`
- `segments_sector_1/2/3`: List of mini-sector codes replaced by its length

**4. Normalization**
- **Laps**: StandardScaler (fitted per session) on lap times, sector times, and speeds, so pace values are z-scores, not seconds
- **Weather**: StandardScaler on temperature, humidity, pressure, wind speed, rainfall
- **Wind direction**: Kept raw (circular variable)

**5. Feature Engineering**

| Feature | Formula | Purpose |
|---------|---------|---------|
| `stint_number` | Stints numbered 1, 2, 3… per driver across all sessions | Track tyre degradation per stint |
| weather join | nearest weather sample to each lap's `date_start` (`merge_asof`, 5 min tolerance) | Give each lap the conditions it was driven in |
| `laps_since_stint_start` | `lap_number - lap_start + 1` | Measure tyre age in current stint |
| `stint_avg_pace` | Leave-one-out mean of lap times | Capture stint-level performance |
| `fuel_corrected_pace` | `lap_duration + 0.035 × laps_remaining` (`laps_remaining = lap_end - lap_number`) | Account for fuel load reduction |

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
| Linear Regression | Underfits non-linear tyre behavior | 0.504 | 0.687 | 0.534 |
| Random Forest | Noisy, stepwise predictions | 0.435 | 0.616 | 0.625 |
| **XGBoost** | Chosen for non-linear learning and smooth predictions | 0.419 | 0.594 | 0.652 |

*Metrics are in standardised (z-score) units, from the current run. They are probably inflated, because `stint_avg_pace` (the top feature) includes the target lap (diagnostic.md #17); see [diagnostic.md](diagnostic.md).*

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
Built by `build_next_lap_target()` in `scripts/feature_config.py`: the same driver's lap `lap_number + 1` in the same stint. The last lap of each stint and laps followed by a cleaning gap are dropped (839 → 514 rows).

**2. Removed Dependent Variables**
- `lap_duration` (sum of sector times)
- `laps_remaining` (used in fuel correction formula)

**3. Features and Split**
- 22 inputs defined in `scripts/feature_config.py` (`MODEL_FEATURE_COLUMNS`)
- Missing values filled with column medians; 80/20 shuffled train/test split (`random_state=42`)

**4. Validation**
- 5-fold cross-validation
- Residual analysis for systematic bias
- Random Forest and XGBoost feature importances

---

## 🏁 Race Simulation & Optimization

### 1. Strategy Space Generation

Enumerate strategies for a 57-lap race (18,318 in total):
- **Stop configurations**: 0-stop, 1-stop, 2-stop
- **Compound sequences**: Every SOFT/MEDIUM/HARD sequence (the two-compound rule is not enforced yet)
- **Pit windows**: Pit laps between lap 10 and lap 46

### 2. Lap Time Prediction

For each compound, once:
1. Predict a lap time with XGBoost for every race lap 1–57 (compound one-hot set, `laps_since_stint_start` = race lap, all other features at 0)
2. Add a fuel penalty of `0.035 × laps of fuel remaining`
3. Store cumulative sums so any stint's time is a single subtraction

### 3. Objective Function

```
Minimize: Total Race Time = Σ(stint lap times) + 22 × number of pit stops
```

### 4. Optimization Engine

- **Method**: Brute force: every strategy is evaluated and the lowest total wins
- **Future enhancement**: Dynamic Programming to prune inferior strategies, Genetic Algorithm for deeper search
- **Output**: `scripts/results/strategy_simulator/` (best strategy JSON, all strategy results CSV)

---

## ✅ Evaluation & Validation

### Model Validation (`validate_strategy.py`)
- **Cross-validation**: 5-fold CV on next-lap predictions (current: MAE 0.698, RMSE 0.943, z-score units)
- **Residual analysis**: Check for systematic bias
- **Feature importance**: Verify alignment with tyre physics
  - Expected top features: tyre age, compound, track temperature
- **t-test**: Placeholder on random sample data
- **Simulation stability** *(planned)*: Predictions remain smooth over multi-lap rollouts

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
| 3 | `process_weather.py` | `data/cleaned_weather.csv` |
| 4 | `process_cardata.py` | `data/cardata_per_lap_clean.csv` (not read by later stages yet) |
| 5 | `feature_engineer.py` | `data/final_engineered_dataset.csv` |
| 6 | `train_model.py` | `scripts/results/train_model/train_model_xgb_model.pkl` |
| 7 | `strategy_simulator.py` | best strategy |
| 8 | `validate_strategy.py` | CV metrics, residual plot |

Each script can also be run on its own, and writes its log and outputs to `scripts/results/<script_name>/`.

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
