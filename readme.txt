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
curl "https://api.openf1.org/v1/laps?session_key=10007&csv=true" -o fp1_laps.csv
curl "https://api.openf1.org/v1/laps?session_key=10008&csv=true" -o fp2_laps.csv
curl "https://api.openf1.org/v1/laps?session_key=10009&csv=true" -o fp3_laps.csv

# Stint data for all three sessions
curl "https://api.openf1.org/v1/stints?session_key=10007&csv=true" -o fp1_stints.csv
curl "https://api.openf1.org/v1/stints?session_key=10008&csv=true" -o fp2_stints.csv
curl "https://api.openf1.org/v1/stints?session_key=10009&csv=true" -o fp3_stints.csv

# Weather data for the weekend
curl "https://api.openf1.org/v1/weather?meeting_key=1253&csv=true" -o weather.csv
```

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
- `clean_laps.py`: Removes outliers and invalid laps
- `clean_stints.py`: Validates stint metadata
- `clean_weather.py`: Filters atmospheric anomalies

### Data Cleaning Process

**1. Redundancy Removal**
- Removed session and meeting keys after merging
- Excluded pit-in and pit-out laps (incomplete laps)

**2. Outlier Detection & Removal**

*Laps:*
- Missing or negative sector times
- Unrealistic lap times (< 30s or ≥ 2× median)
- Aborted laps (speed < 50 km/h)
- Rolling median residual filtering: removed laps outside `[Q1 - 3×IQR, Q3 + 3×IQR]`

*Stints:*
- Logical validation: `lap_start ≤ lap_end`
- Non-negative `tyre_age_at_start`
- Non-null compound values

*Weather:*
- Z-score filtering (|z| > 3) on all continuous variables

**3. Encoding**
- `driver_number`: Kept numerical (tree-friendly, later removed for generalized model)
- `compound`: One-hot encoded → `compound_Soft`, `compound_Medium`, `compound_Hard`

**4. Normalization**
- **Laps**: StandardScaler on lap times, sector times, and speeds
- **Weather**: StandardScaler on temperature, humidity, pressure, wind speed, rainfall
- **Wind direction**: Kept raw (circular variable)

**5. Feature Engineering**

| Feature | Formula | Purpose |
|---------|---------|---------|
| `stint_number` | Assigned from stints data | Track tyre degradation per stint |
| `pseudo_time` | `lap_number` (sorted) | Enable time-alignment with weather |
| `laps_since_stint_start` | `lap_number - lap_start + 1` | Measure tyre age in current stint |
| `stint_avg_pace` | Leave-one-out mean of lap times | Capture stint-level performance |
| `fuel_corrected_pace` | `lap_duration + 0.035 × laps_remaining` | Account for fuel load reduction |

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

After evaluating multiple approaches:

| Model | Result |
|-------|--------|
| Linear Regression | Underfits non-linear tyre behavior |
| Random Forest | Noisy, stepwise predictions |
| K-Means Clustering | No predictive capability |
| **XGBoost** | ✅ **Optimal balance of accuracy and stability** |

**XGBoost Advantages:**
- Captures non-linear tyre physics (warm-up, thermal degradation, cliff effects)
- Stable predictions in autoregressive simulations
- Low overfitting risk
- Fast inference
- Interpretable via SHAP values

### Training Process

**1. Target Variable**
```python
y = fuel_corrected_pace_next_lap
```

**2. Removed Dependent Variables**
- `lap_duration` (sum of sector times)
- `laps_remaining` (used in fuel correction formula)

**3. Validation**
- 5-fold cross-validation
- Residual analysis for systematic bias
- SHAP feature importance verification

---

## 🏁 Race Simulation & Optimization

### 1. Strategy Space Generation

Enumerate all feasible strategies:
- **Stop configurations**: 0-stop, 1-stop, 2-stop, 3-stop
- **Compound sequences**: All valid combinations
- **Pit windows**: Excluding formation laps and extreme early/late stops

### 2. Lap-by-Lap Simulation

For each candidate strategy:
1. Predict next-lap fuel-corrected pace using XGBoost
2. Apply fuel penalty (decreases over race distance)
3. Update state variables:
   - `laps_since_stint_start`
   - `compound`
   - `tyre_age`
4. Accumulate lap times

### 3. Objective Function

```
Minimize: Total Race Time = Σ(lap_times) + pit_time_penalties
```

### 4. Optimization Engine

- **Primary method**: Dynamic Programming (prune inferior strategies)
- **Future enhancement**: Genetic Algorithm for deeper search
- **Output**: Optimal compound sequence + pit lap windows

---

## ✅ Evaluation & Validation

### Model Validation
- **Cross-validation**: 5-fold CV on next-lap predictions
- **Residual analysis**: Check for systematic bias
- **Feature importance**: Verify alignment with tyre physics
  - Expected top features: tyre age, compound, track temperature
- **Simulation stability**: Predictions remain smooth over multi-lap rollouts

### Strategy Validation
- Simulate multiple race conditions with variations
- Compare alternative strategies across pit windows
- Generate heatmaps of predicted race time vs strategy grid
- **Stability check**: Ensure optimal strategy persists under small perturbations

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

### Usage

**1. Data Collection**
```bash
python collect_data.py --country Bahrain --year 2025
```

**2. Data Cleaning**
```bash
python clean_laps.py
python clean_stints.py
python clean_weather.py
```

**3. Feature Engineering**
```bash
python engineer_features.py
```

**4. Model Training**
```bash
python train_model.py
```

**5. Strategy Optimization**
```bash
python optimize_strategy.py --race_distance 57
```

---

## 📈 Key Insights

### Physical Considerations Modeled
- Tyre thermal degradation over stint length
- Compound-specific grip vs durability trade-offs
- Track temperature impact on tyre performance
- Fuel load reduction over race distance
- Track evolution across practice sessions

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
