# F1 Race Strategy Optimiser

> Predicting the optimal tyre strategy for a Formula 1 race from the three practice sessions.

**Data Science Project (CSE558)**  
Indraprastha Institute of Information Technology Delhi  
*Author: Devaj Rathore*

---

## Problem Statement

Use data collected over the 3 practice sessions (FP1, FP2, FP3) of a non sprint Formula 1 weekend to predict the optimal tyre strategy for the race, where at least two of the three dry compounds (soft, medium, hard) have to be used.

The problem statement limits us to the practice sessions, and for this project we stick to that: no qualifying, no race data, and no other weekends' practice data in the model.

The reference weekend is **Bahrain 2025**, and the same pipeline runs on any non-sprint 2024 or 2025 weekend. The model is trained on the whole field, and the race is simulated for one car (`SIM_DRIVER`, default 16, Ferrari) using that car's own tyre sets. Leave `SIM_DRIVER` blank to simulate on fresh tyres (new sets of every compound, no limit on their number).

The full methodology, with every cleaning rule, number and decision, is in [readme.txt](readme.txt). Known issues and open work are in [diagnostic.md](diagnostic.md).

---

## How It Works

**1. Data.** Everything comes from the free [OpenF1 API](https://openf1.org/): laps, stints, weather, entry lists and car telemetry for FP1-FP3. `race_weekend.py` fetches the weekend set in `.env` and keeps every weekend it has fetched in `data/cache/`, so switching back needs no requests.

**2. Cleaning.** Pit-out laps, invalid sectors, aborted laps, residual outliers and non-push laps (cool-down, in- and traffic laps) are removed. The two IQR fences and the two push-lap ratios were tuned over 12 weekends by `explore/tune_iqr.py` and `explore/tune_push.py`. For Bahrain 2025, 1,257 raw laps become 566 clean ones.

**3. Fall-off model.** Practice pace mixes drivers, sessions, fuel plans and quali sims vs long runs, and between stints those are much bigger than the tyre effect. So each lap's fuel-corrected pace is split into a level for its stint and a fall-off with tyre age:

```
fuel_corrected_pace(lap) = level[stint] + fall_off(compound, tyre_age, conditions)
```

- **Stage 1**: a level per stint and a straight-line slope per compound (also the linear baseline)
- **Stage 2**: the model learns the fall-off from what is left (the compound, tyre age and weather only, never the lap's own timing, so nothing leaks)
- **Stage 3**: the gap between compounds on a new set, fitted from each stint's stage-1 level, with no model involved

The model is plugged in last and nothing upstream depends on it. Candidates are compared in `explore/explore_master.py`.

**4. Simulator.** Every legal 1- and 2-stop strategy (two compounds or more, pit window laps 10-46) is priced stint by stint from the fall-off curve of the set the car actually has, plus race fuel and 22 s per stop, and the fastest total wins.

---

## Current Results (Bahrain 2025)

**Model comparison** (`explore/`, run 2026-11-05, 427 laps in 94 multi-lap stints, within-stint shape MAE over 10 outer folds):

| Model (best feature set) | Shape MAE (s) | Curve |
|---|---|---|
| baseline: flat (no fall-off) | 1.035 ± 0.116 | - |
| baseline: linear per compound | 0.905 ± 0.111 | S 0.34, M 0.23, H 0.21 s/lap, rises without limit past age 19 |
| SVR (6 inputs + set age) | 0.877 ± 0.119 | best fit, but gets faster with age on half the steps |
| **MLP (6 inputs + set age)** | 0.888 ± 0.113 | recommended: S 0.18, M 0.17, H 0.17 s/lap, smooth |
| Ridge (tyre only) | 0.890 ± 0.118 | one 0.255 s/lap slope for every compound, most stable |
| ExtraTrees / XGBoost (tuned) / Random Forest | 0.909 / 0.926 / 0.928 | steps of 0.8-1.2 s |
| XGBoost as in `train_model.py` | 1.008-1.060 | the worst, overfits |

Fall-off is real (every model beats no fall-off), but the data hardly separates the better models: the top ones are within 0.03 s against a standard error of ~0.04 s. The shortlist is the MLP, Ridge and the per-compound line, and the choice is still open.

**What the data says** (`eda.py`, tested on stints, never single laps): SOFT falls off +0.34 s/lap and MEDIUM +0.23 s/lap (both significant), HARD +0.21 s/lap (not significant). On a new set MEDIUM is 1.6 s and HARD 2.2 s slower than SOFT for the same driver and session, and FP2 is 2.8 s faster than FP1.

**Strategy for car 16:** SOFT(0)-SOFT(0)-MEDIUM(0), stops on laps 14 and 28, 5625.24 s (set age in brackets). The best 1-stop is 7.53 s slower.

**This is not a valid recommendation yet.** The pipeline still runs XGBoost as a placeholder, which beats neither baseline in validation (shape MAE 1.05 s vs flat 1.03 s and linear 0.90 s). The compound gap is unstable (MEDIUM +1.16 s, HARD +0.30 s on the training split, moving ~1.4 s across folds), and the 29-lap MEDIUM stint runs past the oldest tyre in training (19 laps). See [diagnostic.md](diagnostic.md) #27.

---

## Getting Started

### Prerequisites
Python 3.10. Dependencies are pinned in `requirements.txt` (the versions that produced the committed results):
```bash
pip install -r requirements.txt
```

### Usage

**1. Choose the weekend**: set `RACE_YEAR` and `RACE_WEEKEND` in [.env](.env) (default 2025 `bahrain`, the names are in `scripts/race_weekend.py`).

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
| 5 | `process_cardata.py` | `data/processed/cardata_per_lap_clean.csv` |
| 6 | `feature_engineer.py` | `data/processed/final_engineered_dataset.csv` |
| 7 | `eda.py` | tables and plots in `scripts/results/eda/` |
| 8 | `train_model.py` | `scripts/results/train_model/train_model_fall_off_model.pkl` |
| 9 | `strategy_simulator.py` | best strategy for `SIM_DRIVER` |
| 10 | `validate_strategy.py` | CV metrics, baseline tests, residual plot |

Each script can also be run on its own, and writes its log and outputs to `scripts/results/<script_name>/`.

**3. Model selection and tuning** (from `explore/`):
```bash
cd explore
python explore_master.py --skip-pipeline   # compare the fall-off models on data/processed/
python tune_iqr.py                         # tune the IQR fences over 12 weekends
python tune_push.py                        # tune the push-lap ratios (after tune_iqr.py, never at the same time)
```
Add `--quick` for a smoke run. A full tune run writes its pick into `.env` itself.

**Settings** (race weekend, paths, cleaning thresholds, model hyperparameters, seeds, pit loss, pit window, stop counts) are in [.env](.env). An environment variable with the same name overrides it for one run, e.g. `$env:PIT_LOSS_S="25"; python strategy_simulator.py` (PowerShell).

`data/raw/`, `data/processed/` and `scripts/results/` are tracked in git, so running another weekend rewrites them. Switching back to 2025 `bahrain` restores the committed data from the cache.

---

## Project Status

The current work is #27: make the pipeline independent of the model, then plug the model in last.

- [x] Step 0: check the workflow
- [x] Step 1: make the pipeline independent of the fall-off model (the compound gap, the tuned thresholds and the race pace level no longer depend on it)
- [ ] Step 2: a better compound-gap estimator (partial pooling, run-type controls, an S ≤ M ≤ H constraint, bootstrap CIs)
- [ ] Step 3: dynamic programming in the simulator, so more stops and every set choice can be searched exactly
- [ ] Step 4: choose the fall-off model (MLP, Ridge or the per-compound line)
- [ ] Step 5: docs and results write-up

Details in [diagnostic.md](diagnostic.md).

---

## Future Enhancements

- [ ] Incorporate qualifying position for overtaking probability
- [ ] Add safety car probability modeling
- [ ] Implement real-time strategy adjustment during races
- [ ] Extend to sprint race weekends
- [ ] Driver-specific modeling for personalized strategies
- [ ] Genetic Algorithm integration for larger strategy spaces

---

## License

This project is developed for academic purposes at IIIT Delhi.

---

## Author

**Devaj Rathore**  
*Newly Self-Appointed ML Expert*  
Data Science CSE558  
Indraprastha Institute of Information Technology Delhi

---

## Acknowledgments

- [OpenF1 API](https://openf1.org/) for comprehensive F1 telemetry data
- The data science community for pandas, scikit-learn and XGBoost
- Formula 1 for being the ultimate testing ground for optimization problems

---

*"In racing, they say that your car goes where your eyes go. The driver who cannot tear his eyes away from the wall as he spins out of control will meet that wall; the driver who looks down the track as he struggles to control his car will somehow manage to find his way back to the track."* — Garth Stein
