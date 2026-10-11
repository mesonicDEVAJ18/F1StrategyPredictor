F1 RACE STRATEGY OPTIMISER PIPELINE
Requirement for Data Science CSE558 at Indraprastha Institute of Information Technology Delhi

PROBLEM STATEMENT
Use data collected over 3 practice sessions in a non sprint Formula 1 weekend to predict the optimal tyre strategy for a race where atleast
two compounds are supposed to be used (out of 3, hard, medium and soft). 
The problem statement limits us to the three practice sessions only, we can broaden our scope by going out of this limitation but for this project we wont.

DATA COLLECTION
This analysis has been done for Ferrari in Bahrain 2025, same pipeline is applicable across weekends.
Note: the laps and stints data cover the whole field (~20 drivers per session), and the model is trained on all of them.
Only the car data is Ferrari specific (drivers 7, 16 and 44, see step 5).
All data has been collected from https://openf1.org/ and is stored in data/ (laps_p1-3.csv, stints_p1-3.csv, weather.csv, car_data_p*_*.csv).

1. Identify the 3 practice sessions ids. https://api.openf1.org/v1/sessions?country_name=Bahrain&year=2025 
1	10007	Practice 1	Practice	2025-04-11 11:30	2025-04-11 12:30	1 hr	Friday FP1
2	10008	Practice 2	Practice	2025-04-11 15:00	2025-04-11 16:00	1 hr	Friday FP2
3	10009	Practice 3	Practice	2025-04-12 12:30	2025-04-12 13:30	1 hr	Saturday FP3

2. Download all laps' data for the sessions 
https://api.openf1.org/v1/laps?session_key=10007&csv=true
https://api.openf1.org/v1/laps?session_key=10008&csv=true
https://api.openf1.org/v1/laps?session_key=10009&csv=true

3. Download all stints's data for the sessions
https://api.openf1.org/v1/stints?session_key=10007&csv=true
https://api.openf1.org/v1/stints?session_key=10008&csv=true
https://api.openf1.org/v1/stints?session_key=10009&csv=true

4. Download the weekend's weather related data
https://api.openf1.org/v1/weather?meeting_key=1253&csv=true
KNOWN ISSUE: meeting_key 1253 is Bahrain pre-season testing (sessions 9683-9685, 26-28 Feb 2025), not the Grand Prix weekend.
The laps, stints and car data are from meeting_key 1257 (11-12 Apr 2025). The weather for the race weekend would be
https://api.openf1.org/v1/weather?meeting_key=1257&csv=true

5. Download the car data for the sessions
https://api.openf1.org/v1/car_data?session_key=10007&driver_number=7&csv=true
https://api.openf1.org/v1/car_data?session_key=10007&driver_number=44&csv=true
https://api.openf1.org/v1/car_data?session_key=10008&driver_number=16&csv=true
https://api.openf1.org/v1/car_data?session_key=10008&driver_number=44&csv=true
https://api.openf1.org/v1/car_data?session_key=10009&driver_number=16&csv=true
https://api.openf1.org/v1/car_data?session_key=10009&driver_number=44&csv=true
Saved as car_data_p<session>_<n>.csv (FP1: drivers 7 and 44, FP2 and FP3: drivers 16 and 44). Car data is not used by the pipeline yet.

DATA UNDERSTANDING
Laps
| **Feature**           | **Meaning / Real-World Interpretation**                  | **Type**                   | **Example / Units**     | **Analytical Role**                                                                 |
| --------------------- | -------------------------------------------------------- | -------------------------- | ----------------------- | ----------------------------------------------------------------------------------- |
| **date_start**        | UTC timestamp when the lap began.                        | *Datetime*                 | “2024-03-01 T14:23:15Z” | Enables ordering, track-evolution analysis, and time-based joins with weather data. |
| **driver_number**     | Same as above — driver identifier.                       | *Categorical / identifier* | 16 = Leclerc            | Merge key across tables.                                                            |
| **duration_sector_1** | Time to complete Sector 1.                               | *Numeric (float)*          | 29.374 s                | Used for localized performance analysis (track-segment performance).                |
| **duration_sector_2** | Time to complete Sector 2.                               | *Numeric (float)*          | 38.452 s                | Used for lap-time composition.                                                      |
| **duration_sector_3** | Time to complete Sector 3.                               | *Numeric (float)*          | 26.928 s                | Combined with others to get total lap time.                                         |
| **i1_speed**          | Instantaneous speed at the first intermediate line.      | *Numeric (float)*          | 305 km/h                | Feature for pace modeling; indicates straight-line speed and DRS use.               |
| **i2_speed**          | Instantaneous speed at second intermediate.              | *Numeric (float)*          | 280 km/h                | Captures mid-sector speed differences.                                              |
| **is_pit_out_lap**    | Boolean flag: 1 = lap starts from pit exit.              | *Boolean / binary*         | True / False            | Helps exclude outlaps from analysis.                                                |
| **lap_duration**      | Total lap time (sector1 + 2 + 3) — main target variable. | *Numeric (float)*          | 92.841 s                | Dependent variable for degradation & ML regression models.                          |
| **lap_number**        | Sequential lap index within the session.                 | *Numeric (int)*            | 1 → 60                  | Enables stint segmentation and degradation trends.                                  |
| **meeting_key**       | Same race-weekend identifier.                            | *Categorical / identifier* | 9161                    | Used for multi-GP aggregation.                                                      |
| **segments_sector_1** | Sub-split timings or micro-segments in Sector 1.         | *Array / list (float)*     | [10.1, 9.9, 9.8]        | Optional fine-grained telemetry for advanced modeling.                              |
| **segments_sector_2** | Same for Sector 2.                                       | *Array / list (float)*     | …                       | Optional.                                                                           |
| **segments_sector_3** | Same for Sector 3.                                       | *Array / list (float)*     | …                       | Optional.                                                                           |
| **session_key**       | Session identifier (FP1, FP2, FP3).                      | *Categorical / identifier* | 12345                   | Used to merge with `/stints`.                                                       |
| **st_speed**          | Speed at the start/finish line.                          | *Numeric (float)*          | 310 km/h                | Proxy for DRS usage and straight-line performance.                                  |

Stints
| **Feature**           | **Meaning / Real-World Interpretation**                                                               | **Type**                   | **Example / Units** | **Analytical Role**                                              |
| --------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------- | ------------------- | ---------------------------------------------------------------- |
| **compound**          | Tyre compound used in a stint (Soft, Medium, Hard, Inter, Wet). Determines grip and degradation rate. | *Categorical (nominal)*    | “Soft”              | Key explanatory variable for pace and degradation models.        |
| **driver_number**     | Unique car/driver code assigned during a Grand Prix weekend (e.g., 63 = Russell).                     | *Categorical / identifier* | 44, 1, 16           | Join key across all endpoints; used to merge lap and stint data. |
| **lap_start**         | First lap number where this tyre set was used.                                                        | *Numeric (int)*            | 1 → 57              | Defines stint boundaries for degradation calculations.           |
| **lap_end**           | Last lap number before a pit stop or tyre change.                                                     | *Numeric (int)*            | 18 → 32             | Determines stint length = `lap_end − lap_start + 1`.             |
| **meeting_key**       | Unique race-weekend identifier (e.g., “2023-BahrainGP”).                                              | *Categorical / identifier* | 9161                | Used to group sessions by Grand Prix.                            |
| **session_key**       | Unique identifier for a session (FP1, FP2, FP3).                                                      | *Categorical / identifier* | 12345               | Enables merging across multiple FP sessions.                     |
| **stint_number**      | Sequential number of the stint for a given driver in that session (1, 2, 3 …).                        | *Numeric (int)*            | 1, 2, 3             | Helps analyze pace evolution per stint.                          |
| **tyre_age_at_start** | Tyre age (laps already completed on this set) when the stint began — reused tyres start > 0.          | *Numeric (float)*          | 0 – 25 laps         | Used to adjust degradation modeling (fresh vs used sets).        |

Weather
| **Feature**           | **Meaning / Real-World Interpretation**                                                                                            | **Type**                   | **Example / Units**        | **Analytical Role**                                                                   |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | -------------------------- | ------------------------------------------------------------------------------------- |
| **air_temperature**   | The ambient temperature of the air surrounding the track at the recorded time. Affects engine cooling and aerodynamic performance. | *Numeric (float)*          | 27.3 °C                    | Used to assess how ambient conditions influence lap pace and degradation.             |
| **date**              | UTC timestamp when the weather sample was recorded.                                                                                | *Datetime*                 | `2024-03-01T14:22:00Z`     | Allows time-based alignment with lap start times from `/laps`.                        |
| **humidity**          | Relative humidity of the air. Higher humidity can affect tyre and engine efficiency.                                               | *Numeric (float)*          | 62 %                       | Used in correlation analysis and regression as an explanatory environmental variable. |
| **meeting_key**       | Unique race-weekend identifier (same as in `/laps` and `/stints`).                                                                 | *Categorical / Identifier* | 9161                       | Enables merging of weather data with lap and stint data by race.                      |
| **pressure**          | Atmospheric pressure measured at the circuit.                                                                                      | *Numeric (float)*          | 1012 mbar                  | Optional feature; can indicate altitude and air density effects.                      |
| **rainfall**          | Measured precipitation during the sample interval.                                                                                 | *Numeric (float)*          | 0 mm/h                     | Critical for excluding wet laps or performing wet-vs-dry comparisons.                 |
| **session_key**       | Session identifier (FP1, FP2, FP3, etc.) — consistent with `/laps` and `/stints`.                                                  | *Categorical / Identifier* | 12345                      | Used to filter data for specific practice sessions.                                   |
| **track_temperature** | Temperature of the asphalt surface — directly affects tyre grip and degradation.                                                   | *Numeric (float)*          | 41.7 °C                    | One of the most important predictors for tyre performance.                            |
| **wind_direction**    | Direction from which wind is blowing, usually measured in degrees (0–360).                                                         | *Numeric (float)*          | 250° (wind from WSW)       | Used for advanced aerodynamic or circuit-specific modeling.                           |
| **wind_speed**        | Wind velocity at the track, often measured at a fixed reference point.                                                             | *Numeric (float)*          | 3.6 m/s                    | Can influence drag and straight-line speed variations.                                |

Car Data
| **Feature**       | **Meaning / Real-World Interpretation**                                                                | **Type**                   | **Example / Units**    | **Analytical Role**                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------ | -------------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| **brake**         | Indicates whether the brake pedal is engaged by the driver. `100` means braking, `0` means no braking. | *Binary / Numeric (int)*   | 0 or 100               | Used to analyze braking zones, driver behavior, and deceleration patterns.         |
| **date**          | UTC timestamp when the telemetry data sample was recorded.                                             | *Datetime*                 | `2024-03-01T14:22:05Z` | Enables synchronization with lap, weather, or position data over time.             |
| **driver_number** | The permanent racing number assigned to a Formula 1 driver for the season.                             | *Categorical / Identifier* | 1, 16, 44              | Used to group or filter telemetry data by specific drivers.                        |
| **drs**           | Status of the Drag Reduction System (DRS), indicating whether the rear wing flap is open or closed.    | *Categorical / Numeric*    | 0, 1, 8, 10, 12        | Helps analyze overtaking zones, straight-line speed advantages, and race strategy. |
| **meeting_key**   | Unique identifier for a race weekend (Grand Prix event).                                               | *Categorical / Identifier* | 9161                   | Allows merging telemetry with laps, weather, and stint data for the same event.    |
| **n_gear**        | The currently selected gear in the car’s transmission. `0` represents neutral or no gear engaged.      | *Numeric (int)*            | 0–8                    | Useful for analyzing acceleration patterns, gear shifts, and driver control.       |
| **rpm**           | Engine speed measured as revolutions per minute. Indicates how fast the engine crankshaft is rotating. | *Numeric (int)*            | 11500 rpm              | Used to study engine performance, power delivery, and gear efficiency.             |
| **session_key**   | Unique identifier for the track session (FP1, FP2, FP3, Qualifying, etc.).                             | *Categorical / Identifier* | 12345                  | Enables filtering telemetry for a specific session during a race weekend.          |
| **speed**         | Instantaneous velocity of the car measured by telemetry sensors.                                       | *Numeric (float)*          | 298 km/h               | Critical metric for performance analysis, sector speed comparison, and drag study. |
| **throttle**      | Percentage of maximum engine power being applied by the driver through the throttle pedal.             | *Numeric (float)*          | 0–100 %                | Used to analyze acceleration behavior, power usage, and driver input patterns.     |


Relationship Mapping between these 4 raw tables
(These are the natural join keys. What the pipeline actually implements is described under DATA PREPARATION -> Feature Engineering.)
| **Relationship**                     | **Common Keys**                                                                                             | **Description**                                                                            | **Type**                                        | **Usage**                                                                                       |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Laps ↔ Stints**                    | `meeting_key`, `session_key`, `driver_number`, and lap overlap (`lap_number ∈ [lap_start, lap_end]`)        | Links each lap to the tyre compound, stint number, and tyre age used during that lap.      | **One-to-Many** (one stint → many laps)         | Adds tyre compound and degradation information to lap-level performance analysis.               |
| **Laps ↔ Weather**                   | `meeting_key`, `session_key`, and nearest timestamp (`date_start ≈ date`)                                   | Associates each lap with environmental conditions at the time the lap started.             | **Many-to-One (time-based join)**               | Adds air/track temperature, humidity, wind, etc., to lap-level performance modeling.            |
| **Stints ↔ Weather**                 | `meeting_key`, `session_key`, and midpoint timestamp of stint (`mid_stint_time ≈ date`)                     | Connects tyre stints to the weather conditions experienced during the stint duration.      | **Many-to-One (aggregate join)**                | Used to analyze how weather affects tyre degradation and stint strategy.                        |
| **Laps ↔ Car Data**                  | `meeting_key`, `session_key`, `driver_number`, and timestamp alignment (`date_start ≤ date ≤ lap_end_time`) | Links lap records with high-frequency telemetry such as speed, throttle, braking, and RPM. | **One-to-Many** (one lap → many telemetry rows) | Enables detailed performance analysis like braking zones, acceleration patterns, and DRS usage. |
| **Stints ↔ Car Data**                | `meeting_key`, `session_key`, `driver_number`, and timestamp range within stint duration                    | Maps telemetry data to the tyre stint in which it occurred.                                | **One-to-Many**                                 | Allows analysis of driving style and power usage across different tyre compounds.               |
| **Car Data ↔ Weather**               | `meeting_key`, `session_key`, and nearest timestamp (`date ≈ date`)                                         | Aligns telemetry samples with the weather conditions at that moment.                       | **Many-to-One**                                 | Enables studies of how environmental factors affect speed, throttle usage, and engine RPM.      |
| **Shared Identifiers (Global Keys)** | `meeting_key`, `session_key`, `driver_number`                                                               | Common identifiers present across all datasets ensuring consistent merging.                | —                                               | Allows creation of a unified dataset combining laps, stints, weather, and telemetry data.       |


RUNNING THE PIPELINE
Python 3.10, dependencies pinned in requirements.txt (that file is UTF-16 encoded).
All scripts live in scripts/ and use relative paths, so run them from inside scripts/:

    cd scripts
    python master.py

master.py runs process_laps.py -> process_stints.py -> process_weather.py -> feature_engineer.py -> train_model.py
-> strategy_simulator.py -> validate_strategy.py, and stops at the first script that fails.
Every script writes a log, a manifest and its intermediate outputs to scripts/results/<script_name>/.
Stages hand data to each other through CSVs in data/:
    process_laps.py      -> data/laps_combined_clean.csv
    process_stints.py    -> data/stints_combined_clean.csv
    process_weather.py   -> data/cleaned_weather.csv
    feature_engineer.py  -> data/final_engineered_dataset.csv
    train_model.py       -> scripts/results/train_model/train_model_xgb_model.pkl (read by the simulator and validation)
process_cardata.py is work in progress and is not part of master.py.

DATA PREPARATION
Scripts: process_laps.py, process_stints.py, process_weather.py (process_cardata.py is work in progress, see below)
1. Remove redundant data from all of the data types
These scripts remove session keys and meeting keys, data points irrelevant to the pipeline.
Also process_laps.py removes pit-out laps (is_pit_out_lap = True), i.e. laps that start from the pit exit. There is no explicit in-lap flag in the
data; slow in-laps are caught by the outlier filters below.

2. Remove outliers
Laps (each session file is cleaned separately)
Removed laps with missing sector times and sector times of 0 or below. Also removed laps with unrealistic lap times (duration <= 30s or >= 2x the median lap time of that session).
Also removed laps where i1_speed, i2_speed or st_speed is 50 km/h or less, indicating aborted laps (due to whatever reason).
Removed laps whose residuals fall outside [Q1 - 3xIQR, Q3 + 3xIQR], where residual = lap_duration - (5-lap centred rolling median of that driver's lap_duration). A rolling median compensates for the track improving across a single session.

Stints
No continuous telemetry available, prompting use of logical checks only, removed rows where lap_start > lap_end, negative tyre_age_at_start, and missing compound.

Weather
Applied Z-score filtering (|z| > 3) on continuous atmospheric readings: Air temperature, Track temperature, Humidity, Pressure, Wind speed and Rainfall.
Columns with zero variance are skipped.

CarData (work in progress)
process_cardata.py is not functional yet: it is a copy of process_laps.py, it looks for cardata_p*_*.csv while the files are named car_data_p*_*.csv,
and it filters on lap columns (sector times, i1/i2/st speed) that car data does not have.
Planned: drop samples where brake or throttle read 104, which indicates no valid input (e.g. the car sitting in the garage), then aggregate the
~4 Hz samples per lap by matching timestamps to each lap's date_start.

3. Encoding
Laps
segments_sector_1/2/3 arrive as lists of mini-sector status codes; each list is replaced by its length (the number of mini-sectors).
Driver number was kept numerical as it's friendly to tree based models (decision to use trees explained later) and was used to combine with stints data.
Driver number was removed altogether in feature_engineer.py as it was decided to not build a driver specific model.

Stints
One-hot encoded compound -> compound_HARD, compound_MEDIUM, compound_SOFT (only dry compounds occur in this weekend's data).

Weather
None needed.

4. Normalization
Laps
Applied StandardScaler, fitted separately per session, to lap_duration, duration_sector_1/2/3, i1_speed, i2_speed and st_speed. This ensures all physical signals are on comparable scales.
Because lap_duration is scaled, fuel_corrected_pace and every model prediction are in standardised units (z-scores), not seconds.

Stints
No normalization applied, stints contain mostly integer metadata (stint number, lap ranges).

Weather
Applied StandardScaler to: Air temperature, Track temperature, Humidity, Pressure, Wind speed, and Rainfall.
Wind direction kept raw due to its circular nature.

5. Feature Engineering (feature_engineer.py)
Stint number: stints are sorted by driver and lap_start and numbered 1, 2, 3 ... per driver across all three sessions (global_stint_number), kept session independent to include evolution information too.
Each lap is assigned the global number of the stint whose [lap_start, lap_end] contains its lap_number.

Stint metadata (lap_start, lap_end, tyre_age_at_start, compound) is then merged onto the laps on driver_number + stint_number.

Pseudo time alignment: the laps do carry a date_start timestamp, but the weather file comes from a different event (see diagnostic.md), so
timestamps cannot be matched. Instead a pseudo_time axis is constructed: pseudo_time = lap_number for laps, and the weather rows (in file order)
are spread evenly across the same lap_number range. A nearest-neighbour merge_asof then assigns each lap the closest weather observation.

laps_since_stint_start = lap_number - lap_start + 1 (clipped to at least 1)

stint_avg_pace = mean(lap_duration of all other laps for that driver & stint, excluding the current lap) (Leave-One-Out average -> no target leakage)

fuel_corrected_pace: laps_remaining = lap_end - lap_number (laps left in the stint) and fuel_corrected_pace = lap_duration + k*laps_remaining (k = 0.035 per lap)
The current lap's fuel_corrected_pace is not a model input; the NEXT lap's value is the model's target.

Finally driver_number, date_start, date, lap_duration, pseudo_time and laps_remaining are dropped, and the result is saved as data/final_engineered_dataset.csv.

track evolution was not chosen as an engineered parameter, as it would've been a derived dependant value that would not have contributed to the model.

EXPLORATORY DATA ANALYSIS AND STATISTICAL Inference
Not implemented in the pipeline yet.

ML MODELLING TO COMPUTE DEGRADATION RATE (train_model.py)
1. Remove dependant variables to avoid leakage
lap_duration = duration_sector_1 + duration_sector_2 + duration_sector_3, and was therefore removed as it is clearly a dependant variable.
The sector durations of lap t are kept as inputs, since the target is lap t+1.
The segments_sector_* counts were kept as they may have something to contribute in the model.

laps_remaining = lap_end - lap_number, used to compute/approximate fuel_corrected_pace, not expected for the model.

The 22 model inputs are defined once in scripts/feature_config.py (MODEL_FEATURE_COLUMNS):
sector durations (3), i1/i2/st speed (3), segment counts (3), tyre_age_at_start, compound one-hots (3), weather (7),
laps_since_stint_start and stint_avg_pace.

2. What are we predicting?
Instead of explicitly computing a "degradation rate" label (which becomes circular and leak-prone), the model is trained to predict:
fuel_corrected_pace on the NEXT lap

Fully causal:

All inputs belong to lap t
The model learns how tyre, weather, stint age, and track conditions evolve into lap t+1
No sneaky future-lap info (no target leakage)
Designed for autoregressive race simulation (rolling predictions forward lap by lap)
Degradation is then obtained as an emergent property:

As XGBoost repeatedly predicts future fcp values, the falling trend in predicted pace is degradation.
No slope assumptions.
No fixed compound multipliers.
Just learned tyre physics from FP laps.

Training: the target is built by shifting fuel_corrected_pace back one lap within each stint_number group; rows without a next-lap target are dropped
(1126 -> 613 rows). Missing feature values are filled with the column median, then an 80/20 shuffled train/test split (random_state 42) is used.

3. Model choice
train_model.py fits and compares Linear Regression, Random Forest (200 trees) and XGBoost (300 trees, learning rate 0.05, max depth 7,
subsample 0.8, colsample 0.8), and saves the XGBoost model with its feature list as train_model_xgb_model.pkl.
K-Means was ruled out because it provides no predictive capability (only grouping), so it is not part of the pipeline.

Reasoning:
Linear Regression underfits non-linear tyre behaviour (tyre warm-up phase, thermal degradation, cliffing, etc.).
Random Forest produces noisy, stepwise predictions -> unstable in autoregressive simulations.
XGBoost hits the sweet spot ->
non-linear learning, stable rollouts, fast inference, and direct interpretability through feature importance.

Held-out test results on the current data (standardised units):
| Model             | MAE   | RMSE  | R2    |
| LinearRegression  | 0.784 | 0.961 | 0.306 |
| RandomForest      | 0.779 | 1.011 | 0.232 |
| XGBoost           | 0.817 | 1.094 | 0.102 |
On this data XGBoost scores lowest of the three on the test split; it is kept for the reasons above.

This choice also aligns with how real-world motorsport predictive pipelines are structured:
"predict the next lap's corrected pace and let degradation emerge naturally from repeated predictions."

RACE SIMULATION AND OPTIMISATION (strategy_simulator.py)
1. Generate the strategy space
A 57-lap race is assumed. Strategies are enumerated by brute force over:
0-stop, 1-stop and 2-stop structures
Every compound sequence for that structure (any of SOFT/MEDIUM/HARD per stint, repeats allowed)
Every combination of pit laps between lap 10 and lap 46
This gives 18,318 candidate strategies. The rule that at least two compounds must be used is not enforced yet.

2. Lap time prediction
For each compound, the XGBoost model predicts a lap time for every race lap 1-57:
the compound one-hot is set, laps_since_stint_start is set to the race lap number, and all other features are held at 0 (the average,
since they are standardised).
A fuel penalty of 0.035 x (laps of fuel remaining) is added to each lap.
These per-lap predictions are stored as cumulative sums, so any stint's time is a single subtraction.

3. Objective function: minimize total race time
For each candidate strategy, total race time = sum of predicted lap times of each stint + 22 per pit stop.
This gives a single scalar number for each strategy, allowing direct comparison.

4. Strategy optimisation
All 18,318 strategies are evaluated and the one with the lowest predicted race time is selected.
Results: scripts/results/strategy_simulator/ (all_strategy_results.csv, best_strategy.json, final_summary.txt).
Current output: HARD, no pit stop, total 95.97 (not seconds, see diagnostic.md).

Future work: Dynamic Programming to prune inferior strategies and an optional Genetic Algorithm for deeper search across larger sequences.

EVALUATION AND VALIDATION (validate_strategy.py)
1. Model Validation
Implemented:
5-fold shuffled cross-validation (random_state 42) of the XGBoost configuration on next-lap fuel-corrected pace.
Current result: CV MAE 0.827, CV RMSE 1.093 (standardised units).
Residual plot (predicted vs residual) to check for systematic bias.
Feature importance from Random Forest and XGBoost, saved by train_model.py.
An example two-sample t-test on randomly generated data. This is a placeholder: it does not test anything about the model or the
strategy, and its p-value changes from run to run.

Planned critical checks:
Predictions remain smooth when rolled forward for multiple laps
No drift or instability during stint simulation
Feature influence aligns with expected tyre-degradation behaviour (e.g. tyre age, compound, track temperature matter most)

2. Strategy Validation (planned, not implemented yet)

Once the model is verified, strategy validation ensures the chosen strategy actually makes sense.
This includes:
Running multiple simulated race conditions with small variations
Comparing alternative strategies across pit windows and tyre sequences
Inspecting heatmaps of predicted total race time across the strategy grid
The purpose here is not statistical inference, but checking consistency:
The recommended strategy remains optimal across plausible variations
No alternative strategy becomes better under small perturbations
The solution is stable and not sensitive to minor model noise

KNOWN ISSUES
Known bugs and their effect on the results are documented in diagnostic.md.

THANK YOU
DEVAJ RATHORE
NEWLY SELF APPOINTED ML EXPERT
