F1 RACE STRATEGY OPTIMISER PIPELINE
Requirement for Data Science CSE558 at Indraprastha Institute of Information Technology Delhi

PROBLEM STATEMENT
Use data collected over 3 practice sessions in a non sprint Formula 1 weekend to predict the optimal tyre strategy for a race where atleast
two compounds are supposed to be used (out of 3, hard, medium and soft). 

DATA COLLECTION
This analysis has been done on Bahrain 2025, same pipeline is applicable across weekends.
All data has been collected from https://openf1.org/

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

Relationship Mapping between these 3 raw tables
| Relationship                         | Common Keys                                                                                          | Description                                                                             | Type                                    | Usage                                                                               |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------- |
| **Laps ↔ Stints**                    | `meeting_key`, `session_key`, `driver_number`, and lap overlap (`lap_number ∈ [lap_start, lap_end]`) | Links each lap to the tyre compound, stint number, and tyre age in use during that lap. | **One-to-Many** (one stint → many laps) | To add tyre compound and degradation info to lap-level data.                        |
| **Laps ↔ Weather**                   | `meeting_key`, `session_key`, and nearest timestamp (`date_start ≈ date`)                            | Associates each lap with environmental conditions at its start time.                    | **Many-to-One (approximate join)**      | To add ambient and track temperature, humidity, wind, etc., to lap-level pace data. |
| **Stints ↔ Weather**                 | `meeting_key`, `session_key`, and midpoint timestamp of stint (`mid_stint_time ≈ date`)              | Connects each tyre stint to the average weather conditions during its duration.         | **Many-to-One (aggregate join)**        | To assess how weather influenced tyre wear and stint length.                        |
| **Shared Identifiers (Global Keys)** | `meeting_key`, `session_key`, `driver_number`                                                        | Consistent across all datasets; ensure referential integrity.                           | —                                       | Enables merging of all three datasets into a unified event-level dataset.           |

DATA PREPARATION
Scripts: clean_laps.py, clean_stints.py and clean_weather.py
1. Remove redundant data from all of the data types
These scripts remove session keys and meeeting keys, data points irrelevant to the pipeline.
Also clean_laps.py removes out laps and in laps. (laps that are not complete, involve the driver coming out of the pit or going into the pit are removed.)

2. Remove outliers
Laps
Removed laps with missing sector times and sector times below 0. Also removed laps with unrealsitic lap times (duration < 30s and ≥ 2× the median lap time of that session).
Also removed laps with speed less than 50 km/h indicating aborted laps. (due to whatever reason)
Removed laps whose residuals fall outside [Q1 – 3×IQR, Q3 + 3×IQR] where residuals are a rolling median, as a rolling median would compensate for the track improving across a single session.

Stints
No continous telemetry available, prompting use of logical checks only, removed rows where lap_start > lap_end, negative tyre_age_at_start, and missing compound.

Weather
Applied Z-score filtering (|z| > 3) on continuous atmospheric readings: Air temperature, Track temperature, Humidity, Pressure, Wind speed and Rainfall

3. Encoding
Laps
None needed, driver number was kept numerical as its friendly to tree based models (descision to use trees explained later) and was used to combine with stints data.
Driver number was removed altogether later as it was decided to not build a driver specific model later.

Stints
None needed, driver number was kept numerical as its friendly to tree based models (descision to use trees explained later) and was used to combine with stints data.
One-hot encoded compound → compound_Soft, compound_Medium, compound_Hard

Weather
None needed, driver number was kept numerical as its friendly to tree based models (descision to use trees explained later) and was used to combine with stints data.

4. Normalization
Laps
Applied StandardScaler to continuous telemetry: lap_duration, duration_sector_1/2/3, i1/2/3_speed, this ensures all physical telemetry signals are on comparable scales.

Stints
No normalization applied — stints contain mostly integer metadata (stint number, lap ranges).

Weather
Applied StandardScaler to: Air temperature, Track temperature, Humidity, Pressure, Wind speed, and Rainfall
Wind direction kept raw due to circular nature.

5. Feature Engineering
Stint number, assigned to each lap based on stints data to enable per-stint tyre modeling and degradation analysis, kept session independent to include evolution information too.

Pseudo Time alignment, because laps contain no timestamps, a pseudo_time axis is constructed: Laps sorted by lap_number → pseudo_time = lap_number,A nearest-neighbor merge_asof assigns each lap the closest matching weather observation.
A nearest-neighbor merge_asof assigns each lap the closest matching weather observation

laps_since_stint_start, laps_since_stint_start=lap_number−lap_start+1

stint_avg_pace = mean(lap_duration of all other laps for that driver & stint, excluding the current lap) (Leave-One-Out average → no target leakage)

fuel_corrected_pace, laps_remaining=lap_end−lap_number and fuel_corrected_pace=lap_duration+k⋅laps_remaining(k=0.035 seconds per lap)
This is a dependant variable used in final loss calculations not used in the model as it would cause leakage.

track evolution was not chosen as an engineered parameter, as it would've been a derived dependant value that would not have contributed to the model.

EXPLORATORY DATA ANALYSIS AND STATISTICAL Inference

ML MODELLING TO COMPUTE DEGRADATION RATE
1. Remove dependant variables to avoid leakage
lap_duration = segments_sector_1 + segments_sector_2 + segments_sector_3, and was therefore removed as it is clearly a dependant variable.
Segments were not removed as they individual segment times may have smth to contribute in the model.

laps_remaining is lap_start - lap_end, used to compute/approximate fuel_corrected_pace, not expected for the model.

2. What are we predicitng?
Instead of explicitly computing a “degradation rate” label (which becomes circular and leak-prone), the model is trained to predict:
fuel_corrected_pace on the NEXT lap

Fully causal:

All inputs belong to lap t
The model learns how tyre, weather, stint age, and track conditions evolve into lap t+1
No sneaky future-lap info (no target leakage)
Works perfectly with race simulations (autoregressive rollout)
Degradation is then obtained as an emergent property:

As XGBoost repeatedly predicts future fcp values, the falling trend in predicted pace is degradation.
No slope assumptions.
No fixed compound multipliers.
Just learned tyre physics from FP laps.

3. Model choice
After evaluating linear models, tree ensembles, and clustering-based approaches, the final decision was to use XGBoost as the primary degradation modelling engine.

Reasoning:
Linear Regression underfits non-linear tyre behaviour (tyre warm-up phase, thermal degradation, cliffing, etc.).
Random Forest produces noisy, stepwise predictions → unstable in autoregressive simulations.
K-Means provides no predictive capability (only grouping).
XGBoost hits the sweet spot →
non-linear learning, extremely stable rollouts, low overfit, fast inference, and direct interpretability through feature importance and SHAP.

This choice also aligns with how real-world motorsport predictive pipelines are structured:
“predict the next lap’s corrected pace and let degradation emerge naturally from repeated predictions.”

RACE SIMULATION AND OPTIMISATION
1. Generate the feasible strategy space
All valid race strategies are enumerated by combining:
0-stop, 1-stop, 2-stop, and 3-stop structures

All possible tyre-compound sequences for those structures
Valid pit-lap windows (no pits on the formation laps, no pits too early or too late)
This gives the full set of legal and physically meaningful strategies for the race.

2. Lap-by-lap simulation using the ML model

Each candidate strategy is simulated one lap at a time.
For every lap:
Predict next-lap fuel-corrected pace via XGBoost
Add a fuel penalty that reduces over the race distance
Update all lap-state variables (laps_since_stint_start, tyre compound, tyre age, etc.)

Objective function: minimize total race time

For each candidate strategy, total race time is computed as sum of all lap times + pit_time
This gives a single scalar number for each strategy, allowing direct comparison.

4️. Strategy optimisation
The optimisation engine evaluates and ranks strategies by total race time.
Dynamic Programming is used to prune clearly inferior strategies

Future work: Optional Genetic Algorithm can be applied for deeper search across larger sequences
The best compound order + pit windows are selected based on lowest predicted race time
This produces the final recommended tyre strategy.

EVALUATION AND VALIDATION
1. Model Validation

The predictive model is validated using:
5-fold cross-validation on next-lap fuel-corrected pace
Residual plots to check for systematic bias
SHAP or built-in feature importance to verify that the model follows physics
(e.g., tyre age, compound, track temperature matter most)

Critical checks:
Predictions remain smooth when rolled forward for multiple laps
No drift or instability during stint simulation
Feature influence aligns with expected tyre-degradation behaviour

2. Strategy Validation

Once the model is verified, strategy validation ensures the chosen strategy actually makes sense.
This includes:
Running multiple simulated race conditions with small variations
Comparing alternative strategies across pit windows and tyre sequences
Inspecting heatmaps of predicted total race time across the strategy grid
The purpose here is not statistical inference, but checking consistency:
The recommended strategy remains optimal across plausible variations
No alternative strategy becomes better under small perturbations
The solution is stable and not sensitive to minor model noise

THANK YOU
DEVAJ RATHORE
NEWLY SELF APPOINTED ML EXPERT