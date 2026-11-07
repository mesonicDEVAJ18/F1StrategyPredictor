F1 RACE STRATEGY OPTIMISER PIPELINE
Requirement for Data Science CSE558 at Indraprastha Institute of Information Technology Delhi
Since then updated, a year after original was built; for better results.

PROBLEM STATEMENT
Use data collected over 3 practice sessions in a non sprint Formula 1 weekend to predict the optimal tyre strategy for a race where at least two compounds are supposed to be used (out of 3, hard, medium and soft).
The problem statement limits us to the three practice sessions only. We could broaden our scope beyond this, but for this project we won't: no qualifying, no race data and no other weekends' practice data go into the model.

DATA COLLECTION
The reference analysis is Bahrain 2025, and the same pipeline runs on any non-sprint 2024 or 2025 weekend: set RACE_YEAR and RACE_WEEKEND in .env (e.g. 2024 and saudi_arabia, the names are in scripts/race_weekend.py).
The model is trained on the whole field (20 drivers per session in Bahrain 2025), and the race strategy is simulated for one car, SIM_DRIVER in .env (default 16, Ferrari), using that car's own tyre sets. The only reason to set SIM_DRIVER is to adjust for tyre age and limits on number of tyres available for a specific driver. When unset (SIM_DRIVER= in .env) it is assumed fresh tyres are available: new sets of every compound, as many as a strategy needs, and the field's pace level (#35).

All data comes from https://openf1.org/

scripts/race_weekend.py is the first step of master.py and does steps 1-6 below by itself. If data/raw/ already holds the selected weekend it does nothing, otherwise it fetches it, waiting OPENF1_REQUEST_DELAY_S between requests and retrying HTTP 429/5xx and network errors.

Weekend cache
Every weekend fetched so far is kept in data/cache/<year>_<name>/ (e.g. 2025_bahrain), laid out like data/raw/ (laps/, stints/, drivers/, weather/, car_data/). About 70 files and 67-69 MB per weekend, gitignored.
data/raw/ is only the active weekend, a copy of its cache folder, so no other stage cares which weekend is selected.
On a switch, data/raw/ is moved into the cache folder of the weekend it holds (found from the meeting_key in laps_p1.csv, an unknown meeting goes to meeting_<key>/), the selected weekend's folder is completed from OpenF1, and it is copied into data/raw/. A weekend fetched before needs no requests.
Downloads go to <file>.part and are renamed when complete, so an interrupted run only fetches what is still missing.
A car-data request with no results (HTTP 404) leaves a <file>.nodata marker so it isn't requested again (delete the marker to retry). Laps, stints, drivers or weather with no results stop the run.
To force a re-download, delete the file from data/cache/, and also from data/raw/ if it is the active weekend.
scripts/results/race_weekend/race_weekend_summary.csv records where each run got its data, the file counts and any drivers with no car data.

1. Identify the 3 practice sessions ids. They are hardcoded per weekend in race_weekend.py's WEEKENDS table, together with the meeting_key and the scheduled race laps (the simulator's race distance), taken from https://api.openf1.org/v1/sessions?year=<year>&session_type=Practice
For Bahrain 2025 (meeting 1257):
1	10007	Practice 1	Practice	2025-04-11 11:30	2025-04-11 12:30	1 hr	Friday FP1
2	10008	Practice 2	Practice	2025-04-11 15:00	2025-04-11 16:00	1 hr	Friday FP2
3	10009	Practice 3	Practice	2025-04-12 12:30	2025-04-12 13:30	1 hr	Saturday FP3
Sprint weekends have FP1 only and are rejected.

2. Download all laps' data for the sessions
https://api.openf1.org/v1/laps?session_key=<session_key>&csv=true

3. Download all stints' data for the sessions
https://api.openf1.org/v1/stints?session_key=<session_key>&csv=true

4. Download the weekend's weather related data
https://api.openf1.org/v1/weather?meeting_key=<meeting_key>&csv=true
This covers the whole meeting (practice, qualifying and race), process_weather.py keeps only the practice sessions.

5. Download each session's entry list (driver_number and team_name)
https://api.openf1.org/v1/drivers?session_key=<session_key>&csv=true
tyre_sets.py uses these to find stand-ins. The FP3 lineup is taken as the race cars, and in FP1 and FP2 a driver who is not in it is paired with the one driver of the same team who sat that session out, so their running counts toward that race car's tyre sets.
In Bahrain 2025 six stand-ins ran in FP1 (38 in #16, 34 in #14, 37 in #1, 46 in #55, 50 in #87, 72 in #63), so those six race drivers have no FP1 laps.
The same rule covers a driver replaced before FP3 (Jeddah 2024: Bearman #38 took over Sainz's #55 car, so 55 counts toward 38). A team that can't be paired one-to-one is logged as a warning and its drivers keep their own numbers.

6. Download the car data for every driver in each session (the driver_numbers in that session's laps file)
https://api.openf1.org/v1/car_data?session_key=<session_key>&driver_number=<driver>&csv=true
Saved as data/raw/car_data/car_data_p<session>_<driver>.csv (60 files, ~66 MB for Bahrain 2025). The driver list is read from the laps files after they are downloaded, so it follows each weekend's field.
The car data is the lap's own telemetry, so it is not a model input (see ML Modelling step 1). eda.py and explore/tune_push.py use it.

DATA UNDERSTANDING
Laps
| **Feature**           | **Meaning / Real-World Interpretation**                  | **Type**                   | **Example / Units**     | **Analytical Role**                                                                 |
| --------------------- | -------------------------------------------------------- | -------------------------- | ----------------------- | ----------------------------------------------------------------------------------- |
| **date_start**        | UTC timestamp when the lap began.                        | *Datetime*                 | “2024-03-01 T14:23:15Z” | Enables ordering, track-evolution analysis, and time-based joins with weather data. |
| **driver_number**     | Car/driver number for the weekend.                       | *Categorical / identifier* | 16 = Leclerc            | Merge key across tables.                                                            |
| **duration_sector_1** | Time to complete Sector 1.                               | *Numeric (float)*          | 29.374 s                | Used for localized performance analysis (track-segment performance).                |
| **duration_sector_2** | Time to complete Sector 2.                               | *Numeric (float)*          | 38.452 s                | Used for lap-time composition.                                                      |
| **duration_sector_3** | Time to complete Sector 3.                               | *Numeric (float)*          | 26.928 s                | Combined with others to get total lap time.                                         |
| **i1_speed**          | Instantaneous speed at the first intermediate line.      | *Numeric (float)*          | 305 km/h                | Indicates straight-line speed and DRS use.                                          |
| **i2_speed**          | Instantaneous speed at second intermediate.              | *Numeric (float)*          | 280 km/h                | Captures mid-sector speed differences.                                              |
| **is_pit_out_lap**    | Boolean flag: 1 = lap starts from pit exit.              | *Boolean / binary*         | True / False            | Helps exclude outlaps from analysis.                                                |
| **lap_duration**      | Total lap time (sector1 + 2 + 3).                        | *Numeric (float)*          | 92.841 s                | Basis of fuel_corrected_pace, whose fall-off with tyre age the model learns.        |
| **lap_number**        | Sequential lap index within the session.                 | *Numeric (int)*            | 1 → 60                  | Enables stint segmentation and degradation trends.                                  |
| **meeting_key**       | Same race-weekend identifier.                            | *Categorical / identifier* | 1257                    | Identifies the weekend (used by the weekend cache).                                 |
| **segments_sector_1** | Mini-sector status codes in Sector 1 (2048 yellow, 2049 green, 2051 purple, 2064 pit lane). | *Array / list (int)* | [2049, 2049, 2048] | Replaced by its length (number of mini-sectors), see Encoding.                     |
| **segments_sector_2** | Same for Sector 2.                                       | *Array / list (int)*       | …                       | Same as Sector 1.                                                                   |
| **segments_sector_3** | Same for Sector 3.                                       | *Array / list (int)*       | …                       | Same as Sector 1.                                                                   |
| **session_key**       | Session identifier (FP1, FP2, FP3).                      | *Categorical / identifier* | 10007                   | Used to merge with `/stints`.                                                       |
| **st_speed**          | Speed at the speed trap (where top speeds are usually recorded). | *Numeric (float)*  | 310 km/h                | Proxy for DRS usage and straight-line performance.                                  |

Stints
| **Feature**           | **Meaning / Real-World Interpretation**                                                               | **Type**                   | **Example / Units** | **Analytical Role**                                              |
| --------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------- | ------------------- | ---------------------------------------------------------------- |
| **compound**          | Tyre compound used in a stint (Soft, Medium, Hard, Inter, Wet). Determines grip and degradation rate. | *Categorical (nominal)*    | “Soft”              | Key explanatory variable for pace and degradation models.        |
| **driver_number**     | Unique car/driver code assigned during a Grand Prix weekend (e.g., 63 = Russell).                     | *Categorical / identifier* | 44, 1, 16           | Join key across all endpoints; used to merge lap and stint data. |
| **lap_start**         | First lap number where this tyre set was used.                                                        | *Numeric (int)*            | 1 → 57              | Defines stint boundaries for degradation calculations.           |
| **lap_end**           | Last lap number before a pit stop or tyre change.                                                     | *Numeric (int)*            | 18 → 32             | Determines stint length = `lap_end − lap_start + 1`.             |
| **meeting_key**       | Unique race-weekend identifier.                                                                       | *Categorical / identifier* | 1257                | Used to group sessions by Grand Prix.                            |
| **session_key**       | Unique identifier for a session (FP1, FP2, FP3).                                                      | *Categorical / identifier* | 10007               | Enables merging across multiple FP sessions.                     |
| **stint_number**      | Sequential number of the stint for a given driver in that session (1, 2, 3 …).                        | *Numeric (int)*            | 1, 2, 3             | Helps analyze pace evolution per stint.                          |
| **tyre_age_at_start** | Tyre age (laps already completed on this set) when the stint began — reused tyres start > 0.          | *Numeric (float)*          | 0 – 25 laps         | Used to adjust degradation modeling (fresh vs used sets).        |

Weather
| **Feature**           | **Meaning / Real-World Interpretation**                                                                                            | **Type**                   | **Example / Units**        | **Analytical Role**                                                                   |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | -------------------------- | ------------------------------------------------------------------------------------- |
| **air_temperature**   | The ambient temperature of the air surrounding the track at the recorded time. Affects engine cooling and aerodynamic performance. | *Numeric (float)*          | 27.3 °C                    | Used to assess how ambient conditions influence lap pace and degradation.             |
| **date**              | UTC timestamp when the weather sample was recorded.                                                                                | *Datetime*                 | `2024-03-01T14:22:00Z`     | Allows time-based alignment with lap start times from `/laps`.                        |
| **humidity**          | Relative humidity of the air. Higher humidity can affect tyre and engine efficiency.                                               | *Numeric (float)*          | 62 %                       | Used in correlation analysis and regression as an explanatory environmental variable. |
| **meeting_key**       | Unique race-weekend identifier (same as in `/laps` and `/stints`).                                                                 | *Categorical / Identifier* | 1257                       | Enables merging of weather data with lap and stint data by race.                      |
| **pressure**          | Atmospheric pressure measured at the circuit.                                                                                      | *Numeric (float)*          | 1012 mbar                  | Optional feature; can indicate altitude and air density effects.                      |
| **rainfall**          | Whether there is rainfall (flag; always 0 in Bahrain 2025 practice).                                                               | *Binary / Numeric (int)*   | 0 or 1                     | Critical for excluding wet laps or performing wet-vs-dry comparisons.                 |
| **session_key**       | Session identifier (FP1, FP2, FP3, etc.) — consistent with `/laps` and `/stints`.                                                  | *Categorical / Identifier* | 10007                      | Used to filter data for specific practice sessions.                                   |
| **track_temperature** | Temperature of the asphalt surface — directly affects tyre grip and degradation.                                                   | *Numeric (float)*          | 41.7 °C                    | One of the most important predictors for tyre performance.                            |
| **wind_direction**    | Direction from which wind is blowing, usually measured in degrees (0–360).                                                         | *Numeric (float)*          | 250° (wind from WSW)       | Used for advanced aerodynamic or circuit-specific modeling.                           |
| **wind_speed**        | Wind velocity at the track, often measured at a fixed reference point.                                                             | *Numeric (float)*          | 3.6 m/s                    | Can influence drag and straight-line speed variations.                                |

Car Data
| **Feature**       | **Meaning / Real-World Interpretation**                                                                | **Type**                   | **Example / Units**    | **Analytical Role**                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------ | -------------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| **brake**         | Indicates whether the brake pedal is engaged by the driver. `100` means braking, `0` means no braking. | *Binary / Numeric (int)*   | 0 or 100               | Used to analyze braking zones, driver behavior, and deceleration patterns.         |
| **date**          | UTC timestamp when the telemetry data sample was recorded.                                             | *Datetime*                 | `2024-03-01T14:22:05Z` | Enables synchronization with lap, weather, or position data over time.             |
| **driver_number** | The permanent racing number assigned to a Formula 1 driver for the season.                             | *Categorical / Identifier* | 1, 16, 44              | Used to group or filter telemetry data by specific drivers.                        |
| **drs**           | DRS status code. 0/1 off, 8 eligible in the activation zone, 10/12/14 open; 9, 11, 13, 15 also occur. | *Categorical / Numeric*    | 0, 1, 8-15             | Helps analyze overtaking zones, straight-line speed advantages, and race strategy. |
| **meeting_key**   | Unique identifier for a race weekend (Grand Prix event).                                               | *Categorical / Identifier* | 1257                   | Allows merging telemetry with laps, weather, and stint data for the same event.    |
| **n_gear**        | The currently selected gear in the car’s transmission. `0` represents neutral or no gear engaged.      | *Numeric (int)*            | 0–8                    | Useful for analyzing acceleration patterns, gear shifts, and driver control.       |
| **rpm**           | Engine speed measured as revolutions per minute. Indicates how fast the engine crankshaft is rotating. | *Numeric (int)*            | 11500 rpm              | Used to study engine performance, power delivery, and gear efficiency.             |
| **session_key**   | Unique identifier for the track session (FP1, FP2, FP3, Qualifying, etc.).                             | *Categorical / Identifier* | 10007                  | Enables filtering telemetry for a specific session during a race weekend.          |
| **speed**         | Instantaneous velocity of the car measured by telemetry sensors.                                       | *Numeric (float)*          | 298 km/h               | Critical metric for performance analysis, sector speed comparison, and drag study. |
| **throttle**      | Percentage of maximum engine power being applied by the driver through the throttle pedal.             | *Numeric (float)*          | 0–100 %                | Used to analyze acceleration behavior, power usage, and driver input patterns.     |

Drivers (per-session entry list)
| **Feature**       | **Meaning / Real-World Interpretation**                                                                | **Type**                   | **Example / Units**    | **Analytical Role**                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------ | -------------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| **driver_number** | Number of a driver entered in the session.                                                             | *Categorical / Identifier* | 16, 38                 | Matched against the FP3 lineup to find practice stand-ins.                         |
| **team_name**     | The driver's team in that session.                                                                     | *Categorical*              | Ferrari                | Pairs a stand-in with the team's race driver who sat the session out.              |
| **session_key**   | Session identifier (FP1, FP2, FP3).                                                                    | *Categorical / Identifier* | 10007                  | One file per session (drivers_p1-3.csv).                                           |
(The file has further columns, e.g. full_name and name_acronym, tyre_sets.py only reads driver_number and team_name.)

Relationship Mapping between these 5 raw tables
(These are the natural join keys. What the pipeline actually does is under DATA PREPARATION, Feature Engineering.)
| **Relationship**                     | **Common Keys**                                                                                             | **Description**                                                                            | **Type**                                        | **Usage**                                                                                       |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Laps ↔ Stints**                    | `meeting_key`, `session_key`, `driver_number`, and lap overlap (`lap_number ∈ [lap_start, lap_end]`)        | Links each lap to the tyre compound, stint number, and tyre age used during that lap.      | **One-to-Many** (one stint → many laps)         | Adds tyre compound and degradation information to lap-level performance analysis.               |
| **Laps ↔ Weather**                   | `meeting_key`, `session_key`, and nearest timestamp (`date_start ≈ date`)                                   | Associates each lap with environmental conditions at the time the lap started.             | **Many-to-One (time-based join)**               | Adds air/track temperature, humidity, wind, etc., to lap-level performance modeling.            |
| **Stints ↔ Weather**                 | `meeting_key`, `session_key`, and midpoint timestamp of stint (`mid_stint_time ≈ date`)                     | Connects tyre stints to the weather conditions experienced during the stint duration.      | **Many-to-One (aggregate join)**                | Used to analyze how weather affects tyre degradation and stint strategy.                        |
| **Laps ↔ Car Data**                  | `meeting_key`, `session_key`, `driver_number`, and timestamp alignment (`date_start ≤ date ≤ lap_end_time`) | Links lap records with high-frequency telemetry such as speed, throttle, braking, and RPM. | **One-to-Many** (one lap → many telemetry rows) | Enables detailed performance analysis like braking zones, acceleration patterns, and DRS usage. |
| **Stints ↔ Car Data**                | `meeting_key`, `session_key`, `driver_number`, and timestamp range within stint duration                    | Maps telemetry data to the tyre stint in which it occurred.                                | **One-to-Many**                                 | Allows analysis of driving style and power usage across different tyre compounds.               |
| **Car Data ↔ Weather**               | `meeting_key`, `session_key`, and nearest timestamp (`date ≈ date`)                                         | Aligns telemetry samples with the weather conditions at that moment.                       | **Many-to-One**                                 | Enables studies of how environmental factors affect speed, throttle usage, and engine RPM.      |
| **Drivers ↔ Laps / Stints**          | `session_key`, `driver_number`                                                                              | Gives each driver's team in that session.                                                  | **One-to-Many** (one entry → many laps/stints)  | Maps practice stand-ins to their race car for the tyre-set inventory (tyre_sets.py).            |
| **Shared Identifiers (Global Keys)** | `meeting_key`, `session_key`, `driver_number`                                                               | Common identifiers present across all datasets ensuring consistent merging.                | —                                               | Allows creation of a unified dataset combining laps, stints, weather, and telemetry data.       |

RUNNING THE PIPELINE
Python 3.10, dependencies pinned in requirements.txt (the versions in .venv that produced the committed results):
    pip install -r requirements.txt
All scripts live in scripts/ and use relative paths, so run them from inside scripts/:
    cd scripts
    python master.py
master.py runs race_weekend.py → process_laps.py → process_stints.py → tyre_sets.py → process_weather.py → process_cardata.py → feature_engineer.py → eda.py → train_model.py → strategy_simulator.py → validate_strategy.py, and stops at the first script that fails.
Every script writes a log, a manifest and its outputs to scripts/results/<script_name>/.
Settings (the race weekend, paths, file names, cleaning thresholds, model hyperparameters, seeds and race parameters such as PIT_LOSS_S) live in .env at the repo root and are read through scripts/env_config.py. A missing key is an error, there are no hidden defaults. An environment variable of the same name overrides .env for one run, e.g.
    $env:RACE_YEAR = "2024"; $env:RACE_WEEKEND = "saudi_arabia"; python master.py
runs the whole pipeline on Jeddah 2024.
data/raw/, data/processed/ and scripts/results/ are tracked in git, so running another weekend rewrites them. Switching back to 2025 bahrain restores the committed data from the cache, and the results reproduce.
Stages hand data to each other through CSVs:
    race_weekend.py      → data/raw/ (the selected weekend, from data/cache/ or OpenF1)
    process_laps.py      → data/processed/laps_combined_clean.csv
    process_stints.py    → data/processed/stints_combined_clean.csv
    tyre_sets.py         → data/processed/tyre_sets_race_inventory.csv (read by the simulator)
    process_weather.py   → data/processed/cleaned_weather.csv
    process_cardata.py   → data/processed/cardata_per_lap_clean.csv
    feature_engineer.py  → data/processed/final_engineered_dataset.csv
    train_model.py       → scripts/results/train_model/train_model_fall_off_model.pkl (read by the simulator and validation)
The fall-off model itself is chosen outside the pipeline, in explore/ (see ML Modelling step 3), and the cleaning thresholds are tuned there too (see Data Preparation step 2).

DATA PREPARATION
Scripts: process_laps.py, process_stints.py, process_weather.py, process_cardata.py
1. Remove redundant data from all of the data types
These scripts remove the meeting keys, which are irrelevant to the pipeline.
process_laps.py and process_stints.py keep session_key, because lap numbers and stint numbers restart every session and feature_engineer.py needs it to give each lap its own session's stint. process_weather.py drops it after filtering, since weather is joined on timestamps.
process_laps.py also tags every lap with session_name (FP1, FP2 or FP3, from the laps_p1/p2/p3 file name). It follows practice order, so it means the same on every weekend, unlike session_key.
process_laps.py also removes pit-out laps (is_pit_out_lap = True), i.e. laps that start from the pit exit. There is no in-lap flag in the data, so slow in-laps are caught by the non-push filter below.

2. Remove outliers
Laps (each session is cleaned separately)
A lap missing exactly one sector time, with lap_duration and the other two sectors recorded, gets the missing sector as lap_duration minus the other two (kept only if > 0). This rebuilds 0 / 1 / 1 laps in FP1 / FP2 / FP3 (saved to scripts/results/process_laps/process_laps_rebuilt_sectors_laps_p*.csv). Both are slow (152 s, 174 s) and the IQR and non-push filters remove them later, but on another weekend this keeps push laps.
Removed laps with missing sector times that could not be rebuilt and sector times of 0 or below (0 / 0 / 4 laps). Also removed laps with unrealistic lap times (30 s or less, or 2x the median lap time of that session or more).
Also removed laps where a recorded i1_speed, i2_speed or st_speed is 50 km/h or less, indicating aborted laps (due to whatever reason). This removes 1 lap (FP1).
OpenF1 leaves i1_speed or st_speed empty on some laps (i2_speed never), 24 / 71 / 7 laps reaching this filter. These are kept, a missing speed says nothing about the lap time, and most of them are long-run laps (FP2 median 1.09x the driver's session best).
Removed laps whose residual falls below Q1 - k_lower x IQR (too fast) or above Q3 + k_upper x IQR (too slow), where residual = lap_duration - the 5-lap centred rolling median of that driver's lap_duration. A rolling median compensates for the track improving across a single session.
The two sides have their own multiplier, IQR_LOWER_MULTIPLIER = 62 and IQR_UPPER_MULTIPLIER = 3 in .env (tuned below). This removes 1 / 65 / 1 laps, all of them slow and none fast.
The bounds come from the whole session's residuals, so they depend on its mix of laps. In FP2 the long-run laps make the residuals dense (IQR 5.5 s, upper bound +21.5 s), so many cool-down laps go here rather than in the non-push filter.
Every lap reaching this filter is saved with its residual and its ratio to the driver's best (process_laps_iqr_residuals_laps_p*.csv), and the log gives each session's Q1, Q3, IQR and bounds.
Finally, removed non-push laps (cool-down, in- and traffic laps), using two tests that a lap must both pass:
Session test: laps slower than 1.15x the driver's best lap in that session (PUSH_LAP_MAX_BEST_RATIO). Divided by the driver's session best, laps split into two groups: push laps and long runs at 1.10x or less, and cool-down laps from about 1.3x, with a gap around 1.14-1.16x. The compound isn't known yet at this stage (it comes from the stints), but the gap between compounds (~1%) is well inside the margin. This removes 146 / 69 / 121 laps.
Stint test: laps slower than 1.10x the driver's best lap in the same stint (PUSH_LAP_MAX_STINT_RATIO, stints matched from the raw stints by lap_start / lap_end). This removes 2 / 3 / 2 more laps.
That leaves 566 laps, 186 / 295 / 85 (median 98.3 / 98.9 / 94.9 s, 90th percentile 101.9 / 100.9 / 99.8 s). Every lap reaching this filter is saved with both ratios (process_laps_push_ratios_laps_p*.csv).

Tuning the IQR fences (explore/tune_iqr.py)
Why the two sides differ: a lap faster than its neighbours in practice is a push lap, which is exactly what the model needs. A push lap run between cool-down laps has a cool-down lap as its rolling median, so a normal 92-97 s lap gets a residual of -17 to -64 s, and in a session full of steady long runs (FP2) the IQR is small, so one shared multiplier threw those push laps away. A really wrong fast time is already caught upstream (30 s or less, missing or non-positive sectors). A slow lap on the other hand can be traffic, a mistake or a cool-down lap, so the slow side has real work to do.
Rejected: a skew-adjusted boxplot (medcouple), the textbook way to set asymmetric fences. The residuals are right-skewed (cool-down laps), so it narrows the lower fence to ~0.07 x IQR and removes 34-76 push laps per session, the opposite of what is needed.
Method: tune_iqr.py runs the pipeline's own stages on 12 non-sprint weekends (2024 Hungary, Saudi Arabia, Spain, 2025 Australia, Austria, Bahrain, Britain, Italy, Japan, Netherlands, Singapore, Spain, 36 practice sessions), with the settings passed as environment variables, so nothing in the cleaning is reimplemented.
A. Label sweep: a lap is a push lap when best_ratio <= 1.15. For each session and side it finds the exact k at which the fence starts removing push laps. The lower-side floor is the smallest grid value at least 1.2x the largest k needed in any session.
B. CV sweep: for each pair (k_lower, k_upper) in {3, 6, 10, 20, floor} x {1.5, 2, 3, 5}, process_laps → process_cardata → feature_engineer rerun on every weekend, and the model-free stage-1 linear fall-off is scored by within-stint shape MAE in 5-fold CV grouped by stint (2 seeds). Every pair is scored on the same rows and the same stint folds, and compared with one-sided paired t and Wilcoxon tests. Rule: start from (floor, 3) and move only to a pair that beats it on both tests.
Results: in FP1 and FP3 the lower side needs k = 2.2-6.1, but in FP2 it needs 11.3-16.7 on every weekend with dense long runs, and 51.23 on Spain 2025 (IQR 1.18 s). So the floor is 1.2 x 51.23 → 62. The upper side needs at most 4.21, and at k >= 3 it removes 1 push lap in 36 sessions.
In the CV sweep (1,071 stints, 5,255 common rows) (62, 3) scores 0.9722 s against 0.9725 s for (3, 3), and no pair beats it on both tests.
Winner: IQR_LOWER_MULTIPLIER = 62, IQR_UPPER_MULTIPLIER = 3. A full tune_iqr.py run writes its winner into .env itself (--quick never writes), the reasoning comment above the keys is updated by hand.
Caveat: k_lower = 62 in practice switches the fast side off (it removes no lap on any of the 36 sessions). The k needed is set by the densest FP2 seen so far, so a new weekend can need more. Rerun tune_iqr.py when weekends are added.

Tuning the push-lap filter (explore/tune_push.py)
The question: the session test is relative to the driver's session best, usually a low-fuel SOFT lap, so heavy-fuel long-run laps might sit close to it. Judging each lap against its own stint's best lets long runs be judged against themselves.
Method: tune_push.py is a copy of tune_iqr.py's harness, on the same 12 weekends.
A. Label sweep: the label must not use the ratios being tuned, so it comes from car data. process_laps.py runs with the push filter off and process_cardata.py with its z-score filter off, and each lap's full-throttle share divided by the driver's 90th percentile in the session labels it a push lap (>= 0.8) or a cool-down lap (<= 0.6). That is 11,449 laps: 6,363 push, 4,003 cool-down, 1,063 ambiguous. Each pair in {1.10 … 1.30} x {1.04 … 1.12, off} is scored by Youden's J (share of push laps kept - share of cool-down laps kept).
B. CV sweep: as for the IQR fences, over {1.10 … 1.30} x {1.06 … 1.12, off}.
Results: the best label pair is (1.15, 1.10), push laps kept 99.4% and cool-down laps 0.8% (J 0.986), against J 0.985 with the stint test off. The session ratio already sits at the gap, looser ratios let cool-down laps in quickly and 1.10 loses 2.6% of push laps.
The CV sweep picks the strictest corner, (1.10, 1.06), but this is a bias, not a result: the common rows are the strictest setting's own rows, so it is scored on exactly the laps it trains on, while looser settings also train on laps that are never scored. That strict filter drops 580 laps, 170 of them labelled push laps and 61% of them from runs of 8+ laps, exactly the long runs we wanted to protect.
Decision: the label pick, PUSH_LAP_MAX_BEST_RATIO = 1.15 and PUSH_LAP_MAX_STINT_RATIO = 1.10. A full tune_push.py run writes its label pick into .env itself, the CV winner is for information only. Run tune_iqr.py before tune_push.py, never both at once (they share data/raw/ and data/processed/).

Stints
No continuous telemetry available, prompting use of logical checks only, removed rows where lap_start > lap_end, negative tyre_age_at_start, and missing compound.
Only dry stints (SOFT, MEDIUM, HARD) are kept. INTERMEDIATE, WET and UNKNOWN stints are removed (none on Bahrain 2025, e.g. 30 UNKNOWN on Austria 2025), since their one-hots would be all 0 and read as SOFT.
A session that ran no stint on a compound has no one-hot column for it, so that column is filled with 0 after the sessions are combined (none on Bahrain 2025, Australia 2025 FP1 has no HARD).

Weather
Kept only the practice sessions (the session_keys found in laps_p1-3.csv), qualifying and race weather are dropped.
Applied Z-score filtering (|z| > 3) on continuous atmospheric readings: Air temperature, Track temperature, Humidity, Pressure, Wind speed and Rainfall. Columns with zero variance are skipped.

CarData
process_cardata.py keeps session_key, because it needs it to match samples to laps. Each car_data_p*_*.csv file is cleaned separately:
Removed duplicate timestamps.
Removed samples where brake or throttle read 104. This value means there is no valid input, and these samples are always at about 0 km/h in neutral (the car sitting in the garage). This is roughly a third of all samples.
Removed physically impossible readings: speed outside 0-400 km/h, rpm outside 0-16000, gear outside 0-8, throttle outside 0-100, brake not 0 or 100.
Each ~4.2 Hz sample is then assigned to the lap whose window [date_start, date_start + lap_duration) contains it, matched on session_key and driver_number. The windows come from the raw laps files, so every lap has one, including the laps process_laps.py drops, and each sample lands in the lap it was actually driven in. Samples outside every lap (pit lane, garage) are dropped.
The samples of each lap are aggregated to mean/max speed, mean throttle, share at full throttle (>= 99%), share braking, mean/max rpm, mean gear, gear changes and share with DRS open.
Lap-level filters, in order:
Laps with fewer than half the expected samples (lap_duration x sample rate) are dropped, so gappy telemetry doesn't give misleading means.
Only laps that survive process_laps.py are kept (matched on driver_number + date_start), so the car data describes the same laps as the lap data.
Z-score masking (|z| > 3, per session) on the aggregates, except DRS share, which is mostly 0 so any DRS use would look like an outlier. Only the flagged aggregate is set to NaN, the lap and its other aggregates are kept. This masks 27 values on 20 laps (process_cardata_zscore_masked_cells.csv), and feature_engineer.py fills them from the stint's earlier laps.
The output has all 566 cleaned laps (all 20 drivers in every session), saved to data/processed/cardata_per_lap_clean.csv with session_key, driver_number and lap_number as join keys.

3. Encoding
Laps
segments_sector_1/2/3 arrive as lists of mini-sector status codes, each list is replaced by its length (the number of mini-sectors).
Driver number was kept numerical as it is used to combine with stints data. It is never fed to the model, as it was decided to not build a driver specific model, and stays in the engineered dataset as an identifier only, so stints and driver-session levels can be built per driver.
session_name is one-hot encoded (in feature_engineer.py) → session_FP1, session_FP2, session_FP3. These aren't model inputs either: the model learns fall-off within a stint, and the stint's level absorbs the session's pace (track evolution, run programmes, time of day). eda.py uses them to split its summaries.

Stints
One-hot encoded compound → compound_HARD, compound_MEDIUM, compound_SOFT.

Weather
None needed.

CarData
brake (0 or 100) → braking (0/1). drs status codes → drs_open (1 for 10, 12 or 14, i.e. flap open, 0 otherwise). n_gear kept as an ordinal number.

4. Normalization
Laps
Not scaled. lap_duration, the sector times and the speed traps stay in physical units (seconds, km/h).
Linear Regression, Random Forest and XGBoost give the same fit whether or not a column is rescaled with one mean/std (trees split on the order of values, linear regression just rescales its coefficients), so a global scaler would only change the units.
A per-session scaler would remove each session's pace level, but it would put the target in z-scores, and the race has no session scaler to convert predictions back with, while the fuel and pit penalties are in seconds. So fuel_corrected_pace, every prediction, the simulator and the error metrics are all in seconds, and session-level differences are absorbed by each stint's level in the model, not by scaling.
(A model that needs scaled inputs, e.g. Ridge or an MLP, scales its own inputs inside its Pipeline in train_model.py's MODEL block.)

Stints
No normalization applied, stints contain mostly integer metadata (stint number, lap ranges).

Weather
Applied StandardScaler to: Air temperature, Track temperature, Humidity, Pressure, Wind speed, and Rainfall.
Wind direction kept raw due to its circular nature.

CarData
Not scaled, for the same reason as the laps, so eda.py and tune_push.py's push-lap label read real values (e.g. mean speed 195 km/h).

5. Feature Engineering (feature_engineer.py)
Stint number: stints are sorted by driver, session and per-session stint_number, and numbered 1, 2, 3 ... per driver across all three sessions (global_stint_number), kept session independent to include evolution information too.
Each lap is assigned the global number of the stint from the same session and driver whose [lap_start, lap_end] contains its lap_number. Lap numbers restart every session, so the session has to match as well.
Stint metadata (lap_start, lap_end, tyre_age_at_start, compound) is then merged onto the laps on driver_number + global stint number, which is unique per driver, so the merge is one-to-one: all 566 cleaned laps get a stint and keep exactly one row. A lap that matches no stint is dropped and counted in the log (0 on Bahrain 2025, it happens on weekends whose non-dry stints were removed).

Weather alignment: each lap is joined to the weather sample nearest its date_start (merge_asof on date_start ~ date, 5 minute tolerance). Weather is sampled about once a minute and the sessions are hours apart, so the match always comes from the lap's own session (every lap matches, within about 30 s).
OpenF1 leaves date_start empty on a few laps on some weekends (11 in Spain 2025 FP3), and merge_asof can't take a missing key, so those laps are dropped and counted (0 on Bahrain 2025).

Car data: each lap gets its own per-lap car-data aggregates, left-merged on session_key, driver_number and lap_number (one-to-one). They are not model inputs, the lap's own telemetry would leak its pace (see ML Modelling step 1), but they stay in the dataset for eda.py and tune_push.py.

Missing values: the speed traps, segment counts and car-data aggregates (STINT_FILL_COLUMNS in feature_config.py) are filled with the median of the same driver's earlier laps in the same stint (NaN skipped). Earlier laps only, so a lap never borrows from a later one.
Of the 566 laps, i1_speed is missing on 61 (51 filled), st_speed on 42 (39 filled), the car data on 27 values (18 filled) and the segment counts on 15 (none, no earlier lap of the stint has one). The counts are in feature_engineer_stint_fill_summary.csv.
Considered and not used: medians per session x driver, and predicting i1_speed / st_speed from i2_speed. The stint's earlier laps are the closest group and cover almost every gap.

laps_since_stint_start = lap_number - lap_start + 1 (clipped to at least 1)

tyre_age = tyre_age_at_start + laps_since_stint_start, the real age of the set, i.e. the laps it has done including lap t (1 on the first lap of a new set). laps_since_stint_start alone can't tell a stint on a new set from one on a set already used earlier in the weekend. On the current data it runs from 2 to 19 laps.

run_laps = lap_end - lap_start + 1, the length of the run on that set (3-18 laps), from the stints table. It tells a short quali sim from a long race-pace run, and is used only as a control in the compound-gap fit, not as an input to the fall-off model.

stint_avg_pace = mean(lap_duration of the earlier laps of that driver & stint), in seconds (an expanding mean of the stint so far). On a stint's first lap there are no earlier laps, so it is that lap's own time. It is descriptive only: it is built from the lap times the model predicts, so it is not a model input.

fuel_corrected_pace: laps_remaining = lap_end - lap_number + 1 (laps left in the stint, counting lap t) and fuel_corrected_pace = lap_duration - k*laps_remaining (k = 0.035 seconds per lap). More fuel makes a lap slower, so the fuel effect is subtracted to leave fuel-free pace.
Assumption: practice fuel loads are not recorded, so the laps left in the stint stand in for the fuel on board at the start of lap t (each run fuelled for its own length). The race fuel term in the simulator uses the same convention (laps left in the race, counting the lap), so race fuel is added once, in the simulator only.
fuel_corrected_pace is not a model input, its change with tyre age within a stint is what the model learns.

Finally date_start, date, lap_duration and laps_remaining are dropped, and the result is saved as data/processed/final_engineered_dataset.csv. session_key and driver_number stay in it as identifiers, the models only read the columns listed in feature_config.py.

track evolution was not chosen as an engineered parameter, as it would've been a derived dependent value that would not have contributed to the model. (The stint's level absorbs it anyway.)

EXPLORATORY DATA ANALYSIS AND STATISTICAL INFERENCE (eda.py)
eda.py runs after feature_engineer.py and before train_model.py. It reads final_engineered_dataset.csv, changes nothing downstream, and writes its tables, plots and log to scripts/results/eda/.
lap_duration was dropped by feature_engineer.py, so it is rebuilt as fuel_corrected_pace + k*laps_remaining (it matches the sector sum exactly). Weather columns are z-scores, everything else is in physical units.

1. Summary statistics and missing values per column and session (eda_summary_stats.csv, eda_missing_values.csv)
After the stint fill, 11 columns still have missing values, at most 1.8%: i1_speed and segments_sector_1 on 10 laps each, segments_sector_2 on 4, st_speed on 3, and a few car-data aggregates on 1-2 laps. stint_avg_pace is never missing. segments_sector_1, segments_sector_2 and rainfall are constant, so they carry nothing for a model.

2. Distributions of lap_duration and fuel_corrected_pace by session and compound (eda_pace_distributions.png, eda_pace_by_session_compound.csv)
FP3 is the fastest session (fcp median ~94-97 s against ~97-99 s in FP1/FP2), and SOFT the fastest compound within each session.

3. Pace against tyre age (eda_pace_vs_tyre_age.png, eda_degradation_slopes.csv)
Raw pace mixes drivers, sessions and run plans (the fastest laps cluster at tyre age 2), so each lap's fcp and tyre age are also shown minus their stint's means, which leaves the change within a stint: the fall-off the model learns.
A linear fit gives +0.34 s/lap on SOFT, +0.23 on MEDIUM and +0.21 on HARD, from 33 / 38 / 23 stints with more than one clean lap. This is the same within-stint slope as train_model.py's stage 1. So the data does hold a fall-off trend, steepest on SOFT. The compounds are run on different programmes, so this part is descriptive only (step 6 tests it).

4. Correlations (eda_correlation_matrix*.csv, eda_correlation_heatmap.png, eda_collinear_pairs.csv, eda_target_correlations.csv)
On the 427 laps of multi-lap stints, between the candidate inputs of a fall-off model (compound, tyre ages, weather, never the lap's own timing) and within-stint pace (fcp minus the stint's mean).
Inputs with |Pearson r| >= EDA_COLLINEAR_THRESHOLD (0.8) are grouped: air and track temperature, humidity and pressure (weather changes little within a session).
laps_since_stint_start (+0.38) and tyre_age (+0.25) are the only clear correlations. The weather is ~0 within a stint, and the compound and tyre_age_at_start are exactly 0 because they are constant within a stint (their effect lives in the stint's level).

5. Row coverage (eda_row_coverage.csv, eda_stint_lengths.png)
The 566 clean laps sit in 233 stints, 139 of which have a single clean lap and carry no fall-off. The other 94 stints hold the 427 laps the model learns from (FP1 151, FP2 254, FP3 22, SOFT 174, MEDIUM 168, HARD 85), with tyre age 2-19 (HARD 2-18).

6. Statistical inference (eda_inference_*.csv, eda_inference.png)
Laps within a stint aren't independent, so every test runs on stints or on driver-session means (or resamples whole stints), never on single laps. alpha = SIGNIFICANCE_LEVEL (0.05), 95% intervals, and a Holm correction within each family of three comparisons. scipy only.
Degradation: stage 1's slope per compound, with a stint cluster bootstrap (EDA_BOOTSTRAP_RESAMPLES = 2000 resamples of whole stints within each compound, seeded with RANDOM_STATE).
SOFT +0.341 s/lap [+0.287, +0.440] and MEDIUM +0.227 [+0.135, +0.336] differ from 0 (bootstrap p < 0.0005 each), HARD +0.211 [-0.052, +0.460] does not (p 0.118).
As a check, one OLS slope per stint: 33 of 33 SOFT stints rise (Wilcoxon p < 0.0001), 27 of 38 MEDIUM (p 0.0003), but only 14 of 23 HARD (p 0.39), so HARD's fall-off is the least certain.
No two compounds' slopes differ after Holm (SOFT - MEDIUM +0.11 s/lap, p 0.085, Holm p 0.255).
Compound pace: each stint's pace on a new set (its stage-1 level + its compound's slope at tyre age 1, fitted on all 566 laps, so single-lap stints count) is averaged per driver, session and compound, and compounds are compared within the same driver-session (paired t-test and Wilcoxon), so neither the driver nor the session confounds it.
MEDIUM is 1.61 s slower than SOFT [+0.50, +2.72] over 27 driver-sessions (Wilcoxon Holm p 0.001), HARD 2.23 s slower [+1.51, +2.95] over 16 (Holm p 0.0001). Only one driver-session ran both HARD and MEDIUM, so their order can't be tested.
SOFT runs are shorter (median 4 laps, against 6 on MEDIUM and HARD), so short quali sims still widen the gap. This is the same confound as the compound gap in ML Modelling (stage 3), which gives MEDIUM +0.98 s and HARD +1.23 s with the run_laps control.
Session pace: the same new-set pace, compared between sessions for the same driver on the same compound, so a change in compound mix can't pose as track evolution.
FP2 is 2.77 s faster than FP1 [-3.64, -1.89] (15 pairs, Wilcoxon Holm p 0.0002) and FP3 1.68 s faster than FP1 [-2.35, -1.00] (13 pairs, Holm p 0.002). FP3 and FP2 don't differ (+0.25 s, 21 pairs, p 0.27). Friedman over the 11 driver-compounds that ran all three sessions: p 0.0008.
Track evolution can't be separated from the conditions (FP2 runs in the evening, FP1 and FP3 in the afternoon), fuel loads or engine modes. In the model the stint level absorbs all of them.

ML MODELLING TO COMPUTE DEGRADATION RATE (train_model.py)
1. Remove dependent variables to avoid leakage
The model learns how a lap's own fuel_corrected_pace changes with tyre age, so anything measured on that lap is part of, or a proxy for, the target: lap_duration, the sector durations, the speed traps, the segment counts, the car-data aggregates (the lap's telemetry) and stint_avg_pace (built from the lap times being predicted).
These are LAP_TIMING_COLUMNS in scripts/feature_config.py, and an assert there stops any of them becoming a model input. They stay in the engineered dataset for eda.py and tune_push.py.
laps_remaining = lap_end - lap_number + 1, used to compute/approximate fuel_corrected_pace, not expected for the model.
The model's inputs are one of the feature sets in feature_config.py (FEATURE_SETS, shared with explore/, each one checked against LAP_TIMING_COLUMNS), picked with FALL_OFF_FEATURE_SET in .env.
The current set, core, has 6 inputs: the compound one-hots (3), tyre_age, and the track and air temperature (the conditions the tyre runs in). The other sets are tyre only (compound + tyre_age), all the weather, and core plus tyre_age_at_start or laps_since_stint_start. The session one-hots aren't needed, the stint's level (step 2) absorbs the session.

2. What are we predicting?
Tyre fall-off with age. Each lap's pace is split into a level for its stint and a fall-off that depends on the tyre:
fuel_corrected_pace(lap) = level[stint] + fall_off(compound, tyre_age, conditions)
The stint's level absorbs everything that stays the same over a run: driver, car, session, fuel plan, engine mode, and whether it is a short quali sim or a long race-pace run. Practice data mixes all of these, and between stints they are much bigger than the tyre effect. What is left, the change within a run as the tyre ages, is what the strategy needs.
Not circular: the target is observed pace, and the only inputs are the tyre's compound and age and the weather, none of which comes from the lap's timing.
Not leaky: no input is measured on the lap (step 1).
Degradation is learnt directly, not as a side effect of rolling a next-lap model forward, so errors don't compound over a stint.

Training (train_model.py, with the shared helpers in feature_config.py):
Split: 80/20 by stint (GroupShuffleSplit on driver_number + stint_number, TEST_SIZE 0.20 of the stints, random_state 42), 186 stints / 464 laps train, 47 stints / 102 laps test. A stint's laps are all on one side, because consecutive laps of one stint are highly correlated and the simulator prices stints the model hasn't seen.
Rows: only the laps of stints with two or more clean laps carry fall-off (multi_lap_rows), 356 laps in 78 stints train, 71 in 16 test (427 laps in 94 stints overall). The 139 one-lap stints feed stage 3 only.
Stage 1, stint levels (fit_stint_levels): least squares of fcp on one level per stint plus one tyre_age slope per compound, on the training stints. The slopes are the within-stint fall-off rate, a linear model of the curve: SOFT +0.327, MEDIUM +0.210, HARD +0.215 s per lap of tyre age. They are also the linear baseline every model has to beat.
Stage 2, the model: target fcp - level[stint] (fall_off_target). Inputs still missing get the training rows' median (none on the current data). The curve is measured from a new set's first lap, fall_off(c, age) = g(c, age) - g(c, 1), at the training-median conditions (fall_off_curve).
Stage 3, compound gap (fit_compound_gap): fall-off gives the shape of each curve, not how fast a compound is on a new set, and that needs comparisons between stints.
stint_new_set_levels refits stage 1 on every training lap (one-lap stints too) and gives one row per stint: its pace on a new set (level + slope x anchor age) and its laps. A least-squares fit of that pace on one level per driver and session, a HARD and a MEDIUM indicator and run_laps (to control for the run type), each stint weighted by its laps, gives the gap to SOFT on a new set and each driver-session's pace level.
No fall-off model is involved, so the gap and the levels are the same whichever model is plugged in (swap test: XGBoost and Ridge give identical gaps, levels, slopes and medians). Current: MEDIUM +1.156 s, HARD +0.295 s, run_laps +0.362 s per lap of run, 60 driver-session levels.
Fitting stages 1 and 2 by turns (backfitting) was tried and didn't converge, the levels still moved 0.3-0.9 s after 8 rounds.
The bundle (train_model_fall_off_model.pkl) holds the model, its name, its feature set and feature list, the training medians, the anchor age, the compound gap, the driver-session levels and the stage-1 slopes.

3. Model choice
train_model.py fits one fall-off model, defined in one place, its MODEL block: an sklearn Pipeline (optional scaler + estimator) and MODEL_NAME, on FEATURE_SETS[FALL_OFF_FEATURE_SET]. Nothing else in the pipeline depends on which model it is: the simulator and validation read the bundle's feature list and call predict, and validation fits a clone of the model in each fold.
For now it is XGBoost (300 trees, learning rate 0.05, max depth 7, subsample 0.8, colsample 0.8, XGB_* in .env). It is a placeholder, it beats neither baseline on the fall-off target (see Evaluation), and choosing a replacement is still open.
The model is chosen outside the pipeline, in explore/explore_master.py. It runs the data stages up to eda.py, then compares the candidates in its experiment block on the fall-off target, with nested cross-validation grouped by stint (5 folds x 2 repeats, the hyperparameter search runs inside each outer training fold, stage 1 is fitted per outer fold) and the same within-stint shape score as validate_strategy.py.
Candidates: Ridge (one shared slope), a straight line or polynomial per compound, an isotonic (never faster) curve per compound, Random Forest, ExtraTrees, HistGradientBoosting and XGBoost (each also with a monotone constraint on tyre_age), XGBoost as configured now, SVR, k-nearest neighbours and an MLP.
Feature sets: the 6 inputs, tyre only, all the weather, the 6 + tyre_age_at_start, the 6 + laps_since_stint_start.
Baselines: no fall-off (flat) and the stage-1 straight line per compound (linear).
Each candidate's best configuration is drawn as a curve over the race length with the simulator's own helpers and the one model-free compound gap (stage 3 on all laps: MEDIUM +0.978, HARD +1.228 s), so every candidate's curves share the same gap. It is recommended only if it beats both baselines, its steps stay <= 1 s, it rises on every compound, and it never gets faster with age (MIN_MONOTONE_SHARE).
The winner goes into train_model.py's MODEL block and .env (FALL_OFF_FEATURE_SET and its hyperparameters) by hand.
K-Means was ruled out because it provides no predictive capability (only grouping).

Result on Bahrain 2025 (run 2026-11-05, 427 laps in 94 stints, within-stint shape MAE, mean +- std over the 10 outer folds):
| Model (best feature set)           | Shape MAE (s)   | Curve                                                                    |
| baseline flat (no fall-off)        | 1.035 +- 0.116  | -                                                                        |
| baseline linear (per compound)     | 0.905 +- 0.111  | S 0.34, M 0.23, H 0.21 s/lap, rises without limit past age 19            |
| SVR (6 + tyre_age_at_start)        | 0.877 +- 0.119  | best fit, but gets faster with age on half the steps (RBF reverts)       |
| MLP (6 + tyre_age_at_start)        | 0.888 +- 0.113  | recommended: S 0.18, M 0.17, H 0.17 s/lap, smooth (steps <= 0.18 s)      |
| Ridge (tyre only)                  | 0.890 +- 0.118  | one slope for every compound (0.255 s/lap), most stable across folds     |
| ExtraTrees (6 + tyre_age_at_start) | 0.909           | steps up to 0.8 s                                                        |
| XGBoost / Random Forest (tuned)    | 0.926 / 0.928   | steps of 1.1-1.2 s                                                       |
| XGBoost as in train_model.py       | 1.008-1.060     | the worst: depth 7 overfits, steps up to 3.1 s                           |
Reasoning:
Some fall-off is clearly real: every candidate's best configuration beats no fall-off, by 0.10-0.16 s.
The data doesn't separate the better models: the top ~15 configurations are within 0.03 s, and the standard error of a mean over 10 folds is ~0.04 s.
The MLP is the only candidate that passes the curve checks and beats the linear baseline significantly (better on 53 of 94 stints, Wilcoxon p = 0.038). Ridge does not (p = 0.61). SVR fits best but its curve is wrong past the training ages.
The trees, XGBoost included, are not better here: on ~430 laps they fit steps, not a curve, and can't extrapolate past the oldest tyre in training.
So the shortlist is the MLP (the rule's pick), Ridge (tyre only) and the per-compound line, all within one fold std. The MLP's lead over Ridge is 0.002 s. Its slopes are almost the same on every compound and Ridge's are identical, so either way the strategy rests mostly on the compound gap, which is itself poorly determined.
[TODO #27, P2, see diagnostic.md] Step 2: a better compound-gap estimator (the run type confounds it, and on the training split MEDIUM comes out slower than HARD on a new set). Step 4: choose the model (the MLP, Ridge or the per-compound line) and copy it into train_model.py's MODEL block and .env.
XGBoost's importances, for the record (train_model_feature_importance.csv): tyre_age 0.280, compound_SOFT 0.200, track_temperature 0.159, air_temperature 0.123, compound_MEDIUM 0.119, compound_HARD 0.119.

RACE SIMULATION AND OPTIMISATION (strategy_simulator.py)
1. Generate the strategy space
The race length is the weekend's scheduled race laps from race_weekend.py (57 for Bahrain 2025). Strategies are enumerated by brute force over:
1-stop and 2-stop structures (STOP_COUNTS).
Every compound sequence for that structure that uses at least two different compounds (MIN_COMPOUNDS_USED, the dry-race rule, repeats allowed, e.g. HARD-HARD-SOFT). 0-stop strategies are out, since one stint can only use one compound.
Every combination of pit laps from lap 10 to 11 laps before the end (PIT_WINDOW_START_LAP, PIT_WINDOW_END_BUFFER, laps 10-46 for 57 laps).
For Bahrain 2025 this gives 16,206 candidate strategies (222 one-stop, 15,984 two-stop).

Tyre sets (tyre_sets.py): the simulated car can only fit the sets it has left for the race. OpenF1's stints have no set ID, so tyre_sets.py infers the sets of every car in practice (20) by chaining:
A stint with tyre_age_at_start = 0 takes a new set from the allocation.
A stint that starts used continues the same-compound set that the car's earlier running left at exactly that age. A set's age after a stint is tyre_age_at_start + (lap_end - lap_start), which is how OpenF1 counts it (#44 runs HARD on laps 1-9 in FP1, and the next HARD stint starts at age 8).
Each practice stand-in counts toward its race car (Data Collection step 5, e.g. driver 38 toward car 16).
After each practice session the most-worn sets are returned (TYRE_RETURNS_PER_SESSION = 1, 1, 2), but a new HARD and MEDIUM are always kept (TYRE_RACE_RESERVED). The allocation is TYRE_ALLOCATION = 8 SOFT, 3 MEDIUM, 2 HARD.
All these rules are .env settings and assumptions, they should be checked against the season's sporting regulations.
Across the 20 cars' 275 practice stints, 143 take a new set, 108 match exactly one set, 16 are ambiguous (two sets share a compound and age, so the choice doesn't change the inventory) and 8 are unmatched (no set left at that age, they take a new set at that age). Five cars also run past the MEDIUM or HARD allocation, which adds a set with a warning. Car 30 has no MEDIUM left.
Race inventory of the two Ferraris (set ages in laps):
car 16: SOFT 0, 0, 0, 0, 6, 6 | MEDIUM 0, 0 | HARD 0
car 44: SOFT 0, 0, 4, 5, 6    | MEDIUM 0, 0, 3 | HARD 0
[Limitation] Qualifying is not in the data, so sets used there are counted as still available.

2. Lap time prediction (a fall-off curve per stint)
Every stint is priced from its own set: for each compound and set age that the car has left, the curve is evaluated for stint laps 1..race laps (57) in one model call, and a stint of L laps costs the sum of its first L laps. Tyre age resets to the set's age at each pit stop, so a stint costs the same wherever it starts in the race, and a used set starts further along its curve.
fuel_corrected_pace on stint lap n = race pace level + compound gap + fall_off(compound, set age + n)
fall_off: the model's curve from a new set's first lap (fall_off_curve in feature_config.py, shared with explore/), with the weather at the training medians.
compound gap: stage 3 of train_model.py, model-free (SOFT 0, MEDIUM +1.156 s, HARD +0.295 s on the current data). So the curve is the only place the model's choice enters.
race pace level: the mean of SIM_DRIVER's driver-session levels from stage 3 (94.42 s for car 16), or of every driver-session level when SIM_DRIVER is blank (95.73 s). It is the same for every strategy, so it only shifts the totals, not the ranking.
Nothing is fed back from one lap to the next.
A fuel penalty of 0.035 x (laps of fuel remaining in the race) is added to each race lap a stint covers, so a stint's time is its summed fcp plus the fuel of its race laps.
Per-lap curves are saved to strategy_simulator_stint_curves.csv and the summed fcp per stint length to strategy_simulator_stint_totals.csv. The log has a curve check: fcp and fall-off at stint laps 1, 18 and 57 next to the stage-1 linear fall-off, and the largest step.
Current curves for car 16 (XGBoost, fall-off from a new set's first lap): new SOFT +5.44 s at stint lap 18, then flat to +6.14 s at lap 57, MEDIUM +3.46 → +2.85 s, HARD +5.28 → +4.85 s, the used SOFT (age 6) starts at +2.45 s.
The tyres in training are 2-19 laps old, and a tree can't extrapolate: past age 19 the curve is flat, where the linear model would keep rising (+18.3 s on SOFT at lap 57). The tree's curve also moves in steps of 0.9-1.7 s.

3. Objective function: minimize total race time
For each candidate strategy, total race time (seconds) = sum of predicted lap times of each stint + 22 s per pit stop (PIT_LOSS_S).
This gives a single scalar number for each strategy, allowing direct comparison.
Every race lap is summed exactly once. "Pit on lap p" means the car comes in at the end of lap p: lap p is the in-lap, the last lap of the old stint, and the next stint starts on lap p + 1. The 22 s pit loss covers the in- and out-lap time loss and is added on top of the laps.

4. Strategy optimisation
This is run for the one car in SIM_DRIVER (default 16, an unknown car number raises an error). Strategies that need more sets of a compound than the car has left are skipped (3,996 for car 16, it has one HARD set, so e.g. HARD-HARD-SOFT is out), which leaves 12,210. Each set is fitted at most once, and for each compound sequence and set of pit laps the fastest choice of the car's sets is kept.
Results: scripts/results/strategy_simulator/ (strategy_simulator_all_strategy_results.csv, strategy_simulator_best_strategy.json, strategy_simulator_final_summary.txt).
Current output for SIM_DRIVER = 16 (set ages at the start of each stint in brackets):
SOFT(0)-SOFT(0)-MEDIUM(0), pits on laps 14 and 28, 5625.24 s, tied with SOFT(0)-MEDIUM(0)-SOFT(0) (pits on 14 and 43) and MEDIUM(0)-SOFT(0)-SOFT(0) (29 and 43), the same three stints in another order.
Moving a pit lap by one is 0.11 s slower, the best strategy with the HARD (MEDIUM(0)-SOFT(0)-HARD(0), pits on 29 and 43) 2.35 s, the best 1-stop (SOFT(0)-MEDIUM(0), pit on lap 14) 7.53 s, and the best without the MEDIUM (HARD(0)-SOFT(0)-SOFT(0), pits on 23 and 40) 15.79 s. The used SOFT(6) set is never the best choice of set.
Other cars differ only through their sets and their pace level.
With SIM_DRIVER blank (fresh tyres) the car has a new set of every compound for each stint, so no strategy is skipped (16,206 evaluated) and every set age is 0. On the current data the ranking is the same as car 16's, whose best plans already use only new sets: SOFT(0)-SOFT(0)-MEDIUM(0), pits on laps 14 and 28, 5699.67 s (74.4 s more than car 16 from the slower field pace level alone). The gaps to the alternatives are the same as for car 16 (e.g. the best 1-stop 7.53 s, the best without the MEDIUM 15.79 s).
2 stops win because the curves fall off steeply (3.5-5.4 s by stint lap 18), so fresh tyres are worth a stop.
This is still not a valid recommendation:
The 29-lap MEDIUM stint runs past the oldest tyre in training, where the tree's curve is flat, which flatters long stints.
The compound gap is unstable (on the training split MEDIUM comes out slower than HARD on a new set, and across the CV folds it moves by ~1.4 s).
The pipeline's XGBoost doesn't beat the simple baselines in validation.
[TODO #27, P2, see diagnostic.md] Step 2 (a better compound gap) and Step 4 (choose the fall-off model), the strategy depends on both.

Future work (#27 Step 3): dynamic programming over strategies (race lap, the set on the car, the sets used, the stops made), so that more stops and every set choice can be searched exactly. Optional Genetic Algorithm for deeper search.

EVALUATION AND VALIDATION (validate_strategy.py)
1. Model Validation
5-fold cross-validation grouped by stint (GroupKFold on driver_number + stint_number, shuffled, random_state 42) of the bundle's model, over the 427 laps of the 94 multi-lap stints. It refits a clone of the model per fold, so any model plugged into train_model.py is validated the same way (currently XGBoost). Each fold holds out 18-19 whole stints (64-106 laps), so the score is for stints the model hasn't seen.
Each fold fits its own stage-1 stint levels, input medians and model-free compound gap on its training rows, as train_model.py does with its split. The bundle's are not used here.
What is scored? A held-out stint's level is unknown by construction, so each stint's mean residual is removed and only the shape of the curve within the stint is scored (stint_shape_errors in feature_config.py). Two baselines: no fall-off at all (flat) and the stage-1 slope per compound (linear).
Current result: shape MAE 1.05 s, RMSE 1.57 s (per fold MAE 0.93-1.10 s), against flat 1.03 / 1.54 s and linear 0.90 / 1.39 s.
On train_model.py's held-out split: XGBoost MAE 1.13 s, R2_within 0.13 (the share of the within-stint variation explained), flat 1.20 s, linear 0.95 s, R2_within 0.32.
Significance tests: laps of one stint aren't independent, so each stint is one paired observation: the mean |shape error| of the model against that of a baseline. One-sided paired t-test and Wilcoxon signed-rank test at SIGNIFICANCE_LEVEL (0.05), against each baseline.
Current result: the model is better than flat on 46 of 94 stints (t p = 0.95, Wilcoxon p = 0.93) and than linear on 28 (p = 0.997 / 1.000), so the pipeline's XGBoost (depth 7, untuned) beats neither. The straight line per compound does better, and explore/ compares tuned candidates (ML Modelling step 3). Per-stint values are in validate_strategy_baseline_test_stints.csv, per fold in validate_strategy_cv_fold_metrics.csv.
Residual plot (validate_strategy_residual_plot.png): the out-of-fold within-stint residual against tyre age, by compound. A trend left there is fall-off the model misses.
Curve uncertainty: each fold's fall-off at tyre ages 5, 10 and 15 per compound (validate_strategy_fold_curves.csv). At age 15 it spans SOFT +4.0 to +6.8 s, MEDIUM +2.0 to +5.1 s and HARD +2.3 to +5.2 s, at age 5 MEDIUM straddles 0 (-0.3 to +1.0 s). SOFT falls off most on average (+5.5 s at age 15, against +3.2 / +3.8 s).
The compound gap is logged per fold too: MEDIUM +0.14 to +1.54 s and HARD +0.36 to +1.76 s, so it is poorly determined (#27 Step 2).
This tests the fall-off curve only, not the strategy.
Planned critical checks:
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
Known bugs and open work are documented in diagnostic.md. Open TODOs by priority (IDs as in diagnostic.md, each is tagged [TODO #N, ...] where it applies above):
P1, results are invalid until fixed:  none open
P2, method or validation misleading:  #27 compound gap (Step 2), DP in the simulator (Step 3), model choice (Step 4), reasoning (Step 5)
P3, completeness and cleanup:         #15 results documentation
[TODO #15, P3, see diagnostic.md] Results are not written up yet. Do this once #27 is settled, since the numbers will change.

THANK YOU
DEVAJ RATHORE
NEWLY SELF APPOINTED ML EXPERT
