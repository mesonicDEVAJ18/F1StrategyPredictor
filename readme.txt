F1 RACE STRATEGY OPTIMISER PIPELINE
Requirement for Data Science CSE558 at Indraprastha Institute of Information Technology Delhi

PROBLEM STATEMENT
Use data collected over 3 practice sessions in a non sprint Formula 1 weekend to predict the optimal tyre strategy for a race where atleast
two compounds are supposed to be used (out of 3, hard, medium and soft). 
The problem statement limits us to the three practice sessions only, we can broaden our scope by going out of this limitation but for this project we wont.

DATA COLLECTION
The reference analysis is Bahrain 2025, and the same pipeline runs on any non-sprint 2024 or 2025 weekend: set RACE_YEAR and
RACE_WEEKEND in .env (e.g. 2024 and saudi_arabia; the names are in scripts/race_weekend.py). The model is trained on the whole field, and
the race strategy is simulated for one car, SIM_DRIVER in .env (default 16, Ferrari), using that car's own tyre sets.
Note: the laps, stints and car data cover the whole field (20 drivers per session in Bahrain 2025), and the model is trained on all of them.
All data comes from https://openf1.org/. data/raw/ holds the selected weekend (laps/laps_p1-3.csv, stints/stints_p1-3.csv,
drivers/drivers_p1-3.csv, weather/weather.csv, car_data/car_data_p*_*.csv), and data/cache/<year>_<name>/ keeps every weekend fetched
so far, laid out like data/raw/ (gitignored).

scripts/race_weekend.py is the first step of master.py, and does steps 1-6 below:
- If data/raw/ already holds the complete selected weekend (found from the meeting_key in laps_p1.csv), it does nothing.
- Otherwise it moves data/raw/ into that weekend's cache folder, downloads whatever the selected weekend's cache folder is missing
  (OPENF1_REQUEST_DELAY_S between requests, retrying HTTP 429/5xx and network errors), and copies that folder into data/raw/. A weekend
  fetched before is loaded from the cache with no requests.

Weekend cache (data/cache/, CACHE_DIR in .env):
- One folder per weekend, data/cache/<year>_<name>/ (e.g. 2025_bahrain), with the same laps/, stints/, drivers/, weather/ and car_data/
  subfolders as data/raw/. A weekend is about 70 files and 67-69 MB. The cache is gitignored.
- data/raw/ is only the active weekend, a copy of its cache folder, so no other stage's paths depend on the weekend.
- On a switch, data/raw/ is moved into the cache folder of the weekend it holds, found from the meeting_key in laps_p1.csv (an unknown
  meeting goes to meeting_<key>/, and a data/raw/ without laps_p1.csv to unidentified/). A file the cache already has is kept there and the
  raw copy dropped. The selected weekend's folder is then completed from OpenF1 and copied into data/raw/.
- Downloads are written to <file>.part and renamed when complete, and .part files are never copied into data/raw/, so an interrupted
  run only fetches what is still missing.
- A car-data request with no results (OpenF1 answers HTTP 404) leaves a <file>.nodata marker, which is copied into data/raw/ too, so it
  isn't requested again; delete the marker to retry. Laps, stints, drivers or weather with no results stop the run with an error.
- To force a re-download, delete the file (or the weekend's folder) from data/cache/, and also from data/raw/ if it is the active weekend:
  a complete data/raw/ is left as it is, and a file deleted from data/raw/ only is restored from the cache.
- scripts/results/race_weekend/race_weekend_summary.csv records where each run got its data ("data/raw (unchanged)", "cache", or
  "cache, N file(s) downloaded"), the file counts per folder and any drivers with no car data.

1. Identify the 3 practice sessions ids. They are hardcoded per weekend in race_weekend.py's WEEKENDS table, with the meeting_key and
   the scheduled race laps (the simulator's race distance), from https://api.openf1.org/v1/sessions?year=<year>&session_type=Practice.
   For Bahrain 2025 (meeting 1257):
1	10007	Practice 1	Practice	2025-04-11 11:30	2025-04-11 12:30	1 hr	Friday FP1
2	10008	Practice 2	Practice	2025-04-11 15:00	2025-04-11 16:00	1 hr	Friday FP2
3	10009	Practice 3	Practice	2025-04-12 12:30	2025-04-12 13:30	1 hr	Saturday FP3
Sprint weekends have FP1 only and are rejected.

2. Download all laps' data for the sessions
https://api.openf1.org/v1/laps?session_key=<session_key>&csv=true

3. Download all stints's data for the sessions
https://api.openf1.org/v1/stints?session_key=<session_key>&csv=true

4. Download the weekend's weather related data
https://api.openf1.org/v1/weather?meeting_key=<meeting_key>&csv=true
This covers the whole meeting (practice, qualifying and race); process_weather.py keeps only the practice sessions.

5. Download each session's entry list (driver_number and team_name)
https://api.openf1.org/v1/drivers?session_key=<session_key>&csv=true
tyre_sets.py uses these to find stand-ins. The FP3 lineup is taken as the race cars, and in FP1 and FP2 a driver who is not in it is paired
with the one driver of the same team who sat that session out, so their running counts toward that race car's tyre sets. In Bahrain 2025
six stand-ins ran in FP1 (38 in #16, 34 in #14, 37 in #1, 46 in #55, 50 in #87, 72 in #63), so those six race drivers have no FP1 laps.
The same rule covers a driver replaced before FP3 (Jeddah 2024: Bearman #38 took over Sainz's #55 car, so 55 counts toward 38).
A team that can't be paired one-to-one is logged as a warning and its drivers keep their own numbers.

6. Download the car data for every driver in each session (the driver_numbers in that session's laps_p*.csv)
https://api.openf1.org/v1/car_data?session_key=<session_key>&driver_number=<driver>&csv=true
The driver list is read from the laps files after they are downloaded, so it follows each weekend's field.
Saved as data/raw/car_data/car_data_p<session>_<driver>.csv (CARDATA_DIR in .env; for Bahrain 2025, 20 drivers per session, 60 files,
~66 MB). A driver with no car data leaves a .nodata marker instead, so it isn't requested again.
process_cardata.py cleans, encodes and aggregates the car data per lap (see Data Preparation), and feature_engineer.py merges it as model inputs.

DATA UNDERSTANDING
Laps
| **Feature**           | **Meaning / Real-World Interpretation**                  | **Type**                   | **Example / Units**     | **Analytical Role**                                                                 |
| --------------------- | -------------------------------------------------------- | -------------------------- | ----------------------- | ----------------------------------------------------------------------------------- |
| **date_start**        | UTC timestamp when the lap began.                        | *Datetime*                 | “2024-03-01 T14:23:15Z” | Enables ordering, track-evolution analysis, and time-based joins with weather data. |
| **driver_number**     | Car/driver number for the weekend.                       | *Categorical / identifier* | 16 = Leclerc            | Merge key across tables.                                                            |
| **duration_sector_1** | Time to complete Sector 1.                               | *Numeric (float)*          | 29.374 s                | Used for localized performance analysis (track-segment performance).                |
| **duration_sector_2** | Time to complete Sector 2.                               | *Numeric (float)*          | 38.452 s                | Used for lap-time composition.                                                      |
| **duration_sector_3** | Time to complete Sector 3.                               | *Numeric (float)*          | 26.928 s                | Combined with others to get total lap time.                                         |
| **i1_speed**          | Instantaneous speed at the first intermediate line.      | *Numeric (float)*          | 305 km/h                | Feature for pace modeling; indicates straight-line speed and DRS use.               |
| **i2_speed**          | Instantaneous speed at second intermediate.              | *Numeric (float)*          | 280 km/h                | Captures mid-sector speed differences.                                              |
| **is_pit_out_lap**    | Boolean flag: 1 = lap starts from pit exit.              | *Boolean / binary*         | True / False            | Helps exclude outlaps from analysis.                                                |
| **lap_duration**      | Total lap time (sector1 + 2 + 3).                        | *Numeric (float)*          | 92.841 s                | Basis of fuel_corrected_pace (the target is the next lap's); not a model input.     |
| **lap_number**        | Sequential lap index within the session.                 | *Numeric (int)*            | 1 → 60                  | Enables stint segmentation and degradation trends.                                  |
| **meeting_key**       | Same race-weekend identifier.                            | *Categorical / identifier* | 9161                    | Used for multi-GP aggregation.                                                      |
| **segments_sector_1** | Mini-sector status codes in Sector 1 (2048 yellow, 2049 green, 2051 purple, 2064 pit lane). | *Array / list (int)* | [2049, 2049, 2048] | Replaced by its length (number of mini-sectors), see Encoding.                     |
| **segments_sector_2** | Same for Sector 2.                                       | *Array / list (float)*     | …                       | Optional.                                                                           |
| **segments_sector_3** | Same for Sector 3.                                       | *Array / list (float)*     | …                       | Optional.                                                                           |
| **session_key**       | Session identifier (FP1, FP2, FP3).                      | *Categorical / identifier* | 12345                   | Used to merge with `/stints`.                                                       |
| **st_speed**          | Speed at the speed trap (where top speeds are usually recorded). | *Numeric (float)*  | 310 km/h                | Proxy for DRS usage and straight-line performance.                                  |

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
| **rainfall**          | Whether there is rainfall (flag; always 0 in Bahrain 2025 practice).                                                               | *Binary / Numeric (int)*   | 0 or 1                     | Critical for excluding wet laps or performing wet-vs-dry comparisons.                 |
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
| **drs**           | DRS status code. 0/1 off, 8 eligible in the activation zone, 10/12/14 open; 9, 11, 13, 15 also occur. | *Categorical / Numeric*    | 0, 1, 8-15             | Helps analyze overtaking zones, straight-line speed advantages, and race strategy. |
| **meeting_key**   | Unique identifier for a race weekend (Grand Prix event).                                               | *Categorical / Identifier* | 9161                   | Allows merging telemetry with laps, weather, and stint data for the same event.    |
| **n_gear**        | The currently selected gear in the car’s transmission. `0` represents neutral or no gear engaged.      | *Numeric (int)*            | 0–8                    | Useful for analyzing acceleration patterns, gear shifts, and driver control.       |
| **rpm**           | Engine speed measured as revolutions per minute. Indicates how fast the engine crankshaft is rotating. | *Numeric (int)*            | 11500 rpm              | Used to study engine performance, power delivery, and gear efficiency.             |
| **session_key**   | Unique identifier for the track session (FP1, FP2, FP3, Qualifying, etc.).                             | *Categorical / Identifier* | 12345                  | Enables filtering telemetry for a specific session during a race weekend.          |
| **speed**         | Instantaneous velocity of the car measured by telemetry sensors.                                       | *Numeric (float)*          | 298 km/h               | Critical metric for performance analysis, sector speed comparison, and drag study. |
| **throttle**      | Percentage of maximum engine power being applied by the driver through the throttle pedal.             | *Numeric (float)*          | 0–100 %                | Used to analyze acceleration behavior, power usage, and driver input patterns.     |


Drivers (per-session entry list)
| **Feature**       | **Meaning / Real-World Interpretation**                                                                | **Type**                   | **Example / Units**    | **Analytical Role**                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------ | -------------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| **driver_number** | Number of a driver entered in the session.                                                             | *Categorical / Identifier* | 16, 38                 | Matched against the FP3 lineup to find practice stand-ins.                         |
| **team_name**     | The driver's team in that session.                                                                     | *Categorical*              | Ferrari                | Pairs a stand-in with the team's race driver who sat the session out.              |
| **session_key**   | Session identifier (FP1, FP2, FP3).                                                                    | *Categorical / Identifier* | 10007                  | One file per session (drivers_p1-3.csv).                                           |
(The file has further columns, e.g. full_name and name_acronym; tyre_sets.py only reads driver_number and team_name.)

Relationship Mapping between these 5 raw tables
(These are the natural join keys. What the pipeline actually implements is described under DATA PREPARATION -> Feature Engineering.)
| **Relationship**                     | **Common Keys**                                                                                             | **Description**                                                                            | **Type**                                        | **Usage**                                                                                       |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Laps ↔ Stints**                    | `meeting_key`, `session_key`, `driver_number`, and lap overlap (`lap_number ∈ [lap_start, lap_end]`)        | Links each lap to the tyre compound, stint number, and tyre age used during that lap.      | **One-to-Many** (one stint → many laps)         | Adds tyre compound and degradation information to lap-level performance analysis.               |
| **Laps ↔ Weather**                   | `meeting_key`, `session_key`, and nearest timestamp (`date_start ≈ date`)                                   | Associates each lap with environmental conditions at the time the lap started.             | **Many-to-One (time-based join)**               | Adds air/track temperature, humidity, wind, etc., to lap-level performance modeling.            |
| **Stints ↔ Weather**                 | `meeting_key`, `session_key`, and midpoint timestamp of stint (`mid_stint_time ≈ date`)                     | Connects tyre stints to the weather conditions experienced during the stint duration.      | **Many-to-One (aggregate join)**                | Used to analyze how weather affects tyre degradation and stint strategy.                        |
| **Laps ↔ Car Data**                  | `meeting_key`, `session_key`, `driver_number`, and timestamp alignment (`date_start ≤ date ≤ lap_end_time`) | Links lap records with high-frequency telemetry such as speed, throttle, braking, and RPM. | **One-to-Many** (one lap → many telemetry rows) | Enables detailed performance analysis like braking zones, acceleration patterns, and DRS usage. |
| **Stints ↔ Car Data**                | `meeting_key`, `session_key`, `driver_number`, and timestamp range within stint duration                    | Maps telemetry data to the tyre stint in which it occurred.                                | **One-to-Many**                                 | Allows analysis of driving style and power usage across different tyre compounds.               |
| **Car Data ↔ Weather**               | `meeting_key`, `session_key`, and nearest timestamp (`date ≈ date`)                                         | Aligns telemetry samples with the weather conditions at that moment.                       | **Many-to-One**                                 | Enables studies of how environmental factors affect speed, throttle usage, and engine RPM.      |
| **Drivers ↔ Laps / Stints**         | `session_key`, `driver_number`                                                                              | Gives each driver's team in that session.                                                  | **One-to-Many** (one entry → many laps/stints)  | Maps practice stand-ins to their race car for the tyre-set inventory (tyre_sets.py).            |
| **Shared Identifiers (Global Keys)** | `meeting_key`, `session_key`, `driver_number`                                                               | Common identifiers present across all datasets ensuring consistent merging.                | —                                               | Allows creation of a unified dataset combining laps, stints, weather, and telemetry data.       |


RUNNING THE PIPELINE
Python 3.10, dependencies pinned in requirements.txt (UTF-8; the versions in .venv that produced the committed results):
    pip install -r requirements.txt
All scripts live in scripts/ and use relative paths, so run them from inside scripts/:

    cd scripts
    python master.py

master.py runs race_weekend.py -> process_laps.py -> process_stints.py -> tyre_sets.py -> process_weather.py -> process_cardata.py -> feature_engineer.py -> train_model.py
-> strategy_simulator.py -> validate_strategy.py, and stops at the first script that fails.
Every script writes a log, a manifest and its intermediate outputs to scripts/results/<script_name>/.
Settings (the race weekend, paths, file names, cleaning thresholds, model hyperparameters, seeds and race parameters such as PIT_LOSS_S)
live in .env at the repo root and are read through scripts/env_config.py. An environment variable of the same name overrides .env, e.g.
    $env:RACE_YEAR = "2024"; $env:RACE_WEEKEND = "saudi_arabia"; python master.py
runs the whole pipeline on Jeddah 2024. The race laps come from the weekend table in race_weekend.py.
A weekend fetched before loads from data/cache/ with no requests (see Data Collection, Weekend cache). data/raw/, data/processed/ and
scripts/results/ are tracked in git, so running another weekend rewrites them; switching back to 2025 bahrain restores the committed
data from the cache, and the results reproduce apart from the placeholder t-test and the RandomForest's last digits.
Stages hand data to each other through CSVs in data/processed/:
    race_weekend.py      -> data/raw/ (the selected weekend, from data/cache/ or OpenF1)
    process_laps.py      -> data/processed/laps_combined_clean.csv
    process_stints.py    -> data/processed/stints_combined_clean.csv
    tyre_sets.py         -> data/processed/tyre_sets_race_inventory.csv (read by the simulator)
    process_weather.py   -> data/processed/cleaned_weather.csv
    process_cardata.py   -> data/processed/cardata_per_lap_clean.csv
    feature_engineer.py  -> data/processed/final_engineered_dataset.csv
    train_model.py       -> scripts/results/train_model/train_model_xgb_model.pkl (read by the simulator and validation)

DATA PREPARATION
Scripts: process_laps.py, process_stints.py, process_weather.py, process_cardata.py
1. Remove redundant data from all of the data types
These scripts remove meeting keys, data points irrelevant to the pipeline. process_laps.py and process_stints.py keep session_key, because
lap numbers and stint numbers restart every session and feature_engineer.py needs it to assign each lap to its own session's stint.
process_weather.py drops session_key after filtering, since weather is joined on timestamps.
process_laps.py also tags every lap with session_name (FP1, FP2 or FP3, read from the laps_p1/p2/p3 file name), which is encoded as a model
input (see Encoding). The name follows practice order, so it means the same on every weekend, unlike session_key.
Also process_laps.py removes pit-out laps (is_pit_out_lap = True), i.e. laps that start from the pit exit. There is no explicit in-lap flag in the
data; slow in-laps are caught by the non-push filter below.

2. Remove outliers
Laps (each session file is cleaned separately)
Removed laps with missing sector times and sector times of 0 or below. Also removed laps with unrealistic lap times (duration <= 30s or >= 2x the median lap time of that session).
Also removed laps where i1_speed, i2_speed or st_speed is 50 km/h or less, indicating aborted laps (due to whatever reason).
Removed laps whose residuals fall outside [Q1 - 3xIQR, Q3 + 3xIQR], where residual = lap_duration - (5-lap centred rolling median of that driver's lap_duration). A rolling median compensates for the track improving across a single session.
Finally, removed non-push laps (cool-down, in- and traffic laps): laps slower than 1.15x the driver's best lap in that session
(PUSH_LAP_MAX_BEST_RATIO in .env). Before this filter, laps divided by the driver's session best split into two groups: push laps and long
runs at <= 1.10x, and cool-down laps from about 1.3x, with a gap around 1.14-1.16x. The compound isn't known yet at this stage (it comes from
the stints), but the gap between compounds (~1%) is well inside the margin. This removes 145 / 109 / 118 laps in FP1 / FP2 / FP3 and leaves
467 laps (median 95-99 s, 90th percentile 100-102 s per session).

Stints
No continuous telemetry available, prompting use of logical checks only, removed rows where lap_start > lap_end, negative tyre_age_at_start, and missing compound.

Weather
Kept only the practice sessions (the session_keys found in laps_p1-3.csv); qualifying and race weather are dropped.
Applied Z-score filtering (|z| > 3) on continuous atmospheric readings: Air temperature, Track temperature, Humidity, Pressure, Wind speed and Rainfall.
Columns with zero variance are skipped.

CarData
process_cardata.py keeps session_key, because it needs it to match samples to laps. Each car_data_p*_*.csv file is cleaned separately:
Removed duplicate timestamps.
Removed samples where brake or throttle read 104. This value means there is no valid input, and these samples are always at about 0 km/h in
neutral (the car sitting in the garage). This is roughly a third of all samples.
Removed physically impossible readings: speed outside 0-400 km/h, rpm outside 0-16000, gear outside 0-8, throttle outside 0-100, brake not 0 or 100.
Each ~4.2 Hz sample is then assigned to the lap whose window [date_start, date_start + lap_duration) contains it, matched on session_key and
driver_number. The windows come from the raw laps_p*.csv files, so every lap has one, including the laps process_laps.py drops (pit-out,
invalid, outliers); each sample is matched to the lap it was actually driven in.
Samples outside every lap (pit lane, garage, laps with no lap time) are dropped.
The samples of each lap are aggregated to mean/max speed, mean throttle, share of samples at full throttle (>= 99%), share braking,
mean/max rpm, mean gear, gear changes and share with DRS open.
Lap-level filters, in order:
- Laps with fewer than half the expected samples (lap_duration x sample rate) are dropped, so gappy telemetry doesn't give misleading means.
- Only laps that survive process_laps.py are kept (matched on driver_number + date_start against data/processed/laps_combined_clean.csv). This removes
  pit-out, invalid and outlier laps, so the car data describes the same laps as the lap data.
- Z-score filtering (|z| > 3, per session) on the aggregates, except DRS share, which is mostly 0, so any DRS use would look like an outlier.
The cleaned samples of the kept laps are saved per input file (scripts/results/process_cardata/process_cardata_cleaned_car_data_p*_*.csv).
The output has 448 laps (all 20 drivers in every session, 1-15 per driver and session), saved to data/processed/cardata_per_lap_clean.csv with
session_key, driver_number and lap_number as join keys. lap_duration is dropped from it, because it is the dependent variable.

3. Encoding
Laps
segments_sector_1/2/3 arrive as lists of mini-sector status codes; each list is replaced by its length (the number of mini-sectors).
Driver number was kept numerical as it was used to combine with stints data and has no signifance as its never fed to the model.
Driver number is not a model input, as it was decided to not build a driver specific model. It is kept in the engineered dataset as an
identifier only, so the next-lap target can be built per driver.
session_name is one-hot encoded (in feature_engineer.py) -> session_FP1, session_FP2, session_FP3 (SESSION_COLUMNS in feature_config.py).
These are model inputs: they let the model separate session-level pace offsets (track evolution, run programmes, time of day) from the
effects of tyres and conditions. Lap t and lap t+1 are always in the same session, so this leaks nothing. session_key stays an identifier.

Stints
One-hot encoded compound -> compound_HARD, compound_MEDIUM, compound_SOFT (only dry compounds occur in this weekend's data).

Weather
None needed.

CarData
brake (0 or 100) -> braking (0/1). drs status codes -> drs_open (1 for 10, 12 or 14, i.e. flap open; 0 otherwise).
n_gear kept as an ordinal number.

4. Normalization
Laps
Not scaled. lap_duration, the sector times and the speed traps stay in physical units (seconds, km/h).
Linear Regression, Random Forest and XGBoost give the same fit whether or not a column is rescaled with one mean/std. Trees split on the order
of values, and linear regression just rescales its coefficients. So a global scaler would change only the units. A per-session scaler would
remove each session's pace level, but it would put the target in z-scores, and the race has no session scaler to convert predictions
back with, while the fuel and pit penalties are in seconds. So fuel_corrected_pace, every prediction, the simulator and the error metrics
are all in seconds, and session-level differences are handled by the session one-hot.

Stints
No normalization applied, stints contain mostly integer metadata (stint number, lap ranges).

Weather
Applied StandardScaler to: Air temperature, Track temperature, Humidity, Pressure, Wind speed, and Rainfall.
Wind direction kept raw due to its circular nature.

CarData
Not scaled, for the same reason as the laps: the per-lap aggregates are model inputs and stay in physical units (km/h, %, rpm, gear, shares),
so the training medians the simulator starts from are real values (e.g. mean speed 195 km/h).

5. Feature Engineering (feature_engineer.py)
Stint number: stints are sorted by driver, session and per-session stint_number, and numbered 1, 2, 3 ... per driver across all three sessions
(global_stint_number), kept session independent to include evolution information too.
Each lap is assigned the global number of the stint from the same session (session_key) and driver whose [lap_start, lap_end] contains its
lap_number. Lap numbers restart every session, so the session has to match as well.

Stint metadata (lap_start, lap_end, tyre_age_at_start, compound) is then merged onto the laps on driver_number + global stint number, which is
unique per driver, so the merge is one-to-one: all 467 cleaned laps get a stint and keep exactly one row.
session_key and driver_number are kept in the final dataset as identifiers (not model inputs).

Weather alignment: each lap is joined to the weather sample nearest its date_start (merge_asof on date_start ~ date, 5 minute tolerance).
Weather is sampled about once a minute and the sessions are hours apart, so the match always comes from the lap's own session
(on the current data every lap matches, within about 30 s).

Car data: each lap gets its own per-lap car-data aggregates (CARDATA_FEATURE_COLUMNS in feature_config.py), left-merged on session_key,
driver_number and lap_number (one-to-one). They belong to lap t and the target is lap t+1, so nothing leaks. 448 of the 467 laps have car
data; the other 19 (telemetry dropped as outliers) are filled with the training medians, as other missing inputs are.

Session one-hot: session_name -> session_FP1/FP2/FP3 (see Encoding); session_name itself is then dropped.

laps_since_stint_start = lap_number - lap_start + 1 (clipped to at least 1)

tyre_age = tyre_age_at_start + laps_since_stint_start: the real age of the set, i.e. the laps it has done including lap t (1 on the first lap
of a new set). laps_since_stint_start alone can't tell a stint on a new set from one on a set already used earlier in the weekend.
On the current data it runs from 2 to 19 laps.

stint_avg_pace = mean(lap_duration of the earlier laps of that driver & stint), in seconds (an expanding mean of the stint so far).
It is NaN on a stint's first lap (223 rows), which training fills with the column median, the same value the simulator seeds lap 1 with.
Only laps before t are used: a leave-one-out mean over the whole stint would include lap t+1, whose lap_duration is part of the target.

fuel_corrected_pace: laps_remaining = lap_end - lap_number + 1 (laps left in the stint, counting lap t) and
fuel_corrected_pace = lap_duration - k*laps_remaining (k = 0.035 s per lap), in seconds. More fuel makes a lap slower, so the fuel effect is
subtracted to leave fuel-free pace.
Assumption: practice fuel loads are not recorded, so the laps left in the stint stand in for the fuel on board at the start of lap t (each
run fuelled for its own length). The race fuel term in the simulator uses the same convention (laps left in the race, counting the lap), so
race fuel is added once, in the simulator only.
The current lap's fuel_corrected_pace is not a model input; the NEXT lap's value is the model's target.

Finally date_start, date, lap_duration and laps_remaining are dropped, and the result is saved as data/processed/final_engineered_dataset.csv.
session_key and driver_number stay in it as identifiers for building the target; the models only read the columns listed in feature_config.py.

track evolution was not chosen as an engineered parameter, as it would've been a derived dependant value that would not have contributed to the model.

EXPLORATORY DATA ANALYSIS AND STATISTICAL Inference
[TODO #25, P3, see diagnostic.md] Not implemented yet: no stage looks at the data (distributions, pace against tyre age, correlations,
row coverage) before it is modelled.
[TODO #26, P3, see diagnostic.md] Not implemented yet: no test on the practice laps of compound pace differences, degradation slopes or
session pace. The only test is the placeholder t-test (#5).

ML MODELLING TO COMPUTE DEGRADATION RATE (train_model.py)
1. Remove dependant variables to avoid leakage
lap_duration = duration_sector_1 + duration_sector_2 + duration_sector_3, and was therefore removed as it is clearly a dependant variable.
The sector durations of lap t are kept as inputs, since the target is lap t+1.
The segments_sector_* counts were kept as they may have something to contribute in the model.

laps_remaining = lap_end - lap_number + 1, used to compute/approximate fuel_corrected_pace, not expected for the model.

The 36 model inputs are defined once in scripts/feature_config.py (MODEL_FEATURE_COLUMNS):
sector durations (3, s), i1/i2/st speed (3, km/h), segment counts (3), tyre_age_at_start, compound one-hots (3),
session one-hots (3: session_FP1/FP2/FP3), weather (7), car-data aggregates (10: mean/max speed, mean throttle, full-throttle share,
braking share, mean/max rpm, mean gear, gear changes, DRS share), laps_since_stint_start, tyre_age and stint_avg_pace (s).

2. What are we predicting?
Instead of explicitly computing a "degradation rate" label (which becomes circular and leak-prone), the model is trained to predict:
fuel_corrected_pace on the NEXT lap

Fully causal:

All inputs belong to lap t
The model learns how tyre, weather, stint age, and track conditions evolve into lap t+1
No sneaky future-lap info (no target leakage)
Designed for autoregressive race simulation (rolling predictions forward lap by lap)
Degradation is then obtained as an emergent property:

Training: the target is built by build_next_lap_target() in feature_config.py (shared by train_model.py and validate_strategy.py).
fuel_corrected_pace is shifted back one lap within each driver_number + stint_number group (stint_number is the global per-driver number,
so a group is one stint in one session), and a row is kept only if the next row is exactly lap_number + 1.
This drops the last lap of every stint (223 rows) and laps followed by a gap left by cleaning (91 rows): 467 -> 153 rows.
Practice runs alternate push and cool-down laps, so once the cool-down laps are cleaned out most lap + 1 pairs are broken; the 153 rows
come from 50 stints, mostly FP1 and FP2 long runs (FP3 has 7 rows).
Missing feature values are filled with the column median, then an 80/20 shuffled train/test split (random_state 42) is used.
The medians and each session's share of the target rows (FP1 0.294, FP2 0.660, FP3 0.046) are saved with the model for the simulator.
[TODO #18, P2, see diagnostic.md] The split shuffles single laps, so most stints are in both train and test. Split by stint instead.

3. Model choice
train_model.py fits and compares Linear Regression, Random Forest (200 trees) and XGBoost (300 trees, learning rate 0.05, max depth 7,
subsample 0.8, colsample 0.8), and saves the XGBoost model as train_model_xgb_model.pkl, a joblib bundle with its feature list, the
training medians and the session weights ({"model", "features", "medians", "session_weights"}).
K-Means was ruled out because it provides no predictive capability (only grouping), so it is not part of the pipeline.

Reasoning:
Linear Regression underfits non-linear tyre behaviour (tyre warm-up phase, thermal degradation, cliffing, etc.).
Random Forest produces noisy, stepwise predictions -> unstable in autoregressive simulations.
XGBoost hits the sweet spot ->
non-linear learning, stable rollouts, fast inference, and direct interpretability through feature importance.

Held-out test results on the current data (MAE and RMSE in seconds):
| Model             | MAE (s) | RMSE (s) | R2    |
| LinearRegression  | 1.44    | 1.90     | 0.308 |
| RandomForest      | 1.20    | 1.68     | 0.460 |
| XGBoost           | 1.07    | 1.60     | 0.510 |
XGBoost is the best of the three on this split, and is kept for the reasons above. The test set is only 31 rows, so with 36 inputs the
differences between the models are close to noise.
XGBoost's top inputs are stint_avg_pace (0.145), i2_speed (0.100), mean_gear (0.080) and duration_sector_2 (0.066); tyre_age is 7th (0.039)
and track_temperature 21st. The 10 car-data inputs carry 0.249 of XGB's importance in total.
Random Forest is led by duration_sector_2 (0.201) and i2_speed (0.153).
[TODO #13, P2, see diagnostic.md] Re-run the comparison once #18 is fixed and confirm (or change) the saved model.
[TODO #27, P2, see diagnostic.md] Experiment more: naive baselines, more candidate models, hyperparameter tuning, smaller feature sets and
a rollout comparison of the candidates. Then update the reasoning above to match: the claims about Random Forest's unstable rollouts and
XGBoost's stable ones haven't been tested, since only XGBoost is ever rolled out.

This choice also aligns with how real-world motorsport predictive pipelines are structured:
"predict the next lap's corrected pace and let degradation emerge naturally from repeated predictions."

RACE SIMULATION AND OPTIMISATION (strategy_simulator.py)
1. Generate the strategy space
The race length is the weekend's scheduled race laps from race_weekend.py (57 for Bahrain 2025). Strategies are enumerated by brute force over:
1-stop and 2-stop structures
Every compound sequence for that structure that uses at least two different compounds (the dry-race rule; repeats allowed, e.g.
HARD-HARD-SOFT). Single-compound sequences are dropped, and so are 0-stop strategies, since one stint can only use one compound.
Every combination of pit laps from lap 10 to 11 laps before the end (PIT_WINDOW_START_LAP, PIT_WINDOW_END_BUFFER; laps 10-46 for 57 laps)
For Bahrain 2025 this gives 16,206 candidate strategies (222 one-stop, 15,984 two-stop). The stop counts and the minimum number of compounds are
STOP_COUNTS and MIN_COMPOUNDS_USED in .env.

Tyre sets (tyre_sets.py): the simulated car (SIM_DRIVER) can only fit the sets it has left for the race. tyre_sets.py infers the sets of
every car in practice (20), and the simulator picks SIM_DRIVER's; a car number with no inventory raises an error. OpenF1's stints have no set ID, so the sets are inferred
from practice by chaining. A stint with tyre_age_at_start = 0 takes a new set from the allocation. A stint that starts used continues the
same-compound set that the car's earlier running left at exactly that age. A set's age after a stint is tyre_age_at_start + (lap_end -
lap_start), which is how OpenF1 counts it (#44 runs HARD on laps 1-9 in FP1, and the next HARD stint starts at age 8). Each practice stand-in counts
toward its race car (derived from the drivers files, Data Collection step 5; e.g. driver 38 toward car 16). After each practice session the most-worn sets are returned (TYRE_RETURNS_PER_SESSION = 1, 1, 2), but a
TYRE_RACE_RESERVED new HARD and MEDIUM are always kept. The allocation is TYRE_ALLOCATION = 8 SOFT, 3 MEDIUM, 2 HARD.
All these rules are .env settings and assumptions; check them against the season's sporting regulations.
Across the 20 cars' 275 practice stints, 143 take a new set, 108 match exactly one set, 16 are ambiguous (two sets share a compound and
age, so the choice doesn't change the inventory) and 8 are unmatched (no set left at that age; they take a new set at that age, see the
tyre_sets log). Five cars also run past the MEDIUM or HARD allocation, which adds a set with a warning. Car 30 has no MEDIUM left.
Race inventory of the two Ferraris (set ages in laps):
car 16: SOFT 0, 0, 0, 0, 6, 6 | MEDIUM 0, 0 | HARD 0
car 44: SOFT 0, 0, 4, 5, 6    | MEDIUM 0, 0, 3 | HARD 0
[Limitation] Qualifying is not in the data, so sets used there are counted as still available.

2. Lap time prediction (autoregressive stint rollout)
Every stint is priced from its own set: for each compound and set age that a car has left, and every stint length L = 1..race laps (57), the stint is
rolled forward from stint lap 1. Tyre age resets to the set's age at each pit stop, and a stint costs the same wherever it starts in the race.
Lap 1 of a stint is the seed: the training medians (saved in the model bundle, because the inputs are in physical units, so 0 is not
"average") with the compound one-hot set, tyre_age_at_start = the set's age and tyre_age = set age + 1. Its lap time is the sum of the
median sector times (99.15 s).
From then on the model predicts lap a+1's fuel_corrected_pace from lap a's inputs, exactly the relationship it was trained on. The race is
not one of the practice sessions, so each lap is predicted once per session one-hot (session_FP1, FP2, FP3) and the three predictions are
averaged, weighted by each session's share of the training rows (FP1 0.294, FP2 0.660, FP3 0.046).
Each prediction is fed back as the next lap's inputs:
- it is turned into a raw lap time by adding back the training fuel term: raw = fcp + 0.035 x (L - age + 1), the laps left in the
  stint counting this one (the seed's fcp is 99.15 - 0.035 x L)
  (k is FUEL_EFFECT_K in feature_config.py, shared with feature_engineer.py)
- the sector times are the median sector shares scaled to sum to that raw lap time
- stint_avg_pace is the mean of the stint's earlier raw laps (the same causal definition as in training)
- laps_since_stint_start and tyre_age go up by one
Speeds, segment counts, weather and the car-data aggregates stay at their medians.
In the training rows, laps_since_stint_start runs from 2 to 13 (pit-out laps are cleaned out) and tyre_age from 2 to 18. Beyond these, the
model's response to tyre age is flat, so changes later in a long stint come from the fed-back inputs only.
Predictions stay within 97-102 s. Every compound and set age predicts about 99.5 s on stint lap 2 and about 101 s on lap 3, then stays at
100.5-101.5 s to the end of the stint, all within about 0.4 s of each other. So the model has learnt almost no compound or tyre-age effect from
the 153 training rows.
Per-lap rollouts are saved to scripts/results/strategy_simulator/strategy_simulator_stint_rollouts.csv, and the summed fcp per compound,
set age and stint length to strategy_simulator_stint_totals.csv.
[TODO #12, P3, see diagnostic.md] The starting-compound setting in start_features is overwritten and has no effect.
A fuel penalty of 0.035 x (laps of fuel remaining in the race) is added to each race lap a stint covers, so a stint's time is its summed
rollout fcp plus the fuel of its race laps.

3. Objective function: minimize total race time
For each candidate strategy, total race time (seconds) = sum of predicted lap times of each stint + 22 s per pit stop.
This gives a single scalar number for each strategy, allowing direct comparison.
Every race lap is summed exactly once. "Pit on lap p" means the car comes in at the end of lap p: lap p is the in-lap, the last lap of
the old stint, and the next stint starts on lap p + 1. The 22 s pit loss (PIT_LOSS_S) covers the in- and out-lap time loss and is added on
top of the laps.

4. Strategy optimisation
This is run for the one car in SIM_DRIVER (default 16). Strategies that need more sets of a compound than the car has left are skipped
(3,996 for car 16: it has one HARD set, so e.g. HARD-HARD-SOFT is out), which leaves 12,210. Each set is fitted at most once. For each
compound sequence and set of pit laps, the fastest choice of the car's sets is kept.
Results: scripts/results/strategy_simulator/ (strategy_simulator_all_strategy_results.csv with car_number and set_ages,
strategy_simulator_best_strategy.json, strategy_simulator_final_summary.txt).
Current output for SIM_DRIVER = 16 (set ages at the start of each stint in brackets):
HARD(0)-SOFT(6), pit on lap 45, 5840.85 s (ties with SOFT(6)-HARD(0), pit on lap 12: the same two stints in reverse order).
Neighbouring pit laps are 0.24 s slower, and the best 2-stop (SOFT(0)-SOFT(0)-HARD(0), pits on 16 and 34) is 13.75 s slower.
Other cars differ only through their sets, e.g. car 44: SOFT(4)-HARD(0), pit on lap 14, 5840.39 s; car 30 (no MEDIUM left, 2,072
strategies feasible): SOFT(3)-HARD(0), pit on lap 10, 5839.87 s.
Every lap carries the same race fuel term whatever the strategy, so a total depends only on which stints (compound, set age, length) it uses.
1 stop wins because the rollouts are almost flat (step 2), so an extra stop can't win back its 22 s. The top strategies are within
about 0.3 s, and the compounds barely differ, so the pit lap and compound choice are close to arbitrary. Every new stint also starts with
a fast median seed lap, which narrows the gap to 2 stops. None of this is tyre physics yet.
[TODO #18, P2, see diagnostic.md] The metrics behind this model come from a shuffled split, so this is not a valid recommendation yet.

Future work: Dynamic Programming to prune inferior strategies and an optional Genetic Algorithm for deeper search across larger sequences.

EVALUATION AND VALIDATION (validate_strategy.py)
1. Model Validation
Implemented:
5-fold shuffled cross-validation (random_state 42) of the XGBoost configuration on next-lap fuel-corrected pace.
Missing inputs (stint_avg_pace on a stint's first lap) are filled with the training medians from the model bundle, as in train_model.py.
Current result: CV MAE 1.13 s, CV RMSE 1.60 s.
[TODO #18, P2, see diagnostic.md] The folds shuffle single laps, so a stint's laps land in both train and test folds. Use GroupKFold by stint.
Residual plot (predicted vs residual) to check for systematic bias.
[TODO #6, P2, see diagnostic.md] The plot uses the last fold's model on all rows, including its training rows. Use out-of-fold predictions.
Feature importance from Random Forest and XGBoost, saved by train_model.py.
An example two-sample t-test on randomly generated data. This is a placeholder: it does not test anything about the model or the
strategy, and its p-value changes from run to run.
[TODO #5, P2, see diagnostic.md] Replace with a meaningful test or remove it.

Rollout check (logged by strategy_simulator.py on every run): for a full-race-length stint of each compound and set age SIM_DRIVER has,
fcp at stint laps 1, 18 and the last lap, and the largest lap-to-lap step.
Current result for car 16 (57 laps): SOFT new 97.15 -> 101.03 -> 101.39, SOFT age 6 97.15 -> 101.05 -> 101.41,
MEDIUM new 97.15 -> 101.12 -> 101.49, HARD new 97.15 -> 101.39 -> 101.37 s.
Lap 1 is the seed's fcp (99.15 s raw minus 57 laps of fuel). The largest step is 2.28-2.60 s, on stint laps 1-3.
No runaway drift: every rollout stays within 97.15-101.54 s.
Smooth but nearly flat: after about 4 s over the first three laps (mostly the median seed lap being fast), every rollout gains well under
1 s over the rest of the stint, and compounds and set ages differ by at most about 0.4 s. That is too little to be tyre physics: the model has
learnt little about tyres from 153 training rows, and #18 is still open.

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
Known bugs and their effect on the results are documented in diagnostic.md.
Open TODOs by priority (IDs as in diagnostic.md; each is tagged [TODO #N, ...] where it applies above):
P1, results are invalid until fixed:  none open
P2, method or validation misleading:  #18 splits mix laps of one stint, #6 residual plot leakage, #5 placeholder t-test,
                                      #13 model choice, #27 model experiments and reasoning
P3, completeness and cleanup:         #12 dead starting-compound setting, #25 no EDA, #26 no statistical inference,
                                      #15 results documentation
[TODO #15, P3, see diagnostic.md] Results are not written up yet. Do this once #18 and #13 are settled, since the numbers will change.

THANK YOU
DEVAJ RATHORE
NEWLY SELF APPOINTED ML EXPERT
