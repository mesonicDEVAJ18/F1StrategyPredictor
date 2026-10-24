import pandas as pd
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from env_config import env_str, env_int, env_float, env_list

# Puts the race weekend chosen in .env (RACE_YEAR, RACE_WEEKEND) into data/raw/, from the weekend
# cache in data/cache/ or, for anything missing there, from the OpenF1 API (diagnostic.md #24).
# Other scripts import selected_weekend() for the weekend's constants; importing has no side effects.
SCRIPT_NAME = "race_weekend"
RESULTS_DIR = f"{env_str('RESULTS_ROOT')}/{SCRIPT_NAME}"


# ============================================================
# RACE WEEKENDS (non-sprint, FP1-FP3)
# ============================================================
# (year, name): OpenF1 meeting_key, Practice 1-3 session_keys, race laps.
# Keys from OpenF1 /v1/sessions?session_type=Practice and /v1/meetings; race laps are the
# scheduled distance, checked against the highest race lap in OpenF1 (2025 Austria ran 70).
# Session keys rise FP1 -> FP3 on every weekend, which tyre_sets.py relies on.
WEEKENDS = {
    (2024, "bahrain"):        {"name": "Bahrain Grand Prix",        "meeting_key": 1229, "sessions": (9465, 9466, 9467),    "race_laps": 57},
    (2024, "saudi_arabia"):   {"name": "Saudi Arabian Grand Prix",  "meeting_key": 1230, "sessions": (9473, 9474, 9475),    "race_laps": 50},
    (2024, "australia"):      {"name": "Australian Grand Prix",     "meeting_key": 1231, "sessions": (9481, 9482, 9483),    "race_laps": 58},
    (2024, "japan"):          {"name": "Japanese Grand Prix",       "meeting_key": 1232, "sessions": (9489, 9490, 9491),    "race_laps": 53},
    (2024, "emilia_romagna"): {"name": "Emilia Romagna Grand Prix", "meeting_key": 1235, "sessions": (9508, 9509, 9510),    "race_laps": 63},
    (2024, "monaco"):         {"name": "Monaco Grand Prix",         "meeting_key": 1236, "sessions": (9516, 9517, 9518),    "race_laps": 78},
    (2024, "canada"):         {"name": "Canadian Grand Prix",       "meeting_key": 1237, "sessions": (9524, 9525, 9526),    "race_laps": 70},
    (2024, "spain"):          {"name": "Spanish Grand Prix",        "meeting_key": 1238, "sessions": (9532, 9533, 9534),    "race_laps": 66},
    (2024, "britain"):        {"name": "British Grand Prix",        "meeting_key": 1240, "sessions": (9551, 9552, 9553),    "race_laps": 52},
    (2024, "hungary"):        {"name": "Hungarian Grand Prix",      "meeting_key": 1241, "sessions": (9559, 9560, 9561),    "race_laps": 70},
    (2024, "belgium"):        {"name": "Belgian Grand Prix",        "meeting_key": 1242, "sessions": (9567, 9568, 9569),    "race_laps": 44},
    (2024, "netherlands"):    {"name": "Dutch Grand Prix",          "meeting_key": 1243, "sessions": (9575, 9576, 9577),    "race_laps": 72},
    (2024, "italy"):          {"name": "Italian Grand Prix",        "meeting_key": 1244, "sessions": (9583, 9584, 9585),    "race_laps": 53},
    (2024, "azerbaijan"):     {"name": "Azerbaijan Grand Prix",     "meeting_key": 1245, "sessions": (9591, 9592, 9593),    "race_laps": 51},
    (2024, "singapore"):      {"name": "Singapore Grand Prix",      "meeting_key": 1246, "sessions": (9599, 9600, 9601),    "race_laps": 62},
    (2024, "mexico_city"):    {"name": "Mexico City Grand Prix",    "meeting_key": 1248, "sessions": (9618, 9619, 9620),    "race_laps": 71},
    (2024, "las_vegas"):      {"name": "Las Vegas Grand Prix",      "meeting_key": 1250, "sessions": (9637, 9638, 9639),    "race_laps": 50},
    (2024, "abu_dhabi"):      {"name": "Abu Dhabi Grand Prix",      "meeting_key": 1252, "sessions": (9461, 9656, 9657),    "race_laps": 58},
    (2025, "australia"):      {"name": "Australian Grand Prix",     "meeting_key": 1254, "sessions": (9686, 9687, 9688),    "race_laps": 58},
    (2025, "japan"):          {"name": "Japanese Grand Prix",       "meeting_key": 1256, "sessions": (9999, 10000, 10001),  "race_laps": 53},
    (2025, "bahrain"):        {"name": "Bahrain Grand Prix",        "meeting_key": 1257, "sessions": (10007, 10008, 10009), "race_laps": 57},
    (2025, "saudi_arabia"):   {"name": "Saudi Arabian Grand Prix",  "meeting_key": 1258, "sessions": (10015, 10016, 10017), "race_laps": 50},
    (2025, "emilia_romagna"): {"name": "Emilia Romagna Grand Prix", "meeting_key": 1260, "sessions": (9980, 9981, 9982),    "race_laps": 63},
    (2025, "monaco"):         {"name": "Monaco Grand Prix",         "meeting_key": 1261, "sessions": (9972, 9973, 9974),    "race_laps": 78},
    (2025, "spain"):          {"name": "Spanish Grand Prix",        "meeting_key": 1262, "sessions": (9964, 9965, 9966),    "race_laps": 66},
    (2025, "canada"):         {"name": "Canadian Grand Prix",       "meeting_key": 1263, "sessions": (9956, 9957, 9958),    "race_laps": 70},
    (2025, "austria"):        {"name": "Austrian Grand Prix",       "meeting_key": 1264, "sessions": (9948, 9949, 9950),    "race_laps": 70},
    (2025, "britain"):        {"name": "British Grand Prix",        "meeting_key": 1277, "sessions": (9940, 9941, 9942),    "race_laps": 52},
    (2025, "hungary"):        {"name": "Hungarian Grand Prix",      "meeting_key": 1266, "sessions": (9921, 9922, 9923),    "race_laps": 70},
    (2025, "netherlands"):    {"name": "Dutch Grand Prix",          "meeting_key": 1267, "sessions": (9913, 9914, 9915),    "race_laps": 72},
    (2025, "italy"):          {"name": "Italian Grand Prix",        "meeting_key": 1268, "sessions": (9905, 9906, 9907),    "race_laps": 53},
    (2025, "azerbaijan"):     {"name": "Azerbaijan Grand Prix",     "meeting_key": 1269, "sessions": (9897, 9898, 9899),    "race_laps": 51},
    (2025, "singapore"):      {"name": "Singapore Grand Prix",      "meeting_key": 1270, "sessions": (9889, 9890, 9891),    "race_laps": 62},
    (2025, "mexico_city"):    {"name": "Mexico City Grand Prix",    "meeting_key": 1272, "sessions": (9870, 9871, 9872),    "race_laps": 71},
    (2025, "las_vegas"):      {"name": "Las Vegas Grand Prix",      "meeting_key": 1274, "sessions": (9851, 9852, 9853),    "race_laps": 50},
    (2025, "abu_dhabi"):      {"name": "Abu Dhabi Grand Prix",      "meeting_key": 1276, "sessions": (9832, 9833, 9834),    "race_laps": 58},
}

# Sprint weekends have FP1 only, so the pipeline can't run on them
SPRINT_WEEKENDS = {
    (2024, "china"), (2024, "miami"), (2024, "austria"), (2024, "united_states"), (2024, "sao_paulo"), (2024, "qatar"),
    (2025, "china"), (2025, "miami"), (2025, "belgium"), (2025, "united_states"), (2025, "sao_paulo"), (2025, "qatar"),
}


def weekend_id(year, name):
    """Cache folder name, e.g. 2025_bahrain."""
    return f"{year}_{name}"


def selected_weekend():
    """The weekend in RACE_YEAR / RACE_WEEKEND, with its id, meeting key, session keys and race laps."""
    year = env_int("RACE_YEAR")
    name = env_str("RACE_WEEKEND").strip().lower()
    if (year, name) in SPRINT_WEEKENDS:
        raise ValueError(f"{year} {name} is a sprint weekend (FP1 only); the pipeline needs FP1-FP3")
    if (year, name) not in WEEKENDS:
        valid = sorted(n for y, n in WEEKENDS if y == year)
        raise ValueError(f"Unknown race weekend RACE_YEAR={year}, RACE_WEEKEND={name}. "
                         f"Valid names for {year}: {valid or 'none (years: 2024, 2025)'}")
    return {"id": weekend_id(year, name), "year": year, **WEEKENDS[(year, name)]}


# ============================================================
# CONFIG
# ============================================================
data_dir = Path(env_str("DATA_DIR"))
raw_dir = data_dir / env_str("RAW_DIR")
cache_dir = data_dir / env_str("CACHE_DIR")

# Source folders relative to RAW_DIR, so a cached weekend mirrors data/raw/
LAPS_SUBDIR = Path(env_str("LAPS_DIR")).relative_to(env_str("RAW_DIR"))
STINTS_SUBDIR = Path(env_str("STINTS_DIR")).relative_to(env_str("RAW_DIR"))
DRIVERS_SUBDIR = Path(env_str("DRIVERS_DIR")).relative_to(env_str("RAW_DIR"))
WEATHER_SUBDIR = Path(env_str("WEATHER_DIR")).relative_to(env_str("RAW_DIR"))
CARDATA_SUBDIR = Path(env_str("CARDATA_DIR")).relative_to(env_str("RAW_DIR"))

LAPS_FILES = env_list("LAPS_FILES")
STINTS_FILES = env_list("STINTS_FILES")
DRIVERS_FILES = env_list("DRIVERS_FILES")
WEATHER_FILE = env_str("WEATHER_FILE")
CARDATA_FILE_FORMAT = env_str("CARDATA_FILE_FORMAT")

OPENF1_BASE_URL = env_str("OPENF1_BASE_URL")
REQUEST_DELAY_S = env_float("OPENF1_REQUEST_DELAY_S")
MAX_RETRIES = env_int("OPENF1_MAX_RETRIES")
TIMEOUT_S = env_float("OPENF1_TIMEOUT_S")
RETRY_STATUS = {429, 500, 502, 503, 504}

# A car_data request with no results leaves this marker, so it isn't requested again
NO_DATA_SUFFIX = ".nodata"
PART_SUFFIX = ".part"

# Set up by main(), so that importing selected_weekend() writes nothing
log_file = None
generated_files = []

def log(msg: str):
    # Remove unsupported characters for Windows console/logging
    safe_msg = msg.encode("utf-8", errors="replace").decode("utf-8", "replace")
    print(safe_msg)
    if log_file is not None:
        log_file.write(safe_msg + "\n")


# ============================================================
# FILES A WEEKEND NEEDS
# ============================================================
def base_specs(w):
    """(path relative to the weekend folder, OpenF1 endpoint, query) for laps, stints, drivers and weather."""
    specs = []
    for files, subdir, endpoint in [(LAPS_FILES, LAPS_SUBDIR, "laps"),
                                    (STINTS_FILES, STINTS_SUBDIR, "stints"),
                                    (DRIVERS_FILES, DRIVERS_SUBDIR, "drivers")]:
        for file, session_key in zip(files, w["sessions"]):
            specs.append((subdir / file, endpoint, {"session_key": session_key}))
    # Weather covers the whole meeting; process_weather.py keeps the practice sessions
    specs.append((WEATHER_SUBDIR / WEATHER_FILE, "weather", {"meeting_key": w["meeting_key"]}))
    return specs


def cardata_specs(base, w):
    """Car data for every driver in each session's laps file (needs the laps files in base)."""
    specs = []
    for n, (laps_file, session_key) in enumerate(zip(LAPS_FILES, w["sessions"]), start=1):
        drivers = pd.read_csv(base / LAPS_SUBDIR / laps_file, usecols=["driver_number"])["driver_number"]
        for d in sorted(drivers.dropna().astype(int).unique()):
            file = CARDATA_FILE_FORMAT.format(session=n, driver=d)
            specs.append((CARDATA_SUBDIR / file, "car_data", {"session_key": session_key, "driver_number": d}))
    return specs


def is_present(base, rel):
    path = base / rel
    return path.exists() or path.with_name(path.name + NO_DATA_SUFFIX).exists()


def missing_files(base, w):
    missing = [rel for rel, _, _ in base_specs(w) if not is_present(base, rel)]
    if any(rel.parent == LAPS_SUBDIR for rel in missing):
        return missing + [CARDATA_SUBDIR]   # car data can't be listed without the laps files
    return missing + [rel for rel, _, _ in cardata_specs(base, w) if not is_present(base, rel)]


# ============================================================
# WEEKEND IN data/raw/
# ============================================================
def raw_files():
    return [p for p in raw_dir.rglob("*") if p.is_file()] if raw_dir.exists() else []


def raw_weekend_id():
    """Cache folder for what data/raw/ holds, from the meeting_key in its first laps file."""
    laps_file = raw_dir / LAPS_SUBDIR / LAPS_FILES[0]
    if not laps_file.exists():
        return "unidentified"
    meeting_key = int(pd.read_csv(laps_file, usecols=["meeting_key"])["meeting_key"].dropna().iloc[0])
    for (year, name), info in WEEKENDS.items():
        if info["meeting_key"] == meeting_key:
            return weekend_id(year, name)
    return f"meeting_{meeting_key}"


def move_raw_to_cache(dest):
    """Moves every file in data/raw/ into dest; a file dest already has is kept and the raw copy dropped."""
    moved = dropped = 0
    for src in raw_files():
        target = dest / src.relative_to(raw_dir)
        if target.exists():
            src.unlink()
            dropped += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(target))
            moved += 1
    for d in sorted((p for p in raw_dir.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    log(f"Moved data/raw/ into {dest}: {moved} files moved, {dropped} already cached")


# ============================================================
# OPENF1 DOWNLOAD
# ============================================================
def fetch_csv(endpoint, params):
    """CSV bytes from OpenF1, or None when the query has no results (OpenF1 answers 404)."""
    url = f"{OPENF1_BASE_URL}/{endpoint}?{urllib.parse.urlencode({**params, 'csv': 'true'})}"
    for attempt in range(MAX_RETRIES + 1):
        time.sleep(REQUEST_DELAY_S)   # stay under the free API's rate limit
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code not in RETRY_STATUS or attempt == MAX_RETRIES:
                raise
            reason = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == MAX_RETRIES:
                raise
            reason = str(e)
        wait = REQUEST_DELAY_S * 2 ** (attempt + 1)
        log(f"    {reason} for {url}, retrying in {wait:.1f}s ({attempt + 1}/{MAX_RETRIES})")
        time.sleep(wait)


def download_missing(base, specs):
    """Downloads every spec not in base; returns how many were fetched."""
    todo = [s for s in specs if not is_present(base, s[0])]
    for i, (rel, endpoint, params) in enumerate(todo, start=1):
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        body = fetch_csv(endpoint, params)
        if not body or not body.strip():
            if endpoint != "car_data":
                raise RuntimeError(f"OpenF1 has no {endpoint} data for {params}")
            path.with_name(path.name + NO_DATA_SUFFIX).touch()
            log(f"  [{i}/{len(todo)}] WARNING: no {endpoint} data for {params}, left {path.name}{NO_DATA_SUFFIX}")
            continue
        # Write to .part and rename, so an interrupted download is redone on the next run
        part = path.with_name(path.name + PART_SUFFIX)
        part.write_bytes(body)
        os.replace(part, path)
        log(f"  [{i}/{len(todo)}] {rel.as_posix()} <- {endpoint} {params} ({len(body) / 1024:.0f} KB)")
    return len(todo)


# ============================================================
# MAIN
# ============================================================
def main():
    global log_file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file = open(f"{RESULTS_DIR}/{SCRIPT_NAME}_log.txt", "w", encoding="utf-8", errors="replace")

    log("\n--- RACE WEEKEND START ---\n")
    w = selected_weekend()
    log(f"Selected weekend: {w['year']} {w['name']} ({w['id']}), meeting {w['meeting_key']}, "
        f"FP1-FP3 sessions {list(w['sessions'])}, {w['race_laps']} race laps")

    current = raw_weekend_id() if raw_files() else None
    log(f"data/raw/ holds: {current or 'nothing'}")

    downloaded = 0
    if current == w["id"] and not missing_files(raw_dir, w):
        log("data/raw/ already holds the complete selected weekend; nothing to do.")
        source = "data/raw (unchanged)"
    else:
        if current is not None:
            move_raw_to_cache(cache_dir / current)

        weekend_cache = cache_dir / w["id"]
        missing = missing_files(weekend_cache, w)
        if missing:
            listed = [p for p in missing if p != CARDATA_SUBDIR]
            log(f"\n{weekend_cache} is missing {len(listed)} file(s)"
                + (" plus the car data (listed once the laps files are in)" if CARDATA_SUBDIR in missing else "")
                + f"; downloading from {OPENF1_BASE_URL}")
            downloaded += download_missing(weekend_cache, base_specs(w))
            downloaded += download_missing(weekend_cache, cardata_specs(weekend_cache, w))
            source = f"cache, {downloaded} file(s) downloaded"
        else:
            log(f"\nLoading {w['id']} from the cache ({weekend_cache})")
            source = "cache"

        still_missing = missing_files(weekend_cache, w)
        if still_missing:
            raise RuntimeError(f"{weekend_cache} is still incomplete: {[p.as_posix() for p in still_missing]}")

        # .nodata markers are copied too (CARDATA_GLOB doesn't match them), so the fast path sees them
        shutil.copytree(weekend_cache, raw_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(f"*{PART_SUFFIX}"))
        log(f"Copied {weekend_cache} -> {raw_dir}")

    # ============================================================
    # SUMMARY
    # ============================================================
    counts = {sub.as_posix(): len(list((raw_dir / sub).glob("*.csv")))
              for sub in [LAPS_SUBDIR, STINTS_SUBDIR, DRIVERS_SUBDIR, WEATHER_SUBDIR, CARDATA_SUBDIR]}
    no_data = sorted(p.name for p in raw_dir.rglob(f"*{NO_DATA_SUFFIX}"))
    summary = pd.DataFrame([{"weekend": w["id"], "meeting_key": w["meeting_key"],
                             "session_keys": ",".join(map(str, w["sessions"])), "race_laps": w["race_laps"],
                             "source": source, "downloaded": downloaded,
                             **{f"files_{k}": v for k, v in counts.items()},
                             "no_car_data": ",".join(no_data)}])
    summary_name = f"{SCRIPT_NAME}_summary.csv"
    summary.to_csv(f"{RESULTS_DIR}/{summary_name}", index=False)
    generated_files.append(summary_name)
    log(f"\ndata/raw/ files: {counts}" + (f", no car data: {no_data}" if no_data else ""))

    manifest_path = f"{RESULTS_DIR}/{SCRIPT_NAME}_manifest.txt"
    with open(manifest_path, "w") as mf:
        for item in generated_files:
            mf.write(item + "\n")
    generated_files.append(f"{SCRIPT_NAME}_manifest.txt")

    log("\nPROCESS COMPLETE. Files generated:")
    for f in generated_files:
        log(f"  - {f}")

    log_file.close()


if __name__ == "__main__":
    main()
