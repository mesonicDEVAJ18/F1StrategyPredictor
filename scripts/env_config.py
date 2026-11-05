# Pipeline settings from the repo-root .env.
# A variable already set in the environment overrides .env, e.g.
#   $env:PIT_LOSS_S = "25"; python strategy_simulator.py
# A missing key raises instead of falling back to a hidden default.
import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(path=ENV_PATH):
    """Read KEY=VALUE lines into os.environ (existing variables win)."""
    if not path.exists():
        raise FileNotFoundError(f".env not found at {path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def env_str(key):
    if key not in os.environ:
        raise KeyError(f"{key} is not set in .env ({ENV_PATH})")
    return os.environ[key]


def env_int(key):
    return int(env_str(key))


def env_float(key):
    return float(env_str(key))


def env_list(key, cast=str):
    """Comma-separated list, e.g. STOP_COUNTS=0,1,2 -> [0, 1, 2] with cast=int."""
    return [cast(v.strip()) for v in env_str(key).split(",") if v.strip()]


def write_env(updates, note, path=ENV_PATH):
    """Writes tuned values into .env (explore/tune_*.py): replaces each existing KEY=value line in place
    and keeps one "# <note>" line directly above it (one per run of adjacent keys), replacing the previous
    one from the same script (the note starts with the script's name). Also sets os.environ, which wins
    over .env for this process and the stages it starts. A key missing from .env raises.
    """
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    eol = "\r\n" if "\r\n" in text else "\n"   # keep the file's line endings
    lines = text.split(eol)
    tag = "# " + note.split()[0] + " "

    def key_of(line):
        return None if line.startswith("#") or "=" not in line else line.split("=", 1)[0].strip()

    for key in updates:
        if key not in [key_of(line) for line in lines]:
            raise KeyError(f"{key} is not set in .env ({path})")
    # Notes from earlier runs of this script go first, then one note above each run of adjacent keys
    lines = [line for i, line in enumerate(lines)
             if not (line.startswith(tag) and i + 1 < len(lines) and key_of(lines[i + 1]) in updates)]
    out = []
    for line in lines:
        key = key_of(line)
        if key in updates:
            if not out or key_of(out[-1]) not in updates:
                out.append(f"# {note}")
            line = f"{key}={updates[key]}"
            os.environ[key] = str(updates[key])
        out.append(line)
    lines = out
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(eol.join(lines))


load_env()
