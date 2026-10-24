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


load_env()
