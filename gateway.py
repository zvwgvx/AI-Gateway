"""
Runner for starting the FastAPI app in src.main.

This runner ensures:
 - src/ is on sys.path so `from providers import ...` inside src/main.py works when
   running the script from repository root.
 - CONFIG_PATH is set to the repo's config.json (so src/main.py finds it). If the
   config file is missing, we print a clear error and exit before starting uvicorn.

Usage:
  python gateway.py

Environment variables (optional):
  HOST               default: 0.0.0.0
  PORT               default: 8100
  LOG_LEVEL          default: info
  UVICORN_WORKERS    default: 1
  RELOAD             set to 1/true/yes to enable reload (only when workers=1)
  CONFIG_PATH        if you prefer a different path to config.json, set this.

"""
import os
import sys
import traceback
from pathlib import Path
import uvicorn


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


# File locations
HERE = Path(__file__).resolve().parent
SRC_DIR = str(HERE / "src")
PROJECT_CONFIG = str(HERE / "config.json")

# Ensure `src/` is on sys.path so `providers` (located in src/providers) is
# importable as a top-level module when running this script from the repository root.
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# If the environment didn't already provide CONFIG_PATH, set it to the project's config.json
if "CONFIG_PATH" not in os.environ:
    if os.path.exists(PROJECT_CONFIG):
        os.environ["CONFIG_PATH"] = PROJECT_CONFIG
    else:
        # leave it unset; src.main will raise a helpful error — but we also print one
        print(f"[gateway] WARNING: config.json not found at {PROJECT_CONFIG}. You can set CONFIG_PATH env var to the correct file.")


def run() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = _env_int("PORT", 8100)
    log_level = os.getenv("LOG_LEVEL", "info")
    workers = _env_int("UVICORN_WORKERS", 10)
    reload_opt = _env_bool("RELOAD", False)

    # reload isn't compatible with multiple workers in uvicorn
    if workers > 1 and reload_opt:
        print("[gateway] RELOAD is not compatible with multiple workers; disabling reload.")
        reload_opt = False

    # double-check config presence and advise if missing
    cfg_path = os.getenv("CONFIG_PATH", PROJECT_CONFIG)
    if not os.path.exists(cfg_path):
        print(f"[gateway] ERROR: config.json not found at {cfg_path}. Please create the file or set CONFIG_PATH to its location.")
        sys.exit(1)

    print(f"[gateway] Starting uvicorn at {host}:{port} (workers={workers}, reload={reload_opt})")

    try:
        uvicorn.run("src.main:app", host=host, port=port, log_level=log_level, reload=reload_opt, workers=workers)
    except Exception:
        print("[gateway] uvicorn failed to start:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run()
