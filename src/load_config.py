# load_config.py
import os
import json
from typing import Optional, Set, Dict

# ---------------- Load .env (try python-dotenv, fallback manual) ----------------
def load_dotenv_fallback(path: str = ".env"):
    """
    A fallback function to load .env file if python-dotenv is not installed.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line.split(" ", 1)[1]
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # do not override existing env vars
            if k not in os.environ:
                os.environ[k] = v

def load_env():
    """
    Load environment variables from .env file.
    Prefers python-dotenv if available, otherwise uses a manual fallback.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("[info] python-dotenv not found, using fallback .env loader.")
        load_dotenv_fallback(".env")

# -------- Config loader from config.json --------
DEFAULT_CONFIG_PATH = "../config.json"

def load_config_from_json(path: Optional[str] = None) -> Dict:
    """
    Loads and parses the JSON configuration file.
    """
    path = path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw

def validate_and_normalize_config(raw: Dict) -> Dict:
    """
    Validates the raw config dictionary and normalizes its values.
    """
    cfg = {}
    cfg["ALLOWED_HOSTS"] = set(raw.get("ALLOWED_HOSTS", []))
    cfg["TRUSTED_PROXY"] = set(raw.get("TRUSTED_PROXY", []))
    cfg["ALLOWED_PROVIDERS"] = set(raw.get("ALLOWED_PROVIDERS", []))

    provider_env = raw.get("PROVIDER_ENV_NAMES", {})
    if not isinstance(provider_env, dict):
        raise ValueError("PROVIDER_ENV_NAMES must be an object/dict in config.json")
    cfg["PROVIDER_ENV_NAMES"] = {k: str(v) for k, v in provider_env.items()}

    cfg["PROVIDER_DEFAULT_MODEL"] = dict(raw.get("PROVIDER_DEFAULT_MODEL", {}))

    pam = raw.get("PROVIDER_ALLOWED_MODELS", {})
    if not isinstance(pam, dict):
        raise ValueError("PROVIDER_ALLOWED_MODELS must be an object/dict in config.json")
    cfg["PROVIDER_ALLOWED_MODELS"] = {k: set(v if isinstance(v, list) else []) for k, v in pam.items()}

    for p in cfg["ALLOWED_PROVIDERS"]:
        if p not in cfg["PROVIDER_ENV_NAMES"]:
            print(f"[warn] provider '{p}' has no PROVIDER_ENV_NAMES entry in config.json")
    return cfg

# ---------------- Client keys loading ----------------
def load_client_keys() -> Set[str]:
    """
    Loads client API keys from environment variable and clients.keys file.
    """
    keys = set()
    env = os.getenv("CLIENT_API_KEYS")
    if env:
        for part in env.split(","):
            k = part.strip()
            if k:
                keys.add(k)
    try:
        if os.path.exists("clients.keys"):
            with open("clients.keys", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    keys.add(line)
    except Exception:
        pass
    return keys

# --- Main execution block to load all configurations ---
# This code runs once when the module is imported.

# 1. Load .env first
load_env()

# 2. Load and validate config.json
try:
    _raw_cfg = load_config_from_json()
    _validated_cfg = validate_and_normalize_config(_raw_cfg)
except Exception as e:
    raise RuntimeError(f"Failed to load/validate config.json: {e}")

# 3. Export configuration variables for the main app to use
ALLOWED_HOSTS = _validated_cfg["ALLOWED_HOSTS"]
TRUSTED_PROXY = _validated_cfg["TRUSTED_PROXY"]
ALLOWED_PROVIDERS = _validated_cfg["ALLOWED_PROVIDERS"]
PROVIDER_ENV_NAMES = _validated_cfg["PROVIDER_ENV_NAMES"]
PROVIDER_DEFAULT_MODEL = _validated_cfg["PROVIDER_DEFAULT_MODEL"]
PROVIDER_ALLOWED_MODELS = _validated_cfg["PROVIDER_ALLOWED_MODELS"]

# 4. Load client keys
CLIENT_KEYS = load_client_keys()

# 5. Load provider upstream keys (from env) based on config
PROVIDER_API_KEYS = {p: os.getenv(PROVIDER_ENV_NAMES.get(p, "")) for p in PROVIDER_ENV_NAMES}

print("[info] Configuration loaded successfully.")
