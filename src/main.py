# main.py
import os
import json
from typing import Optional, Set, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn
import datetime
import socket
from urllib.parse import urlparse

from providers import get_provider_forward

# ---------------- Load .env (try python-dotenv, fallback manual) ----------------
def load_dotenv_fallback(path: str = ".env"):
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

try:
    # prefer python-dotenv if available
    from dotenv import load_dotenv
    load_dotenv()  # loads .env into os.environ
except Exception:
    load_dotenv_fallback(".env")

# -------- Config loader --------
DEFAULT_CONFIG_PATH = "../config.json"

def load_config(path: Optional[str] = None) -> Dict:
    path = path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw

def validate_and_normalize_config(raw: Dict) -> Dict:
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

# load config at startup (raise if missing)
try:
    _raw_cfg = load_config()
    CFG = validate_and_normalize_config(_raw_cfg)
except Exception as e:
    raise RuntimeError(f"Failed to load/validate config.json: {e}")

ALLOWED_HOSTS = CFG["ALLOWED_HOSTS"]
TRUSTED_PROXY = CFG["TRUSTED_PROXY"]
ALLOWED_PROVIDERS = CFG["ALLOWED_PROVIDERS"]
PROVIDER_ENV_NAMES = CFG["PROVIDER_ENV_NAMES"]
PROVIDER_DEFAULT_MODEL = CFG["PROVIDER_DEFAULT_MODEL"]
PROVIDER_ALLOWED_MODELS = CFG["PROVIDER_ALLOWED_MODELS"]

# ---------------- Client keys loading ----------------
def load_client_keys() -> Set[str]:
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

CLIENT_KEYS = load_client_keys()

# ---------------- Load provider upstream keys (from env) ----------------
PROVIDER_API_KEYS = {p: os.getenv(PROVIDER_ENV_NAMES.get(p, "")) for p in PROVIDER_ENV_NAMES}

# ---------------- App init ----------------
app = FastAPI(title="LLM API Gateway", version="0.3")

def normalize_host(host: str) -> str:
    if not host:
        return ""
    return host.split(":", 1)[0].lower().strip()

def parse_origin_host(origin: str) -> str:
    try:
        parsed = urlparse(origin)
        return normalize_host(parsed.netloc)
    except Exception:
        return ""

def is_allowed_request(request: Request) -> bool:
    host = normalize_host(request.headers.get("host", ""))
    if host in ALLOWED_HOSTS:
        return True

    client = request.client.host if request.client else None
    xfh = request.headers.get("x-forwarded-host")
    if client in TRUSTED_PROXY and xfh:
        if normalize_host(xfh.split(",")[0]) in ALLOWED_HOSTS:
            return True

    origin_host = parse_origin_host(request.headers.get("origin", "") or request.headers.get("referer", ""))
    if origin_host in ALLOWED_HOSTS:
        return True

    return False

@app.middleware("http")
async def allow_only_domain_middleware(request: Request, call_next):
    if not is_allowed_request(request):
        return PlainTextResponse("Not Found", status_code=404)
    response = await call_next(request)
    return response

@app.get("/", response_model=dict)
async def root():
    return {
        "ok": True,
        "now": datetime.datetime.utcnow().isoformat() + "Z",
        "host": socket.gethostname(),
        "message": "API Gateway running",
        "config_loaded": True
    }

# ---------------- Client auth util ----------------
def extract_client_key(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth:
        parts = auth.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    xk = request.headers.get("x-api-key") or request.headers.get("X-Api-Key")
    if xk:
        return xk.strip()
    return None

def check_client_auth(request: Request) -> bool:
    key = extract_client_key(request)
    if not key:
        return False
    return key in CLIENT_KEYS

# ---------------- Proxy endpoint with provider+model selection + auth ----------------
@app.api_route("/proxy", methods=["POST"])
async def proxy(request: Request):
    if not check_client_auth(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid json body"}, status_code=400)

    provider = None
    if isinstance(data, dict):
        p = data.get("provider")
        if isinstance(p, str) and p.strip():
            provider = p.strip().lower()
        else:
            for k in data.keys():
                if isinstance(k, str) and k.lower() in ALLOWED_PROVIDERS:
                    provider = k.lower()
                    break

    if not provider or provider not in ALLOWED_PROVIDERS:
        return JSONResponse({
            "ok": False,
            "error": "provider not specified or not supported",
            "allowed": sorted(list(ALLOWED_PROVIDERS))
        }, status_code=400)

    # model selection
    requested_model = None
    if isinstance(data, dict):
        m = data.get("model")
        if isinstance(m, str) and m.strip():
            requested_model = m.strip()
    if not requested_model:
        requested_model = PROVIDER_DEFAULT_MODEL.get(provider)

    allowed_set = PROVIDER_ALLOWED_MODELS.get(provider, set())
    if allowed_set and requested_model not in allowed_set:
        return JSONResponse({
            "ok": False,
            "error": "model not allowed for provider",
            "provider": provider,
            "requested_model": requested_model,
            "allowed_models": sorted(list(allowed_set))
        }, status_code=400)

    # upstream key (from env var name defined in config)
    upstream_api_key = PROVIDER_API_KEYS.get(provider)
    if not upstream_api_key:
        # special-case aistudio fallback to GEMINI_API_KEY
        if provider == "aistudio":
            upstream_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("AISTUDIO_API_KEY")
        if not upstream_api_key:
            return JSONResponse({"ok": False, "error": "provider api key not configured"}, status_code=403)

    forward_fn = get_provider_forward(provider)
    if forward_fn is None:
        return JSONResponse({"ok": False, "error": "provider module not implemented"}, status_code=500)

    if isinstance(data, dict):
        data["model"] = requested_model

    return await forward_fn(request, data, upstream_api_key)

# internal debug (optional)
@app.get("/_internal/keys-status")
async def keys_status():
    return {
        "clients_count": len(CLIENT_KEYS),
        "providers": {p: bool(PROVIDER_API_KEYS.get(p) or (p == "aistudio" and bool(os.getenv("GEMINI_API_KEY")))) for p in PROVIDER_API_KEYS}
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8100, log_level="info")
