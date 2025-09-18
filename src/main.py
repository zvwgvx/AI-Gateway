# main.py
import os
import datetime
import socket
from typing import Optional, Any, Dict
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

# Import all configuration variables from the new module
from load_config import (
    ALLOWED_HOSTS,
    TRUSTED_PROXY,
    ALLOWED_PROVIDERS,
    CLIENT_KEYS,
    PROVIDER_API_KEYS,
    PROVIDER_DEFAULT_MODEL,
    PROVIDER_ALLOWED_MODELS
)
from providers import get_provider_forward

# ---------------- App init ----------------
app = FastAPI(title="LLM API Gateway", version="0.4")

# ---------------- Middleware and Request Validation ----------------
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

def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on")
    return False

def normalize_request_payload(data: Dict[str, Any]) -> None:
    """
    Normalize/validate client-sent fields:
      - config: {temperature, top_p, tools, thinking_budget}
      - system_instruction: list[str] (accept str or list)
    This mutates `data`.
    """
    if not isinstance(data, dict):
        return

    # --- config ---
    default_temp = 0.7
    default_top_p = 1.0
    default_thinking = -1  # default unlimited

    cfg = data.get("config", {}) or {}
    if not isinstance(cfg, dict):
        cfg = {}

    # temperature
    temp = cfg.get("temperature", default_temp)
    try:
        temp = float(temp)
    except Exception:
        temp = default_temp
    if temp < 0.0:
        temp = 0.0
    if temp > 2.0:
        temp = 2.0

    # top_p
    top_p = cfg.get("top_p", default_top_p)
    try:
        top_p = float(top_p)
    except Exception:
        top_p = default_top_p
    if top_p < 0.0:
        top_p = 0.0
    if top_p > 1.0:
        top_p = 1.0

    # tools (boolean)
    tools = cfg.get("tools", False)
    tools = _coerce_bool(tools)

    # thinking_budget: accept -1 (unlimited) or 0..24576, clamp otherwise
    tb_raw = cfg.get("thinking_budget", default_thinking)
    try:
        tb = int(tb_raw)
    except Exception:
        tb = default_thinking

    # Normalize
    if tb == -1:
        pass
    else:
        # any negative other than -1 => treat as -1
        if tb < 0:
            tb = default_thinking
        # clamp upper bound
        if tb > 24576:
            tb = 24576

    data["config"] = {
        "temperature": temp,
        "top_p": top_p,
        "tools": tools,
        "thinking_budget": tb
    }

    # --- system_instruction ---
    sys_ins = data.get("system_instruction", [])
    if isinstance(sys_ins, str):
        sys_list = [sys_ins]
    elif isinstance(sys_ins, (list, tuple)):
        sys_list = [str(x) for x in sys_ins if isinstance(x, (str,))]
    else:
        sys_list = []
    sys_list = [s.strip() for s in sys_list if s and s.strip()]
    data["system_instruction"] = sys_list

# ---------------- API Endpoints ----------------
@app.get("/", response_model=dict)
async def root():
    return {
        "ok": True,
        "now": datetime.datetime.utcnow().isoformat() + "Z",
        "host": socket.gethostname(),
        "message": "API Gateway running",
        "config_loaded": True
    }

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

    # ensure model is present in payload
    if isinstance(data, dict):
        data["model"] = requested_model

    # --- NEW: normalize config and system_instruction ---
    normalize_request_payload(data)

    # finally forward the normalized payload to provider-specific forward
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
