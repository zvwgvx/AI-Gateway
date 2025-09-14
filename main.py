# main.py
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
import uvicorn
import datetime
import socket
from urllib.parse import urlparse
import httpx
import asyncio

ALLOWED_HOSTS = {"api.polydevs.uk"}  # chỉ hostname, không chứa port
TRUSTED_PROXY = {"127.0.0.1"}  # nếu bạn có proxy trước gateway, thêm IP ở đây

app = FastAPI(title="Local API Gateway (dev)", version="0.1")

def normalize_host(host: str) -> str:
    if not host:
        return ""
    # host có thể là "api.polydevs.uk:443" -> strip port
    return host.split(":", 1)[0].lower().strip()

def parse_origin_host(origin: str) -> str:
    try:
        parsed = urlparse(origin)
        return normalize_host(parsed.netloc)
    except Exception:
        return ""

def is_allowed_request(request: Request) -> bool:
    # 1) Host header
    host = normalize_host(request.headers.get("host", ""))
    if host in ALLOWED_HOSTS:
        return True

    # 2) X-Forwarded-Host (chỉ khi request tới từ trusted proxy)
    # Kiểm tra remote addr trước
    client = request.client.host if request.client else None
    xfh = request.headers.get("x-forwarded-host")
    if client in TRUSTED_PROXY and xfh:
        if normalize_host(xfh.split(",")[0]) in ALLOWED_HOSTS:
            return True

    # 3) Origin / Referer
    origin_host = parse_origin_host(request.headers.get("origin", "") or request.headers.get("referer", ""))
    if origin_host in ALLOWED_HOSTS:
        return True

    return False

@app.middleware("http")
async def allow_only_domain_middleware(request: Request, call_next):
    if not is_allowed_request(request):
        # Lưu log ngắn gọn hoặc chặn sớm
        return PlainTextResponse("Not Found", status_code=404)

    # xử lý bình thường
    response = await call_next(request)
    return response

@app.get("/", response_model=dict)
async def root():
    return {
        "ok": True,
        "now": datetime.datetime.utcnow().isoformat() + "Z",
        "host": socket.gethostname(),
        "message": "API Gateway running"
    }

# ------------ proxy example ------------
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade"
}

async def filter_request_headers(headers):
    out = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP:
            continue
        # drop host header so httpx will set correct host for target_url
        if lk == "host":
            continue
        out[k] = v
    return out

async def filter_response_headers(headers):
    out = {}
    for k, v in headers.items():
        if k.lower() in HOP_BY_HOP:
            continue
        out[k] = v
    return out

@app.api_route("/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request):
    """
    Ví dụ: forward tất cả request tới backend map theo path.
    - Bạn nên implement mapping model->backend ở đây.
    - target_base lấy từ config hoặc mapping per model.
    """
    # --------- mapping example (replace bằng logic của bạn) ----------
    # Ví dụ: /proxy/openai/ => https://api.openai.com/
    # Đây chỉ là ví dụ; bạn cần config mapping an toàn hơn.
    target_base = "https://api.polydevs.uk/"  # thay bằng actual backend url theo mapping

    # Build target URL
    qs = f"?{request.url.query}" if request.url.query else ""
    target_url = f"{target_base.rstrip('/')}/{path.lstrip('/')}{qs}"

    # prepare headers
    req_headers = await filter_request_headers(request.headers)
    # (tuỳ ứng dụng, có thể chèn thêm authorization header backend ở đây)
    timeout = httpx.Timeout(15.0, connect=5.0)

    # Sử dụng streaming cho body lớn (không load toàn bộ vào RAM)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        try:
            # stream request body from client to backend
            backend_req = client.stream(
                request.method,
                target_url,
                headers=req_headers,
                content=request.stream(),  # starlette stream iterator
            )
            async with backend_req as resp:
                resp_headers = await filter_response_headers(resp.headers)
                # stream response body về client
                return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=resp_headers)
        except httpx.RequestError as e:
            # log error
            return JSONResponse({"error": "upstream error", "detail": str(e)}, status_code=502)

# health endpoints
@app.get("/health", response_model=dict)
async def health():
    return {"ok": True, "now": datetime.datetime.utcnow().isoformat() + "Z", "message": "OK"}

if __name__ == "__main__":
    # CHÚ Ý: production: run uvicorn behind reverse proxy and bind 127.0.0.1
    uvicorn.run("main:app", host="127.0.0.1", port=8100, log_level="info")
