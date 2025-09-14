# providers/openai.py
from typing import Dict
from fastapi import Request
from providers._common import filter_request_headers, forward_streaming
import json

DEFAULT_TARGET = "https://api.openai.com/v1/chat/completions"

async def forward(request: Request, data: Dict, api_key: str):
    """
    Forward request to OpenAI-like endpoint.
    - request: FastAPI Request
    - data: parsed JSON body from client
    - api_key: string (may be None)
    """
    target = data.get("path") or DEFAULT_TARGET
    # prepare headers: copy client headers (filtered) + provider auth
    headers = filter_request_headers(request.headers)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # ensure content-type and body: forward same JSON body
    headers.setdefault("Content-Type", "application/json")
    # forward streaming body (we already parsed JSON in main; for simplicity we re-encode)
    body_bytes = json.dumps(data).encode("utf-8")
    # use forward_streaming (wrap body as bytes)
    return await forward_streaming(request.method, target, headers, body_bytes)
