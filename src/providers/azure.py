# providers/azure.py
from typing import Dict
from fastapi import Request
from providers._common import filter_request_headers, forward_streaming
import json

# Azure endpoint should be configured by data or config (deployment specific)
DEFAULT_TARGET = "https://your-azure-openai-endpoint.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2023-10-01"

async def forward(request: Request, data: Dict, api_key: str):
    target = data.get("path") or DEFAULT_TARGET
    headers = filter_request_headers(request.headers)
    if api_key:
        headers["api-key"] = api_key
    headers.setdefault("Content-Type", "application/json")
    body_bytes = json.dumps(data).encode("utf-8")
    return await forward_streaming(request.method, target, headers, body_bytes)
