# providers/openai.py
import os
import asyncio
import json
from typing import Dict, Optional
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

# cố gắng import openai SDK
try:
    import openai
except Exception as e:
    openai = None
    _IMPORT_ERROR = e

# default model (có thể override bằng data["model"])
DEFAULT_MODEL = "gpt-4o-mini"

def _extract_messages(data: Dict):
    """
    Chuẩn hoá messages cho OpenAI Chat API:
    - Nếu data có 'messages' và là list -> return nó.
    - Nếu data có 'prompt' -> chuyển thành messages [{"role":"user","content": prompt}]
    - Nếu không -> stringify toàn bộ data làm prompt.
    """
    if not isinstance(data, dict):
        return [{"role": "user", "content": json.dumps(data)}]

    msgs = data.get("messages")
    if isinstance(msgs, list) and msgs:
        # đảm bảo mỗi item có 'role' và 'content' (string)
        out = []
        for m in msgs:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content") or m.get("text") or ""
                if isinstance(content, dict):
                    content = content.get("text", "")
                out.append({"role": role, "content": str(content)})
        if out:
            return out

    prompt = data.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt.strip()}]

    # fallback
    return [{"role": "user", "content": json.dumps(data)}]

async def forward(request: Request, data: Dict, api_key: Optional[str]):
    """
    Forward using OpenAI Python SDK and stream back to client.
    - request: FastAPI Request (not used for auth)
    - data: parsed JSON body from client
    - api_key: upstream API key (string)
    """
    if openai is None:
        return JSONResponse({"ok": False, "error": "openai package not installed", "detail": str(_IMPORT_ERROR)}, status_code=500)

    # choose api key: param > env OPENAI_API_KEY
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return JSONResponse({"ok": False, "error": "openai api key not provided"}, status_code=403)

    # set api key on SDK client (this is global in openai lib)
    openai.api_key = key

    # model selection
    model = data.get("model") or DEFAULT_MODEL

    # build messages
    messages = _extract_messages(data)

    # additional params allowed (temperature, max_tokens...) - whitelist if needed
    extra = {}
    for k in ("temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty"):
        if k in data:
            extra[k] = data[k]

    # prepare queue + producer
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def producer():
        """
        Blocking producer: gọi openai.ChatCompletion.create(..., stream=True) và push chunks vào queue.
        Lưu ý: openai SDK stream yields "delta" chunks; chúng ta gom text từ 'choices'
        """
        try:
            # For older/newer SDKs, method name may differ; this uses ChatCompletion.create with stream=True
            # If your SDK version uses client.chat.completions.create or ChatCompletion.acreate, adjust accordingly.
            resp_iter = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=True,
                **extra
            )
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, {"__error": str(e)})
            loop.call_soon_threadsafe(q.put_nowait, None)
            return

        # iterate streaming chunks
        try:
            # resp_iter yields chunks dictionaries
            for chunk in resp_iter:
                # chunk may contain 'choices' list with 'delta' parts
                try:
                    # Many SDKs produce {"choices": [{"delta": {"content": "..."}, "finish_reason": ...}], "id":...}
                    choices = chunk.get("choices") or []
                    text_parts = []
                    for c in choices:
                        delta = c.get("delta") or {}
                        # delta may have 'content' or 'role' etc.
                        content = delta.get("content")
                        if content:
                            text_parts.append(content)
                        # older SDKs may include 'text' directly
                        if "text" in chunk:
                            text_parts.append(chunk.get("text"))
                    if text_parts:
                        s = "".join(text_parts)
                        loop.call_soon_threadsafe(q.put_nowait, s)
                except Exception:
                    # ignore malformed chunk
                    continue
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, {"__error": str(e)})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    # start producer in background thread
    asyncio.create_task(asyncio.to_thread(producer))

    async def streamer():
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, dict) and item.get("__error"):
                    yield (json.dumps({"ok": False, "error": "upstream_error", "detail": item.get("__error")}) + "\n").encode("utf-8")
                    break
                if not isinstance(item, (str, bytes)):
                    item = str(item)
                b = item.encode("utf-8") if isinstance(item, str) else item
                yield b
        except asyncio.CancelledError:
            return

    return StreamingResponse(streamer(), media_type="text/plain; charset=utf-8")
