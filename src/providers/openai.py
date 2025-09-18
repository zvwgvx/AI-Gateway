# providers/openai.py
import os
import asyncio
import json
from typing import Dict, Optional, Any
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

try:
    import openai
except Exception as e:
    openai = None
    _IMPORT_ERROR = e

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

    def _extract_text_from_chunk(chunk: Any):
        """
        Normalize a streaming chunk (could be dict-like or object-like) and
        extract any text pieces from 'delta' or legacy 'text' fields.
        Returns a string (possibly empty).
        """
        text_parts = []

        # try dict-like access
        def dget(obj, path, default=None):
            try:
                cur = obj
                for p in path:
                    if cur is None:
                        return default
                    if isinstance(cur, dict):
                        cur = cur.get(p)
                    else:
                        cur = getattr(cur, p, None)
                return cur
            except Exception:
                return default

        choices = dget(chunk, ("choices",), []) or []
        # choices might be a list of dicts or objects
        for c in choices:
            delta = dget(c, ("delta",), None)
            if delta is not None:
                # delta may be dict or object
                content = None
                if isinstance(delta, dict):
                    content = delta.get("content")
                else:
                    content = getattr(delta, "content", None)
                if content:
                    text_parts.append(content)
            # legacy possibility: choices[].text
            txt = dget(c, ("text",), None)
            if txt:
                text_parts.append(txt)

        # some SDK variants put text at the top-level chunk['text']
        top_text = None
        if isinstance(chunk, dict):
            top_text = chunk.get("text")
        else:
            top_text = getattr(chunk, "text", None)
        if top_text:
            text_parts.append(top_text)

        return "".join(filter(None, text_parts))

    def producer():
        """
        Blocking producer: call the upstream SDK (new or old) and push chunks into queue.
        """
        try:
            # Prefer modern OpenAI client if available (openai.OpenAI)
            if hasattr(openai, "OpenAI"):
                try:
                    # instantiate client with api_key (newer SDK supports passing api_key)
                    client = openai.OpenAI(api_key=key)
                except TypeError:
                    # Some older/newer variants might not accept api_key kw; fallback to setting env
                    client = openai.OpenAI()
                    os.environ["OPENAI_API_KEY"] = key

                # call new client.chat.completions.create with stream=True
                resp_iter = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    **extra
                )
            else:
                # fallback to legacy openai SDK interface
                openai.api_key = key
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
            for chunk in resp_iter:
                try:
                    s = _extract_text_from_chunk(chunk)
                    if s:
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
