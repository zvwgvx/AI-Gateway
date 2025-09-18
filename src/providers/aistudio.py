# providers/aistudio.py
import os
import asyncio
import json
from typing import Dict, Optional, Any, List
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

try:
    from google import genai
    from google.genai import types
except Exception as e:
    genai = None
    types = None
    _IMPORT_ERROR = e

DEFAULT_MODEL = "gemini-2.5-flash-lite"

def _extract_prompt_from_data(data: Dict) -> str:
    """
    Lấy prompt từ data:
    - ưu tiên "prompt" (string)
    - nếu có "messages" (list of {role, content}) -> nối nội dung của các message có role user
    - fallback: stringify toàn bộ data
    """
    if not isinstance(data, dict):
        return json.dumps(data)

    prompt = data.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    messages = data.get("messages")
    if isinstance(messages, list):
        parts = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            # nếu role omitted, vẫn lấy content
            if role is None or (isinstance(role, str) and role.lower() == "user"):
                content = m.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, dict):
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        if parts:
            return "\n".join(parts)

    # fallback: stringify
    return json.dumps(data)

async def forward(request: Request, data: Dict, api_key: Optional[str]):
    """
    Forward (demo) cho AISTUDIO / Gemini bằng google-genai SDK.
    - request: FastAPI Request (không dùng headers của client làm auth)
    - data: JSON body parsed (đã normalize trong main.py)
    - api_key: key từ main.py (nếu None, sẽ fallback env GEMINI_API_KEY / AISTUDIO_API_KEY)
    Trả StreamingResponse streaming bytes (utf-8).
    """
    if genai is None or types is None:
        # import thất bại
        return JSONResponse({"ok": False, "error": "google-genai not installed", "detail": str(_IMPORT_ERROR)}, status_code=500)

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("AISTUDIO_API_KEY")
    if not key:
        return JSONResponse({"ok": False, "error": "gemini/ai studio api key not provided"}, status_code=403)

    prompt_text = _extract_prompt_from_data(data)
    model = data.get("model") or DEFAULT_MODEL

    # --- LẤY CONFIG + SYSTEM INSTRUCTION TỪ `data` ---
    cfg: Dict[str, Any] = data.get("config", {}) or {}
    temperature = cfg.get("temperature", None)
    top_p = cfg.get("top_p", None)
    tools_enabled = bool(cfg.get("tools", False))

    # default thinking config (unlimited / -1 per sample)
    thinking_budget = cfg.get("thinking_budget", -1)

    thinking_cfg = None

    thinking_cfg = types.ThinkingConfig(thinking_budget=thinking_budget)


    # system_instruction là list[str] chuẩn hoá trong main.py
    system_ins_list: List[str] = data.get("system_instruction", []) or []

    # build contents (user input)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text)]
        )
    ]

    # build tools list if enabled
    tools_list = None
    if tools_enabled:
        try:
            tools_list = [
                types.Tool(url_context=types.UrlContext()),
                # depending on SDK this might be the correct usage (sample you provided uses types.Tool(code_execution=types.ToolCodeExecution))
                types.Tool(code_execution=types.ToolCodeExecution),
            ]
        except Exception:
            # conservative fallback: try constructing with callables/constructors if above form fails
            try:
                tools_list = [
                    types.Tool(url_context=types.UrlContext()),
                    types.Tool(code_execution=types.ToolCodeExecution()),
                ]
            except Exception:
                # nếu vẫn lỗi, set None và tiếp tục (tool sẽ bị bỏ)
                tools_list = None

    # build system_instruction parts
    sys_parts = []
    if isinstance(system_ins_list, (list, tuple)):
        for s in system_ins_list:
            if isinstance(s, str) and s.strip():
                sys_parts.append(types.Part.from_text(text=s.strip()))


    # build GenerateContentConfig kwargs carefully (tránh TypeError nếu SDK khác)
    gen_cfg_kwargs = {}
    if temperature is not None:
        try:
            gen_cfg_kwargs["temperature"] = float(temperature)
        except Exception:
            pass
    if top_p is not None:
        try:
            gen_cfg_kwargs["top_p"] = float(top_p)
        except Exception:
            pass
    if thinking_cfg is not None:
        gen_cfg_kwargs["thinking_config"] = thinking_cfg
    if tools_list:
        gen_cfg_kwargs["tools"] = tools_list
    if sys_parts:
        gen_cfg_kwargs["system_instruction"] = sys_parts

    # try create GenerateContentConfig robustly
    generate_cfg = None
    try:
        generate_cfg = types.GenerateContentConfig(**gen_cfg_kwargs)
    except TypeError:
        # If SDK doesn't accept some keys (e.g., top_p), try safer subset
        safe_kwargs = {}
        if "temperature" in gen_cfg_kwargs:
            safe_kwargs["temperature"] = gen_cfg_kwargs["temperature"]
        if "thinking_config" in gen_cfg_kwargs:
            safe_kwargs["thinking_config"] = gen_cfg_kwargs["thinking_config"]
        if "tools" in gen_cfg_kwargs:
            safe_kwargs["tools"] = gen_cfg_kwargs["tools"]
        if "system_instruction" in gen_cfg_kwargs:
            safe_kwargs["system_instruction"] = gen_cfg_kwargs["system_instruction"]
        try:
            generate_cfg = types.GenerateContentConfig(**safe_kwargs)
        except Exception:
            generate_cfg = None

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def producer():
        """
        Chạy trong thread—iterate blocking stream của SDK, đẩy các chunk text vào asyncio.Queue
        Sử dụng loop.call_soon_threadsafe để tương tác an toàn với queue.
        """
        try:
            # gọi generate_content_stream, có hoặc không có config tuỳ SDK
            if generate_cfg is not None:
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_cfg,
                )
            else:
                # fallback: gọi không kèm config nếu tạo config thất bại
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                )
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, {"__error": str(e)})
            loop.call_soon_threadsafe(q.put_nowait, None)
            return

        try:
            for chunk in stream:
                try:
                    if not chunk or chunk.candidates is None:
                        continue
                    cand = chunk.candidates[0]
                    if cand is None or cand.content is None or cand.content.parts is None:
                        continue
                    part = cand.content.parts[0]
                    out_parts = []
                    # text
                    if getattr(part, "text", None):
                        out_parts.append(part.text)
                    # executable code
                    if getattr(part, "executable_code", None):
                        out_parts.append(part.executable_code)
                    # code execution result
                    if getattr(part, "code_execution_result", None):
                        out_parts.append(str(part.code_execution_result))
                    if out_parts:
                        # ghép các phần thành 1 chunk string
                        chunk_str = "".join(out_parts)
                        loop.call_soon_threadsafe(q.put_nowait, chunk_str)
                except Exception:
                    # skip chunk parsing errors but continue stream
                    continue
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, {"__error": str(e)})
        finally:
            # đặt sentinel để async generator biết kết thúc
            loop.call_soon_threadsafe(q.put_nowait, None)

    # tạo client (blocking)
    client = genai.Client(api_key=key)

    # start producer trong thread
    asyncio.create_task(asyncio.to_thread(producer))

    async def streamer():
        """
        Async generator: yield bytes để StreamingResponse trả về.
        Nếu producer đẩy dict {"__error": "..."} -> trả 502 rồi kết thúc.
        """
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, dict) and item.get("__error"):
                    # upstream error
                    err = item.get("__error")
                    # emit as single JSON error chunk then finish
                    yield (json.dumps({"ok": False, "error": "upstream_error", "detail": err}) + "\n").encode("utf-8")
                    break
                # else item is string
                if not isinstance(item, (str, bytes)):
                    item = str(item)
                if isinstance(item, str):
                    b = item.encode("utf-8")
                else:
                    b = item
                # yield chunk (client sẽ nhận theo streaming)
                yield b
        except asyncio.CancelledError:
            # client closed connection; nothing special to do
            return

    # media_type: dùng text/event-stream cho SSE-compatible clients OR plain text
    # ở demo dùng text/plain; nếu muốn SSE, đổi thành "text/event-stream"
    return StreamingResponse(streamer(), media_type="text/plain; charset=utf-8")