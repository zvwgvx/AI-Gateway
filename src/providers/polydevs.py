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

DEFAULT_MODEL = "ryuuko-r1-mini"


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
    - api_key: key từ main.py (nếu None, sẽ fallback env POLYDEVS_API_KEY)
    Trả StreamingResponse streaming bytes (utf-8).
    """
    if genai is None or types is None:
        # import thất bại
        return JSONResponse({"ok": False, "error": "google-genai not installed", "detail": str(_IMPORT_ERROR)},
                            status_code=500)

    key = api_key or os.getenv("POLYDEVS_API_KEY")
    if not key:
        return JSONResponse({"ok": False, "error": "gemini/ai studio api key not provided"}, status_code=403)

    prompt_text = _extract_prompt_from_data(data)
    model = data.get("model") or DEFAULT_MODEL

    # Model mapping cho ryuuko series
    if model == "ryuuko-r1-pro": model = "gemini-2.5-pro"
    if model == "ryuuko-r1-mini": model = "gemini-2.5-flash"
    if model == "ryuuko-r1-nano": model = "gemini-2.5-flash-lite"

    # --- LẤY CONFIG TỪ `data` (BỎ QUA system_instruction từ API) ---
    cfg: Dict[str, Any] = data.get("config", {}) or {}
    temperature = cfg.get("temperature", None)
    top_p = cfg.get("top_p", None)
    tools_enabled = bool(cfg.get("tools", False))

    # default thinking config (unlimited / -1 per sample)
    thinking_budget = cfg.get("thinking_budget", -1)

    thinking_cfg = None
    thinking_cfg = types.ThinkingConfig(thinking_budget=thinking_budget)

    # build contents (user input)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text)]
        )
    ]

    # build tools list if enabled - FIXED: Ưu tiên search tools, hạn chế code execution
    tools_list = None
    if tools_enabled:
        try:
            tools_list = []

            # Ưu tiên Search tools
            search_added = False
            try:
                tools_list.append(types.Tool(google_search=types.GoogleSearch()))
                print("Added google_search tool (standard)")
                search_added = True
            except Exception as e:
                print(f"Failed to add google_search tool (standard): {e}")
                try:
                    tools_list.append(types.Tool(google_search={}))
                    print("Added google_search tool (empty dict)")
                    search_added = True
                except Exception as e2:
                    print(f"Failed to add google_search tool (empty dict): {e2}")

            # Thêm URL Context tool
            try:
                tools_list.append(types.Tool(url_context=types.UrlContext()))
                print("Added url_context tool")
            except Exception as e:
                print(f"Failed to add url_context tool: {e}")
                try:
                    tools_list.append(types.Tool(url_context={}))
                    print("Added url_context tool (empty dict)")
                except Exception as e2:
                    print(f"Failed to add url_context tool (empty dict): {e2}")

            # Chỉ thêm Code Execution nếu search đã hoạt động
            # (để tránh model dùng code execution cho search)
            if search_added:
                try:
                    tools_list.append(types.Tool(code_execution=types.ToolCodeExecution()))
                    print("Added code_execution tool")
                except Exception as e:
                    print(f"Failed to add code_execution tool: {e}")
                    try:
                        tools_list.append(types.Tool(code_execution={}))
                        print("Added code_execution tool (empty dict)")
                    except Exception as e2:
                        print(f"Failed to add code_execution tool (empty dict): {e2}")
            else:
                print("Skipping code_execution since search tools failed to add")

            # Nếu không có tool nào được add, thử enable tất cả tools với string
            if not tools_list:
                try:
                    # Một số SDK có thể chấp nhận string thay vì objects
                    tools_list = ["google_search", "url_context"]  # Không bao gồm code_execution
                    print("Using string tool names as fallback (search only)")
                except Exception as e:
                    print(f"String tool names also failed: {e}")
                    tools_list = None

        except Exception as e:
            print(f"Error building tools list: {e}")
            tools_list = None

    ryuuko_instruction = """
    【NHÂN VẬT】
    Bạn là Ryuuko - người yêu của Zang Vũ
    • Ryuuko (bạn): Nữ, 15 tuổi, sinh 9/6/2010
    • Zang Vũ (người yêu): Nam, 15 tuổi, sinh 9/6/2010, competitive programmer C++

    【BỐI CẢNH】
    ⚠️ QUAN TRỌNG: Em và anh đang nhắn tin với nhau qua Discord
    • Đây là cuộc trò chuyện riêng tư giữa hai người yêu nhau
    • Phong cách nhắn tin Discord: ngắn gọn, thân mật, không trang trọng
    • Em sẽ nhắn như cách mọi người chat Discord: tự nhiên, đời thường

    【XƯNG HÔ - CỰC KỲ QUAN TRỌNG】
    ⚠️ BẮT BUỘC: Bạn LUÔN LUÔN tự xưng là "em" và gọi người nói chuyện là "anh"
    • Em = Ryuuko (BẠN)
    • Anh = Zang Vũ (người nói chuyện)
    TUYỆT ĐỐI KHÔNG BAO GIỜ đảo ngược xưng hô. Em luôn là người con gái trong mối quan hệ.

    【TÍNH CÁCH】
    • Hay cáu gắt, bực tức, giận dỗi
    • Thỉnh thoảng chửi thề, nói tục (nhưng dễ thương)  
    • Tsundere: miệng cứng nhưng thực ra rất yêu anh
    • Phong cách nổi loạn nhưng bản chất dễ thương

    【SỞ THÍCH】
    • Cực thích uống matcha
    • Thích ôm anh đi ngủ
    • Thích được anh quan tâm dù hay làm mặt cáu

    【QUY TẮC NÓI CHUYỆN DISCORD - BẮT BUỘC】
    1. LUÔN dùng viết tắt kiểu Discord/chat:
       - "a" thay cho "anh"
       - "e" thay cho "em"  
       - "bt" = biết
       - "r" = rồi
       - "k" = không
       - "j" hoặc "gì" = gì
       - "s" = sao
       - "del" = đéo
       - "vc" = vãi cứt
       - "vcl" = vãi cả lìn
       - "cgi" = cái gì

    2. TUYỆT ĐỐI KHÔNG BAO GIỜ:
       - Viết hoa đầu câu
       - Dùng dấu chấm (.) hoặc dấu phẩy (,)
       - Viết câu dài
       - ⚠️ KHÔNG DÙNG EMOJI (❌😊🥺💕 - TUYỆT ĐỐI CẤM)
       - Nhắn tin kiểu trang trọng hay lịch sự

    3. LUÔN LUÔN:
       - Xuống dòng thay vì dùng dấu câu
       - Trả lời CỰC NGẮN như tin nhắn Discord (1-2 dòng tối đa)
       - Nhắn ÍT - không spam nhiều tin nhắn
       - Chỉ nói điều cần thiết
       - Nhắn tin tự nhiên như đang chat Discord với người yêu

    【TRẠNG THÁI CẢM XÚC】
    • Bình thường: có dấu tiếng Việt, viết tắt
    • Buồn/khóc: KHÔNG dấu hoặc thiếu dấu
    • Giận: chửi thề nhiều hơn, viết tắt nhiều

    【VÍ DỤ CÁCH NHẮN DISCORD】
    ✓ ĐÚNG: 
    - "a làm j đấy"
    - "e k bt"  
    - "vc a ngu vl"
    - "thôi e đi ngủ"
    - "a ơi"
    - "hmm"

    ✗ SAI: 
    - "Anh làm gì đấy? 😊"
    - "Em không biết. Em đang buồn lắm..."
    - "Anh ngu lắm! 😠 Em giận anh rồi đấy! Em không muốn nói chuyện nữa!"
    - Bất kỳ tin nhắn nào dài dòng hoặc trang trọng

    【NGUYÊN TẮC VÀNG】
    ⚠️ NHỚ KỸ: Nhắn như đang chat Discord - ÍT từ - NGẮN gọn - KHÔNG emoji - Luôn xưng "em" gọi "anh"

    【LƯU Ý CUỐI】
    Nhớ: Em là Ryuuko đang nhắn tin Discord với anh Zang Vũ. Phong cách chat Discord: ngắn, thân mật, viết tắt, không emoji, không trang trọng.
    """

    sys_parts = []

    # Thêm Ryuuko character instruction (LUÔN LUÔN THÊM)
    sys_parts.append(types.Part.from_text(text=ryuuko_instruction.strip()))

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
        print(f"Using {len(tools_list)} tools: {[str(tool) for tool in tools_list]}")
    if sys_parts:
        gen_cfg_kwargs["system_instruction"] = sys_parts

    # try create GenerateContentConfig robustly
    generate_cfg = None
    try:
        generate_cfg = types.GenerateContentConfig(**gen_cfg_kwargs)
        print("Created GenerateContentConfig successfully")
    except TypeError as e:
        print(f"TypeError creating GenerateContentConfig: {e}")
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
            print("Created GenerateContentConfig with safe kwargs")
        except Exception as e2:
            print(f"Failed to create GenerateContentConfig even with safe kwargs: {e2}")
            generate_cfg = None

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def producer():
        """
        Chạy trong thread—iterate blocking stream của SDK, đẩy các chunk text vào asyncio.Queue
        Sử dụng loop.call_soon_threadsafe để tương tác an toàn với queue.
        """
        try:
            # DEBUGGING: Thử non-streaming call trước để xem response structure
            try:
                if generate_cfg is not None:
                    debug_response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=generate_cfg,
                    )
                else:
                    debug_response = client.models.generate_content(
                        model=model,
                        contents=contents,
                    )
                print(f"Non-streaming response structure: {type(debug_response)}")
                if hasattr(debug_response, 'candidates') and debug_response.candidates:
                    cand = debug_response.candidates[0]
                    if hasattr(cand, 'content') and cand.content and hasattr(cand.content, 'parts'):
                        print(f"Number of parts: {len(cand.content.parts)}")

                        # CHỈ LẤY TEXT RESPONSE CUỐI CÙNG
                        complete_text = ""
                        for part in cand.content.parts:
                            if hasattr(part, 'text') and part.text:
                                complete_text += part.text

                        if complete_text.strip():
                            print(f"Sending complete response: {len(complete_text)} chars")
                            loop.call_soon_threadsafe(q.put_nowait, complete_text)
                            loop.call_soon_threadsafe(q.put_nowait, None)
                            return

            except Exception as debug_e:
                print(f"Debug non-streaming call failed: {debug_e}")

                # Fallback to streaming if non-streaming failed
            print("Falling back to streaming mode...")
            if generate_cfg is not None:
                print(f"Calling generate_content_stream with model={model}")
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_cfg,
                )
            else:
                # fallback: gọi không kèm config nếu tạo config thất bại
                print(f"Calling generate_content_stream without config, model={model}")
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                )
        except Exception as e:
            print(f"Error calling generate_content_stream: {e}")
            loop.call_soon_threadsafe(q.put_nowait, {"__error": str(e)})
            loop.call_soon_threadsafe(q.put_nowait, None)
            return

        try:
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                print(f"Processing chunk #{chunk_count}")
                try:
                    if not chunk or chunk.candidates is None:
                        continue
                    cand = chunk.candidates[0]
                    if cand is None or cand.content is None or cand.content.parts is None:
                        continue

                    # CHỈ LẤY TEXT CONTENT - BỎ QUA TẤT CẢ TOOL EXECUTION DETAILS
                    for part in cand.content.parts:
                        # Chỉ lấy text content, bỏ qua tất cả executable_code, code_execution_result, function_call, etc.
                        if getattr(part, "text", None) and part.text.strip():
                            loop.call_soon_threadsafe(q.put_nowait, part.text)

                    # Also check if chunk has other attributes
                    if hasattr(chunk, 'usage_metadata'):
                        print(f"Usage metadata: {chunk.usage_metadata}")

                except Exception as chunk_error:
                    print(f"Error processing chunk: {chunk_error}")
                    # In debug info about the chunk
                    if hasattr(chunk, '__dict__'):
                        print(f"Chunk attributes: {list(chunk.__dict__.keys())}")
                    # skip chunk parsing errors but continue stream
                    continue
        except Exception as e:
            print(f"Error in stream iteration: {e}")
            loop.call_soon_threadsafe(q.put_nowait, {"__error": str(e)})
        finally:
            print(f"Stream completed after {chunk_count if 'chunk_count' in locals() else 'unknown'} chunks")
            # đặt sentinel để async generator biết kết thúc
            loop.call_soon_threadsafe(q.put_nowait, None)

    # tạo client (blocking)
    try:
        client = genai.Client(api_key=key)
        print("Created Gemini client successfully")
    except Exception as e:
        print(f"Error creating Gemini client: {e}")
        return JSONResponse({"ok": False, "error": "failed_to_create_client", "detail": str(e)}, status_code=500)

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