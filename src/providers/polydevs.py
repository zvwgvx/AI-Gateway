import os
import asyncio
import json
from typing import Dict, Optional, Any, List
from pathlib import Path
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

# Global variable để lưu instructions
INSTRUCTIONS = None
INSTRUCTIONS_LOAD_ERROR = None


def load_instructions():
    """Load instructions từ file JSON - BẮT BUỘC phải có file"""
    global INSTRUCTIONS_LOAD_ERROR

    try:
        # Thử nhiều vị trí có thể
        possible_paths = [
            Path(__file__).parent / "instructions.json",  # Cùng thư mục với file Python
            Path.cwd() / "instructions.json",  # Current working directory
            Path(__file__).parent.parent / "instructions.json",  # Thư mục cha
            Path.cwd() / "scripts" / "instructions.json",  # Trong scripts folder
        ]

        instruction_file = None
        tried_paths = []

        for path in possible_paths:
            tried_paths.append(str(path.absolute()))
            print(f"[INSTRUCTION LOADER] Checking: {path.absolute()}")
            if path.exists():
                instruction_file = path
                print(f"[INSTRUCTION LOADER] ✓ Found at: {path.absolute()}")
                break
            else:
                print(f"[INSTRUCTION LOADER] ✗ Not found at: {path.absolute()}")

        if instruction_file is None:
            error_msg = f"CRITICAL: instructions.json not found! Tried paths:\n" + "\n".join(tried_paths)
            print(f"[INSTRUCTION LOADER] {error_msg}")
            INSTRUCTIONS_LOAD_ERROR = error_msg
            return None

        # Load và validate file
        with open(instruction_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"[INSTRUCTION LOADER] Successfully loaded JSON from {instruction_file}")

            # Validate structure
            if not isinstance(data, dict):
                error_msg = f"Invalid instructions.json: Root must be object, got {type(data)}"
                print(f"[INSTRUCTION LOADER] {error_msg}")
                INSTRUCTIONS_LOAD_ERROR = error_msg
                return None

            if "vietnamese" not in data:
                error_msg = "Invalid instructions.json: Missing 'vietnamese' key"
                print(f"[INSTRUCTION LOADER] {error_msg}")
                INSTRUCTIONS_LOAD_ERROR = error_msg
                return None

            if "english" not in data:
                error_msg = "Invalid instructions.json: Missing 'english' key"
                print(f"[INSTRUCTION LOADER] {error_msg}")
                INSTRUCTIONS_LOAD_ERROR = error_msg
                return None

            # Check if instructions are not empty
            vn_instruction = data.get("vietnamese", {})
            en_instruction = data.get("english", {})

            if isinstance(vn_instruction, dict):
                vn_text = vn_instruction.get("system_instruction", "")
            else:
                vn_text = str(vn_instruction)

            if isinstance(en_instruction, dict):
                en_text = en_instruction.get("system_instruction", "")
            else:
                en_text = str(en_instruction)

            if not vn_text or not en_text:
                error_msg = "Invalid instructions.json: Vietnamese or English instruction is empty"
                print(f"[INSTRUCTION LOADER] {error_msg}")
                INSTRUCTIONS_LOAD_ERROR = error_msg
                return None

            print(f"[INSTRUCTION LOADER] ✓ Validated successfully")
            print(f"[INSTRUCTION LOADER] - Vietnamese instruction: {len(vn_text)} chars")
            print(f"[INSTRUCTION LOADER] - English instruction: {len(en_text)} chars")

            return data

    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse instructions.json: {e}"
        print(f"[INSTRUCTION LOADER] {error_msg}")
        INSTRUCTIONS_LOAD_ERROR = error_msg
        return None
    except Exception as e:
        error_msg = f"Unexpected error loading instructions.json: {e}"
        print(f"[INSTRUCTION LOADER] {error_msg}")
        INSTRUCTIONS_LOAD_ERROR = error_msg
        return None


# Load instructions khi khởi động module
print("\n" + "=" * 60)
print("[INSTRUCTION LOADER] Starting instruction loading...")
INSTRUCTIONS = load_instructions()
if INSTRUCTIONS is None:
    print("[INSTRUCTION LOADER] ⚠️ FAILED TO LOAD INSTRUCTIONS!")
    print("[INSTRUCTION LOADER] The service will return errors until instructions.json is properly configured")
else:
    print("[INSTRUCTION LOADER] ✓ Instructions loaded successfully!")
print("=" * 60 + "\n")


def get_instruction_by_model(model: str) -> Optional[str]:
    """
    Trả về instruction phù hợp dựa trên model name
    Returns None nếu không load được instructions
    """
    if INSTRUCTIONS is None:
        print(f"[GET INSTRUCTION] ERROR: Instructions not loaded! {INSTRUCTIONS_LOAD_ERROR}")
        return None

    if model and "eng" in model.lower():
        instruction = INSTRUCTIONS.get("english", {})
        if isinstance(instruction, dict):
            text = instruction.get("system_instruction", "")
            # Handle array format
            if isinstance(text, list):
                text = "\n".join(text)
        else:
            text = str(instruction) if instruction else ""


        if not text:
            print(f"[GET INSTRUCTION] WARNING: English instruction is empty!")
            return None

        print(f"[GET INSTRUCTION] Using English instruction ({len(text)} chars) for model: {model}")
        return text
    else:
        instruction = INSTRUCTIONS.get("vietnamese", {})
        if isinstance(instruction, dict):
            text = instruction.get("system_instruction", "")
            # Handle array format
            if isinstance(text, list):
                text = "\n".join(text)
        else:
            text = str(instruction) if instruction else ""

        if not text:
            print(f"[GET INSTRUCTION] WARNING: Vietnamese instruction is empty!")
            return None

        print(f"[GET INSTRUCTION] Using Vietnamese instruction ({len(text)} chars) for model: {model}")
        return text

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

    # Check instructions đã load chưa
    if INSTRUCTIONS is None:
        error_detail = f"Instructions not loaded. {INSTRUCTIONS_LOAD_ERROR or 'Please ensure instructions.json exists in the correct location.'}"
        print(f"[FORWARD] ERROR: {error_detail}")
        return JSONResponse(
            {"ok": False, "error": "configuration_error", "detail": error_detail},
            status_code=500
        )

    if genai is None or types is None:
        # import thất bại
        return JSONResponse({"ok": False, "error": "google-genai not installed", "detail": str(_IMPORT_ERROR)},
                            status_code=500)

    key = api_key or os.getenv("POLYDEVS_API_KEY")
    if not key:
        return JSONResponse({"ok": False, "error": "gemini/ai studio api key not provided"}, status_code=403)

    prompt_text = _extract_prompt_from_data(data)
    model = data.get("model") or DEFAULT_MODEL
    original_model = model  # Lưu lại model gốc để chọn instruction

    print(f"\n[FORWARD] Processing request with model: {original_model}")
    print(f"[FORWARD] Prompt: {prompt_text[:100]}..." if len(prompt_text) > 100 else f"[FORWARD] Prompt: {prompt_text}")

    # Model mapping cho ryuuko series
    if model == "ryuuko-r1-vnm-pro": model = "gemini-2.5-pro"
    if model == "ryuuko-r1-vnm-mini": model = "gemini-2.5-flash"
    if model == "ryuuko-r1-vnm-nano": model = "gemini-2.5-flash-lite"

    if model == "ryuuko-r1-eng-pro": model = "gemini-2.5-pro"
    if model == "ryuuko-r1-eng-mini": model = "gemini-2.5-flash"
    if model == "ryuuko-r1-eng-nano": model = "gemini-2.5-flash-lite"

    print(f"[FORWARD] Mapped to Gemini model: {model}")

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

    # build tools list if enabled
    tools_list = None
    if tools_enabled:
        try:
            tools_list = []

            # Ưu tiên Search tools
            search_added = False
            try:
                tools_list.append(types.Tool(google_search=types.GoogleSearch()))
                print("[FORWARD] Added google_search tool (standard)")
                search_added = True
            except Exception as e:
                print(f"[FORWARD] Failed to add google_search tool (standard): {e}")
                try:
                    tools_list.append(types.Tool(google_search={}))
                    print("[FORWARD] Added google_search tool (empty dict)")
                    search_added = True
                except Exception as e2:
                    print(f"[FORWARD] Failed to add google_search tool (empty dict): {e2}")

            # Thêm URL Context tool
            try:
                tools_list.append(types.Tool(url_context=types.UrlContext()))
                print("[FORWARD] Added url_context tool")
            except Exception as e:
                print(f"[FORWARD] Failed to add url_context tool: {e}")
                try:
                    tools_list.append(types.Tool(url_context={}))
                    print("[FORWARD] Added url_context tool (empty dict)")
                except Exception as e2:
                    print(f"[FORWARD] Failed to add url_context tool (empty dict): {e2}")

            # Chỉ thêm Code Execution nếu search đã hoạt động
            if search_added:
                try:
                    tools_list.append(types.Tool(code_execution=types.ToolCodeExecution()))
                    print("[FORWARD] Added code_execution tool")
                except Exception as e:
                    print(f"[FORWARD] Failed to add code_execution tool: {e}")
                    try:
                        tools_list.append(types.Tool(code_execution={}))
                        print("[FORWARD] Added code_execution tool (empty dict)")
                    except Exception as e2:
                        print(f"[FORWARD] Failed to add code_execution tool (empty dict): {e2}")
            else:
                print("[FORWARD] Skipping code_execution since search tools failed to add")

            if not tools_list:
                try:
                    tools_list = ["google_search", "url_context"]
                    print("[FORWARD] Using string tool names as fallback (search only)")
                except Exception as e:
                    print(f"[FORWARD] String tool names also failed: {e}")
                    tools_list = None

        except Exception as e:
            print(f"[FORWARD] Error building tools list: {e}")
            tools_list = None

    # Lấy instruction phù hợp dựa trên model gốc
    ryuuko_instruction = get_instruction_by_model(original_model)

    # Check instruction có được load không
    if ryuuko_instruction is None:
        error_msg = f"Failed to get instruction for model {original_model}. Instructions may not be properly loaded."
        print(f"[FORWARD] ERROR: {error_msg}")
        return JSONResponse(
            {"ok": False, "error": "instruction_error", "detail": error_msg},
            status_code=500
        )

    print(f"[FORWARD] Instruction loaded successfully ({len(ryuuko_instruction)} chars)")
    print(f"[FORWARD] First 200 chars: {ryuuko_instruction[:200]}...")

    sys_parts = []
    sys_parts.append(types.Part.from_text(text=ryuuko_instruction.strip()))

    # build GenerateContentConfig kwargs
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
        print(f"[FORWARD] Using {len(tools_list)} tools")
    if sys_parts:
        gen_cfg_kwargs["system_instruction"] = sys_parts
        print(f"[FORWARD] System instruction added to config")

    # try create GenerateContentConfig robustly
    generate_cfg = None
    try:
        generate_cfg = types.GenerateContentConfig(**gen_cfg_kwargs)
        print("[FORWARD] Created GenerateContentConfig successfully")
    except TypeError as e:
        print(f"[FORWARD] TypeError creating GenerateContentConfig: {e}")
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
            print("[FORWARD] Created GenerateContentConfig with safe kwargs")
        except Exception as e2:
            print(f"[FORWARD] Failed to create GenerateContentConfig even with safe kwargs: {e2}")
            generate_cfg = None

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def producer():
        """
        Chạy trong thread—iterate blocking stream của SDK, đẩy các chunk text vào asyncio.Queue
        """
        try:
            # Try non-streaming first
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
                print(f"[PRODUCER] Non-streaming response structure: {type(debug_response)}")
                if hasattr(debug_response, 'candidates') and debug_response.candidates:
                    cand = debug_response.candidates[0]
                    if hasattr(cand, 'content') and cand.content and hasattr(cand.content, 'parts'):
                        print(f"[PRODUCER] Number of parts: {len(cand.content.parts)}")

                        # Get complete text
                        complete_text = ""
                        for part in cand.content.parts:
                            if hasattr(part, 'text') and part.text:
                                complete_text += part.text

                        if complete_text.strip():
                            print(f"[PRODUCER] Sending complete response: {len(complete_text)} chars")
                            loop.call_soon_threadsafe(q.put_nowait, complete_text)
                            loop.call_soon_threadsafe(q.put_nowait, None)
                            return

            except Exception as debug_e:
                print(f"[PRODUCER] Debug non-streaming call failed: {debug_e}")

            # Fallback to streaming
            print("[PRODUCER] Falling back to streaming mode...")
            if generate_cfg is not None:
                print(f"[PRODUCER] Calling generate_content_stream with model={model}")
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_cfg,
                )
            else:
                print(f"[PRODUCER] Calling generate_content_stream without config, model={model}")
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                )
        except Exception as e:
            print(f"[PRODUCER] Error calling generate_content_stream: {e}")
            loop.call_soon_threadsafe(q.put_nowait, {"__error": str(e)})
            loop.call_soon_threadsafe(q.put_nowait, None)
            return

        try:
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                try:
                    if not chunk or chunk.candidates is None:
                        continue
                    cand = chunk.candidates[0]
                    if cand is None or cand.content is None or cand.content.parts is None:
                        continue

                    for part in cand.content.parts:
                        if getattr(part, "text", None) and part.text.strip():
                            loop.call_soon_threadsafe(q.put_nowait, part.text)

                except Exception as chunk_error:
                    print(f"[PRODUCER] Error processing chunk #{chunk_count}: {chunk_error}")
                    continue
        except Exception as e:
            print(f"[PRODUCER] Error in stream iteration: {e}")
            loop.call_soon_threadsafe(q.put_nowait, {"__error": str(e)})
        finally:
            print(f"[PRODUCER] Stream completed after {chunk_count if 'chunk_count' in locals() else 'unknown'} chunks")
            loop.call_soon_threadsafe(q.put_nowait, None)

    # Create client
    try:
        client = genai.Client(api_key=key)
        print("[FORWARD] Created Gemini client successfully")
    except Exception as e:
        print(f"[FORWARD] Error creating Gemini client: {e}")
        return JSONResponse({"ok": False, "error": "failed_to_create_client", "detail": str(e)}, status_code=500)

    # Start producer in thread
    asyncio.create_task(asyncio.to_thread(producer))

    async def streamer():
        """
        Async generator: yield bytes để StreamingResponse trả về.
        """
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, dict) and item.get("__error"):
                    err = item.get("__error")
                    yield (json.dumps({"ok": False, "error": "upstream_error", "detail": err}) + "\n").encode("utf-8")
                    break
                if not isinstance(item, (str, bytes)):
                    item = str(item)
                if isinstance(item, str):
                    b = item.encode("utf-8")
                else:
                    b = item
                yield b
        except asyncio.CancelledError:
            return

    return StreamingResponse(streamer(), media_type="text/plain; charset=utf-8")