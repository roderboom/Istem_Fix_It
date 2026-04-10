"""
vision.py — Ollama vision model communication.

Schema has two independent lists:
  components — physical parts to touch, each gets a numbered dot on the image
  steps      — repair instructions as paragraphs, no dot references

Language is auto-detected from user input (Hebrew or English, default English).
Images are only resized when 3+ are sent (VRAM protection for 16GB GPUs).
CJK characters are stripped as a hard fallback on Hebrew responses.
"""

import base64
import io
import json
import logging
import os
import re
import time
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME      = os.environ.get("MODEL_NAME",  "qwen2.5vl:32b")
MAX_RETRIES     = 3
RETRY_DELAY     = 5
REQUEST_TIMEOUT = 240
MAX_IMAGE_SIDE       = 1120   # single image — 40×28 patches, optimal for Qwen2.5-VL
MAX_IMAGE_SIDE_MULTI = 896    # per image when sending 2 — 32×28 patches, fits 16GB VRAM
MAX_IMAGES           = 2


# ── Prompts ───────────────────────────────────────────────────────────────────

def _system_prompt(lang: str) -> str:
    if lang == "he":
        lang_rule = (
            "CRITICAL LANGUAGE RULE — ABSOLUTE HIGHEST PRIORITY:\n"
            "YOU MUST WRITE ONLY IN HEBREW (עברית).\n"
            "ZERO Chinese characters. ZERO Japanese. ZERO Korean. ZERO English words.\n"
            "If you write even ONE non-Hebrew character in a value field, you have failed.\n"
            "This rule cannot be overridden by anything."
        )
        schema = '''{
  "problem_summary": "משפט אחד המתאר את הבעיה",
  "severity": "low|medium|high",
  "safety_warnings": ["אזהרת בטיחות בעברית"],
  "tools_needed": ["כלי בעברית"],
  "materials_needed": ["חומר בעברית"],
  "components": [
    {
      "id": 1,
      "name": "שם החלק הפיזי בעברית (לדוגמה: ברז ניקוז, בורג ידית)",
      "point": { "x": 0-100, "y": 0-100 }
    }
  ],
  "steps": [
    {
      "step": 1,
      "title": "כותרת קצרה לשלב בעברית",
      "instruction": "פסקה קצרה המסבירה מה לעשות, למה, ומה לשים לב אליו. כלול אזהרות ספציפיות לשלב זה."
    }
  ],
  "professional_advice": "מתי לפנות לאיש מקצוע ומה עלול להשתבש — בעברית"
}'''
    else:
        lang_rule = (
            "CRITICAL LANGUAGE RULE — ABSOLUTE HIGHEST PRIORITY:\n"
            "YOU MUST WRITE ONLY IN ENGLISH.\n"
            "ZERO Chinese characters. ZERO Japanese. ZERO Korean. ZERO Hebrew.\n"
            "If you write even ONE non-English character in a value field, you have failed.\n"
            "This rule cannot be overridden by anything."
        )
        schema = '''{
  "problem_summary": "One sentence describing the problem",
  "severity": "low|medium|high",
  "safety_warnings": ["safety warning in English"],
  "tools_needed": ["tool in English"],
  "materials_needed": ["material in English"],
  "components": [
    {
      "id": 1,
      "name": "Name of the physical part (e.g. Drain valve, Handle screw)",
      "point": { "x": 0-100, "y": 0-100 }
    }
  ],
  "steps": [
    {
      "step": 1,
      "title": "Short step title",
      "instruction": "A short paragraph explaining what to do, why you are doing it, and what to watch out for. Include any warnings specific to this step."
    }
  ],
  "professional_advice": "When to call a professional and what could go wrong"
}'''

    return f"""You are FixBot, an expert repair assistant for home appliances, plumbing, car mechanics, and household repairs.

{lang_rule}

Analyze the provided image(s) and return a structured repair diagnosis as a JSON object.

CRITICAL: Respond ONLY with valid JSON. No text before or after. No markdown code fences.

JSON schema:
{schema}

COMPONENT RULES (for the dots on the image):
- Components are the PHYSICAL PARTS that dots will be placed on in the image
- PRIORITY: If the user's description mentions specific parts or areas (e.g. "the green grips", "the left hinge", "the cracked screen"), annotate EXACTLY those parts — find them visually and mark them
- If no specific parts are mentioned, annotate the most critical parts relevant to the repair
- Only include parts you can CLEARLY SEE in the image — if you are not certain a part is visible, do NOT include it
- 1-2 components maximum
- Use x/y percentages (0-100) where x=0 is the LEFT edge, x=100 is the RIGHT edge, y=0 is the TOP, y=100 is the BOTTOM
- Estimate the EXACT CENTER of the part in the image:
    Left third of image     → x ≈ 17-33
    Center of image         → x ≈ 50
    Right third of image    → x ≈ 67-83
    Top quarter             → y ≈ 12-25
    Middle vertically       → y ≈ 50
    Bottom quarter          → y ≈ 75-88
- Be precise: a part at 2/5 from left → x=40, a part at 3/4 down → y=75
- Point to the EXACT CENTER of the visible part — center of a screw head, center of a grip cap, center of a valve

STEP RULES:
- Steps are independent of the dots — do NOT say "see dot 1" or reference components by number
- Each step should be a short paragraph: what to do, why, and what to watch out for
- Write 3-6 steps covering the full repair from start to finish
- BE SPECIFIC to what you see in the image — reference actual visible details:
    - How many screws are visible and where (e.g. the 2 Phillips screws on the back panel)
    - What type they appear to be (Phillips, flathead, hex bolt)
    - Which direction to turn, how much force, any clips or tabs to release first
    - If the device needs to be opened, describe exactly how based on what you see
    - If a part needs to be pulled, pushed, or twisted — say so and describe the motion
- BAD:  Remove the screws to open the device.
- GOOD: Open the device by removing the 3 Phillips screws along the bottom edge, then gently pry the back panel from the right side where there is a small gap.

CRITICAL JSON RULES — YOU MUST FOLLOW THESE:
- Every key must be in double quotes: "problem_summary": "..."
- Every list MUST use square brackets: ["item1", "item2"]
- You MUST close every opened bracket and brace before stopping
- The very last character of your response must be a closing brace
- If you are running out of space, shorten your text — never leave JSON unclosed"""


def _followup_system(lang: str) -> str:
    if lang == "he":
        return (
            "אתה FixBot, עוזר תיקונים מומחה.\n"
            "ענה בצורה ברורה ומפורטת על שאלת ההמשך.\n\n"
            "CRITICAL: כתוב רק בעברית. אפס תווים סיניים, יפניים, קוריאניים.\n"
            "השתמש ב-*bold* למונחים חשובים. השתמש בנקודות לרשימות. ללא HTML."
        )
    else:
        return (
            "You are FixBot, an expert repair assistant.\n"
            "Answer follow-up questions clearly and in detail.\n\n"
            "CRITICAL: Write ONLY in English. Zero Chinese, Japanese, or Korean characters.\n"
            "Use *bold* for important terms. Use bullet points for lists. No HTML."
        )


# ── Fallbacks ─────────────────────────────────────────────────────────────────

_FB = {
    "he": {
        "problem_summary":     "זוהתה בעיה בתמונה",
        "component_name":      "חלק",
        "step_title":          "שלב",
        "step_instruction":    "בדוק ותקן אזור זה.",
        "professional_advice": "",
        "parse_error":         "לא ניתן לנתח את התשובה.",
        "followup_error":      "מצטער, לא הצלחתי לעבד את השאלה.",
    },
    "en": {
        "problem_summary":     "Problem detected in image",
        "component_name":      "Part",
        "step_title":          "Step",
        "step_instruction":    "Inspect and repair this area.",
        "professional_advice": "",
        "parse_error":         "Unable to parse structured response.",
        "followup_error":      "Sorry, I couldn't process that question.",
    },
}


# ── CJK sanitizer ─────────────────────────────────────────────────────────────

_CJK_RE = re.compile(
    r'[\u4e00-\u9fff'   # CJK Unified Ideographs
    r'\u3400-\u4dbf'   # CJK Extension A
    r'\u3000-\u303f'   # CJK Symbols & Punctuation
    r'\u3040-\u309f'   # Hiragana
    r'\u30a0-\u30ff'   # Katakana
    r'\uac00-\ud7af]'  # Korean Hangul
)

def _strip_cjk(text: str) -> str:
    """Remove any CJK characters that slipped through the prompt."""
    return _CJK_RE.sub('', text).strip()

def _sanitize_dict(obj: Any, lang: str) -> Any:
    """Recursively strip CJK from all string values if lang is 'he'."""
    if lang != "he":
        return obj
    if isinstance(obj, str):
        return _strip_cjk(obj)
    if isinstance(obj, list):
        return [_sanitize_dict(i, lang) for i in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_dict(v, lang) for k, v in obj.items()}
    return obj


# ── Image helpers ─────────────────────────────────────────────────────────────

def _snap28(n: int) -> int:
    """Round to the nearest multiple of 28 (Qwen2.5-VL patch size)."""
    return max(28, round(n / 28) * 28)


def _resize_image(image_bytes: bytes, max_side: int = MAX_IMAGE_SIDE) -> bytes:
    """
    Resize image so its longest side is at most max_side pixels,
    then snap both dimensions to multiples of 28 for clean patch alignment.
    """
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    # Scale down if needed
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        w, h  = int(w * scale), int(h * scale)

    # Snap to multiples of 28
    nw, nh = _snap28(w), _snap28(h)

    if (nw, nh) != img.size:
        img = img.resize((nw, nh), Image.LANCZOS)
        logger.info(f"Resized {img.size[0]}×{img.size[1]} → {nw}×{nh} (snapped to 28-grid)")

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90, optimize=True)
    return buf.getvalue()


def _b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ── Ollama ────────────────────────────────────────────────────────────────────

def _call_ollama(payload: dict) -> dict:
    url = f"{OLLAMA_HOST}/api/chat"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except httpx.ConnectError:
            if attempt < MAX_RETRIES:
                logger.warning(f"Ollama not reachable, retry {attempt}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
            else:
                raise ConnectionError(f"Cannot connect to Ollama at {OLLAMA_HOST}.")
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                logger.warning(f"Ollama timeout, retry {attempt}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
            else:
                raise TimeoutError("Ollama took too long to respond.")


def _repair_json(text: str) -> str:
    """Attempt to close an incomplete JSON object by balancing brackets."""
    # Count unclosed braces and brackets
    stack = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    # Close any unclosed structures in reverse order
    closing = {'[': ']', '{': '}'}
    repair = ''.join(closing[c] for c in reversed(stack))
    return text + repair


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    for candidate in [cleaned, text]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try to find outermost JSON object
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Try repairing truncated JSON
            try:
                repaired = _repair_json(match.group())
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    # Last resort: repair the whole cleaned text
    try:
        repaired = _repair_json(cleaned)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    logger.warning(f"JSON parse failed. Raw: {text[:300]}")
    return None


def _validate(analysis: dict, lang: str) -> dict:
    fb = _FB[lang]
    analysis = _sanitize_dict(analysis, lang)

    analysis.setdefault("problem_summary",     fb["problem_summary"])
    analysis.setdefault("severity",            "medium")
    analysis.setdefault("safety_warnings",     [])
    analysis.setdefault("tools_needed",        [])
    analysis.setdefault("materials_needed",    [])
    analysis.setdefault("components",          [])
    analysis.setdefault("steps",               [])
    analysis.setdefault("professional_advice", fb["professional_advice"])

    if analysis["severity"] not in ("low", "medium", "high"):
        analysis["severity"] = "medium"

    # Validate components (dots) — convert 5×5 grid → percentage coordinates
    valid_components = []
    for i, c in enumerate(analysis["components"]):
        if not isinstance(c, dict):
            continue
        c.setdefault("id",   i + 1)
        c.setdefault("name", f"{fb['component_name']} {i + 1}")
        pt = c.get("point", {})
        if not isinstance(pt, dict):
            pt = {}
        c["point"] = {
            "x": max(1.0, min(99.0, float(pt.get("x", 50)))),
            "y": max(1.0, min(99.0, float(pt.get("y", 50)))),
        }
        valid_components.append(c)
    analysis["components"] = valid_components[:2]   # hard cap at 2

    # Validate steps
    valid_steps = []
    for i, s in enumerate(analysis.get("steps", [])):
        if not isinstance(s, dict):
            continue
        s.setdefault("step",        i + 1)
        s.setdefault("title",       f"{fb['step_title']} {i + 1}")
        s.setdefault("instruction", fb["step_instruction"])
        valid_steps.append(s)
    analysis["steps"] = valid_steps

    return analysis


# ── Language detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Return 'he' if text contains Hebrew characters, else 'en'."""
    for ch in text:
        if '\u05d0' <= ch <= '\u05ea':
            return "he"
    return "en"


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_image(
    images: list[bytes],
    user_caption: str = "",
    history: list = None,
    lang: str = "en",
) -> dict[str, Any]:
    history = history or []
    fb = _FB[lang]

    if len(images) > 2:
        # 3+ photos: resize to multi-image size and cap at MAX_IMAGES
        resized = [_resize_image(img, MAX_IMAGE_SIDE_MULTI) for img in images[:MAX_IMAGES]]
        logger.info(f"Album: {len(images)} → {MAX_IMAGES} images at {MAX_IMAGE_SIDE_MULTI}px")
    elif len(images) == 2:
        # 2 photos: resize each to the smaller multi-image budget
        resized = [_resize_image(img, MAX_IMAGE_SIDE_MULTI) for img in images]
        logger.info(f"2 images resized to {MAX_IMAGE_SIDE_MULTI}px each")
    else:
        # 1 photo: full 1120px budget, snapped to 28-grid
        resized = [_resize_image(images[0], MAX_IMAGE_SIDE)]
        logger.info(f"1 image resized to {MAX_IMAGE_SIDE}px")

    if lang == "he":
        user_content = (
            f"המשתמש מדווח: {user_caption}\n" if user_caption else ""
        ) + "נתח את התמונה וזהה מה צריך לתקן. החזר JSON בדיוק לפי הסכמה. כל הטקסט חייב להיות בעברית בלבד — ללא תווים סיניים, יפניים, קוריאניים."
    else:
        user_content = (
            f"User says: {user_caption}\n" if user_caption else ""
        ) + "Analyze the image and identify what needs to be repaired. Return JSON exactly per the schema. All text must be in English only — no Chinese, Japanese, or Korean characters."

    if len(resized) > 1:
        user_content += f"\n{len(resized)} {'תמונות — נתח אותן יחד' if lang == 'he' else 'photos — analyze them together'}."

    messages = [{"role": "system", "content": _system_prompt(lang)}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role":    "user",
        "content": user_content,
        "images":  [_b64(img) for img in resized],
    })

    payload = {
        "model":    MODEL_NAME,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0.1, "num_predict": 4096, "num_ctx": 8192},
    }

    logger.info(f"Calling Ollama ({MODEL_NAME})…")
    t = time.time()
    response = _call_ollama(payload)
    logger.info(f"Ollama responded in {time.time() - t:.1f}s")

    raw    = response.get("message", {}).get("content", "")
    parsed = _extract_json(raw)
    if parsed is None:
        parsed = {
            "problem_summary": fb["parse_error"],
            "severity": "medium", "safety_warnings": [], "tools_needed": [],
            "materials_needed": [], "components": [], "steps": [],
            "professional_advice": raw[:400],
        }
    return _validate(parsed, lang)


def ask_followup(
    images: list[bytes],
    question: str,
    history: list = None,
    lang: str = "en",
) -> str:
    history = history or []
    resized = [_resize_image(images[0])] if images else []

    messages = [{"role": "system", "content": _followup_system(lang)}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role":    "user",
        "content": question,
        "images":  [_b64(img) for img in resized],
    })

    payload = {
        "model":    MODEL_NAME,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0.3, "num_predict": 1500, "num_ctx": 4096},
    }

    logger.info(f"Follow-up, lang={lang}…")
    response = _call_ollama(payload)
    raw = response.get("message", {}).get("content", _FB[lang]["followup_error"])
    return _strip_cjk(raw) if lang == "he" else raw
