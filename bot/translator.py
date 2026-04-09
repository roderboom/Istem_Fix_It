"""
translator.py — Local translation using the same Ollama model.

No external API, no cost, no account needed.
Ollama already understands repair/technical context in both Hebrew and English.

Trade-off: adds ~20-30 seconds per Hebrew interaction vs near-instant
for a cloud API. Acceptable given the model is already running locally.
"""

import logging
import time

import httpx

from vision import OLLAMA_HOST, MODEL_NAME, _call_ollama

logger = logging.getLogger(__name__)


_TRANSLATE_SYSTEM = (
    "You are a precise technical translator specializing in repair, "
    "electronics, plumbing, and mechanical content. "
    "Translate the given text exactly — preserve all formatting, "
    "bullet points, bold markers (*word*), and line breaks. "
    "Never add explanations. Never change technical meaning. "
    "Output ONLY the translated text, nothing else."
)


def _translate(text: str, target_lang: str) -> str:
    """
    Translate text to target_lang ('Hebrew' or 'English') via Ollama.
    Returns original text on failure.
    """
    if not text or not text.strip():
        return text

    if target_lang == "Hebrew":
        instruction = f"Translate the following English repair text to Hebrew:\n\n{text}"
    else:
        instruction = f"Translate the following Hebrew repair text to English:\n\n{text}"

    payload = {
        "model":    MODEL_NAME,
        "messages": [
            {"role": "system",  "content": _TRANSLATE_SYSTEM},
            {"role": "user",    "content": instruction},
        ],
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": 1500, "num_ctx": 2048},
    }

    try:
        t        = time.time()
        response = _call_ollama(payload)
        result   = response.get("message", {}).get("content", "").strip()
        logger.info(f"Translated to {target_lang} in {time.time()-t:.1f}s ({len(text)}→{len(result)} chars)")
        return result if result else text
    except Exception as e:
        logger.warning(f"Ollama translation failed: {e}")
        return text  # silent fallback — return original


def to_english(text: str) -> str:
    """Translate Hebrew user input to English."""
    return _translate(text, "English")


def to_hebrew(text: str) -> str:
    """Translate English model output to Hebrew."""
    return _translate(text, "Hebrew")


def translate_analysis(analysis: dict) -> dict:
    """
    Translate all user-visible text fields in the analysis dict to Hebrew.
    Batches all strings into one Ollama call for context quality and speed.
    Returns a new dict (original untouched).
    """
    a = dict(analysis)

    # ── Build a numbered block of all strings to translate ───────────────────
    # Format: "1. <text>" so the model returns "1. <translated>"
    # This keeps everything in one call with full context.

    fields  = []   # (index, string) pairs — only non-empty strings
    mapping = []   # how to write back each translated string

    def collect(value, setter):
        if value and str(value).strip():
            fields.append(str(value))
            mapping.append(setter)

    collect(a.get("problem_summary",     ""), lambda v: a.update({"problem_summary": v}))
    collect(a.get("professional_advice", ""), lambda v: a.update({"professional_advice": v}))

    warnings  = list(a.get("safety_warnings",  []))
    tools     = list(a.get("tools_needed",      []))
    materials = list(a.get("materials_needed",  []))
    components= [dict(c) for c in a.get("components", [])]
    steps     = [dict(s) for s in a.get("steps", [])]

    for i, w in enumerate(warnings):
        collect(w, lambda v, i=i: warnings.__setitem__(i, v))
    for i, t in enumerate(tools):
        collect(t, lambda v, i=i: tools.__setitem__(i, v))
    for i, m in enumerate(materials):
        collect(m, lambda v, i=i: materials.__setitem__(i, v))
    for i, c in enumerate(components):
        collect(c.get("name", ""), lambda v, i=i: components[i].update({"name": v}))
    for i, s in enumerate(steps):
        collect(s.get("title",       ""), lambda v, i=i: steps[i].update({"title": v}))
        collect(s.get("instruction", ""), lambda v, i=i: steps[i].update({"instruction": v}))

    if not fields:
        return a

    # Format as numbered list for reliable parsing
    numbered_input = "\n".join(f"{i+1}. {text}" for i, text in enumerate(fields))

    instruction = (
        f"Translate the following {len(fields)} numbered repair guide items from English to Hebrew. "
        f"Return ONLY the numbered list in the same format. Do not add or remove items.\n\n"
        f"{numbered_input}"
    )

    payload = {
        "model":    MODEL_NAME,
        "messages": [
            {"role": "system",  "content": _TRANSLATE_SYSTEM},
            {"role": "user",    "content": instruction},
        ],
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": 2000, "num_ctx": 3000},
    }

    try:
        t        = time.time()
        response = _call_ollama(payload)
        raw      = response.get("message", {}).get("content", "").strip()
        logger.info(f"Batch translation in {time.time()-t:.1f}s")

        # Parse "1. text\n2. text\n..." back into list
        translated = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading "N. " or "N) "
            import re
            m = re.match(r"^\d+[.)]\s*(.+)$", line)
            if m:
                translated.append(m.group(1).strip())

        if len(translated) == len(fields):
            for setter, value in zip(mapping, translated):
                setter(value)
        else:
            logger.warning(
                f"Batch parse mismatch: expected {len(fields)}, got {len(translated)}. "
                "Falling back to field-by-field."
            )
            # Fallback: translate individually
            for i, (text, setter) in enumerate(zip(fields, mapping)):
                result = _translate(text, "Hebrew")
                setter(result)

    except Exception as e:
        logger.warning(f"Batch translation failed: {e} — skipping translation")

    a["safety_warnings"]  = warnings
    a["tools_needed"]     = tools
    a["materials_needed"] = materials
    a["components"]       = components
    a["steps"]            = steps

    return a
