"""
history.py — Persists repair history to disk as JSON.

Each user gets their own file: /data/history/<user_id>.json
The /history command reads from this file and formats it for Telegram.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_DIR = Path(os.environ.get("HISTORY_DIR", "/data/history"))


def _user_file(uid: int) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR / f"{uid}.json"


def _load(uid: int) -> list:
    f = _user_file(uid)
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text())
    except Exception as e:
        logger.warning(f"Could not load history for {uid}: {e}")
        return []


def _save(uid: int, records: list) -> None:
    try:
        _user_file(uid).write_text(json.dumps(records, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Could not save history for {uid}: {e}")


def save_repair(uid: int, analysis: dict) -> None:
    """Append a completed repair analysis to the user's history file."""
    records = _load(uid)
    records.append({
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "problem_summary": analysis.get("problem_summary", ""),
        "severity":        analysis.get("severity", "medium"),
        "tools_needed":    analysis.get("tools_needed", []),
        "professional_advice": analysis.get("professional_advice", ""),
    })
    # Keep only the last 50 repairs
    _save(uid, records[-50:])


def get_history_text(uid: int, lang: str = "en") -> str:
    """Return a Telegram-formatted string of the user's repair history."""
    records = _load(uid)
    if not records:
        return ("📭 עדיין אין היסטוריית תיקונים. שלחו תמונה כדי להתחיל!"
                if lang == "he" else
                "📭 No repair history yet. Send a photo to get started!")

    if lang == "he":
        header = f"🗂 *היסטוריית תיקונים* ({len(records)} תיקונים)\n"
        footer = f"_מציג 10 אחרונים מתוך {len(records)}._"
        unknown = "לא ידוע"
    else:
        header = f"🗂 *Repair History* ({len(records)} repairs)\n"
        footer = f"_Showing 10 most recent of {len(records)}._"
        unknown = "Unknown"

    lines = [header]
    for r in reversed(records[-10:]):
        ts       = r.get("timestamp", "")[:10]
        summary  = r.get("problem_summary", unknown)
        severity = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(r.get("severity", "medium"), "🟡")
        cost     = r.get("estimated_cost", "")

        lines.append(f"{severity} *{summary}*")
        lines.append(f"   📅 {ts}\n")

    if len(records) > 10:
        lines.append(footer)

    return "\n".join(lines)


def clear_history(uid: int) -> int:
    """Delete all history for a user. Returns number of records deleted."""
    records = _load(uid)
    count   = len(records)
    _save(uid, [])
    return count
