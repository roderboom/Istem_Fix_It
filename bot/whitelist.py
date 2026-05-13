"""
whitelist.py — Persistent user whitelist for FixBot.

Allowed users are stored in a JSON file so the list survives restarts.
The initial list can be seeded via the ALLOWED_USERS env var.
The admin user is set via ADMIN_USER_ID env var.

Admin Telegram commands:
  /adduser <user_id>     — allow a user
  /removeuser <user_id>  — revoke a user
  /listusers             — show all allowed users
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_WHITELIST_FILE = Path(os.environ.get("HISTORY_DIR", "/data/history")) / "whitelist.json"
ADMIN_USER_ID   = int(os.environ.get("ADMIN_USER_ID", "0"))

# ── Internal state ────────────────────────────────────────────────────────────

_allowed:    set[int]        = set()
_user_names: dict[int, str]  = {}    # uid → "Full Name (@username)"


def _load() -> None:
    global _allowed
    # Seed from env var (comma-separated IDs)
    env_ids = os.environ.get("ALLOWED_USERS", "")
    if env_ids:
        for raw in env_ids.split(","):
            raw = raw.strip()
            if raw.isdigit():
                _allowed.add(int(raw))

    # Always allow admin
    if ADMIN_USER_ID:
        _allowed.add(ADMIN_USER_ID)

    # Load persisted list
    if _WHITELIST_FILE.exists():
        try:
            data = json.loads(_WHITELIST_FILE.read_text())
            _allowed.update(int(uid) for uid in data.get("allowed", []))
            _user_names.update({int(k): v for k, v in data.get("names", {}).items()})
        except Exception as e:
            logger.warning(f"Could not load whitelist: {e}")

    logger.info(f"Whitelist loaded: {len(_allowed)} users")


def _save() -> None:
    try:
        _WHITELIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"allowed": sorted(_allowed), "names": _user_names}
        _WHITELIST_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"Could not save whitelist: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def is_allowed(user_id: int) -> bool:
    return user_id in _allowed


def is_admin(user_id: int) -> bool:
    return ADMIN_USER_ID != 0 and user_id == ADMIN_USER_ID


def add_user(user_id: int, display_name: str = "") -> bool:
    """Add a user. Returns False if already in list."""
    if user_id in _allowed:
        return False
    _allowed.add(user_id)
    if display_name:
        _user_names[user_id] = display_name
    _save()
    logger.info(f"Whitelist: added {user_id} ({display_name or 'no name'})")
    return True


def remove_user(user_id: int) -> bool:
    """Remove a user. Returns False if not in list. Cannot remove admin."""
    if user_id == ADMIN_USER_ID:
        return False
    if user_id not in _allowed:
        return False
    _allowed.discard(user_id)
    _save()
    logger.info(f"Whitelist: removed {user_id}")
    return True


def list_users() -> list[tuple[int, str]]:
    """Return list of (uid, display_name) sorted by uid."""
    return [(uid, _user_names.get(uid, "")) for uid in sorted(_allowed)]


# ── Ban list ─────────────────────────────────────────────────────────────────

_banned: set[int] = set()
_ban_notified: set[int] = set()  # users who already received the ban message


def _load_banned() -> None:
    global _banned
    if _WHITELIST_FILE.exists():
        try:
            data = json.loads(_WHITELIST_FILE.read_text())
            _banned.update(int(uid) for uid in data.get("banned", []))
        except Exception as e:
            logger.warning(f"Could not load ban list: {e}")


def _save_banned() -> None:
    """Save banned list into the same whitelist.json."""
    try:
        _WHITELIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if _WHITELIST_FILE.exists():
            existing = json.loads(_WHITELIST_FILE.read_text())
        existing["banned"] = sorted(_banned)
        _WHITELIST_FILE.write_text(json.dumps(existing, indent=2))
    except Exception as e:
        logger.warning(f"Could not save ban list: {e}")


def is_banned(user_id: int) -> bool:
    return user_id in _banned


def ban_user(user_id: int) -> bool:
    """Ban a user. Returns False if already banned. Cannot ban admin."""
    if user_id == ADMIN_USER_ID:
        return False
    if user_id in _banned:
        return False
    _banned.add(user_id)
    _allowed.discard(user_id)  # remove from whitelist too
    _save()
    _save_banned()
    logger.info(f"Banned: {user_id}")
    return True


def unban_user(user_id: int) -> bool:
    """Unban a user. Returns False if not banned."""
    if user_id not in _banned:
        return False
    _banned.discard(user_id)
    _ban_notified.discard(user_id)
    _save_banned()
    logger.info(f"Unbanned: {user_id}")
    return True


def was_ban_notified(user_id: int) -> bool:
    return user_id in _ban_notified


def mark_ban_notified(user_id: int) -> None:
    _ban_notified.add(user_id)


# Load on import
_load()
_load_banned()
