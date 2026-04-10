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

_allowed: set[int] = set()


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
        except Exception as e:
            logger.warning(f"Could not load whitelist: {e}")

    logger.info(f"Whitelist loaded: {len(_allowed)} users")


def _save() -> None:
    try:
        _WHITELIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        _WHITELIST_FILE.write_text(
            json.dumps({"allowed": sorted(_allowed)}, indent=2)
        )
    except Exception as e:
        logger.warning(f"Could not save whitelist: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def is_allowed(user_id: int) -> bool:
    return user_id in _allowed


def is_admin(user_id: int) -> bool:
    return ADMIN_USER_ID != 0 and user_id == ADMIN_USER_ID


def add_user(user_id: int) -> bool:
    """Add a user. Returns False if already in list."""
    if user_id in _allowed:
        return False
    _allowed.add(user_id)
    _save()
    logger.info(f"Whitelist: added {user_id}")
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


def list_users() -> list[int]:
    return sorted(_allowed)


# Load on import
_load()
