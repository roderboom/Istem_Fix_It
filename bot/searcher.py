"""
searcher.py — Google Custom Search integration for FixBot.

Free tier: 100 queries/day, no credit card required.

Setup:
  1. https://programmablesearchengine.google.com/ — create engine, enable "Search entire web", copy Search Engine ID
  2. https://console.cloud.google.com/ — enable Custom Search API, create API key

Set in .env:
  GOOGLE_API_KEY=your_api_key
  GOOGLE_CSE_ID=your_search_engine_id
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID  = os.environ.get("GOOGLE_CSE_ID",  "")
GOOGLE_URL     = "https://www.googleapis.com/customsearch/v1"
TIMEOUT        = 10


def search_repair(problem: str) -> str:
    """
    Search Google for repair guides for the given problem string.
    Returns a trimmed context string (max ~2500 chars) ready to inject
    into the model prompt. Returns empty string on any failure.

    Always searches in English — repair guides are far more abundant in English.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.warning("GOOGLE_API_KEY or GOOGLE_CSE_ID not set — skipping web search")
        return ""

    query = f"how to fix {problem} repair guide DIY"
    logger.info(f"Google search: {query!r}")

    try:
        r = httpx.get(
            GOOGLE_URL,
            params={
                "key": GOOGLE_API_KEY,
                "cx":  GOOGLE_CSE_ID,
                "q":   query,
                "num": 5,
                "lr":  "lang_en",
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("Google search: daily quota of 100 queries reached")
        else:
            logger.warning(f"Google search HTTP error: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Google search failed: {e}")
        return ""

    items = data.get("items", [])
    if not items:
        logger.info("Google returned no results")
        return ""

    parts = []
    for item in items[:3]:
        title   = item.get("title", "").strip()
        snippet = item.get("snippet", "").replace("\n", " ").strip()
        if snippet:
            parts.append(f"• {title}\n  {snippet}")

    if not parts:
        return ""

    context = (
        "REPAIR KNOWLEDGE FROM THE WEB (use this to give more accurate, specific advice):\n\n"
        + "\n\n".join(parts)
    )
    logger.info(f"Search context: {len(context)} chars from {len(parts)} results")
    return context[:2500]
