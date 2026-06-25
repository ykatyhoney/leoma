"""
Chutes API client: resolve chute IDs to endpoints.
"""
import asyncio
import time
from typing import Any, Dict

import aiohttp

from leoma.bootstrap import CHUTES_API_URL, CHUTES_API_KEY, CHUTE_CACHE_TTL, emit_log

_chute_info_cache: Dict[str, tuple[Dict[str, Any], float]] = {}
_inflight_locks: Dict[str, asyncio.Lock] = {}
_REQUEST_TIMEOUT_SECONDS = 10


def _lock_for(chute_id: str) -> asyncio.Lock:
    """Per-chute-id lock so concurrent callers for the same id share one upstream request."""
    lock = _inflight_locks.get(chute_id)
    if lock is None:
        lock = asyncio.Lock()
        _inflight_locks[chute_id] = lock
    return lock


def _chutes_auth_headers() -> Dict[str, str]:
    return {"Authorization": CHUTES_API_KEY} if CHUTES_API_KEY else {}


def _get_cached_chute_info(chute_id: str) -> Dict[str, Any] | None:
    cached_entry = _chute_info_cache.get(chute_id)
    if cached_entry is None:
        return None
    info, cached_at = cached_entry
    if time.time() - cached_at < CHUTE_CACHE_TTL:
        return info
    return None


def _set_cached_chute_info(chute_id: str, info: Dict[str, Any]) -> None:
    _chute_info_cache[chute_id] = (info, time.time())


async def get_chute_info(session: aiohttp.ClientSession, chute_id: str) -> Dict[str, Any] | None:
    """Get chute info from Chutes API by chute_id (cached; single-flight per id)."""
    cached_info = _get_cached_chute_info(chute_id)
    if cached_info is not None:
        return cached_info
    # Serialize concurrent fetches for the same id so a cold cache doesn't fan out N×M requests.
    async with _lock_for(chute_id):
        cached_info = _get_cached_chute_info(chute_id)
        if cached_info is not None:
            return cached_info
        try:
            url = f"{CHUTES_API_URL}/chutes/{chute_id}"
            async with session.get(
                url,
                headers=_chutes_auth_headers(),
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status != 200:
                    emit_log(f"Chutes API error for {chute_id}: {resp.status}", "warn")
                    return None
                info = await resp.json()
                _set_cached_chute_info(chute_id, info)
                return info
        except asyncio.TimeoutError:
            emit_log(f"Timeout fetching chute info: {chute_id}", "warn")
            return None
        except Exception as e:
            emit_log(f"Error fetching chute {chute_id}: {e}", "warn")
            return None


def build_chute_endpoint(slug: str) -> str:
    """Build the I2V generation endpoint URL from chute slug."""
    return f"https://{slug}.chutes.ai/generate"
