"""
Live chute-hotness probe (for the 'active miner' dashboard filter).

A miner is *active* when it is valid AND its Chute is currently hot (reachable). Results
go through ``get_chute_info``'s CHUTE_CACHE_TTL cache, so repeated dashboard calls within the
TTL don't re-hit the Chutes API.
"""
import asyncio
import os
from typing import Dict, Iterable

import aiohttp

from leoma.infra.chute_resolver import get_chute_info

# Cap concurrent upstream probes so a cold cache can't fan out to the whole miner set at once.
_PROBE_CONCURRENCY = int(os.environ.get("CHUTE_PROBE_CONCURRENCY", "8"))


async def probe_hot_chutes(chute_ids: Iterable[str]) -> Dict[str, bool]:
    """Return ``{chute_id: is_hot}`` for the given chute ids (cached per CHUTE_CACHE_TTL)."""
    ids = [c for c in dict.fromkeys(chute_ids) if c]  # dedupe, drop empties, keep order
    if not ids:
        return {}
    sem = asyncio.Semaphore(_PROBE_CONCURRENCY)

    async def _one(session: aiohttp.ClientSession, cid: str):
        async with sem:
            return await get_chute_info(session, cid)

    async with aiohttp.ClientSession() as session:
        infos = await asyncio.gather(
            *(_one(session, cid) for cid in ids), return_exceptions=True
        )
    out: Dict[str, bool] = {}
    for cid, info in zip(ids, infos):
        out[cid] = bool(
            info and not isinstance(info, Exception) and isinstance(info, dict) and info.get("hot")
        )
    return out
