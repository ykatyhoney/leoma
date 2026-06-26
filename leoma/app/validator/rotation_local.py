"""Local rotation resolution (owner-api-free).

Whose turn it is to sample is a deterministic function of the current chain block and the hardcoded
validator allowlist, so each validator computes it itself — there is no ``GET /rotation``. Failover is
purely block-derived, and "did the scheduled sampler already produce this rotation?" is read from its
bucket, so there is no claim lease either. Duplicate production during a failover race is harmless:
the window's canonical-sampler resolution deterministically keeps the earliest-failover-order producer.
"""
import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional

from leoma.bootstrap import SAMPLING_ROTATION_INTERVAL, emit_log as log
from leoma.infra.allowlist import load_allowlist
from leoma.infra.peer_registry import load_peers
from leoma.infra.rotation_math import compute_sampler, effective_sampler, grace_blocks_for
from leoma.infra.storage_backend import create_peer_read_client

_GRACE_OVERRIDE = int(os.environ.get("SAMPLER_FAILOVER_GRACE_BLOCKS", "0"))


@dataclass
class RotationView:
    rotation_index: int
    interval: int
    validators: List[str]
    sampler: Optional[str]      # effective sampler (after any failover)
    failover_step: int
    is_your_turn: bool
    produced: bool              # has the current rotation already been produced?


class LocalRotation:
    """Resolves the current sampling turn from chain block + hardcoded allowlist (no owner-api)."""

    def __init__(self, subtensor, my_hotkey: str):
        self._sub = subtensor
        self._me = my_hotkey

    async def _produced(self, rotation_id: int, sampler: Optional[str]) -> bool:
        """True if ``sampler`` has already published a verdict file for ``rotation_id``."""
        if not sampler:
            return False
        peer = load_peers().get(sampler)
        if peer is None:
            return False
        safe = sampler.replace("/", "_").replace("\\", "_")
        key = f"{rotation_id}/evaluation_results/{safe}.json"
        try:
            client = create_peer_read_client(peer)
            resp = await asyncio.to_thread(client.get_object, peer.bucket, key)
            resp.close()
            resp.release_conn()
            return True
        except Exception:
            return False

    async def whose_turn(self) -> Optional[RotationView]:
        """Current rotation view, or ``None`` if the chain block can't be read (caller idles)."""
        try:
            block = int(await self._sub.get_current_block())
        except Exception as e:
            log(f"Could not read current block: {e}", "warn")
            return None
        snap = load_allowlist(SAMPLING_ROTATION_INTERVAL)
        validators, interval = snap.validators, snap.interval
        rid = block // max(1, interval)
        primary = compute_sampler(validators, rid)
        produced = await self._produced(rid, primary)
        grace = grace_blocks_for(interval, _GRACE_OVERRIDE)
        sampler, step = effective_sampler(validators, rid, block - rid * interval, grace, produced)
        return RotationView(
            rotation_index=rid, interval=interval, validators=validators,
            sampler=sampler, failover_step=step,
            is_your_turn=(sampler == self._me), produced=produced,
        )
