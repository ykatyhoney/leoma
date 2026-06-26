"""Local rotation resolution (owner-api-free).

Whose turn it is to sample is a deterministic function of the current chain block and the on-chain
allowlist, so each validator computes it itself — there is no ``GET /rotation``. Failover is purely
block-derived, and "did the scheduled sampler already produce this rotation?" is read from its
bucket, so there is no claim lease either. Duplicate production during a failover race is harmless:
the window's canonical-sampler resolution deterministically keeps the earliest-failover-order producer.
"""
import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional

from leoma.bootstrap import NETUID, SOURCE_BUCKET, emit_log as log
from leoma.infra.onchain_allowlist import AllowlistSnapshot, read_allowlist
from leoma.infra.peer_registry import load_peers
from leoma.infra.rotation_math import compute_sampler, effective_sampler, grace_blocks_for
from leoma.infra.storage_backend import create_peer_read_client, create_source_read_client

_GRACE_OVERRIDE = int(os.environ.get("SAMPLER_FAILOVER_GRACE_BLOCKS", "0"))
# Re-read the on-chain allowlist at most this often (blocks); cheap to cache between.
_ALLOWLIST_REFRESH_BLOCKS = int(os.environ.get("ALLOWLIST_REFRESH_BLOCKS", "300"))


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
    """Resolves the current sampling turn from chain block + on-chain allowlist (no owner-api)."""

    def __init__(self, subtensor, my_hotkey: str, source_read_client=None):
        self._sub = subtensor
        self._me = my_hotkey
        self._source = source_read_client
        self._snap: Optional[AllowlistSnapshot] = None
        self._snap_block = -10**9

    async def _allowlist(self, block: int) -> Optional[AllowlistSnapshot]:
        if self._snap is None or block - self._snap_block >= _ALLOWLIST_REFRESH_BLOCKS:
            snap = await read_allowlist(
                self._sub, NETUID, self._source or create_source_read_client(), SOURCE_BUCKET
            )
            if snap is not None:
                self._snap, self._snap_block = snap, block
        return self._snap

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
        """Current rotation view, or ``None`` if the chain/allowlist can't be read (caller idles)."""
        try:
            block = int(await self._sub.get_current_block())
        except Exception as e:
            log(f"Could not read current block: {e}", "warn")
            return None
        snap = await self._allowlist(block)
        if snap is None:
            return None
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
