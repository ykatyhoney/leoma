"""
Validator-side miner validation (decentralized).

Each validator independently reads the chain (commitments + metagraph), validates every miner
(commit rules, blacklist, HuggingFace model hash, Chutes status, duplicate-model detection), and:
  - keeps a local snapshot used by its own sampler (which miners to call) and weight-setter
    (uid/block lookup), and
  - dual-reports the result to the owner-api, which tallies a majority consensus for the dashboard.

This replaces the owner-api's MinerValidationTask: the owner no longer decides validity, so it
can't unilaterally include/exclude a miner.
"""
import os
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
import bittensor as bt

from leoma.bootstrap import NETUID, NETWORK, WALLET_NAME, HOTKEY_NAME
from leoma.bootstrap import emit_log as log, emit_header as log_header, log_exception
from leoma.domain import MinerInfo
from leoma.infra.commit_parser import parse_commit, validate_commit_fields, validate_commit_count
from leoma.infra.eligibility import validate_miner, detect_plagiarism

MINER_VALIDATION_INTERVAL = int(os.environ.get("MINER_VALIDATION_INTERVAL", "300"))  # 5 min
API_URL = os.environ.get("API_URL", "https://api.leoma.ai")


@dataclass
class ValidationSnapshot:
    """This validator's latest view of the miner set."""

    miners: List[MinerInfo]                       # all validated miners (is_valid set per-miner)
    uid_by_hotkey: Dict[str, int]                 # every metagraph hotkey -> uid
    block_by_hotkey: Dict[str, Optional[int]]     # committed miners -> commit block
    at: float = 0.0


# Module-level snapshot, written by the validation loop and read by the sampler + weight-setter
# (same process, single-threaded asyncio — no lock needed).
_snapshot: Optional[ValidationSnapshot] = None


def current_snapshot() -> Optional[ValidationSnapshot]:
    return _snapshot


def valid_miners() -> List[MinerInfo]:
    """Miners this validator currently considers valid (for sampling). Empty until first validation."""
    if _snapshot is None:
        return []
    return [m for m in _snapshot.miners if m.is_valid]


def _set_snapshot(snapshot: ValidationSnapshot) -> None:
    global _snapshot
    _snapshot = snapshot


async def validate_all_miners(
    subtensor: bt.AsyncSubtensor,
    blacklist: set,
) -> ValidationSnapshot:
    """Read the chain and validate every committed miner. Returns the snapshot (no DB writes)."""
    current_block = await subtensor.get_current_block()
    commits = await subtensor.get_all_revealed_commitments(NETUID, block=current_block)
    meta = await subtensor.metagraph(NETUID)
    hotkeys = list(getattr(meta, "hotkeys", []) or [])

    uid_by_hotkey: Dict[str, int] = {hk: uid for uid, hk in enumerate(hotkeys)}
    block_by_hotkey: Dict[str, Optional[int]] = {}
    miners: List[MinerInfo] = []

    async with aiohttp.ClientSession() as session:
        for uid, hotkey in enumerate(hotkeys):
            commit_data = (commits or {}).get(hotkey)
            if not commit_data:
                continue

            if hotkey in blacklist:
                miners.append(MinerInfo(uid=uid, hotkey=hotkey, is_valid=False, invalid_reason="blacklisted"))
                continue

            commit_block, commit_value = commit_data[-1]
            block_by_hotkey[hotkey] = commit_block

            ok, reason = validate_commit_count(len(commit_data))
            if not ok:
                miners.append(MinerInfo(uid=uid, hotkey=hotkey, block=commit_block, is_valid=False, invalid_reason=reason))
                continue

            parsed = parse_commit(commit_value)
            ok, reason = validate_commit_fields(parsed, hotkey=hotkey)
            if not ok:
                miners.append(MinerInfo(uid=uid, hotkey=hotkey, block=commit_block, is_valid=False, invalid_reason=reason))
                continue

            info = await validate_miner(
                session=session,
                uid=uid,
                hotkey=hotkey,
                model_name=parsed["model_name"],
                model_revision=parsed["model_revision"],
                chute_id=parsed["chute_id"],
                block=commit_block,
            )
            miners.append(info)

    # Re-enable duplicate-model detection: among valid miners sharing a model hash, the earliest
    # (block, uid) keeps validity; the rest are marked duplicate_model.
    miners = detect_plagiarism(miners)
    return ValidationSnapshot(
        miners=miners,
        uid_by_hotkey=uid_by_hotkey,
        block_by_hotkey=block_by_hotkey,
        at=time.time(),
    )


def _to_report_entry(m: MinerInfo) -> dict:
    return {
        "uid": m.uid,
        "miner_hotkey": m.hotkey,
        "model_name": m.model_name or None,
        "model_revision": m.model_revision or None,
        "model_hash": m.model_hash or None,
        "chute_id": m.chute_id or None,
        "chute_slug": m.chute_slug or None,
        "block": m.block or None,
        "is_valid": bool(m.is_valid),
        "invalid_reason": m.invalid_reason,
    }


async def run_validation_loop() -> None:
    """Periodically validate miners, update the local snapshot, and dual-report to the owner-api."""
    from leoma.infra.remote_api import create_api_client_from_wallet

    subtensor = bt.AsyncSubtensor(network=NETWORK)
    api_client = create_api_client_from_wallet(
        wallet_name=WALLET_NAME, hotkey_name=HOTKEY_NAME, api_url=API_URL
    )
    log_header("Validator Miner Validation Starting")
    log(f"Interval: {MINER_VALIDATION_INTERVAL}s  API: {API_URL}", "info")

    while True:
        try:
            try:
                blacklist = set(await api_client.get_blacklisted_miners())
            except Exception:
                blacklist = set()

            snapshot = await validate_all_miners(subtensor, blacklist)
            _set_snapshot(snapshot)

            valid_count = sum(1 for m in snapshot.miners if m.is_valid)
            log(
                f"Validated {len(snapshot.miners)} miners: {valid_count} valid, "
                f"{len(snapshot.miners) - valid_count} invalid",
                "success",
            )

            # Dual-report (best-effort) so the owner-api can compute the dashboard consensus.
            try:
                await api_client.report_miners([_to_report_entry(m) for m in snapshot.miners])
            except Exception as e:
                log(f"Miner report to owner-api failed: {e}", "warn")
        except Exception as e:
            log(f"Miner validation error: {e}", "error")
            log_exception("Miner validation error", e)

        await asyncio.sleep(MINER_VALIDATION_INTERVAL)
