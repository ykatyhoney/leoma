"""
Validator stake-refresh background task.

The validator allowlist is OWNER-MANAGED (added/removed via `leoma validator ...`), not derived
from stake. This task only *refreshes* the stake of already-registered validators from the
metagraph so the dashboard stays accurate — it never adds or removes validators. Runs on a
configurable interval (default 10 minutes).
"""

import asyncio

import bittensor as bt

from leoma.bootstrap import (
    emit_log as log,
    emit_header as log_header,
    log_exception,
    NETUID,
    NETWORK,
    VALIDATOR_SYNC_INTERVAL,
)
from leoma.infra.db.stores import ValidatorStore


def _get_stake(meta, uid: int) -> float:
    """Get stake for a UID from metagraph. Prefer .S then .stake."""
    stake_vec = getattr(meta, "S", None)
    if stake_vec is None:
        stake_vec = getattr(meta, "stake", None)
    if stake_vec is None:
        return 0.0
    try:
        if uid < len(stake_vec):
            return float(stake_vec[uid])
    except (TypeError, IndexError):
        pass
    return 0.0


class ValidatorSyncTask:
    """Background task that refreshes registered validators' stake from the metagraph.

    Membership is owner-managed (CLI); this task never adds or removes validators.
    """

    def __init__(self):
        self.validator_store = ValidatorStore()
        self._running = False

    async def run(self) -> None:
        self._running = True
        log(f"Validator stake-refresh task starting (interval={VALIDATOR_SYNC_INTERVAL}s)", "start")
        await asyncio.sleep(5)
        while self._running:
            try:
                await self._refresh_stakes()
            except Exception as e:
                log(f"Validator stake-refresh error: {e}", "error")
                log_exception("Validator stake-refresh error", e)
            await asyncio.sleep(VALIDATOR_SYNC_INTERVAL)

    def stop(self) -> None:
        self._running = False

    async def _refresh_stakes(self) -> None:
        """Update each registered validator's stake from the metagraph; never adds or removes."""
        registered = await self.validator_store.get_all_validators()
        if not registered:
            return
        log_header("Validator stake refresh (metagraph)")
        subtensor = bt.AsyncSubtensor(network=NETWORK)
        try:
            meta = await subtensor.metagraph(NETUID)
            hotkeys = getattr(meta, "hotkeys", []) or []
            uid_by_hotkey = {hk: uid for uid, hk in enumerate(hotkeys) if isinstance(hk, str)}
            updated = 0
            for v in registered:
                uid = uid_by_hotkey.get(v.hotkey)
                if uid is None:
                    continue  # registered validator not in the metagraph — leave its stake as-is
                await self.validator_store.update_stake(v.hotkey, _get_stake(meta, uid))
                updated += 1
            log(f"Refreshed stake for {updated}/{len(registered)} registered validators", "success")
        finally:
            await subtensor.close()
