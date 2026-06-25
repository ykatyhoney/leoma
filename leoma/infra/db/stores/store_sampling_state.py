"""Sampling state store (decentralized coordination key-value store)."""
import json
from typing import Dict, Tuple

from leoma.infra.db.pool import get_session
from leoma.infra.db.tables import SamplingState
from sqlalchemy import select


KEY_LATEST_TASK_ID = "latest_task_id"
KEY_LATEST_TASK_SAMPLER = "latest_task_sampler"
KEY_ROTATION_INTERVAL = "rotation_interval"
KEY_CLAIM_LEASES = "claim_leases"


class SamplingStateStore:
    """Access layer for sampling_state.

    No central task_id counter: task_id is the block-derived rotation index. The sampling
    validator announces the latest task_id and its hotkey here so peers can discover it.
    """

    async def get_value(self, key: str) -> str | None:
        async with get_session() as session:
            r = await session.execute(select(SamplingState).where(SamplingState.key == key))
            row = r.scalar_one_or_none()
            return row.value if row else None

    async def set_value(self, key: str, value: str) -> None:
        async with get_session() as session:
            r = await session.execute(select(SamplingState).where(SamplingState.key == key))
            row = r.scalar_one_or_none()
            if row:
                row.value = value
            else:
                session.add(SamplingState(key=key, value=value))

    async def get_latest_task_id(self) -> int | None:
        raw = await self.get_value(KEY_LATEST_TASK_ID)
        if raw is None:
            return None
        try:
            return int(raw)
        except (ValueError, TypeError):
            return None

    async def get_latest_task_sampler(self) -> str | None:
        return await self.get_value(KEY_LATEST_TASK_SAMPLER)

    async def announce_task(self, task_id: int, sampler_hotkey: str) -> bool:
        """Record the latest decentralized task and which validator sampled it.

        Ignores out-of-order announcements (task_id older than the current latest) so a
        late upload can't regress the published task. Returns True if applied.
        """
        current = await self.get_latest_task_id()
        if current is not None and task_id < current:
            return False
        async with get_session() as session:
            for key, value in (
                (KEY_LATEST_TASK_ID, str(task_id)),
                (KEY_LATEST_TASK_SAMPLER, sampler_hotkey),
            ):
                r = await session.execute(select(SamplingState).where(SamplingState.key == key))
                row = r.scalar_one_or_none()
                if row:
                    row.value = value
                else:
                    session.add(SamplingState(key=key, value=value))
        return True

    async def load_claim_map(self) -> Dict[int, Tuple[str, float]]:
        """Load the sampling-turn lease map ``{rotation_id: (holder_hotkey, expires_epoch)}``.

        Persisted (not in-memory) so leases survive an owner-api restart — a restart mid-window
        can't drop the lease and let a returning primary and a backup both sample the same window.
        """
        raw = await self.get_value(KEY_CLAIM_LEASES)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return {int(k): (v[0], float(v[1])) for k, v in data.items()}
        except (ValueError, TypeError, KeyError, IndexError):
            return {}

    async def save_claim_map(self, claims: Dict[int, Tuple[str, float]]) -> None:
        serial = {str(k): [holder, exp] for k, (holder, exp) in claims.items()}
        await self.set_value(KEY_CLAIM_LEASES, json.dumps(serial))

    async def get_rotation_interval(self, default: int) -> int:
        raw = await self.get_value(KEY_ROTATION_INTERVAL)
        if raw is None:
            return default
        try:
            value = int(raw)
            return value if value > 0 else default
        except (ValueError, TypeError):
            return default

    async def set_rotation_interval(self, interval: int) -> None:
        await self.set_value(KEY_ROTATION_INTERVAL, str(int(interval)))
