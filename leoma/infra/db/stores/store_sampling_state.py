"""Sampling state store — now only the admin-settable rotation interval shown on the dashboard."""
from leoma.infra.db.pool import get_session
from leoma.infra.db.tables import SamplingState
from sqlalchemy import select


KEY_ROTATION_INTERVAL = "rotation_interval"


class SamplingStateStore:
    """Access layer for sampling_state (rotation interval key-value)."""

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
