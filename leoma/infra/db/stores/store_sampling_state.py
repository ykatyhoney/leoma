"""Sampling state store."""
from leoma.infra.db.pool import get_session
from leoma.infra.db.tables import SamplingState
from sqlalchemy import select


KEY_LATEST_TASK_ID = "latest_task_id"
KEY_NEXT_TASK_ID = "next_task_id"


class SamplingStateStore:
    """Access layer for sampling_state."""

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

    async def set_latest_task_id(self, task_id: int) -> None:
        """Commit successful round: publish latest_task_id and advance next_task_id for observers."""
        async with get_session() as session:
            for key, value in (
                (KEY_LATEST_TASK_ID, str(task_id)),
                (KEY_NEXT_TASK_ID, str(task_id + 1)),
            ):
                r = await session.execute(select(SamplingState).where(SamplingState.key == key))
                row = r.scalar_one_or_none()
                if row:
                    row.value = value
                else:
                    session.add(SamplingState(key=key, value=value))

    async def ensure_next_task_id_synced(self) -> None:
        latest = await self.get_latest_task_id()
        if latest is None:
            return
        async with get_session() as session:
            r = await session.execute(
                select(SamplingState).where(SamplingState.key == KEY_NEXT_TASK_ID)
            )
            row = r.scalar_one_or_none()
            if row is None:
                session.add(SamplingState(key=KEY_NEXT_TASK_ID, value=str(latest + 1)))

    async def peek_next_task_id(self) -> int:
        """Next task id to attempt for this round (read-only).

        Does not advance the counter. On a failed round, the same id is returned again;
        the counter only moves when :meth:`set_latest_task_id` runs after a successful upload.
        """
        latest = await self.get_latest_task_id()
        return (latest or 0) + 1
