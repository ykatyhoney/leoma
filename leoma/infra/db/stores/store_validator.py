"""Validators store."""
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import delete, func, select, update

from leoma.infra.db.pool import get_session
from leoma.infra.db.tables import Validator


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ValidatorStore:
    """Access layer for validators."""

    @staticmethod
    def _by_hotkey(hotkey: str):
        return select(Validator).where(Validator.hotkey == hotkey)

    async def save_validator(
        self,
        uid: int,
        hotkey: str,
        stake: float = 0.0,
        s3_bucket: Optional[str] = None,
    ) -> Validator:
        async with get_session() as session:
            existing = await session.get(Validator, uid)
            if existing:
                existing.hotkey = hotkey
                existing.stake = stake
                if s3_bucket:
                    existing.s3_bucket = s3_bucket
                existing.last_seen_at = _now_utc()
                validator = existing
            else:
                validator = Validator(
                    uid=uid,
                    hotkey=hotkey,
                    stake=stake,
                    s3_bucket=s3_bucket,
                    last_seen_at=_now_utc(),
                )
                session.add(validator)
            await session.flush()
            return validator

    async def get_validator_by_uid(self, uid: int) -> Optional[Validator]:
        async with get_session() as session:
            return await session.get(Validator, uid)

    async def get_validator_by_hotkey(self, hotkey: str) -> Optional[Validator]:
        async with get_session() as session:
            r = await session.execute(self._by_hotkey(hotkey))
            return r.scalar_one_or_none()

    async def get_all_validators(self) -> List[Validator]:
        async with get_session() as session:
            r = await session.execute(select(Validator).order_by(Validator.uid))
            return list(r.scalars().all())

    async def update_last_seen(self, hotkey: str) -> bool:
        async with get_session() as session:
            stmt = (
                update(Validator)
                .where(Validator.hotkey == hotkey)
                .values(last_seen_at=_now_utc())
            )
            r = await session.execute(stmt)
            return r.rowcount > 0

    async def update_stake(self, hotkey: str, stake: float) -> bool:
        async with get_session() as session:
            stmt = update(Validator).where(Validator.hotkey == hotkey).values(stake=stake)
            r = await session.execute(stmt)
            return r.rowcount > 0

    async def get_validator_count(self) -> int:
        async with get_session() as session:
            r = await session.execute(select(func.count(Validator.uid)))
            return r.scalar_one()

    async def get_validators_by_stake(self, min_stake: float = 0.0) -> List[Validator]:
        async with get_session() as session:
            q = (
                select(Validator)
                .where(Validator.stake >= min_stake)
                .order_by(Validator.stake.desc())
            )
            r = await session.execute(q)
            return list(r.scalars().all())

    async def delete_validators_except_uids(self, uids: set[int]) -> int:
        """Delete validators whose uid is not in the given set. Returns deleted count."""
        if not uids:
            return 0
        async with get_session() as session:
            stmt = delete(Validator).where(Validator.uid.not_in(uids))
            r = await session.execute(stmt)
            return r.rowcount or 0

    async def delete_validator_by_hotkey(self, hotkey: str) -> bool:
        """Remove a validator from the owner-managed allowlist by hotkey. Returns True if removed."""
        async with get_session() as session:
            r = await session.execute(delete(Validator).where(Validator.hotkey == hotkey))
            return (r.rowcount or 0) > 0
