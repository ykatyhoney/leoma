"""Miner task rank store."""
from typing import List, Optional, Set

from sqlalchemy import delete, select

from leoma.infra.db.pool import get_session
from leoma.infra.db.tables import MinerTaskRank


class MinerTaskRankStore:
    """Access layer for miner_task_ranks."""

    async def upsert(
        self,
        miner_hotkey: str,
        task_passed_count: int,
        tasks_evaluated: int,
        completeness: float,
        rank: Optional[int] = None,
    ) -> MinerTaskRank:
        async with get_session() as session:
            r = await session.execute(
                select(MinerTaskRank).where(MinerTaskRank.miner_hotkey == miner_hotkey)
            )
            row = r.scalar_one_or_none()
            if row:
                row.task_passed_count = task_passed_count
                row.tasks_evaluated = tasks_evaluated
                row.completeness = completeness
                row.rank = rank
                await session.flush()
                return row
            row = MinerTaskRank(
                miner_hotkey=miner_hotkey,
                task_passed_count=task_passed_count,
                tasks_evaluated=tasks_evaluated,
                completeness=completeness,
                rank=rank,
            )
            session.add(row)
            await session.flush()
            return row

    async def get_all_ranked(self) -> List[MinerTaskRank]:
        async with get_session() as session:
            q = (
                select(MinerTaskRank)
                .where(MinerTaskRank.rank.isnot(None))
                .order_by(MinerTaskRank.rank.asc())
            )
            r = await session.execute(q)
            return list(r.scalars().all())

    async def get_by_miner(self, miner_hotkey: str) -> Optional[MinerTaskRank]:
        async with get_session() as session:
            r = await session.execute(
                select(MinerTaskRank).where(MinerTaskRank.miner_hotkey == miner_hotkey)
            )
            return r.scalar_one_or_none()

    async def delete_miners_not_in(self, keep_hotkeys: Set[str]) -> int:
        """Remove rows for miners no longer meeting completeness (stale eligibility fix).

        After each scorer run, only ``keep_hotkeys`` should remain; everyone else is dropped
        so dashboard eligibility and dominance do not use outdated completeness.
        """
        async with get_session() as session:
            if not keep_hotkeys:
                result = await session.execute(delete(MinerTaskRank))
            else:
                result = await session.execute(
                    delete(MinerTaskRank).where(~MinerTaskRank.miner_hotkey.in_(keep_hotkeys))
                )
            return result.rowcount or 0
