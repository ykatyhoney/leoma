"""Produced-task ledger store (authoritative, gap-free scoring sequence).

The owner-api assigns a monotonic ``task_seq`` on the *first* announce for a ``rotation_id`` so the
scoring window (last N produced tasks) is robust to skipped rotation turns. See ``ProducedTask``.
"""
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from leoma.infra.db.pool import get_session
from leoma.infra.db.tables import ProducedTask
from leoma.infra.scorer_constants import (
    SCORER_MAX_LOOKBACK_BLOCKS,
    SCORER_SETTLE_MARGIN,
    SCORER_TASK_WINDOW,
)


@dataclass
class ProducedTaskRow:
    """A detached row of the produced-task ledger (safe to use after the session closes)."""

    task_seq: int
    rotation_id: int
    sampler_hotkey: str
    block: int


class ProducedTaskStore:
    """Access layer for the ``produced_tasks`` ledger."""

    async def _seq_for_rotation(self, rotation_id: int) -> Optional[int]:
        async with get_session() as session:
            r = await session.execute(
                select(ProducedTask.task_seq).where(ProducedTask.rotation_id == rotation_id)
            )
            return r.scalar_one_or_none()

    async def append(self, rotation_id: int, sampler_hotkey: str, block: int) -> dict:
        """Append a produced task, idempotent per ``rotation_id`` (first announce wins).

        A duplicate (e.g. a late failover announce for an already-recorded rotation_id, or a
        concurrent announce racing on the ``rotation_id`` unique index) is a no-op and returns the
        existing ``task_seq``. Returns ``{"task_seq", "applied"}``.
        """
        existing = await self._seq_for_rotation(rotation_id)
        if existing is not None:
            return {"task_seq": existing, "applied": False}
        try:
            async with get_session() as session:
                obj = ProducedTask(
                    rotation_id=rotation_id, sampler_hotkey=sampler_hotkey, block=block
                )
                session.add(obj)
                await session.flush()  # populate the autoincrement task_seq
                seq = obj.task_seq
            return {"task_seq": seq, "applied": True}
        except IntegrityError:
            # A concurrent announce won the unique(rotation_id) race; return the winner's seq.
            existing = await self._seq_for_rotation(rotation_id)
            return {"task_seq": existing, "applied": False}

    async def has_rotation(self, rotation_id: int) -> bool:
        return (await self._seq_for_rotation(rotation_id)) is not None

    async def window(
        self,
        as_of_block: int,
        n: int = SCORER_TASK_WINDOW,
        margin: int = SCORER_SETTLE_MARGIN,
        max_lookback: int = SCORER_MAX_LOOKBACK_BLOCKS,
    ) -> List[ProducedTaskRow]:
        """The last ``n`` produced tasks at ``as_of_block``, dropping the newest ``margin``.

        Anchored to ``as_of_block`` (``block <= as_of_block``) so validators sharing the epoch block
        compute the identical window over immutable rows. The newest ``margin`` produced rows are
        excluded (settle margin: a sampler may still be writing the latest verdicts). Optionally
        floored to ``as_of_block - max_lookback`` so the window can't reach into ancient history.
        Returned ascending by ``task_seq``.
        """
        async with get_session() as session:
            stmt = select(ProducedTask).where(ProducedTask.block <= as_of_block)
            if max_lookback and max_lookback > 0:
                stmt = stmt.where(ProducedTask.block >= as_of_block - max_lookback)
            stmt = (
                stmt.order_by(ProducedTask.task_seq.desc())
                .offset(max(0, margin))
                .limit(max(0, n))
            )
            rows = (await session.execute(stmt)).scalars().all()
        rows = list(reversed(rows))  # ascending by task_seq
        return [
            ProducedTaskRow(
                task_seq=r.task_seq,
                rotation_id=r.rotation_id,
                sampler_hotkey=r.sampler_hotkey,
                block=r.block,
            )
            for r in rows
        ]

    async def count(self) -> int:
        async with get_session() as session:
            r = await session.execute(select(func.count()).select_from(ProducedTask))
            return int(r.scalar_one() or 0)

    async def backfill(self, entries: List[dict]) -> int:
        """One-time seed of the ledger from historical data (idempotent per rotation_id).

        ``entries`` is an ordered list of ``{rotation_id, sampler_hotkey, block}`` (caller orders by
        rotation_id so ``task_seq`` follows chronology). Skips rotation_ids already present. Returns
        the number of rows inserted.
        """
        inserted = 0
        for e in entries:
            res = await self.append(
                rotation_id=int(e["rotation_id"]),
                sampler_hotkey=e["sampler_hotkey"],
                block=int(e["block"]),
            )
            if res["applied"]:
                inserted += 1
        return inserted
