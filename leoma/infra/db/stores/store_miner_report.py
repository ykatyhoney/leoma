"""Validator miner-report store (decentralized validation → consensus)."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select

from leoma.infra.db.pool import get_session
from leoma.infra.db.tables import ValidatorMinerReport


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class MinerReportStore:
    """Access layer for validator_miner_reports."""

    async def replace_validator_report(
        self, validator_hotkey: str, entries: List[Dict[str, Any]]
    ) -> int:
        """Replace this validator's entire report (delete prior rows, insert the new set)."""
        async with get_session() as session:
            await session.execute(
                delete(ValidatorMinerReport).where(
                    ValidatorMinerReport.validator_hotkey == validator_hotkey
                )
            )
            now = _now_utc()
            for e in entries:
                session.add(
                    ValidatorMinerReport(
                        validator_hotkey=validator_hotkey,
                        miner_hotkey=e["miner_hotkey"],
                        uid=int(e["uid"]),
                        block=e.get("block"),
                        model_name=e.get("model_name"),
                        model_revision=e.get("model_revision"),
                        model_hash=e.get("model_hash"),
                        chute_id=e.get("chute_id"),
                        chute_slug=e.get("chute_slug"),
                        is_valid=bool(e.get("is_valid", False)),
                        invalid_reason=e.get("invalid_reason"),
                        reported_at=now,
                    )
                )
            return len(entries)

    async def get_all_reports(
        self, since: Optional[datetime] = None
    ) -> List[ValidatorMinerReport]:
        """All reports, optionally only those reported at/after ``since`` (drop stale validators)."""
        async with get_session() as session:
            q = select(ValidatorMinerReport)
            if since is not None:
                q = q.where(ValidatorMinerReport.reported_at >= since)
            r = await session.execute(q)
            return list(r.scalars().all())

    async def delete_validator_report(self, validator_hotkey: str) -> int:
        async with get_session() as session:
            r = await session.execute(
                delete(ValidatorMinerReport).where(
                    ValidatorMinerReport.validator_hotkey == validator_hotkey
                )
            )
            return r.rowcount
