"""
Miner consensus background task (replaces the owner-api's MinerValidationTask).

The owner-api no longer validates miners — validators do. This task simply tallies the validators'
reports (validator_miner_reports) into a MAJORITY consensus and writes the existing ``valid_miners``
table, so the dashboard + /miners endpoints keep working unchanged. It does NO chain reads, HF, or
Chutes calls.
"""
import os
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from leoma.bootstrap import emit_log as log, emit_header as log_header, log_exception
from leoma.infra.db.stores import MinerReportStore, ParticipantStore
from leoma.delivery.http.routes.health import update_last_sync

MINER_CONSENSUS_INTERVAL = int(os.environ.get("MINER_VALIDATION_INTERVAL", "300"))
# Drop reports older than this many intervals (a validator that stopped reporting stops counting).
STALE_REPORT_INTERVALS = int(os.environ.get("MINER_REPORT_STALE_INTERVALS", "3"))


class MinerConsensusTask:
    """Tally validator miner-reports into a majority consensus → valid_miners."""

    def __init__(self):
        self.report_dao = MinerReportStore()
        self.valid_miners_dao = ParticipantStore()
        self._running = False

    async def run(self) -> None:
        self._running = True
        log(f"Miner consensus task starting (interval={MINER_CONSENSUS_INTERVAL}s)", "start")
        await asyncio.sleep(5)
        while self._running:
            try:
                await self._compute_consensus()
            except Exception as e:
                log(f"Miner consensus error: {e}", "error")
                log_exception("Miner consensus error", e)
            await asyncio.sleep(MINER_CONSENSUS_INTERVAL)

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _consensus_for_miner(rows: List[Any]) -> Dict[str, Any]:
        """Majority is_valid; representative metadata from the most-recent valid (else recent) report."""
        total = len(rows)
        valid_votes = sum(1 for r in rows if r.is_valid)
        is_valid = valid_votes * 2 > total  # strict majority of reporting validators

        rep = sorted(rows, key=lambda r: (r.is_valid, r.reported_at), reverse=True)[0]
        invalid_reason = None
        if not is_valid:
            reasons = [r.invalid_reason for r in rows if not r.is_valid and r.invalid_reason]
            invalid_reason = Counter(reasons).most_common(1)[0][0] if reasons else "consensus_invalid"

        return {
            "uid": rep.uid,
            "miner_hotkey": rep.miner_hotkey,
            "block": rep.block,
            "model_name": rep.model_name,
            "model_revision": rep.model_revision,
            "model_hash": rep.model_hash,
            "chute_id": rep.chute_id,
            "chute_slug": rep.chute_slug,
            "is_valid": is_valid,
            "invalid_reason": invalid_reason,
        }

    async def _compute_consensus(self) -> None:
        log_header("Miner Consensus (validator reports)")
        since = datetime.now(timezone.utc) - timedelta(
            seconds=STALE_REPORT_INTERVALS * MINER_CONSENSUS_INTERVAL
        )
        reports = await self.report_dao.get_all_reports(since=since)
        if not reports:
            log("No fresh validator reports; leaving valid_miners unchanged", "info")
            return

        by_miner: Dict[str, List[Any]] = {}
        for r in reports:
            by_miner.setdefault(r.miner_hotkey, []).append(r)

        consensus = [self._consensus_for_miner(rows) for rows in by_miner.values()]
        await self.valid_miners_dao.batch_upsert_miners(consensus)
        await self.valid_miners_dao.delete_stale_miners([c["uid"] for c in consensus])
        update_last_sync(datetime.now(timezone.utc))

        valid_count = sum(1 for c in consensus if c["is_valid"])
        reporting = len({r.validator_hotkey for r in reports})
        log(
            f"Consensus over {reporting} validators: {len(consensus)} miners "
            f"({valid_count} valid, {len(consensus) - valid_count} invalid)",
            "success",
        )
