"""
Score calculation background task for dashboard.

1) Per-validator scores: from validator_samples into rank_scores (dashboard).
2) Equal-weight scorer: every SCORER_INTERVAL, use a **consecutive** task_id window of
   length SCORER_TASK_WINDOW ending at max(task_id) in validator_samples, i.e.
   [max_id - N + 1, max_id]. Scoring does not run until max_id >= N (full window exists).
   Eligibility: completeness = (tasks evaluated in that window) / N must be >=
   COMPLETENESS_ELIGIBILITY_THRESHOLD (default 80%). Pass/fail per task uses equal-weight
   validator votes (one validator = one vote, matching the on-chain local aggregation);
   rank eligible miners by task_passed_count (win rate over evaluated tasks).
3) Rank (dominance): same cycle, among eligible miners compute top by dominance
   (block order + DOMINANCE_THRESHOLD), persist to miner_ranks for /weights and /rank.
"""

import os
import asyncio
from typing import Dict, Set, List, Tuple, Optional

from leoma.bootstrap import emit_log as log, emit_header as log_header, log_exception
from leoma.infra.aggregate import compute_miner_aggregates, rank_from_aggregates, Verdicts
from leoma.infra.scorer_constants import (
    COMPLETENESS_ELIGIBILITY_THRESHOLD,
    SCORER_TASK_WINDOW,
    required_distinct_validators,
)
from leoma.delivery.http.routes.rotation import (
    current_scoring_window,
    current_scoring_window_rows,
)
from leoma.infra.db.stores import (
    MinerRankStore,
    MinerTaskRankStore,
    ParticipantStore,
    RankStore,
    SampleStore,
    ValidatorStore,
)


# Configuration
SCORE_CALCULATION_INTERVAL = int(os.environ.get("SCORE_CALCULATION_INTERVAL", "300"))  # 5 min
SCORER_INTERVAL = int(os.environ.get("SCORER_INTERVAL", "1800"))  # 30 mins
# Dominance: late miner must beat each earlier miner's pass_rate by this threshold to be top
DOMINANCE_THRESHOLD = float(os.environ.get("DOMINANCE_THRESHOLD", "0.05"))


class ScoreCalculationTask:
    """Per-validator, per-miner dashboard scores from validator_samples (no cross-validator aggregation)."""

    def __init__(self):
        self.validator_samples_dao = SampleStore()
        self.rank_scores_dao = RankStore()
        self.validators_dao = ValidatorStore()
        self.valid_miners_dao = ParticipantStore()
        self.miner_task_rank_dao = MinerTaskRankStore()
        self.miner_rank_dao = MinerRankStore()
        self._running = False
        self._last_scorer_run = 0.0

    async def run(self) -> None:
        self._running = True
        log(f"Score calculation task starting (interval={SCORE_CALCULATION_INTERVAL}s, scorer={SCORER_INTERVAL}s)", "start")
        await asyncio.sleep(10)

        while self._running:
            try:
                await self._calculate_and_store_scores()
            except Exception as e:
                log(f"Score calculation error: {e}", "error")
                log_exception("Score calculation error", e)

            import time
            now = time.monotonic()
            if now - self._last_scorer_run >= SCORER_INTERVAL:
                try:
                    await self._run_scorer()
                    self._last_scorer_run = now
                except Exception as e:
                    log(f"Per-validator-average scorer error: {e}", "error")
                    log_exception("Per-validator-average scorer error", e)

            await asyncio.sleep(SCORE_CALCULATION_INTERVAL)
    
    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _build_scores_for_validator(
        stats: Dict[str, Dict[str, int | float]],
        valid_hotkeys: Set[str],
    ) -> Dict[str, Dict[str, int | float]]:
        """Build batch score payload for a validator, filtered by valid miners."""
        scores_to_save: Dict[str, Dict[str, int | float]] = {}
        for miner_hotkey, miner_stats in stats.items():
            if miner_hotkey not in valid_hotkeys:
                continue

            passed_count = miner_stats.get("passed_count", 0)
            total = miner_stats.get("total", 0)
            pass_rate = miner_stats.get("pass_rate", 0.0)
            scores_to_save[miner_hotkey] = {
                "score": pass_rate,
                "total_samples": total,
                "total_passed": passed_count,
                "pass_rate": pass_rate,
            }
        return scores_to_save

    @staticmethod
    def _find_leader(miner_totals: Dict[str, Dict[str, int]]) -> tuple[str | None, float]:
        """Find miner with highest aggregated pass rate."""
        leader: str | None = None
        leader_rate = 0.0
        for hotkey, totals in miner_totals.items():
            if totals["total"] <= 0:
                continue
            rate = totals["passed_count"] / totals["total"]
            if rate > leader_rate:
                leader = hotkey
                leader_rate = rate
        return leader, leader_rate
    
    async def _get_valid_miner_hotkeys(self) -> Set[str]:
        """Get set of valid miner hotkeys."""
        valid_miners = await self.valid_miners_dao.get_valid_miners()
        return {m.miner_hotkey for m in valid_miners}
    
    async def _cleanup_invalid_miner_scores(self, valid_hotkeys: Set[str]) -> int:
        """Remove scores for miners that are no longer valid; returns the count removed."""
        all_scores = await self.rank_scores_dao.get_all_scores()
        removed = 0
        
        for score in all_scores:
            if score.miner_hotkey not in valid_hotkeys:
                await self.rank_scores_dao.delete_scores_by_miner(score.miner_hotkey)
                removed += 1
        
        return removed
    
    async def _calculate_and_store_scores(self) -> None:
        """Per validator, compute each valid miner's pass rate from validator_samples and store one
        rank_scores row per validator-miner pair."""
        log_header("Dashboard Score Calculation")

        valid_hotkeys = await self._get_valid_miner_hotkeys()

        if not valid_hotkeys:
            log("No valid miners found", "info")
            return

        log(f"Found {len(valid_hotkeys)} valid miners", "info")

        removed = await self._cleanup_invalid_miner_scores(valid_hotkeys)
        if removed > 0:
            log(f"Removed {removed} scores for invalid miners", "info")

        validators = await self.validators_dao.get_all_validators()

        if not validators:
            log("No validators found", "info")
            return

        total_scores = 0

        for validator in validators:
            stats = await self.validator_samples_dao.get_miner_stats_by_validator(
                validator.hotkey
            )

            if not stats:
                continue

            scores_to_save = self._build_scores_for_validator(stats, valid_hotkeys)

            if not scores_to_save:
                continue

            count = await self.rank_scores_dao.batch_save_scores(
                validator_hotkey=validator.hotkey,
                scores=scores_to_save,
            )
            total_scores += count

        log(f"Updated {total_scores} dashboard scores from {len(validators)} validators", "success")

        # Log current leader (based on simple pass rate, not weighted)
        all_scores = await self.rank_scores_dao.get_all_scores()
        if all_scores:
            miner_hotkeys = {s.miner_hotkey for s in all_scores}
            window = await current_scoring_window()
            sampling = await self.validator_samples_dao.get_miner_sampling_stats_by_hotkeys(
                miner_hotkeys, task_ids=window
            )
            miner_totals = {
                hk: {"passed_count": s["passed_tasks"], "total": s["total_tasks"]}
                for hk, s in sampling.items()
                if s["total_tasks"] > 0
            }
            leader, leader_rate = self._find_leader(miner_totals)
            
            if leader:
                t = miner_totals[leader]
                log(
                    f"Dashboard leader: {leader[:8]}... "
                    f"({t['passed_count']}/{t['total']} passes, {leader_rate:.1%})",
                    "info"
                )

    async def _run_scorer(self) -> None:
        """Per-validator-average scorer over a consecutive task_id window of SCORER_TASK_WINDOW.

        Self-evaluation model: each task has one verdict (from its sampler). To weight validators
        equally, a miner's score is the MEAN of each validator's pass-rate over that validator's
        own tasks (identical to the on-chain ``aggregate_per_validator_average``). Window =
        [max_task_id - N + 1, max_task_id]; no rankings until max_task_id >= N. Writes
        miner_task_ranks (completeness/eligibility) and miner_ranks (dominance rank) for the
        dashboard's /scores/rank and /weights.
        """
        log_header(f"Per-validator-average Scorer ({SCORER_TASK_WINDOW}-task window)")
        # Production-based settled window from the ledger — same construction as the on-chain
        # aggregation (last N *produced* tasks), so skipped turns don't dilute the window.
        task_ids, active_validators = await current_scoring_window_rows()
        if not task_ids:
            log("Scorer skipped: no settled scoring window yet", "info")
            await self.miner_task_rank_dao.delete_miners_not_in(set())
            await self.miner_rank_dao.replace_all([])
            return
        # Winner-take-all guard: require a majority of active validators (None on the legacy
        # block-derived fallback, where the active set is unknown -> gate disabled).
        min_distinct = (
            required_distinct_validators(len(active_validators)) if active_validators else 1
        )

        samples = await self.validator_samples_dao.get_samples_in_task_window(task_ids)

        # (validator_hotkey, task_id, miner_hotkey) -> passed; one verdict per task (self-eval).
        verdicts: Verdicts = {}
        for s in samples:
            verdicts[(s.validator_hotkey, s.task_id, s.miner_hotkey)] = bool(s.passed)

        # Block from the full consensus table; normalize identically to the on-chain path
        # (treat 0 and None the same -> sorts last in dominance) so the dashboard matches on-chain.
        all_miner_objs = await self.valid_miners_dao.get_all_miners()
        block_by_hotkey: Dict[str, Optional[int]] = {
            m.miner_hotkey: (m.block or None) for m in all_miner_objs
        }

        aggregates = compute_miner_aggregates(verdicts, task_ids, block_by_hotkey)
        # Implicit-via-sampling eligibility (matches on-chain aggregate_local): no explicit valid
        # filter — only sampled miners have verdicts, and the completeness gate guards manipulation.
        # The min-distinct-validators gate guards winner-take-all in a skip-thinned window.
        winner_hotkey, rank_entries = rank_from_aggregates(
            aggregates, set(), DOMINANCE_THRESHOLD, min_distinct_validators=min_distinct
        )

        # miner_ranks: eligible miners ranked by mean per-validator rate (drives /weights, /rank).
        await self.miner_rank_dao.replace_all(rank_entries)

        # miner_task_ranks: completeness + totals for the eligible set (eligibility display).
        keep: Set[str] = set()
        for entry in rank_entries:
            hk = entry["miner_hotkey"]
            agg = aggregates.get(hk)
            if agg is None:
                continue
            await self.miner_task_rank_dao.upsert(
                miner_hotkey=hk,
                task_passed_count=agg.total_passed,
                tasks_evaluated=agg.total_evaluated,
                completeness=agg.completeness,
                rank=entry["rank"],
            )
            keep.add(hk)
        await self.miner_task_rank_dao.delete_miners_not_in(keep)

        log(
            f"Per-validator-average scorer: {len(rank_entries)} eligible miners ranked "
            f"(window {task_ids[0]}…{task_ids[-1]}, completeness>={COMPLETENESS_ELIGIBILITY_THRESHOLD}), "
            f"winner={winner_hotkey[:12] + '...' if winner_hotkey else 'None'}",
            "success",
        )
