"""
Score calculation background task for dashboard.

1) Per-validator scores: from validator_samples into rank_scores (dashboard).
2) Stake-weighted scorer: every SCORER_INTERVAL, use a **consecutive** task_id window of
   length SCORER_TASK_WINDOW ending at max(task_id) in validator_samples, i.e.
   [max_id - N + 1, max_id]. Scoring does not run until max_id >= N (full window exists).
   Eligibility: completeness = (tasks evaluated in that window) / N must be >=
   COMPLETENESS_ELIGIBILITY_THRESHOLD (default 80%). Pass/fail per task uses stake-weighted
   validator votes; rank eligible miners by task_passed_count (win rate over evaluated tasks).
3) Rank (dominance): same cycle, among eligible miners compute top by dominance
   (block order + DOMINANCE_THRESHOLD), persist to miner_ranks for /weights and /rank.
"""

import os
import asyncio
from typing import Dict, Set, List, Tuple, Optional

from leoma.bootstrap import emit_log as log, emit_header as log_header, log_exception
from leoma.infra.rank import find_dominant_winner, compute_rank_from_miner_stats
from leoma.infra.scorer_constants import (
    COMPLETENESS_ELIGIBILITY_THRESHOLD,
    SCORER_TASK_WINDOW,
    scoring_window_task_ids,
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
    """Background task for calculating dashboard scores from submitted samples.
    
    Calculates per-validator, per-miner scores from validator_samples.
    No cross-validator aggregation - each validator's scores stored separately.
    """
    
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
        """Run the score calculation task loop."""
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
                    await self._run_stake_weighted_scorer()
                    await self._run_rank_update()
                    self._last_scorer_run = now
                except Exception as e:
                    log(f"Stake-weighted scorer / rank error: {e}", "error")
                    log_exception("Stake-weighted scorer / rank error", e)

            await asyncio.sleep(SCORE_CALCULATION_INTERVAL)
    
    def stop(self) -> None:
        """Stop the task."""
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
        """Remove scores for miners that are no longer valid.
        
        Args:
            valid_hotkeys: Set of currently valid miner hotkeys
            
        Returns:
            Number of scores removed
        """
        all_scores = await self.rank_scores_dao.get_all_scores()
        removed = 0
        
        for score in all_scores:
            if score.miner_hotkey not in valid_hotkeys:
                await self.rank_scores_dao.delete_scores_by_miner(score.miner_hotkey)
                removed += 1
        
        return removed
    
    async def _calculate_and_store_scores(self) -> None:
        """Calculate scores from samples and store in rank_scores for dashboard.
        
        For each validator:
        - Get their submitted samples from validator_samples
        - Calculate pass rate per miner
        - Store in rank_scores (one row per validator-miner pair)
        
        Only includes scores for valid miners.
        """
        log_header("Dashboard Score Calculation")
        
        # Get valid miner hotkeys
        valid_hotkeys = await self._get_valid_miner_hotkeys()
        
        if not valid_hotkeys:
            log("No valid miners found", "info")
            return
        
        log(f"Found {len(valid_hotkeys)} valid miners", "info")
        
        # Clean up scores for invalid miners
        removed = await self._cleanup_invalid_miner_scores(valid_hotkeys)
        if removed > 0:
            log(f"Removed {removed} scores for invalid miners", "info")
        
        # Get all validators
        validators = await self.validators_dao.get_all_validators()
        
        if not validators:
            log("No validators found", "info")
            return
        
        total_scores = 0
        
        for validator in validators:
            # Get miner stats for this validator from their submitted samples
            stats = await self.validator_samples_dao.get_miner_stats_by_validator(
                validator.hotkey
            )
            
            if not stats:
                continue
            
            scores_to_save = self._build_scores_for_validator(stats, valid_hotkeys)
            
            if not scores_to_save:
                continue
            
            # Save this validator's scores (one row per miner)
            count = await self.rank_scores_dao.batch_save_scores(
                validator_hotkey=validator.hotkey,
                scores=scores_to_save,
            )
            total_scores += count
        
        log(f"Updated {total_scores} dashboard scores from {len(validators)} validators", "success")
        
        # Log current leader (based on simple pass rate, not weighted)
        all_scores = await self.rank_scores_dao.get_all_scores()
        if all_scores:
            validators = await self.validators_dao.get_all_validators()
            stake_map = {v.hotkey: max(0.0, float(v.stake)) for v in validators}
            miner_hotkeys = {s.miner_hotkey for s in all_scores}
            max_tid = await self.validator_samples_dao.get_max_evaluated_task_id()
            window = scoring_window_task_ids(max_tid)
            sampling = await self.validator_samples_dao.get_miner_sampling_stats_by_hotkeys(
                stake_map, miner_hotkeys, task_ids=window
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

    async def _run_stake_weighted_scorer(self) -> None:
        """Stake-weighted scorer: consecutive task_id window of length SCORER_TASK_WINDOW.

        Window = [max_task_id - N + 1, max_task_id]. No rankings until max_task_id >= N.
        Eligible miners must have evaluations on every task_id in that window.
        """
        log_header(f"Stake-weighted Scorer ({SCORER_TASK_WINDOW}-task consecutive window)")
        max_tid = await self.validator_samples_dao.get_max_evaluated_task_id()
        if max_tid is None or max_tid < SCORER_TASK_WINDOW:
            log(
                f"Scorer skipped: max_task_id={max_tid}, need max_task_id>={SCORER_TASK_WINDOW} "
                f"to form window [max-{SCORER_TASK_WINDOW - 1} … max]",
                "info",
            )
            await self.miner_task_rank_dao.delete_miners_not_in(set())
            return

        task_ids = list(range(max_tid - SCORER_TASK_WINDOW + 1, max_tid + 1))
        window_set = frozenset(task_ids)
        window_size = len(task_ids)
        samples = await self.validator_samples_dao.get_samples_in_task_window(task_ids)
        validators = await self.validators_dao.get_all_validators()
        stake_by_validator: Dict[str, float] = {v.hotkey: max(0.0, float(v.stake)) for v in validators}

        # (task_id, miner_hotkey) -> list of (validator_hotkey, passed_flag 0/1)
        votes: Dict[Tuple[int, str], List[Tuple[str, int]]] = {}
        for s in samples:
            key = (s.task_id, s.miner_hotkey)
            if key not in votes:
                votes[key] = []
            passed_flag = 1 if s.passed else 0
            votes[key].append((s.validator_hotkey, passed_flag))

        # Per (task_id, miner): avg_score = sum(stake * passed_flag) / total_stake; task passes when avg_score > 0.5
        task_passes: Dict[str, Set[int]] = {}  # miner_hotkey -> set of task_ids that are pass
        task_evaluated: Dict[str, Set[int]] = {}  # miner_hotkey -> set of task_ids evaluated
        for (task_id, miner_hotkey), vlist in votes.items():
            if miner_hotkey not in task_evaluated:
                task_evaluated[miner_hotkey] = set()
                task_passes[miner_hotkey] = set()
            task_evaluated[miner_hotkey].add(task_id)
            total_stake = sum(stake_by_validator.get(vh, 0.0) for vh, _ in vlist)
            if total_stake <= 0:
                continue
            weighted_sum = sum(stake_by_validator.get(vh, 0.0) * w for vh, w in vlist)
            avg_score = weighted_sum / total_stake
            if avg_score > 0.5:
                task_passes[miner_hotkey].add(task_id)

        # Eligible if evaluated task count in window / window_size >= completeness threshold (default 80%).
        valid_miners = await self._get_valid_miner_hotkeys()
        eligible: List[Tuple[str, int, int, float]] = []
        for miner_hotkey, ev_set in task_evaluated.items():
            if miner_hotkey not in valid_miners:
                continue
            ev_in_window = ev_set & window_set
            evaluated = len(ev_in_window)
            completeness = evaluated / window_size if window_size else 0.0
            if completeness < COMPLETENESS_ELIGIBILITY_THRESHOLD - 1e-12:
                continue
            # Passes only among tasks that both were evaluated and passed stake vote
            passed_count = len(task_passes.get(miner_hotkey, set()) & ev_in_window)
            eligible.append((miner_hotkey, passed_count, evaluated, completeness))
        eligible.sort(key=lambda x: (-x[1], -x[2]))

        for r, (miner_hotkey, passed_count, evaluated, completeness) in enumerate(eligible, start=1):
            await self.miner_task_rank_dao.upsert(
                miner_hotkey=miner_hotkey,
                task_passed_count=passed_count,
                tasks_evaluated=evaluated,
                completeness=completeness,
                rank=r,
            )
        keep = {hk for hk, _, _, _ in eligible}
        await self.miner_task_rank_dao.delete_miners_not_in(keep)
        log(
            f"Stake-weighted scorer: {len(eligible)} miners ranked "
            f"(window task_ids {task_ids[0]}…{task_ids[-1]}, size={window_size}, "
            f"completeness>={COMPLETENESS_ELIGIBILITY_THRESHOLD})",
            "success",
        )
        if eligible:
            top = eligible[0]
            log(f"Top miner: {top[0][:12]}... ({top[1]} task passes, rank 1)", "info")

    async def _run_rank_update(self) -> None:
        """Compute rank by dominance (block + 5% threshold), persist to miner_ranks.
        Rank 1 = top-ranked miner; rest by passed_count desc. Used by GET /weights and /rank.
        
        Only includes miners present in miner_task_ranks (met completeness threshold).
        """
        log_header("Rank update (dominance)")
        
        # Get miners from miner_task_ranks (already has completeness check)
        task_ranked_miners = await self.miner_task_rank_dao.get_all_ranked()
        if not task_ranked_miners:
            log("No eligible miners in miner_task_ranks", "info")
            # Clear miner_ranks if no eligible miners
            await self.miner_rank_dao.replace_all([])
            return
        
        # Get block info from valid_miners
        valid_miners = await self.valid_miners_dao.get_valid_miners()
        block_by_hotkey: Dict[str, Optional[int]] = {m.miner_hotkey: m.block for m in valid_miners}
        
        miner_stats: List[Tuple[str, int, float, Optional[int]]] = []
        for m in task_ranked_miners:
            hotkey = m.miner_hotkey
            passed_count = m.task_passed_count or 0
            pass_rate = passed_count / m.tasks_evaluated if m.tasks_evaluated else 0.0
            miner_stats.append((hotkey, passed_count, float(pass_rate), block_by_hotkey.get(hotkey)))
        
        winner_hotkey, rank_entries = compute_rank_from_miner_stats(miner_stats, DOMINANCE_THRESHOLD)
        await self.miner_rank_dao.replace_all(rank_entries)
        log(f"Rank updated: {len(rank_entries)} eligible miners, top_hotkey={winner_hotkey[:12] if winner_hotkey else 'None'}...", "success")
