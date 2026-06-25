"""
Weights endpoint for validators.

Top-ranked-only weighting: returns winner_uid and per-miner list (hotkey, uid, pass_rate, weight).
Validators set weight 1.0 for winner_uid, 0 for others. If no top-ranked miner, winner_uid=0 (burn alpha).
"""

from typing import List

from fastapi import APIRouter

from leoma.delivery.http.contracts import WeightsResponse, MinerWeightEntry
from leoma.infra.db.stores import MinerRankStore, ParticipantStore


router = APIRouter()
miner_rank_dao = MinerRankStore()
valid_miners_dao = ParticipantStore()


@router.get("", response_model=WeightsResponse)
async def get_weights() -> WeightsResponse:
    """Return top-ranked UID (`winner_uid`) and each miner's hotkey, uid, pass_rate, and weight (1.0 or 0).
    
    Validators use this to set on-chain weights: only winner_uid gets 1.0, others 0.
    miners list gives each miner's score (pass_rate) and assigned weight for transparency.
    If no top-ranked miner, winner_uid=0 so validators set weight to UID 0 (burn alpha).
    """
    from leoma.bootstrap import emit_log as log
    
    rows = await miner_rank_dao.get_all_ordered_by_rank()
    winner_hotkey = await miner_rank_dao.get_winner_hotkey()

    log(f"Weights endpoint: {len(rows)} rows in miner_ranks, winner_hotkey={winner_hotkey[:12] if winner_hotkey else 'None'}...", "info")

    # Batch-load miners once (avoid a per-row N+1).
    miner_by_hotkey = {m.miner_hotkey: m for m in await valid_miners_dao.get_all_miners()}

    winner_uid = 0
    if winner_hotkey:
        miner = miner_by_hotkey.get(winner_hotkey)
        if miner and miner.is_valid:
            winner_uid = miner.uid
            log(f"Found winner: hotkey={winner_hotkey[:12]}..., uid={winner_uid}", "info")
        elif miner:
            log(f"Winner hotkey {winner_hotkey[:12]}... is not valid (is_valid=False), using winner_uid=0", "warn")
        else:
            log(f"Winner hotkey {winner_hotkey[:12]}... not found in valid_miners table", "warn")

    miners: List[MinerWeightEntry] = []
    for r in rows:
        miner = miner_by_hotkey.get(r.miner_hotkey)
        uid = miner.uid if miner else 0
        weight = 1.0 if r.rank == 1 and winner_uid and uid == winner_uid else 0.0
        miners.append(
            MinerWeightEntry(
                miner_hotkey=r.miner_hotkey,
                uid=uid,
                pass_rate=r.pass_rate,
                weight=weight,
            )
        )
    return WeightsResponse(winner_uid=winner_uid, miners=miners)
