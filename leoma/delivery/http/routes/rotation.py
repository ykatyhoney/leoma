"""
Owner-api dashboard helpers + the rotation-interval admin endpoint.

Rotation itself is computed locally by each validator from the on-chain allowlist
(see app/validator/rotation_local.py); the owner-api no longer decides whose turn it is. What
remains here is read-only support for the public dashboard: the validator order, the settled
scoring window (derived from the dual-reported ``validator_samples``), and the admin-settable
display interval.
"""
from typing import Annotated, List, Optional, Tuple

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from leoma.bootstrap import emit_log as log
from leoma.delivery.http.verifier import get_current_admin
from leoma.infra.db.stores import SampleStore, SamplingStateStore, ValidatorStore
from leoma.infra.scorer_constants import SCORER_SETTLE_MARGIN, SCORER_TASK_WINDOW


router = APIRouter()
sampling_state_dao = SamplingStateStore()
validator_samples_dao = SampleStore()
validator_store = ValidatorStore()


class IntervalUpdate(BaseModel):
    interval: int = Field(..., gt=0, description="Rotation interval in blocks (must be > 0).")


async def ordered_validators() -> List[str]:
    """Deterministic order of the owner-managed validator allowlist, sorted by hotkey."""
    validators = await validator_store.get_all_validators()
    return sorted(v.hotkey for v in validators)


async def current_scoring_window_rows() -> Tuple[List[int], List[str]]:
    """The settled scoring window as ``(task_ids, active_samplers)`` for the dashboard.

    Derived from the dual-reported ``validator_samples``: the last N distinct task_ids (dropping the
    newest settle margin) and the validators that sampled them. Informational only — validators set
    weights from their own locally-derived window.
    """
    return await validator_samples_dao.get_recent_task_window(SCORER_TASK_WINDOW, SCORER_SETTLE_MARGIN)


async def current_scoring_window() -> Optional[List[int]]:
    """Scoring-window task_ids only (settled), shared by the dashboard + scorer."""
    ids, _ = await current_scoring_window_rows()
    return ids or None


@router.post("/interval")
async def set_rotation_interval(
    body: IntervalUpdate,
    hotkey: Annotated[str, Depends(get_current_admin)],
) -> dict:
    """Set the rotation interval shown on the dashboard (blocks). Admin only."""
    await sampling_state_dao.set_rotation_interval(body.interval)
    log(f"Rotation interval set to {body.interval} blocks by {hotkey[:12]}...", "success")
    return {"interval": body.interval}
