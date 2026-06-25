"""
Network overview for the dashboard header (decentralized snapshot).
"""
from fastapi import APIRouter

from leoma.bootstrap import SAMPLING_ROTATION_INTERVAL
from leoma.delivery.http.contracts import OverviewResponse
from leoma.delivery.http.routes.rotation import current_scoring_window
from leoma.infra.chute_status import probe_hot_chutes
from leoma.infra.db.stores import (
    ParticipantStore,
    SampleStore,
    SamplingStateStore,
    ValidatorStore,
)


router = APIRouter()
valid_miners_dao = ParticipantStore()
validators_dao = ValidatorStore()
sampling_state_dao = SamplingStateStore()
validator_samples_dao = SampleStore()


@router.get("", response_model=OverviewResponse)
async def get_overview() -> OverviewResponse:
    """Network snapshot: active/valid miner counts, validator counts, rotation state, window."""
    valid = await valid_miners_dao.get_valid_miners()
    hot = await probe_hot_chutes([m.chute_id for m in valid if m.chute_id])
    active = sum(1 for m in valid if m.chute_id and hot.get(m.chute_id))
    total_miners = await valid_miners_dao.get_total_count()

    validators = await validators_dao.get_all_validators()
    interval = await sampling_state_dao.get_rotation_interval(SAMPLING_ROTATION_INTERVAL)
    latest = await validator_samples_dao.get_latest_task()
    latest_task_id = latest[0] if latest else None
    current_sampler = latest[1] if latest else None

    window = await current_scoring_window()
    window_start = window[0] if window else None
    window_end = window[-1] if window else None

    return OverviewResponse(
        active_miners=active,
        valid_miners=len(valid),
        total_miners=total_miners,
        total_validators=len(validators),
        # Every owner-managed validator is permissioned in the unified design.
        permissioned_validators=len(validators),
        rotation_interval=interval,
        current_sampler=current_sampler,
        latest_task_id=latest_task_id,
        window_start=window_start,
        window_end=window_end,
    )
