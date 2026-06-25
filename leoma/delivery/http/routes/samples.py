"""Samples routes for Leoma API."""
from typing import Annotated, Any, List

from fastapi import APIRouter, Depends, HTTPException, Query

from leoma.delivery.http.verifier import verify_signature
from leoma.delivery.http.contracts import SampleSubmission, SampleResponse, SampleBatchSubmission
from leoma.delivery.http.validators import validate_miner_hotkey, validate_validator_hotkey
from leoma.infra.db.stores import EvaluationSignatureStore, SampleStore


router = APIRouter()
validator_samples_dao = SampleStore()
evaluation_signature_dao = EvaluationSignatureStore()
MAX_BATCH_SIZE = 100  # Maximum samples allowed per batch request
MAX_LIST_LIMIT = 500  # Maximum limit for list endpoints (DoS mitigation)


def _to_sample_response(sample: Any) -> SampleResponse:
    """Convert sample ORM/entity record to API response model."""
    return SampleResponse(
        id=sample.id,
        task_id=sample.task_id,
        validator_hotkey=sample.validator_hotkey,
        miner_hotkey=sample.miner_hotkey,
        prompt=sample.prompt,
        passed=sample.passed,
        confidence=sample.confidence,
        reasoning=sample.reasoning,
        evaluated_at=sample.evaluated_at,
        latency_ms=getattr(sample, "latency_ms", None),
    )


async def _save_submitted_sample(
    sample: SampleSubmission,
    validator_hotkey: str,
    *,
    evaluation_signature: str | None = None,
) -> Any:
    """Persist a submitted sample through the sample store (dashboard DB only).

    Decentralized model: validators upload the evaluation_results JSON to their OWN
    bucket directly; the API just records the sample for the dashboard, so there is no
    central-bucket upload here.
    """
    passed = sample.passed
    saved = await validator_samples_dao.save_sample(
        validator_hotkey=validator_hotkey,
        task_id=sample.task_id,
        miner_hotkey=sample.miner_hotkey,
        s3_bucket=sample.s3_bucket,
        s3_prefix=sample.s3_prefix,
        passed=passed,
        prompt=sample.prompt,
        confidence=sample.confidence,
        reasoning=sample.reasoning,
        latency_ms=sample.latency_ms,
        original_artifacts=sample.original_artifacts,
        generated_artifacts=sample.generated_artifacts,
        presentation_order=sample.presentation_order,
    )
    sig = evaluation_signature or getattr(sample, "evaluation_signature", None)
    if sig:
        await evaluation_signature_dao.set_signature(sample.task_id, validator_hotkey, sig)
    return saved


@router.post("", response_model=SampleResponse)
async def submit_sample(
    sample: SampleSubmission,
    hotkey: Annotated[str, Depends(verify_signature)],
) -> SampleResponse:
    """Submit sample metadata after evaluating a sample. Requires validator signature authentication."""
    result = await _save_submitted_sample(
        sample, validator_hotkey=hotkey,
        evaluation_signature=getattr(sample, "evaluation_signature", None),
    )
    return _to_sample_response(result)


@router.post("/batch", response_model=List[SampleResponse])
async def submit_samples_batch(
    body: SampleBatchSubmission,
    hotkey: Annotated[str, Depends(verify_signature)],
) -> List[SampleResponse]:
    """Submit multiple samples in batch (max 100). Requires validator signature authentication.

    Optional signature is over the evaluation payload (same shape as S3 "data") for verification.
    """
    samples = body.samples
    if len(samples) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit of {MAX_BATCH_SIZE} samples",
        )
    if body.signature and samples:
        task_ids = {s.task_id for s in samples if s.task_id is not None}
        if len(task_ids) != 1:
            raise HTTPException(
                status_code=400,
                detail="When signature is provided, all samples must have the same task_id",
            )
        tid = next(iter(task_ids))
        await evaluation_signature_dao.set_signature(tid, hotkey, body.signature)
    
    results = []
    for sample in samples:
        result = await _save_submitted_sample(sample, validator_hotkey=hotkey)
        results.append(_to_sample_response(result))

    return results


@router.get("/list", response_model=List[SampleResponse])
async def list_recent_samples(
    limit: int = Query(200, ge=1, le=MAX_LIST_LIMIT, description="Max items to return"),
) -> List[SampleResponse]:
    """List recent samples across all validators (public, for dashboard task list)."""
    samples = await validator_samples_dao.get_recent_samples(limit=limit)
    return [_to_sample_response(s) for s in samples]


@router.get("/task", response_model=List[SampleResponse])
async def get_task_samples(
    validator_hotkey: str,
    task_id: int,
    _hotkey: Annotated[str, Depends(verify_signature)],
) -> List[SampleResponse]:
    """Get all samples for one evaluation task from one validator. Validator-signature only.

    Exposes a validator's raw verdicts; gated like the sibling ``/samples/validator/{hk}`` and
    ``/samples/miner/{hk}`` reads (the public dashboard uses ``/tasks/{id}`` instead).
    """
    validator_hotkey = validate_validator_hotkey(validator_hotkey)
    samples = await validator_samples_dao.get_samples_by_validator_and_task_id(
        validator_hotkey=validator_hotkey,
        task_id=task_id,
    )
    return [_to_sample_response(s) for s in samples]


@router.get("/validator/{validator_hotkey}", response_model=List[SampleResponse])
async def get_validator_samples(
    validator_hotkey: str,
    hotkey: Annotated[str, Depends(verify_signature)],
    limit: int = Query(100, ge=1, le=MAX_LIST_LIMIT, description="Max items to return"),
) -> List[SampleResponse]:
    """Get samples submitted by a validator. Requires validator signature authentication."""
    validator_hotkey = validate_validator_hotkey(validator_hotkey)
    samples = await validator_samples_dao.get_samples_by_validator(
        validator_hotkey=validator_hotkey,
        limit=limit,
    )
    
    return [_to_sample_response(sample) for sample in samples]


@router.get("/miner/{miner_hotkey}", response_model=List[SampleResponse])
async def get_miner_samples(
    miner_hotkey: str,
    hotkey: Annotated[str, Depends(verify_signature)],
    limit: int = Query(100, ge=1, le=MAX_LIST_LIMIT, description="Max items to return"),
) -> List[SampleResponse]:
    """Get samples for a miner across all validators. Requires validator signature authentication."""
    miner_hotkey = validate_miner_hotkey(miner_hotkey)
    samples = await validator_samples_dao.get_samples_by_miner(
        miner_hotkey=miner_hotkey,
        limit=limit,
    )
    
    return [_to_sample_response(sample) for sample in samples]
