"""One-time backfill of the produced-task ledger from historical validator_samples.

The scoring window reads the produced-task ledger ("last N produced tasks"). On a deploy that adds
the ledger, the table is empty, so the window would be empty until ~N fresh tasks accrue. This seeds
it from existing samples — each distinct task_id mapped to the validator that produced it (self-eval),
block approximated as ``task_id × interval`` — so scoring doesn't blackout. Idempotent (skips
rotation_ids already present), so it is safe to run on every startup.
"""
from leoma.bootstrap import SAMPLING_ROTATION_INTERVAL, emit_log as log
from leoma.infra.db.stores import ProducedTaskStore, SampleStore, SamplingStateStore


async def backfill_produced_task_ledger(only_if_empty: bool = True) -> int:
    """Seed ``produced_tasks`` from ``validator_samples``. Returns the number of rows inserted.

    With ``only_if_empty`` (the startup default), it no-ops once the ledger has any rows — so it
    self-disables after the first seed or as soon as live announces populate the ledger. The CLI
    passes ``only_if_empty=False`` to force a (still idempotent) full reconciliation pass.
    """
    produced = ProducedTaskStore()
    if only_if_empty and await produced.count() > 0:
        return 0

    interval = await SamplingStateStore().get_rotation_interval(SAMPLING_ROTATION_INTERVAL)
    pairs = await SampleStore().get_distinct_task_samplers()
    entries = [
        {"rotation_id": tid, "sampler_hotkey": hk, "block": tid * interval}
        for tid, hk in pairs
    ]
    inserted = await produced.backfill(entries)
    if inserted:
        log(
            f"Produced-task ledger: backfilled {inserted} rows from {len(pairs)} historical tasks "
            f"(interval={interval})",
            "success",
        )
    return inserted
