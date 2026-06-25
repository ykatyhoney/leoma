"""Tests for the automatic produced-task ledger backfill (startup seed from validator_samples)."""
import pytest

from leoma.infra.db.stores import ProducedTaskStore
from leoma.infra.db.tables import ValidatorSample
from leoma.infra.ledger_backfill import backfill_produced_task_ledger


def _sample(task_id: int, validator: str, miner: str = "m") -> ValidatorSample:
    return ValidatorSample(
        validator_hotkey=validator,
        task_id=task_id,
        miner_hotkey=miner,
        s3_bucket="b",
        s3_prefix=f"t/{task_id}",
        passed=True,
    )


async def _all_rows():
    # max_lookback=0 so the block-age floor never hides backfilled rows in the assertion.
    return await ProducedTaskStore().window(as_of_block=10**12, n=1000, margin=0, max_lookback=0)


async def test_seeds_from_samples_when_empty(mock_get_session, db_session):
    for tid, v in [(10, "A"), (11, "A"), (13, "B")]:  # note: 12 skipped (gappy)
        db_session.add(_sample(tid, v))
    await db_session.commit()

    inserted = await backfill_produced_task_ledger(only_if_empty=True)
    assert inserted == 3

    rows = await _all_rows()
    assert [r.rotation_id for r in rows] == [10, 11, 13]
    assert {r.rotation_id: r.sampler_hotkey for r in rows} == {10: "A", 11: "A", 13: "B"}


async def test_noop_when_ledger_already_populated(mock_get_session, db_session):
    await ProducedTaskStore().append(rotation_id=1, sampler_hotkey="X", block=100)
    db_session.add(_sample(10, "A"))
    await db_session.commit()

    inserted = await backfill_produced_task_ledger(only_if_empty=True)
    assert inserted == 0  # ledger non-empty -> startup seed self-disables


async def test_force_runs_even_when_nonempty(mock_get_session, db_session):
    await ProducedTaskStore().append(rotation_id=1, sampler_hotkey="X", block=100)
    db_session.add(_sample(10, "A"))
    await db_session.commit()

    inserted = await backfill_produced_task_ledger(only_if_empty=False)
    assert inserted == 1  # task 10 seeded; rotation_id 1 already present -> skipped


async def test_idempotent_on_repeat(mock_get_session, db_session):
    db_session.add(_sample(10, "A"))
    db_session.add(_sample(11, "B"))
    await db_session.commit()

    assert await backfill_produced_task_ledger(only_if_empty=False) == 2
    assert await backfill_produced_task_ledger(only_if_empty=False) == 0  # nothing new
