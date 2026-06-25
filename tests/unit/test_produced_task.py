"""
Unit tests for the produced-task ledger store (gap-free scoring sequence).

Covers: monotonic ``task_seq`` assignment, idempotency per ``rotation_id``, and the production-based
window — the last N *produced* tasks (skip-robust, gappy in rotation_id), settle margin, as_of_block
filtering (determinism), max-lookback bound, and idempotent backfill.
"""
import pytest

from leoma.infra.db.stores.store_produced_task import ProducedTaskStore

# interval used to map rotation_id -> block in these tests
INTERVAL = 100


@pytest.fixture
def store(mock_get_session):
    """A ProducedTaskStore wired to the in-memory test DB."""
    return ProducedTaskStore()


async def _seed(store, rotation_ids, sampler="V1"):
    """Append a list of rotation_ids (block = rotation_id * INTERVAL) in order."""
    out = []
    for rid in rotation_ids:
        out.append(await store.append(rotation_id=rid, sampler_hotkey=sampler, block=rid * INTERVAL))
    return out


class TestAppend:
    async def test_assigns_monotonic_task_seq(self, store):
        res = await _seed(store, [10, 11, 13])  # note: 12 skipped
        assert [r["task_seq"] for r in res] == [1, 2, 3]
        assert all(r["applied"] for r in res)

    async def test_idempotent_per_rotation_id(self, store):
        first = await store.append(rotation_id=10, sampler_hotkey="V1", block=1000)
        again = await store.append(rotation_id=10, sampler_hotkey="V2", block=1000)
        assert first["applied"] is True
        assert again["applied"] is False
        assert again["task_seq"] == first["task_seq"]  # returns the existing seq
        assert await store.count() == 1  # no duplicate row

    async def test_has_rotation(self, store):
        await store.append(rotation_id=42, sampler_hotkey="V1", block=4200)
        assert await store.has_rotation(42) is True
        assert await store.has_rotation(43) is False


class TestWindow:
    async def test_last_n_produced_is_skip_robust(self, store):
        # Gappy rotation_ids (turns skipped): the window is the last N *produced* by seq,
        # contiguous in seq but gappy in rotation_id — NOT the last N numbers.
        await _seed(store, [10, 11, 13, 17, 20])
        rows = await store.window(as_of_block=10_000, n=3, margin=0)
        assert [r.rotation_id for r in rows] == [13, 17, 20]  # ascending, last 3 produced
        assert [r.task_seq for r in rows] == [3, 4, 5]

    async def test_margin_drops_newest_produced(self, store):
        await _seed(store, [10, 11, 13, 17, 20])
        rows = await store.window(as_of_block=10_000, n=10, margin=2)
        # Drops the 2 newest produced (20, 17); keeps the rest.
        assert [r.rotation_id for r in rows] == [10, 11, 13]

    async def test_as_of_block_excludes_newer_rows(self, store):
        await _seed(store, [10, 11, 13, 17, 20])  # blocks 1000,1100,1300,1700,2000
        rows = await store.window(as_of_block=1300, n=10, margin=0)
        assert [r.rotation_id for r in rows] == [10, 11, 13]  # 17,20 are beyond as_of_block

    async def test_determinism_same_as_of_block(self, store):
        await _seed(store, [10, 11, 13, 17, 20])
        a = await store.window(as_of_block=1700, n=100, margin=1)
        b = await store.window(as_of_block=1700, n=100, margin=1)
        assert [r.rotation_id for r in a] == [r.rotation_id for r in b]

    async def test_max_lookback_floor(self, store):
        await _seed(store, [10, 11, 13, 17, 20])  # blocks up to 2000
        # as_of_block 2000, lookback 500 -> floor block 1500 -> only 17 (1700) and 20 (2000).
        rows = await store.window(as_of_block=2000, n=100, margin=0, max_lookback=500)
        assert [r.rotation_id for r in rows] == [17, 20]

    async def test_empty_when_nothing_produced(self, store):
        rows = await store.window(as_of_block=10_000, n=100, margin=0)
        assert rows == []

    async def test_active_validators_derivable_from_window(self, store):
        await store.append(rotation_id=1, sampler_hotkey="A", block=100)
        await store.append(rotation_id=2, sampler_hotkey="B", block=200)
        await store.append(rotation_id=3, sampler_hotkey="A", block=300)
        rows = await store.window(as_of_block=10_000, n=100, margin=0)
        assert sorted({r.sampler_hotkey for r in rows}) == ["A", "B"]


class TestBackfill:
    async def test_backfill_is_idempotent(self, store):
        entries = [
            {"rotation_id": 5, "sampler_hotkey": "A", "block": 500},
            {"rotation_id": 6, "sampler_hotkey": "A", "block": 600},
            {"rotation_id": 8, "sampler_hotkey": "B", "block": 800},
        ]
        assert await store.backfill(entries) == 3
        assert await store.backfill(entries) == 0  # re-run inserts nothing
        assert await store.count() == 3
        rows = await store.window(as_of_block=10_000, n=100, margin=0)
        assert [r.rotation_id for r in rows] == [5, 6, 8]
