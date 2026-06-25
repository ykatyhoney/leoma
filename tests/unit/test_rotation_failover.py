"""
Unit tests for sampler failover (Phase 2): deterministic turn advance + the claim lease.

Covers the pure ``failover_step`` and ``_apply_claim`` helpers, the ``POST /tasks/claim`` lease
endpoint (grant / deny / already-produced), and ``GET /rotation`` advancing the sampler to a backup
after the grace period when the window hasn't been produced.
"""
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from leoma.delivery.http.routes import rotation as rotation_mod
from leoma.delivery.http.routes import tasks as tasks_mod
from leoma.delivery.http.routes.rotation import failover_step
from leoma.delivery.http.routes.tasks import _apply_claim
from leoma.delivery.http.verifier import verify_permissioned_validator

VALIDATORS = ["v_a", "v_b", "v_c", "v_d"]  # already sorted


class TestFailoverStep:
    def test_within_grace_is_primary(self):
        assert failover_step(0, 50, 4) == 0
        assert failover_step(49, 50, 4) == 0

    def test_advances_one_per_grace_period(self):
        assert failover_step(50, 50, 4) == 1
        assert failover_step(99, 50, 4) == 1
        assert failover_step(100, 50, 4) == 2

    def test_capped_at_ring_size(self):
        assert failover_step(10_000, 50, 4) == 3  # never wraps past N-1

    def test_disabled_when_grace_zero_or_single_validator(self):
        assert failover_step(999, 0, 4) == 0
        assert failover_step(999, 50, 1) == 0


class TestApplyClaim:
    def test_grants_to_first_caller(self):
        claims = {}
        res = _apply_claim(claims, 5, "v_a", now=100.0, ttl=600.0)
        assert res == {"granted": True, "holder": "v_a"}
        assert claims[5][0] == "v_a"

    def test_denies_other_caller_while_active(self):
        claims = {}
        _apply_claim(claims, 5, "v_a", now=100.0, ttl=600.0)
        res = _apply_claim(claims, 5, "v_b", now=200.0, ttl=600.0)
        assert res == {"granted": False, "holder": "v_a"}

    def test_same_holder_refreshes(self):
        claims = {}
        _apply_claim(claims, 5, "v_a", now=100.0, ttl=600.0)
        res = _apply_claim(claims, 5, "v_a", now=300.0, ttl=600.0)
        assert res["granted"] is True
        assert claims[5][1] == 300.0 + 600.0  # expiry refreshed

    def test_expired_lease_is_reclaimable(self):
        claims = {}
        _apply_claim(claims, 5, "v_a", now=100.0, ttl=600.0)  # expires at 700
        res = _apply_claim(claims, 5, "v_b", now=800.0, ttl=600.0)  # past expiry
        assert res == {"granted": True, "holder": "v_b"}


async def test_claim_map_persists_round_trip(mock_get_session):
    """The lease map round-trips through the DB (int keys + float expiry), so it survives restart."""
    from leoma.infra.db.stores import SamplingStateStore

    s = SamplingStateStore()
    assert await s.load_claim_map() == {}
    await s.save_claim_map({7: ("v_a", 1234.5), 8: ("v_b", 9999.0)})
    assert await s.load_claim_map() == {7: ("v_a", 1234.5), 8: ("v_b", 9999.0)}


# --------------------------------------------------------------------------- #
# Endpoint tests                                                              #
# --------------------------------------------------------------------------- #


@pytest.fixture
def caller():
    """Mutable holder of the authenticated caller hotkey (swap between requests)."""
    return {"hotkey": "v_a"}


@pytest.fixture
def claim_client(caller, monkeypatch):
    app = FastAPI()
    app.include_router(tasks_mod.router, prefix="/tasks")

    async def _auth():
        return caller["hotkey"]

    app.dependency_overrides[verify_permissioned_validator] = _auth
    monkeypatch.setattr(tasks_mod.produced_task_dao, "has_rotation", AsyncMock(return_value=False))

    # Fake the persisted lease map with an in-test dict so the endpoint logic is exercised without DB.
    store: dict = {}

    async def _load():
        return dict(store)

    async def _save(m):
        store.clear()
        store.update(m)

    monkeypatch.setattr(tasks_mod.sampling_state_dao, "load_claim_map", _load)
    monkeypatch.setattr(tasks_mod.sampling_state_dao, "save_claim_map", _save)
    transport = ASGITransport(app=app)
    return app, transport


async def test_claim_grants_then_denies_other(claim_client, caller):
    app, transport = claim_client
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        caller["hotkey"] = "v_a"
        r1 = await ac.post("/tasks/claim", json={"rotation_id": 7})
        assert r1.json()["granted"] is True
        caller["hotkey"] = "v_b"
        r2 = await ac.post("/tasks/claim", json={"rotation_id": 7})
        body = r2.json()
        assert body["granted"] is False and body["holder"] == "v_a"
        # original holder can still re-claim (refresh)
        caller["hotkey"] = "v_a"
        r3 = await ac.post("/tasks/claim", json={"rotation_id": 7})
        assert r3.json()["granted"] is True


async def test_claim_denied_when_already_produced(claim_client, caller, monkeypatch):
    app, transport = claim_client
    monkeypatch.setattr(tasks_mod.produced_task_dao, "has_rotation", AsyncMock(return_value=True))
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/tasks/claim", json={"rotation_id": 7})
        body = r.json()
        assert body["granted"] is False
        assert body["already_produced"] is True


@pytest.fixture
def rotation_client(caller, monkeypatch):
    app = FastAPI()
    app.include_router(rotation_mod.router, prefix="/rotation")

    async def _auth():
        return caller["hotkey"]

    app.dependency_overrides[verify_permissioned_validator] = _auth

    async def _ordered():
        return list(VALIDATORS)

    monkeypatch.setattr(rotation_mod, "ordered_validators", _ordered)  # owner-managed allowlist
    monkeypatch.setattr(rotation_mod, "FAILOVER_GRACE_BLOCKS", 0)  # derive interval//2 = 50
    monkeypatch.setattr(
        rotation_mod.sampling_state_dao, "get_rotation_interval", AsyncMock(return_value=100)
    )
    monkeypatch.setattr(
        rotation_mod.produced_task_dao, "has_rotation", AsyncMock(return_value=False)
    )
    transport = ASGITransport(app=app)
    return app, transport


async def test_rotation_primary_within_grace(rotation_client, monkeypatch):
    app, transport = rotation_client
    # block 18000 -> rotation_index 180, offset 0 -> primary = VALIDATORS[180 % 4] = v_a
    monkeypatch.setattr(rotation_mod, "_current_block", AsyncMock(return_value=18000))
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.get("/rotation")
        body = r.json()
        assert body["rotation_index"] == 180
        assert body["failover_step"] == 0
        assert body["sampler_hotkey"] == "v_a"
        assert body["primary_sampler"] == "v_a"


async def test_rotation_advances_to_backup_after_grace(rotation_client, caller, monkeypatch):
    app, transport = rotation_client
    # block 18060 -> offset 60 >= grace 50 -> step 1 -> sampler = VALIDATORS[181 % 4] = v_b
    monkeypatch.setattr(rotation_mod, "_current_block", AsyncMock(return_value=18060))
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        caller["hotkey"] = "v_b"
        r = await ac.get("/rotation")
        body = r.json()
        assert body["failover_step"] == 1
        assert body["sampler_hotkey"] == "v_b"
        assert body["primary_sampler"] == "v_a"
        assert body["is_your_turn"] is True


async def test_rotation_no_failover_once_produced(rotation_client, monkeypatch):
    app, transport = rotation_client
    # Even past the grace period, a produced window does not fail over.
    monkeypatch.setattr(rotation_mod, "_current_block", AsyncMock(return_value=18060))
    monkeypatch.setattr(rotation_mod.produced_task_dao, "has_rotation", AsyncMock(return_value=True))
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.get("/rotation")
        body = r.json()
        assert body["failover_step"] == 0
        assert body["sampler_hotkey"] == "v_a"  # back to primary
