"""
Tests for owner-managed validator allowlist (replaces stake-based auto-admission).

Covers: DB-backed `is_permissioned`, the admin register/remove API routes, and the stake-refresh
task being refresh-only (never adds or removes validators).
"""
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from leoma.delivery.http.routes import validators as validators_mod
from leoma.delivery.http.tasks import validator_sync as sync_mod
from leoma.delivery.http.verifier import SignatureVerifier, verify_admin_signature
from leoma.infra.db.stores import ValidatorStore


# --------------------------------------------------------------------------- #
# is_permissioned -> owner-managed validators table                           #
# --------------------------------------------------------------------------- #


class TestIsPermissioned:
    async def test_registered_validator_is_permissioned(self, mock_get_session, test_hotkeys, monkeypatch):
        monkeypatch.setattr("leoma.delivery.http.verifier.ADMIN_HOTKEYS", [])
        await ValidatorStore().save_validator(uid=0, hotkey=test_hotkeys[0], stake=1.0)
        verifier = SignatureVerifier()
        assert await verifier.is_permissioned(test_hotkeys[0]) is True

    async def test_unregistered_validator_is_not_permissioned(self, mock_get_session, test_hotkeys, monkeypatch):
        monkeypatch.setattr("leoma.delivery.http.verifier.ADMIN_HOTKEYS", [])
        verifier = SignatureVerifier()
        assert await verifier.is_permissioned(test_hotkeys[0]) is False

    async def test_admin_is_always_permissioned(self, mock_get_session, test_hotkeys, monkeypatch):
        monkeypatch.setattr("leoma.delivery.http.verifier.ADMIN_HOTKEYS", [test_hotkeys[3]])
        verifier = SignatureVerifier()
        assert await verifier.is_permissioned(test_hotkeys[3]) is True  # admin, not in validators


# --------------------------------------------------------------------------- #
# Admin register / remove routes                                              #
# --------------------------------------------------------------------------- #


ADMIN = "5C62W7ELLAAfjCQeBU3me9nTXXqjVwN4kQY8w8gM9nJ8K4pL"


@pytest.fixture
def admin_client(monkeypatch):
    app = FastAPI()
    app.include_router(validators_mod.router, prefix="/validators")

    async def _admin():
        return ADMIN

    app.dependency_overrides[verify_admin_signature] = _admin
    transport = ASGITransport(app=app)
    yield transport, monkeypatch
    app.dependency_overrides.clear()


async def test_register_validator(admin_client, test_hotkeys):
    transport, monkeypatch = admin_client
    from leoma.infra.db.tables import Validator

    saved = Validator(uid=5, hotkey=test_hotkeys[0], stake=12345.0)
    save = AsyncMock(return_value=saved)
    monkeypatch.setattr(validators_mod.validators_dao, "save_validator", save)

    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/validators", json={"uid": 5, "hotkey": test_hotkeys[0], "stake": 12345.0})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True and body["uid"] == 5 and body["added_by"] == ADMIN
    save.assert_awaited_once()


async def test_register_validator_auto_resolves_uid_and_stake(admin_client, test_hotkeys):
    transport, monkeypatch = admin_client
    from leoma.infra.db.tables import Validator

    # uid omitted -> resolved from the metagraph (uid 7, stake 333.0).
    monkeypatch.setattr(validators_mod, "_resolve_from_metagraph", AsyncMock(return_value=(7, 333.0)))
    save = AsyncMock(return_value=Validator(uid=7, hotkey=test_hotkeys[0], stake=333.0))
    monkeypatch.setattr(validators_mod.validators_dao, "save_validator", save)

    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/validators", json={"hotkey": test_hotkeys[0]})  # no uid/stake
    assert r.status_code == 200
    body = r.json()
    assert body["uid"] == 7 and body["stake"] == 333.0
    assert save.await_args.kwargs["uid"] == 7 and save.await_args.kwargs["stake"] == 333.0


async def test_register_validator_unknown_hotkey_404(admin_client, test_hotkeys):
    transport, monkeypatch = admin_client
    monkeypatch.setattr(validators_mod, "_resolve_from_metagraph", AsyncMock(return_value=None))
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/validators", json={"hotkey": test_hotkeys[0]})  # not on metagraph
    assert r.status_code == 404


async def test_resolve_from_metagraph(test_hotkeys, monkeypatch):
    monkeypatch.setattr(
        validators_mod.bt, "AsyncSubtensor",
        lambda *a, **k: _FakeSubtensor([test_hotkeys[0], test_hotkeys[1]], [10.0, 20.0]),
    )
    assert await validators_mod._resolve_from_metagraph(test_hotkeys[1]) == (1, 20.0)
    assert await validators_mod._resolve_from_metagraph(test_hotkeys[2]) is None  # not registered


async def test_remove_validator(admin_client, test_hotkeys):
    transport, monkeypatch = admin_client
    monkeypatch.setattr(
        validators_mod.validators_dao, "delete_validator_by_hotkey", AsyncMock(return_value=True)
    )
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.delete(f"/validators/{test_hotkeys[0]}")
    assert r.status_code == 200
    assert r.json()["success"] is True


async def test_remove_unregistered_validator_404(admin_client, test_hotkeys):
    transport, monkeypatch = admin_client
    monkeypatch.setattr(
        validators_mod.validators_dao, "delete_validator_by_hotkey", AsyncMock(return_value=False)
    )
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.delete(f"/validators/{test_hotkeys[0]}")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Stake refresh is refresh-only (no add / no remove)                          #
# --------------------------------------------------------------------------- #


class _FakeMeta:
    def __init__(self, hotkeys, stakes):
        self.hotkeys = hotkeys
        self.S = stakes


class _FakeSubtensor:
    def __init__(self, hotkeys, stakes):
        self._meta = _FakeMeta(hotkeys, stakes)

    async def metagraph(self, netuid):
        return self._meta

    async def close(self):
        pass


async def test_refresh_only_updates_existing_never_adds_or_removes(
    mock_get_session, test_hotkeys, monkeypatch
):
    store = ValidatorStore()
    # Owner-added validators with stale stake 0.
    await store.save_validator(uid=0, hotkey=test_hotkeys[0], stake=0.0)
    await store.save_validator(uid=1, hotkey=test_hotkeys[1], stake=0.0)

    # Metagraph also contains a high-stake hotkey (test_hotkeys[2]) the owner did NOT add.
    meta_hotkeys = [test_hotkeys[0], test_hotkeys[1], test_hotkeys[2]]
    meta_stakes = [100.0, 200.0, 999999.0]
    monkeypatch.setattr(
        sync_mod.bt, "AsyncSubtensor", lambda *a, **k: _FakeSubtensor(meta_hotkeys, meta_stakes)
    )

    await sync_mod.ValidatorSyncTask()._refresh_stakes()

    all_v = await store.get_all_validators()
    assert {v.hotkey for v in all_v} == {test_hotkeys[0], test_hotkeys[1]}  # no add, no remove
    by_hk = {v.hotkey: v.stake for v in all_v}
    assert by_hk[test_hotkeys[0]] == 100.0  # stake refreshed
    assert by_hk[test_hotkeys[1]] == 200.0
