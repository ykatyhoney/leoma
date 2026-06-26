"""is_permissioned reads the hardcoded repo allowlist (single source of truth, no DB)."""
from leoma.delivery.http import verifier as v


async def test_in_allowlist_is_permissioned(monkeypatch):
    monkeypatch.setattr(v, "VALIDATOR_ALLOWLIST", ["5Good"])
    monkeypatch.setattr(v, "ADMIN_HOTKEYS", [])
    assert await v.SignatureVerifier().is_permissioned("5Good") is True


async def test_not_in_allowlist_is_not_permissioned(monkeypatch):
    monkeypatch.setattr(v, "VALIDATOR_ALLOWLIST", ["5Good"])
    monkeypatch.setattr(v, "ADMIN_HOTKEYS", [])
    assert await v.SignatureVerifier().is_permissioned("5Other") is False


async def test_admin_is_always_permissioned(monkeypatch):
    monkeypatch.setattr(v, "VALIDATOR_ALLOWLIST", [])          # not in the allowlist
    monkeypatch.setattr(v, "ADMIN_HOTKEYS", ["5Admin"])
    assert await v.SignatureVerifier().is_permissioned("5Admin") is True
