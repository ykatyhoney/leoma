"""
Chain/allowlist-outage resilience in weight setting.

When the on-chain allowlist can't be read (chain unreachable), the validator repeats the last winner
it computed instead of burning the epoch — but only when the allowlist is *unreachable*, not when it
is readable and there is legitimately no settled window. Also covers the persisted last-winner store.
"""
from leoma.app.validator import aggregate_local, last_winner
from leoma.infra.onchain_allowlist import AllowlistSnapshot


def _areturn(val):
    async def f(*a, **k):
        return val
    return f


async def test_repeats_last_winner_when_chain_unreachable(monkeypatch):
    monkeypatch.setattr(aggregate_local, "load_peers", lambda: {"V": object()})
    monkeypatch.setattr(aggregate_local, "create_source_read_client", lambda: object())
    monkeypatch.setattr(aggregate_local, "read_allowlist", _areturn(None))   # chain down
    monkeypatch.setattr(aggregate_local, "load_last_winner", lambda: (5, "winner_hotkey_xyz"))
    uid, hk = await aggregate_local.compute_local_winner(object(), epoch_block=18000)
    assert (uid, hk) == (5, "winner_hotkey_xyz")


async def test_burns_when_down_and_no_last_winner(monkeypatch):
    monkeypatch.setattr(aggregate_local, "load_peers", lambda: {"V": object()})
    monkeypatch.setattr(aggregate_local, "create_source_read_client", lambda: object())
    monkeypatch.setattr(aggregate_local, "read_allowlist", _areturn(None))
    monkeypatch.setattr(aggregate_local, "load_last_winner", lambda: None)
    uid, hk = await aggregate_local.compute_local_winner(object(), epoch_block=18000)
    assert (uid, hk) == (0, None)


async def test_readable_but_empty_window_is_legitimate_burn_not_fallback(monkeypatch):
    # Allowlist reads fine, but nothing has been produced -> empty window -> burn, NOT last winner.
    snap = AllowlistSnapshot(validators=["V"], interval=100, digest="d")
    monkeypatch.setattr(aggregate_local, "load_peers", lambda: {"V": object()})
    monkeypatch.setattr(aggregate_local, "create_source_read_client", lambda: object())
    monkeypatch.setattr(aggregate_local, "read_allowlist", _areturn(snap))
    monkeypatch.setattr(aggregate_local, "_discover_produced", _areturn(({"V": set()}, {})))
    fallback_calls = {"n": 0}
    monkeypatch.setattr(
        aggregate_local, "load_last_winner", lambda: (fallback_calls.__setitem__("n", 1) or (5, "x"))
    )
    uid, hk = await aggregate_local.compute_local_winner(object(), epoch_block=18000)
    assert (uid, hk) == (0, None)
    assert fallback_calls["n"] == 0   # did NOT fall back to last winner


def test_last_winner_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LEOMA_STATE_DIR", str(tmp_path))
    assert last_winner.load_last_winner() is None
    last_winner.save_last_winner(7, "hk_abc", epoch_block=18000)
    assert last_winner.load_last_winner() == (7, "hk_abc")


def test_last_winner_ignores_uid_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("LEOMA_STATE_DIR", str(tmp_path))
    last_winner.save_last_winner(0, "hk", epoch_block=1)   # UID 0 is a burn, not a real winner
    assert last_winner.load_last_winner() is None
