"""
Peer-bucket-outage resilience in weight setting.

The validator set is hardcoded in the repo, so it's always available — the resilience case is a
transient peer-bucket outage: a settled window exists but no verdict files can be read. Then the
validator repeats the last winner it computed instead of burning the epoch, while a legitimately
empty window (nothing produced yet) is a real burn, not a fallback. Also covers the last-winner store.
"""
from leoma.app.validator import aggregate_local, last_winner
from leoma.infra.allowlist import AllowlistSnapshot


def _areturn(val):
    async def f(*a, **k):
        return val
    return f


class _Peer:
    hotkey = "V"
    bucket = "bucket::V"


class _FailClient:
    def get_object(self, bucket, key):
        raise FileNotFoundError(key)


def _patch_window(monkeypatch):
    """A non-empty settled window whose verdict files all fail to read (bucket outage)."""
    monkeypatch.setattr(aggregate_local, "load_peers", lambda: {"V": _Peer()})
    monkeypatch.setattr(aggregate_local, "load_allowlist", lambda interval=100: AllowlistSnapshot(["V"], 100))
    monkeypatch.setattr(
        aggregate_local, "_discover_produced", _areturn(({"V": {1, 2, 3, 4, 5}}, {"V": _FailClient()}))
    )


async def test_repeats_last_winner_when_verdicts_unreadable(monkeypatch):
    _patch_window(monkeypatch)
    monkeypatch.setattr(aggregate_local, "load_last_winner", lambda: (5, "winner_hotkey_xyz"))
    uid, hk = await aggregate_local.compute_local_winner(epoch_block=18000)
    assert (uid, hk) == (5, "winner_hotkey_xyz")


async def test_burns_when_unreadable_and_no_last_winner(monkeypatch):
    _patch_window(monkeypatch)
    monkeypatch.setattr(aggregate_local, "load_last_winner", lambda: None)
    uid, hk = await aggregate_local.compute_local_winner(epoch_block=18000)
    assert (uid, hk) == (0, None)


async def test_empty_window_is_legitimate_burn_not_fallback(monkeypatch):
    # Nothing produced -> empty window -> burn, and the last-winner fallback is NOT consulted.
    monkeypatch.setattr(aggregate_local, "load_peers", lambda: {"V": _Peer()})
    monkeypatch.setattr(aggregate_local, "load_allowlist", lambda interval=100: AllowlistSnapshot(["V"], 100))
    monkeypatch.setattr(aggregate_local, "_discover_produced", _areturn(({"V": set()}, {})))
    calls = {"n": 0}
    monkeypatch.setattr(aggregate_local, "load_last_winner", lambda: calls.__setitem__("n", 1) or (5, "x"))
    uid, hk = await aggregate_local.compute_local_winner(epoch_block=18000)
    assert (uid, hk) == (0, None)
    assert calls["n"] == 0   # did NOT fall back to last winner


def test_last_winner_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LEOMA_STATE_DIR", str(tmp_path))
    assert last_winner.load_last_winner() is None
    last_winner.save_last_winner(7, "hk_abc", epoch_block=18000)
    assert last_winner.load_last_winner() == (7, "hk_abc")


def test_last_winner_ignores_uid_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("LEOMA_STATE_DIR", str(tmp_path))
    last_winner.save_last_winner(0, "hk", epoch_block=1)   # UID 0 is a burn, not a real winner
    assert last_winner.load_last_winner() is None
