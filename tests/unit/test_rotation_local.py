"""LocalRotation: compute whose turn it is from chain block + on-chain allowlist (no owner-api)."""
from leoma.app.validator import rotation_local as rl
from leoma.infra.onchain_allowlist import AllowlistSnapshot

VALS = ["A", "B", "C", "D"]


class _Sub:
    def __init__(self, block): self._b = block
    async def get_current_block(self): return self._b


def _areturn(val):
    async def f(*a, **k):
        return val
    return f


def _patch(monkeypatch, snap):
    monkeypatch.setattr(rl, "read_allowlist", _areturn(snap))
    monkeypatch.setattr(rl, "load_peers", lambda: {})   # no peers -> _produced returns False
    monkeypatch.setattr(rl, "create_source_read_client", lambda: object())


async def test_primary_turn(monkeypatch):
    _patch(monkeypatch, AllowlistSnapshot(validators=VALS, interval=100, digest="d"))
    view = await rl.LocalRotation(_Sub(1000), my_hotkey="C").whose_turn()  # rid 10 -> V[10%4]="C"
    assert view.rotation_index == 10
    assert view.sampler == "C" and view.failover_step == 0 and view.is_your_turn is True


async def test_not_my_turn(monkeypatch):
    _patch(monkeypatch, AllowlistSnapshot(validators=VALS, interval=100, digest="d"))
    view = await rl.LocalRotation(_Sub(1000), my_hotkey="A").whose_turn()
    assert view.sampler == "C" and view.is_your_turn is False


async def test_failover_backup_turn(monkeypatch):
    # primary C didn't produce (no peers) and the grace period (50) lapsed -> backup D (step 1).
    _patch(monkeypatch, AllowlistSnapshot(validators=VALS, interval=100, digest="d"))
    view = await rl.LocalRotation(_Sub(1060), my_hotkey="D").whose_turn()  # rid 10, offset 60
    assert view.failover_step == 1 and view.sampler == "D" and view.is_your_turn is True


async def test_none_when_allowlist_unreadable(monkeypatch):
    monkeypatch.setattr(rl, "read_allowlist", _areturn(None))
    monkeypatch.setattr(rl, "create_source_read_client", lambda: object())
    assert await rl.LocalRotation(_Sub(1000), my_hotkey="A").whose_turn() is None
