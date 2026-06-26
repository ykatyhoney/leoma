"""LocalRotation: compute whose turn it is from chain block + hardcoded allowlist (no owner-api)."""
from leoma.app.validator import rotation_local as rl
from leoma.infra.allowlist import AllowlistSnapshot

VALS = ["A", "B", "C", "D"]


class _Sub:
    def __init__(self, block): self._b = block
    async def get_current_block(self): return self._b


class _SubErr:
    async def get_current_block(self): raise RuntimeError("chain down")


def _patch(monkeypatch):
    monkeypatch.setattr(rl, "load_allowlist", lambda interval=100: AllowlistSnapshot(VALS, 100))
    monkeypatch.setattr(rl, "load_peers", lambda: {})   # no peers -> _produced returns False


async def test_primary_turn(monkeypatch):
    _patch(monkeypatch)
    view = await rl.LocalRotation(_Sub(1000), my_hotkey="C").whose_turn()  # rid 10 -> V[10%4]="C"
    assert view.rotation_index == 10
    assert view.sampler == "C" and view.failover_step == 0 and view.is_your_turn is True


async def test_not_my_turn(monkeypatch):
    _patch(monkeypatch)
    view = await rl.LocalRotation(_Sub(1000), my_hotkey="A").whose_turn()
    assert view.sampler == "C" and view.is_your_turn is False


async def test_failover_backup_turn(monkeypatch):
    # primary C didn't produce (no peers) and the grace period (50) lapsed -> backup D (step 1).
    _patch(monkeypatch)
    view = await rl.LocalRotation(_Sub(1060), my_hotkey="D").whose_turn()  # rid 10, offset 60
    assert view.failover_step == 1 and view.sampler == "D" and view.is_your_turn is True


async def test_none_when_block_unreadable(monkeypatch):
    _patch(monkeypatch)
    assert await rl.LocalRotation(_SubErr(), my_hotkey="A").whose_turn() is None
