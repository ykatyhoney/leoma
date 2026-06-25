"""
Tests for owner-api-outage resilience in weight setting.

When the scoring window can't be fetched (owner-api down), the validator retries, then repeats the
last winner it computed instead of burning the epoch — but only when the owner-api is *unreachable*,
not when it is reachable and there is legitimately no eligible miner. Also covers the persisted
last-winner store.
"""
import pytest

from leoma.app.validator import aggregate_local, last_winner


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    # Avoid real backoff sleeps in tests.
    monkeypatch.setattr(aggregate_local, "_WINDOW_FETCH_ATTEMPTS", 1)
    monkeypatch.setattr(aggregate_local, "_WINDOW_FETCH_BACKOFF", 0.0)


class _DownClient:
    async def get_task_window(self, as_of_block=None):
        raise ConnectionError("owner-api down")

    async def close(self):
        pass


async def test_repeats_last_winner_when_owner_api_down(monkeypatch):
    monkeypatch.setattr(aggregate_local, "load_peers", lambda: {"V": object()})
    monkeypatch.setattr(aggregate_local, "load_last_winner", lambda: (5, "winner_hotkey_xyz"))
    uid, hk = await aggregate_local.compute_local_winner(_DownClient(), epoch_block=18000)
    assert (uid, hk) == (5, "winner_hotkey_xyz")


async def test_burns_when_down_and_no_last_winner(monkeypatch):
    monkeypatch.setattr(aggregate_local, "load_peers", lambda: {"V": object()})
    monkeypatch.setattr(aggregate_local, "load_last_winner", lambda: None)
    uid, hk = await aggregate_local.compute_local_winner(_DownClient(), epoch_block=18000)
    assert (uid, hk) == (0, None)


async def test_reachable_empty_window_is_legitimate_burn_not_fallback(monkeypatch):
    # A transient blip recovers on retry; the window is reachable but empty -> burn, NOT last winner.
    monkeypatch.setattr(aggregate_local, "_WINDOW_FETCH_ATTEMPTS", 2)
    monkeypatch.setattr(aggregate_local, "load_peers", lambda: {"V": object()})
    fallback_calls = {"n": 0}
    monkeypatch.setattr(
        aggregate_local, "load_last_winner", lambda: (fallback_calls.__setitem__("n", 1) or (5, "x"))
    )

    calls = {"n": 0}

    class _Flaky:
        async def get_task_window(self, as_of_block=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("blip")
            return {"window": [], "active_validators": []}

        async def close(self):
            pass

    uid, hk = await aggregate_local.compute_local_winner(_Flaky(), epoch_block=18000)
    assert (uid, hk) == (0, None)
    assert calls["n"] == 2           # retried after the blip
    assert fallback_calls["n"] == 0  # did NOT fall back to last winner


def test_last_winner_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LEOMA_STATE_DIR", str(tmp_path))
    assert last_winner.load_last_winner() is None
    last_winner.save_last_winner(7, "hk_abc", epoch_block=18000)
    assert last_winner.load_last_winner() == (7, "hk_abc")


def test_last_winner_ignores_uid_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("LEOMA_STATE_DIR", str(tmp_path))
    last_winner.save_last_winner(0, "hk", epoch_block=1)  # UID 0 is a burn, not a real winner
    assert last_winner.load_last_winner() is None
