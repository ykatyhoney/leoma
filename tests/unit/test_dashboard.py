"""
Unit tests for the public dashboard.json payload builder.
"""

from leoma.app.validator.state_store import KingState
from leoma.app.validator.dashboard import build_dashboard
from leoma.app.validator import king as K


def _digest(c):
    return "sha256:" + c * 64


def _two_king_state():
    st = KingState()
    st.king, st.king_chain = K.crown(
        None, [], hotkey="A", model_repo="u/leoma-A", model_digest=_digest("a"),
        block=100, challenge_id="seed", crowned_at="t0",
    )
    st.king, st.king_chain = K.crown(
        st.king, st.king_chain, hotkey="B", model_repo="u/leoma-B", model_digest=_digest("b"),
        block=200, challenge_id="eval-1", crowned_at="t1",
    )
    return st


class TestBuildDashboard:
    def test_king_carries_uid_and_reign(self):
        st = _two_king_state()
        d = build_dashboard(st, {"A": 10, "B": 20}, chain_meta={"name": "leoma"},
                            duel_params={}, updated_at="now")
        assert d["king"]["hotkey"] == "B"
        assert d["king"]["uid"] == 20
        assert d["king"]["reign_number"] == 1
        assert d["king"]["previous_repo"] == "u/leoma-A"

    def test_equal_weight_across_registered_kings(self):
        st = _two_king_state()
        d = build_dashboard(st, {"A": 10, "B": 20}, chain_meta={}, duel_params={}, updated_at="now")
        rows = {r["hotkey"]: r for r in d["king_chain"]}
        assert [r["hotkey"] for r in d["king_chain"]] == ["B", "A"]  # current king first
        assert rows["A"]["weight"] == 0.5
        assert rows["B"]["weight"] == 0.5
        assert rows["A"]["uid"] == 10

    def test_unregistered_king_gets_null_weight(self):
        st = _two_king_state()
        d = build_dashboard(st, {"B": 20}, chain_meta={}, duel_params={}, updated_at="now")
        rows = {r["hotkey"]: r for r in d["king_chain"]}
        assert rows["B"]["weight"] == 1.0   # only B registered -> 100%
        assert rows["A"]["weight"] is None

    def test_no_king_burns_empty(self):
        st = KingState()
        d = build_dashboard(st, {}, chain_meta={}, duel_params={}, updated_at="now")
        assert d["king"] == {}
        assert d["king_chain"] == []

    def test_history_and_queue_passthrough(self):
        st = _two_king_state()
        st.record_duel({"hotkey": "B", "verdict": "challenger", "accepted": True})
        st.record_duel({"hotkey": "C", "verdict": "king", "accepted": False})
        q = [{"hotkey": "C", "status": "unseen"}]
        d = build_dashboard(st, {"A": 10, "B": 20}, chain_meta={}, duel_params={}, updated_at="now", queue=q)
        assert [h["hotkey"] for h in d["history"]] == ["C", "B"]  # newest first
        assert d["queue"] == q

    def test_carries_meta_and_params(self):
        st = _two_king_state()
        d = build_dashboard(
            st, {"A": 10, "B": 20},
            chain_meta={"name": "leoma", "netuid": 99},
            duel_params={"metric": "lpips", "delta_threshold": 0.0025},
            updated_at="2026-01-01T00:00:00Z",
        )
        assert d["chain"]["name"] == "leoma"
        assert d["duel_params"]["metric"] == "lpips"
        assert d["updated_at"] == "2026-01-01T00:00:00Z"
        assert d["stats"] == {"accepted": 0, "rejected": 0, "failed": 0}
