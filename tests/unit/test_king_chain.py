"""
Unit tests for king-of-the-hill state transitions and weight targets.
"""

from leoma.app.validator import king as K


def _digest(c):
    return "sha256:" + c * 64


class TestCrown:
    def test_seed_king_no_reign_bump_no_chain(self):
        k, chain = K.crown(
            None, [], hotkey="", model_repo="org/leoma-genesis", model_digest=_digest("a"),
            block=100, challenge_id="seed", crowned_at="t0",
        )
        assert k["reign_number"] == 0
        assert k["previous_repo"] == ""
        assert chain == []

    def test_dethrone_pushes_prior_king_and_bumps_reign(self):
        k1, ch = K.crown(None, [], hotkey="A", model_repo="u/leoma-A", model_digest=_digest("a"),
                         block=100, challenge_id="seed", crowned_at="t0")
        k2, ch = K.crown(k1, ch, hotkey="B", model_repo="u/leoma-B", model_digest=_digest("b"),
                         block=200, challenge_id="eval-1", crowned_at="t1")
        assert k2["hotkey"] == "B"
        assert k2["reign_number"] == 1
        assert k2["previous_repo"] == "u/leoma-A"
        assert [e["hotkey"] for e in ch] == ["A"]

    def test_chain_capped_at_size_minus_one(self):
        k, ch = None, []
        # crown 8 distinct kings; chain holds at most KING_CHAIN_SIZE-1 prior kings
        for i in range(8):
            k, ch = K.crown(k, ch, hotkey=f"H{i}", model_repo=f"u/leoma-{i}",
                            model_digest=_digest(chr(97 + i)), block=100 + i,
                            challenge_id=f"eval-{i}", crowned_at=f"t{i}")
        assert len(ch) == K.KING_CHAIN_SIZE - 1
        # most recent deposed king is at the front
        assert ch[0]["hotkey"] == "H6"

    def test_inputs_not_mutated(self):
        k1, ch = K.crown(None, [], hotkey="A", model_repo="u/leoma-A", model_digest=_digest("a"),
                         block=1, challenge_id="seed", crowned_at="t0")
        orig_chain = list(ch)
        K.crown(k1, ch, hotkey="B", model_repo="u/leoma-B", model_digest=_digest("b"),
                block=2, challenge_id="eval-1", crowned_at="t1")
        assert ch == orig_chain  # crown returns a new chain, doesn't mutate input


class TestWeightTargets:
    def test_equal_split_among_registered_kings(self):
        king = {"hotkey": "A"}
        chain = [{"hotkey": "B"}, {"hotkey": "C"}]
        uid_map = {"A": 10, "B": 20, "C": 30}
        uids, weights, label = K.weight_targets(king, chain, uid_map)
        assert uids == [10, 20, 30]
        assert weights == [round(1 / 3, 9)] * 3
        assert label == "A"

    def test_skips_kings_not_on_metagraph(self):
        king = {"hotkey": "A"}
        chain = [{"hotkey": "B"}]  # B not registered
        uid_map = {"A": 10}
        uids, weights, _ = K.weight_targets(king, chain, uid_map)
        assert uids == [10]
        assert weights == [1.0]

    def test_burns_when_no_king_hotkey_registered(self):
        uids, weights, label = K.weight_targets({"hotkey": "Z"}, [], {"A": 1}, burn_uid=0)
        assert uids == [0]
        assert weights == [1.0]
        assert label == "burn:uid=0"

    def test_burns_when_no_king(self):
        uids, weights, label = K.weight_targets(None, [], {"A": 1})
        assert uids == [K.BURN_UID]
        assert label.startswith("burn:")

    def test_dedupes_hotkeys(self):
        # A appears as both king and chain entry -> counted once.
        king = {"hotkey": "A"}
        chain = [{"hotkey": "A"}, {"hotkey": "B"}]
        uids, weights, _ = K.weight_targets(king, chain, {"A": 1, "B": 2})
        assert sorted(uids) == [1, 2]
        assert weights == [0.5, 0.5]
