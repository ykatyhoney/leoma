"""
Unit tests for the persisted king state (JSON KV over an object bucket).
"""

import io
import json

from leoma.app.validator.state_store import (
    JsonBucketStore,
    KingState,
    KEY_KING,
    KEY_KING_CHAIN,
    KEY_VALIDATOR_STATE,
    KEY_SEEN,
)


class FakeMinio:
    """In-memory stand-in for the Minio client's put/get object API."""

    def __init__(self):
        self.blobs: dict[tuple, bytes] = {}

    def put_object(self, bucket, key, data, length, content_type=None):
        payload = data.read()
        assert len(payload) == length
        self.blobs[(bucket, key)] = payload

    def get_object(self, bucket, key):
        if (bucket, key) not in self.blobs:
            raise KeyError(key)
        blob = self.blobs[(bucket, key)]

        class _Resp:
            def read(self_inner):
                return blob

            def close(self_inner):
                pass

            def release_conn(self_inner):
                pass

        return _Resp()


def _store():
    return JsonBucketStore(FakeMinio(), "own-bucket")


class TestJsonBucketStore:
    def test_put_get_roundtrip(self):
        s = _store()
        s.put("k.json", {"a": 1, "b": [2, 3]})
        assert s.get("k.json") == {"a": 1, "b": [2, 3]}

    def test_missing_key_returns_none(self):
        assert _store().get("absent.json") is None

    def test_writes_expected_keys(self):
        s = _store()
        st = KingState(king={"hotkey": "A"}, king_chain=[{"hotkey": "B"}])
        st.flush(s)
        for key in (KEY_KING, KEY_KING_CHAIN, KEY_VALIDATOR_STATE, KEY_SEEN):
            assert ("own-bucket", key) in s.client.blobs


class TestKingState:
    def test_flush_then_load_roundtrip(self):
        s = _store()
        st = KingState()
        st.king = {"hotkey": "A", "model_repo": "u/leoma-A", "reign_number": 3}
        st.king_chain = [{"hotkey": "B"}, {"hotkey": "C"}]
        st.last_weight_block = 4321
        st.last_winner_hotkey = "A"
        st.counter = 9
        st.stats = {"accepted": 2, "rejected": 1, "failed": 0}
        st.mark_seen("A|sha256:aaa")
        st.mark_seen("B|sha256:bbb")
        st.flush(s)

        loaded = KingState.load(s)
        assert loaded.king == st.king
        assert [e["hotkey"] for e in loaded.king_chain] == ["B", "C"]
        assert loaded.last_weight_block == 4321
        assert loaded.last_winner_hotkey == "A"
        assert loaded.counter == 9
        assert loaded.stats == {"accepted": 2, "rejected": 1, "failed": 0}
        assert loaded.seen_hotkeys == {"A|sha256:aaa", "B|sha256:bbb"}

    def test_load_empty_bucket_defaults(self):
        loaded = KingState.load(_store())
        assert loaded.king == {}
        assert loaded.king_chain == []
        assert loaded.last_weight_block == 0
        assert loaded.seen_hotkeys == set()

    def test_next_eval_id_increments_and_persists_counter(self):
        s = _store()
        st = KingState()
        assert st.next_eval_id() == "eval-0001"
        assert st.next_eval_id() == "eval-0002"
        st.flush(s)
        assert KingState.load(s).counter == 2

    def test_history_roundtrip_and_newest_first(self):
        s = _store()
        st = KingState()
        st.record_duel({"hotkey": "A", "verdict": "king"})
        st.record_duel({"hotkey": "B", "verdict": "challenger"})
        st.flush(s)
        loaded = KingState.load(s)
        assert [h["hotkey"] for h in loaded.history] == ["B", "A"]  # newest first

    def test_history_is_bounded(self):
        from leoma.app.validator.state_store import HISTORY_LIMIT
        st = KingState()
        for i in range(HISTORY_LIMIT + 25):
            st.record_duel({"hotkey": f"h{i}"})
        assert len(st.history) == HISTORY_LIMIT
        assert st.history[0]["hotkey"] == f"h{HISTORY_LIMIT + 24}"  # most recent kept
