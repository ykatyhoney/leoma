"""
Unit tests for the peer validator bucket registry (PEER_VALIDATORS parsing) and the
rotation sampler assignment.
"""

import json

import pytest

from leoma.infra import peer_registry
from leoma.bootstrap.runtime import settings


@pytest.fixture
def set_peer_env(monkeypatch):
    """Set settings.peer_validators to a given raw JSON string."""
    def _apply(raw: str):
        monkeypatch.setattr(settings, "peer_validators", raw)
    return _apply


def _peer(hotkey, bucket, uid=0):
    return {
        "hotkey": hotkey,
        "uid": uid,
        "bucket": bucket,
        "endpoint": "https://acct.r2.cloudflarestorage.com",
        "region": "auto",
        "read_access_key": "ak",
        "read_secret_key": "sk",
    }


class TestLoadPeers:
    def test_empty_returns_empty_map(self, set_peer_env):
        set_peer_env("")
        assert peer_registry.load_peers() == {}

    def test_invalid_json_returns_empty(self, set_peer_env):
        set_peer_env("{not json")
        assert peer_registry.load_peers() == {}

    def test_non_list_returns_empty(self, set_peer_env):
        set_peer_env(json.dumps({"hotkey": "x"}))
        assert peer_registry.load_peers() == {}

    def test_valid_entries_parsed(self, set_peer_env):
        set_peer_env(json.dumps([_peer("5A", "bucket-a", 1), _peer("5B", "bucket-b", 2)]))
        peers = peer_registry.load_peers()
        assert set(peers.keys()) == {"5A", "5B"}
        assert peers["5A"].bucket == "bucket-a"
        assert peers["5A"].read_access_key == "ak"
        assert peers["5B"].uid == 2

    def test_entry_missing_required_field_skipped(self, set_peer_env):
        bad = _peer("5C", "bucket-c")
        del bad["read_secret_key"]
        set_peer_env(json.dumps([_peer("5A", "bucket-a"), bad]))
        peers = peer_registry.load_peers()
        assert "5A" in peers
        assert "5C" not in peers

    def test_region_defaults_to_auto(self, set_peer_env):
        entry = _peer("5A", "bucket-a")
        del entry["region"]
        set_peer_env(json.dumps([entry]))
        assert peer_registry.load_peers()["5A"].region == "auto"

    def test_get_peer_and_hotkeys(self, set_peer_env):
        set_peer_env(json.dumps([_peer("5A", "bucket-a"), _peer("5B", "bucket-b")]))
        assert peer_registry.get_peer("5A").bucket == "bucket-a"
        assert peer_registry.get_peer("missing") is None
        assert sorted(peer_registry.peer_hotkeys()) == ["5A", "5B"]


class TestRotationSampler:
    """Sampler assignment is deterministic across the sorted permissioned set."""

    def test_compute_sampler_round_robin(self):
        from leoma.infra.rotation_math import compute_sampler

        validators = ["5A", "5B", "5C"]
        assert compute_sampler(validators, 0) == "5A"
        assert compute_sampler(validators, 1) == "5B"
        assert compute_sampler(validators, 2) == "5C"
        assert compute_sampler(validators, 3) == "5A"
        # task_id == rotation_index, so consecutive windows rotate evenly.

    def test_compute_sampler_empty(self):
        from leoma.infra.rotation_math import compute_sampler

        assert compute_sampler([], 5) is None
