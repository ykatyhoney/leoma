"""
Unit tests for decentralized miner validation:
- MinerConsensusTask._consensus_for_miner: strict-majority is_valid + representative metadata.
- miner_validation._to_report_entry: MinerInfo -> report dict mapping.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from leoma.delivery.http.tasks.miner_consensus import MinerConsensusTask
from leoma.app.validator.miner_validation import _to_report_entry
from leoma.domain import MinerInfo


_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _row(validator, is_valid, when_offset=0, invalid_reason=None, uid=5, block=100, model="leoma/x"):
    return SimpleNamespace(
        validator_hotkey=validator,
        miner_hotkey="5MINER",
        uid=uid,
        block=block,
        model_name=model,
        model_revision="rev",
        model_hash="hash",
        chute_id="cid",
        chute_slug="slug",
        is_valid=is_valid,
        invalid_reason=invalid_reason,
        reported_at=_T0 + timedelta(seconds=when_offset),
    )


class TestConsensusForMiner:
    def test_strict_majority_valid(self):
        rows = [_row("A", True), _row("B", True), _row("C", False, invalid_reason="chute_not_running")]
        c = MinerConsensusTask._consensus_for_miner(rows)
        assert c["is_valid"] is True
        assert c["invalid_reason"] is None

    def test_tie_is_not_majority(self):
        # 2 valid, 2 invalid -> valid*2 (4) is not > total (4) -> invalid.
        rows = [_row("A", True), _row("B", True), _row("C", False), _row("D", False)]
        c = MinerConsensusTask._consensus_for_miner(rows)
        assert c["is_valid"] is False

    def test_majority_invalid_uses_most_common_reason(self):
        rows = [
            _row("A", False, invalid_reason="chute_not_running"),
            _row("B", False, invalid_reason="chute_not_running"),
            _row("C", False, invalid_reason="hf_model_fetch_failed"),
            _row("D", True),
        ]
        c = MinerConsensusTask._consensus_for_miner(rows)
        assert c["is_valid"] is False
        assert c["invalid_reason"] == "chute_not_running"

    def test_representative_is_most_recent_valid(self):
        # Metadata should come from the most-recent VALID report.
        rows = [
            _row("A", True, when_offset=10, uid=1, block=50),
            _row("B", True, when_offset=99, uid=2, block=77),   # most recent valid
            _row("C", False, when_offset=200, uid=9, block=999),  # newer but invalid
        ]
        c = MinerConsensusTask._consensus_for_miner(rows)
        assert c["uid"] == 2 and c["block"] == 77

    def test_single_reporter_is_consensus(self):
        c = MinerConsensusTask._consensus_for_miner([_row("A", True)])
        assert c["is_valid"] is True


class TestReportEntryMapping:
    def test_valid_miner_maps_fields(self):
        m = MinerInfo(
            uid=3, hotkey="5MINER", model_name="leoma/x", model_revision="rev",
            model_hash="hash", chute_id="cid", chute_slug="slug", block=120, is_valid=True,
        )
        e = _to_report_entry(m)
        assert e == {
            "uid": 3, "miner_hotkey": "5MINER", "model_name": "leoma/x",
            "model_revision": "rev", "model_hash": "hash", "chute_id": "cid",
            "chute_slug": "slug", "block": 120, "is_valid": True, "invalid_reason": None,
        }

    def test_invalid_miner_empty_strings_become_none(self):
        m = MinerInfo(uid=4, hotkey="5BAD", is_valid=False, invalid_reason="blacklisted")
        e = _to_report_entry(m)
        assert e["is_valid"] is False
        assert e["invalid_reason"] == "blacklisted"
        # MinerInfo defaults empty strings -> reported as None
        assert e["model_name"] is None and e["chute_id"] is None and e["block"] is None
