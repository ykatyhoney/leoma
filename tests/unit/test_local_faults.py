"""A validator's own misconfiguration must never be charged to a miner.

The attempt ledger quarantines an artifact once its attempts are `exhausted` —
whatever the failure class. So a validator whose *own* corpus was broken, or whose
eval box ran a stale `chain.toml`, would fail every duel transiently, four times
each, and then **quarantine every honest miner on the subnet** — permanently locking
them out over a mistake that was entirely its own.

That is the worst outcome the system can produce: it is silent, it is total, and the
victims did nothing wrong. Hence `ErrorClass.LOCAL`: faults that are ours cost the
challenger nothing and stop the validator instead.
"""

import pytest

import leoma.app.validator.main as vmain
from leoma.app.validator.failures import (
    ErrorClass,
    EvalJobFailed,
    classify,
    classify_remote,
)
from leoma.app.validator.reveal_scan import ChallengerEntry
from leoma.app.validator.state_store import KingState, MAX_DUEL_ATTEMPTS
from leoma.eval.errors import (
    ConsensusConfigError,
    CorpusIntegrityError,
    DegenerateGeneration,
    DuelCancelled,
    TransientDuelError,
)


def _entry(hotkey="5a", digest="sha256:" + "a" * 64, block=1):
    return ChallengerEntry(
        hotkey=hotkey, model_repo="u/leoma-c", model_digest=digest, block=block
    )


def _state():
    return KingState(
        king={"hotkey": "5KING", "model_repo": "u/leoma-king",
              "model_digest": "sha256:" + "k" * 64, "block": 0, "reign_number": 1},
        king_chain=[],
    )


class FakeSubtensor:
    async def get_block_hash(self, block):
        return f"0x{block:064x}"


class TestClassification:
    @pytest.mark.parametrize("exc", [
        CorpusIntegrityError("decoded ground truth does not match the manifest"),
        ConsensusConfigError("chain.toml [corpus].manifest_digest is not pinned"),
    ])
    def test_our_faults_are_LOCAL(self, exc):
        assert classify(exc).kind is ErrorClass.LOCAL

    def test_a_stale_eval_box_is_LOCAL(self):
        failure = classify(EvalJobFailed(
            "eval server pins a different consensus surface", reason="consensus_mismatch"
        ))
        assert failure.kind is ErrorClass.LOCAL

    def test_an_echo_mismatch_is_LOCAL(self):
        failure = classify(EvalJobFailed("spec drifted", reason="consensus_echo_mismatch"))
        assert failure.kind is ErrorClass.LOCAL

    def test_a_broken_model_is_still_PERMANENT(self):
        """LOCAL must not swallow real challenger faults — a degenerate model is the
        miner's problem, and it should be quarantined."""
        assert classify(DegenerateGeneration("1 frame")).kind is ErrorClass.PERMANENT

    def test_a_cancelled_duel_is_TRANSIENT_not_permanent(self):
        """DuelCancelled subclasses TransientDuelError, but it is checked before the
        ChallengerFault branch — a watchdog stall says nothing about the model."""
        assert classify(DuelCancelled("watchdog")).kind is ErrorClass.TRANSIENT

    def test_a_network_blip_is_still_TRANSIENT(self):
        assert classify(TransientDuelError("connection reset")).kind is ErrorClass.TRANSIENT

    def test_an_unknown_failure_still_fails_open_into_retry(self):
        assert classify_remote("something weird happened").kind is ErrorClass.TRANSIENT


class TestTheLedgerIsNeverCharged:
    async def test_a_broken_corpus_does_NOT_quarantine_a_single_miner(self, monkeypatch, duel_ready):
        """THE bug this class exists for. Four transient failures = 'exhausted' =
        quarantined. With a broken corpus every duel fails, so every miner on the
        subnet gets locked out — for our mistake."""
        state = _state()
        store = object()

        async def broken_corpus(entry, king, block_hash):
            raise CorpusIntegrityError("clip-0007: decoded ground truth does not match")

        monkeypatch.setattr(vmain, "dispatch_duel", broken_corpus)

        entries = [_entry(hotkey=f"5m{i}", digest="sha256:" + f"{i:064x}") for i in range(3)]

        # Run enough ticks to exhaust the attempt budget several times over.
        for block in range(1, MAX_DUEL_ATTEMPTS * 3):
            await vmain.process_challengers(
                FakeSubtensor(), None, state, {}, store, entries, block
            )

        assert state.attempts == {}, "a local fault was charged to the challengers' ledger"
        assert state.banned_hotkeys() == set()
        for entry in entries:
            key = vmain._seen_key(entry.hotkey, entry.model_digest)
            assert not state.is_quarantined(key)

    async def test_it_stops_dueling_and_says_why(self, monkeypatch, duel_ready):
        state = _state()

        async def stale_box(entry, king, block_hash):
            raise EvalJobFailed("box pins a different surface", reason="consensus_mismatch")

        monkeypatch.setattr(vmain, "dispatch_duel", stale_box)
        entries = [_entry(hotkey=f"5m{i}", digest="sha256:" + f"{i:064x}") for i in range(3)]

        await vmain.process_challengers(FakeSubtensor(), None, state, {}, object(), entries, 1)

        # Degraded, and it did not even try the other two — they would hit the same wall.
        assert state.degraded == "consensus_mismatch"
        assert state.stats.get("failed", 0) == 0

    async def test_a_genuinely_bad_model_IS_still_quarantined(self, monkeypatch, duel_ready):
        """The other half: LOCAL must not become a blanket amnesty. A model that is
        actually broken still gets quarantined, and the miners behind it still run."""
        state = _state()
        flushed = []

        class Store:
            async def put(self, key, value):
                flushed.append(key)

        async def bad_model(entry, king, block_hash):
            raise EvalJobFailed("repo does not exist", reason="model_not_found")

        monkeypatch.setattr(vmain, "dispatch_duel", bad_model)
        entry = _entry()

        for block in (1, 100):   # two sightings -> permanent quarantine
            await vmain.process_challengers(
                FakeSubtensor(), None, state, {}, Store(), [entry], block
            )

        key = vmain._seen_key(entry.hotkey, entry.model_digest)
        assert state.is_quarantined(key)
        assert state.stats["failed"] == 1
