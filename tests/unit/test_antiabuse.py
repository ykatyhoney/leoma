"""Copying the king, and monopolizing the GPU.

Two free attacks, both costing the attacker nothing:

* **Copy the king.** `_is_current_king` required the *hotkey* to match as well as the
  digest, so a **different** hotkey re-committing the king's exact digest was treated
  as a novel challenger and handed a full multi-hour duel. It could never actually win
  (a copy ties exactly, and the threshold demands strictly better) — it was simply
  free to repeat, forever.

* **Mint fresh digests.** The seen-set (`hotkey|digest`) stops the same *artifact*
  being dueled twice, but a hotkey can change one byte, re-upload, and get a brand-new
  key — and a brand-new **free multi-hour duel** on the only GPU in the subnet.
"""

import numpy as np
import pytest

import leoma.app.validator.main as vmain
from leoma.app.validator import rate_limit as RL
from leoma.app.validator.reveal_scan import ChallengerEntry
from leoma.app.validator.state_store import JsonBucketStore, KingState
from leoma.eval.metrics import mse
from leoma.eval.video_runner import Clip, GenParams, run_duel

from tests.unit.conftest import FakeEvalBox, FakeMinio, make_verdict

KING_DIGEST = "sha256:" + "k" * 64
PARAMS = GenParams(num_frames=4, fps=2, width=8, height=8)


def _store():
    return JsonBucketStore(FakeMinio(), "own", backoff=0)


def _state():
    st = KingState()
    st.king = {"hotkey": "5KING", "model_repo": "u/leoma-king",
               "model_digest": KING_DIGEST, "reign_number": 1}
    return st


def _entry(hotkey, digest, block=100):
    return ChallengerEntry(hotkey=hotkey, model_repo="u/leoma-x",
                           model_digest=digest, block=block)


class TestCopyOfKing:
    async def test_a_copy_of_the_king_is_rejected_WITHOUT_burning_a_duel(self, monkeypatch, duel_ready):
        """It cost hours of GPU every time, for a model that could never win."""
        st, store = _state(), _store()
        box = FakeEvalBox(monkeypatch, lambda e: AssertionError("must not dispatch"), duel_ready)

        thief = _entry("5THIEF", KING_DIGEST)     # the king's exact weights, new hotkey
        await box.drive(st, store, [thief], block=200, ticks=1)

        assert box.dispatched == [], "a copy of the king was handed the GPU"
        key = vmain._seen_key(thief.hotkey, thief.model_digest)
        assert st.attempts[key]["last_reason"] == "copy_of_king"

    async def test_the_king_itself_is_not_flagged_as_a_copy_of_itself(self, monkeypatch, duel_ready):
        st, store = _state(), _store()
        box = FakeEvalBox(monkeypatch, lambda e: AssertionError("must not dispatch"), duel_ready)

        incumbent = _entry("5KING", KING_DIGEST)   # the king re-revealing itself
        await box.drive(st, store, [incumbent], block=200, ticks=1)

        assert box.dispatched == []
        assert st.attempts == {}, "the incumbent was penalized for being the incumbent"

    async def test_copying_a_FORMER_king_is_also_plagiarism(self, monkeypatch, duel_ready):
        st, store = _state(), _store()
        old = "sha256:" + "0" * 64
        st.king_chain = [{"hotkey": "5OLD", "model_repo": "u/leoma-old", "model_digest": old}]

        box = FakeEvalBox(monkeypatch, lambda e: AssertionError("must not dispatch"), duel_ready)
        await box.drive(st, store, [_entry("5THIEF", old)], block=200, ticks=1)

        assert box.dispatched == []

    async def test_it_costs_the_thief_a_strike(self, monkeypatch, duel_ready):
        st, store = _state(), _store()
        box = FakeEvalBox(monkeypatch, lambda e: None, duel_ready)
        await box.drive(st, store, [_entry("5THIEF", KING_DIGEST)], block=200, ticks=1)

        assert st.duels["5THIEF"]["strikes"] == 1

    def test_bit_identical_generations_are_caught_at_duel_time_too(self):
        """Defence in depth, and free. The duel is deterministic: the same weights,
        seed and clip produce bit-identical frames. So a copy repackaged under a new
        digest is caught by what it DOES, not by what it claims to be — no repackaging
        survives this."""
        rng = np.random.default_rng(0)
        clips = [
            Clip(clip_index=i, clip_id=f"c{i}", first_frame=t[0], prompt="p",
                 truth_frames=t, params=PARAMS)
            for i, t in enumerate(
                rng.integers(0, 255, size=(6, 4, 8, 8, 3)).astype("uint8")
            )
        ]

        def same_model(clip, seed):
            return np.clip(clip.truth_frames.astype(int) + 5, 0, 255).astype("uint8")

        v = run_duel(clips, generate_king=same_model, generate_challenger=same_model,
                     distance_fn=mse, master_seed=1, delta_threshold=0.0025,
                     alpha=0.05, n_bootstrap=200)

        assert v["accepted"] is False
        assert v["rejected_by"] == "copy_of_king"
        assert "repackaged" in v["reason"]

    def test_two_genuinely_different_models_are_not_flagged(self):
        rng = np.random.default_rng(1)
        clips = [
            Clip(clip_index=i, clip_id=f"c{i}", first_frame=t[0], prompt="p",
                 truth_frames=t, params=PARAMS)
            for i, t in enumerate(rng.integers(0, 255, size=(6, 4, 8, 8, 3)).astype("uint8"))
        ]

        v = run_duel(
            clips,
            generate_king=lambda c, s: np.clip(c.truth_frames.astype(int) + 40, 0, 255).astype("uint8"),
            generate_challenger=lambda c, s: np.clip(c.truth_frames.astype(int) + 3, 0, 255).astype("uint8"),
            distance_fn=mse, master_seed=1, delta_threshold=0.0025, alpha=0.05, n_bootstrap=200,
        )
        assert "rejected_by" not in v
        assert v["accepted"] is True


class TestRateLimit:
    def test_a_fresh_hotkey_may_duel(self):
        assert RL.check({}, "5a", king={"reign_number": 1}, block=1000) is None

    def test_the_cooldown_bites_right_after_a_verdict(self):
        duels = {}
        RL.record_verdict(duels, "5a", king={"reign_number": 1}, block=1000)

        assert RL.check(duels, "5a", king={"reign_number": 1}, block=1001) is not None
        assert "cooldown" in RL.check(duels, "5a", king={"reign_number": 1}, block=1001)

    def test_the_cooldown_expires(self):
        duels = {}
        RL.record_verdict(duels, "5a", king={"reign_number": 1}, block=1000)
        later = 1000 + RL.COOLDOWN_BLOCKS
        assert RL.check(duels, "5a", king={"reign_number": 1}, block=later) is None

    def test_the_per_reign_cap_stops_a_hotkey_monopolizing_one_king(self):
        """Each duel costs HOURS of GPU. Without this, one miner minting fresh digests
        starves everyone else, at no cost to themselves."""
        duels = {}
        king = {"reign_number": 1}
        block = 1000
        for _ in range(RL.MAX_CHALLENGES_PER_REIGN):
            RL.record_verdict(duels, "5a", king=king, block=block)
            block += RL.COOLDOWN_BLOCKS       # respect the cooldown each time

        limited = RL.check(duels, "5a", king=king, block=block)
        assert limited is not None and "reign cap" in limited

    def test_a_NEW_king_gives_everyone_a_fresh_allowance(self):
        """The cap is per-reign: dethroning the king resets the contest."""
        duels = {}
        block = 1000
        for _ in range(RL.MAX_CHALLENGES_PER_REIGN):
            RL.record_verdict(duels, "5a", king={"reign_number": 1}, block=block)
            block += RL.COOLDOWN_BLOCKS

        assert RL.check(duels, "5a", king={"reign_number": 1}, block=block) is not None
        assert RL.check(duels, "5a", king={"reign_number": 2}, block=block) is None

    def test_a_long_reign_does_not_lock_everyone_out_forever(self):
        """Without the refresh, a durable king freezes the subnet: nobody can ever
        challenge it again and the incumbent reigns by default."""
        duels = {}
        king = {"reign_number": 1}
        block = 1000
        for _ in range(RL.MAX_CHALLENGES_PER_REIGN):
            RL.record_verdict(duels, "5a", king=king, block=block)
            block += RL.COOLDOWN_BLOCKS

        assert RL.check(duels, "5a", king=king, block=block) is not None

        much_later = 1000 + RL.REIGN_REFRESH_BLOCKS + 1
        assert RL.check(duels, "5a", king=king, block=much_later) is None

    def test_the_cap_is_per_hotkey_not_global(self):
        duels = {}
        king = {"reign_number": 1}
        block = 1000
        for _ in range(RL.MAX_CHALLENGES_PER_REIGN):
            RL.record_verdict(duels, "5a", king=king, block=block)
            block += RL.COOLDOWN_BLOCKS

        assert RL.check(duels, "5a", king=king, block=block) is not None
        assert RL.check(duels, "5b", king=king, block=block) is None   # unaffected


class TestStrikes:
    def test_LOSING_a_duel_is_never_a_strike(self):
        """A miner whose honest model isn't good enough has done nothing wrong.
        Penalizing them for trying would deter the people the subnet exists for."""
        duels = {}
        RL.record_strike(duels, "5a", "")            # a fair loss carries no reason
        RL.record_strike(duels, "5a", "lost")
        assert duels["5a"]["strikes"] == 0

    @pytest.mark.parametrize("reason", sorted(RL.STRIKEABLE))
    def test_gate_rejections_ARE_strikes(self, reason):
        """These all mean "you should never have been dispatched"."""
        duels = {}
        RL.record_strike(duels, "5a", reason)
        assert duels["5a"]["strikes"] == 1

    def test_repeat_offenders_stop_being_scanned(self):
        duels = {}
        for _ in range(RL.MAX_STRIKES):
            RL.record_strike(duels, "5spam", "copy_of_king")
        assert "5spam" in RL.struck_out(duels)

    def test_an_honest_loser_is_never_struck_out(self):
        duels = {}
        for block in range(0, 10_000, RL.COOLDOWN_BLOCKS):
            RL.record_verdict(duels, "5honest", king={"reign_number": 1}, block=block)
            RL.record_strike(duels, "5honest", "")   # lost, fairly, every time
        assert RL.struck_out(duels) == set()


class TestPersistence:
    async def test_the_duel_budget_survives_a_restart(self):
        store = _store()
        st = _state()
        RL.record_verdict(st.duels, "5a", king=st.king, block=1000)
        st.touch()
        await st.flush(store)

        revived = await KingState.load(store)
        assert revived.duels["5a"]["last_verdict_block"] == 1000

    async def test_a_bucket_written_before_this_feature_still_loads(self):
        """Live validators have existing state.json objects with no `duels` key. They
        must load, not crash — a migration that bricks the fleet is not a migration."""
        store = _store()
        st = KingState()
        st.king = {"hotkey": "5KING", "model_digest": KING_DIGEST}
        st.touch()
        await st.flush(store)

        doc = await store.get("state/state.json")
        doc.pop("duels", None)                      # the old shape, verbatim
        await store.put("state/state.json", doc)

        revived = await KingState.load(store)
        assert revived.duels == {}
        assert revived.king["hotkey"] == "5KING"
