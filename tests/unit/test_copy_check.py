"""OCI-layer copy detection + earliest-author displacement.

The free exact-digest check (`main._copies_a_king`) catches the crude copy — a
different hotkey re-committing the king's *exact* digest. It misses the copy that
matters: identical WEIGHTS repackaged with a changed README or tokenizer, which
yields a new top-level digest but the SAME per-layer safetensor digests. On a
content-addressed registry that is the king's weights in a disguise, and under the
old gate it burned a full multi-hour duel before tying and losing.

`check_model_copy` closes that for one manifest fetch (a few KB, no weight download),
and while it is at it, displaces the king with a byte-identical model that was pushed
to the registry EARLIER — the true original, front-run by whoever got crowned first.
"""

import leoma.app.validator.main as vmain
from leoma.app.validator.copy_check import _parse_registry_timestamp, check_model_copy
from leoma.app.validator.reveal_scan import ChallengerEntry
from leoma.app.validator.state_store import JsonBucketStore, KingState

from tests.unit.conftest import FakeEvalBox, FakeMinio

KING_REPO = "u/leoma-king"
KING_DIGEST = "sha256:" + "a" * 64
LAYERS = {"transformer/model-00001.safetensors": "sha256:aaa",
          "vae/model-00001.safetensors": "sha256:bbb"}


def _info(layers, *, ts, src):
    return {"safetensor_layers": dict(layers), "committed_at": ts, "timestamp_source": src}


def _fetcher(mapping):
    """Build a fetch(ref)->info stub keyed by digest."""
    return lambda ref: mapping.get(ref.digest)


class TestDecision:
    def test_genuinely_different_weights_are_not_a_copy(self):
        other = {"transformer/model-00001.safetensors": "sha256:XXX",
                 "vae/model-00001.safetensors": "sha256:bbb"}
        fetch = _fetcher({
            KING_DIGEST: _info(LAYERS, ts="2026-01-01T00:00:00Z", src="harbor_artifact.push_time"),
            "sha256:" + "c" * 64: _info(other, ts="2026-02-01T00:00:00Z", src="harbor_artifact.push_time"),
        })
        assert check_model_copy("u/c", "sha256:" + "c" * 64, KING_REPO, KING_DIGEST, fetch=fetch) is None

    def test_a_different_layer_COUNT_is_not_a_copy(self):
        fewer = {"transformer/model-00001.safetensors": "sha256:aaa"}
        fetch = _fetcher({
            KING_DIGEST: _info(LAYERS, ts="2026-01-01T00:00:00Z", src="harbor_artifact.push_time"),
            "sha256:" + "c" * 64: _info(fewer, ts="2026-02-01T00:00:00Z", src="harbor_artifact.push_time"),
        })
        assert check_model_copy("u/c", "sha256:" + "c" * 64, KING_REPO, KING_DIGEST, fetch=fetch) is None

    def test_identical_layers_committed_LATER_is_rejected(self):
        """The repackaged copy the exact-digest check misses: same weights, new top
        digest, pushed after the king."""
        c = "sha256:" + "c" * 64
        fetch = _fetcher({
            KING_DIGEST: _info(LAYERS, ts="2026-01-01T00:00:00Z", src="harbor_artifact.push_time"),
            c: _info(LAYERS, ts="2026-02-01T00:00:00Z", src="harbor_artifact.push_time"),
        })
        v = check_model_copy("u/c", c, KING_REPO, KING_DIGEST, fetch=fetch)
        assert v["action"] == "reject"
        assert "identical OCI digests" in v["reason"]

    def test_identical_layers_committed_EARLIER_displaces_the_king(self):
        """The original author, front-run: byte-identical weights, earlier push."""
        c = "sha256:" + "c" * 64
        fetch = _fetcher({
            KING_DIGEST: _info(LAYERS, ts="2026-02-01T00:00:00Z", src="harbor_artifact.push_time"),
            c: _info(LAYERS, ts="2026-01-01T00:00:00Z", src="harbor_artifact.push_time"),  # earlier
        })
        v = check_model_copy("u/c", c, KING_REPO, KING_DIGEST, fetch=fetch)
        assert v["action"] == "crown_earlier"
        assert "original author" in v["reason"]

    def test_the_exact_reigning_king_is_rejected_without_a_fetch(self):
        called = []
        fetch = lambda ref: called.append(ref) or None
        v = check_model_copy(KING_REPO, KING_DIGEST, KING_REPO, KING_DIGEST, fetch=fetch)
        assert v["action"] == "reject"
        assert called == [], "no metadata fetch needed for an exact re-commit"


class TestFailSafeAndFailOpen:
    def test_missing_timestamps_REJECT_never_crown(self):
        """Fail-safe: identical weights but no registry timestamp => reject. Turning a
        real author away is the safe direction; enthroning a plagiarist on backdated
        metadata is not."""
        c = "sha256:" + "c" * 64
        fetch = _fetcher({
            KING_DIGEST: _info(LAYERS, ts=None, src=None),
            c: _info(LAYERS, ts=None, src=None),
        })
        v = check_model_copy("u/c", c, KING_REPO, KING_DIGEST, fetch=fetch)
        assert v["action"] == "reject"
        assert "timestamps unavailable" in v["reason"]

    def test_a_client_supplied_only_timestamp_does_not_earn_crown_earlier(self):
        """If the earlier side has no registry source, it cannot displace — even though
        its timestamp string is earlier. (The production fetcher never populates a
        client annotation as committed_at; this guards the decision logic regardless.)"""
        c = "sha256:" + "c" * 64
        fetch = _fetcher({
            KING_DIGEST: _info(LAYERS, ts="2026-02-01T00:00:00Z", src="harbor_artifact.push_time"),
            c: _info(LAYERS, ts="2026-01-01T00:00:00Z", src=None),  # earlier but no source
        })
        v = check_model_copy("u/c", c, KING_REPO, KING_DIGEST, fetch=fetch)
        assert v["action"] == "reject"

    def test_a_metadata_hiccup_fails_OPEN(self):
        """Fetch returns None (registry unreachable) => the check returns None => the
        caller lets the model duel. A registry blip must never block a valid submission."""
        fetch = lambda ref: None
        assert check_model_copy("u/c", "sha256:" + "c" * 64, KING_REPO, KING_DIGEST, fetch=fetch) is None

    def test_no_king_means_nothing_to_copy(self):
        assert check_model_copy("u/c", "sha256:" + "c" * 64, "", "", fetch=lambda r: None) is None


class TestTimestampParsing:
    def test_iso_8601_with_z(self):
        assert _parse_registry_timestamp("2026-01-01T00:00:00Z") is not None

    def test_rfc_2822_last_modified(self):
        # The manifest Last-Modified header form.
        assert _parse_registry_timestamp("Wed, 01 Jan 2026 00:00:00 GMT") is not None

    def test_earlier_really_compares_earlier_across_formats(self):
        iso = _parse_registry_timestamp("2026-01-01T00:00:00Z")
        rfc = _parse_registry_timestamp("Wed, 01 Feb 2026 00:00:00 GMT")
        assert iso < rfc

    def test_garbage_is_none(self):
        assert _parse_registry_timestamp("not a date") is None
        assert _parse_registry_timestamp(None) is None


class TestWiredIntoDispatch:
    async def test_a_repackaged_copy_never_reaches_the_GPU(self, monkeypatch, duel_ready):
        """The whole point: a copy that changed only its README costs one manifest
        fetch, not a multi-hour duel."""
        monkeypatch.setattr(vmain, "COPY_CHECK_ENABLED", True)
        c = "sha256:" + "c" * 64
        monkeypatch.setattr(vmain, "check_model_copy",
                            lambda cr, cd, kr, kd: {"action": "reject", "reason": "repackaged king"})

        box = FakeEvalBox(monkeypatch, lambda e: AssertionError("must not dispatch"), duel_ready)
        st = KingState()
        st.king = {"hotkey": "5KING", "model_repo": KING_REPO, "model_digest": KING_DIGEST, "reign_number": 1}
        store = JsonBucketStore(FakeMinio(), "own", backoff=0)
        entry = ChallengerEntry(hotkey="5thief", model_repo="u/leoma-copy", model_digest=c, block=100)

        await box.drive(st, store, [entry], block=200, ticks=1)

        assert box.dispatched == []
        key = vmain._seen_key(entry.hotkey, entry.model_digest)
        assert st.attempts[key]["last_reason"] == "copy_of_king"
        assert st.duels["5thief"]["strikes"] == 1

    async def test_an_earlier_author_takes_the_crown_with_no_duel(self, monkeypatch, duel_ready):
        monkeypatch.setattr(vmain, "COPY_CHECK_ENABLED", True)
        c = "sha256:" + "c" * 64
        monkeypatch.setattr(vmain, "check_model_copy", lambda cr, cd, kr, kd: {
            "action": "crown_earlier", "reason": "earlier push",
            "challenger_committed_at": "2026-01-01T00:00:00Z", "king_committed_at": "2026-02-01T00:00:00Z",
        })

        box = FakeEvalBox(monkeypatch, lambda e: AssertionError("must not duel a proven copy"), duel_ready)
        st = KingState()
        st.king = {"hotkey": "5KING", "model_repo": KING_REPO, "model_digest": KING_DIGEST, "reign_number": 1}
        store = JsonBucketStore(FakeMinio(), "own", backoff=0)
        entry = ChallengerEntry(hotkey="5author", model_repo="u/leoma-orig", model_digest=c, block=100)

        await box.drive(st, store, [entry], block=200, ticks=1)

        assert box.dispatched == []                       # no duel
        assert st.king["hotkey"] == "5author"             # the original author reigns
        assert st.king["model_digest"] == c
        assert st.king["reign_number"] == 2               # a genuine dethrone
        assert st.history[0]["verdict"] == "crown_earlier"
        assert st.stats["accepted"] == 1

    async def test_a_metadata_hiccup_lets_the_model_duel(self, monkeypatch, duel_ready):
        """Fail-open at the call site too: None => proceed to the normal duel."""
        monkeypatch.setattr(vmain, "COPY_CHECK_ENABLED", True)
        monkeypatch.setattr(vmain, "check_model_copy", lambda *a: None)

        box = FakeEvalBox(monkeypatch, lambda e: {"status": "running"}, duel_ready)
        st = KingState()
        st.king = {"hotkey": "5KING", "model_repo": KING_REPO, "model_digest": KING_DIGEST, "reign_number": 1}
        store = JsonBucketStore(FakeMinio(), "own", backoff=0)
        entry = ChallengerEntry(hotkey="5new", model_repo="u/leoma-new",
                                model_digest="sha256:" + "n" * 64, block=100)

        await box.drive(st, store, [entry], block=200, ticks=1)
        assert box.dispatched == ["5new"]

    async def test_the_check_crashing_does_not_crash_the_tick(self, monkeypatch, duel_ready):
        monkeypatch.setattr(vmain, "COPY_CHECK_ENABLED", True)

        def boom(*a):
            raise RuntimeError("registry exploded")

        monkeypatch.setattr(vmain, "check_model_copy", boom)
        box = FakeEvalBox(monkeypatch, lambda e: {"status": "running"}, duel_ready)
        st = KingState()
        st.king = {"hotkey": "5KING", "model_repo": KING_REPO, "model_digest": KING_DIGEST, "reign_number": 1}
        store = JsonBucketStore(FakeMinio(), "own", backoff=0)
        entry = ChallengerEntry(hotkey="5new", model_repo="u/leoma-new",
                                model_digest="sha256:" + "n" * 64, block=100)

        await box.drive(st, store, [entry], block=200, ticks=1)   # must not raise
        assert box.dispatched == ["5new"]                          # failed open
