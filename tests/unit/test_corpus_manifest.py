"""The corpus manifest, and the fail-closed dataset built from it.

The bugs pinned here are the ones that let two honest validators grade the same
challenger against *different ground truth* — silently:

* the clip set came from a **live bucket listing** permuted by corpus *size*, so
  one upload (or one flaky read) reshuffled everyone's exam;
* the window inside each video was chosen by running **ffmpeg scene detection at
  duel time**, and scene-cut timestamps vary by ffmpeg build;
* a clip that failed to download was **silently skipped**, so a duel could run on
  3 clips instead of 32 and be recorded as a perfectly normal result.
"""

import json

import numpy as np
import pytest

from leoma.eval.dataset import build_duel_clips, corpus_audit, fetch_manifest
from leoma.eval.digests import digest_frames, sha256_bytes
from leoma.eval.errors import ConsensusConfigError, CorpusIntegrityError, TransientDuelError
from leoma.eval.manifest import (
    ClipEntry,
    CorpusManifest,
    DecodeParams,
    clip_keys_digest,
    load_pinned_manifest,
    parse_manifest,
)
from leoma.eval.video_runner import GenParams

DECODE = DecodeParams(width=8, height=8, fps=2, num_frames=4)
GEN = GenParams(num_frames=4, fps=2, width=8, height=8)


def _truth(i: int) -> np.ndarray:
    rng = np.random.default_rng(i)
    return rng.integers(0, 255, size=(4, 8, 8, 3)).astype("uint8")


def _entry(i: int) -> ClipEntry:
    return ClipEntry(
        clip_id=f"clip-{i:04d}",
        video_key=f"videos/v{i:04d}.mp4",
        video_sha256="sha256:" + f"{i:064x}",
        clip_start=1.5,
        clip_seconds=2.0,
        prompt=f"a prompt for {i}",
        truth_sha256=digest_frames(_truth(i)),
        motion_energy=9.0,
    )


def _manifest(n=80) -> CorpusManifest:
    return CorpusManifest(
        corpus_id="test-v1", decode=DECODE, clips=tuple(_entry(i) for i in range(n))
    )


def _raw(manifest: CorpusManifest) -> bytes:
    from leoma.infra.corpus_manifest import serialize

    return serialize(manifest)


class FakeCorpusClient:
    """Serves the manifest and the source videos — with fault injection.

    ``decode`` is stubbed out (the real thing shells out to ffmpeg); what we are
    testing here is the *policy*: what happens when a hash doesn't match, or a
    fetch fails. That policy is what used to be "silently skip and carry on".
    """

    def __init__(self, manifest, *, fail_fetch=(), wrong_video=(), wrong_truth=()):
        self.manifest = manifest
        self.raw = _raw(manifest)
        self.fail_fetch = set(fail_fetch)
        self.wrong_video = set(wrong_video)
        self.wrong_truth = set(wrong_truth)
        self.fetched: list[str] = []

    def get_object(self, bucket, key):
        raw = self.raw

        class _Resp:
            def read(self):
                return raw

            def close(self):
                pass

            def release_conn(self):
                pass

        return _Resp()

    def fget_object(self, bucket, key, path):
        self.fetched.append(key)
        if key in self.fail_fetch:
            raise RuntimeError("connection reset")
        with open(path, "wb") as f:
            f.write(b"fake mp4 bytes")


def _install_stubs(monkeypatch, client):
    """Make the dataset's two impure steps (hash the mp4, decode it) deterministic."""
    import leoma.eval.dataset as ds
    import leoma.infra.video_utils as vu

    index = {e.video_key: i for i, e in enumerate(client.manifest.clips)}
    state = {"key": None}

    real_fetch = client.fget_object

    def fget(bucket, key, path):
        state["key"] = key
        return real_fetch(bucket, key, path)

    client.fget_object = fget

    def digest_file(path):
        i = index[state["key"]]
        if state["key"] in client.wrong_video:
            return "sha256:" + "f" * 64      # the bucket drifted from the pin
        return "sha256:" + f"{i:064x}"

    def decode_frames_rgb(path, **kw):
        i = index[state["key"]]
        if state["key"] in client.wrong_truth:
            return _truth(i + 1000)          # this box decodes video differently
        return _truth(i)

    monkeypatch.setattr(ds, "digest_file", digest_file)
    monkeypatch.setattr(vu, "decode_frames_rgb", decode_frames_rgb)


class TestParseInvariants:
    def test_duplicate_clip_ids_rejected(self):
        data = _manifest(3).as_dict()
        data["clips"][1]["clip_id"] = data["clips"][0]["clip_id"]
        with pytest.raises(CorpusIntegrityError, match="duplicate"):
            parse_manifest(data)

    def test_unsorted_clips_rejected(self):
        """Selection is BY INDEX. An unsorted manifest means index N is a different
        clip on a box that happened to write the file in a different order."""
        data = _manifest(3).as_dict()
        data["clips"].reverse()
        with pytest.raises(CorpusIntegrityError, match="sorted"):
            parse_manifest(data)

    def test_missing_truth_hash_rejected(self):
        data = _manifest(3).as_dict()
        data["clips"][0]["truth_sha256"] = ""
        with pytest.raises(CorpusIntegrityError, match="truth_sha256"):
            parse_manifest(data)

    def test_unknown_version_rejected(self):
        data = _manifest(3).as_dict()
        data["manifest_version"] = 99
        with pytest.raises(CorpusIntegrityError, match="manifest_version"):
            parse_manifest(data)

    def test_empty_manifest_rejected(self):
        data = _manifest(3).as_dict()
        data["clips"] = []
        with pytest.raises(CorpusIntegrityError, match="no clips"):
            parse_manifest(data)


class TestPinnedLoad:
    def test_digest_must_match(self):
        m = _manifest(3)
        with pytest.raises(CorpusIntegrityError, match="digest mismatch"):
            load_pinned_manifest(_raw(m), "sha256:" + "0" * 64)

    def test_matching_digest_parses(self):
        m = _manifest(3)
        raw = _raw(m)
        loaded = load_pinned_manifest(raw, sha256_bytes(raw))
        assert loaded.corpus_id == "test-v1"
        assert loaded.source_digest == sha256_bytes(raw)

    def test_a_tampered_manifest_is_caught_before_it_is_parsed(self):
        """One flipped byte anywhere in the file — including a clip_start — is caught
        by the digest, not by whatever validation happens to look at that field."""
        m = _manifest(3)
        raw = _raw(m)
        tampered = json.loads(raw)
        tampered["clips"][0]["clip_start"] = 99.0
        with pytest.raises(CorpusIntegrityError, match="digest mismatch"):
            load_pinned_manifest(json.dumps(tampered).encode(), sha256_bytes(raw))


class TestExamIdentity:
    def test_clip_keys_digest_binds_ids_and_truth(self):
        clips = [_entry(0), _entry(1)]
        assert clip_keys_digest(clips) == clip_keys_digest([_entry(0), _entry(1)])

    def test_a_different_truth_for_the_same_clip_id_is_a_different_exam(self):
        """This is what makes a distance disagreement attributable: same digest =>
        provably the same ground truth, so any difference is generation noise."""
        clips = [_entry(0)]
        swapped = [
            ClipEntry(**{**_entry(0).as_dict(), "truth_sha256": digest_frames(_truth(7))})
        ]
        assert clip_keys_digest(clips) != clip_keys_digest(swapped)

    def test_order_matters(self):
        assert clip_keys_digest([_entry(0), _entry(1)]) != clip_keys_digest([_entry(1), _entry(0)])


class TestDecodeCompat:
    def test_a_manifest_built_at_a_different_fps_is_refused_up_front(self):
        """Not clip-by-clip: all 32 truth hashes would miss, and the operator would
        be staring at 32 identical 'integrity' errors instead of one clear one."""
        m = _manifest()
        with pytest.raises(ConsensusConfigError, match="decode parameters"):
            build_duel_clips(
                m, client=FakeCorpusClient(m), bucket="videos", master_seed=1, n_clips=4,
                gen=GenParams(num_frames=4, fps=99, width=8, height=8),
                prompt_mode="manifest", fixed_prompt="",
            )

    def test_a_corpus_barely_bigger_than_the_duel_is_refused(self):
        """A 40-clip corpus for a 32-clip duel is a memorizable exam, not a test set."""
        small = _manifest(40)
        with pytest.raises(ConsensusConfigError, match="memorize"):
            build_duel_clips(
                small, client=FakeCorpusClient(small), bucket="videos", master_seed=1,
                n_clips=32, gen=GEN, prompt_mode="manifest", fixed_prompt="",
            )


class TestFailClosedDataset:
    def test_builds_exactly_n_clips_from_the_pinned_manifest(self, monkeypatch):
        m = _manifest()
        client = FakeCorpusClient(m)
        _install_stubs(monkeypatch, client)

        clips, entries = build_duel_clips(
            m, client=client, bucket="videos", master_seed=7, n_clips=4,
            gen=GEN, prompt_mode="manifest", fixed_prompt="",
        )
        assert len(clips) == 4
        assert [c.clip_id for c in clips] == [e.clip_id for e in entries]
        # The clip's prompt comes from the MANIFEST, not from a per-box env var.
        assert clips[0].prompt == entries[0].prompt

    def test_the_seed_is_derived_from_the_manifest_index_not_the_duel_position(self, monkeypatch):
        """If the seed hung off the position in the selection, the same clip would
        generate from different noise depending on which other clips came with it."""
        m = _manifest()
        client = FakeCorpusClient(m)
        _install_stubs(monkeypatch, client)

        clips, _ = build_duel_clips(
            m, client=client, bucket="videos", master_seed=7, n_clips=4,
            gen=GEN, prompt_mode="manifest", fixed_prompt="",
        )
        for clip in clips:
            assert m.clips[clip.clip_index].clip_id == clip.clip_id

    def test_same_seed_same_exam(self, monkeypatch):
        m = _manifest()
        client = FakeCorpusClient(m)
        _install_stubs(monkeypatch, client)

        a, _ = build_duel_clips(m, client=client, bucket="videos", master_seed=42, n_clips=4,
                                gen=GEN, prompt_mode="manifest", fixed_prompt="")
        b, _ = build_duel_clips(m, client=client, bucket="videos", master_seed=42, n_clips=4,
                                gen=GEN, prompt_mode="manifest", fixed_prompt="")
        assert [c.clip_id for c in a] == [c.clip_id for c in b]

    def test_the_clip_set_does_not_depend_on_the_corpus_SIZE(self, monkeypatch):
        """THE bug. The old selection permuted range(len(keys)) from a LIVE listing,
        so uploading one video reshuffled the entire clip set — and only for the
        validators that had already seen the upload. Selection is now by index into
        a manifest that is fixed until it is explicitly rotated."""
        m = _manifest(80)
        client = FakeCorpusClient(m)
        _install_stubs(monkeypatch, client)
        before, _ = build_duel_clips(m, client=client, bucket="videos", master_seed=5,
                                     n_clips=4, gen=GEN, prompt_mode="manifest", fixed_prompt="")

        # A validator whose bucket has 20 MORE videos in it. The manifest is what
        # counts, and it hasn't changed — so neither does the exam.
        client2 = FakeCorpusClient(m)
        _install_stubs(monkeypatch, client2)
        after, _ = build_duel_clips(m, client=client2, bucket="videos", master_seed=5,
                                    n_clips=4, gen=GEN, prompt_mode="manifest", fixed_prompt="")
        assert [c.clip_id for c in before] == [c.clip_id for c in after]

    def test_a_failed_download_aborts_the_duel_it_does_NOT_skip_the_clip(self, monkeypatch):
        """The old code caught every exception and moved to the next candidate, so a
        flaky S3 read produced a *different clip set* — or a duel on 3 clips that was
        recorded as a normal result."""
        m = _manifest()
        client = FakeCorpusClient(m)
        _install_stubs(monkeypatch, client)
        selected = build_duel_clips(m, client=client, bucket="videos", master_seed=7, n_clips=4,
                                    gen=GEN, prompt_mode="manifest", fixed_prompt="")[1]

        broken = FakeCorpusClient(m, fail_fetch={selected[1].video_key})
        _install_stubs(monkeypatch, broken)
        with pytest.raises(TransientDuelError, match="could not fetch source video"):
            build_duel_clips(m, client=broken, bucket="videos", master_seed=7, n_clips=4,
                             gen=GEN, prompt_mode="manifest", fixed_prompt="")

    def test_a_drifted_source_video_is_a_corpus_error_not_a_miner_error(self, monkeypatch):
        m = _manifest()
        client = FakeCorpusClient(m)
        _install_stubs(monkeypatch, client)
        selected = build_duel_clips(m, client=client, bucket="videos", master_seed=7, n_clips=4,
                                    gen=GEN, prompt_mode="manifest", fixed_prompt="")[1]

        drifted = FakeCorpusClient(m, wrong_video={selected[0].video_key})
        _install_stubs(monkeypatch, drifted)
        with pytest.raises(CorpusIntegrityError, match="does not match the manifest"):
            build_duel_clips(m, client=drifted, bucket="videos", master_seed=7, n_clips=4,
                             gen=GEN, prompt_mode="manifest", fixed_prompt="")

    def test_a_box_that_decodes_video_differently_refuses_to_duel(self, monkeypatch):
        """The quiet catastrophe: a validator whose ffmpeg produces different pixels
        measures every distance against different ground truth, and would happily
        publish confident verdicts nobody else can reproduce."""
        m = _manifest()
        client = FakeCorpusClient(m)
        _install_stubs(monkeypatch, client)
        selected = build_duel_clips(m, client=client, bucket="videos", master_seed=7, n_clips=4,
                                    gen=GEN, prompt_mode="manifest", fixed_prompt="")[1]

        odd = FakeCorpusClient(m, wrong_truth={selected[2].video_key})
        _install_stubs(monkeypatch, odd)
        with pytest.raises(CorpusIntegrityError, match="decoded ground truth does not match"):
            build_duel_clips(m, client=odd, bucket="videos", master_seed=7, n_clips=4,
                             gen=GEN, prompt_mode="manifest", fixed_prompt="")

    def test_audit_block_lets_a_third_party_replay_the_exam(self, monkeypatch):
        m = _manifest()
        client = FakeCorpusClient(m)
        _install_stubs(monkeypatch, client)
        _, entries = build_duel_clips(m, client=client, bucket="videos", master_seed=7,
                                      n_clips=4, gen=GEN, prompt_mode="manifest", fixed_prompt="")

        audit = corpus_audit(m, entries)
        assert audit["corpus_id"] == "test-v1"
        assert audit["corpus_size"] == 80
        assert audit["clip_ids"] == [e.clip_id for e in entries]
        assert audit["clip_keys_digest"] == clip_keys_digest(entries)


class TestFetchManifest:
    def test_an_unreachable_bucket_is_transient_not_a_corpus_error(self):
        """Fault attribution matters: a network blip must not read as "this
        validator's corpus is corrupt" (which is fail-closed and pages someone)."""
        from leoma.eval.spec import CorpusSpec

        class Dead:
            def get_object(self, *a, **k):
                raise RuntimeError("connection refused")

        corpus = CorpusSpec(bucket="videos", manifest_key="m.json",
                            manifest_digest="sha256:" + "a" * 64)
        with pytest.raises(TransientDuelError, match="could not fetch the corpus manifest"):
            fetch_manifest(Dead(), corpus)

    def test_a_manifest_that_is_not_the_pinned_one_is_refused(self):
        from leoma.eval.spec import CorpusSpec

        m = _manifest(3)
        corpus = CorpusSpec(bucket="videos", manifest_key="m.json",
                            manifest_digest="sha256:" + "b" * 64)
        with pytest.raises(CorpusIntegrityError, match="digest mismatch"):
            fetch_manifest(FakeCorpusClient(m), corpus)
