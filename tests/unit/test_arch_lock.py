"""The architecture lock, and the prescreen that runs it before the GPU is touched.

The reference subnet locks a flat transformers `config.json`. **A diffusers snapshot
has no root config.json at all** — it has `model_index.json` (a component→class map)
plus a separate config inside each component directory, and the shape-critical numbers
live in `transformer/config.json` and `vae/config.json`. So the lock is nested and
per-component, and it validates by **diffing against the base repo's own configs**
rather than against numbers hand-copied into chain.toml (which rot the moment the
pinned base is bumped).

The prescreen runs it on a **config-only fetch** (~200 KB), on the validator, before
dispatch. A wrong-architecture model costs ~5 seconds instead of hours of GPU with the
eval server's lock held.
"""

import json

import pytest

import leoma.app.validator.main as vmain
from leoma.app.validator import prescreen as ps
from leoma.app.validator.failures import ErrorClass, classify
from leoma.app.validator.reveal_scan import ChallengerEntry
from leoma.app.validator.state_store import JsonBucketStore, KingState
from leoma.eval.arch_lock import (
    ArchMismatch,
    check_size,
    load_model_index,
    validate,
)

from tests.unit.conftest import FakeEvalBox, FakeMinio

PIPELINE = "WanImageToVideoPipeline"

BASE_INDEX = {
    "_class_name": PIPELINE,
    "_diffusers_version": "0.31.0",
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
}
BASE_TRANSFORMER = {
    "num_layers": 40, "num_attention_heads": 40, "attention_head_dim": 128,
    "in_channels": 36, "out_channels": 16, "patch_size": [1, 2, 2],
    "freq_dim": 256, "ffn_dim": 13824, "text_dim": 4096,
}
BASE_VAE = {"latent_channels": 16, "z_dim": 16, "base_dim": 96, "dim_mult": [1, 2, 4, 4]}


def _snapshot(root, *, index=None, transformer=None, vae=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps(index or BASE_INDEX))
    for name, config in (
        ("transformer", transformer if transformer is not None else BASE_TRANSFORMER),
        ("vae", vae if vae is not None else BASE_VAE),
    ):
        sub = root / name
        sub.mkdir(exist_ok=True)
        (sub / "config.json").write_text(json.dumps(config))
    return root


@pytest.fixture
def base(tmp_path):
    return _snapshot(tmp_path / "base")


class TestPipelineClass:
    def test_the_pinned_pipeline_passes(self, tmp_path, base):
        good = _snapshot(tmp_path / "good")
        assert validate(good, base, pipeline=PIPELINE)["pipeline"] == PIPELINE

    def test_a_different_pipeline_is_rejected(self, tmp_path, base):
        other = _snapshot(tmp_path / "other",
                          index={**BASE_INDEX, "_class_name": "StableVideoDiffusionPipeline"})
        with pytest.raises(ArchMismatch, match="pinned to"):
            validate(other, base, pipeline=PIPELINE)

    def test_a_transformers_style_repo_is_rejected_with_a_useful_message(self, tmp_path, base):
        """The most likely honest mistake: a miner uploads a transformers checkpoint.
        A bare 'model_index.json missing' would leave them guessing."""
        wrong = tmp_path / "transformers-style"
        wrong.mkdir()
        (wrong / "config.json").write_text(json.dumps({"model_type": "llama"}))

        with pytest.raises(ArchMismatch, match="root config.json is NOT what this subnet expects"):
            validate(wrong, base, pipeline=PIPELINE)


class TestComponents:
    def test_disabled_optional_components_are_not_executable_components(self, tmp_path):
        index = {
            **BASE_INDEX,
            "image_encoder": [None, None],
            "image_processor": [None, None],
        }
        base = _snapshot(tmp_path / "disabled-base", index=index)
        mine = _snapshot(tmp_path / "disabled-mine", index=index)

        result = validate(mine, base, pipeline=PIPELINE)

        assert "image_encoder" not in result["components"]
        assert "image_processor" not in result["components"]

    def test_a_missing_component_is_rejected(self, tmp_path, base):
        index = {k: v for k, v in BASE_INDEX.items() if k != "vae"}
        snap = _snapshot(tmp_path / "novae", index=index)
        with pytest.raises(ArchMismatch, match="missing pipeline components: vae"):
            validate(snap, base, pipeline=PIPELINE)

    def test_an_EXTRA_component_is_rejected(self, tmp_path, base):
        """Not pedantry: an unexpected component is a way to smuggle code onto the box."""
        index = {**BASE_INDEX, "custom_module": ["diffusers", "SomethingElse"]}
        snap = _snapshot(tmp_path / "extra", index=index)
        with pytest.raises(ArchMismatch, match="unexpected pipeline components"):
            validate(snap, base, pipeline=PIPELINE)

    def test_a_swapped_component_class_is_rejected(self, tmp_path, base):
        index = {**BASE_INDEX, "vae": ["diffusers", "AutoencoderKL"]}   # not the Wan VAE
        snap = _snapshot(tmp_path / "swapped", index=index)
        with pytest.raises(ArchMismatch, match="pinned architecture uses"):
            validate(snap, base, pipeline=PIPELINE)

    def test_an_exotic_library_is_rejected(self, tmp_path, base):
        index = {**BASE_INDEX, "vae": ["evil_pkg", "AutoencoderKLWan"]}
        snap = _snapshot(tmp_path / "exotic", index=index)
        with pytest.raises(ArchMismatch, match="only.*are allowed"):
            validate(snap, base, pipeline=PIPELINE)


class TestLockedKeys:
    @pytest.mark.parametrize("key,bad", [
        ("num_layers", 28),
        ("num_attention_heads", 16),
        ("attention_head_dim", 64),
        ("in_channels", 4),
    ])
    def test_a_changed_transformer_shape_is_rejected(self, tmp_path, base, key, bad):
        snap = _snapshot(tmp_path / "resized", transformer={**BASE_TRANSFORMER, key: bad})
        with pytest.raises(ArchMismatch, match=f"transformer/config.json {key}"):
            validate(snap, base, pipeline=PIPELINE)

    def test_a_changed_vae_shape_is_rejected(self, tmp_path, base):
        snap = _snapshot(tmp_path / "vae", vae={**BASE_VAE, "latent_channels": 4})
        with pytest.raises(ArchMismatch, match="vae/config.json latent_channels"):
            validate(snap, base, pipeline=PIPELINE)

    def test_the_message_tells_the_miner_exactly_what_to_fix(self, tmp_path, base):
        snap = _snapshot(tmp_path / "x", transformer={**BASE_TRANSFORMER, "num_layers": 28})
        with pytest.raises(ArchMismatch, match=r"num_layers=28 but the pinned architecture locks it to 40"):
            validate(snap, base, pipeline=PIPELINE)

    def test_a_key_the_BASE_does_not_define_is_not_a_constraint(self, tmp_path, base):
        """Otherwise the lock breaks every time diffusers adds a config field — it
        would start rejecting the very architecture it is meant to enforce."""
        snap = _snapshot(tmp_path / "extrakey",
                         transformer={**BASE_TRANSFORMER, "some_new_diffusers_field": 7})
        validate(snap, base, pipeline=PIPELINE)   # does not raise

    def test_unrelated_finetuning_metadata_is_allowed(self, tmp_path, base):
        """Miners must be free to fine-tune. Only SHAPE is locked."""
        snap = _snapshot(tmp_path / "ft",
                         transformer={**BASE_TRANSFORMER, "_name_or_path": "my-finetune-v3"})
        validate(snap, base, pipeline=PIPELINE)

    def test_wan22_second_transformer_is_locked(self, tmp_path):
        index = {**BASE_INDEX, "transformer_2": ["diffusers", "WanTransformer3DModel"]}
        good = _snapshot(tmp_path / "dual-base", index=index)
        second = good / "transformer_2"
        second.mkdir()
        second.joinpath("config.json").write_text(json.dumps(BASE_TRANSFORMER))

        bad = _snapshot(tmp_path / "dual-bad", index=index)
        bad_second = bad / "transformer_2"
        bad_second.mkdir()
        bad_second.joinpath("config.json").write_text(
            json.dumps({**BASE_TRANSFORMER, "num_layers": 39})
        )

        with pytest.raises(ArchMismatch, match="transformer_2/config.json num_layers=39"):
            validate(bad, good, pipeline=PIPELINE)


class TestSizeBound:
    def test_a_stub_repo_is_rejected(self):
        with pytest.raises(ArchMismatch, match="stub repo"):
            check_size(1024)

    def test_a_disk_filling_repo_is_rejected(self):
        with pytest.raises(ArchMismatch, match="fill the eval box's disk"):
            check_size(500 * 1024**3)

    def test_a_plausible_14B_model_passes(self):
        check_size(30 * 1024**3)


class TestFaultAttribution:
    def test_prescreen_base_ref_uses_the_consensus_pinned_digest(self):
        ref = ps._base_ref()
        assert ref is not None
        assert ref.repo == ps.SPEC.arch.base_repo
        assert ref.digest == ps.SPEC.arch.base_digest

    def test_an_arch_mismatch_is_PERMANENT_so_it_is_quarantined(self):
        """The artifact is immutable: a wrong-architecture model will be wrong forever.
        Retrying it four times would just waste four prescreens."""
        assert classify(ArchMismatch("wrong shape")).kind is ErrorClass.PERMANENT
        assert classify(ArchMismatch("wrong shape")).reason == "arch_mismatch"

    def test_a_base_repo_we_cannot_fetch_is_OUR_problem_not_the_miners(self, tmp_path, monkeypatch):
        """Fail-open on infrastructure. If the prescreen can't run, that says nothing
        about the challenger — and quarantining every miner because our Hub token
        expired would be catastrophic."""
        from leoma.eval.errors import TransientDuelError

        monkeypatch.setattr(ps, "materialize_model", lambda *a, **k: str(tmp_path))
        monkeypatch.setattr(ps, "_base_ref", lambda: None)

        with pytest.raises(TransientDuelError):
            ps.prescreen("u/leoma-x", "sha256:" + "a" * 64)


class TestPrescreenIsWiredIn:
    async def test_a_wrong_arch_model_never_reaches_the_GPU(self, monkeypatch, duel_ready):
        """THE point of the prescreen. Without it, this model costs a full dispatch:
        tens of GB downloaded, two 14B pipelines loaded, the eval lock held the whole
        time — and only THEN does anyone discover it was never loadable."""
        monkeypatch.setattr(vmain, "PRESCREEN_ENABLED", True)

        def reject(repo, digest):
            raise ArchMismatch("transformer/config.json num_layers=28 (locked to 40)")

        monkeypatch.setattr(vmain, "prescreen", reject)

        box = FakeEvalBox(monkeypatch, lambda e: AssertionError("must not dispatch"), duel_ready)
        st = KingState()
        st.king = {"hotkey": "5KING", "model_repo": "u/leoma-king",
                   "model_digest": "sha256:" + "k" * 64, "reign_number": 1}
        store = JsonBucketStore(FakeMinio(), "own", backoff=0)
        entry = ChallengerEntry(hotkey="5bad", model_repo="u/leoma-bad",
                                model_digest="sha256:" + "b" * 64, block=100)

        await box.drive(st, store, [entry], block=200, ticks=1)

        assert box.dispatched == [], "a wrong-architecture model was handed the GPU"
        key = vmain._seen_key(entry.hotkey, entry.model_digest)
        assert st.attempts[key]["last_reason"] == "arch_mismatch"
        assert st.duels["5bad"]["strikes"] == 1     # a gate rejection IS a strike

    async def test_a_good_model_passes_the_prescreen_and_is_dispatched(self, monkeypatch, duel_ready):
        monkeypatch.setattr(vmain, "PRESCREEN_ENABLED", True)
        monkeypatch.setattr(vmain, "prescreen", lambda repo, digest: {"pipeline": PIPELINE})

        box = FakeEvalBox(monkeypatch, lambda e: {"status": "running"}, duel_ready)
        st = KingState()
        st.king = {"hotkey": "5KING", "model_repo": "u/leoma-king",
                   "model_digest": "sha256:" + "k" * 64, "reign_number": 1}
        store = JsonBucketStore(FakeMinio(), "own", backoff=0)
        entry = ChallengerEntry(hotkey="5good", model_repo="u/leoma-good",
                                model_digest="sha256:" + "g" * 64, block=100)

        await box.drive(st, store, [entry], block=200, ticks=1)
        assert box.dispatched == ["5good"]


class TestLoadModelIndex:
    def test_a_missing_index_raises_rather_than_returning_none(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ArchMismatch):
            load_model_index(empty)
