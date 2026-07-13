"""The consensus surface: no defaults, no extras, no silent drift.

Every test here pins a way two honest validators could have reached different
verdicts on the same challenger — which is the failure mode that matters most,
because it is silent, plausible, and destroys the subnet's credibility without
anyone noticing.
"""

import pytest
from pydantic import ValidationError

from leoma.eval.digests import canonical_json, digest_obj
from leoma.eval.errors import ConsensusConfigError
from leoma.eval.spec import ConsensusSpec, CorpusSpec, DuelSpec, GenSpec, verify_echo

from .conftest import pinned_spec


class TestNoDefaults:
    """A field with a default is a field a validator can silently forget."""

    @pytest.mark.parametrize("missing", ["metric", "metric_device", "n_clips",
                                         "delta_threshold", "alpha", "n_bootstrap"])
    def test_a_missing_duel_field_raises(self, missing):
        spec = pinned_spec()
        fields = spec.duel.model_dump()
        fields.pop(missing)
        with pytest.raises(ValidationError):
            DuelSpec(**fields)

    @pytest.mark.parametrize("missing", ["num_frames", "fps", "width", "height",
                                         "guidance_scale", "num_inference_steps",
                                         "negative_prompt", "prompt_mode", "prompt", "dtype"])
    def test_a_missing_gen_field_raises(self, missing):
        spec = pinned_spec()
        fields = spec.gen.model_dump()
        fields.pop(missing)
        with pytest.raises(ValidationError):
            GenSpec(**fields)

    def test_an_unknown_field_raises(self):
        spec = pinned_spec()
        with pytest.raises(ValidationError):
            DuelSpec(**spec.duel.model_dump(), future_knob=1)


class TestDigest:
    def test_same_spec_same_digest(self):
        assert pinned_spec().digest() == pinned_spec().digest()

    def test_any_field_changes_the_digest(self):
        base = pinned_spec()
        moved = base.model_copy(
            update={"duel": base.duel.model_copy(update={"n_clips": base.duel.n_clips + 1})}
        )
        assert moved.digest() != base.digest()

    def test_digest_survives_a_json_round_trip(self):
        """The spec crosses HTTP. If a float came back with a different last bit the
        digest would flip and every duel would be refused — so floats are quantized."""
        import json

        base = pinned_spec()
        round_tripped = ConsensusSpec.model_validate(json.loads(json.dumps(base.model_dump(mode="json"))))
        assert round_tripped.digest() == base.digest()

    def test_key_order_does_not_matter(self):
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
        assert digest_obj({"b": 1, "a": 2}) == digest_obj({"a": 2, "b": 1})


class TestVerifyEcho:
    """The check that catches "the field wasn't sent, so the box used its default"."""

    def test_matching_echo_passes(self):
        spec = pinned_spec()
        verify_echo(spec, spec.model_dump(mode="json"))

    def test_no_echo_at_all_is_refused(self):
        # An eval box on a build that predates the consensus surface. It may well
        # have used its own generation parameters — and it wouldn't know to say so.
        with pytest.raises(ConsensusConfigError, match="no consensus echo"):
            verify_echo(pinned_spec(), None)

    def test_a_box_that_ran_a_different_metric_is_caught(self):
        spec = pinned_spec()
        echoed = spec.model_dump(mode="json")
        echoed["duel"]["metric"] = "mse"
        with pytest.raises(ConsensusConfigError, match="DIFFERENT consensus spec"):
            verify_echo(spec, echoed)

    def test_the_error_names_the_field_that_drifted(self):
        spec = pinned_spec()
        echoed = spec.model_dump(mode="json")
        echoed["gen"]["num_frames"] = 17
        with pytest.raises(ConsensusConfigError, match=r"gen\.num_frames: sent=81 echoed=17"):
            verify_echo(spec, echoed)

    def test_a_malformed_echo_is_refused(self):
        with pytest.raises(ConsensusConfigError, match="malformed"):
            verify_echo(pinned_spec(), {"duel": "not a spec"})


class TestUnpinnedCorpusFailsClosed:
    def test_the_shipped_chain_toml_is_not_duel_ready(self):
        """It ships unpinned ON PURPOSE. A validator must not duel on "whatever the
        bucket holds today" — the corpus has to be published and pinned first."""
        from leoma.infra.chain_config import SPEC

        assert not SPEC.corpus.pinned
        with pytest.raises(ConsensusConfigError, match="not pinned"):
            SPEC.require_duel_ready()

    def test_a_pinned_corpus_is_duel_ready(self):
        pinned_spec().require_duel_ready()  # does not raise

    def test_a_garbage_digest_is_rejected_outright(self):
        with pytest.raises(ValidationError):
            CorpusSpec(bucket="videos", manifest_key="m.json", manifest_digest="not-a-hash")


class TestEarlyStopBound:
    def test_derived_from_the_threshold_so_it_tracks_recalibration(self):
        spec = pinned_spec()
        assert spec.early_stop_max_advantage == pytest.approx(
            spec.duel.early_stop_factor * spec.duel.delta_threshold
        )

    def test_at_20x_it_cannot_touch_a_marginal_challenger(self):
        """Sanity-check the bound is loose enough to only kill hopeless models.

        With 32 clips and delta=0.0025, early stop can only fire if the challenger
        is already so far behind that even a perfect run of the remaining clips
        can't reach the threshold. A challenger that is merely *slightly* worse
        must always be scored to the end — otherwise the bound is silently changing
        verdicts, not just saving GPU.
        """
        from leoma.eval.bootstrap import can_still_win

        spec = pinned_spec()
        bound = spec.early_stop_max_advantage
        n = spec.duel.n_clips

        # Halfway in, and marginally behind: must keep going.
        marginal = [-0.001] * (n // 2)
        assert can_still_win(
            [0.0] * (n // 2), [-d for d in marginal], remaining=n // 2,
            delta_threshold=spec.duel.delta_threshold, best_possible_advantage=bound,
        )

        # Halfway in, and hopelessly behind: stop.
        assert not can_still_win(
            [0.0] * (n // 2), [10.0] * (n // 2), remaining=n // 2,
            delta_threshold=spec.duel.delta_threshold, best_possible_advantage=bound,
        )
