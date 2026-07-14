"""
Unit tests for the flow (optical-flow) and clip (semantic) metrics.

OpenCV / open_clip aren't installed in this env, so the learned backends are
imported lazily (only when the metric is *called*). Here we test the pure-numpy
math they rely on (flow endpoint error, cosine distance), the registry wiring,
and that importing the metric modules never pulls in cv2/torch.
"""

import numpy as np
import pytest

from leoma.eval.metrics import (
    get_metric,
    flow_distance,
    clip_distance,
    make_composite,
    mse,
    temporal_distance,
)
from leoma.eval._flow import flow_endpoint_error
from leoma.eval._clip import cosine_distance


class TestFlowEndpointError:
    def test_identical_flow_is_zero(self):
        rng = np.random.default_rng(0)
        f = rng.standard_normal((3, 4, 4, 2))
        assert flow_endpoint_error(f, f) == 0.0

    def test_shift_gives_known_error(self):
        # Every flow vector differs by (3, 4) -> endpoint error = 5 everywhere.
        a = np.zeros((2, 4, 4, 2))
        b = np.full((2, 4, 4, 2), 0.0)
        b[..., 0] = 3.0
        b[..., 1] = 4.0
        assert flow_endpoint_error(a, b) == pytest.approx(5.0)

    def test_empty_is_zero(self):
        empty = np.zeros((0, 4, 4, 2))
        assert flow_endpoint_error(empty, empty) == 0.0


class TestCosineDistance:
    def test_identical_is_zero(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 1.0]])
        assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_is_one(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        assert cosine_distance(a, b) == pytest.approx(1.0)

    def test_opposite_is_two(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        assert cosine_distance(a, b) == pytest.approx(2.0)

    def test_magnitude_invariant(self):
        # Cosine ignores magnitude: scaling one set doesn't change the distance.
        a = np.array([[1.0, 2.0, 2.0]])
        b = np.array([[2.0, 4.0, 4.0]])
        assert cosine_distance(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_empty_is_zero(self):
        assert cosine_distance(np.empty((0, 4)), np.empty((0, 4))) == 0.0


class TestRegistryWiring:
    def test_flow_and_clip_registered(self):
        assert get_metric("flow") is flow_distance
        assert get_metric("clip") is clip_distance

    def test_all_axes_present(self):
        from leoma.eval.metrics import _METRICS
        assert set(_METRICS) == {"mse", "ssim", "temporal", "flow", "lpips", "clip"}

    def test_composite_with_flow_is_lazy(self):
        # Building a composite that includes flow/clip must NOT import cv2/torch
        # (they're only pulled when the metric is actually called on frames).
        m = make_composite({"mse": 1.0, "flow": 0.5, "clip": 0.25})
        assert callable(m)

    def test_composite_spec_with_flow_clip(self):
        m = get_metric("composite:mse=1.0,flow=0.5")
        assert callable(m)


class TestImportSafety:
    def test_metric_modules_import_without_heavy_deps(self):
        # These imports must succeed even though cv2/open_clip aren't installed.
        import importlib
        import leoma.eval._flow as f
        import leoma.eval._clip as c
        importlib.reload(f)
        importlib.reload(c)
        assert hasattr(f, "flow_video_distance")
        assert hasattr(c, "clip_video_distance")


class TestShortGenerationIsNotFree:
    """THE freeze/1-frame cheat, closed at the root.

    Before: `_align` truncated the TRUTH down to the generation's length, so a
    1-frame output was scored against frame 0 of the truth — which IS the
    conditioning frame the model was handed. Every metric, including the
    production default (lpips), returned a near-perfect score.
    """

    @pytest.mark.parametrize("metric", ["mse", "ssim", "temporal", "flow", "lpips", "clip"])
    def test_one_frame_generation_is_rejected_by_every_metric(self, metric):
        rng = np.random.default_rng(0)
        truth = rng.integers(0, 255, size=(8, 8, 8, 3)).astype("uint8")
        one_frame = truth[:1].copy()          # the cheat: echo the conditioning frame

        fn = get_metric(metric)
        # The guard sits ABOVE the lazy import, so this runs with no torch/cv2.
        with pytest.raises(ValueError, match="too short|at least 2 frames"):
            fn(one_frame, truth)

    @pytest.mark.parametrize("metric", ["mse", "ssim", "temporal", "flow", "lpips", "clip"])
    def test_partial_generation_is_rejected(self, metric):
        rng = np.random.default_rng(1)
        truth = rng.integers(0, 255, size=(8, 6, 6, 3)).astype("uint8")
        short = truth[:5].copy()              # 5 < 8: still short, still a cheat
        with pytest.raises(ValueError, match="too short"):
            get_metric(metric)(short, truth)

    def test_composite_propagates_the_rejection(self):
        rng = np.random.default_rng(2)
        truth = rng.integers(0, 255, size=(6, 5, 5, 3)).astype("uint8")
        with pytest.raises(ValueError, match="too short"):
            get_metric("composite:mse=1.0,temporal=0.5")(truth[:1], truth)

    def test_full_length_generation_still_scores(self):
        rng = np.random.default_rng(3)
        truth = rng.integers(0, 255, size=(6, 5, 5, 3)).astype("uint8")
        assert mse(truth, truth) == 0.0                    # identical -> 0
        assert temporal_distance(truth, truth) == 0.0

    def test_longer_generation_is_truncated_not_rejected(self):
        rng = np.random.default_rng(4)
        truth = rng.integers(0, 255, size=(4, 5, 5, 3)).astype("uint8")
        longer = np.concatenate([truth, truth[:2]], axis=0)   # 6 frames vs 4
        assert mse(longer, truth) == 0.0                      # truncated to the truth
