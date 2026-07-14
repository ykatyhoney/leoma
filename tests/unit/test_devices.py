"""Device assignment for concurrent king/challenger generation.

A throughput setting, not a consensus one — see eval/devices.py's module docstring for
why. These tests pin the decision logic: when concurrency is actually honored, and the
several ways it safely falls back to today's single-device behavior instead of
guessing.
"""

import pytest

from leoma.eval.devices import resolve_duel_devices


class TestDisabled:
    def test_disabled_means_no_devices_assigned(self):
        d = resolve_duel_devices(concurrent_enabled=False, cuda_device_count=8)
        assert d.king_device is None
        assert d.challenger_device is None
        assert d.concurrent is False


class TestInsufficientDevices:
    @pytest.mark.parametrize("count", [0, 1])
    def test_fewer_than_two_devices_falls_back(self, count):
        d = resolve_duel_devices(concurrent_enabled=True, cuda_device_count=count)
        assert d.concurrent is False
        assert "falling back" in d.note

    def test_the_fallback_explains_itself(self):
        d = resolve_duel_devices(concurrent_enabled=True, cuda_device_count=1)
        assert "1 CUDA device" in d.note


class TestHappyPath:
    def test_two_devices_assigns_king_and_challenger_separately(self):
        d = resolve_duel_devices(concurrent_enabled=True, cuda_device_count=2)
        assert d.concurrent is True
        assert d.king_device == "cuda:0"
        assert d.challenger_device == "cuda:1"

    def test_more_than_two_devices_still_uses_the_first_two(self):
        d = resolve_duel_devices(concurrent_enabled=True, cuda_device_count=8)
        assert d.king_device == "cuda:0"
        assert d.challenger_device == "cuda:1"
        assert d.concurrent is True


class TestOverrides:
    def test_explicit_overrides_are_respected(self):
        d = resolve_duel_devices(
            concurrent_enabled=True, cuda_device_count=8,
            king_device_override="cuda:2", challenger_device_override="cuda:3",
        )
        assert d.king_device == "cuda:2"
        assert d.challenger_device == "cuda:3"
        assert d.concurrent is True

    def test_both_overrides_are_trusted_even_with_a_low_reported_count(self):
        """torch.cuda.device_count() might undercount what an operator knows is usable
        (MIG partitions, CUDA_VISIBLE_DEVICES tricks) — an explicit pair of overrides
        is trusted rather than second-guessed."""
        d = resolve_duel_devices(
            concurrent_enabled=True, cuda_device_count=1,
            king_device_override="cuda:0", challenger_device_override="cuda:1",
        )
        assert d.concurrent is True

    def test_a_single_override_with_too_few_devices_still_falls_back(self):
        """Only ONE side pinned isn't enough information to safely pick the other."""
        d = resolve_duel_devices(
            concurrent_enabled=True, cuda_device_count=1, king_device_override="cuda:0",
        )
        assert d.concurrent is False

    def test_overrides_colliding_on_the_same_device_falls_back(self):
        """Generating 'concurrently' on the same device just serializes on that
        device's queue — no overlap, so it must not be reported as concurrent."""
        d = resolve_duel_devices(
            concurrent_enabled=True, cuda_device_count=8,
            king_device_override="cuda:0", challenger_device_override="cuda:0",
        )
        assert d.concurrent is False
        assert "not overlap" in d.note

    def test_a_default_challenger_colliding_with_an_overridden_king_falls_back(self):
        """King is pinned to cuda:1 (the challenger's default) with no explicit
        challenger override — the collision must still be caught."""
        d = resolve_duel_devices(
            concurrent_enabled=True, cuda_device_count=8, king_device_override="cuda:1",
        )
        assert d.challenger_device == "cuda:1"
        assert d.concurrent is False


class TestNoteIsAlwaysPresent:
    def test_every_outcome_explains_itself(self):
        for kwargs in (
            dict(concurrent_enabled=False, cuda_device_count=8),
            dict(concurrent_enabled=True, cuda_device_count=0),
            dict(concurrent_enabled=True, cuda_device_count=2),
        ):
            d = resolve_duel_devices(**kwargs)
            assert d.note and isinstance(d.note, str)
