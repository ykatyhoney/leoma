"""Pure rotation math (shared by the owner-api and each validator's local resolver)."""
from leoma.infra.rotation_math import (
    compute_sampler,
    effective_sampler,
    failover_step,
    grace_blocks_for,
    rotation_index_for_block,
)

V = ["A", "B", "C", "D"]


def test_rotation_index_and_sampler():
    assert rotation_index_for_block(1000, 100) == 10
    assert compute_sampler(V, 10) == V[10 % 4]   # "C"
    assert compute_sampler(V, 13) == "B"
    assert compute_sampler([], 5) is None


def test_failover_step_boundaries():
    # within grace -> primary
    assert failover_step(0, 50, 4) == 0
    assert failover_step(49, 50, 4) == 0
    # after one / two grace periods
    assert failover_step(50, 50, 4) == 1
    assert failover_step(120, 50, 4) == 2
    # capped at n-1, never wraps the ring
    assert failover_step(10_000, 50, 4) == 3
    # degenerate inputs
    assert failover_step(100, 0, 4) == 0
    assert failover_step(100, 50, 1) == 0


def test_grace_blocks_for():
    assert grace_blocks_for(100, 0) == 50      # default: half a window
    assert grace_blocks_for(100, 30) == 30     # configured wins


def test_effective_sampler_produced_vs_failover():
    # produced -> primary, no failover
    s, step = effective_sampler(V, 10, block_offset=999, grace_blocks=50, produced=True)
    assert (s, step) == (compute_sampler(V, 10), 0)
    # not produced, within grace -> still primary
    s, step = effective_sampler(V, 10, block_offset=10, grace_blocks=50, produced=False)
    assert (s, step) == (compute_sampler(V, 10), 0)
    # not produced, one grace lapsed -> first backup
    s, step = effective_sampler(V, 10, block_offset=60, grace_blocks=50, produced=False)
    assert step == 1 and s == compute_sampler(V, 11)
