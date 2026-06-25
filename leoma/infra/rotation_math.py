"""Pure rotation math — shared by the owner-api route and each validator's local resolver.

The sampling schedule is a deterministic function of the current chain block and the ordered
validator allowlist, so every validator computes the identical turn with no coordinator:

    rotation_index = block // interval
    sampler        = validators[rotation_index % len(validators)]

Failover advances the turn to the next validator(s) when the primary hasn't produced within a
grace period — again purely block-derived — so a missed turn rarely leaves a gap.
"""
from typing import List, Optional, Tuple


def rotation_index_for_block(block: int, interval: int) -> int:
    return block // max(1, interval)


def compute_sampler(validators: List[str], rotation_index: int) -> Optional[str]:
    if not validators:
        return None
    return validators[rotation_index % len(validators)]


def failover_step(block_offset: int, grace_blocks: int, n_validators: int) -> int:
    """Rotation positions to advance given blocks elapsed in the window without a produced task.

    ``0`` within the first ``grace_blocks`` (primary's turn); ``1`` after one grace period, ``2``
    after two, capped at ``n_validators - 1`` so it never wraps the whole ring.
    """
    if grace_blocks <= 0 or n_validators <= 1 or block_offset < grace_blocks:
        return 0
    return min(block_offset // grace_blocks, n_validators - 1)


def grace_blocks_for(interval: int, configured: int = 0) -> int:
    """Failover grace period in blocks; ``configured`` (>0) wins, else half a window."""
    return configured if configured > 0 else max(1, interval // 2)


def effective_sampler(
    validators: List[str],
    rotation_index: int,
    block_offset: int,
    grace_blocks: int,
    produced: bool,
) -> Tuple[Optional[str], int]:
    """The validator whose turn it currently is and the failover step applied.

    Returns the primary (step 0) when the task is already produced or there is no failover yet;
    otherwise advances to the deterministic backup for how long the window has gone unproduced.
    """
    if produced or not validators:
        return compute_sampler(validators, rotation_index), 0
    step = failover_step(block_offset, grace_blocks, len(validators))
    return compute_sampler(validators, rotation_index + step), step
