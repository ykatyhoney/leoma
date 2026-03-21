"""Stake-weighted pass/fail from multiple validator evaluations (one task, one miner)."""


def stake_weighted_pass(samples_with_stakes: list[tuple[bool, float]]) -> bool:
    """Return True when stake-weighted average score is above 0.5."""
    total_stake = sum(stake for _, stake in samples_with_stakes)
    if total_stake <= 0:
        return False
    weighted = sum((1 if passed else 0) * stake for passed, stake in samples_with_stakes)
    return (weighted / total_stake) > 0.5
