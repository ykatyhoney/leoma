"""Validator allowlist — hardcoded in the repo (single source of truth).

The permissioned validator set is committed directly in source control: public ss58 hotkeys, version
controlled and auditable via git history. Every validator running this code derives the identical
rotation order and equal-weight voter set from it — no R2 bucket, no on-chain commitment. Change the
set by editing this list and shipping a release; all validators must run the same version for
consensus to agree.
"""
from dataclasses import dataclass
from typing import List

from leoma.bootstrap import SAMPLING_ROTATION_INTERVAL

# Permissioned validator hotkeys (ss58). Edit + release to change the set.
VALIDATOR_ALLOWLIST: List[str] = [
    "5C7LM2i42XgL2oB4x3rcmB7KDiof4B92KZzUpg5miZ6DogjU",
    "5DJ76XJdWvU7PcmKmBjzoAKYC3i4YjhdR92uVYGA7FthyCv2",
    "5GW8VcE7gLU8pJFWvYYC378RyDWxN8rTCm1fNYX6AxzDV1de",
    "5CrGhhemVi8e77LRpogbQEvuqvBssaEYz2EzrUfNR5bJ1s99",
]


@dataclass(frozen=True)
class AllowlistSnapshot:
    validators: List[str]   # sorted permissioned hotkeys = rotation order + equal-weight voter set
    interval: int           # rotation interval in blocks


def load_allowlist(interval: int = SAMPLING_ROTATION_INTERVAL) -> AllowlistSnapshot:
    """The hardcoded validator allowlist as a snapshot (sorted, deduped)."""
    return AllowlistSnapshot(validators=sorted(set(VALIDATOR_ALLOWLIST)), interval=int(interval))
