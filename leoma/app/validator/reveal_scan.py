"""Validator-side discovery of miner model submissions from on-chain reveals.

The validator reads the revealed commitments from chain, parses each into an
immutable ``(repo, digest)`` model reference, and queues the challenger for
evaluation. It downloads the weights itself (``model_store``) — no miner-hosted
endpoint is ever contacted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from leoma.infra.model_store import parse_reveal_v4
from leoma.infra.commit_parser import validate_repo_name


@dataclass(frozen=True)
class ChallengerEntry:
    """A discovered miner model submission ready for evaluation."""

    hotkey: str
    block: int
    model_repo: str
    model_digest: str


def scan_reveals(
    commits: Optional[Dict[str, Sequence[Tuple[int, str]]]],
    *,
    blacklist: Optional[set] = None,
) -> List[ChallengerEntry]:
    """Parse revealed commitments into challenger entries (latest valid per hotkey).

    ``commits`` is the shape returned by
    ``AsyncSubtensor.get_all_revealed_commitments``: ``{hotkey: [(block, payload), ...]}``.
    For each hotkey we take the latest payload, parse the ``v4`` reveal, and:
      - drop blacklisted hotkeys,
      - require the payload's author hotkey to match the chain signer (chain wins),
      - enforce the repo naming rule (starts "leoma", ends with the hotkey),
      - skip legacy/malformed/non-v4 payloads (e.g. the old JSON commit format).

    Entries are returned in ascending commit-block order (older submissions
    first), so a stable, deterministic queue across validators.
    """
    blk = blacklist or set()
    out: List[ChallengerEntry] = []

    for hotkey, history in (commits or {}).items():
        if not history or hotkey in blk:
            continue

        block, payload = history[-1]
        try:
            ref, author_hotkey = parse_reveal_v4(payload)
        except ValueError:
            continue  # legacy JSON / malformed / non-v4 reveal

        if author_hotkey != hotkey:
            continue  # payload author must match the chain commitment signer

        ok, _reason = validate_repo_name(ref.repo, hotkey=hotkey)
        if not ok:
            continue

        out.append(
            ChallengerEntry(
                hotkey=hotkey,
                block=int(block),
                model_repo=ref.repo,
                model_digest=ref.digest,
            )
        )

    out.sort(key=lambda e: (e.block, e.hotkey))
    return out
