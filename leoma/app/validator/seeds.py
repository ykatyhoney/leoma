"""Deterministic seed derivation for the king-of-the-hill duel.

Every validator must sample the same held-out clips and generate from the same
noise for a given challenge, so the duel is reproducible and consensus converges
without any cross-validator coordination. Seeds are derived from the reveal's
block hash + the challenger hotkey (Teutonic's scheme): the block hash is
unpredictable until mined — so miners can't overfit to the test set — yet fully
reproducible afterwards.
"""
from __future__ import annotations

import hashlib


def eval_seed_material(block_hash: str, hotkey: str, base_seed: int = 0) -> str:
    """Canonical string mixed into the duel seed.

    Falls back to just ``base_seed`` when no usable block hash is available, so a
    degraded run is still deterministic (though not chain-anchored).
    """
    bh = (block_hash or "").strip()
    hk = (hotkey or "").strip()
    if bh and bh != "default":
        return f"block_hash={bh}|hotkey={hk}|base_seed={base_seed}"
    return f"base_seed={base_seed}"


def _blake_int(material: str, digest_size: int = 8) -> int:
    digest = hashlib.blake2b(material.encode(), digest_size=digest_size).digest()
    return int.from_bytes(digest, "little")


def eval_seed(block_hash: str, hotkey: str, base_seed: int = 0) -> int:
    """Master duel seed for (block_hash, hotkey) — drives clip selection."""
    return _blake_int(eval_seed_material(block_hash, hotkey, base_seed))


def clip_generation_seed(master_seed: int, clip_index: int) -> int:
    """Per-clip generation seed, so king and challenger generate from the SAME
    noise on clip ``clip_index`` while different clips use different noise."""
    return _blake_int(f"seed={master_seed}|clip={clip_index}")


def select_clip_indices(master_seed: int, total_clips: int, n: int) -> list[int]:
    """Deterministically pick ``n`` distinct clip indices out of ``total_clips``.

    Same seed ⇒ same set (order-stable, ascending). Selects all clips when
    ``n >= total_clips``. The reference dataset is treated as a stable, sorted
    key list by the caller (as in the block-hash source sampling already shipped).
    """
    if total_clips <= 0 or n <= 0:
        return []
    if n >= total_clips:
        return list(range(total_clips))
    import numpy as np

    rng = np.random.default_rng(master_seed)
    chosen = rng.choice(total_clips, size=n, replace=False)
    return sorted(int(i) for i in chosen)
