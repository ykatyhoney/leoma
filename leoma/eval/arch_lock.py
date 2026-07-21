"""The architecture lock, for a diffusers layout.

Every miner submits **weights for one pinned architecture**. A model that isn't that
architecture cannot be dueled fairly — it will load wrong, generate wrong, or fail on
the GPU after we have already spent an hour downloading 70 GB of it.

The reference subnet locks a flat transformers ``config.json``. **We can't.** A
diffusers snapshot has *no root config.json at all*: it has ``model_index.json`` (a
map of component → [library, class]) plus a separate config inside each component
directory, and the shape-critical numbers live in ``transformer/config.json`` and
``vae/config.json``. So the lock has to be **nested and per-component**.

It also validates by **diffing against the base repo's own configs** rather than
against numbers transcribed by hand into ``chain.toml``. Hand-transcribed shapes rot:
the day someone bumps the pinned base model, every hand-copied number silently becomes
a lie, and the lock starts rejecting the very architecture it is supposed to enforce.
Diffing against the base repo cannot drift, because the base repo *is* the definition.

Five checks, cheapest first:

1. ``model_index.json`` exists and its ``_class_name`` is the pinned pipeline.
2. Every locked component is present with exactly its ``[library, class]``.
3. No **extra** components (a smuggled-in module is a code-execution surface).
4. Per-component **locked keys** deep-compare against the base.
5. **Size bound** — a stub repo (DoS: dispatch a 1 KB "model" over and over) and a
   500 GB repo (DoS: fill the disk) are both rejected.

All of it runs on a **config-only fetch** (~200 KB), on the *validator*, **before**
the challenger is ever dispatched. A bad model costs ~5 seconds instead of hours of
GPU — which is the actual mitigation for "one bad repo blocks everyone", and it
finally consumes the three long-dead hooks (``materialize_model(config_only=True)``,
``EXTRA_LOCK_KEYS``, ``ARCH_BASE_REPO``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from leoma.eval.errors import ChallengerFault

#: Component configs whose shape decides whether the weights can load at all.
#: Keys are compared against the BASE REPO's values, never against a copy in chain.toml.
LOCKED_KEYS: dict[str, tuple[str, ...]] = {
    "transformer": (
        "num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "patch_size",
        "freq_dim",
        "ffn_dim",
        "text_dim",
    ),
    # Wan2.2 A14B is a mixture-of-experts pipeline with a second transformer.
    # It has the same shape-critical surface and must be locked independently;
    # checking only the first expert would allow incompatible weights through.
    "transformer_2": (
        "num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "patch_size",
        "freq_dim",
        "ffn_dim",
        "text_dim",
    ),
    "vae": (
        "latent_channels",
        "z_dim",
        "base_dim",
        "dim_mult",
    ),
}

#: Libraries a component may come from. Anything else is a smuggled dependency.
ALLOWED_LIBRARIES = frozenset({"diffusers", "transformers"})

MIN_SNAPSHOT_BYTES = int(os.environ.get("LEOMA_MIN_SNAPSHOT_BYTES", str(1 * 1024**3)))
MAX_SNAPSHOT_BYTES = int(os.environ.get("LEOMA_MAX_SNAPSHOT_BYTES", str(200 * 1024**3)))


class ArchMismatch(ChallengerFault):
    """The challenger is not the pinned architecture. Miner-facing, and specific."""

    reason = "arch_mismatch"


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "rb") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_model_index(snapshot: str | os.PathLike[str]) -> dict:
    """The diffusers pipeline manifest. Its absence is itself a rejection."""
    index = _read_json(Path(snapshot) / "model_index.json")
    if not isinstance(index, dict):
        raise ArchMismatch(
            "model_index.json is missing or unreadable. A diffusers pipeline is defined "
            "by that file; without it this is not a loadable model. (Note: a transformers "
            "-style root config.json is NOT what this subnet expects.)"
        )
    return index


def components(index: dict) -> dict[str, list]:
    """The enabled component map: name -> [library, class].

    Diffusers serializes intentionally disabled optional components as
    ``[null, null]``. They load no code and carry no weights, so they are metadata,
    not executable components. Partially-null pairs remain visible and fail the
    library/class checks below.
    """
    return {
        name: value
        for name, value in index.items()
        if (
            not name.startswith("_")
            and isinstance(value, list)
            and len(value) == 2
            and value != [None, None]
        )
    }


def check_pipeline_class(index: dict, *, pipeline: str) -> None:
    actual = index.get("_class_name")
    if actual != pipeline:
        raise ArchMismatch(
            f"model_index.json declares _class_name={actual!r}, but this subnet is pinned to "
            f"{pipeline!r}. Fine-tune the pinned base architecture; do not substitute a "
            "different pipeline."
        )


def check_components(index: dict, base_index: dict) -> None:
    """Same components, same classes, no extras, no exotic libraries."""
    mine = components(index)
    base = components(base_index)

    missing = sorted(set(base) - set(mine))
    if missing:
        raise ArchMismatch(f"missing pipeline components: {', '.join(missing)}")

    extra = sorted(set(mine) - set(base))
    if extra:
        raise ArchMismatch(
            f"unexpected pipeline components: {', '.join(extra)}. The pipeline is pinned; "
            "extra components are not loaded, and shipping one is a way to smuggle code onto "
            "the eval box."
        )

    for name, (library, cls) in sorted(mine.items()):
        if library not in ALLOWED_LIBRARIES:
            raise ArchMismatch(
                f"component {name!r} comes from library {library!r}; only "
                f"{sorted(ALLOWED_LIBRARIES)} are allowed."
            )
        base_library, base_cls = base[name]
        if [library, cls] != [base_library, base_cls]:
            raise ArchMismatch(
                f"component {name!r} is {library}.{cls}, but the pinned architecture uses "
                f"{base_library}.{base_cls}."
            )


def check_component_configs(
    snapshot: str | os.PathLike[str],
    base_snapshot: str | os.PathLike[str],
    *,
    extra_keys: tuple[str, ...] = (),
) -> None:
    """Deep-compare the shape-critical keys against the BASE REPO's own configs.

    Only keys the base actually defines are compared. A key the base doesn't have is
    not a shape constraint — it is a diffusers version difference, and rejecting on it
    would mean the lock breaks every time diffusers adds a field.
    """
    root = Path(snapshot)
    base_root = Path(base_snapshot)

    for component, keys in LOCKED_KEYS.items():
        base_config = _read_json(base_root / component / "config.json")
        if base_config is None:
            continue  # the base has no such component; nothing to lock against

        config = _read_json(root / component / "config.json")
        if config is None:
            raise ArchMismatch(f"{component}/config.json is missing or unreadable")

        for key in (*keys, *extra_keys):
            if key not in base_config:
                continue
            expected = base_config[key]
            actual = config.get(key, "<missing>")
            if actual != expected:
                raise ArchMismatch(
                    f"{component}/config.json {key}={actual!r} but the pinned architecture "
                    f"locks it to {expected!r}. Submit WEIGHTS for the pinned architecture — "
                    "changing its shape means the weights cannot be loaded by the duel."
                )


def check_size(total_bytes: int) -> None:
    """Reject a stub repo and a disk-filling one. Both are cheap DoS on the eval box."""
    if total_bytes < MIN_SNAPSHOT_BYTES:
        raise ArchMismatch(
            f"model is {total_bytes / 1024**3:.2f} GB, below the {MIN_SNAPSHOT_BYTES / 1024**3:.0f} GB "
            "floor for this architecture. A stub repo cannot be the pinned 14B model."
        )
    if total_bytes > MAX_SNAPSHOT_BYTES:
        raise ArchMismatch(
            f"model is {total_bytes / 1024**3:.0f} GB, above the {MAX_SNAPSHOT_BYTES / 1024**3:.0f} GB "
            "ceiling. Refusing to fill the eval box's disk."
        )


def validate(
    snapshot: str | os.PathLike[str],
    base_snapshot: str | os.PathLike[str],
    *,
    pipeline: str,
    extra_keys: tuple[str, ...] = (),
    total_bytes: Optional[int] = None,
) -> dict:
    """Run the whole lock. Raises :class:`ArchMismatch` with a miner-facing reason."""
    index = load_model_index(snapshot)
    base_index = load_model_index(base_snapshot)

    check_pipeline_class(index, pipeline=pipeline)
    check_components(index, base_index)
    check_component_configs(snapshot, base_snapshot, extra_keys=extra_keys)
    if total_bytes is not None:
        check_size(total_bytes)

    return {"pipeline": pipeline, "components": sorted(components(index))}


__all__ = [
    "ALLOWED_LIBRARIES",
    "LOCKED_KEYS",
    "MAX_SNAPSHOT_BYTES",
    "MIN_SNAPSHOT_BYTES",
    "ArchMismatch",
    "check_component_configs",
    "check_components",
    "check_pipeline_class",
    "check_size",
    "components",
    "load_model_index",
    "validate",
]
