"""Single source of truth for the pinned architecture + genesis king.

Reads ``chain.toml`` at the repo root and exposes the constants used by the
miner (default challenger namespace), the validator (repo naming + config-lock),
and the eval server (arch loading + seed king). To swap the pinned base model or
the genesis king, edit ``chain.toml`` — no code edits should be necessary.

Override knob: ``LEOMA_CHAIN_OVERRIDE`` env var, when set, points at an
alternate TOML (relative to the repo root or an absolute path). Used by local
testing and archived alternate configs so the default ``chain.toml`` can stay
pointed at the live chain.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import re
import tomllib
from types import ModuleType

# repo root = two parents up from this file (leoma/infra/chain_config.py -> repo/)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_OVERRIDE = os.environ.get("LEOMA_CHAIN_OVERRIDE", "").strip()
if _OVERRIDE:
    _candidate = pathlib.Path(_OVERRIDE)
    _TOML_PATH = _candidate if _candidate.is_absolute() else (_REPO_ROOT / _candidate)
else:
    _TOML_PATH = _REPO_ROOT / "chain.toml"

with open(_TOML_PATH, "rb") as _f:
    _doc = tomllib.load(_f)

_chain = _doc.get("chain", {})
_arch = _doc.get("arch", {})
_seed = _doc.get("seed", {})

_VALID_SEED_REPO_BACKENDS = {"hf", "hippius"}


def _default_seed_repo_backend(seed_digest: str) -> str:
    digest = (seed_digest or "").strip()
    if digest.startswith("sha256:"):
        return "hippius"
    if digest.startswith("hf:"):
        return "hf"
    return "hf"


NAME: str = _chain["name"]
SEED_REPO: str = _chain["seed_repo"]
REPO_PATTERN: str = _chain.get("repo_pattern") or rf"^[^/]+/{re.escape(NAME)}-.+$"

# Optional custom-code module whose import side effect registers config/model
# classes (empty for stock diffusers pipelines resolved by class name).
ARCH_MODULE: str = _arch.get("module", "")
# The pinned video-gen base: every challenger is weights for THIS architecture.
ARCH_BASE_REPO: str = _arch.get("base_repo", "")
# diffusers pipeline class used to load king + challenger (e.g. an I2V pipeline).
ARCH_PIPELINE: str = _arch.get("pipeline", "")
# Config keys locked to the base arch's values (challenger config must match).
EXTRA_LOCK_KEYS: tuple[str, ...] = tuple(_arch.get("extra_lock_keys", []))

SEED_DIGEST: str = _seed.get("seed_digest", "")
SEED_REPO_BACKEND: str = (_seed.get("repo_backend") or _default_seed_repo_backend(SEED_DIGEST)).strip().lower()
if SEED_REPO_BACKEND not in _VALID_SEED_REPO_BACKENDS:
    raise RuntimeError(
        f"chain.toml [seed].repo_backend must be one of "
        f"{sorted(_VALID_SEED_REPO_BACKENDS)}, got {SEED_REPO_BACKEND!r}"
    )

# Namespace inferred from the seed repo. Miners default their challenger repo to
# "<namespace>/<NAME>-<suffix>" though they can override to publish under their
# own account.
SEED_NAMESPACE: str = SEED_REPO.split("/", 1)[0] if "/" in SEED_REPO else ""


def load_arch() -> ModuleType:
    """Import the configured custom-arch module (registers HF/diffusers classes).

    Only needed for architectures that ship custom modeling code; stock
    diffusers pipelines are resolved by ``ARCH_PIPELINE`` class name instead.
    """
    if not ARCH_MODULE:
        raise RuntimeError("chain.toml [arch].module is not set (stock pipeline?)")
    return importlib.import_module(ARCH_MODULE)


__all__ = [
    "NAME",
    "SEED_REPO",
    "REPO_PATTERN",
    "ARCH_MODULE",
    "ARCH_BASE_REPO",
    "ARCH_PIPELINE",
    "EXTRA_LOCK_KEYS",
    "SEED_DIGEST",
    "SEED_REPO_BACKEND",
    "SEED_NAMESPACE",
    "load_arch",
]
