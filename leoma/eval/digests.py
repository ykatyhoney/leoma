"""Canonical digests — the one hashing convention the whole consensus surface uses.

Every digest that appears in a verdict, a manifest, or ``chain.toml`` is computed
here, so two validators hashing "the same thing" cannot disagree because one of
them serialized a float differently or ordered a dict differently.

The rules, in one place:

* **JSON** is dumped with sorted keys, no whitespace, and ``ensure_ascii`` — so
  the byte string is a pure function of the value, not of the writer.
* **Floats** are quantized before hashing (:func:`canonical_float`). A float that
  survives a TOML → JSON → HTTP → JSON round-trip can pick up a last-bit
  difference; a consensus digest must not be that fragile.
* **Frames** hash their exact ``uint8`` bytes plus their shape, so two arrays
  that differ only in shape can never collide.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Digest of a value that is deliberately absent (vs. one we failed to compute).
EMPTY = "sha256:" + hashlib.sha256(b"").hexdigest()

# Floats are rounded to this many decimals before hashing. Generous enough for
# every consensus knob we pin (guidance 5.0, delta 0.0025, alpha 0.001) and tight
# enough that float-repr noise cannot change a digest.
FLOAT_PRECISION = 9


def canonical_float(value: float) -> float:
    """Quantize a float so it hashes identically after a JSON round-trip."""
    rounded = round(float(value), FLOAT_PRECISION)
    # Normalize -0.0 to 0.0: they compare equal but serialize differently.
    return rounded + 0.0


def _canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        return canonical_float(value)
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return value


def canonical_json(value: Any) -> bytes:
    """Serialize to the one byte string every validator must agree on."""
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_obj(value: Any) -> str:
    """``sha256:`` digest of any JSON-able value, canonically serialized."""
    return sha256_bytes(canonical_json(value))


def digest_frames(frames) -> str:
    """Digest a frame array by its exact uint8 bytes *and* its shape.

    Hashing the bytes alone would let a (2, 4, 4, 3) array collide with a
    (4, 2, 4, 3) one holding the same values.
    """
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(frames, dtype="uint8"))
    h = hashlib.sha256()
    h.update(canonical_json(list(arr.shape)))
    h.update(arr.tobytes())
    return "sha256:" + h.hexdigest()


def digest_file(path: str, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return "sha256:" + h.hexdigest()


__all__ = [
    "EMPTY",
    "FLOAT_PRECISION",
    "canonical_float",
    "canonical_json",
    "sha256_bytes",
    "digest_obj",
    "digest_frames",
    "digest_file",
]
