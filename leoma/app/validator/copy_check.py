"""Weight-for-weight copy detection, from registry metadata alone.

``main._copies_a_king`` catches the crude copy: a different hotkey re-committing a
king's *exact* digest. It misses the copy that matters more — identical **weights**
repackaged with a changed README or tokenizer, which produces a *new* top-level
manifest digest but the **same per-layer safetensor digests**. On a content-addressed
registry, identical layer digests mean identical bytes, so that model is the king's
weights wearing a disguise. Under the old exact-digest-only gate it sailed through and
burned a **full multi-hour duel** before tying the king and losing.

This module closes that hole for the cost of one manifest fetch (a few KB) — **no
weight download**. It is the anti-abuse gate that matters most for a subnet whose
entire thesis is that GPU time is the scarce resource. Ported from Teutonic's
``check_model_copy`` (validator.py), adapted to Leoma's ``ModelRef``.

It does two jobs at once:

* **Reject** a copy committed *after* the king — plagiarism of the incumbent.
* **crown_earlier**: displace the king with a byte-identical model that was pushed
  to the registry *earlier*. That is the true original author, front-run by whoever
  got crowned first; they should hold the crown, and no duel is needed because the
  weights are provably identical to the reigning king's.

The earlier-author decision is **consensus-safe** because it rests only on the
registry's *own* observed push time (Harbor ``push_time`` / manifest
``Last-Modified``), which every validator reads identically — never on a
client-supplied annotation a miner could backdate. And it is **fail-safe**: if the
weights are identical but the timestamps can't be established, it *rejects* rather
than crowning, so the worst case is that a legitimate earlier author is turned away,
never that a plagiarist is enthroned.
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

from leoma.infra.model_store import ModelRef, fetch_oci_copy_info


def _parse_registry_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 or RFC-2822 timestamp to an aware UTC datetime, or None."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(ts)
        except (TypeError, ValueError):
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def check_model_copy(
    challenger_repo: str,
    challenger_digest: str,
    king_repo: str,
    king_digest: str,
    *,
    fetch=fetch_oci_copy_info,
) -> Optional[dict]:
    """Is the challenger a weight-for-weight copy of the king? What to do about it.

    Returns ``None`` when the models genuinely differ **or** when the check cannot be
    performed (fail open — a metadata hiccup must never block a valid submission).

    On a copy, returns ``{"action": "reject"|"crown_earlier", "reason": str, ...}``.

    ``fetch`` is injectable for testing; in production it is
    :func:`~leoma.infra.model_store.fetch_oci_copy_info`.
    """
    if not king_repo or not king_digest:
        return None

    # Exact re-commit of the reigning king. No metadata fetch needed, and never a
    # crown_earlier candidate: an identical top-level digest is the identical upload,
    # so there is no distinct "earlier author" to promote.
    if challenger_repo == king_repo and challenger_digest == king_digest:
        return {
            "action": "reject",
            "reason": f"challenger is the reigning king verbatim ({challenger_digest[:19]}...)",
            "challenger_committed_at": None,
            "king_committed_at": None,
        }

    challenger_info = fetch(ModelRef(challenger_repo, challenger_digest))
    if not challenger_info:
        return None
    king_info = fetch(ModelRef(king_repo, king_digest))
    if not king_info:
        return None

    challenger_layers = challenger_info["safetensor_layers"]
    king_layers = king_info["safetensor_layers"]

    # A different layer count, or any single differing layer digest, means these are
    # genuinely different weights. Not a copy — let it duel.
    if not challenger_layers or len(challenger_layers) != len(king_layers):
        return None
    if any(king_layers.get(title) != digest for title, digest in challenger_layers.items()):
        return None

    # Every weight layer is byte-identical. Decide by registry-observed push time only.
    n = len(challenger_layers)
    c_ts, k_ts = challenger_info.get("committed_at"), king_info.get("committed_at")
    c_src, k_src = challenger_info.get("timestamp_source"), king_info.get("timestamp_source")
    c_dt, k_dt = _parse_registry_timestamp(c_ts), _parse_registry_timestamp(k_ts)

    base = (f"all {n} .safetensors layers have identical OCI digests; "
            f"challenger pushed_at={c_ts} ({c_src}), king pushed_at={k_ts} ({k_src})")
    meta = {
        "challenger_committed_at": c_ts, "king_committed_at": k_ts,
        "challenger_timestamp_source": c_src, "king_timestamp_source": k_src,
    }

    # Fail-safe: never crown an "earlier" model unless BOTH timestamps came from the
    # registry itself. Missing/unparseable => reject (turn a real author away rather
    # than risk enthroning a plagiarist on backdated metadata).
    if c_dt is None or k_dt is None or not c_src or not k_src:
        return {"action": "reject",
                "reason": f"copy of the king; registry timestamps unavailable, cannot verify authorship: {base}",
                **meta}

    if c_dt < k_dt:
        return {"action": "crown_earlier",
                "reason": f"identical weights, earlier registry push time ({c_ts} < {k_ts}); "
                          f"the challenger is the original author, front-run by the king. {base}",
                **meta}

    return {"action": "reject", "reason": f"copy of the king, not earlier than it: {base}", **meta}


__all__ = ["check_model_copy"]
