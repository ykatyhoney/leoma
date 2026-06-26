"""On-chain anchored validator allowlist (decentralized rotation, Option B).

The owner permissions the validator set, but validators must agree on it WITHOUT trusting the
owner-api process. So the owner anchors the allowlist on-chain:

  - the full list lives in the shared source bucket at ``allowlist/v1.json``;
  - a ``sha256`` of that file is committed on-chain by the subnet-owner hotkey
    (``set_commitment`` -> ``leoma:allowlist:v1:<digest>``).

Every validator reads the subnet owner's commitment, fetches the file, and verifies the hash.
The owner can only change the set by re-committing on-chain (a visible, auditable tx) and the file
can't be tampered with off-chain (the on-chain hash pins it). Returns ``None`` on any failure so the
caller can fall back rather than sample against an unverified list.
"""
import asyncio
import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any, List, Optional

from leoma.bootstrap import emit_log as log

ALLOWLIST_OBJECT_KEY = "allowlist/v1.json"
COMMIT_PREFIX = "leoma:allowlist:v1:"


def canonical_payload(validators: List[str], interval: int) -> bytes:
    """Deterministic JSON bytes for the allowlist (sorted hotkeys) — what gets hashed and stored."""
    body = {"version": 1, "interval": int(interval), "validators": sorted(set(validators))}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_commitment(data: Optional[str]) -> Optional[str]:
    """Extract the committed digest from a commitment string, or ``None`` if it isn't ours."""
    if not data or not data.startswith(COMMIT_PREFIX):
        return None
    digest = data[len(COMMIT_PREFIX):].strip()
    return digest or None


@dataclass(frozen=True)
class AllowlistSnapshot:
    validators: List[str]   # sorted permissioned hotkeys = rotation order
    interval: int           # rotation interval in blocks
    digest: str             # sha256 of the published file (matches the on-chain commitment)


async def _put_bytes(client: Any, bucket: str, key: str, data: bytes) -> None:
    await asyncio.to_thread(
        client.put_object, bucket, key, io.BytesIO(data), len(data), "application/json"
    )


async def _get_bytes(client: Any, bucket: str, key: str) -> bytes:
    resp = await asyncio.to_thread(client.get_object, bucket, key)
    try:
        return await asyncio.to_thread(resp.read)
    finally:
        resp.close()
        resp.release_conn()


async def publish_allowlist(
    subtensor: Any,
    wallet: Any,
    netuid: int,
    write_client: Any,
    source_bucket: str,
    validators: List[str],
    interval: int,
) -> str:
    """Owner-side: write ``allowlist/v1.json`` to the source bucket and commit its hash on-chain.

    Must be signed by the subnet-owner wallet (the on-chain authority validators trust). Returns
    the published digest.
    """
    payload = canonical_payload(validators, interval)
    digest = digest_of(payload)
    await _put_bytes(write_client, source_bucket, ALLOWLIST_OBJECT_KEY, payload)
    await subtensor.set_commitment(wallet, netuid, COMMIT_PREFIX + digest)
    log(f"Published allowlist: {len(json.loads(payload)['validators'])} validators, digest {digest[:12]}…", "success")
    return digest


async def read_allowlist(
    subtensor: Any,
    netuid: int,
    read_client: Any,
    source_bucket: str,
) -> Optional[AllowlistSnapshot]:
    """Validator-side: read the owner's on-chain commitment, fetch the file, verify the hash.

    Returns ``None`` (and logs) if the commitment is missing/foreign, the file is unreadable, or the
    hash doesn't match — so a caller never samples against an unverified or tampered allowlist.
    """
    try:
        owner_hotkey = await subtensor.get_subnet_owner_hotkey(netuid)
        if not owner_hotkey:
            log("No subnet owner hotkey on-chain; cannot read allowlist", "warn")
            return None
        commitments = await subtensor.get_all_commitments(netuid)
        digest = parse_commitment((commitments or {}).get(owner_hotkey))
        if digest is None:
            log("Subnet owner has not committed an allowlist on-chain yet", "warn")
            return None
        payload = await _get_bytes(read_client, source_bucket, ALLOWLIST_OBJECT_KEY)
    except Exception as e:
        log(f"Could not read on-chain allowlist: {e}", "warn")
        return None

    if digest_of(payload) != digest:
        log("Allowlist file hash does not match the on-chain commitment; rejecting", "error")
        return None
    try:
        body = json.loads(payload)
        validators = sorted({str(h) for h in body.get("validators", []) if h})
        interval = int(body.get("interval", 0))
    except Exception as e:
        log(f"Allowlist file is malformed: {e}", "warn")
        return None
    if not validators or interval <= 0:
        log("Allowlist file is empty or has no interval; rejecting", "warn")
        return None
    return AllowlistSnapshot(validators=validators, interval=interval, digest=digest)
