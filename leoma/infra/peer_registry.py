"""
Peer validator bucket registry (decentralized aggregation).

Each permissioned validator owns an R2 result bucket. Read credentials are shared
peer-to-peer via the ``PEER_VALIDATORS`` env var (a JSON list), so any validator can
download any peer's evaluation results to aggregate scores locally.

PEER_VALIDATORS schema (one entry per permissioned validator, INCLUDING self):

    [
      {
        "hotkey": "5F...",            # validator SS58 hotkey
        "uid": 3,                      # metagraph uid (optional, informational)
        "bucket": "leoma-val-3",       # that validator's result bucket
        "endpoint": "https://<acct>.r2.cloudflarestorage.com",
        "region": "auto",             # optional, defaults to "auto"
        "read_access_key": "...",      # read-only key for that bucket
        "read_secret_key": "..."
      },
      ...
    ]
"""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from leoma.bootstrap import emit_log
from leoma.bootstrap.runtime import settings


@dataclass(frozen=True)
class PeerBucket:
    """A peer validator's result bucket and read credentials."""

    hotkey: str
    bucket: str
    endpoint: str
    region: str
    read_access_key: str
    read_secret_key: str
    uid: Optional[int] = None


def _parse_peers(raw: str) -> Dict[str, PeerBucket]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as e:
        emit_log(f"PEER_VALIDATORS is not valid JSON: {e}", "error")
        return {}
    if not isinstance(entries, list):
        emit_log("PEER_VALIDATORS must be a JSON list of peer objects", "error")
        return {}
    peers: Dict[str, PeerBucket] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        hotkey = entry.get("hotkey")
        bucket = entry.get("bucket")
        endpoint = entry.get("endpoint")
        ak = entry.get("read_access_key")
        sk = entry.get("read_secret_key")
        if not (hotkey and bucket and endpoint and ak and sk):
            emit_log(
                f"PEER_VALIDATORS entry missing required fields (hotkey/bucket/endpoint/read_access_key/read_secret_key): {entry.get('hotkey', '?')}",
                "warn",
            )
            continue
        peers[hotkey] = PeerBucket(
            hotkey=hotkey,
            bucket=bucket,
            endpoint=endpoint,
            region=entry.get("region") or "auto",
            read_access_key=ak,
            read_secret_key=sk,
            uid=entry.get("uid"),
        )
    return peers


def load_peers() -> Dict[str, PeerBucket]:
    """Parse the ``PEER_VALIDATORS`` env JSON into a ``{hotkey -> PeerBucket}`` map."""
    return _parse_peers(settings.peer_validators)


def get_peer(hotkey: str) -> Optional[PeerBucket]:
    """Resolve a single peer's bucket + read creds by hotkey (None if not in the ring)."""
    return load_peers().get(hotkey)


def peer_hotkeys() -> List[str]:
    """All permissioned validator hotkeys present in the peer registry."""
    return list(load_peers().keys())


def own_bucket() -> Optional[str]:
    """This validator's own result bucket name (write target)."""
    return settings.r2_own_bucket
