"""Persisted king-of-the-hill state (this validator's own bucket).

Each validator keeps its king state in its own object bucket so it survives
restarts and is auditable. State is deterministic given the chain + reveals, so
validators converge on the same king without a shared store; the bucket is just
durable local memory (Teutonic's R2 role). Keys:

  king/current.json          the reigning king
  state/king_chain.json      recent prior kings sharing emission
  state/validator_state.json counters + last weight block + stats
  state/seen_hotkeys.json    hotkeys already evaluated (dedup)
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Optional

KEY_KING = "king/current.json"
KEY_KING_CHAIN = "state/king_chain.json"
KEY_VALIDATOR_STATE = "state/validator_state.json"
KEY_SEEN = "state/seen_hotkeys.json"
KEY_HISTORY = "state/history.json"

# Recent duel verdicts kept for the dashboard (newest first).
HISTORY_LIMIT = 200


class JsonBucketStore:
    """Tiny JSON KV over a Minio/S3 bucket (get returns None on miss)."""

    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    def get(self, key: str) -> Optional[dict]:
        try:
            resp = self.client.get_object(self.bucket, key)
        except Exception:
            return None
        try:
            data = resp.read()
        finally:
            try:
                resp.close()
                resp.release_conn()
            except Exception:
                pass
        try:
            return json.loads(data)
        except (ValueError, TypeError):
            return None

    def put(self, key: str, obj: Any) -> None:
        payload = json.dumps(obj, default=str, sort_keys=True).encode("utf-8")
        self.client.put_object(
            self.bucket,
            key,
            io.BytesIO(payload),
            length=len(payload),
            content_type="application/json",
        )


@dataclass
class KingState:
    """In-memory view of the persisted king state."""

    king: dict = field(default_factory=dict)
    king_chain: list = field(default_factory=list)
    last_weight_block: int = 0
    last_winner_hotkey: Optional[str] = None
    counter: int = 0
    stats: dict = field(default_factory=lambda: {"accepted": 0, "rejected": 0, "failed": 0})
    seen_hotkeys: set = field(default_factory=set)
    history: list = field(default_factory=list)  # recent duel verdicts, newest first

    # ---- persistence -----------------------------------------------------
    @classmethod
    def load(cls, store: JsonBucketStore) -> "KingState":
        self = cls()
        k = store.get(KEY_KING)
        if k:
            self.king = k
        kc = store.get(KEY_KING_CHAIN)
        if kc:
            self.king_chain = kc.get("chain", [])
        st = store.get(KEY_VALIDATOR_STATE)
        if st:
            self.last_weight_block = st.get("last_weight_block", 0)
            self.last_winner_hotkey = st.get("last_winner_hotkey")
            self.counter = st.get("counter", 0)
            self.stats = st.get("stats", self.stats)
        seen = store.get(KEY_SEEN)
        if seen:
            self.seen_hotkeys = set(seen.get("hotkeys", []))
        hist = store.get(KEY_HISTORY)
        if hist:
            self.history = hist.get("history", [])
        return self

    def flush(self, store: JsonBucketStore) -> None:
        store.put(KEY_KING, self.king)
        store.put(KEY_KING_CHAIN, {"chain": self.king_chain})
        store.put(KEY_VALIDATOR_STATE, {
            "last_weight_block": self.last_weight_block,
            "last_winner_hotkey": self.last_winner_hotkey,
            "counter": self.counter,
            "stats": self.stats,
        })
        store.put(KEY_SEEN, {"hotkeys": sorted(self.seen_hotkeys)})
        store.put(KEY_HISTORY, {"history": self.history})

    # ---- helpers ---------------------------------------------------------
    def next_eval_id(self) -> str:
        self.counter += 1
        return f"eval-{self.counter:04d}"

    def mark_seen(self, hotkey: str) -> None:
        self.seen_hotkeys.add(hotkey)

    def record_duel(self, entry: dict) -> None:
        """Prepend a duel verdict to the bounded history (newest first)."""
        self.history.insert(0, entry)
        del self.history[HISTORY_LIMIT:]
