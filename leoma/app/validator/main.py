"""
Validator service for Leoma — king of the hill.

Each permissioned validator independently:
  1. scans the chain's revealed commitments for miner model submissions
     (``reveal_scan``) — no owner-api, no rotation,
  2. duels each new challenger against the reigning king on a GPU eval server
     (deterministic, block-hash-seeded), and
  3. crowns a challenger that wins by a confident margin, then sets equal weights
     across the king + recent prior kings (else burns to UID 0).

Because the duel is deterministic given the chain + reveals, every validator
converges on the same king with no cross-validator coordination. The king state
persists to this validator's own bucket (``state_store``).
"""

import os
import json
import asyncio
from typing import Optional

import bittensor as bt

from leoma.bootstrap import (
    NETUID,
    NETWORK,
    WALLET_NAME,
    HOTKEY_NAME,
    R2_OWN_BUCKET,
)
from leoma.bootstrap import emit_log as log, emit_header as log_header, log_exception
from leoma.infra.storage_backend import create_own_write_client, ensure_bucket_exists
from leoma.infra.chain_config import SEED_REPO, SEED_DIGEST
from leoma.app.validator.reveal_scan import scan_reveals, ChallengerEntry
from leoma.app.validator.state_store import JsonBucketStore, KingState
from leoma.app.validator import king as K

EVAL_SERVER_URL = os.environ.get("EVAL_SERVER_URL", "http://localhost:9000")
CHALLENGE_POLL_INTERVAL = int(os.environ.get("LEOMA_CHALLENGE_POLL_INTERVAL", "60"))

# Duel parameters (must match across validators for consensus).
DUEL_METRIC = os.environ.get("LEOMA_DUEL_METRIC", "lpips")
DUEL_N_CLIPS = int(os.environ.get("LEOMA_DUEL_N_CLIPS", "32"))
DELTA_THRESHOLD = float(os.environ.get("LEOMA_DELTA_THRESHOLD", "0.0025"))
ALPHA = float(os.environ.get("LEOMA_ALPHA", "0.001"))
N_BOOTSTRAP = int(os.environ.get("LEOMA_N_BOOTSTRAP", "10000"))

# Eval dispatch: connect fast, but allow a long duel (N clips x 2 generations).
_EVAL_CONNECT_TIMEOUT = 30.0
_EVAL_READ_TIMEOUT = float(os.environ.get("LEOMA_EVAL_TIMEOUT", "3600"))


def _seen_key(hotkey: str, digest: str) -> str:
    return f"{hotkey}|{digest}"


def _is_current_king(state: KingState, entry: ChallengerEntry) -> bool:
    k = state.king or {}
    return k.get("hotkey") == entry.hotkey and k.get("model_digest") == entry.model_digest


async def refresh_uid_map(subtensor: bt.AsyncSubtensor) -> dict[str, int]:
    meta = await subtensor.metagraph(NETUID)
    hotkeys = list(getattr(meta, "hotkeys", []) or [])
    return {hk: uid for uid, hk in enumerate(hotkeys)}


async def dispatch_duel(
    entry: ChallengerEntry,
    king: dict,
    block_hash: str,
) -> Optional[dict]:
    """POST a duel to the eval server and stream its verdict.

    Returns the verdict dict, or None when the server is busy (409) so the
    caller retries later. Raises on an eval-server error event.
    """
    import httpx

    payload = {
        "king_repo": king["model_repo"],
        "king_digest": king["model_digest"],
        "challenger_repo": entry.model_repo,
        "challenger_digest": entry.model_digest,
        "block_hash": block_hash,
        "hotkey": entry.hotkey,
        "metric": DUEL_METRIC,
        "n_clips": DUEL_N_CLIPS,
        "delta_threshold": DELTA_THRESHOLD,
        "alpha": ALPHA,
        "n_bootstrap": N_BOOTSTRAP,
    }
    timeout = httpx.Timeout(_EVAL_READ_TIMEOUT, connect=_EVAL_CONNECT_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{EVAL_SERVER_URL}/eval", json=payload)
        if resp.status_code == 409:
            log("Eval server busy; will retry", "warn")
            return None
        resp.raise_for_status()
        eval_id = resp.json()["eval_id"]

        verdict: Optional[dict] = None
        async with client.stream("GET", f"{EVAL_SERVER_URL}/eval/{eval_id}/stream") as stream:
            async for line in stream.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                phase = event.get("phase")
                if phase == "verdict":
                    verdict = event
                elif phase == "error":
                    raise RuntimeError(f"eval server error: {event.get('error')}")
    return verdict


async def maybe_set_weights(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    *,
    force: bool = False,
) -> bool:
    """Set equal weights over the king chain (else burn UID 0), rate-limited."""
    current_block = await subtensor.get_current_block()
    if not force and current_block - state.last_weight_block < K.WEIGHT_INTERVAL:
        return False

    uids, weights, label = K.weight_targets(state.king, state.king_chain, uid_map)
    log(f"[{current_block}] set_weights -> {label}: uids={uids} weights={[round(w,4) for w in weights]}", "info")
    try:
        await subtensor.set_weights(
            wallet=wallet, netuid=NETUID, uids=uids, weights=weights, wait_for_inclusion=True
        )
    except Exception as e:
        log(f"[{current_block}] set_weights failed: {e}", "error")
        return False

    state.last_weight_block = current_block
    state.last_winner_hotkey = label
    state.flush(store)
    return True


def ensure_genesis_king(state: KingState, block: int) -> None:
    """Seed the genesis king from chain.toml on first run (empty hotkey ⇒ burns
    emission until a miner's challenger dethrones the base model)."""
    if state.king:
        return
    if not SEED_DIGEST:
        log("No seed_digest in chain.toml; king unset (burning) until first crown", "warn")
        return
    king, chain = K.crown(
        None, [], hotkey="", model_repo=SEED_REPO, model_digest=SEED_DIGEST,
        block=block, challenge_id=K.SEED_CHALLENGE_ID,
    )
    state.king, state.king_chain = king, chain
    log(f"Seeded genesis king from {SEED_REPO}@{SEED_DIGEST[:16]}...", "success")


async def process_challengers(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    entries: list[ChallengerEntry],
) -> None:
    """Duel each new challenger; crown the winners and refresh weights."""
    for entry in entries:
        key = _seen_key(entry.hotkey, entry.model_digest)
        if key in state.seen_hotkeys or _is_current_king(state, entry):
            continue
        if not state.king:
            # No king to duel against and no seed configured; the first valid
            # challenger takes the crown unopposed.
            state.king, state.king_chain = K.crown(
                None, state.king_chain, hotkey=entry.hotkey, model_repo=entry.model_repo,
                model_digest=entry.model_digest, block=entry.block, challenge_id="first",
            )
            state.seen_hotkeys.add(key)
            state.stats["accepted"] = state.stats.get("accepted", 0) + 1
            state.flush(store)
            log(f"Crowned first king {entry.hotkey[:12]}... ({entry.model_repo})", "success")
            await maybe_set_weights(subtensor, wallet, state, uid_map, store, force=True)
            continue

        block_hash = await subtensor.get_block_hash(entry.block)
        log(f"Dueling challenger {entry.hotkey[:12]}... vs king {state.king.get('hotkey','')[:12] or 'base'}...", "info")
        try:
            verdict = await dispatch_duel(entry, state.king, block_hash)
        except Exception as e:
            log(f"Duel dispatch failed for {entry.hotkey[:12]}...: {e}", "error")
            return  # leave unseen so it retries next tick

        if verdict is None:
            return  # server busy; retry later, keep entry unseen

        state.seen_hotkeys.add(key)
        if verdict.get("accepted"):
            state.king, state.king_chain = K.crown(
                state.king, state.king_chain, hotkey=entry.hotkey, model_repo=entry.model_repo,
                model_digest=entry.model_digest, block=entry.block,
                challenge_id=verdict.get("challenge_id") or f"block-{entry.block}",
            )
            state.stats["accepted"] = state.stats.get("accepted", 0) + 1
            state.flush(store)
            log(f"Challenger {entry.hotkey[:12]}... CROWNED (lcb={verdict.get('lcb')})", "success")
            await maybe_set_weights(subtensor, wallet, state, uid_map, store, force=True)
        else:
            state.stats["rejected"] = state.stats.get("rejected", 0) + 1
            state.flush(store)
            log(f"Challenger {entry.hotkey[:12]}... rejected (lcb={verdict.get('lcb')})", "info")


async def tick(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    store: JsonBucketStore,
) -> None:
    block = await subtensor.get_current_block()
    uid_map = await refresh_uid_map(subtensor)
    ensure_genesis_king(state, block)

    commits = await subtensor.get_all_revealed_commitments(NETUID, block=block)
    entries = scan_reveals(commits)
    if entries:
        await process_challengers(subtensor, wallet, state, uid_map, store, entries)

    # Periodic weight refresh (also keeps the king chain aligned as UIDs change).
    await maybe_set_weights(subtensor, wallet, state, uid_map, store)


async def main() -> None:
    """Run the king-of-the-hill validator loop."""
    log_header("Leoma Validator Starting (king of the hill)")

    if not R2_OWN_BUCKET:
        log("R2_OWN_BUCKET not set; validator disabled (cannot persist king state)", "error")
        return

    subtensor = bt.AsyncSubtensor(network=NETWORK)
    wallet = bt.Wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
    log(f"Wallet: {WALLET_NAME}/{HOTKEY_NAME}  Network: {NETWORK}  NetUID: {NETUID}", "info")
    log(f"Eval server: {EVAL_SERVER_URL}  metric={DUEL_METRIC} n_clips={DUEL_N_CLIPS} delta={DELTA_THRESHOLD}", "info")

    own_client = create_own_write_client()
    await ensure_bucket_exists(own_client, R2_OWN_BUCKET)
    store = JsonBucketStore(own_client, R2_OWN_BUCKET)
    state = KingState.load(store)
    log(f"Loaded king: {(state.king or {}).get('model_repo', 'none')} chain={len(state.king_chain)}", "info")

    while True:
        try:
            await tick(subtensor, wallet, state, store)
        except Exception as e:
            log(f"Validator tick error: {e}", "error")
            log_exception("Validator tick error", e)
        await asyncio.sleep(CHALLENGE_POLL_INTERVAL)


def main_sync() -> None:
    """Synchronous entry point for CLI."""
    asyncio.run(main())
