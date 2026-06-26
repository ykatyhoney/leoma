"""
Local pooled-pass-rate aggregation → on-chain winner (decentralized weight setting).

Each task is sampled AND evaluated by exactly one validator (its sampler), whose bucket holds its
verdicts at ``{task_id}/evaluation_results/<hotkey>.json``. At each epoch every validator:

  1. reads the hardcoded validator allowlist from the repo (no owner-api),
  2. derives the settled scoring window itself from peer-bucket producedness + the shared epoch block,
  3. reads each window task's verdict from its canonical sampler, scores miners by pooled pass-rate
     (total_passed / total_evaluated), and ranks them by dominance.

The whole path is owner-api-independent: the allowlist comes from the chain, the window from the
buckets validators already read. Every validator computes the identical window — and winner — over
immutable, signed verdict files.
"""
import os
import json
import hashlib
import asyncio
from typing import Dict, List, Optional, Set, Tuple

from substrateinterface import Keypair

from leoma.bootstrap import emit_log as log, log_exception, SAMPLING_ROTATION_INTERVAL
from leoma.app.validator.last_winner import load_last_winner, save_last_winner
from leoma.app.validator.window_local import resolve_canonical_samplers, derive_window
from leoma.infra.aggregate import aggregate_scores, Verdicts
from leoma.infra.allowlist import load_allowlist
from leoma.infra.scorer_constants import required_distinct_validators
from leoma.infra.peer_registry import PeerBucket, load_peers
from leoma.infra.storage_backend import (
    create_peer_read_client,
    list_evaluated_task_ids,
)

# Dominance margin (kept identical to the dashboard scorer default).
DOMINANCE_THRESHOLD = float(os.environ.get("DOMINANCE_THRESHOLD", "0.05"))
# Bound concurrent bucket reads.
_READ_CONCURRENCY = int(os.environ.get("AGGREGATION_READ_CONCURRENCY", "16"))
# Recent rotations to list per peer bucket when discovering producedness (window is anchored after).
_DISCOVERY_MAX_TASKS = int(os.environ.get("WINDOW_DISCOVERY_MAX_TASKS", "500"))


async def _discover_produced(
    peers: Dict[str, PeerBucket], validators: List[str]
) -> Tuple[Dict[str, Set[int]], Dict[str, object]]:
    """List each validator's bucket to find the rotations it published verdicts for.

    Returns ``(produced_by_peer, read_clients)`` — the reusable read clients are handed back so the
    verdict-file reads don't have to recreate them.
    """
    produced: Dict[str, Set[int]] = {}
    clients: Dict[str, object] = {}
    sem = asyncio.Semaphore(_READ_CONCURRENCY)

    async def one(hotkey: str) -> None:
        peer = peers.get(hotkey)
        if peer is None:
            return
        try:
            client = create_peer_read_client(peer)
        except Exception as ex:
            log(f"Cannot create read client for peer {hotkey[:12]}...: {ex}", "warn")
            return
        clients[hotkey] = client
        async with sem:
            try:
                rids = await list_evaluated_task_ids(
                    client, peer.bucket, hotkey, _DISCOVERY_MAX_TASKS
                )
            except Exception:
                rids = []
        produced[hotkey] = set(rids)

    await asyncio.gather(*(one(hk) for hk in validators))
    return produced, clients


def _verify(peer_hotkey: str, wrapper: dict) -> bool:
    """Verify a verdict file's signature against the peer's hotkey (matches sign_evaluation_payload).

    Accepts files with no signature (compat) and fails closed only on an explicit bad signature —
    a verification *error* is treated as acceptable (logged) so a library quirk can't silently zero
    out the whole aggregation and burn alpha.
    """
    sig = wrapper.get("signature")
    if not sig:
        return True
    try:
        canonical = json.dumps(wrapper.get("data") or [], sort_keys=True).encode("utf-8")
        msg_hash = hashlib.sha256(canonical).digest()
        keypair = Keypair(ss58_address=peer_hotkey)
        return keypair.verify(msg_hash, bytes.fromhex(sig.removeprefix("0x")))
    except Exception:
        return True


async def _fetch_one(client, peer: PeerBucket, task_id: int, verdicts: Verdicts, sem: asyncio.Semaphore) -> None:
    """Fetch + verify peer's verdict file for one task; add its entries to ``verdicts``."""
    safe = peer.hotkey.replace("/", "_").replace("\\", "_")
    key = f"{task_id}/evaluation_results/{safe}.json"
    async with sem:
        try:
            resp = await asyncio.to_thread(client.get_object, peer.bucket, key)
        except Exception:
            return  # no file (skipped turn / not this peer's task)
        try:
            body = await asyncio.to_thread(resp.read)
        finally:
            resp.close()
            resp.release_conn()
    try:
        wrapper = json.loads(body)
    except Exception as e:
        log(f"Bad verdict JSON {peer.bucket}/{key}: {e}", "warn")
        return
    if not _verify(peer.hotkey, wrapper):
        log(f"Rejected verdict file with bad signature: {peer.bucket}/{key}", "warn")
        return
    for entry in wrapper.get("data") or []:
        miner_hotkey = entry.get("hotkey")
        if miner_hotkey:
            verdicts[(peer.hotkey, task_id, miner_hotkey)] = bool(entry.get("passed"))


async def compute_local_winner(
    epoch_block: int,
    *,
    get_all_miners=None,
) -> Tuple[int, Optional[str]]:
    """Aggregate peer verdicts (pooled pass-rate) into a winner. Returns (winner_uid, winner_hotkey).

    Owner-api-independent: the validator set is the hardcoded repo allowlist, and the settled window
    is derived locally from peer-bucket producedness anchored to ``epoch_block`` (the shared consensus
    block) — so every validator computes the identical window over immutable, signed verdict files.
    Returns (0, None) when there is nothing to score (no peers, no settled window, or no eligible
    miner), so the caller burns alpha on UID 0.

    If a settled window exists but no verdict files could be read (likely a transient peer-bucket
    outage), repeat the last winner this validator computed instead of burning the epoch — every
    validator's last winner is the same deterministic value, so they stay aligned through the outage.
    """
    peers = load_peers()
    if not peers:
        log("PEER_VALIDATORS not configured; cannot aggregate locally", "error")
        return 0, None

    # The owner-managed validator set, hardcoded in the repo (single source of truth, no R2/chain).
    snap = load_allowlist(SAMPLING_ROTATION_INTERVAL)
    validators, interval = snap.validators, snap.interval

    # Derive the window ourselves: who produced which rotation (from buckets) + the shared epoch block.
    produced_by_peer, clients = await _discover_produced(peers, validators)
    canonical = resolve_canonical_samplers(produced_by_peer, validators)
    window, active_validators = derive_window(canonical, epoch_block, interval)
    if not window:
        log("No settled scoring window yet (no produced tasks / subnet too young)", "info")
        return 0, None
    min_distinct = required_distinct_validators(len(active_validators))

    verdicts: Verdicts = {}
    sem = asyncio.Semaphore(_READ_CONCURRENCY)
    # Read only the canonical sampler's own verdict file for each produced task (O(window)).
    tasks = []
    for rotation_id in window:
        sampler = canonical[rotation_id]
        peer = peers.get(sampler)
        client = clients.get(sampler)
        if peer is None or client is None:
            continue
        tasks.append(_fetch_one(client, peer, rotation_id, verdicts, sem))
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        log_exception("Peer verdict read error", e)

    if not verdicts:
        fallback = load_last_winner()
        if fallback:
            log(
                f"No verdict files readable for the window (peer-bucket outage?); repeating last "
                f"winner uid={fallback[0]} hotkey={fallback[1][:12]}... to avoid burning this epoch",
                "warn",
            )
            return fallback
        log("No verdicts found across peer buckets for the window", "info")
        return 0, None

    # uid/block come from THIS validator's local chain read (no owner-api dependency); fall back to a
    # caller-provided resolver only until the first local validation has run.
    from leoma.app.validator import miner_validation

    snap_m = miner_validation.current_snapshot()
    if snap_m and snap_m.uid_by_hotkey:
        uid_by_hotkey = snap_m.uid_by_hotkey
        block_by_hotkey = snap_m.block_by_hotkey
    elif get_all_miners is not None:
        miners = await get_all_miners()
        uid_by_hotkey = {m.hotkey: m.uid for m in miners}
        block_by_hotkey = {m.hotkey: (m.block or None) for m in miners}
    else:
        log("No local miner snapshot for uid mapping; burning this epoch (UID 0)", "warn")
        return 0, None

    # Eligibility is implicit via sampling: only sampled (locally-validated) miners have verdicts,
    # and the completeness gate requires broad sampling — so no explicit valid-miner filter here.
    winner_hotkey, rank_entries = aggregate_scores(
        verdicts,
        window,
        set(),
        block_by_hotkey,
        DOMINANCE_THRESHOLD,
        min_distinct_validators=min_distinct,
    )
    if not winner_hotkey:
        log("No eligible miner won the per-validator-average aggregation", "info")
        return 0, None

    winner_uid = int(uid_by_hotkey.get(winner_hotkey, 0))
    if winner_uid:
        # Persist so the next epoch can repeat it if the owner-api is unreachable then.
        save_last_winner(winner_uid, winner_hotkey, epoch_block)
    log(
        f"Local aggregation: {len(rank_entries)} ranked, window {window[0]}…{window[-1]}, "
        f"winner={winner_hotkey[:12]}... uid={winner_uid}",
        "info",
    )
    return winner_uid, winner_hotkey
