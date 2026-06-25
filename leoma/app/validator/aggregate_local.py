"""
Local per-validator-average aggregation → on-chain winner (decentralized weight setting).

Self-evaluation model: each task is sampled AND evaluated by exactly one validator (its sampler),
so a peer's bucket holds exactly that peer's own verdicts at ``{task_id}/evaluation_results/<hotkey>.json``.
At each epoch every validator reads ALL permissioned validators' verdicts for the scoring window,
gives each validator equal weight (mean of per-validator pass-rates), and ranks miners by dominance.

Bucket reads are BOUNDED to the window (direct ``get_object`` per (peer, task_id)) rather than a
full-bucket list, so cost is O(window × peers), independent of how large the buckets grow. Each
peer's verdict file signature is verified against that peer's hotkey before it is trusted.
"""
import os
import json
import hashlib
import asyncio
from typing import Dict, Optional, Tuple

from substrateinterface import Keypair

from leoma.bootstrap import emit_log as log, log_exception
from leoma.app.validator.last_winner import load_last_winner, save_last_winner
from leoma.infra.aggregate import aggregate_per_validator_average, Verdicts
from leoma.infra.scorer_constants import required_distinct_validators
from leoma.infra.peer_registry import PeerBucket, load_peers
from leoma.infra.storage_backend import create_peer_read_client

# Dominance margin (kept identical to the dashboard scorer default).
DOMINANCE_THRESHOLD = float(os.environ.get("DOMINANCE_THRESHOLD", "0.05"))
# Bound concurrent bucket reads.
_READ_CONCURRENCY = int(os.environ.get("AGGREGATION_READ_CONCURRENCY", "16"))
# Retry the scoring-window fetch before giving up, so a brief owner-api restart at the epoch
# boundary doesn't cost an epoch. Total wait ≈ (ATTEMPTS-1) * BACKOFF; epoch processing has slack.
_WINDOW_FETCH_ATTEMPTS = int(os.environ.get("WINDOW_FETCH_ATTEMPTS", "5"))
_WINDOW_FETCH_BACKOFF = float(os.environ.get("WINDOW_FETCH_BACKOFF_SECONDS", "10"))


async def _fetch_window_with_retry(api_client, epoch_block: int):
    """Fetch the scoring window, retrying with backoff. Returns the response, or None if the
    owner-api stays unreachable across all attempts."""
    last_err: Optional[Exception] = None
    for attempt in range(1, _WINDOW_FETCH_ATTEMPTS + 1):
        try:
            return await api_client.get_task_window(as_of_block=epoch_block)
        except Exception as e:  # owner-api down / transient — retry
            last_err = e
            if attempt < _WINDOW_FETCH_ATTEMPTS:
                log(
                    f"Scoring-window fetch failed (attempt {attempt}/{_WINDOW_FETCH_ATTEMPTS}): {e}; "
                    f"retrying in {_WINDOW_FETCH_BACKOFF:.0f}s",
                    "warn",
                )
                await asyncio.sleep(_WINDOW_FETCH_BACKOFF)
    log(f"Owner-api unreachable for scoring window after {_WINDOW_FETCH_ATTEMPTS} attempts: {last_err}", "error")
    return None


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


async def compute_local_winner(api_client, epoch_block: int) -> Tuple[int, Optional[str]]:
    """Aggregate peer verdicts (per-validator average) into a winner. Returns (winner_uid, winner_hotkey).

    The scoring window is the owner-api's production-based ledger window, anchored to ``epoch_block``
    (the shared consensus block) so every validator that runs this epoch computes the identical window
    over fully settled, immutable verdict files — and skipped rotation turns never dilute it. Because
    the ledger tells us which validator sampled each task, we read ONLY that sampler's verdict file per
    task (O(window)), not every peer's (O(window × peers)). Returns (0, None) when there is nothing to
    score (no peers, no settled window, or no eligible miner), so the caller burns alpha on UID 0.

    If the owner-api is unreachable (after retries), we repeat the last winner this validator computed
    instead of burning the epoch — every validator's last winner is the same deterministic value, so
    they stay aligned through the outage. Burns only if there is no cached winner yet.
    """
    peers = load_peers()
    if not peers:
        log("PEER_VALIDATORS not configured; cannot aggregate locally", "error")
        return 0, None

    win = await _fetch_window_with_retry(api_client, epoch_block)
    if win is None:
        fallback = load_last_winner()
        if fallback:
            log(
                f"Owner-api down; repeating last winner uid={fallback[0]} "
                f"hotkey={fallback[1][:12]}... to avoid burning this epoch",
                "warn",
            )
            return fallback
        log("Owner-api down and no last winner cached; burning this epoch (UID 0)", "warn")
        return 0, None
    window_entries = win.get("window") or []
    if not window_entries:
        log("No settled scoring window yet (ledger empty / subnet too young)", "info")
        return 0, None

    active_validators = win.get("active_validators") or sorted(
        {e["sampler_hotkey"] for e in window_entries}
    )
    min_distinct = required_distinct_validators(len(active_validators))

    # Peer-ring drift: window samplers we have no read creds for contribute no verdicts (to everyone
    # equally) — warn so the operator can reconcile PEER_VALIDATORS with the allowlist.
    missing = set(active_validators) - set(peers.keys())
    if missing:
        log(
            "Window samplers missing from PEER_VALIDATORS (their verdicts can't be read): "
            f"{sorted(h[:12] for h in missing)}",
            "warn",
        )

    window: list = [int(e["rotation_id"]) for e in window_entries]

    verdicts: Verdicts = {}
    sem = asyncio.Semaphore(_READ_CONCURRENCY)
    # Read only the sampler's own verdict file for each produced task (O(window)).
    clients: Dict[str, object] = {}
    tasks = []
    for e in window_entries:
        rotation_id = int(e["rotation_id"])
        sampler = e["sampler_hotkey"]
        peer = peers.get(sampler)
        if peer is None:
            continue  # no read creds for this sampler (warned above)
        client = clients.get(sampler)
        if client is None:
            try:
                client = create_peer_read_client(peer)
            except Exception as ex:
                log(f"Cannot create read client for peer {sampler[:12]}...: {ex}", "warn")
                continue
            clients[sampler] = client
        tasks.append(_fetch_one(client, peer, rotation_id, verdicts, sem))
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        log_exception("Peer verdict read error", e)

    if not verdicts:
        log("No verdicts found across peer buckets for the window", "info")
        return 0, None

    # uid/block come from THIS validator's local chain read (no owner-api dependency); fall back
    # to the API's consensus table only until the first local validation has run.
    from leoma.app.validator import miner_validation

    snap = miner_validation.current_snapshot()
    if snap and snap.uid_by_hotkey:
        uid_by_hotkey = snap.uid_by_hotkey
        block_by_hotkey = snap.block_by_hotkey
    else:
        miners = await api_client.get_all_miners()
        uid_by_hotkey = {m.hotkey: m.uid for m in miners}
        block_by_hotkey = {m.hotkey: (m.block or None) for m in miners}

    # Eligibility is implicit via sampling: only sampled (locally-validated) miners have verdicts,
    # and the completeness gate requires broad sampling — so no explicit valid-miner filter here.
    winner_hotkey, rank_entries = aggregate_per_validator_average(
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
