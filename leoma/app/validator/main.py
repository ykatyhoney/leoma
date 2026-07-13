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
from datetime import datetime, timezone
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
from leoma.infra.chain_config import (
    CONSENSUS_DIGEST,
    NAME as CHAIN_NAME,
    SEED_REPO,
    SEED_DIGEST,
    SPEC,
)
from leoma.eval.codehash import eval_code_digest
from leoma.eval.errors import ConsensusConfigError
from leoma.eval.spec import verify_echo
from leoma.app.validator.reveal_scan import scan_reveals, ChallengerEntry
from leoma.app.validator.state_store import (
    JsonBucketStore,
    KingState,
    StateInconsistent,
    StoreUnavailable,
)
from leoma.app.validator.dashboard import build_dashboard, publish_dashboard
from leoma.app.validator.failures import (
    DuelFailure,
    EvalBusy,
    EvalJobFailed,
    classify,
    classify_remote,
)
from leoma.app.validator.state_store import MAX_DUEL_ATTEMPTS
from leoma.app.validator import king as K

EVAL_SERVER_URL = os.environ.get("EVAL_SERVER_URL", "http://localhost:9000")
CHALLENGE_POLL_INTERVAL = int(os.environ.get("LEOMA_CHALLENGE_POLL_INTERVAL", "60"))

# The duel parameters (metric, n_clips, delta, alpha, bootstrap, generation knobs)
# are NOT read from the environment any more. They are the consensus surface, and
# they live in chain.toml as `SPEC`. An env var is per-box; a per-box exam is not
# consensus. LEOMA_DUEL_METRIC / _N_CLIPS / _DELTA_THRESHOLD / _ALPHA / _N_BOOTSTRAP
# are deliberately gone — setting one no longer does anything, which is the point.

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


async def preflight_eval_server(client) -> None:
    """Refuse a box whose pinned exam or scoring code has drifted from ours.

    A stale eval box is the most *likely* consensus failure in the whole system —
    not an attack, just an operator who redeployed three machines out of four. It
    produces confident, plausible verdicts that no other validator can reproduce.
    One cheap GET catches it; the alternative is finding out after an hour of GPU.
    """
    resp = await client.get(f"{EVAL_SERVER_URL}/health")
    resp.raise_for_status()
    health = resp.json()

    theirs = health.get("consensus_digest")
    if theirs != CONSENSUS_DIGEST:
        raise EvalJobFailed(
            f"eval server pins a different consensus surface (box {theirs}, "
            f"validator {CONSENSUS_DIGEST}). One of us is running a stale chain.toml.",
            reason="consensus_mismatch",
        )
    their_code = health.get("eval_code_digest")
    if their_code and their_code != eval_code_digest():
        raise EvalJobFailed(
            f"eval server runs different scoring code (box {their_code}, validator "
            f"{eval_code_digest()}). Its distances would not be reproducible.",
            reason="code_mismatch",
        )


async def dispatch_duel(
    entry: ChallengerEntry,
    king: dict,
    block_hash: str,
) -> dict:
    """POST a duel to the eval server and stream its verdict.

    Sends the **whole pinned consensus surface** (``SPEC``) rather than a handful of
    loose knobs, and verifies the echo before returning — so a verdict produced under
    different parameters can never reach the crowning code.

    Raises ``EvalBusy`` when the server is already running a duel (409), and
    ``EvalJobFailed`` on a terminal error — including a stream that ends without a
    verdict. It used to return ``None`` for BOTH cases, so "the server is busy" was
    indistinguishable from "the duel broke", and the caller treated a broken duel as
    a reason to stop processing everyone else.
    """
    import httpx

    payload = {
        "king_repo": king["model_repo"],
        "king_digest": king["model_digest"],
        "challenger_repo": entry.model_repo,
        "challenger_digest": entry.model_digest,
        "block_hash": block_hash,
        "hotkey": entry.hotkey,
        "spec": SPEC.model_dump(mode="json"),
        "consensus_digest": CONSENSUS_DIGEST,
    }
    timeout = httpx.Timeout(_EVAL_READ_TIMEOUT, connect=_EVAL_CONNECT_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await preflight_eval_server(client)

        resp = await client.post(f"{EVAL_SERVER_URL}/eval", json=payload)
        if resp.status_code == 409:
            raise EvalBusy("eval server is already running a duel")
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
                    raise EvalJobFailed(
                        str(event.get("error") or "eval server error"),
                        reason=str(event.get("reason") or ""),
                    )

    if verdict is None:
        # The stream ended with neither a verdict nor an error. That is a broken
        # duel, not a busy server — surface it as an explicit transient failure.
        raise EvalJobFailed(
            "eval stream ended without a terminal verdict", reason="stream_no_terminal"
        )

    # Fail closed: the box must have run EXACTLY the exam we set. This is the check
    # that catches "the field wasn't sent, so the box used its own default" — the
    # entire bug class that made two honest validators disagree.
    try:
        verify_echo(SPEC, verdict.get("echo"))
    except ConsensusConfigError as e:
        raise EvalJobFailed(str(e), reason="consensus_echo_mismatch") from e

    return verdict


# bittensor 9.12.2's set_weights returns (success: bool, message: str). Its
# rate-limit path returns this literal WITHOUT ever submitting an extrinsic.
_RATE_LIMIT_TOKENS = (
    "no attempt made",
    "too soon",
    "rate limit",
    "settingweightstoofast",
)


def _unpack_set_weights(result) -> tuple[bool, str]:
    """Tolerate (bool, str), a bare bool, or an ExtrinsicResponse-like object."""
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0]), str(result[1])
    if hasattr(result, "success"):
        return bool(result.success), str(getattr(result, "message", "") or "")
    return bool(result), ""


def _is_rate_limited(message: str) -> bool:
    lowered = (message or "").lower()
    return any(token in lowered for token in _RATE_LIMIT_TOKENS)


async def _chain_says_too_soon(subtensor: bt.AsyncSubtensor, wallet: bt.Wallet) -> bool:
    """Ask the CHAIN whether it is too soon to set weights.

    The chain is the queue, so the chain is also the weight clock. Trusting our
    own ``last_weight_block`` is what made "state says the weights landed but they
    didn't" possible; asking the chain makes that structurally impossible.
    Degrades to False (i.e. attempt the set) if the node doesn't expose these.
    """
    try:
        uid = await subtensor.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, NETUID)
        if uid is None:
            return False
        since = await subtensor.blocks_since_last_update(NETUID, uid)
        limit = await subtensor.weights_rate_limit(NETUID)
    except Exception:
        return False
    if since is None or limit is None:
        return False
    return since <= limit


def _weight_failed(state: KingState, block: int, message: str) -> bool:
    state.weight_failures += 1
    # Exponential block backoff, capped — a genuine failure should be retried
    # soon, NOT after a full WEIGHT_INTERVAL.
    state.next_weight_block = block + min(2**state.weight_failures, 20)
    state.touch()
    log(f"[{block}] set_weights FAILED ({state.weight_failures}x): {message}", "error")
    return False


async def maybe_set_weights(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    *,
    force: bool = False,
) -> bool:
    """Set equal weights over the king chain (else burn UID 0), rate-limited.

    The return value of ``set_weights`` used to be discarded. bittensor returns
    ``(success, message)`` and its rate-limit path returns ``False`` *without
    submitting anything* — so a no-op advanced ``last_weight_block`` as if the
    weights had landed, blocking any retry for a full WEIGHT_INTERVAL (~1h) and
    misreporting the state. We now only advance on a real success.
    """
    current_block = await subtensor.get_current_block()

    if not force:
        if current_block < state.next_weight_block:
            return False  # backing off after a genuine failure
        if current_block - state.last_weight_block < K.WEIGHT_INTERVAL:
            return False
        if await _chain_says_too_soon(subtensor, wallet):
            return False

    uids, weights, label = K.weight_targets(
        state.king, state.king_chain, uid_map, burn_uid=K.BURN_UID
    )
    log(
        f"[{current_block}] set_weights -> {label}: uids={uids} "
        f"weights={[round(w, 4) for w in weights]}",
        "info",
    )
    try:
        result = await subtensor.set_weights(
            wallet=wallet, netuid=NETUID, uids=uids, weights=weights, wait_for_inclusion=True
        )
    except Exception as e:
        return _weight_failed(state, current_block, f"exception: {e}")

    success, message = _unpack_set_weights(result)
    if not success:
        if _is_rate_limited(message):
            # A no-op, not a failure: do NOT advance last_weight_block.
            log(f"[{current_block}] set_weights rate-limited ({message}); retrying next tick", "warn")
            return False
        return _weight_failed(state, current_block, message)

    state.last_weight_block = current_block
    state.last_winner_hotkey = label
    state.weight_failures = 0
    state.next_weight_block = 0
    state.touch()
    await state.flush(store)
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
    state.touch()
    log(f"Seeded genesis king from {SEED_REPO}@{SEED_DIGEST[:16]}...", "success")


def _duel_history_entry(entry: ChallengerEntry, verdict: dict, uid_map: dict[str, int]) -> dict:
    """A dashboard history row from a completed duel verdict."""
    return {
        "challenge_id": verdict.get("challenge_id") or f"block-{entry.block}",
        "hotkey": entry.hotkey,
        "uid": uid_map.get(entry.hotkey),
        "model_repo": entry.model_repo,
        "model_digest": entry.model_digest,
        "accepted": bool(verdict.get("accepted")),
        "verdict": verdict.get("verdict", "unknown"),
        "mu_hat": verdict.get("mu_hat"),
        "lcb": verdict.get("lcb"),
        "delta": verdict.get("delta_threshold"),
        "avg_king_distance": verdict.get("avg_king_distance"),
        "avg_challenger_distance": verdict.get("avg_challenger_distance"),
        "metric": SPEC.duel.metric,
        "n_clips": verdict.get("n_clips"),
        "block": entry.block,
        # The eval server stamps produced_at OUTSIDE the digested surface, so two
        # validators that agree still produce identical verdict_digests.
        "timestamp": verdict.get("produced_at"),
        "early_stopped": bool(verdict.get("early_stopped")),
        # The audit anchors: anyone can now check that two validators graded the
        # same exam (clip_keys_digest) and reached the same call (verdict_digest).
        "verdict_digest": verdict.get("verdict_digest"),
        "consensus_digest": (verdict.get("audit") or {}).get("consensus_digest"),
        "clip_keys_digest": ((verdict.get("audit") or {}).get("corpus") or {}).get("clip_keys_digest"),
    }


def _build_queue(state: KingState, entries: list[ChallengerEntry], uid_map: dict[str, int]) -> list[dict]:
    """Pending-challenger view for the dashboard (excludes the reigning king)."""
    queue = []
    for e in entries:
        if _is_current_king(state, e):
            continue
        seen = _seen_key(e.hotkey, e.model_digest) in state.seen_hotkeys
        queue.append({
            "hotkey": e.hotkey,
            "uid": uid_map.get(e.hotkey),
            "model_repo": e.model_repo,
            "model_digest": e.model_digest,
            "block": e.block,
            "status": "seen" if seen else "unseen",
        })
    return queue


async def _publish_dashboard(
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    queue: list[dict],
) -> None:
    """Build + publish the public dashboard.json snapshot (best-effort)."""
    try:
        payload = build_dashboard(
            state,
            uid_map,
            chain_meta={
                "name": CHAIN_NAME,
                "seed_repo": SEED_REPO,
                "seed_digest": SEED_DIGEST,
                "netuid": NETUID,
            },
            duel_params={
                "metric": SPEC.duel.metric,
                "metric_device": SPEC.duel.metric_device,
                "n_clips": SPEC.duel.n_clips,
                "delta_threshold": SPEC.duel.delta_threshold,
                "alpha": SPEC.duel.alpha,
                "n_bootstrap": SPEC.duel.n_bootstrap,
                "consensus_digest": CONSENSUS_DIGEST,
                "corpus_pinned": SPEC.corpus.pinned,
            },
            updated_at=datetime.now(timezone.utc).isoformat(),
            queue=queue,
        )
        await publish_dashboard(store, payload)
    except Exception as e:
        log(f"Dashboard publish failed: {e}", "warn")


def _error_history_entry(
    entry: ChallengerEntry,
    failure: DuelFailure,
    uid_map: dict[str, int],
    row: dict,
) -> dict:
    """A dashboard history row for a QUARANTINED challenger.

    The frontend has typed `DuelHistoryEntry.error` since day one and it never
    had a producer: dispatch failures were only logged, so the dashboard could
    never explain why a miner's model was ignored.
    """
    return {
        "challenge_id": f"block-{entry.block}",
        "hotkey": entry.hotkey,
        "uid": uid_map.get(entry.hotkey),
        "model_repo": entry.model_repo,
        "model_digest": entry.model_digest,
        "accepted": False,
        "verdict": "error",
        "error": (failure.detail or "")[:500],
        "error_reason": failure.reason,
        "attempts": row.get("attempts"),
        "metric": SPEC.duel.metric,
        "block": entry.block,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _note_failure(
    state: KingState,
    store: JsonBucketStore,
    uid_map: dict[str, int],
    entry: ChallengerEntry,
    key: str,
    failure: DuelFailure,
    block: int,
) -> None:
    """Record a failed attempt; quarantine + surface it when the budget is spent."""
    row = state.record_failure(key, block=block, failure=failure)

    if not row["quarantined"]:
        state.stats["transient_errors"] = state.stats.get("transient_errors", 0) + 1
        state.touch()
        log(
            f"Duel failed ({failure.reason}) for {entry.hotkey[:12]}...: "
            f"attempt {row['attempts']}/{MAX_DUEL_ATTEMPTS}, retry at block {row['next_retry_block']}",
            "warn",
        )
    else:
        # stats.failed counts QUARANTINED challengers, once each. Retries go to
        # transient_errors so the dashboard number doesn't inflate with backoff.
        state.stats["failed"] = state.stats.get("failed", 0) + 1
        state.touch()
        state.record_duel(_error_history_entry(entry, failure, uid_map, row))
        state.mark_seen(key)  # never retried
        log(
            f"Challenger {entry.hotkey[:12]}... QUARANTINED "
            f"({row['quarantine_reason']}: {failure.reason})",
            "error",
        )

    await state.flush(store)


async def process_challengers(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    entries: list[ChallengerEntry],
    block: int,
) -> None:
    """Duel each new challenger; crown the winners and refresh weights.

    A failing challenger must NEVER block the ones behind it. This loop used to
    ``return`` on any duel error, so one challenger whose repo 404s (or whose
    weights crash the pipeline) permanently blocked every later challenger — a
    free griefing vector. Now:

      * BUSY  -> ``break``    (a property of the SERVER: continuing would just 409)
      * error -> ``continue`` (a property of the CHALLENGER: everyone else proceeds)
    """
    # An unpinned corpus is a subnet-wide condition, not a per-challenger one: the
    # exam itself doesn't exist yet. Refuse every duel and burn, exactly as with a
    # missing seed_digest — and say so once, not once per challenger.
    try:
        SPEC.require_duel_ready()
    except ConsensusConfigError as e:
        log(str(e), "error")
        state.degraded = "corpus_unpinned"
        return

    for entry in entries:
        key = _seen_key(entry.hotkey, entry.model_digest)

        if key in state.seen_hotkeys or _is_current_king(state, entry):
            continue
        if state.is_quarantined(key):
            continue  # this artifact can never be evaluated
        if block < state.next_retry_block(key):
            continue  # backoff has not elapsed

        if not state.king:
            # No king AND no chain.toml seed_digest. We refuse to crown an
            # unevaluated model — the burn path already exists and is correct
            # (king_hotkeys -> [] -> weight_targets -> burn UID 0). The old code
            # short-circuited it by crowning the first reveal it happened to see.
            log(
                "No king and no chain.toml seed_digest — refusing to crown an "
                "unevaluated challenger. Burning to UID 0 until seed_digest is pinned.",
                "error",
            )
            state.degraded = "no_seed_digest"
            break

        try:
            block_hash = await subtensor.get_block_hash(entry.block)
            log(
                f"Dueling challenger {entry.hotkey[:12]}... vs king "
                f"{state.king.get('hotkey','')[:12] or 'base'}...",
                "info",
            )
            verdict = await dispatch_duel(entry, state.king, block_hash)
        except EvalBusy:
            log("Eval server busy; remaining challengers deferred to next tick", "warn")
            break
        except Exception as e:  # noqa: BLE001 — deliberate: classify, never crash the tick
            failure = classify(e)

            if failure.is_local:
                # OUR fault: a corpus that doesn't match the pinned manifest, an eval
                # box on a stale chain.toml. It would fail identically for every
                # challenger, so it is not evidence about this one — and charging it
                # to the ledger would, after four attempts, quarantine every honest
                # miner on the subnet for our own misconfiguration. Stop dueling,
                # say so loudly, and burn until an operator fixes it.
                log(f"LOCAL FAULT ({failure.reason}) — this validator cannot duel: "
                    f"{failure.detail}", "error")
                state.degraded = failure.reason
                break

            await _note_failure(state, store, uid_map, entry, key, failure, block)
            continue  # <<< THE FIX: one bad challenger no longer blocks the rest

        # A completed duel clears the artifact's failure history.
        state.clear_attempts(key)
        state.mark_seen(key)
        state.record_duel(_duel_history_entry(entry, verdict, uid_map))

        if verdict.get("accepted"):
            state.king, state.king_chain = K.crown(
                state.king, state.king_chain, hotkey=entry.hotkey, model_repo=entry.model_repo,
                model_digest=entry.model_digest, block=entry.block,
                challenge_id=verdict.get("challenge_id") or f"block-{entry.block}",
            )
            state.stats["accepted"] = state.stats.get("accepted", 0) + 1
            state.touch()
            await state.flush(store)
            log(f"Challenger {entry.hotkey[:12]}... CROWNED (lcb={verdict.get('lcb')})", "success")
            await maybe_set_weights(subtensor, wallet, state, uid_map, store, force=True)
        else:
            state.stats["rejected"] = state.stats.get("rejected", 0) + 1
            state.touch()
            await state.flush(store)
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
    # Hotkeys with several quarantined artifacts are dropped at scan time — this
    # finally wires reveal_scan's long-dead `blacklist=` hook.
    entries = scan_reveals(commits, blacklist=state.banned_hotkeys())
    if entries:
        await process_challengers(subtensor, wallet, state, uid_map, store, entries, block)

    # Periodic weight refresh (also keeps the king chain aligned as UIDs change).
    await maybe_set_weights(subtensor, wallet, state, uid_map, store)

    # Publish the public dashboard snapshot for the website.
    await _publish_dashboard(state, uid_map, store, _build_queue(state, entries, uid_map))


async def main() -> None:
    """Run the king-of-the-hill validator loop."""
    log_header("Leoma Validator Starting (king of the hill)")

    if not R2_OWN_BUCKET:
        log("R2_OWN_BUCKET not set; validator disabled (cannot persist king state)", "error")
        return

    subtensor = bt.AsyncSubtensor(network=NETWORK)
    wallet = bt.Wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
    log(f"Wallet: {WALLET_NAME}/{HOTKEY_NAME}  Network: {NETWORK}  NetUID: {NETUID}", "info")
    log(
        f"Eval server: {EVAL_SERVER_URL}  metric={SPEC.duel.metric}@{SPEC.duel.metric_device} "
        f"n_clips={SPEC.duel.n_clips} delta={SPEC.duel.delta_threshold}",
        "info",
    )
    log(f"Consensus digest: {CONSENSUS_DIGEST}  (eval code: {eval_code_digest()})", "info")

    own_client = create_own_write_client()
    await ensure_bucket_exists(own_client, R2_OWN_BUCKET)
    store = JsonBucketStore(own_client, R2_OWN_BUCKET)

    # A validator that cannot read its state must NOT run: on a chain-derived
    # system, running on blank state re-duels every past challenger and re-seeds
    # genesis over the reigning king — and the first flush would then overwrite
    # the good bucket state with the blank one.
    try:
        state = await KingState.load(store)
    except (StoreUnavailable, StateInconsistent) as e:
        if os.environ.get("LEOMA_FORCE_FRESH_STATE") == "1":
            log(f"State unreadable ({e}); LEOMA_FORCE_FRESH_STATE=1 — starting BLANK", "warn")
            state = KingState()
        else:
            log(f"Cannot read validator state: {e}", "critical")
            log("Refusing to start on unknown state. Restore the bucket, or set "
                "LEOMA_FORCE_FRESH_STATE=1 to deliberately start blank.", "critical")
            raise SystemExit(1)

    log(f"Loaded king: {(state.king or {}).get('model_repo', 'none')} chain={len(state.king_chain)}", "info")

    # Re-assert weights immediately on startup. Without this, a restarted
    # validator with a recently-persisted last_weight_block sits idle for up to a
    # full WEIGHT_INTERVAL (~1h) before it sets weights again.
    try:
        block = await subtensor.get_current_block()
        ensure_genesis_king(state, block)
        uid_map = await refresh_uid_map(subtensor)
        await maybe_set_weights(subtensor, wallet, state, uid_map, store, force=True)
    except Exception as e:
        log(f"Startup weight-set failed (will retry on the first tick): {e}", "warn")

    while True:
        try:
            await tick(subtensor, wallet, state, store)
        except (StoreUnavailable, StateInconsistent) as e:
            # Losing the store mid-run means we can no longer persist crowns.
            log(f"State store unavailable: {e}", "critical")
            log_exception("State store unavailable — exiting for supervisor restart", e)
            raise SystemExit(1)
        except Exception as e:
            log(f"Validator tick error: {e}", "error")
            log_exception("Validator tick error", e)
        await asyncio.sleep(CHALLENGE_POLL_INTERVAL)


def main_sync() -> None:
    """Synchronous entry point for CLI."""
    asyncio.run(main())
