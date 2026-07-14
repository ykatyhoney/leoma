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
    ErrorClass,
    EvalBusy,
    EvalJobFailed,
    classify,
    classify_remote,
)
from leoma.app.validator import rate_limit as RL
from leoma.app.validator.prescreen import prescreen
from leoma.app.validator.state_store import MAX_DUEL_ATTEMPTS
from leoma.app.validator import king as K

EVAL_SERVER_URL = os.environ.get("EVAL_SERVER_URL", "http://localhost:9000")
CHALLENGE_POLL_INTERVAL = int(os.environ.get("LEOMA_CHALLENGE_POLL_INTERVAL", "60"))

# The pre-dispatch architecture check. On by default: it is the difference between a
# bad model costing ~5 seconds and costing hours of GPU with the lock held. Off only
# as an operator escape hatch if the Hub's config endpoint is misbehaving.
PRESCREEN_ENABLED = os.environ.get("LEOMA_PRESCREEN", "1") != "0"

# The duel parameters (metric, n_clips, delta, alpha, bootstrap, generation knobs)
# are NOT read from the environment any more. They are the consensus surface, and
# they live in chain.toml as `SPEC`. An env var is per-box; a per-box exam is not
# consensus. LEOMA_DUEL_METRIC / _N_CLIPS / _DELTA_THRESHOLD / _ALPHA / _N_BOOTSTRAP
# are deliberately gone — setting one no longer does anything, which is the point.

# Every call to the eval server is now short: dispatch, or poll. Neither waits for a
# duel. LEOMA_EVAL_TIMEOUT is gone — the validator no longer has to guess how long a
# 14B video duel takes, which is a guess it could only ever get wrong. The wall-clock
# bound lives on the eval server, where the phase information actually is.
_EVAL_CONNECT_TIMEOUT = 30.0
_EVAL_POLL_TIMEOUT = 60.0


def _seen_key(hotkey: str, digest: str) -> str:
    return f"{hotkey}|{digest}"


def _is_current_king(state: KingState, entry: ChallengerEntry) -> bool:
    k = state.king or {}
    return k.get("hotkey") == entry.hotkey and k.get("model_digest") == entry.model_digest


def _copies_a_king(state: KingState, entry: ChallengerEntry) -> Optional[str]:
    """Is this a *different* hotkey submitting a king's exact weights? Returns whose.

    ``_is_current_king`` requires the hotkey to match as well as the digest — which is
    right for "don't re-duel the incumbent against itself", but it left a hole: a
    **different** hotkey re-committing the king's exact digest was treated as a novel
    challenger and handed a full multi-hour duel on the subnet's only GPU. It would
    then tie the king exactly and lose (the threshold requires strictly better), so it
    could never actually win — but it cost hours of GPU every time, for free, and
    could be repeated indefinitely.

    The digest is content-addressed, so identical digest means identical bytes. There
    is nothing to evaluate. Reject before dispatch.

    Prior kings count too: submitting a *previous* king's weights is the same
    plagiarism, and it has already been beaten by the current king.
    """
    for who in ([state.king] if state.king else []) + list(state.king_chain or []):
        if not who:
            continue
        if who.get("model_digest") == entry.model_digest and who.get("hotkey") != entry.hotkey:
            return str(who.get("hotkey", ""))
    return None


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


async def start_duel(entry: ChallengerEntry, king: dict, block_hash: str) -> str:
    """POST a duel and return its ``eval_id``. Returns in **seconds**, not hours.

    The validator used to dispatch a duel and then *stream it to completion* — an
    ``await`` that blocked the entire tick loop for as long as the duel took. For a
    32-clip duel of two 14B video models, that is hours in which the validator sets no
    weights, publishes no dashboard, and looks dead to the chain. The duel was not the
    only thing that stopped; the validator did.

    Now the tick is bounded: dispatch, persist the slot, return. The verdict is
    collected on a later tick by :func:`settle_inflight`.

    Sends the **whole pinned consensus surface** (``SPEC``) rather than a handful of
    loose knobs. Raises ``EvalBusy`` on 409 (a property of the *server*) and
    ``EvalJobFailed`` otherwise (a property of the *challenger*, or of us).
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
    timeout = httpx.Timeout(_EVAL_POLL_TIMEOUT, connect=_EVAL_CONNECT_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await preflight_eval_server(client)

        resp = await client.post(f"{EVAL_SERVER_URL}/eval", json=payload)
        if resp.status_code == 409:
            raise EvalBusy("eval server is already running a duel")
        resp.raise_for_status()
        return resp.json()["eval_id"]


async def poll_duel(eval_id: str) -> dict:
    """Ask the eval server how a dispatched duel is going.

    Returns ``{status, phase, verdict, error, reason}``. ``status`` is one of
    ``running`` / ``done`` / ``error`` / ``cancelled``.

    A **404** means the eval box no longer knows about this duel — it restarted, or
    the job aged out. That is transient and ours to retry; it must never be read as a
    verdict, and it must never be charged to the miner.
    """
    import httpx

    timeout = httpx.Timeout(_EVAL_POLL_TIMEOUT, connect=_EVAL_CONNECT_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(f"{EVAL_SERVER_URL}/eval/{eval_id}")
        if resp.status_code == 404:
            raise EvalJobFailed(
                f"eval server no longer knows about {eval_id} (it restarted, or the job "
                "aged out). The duel is lost; it will be re-dispatched.",
                reason="eval_job_lost",
            )
        resp.raise_for_status()
        return resp.json()


def _verified_verdict(verdict: Optional[dict]) -> dict:
    """Fail closed unless the box ran EXACTLY the exam we set.

    This is the check that catches "the field wasn't sent, so the box used its own
    default" — the entire bug class that made two honest validators disagree. It runs
    **before crowning**, so a verdict produced under different parameters can never
    take the crown.
    """
    if not verdict:
        raise EvalJobFailed(
            "eval server reported success but returned no verdict", reason="stream_no_terminal"
        )
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
        # Why a challenger that beat the king still didn't take the crown: it copied
        # the king, or it beat a mediocre king only by holding the conditioning frame.
        "rejected_by": verdict.get("rejected_by"),
        "gates": verdict.get("gates"),
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


def _slot_entry(slot: dict) -> ChallengerEntry:
    """Rebuild the challenger from the persisted slot.

    Deliberately reconstructed from the slot rather than looked up in this tick's
    reveals: the slot is the authoritative record of *what we dispatched*. A reveal
    that has since scrolled out of the commitment window must still be settled — the
    duel ran, and the miner is owed its verdict.
    """
    return ChallengerEntry(
        hotkey=slot["hotkey"],
        model_repo=slot["model_repo"],
        model_digest=slot["model_digest"],
        block=int(slot.get("block", 0)),
    )


async def _local_fault(state: KingState, failure) -> None:
    """OUR fault: stop dueling, say so loudly, and burn until an operator fixes it.

    A corpus that doesn't match the pinned manifest, an eval box on a stale
    chain.toml. It would fail identically for *every* challenger — including the
    reigning king — so it is not evidence about any one of them. And charging it to
    the ledger would, after four attempts, quarantine every honest miner on the
    subnet for our own misconfiguration.
    """
    log(f"LOCAL FAULT ({failure.reason}) — this validator cannot duel: {failure.detail}", "error")
    state.degraded = failure.reason


async def settle_inflight(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    block: int,
) -> bool:
    """Collect a dispatched duel's verdict. Returns True once the slot is free.

    **Restart-safe.** The slot is persisted, so a validator that restarts mid-duel
    re-attaches to the job it left running instead of orphaning it. Before this, a
    restart meant the eval box spent hours on a duel nobody would ever read, and
    answered 409 to everyone else the entire time.
    """
    slot = state.inflight
    if not slot:
        return True

    entry = _slot_entry(slot)
    key = _seen_key(entry.hotkey, entry.model_digest)

    try:
        result = await poll_duel(slot["eval_id"])
    except Exception as e:  # noqa: BLE001 — classify, never crash the tick
        failure = classify(e)
        if failure.is_local:
            await _local_fault(state, failure)
            return False        # keep the slot; the duel may still be fine once we are
        state.inflight = None
        state.touch()
        await _note_failure(state, store, uid_map, entry, key, failure, block)
        return True

    status = result.get("status")
    if status == "running":
        log(f"Duel {slot['eval_id']} still running (phase={result.get('phase')})", "info")
        return False

    # Terminal, one way or another: the slot is free from here on.
    state.inflight = None
    state.touch()

    if status != "done":
        failure = classify_remote(str(result.get("error") or status), str(result.get("reason") or ""))
        if failure.is_local:
            await _local_fault(state, failure)
            await state.flush(store)
            return True
        await _note_failure(state, store, uid_map, entry, key, failure, block)
        return True

    try:
        verdict = _verified_verdict(result.get("verdict"))
    except Exception as e:  # noqa: BLE001
        failure = classify(e)
        if failure.is_local:
            await _local_fault(state, failure)
            await state.flush(store)
            return True
        await _note_failure(state, store, uid_map, entry, key, failure, block)
        return True

    # The king may have changed while this duel was in flight (a crown from a
    # different challenger, or an operator re-seed). The verdict measured the
    # challenger against a king that no longer reigns, so it cannot be acted on —
    # and it must NOT be marked seen, or the challenger would never get a fair duel
    # against the king it actually has to beat.
    current_king_digest = (state.king or {}).get("model_digest")
    if slot.get("king_digest") != current_king_digest:
        log(
            f"King changed while {entry.hotkey[:12]}...'s duel was in flight — discarding "
            "the stale verdict; it will be re-dueled against the current king.",
            "warn",
        )
        await state.flush(store)
        return True

    # A completed duel clears the artifact's failure history and charges the hotkey's
    # GPU budget — whatever the outcome. The duel cost hours either way.
    state.clear_attempts(key)
    state.mark_seen(key)
    RL.record_verdict(state.duels, entry.hotkey, king=state.king, block=block)

    # A gate rejection is a strike; LOSING a duel fairly is not. A miner whose honest
    # model simply isn't good enough has done nothing wrong, and penalizing them for
    # trying would deter exactly the people the subnet exists to attract.
    rejected_by = str(verdict.get("rejected_by") or "")
    if rejected_by:
        RL.record_strike(state.duels, entry.hotkey, rejected_by)

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

    return True


async def process_challengers(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    entries: list[ChallengerEntry],
    block: int,
) -> None:
    """Settle the in-flight duel, then dispatch at most one more. Returns in seconds.

    This used to duel every challenger *inline*, streaming each one to completion —
    so a single tick could take hours, during which the validator set no weights and
    published no dashboard. The duel wasn't the only thing blocked; the validator was.

    Now the tick is bounded: settle whatever is in flight, dispatch the next one, come
    back. Everything else in ``tick`` (weights, dashboard) therefore runs *every* tick,
    even mid-duel.

    A failing challenger must never block the ones behind it, so the dispatch loop
    still distinguishes:

      * BUSY  -> ``break``    (a property of the SERVER: continuing would just 409)
      * LOCAL -> ``break``    (a property of US: every challenger would fail the same)
      * error -> ``continue`` (a property of the CHALLENGER: everyone else proceeds)
    """
    # An unpinned corpus is a subnet-wide condition, not a per-challenger one: the exam
    # itself does not exist yet. Refuse every duel and burn, exactly as with a missing
    # seed_digest — and say so once, not once per challenger.
    try:
        SPEC.require_duel_ready()
    except ConsensusConfigError as e:
        log(str(e), "error")
        state.degraded = "corpus_unpinned"
        return

    if not await settle_inflight(subtensor, wallet, state, uid_map, store, block):
        return  # a duel is still running; the tick moves on to weights + dashboard

    for entry in entries:
        key = _seen_key(entry.hotkey, entry.model_digest)

        if key in state.seen_hotkeys or _is_current_king(state, entry):
            continue
        if state.is_quarantined(key):
            continue  # this artifact can never be evaluated
        if block < state.next_retry_block(key):
            continue  # backoff has not elapsed

        # A different hotkey submitting a king's exact weights. Content-addressed
        # digests mean identical digest = identical bytes: there is nothing to
        # evaluate, and dispatching it would burn hours of GPU on a model that
        # already reigns. It could never win anyway (the threshold demands strictly
        # better, and a copy ties exactly) — it was simply free to repeat.
        plagiarized = _copies_a_king(state, entry)
        if plagiarized:
            failure = DuelFailure(
                ErrorClass.PERMANENT, "copy_of_king",
                f"{entry.model_digest} is {plagiarized[:12]}...'s model, repackaged. "
                "Copying a king is not an improvement on it.",
            )
            RL.record_strike(state.duels, entry.hotkey, failure.reason)
            await _note_failure(state, store, uid_map, entry, key, failure, block)
            continue

        # The GPU is the scarcest thing in the subnet. A hotkey that has just been
        # dueled, or has already spent its allowance against this king, waits.
        limited = RL.check(state.duels, entry.hotkey, king=state.king, block=block)
        if limited:
            log(f"Rate-limited {entry.hotkey[:12]}...: {limited}", "info")
            continue

        if not state.king:
            # No king AND no chain.toml seed_digest. We refuse to crown an unevaluated
            # model — the burn path already exists and is correct (king_hotkeys -> [] ->
            # weight_targets -> burn UID 0). The old code short-circuited it by crowning
            # the first reveal it happened to see.
            log(
                "No king and no chain.toml seed_digest — refusing to crown an "
                "unevaluated challenger. Burning to UID 0 until seed_digest is pinned.",
                "error",
            )
            state.degraded = "no_seed_digest"
            return

        # Configs only (~200 KB), on this box, before the GPU is touched. A
        # wrong-architecture model costs seconds here instead of hours of download +
        # load, during which the eval server's lock is held and every honest
        # challenger waits behind it.
        if PRESCREEN_ENABLED:
            try:
                await asyncio.to_thread(prescreen, entry.model_repo, entry.model_digest)
            except Exception as e:  # noqa: BLE001
                failure = classify(e)
                if failure.is_local:
                    await _local_fault(state, failure)
                    return
                if failure.is_permanent:
                    RL.record_strike(state.duels, entry.hotkey, failure.reason)
                await _note_failure(state, store, uid_map, entry, key, failure, block)
                continue

        try:
            block_hash = await subtensor.get_block_hash(entry.block)
            log(
                f"Dispatching duel: {entry.hotkey[:12]}... vs king "
                f"{state.king.get('hotkey','')[:12] or 'base'}...",
                "info",
            )
            eval_id = await start_duel(entry, state.king, block_hash)
        except EvalBusy:
            log("Eval server busy; challengers deferred to next tick", "warn")
            return
        except Exception as e:  # noqa: BLE001 — deliberate: classify, never crash the tick
            failure = classify(e)
            if failure.is_local:
                await _local_fault(state, failure)
                return
            await _note_failure(state, store, uid_map, entry, key, failure, block)
            continue  # one bad challenger no longer blocks the rest

        # Persist the slot BEFORE returning: a crash between the POST and the flush
        # would otherwise orphan a duel that is already burning GPU hours, and the
        # eval box would 409 everyone until it finished a job nobody was waiting for.
        state.inflight = {
            "eval_id": eval_id,
            "hotkey": entry.hotkey,
            "model_repo": entry.model_repo,
            "model_digest": entry.model_digest,
            "block": entry.block,
            "king_digest": (state.king or {}).get("model_digest"),
            "dispatched_block": block,
        }
        state.touch()
        await state.flush(store)
        log(f"Duel {eval_id} dispatched; the tick continues (weights + dashboard stay live)", "info")
        return  # one duel at a time — the eval server has one GPU


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
    # Two distinct reasons a hotkey stops being scanned: several unevaluable artifacts
    # (it keeps uploading broken models), or several GATE rejections (it keeps uploading
    # models that should never have been dispatched — copies, wrong architectures).
    # Neither counts a fair loss.
    entries = scan_reveals(
        commits, blacklist=state.banned_hotkeys() | RL.struck_out(state.duels)
    )

    # Bounded: settles the in-flight duel and dispatches at most one more. Always
    # returns in seconds, even when a multi-hour duel is running on the GPU box.
    await process_challengers(subtensor, wallet, state, uid_map, store, entries, block)

    # These two now run on EVERY tick, including mid-duel. They used to be starved for
    # hours behind an inline duel: no weights (the chain concludes the validator is
    # dead) and a dashboard frozen on whatever was true when the duel started.
    await maybe_set_weights(subtensor, wallet, state, uid_map, store)
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
