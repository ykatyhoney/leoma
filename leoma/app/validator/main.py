"""
Validator service for Leoma (fully decentralized).

Runs three loops in one process:
1. Sampler: on this validator's rotation turn (GET /rotation), sample valid miners, upload the
   task to its own R2 bucket, and announce it (POST /tasks/announce).
2. Evaluator: poll GET /tasks/latest, download the task from the sampler's bucket, run Gemini per
   miner, publish results to its own bucket, and dual-report to the API for the dashboard.
3. Weight-setter: at each epoch boundary, read every peer's results from their buckets, aggregate
   with equal weight (one validator = one vote), and set top-ranked-only weights on-chain
   (weight 1.0 for winner_uid, 0 for others; if no eligible miner, UID 0 to burn alpha).
"""

import os
import asyncio
from typing import List

import bittensor as bt

from leoma.bootstrap import (
    NETUID,
    EPOCH_LEN,
    OBJECT_STORAGE_BACKEND,
    GEMINI_API_KEY,
    WALLET_NAME,
    HOTKEY_NAME,
    NETWORK,
)
from leoma.bootstrap import emit_log as log, emit_header as log_header
from leoma.app.sampler.loop import run_sampler_loop
from leoma.app.validator.miner_validation import run_validation_loop


API_URL = os.environ.get("API_URL", "https://api.leoma.ai")


def _build_weight_payload(winner_uid: int) -> tuple[List[int], List[float]]:
    """Build (uids, weights) for top-ranked-only weighting: only winner_uid gets 1.0. If winner_uid=0, burn alpha."""
    return [winner_uid], [1.0]


async def run_epoch(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    block: int,
) -> None:
    """Aggregate peer evaluation results locally (equal weight) and set top-ranked-only weights.

    Reads every permissioned validator's results from their buckets, computes the winner with
    one-vote-per-validator, and sets weight 1.0 on the winner. If aggregation fails or no miner
    is eligible, UID 0 burns alpha.
    """
    log(f"[{block}] Aggregating peer evaluation results locally (equal weight)", "info")
    winner_uid = 0
    try:
        from leoma.app.validator.aggregate_local import compute_local_winner

        # block is the epoch-boundary block (block % EPOCH_LEN == 0), identical across validators
        # running this epoch, so the hardcoded allowlist + block-derived window are identical for all.
        winner_uid, winner_hotkey = await compute_local_winner(epoch_block=block)
        if winner_hotkey:
            log(f"[{block}] Local winner: uid={winner_uid} hotkey={winner_hotkey[:12]}...", "info")
    except Exception as e:
        log(f"[{block}] Local aggregation failed: {e}; setting UID 0 (burn alpha)", "error")

    # Top-ranked-only weighting: set weight 1.0 for winner_uid only (or UID 0 to burn alpha)
    uids, weights = _build_weight_payload(winner_uid)
    try:
        await subtensor.set_weights(wallet=wallet, netuid=NETUID, uids=uids, weights=weights, wait_for_inclusion=True)
        if winner_uid == 0:
            log(f"[{block}] Set weights: no top-ranked miner, UID 0 (burn alpha)", "success")
        else:
            log(f"[{block}] Set weights: top-ranked-only UID {winner_uid}", "success")
    except Exception as e:
        log(f"[{block}] Failed to set weights: {e}", "error")


async def step(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
) -> int | None:
    """Wait for the epoch boundary and run weight setting. Returns the processed epoch, or None if still waiting."""
    current_block = await subtensor.get_current_block()
    current_epoch = current_block // EPOCH_LEN

    if current_block % EPOCH_LEN != 0:
        remaining = EPOCH_LEN - (current_block % EPOCH_LEN)
        wait_time = 12 * remaining
        log(f"Block {current_block}: waiting {remaining} blocks (~{wait_time}s) until epoch", "info")
        await asyncio.sleep(wait_time)
        return None

    log_header(f"Leoma Epoch #{current_epoch} (block {current_block})")

    await run_epoch(subtensor, wallet, current_block)

    wait_time = EPOCH_LEN * 12
    log(f"Waiting {wait_time}s until next epoch", "info")
    await asyncio.sleep(wait_time)

    return current_epoch


async def main() -> None:
    """Main entry point: run the sampler (sample + self-evaluate) + weight-setting loop."""
    log_header("Leoma Validator Starting (sampler + self-evaluator + weight-setter)")

    if not GEMINI_API_KEY:
        log("Sampler/evaluator requires GEMINI_API_KEY", "error")
        return

    log(f"Coordinator API: {API_URL}", "info")
    log(f"Object storage backend: {OBJECT_STORAGE_BACKEND}", "info")

    subtensor = bt.AsyncSubtensor(network=NETWORK)
    wallet = bt.Wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)

    log(f"Wallet: {WALLET_NAME}/{HOTKEY_NAME}", "info")
    log(f"Network: {NETWORK}", "info")
    log(f"NetUID: {NETUID}", "info")
    log(f"Epoch length: {EPOCH_LEN} blocks (~{EPOCH_LEN * 12}s)", "info")

    def _log_background_exception(name: str):
        def _cb(task: asyncio.Task) -> None:
            try:
                exc = task.exception()
                if exc is not None:
                    log(f"Background {name} task failed: {exc}", "error")
            except asyncio.CancelledError:
                log(f"Background {name} task was cancelled", "warn")
        return _cb

    log("Starting miner-validation loop in background (validates miners + reports to owner-api)...", "start")
    validation_task = asyncio.create_task(run_validation_loop())
    validation_task.add_done_callback(_log_background_exception("miner-validation"))

    log("Starting sampler loop in background (samples + self-evaluates own tasks)...", "start")
    sampler_task = asyncio.create_task(run_sampler_loop())
    sampler_task.add_done_callback(_log_background_exception("sampler"))

    log("Starting weight-setting loop...", "start")
    while True:
        try:
            await step(subtensor, wallet)
        except Exception as e:
            log(f"Weight-setting loop error: {e}", "error")
            await asyncio.sleep(10)


def main_sync() -> None:
    """Synchronous entry point for CLI."""
    asyncio.run(main())
