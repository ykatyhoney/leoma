"""
Validator service for Leoma.

Runs two loops in one process:
1. Evaluator: poll GET /tasks/latest, download task from object storage (R2 or Hippius), run Gemini per miner, POST results to API.
2. Weight-setter: at each epoch boundary, GET /weights from API, set top-ranked-only weights on-chain
   (weight 1.0 for winner_uid, 0 for others; if no top-ranked miner, UID 0 to burn alpha).
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
    SAMPLES_BUCKET,
)
from leoma.bootstrap import emit_log as log, emit_header as log_header
from leoma.app.evaluator.main import run_evaluator_loop


# API configuration
API_URL = os.environ.get("API_URL", "https://api.leoma.ai")


def _build_weight_payload(winner_uid: int) -> tuple[List[int], List[float]]:
    """Build (uids, weights) for top-ranked-only weighting: only winner_uid gets 1.0. If winner_uid=0, burn alpha."""
    return [winner_uid], [1.0]


async def run_epoch(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    block: int,
) -> None:
    """Get winner_uid from API /weights, set top-ranked-only weights on-chain. If no top-ranked miner or API fails, UID 0 burns alpha."""
    log(f"[{block}] Fetching weights from API", "info")
    winner_uid = 0
    try:
        from leoma.infra.remote_api import create_api_client_from_wallet
        api_client = create_api_client_from_wallet(
            wallet_name=WALLET_NAME,
            hotkey_name=HOTKEY_NAME,
            api_url=API_URL,
        )
        try:
            data = await api_client.get_weights()
            winner_uid = int(data.get("winner_uid", 0))
            miners = data.get("miners") or []
            if miners:
                log(f"[{block}] Weights API: {len(miners)} miners, top_uid(winner_uid)={winner_uid}", "info")
        finally:
            await api_client.close()
    except Exception as e:
        log(f"[{block}] Failed to get weights from API: {e}; setting UID 0 (burn alpha)", "error")

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
    """Wait for epoch boundary and run weight setting.

    Args:
        subtensor: Bittensor async subtensor instance
        wallet: Bittensor wallet for signing transactions

    Returns:
        The epoch number that was processed, or None if waiting
    """
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
    """Main entry point: run evaluator (background) + weight-setting loop."""
    log_header("Leoma Validator Starting (evaluator + weight-setter)")

    if not GEMINI_API_KEY:
        log("Evaluator requires GEMINI_API_KEY", "error")
        return

    log(f"Using centralized API: {API_URL}", "info")
    log(
        f"Evaluator object storage: backend={OBJECT_STORAGE_BACKEND}, samples_bucket={SAMPLES_BUCKET}",
        "info",
    )

    subtensor = bt.AsyncSubtensor(network=NETWORK)
    wallet = bt.Wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)

    log(f"Wallet: {WALLET_NAME}/{HOTKEY_NAME}", "info")
    log(f"Network: {NETWORK}", "info")
    log(f"NetUID: {NETUID}", "info")
    log(f"Epoch length: {EPOCH_LEN} blocks (~{EPOCH_LEN * 12}s)", "info")

    log("Starting evaluator loop in background...", "start")
    evaluator_task = asyncio.create_task(run_evaluator_loop())

    def handle_evaluator_exception(task: asyncio.Task) -> None:
        try:
            exc = task.exception()
            if exc is not None:
                log(f"Background evaluator task failed: {exc}", "error")
        except asyncio.CancelledError:
            log("Background evaluator task was cancelled", "warn")

    evaluator_task.add_done_callback(handle_evaluator_exception)

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
