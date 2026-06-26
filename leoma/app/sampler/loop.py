"""
Per-validator sampler loop (decentralized task generation).

Each permissioned validator runs this loop. Whose turn it is to sample is computed LOCALLY from the
current chain block + the hardcoded validator allowlist (no owner-api): on its turn it samples its
locally-validated miners once for the window (``task_id == rotation_index``), self-evaluates, and
publishes the artifacts + signed verdicts to its OWN bucket. It then best-effort announces and
dual-reports to the owner-api for the dashboard only. Only one validator samples per window.
"""
import os
import asyncio

import bittensor as bt
from google import genai

from leoma.bootstrap import (
    GEMINI_API_KEY,
    R2_OWN_BUCKET,
    WALLET_NAME,
    HOTKEY_NAME,
    NETWORK,
)
from leoma.bootstrap import emit_log as log, emit_header as log_header, log_exception
from leoma.infra.storage_backend import (
    create_source_read_client,
    create_own_write_client,
    ensure_bucket_exists,
    upload_task_artifacts,
    upload_evaluation_result_json,
)
from leoma.app.sampler.core import sample_once, build_generation_miners, cleanup
from leoma.app.evaluator.main import evaluate_sampled_task
from leoma.app.validator.miner_validation import valid_miners as local_valid_miners

API_URL = os.environ.get("API_URL", "https://api.leoma.ai")
# How often to check the rotation schedule. The turn lasts a whole window (default 100
# blocks ≈ 20 min), so polling once a minute samples promptly without spamming the API.
SAMPLER_POLL_INTERVAL = int(os.environ.get("SAMPLER_POLL_INTERVAL", "60"))


async def run_sampler_loop() -> None:
    """Sample miners on this validator's rotation turn and publish to its own bucket."""
    from leoma.infra.remote_api import create_api_client_from_wallet

    if not R2_OWN_BUCKET:
        log("R2_OWN_BUCKET not set; sampler disabled (cannot publish tasks)", "error")
        return

    gemini_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        log("Sampler requires GEMINI_API_KEY (clip description)", "error")
        return
    gemini_client = genai.Client(api_key=gemini_key)

    try:
        source_client = create_source_read_client()
        own_client = create_own_write_client()
    except ValueError as e:
        log(f"Sampler cannot create storage clients: {e}", "error")
        return
    await ensure_bucket_exists(own_client, R2_OWN_BUCKET)

    api_client = create_api_client_from_wallet(
        wallet_name=WALLET_NAME, hotkey_name=HOTKEY_NAME, api_url=API_URL
    )

    from leoma.app.validator.rotation_local import LocalRotation
    subtensor = bt.AsyncSubtensor(network=NETWORK)
    local_rotation = LocalRotation(subtensor, api_client.hotkey)

    log_header("Validator Sampler Starting")
    log(f"Own result bucket: {R2_OWN_BUCKET}", "info")
    log(f"Rotation: hardcoded allowlist (local); poll={SAMPLER_POLL_INTERVAL}s", "info")

    last_sampled_index: int | None = None

    while True:
        try:
            # Whose turn is computed locally from the chain block + hardcoded allowlist (no owner-api).
            view = await local_rotation.whose_turn()
            if view is None:
                # Chain / allowlist unreadable right now (or our hotkey isn't allowlisted yet); idle.
                await asyncio.sleep(SAMPLER_POLL_INTERVAL)
                continue
            rotation_index = view.rotation_index
            if not view.is_your_turn:
                await asyncio.sleep(SAMPLER_POLL_INTERVAL)
                continue
            if rotation_index == last_sampled_index:
                await asyncio.sleep(SAMPLER_POLL_INTERVAL)
                continue
            if view.produced:
                # Already produced this window (e.g. we restarted after sampling it); stand down.
                last_sampled_index = rotation_index
                await asyncio.sleep(SAMPLER_POLL_INTERVAL)
                continue

            log_header(f"Sampler turn – task_id={rotation_index} (failover step {view.failover_step})")

            # Locally-validated miners (this validator's own validation, not the owner-api's).
            valid_miners = local_valid_miners()
            if not valid_miners:
                log("No locally-validated miners yet; skipping this window", "warn")
                last_sampled_index = rotation_index
                await asyncio.sleep(SAMPLER_POLL_INTERVAL)
                continue

            miners = build_generation_miners(
                [
                    {
                        "hotkey": m.hotkey,
                        "chute_id": m.chute_id,
                        "chute_slug": m.chute_slug,
                        "model_name": m.model_name,
                        "model_revision": m.model_revision,
                        "model_hash": m.model_hash,
                        "block": m.block,
                    }
                    for m in valid_miners
                ]
            )
            log(f"Sampling {len(miners)} valid miners", "info")

            result = await sample_once(rotation_index, miners, source_client, gemini_client)
            if result is None:
                log("Sampling produced no usable task this window", "warn")
                last_sampled_index = rotation_index
                await asyncio.sleep(SAMPLER_POLL_INTERVAL)
                continue

            try:
                # 1. Publish the task artifacts (videos + metadata) to our own bucket.
                await upload_task_artifacts(
                    own_client,
                    R2_OWN_BUCKET,
                    result.task_id,
                    result.clip_path,
                    result.frame_path,
                    result.metadata,
                    result.miner_paths,
                )
                # 2. Self-evaluate our own task (no cross-validation) while videos are local.
                samples_payload, eval_entries = await evaluate_sampled_task(
                    gemini_client, result, R2_OWN_BUCKET
                )
                # 3. Publish our verdicts to our own bucket (peers read these to aggregate).
                if eval_entries:
                    signature = api_client.sign_evaluation_payload(eval_entries)
                    await upload_evaluation_result_json(
                        own_client,
                        R2_OWN_BUCKET,
                        result.task_id,
                        api_client.hotkey,
                        eval_entries,
                        signature=signature,
                    )
                    # 4. Best-effort dual-report to the owner-api for the dashboard (the dashboard
                    #    derives the produced-task window from these samples; not consensus-critical).
                    try:
                        await api_client.submit_samples_batch(
                            samples_payload, evaluation_signature=signature
                        )
                    except Exception as e:
                        log(f"Dashboard dual-report failed: {e}", "warn")
                last_sampled_index = rotation_index
                log(
                    f"Task {result.task_id}: published + self-evaluated "
                    f"({len(eval_entries)}/{len(result.miner_paths)} miners)",
                    "success",
                )
            finally:
                cleanup(result)

        except Exception as e:
            log(f"Sampler error: {e}", "error")
            log_exception("Sampler error", e)

        await asyncio.sleep(SAMPLER_POLL_INTERVAL)
