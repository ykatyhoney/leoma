"""Miner command implementations.

Two steps replace the old Chutes deploy:
  push   — upload the model-weights folder to Hippius Hub -> immutable ModelRef
  commit — reveal ``v4|<repo>|<digest>|<hotkey>`` on-chain for validator discovery

Miners no longer host inference; validators download these weights and run the
model themselves. The Hippius repo name must start with "leoma" and end with the
miner's hotkey ss58 (the validator enforces this on discovery).
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any

from leoma.bootstrap import NETUID, WALLET_NAME, HOTKEY_NAME, NETWORK
from leoma.bootstrap import emit_log as log
from leoma.infra.model_store import ModelRef, build_reveal_v4, upload_model_folder
from leoma.infra.commit_parser import validate_repo_name


async def push_command(
    model_dir: str,
    repo: str,
    revision: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a local model-weights folder to Hippius Hub.

    Returns a result dict with the immutable ``repo`` + ``digest`` to commit.
    """
    try:
        # hippius_hub.upload_folder is blocking; run it off the event loop.
        ref = await asyncio.to_thread(
            upload_model_folder,
            model_dir,
            repo,
            revision,
            commit_message,
        )
        log(f"Uploaded {model_dir} -> {ref.immutable_ref}", "success")
        return {
            "success": True,
            "repo": ref.repo,
            "digest": ref.digest,
            "immutable_ref": ref.immutable_ref,
        }
    except Exception as e:
        log(f"Upload failed: {e}", "error")
        return {"success": False, "error": str(e)}


async def commit_command(
    repo: str,
    digest: str,
    coldkey: Optional[str] = None,
    hotkey: Optional[str] = None,
) -> Dict[str, Any]:
    """Commit the model reveal to the chain. Returns a result dict with success status."""
    import bittensor as bt

    cold = coldkey or WALLET_NAME
    hot = hotkey or HOTKEY_NAME
    wallet = bt.Wallet(name=cold, hotkey=hot)
    author_hotkey = wallet.hotkey.ss58_address

    ok, reason = validate_repo_name(repo, hotkey=author_hotkey)
    if not ok:
        log(f"Repo name {repo!r} rejected: {reason} (must start with 'leoma' and end with your hotkey)", "error")
        return {"success": False, "error": reason}

    try:
        ref = ModelRef(repo, digest)
        payload = build_reveal_v4(ref, author_hotkey)
    except ValueError as e:
        log(f"Invalid model reference: {e}", "error")
        return {"success": False, "error": str(e)}

    log(f"Committing reveal for {ref.immutable_ref}", "info")
    log(f"Using wallet hotkey: {author_hotkey[:16]}...", "info")

    async def _commit() -> bool:
        subtensor = bt.AsyncSubtensor(network=NETWORK)
        log(f"Subtensor network configured to {NETWORK}", "info")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                await subtensor.set_reveal_commitment(
                    wallet=wallet,
                    netuid=NETUID,
                    data=payload,
                    blocks_until_reveal=1,
                )
                return True
            except Exception as e:
                if "SpaceLimitExceeded" in str(e):
                    log("Space limit exceeded, waiting for next block...", "warn")
                    await asyncio.sleep(12)
                elif attempt < max_retries - 1:
                    log(f"Commit attempt {attempt + 1} failed: {e}", "warn")
                    await asyncio.sleep(6)
                else:
                    raise
        return False

    try:
        success = await _commit()
        if success:
            log("Commit successful", "success")
            return {"success": True, "repo": ref.repo, "digest": ref.digest}
        log("Commit failed", "error")
        return {"success": False, "error": "Commit failed after retries"}
    except Exception as e:
        log(f"Commit failed: {e}", "error")
        return {"success": False, "error": str(e)}
