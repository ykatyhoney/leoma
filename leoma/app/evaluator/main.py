"""
Validator evaluator loop: poll GET /tasks/latest, download task from S3, run Gemini, POST results.

Runs as the validator-side process. On startup, initializes evaluated task_id list from object storage
(S3-compatible: Hippius or Cloudflare R2 per OBJECT_STORAGE_BACKEND; lists task_ids that have
evaluation_results/{validator_hotkey}.json). Then in a loop:
- GET /tasks/latest
- If task_id already evaluated, decrement task_id until unevaluated (max 100 in list)
- Download task artifacts (metadata, first_frame, generated_videos per miner)
- For each miner: single-video benchmark evaluation, then POST evaluation to API
- Add task_id to evaluated list
"""

import json
import os
import asyncio
import tempfile
from typing import Set

from google import genai
from openai import AsyncOpenAI

from leoma.bootstrap import (
    GEMINI_API_KEY,
    OBJECT_STORAGE_BACKEND,
    OPENAI_API_KEY,
    SAMPLES_BUCKET,
)
from leoma.bootstrap import emit_log as log, emit_header as log_header, log_exception
from leoma.infra.storage_backend import (
    create_samples_read_client,
    list_evaluated_task_ids,
    download_task_artifacts,
)
from leoma.infra.video_utils import extract_frames, frames_to_base64
from leoma.infra.judge import evaluate_generated_video_async

EVALUATOR_POLL_INTERVAL = int(os.environ.get("EVALUATOR_POLL_INTERVAL", "60"))
EVALUATED_LIST_MAX = 100
API_URL = os.environ.get("API_URL", "https://api.leoma.ai")
EVALUATION_MAX_FRAMES = int(os.environ.get("EVALUATION_MAX_FRAMES", "12"))
EVALUATION_FRAME_FPS = float(os.environ.get("EVALUATION_FRAME_FPS", "3"))
EVALUATION_PASS_THRESHOLD = int(os.environ.get("EVALUATION_PASS_THRESHOLD", "75"))
EVALUATION_CRITICAL_THRESHOLD = int(os.environ.get("EVALUATION_CRITICAL_THRESHOLD", "50"))


def _remove_file(path: str | None) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _remove_directory(path: str | None) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        for f in os.listdir(path):
            _remove_file(os.path.join(path, f))
        os.rmdir(path)
    except OSError:
        pass


async def run_evaluator_loop() -> None:
    """
    Validator evaluation loop: poll API for latest task_id, evaluate unevaluated tasks, POST results.
    """
    from leoma.infra.remote_api import create_api_client_from_wallet
    from leoma.bootstrap import WALLET_NAME, HOTKEY_NAME

    gemini_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    openai_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not gemini_key and not openai_key:
        log("Evaluator requires GEMINI_API_KEY and/or OPENAI_API_KEY (fallback)", "error")
        return
    gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    if gemini_client and openai_client:
        log("Evaluator LLM: Gemini primary, GPT-4o fallback", "info")
    elif gemini_client:
        log("Evaluator LLM: Gemini only (no GPT-4o fallback configured)", "info")
    else:
        log("Evaluator LLM: GPT-4o only (GEMINI_API_KEY not set)", "info")
    try:
        s3_client = create_samples_read_client()
    except ValueError as e:
        log(
            f"Cannot create samples read client (backend={OBJECT_STORAGE_BACKEND}): {e}",
            "error",
        )
        return
    api_client = create_api_client_from_wallet(
        wallet_name=WALLET_NAME,
        hotkey_name=HOTKEY_NAME,
        api_url=API_URL,
    )
    validator_hotkey = api_client.hotkey
    log_header("Validator Evaluator Starting")
    log(f"Validator hotkey: {validator_hotkey[:16]}...", "info")
    log(f"API: {API_URL}", "info")
    log(
        f"Samples read client: backend={OBJECT_STORAGE_BACKEND}, bucket={SAMPLES_BUCKET}",
        "info",
    )

    evaluated: Set[int] = set()
    try:
        task_ids = await list_evaluated_task_ids(
            s3_client, SAMPLES_BUCKET, validator_hotkey, max_tasks=EVALUATED_LIST_MAX
        )
        evaluated = set(task_ids)
        log(
            f"Initialized evaluated list: {len(evaluated)} task_ids from object storage ({OBJECT_STORAGE_BACKEND})",
            "info",
        )
    except Exception as e:
        log(
            f"Could not load evaluated list from object storage ({OBJECT_STORAGE_BACKEND}): {e}",
            "warn",
        )

    round_num = 0
    while True:
        round_num += 1
        try:
            latest = await api_client.get_latest_task_id()
            if latest is None:
                log("No latest task_id from API", "info")
                await asyncio.sleep(EVALUATOR_POLL_INTERVAL)
                continue

            task_id = latest
            while task_id in evaluated and task_id >= 1:
                task_id -= 1
            if task_id < 1 or task_id <= latest - EVALUATED_LIST_MAX:
                log("All tasks up to latest already evaluated", "info")
                await asyncio.sleep(EVALUATOR_POLL_INTERVAL)
                continue

            log_header(f"Evaluator Round #{round_num} – task_id={task_id}")
            dest_dir = tempfile.mkdtemp(prefix=f"eval_{task_id}_")
            try:
                artifacts = await download_task_artifacts(
                    s3_client,
                    SAMPLES_BUCKET,
                    task_id,
                    dest_dir,
                    include_original_clip=False,
                )
            except Exception as e:
                log(f"Failed to download task {task_id}: {e}", "error")
                _remove_directory(dest_dir)
                evaluated.add(task_id)
                if len(evaluated) > EVALUATED_LIST_MAX:
                    evaluated = set(sorted(evaluated, reverse=True)[:EVALUATED_LIST_MAX])
                await asyncio.sleep(EVALUATOR_POLL_INTERVAL)
                continue

            metadata = artifacts["metadata"]
            prompt_meta = metadata.get("prompt")
            description = (
                prompt_meta.get("text", "")
                if isinstance(prompt_meta, dict)
                else str(prompt_meta or "")
            )
            miner_latencies_ms = (metadata.get("miner_latencies_ms") or {}) if isinstance(metadata.get("miner_latencies_ms"), dict) else {}
            first_frame_path = artifacts["first_frame"]
            generated_videos = artifacts.get("generated_videos") or {}
            if not generated_videos:
                log(f"Task {task_id}: no generated videos", "warn")
                _remove_directory(dest_dir)
                evaluated.add(task_id)
                await asyncio.sleep(EVALUATOR_POLL_INTERVAL)
                continue

            first_frame_b64 = frames_to_base64([first_frame_path])
            s3_prefix = str(task_id)

            samples_payload: list = []
            eval_entries: list = []
            for miner_hotkey, gen_video_path in generated_videos.items():
                gen_frames_dir = os.path.join(dest_dir, f"gen_frames_{miner_hotkey[:8]}")
                try:
                    latency_ms = miner_latencies_ms.get(miner_hotkey)
                    gen_frames = await extract_frames(
                        gen_video_path,
                        gen_frames_dir,
                        max_frames=EVALUATION_MAX_FRAMES,
                        fps=EVALUATION_FRAME_FPS,
                    )
                    gen_frames_b64 = frames_to_base64(gen_frames)
                    comparison = await evaluate_generated_video_async(
                        first_frame_b64,
                        gen_frames_b64,
                        description,
                        gemini_client=gemini_client,
                        openai_client=openai_client,
                        pass_threshold=EVALUATION_PASS_THRESHOLD,
                        critical_threshold=EVALUATION_CRITICAL_THRESHOLD,
                    )
                    passed = bool(comparison.get("passed", False))
                    overall_score = comparison.get("overall_score")
                    aspect_scores = comparison.get("aspect_scores", {})
                    base_reasoning = comparison.get("reasoning") or ""
                    reasoning = (
                        f"{base_reasoning} | overall_score={overall_score} "
                        f"| aspect_scores={json.dumps(aspect_scores, separators=(',', ':'))}"
                    )
                    orig_art = json.dumps(comparison.get("original_artifacts", []))
                    gen_art = json.dumps(
                        {
                            "major_issues": comparison.get("major_issues", []),
                            "strengths": comparison.get("strengths", []),
                            "aspect_scores": aspect_scores,
                            "overall_score": overall_score,
                            "raw_generated_artifacts": comparison.get("generated_artifacts", []),
                        }
                    )
                    pres_order = comparison.get("presentation_order")
                    samples_payload.append({
                        "task_id": task_id,
                        "miner_hotkey": miner_hotkey,
                        "s3_bucket": SAMPLES_BUCKET,
                        "s3_prefix": s3_prefix,
                        "passed": passed,
                        "prompt": description,
                        "confidence": comparison.get("confidence"),
                        "reasoning": reasoning,
                        "latency_ms": latency_ms,
                        "original_artifacts": orig_art,
                        "generated_artifacts": gen_art,
                        "presentation_order": pres_order,
                    })
                    eval_entries.append({
                        "hotkey": miner_hotkey,
                        "passed": passed,
                        "status": "passed" if passed else "failed",
                        "confidence": comparison.get("confidence"),
                        "reasoning": reasoning,
                        "latency_ms": latency_ms,
                        "original_artifacts": orig_art,
                        "generated_artifacts": gen_art,
                        "presentation_order": pres_order,
                    })
                    log(f"Miner {miner_hotkey[:12]}...: {'PASSED' if passed else 'FAILED'}", "info")
                except Exception as e:
                    log(f"Miner {miner_hotkey[:12]}... evaluation failed: {e}", "error")
                finally:
                    if os.path.exists(gen_frames_dir):
                        _remove_directory(gen_frames_dir)

            if samples_payload:
                signature = api_client.sign_evaluation_payload(eval_entries)
                await api_client.submit_samples_batch(samples_payload, evaluation_signature=signature)
            _remove_directory(dest_dir)
            evaluated.add(task_id)
            if len(evaluated) > EVALUATED_LIST_MAX:
                evaluated = set(sorted(evaluated, reverse=True)[:EVALUATED_LIST_MAX])
            log(f"Task {task_id}: submitted {len(samples_payload)} evaluations", "success")

        except Exception as e:
            log(f"Evaluator error: {e}", "error")
            log_exception("Evaluator error", e)

        await asyncio.sleep(EVALUATOR_POLL_INTERVAL)
