"""
Self-evaluation (decentralized, no cross-validation).

A validator evaluates ONLY the tasks it sampled itself, right after sampling — it already has
the generated videos locally. ``evaluate_sampled_task`` runs the Gemini benchmark per miner and
returns both the API dual-report payload and the bucket ``evaluation_results`` entries. The
sampler loop ([app/sampler/loop.py]) calls it; there is no separate polling evaluator anymore.
"""

import os
import json
from typing import TYPE_CHECKING, Any, List, Tuple

from leoma.bootstrap import emit_log as log
from leoma.infra.video_utils import frames_to_base64
from leoma.infra.judge import evaluate_generated_video_async

if TYPE_CHECKING:
    from leoma.app.sampler.core import SampledTask

EVALUATION_PASS_THRESHOLD = int(os.environ.get("EVALUATION_PASS_THRESHOLD", "75"))
EVALUATION_CRITICAL_THRESHOLD = int(os.environ.get("EVALUATION_CRITICAL_THRESHOLD", "50"))


async def evaluate_sampled_task(
    gemini_client: Any,
    result: "SampledTask",
    own_bucket: str,
) -> Tuple[List[dict], List[dict]]:
    """Evaluate this validator's own sampled task. Returns (samples_payload, eval_entries).

    - ``samples_payload``: rows for the API dashboard dual-report (POST /samples/batch).
    - ``eval_entries``: the ``data`` list published to this validator's own bucket
      (``{task_id}/evaluation_results/<hotkey>.json``), read by peers for aggregation.
    """
    prompt_meta = result.metadata.get("prompt")
    description = prompt_meta.get("text", "") if isinstance(prompt_meta, dict) else str(prompt_meta or "")
    latencies = result.metadata.get("miner_latencies_ms") or {}
    first_frame_b64 = frames_to_base64([result.frame_path])
    task_id = result.task_id
    s3_prefix = str(task_id)

    samples_payload: List[dict] = []
    eval_entries: List[dict] = []
    for miner_hotkey, gen_video_path in result.miner_paths.items():
        try:
            comparison = await evaluate_generated_video_async(
                first_frame_b64,
                gen_video_path,
                description,
                gemini_client=gemini_client,
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
            latency_ms = latencies.get(miner_hotkey)
            samples_payload.append({
                "task_id": task_id,
                "miner_hotkey": miner_hotkey,
                "s3_bucket": own_bucket,
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

    return samples_payload, eval_entries
