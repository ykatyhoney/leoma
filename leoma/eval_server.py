"""FastAPI eval server for the video king-of-the-hill duel.

The validator POSTs a duel (king vs challenger, both by immutable ``repo@digest``)
and streams progress + the final verdict over SSE. One duel runs at a time
(global lock): the GPU box downloads both models, generates on the deterministic
held-out clips, scores against the ground-truth continuations, and returns the
paired-bootstrap verdict.

The heavy work is a pluggable *runner* (``create_app(runner=...)``) so the HTTP
contract, the single-flight lock, and the SSE stream are unit-testable with a
fake runner and no GPU. The default runner is the production
``run_eval_job`` (torch/diffusers imported lazily).
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    """A duel job posted by a validator."""

    king_repo: str
    king_digest: str
    challenger_repo: str
    challenger_digest: str
    block_hash: str = ""
    hotkey: str = ""
    metric: str = "lpips"
    n_clips: int = Field(default=32, ge=1)
    delta_threshold: float = 0.0025
    alpha: float = 0.001
    n_bootstrap: int = Field(default=10_000, ge=1)
    base_seed: int = 0
    # Generation knobs (kept identical for king + challenger).
    num_frames: int = 81
    num_inference_steps: int = 30
    guidance_scale: float = 5.0


# A runner turns a request into a verdict, emitting progress events along the way.
Emit = Callable[[dict], None]
Runner = Callable[[EvalRequest, Emit], dict]

_SENTINEL = object()


@dataclass
class _Job:
    eval_id: str
    status: str = "running"          # running | done | error
    events: "queue.Queue" = field(default_factory=queue.Queue)
    verdict: Optional[dict] = None
    error: Optional[str] = None


def create_app(runner: Optional[Runner] = None) -> FastAPI:
    app = FastAPI(title="leoma-eval-server")
    run_job: Runner = runner or run_eval_job

    lock = threading.Lock()
    jobs: dict[str, _Job] = {}
    jobs_guard = threading.Lock()

    def _execute(job: _Job, req: EvalRequest) -> None:
        def emit(event: dict) -> None:
            job.events.put(event)

        try:
            verdict = run_job(req, emit)
            job.verdict = verdict
            job.status = "done"
            emit({"phase": "verdict", **verdict})
        except Exception as e:  # pragma: no cover - defensive
            job.status = "error"
            job.error = str(e)
            emit({"phase": "error", "error": str(e)})
        finally:
            job.events.put(_SENTINEL)
            lock.release()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "busy": lock.locked()}

    @app.post("/eval")
    def start_eval(req: EvalRequest):
        if not lock.acquire(blocking=False):
            return JSONResponse(status_code=409, content={"error": "an eval is already running"})
        eval_id = f"eval-{uuid.uuid4().hex[:12]}"
        job = _Job(eval_id=eval_id)
        with jobs_guard:
            jobs[eval_id] = job
        threading.Thread(target=_execute, args=(job, req), daemon=True).start()
        return {"eval_id": eval_id}

    @app.get("/eval/{eval_id}")
    def get_eval(eval_id: str):
        with jobs_guard:
            job = jobs.get(eval_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "unknown eval_id"})
        return {"eval_id": eval_id, "status": job.status, "verdict": job.verdict, "error": job.error}

    @app.get("/eval/{eval_id}/stream")
    def stream_eval(eval_id: str):
        with jobs_guard:
            job = jobs.get(eval_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "unknown eval_id"})

        def gen():
            while True:
                event = job.events.get()
                if event is _SENTINEL:
                    break
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def run_eval_job(req: EvalRequest, emit: Emit) -> dict:
    """Production duel: download both models, generate on held-out clips, score.

    Heavy deps (torch/diffusers, the corpus) are imported here, lazily, so the
    server module stays import-safe. Raises on any failure (the app turns that
    into an ``error`` SSE event).
    """
    from leoma.infra.model_store import ModelRef, materialize_model
    from leoma.eval.video_runner import GenParams, load_video_pipeline, duel
    from leoma.eval.dataset import build_duel_clips
    from leoma.app.validator.seeds import eval_seed

    king_ref = ModelRef(req.king_repo, req.king_digest)
    chall_ref = ModelRef(req.challenger_repo, req.challenger_digest)

    emit({"phase": "materialize", "which": "king", "ref": king_ref.immutable_ref})
    king_dir = materialize_model(king_ref)
    emit({"phase": "materialize", "which": "challenger", "ref": chall_ref.immutable_ref})
    chall_dir = materialize_model(chall_ref)

    emit({"phase": "load", "which": "king"})
    king_pipe = load_video_pipeline(king_dir)
    emit({"phase": "load", "which": "challenger"})
    chall_pipe = load_video_pipeline(chall_dir)

    master_seed = eval_seed(req.block_hash, req.hotkey, req.base_seed)
    params = GenParams(
        num_frames=req.num_frames,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
    )
    emit({"phase": "sample_clips", "n_clips": req.n_clips, "seed": master_seed})
    clips = build_duel_clips(master_seed, req.n_clips, params)

    emit({"phase": "duel", "n_clips": len(clips), "metric": req.metric})
    verdict = duel(
        king_pipe, chall_pipe, clips,
        master_seed=master_seed, metric=req.metric,
        delta_threshold=req.delta_threshold, alpha=req.alpha, n_bootstrap=req.n_bootstrap,
        on_phase=emit,
    )
    verdict["king_digest"] = req.king_digest
    verdict["challenger_digest"] = req.challenger_digest
    verdict["challenger_repo"] = req.challenger_repo
    return verdict


def main() -> None:  # pragma: no cover - entrypoint
    import os
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("EVAL_SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("EVAL_SERVER_PORT", "9000")),
    )


app = None  # set by uvicorn factory if needed


if __name__ == "__main__":  # pragma: no cover
    main()
