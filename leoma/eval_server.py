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
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from leoma.eval.spec import ConsensusSpec


class EvalRequest(BaseModel):
    """A duel job posted by a validator.

    **The request carries the entire consensus surface.** It used to carry a
    handful of loose knobs (``metric``, ``n_clips``, ``num_frames``…) each with a
    *default* — so a validator that forgot one, or an eval box running an older
    build that ignored one, silently ran a different exam and produced a verdict
    nobody else could reproduce. Worse, the parameters it *didn't* carry (prompt,
    resolution, negative prompt, fps) came from the eval box's own environment.

    Now the validator sends the pinned :class:`~leoma.eval.spec.ConsensusSpec` and
    the server executes exactly that. The server reads **nothing** about how to
    duel from its own environment, and echoes the spec back in the verdict so the
    validator can verify what it actually ran before crowning anyone.
    """

    model_config = ConfigDict(extra="forbid")

    king_repo: str
    king_digest: str
    challenger_repo: str
    challenger_digest: str
    block_hash: str = ""
    hotkey: str = ""
    #: The pinned exam. No defaults — a missing field is a 422, not a guess.
    spec: ConsensusSpec
    #: The validator's own digest of ``spec``. Guards against a spec that was
    #: mangled in transit into something that still parses.
    consensus_digest: str


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
        """Enough for a validator to reject a stale box *before* handing it a duel.

        ``consensus_digest`` and ``eval_code_digest`` are the two that matter: a box
        whose chain.toml or scoring code has drifted will produce distances nobody
        can reproduce, and an hours-long duel is a very expensive way to find that
        out. The validator preflights this endpoint instead.
        """
        from leoma.infra.chain_config import CONSENSUS_DIGEST
        from leoma.eval.codehash import eval_code_digest

        return {
            "status": "ok",
            "busy": lock.locked(),
            "consensus_digest": CONSENSUS_DIGEST,
            "eval_code_digest": eval_code_digest(),
        }

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


def check_request(req: EvalRequest) -> None:
    """Refuse a duel this box cannot run reproducibly. Called before any GPU work.

    Three ways a duel is dead on arrival, all of them cheap to detect and
    catastrophic to miss:

    * the validator's ``consensus_digest`` doesn't match the spec it sent (mangled
      in transit, or a validator computing digests differently);
    * this box's ``chain.toml`` pins a *different* exam than the validator's (an
      operator forgot to redeploy one machine — the most likely failure by far);
    * the corpus isn't pinned at all.

    Any of them and we stop here, rather than spending an hour of GPU producing a
    verdict nobody else can reproduce.
    """
    from leoma.infra.chain_config import CONSENSUS_DIGEST
    from leoma.eval.errors import ConsensusConfigError

    if req.spec.digest() != req.consensus_digest:
        raise ConsensusConfigError(
            f"request consensus_digest {req.consensus_digest} does not match the spec it "
            f"carries ({req.spec.digest()}) — the request was altered in transit"
        )
    if req.consensus_digest != CONSENSUS_DIGEST:
        raise ConsensusConfigError(
            f"this eval box pins a different consensus surface than the validator "
            f"(box {CONSENSUS_DIGEST}, validator {req.consensus_digest}). One of the two "
            "is running a stale chain.toml. Refusing the duel rather than producing a "
            "verdict the rest of the subnet cannot reproduce."
        )
    req.spec.require_duel_ready()


def run_eval_job(req: EvalRequest, emit: Emit) -> dict:
    """Production duel: download both models, generate on held-out clips, score.

    A **pure executor of the request**. Every parameter comes from ``req.spec``;
    this function reads nothing about how to duel from its own environment, which
    is what makes the verdict a function of the chain rather than of the box.

    Heavy deps (torch/diffusers, the corpus) are imported here, lazily, so the
    server module stays import-safe. Raises on any failure (the app turns that into
    an ``error`` event).
    """
    from leoma.infra.model_store import ModelRef, materialize_model
    from leoma.infra.storage_backend import create_source_read_client
    from leoma.eval.video_runner import GenParams, load_video_pipeline, duel
    from leoma.eval.codehash import eval_code_digest
    from leoma.eval.dataset import build_duel_clips, corpus_audit, fetch_manifest
    from leoma.eval.determinism import apply_determinism, runtime_env
    from leoma.eval.digests import digest_obj
    from leoma.app.validator.seeds import eval_seed

    check_request(req)
    spec = req.spec
    apply_determinism(spec.determinism)

    king_ref = ModelRef(req.king_repo, req.king_digest)
    chall_ref = ModelRef(req.challenger_repo, req.challenger_digest)

    emit({"phase": "materialize", "which": "king", "ref": king_ref.immutable_ref})
    king_dir = materialize_model(king_ref)
    emit({"phase": "materialize", "which": "challenger", "ref": chall_ref.immutable_ref})
    chall_dir = materialize_model(chall_ref)

    emit({"phase": "load", "which": "king"})
    king_pipe = load_video_pipeline(king_dir, gen=spec.gen)
    emit({"phase": "load", "which": "challenger"})
    chall_pipe = load_video_pipeline(chall_dir, gen=spec.gen)

    master_seed = eval_seed(req.block_hash, req.hotkey, spec.duel.base_seed)
    params = GenParams.from_spec(spec.gen)

    emit({"phase": "sample_clips", "n_clips": spec.duel.n_clips, "seed": master_seed})
    client = create_source_read_client()
    manifest = fetch_manifest(client, spec.corpus)
    clips, entries = build_duel_clips(
        manifest,
        client=client,
        bucket=spec.corpus.bucket,
        master_seed=master_seed,
        n_clips=spec.duel.n_clips,
        gen=params,
        prompt_mode=spec.gen.prompt_mode,
        fixed_prompt=spec.gen.prompt,
        # The clip phase is otherwise silent for minutes; this is the watchdog's
        # evidence that the box is making forward progress rather than hung.
        on_progress=lambda done, total, entry: emit({
            "phase": "clip_ready", "done": done, "total": total, "clip_id": entry.clip_id,
        }),
    )

    emit({"phase": "duel", "n_clips": len(clips), "metric": spec.duel.metric})
    verdict = duel(
        king_pipe, chall_pipe, clips,
        master_seed=master_seed,
        metric=spec.duel.metric,
        metric_device=spec.duel.metric_device,
        delta_threshold=spec.duel.delta_threshold,
        alpha=spec.duel.alpha,
        n_bootstrap=spec.duel.n_bootstrap,
        on_phase=emit,
        early_stop_max_advantage=spec.early_stop_max_advantage,
    )

    verdict["king_digest"] = req.king_digest
    verdict["challenger_digest"] = req.challenger_digest
    verdict["challenger_repo"] = req.challenger_repo

    # The spec, verbatim. The validator re-parses this and refuses to crown unless
    # it matches what it sent — which is what makes "field silently defaulted" a
    # caught error instead of an invisible fork.
    verdict["echo"] = spec.model_dump(mode="json")

    # Everything a third party needs to replay this duel and get the same numbers.
    verdict["audit"] = {
        "master_seed": master_seed,
        "block_hash": req.block_hash,
        "hotkey": req.hotkey,
        "consensus_digest": req.consensus_digest,
        "eval_code_digest": eval_code_digest(),
        "corpus": corpus_audit(manifest, entries),
        "env": runtime_env(),
    }
    # Hashes ONLY the consensus surface — the decision, the exam, the parameters.
    # Deliberately excludes env and produced_at: two validators on different GPUs
    # SHOULD produce the same verdict_digest when they agree, and a wall-clock or a
    # GPU name inside it would guarantee they never do.
    verdict["verdict_digest"] = digest_obj({
        "accepted": verdict["accepted"],
        "consensus_digest": req.consensus_digest,
        "master_seed": master_seed,
        "king_digest": req.king_digest,
        "challenger_digest": req.challenger_digest,
        "clip_keys_digest": verdict["audit"]["corpus"]["clip_keys_digest"],
        "per_clip": [
            {
                "clip_id": c["clip_id"],
                "king_distance": c["king_distance"],
                "challenger_distance": c["challenger_distance"],
            }
            for c in verdict["per_clip"]
        ],
    })
    verdict["produced_at"] = datetime.now(timezone.utc).isoformat()
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
