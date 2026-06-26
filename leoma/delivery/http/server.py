"""
FastAPI application for Leoma API Service.
"""
import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from leoma import __version__
from leoma.bootstrap import emit_log, emit_header, logger as leoma_logger
from leoma.infra.db.pool import init_database, close_database, create_tables
from leoma.delivery.http.routes import (
    miners_router,
    overview_router,
    rotation_router,
    samples_router,
    scores_router,
    blacklist_router,
    health_router,
    tasks_router,
    validators_router,
    weights_router,
)
from leoma.delivery.http.tasks import (
    MinerConsensusTask,
    ScoreCalculationTask,
    ValidatorSyncTask,
)

_background_tasks: list[asyncio.Task] = []


def _env_flag(name: str, default: bool = False) -> bool:
    fallback = "true" if default else "false"
    return os.environ.get(name, fallback).lower() == "true"


async def _create_tables_if_needed() -> None:
    await create_tables()


def _start_background_tasks() -> None:
    miner_task = MinerConsensusTask()
    score_task = ScoreCalculationTask()
    validator_sync_task = ValidatorSyncTask()
    _background_tasks.append(asyncio.create_task(miner_task.run()))
    _background_tasks.append(asyncio.create_task(score_task.run()))
    _background_tasks.append(asyncio.create_task(validator_sync_task.run()))


async def _stop_background_tasks() -> None:
    for task in _background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    emit_header("Leoma API Service Starting")
    await init_database()
    emit_log("Database initialized", "success")
    await _create_tables_if_needed()
    _start_background_tasks()
    emit_log("Background tasks started", "success")
    yield
    emit_log("Shutting down...", "info")
    await _stop_background_tasks()
    await close_database()
    emit_log("Shutdown complete", "success")


def _cors_origins() -> list[str]:
    """CORS allow_origins: from CORS_ORIGINS env (comma-separated) or ['*'] for dev."""
    raw = os.environ.get("CORS_ORIGINS", "").strip()
    if not raw:
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="Leoma API",
    description="Centralized API for Leoma subnet validators",
    version=__version__,
    lifespan=lifespan,
)


def _cors_headers_for_request(request: Request) -> dict[str, str]:
    """Add CORS headers so error responses (e.g. 500) are visible to the frontend."""
    origin = request.headers.get("origin")
    if not origin:
        return {}
    origins = _cors_origins()
    if "*" in origins:
        return {"Access-Control-Allow-Origin": origin, "Access-Control-Allow-Credentials": "true"}
    if origin in origins:
        return {"Access-Control-Allow-Origin": origin, "Access-Control-Allow-Credentials": "true"}
    return {}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return generic error in response; log full detail server-side."""
    leoma_logger.exception("Unhandled exception")
    headers = _cors_headers_for_request(request)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred"},
        headers=headers,
    )

# Request body size limit (DoS mitigation); set MAX_BODY_SIZE env (bytes, default 2MB)
_MAX_BODY_SIZE = int(os.environ.get("MAX_BODY_SIZE", "2097152"))


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with Content-Length exceeding MAX_BODY_SIZE."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > _MAX_BODY_SIZE:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": "Request body too large"},
                        )
                except ValueError:
                    pass
        return await call_next(request)


_cors_origins_list = _cors_origins()
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, tags=["Health"])
app.include_router(miners_router, prefix="/miners", tags=["Miners"])
app.include_router(samples_router, prefix="/samples", tags=["Samples"])
app.include_router(scores_router, prefix="/scores", tags=["Scores"])
app.include_router(blacklist_router, prefix="/blacklist", tags=["Blacklist"])
app.include_router(tasks_router, prefix="/tasks", tags=["Tasks"])
app.include_router(weights_router, prefix="/weights", tags=["Weights"])
app.include_router(rotation_router, prefix="/rotation", tags=["Rotation"])
app.include_router(validators_router, prefix="/validators", tags=["Validators"])
app.include_router(overview_router, prefix="/overview", tags=["Overview"])


def main() -> None:
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(
        "leoma.delivery.http.server:app",
        host=host,
        port=port,
        reload=_env_flag("API_RELOAD"),
    )


if __name__ == "__main__":
    main()
