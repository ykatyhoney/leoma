"""
Health check route for Leoma API.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from sqlalchemy import text

from leoma.delivery.http.contracts import HealthResponse
from leoma import __version__


router = APIRouter()

# Track last metagraph sync
_last_metagraph_sync: Optional[datetime] = None


async def _is_database_healthy() -> bool:
    """Check database connectivity via lightweight query."""
    try:
        from leoma.infra.db.pool import get_session

        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def update_last_sync(sync_time: datetime) -> None:
    """Update the last metagraph sync timestamp."""
    global _last_metagraph_sync
    _last_metagraph_sync = sync_time


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health status, including database and metagraph sync state."""
    db_healthy = await _is_database_healthy()
    
    return HealthResponse(
        status="healthy" if db_healthy else "degraded",
        version=__version__,
        database=db_healthy,
        metagraph_synced=_last_metagraph_sync is not None,
        last_sync=_last_metagraph_sync,
    )
