"""Blacklist routes for Leoma API: manage blacklisted miners."""

from typing import Annotated, Any, List

from fastapi import APIRouter, Depends, HTTPException

from leoma.delivery.http.verifier import verify_admin_signature
from leoma.delivery.http.contracts import BlacklistEntry, BlacklistResponse
from leoma.delivery.http.validators import validate_hotkey
from leoma.infra.db.stores import BlacklistStore


router = APIRouter()
blacklist_dao = BlacklistStore()


def _to_blacklist_response(entry: Any) -> BlacklistResponse:
    """Convert blacklist ORM/entity record to API response model."""
    return BlacklistResponse(
        hotkey=entry.hotkey,
        reason=entry.reason,
        added_by=entry.added_by,
        created_at=entry.created_at,
    )


@router.get("", response_model=List[BlacklistResponse])
async def get_blacklist() -> List[BlacklistResponse]:
    """Get all blacklisted miners. Public (no authentication required)."""
    entries = await blacklist_dao.get_all()
    
    return [_to_blacklist_response(entry) for entry in entries]


@router.get("/miners", response_model=List[str])
async def get_blacklisted_miners() -> List[str]:
    """Get list of blacklisted miner hotkeys. Public (no authentication required)."""
    return await blacklist_dao.get_hotkeys()


@router.get("/{hotkey}", response_model=BlacklistResponse)
async def get_blacklist_entry(
    hotkey: str,
) -> BlacklistResponse:
    """Get blacklist entry for a specific miner. Public (no authentication required)."""
    hotkey = validate_hotkey(hotkey)
    entry = await blacklist_dao.get(hotkey)
    
    if not entry:
        raise HTTPException(status_code=404, detail="Miner not blacklisted")
    
    return _to_blacklist_response(entry)


@router.post("", response_model=BlacklistResponse)
async def add_to_blacklist(
    entry: BlacklistEntry,
    admin_hotkey: Annotated[str, Depends(verify_admin_signature)],
) -> BlacklistResponse:
    """Add a miner to the blacklist. Requires admin signature authentication."""
    result = await blacklist_dao.add(
        hotkey=entry.hotkey,
        reason=entry.reason,
        added_by=admin_hotkey,
    )
    
    return _to_blacklist_response(result)


@router.delete("/{hotkey}")
async def remove_from_blacklist(
    hotkey: str,
    admin_hotkey: Annotated[str, Depends(verify_admin_signature)],
) -> dict:
    """Remove a miner from the blacklist. Requires admin signature authentication."""
    hotkey = validate_hotkey(hotkey)
    removed = await blacklist_dao.remove(hotkey)
    
    if not removed:
        raise HTTPException(status_code=404, detail="Miner not blacklisted")
    
    return {
        "success": True,
        "message": f"Removed miner {hotkey[:8]}... from blacklist",
        "removed_by": admin_hotkey,
    }
