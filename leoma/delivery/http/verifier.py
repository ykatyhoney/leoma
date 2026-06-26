"""
Hotkey signature verification for API authentication.
"""
import json
import os
import time
import hashlib
from typing import Any, List, Optional, Tuple, Annotated

from fastapi import HTTPException, Header, Request, Depends
from substrateinterface import Keypair

from leoma.bootstrap import emit_log
from leoma.infra.allowlist import VALIDATOR_ALLOWLIST
from leoma.infra.db.stores import BlacklistStore

SIGNATURE_EXPIRY_SECONDS = int(os.environ.get("SIGNATURE_EXPIRY_SECONDS", "300"))


def _load_admin_hotkeys() -> list[str]:
    return [h for h in os.environ.get("ADMIN_HOTKEYS", "").split(",") if h]


ADMIN_HOTKEYS = _load_admin_hotkeys()


class SignatureVerifier:
    def __init__(self):
        self.blacklist_store = BlacklistStore()

    @staticmethod
    def _validate_timestamp(timestamp: str) -> Tuple[bool, Optional[str]]:
        try:
            parsed_timestamp = int(timestamp)
        except ValueError:
            return False, "Invalid timestamp format"
        now = int(time.time())
        if abs(now - parsed_timestamp) > SIGNATURE_EXPIRY_SECONDS:
            return False, f"Timestamp expired (max {SIGNATURE_EXPIRY_SECONDS}s)"
        return True, None

    @staticmethod
    def _signature_bytes(signature: str) -> bytes:
        return bytes.fromhex(signature.removeprefix("0x"))

    async def verify_request(
        self,
        body: bytes,
        hotkey: str,
        signature: str,
        timestamp: str,
    ) -> Tuple[bool, Optional[str]]:
        is_timestamp_valid, timestamp_error = self._validate_timestamp(timestamp)
        if not is_timestamp_valid:
            return False, timestamp_error
        if await self.blacklist_store.is_blacklisted(hotkey):
            return False, "Hotkey is blacklisted"
        try:
            message = self._create_message(body, timestamp)
            keypair = Keypair(ss58_address=hotkey)
            sig_bytes = self._signature_bytes(signature)
            is_valid = keypair.verify(message, sig_bytes)
            if not is_valid:
                return False, "Invalid signature"
            # Admin hotkeys bypass the allowlist; everyone else must be in the hardcoded allowlist.
            if self.is_admin(hotkey) or hotkey in VALIDATOR_ALLOWLIST:
                return True, None
            return False, "Not a permissioned validator (hotkey is not in the repo allowlist)."
        except Exception as e:
            emit_log(f"Signature verification error: {e}", "warn")
            return False, "Signature verification failed"

    @staticmethod
    def _create_message(body: bytes, timestamp: str) -> bytes:
        body_hash = hashlib.sha256(body).hexdigest()
        return f"{body_hash}:{timestamp}".encode("utf-8")

    def is_admin(self, hotkey: str) -> bool:
        return hotkey in ADMIN_HOTKEYS

    async def is_permissioned(self, hotkey: str) -> bool:
        """A validator is permissioned iff it is in the hardcoded repo allowlist (or an admin).

        The repo ``VALIDATOR_ALLOWLIST`` is the single source of truth for membership — the same list
        that drives rotation and the equal-weight voter set — so auth never drifts from consensus.
        """
        return self.is_admin(hotkey) or hotkey in VALIDATOR_ALLOWLIST


_verifier: Optional[SignatureVerifier] = None


def get_verifier() -> SignatureVerifier:
    global _verifier
    if _verifier is None:
        _verifier = SignatureVerifier()
    return _verifier


async def verify_signature(
    request: Request,
    x_validator_hotkey: Annotated[str, Header()],
    x_signature: Annotated[str, Header()],
    x_timestamp: Annotated[str, Header()],
) -> str:
    verifier = get_verifier()
    body = await request.body()
    is_valid, error = await verifier.verify_request(
        body=body,
        hotkey=x_validator_hotkey,
        signature=x_signature,
        timestamp=x_timestamp,
    )
    if not is_valid:
        raise HTTPException(status_code=401, detail=error or "Authentication failed")
    return x_validator_hotkey


async def verify_admin_signature(
    request: Request,
    x_validator_hotkey: Annotated[str, Header()],
    x_signature: Annotated[str, Header()],
    x_timestamp: Annotated[str, Header()],
) -> str:
    hotkey = await verify_signature(request, x_validator_hotkey, x_signature, x_timestamp)
    if not get_verifier().is_admin(hotkey):
        raise HTTPException(status_code=403, detail="Admin access required")
    return hotkey


async def verify_permissioned_validator(
    request: Request,
    x_validator_hotkey: Annotated[str, Header()],
    x_signature: Annotated[str, Header()],
    x_timestamp: Annotated[str, Header()],
) -> str:
    """Authenticate, then require the hotkey to be in the permissioned sampling allowlist."""
    hotkey = await verify_signature(request, x_validator_hotkey, x_signature, x_timestamp)
    if not await get_verifier().is_permissioned(hotkey):
        raise HTTPException(status_code=403, detail="Not a permissioned sampling validator")
    return hotkey


async def get_current_validator(
    hotkey: Annotated[str, Depends(verify_signature)],
) -> str:
    return hotkey


async def get_current_admin(
    hotkey: Annotated[str, Depends(verify_admin_signature)],
) -> str:
    return hotkey


def sign_message(keypair: Keypair, body: bytes, timestamp: str) -> str:
    message = SignatureVerifier._create_message(body, timestamp)
    signature = keypair.sign(message)
    return "0x" + signature.hex()


def verify_evaluation_signature(
    validator_hotkey_ss58: str,
    signature_hex: str,
    data: List[Any],
) -> bool:
    canonical = json.dumps(data, sort_keys=True).encode("utf-8")
    msg_hash = hashlib.sha256(canonical).digest()
    keypair = Keypair(ss58_address=validator_hotkey_ss58)
    sig_bytes = bytes.fromhex(signature_hex.removeprefix("0x"))
    return keypair.verify(msg_hash, sig_bytes)
