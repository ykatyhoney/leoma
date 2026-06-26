"""
Validator API client for the centralized Leoma API.
"""
import asyncio
import json
import os
import time
import hashlib
from typing import Any, Dict, List, Optional

import aiohttp
from substrateinterface import Keypair

from leoma.bootstrap import emit_log
from leoma.domain import MinerInfo

API_URL = os.environ.get("API_URL", "https://api.leoma.ai")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.environ.get("API_MAX_RETRIES", "3"))
_SUCCESS_STATUS_CODES = {200, 201}


class APIClient:
    def __init__(
        self,
        api_url: str = API_URL,
        keypair: Optional[Keypair] = None,
        hotkey_path: Optional[str] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self._keypair = keypair
        self._hotkey_path = hotkey_path
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def keypair(self) -> Keypair:
        if self._keypair is None:
            if self._hotkey_path:
                self._keypair = Keypair.create_from_uri(self._hotkey_path)
            else:
                raise ValueError("No keypair or hotkey_path provided")
        return self._keypair

    @property
    def hotkey(self) -> str:
        return self.keypair.ss58_address

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def sign_evaluation_payload(self, payload_data: list) -> str:
        canonical = json.dumps(payload_data, sort_keys=True).encode("utf-8")
        msg_hash = hashlib.sha256(canonical).digest()
        sig = self.keypair.sign(msg_hash)
        return "0x" + sig.hex()

    def _sign_request(self, body: bytes) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{body_hash}:{timestamp}".encode("utf-8")
        signature = self.keypair.sign(message)
        return {
            "X-Validator-Hotkey": self.hotkey,
            "X-Signature": "0x" + signature.hex(),
            "X-Timestamp": timestamp,
        }

    def _encode_body(self, data: Any) -> bytes:
        if data:
            return json.dumps(data).encode("utf-8")
        return b""

    def _build_headers(self, body: bytes, require_auth: bool) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if require_auth:
            headers.update(self._sign_request(body))
        return headers

    @staticmethod
    async def _parse_response(response: aiohttp.ClientResponse, endpoint: str) -> Dict[str, Any]:
        if response.status in _SUCCESS_STATUS_CODES:
            return await response.json()
        if response.status == 401:
            raise PermissionError("Authentication failed")
        if response.status == 403:
            raise PermissionError("Access denied")
        if response.status == 404:
            raise ValueError(f"Not found: {endpoint}")
        error_text = await response.text()
        raise Exception(f"API error {response.status}: {error_text}")

    @staticmethod
    def _miner_from_payload(payload: Dict[str, Any]) -> MinerInfo:
        def _str(v: Any) -> str:
            return "" if v is None else str(v)
        return MinerInfo(
            uid=payload["uid"],
            hotkey=payload["hotkey"],
            model_name=_str(payload.get("model_name")),
            model_revision=_str(payload.get("model_revision")),
            model_hash=_str(payload.get("model_hash")),
            chute_id=_str(payload.get("chute_id")),
            chute_slug=_str(payload.get("chute_slug")),
            is_valid=payload.get("is_valid", False),
            invalid_reason=payload.get("invalid_reason"),
            block=payload.get("block") if payload.get("block") is not None else 0,
        )

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Any] = None,
        require_auth: bool = True,
    ) -> Dict[str, Any]:
        url = f"{self.api_url}{endpoint}"
        session = await self._get_session()
        body = self._encode_body(data)
        headers = self._build_headers(body, require_auth=require_auth)
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                async with session.request(method, url, data=body if body else None, headers=headers) as response:
                    return await self._parse_response(response, endpoint)
            except aiohttp.ClientError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    emit_log(f"API request failed, retrying... ({e})", "warn")
                    await asyncio.sleep(2 ** attempt)
        raise last_error or Exception("Request failed")

    async def get_valid_miners(self) -> List[MinerInfo]:
        data = await self._request("GET", "/miners/valid")
        return [self._miner_from_payload(p) for p in data["miners"]]

    async def get_all_miners(self) -> List[MinerInfo]:
        data = await self._request("GET", "/miners/all")
        return [self._miner_from_payload(p) for p in data["miners"]]

    async def get_miner(self, hotkey: str) -> MinerInfo:
        data = await self._request("GET", f"/miners/{hotkey}")
        return self._miner_from_payload(data)

    async def report_miners(self, miners: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Report this validator's miner-validation results (permissioned). Replaces prior report."""
        return await self._request("POST", "/miners/report", data={"miners": miners}, require_auth=True)

    async def submit_sample(
        self,
        task_id: int,
        miner_hotkey: str,
        s3_bucket: str,
        s3_prefix: str,
        passed: bool,
        prompt: Optional[str] = None,
        confidence: Optional[int] = None,
        reasoning: Optional[str] = None,
        latency_ms: Optional[int] = None,
        original_artifacts: Optional[str] = None,
        generated_artifacts: Optional[str] = None,
        presentation_order: Optional[str] = None,
    ) -> Dict[str, Any]:
        data = {
            "task_id": task_id,
            "miner_hotkey": miner_hotkey,
            "s3_bucket": s3_bucket,
            "s3_prefix": s3_prefix,
            "passed": bool(passed),
        }
        if prompt:
            data["prompt"] = prompt
        if confidence is not None:
            data["confidence"] = confidence
        if reasoning:
            data["reasoning"] = reasoning
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        if original_artifacts is not None:
            data["original_artifacts"] = original_artifacts
        if generated_artifacts is not None:
            data["generated_artifacts"] = generated_artifacts
        if presentation_order is not None:
            data["presentation_order"] = presentation_order
        return await self._request("POST", "/samples", data)

    async def submit_samples_batch(
        self,
        samples: List[Dict[str, Any]],
        evaluation_signature: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {"samples": samples}
        if evaluation_signature:
            body["signature"] = evaluation_signature
        return await self._request("POST", "/samples/batch", body)

    async def get_scores(self) -> Dict[str, Any]:
        return await self._request("GET", "/scores", require_auth=False)

    async def get_miner_score(self, miner_hotkey: str) -> Dict[str, Any]:
        return await self._request("GET", f"/scores/miner/{miner_hotkey}", require_auth=False)

    async def get_weights(self) -> Dict[str, Any]:
        return await self._request("GET", "/weights", require_auth=False)

    async def get_rank(self) -> Dict[str, Any]:
        return await self._request("GET", "/scores/rank", require_auth=False)

    async def get_blacklist(self) -> List[Dict[str, Any]]:
        return await self._request("GET", "/blacklist", require_auth=False)

    async def get_blacklisted_miners(self) -> List[str]:
        return await self._request("GET", "/blacklist/miners", require_auth=False)

    async def is_blacklisted(self, hotkey: str) -> bool:
        try:
            await self._request("GET", f"/blacklist/{hotkey}", require_auth=False)
            return True
        except ValueError:
            return False

    async def add_to_blacklist(self, hotkey: str, reason: Optional[str] = None) -> Dict[str, Any]:
        data = {"hotkey": hotkey}
        if reason:
            data["reason"] = reason
        return await self._request("POST", "/blacklist", data=data, require_auth=True)

    async def remove_from_blacklist(self, hotkey: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/blacklist/{hotkey}", require_auth=True)

    async def health_check(self) -> Dict[str, Any]:
        return await self._request("GET", "/health", require_auth=False)


def create_api_client_from_wallet(
    wallet_name: str = "default",
    hotkey_name: str = "default",
    api_url: str = API_URL,
) -> APIClient:
    import bittensor as bt
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    return APIClient(api_url=api_url, keypair=wallet.hotkey)
