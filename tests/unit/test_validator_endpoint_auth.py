"""
Security guard: every validator-operation endpoint MUST reject unauthenticated access,
regardless of HTTP method. Validator coordination, inputs, and validator-specific reads require a
validator hotkey signature; admin operations require an admin signature. Public dashboard reads are
listed separately and intentionally open (the website cannot sign).

This locks in the invariant — if someone adds a validator-operation route without an auth
dependency (so an unauthenticated caller reaches the handler / gets 200), this test fails.
"""
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from leoma.delivery.http.routes import (
    miners_router,
    rotation_router,
    samples_router,
    tasks_router,
)

HK = "5C62W7ELLAAfjCQeBU3me9nTXXqjVwN4kQY8w8gM9nJ8K4pL"  # arbitrary ss58 path param

# (method, path) for every endpoint that performs/exposes a VALIDATOR operation.
VALIDATOR_OP_ENDPOINTS = [
    ("GET", "/tasks/latest"),            # sampling coordination (was unauthenticated — now gated)
    ("GET", "/tasks/window"),
    ("POST", "/tasks/announce"),
    ("POST", "/tasks/claim"),
    ("GET", "/rotation"),
    ("POST", "/miners/report"),
    ("GET", "/miners/valid"),
    ("GET", "/miners/all"),
    ("POST", "/samples"),
    ("POST", "/samples/batch"),
    ("GET", f"/samples/task?validator_hotkey={HK}&task_id=1"),  # was unauthenticated — now gated
    ("GET", f"/samples/validator/{HK}"),
    ("GET", f"/samples/miner/{HK}"),
    ("GET", f"/miners/{HK}"),            # validator-facing miner read (public twin is /miners/info/{hk})
]

# Admin-only operations: unauthenticated callers must also be rejected here.
ADMIN_OP_ENDPOINTS = [
    ("POST", "/rotation/interval"),
]


@pytest.fixture
def app_client():
    app = FastAPI()
    app.include_router(miners_router, prefix="/miners")
    app.include_router(samples_router, prefix="/samples")
    app.include_router(tasks_router, prefix="/tasks")
    app.include_router(rotation_router, prefix="/rotation")
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


@pytest.mark.parametrize("method,path", VALIDATOR_OP_ENDPOINTS + ADMIN_OP_ENDPOINTS)
async def test_validator_op_rejects_unauthenticated(app_client, method, path):
    """No signature headers -> the request must be rejected before reaching the handler.

    422 (FastAPI rejects the required X-Validator-Hotkey/X-Signature/X-Timestamp headers) or
    401/403 (verifier rejects) are all acceptable; a 200 would be a security hole.
    """
    async with app_client as ac:
        r = await ac.request(method, path, json={} if method == "POST" else None)
    assert r.status_code in (401, 403, 422), f"{method} {path} returned {r.status_code} without auth"


@pytest.mark.parametrize("method,path", VALIDATOR_OP_ENDPOINTS + ADMIN_OP_ENDPOINTS)
async def test_validator_op_rejects_bad_signature(app_client, method, path):
    """Bogus signature headers -> 401 (verifier runs and fails). Never 200."""
    headers = {
        "X-Validator-Hotkey": HK,
        "X-Signature": "0x" + "00" * 64,
        "X-Timestamp": "9999999999",  # also far-future, but signature check fails first/too
    }
    async with app_client as ac:
        r = await ac.request(method, path, headers=headers, json={} if method == "POST" else None)
    assert r.status_code != 200, f"{method} {path} returned 200 with a bogus signature"
    assert r.status_code in (401, 403, 422), f"{method} {path} returned {r.status_code} with bad sig"
