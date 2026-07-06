"""
Unit tests for the eval-server HTTP contract (fake runner, no GPU).

Covers: POST /eval -> eval_id, SSE stream of phases + verdict, status poll,
single-flight 409 while a duel is running, and 404 for unknown ids.
"""

import json
import threading

import pytest
from starlette.testclient import TestClient

from leoma.eval_server import create_app


REQ = dict(
    king_repo="u/leoma-k", king_digest="sha256:" + "a" * 64,
    challenger_repo="u/leoma-c", challenger_digest="sha256:" + "b" * 64,
    block_hash="0xabc", hotkey="5C7L", n_clips=8, n_bootstrap=100,
)


def _stream_events(client, eval_id):
    events = []
    with client.stream("GET", f"/eval/{eval_id}/stream") as s:
        for line in s.iter_lines():
            if line and line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_health():
    c = TestClient(create_app(runner=lambda req, emit: {"accepted": False}))
    body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["busy"] is False


def test_full_duel_flow():
    def runner(req, emit):
        emit({"phase": "materialize", "which": "king"})
        emit({"phase": "duel", "n_clips": req.n_clips})
        return {"accepted": True, "verdict": "challenger", "lcb": 0.09, "n_clips": req.n_clips}

    c = TestClient(create_app(runner=runner))
    r = c.post("/eval", json=REQ)
    assert r.status_code == 200
    eval_id = r.json()["eval_id"]

    events = _stream_events(c, eval_id)
    phases = [e["phase"] for e in events]
    assert phases == ["materialize", "duel", "verdict"]
    assert events[-1]["accepted"] is True
    assert events[-1]["verdict"] == "challenger"

    poll = c.get(f"/eval/{eval_id}").json()
    assert poll["status"] == "done"
    assert poll["verdict"]["accepted"] is True


def test_error_in_runner_becomes_error_event():
    def runner(req, emit):
        raise RuntimeError("boom")

    c = TestClient(create_app(runner=runner))
    eval_id = c.post("/eval", json=REQ).json()["eval_id"]
    events = _stream_events(c, eval_id)
    assert events[-1]["phase"] == "error"
    assert "boom" in events[-1]["error"]
    assert c.get(f"/eval/{eval_id}").json()["status"] == "error"


def test_single_flight_409_while_busy():
    release = threading.Event()

    def blocking_runner(req, emit):
        release.wait(timeout=5)
        return {"accepted": False, "verdict": "king"}

    c = TestClient(create_app(runner=blocking_runner))
    first = c.post("/eval", json=REQ)
    assert first.status_code == 200

    # Second request while the first still holds the lock -> 409.
    second = c.post("/eval", json=REQ)
    assert second.status_code == 409
    assert "already running" in second.json()["error"]

    # Let the first finish, drain its stream, then a new eval is accepted.
    release.set()
    _stream_events(c, first.json()["eval_id"])
    third = c.post("/eval", json=REQ)
    assert third.status_code == 200


def test_unknown_eval_id_404():
    c = TestClient(create_app(runner=lambda req, emit: {"accepted": False}))
    assert c.get("/eval/does-not-exist").status_code == 404
    assert c.get("/eval/does-not-exist/stream").status_code == 404
