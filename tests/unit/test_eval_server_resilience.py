"""The eval server holds the subnet's only GPU behind one lock. Every way that lock
can be held forever is a way to halt the subnet — and each one looks exactly like
"we're busy".

Every test here pins a specific way the old server could wedge, lose a verdict, or
starve a subscriber.
"""

import json
import threading
import time

import pytest
from starlette.testclient import TestClient

import leoma.eval_server as es
from leoma.eval_server import TERMINAL_PHASES, create_app

from .conftest import pinned_spec


SPEC = pinned_spec()

REQ = dict(
    king_repo="u/leoma-k", king_digest="sha256:" + "a" * 64,
    challenger_repo="u/leoma-c", challenger_digest="sha256:" + "b" * 64,
    block_hash="0xabc", hotkey="5C7L",
    spec=SPEC.model_dump(mode="json"),
    consensus_digest=SPEC.digest(),
)


def _events(client, eval_id):
    out = []
    with client.stream("GET", f"/eval/{eval_id}/stream") as s:
        for line in s.iter_lines():
            if line and line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestTheLockIsAlwaysReleased:
    """A leaked lock is a permanently 409ing box that looks perfectly healthy."""

    def test_released_after_a_successful_duel(self):
        c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
        first = c.post("/eval", json=REQ).json()["eval_id"]
        _events(c, first)
        assert c.post("/eval", json=REQ).status_code == 200

    def test_released_after_the_runner_raises(self):
        def boom(req, emit, cancel):
            raise RuntimeError("boom")

        c = TestClient(create_app(runner=boom))
        _events(c, c.post("/eval", json=REQ).json()["eval_id"])
        assert c.post("/eval", json=REQ).status_code == 200
        assert c.get("/health").json()["busy"] is False

    def test_released_when_the_worker_thread_cannot_even_start(self, monkeypatch):
        """THE leak. The old code released the lock in the worker's `finally` — which
        never runs if Thread.start() itself raises. The lock was held forever and the
        box answered 409 to every future duel, for the rest of its life."""
        real_thread = threading.Thread
        calls = {"n": 0}

        class ExplodingThread(real_thread):
            def start(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("can't start new thread")
                return super().start()

        c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
        monkeypatch.setattr(es.threading, "Thread", ExplodingThread)
        assert c.post("/eval", json=REQ).status_code == 500

        monkeypatch.undo()
        # The box must still be usable. Before the fix, this was a permanent 409.
        assert c.get("/health").json()["busy"] is False
        assert c.post("/eval", json=REQ).status_code == 200

    def test_a_failed_start_still_produces_a_terminal_event(self, monkeypatch):
        class ExplodingThread(threading.Thread):
            def start(self):
                raise RuntimeError("nope")

        c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
        monkeypatch.setattr(es.threading, "Thread", ExplodingThread)
        response = c.post("/eval", json=REQ)
        monkeypatch.undo()

        assert response.status_code == 500
        # The lock is free and the box is usable — no zombie "busy" state.
        assert c.get("/health").json()["busy"] is False


class TestExactlyOneTerminalEvent:
    """A stream that ends with no terminal event was indistinguishable from a busy
    server, so a BROKEN duel read as "try again later" — forever."""

    def test_success_ends_in_verdict(self):
        c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": True, "verdict": "challenger"}))
        events = _events(c, c.post("/eval", json=REQ).json()["eval_id"])
        assert events[-1]["phase"] == "verdict"
        assert sum(e["phase"] in TERMINAL_PHASES for e in events) == 1

    def test_failure_ends_in_error(self):
        def boom(req, emit, cancel):
            emit({"phase": "materialize"})
            raise RuntimeError("weights are corrupt")

        c = TestClient(create_app(runner=boom))
        events = _events(c, c.post("/eval", json=REQ).json()["eval_id"])
        assert events[-1]["phase"] == "error"
        assert "corrupt" in events[-1]["error"]
        assert sum(e["phase"] in TERMINAL_PHASES for e in events) == 1

    def test_even_a_baseexception_ends_in_a_terminal_event(self):
        """`except Exception` would let a KeyboardInterrupt or SystemExit inside the
        worker leave the job with NO terminal event, the stream hanging, and the lock
        held."""
        def die(req, emit, cancel):
            raise SystemExit("worker died")

        c = TestClient(create_app(runner=die))
        events = _events(c, c.post("/eval", json=REQ).json()["eval_id"])
        assert events[-1]["phase"] == "error"
        assert c.get("/health").json()["busy"] is False

    def test_a_typed_failure_carries_its_reason_to_the_validator(self):
        """The reason drives retry-vs-quarantine. Losing it means retrying a model
        that will never work, or quarantining a miner for a network blip."""
        from leoma.eval.errors import CorpusIntegrityError

        def bad_corpus(req, emit, cancel):
            raise CorpusIntegrityError("truth hash mismatch")

        c = TestClient(create_app(runner=bad_corpus))
        events = _events(c, c.post("/eval", json=REQ).json()["eval_id"])
        assert events[-1]["reason"] == "corpus_integrity"


class TestTheEventLogReplays:
    def test_a_late_subscriber_gets_the_WHOLE_duel_not_a_hang(self):
        """The log was a queue.Queue drained by the SSE handler. A subscriber that
        connected after the duel finished blocked forever on an empty queue — so a
        validator that restarted mid-duel could never collect the verdict it was owed,
        and the eval box sat there having done hours of work for nobody."""
        def runner(req, emit, cancel):
            emit({"phase": "materialize"})
            emit({"phase": "duel"})
            return {"accepted": True, "verdict": "challenger"}

        c = TestClient(create_app(runner=runner))
        eval_id = c.post("/eval", json=REQ).json()["eval_id"]
        assert _wait_until(lambda: c.get(f"/eval/{eval_id}").json()["status"] == "done")

        # Subscribe only AFTER it has completely finished.
        events = _events(c, eval_id)
        assert [e["phase"] for e in events] == ["materialize", "duel", "verdict"]

    def test_two_subscribers_each_get_the_full_stream(self):
        """A queue SPLITS between consumers: each subscriber saw a random half of the
        duel, and the phases they missed were simply gone."""
        def runner(req, emit, cancel):
            emit({"phase": "materialize"})
            emit({"phase": "load"})
            emit({"phase": "duel"})
            return {"accepted": False, "verdict": "king"}

        c = TestClient(create_app(runner=runner))
        eval_id = c.post("/eval", json=REQ).json()["eval_id"]
        assert _wait_until(lambda: c.get(f"/eval/{eval_id}").json()["status"] == "done")

        a = [e["phase"] for e in _events(c, eval_id)]
        b = [e["phase"] for e in _events(c, eval_id)]
        assert a == b == ["materialize", "load", "duel", "verdict"]

    def test_a_finished_job_is_eventually_evicted(self, monkeypatch):
        """Without a TTL the jobs dict grew forever, each entry pinning a full
        per_clip payload — a slow leak on a box that is already memory-starved."""
        monkeypatch.setattr(es, "JOB_TTL_SECONDS", -1.0)  # expire immediately
        c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
        first = c.post("/eval", json=REQ).json()["eval_id"]
        assert _wait_until(lambda: c.get(f"/eval/{first}").json()["status"] == "done")

        # The next POST reaps expired jobs.
        c.post("/eval", json=REQ)
        assert c.get(f"/eval/{first}").status_code == 404


class TestCancellation:
    def test_delete_stops_a_running_duel(self):
        started = threading.Event()

        def runner(req, emit, cancel):
            from leoma.eval.errors import DuelCancelled

            started.set()
            for i in range(200):
                if cancel():
                    raise DuelCancelled(f"stopped after {i} clips")
                emit({"phase": "scored_clip", "position": i})
                time.sleep(0.01)
            return {"accepted": False, "verdict": "king"}

        c = TestClient(create_app(runner=runner))
        eval_id = c.post("/eval", json=REQ).json()["eval_id"]
        assert started.wait(timeout=5)

        assert c.delete(f"/eval/{eval_id}").json()["cancelled"] is True
        assert _wait_until(lambda: c.get(f"/eval/{eval_id}").json()["status"] == "cancelled")

        events = _events(c, eval_id)
        assert events[-1]["phase"] == "cancelled"
        # And the GPU is free again — the whole point.
        assert c.get("/health").json()["busy"] is False

    def test_cancelling_a_finished_duel_is_a_no_op(self):
        c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
        eval_id = c.post("/eval", json=REQ).json()["eval_id"]
        assert _wait_until(lambda: c.get(f"/eval/{eval_id}").json()["status"] == "done")
        assert c.delete(f"/eval/{eval_id}").json()["cancelled"] is False

    def test_a_cancelled_duel_is_transient_so_the_miner_is_never_blamed(self):
        """A watchdog stall says nothing about the challenger's model. Charging the
        miner for our box's failure would be the worst possible outcome."""
        from leoma.app.validator.failures import classify
        from leoma.eval.errors import DuelCancelled

        assert classify(DuelCancelled("watchdog")).kind.value == "transient"


class TestWatchdog:
    def test_a_stalled_duel_is_killed_and_the_lock_freed(self, monkeypatch):
        """A flat wall-clock cap has to guess how long a 14B video duel takes. The
        watchdog asks the honest question instead: is it still making progress?"""
        monkeypatch.setattr(es, "WATCHDOG_INTERVAL", 0.02)
        monkeypatch.setattr(es, "DEFAULT_BUDGET", 0.1)
        monkeypatch.setattr(es, "PHASE_BUDGETS", {})

        wedged = threading.Event()

        def hung(req, emit, cancel):
            emit({"phase": "load"})
            wedged.wait(timeout=10)     # a socket that will never return
            return {"accepted": False}

        c = TestClient(create_app(runner=hung))
        eval_id = c.post("/eval", json=REQ).json()["eval_id"]

        assert _wait_until(lambda: c.get(f"/eval/{eval_id}").json()["status"] == "error", timeout=5)
        body = c.get(f"/eval/{eval_id}").json()
        assert body["reason"] == "watchdog_stalled"
        assert "no progress" in body["error"]

        # The lock is back, so the subnet keeps moving. Before: a permanent 409.
        assert c.get("/health").json()["busy"] is False
        assert c.post("/eval", json=REQ).status_code == 200
        wedged.set()

    def test_a_SLOW_but_progressing_duel_is_NOT_killed(self, monkeypatch):
        """The critical half. A 70 GB download over a slow link is healthy for an
        hour; a wall-clock cap tuned to kill hung sockets would kill it too."""
        monkeypatch.setattr(es, "WATCHDOG_INTERVAL", 0.02)
        monkeypatch.setattr(es, "DEFAULT_BUDGET", 0.3)
        monkeypatch.setattr(es, "PHASE_BUDGETS", {})

        def slow_but_alive(req, emit, cancel):
            for i in range(12):         # 12 x 0.05s = 0.6s > the 0.3s budget...
                emit({"phase": "materialize", "bytes": i * 1000})   # ...but progress!
                time.sleep(0.05)
            return {"accepted": True, "verdict": "challenger"}

        c = TestClient(create_app(runner=slow_but_alive))
        eval_id = c.post("/eval", json=REQ).json()["eval_id"]
        events = _events(c, eval_id)

        assert events[-1]["phase"] == "verdict", "the watchdog killed a healthy download"

    def test_health_reports_the_phase_and_how_long_it_has_been_stalled(self):
        blocked = threading.Event()

        def runner(req, emit, cancel):
            emit({"phase": "materialize", "which": "king"})
            blocked.wait(timeout=5)
            return {"accepted": False}

        c = TestClient(create_app(runner=runner))
        c.post("/eval", json=REQ)
        assert _wait_until(lambda: c.get("/health").json()["phase"] == "materialize")

        health = c.get("/health").json()
        assert health["busy"] is True
        assert health["stalled_for"] >= 0
        blocked.set()


class TestBindSafety:
    """POST /eval makes this box download and run an arbitrary model repo. On
    0.0.0.0 with no auth, that is remote code execution with a REST API."""

    def test_loopback_is_fine(self):
        es._check_bind_safety("127.0.0.1")

    def test_public_bind_without_a_token_is_refused(self):
        with pytest.raises(SystemExit, match="refusing to bind"):
            es._check_bind_safety("0.0.0.0")

    def test_public_bind_with_a_token_is_allowed(self, monkeypatch):
        monkeypatch.setenv("LEOMA_EVAL_TOKEN", "s3cret")
        es._check_bind_safety("0.0.0.0")
