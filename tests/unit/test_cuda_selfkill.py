"""CUDA-context-poison self-kill.

Once a CUDA context is corrupted — an illegal memory access, a device-side assert, a
cuBLAS execution failure — the allocator and every stream on the process are poisoned,
and every subsequent load/generate keeps raising against the dead context. The
watchdog's `os._exit` only fires when a duel *stalls*; a duel that FAILS FAST on a CUDA
fault would release the lock and hand the next challenger a poisoned box, mis-rejecting
an honest model as broken. For Leoma that wastes *hours* of the next duel, not minutes.

The only recovery is to exit and let the supervisor restart with a fresh context. This
is safe precisely because the chain is the queue: the restart loses at most the current
duel, which the validator re-dispatches. Ported from Teutonic's self-kill.
"""

import threading
import time

import pytest
from starlette.testclient import TestClient

import leoma.eval_server as es
from leoma.eval_server import create_app, is_cuda_fatal

from .conftest import pinned_spec


SPEC = pinned_spec()
REQ = dict(
    king_repo="u/leoma-k", king_digest="sha256:" + "a" * 64,
    challenger_repo="u/leoma-c", challenger_digest="sha256:" + "b" * 64,
    block_hash="0xabc", hotkey="5C7L",
    spec=SPEC.model_dump(mode="json"), consensus_digest=SPEC.digest(),
)


@pytest.fixture
def caught_exit(monkeypatch):
    """Replace os._exit with a recorder and reset the idempotency latch."""
    monkeypatch.setattr(es, "_self_kill_scheduled", threading.Event())
    monkeypatch.setattr(es, "CUDA_FATAL_EXIT_DELAY_S", 0.0)
    calls: list[int] = []
    done = threading.Event()
    monkeypatch.setattr(es.os, "_exit", lambda code: (calls.append(code), done.set()))
    return calls, done


class TestIsCudaFatal:
    @pytest.mark.parametrize("msg", [
        "RuntimeError: CUDA error: an illegal memory access was encountered",
        "cudaErrorIllegalAddress",
        "CUDA error: device-side assert triggered",
        "CUDA error: misaligned address",
        "CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx",
        "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED",
        "Bus error",
        "Segmentation fault",
    ])
    def test_fatal_tokens_are_recognized(self, msg):
        assert is_cuda_fatal(msg) is True

    def test_case_insensitive(self):
        assert is_cuda_fatal("cuda error: MISALIGNED ADDRESS") is True

    @pytest.mark.parametrize("msg", [
        "CUDA out of memory. Tried to allocate 2.00 GiB",   # recoverable — empty_cache + retry
        "RepositoryNotFoundError: repo missing",
        "connection reset by peer",
        "ValueError: generation too short",
        "",
    ])
    def test_recoverable_and_unrelated_errors_are_NOT_fatal(self, msg):
        # OOM especially must not self-kill: it is transient and the box recovers.
        assert is_cuda_fatal(msg) is False

    def test_accepts_an_exception_object(self):
        assert is_cuda_fatal(RuntimeError("CUDA error: an illegal memory access")) is True


class TestScheduleSelfKill:
    def test_it_eventually_exits_with_the_configured_code(self, caught_exit):
        calls, done = caught_exit
        es.schedule_self_kill("test")
        assert done.wait(timeout=5)
        assert calls == [es.CUDA_FATAL_EXIT_CODE]

    def test_it_is_idempotent(self, caught_exit):
        calls, done = caught_exit
        es.schedule_self_kill("first")
        es.schedule_self_kill("second")
        es.schedule_self_kill("third")
        assert done.wait(timeout=5)
        time.sleep(0.05)
        assert calls == [es.CUDA_FATAL_EXIT_CODE], "scheduled the exit more than once"


class TestWiredIntoTheWorker:
    def _events(self, client, eval_id):
        out = []
        with client.stream("GET", f"/eval/{eval_id}/stream") as s:
            for line in s.iter_lines():
                if line and line.startswith("data: "):
                    import json
                    out.append(json.loads(line[6:]))
        return out

    def test_a_cuda_fatal_duel_reports_its_error_AND_self_kills(self, caught_exit):
        """Two things must both happen: the validator still gets a terminal error event
        (so it can retry), and the box schedules its own restart."""
        calls, done = caught_exit

        def poisoned(req, emit, cancel):
            emit({"phase": "load"})
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        c = TestClient(create_app(runner=poisoned))
        eval_id = c.post("/eval", json=REQ).json()["eval_id"]
        events = self._events(c, eval_id)

        # The validator is told — as a normal terminal error, so it retries elsewhere.
        assert events[-1]["phase"] == "error"
        assert "illegal memory access" in events[-1]["error"]
        # And the box is taking itself down for a fresh context.
        assert done.wait(timeout=5)
        assert calls == [es.CUDA_FATAL_EXIT_CODE]

    def test_an_ordinary_duel_failure_does_NOT_self_kill(self, caught_exit):
        """A 404 or a bad model is not a poisoned context — the box must keep serving."""
        calls, done = caught_exit

        def broken_model(req, emit, cancel):
            raise RuntimeError("RepositoryNotFoundError: repo missing")

        c = TestClient(create_app(runner=broken_model))
        eval_id = c.post("/eval", json=REQ).json()["eval_id"]
        self._events(c, eval_id)

        assert not done.wait(timeout=0.5)
        assert calls == []
        # And the box is still usable.
        assert c.get("/health").json()["busy"] is False

    def test_an_oom_does_NOT_self_kill(self, caught_exit):
        """The one that would be easy to get wrong: OOM is transient, not fatal."""
        calls, done = caught_exit

        def oom(req, emit, cancel):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        c = TestClient(create_app(runner=oom))
        eval_id = c.post("/eval", json=REQ).json()["eval_id"]
        self._events(c, eval_id)

        assert not done.wait(timeout=0.5)
        assert calls == []
