"""Shared unit-test fixtures.

The in-memory Minio stub lives here (it used to be local to
test_king_state_store.py) because several suites now need it — and because the
state-integrity tests need it to **inject faults**. That is the whole point of
that work: a transport error must no longer be indistinguishable from an empty
bucket.
"""

import pytest


class FakeS3Error(Exception):
    """Mirrors minio.error.S3Error's `.code` contract."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


class FakeMinio:
    """In-memory stand-in for the Minio client's put/get object API.

    Fault injection:
      fail_get / fail_put: {key: error_code} — raise FakeS3Error(code) for that key.
                           Use code="NoSuchKey" to simulate a genuine miss.
      flaky_get:           {key: n} — fail the first n GETs for that key, then succeed.
    """

    def __init__(self, *, fail_get=None, fail_put=None, flaky_get=None):
        self.blobs: dict[tuple, bytes] = {}
        self.fail_get = dict(fail_get or {})
        self.fail_put = dict(fail_put or {})
        self.flaky_get = dict(flaky_get or {})
        self.get_calls: list[str] = []
        self.put_calls: list[str] = []

    # ---- minio API surface -------------------------------------------------
    def put_object(self, bucket, key, data, length, content_type=None):
        self.put_calls.append(key)
        if key in self.fail_put:
            raise FakeS3Error(self.fail_put[key])
        payload = data.read()
        assert len(payload) == length
        self.blobs[(bucket, key)] = payload

    def get_object(self, bucket, key):
        self.get_calls.append(key)

        remaining = self.flaky_get.get(key, 0)
        if remaining:
            self.flaky_get[key] = remaining - 1
            raise FakeS3Error("InternalError", "transient")

        if key in self.fail_get:
            raise FakeS3Error(self.fail_get[key])

        if (bucket, key) not in self.blobs:
            raise FakeS3Error("NoSuchKey")

        blob = self.blobs[(bucket, key)]

        class _Resp:
            def read(self_inner):
                return blob

            def close(self_inner):
                pass

            def release_conn(self_inner):
                pass

        return _Resp()

    # ---- test helpers ------------------------------------------------------
    def seed_raw(self, bucket: str, key: str, raw: bytes) -> None:
        """Put raw bytes directly (e.g. to seed corrupt JSON)."""
        self.blobs[(bucket, key)] = raw


@pytest.fixture
def fake_minio() -> FakeMinio:
    return FakeMinio()
