"""On-chain anchored allowlist: publish/read round-trip + tamper rejection."""
import io

import pytest

from leoma.infra import onchain_allowlist as oa

OWNER = "5Owner000000000000000000000000000000000000000"
VALS = ["5Bbb", "5Aaa", "5Ccc"]


class _Resp:
    def __init__(self, data): self._d = data
    def read(self): return self._d
    def close(self): pass
    def release_conn(self): pass


class _FakeBucket:
    """In-memory minio stand-in: {(bucket, key): bytes}."""
    def __init__(self): self.store = {}
    def put_object(self, bucket, key, data, length, content_type=None):
        self.store[(bucket, key)] = data.read()
    def get_object(self, bucket, key):
        if (bucket, key) not in self.store:
            raise FileNotFoundError(key)
        return _Resp(self.store[(bucket, key)])


class _FakeSubtensor:
    def __init__(self): self.commitments = {}
    async def set_commitment(self, wallet, netuid, data, period=32):
        self.commitments[OWNER] = data   # signed by the owner wallet
        return True
    async def get_subnet_owner_hotkey(self, netuid, block=None): return OWNER
    async def get_all_commitments(self, netuid, **kw): return dict(self.commitments)


def test_canonical_payload_is_sorted_and_deterministic():
    a = oa.canonical_payload(["5Ccc", "5Aaa", "5Bbb"], 100)
    b = oa.canonical_payload(["5Aaa", "5Bbb", "5Ccc"], 100)
    assert a == b                                  # order-independent
    assert oa.digest_of(a) == oa.digest_of(b)
    import json
    assert json.loads(a)["validators"] == ["5Aaa", "5Bbb", "5Ccc"]


def test_parse_commitment():
    assert oa.parse_commitment(oa.COMMIT_PREFIX + "abc123") == "abc123"
    assert oa.parse_commitment("something-else") is None
    assert oa.parse_commitment(None) is None


async def test_publish_then_read_roundtrip():
    sub, bucket = _FakeSubtensor(), _FakeBucket()
    digest = await oa.publish_allowlist(sub, wallet=None, netuid=99,
                                        write_client=bucket, source_bucket="src",
                                        validators=VALS, interval=100)
    assert sub.commitments[OWNER] == oa.COMMIT_PREFIX + digest

    snap = await oa.read_allowlist(sub, 99, bucket, "src")
    assert snap is not None
    assert snap.validators == sorted(VALS)         # sorted = rotation order
    assert snap.interval == 100
    assert snap.digest == digest


async def test_read_rejects_tampered_file():
    sub, bucket = _FakeSubtensor(), _FakeBucket()
    await oa.publish_allowlist(sub, None, 99, bucket, "src", VALS, 100)
    # Tamper with the off-chain file without updating the on-chain hash.
    bucket.store[("src", oa.ALLOWLIST_OBJECT_KEY)] = oa.canonical_payload(VALS + ["5Evil"], 100)
    assert await oa.read_allowlist(sub, 99, bucket, "src") is None


async def test_read_none_when_no_commitment():
    sub, bucket = _FakeSubtensor(), _FakeBucket()
    # nothing committed yet
    assert await oa.read_allowlist(sub, 99, bucket, "src") is None
