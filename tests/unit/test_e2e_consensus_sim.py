"""
End-to-end DECENTRALIZED consensus simulation (production launch validation).

Owner-api-FREE: the validator set is read from the on-chain-anchored allowlist and the settled
scoring window is derived by each validator itself from peer-bucket producedness — there is no
/rotation and no /tasks/window. Each validator independently runs the real ``compute_local_winner``
over the shared (in-memory) buckets, and they all select the IDENTICAL winner UID.

The story also proves: per-validator-average (not task-pooled), the dominance hold (the
earliest-registered miner keeps #1 over a higher-scoring rival within the 5% margin), the
completeness gate, the min-distinct-validators gate, and the settle margin (filler tasks dropped).

Run with ``-s`` to see the report.
"""
import json
from collections import defaultdict

import pytest

from leoma.app.validator import aggregate_local, miner_validation as mv
from leoma.app.validator.aggregate_local import compute_local_winner
from leoma.bootstrap import NETUID, SOURCE_BUCKET
from leoma.infra import onchain_allowlist as oa
from leoma.infra.aggregate import compute_miner_aggregates, rank_from_aggregates
from leoma.infra.peer_registry import PeerBucket
from leoma.infra.scorer_constants import (
    COMPLETENESS_ELIGIBILITY_THRESHOLD,
    required_distinct_validators,
)

N_VALIDATORS = 4
TASKS_PER_VALIDATOR = 8
SETTLED = N_VALIDATORS * TASKS_PER_VALIDATOR   # 32 settled tasks
FILLER = 2                                     # 2 newest produced -> dropped by the settle margin
INTERVAL = 50
EPOCH_BLOCK = 2000                             # epoch_rid = 40 >= SETTLED+FILLER, so the margin (not
                                               # the block anchor) is what drops the fillers
OWNER = "5Owner0000000000000000000000000000000000000a"

VALIDATORS = [f"VAL_{i}" for i in range(N_VALIDATORS)]  # already sorted


class _Miner:
    def __init__(self, name, uid, block, eval_vals, predicate, passes):
        self.name, self.uid, self.block = name, uid, block
        self.hk = f"MINER_{name}"
        self.eval_vals = eval_vals
        self.predicate = predicate
        self.passes = passes
        self.passed_tasks = {}
        self.eval_tasks = {}


def _story():
    allv = set(range(N_VALIDATORS))
    return [
        _Miner("champ", uid=3, block=100, eval_vals=allv, predicate=lambda i: True,
               passes={0: 6, 1: 6, 2: 6, 3: 6}),                          # 0.75 -> holds #1
        _Miner("rival", uid=7, block=700, eval_vals=allv, predicate=lambda i: True,
               passes={0: 7, 1: 6, 2: 6, 3: 6}),                          # ~0.78 but within 5% -> #2
        _Miner("mid", uid=11, block=300, eval_vals=allv, predicate=lambda i: True,
               passes={0: 4, 1: 4, 2: 4, 3: 4}),
        _Miner("solo", uid=5, block=500, eval_vals={0}, predicate=lambda i: True,
               passes={0: 8}),                                            # 1 validator -> gated
        _Miner("incomplete", uid=2, block=200, eval_vals=allv, predicate=lambda i: i < SETTLED // 2,
               passes={0: 4, 1: 4, 2: 4, 3: 4}),                          # 50% completeness -> gated
    ]


def _generate(miners):
    """Per-(validator, task) verdict files (in-memory buckets) + the ground-truth verdict map."""
    vtasks = {idx: [i for i in range(SETTLED) if i % N_VALIDATORS == idx] for idx in range(N_VALIDATORS)}
    for m in miners:
        for idx in m.eval_vals:
            evaled = [i for i in vtasks[idx] if m.predicate(i)]
            m.eval_tasks[idx] = evaled
            m.passed_tasks[idx] = set(evaled[: m.passes[idx]])

    store: dict = defaultdict(dict)
    verdicts: dict = {}
    files = defaultdict(list)
    for i in range(SETTLED):
        idx = i % N_VALIDATORS
        vhk = VALIDATORS[idx]
        for m in miners:
            if idx in m.eval_vals and m.predicate(i):
                passed = i in m.passed_tasks[idx]
                files[(i, vhk)].append({"hotkey": m.hk, "passed": passed})
                verdicts[(vhk, i, m.hk)] = passed
    for (i, vhk), entries in files.items():
        store[f"bucket::{vhk}"][f"{i}/evaluation_results/{vhk}.json"] = json.dumps({"data": entries}).encode()
    # Filler produced tasks (newest) with no miner data: present in the buckets so the settle margin
    # drops exactly these, leaving the SETTLED window.
    for i in range(SETTLED, SETTLED + FILLER):
        vhk = VALIDATORS[i % N_VALIDATORS]
        store[f"bucket::{vhk}"][f"{i}/evaluation_results/{vhk}.json"] = json.dumps({"data": []}).encode()
    return store, verdicts


class _Resp:
    def __init__(self, data): self._d = data
    def read(self): return self._d
    def close(self): pass
    def release_conn(self): pass


class _Obj:
    def __init__(self, name): self.object_name = name


class _FakeBucketClient:
    """Shared in-memory minio: {bucket: {key: bytes}} with get/put/list."""
    def __init__(self, store): self._store = store
    def get_object(self, bucket, key):
        try:
            return _Resp(self._store[bucket][key])
        except KeyError:
            raise FileNotFoundError(f"{bucket}/{key}")
    def put_object(self, bucket, key, data, length, content_type=None):
        self._store.setdefault(bucket, {})[key] = data.read()
    def list_objects(self, bucket, prefix="", recursive=True):
        return [_Obj(k) for k in self._store.get(bucket, {}) if k.startswith(prefix or "")]


class _FakeSubtensor:
    def __init__(self): self.commitments = {}
    async def set_commitment(self, wallet, netuid, data, period=32):
        self.commitments[OWNER] = data
        return True
    async def get_subnet_owner_hotkey(self, netuid, block=None): return OWNER
    async def get_all_commitments(self, netuid, **kw): return dict(self.commitments)


class _MockMiner:
    def __init__(self, hotkey, uid, block):
        self.hotkey, self.uid, self.block = hotkey, uid, block


@pytest.fixture
def sim(tmp_path, monkeypatch):
    monkeypatch.setenv("LEOMA_STATE_DIR", str(tmp_path))       # isolate last-winner persistence
    monkeypatch.setattr(mv, "current_snapshot", lambda: None)  # force the get_all_miners path

    miners = _story()
    store, verdicts = _generate(miners)

    peers = {
        hk: PeerBucket(hotkey=hk, bucket=f"bucket::{hk}", endpoint="x", region="auto",
                       read_access_key="x", read_secret_key="x")
        for hk in VALIDATORS
    }
    monkeypatch.setattr(aggregate_local, "load_peers", lambda: peers)
    monkeypatch.setattr(aggregate_local, "create_peer_read_client", lambda peer: _FakeBucketClient(store))

    sub = _FakeSubtensor()
    source_client = _FakeBucketClient(store)
    mock_miners = [_MockMiner(m.hk, m.uid, m.block) for m in miners]

    async def _setup():
        # Owner anchors the allowlist on-chain (hash) + to the shared source bucket (the file).
        await oa.publish_allowlist(sub, None, NETUID, source_client, SOURCE_BUCKET, VALIDATORS, INTERVAL)

    return {"miners": miners, "verdicts": verdicts, "mock_miners": mock_miners,
            "sub": sub, "source_client": source_client, "setup": _setup}


async def test_e2e_decentralized_consensus(sim, capsys):
    await sim["setup"]()
    miners = sim["miners"]
    verdicts = sim["verdicts"]
    by_uid = {m.uid: m for m in miners}

    async def _get_all_miners():
        return sim["mock_miners"]

    # --- Each validator INDEPENDENTLY derives the winner: on-chain allowlist + bucket-derived window ---
    results = {}
    for vhk in VALIDATORS:
        uid, hotkey = await compute_local_winner(
            sim["sub"], epoch_block=EPOCH_BLOCK,
            source_read_client=sim["source_client"], get_all_miners=_get_all_miners,
        )
        results[vhk] = (uid, hotkey)
    selected_uids = {uid for uid, _ in results.values()}

    # --- Independent report via the real aggregation (transparency) -------------------------------
    block_by_hotkey = {m.hk: m.block for m in miners}
    window_ids = list(range(SETTLED))
    aggs = compute_miner_aggregates(verdicts, window_ids, block_by_hotkey)
    active = {VALIDATORS[i % N_VALIDATORS] for i in range(SETTLED)}
    min_distinct = required_distinct_validators(len(active))
    winner_hotkey, ranking = rank_from_aggregates(aggs, set(), 0.05, min_distinct_validators=min_distinct)

    print("\n" + "=" * 78)
    print(f"DECENTRALIZED CONSENSUS (owner-api-free) — {N_VALIDATORS} validators, {SETTLED} settled "
          f"tasks (+{FILLER} dropped by settle margin); allowlist read from chain")
    print("=" * 78)
    for vhk, (uid, _) in results.items():
        print(f"  {vhk}: winner uid={uid}")
    print(f"=> CONSENSUS: {selected_uids} ({'AGREE' if len(selected_uids) == 1 else 'DISAGREE!'})")

    # 1) Every validator independently selects the IDENTICAL winner UID (on-chain determinism).
    assert len(selected_uids) == 1, f"validators disagreed: {results}"
    winner_uid = selected_uids.pop()
    # 2) Winner is the champion (uid 3), held #1 by dominance despite the rival scoring higher.
    assert winner_uid == 3
    assert winner_hotkey == by_uid[3].hk
    assert aggs[by_uid[7].hk].avg_rate > aggs[by_uid[3].hk].avg_rate
    assert ranking[0]["miner_hotkey"] == by_uid[3].hk
    # 3) Gates exclude the bad miners; ranking is champ > rival > mid.
    ranked_uids = [next(m for m in miners if m.hk == e["miner_hotkey"]).uid for e in ranking]
    assert ranked_uids == [3, 7, 11]
    assert by_uid[5].hk not in {e["miner_hotkey"] for e in ranking}   # solo: too few validators
    assert by_uid[2].hk not in {e["miner_hotkey"] for e in ranking}   # incomplete: completeness
