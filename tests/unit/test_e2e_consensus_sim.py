"""
End-to-end consensus simulation (production launch validation).

Exercises the REAL decentralized weight-setting pipeline with mock evaluation data:

  1. N validators registered in the owner-managed allowlist.
  2. The owner-api produced-task ledger is built (rotating sampler per task).
  3. Each validator writes its own mock verdicts to its own (in-memory) bucket.
  4. Each validator INDEPENDENTLY runs the real ``compute_local_winner``: fetches the settled window
     from the real owner-api logic, reads every peer's shared verdicts, aggregates per-validator
     (equal weight), applies the completeness + min-distinct gates, ranks by dominance.
  5. We assert all validators select the IDENTICAL winner UID (the value that would be set on-chain)
     and that the gates exclude the bad miners. No real set_weights call — just the UID selection.

The designed story also proves: per-validator-average (not task-pooled), the dominance hold (the
earliest-registered miner keeps #1 over a higher-scoring rival within the 5% margin), the
completeness gate, the min-distinct-validators gate, and the settle margin.
"""
import json
from collections import defaultdict

import pytest

from leoma.app.validator import aggregate_local
from leoma.app.validator.aggregate_local import compute_local_winner
from leoma.infra.aggregate import compute_miner_aggregates, rank_from_aggregates
from leoma.infra.db.stores import ProducedTaskStore, ValidatorStore
from leoma.infra.peer_registry import PeerBucket
from leoma.infra.scorer_constants import (
    COMPLETENESS_ELIGIBILITY_THRESHOLD,
    required_distinct_validators,
)

N_VALIDATORS = 4
TASKS_PER_VALIDATOR = 8
SETTLED = N_VALIDATORS * TASKS_PER_VALIDATOR  # 32 settled tasks
FILLER = 2                                    # 2 newest -> dropped by the settle margin
EPOCH_BLOCK = 2000

VALIDATORS = [f"VAL_{i}" for i in range(N_VALIDATORS)]  # already sorted


class _Miner:
    def __init__(self, name, uid, block, eval_vals, predicate, passes):
        self.name = name
        self.hk = f"MINER_{name}"
        self.uid = uid
        self.block = block
        self.eval_vals = eval_vals          # validator indices that evaluate this miner
        self.predicate = predicate          # which task indices it's evaluated on
        self.passes = passes                # {validator_idx: pass_count among its evaluated tasks}
        self.passed_tasks = {}              # filled during generation: idx -> set(task_i that pass)
        self.eval_tasks = {}                # idx -> list(task_i evaluated)


def _story():
    allv = set(range(N_VALIDATORS))
    return [
        # champ: earliest block, 0.75 everywhere -> holds #1 by dominance.
        _Miner("champ", uid=3, block=100, eval_vals=allv, predicate=lambda i: True,
               passes={0: 6, 1: 6, 2: 6, 3: 6}),
        # rival: later block, scores HIGHER (mean ~0.78) but not by the 5% margin -> stays #2.
        _Miner("rival", uid=7, block=700, eval_vals=allv, predicate=lambda i: True,
               passes={0: 7, 1: 6, 2: 6, 3: 6}),
        # mid: clearly third.
        _Miner("mid", uid=11, block=300, eval_vals=allv, predicate=lambda i: True,
               passes={0: 4, 1: 4, 2: 4, 3: 4}),
        # solo: only ONE validator evaluates it -> gated by min-distinct (and completeness).
        _Miner("solo", uid=5, block=500, eval_vals={0}, predicate=lambda i: True,
               passes={0: 8}),
        # incomplete: all validators, but only the first half of the window -> gated by completeness.
        _Miner("incomplete", uid=2, block=200, eval_vals=allv, predicate=lambda i: i < SETTLED // 2,
               passes={0: 4, 1: 4, 2: 4, 3: 4}),
    ]


def _generate(miners):
    """Build per-(validator, task) verdict files (in-memory buckets) + the ground-truth verdict map."""
    vtasks = {idx: [i for i in range(SETTLED) if i % N_VALIDATORS == idx] for idx in range(N_VALIDATORS)}
    for m in miners:
        for idx in m.eval_vals:
            evaled = [i for i in vtasks[idx] if m.predicate(i)]
            m.eval_tasks[idx] = evaled
            m.passed_tasks[idx] = set(evaled[: m.passes[idx]])

    # bucket store: {bucket_name: {object_key: bytes}}; verdict map: {(val_hk, task_i, miner_hk): passed}
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
        store[f"bucket::{vhk}"][f"{i}/evaluation_results/{vhk}.json"] = json.dumps(
            {"data": entries}
        ).encode()
    return store, verdicts


class _Resp:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def close(self):
        pass

    def release_conn(self):
        pass


class _FakeBucketClient:
    """Stand-in for the minio peer-read client, backed by the shared in-memory bucket store."""

    def __init__(self, store):
        self._store = store

    def get_object(self, bucket, key):
        try:
            return _Resp(self._store[bucket][key])
        except KeyError:
            raise FileNotFoundError(f"{bucket}/{key}")


class _MockMiner:
    def __init__(self, hotkey, uid, block):
        self.hotkey = hotkey
        self.uid = uid
        self.block = block


class _SimAPIClient:
    """Minimal owner-api client for one validator: real window logic + mock miner registry."""

    def __init__(self, miners):
        self._miners = miners

    async def get_task_window(self, as_of_block=None):
        # Calls the REAL owner-api route logic (produced-task ledger window, server-default settle margin).
        from leoma.delivery.http.routes.tasks import get_scoring_window

        return await get_scoring_window(hotkey="caller", as_of_block=as_of_block)

    async def get_all_miners(self):
        return self._miners

    async def close(self):
        pass


@pytest.fixture
def sim(mock_get_session, tmp_path, monkeypatch):
    """Wire the simulation: register validators, build the ledger, stage mock verdicts."""
    monkeypatch.setenv("LEOMA_STATE_DIR", str(tmp_path))  # isolate last-winner persistence

    miners = _story()
    store, verdicts = _generate(miners)

    # Peer ring: every validator's bucket, read by all (shared evaluation data).
    peers = {
        hk: PeerBucket(hotkey=hk, bucket=f"bucket::{hk}", endpoint="x", region="auto",
                       read_access_key="x", read_secret_key="x")
        for hk in VALIDATORS
    }
    monkeypatch.setattr(aggregate_local, "load_peers", lambda: peers)
    monkeypatch.setattr(aggregate_local, "create_peer_read_client", lambda peer: _FakeBucketClient(store))

    mock_miners = [_MockMiner(m.hk, m.uid, m.block) for m in miners]

    async def _setup():
        vstore = ValidatorStore()
        for idx, hk in enumerate(VALIDATORS):
            await vstore.save_validator(uid=idx, hotkey=hk, stake=0.0)
        ledger = ProducedTaskStore()
        # Append settled tasks (rotating sampler), then 2 newest fillers (dropped by settle margin).
        for i in range(SETTLED + FILLER):
            await ledger.append(rotation_id=i, sampler_hotkey=VALIDATORS[i % N_VALIDATORS], block=1000 + i)

    return {"miners": miners, "verdicts": verdicts, "mock_miners": mock_miners, "setup": _setup}


async def test_e2e_decentralized_consensus(sim, capsys):
    await sim["setup"]()
    miners = sim["miners"]
    verdicts = sim["verdicts"]
    by_uid = {m.uid: m for m in miners}

    # --- Each validator independently computes the winner from the shared verdicts ---------------
    results = {}
    for vhk in VALIDATORS:
        uid, hotkey = await compute_local_winner(_SimAPIClient(sim["mock_miners"]), epoch_block=EPOCH_BLOCK)
        results[vhk] = (uid, hotkey)

    selected_uids = {uid for uid, _ in results.values()}

    # --- Independent report via the real aggregation (transparency) -------------------------------
    block_by_hotkey = {m.hk: m.block for m in miners}
    window_ids = list(range(SETTLED))
    aggs = compute_miner_aggregates(verdicts, window_ids, block_by_hotkey)
    active = {VALIDATORS[i % N_VALIDATORS] for i in range(SETTLED)}
    min_distinct = required_distinct_validators(len(active))
    winner_hotkey, ranking = rank_from_aggregates(aggs, set(), 0.05, min_distinct_validators=min_distinct)

    # ============================ REPORT ============================
    print("\n" + "=" * 78)
    print(f"DECENTRALIZED CONSENSUS SIMULATION — {N_VALIDATORS} validators, "
          f"{SETTLED} settled tasks (+{FILLER} dropped by settle margin)")
    print(f"min distinct validators required = {min_distinct} (of {len(active)} active)")
    print("=" * 78)
    print(f"\n{'miner':<12}{'uid':>4}{'block':>7}   per-validator pass-rates      "
          f"{'mean':>6}{'compl':>7}  {'#val':>4}  status")
    print("-" * 78)
    for m in sorted(miners, key=lambda x: x.block):
        a = aggs.get(m.hk)
        rates = "  ".join(f"{a.per_validator_rate.get(v, float('nan')):.2f}" if a and v in a.per_validator_rate
                          else " -- " for v in VALIDATORS)
        mean = f"{a.avg_rate:.3f}" if a else "  -  "
        compl = f"{a.completeness:.2f}" if a else " - "
        nval = len(a.per_validator_rate) if a else 0
        reasons = []
        if a is not None and a.completeness < COMPLETENESS_ELIGIBILITY_THRESHOLD:
            reasons.append(f"incomplete {a.completeness:.0%}")
        if a is not None and nval < min_distinct:
            reasons.append(f"only {nval} validator(s)")
        if a is None:
            status = "no data"
        elif reasons:
            status = "GATED: " + ", ".join(reasons)
        else:
            rank = next((e["rank"] for e in ranking if e["miner_hotkey"] == m.hk), None)
            status = f"eligible (rank #{rank})"
        print(f"{m.name:<12}{m.uid:>4}{m.block:>7}   {rates}   {mean:>6}{compl:>7}  {nval:>4}  {status}")

    print("\nRanking (eligible miners):")
    for e in ranking:
        mm = next(m for m in miners if m.hk == e["miner_hotkey"])
        print(f"  #{e['rank']}  {mm.name:<10} uid={mm.uid:<3} rate={e['pass_rate']:.3f} "
              f"passed={e['passed_count']} block={e['block']}")

    print("\nIndependent winner UID per validator (what each would set on-chain):")
    for vhk, (uid, hk) in results.items():
        print(f"  {vhk}: winner uid={uid}")
    print(f"\n=> CONSENSUS: all validators selected uid={selected_uids} "
          f"({'AGREE' if len(selected_uids) == 1 else 'DISAGREE!'})")
    print("=" * 78)

    # ============================ ASSERTIONS ============================
    # 1) Every validator independently selects the IDENTICAL winner UID (the on-chain determinism guarantee).
    assert len(selected_uids) == 1, f"validators disagreed: {results}"
    winner_uid = selected_uids.pop()

    # 2) The winner is the champion (uid 3) — held #1 by dominance despite the rival scoring higher.
    assert winner_uid == 3
    assert winner_hotkey == by_uid[3].hk
    assert aggs[by_uid[7].hk].avg_rate > aggs[by_uid[3].hk].avg_rate  # rival scores higher...
    assert ranking[0]["miner_hotkey"] == by_uid[3].hk                 # ...yet champ is #1

    # 3) Gates exclude the bad miners; ranking is champ > rival > mid.
    ranked_uids = [by_uid_for(miners, e["miner_hotkey"]).uid for e in ranking]
    assert ranked_uids == [3, 7, 11]
    assert by_uid[5].hk not in {e["miner_hotkey"] for e in ranking}   # solo: too few validators
    assert by_uid[2].hk not in {e["miner_hotkey"] for e in ranking}   # incomplete: completeness


def by_uid_for(miners, hotkey):
    return next(m for m in miners if m.hk == hotkey)
