"""Local scoring-window derivation from peer-bucket producedness."""
from leoma.app.validator.window_local import derive_window, resolve_canonical_samplers

V = ["A", "B", "C", "D"]  # rotation order: sampler(rid) = V[rid % 4]


def test_canonical_sampler_is_the_scheduled_primary():
    # each validator produced exactly its own scheduled rotations
    produced = {
        "A": {0, 4, 8}, "B": {1, 5}, "C": {2, 6}, "D": {3, 7},
    }
    canon = resolve_canonical_samplers(produced, V)
    for rid, hk in canon.items():
        assert hk == V[rid % 4]


def test_failover_backup_wins_only_when_primary_absent():
    # rid 0 primary is A; A skipped it, backup B (step 1) produced it.
    produced = {"B": {0, 1}}
    canon = resolve_canonical_samplers(produced, V)
    assert canon[0] == "B"   # B is V[(0+1)%4], the first failover backup
    # rid 1 primary is B and B produced -> primary wins
    assert canon[1] == "B"


def test_primary_wins_over_late_backup_duplicate():
    # both A (primary, step 0) and B (step 1) produced rid 0 -> primary A is canonical
    produced = {"A": {0}, "B": {0}}
    assert resolve_canonical_samplers(produced, V)[0] == "A"


def test_illegitimate_producer_is_ignored():
    # C is neither primary nor an early backup for rid 0 within reach? C = V[(0+2)%4] is a backup,
    # but only counts if no earlier-order producer exists. With ONLY C producing rid 0, C wins
    # (it's the earliest-order producer present). A producer off the ring can't happen (set = V).
    assert resolve_canonical_samplers({"C": {0}}, V)[0] == "C"
    # empty validator set -> nothing resolvable
    assert resolve_canonical_samplers({"A": {0}}, []) == {}


def test_derive_window_anchor_margin_and_order():
    interval = 100
    canon = {r: V[r % 4] for r in range(0, 12)}     # rotations 0..11 produced
    # epoch at block 900 -> epoch_rid 9, so rids 10,11 (block 1000,1100) are excluded by the anchor.
    window, active = derive_window(canon, epoch_block=900, interval=interval, n=100, margin=2, max_lookback=0)
    # anchored rids = 0..9; drop newest margin (9,8) -> 0..7 ascending
    assert window == list(range(0, 8))
    assert active == ["A", "B", "C", "D"]


def test_derive_window_caps_to_n():
    interval = 100
    canon = {r: V[r % 4] for r in range(0, 50)}
    window, _ = derive_window(canon, epoch_block=10_000, interval=interval, n=10, margin=2, max_lookback=0)
    assert len(window) == 10
    assert window == sorted(window)                 # ascending
