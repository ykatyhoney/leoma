"""
Unit tests for deterministic duel seeds.

Every validator must derive the same clip set + generation seeds from the block
hash + hotkey, so the duel is reproducible and consensus converges.
"""

from leoma.app.validator.seeds import (
    eval_seed,
    eval_seed_material,
    clip_generation_seed,
    select_clip_indices,
)


class TestEvalSeed:
    def test_deterministic(self):
        assert eval_seed("0xabc", "5C7L", 0) == eval_seed("0xabc", "5C7L", 0)

    def test_block_hash_changes_seed(self):
        assert eval_seed("0xabc", "5C7L") != eval_seed("0xabd", "5C7L")

    def test_hotkey_changes_seed(self):
        assert eval_seed("0xabc", "A") != eval_seed("0xabc", "B")

    def test_base_seed_changes_seed(self):
        assert eval_seed("0xabc", "A", 0) != eval_seed("0xabc", "A", 1)

    def test_material_falls_back_without_block_hash(self):
        assert eval_seed_material("", "hk", 5) == "base_seed=5"
        assert eval_seed_material("default", "hk", 5) == "base_seed=5"
        assert "block_hash=0x1" in eval_seed_material("0x1", "hk", 5)


class TestClipSeeds:
    def test_same_clip_same_seed_for_both_models(self):
        master = eval_seed("0xabc", "hk")
        # king and challenger both call clip_generation_seed(master, i) -> identical
        assert clip_generation_seed(master, 3) == clip_generation_seed(master, 3)

    def test_different_clips_differ(self):
        master = eval_seed("0xabc", "hk")
        assert clip_generation_seed(master, 3) != clip_generation_seed(master, 4)


class TestSelectClipIndices:
    def test_deterministic_and_sorted(self):
        s = eval_seed("0xabc", "hk")
        a = select_clip_indices(s, 100, 10)
        b = select_clip_indices(s, 100, 10)
        assert a == b
        assert a == sorted(a)
        assert len(a) == 10
        assert len(set(a)) == 10  # distinct

    def test_all_when_n_ge_total(self):
        s = eval_seed("0xabc", "hk")
        assert select_clip_indices(s, 5, 10) == [0, 1, 2, 3, 4]

    def test_empty_edge_cases(self):
        s = eval_seed("0xabc", "hk")
        assert select_clip_indices(s, 0, 5) == []
        assert select_clip_indices(s, 10, 0) == []

    def test_different_seed_different_selection(self):
        a = select_clip_indices(eval_seed("0xa", "hk"), 100, 10)
        b = select_clip_indices(eval_seed("0xb", "hk"), 100, 10)
        assert a != b
