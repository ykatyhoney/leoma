"""
Unit tests for miner repo-name validation.

The repo name (after the last "/") must start with "leoma" and end with the
miner's hotkey — shared by the miner and the validator's reveal scan.
"""

from leoma.infra.commit_parser import validate_repo_name

HOTKEY = "5C7LM2i42XgL2oB4x3rcmB7KDiof4B92KZzUpg5miZ6DogjU"


class TestValidateRepoName:
    def test_valid_with_hotkey_suffix(self):
        ok, reason = validate_repo_name(f"user/leoma-mymodel-{HOTKEY}", hotkey=HOTKEY)
        assert ok is True
        assert reason is None

    def test_valid_without_hotkey_check(self):
        ok, _ = validate_repo_name("user/leoma-anything")
        assert ok is True

    def test_case_insensitive_prefix(self):
        ok, _ = validate_repo_name("user/LEOMA-Model", hotkey=None)
        assert ok is True

    def test_rejects_non_leoma_prefix(self):
        ok, reason = validate_repo_name(f"user/notleoma-{HOTKEY}", hotkey=HOTKEY)
        assert ok is False
        assert reason == "model_name_must_start_with_leoma"

    def test_rejects_wrong_hotkey_suffix(self):
        ok, reason = validate_repo_name("user/leoma-model-otherhotkey", hotkey=HOTKEY)
        assert ok is False
        assert reason == "model_name_must_end_with_hotkey"

    def test_username_is_excluded_from_prefix_check(self):
        # The username ("leomacorp") does not count; the repo name must start with leoma.
        ok, reason = validate_repo_name("leomacorp/model-x", hotkey=None)
        assert ok is False
        assert reason == "model_name_must_start_with_leoma"

    def test_bare_repo_without_username(self):
        ok, _ = validate_repo_name(f"leoma-model-{HOTKEY}", hotkey=HOTKEY)
        assert ok is True

    def test_empty_is_invalid(self):
        ok, reason = validate_repo_name("", hotkey=None)
        assert ok is False
        assert reason == "model_name_must_start_with_leoma"
