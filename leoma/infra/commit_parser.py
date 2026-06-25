"""
Parse and validate miner chain commitments.

Hugging Face model_name is "username/repo_name" (e.g. your_username/leoma-hh-5GRWVAEF5Z...).
We validate only the repo name (part after the last "/"), excluding the username.

Repo name rules (case-insensitive):
- Must start with "leoma"
- Must end with the miner's hotkey (when hotkey is provided)
"""
import json
from typing import Any, Dict, Optional, Tuple

_COMMIT_REQUIRED_FIELDS = (
    ("model_name", "missing_model_name"),
    ("model_revision", "missing_model_revision"),
    ("chute_id", "missing_chute_id"),
)

MODEL_NAME_PREFIX = "leoma"
MAX_COMMITS_PER_HOTKEY = 2


def _repo_name_from_model_name(model_name: str) -> str:
    """Return the repo name (part after the last '/'); excludes Hugging Face username."""
    s = (model_name or "").strip()
    if "/" in s:
        return s.rsplit("/", 1)[-1].strip()
    return s


def _parse_commit_json(commit_value: str) -> Dict[str, Any] | None:
    try:
        parsed = json.loads(commit_value)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_commit(commit_value: str) -> Dict[str, Any]:
    """Parse a miner's chain commitment. Returns dict with model_name, model_revision, chute_id or empty."""
    if not commit_value:
        return {}
    parsed = _parse_commit_json(commit_value)
    return parsed or {}


def validate_commit_fields(
    commit: Dict[str, Any],
    hotkey: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate that commit has all required fields and model_name rules.

    Hugging Face model_name is username/repo_name. We validate only the repo name
    (part after the last "/"): it must start with "leoma" (case-insensitive) and,
    when hotkey is provided, must end with that hotkey (case-insensitive).

    Returns (is_valid, error_reason).
    """
    for field, error_reason in _COMMIT_REQUIRED_FIELDS:
        if not commit.get(field, ""):
            return False, error_reason

    model_name = (commit.get("model_name") or "").strip()
    repo_name = _repo_name_from_model_name(model_name)
    repo_name_lower = repo_name.lower()

    if not repo_name_lower.startswith(MODEL_NAME_PREFIX.lower()):
        return False, "model_name_must_start_with_leoma"

    if hotkey and hotkey.strip():
        hotkey_lower = hotkey.strip().lower()
        if not repo_name_lower.endswith(hotkey_lower):
            return False, "model_name_must_end_with_hotkey"

    return True, None


def validate_commit_count(
    commit_history_len: int,
    max_commits: int = MAX_COMMITS_PER_HOTKEY,
) -> Tuple[bool, Optional[str]]:
    """Validate that a hotkey has not exceeded the maximum allowed commits.

    Returns (is_valid, error_reason); is_valid is False when commit_history_len > max_commits.
    """
    if commit_history_len > max_commits:
        return False, f"max_commits_exceeded_{max_commits}"
    return True, None
