"""
Validate miner model-repo names.

A miner's Hippius Hub repo is "username/repo_name" (e.g. your_username/leoma-hh-5GRWVAEF5Z...).
We validate only the repo name (part after the last "/"), excluding the username:

- Must start with "leoma" (case-insensitive)
- Must end with the miner's hotkey (when a hotkey is provided)

Shared by the miner (name its upload correctly) and the validator's reveal scan
(accept a challenger submission).
"""
from typing import Optional, Tuple

MODEL_NAME_PREFIX = "leoma"


def _repo_name_from_model_name(model_name: str) -> str:
    """Return the repo name (part after the last '/'); excludes the username."""
    s = (model_name or "").strip()
    if "/" in s:
        return s.rsplit("/", 1)[-1].strip()
    return s


def validate_repo_name(
    model_name: str,
    hotkey: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate a model repo id against the naming rules.

    Returns (is_valid, error_reason). The repo name must start with "leoma" and,
    when a hotkey is provided, end with that hotkey (both case-insensitive).
    """
    repo_name_lower = _repo_name_from_model_name(model_name).lower()

    if not repo_name_lower.startswith(MODEL_NAME_PREFIX.lower()):
        return False, "model_name_must_start_with_leoma"

    if hotkey and hotkey.strip():
        if not repo_name_lower.endswith(hotkey.strip().lower()):
            return False, "model_name_must_end_with_hotkey"

    return True, None
