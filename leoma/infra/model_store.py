"""Hippius Hub model references and local materialization.

Miners upload their video-generation model weights (safetensors + config) to
Hippius Hub — a content-addressed OCI registry — and commit a compact
``v4|<repo>|<digest>|<hotkey>`` reveal on-chain. Validators read the reveal,
resolve it to an immutable ``repo@digest`` reference, and download the exact
bytes themselves (no miner-hosted inference endpoint).

``hippius_hub`` is imported lazily inside the functions that need it so that
``ModelRef`` and the reveal parsing/serialisation (the parts exercised by unit
tests and by the on-chain scan) import cleanly without the package installed.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

MODEL_CACHE_DIR = os.environ.get("LEOMA_MODEL_CACHE_DIR", "/tmp/leoma/hippius_models")
HUB_TOKEN_PATH = Path("~/.cache/hippius/hub/token").expanduser()

REVEAL_V4_PREFIX = "v4"
REPO_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._/-]*$")
# Two digest shapes accepted:
#   - "sha256:<64hex>"  Hippius OCI manifest digest (challenger uploads via
#                       hippius_hub, the canonical Hippius reference)
#   - "hf:<40hex>"      HuggingFace commit SHA (a genesis king pinned to a
#                       vanilla HF repo without a Hippius mirror)
DIGEST_RE = re.compile(r"^(sha256:[0-9a-f]{64}|hf:[0-9a-f]{40})$")
SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{47,48}$")

# Files pulled for a model snapshot. Covers diffusers layout (config.json,
# model_index.json, scheduler/*, text_encoder + tokenizer configs) plus the
# safetensors weights. No code files are ever downloaded/executed.
ALLOW_PATTERNS = ["*.safetensors", "*.json", "tokenizer*", "special_tokens*", "*.model", "*.txt"]
CONFIG_ONLY_PATTERNS = ALLOW_PATTERNS[1:]

HUB_TOKEN_ENV_NAMES = (
    "HIPPIUS_HUB_TOKEN",
    "HIPPIUS_TOKEN",
    "LEOMA_HIPPIUS_TOKEN",
)
# S3-style Hippius creds (used for the video corpus + state buckets) are NOT
# valid for Hub/OCI registry auth; detected only to produce a clearer error.
S3_ONLY_ENV_NAMES = (
    "HIPPIUS_VIDEOS_READ_ACCESS_KEY",
    "HIPPIUS_VIDEOS_READ_SECRET_KEY",
    "HIPPIUS_VIDEOS_WRITE_ACCESS_KEY",
    "HIPPIUS_VIDEOS_WRITE_SECRET_KEY",
    "HIPPIUS_ACCESS_KEY",
    "HIPPIUS_SECRET_KEY",
)
HUB_USERNAME_ENV_NAMES = (
    "HIPPIUS_HUB_USERNAME",
    "HIPPIUS_REGISTRY_USERNAME",
    "LEOMA_HIPPIUS_USERNAME",
)
HUB_PASSWORD_ENV_NAMES = (
    "HIPPIUS_HUB_PASSWORD",
    "HIPPIUS_REGISTRY_PASSWORD",
    "LEOMA_HIPPIUS_PASSWORD",
)


class HippiusHubAuthError(RuntimeError):
    """Raised when Hub/registry auth is unavailable or clearly misconfigured."""


def _get_first_env(names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def get_hub_token() -> Optional[str]:
    token = _get_first_env(HUB_TOKEN_ENV_NAMES)
    if token:
        return token
    if HUB_TOKEN_PATH.exists():
        cached = HUB_TOKEN_PATH.read_text().strip()
        if cached:
            return cached
    return None


def get_hub_basic_auth() -> Optional[tuple[str, str]]:
    username = _get_first_env(HUB_USERNAME_ENV_NAMES)
    password = _get_first_env(HUB_PASSWORD_ENV_NAMES)
    if username and password:
        return username, password
    return None


def _s3_auth_detail() -> str:
    present = [name for name in S3_ONLY_ENV_NAMES if (os.environ.get(name) or "").strip()]
    if not present:
        return ""
    return (
        " Found only S3-style Hippius credentials "
        f"({', '.join(present)}), which are not valid for Hub/OCI registry auth."
    )


def _resolve_hub_token(action: Optional[str] = None) -> Optional[str]:
    token = get_hub_token()
    if token:
        return token

    basic_auth = get_hub_basic_auth()
    if basic_auth:
        from hippius_hub import login as hub_login

        username, password = basic_auth
        hub_login(username=username, password=password)
        token = get_hub_token()
        if token:
            return token
        if action:
            raise HippiusHubAuthError(f"{action} could not read cached Hippius Hub auth after login.")
        return None

    if action:
        raise HippiusHubAuthError(
            f"{action} requires Hippius Hub auth via token {HUB_TOKEN_ENV_NAMES} "
            f"or username/password envs {HUB_USERNAME_ENV_NAMES} + {HUB_PASSWORD_ENV_NAMES}."
            f"{_s3_auth_detail()}"
        )
    return None


def _prepare_upload_token(action: str) -> Optional[str]:
    basic_auth = get_hub_basic_auth()
    if basic_auth:
        from hippius_hub import login as hub_login

        username, password = basic_auth
        hub_login(username=username, password=password)
        return None

    token = get_hub_token()
    if token:
        return token

    raise HippiusHubAuthError(
        f"{action} requires Hippius Hub auth via token {HUB_TOKEN_ENV_NAMES} "
        f"or username/password envs {HUB_USERNAME_ENV_NAMES} + {HUB_PASSWORD_ENV_NAMES}."
        f"{_s3_auth_detail()}"
    )


@dataclass(frozen=True)
class ModelRef:
    """Immutable Hippius Hub model reference (``repo@digest``)."""

    repo: str
    digest: str

    def __post_init__(self) -> None:
        repo = (self.repo or "").strip()
        digest = (self.digest or "").strip()
        if not REPO_RE.match(repo):
            raise ValueError(f"invalid Hippius repo id: {self.repo!r}")
        if not DIGEST_RE.match(digest):
            raise ValueError(f"invalid Hippius OCI digest: {self.digest!r}")
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "digest", digest)

    @property
    def immutable_ref(self) -> str:
        return f"{self.repo}@{self.digest}"


def _normalise_digest(value: str) -> str:
    digest = (value or "").strip()
    if not DIGEST_RE.match(digest):
        raise ValueError(f"invalid OCI digest: {value!r}")
    return digest


# v4 payload: `v4|<challenger_repo>|<challenger_digest>|<author_hotkey>`.
# challenger_digest carries its format prefix (sha256:/hf:) so the validator can
# dispatch to the right download path. author_hotkey is the 48-char ss58 of the
# submitter, kept for cross-check against the chain-side commitment signer key.
# Longest case: `v4|<repo-50>|sha256:<64>|<ss58-48>` ≈ 160 chars.

def build_reveal_v4(challenger_ref: ModelRef, author_hotkey: str) -> str:
    hk = (author_hotkey or "").strip()
    if not SS58_RE.match(hk):
        raise ValueError(f"invalid author hotkey ss58: {author_hotkey!r}")
    return f"{REVEAL_V4_PREFIX}|{challenger_ref.repo}|{challenger_ref.digest}|{hk}"


def parse_reveal_v4(payload: str) -> tuple[ModelRef, str]:
    """Returns (ModelRef(challenger_repo, challenger_digest), author_hotkey).

    Raises ValueError for any non-v4 / malformed payload (e.g. a legacy JSON
    commitment), so the scanner can treat those as skippable.
    """
    parts = (payload or "").strip().split("|")
    if len(parts) != 4 or parts[0] != REVEAL_V4_PREFIX:
        raise ValueError("expected v4|repo|challenger_digest|author_hotkey reveal")
    hk = parts[3].strip()
    if not SS58_RE.match(hk):
        raise ValueError(f"invalid v4 author hotkey: {parts[3]!r}")
    return ModelRef(parts[1], _normalise_digest(parts[2])), hk


def _cache_snapshot_path(ref: ModelRef) -> Path:
    repo_key = ref.repo.replace("/", "--")
    digest_key = ref.digest.replace(":", "-")
    return Path(MODEL_CACHE_DIR) / repo_key / "snapshots" / digest_key


def local_snapshot_path(ref: ModelRef) -> str:
    path = _cache_snapshot_path(ref)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return str(path)


def cache_path(ref: ModelRef, *, config_only: bool = False) -> str:
    """Where this ref's snapshot lives — whether or not it has been downloaded yet.

    Unlike :func:`local_snapshot_path` this does not require the directory to exist,
    because the caller that needs it most is the download-progress watcher: it polls
    the directory *while it is being filled*, which is precisely the window in which
    "does it exist yet" is the wrong question.
    """
    base = _cache_snapshot_path(ref)
    return str(base.with_name(base.name + "_cfg") if config_only else base)


def _call_snapshot_download(ref: ModelRef, local_dir: Optional[str], max_workers: Optional[int],
                            *, allow_patterns=ALLOW_PATTERNS) -> str:
    if ref.digest.startswith("hf:"):
        from huggingface_hub import snapshot_download as hf_snapshot_download

        return str(hf_snapshot_download(
            repo_id=ref.repo, revision=ref.digest[3:], local_dir=local_dir,
            allow_patterns=allow_patterns, max_workers=max_workers or 8,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"),
        ))
    from hippius_hub import snapshot_download

    return str(snapshot_download(
        repo_id=ref.repo, revision=ref.digest, local_dir=local_dir,
        allow_patterns=allow_patterns, max_workers=max_workers or 8,
        token=_resolve_hub_token(f"Downloading {ref.immutable_ref}"),
    ))


def materialize_model(ref: ModelRef, local_dir: Optional[str] = None, max_workers: Optional[int] = None,
                      *, config_only: bool = False) -> str:
    """Download or reuse an immutable Hippius Hub snapshot; returns the local dir.

    ``config_only=True`` skips the large ``*.safetensors`` files — use for the
    validator's per-challenger arch/lock validation which only needs config.json.
    Cache dir is suffixed with ``_cfg`` so a config-only fetch doesn't pollute a
    later full-fetch's cache state.
    """
    if config_only:
        base = Path(local_dir) if local_dir else _cache_snapshot_path(ref)
        target = base.with_name(base.name + "_cfg")
    else:
        target = Path(local_dir) if local_dir else _cache_snapshot_path(ref)
    if target.exists() and (target / "config.json").exists():
        if config_only or any(target.glob("*.safetensors")):
            return str(target)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    patterns = CONFIG_ONLY_PATTERNS if config_only else ALLOW_PATTERNS
    return _call_snapshot_download(ref, str(target), max_workers, allow_patterns=patterns)


def list_snapshot_files(snapshot: str | os.PathLike[str]) -> list[str]:
    root = Path(snapshot)
    return sorted(
        str(p.relative_to(root)).replace(os.sep, "/")
        for p in root.rglob("*")
        if p.is_file()
    )


def snapshot_size(snapshot: str | os.PathLike[str], files: Optional[Iterable[str]] = None) -> int:
    root = Path(snapshot)
    paths = (root / f for f in files) if files is not None else (p for p in root.rglob("*") if p.is_file())
    total = 0
    for path in paths:
        try:
            total += Path(path).stat().st_size
        except FileNotFoundError:
            continue
    return total


def sha256_safetensors(path: str | os.PathLike[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in sorted(Path(path).glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def upload_model_folder(
    folder_path: str | os.PathLike[str],
    repo: str,
    revision: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> ModelRef:
    """Upload a model folder to Hippius Hub and return its immutable digest."""
    from hippius_hub import upload_folder

    token = _prepare_upload_token(f"Uploading {folder_path} to {repo}")
    result = upload_folder(
        repo_id=repo, folder_path=str(folder_path), revision=revision,
        commit_message=commit_message, allow_patterns=ALLOW_PATTERNS, token=token,
    )
    return ModelRef(repo, _normalise_digest(str(result.oid)))
