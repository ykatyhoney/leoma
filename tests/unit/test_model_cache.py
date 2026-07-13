"""Disk and VRAM: the eval box degrades and cannot recover.

Two bugs here are worth the whole file:

* **`materialize_model`'s cache check never hit on a diffusers layout**, so every
  duel `rmtree`d the cache and re-downloaded the king's ~30-70 GB of weights. The
  king is the *same model* for every challenger in the queue.
* **`sha256_safetensors` returned the sha256 of the empty string** on a diffusers
  layout — a constant, for every model on earth. As the copy-of-king detector it
  would have flagged every model as identical to every other one.

Both come from the same root cause: code written for a **transformers** layout (root
`config.json`, weights at the top level) running against a **diffusers** one
(`model_index.json`, weights in `transformer/`, `vae/`, `text_encoder/` subfolders).
"""

import hashlib
import json

import pytest

from leoma.infra import model_store as ms
from leoma.infra.model_store import (
    COMPLETION_MARKER,
    ModelRef,
    cache_path,
    evict_snapshots,
    is_complete,
    materialize_model,
    sha256_safetensors,
)

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def _diffusers_snapshot(root, *, weights=b"WEIGHTS"):
    """A real diffusers layout: NO root config.json, weights in subfolders."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "WanImageToVideoPipeline"}))
    for component in ("transformer", "vae", "text_encoder"):
        sub = root / component
        sub.mkdir(exist_ok=True)
        (sub / "config.json").write_text("{}")
        (sub / "diffusion_pytorch_model.safetensors").write_bytes(weights + component.encode())
    return root


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "MODEL_CACHE_DIR", str(tmp_path))
    return tmp_path


class TestTheKingIsNotRedownloadedEveryDuel:
    def test_a_complete_diffusers_snapshot_is_reused(self, cache_dir, monkeypatch):
        """THE bug. The old check was `config.json exists AND glob('*.safetensors')`.
        A diffusers snapshot has neither at the top level, so it NEVER hit: every duel
        deleted the cache and pulled tens of gigabytes again."""
        ref = ModelRef("u/leoma-king", DIGEST_A)
        target = _diffusers_snapshot(cache_dir / "u--leoma-king" / "snapshots" / f"sha256-{'a' * 64}")
        (target / COMPLETION_MARKER).write_text("{}")

        downloads = []
        monkeypatch.setattr(
            ms, "_call_snapshot_download",
            lambda *a, **k: downloads.append(1) or str(target),
        )

        assert materialize_model(ref) == str(target)
        assert downloads == [], "the king was re-downloaded from a perfectly good cache"

    def test_an_interrupted_download_is_NOT_mistaken_for_a_cache(self, cache_dir, monkeypatch):
        """The deeper bug the marker fixes: a half-finished snapshot has files in it,
        and 'has files' was the entire cache test. A partial model that *looks* usable
        is worse than none at all."""
        ref = ModelRef("u/leoma-king", DIGEST_A)
        target = _diffusers_snapshot(cache_dir / "u--leoma-king" / "snapshots" / f"sha256-{'a' * 64}")
        # No completion marker: the download died halfway.

        downloads = []

        def fake_download(ref_, local_dir, *a, **k):
            downloads.append(local_dir)
            return str(_diffusers_snapshot(ms.Path(local_dir)))

        monkeypatch.setattr(ms, "_call_snapshot_download", fake_download)

        materialize_model(ref)
        assert len(downloads) == 1, "an interrupted download was reused as if it were complete"
        assert is_complete(target)

    def test_a_finished_download_is_marked_complete(self, cache_dir, monkeypatch):
        ref = ModelRef("u/leoma-c", DIGEST_B)

        def fake_download(ref_, local_dir, *a, **k):
            return str(_diffusers_snapshot(ms.Path(local_dir)))

        monkeypatch.setattr(ms, "_call_snapshot_download", fake_download)
        path = materialize_model(ref)

        assert is_complete(path)
        marker = json.loads((ms.Path(path) / COMPLETION_MARKER).read_text())
        assert marker["ref"] == ref.immutable_ref
        assert marker["bytes"] > 0

    def test_the_marker_does_not_care_about_the_layout(self, cache_dir):
        """A check that inspects file names has to know whether it is looking at a
        transformers repo or a diffusers one — and gets it wrong the moment the
        pinned architecture changes."""
        transformers_style = cache_dir / "t"
        transformers_style.mkdir()
        (transformers_style / "config.json").write_text("{}")
        (transformers_style / "model.safetensors").write_bytes(b"w")
        assert not is_complete(transformers_style)

        (transformers_style / COMPLETION_MARKER).write_text("{}")
        assert is_complete(transformers_style)


class TestEviction:
    def _snapshot(self, cache_dir, repo, digest, mtime):
        import os

        path = cache_dir / repo.replace("/", "--") / "snapshots" / digest.replace(":", "-")
        _diffusers_snapshot(path)
        marker = path / COMPLETION_MARKER
        marker.write_text("{}")
        os.utime(marker, (mtime, mtime))
        return path

    def test_the_oldest_snapshots_go_first(self, cache_dir, monkeypatch):
        monkeypatch.setattr(ms, "MAX_CACHED_SNAPSHOTS", 2)
        monkeypatch.setattr(ms, "MIN_FREE_BYTES", 0)   # only the count budget applies

        old = self._snapshot(cache_dir, "u/old", DIGEST_A, mtime=1000)
        mid = self._snapshot(cache_dir, "u/mid", DIGEST_B, mtime=2000)
        new = self._snapshot(cache_dir, "u/new", "sha256:" + "c" * 64, mtime=3000)
        newest = self._snapshot(cache_dir, "u/newest", "sha256:" + "d" * 64, mtime=4000)

        evicted = evict_snapshots(root=str(cache_dir))

        # 4 snapshots, budget of 2: the two coldest go, in age order.
        assert [str(old), str(mid)] == evicted
        assert not old.exists() and not mid.exists()
        assert new.exists() and newest.exists()

    def test_the_king_is_never_evicted_however_cold_it_looks(self, cache_dir, monkeypatch):
        """A king that has reigned for weeks has the oldest mtime in the cache — and
        is the single snapshot that must survive."""
        monkeypatch.setattr(ms, "MAX_CACHED_SNAPSHOTS", 1)
        monkeypatch.setattr(ms, "MIN_FREE_BYTES", 0)

        king = self._snapshot(cache_dir, "u/king", DIGEST_A, mtime=1)      # ancient
        chall = self._snapshot(cache_dir, "u/chall", DIGEST_B, mtime=9999)  # brand new

        evicted = evict_snapshots(keep=[str(king), str(chall)], root=str(cache_dir))

        assert evicted == []
        assert king.exists() and chall.exists()

    def test_an_incomplete_snapshot_is_not_treated_as_a_cache_entry(self, cache_dir):
        junk = cache_dir / "u--junk" / "snapshots" / "sha256-deadbeef"
        _diffusers_snapshot(junk)          # no marker: debris
        assert ms._cached_snapshots(cache_dir) == []


class TestSha256Safetensors:
    def test_it_finds_weights_in_diffusers_subfolders(self, tmp_path):
        """It used to glob('*.safetensors') NON-recursively, match zero files, and
        return the sha256 of the empty string — the same constant for every model."""
        snapshot = _diffusers_snapshot(tmp_path / "m")
        digest = sha256_safetensors(snapshot)

        assert digest != hashlib.sha256(b"").hexdigest(), "hashed nothing at all"
        assert len(digest) == 64

    def test_different_weights_hash_differently(self, tmp_path):
        a = sha256_safetensors(_diffusers_snapshot(tmp_path / "a", weights=b"AAA"))
        b = sha256_safetensors(_diffusers_snapshot(tmp_path / "b", weights=b"BBB"))
        assert a != b

    def test_identical_weights_hash_identically(self, tmp_path):
        """This is the property the copy-of-king detector actually rests on."""
        a = sha256_safetensors(_diffusers_snapshot(tmp_path / "a", weights=b"SAME"))
        b = sha256_safetensors(_diffusers_snapshot(tmp_path / "b", weights=b"SAME"))
        assert a == b

    def test_the_same_bytes_under_a_different_component_name_is_a_different_model(self, tmp_path):
        snapshot = _diffusers_snapshot(tmp_path / "m")
        before = sha256_safetensors(snapshot)
        (snapshot / "transformer" / "diffusion_pytorch_model.safetensors").rename(
            snapshot / "transformer" / "renamed.safetensors"
        )
        assert sha256_safetensors(snapshot) != before

    def test_a_snapshot_with_no_weights_RAISES_rather_than_hashing_nothing(self, tmp_path):
        """Returning a valid-looking digest for a snapshot with no weights is exactly
        how the original bug stayed invisible."""
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "model_index.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="no .safetensors"):
            sha256_safetensors(empty)


class TestCachePath:
    def test_it_does_not_require_the_snapshot_to_exist(self, cache_dir):
        """The download-progress watcher polls the directory WHILE it is being filled,
        which is precisely the window in which 'does it exist yet' is the wrong
        question."""
        ref = ModelRef("u/leoma-x", DIGEST_A)
        path = cache_path(ref)
        assert not ms.Path(path).exists()
        assert DIGEST_A.replace(":", "-") in path

    def test_config_only_lives_somewhere_else_entirely(self, cache_dir):
        ref = ModelRef("u/leoma-x", DIGEST_A)
        assert cache_path(ref, config_only=True) != cache_path(ref)
        assert cache_path(ref, config_only=True).endswith("_cfg")
