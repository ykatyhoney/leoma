"""The corpus manifest: the duel's exam paper, fixed in advance and hash-anchored.

Today the clip set is derived from a **live bucket listing**, and the window inside
each video is chosen by **running ffmpeg scene detection at duel time**. Both are
consensus holes:

* the try-order permutation is a function of the corpus *size*, so one extra video
  — or one flaky S3 read that makes a clip get skipped — changes the entire clip
  set, and
* scene-cut timestamps vary across ffmpeg builds, so two validators can carve a
  *different window out of the same video* and score against different ground
  truth, silently.

A manifest closes both. It is built **once, offline**, and lists every clip by
``clip_id`` with the window already chosen (``clip_start``) and the decoded ground
truth already hashed (``truth_sha256``). At duel time nobody detects, searches, or
lists anything: the validator selects clip *ids* from a fixed-size list, decodes
exactly the pinned window, and **checks the truth against its hash**. A validator
whose ffmpeg decodes differently now *finds out* instead of quietly diverging.

Pre-filtering offline is sound because the reasons a video is unusable — too short,
no single-shot window long enough — are **seed-independent** properties of the
video. The seed only ever picked *among* viable candidates. So "is this video
usable" is decidable once, at build time, which turns "silently skip it and shift
everyone else's index" into "it was never in the list".

The manifest also carries the ``decode`` block it was hashed under. A manifest built
at 16 fps × 81 frames cannot be used to run a duel at 24 fps: the truth hashes
would all miss. Rather than discover that clip by clip, :func:`check_decode_compat`
rejects it up front.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from leoma.eval.digests import digest_obj, sha256_bytes
from leoma.eval.errors import ConsensusConfigError, CorpusIntegrityError

MANIFEST_VERSION = 1

#: A manifest must be much larger than one duel's clip count, or a miner can simply
#: memorize the whole exam. The duel samples ``n_clips`` from it per challenge.
MIN_CORPUS_MULTIPLE = 20


@dataclass(frozen=True)
class DecodeParams:
    """The decode settings ``truth_sha256`` was computed under.

    Part of the manifest, not of ``chain.toml``, because the hashes are only
    meaningful *relative to these numbers*. Keeping them together makes a
    mismatched pair impossible to construct by accident.
    """

    width: int
    height: int
    fps: int
    num_frames: int

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fps": self.fps, "num_frames": self.num_frames}


@dataclass(frozen=True)
class ClipEntry:
    """One held-out duel item, fully determined before any duel runs."""

    clip_id: str
    video_key: str
    video_sha256: str      # the source .mp4's hash — proves we fetched the right file
    clip_start: float      # seconds; CHOSEN OFFLINE, never re-detected at duel time
    clip_seconds: float
    prompt: str
    truth_sha256: str      # hash of the decoded RGB ground truth under DecodeParams
    motion_energy: float   # mean abs inter-frame delta; static clips are excluded

    def as_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "video_key": self.video_key,
            "video_sha256": self.video_sha256,
            "clip_start": self.clip_start,
            "clip_seconds": self.clip_seconds,
            "prompt": self.prompt,
            "truth_sha256": self.truth_sha256,
            "motion_energy": self.motion_energy,
        }


@dataclass(frozen=True)
class CorpusManifest:
    """A versioned, hash-anchored list of duel clips."""

    corpus_id: str
    decode: DecodeParams
    clips: tuple[ClipEntry, ...]
    manifest_version: int = MANIFEST_VERSION
    #: Digest of the manifest bytes as loaded. Set by :func:`load_pinned_manifest`;
    #: empty for a manifest built in memory (nothing has been serialized to hash yet).
    source_digest: str = field(default="", compare=False)

    def __len__(self) -> int:
        return len(self.clips)

    def as_dict(self) -> dict:
        return {
            "manifest_version": self.manifest_version,
            "corpus_id": self.corpus_id,
            "decode": self.decode.as_dict(),
            "clips": [c.as_dict() for c in self.clips],
        }

    def select(self, indices: Sequence[int]) -> list[ClipEntry]:
        """The clips at ``indices`` — the duel's actual exam."""
        try:
            return [self.clips[i] for i in indices]
        except IndexError as e:
            raise CorpusIntegrityError(
                f"clip index out of range for a corpus of {len(self.clips)} clips: {e}"
            ) from e


def clip_keys_digest(clips: Sequence[ClipEntry]) -> str:
    """Digest identifying *exactly which exam* was set.

    Binds each selected clip's id **and** its expected truth hash, in order. Two
    validators publishing the same ``clip_keys_digest`` have provably scored
    against the same ground truth — which is what makes a distance disagreement
    attributable to generation noise rather than to a different question.
    """
    return digest_obj([[c.clip_id, c.truth_sha256] for c in clips])


def parse_manifest(data: Any, *, source_digest: str = "") -> CorpusManifest:
    """Turn manifest JSON into a :class:`CorpusManifest`, fail-closed.

    Every invariant here exists because violating it would let two validators build
    different clip sets from the same file: duplicate ids (ambiguous selection),
    unsorted clips (index N means a different clip), a missing hash (nothing to
    verify against).
    """
    if not isinstance(data, dict):
        raise CorpusIntegrityError("manifest must be a JSON object")

    version = data.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise CorpusIntegrityError(
            f"unsupported manifest_version {version!r} (this build reads {MANIFEST_VERSION})"
        )

    corpus_id = str(data.get("corpus_id") or "").strip()
    if not corpus_id:
        raise CorpusIntegrityError("manifest has no corpus_id (needed to rotate the corpus)")

    raw_decode = data.get("decode")
    if not isinstance(raw_decode, dict):
        raise CorpusIntegrityError("manifest has no decode block")
    try:
        decode = DecodeParams(
            width=int(raw_decode["width"]),
            height=int(raw_decode["height"]),
            fps=int(raw_decode["fps"]),
            num_frames=int(raw_decode["num_frames"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise CorpusIntegrityError(f"manifest decode block is invalid: {e}") from e

    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise CorpusIntegrityError("manifest has no clips")

    clips: list[ClipEntry] = []
    for i, raw in enumerate(raw_clips):
        if not isinstance(raw, dict):
            raise CorpusIntegrityError(f"clip {i} is not an object")
        try:
            entry = ClipEntry(
                clip_id=str(raw["clip_id"]),
                video_key=str(raw["video_key"]),
                video_sha256=str(raw["video_sha256"]),
                clip_start=float(raw["clip_start"]),
                clip_seconds=float(raw["clip_seconds"]),
                prompt=str(raw.get("prompt", "")),
                truth_sha256=str(raw["truth_sha256"]),
                motion_energy=float(raw.get("motion_energy", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CorpusIntegrityError(f"clip {i} is invalid: {e}") from e
        if not entry.truth_sha256.startswith("sha256:"):
            raise CorpusIntegrityError(f"clip {entry.clip_id} has no usable truth_sha256")
        if entry.clip_start < 0:
            raise CorpusIntegrityError(f"clip {entry.clip_id} has a negative clip_start")
        clips.append(entry)

    ids = [c.clip_id for c in clips]
    if len(set(ids)) != len(ids):
        raise CorpusIntegrityError("manifest contains duplicate clip_ids")
    if ids != sorted(ids):
        raise CorpusIntegrityError(
            "manifest clips must be sorted by clip_id — selection is by index, so an "
            "unsorted manifest means index N is a different clip on different boxes"
        )

    return CorpusManifest(
        corpus_id=corpus_id,
        decode=decode,
        clips=tuple(clips),
        manifest_version=MANIFEST_VERSION,
        source_digest=source_digest,
    )


def load_pinned_manifest(raw: bytes, expected_digest: str) -> CorpusManifest:
    """Verify the manifest **bytes** against the pinned digest, then parse.

    Hash first, parse second. Parsing an unverified manifest to "see what's in it"
    would mean the file has already influenced our behavior (allocations, error
    paths) before we know it is the file the chain pinned.
    """
    import json

    actual = sha256_bytes(raw)
    if actual != expected_digest:
        raise CorpusIntegrityError(
            f"corpus manifest digest mismatch: pinned {expected_digest}, fetched {actual}. "
            "The bucket's manifest is not the one chain.toml pins — refusing to duel "
            "on an unknown corpus."
        )
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise CorpusIntegrityError(f"corpus manifest is not valid JSON: {e}") from e
    return parse_manifest(data, source_digest=actual)


def check_decode_compat(manifest: CorpusManifest, gen) -> None:
    """The manifest's truth hashes are only valid under its own decode params."""
    d = manifest.decode
    mismatched = [
        f"{name}: manifest={mine} chain={theirs}"
        for name, mine, theirs in (
            ("width", d.width, gen.width),
            ("height", d.height, gen.height),
            ("fps", d.fps, gen.fps),
            ("num_frames", d.num_frames, gen.num_frames),
        )
        if mine != theirs
    ]
    if mismatched:
        raise ConsensusConfigError(
            "corpus manifest was built under different decode parameters than "
            f"chain.toml [gen] pins ({'; '.join(mismatched)}). Every truth_sha256 in "
            "the manifest would fail to verify. Rebuild the manifest or fix [gen]."
        )


def check_corpus_size(manifest: CorpusManifest, n_clips: int) -> None:
    """A corpus barely larger than one duel is a memorizable exam, not a test set."""
    needed = MIN_CORPUS_MULTIPLE * n_clips
    if len(manifest) < needed:
        raise ConsensusConfigError(
            f"corpus has {len(manifest)} clips but a {n_clips}-clip duel needs at least "
            f"{needed} ({MIN_CORPUS_MULTIPLE}x) so miners cannot memorize the held-out set"
        )


__all__ = [
    "MANIFEST_VERSION",
    "MIN_CORPUS_MULTIPLE",
    "ClipEntry",
    "CorpusManifest",
    "DecodeParams",
    "check_corpus_size",
    "check_decode_compat",
    "clip_keys_digest",
    "load_pinned_manifest",
    "parse_manifest",
]
