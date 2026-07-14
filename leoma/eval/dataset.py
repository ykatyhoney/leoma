"""Build a duel's clips from the pinned corpus manifest — deterministically, or not at all.

What this module used to do, and why each step was a consensus hole:

1. ``list_objects`` on a live bucket, then permute ``range(len(keys))``. The try
   order was a function of the corpus *size*, so uploading one video reshuffled
   every validator's clip set — and only for the validators who had seen the
   upload.
2. Run ffmpeg **scene detection at duel time** to pick the window inside each
   video. Scene-cut timestamps vary by ffmpeg build, so two validators could carve
   a *different five seconds out of the same video* and never know.
3. ``except Exception: return None`` on any clip failure, then move to the next
   candidate. One flaky S3 read shifted the whole clip set — and a duel could
   quietly run on 3 clips instead of 32 and be recorded as a normal result.

Now: the manifest fixes the clip list and the window *offline*; the seed picks
**indices into a fixed-size list**; each clip's ground truth is **verified against
its hash**; and any failure **aborts the duel**. A clip is never skipped and never
substituted — if we cannot build the exact exam the chain specifies, we do not
hand out a grade.

Fault attribution is the other half. A download that fails is
:class:`~leoma.eval.errors.TransientDuelError` — retry, and never blame the miner.
A truth that decodes to the *wrong hash* is
:class:`~leoma.eval.errors.CorpusIntegrityError` — this validator's ffmpeg or bucket
is wrong, and it must not emit a verdict at all.
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable, Optional, Sequence

from leoma.app.validator.seeds import select_clip_indices
from leoma.eval.digests import digest_frames, digest_file
from leoma.eval.errors import CorpusIntegrityError, TransientDuelError
from leoma.eval.manifest import (
    ClipEntry,
    CorpusManifest,
    check_corpus_size,
    check_decode_compat,
    clip_keys_digest,
    load_pinned_manifest,
)
from leoma.eval.video_runner import Clip, GenParams

#: Called after each clip is built — the eval server's forward-progress heartbeat
#: for a phase that is otherwise silent for minutes.
ProgressFn = Callable[[int, int, ClipEntry], None]


def fetch_manifest(client, corpus) -> CorpusManifest:
    """Fetch the manifest from the corpus bucket and verify it against the pin."""
    try:
        response = client.get_object(corpus.bucket, corpus.manifest_key)
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()
    except Exception as e:  # noqa: BLE001 — a fetch failure is the network's fault
        raise TransientDuelError(
            f"could not fetch the corpus manifest {corpus.bucket}/{corpus.manifest_key}: {e}"
        ) from e
    return load_pinned_manifest(raw, corpus.manifest_digest)


def _fetch_and_decode(client, bucket: str, entry: ClipEntry, gen: GenParams):
    """Download one source video and decode its pinned window, verifying both hashes."""
    from leoma.infra.video_utils import decode_frames_rgb

    tmpdir = tempfile.mkdtemp(prefix="leoma-duel-")
    src = os.path.join(tmpdir, "src.mp4")
    try:
        try:
            client.fget_object(bucket, entry.video_key, src)
        except Exception as e:  # noqa: BLE001
            raise TransientDuelError(
                f"could not fetch source video {entry.video_key} for clip {entry.clip_id}: {e}"
            ) from e

        actual_video = digest_file(src)
        if actual_video != entry.video_sha256:
            raise CorpusIntegrityError(
                f"clip {entry.clip_id}: source video {entry.video_key} does not match the "
                f"manifest (pinned {entry.video_sha256}, got {actual_video}). The bucket's "
                "contents have drifted from the pinned corpus."
            )

        try:
            truth = decode_frames_rgb(
                src,
                start_seconds=entry.clip_start,
                duration_seconds=entry.clip_seconds,
                fps=gen.fps,
                num_frames=gen.num_frames,
                width=gen.width,
                height=gen.height,
            )
        except Exception as e:  # noqa: BLE001 — ffmpeg failed on a file that hashed correctly
            raise CorpusIntegrityError(
                f"clip {entry.clip_id}: ffmpeg could not decode the pinned window "
                f"({entry.clip_start:.3f}s +{entry.clip_seconds:.3f}s): {e}"
            ) from e

        actual_truth = digest_frames(truth)
        if actual_truth != entry.truth_sha256:
            raise CorpusIntegrityError(
                f"clip {entry.clip_id}: decoded ground truth does not match the manifest "
                f"(pinned {entry.truth_sha256}, got {actual_truth}). This box decodes video "
                "differently from the one that built the corpus — its distances would be "
                "measured against different ground truth. Refusing to duel.\n"
                "Run `leoma corpus verify` to check this box's ffmpeg against the manifest."
            )
        return truth
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def build_duel_clips(
    manifest: CorpusManifest,
    *,
    client,
    bucket: str,
    master_seed: int,
    n_clips: int,
    gen: GenParams,
    prompt_mode: str,
    fixed_prompt: str,
    on_progress: Optional[ProgressFn] = None,
) -> tuple[list[Clip], list[ClipEntry]]:
    """The exam: ``n_clips`` clips selected by seed from the pinned manifest.

    Returns the runnable clips **and** the manifest entries they came from (the
    caller needs the entries to publish the audit block). Raises rather than
    returning a short list — a duel on fewer clips than the chain specifies is not
    a valid duel, and silently accepting one is how the old code turned a flaky
    bucket into a crown.
    """
    check_decode_compat(manifest, gen)
    check_corpus_size(manifest, n_clips)

    indices = select_clip_indices(master_seed, len(manifest), n_clips)
    if len(indices) != n_clips:
        raise CorpusIntegrityError(
            f"selected {len(indices)} clips from a corpus of {len(manifest)} but the duel "
            f"requires exactly {n_clips}"
        )
    entries = manifest.select(indices)

    clips: list[Clip] = []
    for position, (index, entry) in enumerate(zip(indices, entries)):
        truth = _fetch_and_decode(client, bucket, entry, gen)
        prompt = fixed_prompt if prompt_mode == "fixed" else entry.prompt
        clips.append(
            Clip(
                # The manifest index — NOT the position in this duel. The seed is
                # derived from it, so it must identify the clip in the corpus, or
                # the same clip would generate from different noise depending on
                # where it happened to land in the selection.
                clip_index=index,
                clip_id=entry.clip_id,
                first_frame=truth[0],
                prompt=prompt,
                truth_frames=truth,
                params=gen,
            )
        )
        if on_progress:
            on_progress(position + 1, n_clips, entry)

    return clips, list(entries)


def corpus_audit(manifest: CorpusManifest, entries: Sequence[ClipEntry]) -> dict:
    """The corpus half of the verdict's audit block — enough to replay the exam."""
    return {
        "corpus_id": manifest.corpus_id,
        "manifest_digest": manifest.source_digest,
        "corpus_size": len(manifest),
        "clip_ids": [e.clip_id for e in entries],
        "clip_keys_digest": clip_keys_digest(entries),
    }


__all__ = ["ProgressFn", "build_duel_clips", "corpus_audit", "fetch_manifest"]
