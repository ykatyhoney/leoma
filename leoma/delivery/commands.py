"""
CLI interface for Leoma.

Provides commands for running the validator and individual services.
"""

import asyncio

import click

from leoma import __version__


def _run_async(coroutine) -> None:
    """Run an async coroutine from CLI commands."""
    asyncio.run(coroutine)


@click.group()
@click.version_option(version=__version__, prog_name="leoma")
def cli():
    """Leoma - Image-to-Video Generation Subnet

    A king-of-the-hill TI2V subnet: miners upload model weights to Hippius Hub,
    validators download them and duel challenger vs king on held-out clips.
    """


@cli.command()
def serve():
    """Start the king-of-the-hill validator.

    Scans on-chain miner reveals, duels each new challenger against the reigning
    king on the GPU eval server (deterministic, block-hash-seeded), crowns
    winners, and sets equal weights across the king chain (else burns UID 0).
    Requires R2_OWN_BUCKET and a reachable EVAL_SERVER_URL.
    """
    from leoma.app.validator.main import main
    _run_async(main())


@cli.group()
def servers():
    """Start individual services.
    
    Use these commands to run specific components separately.
    """


@servers.command("validator")
def start_validator():
    """Start the king-of-the-hill validator (duel + weight-setter).

    Scans on-chain reveals, duels new challengers against the king on the eval
    server, crowns winners, and sets equal weights across the king chain (else
    burns to UID 0). Same as `leoma serve`.
    """
    from leoma.app.validator.main import main
    _run_async(main())


@servers.command("eval-server")
def start_eval_server():
    """Start the GPU eval server (video-generation duel).

    FastAPI service the validator dispatches duels to (POST /eval, SSE stream).
    Listens on EVAL_SERVER_HOST:EVAL_SERVER_PORT (default 0.0.0.0:9000).
    """
    from leoma.eval_server import main as eval_main
    eval_main()


@cli.group()
def corpus():
    """Video corpus management commands.
    
    Commands for ingesting, listing, and managing videos in the
    Hippius "videos" bucket used for I2V evaluation.
    """

@corpus.command("expand")
@click.option("--count", "-n", default=10, help="Number of videos to add")
@click.option("--concurrent", "-c", default=2, help="Max concurrent downloads")
@click.option("--query", "-q", multiple=True, help="Custom search queries (can specify multiple)")
def corpus_expand(count, concurrent, query):
    """Expand corpus with random YouTube videos.
    
    Searches YouTube using diverse queries and ingests suitable videos.
    Uses default queries for nature, cooking, travel, dance, sports, etc.
    
    Examples:
    
        leoma corpus expand --count 20
        
        leoma corpus expand -n 10 -q "nature documentary" -q "cooking tutorial"
    """
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.infra.storage_backend import create_source_write_client
    from leoma.infra.corpus import expand_corpus_random
    
    queries = list(query) if query else None
    
    async def run():
        log_header("Expanding Video Corpus")
        log(f"Target: {count} videos, {concurrent} concurrent downloads", "info")
        if queries:
            log(f"Custom queries: {queries}", "info")
        
        minio_client = create_source_write_client()
        results = await expand_corpus_random(minio_client, count, queries, concurrent)
        
        log_header("Expansion Results")
        log(f"Success: {results['success']}/{results['total']}", "success")
        
        if results["uploaded"]:
            log(f"Uploaded {len(results['uploaded'])} videos", "info")
        
        if results["errors"]:
            log(f"Failed: {results['failed']}", "warn")
            for err in results["errors"][:5]:
                url = err.get("url", "unknown")[:50]
                log(f"  {url}: {err['error']}", "warn")
            if len(results["errors"]) > 5:
                log(f"  ... and {len(results['errors']) - 5} more errors", "warn")
    
    _run_async(run())


@corpus.command("build-manifest")
@click.option("--corpus-id", required=True, help="Version label for this corpus, e.g. leoma-corpus-v1")
@click.option("--out", "-o", default="manifest.json", help="Where to write the manifest")
@click.option("--limit", type=int, default=None, help="Only consider the first N source videos (testing)")
def corpus_build_manifest(corpus_id, out, limit):
    """Build the pinned corpus manifest from the source bucket.

    Decides each clip's window ONCE, decodes its ground truth, and records the
    hashes. This is what makes a duel reproducible: at duel time nobody detects
    scenes, lists buckets, or skips a clip — they just decode the pinned window and
    check it against the hash recorded here.

    Prints the manifest digest to paste into chain.toml [corpus].manifest_digest.
    """
    from leoma.bootstrap import (
        MAX_VIDEO_SIZE,
        MIN_VIDEO_SIZE,
        SOURCE_BUCKET,
        emit_log as log,
        emit_header as log_header,
    )
    from leoma.eval.manifest import DecodeParams
    from leoma.infra.chain_config import SPEC
    from leoma.infra.corpus_manifest import build_manifest, write_manifest
    from leoma.infra.storage_backend import create_source_read_client

    log_header("Building Corpus Manifest")

    # The decode params come from the PINNED [gen] block, never from flags: the
    # truth hashes are only meaningful relative to them, and a manifest built under
    # different numbers would fail every hash check at duel time.
    decode = DecodeParams(
        width=SPEC.gen.width,
        height=SPEC.gen.height,
        fps=SPEC.gen.fps,
        num_frames=SPEC.gen.num_frames,
    )
    log(f"Decode: {decode.width}x{decode.height} @ {decode.fps}fps x {decode.num_frames} frames", "info")

    client = create_source_read_client()
    keys = [
        obj.object_name
        for obj in client.list_objects(SOURCE_BUCKET, recursive=True)
        if obj.object_name.endswith(".mp4") and MIN_VIDEO_SIZE < obj.size < MAX_VIDEO_SIZE
    ]
    keys.sort()
    if limit:
        keys = keys[:limit]
    log(f"Considering {len(keys)} source videos from {SOURCE_BUCKET}", "info")

    manifest = build_manifest(
        client, SOURCE_BUCKET, corpus_id=corpus_id, decode=decode, keys=keys,
        log=lambda m: log(m, "info"),
    )
    digest = write_manifest(manifest, out)

    log_header("Manifest Built")
    log(f"{len(manifest)} clips -> {out}", "success")
    log(f"digest: {digest}", "success")
    log("Paste into chain.toml [corpus].manifest_digest, then `leoma corpus publish-manifest`", "info")


@corpus.command("publish-manifest")
@click.argument("path", type=click.Path(exists=True))
def corpus_publish_manifest(path):
    """Upload a built manifest to the corpus bucket at the pinned key."""
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.infra.chain_config import SPEC
    from leoma.infra.corpus_manifest import publish_manifest, read_manifest
    from leoma.infra.storage_backend import create_source_write_client

    log_header("Publishing Corpus Manifest")
    manifest = read_manifest(path)
    client = create_source_write_client()
    digest = publish_manifest(client, SPEC.corpus.bucket, SPEC.corpus.manifest_key, manifest)

    log(f"Published {len(manifest)} clips to {SPEC.corpus.bucket}/{SPEC.corpus.manifest_key}", "success")
    log(f"digest: {digest}", "success")
    if digest != SPEC.corpus.manifest_digest:
        log("chain.toml [corpus].manifest_digest does NOT match this manifest — "
            "validators will refuse to duel until it is pinned to the digest above", "warn")


@corpus.command("verify")
@click.option("--sample", type=int, default=4, help="How many clips to re-decode (0 = all)")
def corpus_verify(sample):
    """Prove THIS box decodes the pinned corpus byte-identically.

    Run this on every new eval box before it duels. A box whose ffmpeg produces even
    slightly different pixels measures every distance against different ground truth
    — silently, confidently, and wrongly. A minute here, or a broken consensus in
    production.
    """
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.eval.dataset import fetch_manifest
    from leoma.infra.chain_config import SPEC
    from leoma.infra.corpus_manifest import verify_manifest
    from leoma.infra.storage_backend import create_source_read_client

    log_header("Verifying Corpus")
    SPEC.require_duel_ready()

    client = create_source_read_client()
    manifest = fetch_manifest(client, SPEC.corpus)
    log(f"Manifest {manifest.corpus_id}: {len(manifest)} clips, digest matches chain.toml", "success")

    checked = verify_manifest(
        client, SPEC.corpus.bucket, manifest,
        sample=None if sample == 0 else sample,
        log=lambda m: log(m, "info"),
    )
    log_header("Corpus Verified")
    log(f"{checked} clips decoded byte-identically to the manifest — this box can duel", "success")


@cli.group()
def miner():
    """Miner management commands.

    Commands for uploading model weights to Hippius Hub and committing
    the model reveal to the blockchain for validator discovery.
    """


@miner.command("push")
@click.option("--model-dir", required=True, help="Local folder with the model weights (safetensors + config)")
@click.option("--repo", required=True, help="Hippius Hub repo id (must start with 'leoma' and end with your hotkey), e.g. user/leoma-<name>-<hotkey>")
@click.option("--revision", default=None, help="Optional revision/branch label for the upload")
@click.option("--message", "commit_message", default=None, help="Optional upload commit message")
def miner_push(model_dir, repo, revision, commit_message):
    """Upload model weights to Hippius Hub.

    Uploads the safetensors + config from a local folder to Hippius Hub and
    prints the immutable repo@digest to commit.

    Example:

        leoma miner push --model-dir ./out --repo user/leoma-mymodel-5GRW...
    """
    import json
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.app.miner.main import push_command

    async def run():
        log_header("Uploading to Hippius Hub")
        log(f"Model dir: {model_dir}", "info")
        log(f"Repo: {repo}", "info")

        result = await push_command(
            model_dir=model_dir,
            repo=repo,
            revision=revision,
            commit_message=commit_message,
        )

        print(json.dumps(result, indent=2))

        if result.get("success"):
            log_header("Upload Complete")
            log(f"Immutable ref: {result.get('immutable_ref')}", "success")
            log("Next: leoma miner commit --repo <repo> --digest <digest>", "info")
        else:
            log(f"Upload failed: {result.get('error')}", "error")

    _run_async(run())


@miner.command("commit")
@click.option("--repo", required=True, help="Hippius Hub repo id (from push output)")
@click.option("--digest", required=True, help="Immutable digest, e.g. sha256:<64hex> (from push output)")
@click.option("--coldkey", help="Wallet coldkey name (optional, from env WALLET_NAME)")
@click.option("--hotkey", help="Wallet hotkey name (optional, from env HOTKEY_NAME)")
def miner_commit(repo, digest, coldkey, hotkey):
    """Commit the model reveal to the blockchain.

    Reveals ``v4|<repo>|<digest>|<hotkey>`` on-chain so validators can discover
    and download the exact model weights.

    Example:

        leoma miner commit --repo user/leoma-mymodel-5GRW... --digest sha256:abc...
    """
    import json
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.app.miner.main import commit_command

    async def run():
        log_header("Committing to Chain")
        log(f"Repo: {repo}", "info")
        log(f"Digest: {digest[:24]}...", "info")

        result = await commit_command(
            repo=repo,
            digest=digest,
            coldkey=coldkey,
            hotkey=hotkey,
        )

        print(json.dumps(result, indent=2))

        if result.get("success"):
            log_header("Commit Complete")
            log("Model reveal committed to chain", "success")
        else:
            log(f"Commit failed: {result.get('error')}", "error")

    _run_async(run())


if __name__ == "__main__":
    cli()
