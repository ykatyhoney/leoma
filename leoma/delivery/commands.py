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


def _api_url() -> str:
    """Return configured API base URL."""
    import os

    return os.environ.get("API_URL", "https://api.leoma.ai")


@click.group()
@click.version_option(version=__version__, prog_name="leoma")
def cli():
    """Leoma - Image-to-Video Generation Subnet
    
    A TI2V subnet where miners run I2V generators and validators
    challenge them using GPT-4o pass/fail evaluation.
    """


@cli.command()
def api():
    """Start the API service (with background tasks).
    
    Runs FastAPI + miner validation, score calculation, and rank update tasks.
    Requires DATABASE_URL (or POSTGRES_*). Listens on API_HOST:API_PORT (default 0.0.0.0:8000).
    """
    from leoma.delivery.http.server import main as api_main
    api_main()


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


@servers.command("sampler")
def start_sampler():
    """Start the validator sampler + self-evaluator (separate process).

    On this validator's rotation turn (GET /rotation): samples valid miners, uploads the task
    artifacts to its own bucket (R2_OWN_BUCKET), self-evaluates them (no cross-validation),
    publishes verdicts to its bucket, dual-reports to the dashboard, and announces the task.
    """
    from leoma.app.sampler.loop import run_sampler_loop
    _run_async(run_sampler_loop())


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
def db():
    """Database management commands.
    
    Commands for initializing and managing the PostgreSQL database.
    """


@db.command("init")
def db_init():
    """Initialize database tables.
    
    Creates all tables from the ORM (leoma.infra.db.tables). Use for fresh installs.
    """
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.infra.db.pool import init_database, create_tables, close_database
    
    async def run():
        log_header("Database Initialization")
        await init_database()
        await create_tables()
        await close_database()
        log("Database initialization complete", "success")
    
    _run_async(run())


@cli.group()
def blacklist():
    """Blacklist management commands (via API)."""


@blacklist.command("list")
def blacklist_list():
    """Show blacklisted miner hotkeys."""
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    
    async def run():
        log_header("Blacklist")
        
        api_url = _api_url()
        
        from leoma.infra.remote_api import APIClient
        
        client = APIClient(api_url=api_url)
        try:
            hotkeys = await client.get_blacklisted_miners()
            
            if not hotkeys:
                log("Blacklist is empty", "info")
            else:
                for hotkey in hotkeys:
                    log(f"  {hotkey}", "info")
                log(f"Total: {len(hotkeys)} miners", "info")
        finally:
            await client.close()
    
    _run_async(run())


@blacklist.command("add")
@click.argument("hotkey")
@click.option("--reason", "-r", default=None, help="Reason for blacklisting")
def blacklist_add(hotkey: str, reason: str):
    """Add a miner to the blacklist (requires admin wallet)."""
    from leoma.bootstrap import emit_log as log
    from leoma.bootstrap import WALLET_NAME, HOTKEY_NAME
    
    async def run():
        api_url = _api_url()
        
        from leoma.infra.remote_api import create_api_client_from_wallet
        
        client = create_api_client_from_wallet(
            wallet_name=WALLET_NAME,
            hotkey_name=HOTKEY_NAME,
            api_url=api_url,
        )
        try:
            await client.add_to_blacklist(hotkey=hotkey, reason=reason)
            log(f"Added {hotkey} to blacklist", "success")
        finally:
            await client.close()
    
    _run_async(run())


@blacklist.command("remove")
@click.argument("hotkey")
def blacklist_remove(hotkey: str):
    """Remove a miner from the blacklist (requires admin wallet)."""
    from leoma.bootstrap import emit_log as log
    from leoma.bootstrap import WALLET_NAME, HOTKEY_NAME
    
    async def run():
        api_url = _api_url()
        
        from leoma.infra.remote_api import create_api_client_from_wallet
        
        client = create_api_client_from_wallet(
            wallet_name=WALLET_NAME,
            hotkey_name=HOTKEY_NAME,
            api_url=api_url,
        )
        try:
            await client.remove_from_blacklist(hotkey)
            log(f"Removed {hotkey} from blacklist", "success")
        finally:
            await client.close()

    _run_async(run())


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


@cli.command("get-rank")
def get_rank():
    """Print miner rank list from the API (same data as the dashboard).
    
    Requires API_URL. Use GET /scores/rank under the hood.
    """
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.infra.remote_api import APIClient

    async def run():
        api_url = _api_url()
        log_header("Miner rank")
        client = APIClient(api_url=api_url)
        try:
            data = await client.get_rank()
            ranks = data.get("ranks") or []
            if not ranks:
                log("No ranks yet", "info")
                return
            for r in ranks:
                hotkey = r.get("miner_hotkey", "?")[:12]
                uid = r.get("uid", "?")
                rank = r.get("rank", "?")
                passed_count = r.get("passed_count", 0)
                pass_rate = r.get("pass_rate", 0.0)
                eligible = r.get("eligible", False)
                block = r.get("block")
                block_str = str(block) if block is not None else "—"
                log(f"  #{rank}  UID {uid}  block {block_str}  {hotkey}...  {passed_count} passes ({pass_rate:.1%})  eligible={eligible}", "info")
        finally:
            await client.close()

    _run_async(run())


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
