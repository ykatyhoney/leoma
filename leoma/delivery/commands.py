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
    """Start the validator (sampler + evaluator + weight setter).

    Runs the complete decentralized Leoma validator in one process:
    - sampler loop: on this validator's rotation turn, sample miners and publish to its own bucket
    - evaluator loop: download the latest task from the sampler's bucket, Gemini-evaluate, publish results
    - weight-setting loop: aggregate all peers' results equally and set on-chain weights
    Requires R2_OWN_BUCKET, PEER_VALIDATORS, source-bucket read creds, CHUTES_API_KEY, GEMINI_API_KEY.
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
    """Start weight-setting service only.
    
    Polls API GET /weights and sets top-ranked-only weights on-chain. For full validator
    (evaluator + weight-setter), use leoma serve.
    """
    import bittensor as bt
    
    from leoma.bootstrap import NETWORK, WALLET_NAME, HOTKEY_NAME, emit_log as log, emit_header as log_header
    from leoma.app.validator.main import step
    
    async def run_validator():
        log_header("Leoma Validator (Weight Setter) Starting")
        
        subtensor = bt.AsyncSubtensor(network=NETWORK)
        wallet = bt.Wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
        log(f"Wallet: {WALLET_NAME}/{HOTKEY_NAME}", "info")
        log(f"Network: {NETWORK}", "info")
        
        while True:
            await step(subtensor, wallet)
    
    _run_async(run_validator())


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
    
    Commands for deploying I2V models to Chutes and committing
    model info to the blockchain.
    """


@miner.command("push")
@click.option("--model-name", required=True, help="HuggingFace repository ID (e.g., user/model-name)")
@click.option("--model-revision", required=True, help="HuggingFace commit SHA")
@click.option("--chutes-api-key", help="Chutes API key (optional, from env CHUTES_API_KEY)")
@click.option("--chute-user", help="Chutes username (optional, from env CHUTE_USER)")
def miner_push(model_name, model_revision, chutes_api_key, chute_user):
    """Deploy I2V model to Chutes.
    
    Generates a Chute configuration for the I2V model and deploys it.
    
    Example:
    
        leoma miner push --model-name user/model --model-revision abc123 --chutes-api-key api-key --chute-user myuser
    """
    import json
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.app.miner.main import push_command
    
    async def run():
        log_header("Deploying to Chutes")
        log(f"Repository: {model_name}", "info")
        log(f"Revision: {model_revision[:16]}...", "info")
        
        result = await push_command(
            model_name=model_name,
            model_revision=model_revision,
            chutes_api_key=chutes_api_key,
            chute_user=chute_user,
        )
        
        print(json.dumps(result, indent=2))
        
        if result.get("success"):
            log_header("Deployment Complete")
            log(f"Chute ID: {result.get('chute_id')}", "success")
        else:
            log(f"Deployment failed: {result.get('error')}", "error")
    
    _run_async(run())


@miner.command("commit")
@click.option("--model-name", required=True, help="HuggingFace repository ID")
@click.option("--model-revision", required=True, help="HuggingFace commit SHA")
@click.option("--chute-id", required=True, help="Chutes deployment ID")
@click.option("--coldkey", help="Wallet coldkey name (optional, from env WALLET_NAME)")
@click.option("--hotkey", help="Wallet hotkey name (optional, from env HOTKEY_NAME)")
def miner_commit(model_name, model_revision, chute_id, coldkey, hotkey):
    """Commit model info to blockchain.
    
    Commits the model repository, revision, and chute ID to the
    Bittensor chain for validator discovery.
    
    Example:
    
        leoma miner commit --model-name user/model --model-revision abc123 --chute-id xyz789 --coldkey default --hotkey default
    """
    import json
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.app.miner.main import commit_command
    
    async def run():
        log_header("Committing to Chain")
        log(f"Repository: {model_name}", "info")
        log(f"Revision: {model_revision[:16]}...", "info")
        log(f"Chute ID: {chute_id}", "info")
        
        result = await commit_command(
            model_name=model_name,
            model_revision=model_revision,
            chute_id=chute_id,
            coldkey=coldkey,
            hotkey=hotkey,
        )
        
        print(json.dumps(result, indent=2))
        
        if result.get("success"):
            log_header("Commit Complete")
            log("Model info committed to chain", "success")
        else:
            log(f"Commit failed: {result.get('error')}", "error")
    
    _run_async(run())


if __name__ == "__main__":
    cli()
