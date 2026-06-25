"""
Shared test fixtures for Leoma test suite.

Provides:
- Async database session (SQLite in-memory for unit tests)
- PostgreSQL test database (for integration tests via Docker)
- Mock S3 client
- Sample data factories
"""

import os

# Production default is R2; tests and fixtures assume Hippius unless the suite sets this.
os.environ.setdefault("OBJECT_STORAGE_BACKEND", "hippius")

import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from leoma.infra.db.tables import Base, ValidMiner, ValidatorSample, RankScore, Blacklist, Validator


class AsyncSessionAdapter:
    """Thin async wrapper around a sync SQLAlchemy session for unit tests.

    The production code expects async session methods, but unit tests only need
    local SQLite behavior. This avoids the aiosqlite hangs seen in this env.
    """

    def __init__(self, session: Session):
        self._session = session

    def add(self, instance: Any) -> None:
        self._session.add(instance)

    async def execute(self, *args: Any, **kwargs: Any):
        return self._session.execute(*args, **kwargs)

    async def get(self, *args: Any, **kwargs: Any):
        return self._session.get(*args, **kwargs)

    async def flush(self) -> None:
        self._session.flush()

    async def commit(self) -> None:
        self._session.commit()

    async def rollback(self) -> None:
        self._session.rollback()

    async def refresh(self, instance: Any) -> None:
        self._session.refresh(instance)

    async def close(self) -> None:
        self._session.close()


# -----------------------------------------------------------------------------
# Database Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
async def sqlite_engine() -> AsyncGenerator[Engine, None]:
    """Create an in-memory SQLite engine for unit tests.
    
    This provides fast, isolated database tests without external dependencies.
    """
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Cleanup
    engine.dispose()


@pytest.fixture
async def db_session(sqlite_engine: Engine) -> AsyncGenerator[AsyncSessionAdapter, None]:
    """Create a session adapter for testing.
    
    Each test gets a fresh session with automatic rollback on cleanup.
    """
    session_factory = sessionmaker(
        sqlite_engine,
        expire_on_commit=False,
    )
    
    session = AsyncSessionAdapter(session_factory())
    try:
        yield session
    finally:
        await session.rollback()
        await session.close()


@pytest.fixture
def mock_get_session(db_session: AsyncSessionAdapter, monkeypatch: pytest.MonkeyPatch):
    """Patch the global get_session to use the test database session.
    
    This allows DAO classes to work transparently with the test database.
    """
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def _mock_get_session():
        yield db_session
        # Don't commit in tests - let the test control this
    
    # Patch pool and every store module that imports get_session
    monkeypatch.setattr("leoma.infra.db.pool.get_session", _mock_get_session)
    for mod in (
        "leoma.infra.db.stores.store_participant",
        "leoma.infra.db.stores.store_sample",
        "leoma.infra.db.stores.store_produced_task",
        "leoma.infra.db.stores.store_rank",
        "leoma.infra.db.stores.store_blacklist",
        "leoma.infra.db.stores.store_validator",
        "leoma.infra.db.stores.store_sampling_state",
        "leoma.infra.db.stores.store_miner_rank",
        "leoma.infra.db.stores.store_miner_task_rank",
        "leoma.infra.db.stores.store_evaluation_signature",
    ):
        monkeypatch.setattr(f"{mod}.get_session", _mock_get_session)
    
    return db_session


# -----------------------------------------------------------------------------
# Mock S3/Minio Client Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_minio_client() -> MagicMock:
    """Create a mock Minio client for S3 operations.
    
    Simulates S3 bucket operations without requiring actual S3.
    """
    client = MagicMock()
    
    # Mock bucket operations
    client.bucket_exists.return_value = True
    client.make_bucket.return_value = None
    
    # Mock object operations
    client.put_object.return_value = MagicMock(
        object_name="test-object",
        etag="test-etag",
        version_id=None,
    )
    client.get_object.return_value = MagicMock(
        read=MagicMock(return_value=b"test data"),
        close=MagicMock(),
    )
    client.stat_object.return_value = MagicMock(
        size=1024,
        etag="test-etag",
        content_type="video/mp4",
        last_modified=datetime.now(timezone.utc),
    )
    client.remove_object.return_value = None
    
    # Mock list operations
    client.list_objects.return_value = iter([])
    
    return client


@pytest.fixture
def mock_hippius_client(mock_minio_client: MagicMock) -> MagicMock:
    """Create a mock Hippius S3 client wrapper.
    
    Wraps the mock Minio client with Hippius-specific behavior.
    """
    hippius = MagicMock()
    hippius.client = mock_minio_client
    hippius.bucket = "test-bucket"
    
    # Mock download method
    async def mock_download(bucket: str, key: str, destination: str):
        # Create a dummy file at destination
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as f:
            f.write(b"mock video data")
        return destination
    
    hippius.download_file = AsyncMock(side_effect=mock_download)
    hippius.upload_file = AsyncMock(return_value="s3://test-bucket/test-key")
    hippius.file_exists = AsyncMock(return_value=True)
    hippius.list_files = AsyncMock(return_value=[])
    
    return hippius


# -----------------------------------------------------------------------------
# Test Data Factories
# -----------------------------------------------------------------------------


class ValidMinerFactory:
    """Factory for creating test ValidMiner objects."""
    
    @staticmethod
    def create(
        uid: int,
        miner_hotkey: Optional[str] = None,
        is_valid: bool = True,
        model_name: str = "",
        model_revision: str = "",
        model_hash: Optional[str] = None,
        chute_id: str = "",
        chute_slug: str = "",
        **kwargs: Any,
    ) -> ValidMiner:
        """Create a ValidMiner instance with sensible defaults."""
        if miner_hotkey is None:
            miner_hotkey = f"5{'F' * 47}"  # Valid-looking hotkey format
        
        return ValidMiner(
            uid=uid,
            miner_hotkey=miner_hotkey,
            block=kwargs.get("block", 1000000),
            model_name=model_name,
            model_revision=model_revision,
            model_hash=model_hash,
            chute_id=chute_id,
            chute_slug=chute_slug,
            is_valid=is_valid,
            invalid_reason=kwargs.get("invalid_reason"),
            last_validated_at=kwargs.get("last_validated_at"),
        )


class ValidatorSampleFactory:
    """Factory for creating test ValidatorSample objects."""
    
    @staticmethod
    def create(
        validator_hotkey: Optional[str] = None,
        task_id: Optional[int] = None,
        miner_hotkey: Optional[str] = None,
        passed: bool = True,
        confidence: int = 85,
        s3_bucket: str = "test-bucket",
        s3_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> ValidatorSample:
        """Create a ValidatorSample instance with sensible defaults."""
        if validator_hotkey is None:
            validator_hotkey = f"5{'V' * 47}"  # Validator hotkey format
        if miner_hotkey is None:
            miner_hotkey = f"5{'M' * 47}"  # Miner hotkey format
        if task_id is None:
            task_id = int(uuid.uuid4().int % 1000000)
        if s3_prefix is None:
            s3_prefix = f"tasks/{task_id}"
        
        return ValidatorSample(
            validator_hotkey=validator_hotkey,
            task_id=task_id,
            miner_hotkey=miner_hotkey,
            prompt=kwargs.get("prompt", "A skateboarder crossing an empty plaza"),
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            passed=passed,
            confidence=confidence,
            reasoning=kwargs.get(
                "reasoning",
                "The generated clip matches the prompt and stays coherent.",
            ),
        )


class RankScoreFactory:
    """Factory for creating test RankScore objects."""
    
    @staticmethod
    def create(
        miner_hotkey: Optional[str] = None,
        validator_hotkey: Optional[str] = None,
        score: float = 0.75,
        total_samples: int = 20,
        total_passed: int = 15,
        pass_rate: float = 0.75,
        **kwargs: Any,
    ) -> RankScore:
        """Create a RankScore instance with sensible defaults."""
        if miner_hotkey is None:
            miner_hotkey = f"5{'M' * 47}"
        if validator_hotkey is None:
            validator_hotkey = f"5{'V' * 47}"
        
        return RankScore(
            miner_hotkey=miner_hotkey,
            validator_hotkey=validator_hotkey,
            score=score,
            total_samples=total_samples,
            total_passed=total_passed,
            pass_rate=pass_rate,
        )


class BlacklistFactory:
    """Factory for creating test Blacklist objects."""
    
    @staticmethod
    def create(
        hotkey: Optional[str] = None,
        reason: Optional[str] = "Test blacklist reason",
        added_by: Optional[str] = None,
        **kwargs: Any,
    ) -> Blacklist:
        """Create a Blacklist instance with sensible defaults."""
        if hotkey is None:
            hotkey = f"5{'B' * 47}"  # Blacklisted hotkey format
        
        return Blacklist(
            hotkey=hotkey,
            reason=reason,
            added_by=added_by,
        )


class ValidatorFactory:
    """Factory for creating test Validator objects."""
    
    @staticmethod
    def create(
        uid: int,
        hotkey: Optional[str] = None,
        stake: float = 10000.0,
        s3_bucket: Optional[str] = None,
        **kwargs: Any,
    ) -> Validator:
        """Create a Validator instance with sensible defaults."""
        if hotkey is None:
            hotkey = f"5{'V' * 47}"  # Validator hotkey format
        
        return Validator(
            uid=uid,
            hotkey=hotkey,
            stake=stake,
            s3_bucket=s3_bucket,
            last_seen_at=kwargs.get("last_seen_at"),
        )


@pytest.fixture
def valid_miner_factory() -> type[ValidMinerFactory]:
    """Provide access to the ValidMinerFactory class."""
    return ValidMinerFactory


@pytest.fixture
def validator_sample_factory() -> type[ValidatorSampleFactory]:
    """Provide access to the ValidatorSampleFactory class."""
    return ValidatorSampleFactory


@pytest.fixture
def rank_score_factory() -> type[RankScoreFactory]:
    """Provide access to the RankScoreFactory class."""
    return RankScoreFactory


@pytest.fixture
def blacklist_factory() -> type[BlacklistFactory]:
    """Provide access to the BlacklistFactory class."""
    return BlacklistFactory


@pytest.fixture
def validator_factory() -> type[ValidatorFactory]:
    """Provide access to the ValidatorFactory class."""
    return ValidatorFactory


# -----------------------------------------------------------------------------
# Shared Test Data Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def test_hotkeys() -> list[str]:
    """Provide a reusable list of hotkeys for tests."""
    return [
        "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y",
        "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy",
        "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw",
    ]


@pytest.fixture
def sample_hotkeys(test_hotkeys: list[str]) -> list[str]:
    """Backward-compatible alias for integration tests still using the old name."""
    return test_hotkeys


@pytest.fixture
def evaluation_result_payload() -> Dict[str, Any]:
    """Provide a validator evaluation result payload."""
    return {
        "passed": True,
        "confidence": 85,
        "reasoning": "The generated clip stays coherent and follows the prompt.",
        "original_artifacts": [],
        "generated_artifacts": [],
        "presentation_order": "generated_first",
    }


@pytest.fixture
def sample_evaluation_result(evaluation_result_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible alias for the evaluation payload fixture."""
    return evaluation_result_payload


@pytest.fixture
def source_video_info() -> Dict[str, Any]:
    """Provide source video information."""
    return {
        "bucket": "hippius-videos",
        "key": "source-videos/nature/forest_walk.mp4",
        "full_duration_seconds": 120.5,
        "clip_start_seconds": 30.0,
        "clip_duration_seconds": 5.0,
    }


@pytest.fixture
def sample_source_info(source_video_info: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible alias for source video information."""
    return source_video_info


@pytest.fixture
def prompt_info() -> Dict[str, Any]:
    """Provide prompt information."""
    return {
        "model": "gpt-4o",
        "text": "A serene forest path with dappled sunlight filtering through the canopy.",
    }


@pytest.fixture
def sample_prompt_info(prompt_info: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible alias for prompt information."""
    return prompt_info


@pytest.fixture
def generation_info() -> Dict[str, Any]:
    """Provide generation information."""
    return {
        "model": "leoma-video",
        "endpoint": "https://api.chutes.ai/v1/generate",
        "parameters": {
            "fps": 16,
            "frames": 81,
            "resolution": "480p",
            "fast": True,
        },
    }


@pytest.fixture
def sample_generation_info(generation_info: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible alias for generation information."""
    return generation_info


# -----------------------------------------------------------------------------
# Environment Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def test_env(monkeypatch: pytest.MonkeyPatch) -> Dict[str, str]:
    """Set up test environment variables."""
    env_vars = {
        "OBJECT_STORAGE_BACKEND": "hippius",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5433",
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_DB": "leoma_test",
        "HIPPIUS_ENDPOINT": "test.hippius.com",
        "HIPPIUS_REGION": "decentralized",
        "HIPPIUS_SOURCE_BUCKET": "videos",
        "HIPPIUS_VIDEOS_READ_ACCESS_KEY": "test-videos-read-key",
        "HIPPIUS_VIDEOS_READ_SECRET_KEY": "test-videos-read-secret",
        "HIPPIUS_VIDEOS_WRITE_ACCESS_KEY": "test-videos-write-key",
        "HIPPIUS_VIDEOS_WRITE_SECRET_KEY": "test-videos-write-secret",
        "OPENAI_API_KEY": "test-openai-key",
        "NETUID": "999",
        "WALLET_NAME": "test_wallet",
        "WALLET_HOTKEY": "test_hotkey",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


# -----------------------------------------------------------------------------
# Async Helper Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    """Specify the async backend for anyio."""
    return "asyncio"
