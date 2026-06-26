"""Integration-test fixtures for PostgreSQL-backed Leoma flows."""

import os
from typing import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from leoma.infra.db.tables import Base


# -----------------------------------------------------------------------------
# PostgreSQL Fixtures
# -----------------------------------------------------------------------------


def get_test_postgres_url() -> str:
    """Get PostgreSQL test database URL from environment.
    
    Uses docker-compose.test.yml settings by default:
    - Host: localhost
    - Port: 5433 (different from production 5432)
    - User/Password/DB: test
    """
    host = os.environ.get("TEST_POSTGRES_HOST", "localhost")
    port = os.environ.get("TEST_POSTGRES_PORT", "5433")
    user = os.environ.get("TEST_POSTGRES_USER", "test")
    password = os.environ.get("TEST_POSTGRES_PASSWORD", "test")
    database = os.environ.get("TEST_POSTGRES_DB", "leoma_test")
    
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """Provide the PostgreSQL test database URL."""
    return get_test_postgres_url()


@pytest.fixture(scope="function")
async def postgres_engine(postgres_url: str) -> AsyncGenerator[AsyncEngine, None]:
    """Create a PostgreSQL async engine for integration tests.
    
    Each test gets a fresh database to ensure isolation.
    Tables are created at the start and dropped on cleanup.
    """
    engine = create_async_engine(
        postgres_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Drop all tables on cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def postgres_session(
    postgres_engine: AsyncEngine,
) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session connected to PostgreSQL test database.
    
    Each test gets its own session with automatic cleanup.
    """
    session_factory = async_sessionmaker(
        postgres_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with session_factory() as session:
        yield session
        # Rollback any uncommitted changes
        await session.rollback()


@pytest.fixture
async def clean_postgres_tables(postgres_engine: AsyncEngine):
    """Ensure clean tables before each integration test.
    
    Truncates all tables while keeping the schema intact.
    """
    from sqlalchemy import text
    
    async with postgres_engine.begin() as conn:
        # Disable foreign key checks temporarily for truncation
        await conn.execute(text("SET session_replication_role = 'replica'"))
        
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(text(f"TRUNCATE TABLE {table.name} CASCADE"))
        
        # Re-enable foreign key checks
        await conn.execute(text("SET session_replication_role = 'origin'"))


@pytest.fixture
def mock_postgres_get_session(
    postgres_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
):
    """Patch the global get_session to use the PostgreSQL test database.
    
    Creates new sessions for each call (like the real get_session) but connected
    to the test database. This allows concurrent operations to work correctly.
    """
    from contextlib import asynccontextmanager
    
    # Create a session factory from the test engine
    test_session_factory = async_sessionmaker(
        postgres_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    @asynccontextmanager
    async def _mock_get_session():
        session = test_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    # Patch in the client module AND in each DAO module where it's imported
    # Python imports bind references at import time, so we must patch where it's USED
    monkeypatch.setattr("leoma.infra.db.pool.get_session", _mock_get_session)


# -----------------------------------------------------------------------------
# Store Fixtures for Integration Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def integration_participant_store(mock_postgres_get_session):
    """Provide a ParticipantStore instance connected to PostgreSQL."""
    from leoma.infra.db.stores.store_participant import ParticipantStore
    return ParticipantStore()


@pytest.fixture
def integration_evaluation_store(mock_postgres_get_session):
    """Provide an evaluation store instance connected to PostgreSQL."""
    from leoma.infra.db.stores.store_sample import SampleStore as EvaluationStore
    return EvaluationStore()


@pytest.fixture
def integration_rank_store(mock_postgres_get_session):
    """Provide a RankStore instance connected to PostgreSQL."""
    from leoma.infra.db.stores.store_rank import RankStore
    return RankStore()


@pytest.fixture
def integration_blacklist_store(mock_postgres_get_session):
    """Provide a BlacklistStore instance connected to PostgreSQL."""
    from leoma.infra.db.stores.store_blacklist import BlacklistStore
    return BlacklistStore()


# -----------------------------------------------------------------------------
# Docker Compose Helper Markers
# -----------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers for integration tests."""
    config.addinivalue_line(
        "markers",
        "requires_postgres: mark test as requiring PostgreSQL (via docker-compose.test.yml)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests if PostgreSQL is not available."""
    import socket
    
    # Check if PostgreSQL is available
    host = os.environ.get("TEST_POSTGRES_HOST", "localhost")
    port = int(os.environ.get("TEST_POSTGRES_PORT", "5433"))
    
    postgres_available = False
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        postgres_available = result == 0
        sock.close()
    except Exception:
        pass
    
    if not postgres_available:
        skip_postgres = pytest.mark.skip(
            reason=f"PostgreSQL not available at {host}:{port}. "
            "Run 'docker compose -f docker-compose.test.yml up -d' first."
        )
        for item in items:
            if "requires_postgres" in item.keywords or "integration" in str(item.fspath):
                item.add_marker(skip_postgres)
