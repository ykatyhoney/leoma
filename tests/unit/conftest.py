"""Unit-test fixtures for fast local database and service isolation."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from leoma.infra.db.tables import ValidMiner, ValidatorSample, RankScore, Blacklist, Validator


# -----------------------------------------------------------------------------
# Store Fixtures with Patched Sessions
# -----------------------------------------------------------------------------


@pytest.fixture
def participant_store(mock_get_session):
    """Provide a ParticipantStore instance that uses the test database."""
    from leoma.infra.db.stores.store_participant import ParticipantStore
    return ParticipantStore()


@pytest.fixture
def evaluation_store(mock_get_session):
    """Provide a store instance for validator evaluation records."""
    from leoma.infra.db.stores.store_sample import SampleStore as EvaluationStore
    return EvaluationStore()


@pytest.fixture
def rank_store(mock_get_session):
    """Provide a RankStore instance that uses the test database."""
    from leoma.infra.db.stores.store_rank import RankStore
    return RankStore()


@pytest.fixture
def blacklist_store(mock_get_session):
    """Provide a BlacklistStore instance that uses the test database."""
    from leoma.infra.db.stores.store_blacklist import BlacklistStore
    return BlacklistStore()


@pytest.fixture
def validator_store(mock_get_session):
    """Provide a ValidatorStore instance that uses the test database."""
    from leoma.infra.db.stores.store_validator import ValidatorStore
    return ValidatorStore()


# -----------------------------------------------------------------------------
# Pre-populated Database Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
async def db_with_valid_miner(
    db_session: AsyncSession,
    valid_miner_factory,
    test_hotkeys: list[str],
) -> ValidMiner:
    """Provide a database session with one valid miner already created."""
    miner = valid_miner_factory.create(
        uid=0,
        miner_hotkey=test_hotkeys[0],
        is_valid=True,
    )
    db_session.add(miner)
    await db_session.commit()
    await db_session.refresh(miner)
    return miner


@pytest.fixture
async def db_with_valid_miners(
    db_session: AsyncSession,
    valid_miner_factory,
    test_hotkeys: list[str],
) -> list[ValidMiner]:
    """Provide a database session with multiple valid miners."""
    miners = []
    for i, hotkey in enumerate(test_hotkeys):
        miner = valid_miner_factory.create(
            uid=i,
            miner_hotkey=hotkey,
            is_valid=(i % 2 == 0),  # Even UIDs are valid
        )
        db_session.add(miner)
        miners.append(miner)
    
    await db_session.commit()
    for miner in miners:
        await db_session.refresh(miner)
    
    return miners


@pytest.fixture
async def db_with_validator_evaluation(
    db_session: AsyncSession,
    validator_sample_factory,
    test_hotkeys: list[str],
) -> ValidatorSample:
    """Provide a database session with one validator evaluation."""
    evaluation = validator_sample_factory.create(
        validator_hotkey=test_hotkeys[0],
        task_id=1,
        miner_hotkey=test_hotkeys[1],
        passed=True,
    )
    db_session.add(evaluation)
    await db_session.commit()
    await db_session.refresh(evaluation)
    return evaluation


@pytest.fixture
async def db_with_validator_evaluations(
    db_session: AsyncSession,
    validator_sample_factory,
    test_hotkeys: list[str],
) -> list[ValidatorSample]:
    """Provide a database session with multiple validator evaluations."""
    evaluations = []
    validator_hotkey = test_hotkeys[0]
    
    for i, miner_hotkey in enumerate(test_hotkeys[1:4]):  # 3 miners
        evaluation = validator_sample_factory.create(
            validator_hotkey=validator_hotkey,
            task_id=i + 1,
            miner_hotkey=miner_hotkey,
            passed=(i % 2 == 0),
            confidence=80 + i * 5,
        )
        db_session.add(evaluation)
        evaluations.append(evaluation)
    
    await db_session.commit()
    for evaluation in evaluations:
        await db_session.refresh(evaluation)
    
    return evaluations


@pytest.fixture
async def db_with_rank_score(
    db_session: AsyncSession,
    rank_score_factory,
    test_hotkeys: list[str],
) -> RankScore:
    """Provide a database session with one rank score."""
    score = rank_score_factory.create(
        miner_hotkey=test_hotkeys[0],
        validator_hotkey=test_hotkeys[1],
        score=0.75,
        total_samples=20,
        total_passed=15,
        pass_rate=0.75,
    )
    db_session.add(score)
    await db_session.commit()
    await db_session.refresh(score)
    return score


@pytest.fixture
async def db_with_rank_scores(
    db_session: AsyncSession,
    rank_score_factory,
    test_hotkeys: list[str],
) -> list[RankScore]:
    """Provide a database session with multiple rank scores."""
    scores = []
    validator_hotkey = test_hotkeys[0]
    
    for i, miner_hotkey in enumerate(test_hotkeys[1:4]):  # 3 miners
        score = rank_score_factory.create(
            miner_hotkey=miner_hotkey,
            validator_hotkey=validator_hotkey,
            score=0.5 + (i * 0.1),
            total_samples=20,
            total_passed=10 + i * 2,
            pass_rate=0.5 + (i * 0.1),
        )
        db_session.add(score)
        scores.append(score)
    
    await db_session.commit()
    for score in scores:
        await db_session.refresh(score)
    
    return scores


@pytest.fixture
async def db_with_blacklist(
    db_session: AsyncSession,
    blacklist_factory,
) -> list[Blacklist]:
    """Provide a database session with blacklisted entries."""
    entries = []
    for i in range(3):
        entry = blacklist_factory.create(
            hotkey=f"5{'B' * 46}{i}",  # Unique blacklisted hotkeys
            reason=f"Moderation category {i}",
        )
        db_session.add(entry)
        entries.append(entry)
    
    await db_session.commit()
    for entry in entries:
        await db_session.refresh(entry)
    
    return entries


@pytest.fixture
async def db_with_validators(
    db_session: AsyncSession,
    validator_factory,
    test_hotkeys: list[str],
) -> list[Validator]:
    """Provide a database session with multiple validators."""
    validators = []
    for i, hotkey in enumerate(test_hotkeys[:3]):  # 3 validators
        validator = validator_factory.create(
            uid=i,
            hotkey=hotkey,
            stake=10000.0 + (i * 5000),
        )
        db_session.add(validator)
        validators.append(validator)
    
    await db_session.commit()
    for validator in validators:
        await db_session.refresh(validator)
    
    return validators


# -----------------------------------------------------------------------------
# Mock External Service Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_openai_client(mocker):
    """Create a mock OpenAI client for description generation tests (sampler side)."""
    from unittest.mock import AsyncMock

    mock_client = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.choices = [
        mocker.MagicMock(
            message=mocker.MagicMock(
                content='{"passed": true, "confidence": 85, "reasoning": "Test"}'
            )
        )
    ]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    return mock_client


@pytest.fixture
def mock_gemini_client(mocker):
    """Create a mock Gemini client for validator evaluation tests."""
    from unittest.mock import AsyncMock

    mock_client = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.text = '{"overall_score": 0}'
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    return mock_client


@pytest.fixture
def mock_chutes_client(mocker):
    """Create a mock Chutes API client for generation tests."""
    from unittest.mock import AsyncMock
    
    mock_client = mocker.MagicMock()
    mock_client.generate = AsyncMock(return_value={
        "video_url": "https://test.chutes.ai/video.mp4",
        "status": "completed",
    })
    mock_client.check_status = AsyncMock(return_value="completed")
    
    return mock_client


@pytest.fixture
def mock_bittensor(mocker):
    """Create mock bittensor objects for chain tests."""
    mock_subtensor = mocker.MagicMock()
    mock_subtensor.get_block_number.return_value = 1000000
    mock_subtensor.query_map_subtensor.return_value = {}
    
    mock_metagraph = mocker.MagicMock()
    mock_metagraph.hotkeys = [f"5{'F' * 47}" for _ in range(10)]
    mock_metagraph.uids = list(range(10))
    
    return {
        "subtensor": mock_subtensor,
        "metagraph": mock_metagraph,
    }
