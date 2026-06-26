"""Integration tests for PostgreSQL-backed persistence flows."""

import asyncio
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from leoma.infra.db.tables import ValidMiner


pytestmark = pytest.mark.requires_postgres


class TestEvaluationPersistenceFlow:
    """Tests complete evaluation persistence and score writing."""

    async def test_task_evaluation_flow_and_score_persistence(
        self,
        integration_participant_store,
        integration_evaluation_store,
        integration_rank_store,
        test_hotkeys,
    ):
        """Persist task evaluations, aggregate stats, and write scores."""
        validator_hotkey = test_hotkeys[0]
        miner_hotkeys = test_hotkeys[1:4]

        # 1. Create miners
        for i, hotkey in enumerate(miner_hotkeys):
            await integration_participant_store.save_miner(
                uid=i,
                miner_hotkey=hotkey,
                model_name=f"user/leoma-video-model-{i}",
                model_revision=f"rev{i}abc123",
                chute_id=f"3182321e-3e58-55da-ba44-05168{i}ddbfe5",
                chute_slug=f"chutes-leoma-video-miner-{i}",
                is_valid=True,
            )

        # 3. Persist evaluations for each miner
        for i, miner_hotkey in enumerate(miner_hotkeys):
            await integration_evaluation_store.save_sample(
                validator_hotkey=validator_hotkey,
                task_id=i + 1,
                miner_hotkey=miner_hotkey,
                s3_bucket="test-bucket",
                s3_prefix=f"tasks/lifecycle-{i + 1}",
                passed=i % 2 == 0,
                confidence=80 + i * 5,
                reasoning=f"Validator review for miner slot {i}",
            )

        evaluations = await integration_evaluation_store.get_samples_by_validator(
            validator_hotkey
        )
        assert len(evaluations) == 3

        stats = await integration_evaluation_store.get_miner_stats_by_validator(
            validator_hotkey
        )
        assert len(stats) == 3

        # 6. Save scores for each miner
        for miner_hotkey, miner_stats in stats.items():
            await integration_rank_store.save_score(
                miner_hotkey=miner_hotkey,
                validator_hotkey=validator_hotkey,
                score=miner_stats["pass_rate"],
                total_samples=miner_stats["total"],
                total_passed=miner_stats["passed_count"],
                pass_rate=miner_stats["pass_rate"],
            )

        # 7. Verify scores were saved
        scores = await integration_rank_store.get_scores_by_validator(validator_hotkey)
        assert len(scores) == 3

    async def test_multiple_validators_persist_task_reviews_for_one_miner(
        self,
        integration_participant_store,
        integration_evaluation_store,
        test_hotkeys,
    ):
        """Multiple validators can evaluate the same miner across different tasks."""
        miner_hotkey = test_hotkeys[0]
        validator_hotkeys = test_hotkeys[1:4]

        # Create miner
        await integration_participant_store.save_miner(
            uid=0,
            miner_hotkey=miner_hotkey,
            model_name="user/leoma-video-model",
            is_valid=True,
        )

        # Persist evaluations from multiple validators
        for i, validator_hotkey in enumerate(validator_hotkeys):
            for j in range(3):
                await integration_evaluation_store.save_sample(
                    validator_hotkey=validator_hotkey,
                    task_id=(i * 10) + j + 1,
                    miner_hotkey=miner_hotkey,
                    s3_bucket="test-bucket",
                    s3_prefix=f"tasks/review-{i}-{j}",
                    passed=j % 2 == 0,
                    confidence=85,
                )

        total_count = await integration_evaluation_store.get_total_sample_count()
        assert total_count == 9

        stats = await integration_evaluation_store.get_all_miner_stats()
        assert miner_hotkey in stats
        assert stats[miner_hotkey]["total"] == 9
        assert stats[miner_hotkey]["validator_count"] == 3


class TestConcurrentWrites:
    """Tests for concurrent database operations."""

    async def test_concurrent_task_evaluation_writes(
        self,
        integration_evaluation_store,
        test_hotkeys,
    ):
        """Concurrent evaluation writes should not conflict."""
        validator_hotkey = test_hotkeys[0]
        miner_hotkeys = test_hotkeys[1:4]

        # Concurrent writes
        async def write_evaluation(miner_hotkey: str, index: int):
            await integration_evaluation_store.save_sample(
                validator_hotkey=validator_hotkey,
                task_id=index + 1,
                miner_hotkey=miner_hotkey,
                s3_bucket="test-bucket",
                s3_prefix=f"tasks/concurrent-{index + 1}",
                passed=index % 2 == 0,
                confidence=80,
                reasoning=f"Concurrent validator review {index}",
            )

        # Execute concurrently
        await asyncio.gather(*[
            write_evaluation(miner_hotkey, i)
            for i, miner_hotkey in enumerate(miner_hotkeys)
        ])

        evaluations = await integration_evaluation_store.get_samples_by_validator(
            validator_hotkey
        )
        assert len(evaluations) == 3

    async def test_concurrent_miner_updates(
        self,
        integration_participant_store,
        test_hotkeys,
    ):
        """Test concurrent miner validation updates."""
        # Create miners
        for i, hotkey in enumerate(test_hotkeys[:3]):
            await integration_participant_store.save_miner(
                uid=i,
                miner_hotkey=hotkey,
                is_valid=False,
            )

        # Concurrent validation updates
        async def update_validation(uid: int, is_valid: bool):
            await integration_participant_store.set_validation_status(
                uid=uid,
                is_valid=is_valid,
                invalid_reason=None if is_valid else "Validator probe failed",
            )

        await asyncio.gather(*[
            update_validation(0, True),
            update_validation(1, True),
            update_validation(2, False),
        ])

        # Verify updates
        all_miners = await integration_participant_store.get_all_miners()
        assert len(all_miners) == 3

        valid_miners = await integration_participant_store.get_valid_miners()
        assert len(valid_miners) == 2

    async def test_concurrent_score_reads(
        self,
        integration_rank_store,
        test_hotkeys,
    ):
        """Test concurrent score reads don't interfere."""
        validator_hotkey = test_hotkeys[0]
        miner_hotkeys = test_hotkeys[1:4]

        # Create scores
        for i, miner_hotkey in enumerate(miner_hotkeys):
            await integration_rank_store.save_score(
                miner_hotkey=miner_hotkey,
                validator_hotkey=validator_hotkey,
                score=0.5 + i * 0.1,
                total_samples=10,
                total_passed=5 + i,
                pass_rate=0.5 + i * 0.1,
            )

        # Concurrent reads
        async def read_score(miner_hotkey: str):
            return await integration_rank_store.get_scores_by_miner(miner_hotkey)

        results = await asyncio.gather(*[
            read_score(miner_hotkey)
            for miner_hotkey in miner_hotkeys
        ])

        assert all(len(r) == 1 for r in results)


class TestDatabaseConstraints:
    """Tests for database constraints and integrity."""

    async def test_unique_miner_uid_constraint(
        self,
        integration_participant_store,
        test_hotkeys,
    ):
        """Test miner UID uniqueness (should update instead of fail)."""
        # Create miner
        await integration_participant_store.save_miner(
            uid=100,
            miner_hotkey=test_hotkeys[0],
        )

        # Save with same UID should update
        updated = await integration_participant_store.save_miner(
            uid=100,
            miner_hotkey=test_hotkeys[1],
            model_name="user/new-model",
            model_revision="newrev456",
            chute_id="11111111-2222-3333-4444-555555555555",
            chute_slug="chutes-new-slug",
        )

        assert updated.uid == 100
        assert updated.miner_hotkey == test_hotkeys[1]
        assert updated.model_name == "user/new-model"
        assert updated.chute_slug == "chutes-new-slug"

    async def test_unique_miner_hotkey_constraint(
        self,
        integration_participant_store,
        test_hotkeys,
    ):
        """Test that miner hotkey must be unique."""
        # Create miner
        await integration_participant_store.save_miner(
            uid=0,
            miner_hotkey=test_hotkeys[0],
        )

        # Attempt to create with same hotkey but different UID should fail
        with pytest.raises(Exception):  # IntegrityError
            await integration_participant_store.save_miner(
                uid=1,
                miner_hotkey=test_hotkeys[0],
            )

    async def test_unique_task_evaluation_constraint(
        self,
        integration_evaluation_store,
        test_hotkeys,
    ):
        """Validator/task/miner writes should upsert instead of duplicating rows."""
        validator_hotkey = test_hotkeys[0]
        miner_hotkey = test_hotkeys[1]

        first_evaluation = await integration_evaluation_store.save_sample(
            validator_hotkey=validator_hotkey,
            task_id=1,
            miner_hotkey=miner_hotkey,
            s3_bucket="test-bucket",
            s3_prefix="tasks/unique-1",
            passed=True,
            confidence=85,
        )

        updated_evaluation = await integration_evaluation_store.save_sample(
            validator_hotkey=validator_hotkey,
            task_id=1,
            miner_hotkey=miner_hotkey,
            s3_bucket="test-bucket",
            s3_prefix="tasks/unique-1",
            passed=False,
            confidence=90,
        )

        assert first_evaluation.id == updated_evaluation.id
        evaluations = await integration_evaluation_store.get_samples_by_validator(
            validator_hotkey
        )
        assert len(evaluations) == 1
        assert evaluations[0].passed is False
        assert evaluations[0].confidence == 90

class TestDeleteOperations:
    """Tests for delete operations."""

    async def test_delete_samples_by_validator(
        self,
        integration_evaluation_store,
        test_hotkeys,
    ):
        """Delete every evaluation submitted by one validator."""
        validator_hotkey = test_hotkeys[0]
        miner_hotkeys = test_hotkeys[1:4]

        # Create samples
        for i, miner_hotkey in enumerate(miner_hotkeys):
            await integration_evaluation_store.save_sample(
                validator_hotkey=validator_hotkey,
                task_id=i + 1,
                miner_hotkey=miner_hotkey,
                s3_bucket="test-bucket",
                s3_prefix=f"tasks/delete-{i + 1}",
                passed=True,
            )

        # Verify samples exist
        evaluations = await integration_evaluation_store.get_samples_by_validator(
            validator_hotkey
        )
        assert len(evaluations) == 3

        # Delete samples
        deleted = await integration_evaluation_store.delete_samples_by_validator(
            validator_hotkey
        )
        assert deleted == 3

        # Verify samples are gone
        evaluations = await integration_evaluation_store.get_samples_by_validator(
            validator_hotkey
        )
        assert len(evaluations) == 0

    async def test_delete_samples_by_miner(
        self,
        integration_evaluation_store,
        test_hotkeys,
    ):
        """Delete every evaluation associated with one miner."""
        miner_hotkey = test_hotkeys[0]
        validator_hotkeys = test_hotkeys[1:4]

        # Create samples from different validators
        for i, validator_hotkey in enumerate(validator_hotkeys):
            await integration_evaluation_store.save_sample(
                validator_hotkey=validator_hotkey,
                task_id=i + 1,
                miner_hotkey=miner_hotkey,
                s3_bucket="test-bucket",
                s3_prefix=f"tasks/miner-delete-{i + 1}",
                passed=True,
            )

        # Delete samples for miner
        deleted = await integration_evaluation_store.delete_samples_by_miner(
            miner_hotkey
        )
        assert deleted == 3

        # Verify samples are gone
        evaluations = await integration_evaluation_store.get_samples_by_miner(
            miner_hotkey
        )
        assert len(evaluations) == 0

    async def test_delete_stale_miners(
        self,
        integration_participant_store,
        test_hotkeys,
    ):
        """Test deleting miners not in the active list."""
        # Create miners
        for i, hotkey in enumerate(test_hotkeys[:5]):
            await integration_participant_store.save_miner(
                uid=i,
                miner_hotkey=hotkey,
                is_valid=True,
            )

        # Delete stale miners (keeping only first 3)
        active_uids = [0, 1, 2]
        deleted = await integration_participant_store.delete_stale_miners(active_uids)
        assert deleted == 2

        # Verify only active miners remain
        all_miners = await integration_participant_store.get_all_miners()
        assert len(all_miners) == 3
        assert all(m.uid in active_uids for m in all_miners)


class TestTransactionBehavior:
    """Tests for transaction handling."""

    async def test_rollback_on_error(
        self,
        postgres_session: AsyncSession,
        valid_miner_factory,
    ):
        """Test that errors trigger proper rollback."""
        # Start a transaction
        miner = valid_miner_factory.create(uid=999, miner_hotkey="5" + "X" * 47)
        postgres_session.add(miner)

        # Force a rollback
        await postgres_session.rollback()

        # Verify miner was not persisted
        result = await postgres_session.execute(
            select(ValidMiner).where(ValidMiner.uid == 999)
        )
        assert result.scalar_one_or_none() is None


class TestBlacklistOperations:
    """Tests for blacklist functionality."""

    async def test_blacklist_add_and_check(
        self,
        integration_blacklist_store,
        test_hotkeys,
    ):
        """Test adding to blacklist and checking status."""
        hotkey = test_hotkeys[0]

        # Initially not blacklisted
        assert await integration_blacklist_store.is_blacklisted(hotkey) is False

        # Add to blacklist
        entry = await integration_blacklist_store.add(
            hotkey=hotkey,
            reason="Manual moderation review",
            added_by="root-admin",
        )
        assert entry.hotkey == hotkey

        # Now blacklisted
        assert await integration_blacklist_store.is_blacklisted(hotkey) is True

    async def test_blacklist_remove(
        self,
        integration_blacklist_store,
        test_hotkeys,
    ):
        """Test removing from blacklist."""
        hotkey = test_hotkeys[0]

        # Add to blacklist
        await integration_blacklist_store.add(hotkey=hotkey, reason="Temporary quarantine")

        # Remove from blacklist
        removed = await integration_blacklist_store.remove(hotkey)
        assert removed is True

        # No longer blacklisted
        assert await integration_blacklist_store.is_blacklisted(hotkey) is False

    async def test_blacklist_get_all(
        self,
        integration_blacklist_store,
        test_hotkeys,
    ):
        """Test getting all blacklisted entries."""
        # Add multiple entries
        for i, hotkey in enumerate(test_hotkeys[:3]):
            await integration_blacklist_store.add(
                hotkey=hotkey,
                reason=f"Moderation category {i}",
            )

        # Get all
        all_entries = await integration_blacklist_store.get_all()
        assert len(all_entries) == 3

        # Get just hotkeys
        hotkeys = await integration_blacklist_store.get_hotkeys()
        assert len(hotkeys) == 3
        assert all(hk in test_hotkeys[:3] for hk in hotkeys)


class TestDataTypes:
    """Tests for PostgreSQL-specific data types."""

    async def test_timestamp_precision(
        self,
        integration_rank_store,
        test_hotkeys,
    ):
        """Test timestamp precision in PostgreSQL."""
        score = await integration_rank_store.save_score(
            miner_hotkey=test_hotkeys[0],
            validator_hotkey=test_hotkeys[1],
            score=0.75,
            total_samples=20,
            total_passed=15,
            pass_rate=0.75,
        )

        assert score.updated_at is not None
        # Should have microsecond precision
        assert score.updated_at.microsecond is not None

    async def test_float_precision(
        self,
        integration_rank_store,
        test_hotkeys,
    ):
        """Test float precision for scores."""
        precise_score = 0.123456789

        score = await integration_rank_store.save_score(
            miner_hotkey=test_hotkeys[0],
            validator_hotkey=test_hotkeys[1],
            score=precise_score,
            total_samples=100,
            total_passed=12,
            pass_rate=precise_score,
        )

        # Retrieve and check precision
        scores = await integration_rank_store.get_scores_by_miner(test_hotkeys[0])
        assert len(scores) == 1
        assert abs(scores[0].score - precise_score) < 1e-6
