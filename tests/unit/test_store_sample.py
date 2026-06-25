"""Unit tests for validator evaluation persistence in `SampleStore`."""

from leoma.infra.db.stores.store_sample import SampleStore as EvaluationStore
from leoma.infra.db.tables import ValidatorSample


class TestEvaluationStoreSave:
    """Tests for evaluation upserts."""

    async def test_save_evaluation_creates_record(
        self,
        evaluation_store: EvaluationStore,
        test_hotkeys: list[str],
    ):
        """Test that `save_sample` creates a new evaluation record."""
        evaluation = await evaluation_store.save_sample(
            validator_hotkey=test_hotkeys[0],
            task_id=1,
            miner_hotkey=test_hotkeys[1],
            s3_bucket="test-bucket",
            s3_prefix="tasks/1",
            passed=True,
            prompt="A skateboarder crossing an empty plaza",
            confidence=85,
            reasoning="The generated clip matches the prompt and stays coherent.",
        )

        assert evaluation is not None
        assert evaluation.validator_hotkey == test_hotkeys[0]
        assert evaluation.task_id == 1
        assert evaluation.miner_hotkey == test_hotkeys[1]
        assert evaluation.s3_bucket == "test-bucket"
        assert evaluation.s3_prefix == "tasks/1"
        assert evaluation.passed is True
        assert evaluation.prompt == "A skateboarder crossing an empty plaza"
        assert evaluation.confidence == 85
        assert evaluation.reasoning == "The generated clip matches the prompt and stays coherent."

    async def test_save_evaluation_updates_existing_record(
        self,
        evaluation_store: EvaluationStore,
        test_hotkeys: list[str],
    ):
        """Test that the task-scoped unique key performs an upsert."""
        await evaluation_store.save_sample(
            validator_hotkey=test_hotkeys[0],
            task_id=1,
            miner_hotkey=test_hotkeys[1],
            s3_bucket="test-bucket",
            s3_prefix="tasks/1",
            passed=True,
            confidence=80,
        )

        updated = await evaluation_store.save_sample(
            validator_hotkey=test_hotkeys[0],
            task_id=1,
            miner_hotkey=test_hotkeys[1],
            s3_bucket="updated-bucket",
            s3_prefix="tasks/1/retry",
            passed=False,
            confidence=90,
        )

        assert updated.s3_bucket == "updated-bucket"
        assert updated.s3_prefix == "tasks/1/retry"
        assert updated.passed is False
        assert updated.confidence == 90

    async def test_save_evaluation_with_required_fields_only(
        self,
        evaluation_store: EvaluationStore,
        test_hotkeys: list[str],
    ):
        """Test saving an evaluation with only required fields."""
        evaluation = await evaluation_store.save_sample(
            validator_hotkey=test_hotkeys[0],
            task_id=2,
            miner_hotkey=test_hotkeys[1],
            s3_bucket="test-bucket",
            s3_prefix="tasks/2",
            passed=False,
        )

        assert evaluation is not None
        assert evaluation.prompt is None
        assert evaluation.confidence is None
        assert evaluation.reasoning is None


class TestEvaluationStoreReads:
    """Tests for evaluation retrieval methods."""

    async def test_get_samples_by_validator(
        self,
        evaluation_store: EvaluationStore,
        db_with_validator_evaluations: list[ValidatorSample],
    ):
        """Test retrieving evaluations by validator hotkey."""
        validator_hotkey = db_with_validator_evaluations[0].validator_hotkey
        evaluations = await evaluation_store.get_samples_by_validator(validator_hotkey)

        assert len(evaluations) == len(db_with_validator_evaluations)
        assert all(s.validator_hotkey == validator_hotkey for s in evaluations)

    async def test_get_samples_by_validator_respects_limit(
        self,
        evaluation_store: EvaluationStore,
        db_with_validator_evaluations: list[ValidatorSample],
    ):
        """Test that limit parameter is respected."""
        validator_hotkey = db_with_validator_evaluations[0].validator_hotkey
        evaluations = await evaluation_store.get_samples_by_validator(
            validator_hotkey, limit=1
        )

        assert len(evaluations) <= 1

    async def test_get_samples_by_validator_empty(
        self,
        evaluation_store: EvaluationStore,
    ):
        """Test validator lookup returns an empty list for an unknown hotkey."""
        evaluations = await evaluation_store.get_samples_by_validator("unknown-validator")

        assert evaluations == []

    async def test_get_samples_by_miner(
        self,
        evaluation_store: EvaluationStore,
        db_with_validator_evaluations: list[ValidatorSample],
    ):
        """Test retrieving evaluations by miner hotkey."""
        miner_hotkey = db_with_validator_evaluations[0].miner_hotkey
        evaluations = await evaluation_store.get_samples_by_miner(miner_hotkey)

        assert len(evaluations) >= 1
        assert all(s.miner_hotkey == miner_hotkey for s in evaluations)

    async def test_get_samples_by_miner_respects_limit(
        self,
        evaluation_store: EvaluationStore,
        db_with_validator_evaluations: list[ValidatorSample],
    ):
        """Test that limit parameter works for miner samples."""
        miner_hotkey = db_with_validator_evaluations[0].miner_hotkey
        evaluations = await evaluation_store.get_samples_by_miner(miner_hotkey, limit=1)

        assert len(evaluations) <= 1


class TestEvaluationStoreStats:
    """Tests for validator and miner evaluation statistics."""

    async def test_get_miner_stats_by_validator(
        self,
        evaluation_store: EvaluationStore,
        db_with_validator_evaluations: list[ValidatorSample],
    ):
        """Test getting miner stats for a specific validator."""
        validator_hotkey = db_with_validator_evaluations[0].validator_hotkey
        stats = await evaluation_store.get_miner_stats_by_validator(validator_hotkey)

        # Should have stats for each miner
        assert len(stats) > 0
        
        for miner_hotkey, miner_stats in stats.items():
            assert "passed_count" in miner_stats
            assert "total" in miner_stats
            assert "pass_rate" in miner_stats
            assert miner_stats["total"] >= 1
            assert 0.0 <= miner_stats["pass_rate"] <= 1.0

    async def test_get_miner_stats_by_validator_empty(
        self,
        evaluation_store: EvaluationStore,
    ):
        """Test stats for unknown validator returns empty dict."""
        stats = await evaluation_store.get_miner_stats_by_validator("unknown-validator")

        assert stats == {}

    async def test_get_all_miner_stats(
        self,
        evaluation_store: EvaluationStore,
        db_with_validator_evaluations: list[ValidatorSample],
    ):
        """Test getting aggregated miner stats across all validators."""
        stats = await evaluation_store.get_all_miner_stats()

        assert len(stats) > 0
        
        for miner_hotkey, miner_stats in stats.items():
            assert "passed_count" in miner_stats
            assert "total" in miner_stats
            assert "pass_rate" in miner_stats
            assert "validator_count" in miner_stats
            assert miner_stats["validator_count"] >= 1

    async def test_get_all_miner_stats_pass_rate_calculation(
        self,
        evaluation_store: EvaluationStore,
        test_hotkeys: list[str],
    ):
        """Test that pass rate is correctly calculated."""
        # Create samples with known outcomes
        validator_hotkey = test_hotkeys[0]
        miner_hotkey = test_hotkeys[1]
        
        # 3 passes, 2 failures = 60% pass rate
        for i in range(5):
            await evaluation_store.save_sample(
                validator_hotkey=validator_hotkey,
                task_id=100 + i,
                miner_hotkey=miner_hotkey,
                s3_bucket="test-bucket",
                s3_prefix=f"tasks/{100 + i}",
                passed=(i < 3),
            )

        stats = await evaluation_store.get_all_miner_stats()
        
        assert miner_hotkey in stats
        assert stats[miner_hotkey]["passed_count"] == 3
        assert stats[miner_hotkey]["total"] == 5
        assert stats[miner_hotkey]["pass_rate"] == 0.6


class TestEvaluationStoreCounts:
    """Tests for evaluation count operations."""

    async def test_get_evaluation_count_by_validator(
        self,
        evaluation_store: EvaluationStore,
        db_with_validator_evaluations: list[ValidatorSample],
    ):
        """Test counting samples for a specific validator."""
        validator_hotkey = db_with_validator_evaluations[0].validator_hotkey
        count = await evaluation_store.get_sample_count_by_validator(validator_hotkey)

        assert count == len(db_with_validator_evaluations)

    async def test_get_evaluation_count_by_validator_zero(
        self,
        evaluation_store: EvaluationStore,
    ):
        """Test count for unknown validator returns zero."""
        count = await evaluation_store.get_sample_count_by_validator("unknown-validator")

        assert count == 0

    async def test_get_total_evaluation_count(
        self,
        evaluation_store: EvaluationStore,
        db_with_validator_evaluations: list[ValidatorSample],
    ):
        """Test counting total samples across all validators."""
        count = await evaluation_store.get_total_sample_count()

        assert count == len(db_with_validator_evaluations)

    async def test_get_total_evaluation_count_empty(
        self,
        evaluation_store: EvaluationStore,
    ):
        """Test total count on empty database."""
        count = await evaluation_store.get_total_sample_count()

        assert count == 0


class TestEvaluationStoreDeletes:
    """Tests for delete operations."""

    async def test_delete_samples_by_validator(
        self,
        evaluation_store: EvaluationStore,
        db_with_validator_evaluations: list[ValidatorSample],
    ):
        """Test deleting all samples from a validator."""
        validator_hotkey = db_with_validator_evaluations[0].validator_hotkey
        deleted = await evaluation_store.delete_samples_by_validator(validator_hotkey)

        assert deleted == len(db_with_validator_evaluations)

        # Verify deletion
        remaining = await evaluation_store.get_samples_by_validator(validator_hotkey)
        assert len(remaining) == 0

    async def test_delete_samples_by_validator_not_found(
        self,
        evaluation_store: EvaluationStore,
    ):
        """Test delete returns 0 for unknown validator."""
        deleted = await evaluation_store.delete_samples_by_validator("unknown-validator")

        assert deleted == 0

    async def test_delete_samples_by_miner(
        self,
        evaluation_store: EvaluationStore,
        test_hotkeys: list[str],
    ):
        """Test deleting all evaluations for a miner."""
        miner_hotkey = test_hotkeys[0]
        
        for i, validator_hotkey in enumerate(test_hotkeys[1:4]):
            await evaluation_store.save_sample(
                validator_hotkey=validator_hotkey,
                task_id=200 + i,
                miner_hotkey=miner_hotkey,
                s3_bucket="test-bucket",
                s3_prefix=f"tasks/{200 + i}",
                passed=True,
            )

        deleted = await evaluation_store.delete_samples_by_miner(miner_hotkey)

        assert deleted == 3

        # Verify deletion
        remaining = await evaluation_store.get_samples_by_miner(miner_hotkey)
        assert len(remaining) == 0


class TestEvaluationStoreTaskKeying:
    """Tests for task-scoped uniqueness and fanout."""

    async def test_multiple_validators_can_store_same_task_for_one_miner(
        self,
        evaluation_store: EvaluationStore,
        test_hotkeys: list[str],
    ):
        """Test that multiple validators can store results for the same task/miner combo."""
        miner_hotkey = test_hotkeys[0]
        task_id = 300
        
        await evaluation_store.save_sample(
            validator_hotkey=test_hotkeys[1],
            task_id=task_id,
            miner_hotkey=miner_hotkey,
            s3_bucket="bucket1",
            s3_prefix="prefix1",
            passed=True,
        )
        
        await evaluation_store.save_sample(
            validator_hotkey=test_hotkeys[2],
            task_id=task_id,
            miner_hotkey=miner_hotkey,
            s3_bucket="bucket2",
            s3_prefix="prefix2",
            passed=False,
        )

        miner_evaluations = await evaluation_store.get_samples_by_miner(miner_hotkey)
        assert len(miner_evaluations) == 2
        
        # Check different validators
        validators = {s.validator_hotkey for s in miner_evaluations}
        assert len(validators) == 2

    async def test_same_validator_can_store_one_task_for_multiple_miners(
        self,
        evaluation_store: EvaluationStore,
        test_hotkeys: list[str],
    ):
        """Test that one validator can store results for multiple miners on one task."""
        validator_hotkey = test_hotkeys[0]
        task_id = 400
        
        for i, miner_hotkey in enumerate(test_hotkeys[1:4]):
            await evaluation_store.save_sample(
                validator_hotkey=validator_hotkey,
                task_id=task_id,
                miner_hotkey=miner_hotkey,
                s3_bucket="test-bucket",
                s3_prefix=f"prefix-{i}",
                passed=(i % 2 == 0),
            )

        validator_samples = await evaluation_store.get_samples_by_validator(validator_hotkey)
        assert len(validator_samples) == 3
        
        # All should have same task_id
        assert all(s.task_id == task_id for s in validator_samples)


class TestSampleDerivedWindow:
    """The dashboard's scoring window + latest task now come straight from validator_samples."""

    async def _seed(self, store, test_hotkeys):
        # (task_id -> sampler) across three validators; one validator samples two tasks.
        v0, v1, v2 = test_hotkeys[0], test_hotkeys[1], test_hotkeys[2]
        miner = test_hotkeys[3]
        for task_id, sampler in [(10, v0), (11, v1), (12, v2), (13, v0), (14, v1)]:
            await store.save_sample(
                validator_hotkey=sampler, task_id=task_id, miner_hotkey=miner,
                s3_bucket="b", s3_prefix=f"tasks/{task_id}", passed=True,
            )
        return v0, v1, v2

    async def test_get_recent_task_window_drops_margin_and_caps_n(
        self, evaluation_store: EvaluationStore, test_hotkeys: list[str]
    ):
        v0, v1, v2 = await self._seed(evaluation_store, test_hotkeys)
        # distinct ids desc [14,13,12,11,10] -> drop newest 1 -> [13,12,11,10] -> take 3 -> [11,12,13]
        window, active = await evaluation_store.get_recent_task_window(n=3, margin=1)
        assert window == [11, 12, 13]
        assert active == sorted({v0, v1, v2})  # samplers of 11(v1),12(v2),13(v0)

    async def test_get_recent_task_window_empty(
        self, evaluation_store: EvaluationStore
    ):
        assert await evaluation_store.get_recent_task_window(n=5, margin=0) == ([], [])

    async def test_get_latest_task(
        self, evaluation_store: EvaluationStore, test_hotkeys: list[str]
    ):
        _, v1, _ = await self._seed(evaluation_store, test_hotkeys)
        assert await evaluation_store.get_latest_task() == (14, v1)

    async def test_get_latest_task_none_when_empty(
        self, evaluation_store: EvaluationStore
    ):
        assert await evaluation_store.get_latest_task() is None
