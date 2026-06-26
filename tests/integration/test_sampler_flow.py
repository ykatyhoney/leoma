"""Integration tests for validator evaluation and ranking flows."""

import pytest


pytestmark = pytest.mark.requires_postgres


class TestValidatorEvaluationFlow:
    """Tests end-to-end validator evaluation persistence."""

    async def test_validator_persists_task_reviews_for_miners(
        self,
        integration_participant_store,
        integration_evaluation_store,
        test_hotkeys,
    ):
        """Persist validator task reviews and verify derived stats."""
        validator_hotkey = test_hotkeys[0]
        miner_hotkeys = test_hotkeys[1:4]

        # 1. Setup: Create valid miners
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

        # 3. Verify miners are valid
        valid_miners = await integration_participant_store.get_valid_miners()
        assert len(valid_miners) == 3

        # 4. Validator writes task evaluations
        evaluation_results = [
            {"passed": True, "confidence": 85, "reasoning": "Prompt alignment is strong"},
            {"passed": False, "confidence": 70, "reasoning": "Temporal drift is visible"},
            {"passed": True, "confidence": 90, "reasoning": "Motion stays coherent"},
        ]

        for i, miner in enumerate(valid_miners):
            await integration_evaluation_store.save_sample(
                validator_hotkey=validator_hotkey,
                task_id=i + 1,
                miner_hotkey=miner.miner_hotkey,
                s3_bucket="validator-bucket",
                s3_prefix=f"tasks/evaluator-flow-{i + 1}",
                passed=evaluation_results[i]["passed"],
                confidence=evaluation_results[i]["confidence"],
                reasoning=evaluation_results[i]["reasoning"],
                prompt="A serene forest path with sunlight",
            )

        evaluations = await integration_evaluation_store.get_samples_by_validator(
            validator_hotkey
        )
        assert len(evaluations) == 3

        stats = await integration_evaluation_store.get_miner_stats_by_validator(
            validator_hotkey
        )
        assert len(stats) == 3

        # Check pass counts (2 out of 3)
        total_passed = sum(s["passed_count"] for s in stats.values())
        assert total_passed == 2

    async def test_batch_task_evaluation_submission(
        self,
        integration_evaluation_store,
        test_hotkeys,
    ):
        """Persist a batch of task evaluations for one validator."""
        validator_hotkey = test_hotkeys[0]

        batch_size = 10
        for i in range(batch_size):
            miner_idx = i % 3 + 1  # Rotate through miners
            await integration_evaluation_store.save_sample(
                validator_hotkey=validator_hotkey,
                task_id=i + 1,
                miner_hotkey=test_hotkeys[miner_idx],
                s3_bucket="test-bucket",
                s3_prefix=f"tasks/batch-{i + 1}",
                passed=i % 2 == 0,
                confidence=80 + (i % 10),
            )

        count = await integration_evaluation_store.get_sample_count_by_validator(
            validator_hotkey
        )
        assert count == batch_size


class TestScoreAggregation:
    """Tests for score aggregation from persisted evaluations."""

    async def test_aggregate_scores_multiple_validators(
        self,
        integration_participant_store,
        integration_evaluation_store,
        integration_rank_store,
        test_hotkeys,
    ):
        """Test score aggregation across multiple validators."""
        miner_hotkey = test_hotkeys[0]
        validator_hotkeys = test_hotkeys[1:4]

        # Create miner
        await integration_participant_store.save_miner(
            uid=0,
            miner_hotkey=miner_hotkey,
            model_name="user/leoma-video-model",
            is_valid=True,
        )

        # Each validator submits samples and we calculate scores
        for i, validator_hotkey in enumerate(validator_hotkeys):
            # Submit samples with varying pass rates
            samples_count = 10
            pass_target = 5 + i  # Different pass rates per validator

            for j in range(samples_count):
                await integration_evaluation_store.save_sample(
                    validator_hotkey=validator_hotkey,
                    task_id=(i * 100) + j + 1,
                    miner_hotkey=miner_hotkey,
                    s3_bucket="test-bucket",
                    s3_prefix=f"tasks/aggregate-{i}-{j}",
                    passed=j < pass_target,
                    confidence=85,
                )

            # Calculate and save scores for this validator
            stats = await integration_evaluation_store.get_miner_stats_by_validator(
                validator_hotkey
            )

            for m_hotkey, m_stats in stats.items():
                await integration_rank_store.save_score(
                    miner_hotkey=m_hotkey,
                    validator_hotkey=validator_hotkey,
                    score=m_stats["pass_rate"],
                    total_samples=m_stats["total"],
                    total_passed=m_stats["passed_count"],
                    pass_rate=m_stats["pass_rate"],
                )

        # Get aggregated scores across all validators
        aggregated = await integration_rank_store.get_aggregated_scores()

        assert miner_hotkey in aggregated
        assert aggregated[miner_hotkey]["validator_count"] == 3
        assert aggregated[miner_hotkey]["total_samples"] == 30  # 10 * 3 validators

    async def test_score_calculation_from_task_evaluations(
        self,
        integration_participant_store,
        integration_evaluation_store,
        integration_rank_store,
        test_hotkeys,
    ):
        """Derive pass-rate scores from task evaluation rows."""
        validator_hotkey = test_hotkeys[0]
        miner_hotkeys = test_hotkeys[1:4]

        # Setup miners
        for i, hotkey in enumerate(miner_hotkeys):
            await integration_participant_store.save_miner(
                uid=i,
                miner_hotkey=hotkey,
                is_valid=True,
            )

        # Create samples with different pass patterns:
        # Miner 0: Always passed_count (5/5)
        # Miner 1: Alternates (3/5 or 2/5)
        # Miner 2: Always loses (0/5)
        for task_group in range(5):
            results = [
                (miner_hotkeys[0], True),
                (miner_hotkeys[1], task_group % 2 == 0),
                (miner_hotkeys[2], False),
            ]

            for result_index, (miner_hotkey, passed) in enumerate(results):
                await integration_evaluation_store.save_sample(
                    validator_hotkey=validator_hotkey,
                    task_id=(task_group * 10) + result_index + 1,
                    miner_hotkey=miner_hotkey,
                    s3_bucket="test-bucket",
                    s3_prefix=f"tasks/score-{task_group + 1}",
                    passed=passed,
                    confidence=80,
                )

        # Calculate scores
        stats = await integration_evaluation_store.get_miner_stats_by_validator(
            validator_hotkey
        )

        # Verify stats
        assert stats[miner_hotkeys[0]]["pass_rate"] == 1.0  # 5/5
        assert stats[miner_hotkeys[1]]["pass_rate"] == 0.6  # 3/5
        assert stats[miner_hotkeys[2]]["pass_rate"] == 0.0  # 0/5

        # Save scores
        for miner_hotkey, miner_stats in stats.items():
            await integration_rank_store.save_score(
                miner_hotkey=miner_hotkey,
                validator_hotkey=validator_hotkey,
                score=miner_stats["pass_rate"],
                total_samples=miner_stats["total"],
                total_passed=miner_stats["passed_count"],
                pass_rate=miner_stats["pass_rate"],
            )

        scores = await integration_rank_store.get_scores_by_validator(validator_hotkey)
        top_ranked = max(scores, key=lambda s: s.pass_rate)
        assert top_ranked.miner_hotkey == miner_hotkeys[0]
        assert top_ranked.pass_rate == 1.0


class TestBlacklistInFlow:
    """Tests for blacklist integration in validator flow."""

    async def test_blacklist_filters_miners(
        self,
        integration_participant_store,
        integration_blacklist_store,
        test_hotkeys,
    ):
        """Test blacklist integration in the flow."""
        miner_hotkeys = test_hotkeys[:3]

        # Create miners
        for i, hotkey in enumerate(miner_hotkeys):
            await integration_participant_store.save_miner(
                uid=i,
                miner_hotkey=hotkey,
                is_valid=True,
            )

        # Blacklist one miner
        await integration_blacklist_store.add(
            hotkey=miner_hotkeys[1],
            reason="Copied model weights",
        )

        # Get valid miners and filter by blacklist
        valid_miners = await integration_participant_store.get_valid_miners()
        blacklisted = await integration_blacklist_store.get_hotkeys()

        active_miners = [
            m for m in valid_miners
            if m.miner_hotkey not in blacklisted
        ]

        assert len(active_miners) == 2
        assert miner_hotkeys[1] not in [m.miner_hotkey for m in active_miners]

    async def test_blacklist_reason_tracking(
        self,
        integration_blacklist_store,
        test_hotkeys,
    ):
        """Test that blacklist reasons are properly tracked."""
        hotkey = test_hotkeys[0]
        reason = "Detected copied model weights from another miner"
        admin_hotkey = test_hotkeys[1]

        # Add with reason
        entry = await integration_blacklist_store.add(
            hotkey=hotkey,
            reason=reason,
            added_by=admin_hotkey,
        )

        # Retrieve and verify
        retrieved = await integration_blacklist_store.get(hotkey)
        assert retrieved.reason == reason
        assert retrieved.added_by == admin_hotkey


class TestMinerValidation:
    """Tests for miner validation state management."""

    async def test_miner_validation_lifecycle(
        self,
        integration_participant_store,
        test_hotkeys,
    ):
        """Test miner validation state changes."""
        hotkey = test_hotkeys[0]

        # Create miner as initially invalid
        await integration_participant_store.save_miner(
            uid=0,
            miner_hotkey=hotkey,
            model_name="user/leoma-video-model",
            is_valid=False,
            invalid_reason="Pending verification",
        )

        # Verify initial state
        miner = await integration_participant_store.get_miner_by_hotkey(hotkey)
        assert miner.is_valid is False
        assert miner.invalid_reason == "Pending verification"

        # Validate miner
        await integration_participant_store.set_validation_status(
            uid=0,
            is_valid=True,
            invalid_reason=None,
        )

        # Verify updated state
        miner = await integration_participant_store.get_miner_by_hotkey(hotkey)
        assert miner.is_valid is True
        assert miner.invalid_reason is None

        # Invalidate miner
        await integration_participant_store.set_validation_status(
            uid=0,
            is_valid=False,
            invalid_reason="Model hash mismatch",
        )

        # Verify final state
        miner = await integration_participant_store.get_miner_by_hotkey(hotkey)
        assert miner.is_valid is False
        assert miner.invalid_reason == "Model hash mismatch"

    async def test_batch_miner_upsert(
        self,
        integration_participant_store,
        test_hotkeys,
    ):
        """Test batch miner upsert from metagraph sync."""
        miners_data = [
            {
                "uid": i,
                "miner_hotkey": hotkey,
                "model_name": f"user/model-{i}",
                "model_revision": f"rev{i}abc",
                "chute_id": f"chute-{i}",
                "chute_slug": f"chutes-model-{i}",
                "is_valid": True,
            }
            for i, hotkey in enumerate(test_hotkeys[:5])
        ]

        # Batch upsert
        count = await integration_participant_store.batch_upsert_miners(miners_data)
        assert count == 5

        # Verify all miners created
        all_miners = await integration_participant_store.get_all_miners()
        assert len(all_miners) == 5

        # Verify counts
        valid_count = await integration_participant_store.get_valid_count()
        assert valid_count == 5
