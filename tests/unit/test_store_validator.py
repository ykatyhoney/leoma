"""
Unit tests for ValidatorStore.

Tests CRUD operations for validator records using SQLite in-memory database.
"""

from leoma.infra.db.stores.store_validator import ValidatorStore
from leoma.infra.db.tables import Validator


async def _save_validators_with_stakes(
    validator_store: ValidatorStore,
    test_hotkeys: list[str],
    stakes: list[float],
) -> None:
    """Create validators with corresponding stake values."""
    for uid, (hotkey, stake) in enumerate(zip(test_hotkeys, stakes)):
        await validator_store.save_validator(uid=uid, hotkey=hotkey, stake=stake)


class TestValidatorStoreSaveValidator:
    """Tests for save_validator method."""

    async def test_save_validator_creates_record(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test that save_validator creates a new validator record."""
        validator = await validator_store.save_validator(
            uid=0,
            hotkey=test_hotkeys[0],
            stake=10000.0,
            s3_bucket="test-bucket",
        )

        assert validator is not None
        assert validator.uid == 0
        assert validator.hotkey == test_hotkeys[0]
        assert validator.stake == 10000.0
        assert validator.s3_bucket == "test-bucket"
        assert validator.last_seen_at is not None

    async def test_save_validator_updates_existing(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test that save_validator updates existing validator record."""
        # Create initial validator
        await validator_store.save_validator(
            uid=0,
            hotkey=test_hotkeys[0],
            stake=5000.0,
        )

        # Update the validator
        updated = await validator_store.save_validator(
            uid=0,
            hotkey=test_hotkeys[0],
            stake=15000.0,
            s3_bucket="new-bucket",
        )

        assert updated.uid == 0
        assert updated.hotkey == test_hotkeys[0]
        assert updated.stake == 15000.0
        assert updated.s3_bucket == "new-bucket"

    async def test_save_validator_without_s3_bucket(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test saving validator without S3 bucket."""
        validator = await validator_store.save_validator(
            uid=0,
            hotkey=test_hotkeys[0],
            stake=10000.0,
        )

        assert validator is not None
        assert validator.s3_bucket is None

    async def test_save_validator_updates_last_seen(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test that save_validator updates last_seen_at timestamp."""
        # Create initial validator
        initial = await validator_store.save_validator(
            uid=0,
            hotkey=test_hotkeys[0],
            stake=10000.0,
        )
        initial_time = initial.last_seen_at

        # Update the validator
        updated = await validator_store.save_validator(
            uid=0,
            hotkey=test_hotkeys[0],
            stake=10000.0,
        )

        # last_seen_at should be updated (or same if very fast)
        assert updated.last_seen_at is not None
        assert updated.last_seen_at >= initial_time


class TestValidatorStoreGetValidator:
    """Tests for get_validator_by_uid and get_validator_by_hotkey methods."""

    async def test_get_validator_by_uid(
        self,
        validator_store: ValidatorStore,
        db_with_validators: list[Validator],
    ):
        """Test retrieving validator by UID."""
        validator = await validator_store.get_validator_by_uid(0)

        assert validator is not None
        assert validator.uid == 0

    async def test_get_validator_by_uid_not_found(
        self,
        validator_store: ValidatorStore,
    ):
        """Test get_validator_by_uid returns None for non-existent UID."""
        validator = await validator_store.get_validator_by_uid(999)

        assert validator is None

    async def test_get_validator_by_hotkey(
        self,
        validator_store: ValidatorStore,
        db_with_validators: list[Validator],
    ):
        """Test retrieving validator by hotkey."""
        expected_hotkey = db_with_validators[0].hotkey
        validator = await validator_store.get_validator_by_hotkey(expected_hotkey)

        assert validator is not None
        assert validator.hotkey == expected_hotkey

    async def test_get_validator_by_hotkey_not_found(
        self,
        validator_store: ValidatorStore,
    ):
        """Test get_validator_by_hotkey returns None for unknown hotkey."""
        validator = await validator_store.get_validator_by_hotkey("unknown-hotkey")

        assert validator is None


class TestValidatorStoreGetAllValidators:
    """Tests for get_all_validators method."""

    async def test_get_all_validators(
        self,
        validator_store: ValidatorStore,
        db_with_validators: list[Validator],
    ):
        """Test that get_all_validators returns all validators."""
        validators = await validator_store.get_all_validators()

        assert len(validators) == len(db_with_validators)

    async def test_get_all_validators_ordered_by_uid(
        self,
        validator_store: ValidatorStore,
        db_with_validators: list[Validator],
    ):
        """Test that validators are ordered by UID."""
        validators = await validator_store.get_all_validators()

        uids = [v.uid for v in validators]
        assert uids == sorted(uids)

    async def test_get_all_validators_empty(
        self,
        validator_store: ValidatorStore,
    ):
        """Test get_all_validators on empty database."""
        validators = await validator_store.get_all_validators()

        assert validators == []


class TestValidatorStoreUpdateLastSeen:
    """Tests for update_last_seen method."""

    async def test_update_last_seen(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test updating validator's last seen timestamp."""
        # Create validator
        await validator_store.save_validator(
            uid=0,
            hotkey=test_hotkeys[0],
            stake=10000.0,
        )

        # Update last seen
        result = await validator_store.update_last_seen(test_hotkeys[0])

        assert result is True

        # Verify timestamp was updated
        validator = await validator_store.get_validator_by_hotkey(test_hotkeys[0])
        assert validator.last_seen_at is not None

    async def test_update_last_seen_not_found(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test update_last_seen returns False for unknown hotkey."""
        result = await validator_store.update_last_seen(test_hotkeys[0])

        assert result is False


class TestValidatorStoreUpdateStake:
    """Tests for update_stake method."""

    async def test_update_stake(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test updating validator's stake."""
        # Create validator with initial stake
        await validator_store.save_validator(
            uid=0,
            hotkey=test_hotkeys[0],
            stake=5000.0,
        )

        # Update stake
        result = await validator_store.update_stake(test_hotkeys[0], 25000.0)

        assert result is True

        # Verify stake was updated
        validator = await validator_store.get_validator_by_hotkey(test_hotkeys[0])
        assert validator.stake == 25000.0

    async def test_update_stake_not_found(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test update_stake returns False for unknown hotkey."""
        result = await validator_store.update_stake(test_hotkeys[0], 25000.0)

        assert result is False


class TestValidatorStoreGetValidatorCount:
    """Tests for get_validator_count method."""

    async def test_get_validator_count(
        self,
        validator_store: ValidatorStore,
        db_with_validators: list[Validator],
    ):
        """Test counting validators."""
        count = await validator_store.get_validator_count()

        assert count == len(db_with_validators)

    async def test_get_validator_count_empty(
        self,
        validator_store: ValidatorStore,
    ):
        """Test count on empty database."""
        count = await validator_store.get_validator_count()

        assert count == 0


class TestValidatorStoreGetValidatorsByStake:
    """Tests for get_validators_by_stake method."""

    async def test_get_validators_by_stake_no_minimum(
        self,
        validator_store: ValidatorStore,
        db_with_validators: list[Validator],
    ):
        """Test getting all validators with no minimum stake."""
        validators = await validator_store.get_validators_by_stake(min_stake=0.0)

        assert len(validators) == len(db_with_validators)

    async def test_get_validators_by_stake_with_minimum(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test getting validators with minimum stake threshold."""
        # Create validators with different stakes
        await _save_validators_with_stakes(
            validator_store=validator_store,
            test_hotkeys=test_hotkeys,
            stakes=[5000.0, 15000.0, 25000.0],
        )

        # Get validators with stake >= 10000
        validators = await validator_store.get_validators_by_stake(min_stake=10000.0)

        assert len(validators) == 2
        stakes = [v.stake for v in validators]
        assert all(s >= 10000.0 for s in stakes)

    async def test_get_validators_by_stake_ordered_descending(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test validators are ordered by stake descending."""
        # Create validators with different stakes
        await _save_validators_with_stakes(
            validator_store=validator_store,
            test_hotkeys=test_hotkeys,
            stakes=[5000.0, 25000.0, 15000.0],
        )

        validators = await validator_store.get_validators_by_stake()

        stakes = [v.stake for v in validators]
        assert stakes == sorted(stakes, reverse=True)

    async def test_get_validators_by_stake_high_minimum(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test getting validators with very high minimum stake."""
        # Create validators
        await _save_validators_with_stakes(
            validator_store=validator_store,
            test_hotkeys=test_hotkeys,
            stakes=[5000.0, 10000.0],
        )

        # Get validators with stake >= 100000 (none should match)
        validators = await validator_store.get_validators_by_stake(min_stake=100000.0)

        assert len(validators) == 0


class TestValidatorStoreDeleteValidatorsExceptUids:
    """Tests for delete_validators_except_uids method."""

    async def test_delete_validators_except_uids_removes_stale(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test that validators not in the given uid set are deleted."""
        await _save_validators_with_stakes(
            validator_store=validator_store,
            test_hotkeys=test_hotkeys[:3],
            stakes=[1000.0, 2000.0, 3000.0],
        )
        keep_uids = {0, 2}
        deleted = await validator_store.delete_validators_except_uids(keep_uids)
        assert deleted == 1
        all_v = await validator_store.get_all_validators()
        assert len(all_v) == 2
        uids = {v.uid for v in all_v}
        assert uids == keep_uids

    async def test_delete_validators_except_uids_empty_set(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        """Test delete_validators_except_uids with empty set returns 0 and deletes nothing."""
        await validator_store.save_validator(uid=0, hotkey=test_hotkeys[0], stake=1000.0)
        deleted = await validator_store.delete_validators_except_uids(set())
        assert deleted == 0
        all_v = await validator_store.get_all_validators()
        assert len(all_v) == 1


class TestValidatorStoreDeleteByHotkey:
    """Tests for delete_validator_by_hotkey (owner-managed removal)."""

    async def test_delete_validator_by_hotkey(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        await validator_store.save_validator(uid=0, hotkey=test_hotkeys[0], stake=1.0)
        await validator_store.save_validator(uid=1, hotkey=test_hotkeys[1], stake=2.0)

        assert await validator_store.delete_validator_by_hotkey(test_hotkeys[0]) is True
        remaining = await validator_store.get_all_validators()
        assert [v.hotkey for v in remaining] == [test_hotkeys[1]]

    async def test_delete_validator_by_hotkey_absent_returns_false(
        self,
        validator_store: ValidatorStore,
        test_hotkeys: list[str],
    ):
        assert await validator_store.delete_validator_by_hotkey(test_hotkeys[0]) is False
