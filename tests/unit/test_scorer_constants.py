"""Tests for scorer window helpers."""

from leoma.infra.scorer_constants import required_distinct_validators


class TestRequiredDistinctValidators:
    def test_majority_of_active(self):
        assert required_distinct_validators(4) == 2  # ceil(4*0.5)
        assert required_distinct_validators(5) == 3  # ceil(5*0.5)
        assert required_distinct_validators(3) == 2  # ceil(3*0.5)

    def test_floored_at_one(self):
        # A single live validator (or none) must never deadlock the gate.
        assert required_distinct_validators(1) == 1
        assert required_distinct_validators(0) == 1
        assert required_distinct_validators(-3) == 1

    def test_two_active_requires_both(self):
        assert required_distinct_validators(2) == 1  # ceil(2*0.5)=1 -> gate effectively off at 2
