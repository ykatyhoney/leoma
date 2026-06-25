"""Unit tests for validator evaluation API routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from leoma.delivery.http.routes.samples import router, MAX_BATCH_SIZE
from leoma.infra.db.tables import ValidatorSample


# Module-local SS58-like identities for evaluation route tests.
EVALUATION_VALIDATOR_HOTKEY = "5C62W7ELLAAfjCQeBU3me9nTXXqjVwN4kQY8w8gM9nJ8K4pL"
EVALUATION_MINER_HOTKEY = "5D9KxqM4nTa8cJrL2WvY6hP3sFzU1bN7eQmR5tHkC8yLpXaZ"


@pytest.fixture
def app():
    """Create a test FastAPI application with the evaluation router."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/samples")
    return test_app


@pytest.fixture
def mock_auth(app):
    """Mock the authentication dependency to always succeed."""
    from leoma.delivery.http.verifier import verify_signature
    
    async def _mock_verify_signature():
        return EVALUATION_VALIDATOR_HOTKEY

    app.dependency_overrides[verify_signature] = _mock_verify_signature
    yield EVALUATION_VALIDATOR_HOTKEY
    app.dependency_overrides.clear()


@pytest.fixture
def evaluation_submission():
    """Create a validator evaluation payload."""
    return {
        "task_id": 1,
        "miner_hotkey": EVALUATION_MINER_HOTKEY,
        "prompt": "A skateboarder crossing an empty plaza",
        "s3_bucket": "test-bucket",
        "s3_prefix": "tasks/1",
        "passed": True,
        "confidence": 85,
        "reasoning": "The generated clip matches the prompt and stays coherent.",
    }


@pytest.fixture
def stored_evaluation():
    """Create a stored validator evaluation record."""
    return ValidatorSample(
        id=1,
        validator_hotkey=EVALUATION_VALIDATOR_HOTKEY,
        task_id=1,
        miner_hotkey=EVALUATION_MINER_HOTKEY,
        prompt="A skateboarder crossing an empty plaza",
        s3_bucket="test-bucket",
        s3_prefix="tasks/1",
        passed=True,
        confidence=85,
        reasoning="The generated clip matches the prompt and stays coherent.",
        evaluated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def stored_evaluations(test_hotkeys):
    """Create a list of stored validator evaluation records."""
    evaluations = []
    for i in range(5):
        evaluations.append(ValidatorSample(
            id=i + 1,
            validator_hotkey=EVALUATION_VALIDATOR_HOTKEY,
            task_id=i + 1,
            miner_hotkey=test_hotkeys[i % len(test_hotkeys)],
            prompt=f"Prompt {i}",
            s3_bucket="test-bucket",
            s3_prefix=f"tasks/{i + 1}",
            passed=(i % 2 == 0),
            confidence=80 + i,
            reasoning=f"Evaluation reasoning {i}",
            evaluated_at=datetime.now(timezone.utc),
        ))
    return evaluations


class TestSubmitEvaluation:
    """Tests for `POST /samples`."""

    async def test_submit_evaluation_success(
        self,
        app: FastAPI,
        mock_auth,
        evaluation_submission: dict,
        stored_evaluation: ValidatorSample,
        monkeypatch,
    ):
        """Test successful evaluation submission."""
        mock_dao = MagicMock()
        mock_dao.save_sample = AsyncMock(return_value=stored_evaluation)
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/samples", json=evaluation_submission)

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == stored_evaluation.task_id
        assert data["miner_hotkey"] == stored_evaluation.miner_hotkey
        assert data["passed"] == stored_evaluation.passed
        assert data["confidence"] == stored_evaluation.confidence

    async def test_submit_evaluation_calls_store_with_expected_fields(
        self,
        app: FastAPI,
        mock_auth,
        evaluation_submission: dict,
        stored_evaluation: ValidatorSample,
        monkeypatch,
    ):
        """Test that the route forwards the evaluation payload unchanged."""
        mock_dao = MagicMock()
        mock_dao.save_sample = AsyncMock(return_value=stored_evaluation)
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            await client.post("/samples", json=evaluation_submission)

        mock_dao.save_sample.assert_called_once()
        call_kwargs = mock_dao.save_sample.call_args.kwargs
        assert call_kwargs["validator_hotkey"] == EVALUATION_VALIDATOR_HOTKEY
        assert call_kwargs["task_id"] == evaluation_submission["task_id"]
        assert call_kwargs["miner_hotkey"] == evaluation_submission["miner_hotkey"]
        assert call_kwargs["passed"] == evaluation_submission["passed"]

    async def test_submit_evaluation_with_required_fields_only(
        self,
        app: FastAPI,
        mock_auth,
        stored_evaluation: ValidatorSample,
        monkeypatch,
    ):
        """Test evaluation submission with only required fields."""
        minimal_evaluation = {
            "task_id": 1,
            "miner_hotkey": EVALUATION_MINER_HOTKEY,
            "prompt": "A short benchmark prompt",
            "s3_bucket": "test-bucket",
            "s3_prefix": "tasks/1",
            "passed": False,
        }

        stored_evaluation.passed = False
        stored_evaluation.confidence = None
        stored_evaluation.reasoning = None

        mock_dao = MagicMock()
        mock_dao.save_sample = AsyncMock(return_value=stored_evaluation)
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/samples", json=minimal_evaluation)

        assert response.status_code == 200

    async def test_submit_evaluation_rejects_invalid_miner_hotkey(
        self,
        app: FastAPI,
        mock_auth,
        evaluation_submission: dict,
        monkeypatch,
    ):
        """Test validation error for an invalid miner hotkey."""
        evaluation_submission["miner_hotkey"] = "invalid-hotkey"

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/samples", json=evaluation_submission)

        assert response.status_code == 422  # Validation error


class TestSubmitEvaluationBatch:
    """Tests for `POST /samples/batch`."""

    async def test_submit_evaluation_batch_success(
        self,
        app: FastAPI,
        mock_auth,
        stored_evaluations: list[ValidatorSample],
        test_hotkeys: list[str],
        monkeypatch,
    ):
        """Test successful batch evaluation submission."""
        batch = [
            {
                "task_id": i + 1,
                "miner_hotkey": test_hotkeys[i % len(test_hotkeys)],
                "prompt": f"Prompt {i}",
                "s3_bucket": "test-bucket",
                "s3_prefix": f"tasks/{i + 1}",
                "passed": i % 2 == 0,
            }
            for i in range(3)
        ]

        # Mock DAO to return samples
        call_count = 0
        async def mock_save(*args, **kwargs):
            nonlocal call_count
            result = stored_evaluations[call_count % len(stored_evaluations)]
            call_count += 1
            return result

        mock_dao = MagicMock()
        mock_dao.save_sample = AsyncMock(side_effect=mock_save)
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/samples/batch", json={"samples": batch})

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    async def test_submit_evaluation_batch_rejects_oversized_request(
        self,
        app: FastAPI,
        mock_auth,
        test_hotkeys: list[str],
    ):
        """Test batch submission exceeding the maximum size."""
        batch = [
            {
                "task_id": i + 1,
                "miner_hotkey": test_hotkeys[0],
                "prompt": f"Prompt {i}",
                "s3_bucket": "test-bucket",
                "s3_prefix": f"tasks/{i + 1}",
                "passed": True,
            }
            for i in range(MAX_BATCH_SIZE + 1)
        ]

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/samples/batch", json={"samples": batch})

        assert response.status_code == 400
        assert "exceeds limit" in response.json()["detail"].lower()

    async def test_submit_evaluation_batch_accepts_empty_list(
        self,
        app: FastAPI,
        mock_auth,
        monkeypatch,
    ):
        """Test submitting an empty batch."""
        mock_dao = MagicMock()
        mock_dao.save_sample = AsyncMock()
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/samples/batch", json={"samples": []})

        assert response.status_code == 200
        assert response.json() == []

    async def test_submit_evaluation_batch_accepts_maximum_size(
        self,
        app: FastAPI,
        mock_auth,
        test_hotkeys: list[str],
        stored_evaluations: list[ValidatorSample],
        monkeypatch,
    ):
        """Test batch submission at exactly the maximum size."""
        batch = [
            {
                "task_id": i + 1,
                "miner_hotkey": test_hotkeys[i % len(test_hotkeys)],
                "prompt": f"Prompt {i}",
                "s3_bucket": "test-bucket",
                "s3_prefix": f"tasks/{i + 1}",
                "passed": True,
            }
            for i in range(MAX_BATCH_SIZE)
        ]

        mock_dao = MagicMock()
        mock_dao.save_sample = AsyncMock(return_value=stored_evaluations[0])
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post("/samples/batch", json={"samples": batch})

        assert response.status_code == 200
        assert len(response.json()) == MAX_BATCH_SIZE


class TestGetValidatorEvaluations:
    """Tests for `GET /samples/validator/{validator_hotkey}`."""

    async def test_get_validator_evaluations_success(
        self,
        app: FastAPI,
        mock_auth,
        stored_evaluations: list[ValidatorSample],
        monkeypatch,
    ):
        """Test successful retrieval of validator evaluations."""
        mock_dao = MagicMock()
        mock_dao.get_samples_by_validator = AsyncMock(return_value=stored_evaluations)
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(f"/samples/validator/{EVALUATION_VALIDATOR_HOTKEY}")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == len(stored_evaluations)

    async def test_get_validator_evaluations_respect_limit(
        self,
        app: FastAPI,
        mock_auth,
        stored_evaluations: list[ValidatorSample],
        monkeypatch,
    ):
        """Test retrieval with a custom limit."""
        mock_dao = MagicMock()
        mock_dao.get_samples_by_validator = AsyncMock(return_value=stored_evaluations[:2])
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/samples/validator/{EVALUATION_VALIDATOR_HOTKEY}",
                params={"limit": 2},
            )

        assert response.status_code == 200
        mock_dao.get_samples_by_validator.assert_called_once_with(
            validator_hotkey=EVALUATION_VALIDATOR_HOTKEY,
            limit=2,
        )

    async def test_get_validator_evaluations_empty(
        self,
        app: FastAPI,
        mock_auth,
        monkeypatch,
    ):
        """Test retrieval when no evaluations exist."""
        mock_dao = MagicMock()
        mock_dao.get_samples_by_validator = AsyncMock(return_value=[])
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(f"/samples/validator/{EVALUATION_VALIDATOR_HOTKEY}")

        assert response.status_code == 200
        assert response.json() == []


class TestGetMinerEvaluations:
    """Tests for `GET /samples/miner/{miner_hotkey}`."""

    async def test_get_miner_evaluations_success(
        self,
        app: FastAPI,
        mock_auth,
        stored_evaluations: list[ValidatorSample],
        monkeypatch,
    ):
        """Test successful retrieval of miner evaluations."""
        mock_dao = MagicMock()
        mock_dao.get_samples_by_miner = AsyncMock(return_value=stored_evaluations)
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(f"/samples/miner/{EVALUATION_MINER_HOTKEY}")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == len(stored_evaluations)

    async def test_get_miner_evaluations_respect_limit(
        self,
        app: FastAPI,
        mock_auth,
        stored_evaluations: list[ValidatorSample],
        monkeypatch,
    ):
        """Test retrieval with a custom limit."""
        mock_dao = MagicMock()
        mock_dao.get_samples_by_miner = AsyncMock(return_value=stored_evaluations[:3])
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/samples/miner/{EVALUATION_MINER_HOTKEY}",
                params={"limit": 3},
            )

        assert response.status_code == 200
        mock_dao.get_samples_by_miner.assert_called_once_with(
            miner_hotkey=EVALUATION_MINER_HOTKEY,
            limit=3,
        )

    async def test_get_miner_evaluations_response_shape(
        self,
        app: FastAPI,
        mock_auth,
        stored_evaluation: ValidatorSample,
        monkeypatch,
    ):
        """Test response contains the expected evaluation fields."""
        mock_dao = MagicMock()
        mock_dao.get_samples_by_miner = AsyncMock(return_value=[stored_evaluation])
        monkeypatch.setattr("leoma.delivery.http.routes.samples.validator_samples_dao", mock_dao)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(f"/samples/miner/{EVALUATION_MINER_HOTKEY}")

        assert response.status_code == 200
        evaluation_data = response.json()[0]

        # Verify all expected fields
        assert evaluation_data["id"] == stored_evaluation.id
        assert evaluation_data["task_id"] == stored_evaluation.task_id
        assert evaluation_data["miner_hotkey"] == stored_evaluation.miner_hotkey
        assert evaluation_data["prompt"] == stored_evaluation.prompt
        # s3_bucket/s3_prefix are intentionally NOT exposed in the API response (operational/infra).
        assert "s3_bucket" not in evaluation_data
        assert "s3_prefix" not in evaluation_data
        assert evaluation_data["passed"] == stored_evaluation.passed
        assert evaluation_data["confidence"] == stored_evaluation.confidence
        assert evaluation_data["reasoning"] == stored_evaluation.reasoning


class TestEvaluationAuthentication:
    """Tests for authentication on evaluation endpoints."""

    async def test_submit_evaluation_requires_auth(
        self,
        app: FastAPI,
        evaluation_submission: dict,
    ):
        """Test POST /samples requires authentication."""
        from fastapi import HTTPException
        from leoma.delivery.http.verifier import verify_signature

        async def _mock_verify_fail():
            raise HTTPException(status_code=401, detail="Authentication failed")

        app.dependency_overrides[verify_signature] = _mock_verify_fail

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post("/samples", json=evaluation_submission)

            assert response.status_code == 401
        finally:
            app.dependency_overrides.clear()

    async def test_submit_evaluation_batch_requires_auth(
        self,
        app: FastAPI,
        evaluation_submission: dict,
    ):
        """Test POST /samples/batch requires authentication."""
        from fastapi import HTTPException
        from leoma.delivery.http.verifier import verify_signature

        async def _mock_verify_fail():
            raise HTTPException(status_code=401, detail="Authentication failed")

        app.dependency_overrides[verify_signature] = _mock_verify_fail

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post("/samples/batch", json=[evaluation_submission])

            assert response.status_code == 401
        finally:
            app.dependency_overrides.clear()

    async def test_get_validator_evaluations_requires_auth(
        self,
        app: FastAPI,
    ):
        """Test GET /samples/validator/{hotkey} requires authentication."""
        from fastapi import HTTPException
        from leoma.delivery.http.verifier import verify_signature

        async def _mock_verify_fail():
            raise HTTPException(status_code=401, detail="Authentication failed")

        app.dependency_overrides[verify_signature] = _mock_verify_fail

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get(f"/samples/validator/{EVALUATION_VALIDATOR_HOTKEY}")

            assert response.status_code == 401
        finally:
            app.dependency_overrides.clear()

    async def test_get_miner_evaluations_requires_auth(
        self,
        app: FastAPI,
    ):
        """Test GET /samples/miner/{hotkey} requires authentication."""
        from fastapi import HTTPException
        from leoma.delivery.http.verifier import verify_signature

        async def _mock_verify_fail():
            raise HTTPException(status_code=401, detail="Authentication failed")

        app.dependency_overrides[verify_signature] = _mock_verify_fail

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get(f"/samples/miner/{EVALUATION_MINER_HOTKEY}")

            assert response.status_code == 401
        finally:
            app.dependency_overrides.clear()
