"""
Unit tests for API authentication.

Tests signature verification, admin checks, and FastAPI authentication dependencies.
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException

from leoma.delivery.http.verifier import (
    SignatureVerifier,
    sign_message,
    verify_signature,
    verify_admin_signature,
    get_verifier,
    SIGNATURE_EXPIRY_SECONDS,
)


# Static SS58-like identities used only inside this test module.
NETWORK_VALIDATOR_HOTKEY = "5C62W7ELLAAfjCQeBU3me9nTXXqjVwN4kQY8w8gM9nJ8K4pL"
ROOT_ADMIN_HOTKEY = "5D9KxqM4nTa8cJrL2WvY6hP3sFzU1bN7eQmR5tHkC8yLpXaZ"


class TestSignatureVerifierCreateMessage:
    """Tests for _create_message method."""

    def test_create_message_consistent(self):
        """Test that message creation is consistent."""
        verifier = SignatureVerifier()
        body = b'{"test": "data"}'
        timestamp = "1700000000"

        message1 = verifier._create_message(body, timestamp)
        message2 = verifier._create_message(body, timestamp)

        assert message1 == message2

    def test_create_message_different_body(self):
        """Test different bodies produce different messages."""
        verifier = SignatureVerifier()
        timestamp = "1700000000"

        message1 = verifier._create_message(b'{"test": "data1"}', timestamp)
        message2 = verifier._create_message(b'{"test": "data2"}', timestamp)

        assert message1 != message2

    def test_create_message_different_timestamp(self):
        """Test different timestamps produce different messages."""
        verifier = SignatureVerifier()
        body = b'{"test": "data"}'

        message1 = verifier._create_message(body, "1700000000")
        message2 = verifier._create_message(body, "1700000001")

        assert message1 != message2


class TestSignatureVerifierIsAdmin:
    """Tests for is_admin method."""

    def test_is_admin_returns_false_for_non_admin(self):
        """Test is_admin returns False for non-admin hotkey."""
        verifier = SignatureVerifier()

        result = verifier.is_admin(NETWORK_VALIDATOR_HOTKEY)

        assert result is False

    def test_is_admin_returns_false_for_empty_string(self):
        """Test is_admin returns False for empty string."""
        verifier = SignatureVerifier()

        result = verifier.is_admin("")

        assert result is False

    def test_is_admin_returns_true_for_admin(self, monkeypatch):
        """Test is_admin returns True for admin hotkey."""
        monkeypatch.setattr(
            "leoma.delivery.http.verifier.ADMIN_HOTKEYS",
            [ROOT_ADMIN_HOTKEY],
        )
        verifier = SignatureVerifier()

        result = verifier.is_admin(ROOT_ADMIN_HOTKEY)

        assert result is True


class TestSignatureVerifierVerifyRequest:
    """Tests for verify_request method."""

    async def test_verify_request_expired_timestamp(self):
        """Test verify_request rejects expired timestamp."""
        verifier = SignatureVerifier()
        body = b'{"test": "data"}'
        # Use a timestamp from the distant past
        expired_timestamp = str(int(time.time()) - SIGNATURE_EXPIRY_SECONDS - 100)

        is_valid, error = await verifier.verify_request(
            body=body,
            hotkey=NETWORK_VALIDATOR_HOTKEY,
            signature="0x" + "00" * 64,
            timestamp=expired_timestamp,
        )

        assert is_valid is False
        assert "expired" in error.lower()

    async def test_verify_request_invalid_timestamp_format(self):
        """Test verify_request rejects invalid timestamp format."""
        verifier = SignatureVerifier()
        body = b'{"test": "data"}'

        is_valid, error = await verifier.verify_request(
            body=body,
            hotkey=NETWORK_VALIDATOR_HOTKEY,
            signature="0x" + "00" * 64,
            timestamp="not-a-number",
        )

        assert is_valid is False
        assert "invalid timestamp" in error.lower()

    async def test_verify_request_blacklisted_hotkey(self):
        """Test verify_request rejects blacklisted hotkey."""
        mock_store = MagicMock()
        mock_store.is_blacklisted = AsyncMock(return_value=True)

        verifier = SignatureVerifier()
        verifier.blacklist_store = mock_store

        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))

        is_valid, error = await verifier.verify_request(
            body=body,
            hotkey=NETWORK_VALIDATOR_HOTKEY,
            signature="0x" + "00" * 64,
            timestamp=timestamp,
        )

        assert is_valid is False
        assert "blacklisted" in error.lower()

    async def test_verify_request_invalid_signature(self):
        """Test verify_request rejects invalid signature."""
        mock_store = MagicMock()
        mock_store.is_blacklisted = AsyncMock(return_value=False)

        verifier = SignatureVerifier()
        verifier.blacklist_store = mock_store

        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))

        is_valid, error = await verifier.verify_request(
            body=body,
            hotkey=NETWORK_VALIDATOR_HOTKEY,
            signature="0x" + "00" * 64,  # Invalid signature
            timestamp=timestamp,
        )

        assert is_valid is False
        # Either "Invalid signature" or "verification failed"
        assert "signature" in error.lower() or "failed" in error.lower()


class TestSignatureVerifierWithRealKeypair:
    """Tests using real keypair for signature generation and verification."""

    @pytest.fixture
    def keypair(self):
        """Create a test keypair."""
        try:
            from substrateinterface import Keypair
            # Use Alice's well-known mnemonic for testing
            return Keypair.create_from_uri("//Alice")
        except ImportError:
            pytest.skip("substrateinterface not available")

    def _verifier(self, monkeypatch, allowlist, admins=()):
        monkeypatch.setattr("leoma.delivery.http.verifier.VALIDATOR_ALLOWLIST", list(allowlist))
        monkeypatch.setattr("leoma.delivery.http.verifier.ADMIN_HOTKEYS", list(admins))
        verifier = SignatureVerifier()
        mock_blacklist = MagicMock()
        mock_blacklist.is_blacklisted = AsyncMock(return_value=False)
        verifier.blacklist_store = mock_blacklist
        return verifier

    async def test_verify_request_with_valid_signature(self, keypair, monkeypatch):
        """verify_request accepts a valid signature when the hotkey is in the hardcoded allowlist."""
        verifier = self._verifier(monkeypatch, allowlist=[keypair.ss58_address])
        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))
        signature = sign_message(keypair, body, timestamp)

        is_valid, error = await verifier.verify_request(
            body=body, hotkey=keypair.ss58_address, signature=signature, timestamp=timestamp,
        )
        assert is_valid is True
        assert error is None

    async def test_verify_request_rejects_valid_signature_when_not_in_allowlist(self, keypair, monkeypatch):
        """A valid signature from a hotkey absent from the allowlist is rejected."""
        verifier = self._verifier(monkeypatch, allowlist=[])
        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))
        signature = sign_message(keypair, body, timestamp)

        is_valid, error = await verifier.verify_request(
            body=body, hotkey=keypair.ss58_address, signature=signature, timestamp=timestamp,
        )
        assert is_valid is False
        assert "Not a permissioned validator" in (error or "")

    async def test_verify_request_allows_admin_without_allowlist(self, keypair, monkeypatch):
        """An admin hotkey is accepted even when absent from the validator allowlist."""
        verifier = self._verifier(monkeypatch, allowlist=[], admins=[keypair.ss58_address])
        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))
        signature = sign_message(keypair, body, timestamp)

        is_valid, error = await verifier.verify_request(
            body=body, hotkey=keypair.ss58_address, signature=signature, timestamp=timestamp,
        )
        assert is_valid is True
        assert error is None


class TestSignMessage:
    """Tests for sign_message utility function."""

    @pytest.fixture
    def keypair(self):
        """Create a test keypair."""
        try:
            from substrateinterface import Keypair
            return Keypair.create_from_uri("//Alice")
        except ImportError:
            pytest.skip("substrateinterface not available")

    def test_sign_message_returns_hex_string(self, keypair):
        """Test sign_message returns hex-encoded signature."""
        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))

        signature = sign_message(keypair, body, timestamp)

        assert signature.startswith("0x")
        # Verify it's valid hex
        bytes.fromhex(signature.removeprefix("0x"))

    def test_sign_message_deterministic(self, keypair):
        """Test sign_message produces consistent signatures."""
        body = b'{"test": "data"}'
        timestamp = "1700000000"

        sig1 = sign_message(keypair, body, timestamp)
        sig2 = sign_message(keypair, body, timestamp)

        # Note: Some signing schemes are deterministic, some are not
        # SR25519 is typically not deterministic, so we just verify format
        assert sig1.startswith("0x")
        assert sig2.startswith("0x")


class TestVerifySignatureDependency:
    """Tests for verify_signature FastAPI dependency."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = MagicMock()
        request.body = AsyncMock(return_value=b'{"test": "data"}')
        return request

    async def test_verify_signature_raises_on_invalid(self, mock_request, monkeypatch):
        """Test verify_signature raises HTTPException on invalid signature."""
        # Mock verifier to return invalid
        mock_verifier = MagicMock()
        mock_verifier.verify_request = AsyncMock(return_value=(False, "Test error"))
        monkeypatch.setattr("leoma.delivery.http.verifier.get_verifier", lambda: mock_verifier)

        with pytest.raises(HTTPException) as exc_info:
            await verify_signature(
                request=mock_request,
                x_validator_hotkey=NETWORK_VALIDATOR_HOTKEY,
                x_signature="0x" + "00" * 64,
                x_timestamp=str(int(time.time())),
            )

        assert exc_info.value.status_code == 401
        assert "Test error" in exc_info.value.detail

    async def test_verify_signature_returns_hotkey_on_valid(self, mock_request, monkeypatch):
        """Test verify_signature returns hotkey on valid signature."""
        # Mock verifier to return valid
        mock_verifier = MagicMock()
        mock_verifier.verify_request = AsyncMock(return_value=(True, None))
        monkeypatch.setattr("leoma.delivery.http.verifier.get_verifier", lambda: mock_verifier)

        result = await verify_signature(
            request=mock_request,
            x_validator_hotkey=NETWORK_VALIDATOR_HOTKEY,
            x_signature="0x" + "00" * 64,
            x_timestamp=str(int(time.time())),
        )

        assert result == NETWORK_VALIDATOR_HOTKEY


class TestVerifyAdminSignatureDependency:
    """Tests for verify_admin_signature FastAPI dependency."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = MagicMock()
        request.body = AsyncMock(return_value=b'{"test": "data"}')
        return request

    async def test_verify_admin_signature_raises_on_non_admin(self, mock_request, monkeypatch):
        """Test verify_admin_signature raises HTTPException for non-admin."""
        # Mock verifier to return valid but not admin
        mock_verifier = MagicMock()
        mock_verifier.verify_request = AsyncMock(return_value=(True, None))
        mock_verifier.is_admin = MagicMock(return_value=False)
        monkeypatch.setattr("leoma.delivery.http.verifier.get_verifier", lambda: mock_verifier)

        with pytest.raises(HTTPException) as exc_info:
            await verify_admin_signature(
                request=mock_request,
                x_validator_hotkey=NETWORK_VALIDATOR_HOTKEY,
                x_signature="0x" + "00" * 64,
                x_timestamp=str(int(time.time())),
            )

        assert exc_info.value.status_code == 403
        assert "admin" in exc_info.value.detail.lower()

    async def test_verify_admin_signature_returns_hotkey_for_admin(self, mock_request, monkeypatch):
        """Test verify_admin_signature returns hotkey for admin."""
        # Mock verifier to return valid and admin
        mock_verifier = MagicMock()
        mock_verifier.verify_request = AsyncMock(return_value=(True, None))
        mock_verifier.is_admin = MagicMock(return_value=True)
        monkeypatch.setattr("leoma.delivery.http.verifier.get_verifier", lambda: mock_verifier)

        result = await verify_admin_signature(
            request=mock_request,
            x_validator_hotkey=ROOT_ADMIN_HOTKEY,
            x_signature="0x" + "00" * 64,
            x_timestamp=str(int(time.time())),
        )

        assert result == ROOT_ADMIN_HOTKEY


class TestGetVerifier:
    """Tests for get_verifier function."""

    def test_get_verifier_returns_singleton(self):
        """Test get_verifier returns the same instance."""
        # Reset the global verifier
        import leoma.delivery.http.verifier as sig_module
        sig_module._verifier = None

        verifier1 = get_verifier()
        verifier2 = get_verifier()

        assert verifier1 is verifier2

    def test_get_verifier_creates_instance(self):
        """Test get_verifier creates a SignatureVerifier instance."""
        # Reset the global verifier
        import leoma.delivery.http.verifier as sig_module
        sig_module._verifier = None

        verifier = get_verifier()

        assert isinstance(verifier, SignatureVerifier)
