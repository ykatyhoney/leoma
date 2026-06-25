"""
Verify that peer verdict-file signatures round-trip: a payload signed the way the sampler signs
it (APIClient.sign_evaluation_payload) passes aggregate_local._verify, and tampering is rejected.
"""
import json
import hashlib

from substrateinterface import Keypair

from leoma.app.validator.aggregate_local import _verify


def _sign(keypair: Keypair, data: list) -> str:
    """Mirror APIClient.sign_evaluation_payload."""
    canonical = json.dumps(data, sort_keys=True).encode("utf-8")
    msg_hash = hashlib.sha256(canonical).digest()
    return "0x" + keypair.sign(msg_hash).hex()


def _data():
    return [
        {"hotkey": "5MinerA", "passed": True, "status": "passed"},
        {"hotkey": "5MinerB", "passed": False, "status": "failed"},
    ]


class TestVerifyVerdictSignature:
    def test_valid_signature_accepted(self):
        kp = Keypair.create_from_uri("//Alice")
        data = _data()
        wrapper = {"signature": _sign(kp, data), "data": data}
        assert _verify(kp.ss58_address, wrapper) is True

    def test_tampered_data_rejected(self):
        kp = Keypair.create_from_uri("//Alice")
        data = _data()
        wrapper = {"signature": _sign(kp, data), "data": data}
        # Flip a verdict after signing.
        wrapper["data"][0]["passed"] = False
        assert _verify(kp.ss58_address, wrapper) is False

    def test_wrong_signer_rejected(self):
        signer = Keypair.create_from_uri("//Alice")
        other = Keypair.create_from_uri("//Bob")
        data = _data()
        wrapper = {"signature": _sign(signer, data), "data": data}
        # Verifying against Bob's address must fail (signed by Alice).
        assert _verify(other.ss58_address, wrapper) is False

    def test_missing_signature_accepted(self):
        # No signature -> accept (compat); aggregation still works.
        assert _verify("5Whatever", {"data": _data()}) is True
        assert _verify("5Whatever", {"signature": "", "data": _data()}) is True

    def test_malformed_signature_does_not_raise(self):
        # A garbage signature must not crash aggregation; _verify swallows errors -> True.
        wrapper = {"signature": "0xnothex", "data": _data()}
        assert _verify("5Whatever", wrapper) is True
