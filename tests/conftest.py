"""Shared test configuration for the Leoma test suite.

The database, owner-api, and hosted-inference fixtures were retired with the
pooled-scoring stack. The king-of-the-hill suite is self-contained (pure logic + a FastAPI
TestClient with a fake runner), so only the storage-backend default is set here.
"""
import os

# Tests and fixtures assume the Hippius backend unless a suite overrides it.
os.environ.setdefault("OBJECT_STORAGE_BACKEND", "hippius")
