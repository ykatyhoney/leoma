"""Unit-test fixtures.

The database + owner-api fixtures were retired with the pooled-scoring stack;
the king-of-the-hill test suite is self-contained (pure logic + a FastAPI
TestClient with a fake runner), so no shared fixtures are needed here.
"""
