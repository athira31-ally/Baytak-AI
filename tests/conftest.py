"""Build a small, isolated set of artifacts once per test session."""
import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="dhm-test-")
for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_SEARCH_ENDPOINT", "COSMOS_ENDPOINT", "APPLICATIONINSIGHTS_CONNECTION_STRING"):
    os.environ.pop(var, None)
os.environ["AZURE_OPENAI_ENDPOINT"] = ""  # force offline mode even if a local .env sets it
os.environ["AZURE_SEARCH_ENDPOINT"] = ""
os.environ["COSMOS_ENDPOINT"] = ""
os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"] = ""

import pytest  # noqa: E402

from scripts.bootstrap import main as bootstrap  # noqa: E402

bootstrap(n_listings=1200, n_users=400)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c
