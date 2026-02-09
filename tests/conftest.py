"""
Shared fixtures for FraudLens AI tests.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_claim_path():
    return str(PROJECT_ROOT / "sample_claim.json")


@pytest.fixture
def sample_claim_data():
    path = PROJECT_ROOT / "sample_claim.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def claim_data(sample_claim_data):
    """Just the claim_data portion."""
    return sample_claim_data["claim_data"]


@pytest.fixture
def raw_text(sample_claim_data):
    return sample_claim_data["raw_text"]


# ---------------------------------------------------------------------------
# Mock NIM client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_nim_client():
    """A mock NIMClient that returns plausible dummy responses."""
    client = AsyncMock()
    client.chat = AsyncMock(return_value="No inconsistencies detected.")
    client.embed = AsyncMock(return_value=[[0.1] * 1024])
    client.rerank = AsyncMock(return_value=[])
    client.parse_document = AsyncMock(return_value={"text": "parsed text"})
    return client


@pytest.fixture(autouse=True)
def patch_nim_client(mock_nim_client):
    """Automatically patch get_nim_client so no real API calls are made."""
    with patch("core.nim_client.get_nim_client", return_value=mock_nim_client):
        with patch("core.nim_client._nim_client", mock_nim_client):
            yield mock_nim_client
