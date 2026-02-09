"""Tests for the DocumentAgent."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.document_agent import DocumentAgent, ClaimData


@pytest.fixture
def agent(patch_nim_client):
    return DocumentAgent()


class TestClaimData:
    def test_default_values(self):
        cd = ClaimData()
        assert cd.claimant_name == ""
        assert cd.claim_amount == 0.0
        assert cd.injuries_reported == []

    def test_to_dict_structure(self):
        cd = ClaimData(claimant_name="Test", claim_amount=5000)
        d = cd.to_dict()
        assert d["claimant"]["name"] == "Test"
        assert d["claim"]["amount"] == 5000


class TestDocumentAgent:
    def test_parse_amount(self, agent):
        assert agent._parse_amount("$1,234.56") == 1234.56
        assert agent._parse_amount("5000") == 5000.0
        assert agent._parse_amount("invalid") == 0.0
        assert agent._parse_amount("$0") == 0.0

    def test_parse_llm_response_claimant(self, agent):
        response = """
1. CLAIMANT INFORMATION:
   - Full Name: John Doe
   - Address: 123 Main St
   - Phone Number: 555-1234

2. POLICY INFORMATION:
   - Policy Number: POL-001
   - Policy Type: auto

3. INCIDENT INFORMATION:
   - Date of Incident: 2025-01-01
   - Description: Rear-ended at intersection

4. CLAIM DETAILS:
   - Claim Number: CLM-001
   - Amount Claimed: $50,000

5. VEHICLE INFORMATION:
   - Make: Toyota
   - Model: Camry
   - Year: 2022
"""
        claim = agent._parse_llm_response(response)
        assert claim.claimant_name == "John Doe"
        assert claim.policy_number == "POL-001"
        assert claim.incident_date == "2025-01-01"
        assert claim.claim_amount == 50000.0
        assert claim.vehicle_make == "Toyota"

    def test_parse_llm_response_medical(self, agent):
        response = """
6. MEDICAL INFORMATION:
   - Injuries Reported: whiplash, back pain, neck strain
   - Treatment Providers: Dr. Smith, QuickCare
   - Medical Costs: $25,000
"""
        claim = agent._parse_llm_response(response)
        assert len(claim.injuries_reported) == 3
        assert "whiplash" in claim.injuries_reported
        assert claim.medical_costs == 25000.0

    @pytest.mark.asyncio
    async def test_process_returns_error_for_missing_file(self, agent):
        result = await agent.process("/nonexistent/file.pdf")
        assert result["status"] == "error"
