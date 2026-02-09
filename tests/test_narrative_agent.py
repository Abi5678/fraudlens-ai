"""Tests for the NarrativeAgent."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.narrative_agent import NarrativeAgent


@pytest.fixture
def agent(patch_nim_client):
    return NarrativeAgent()


class TestNarrativeAgent:
    def test_build_prompt(self, agent):
        claim_data = {
            "claim_data": {
                "claimant": {"name": "John Doe"},
                "claim": {"amount": 50000},
            }
        }
        fraud_score = {"overall_score": 72, "risk_level": "high"}
        inconsistencies = {"inconsistencies": [{"desc": "a"}]}
        pattern_matches = {"matched_patterns": [{"name": "staged"}]}

        prompt = agent._build_prompt(claim_data, fraud_score, inconsistencies, pattern_matches, None)
        assert "John Doe" in prompt
        assert "72" in prompt
        assert "HIGH" in prompt

    def test_parse_sections(self, agent):
        narrative = "Some text here"
        result = agent._parse_sections(narrative)
        assert "full_text" in result

    @pytest.mark.asyncio
    async def test_generate_returns_structure(self, agent, patch_nim_client):
        patch_nim_client.chat.return_value = "EXECUTIVE SUMMARY\nThis claim shows moderate risk."

        result = await agent.generate(
            {"claim_data": {"claimant": {"name": "Test"}, "claim": {"amount": 1000}}},
            {"overall_score": 45, "risk_level": "medium"},
            {"inconsistencies": []},
            {"matched_patterns": []},
        )
        assert result["status"] == "success"
        assert "full_narrative" in result

    @pytest.mark.asyncio
    async def test_generate_handles_error(self, agent, patch_nim_client):
        patch_nim_client.chat.side_effect = Exception("API error")
        result = await agent.generate(
            {"claim_data": {"claimant": {"name": "Test"}, "claim": {"amount": 0}}},
            {"overall_score": 0, "risk_level": "low"},
            {"inconsistencies": []},
            {"matched_patterns": []},
        )
        assert result["status"] == "error"
