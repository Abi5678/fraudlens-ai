"""Tests for the InconsistencyAgent."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.inconsistency_agent import InconsistencyAgent, Inconsistency


@pytest.fixture
def agent(patch_nim_client):
    return InconsistencyAgent()


class TestInconsistency:
    def test_to_dict(self):
        inc = Inconsistency(
            type="timeline", description="test",
            severity="high", confidence=0.9, evidence=["e1"],
        )
        d = inc.to_dict()
        assert d["type"] == "timeline"
        assert d["severity"] == "high"
        assert len(d["evidence"]) == 1


class TestInconsistencyAgent:
    def test_calculate_score_empty(self, agent):
        assert agent._calculate_score([]) == 0.0

    def test_calculate_score_critical(self, agent):
        incs = [Inconsistency("timeline", "desc", "critical", 1.0)]
        score = agent._calculate_score(incs)
        assert score == 30.0

    def test_calculate_score_caps_at_100(self, agent):
        incs = [Inconsistency("t", "d", "critical", 1.0) for _ in range(10)]
        score = agent._calculate_score(incs)
        assert score == 100.0

    def test_generate_summary_no_issues(self, agent):
        summary = agent._generate_summary([])
        assert "No significant" in summary

    def test_generate_summary_with_issues(self, agent):
        incs = [
            Inconsistency("t", "d", "critical", 1.0),
            Inconsistency("t", "d", "high", 0.9),
        ]
        summary = agent._generate_summary(incs)
        assert "critical" in summary.lower() or "2" in summary

    def test_parse_no_inconsistencies(self, agent):
        response = "No timeline inconsistencies detected."
        result = agent._parse_inconsistencies(response, "timeline")
        assert result == []

    def test_parse_numbered_items(self, agent):
        response = """
1. Claim was filed before incident date
2. Treatment started before injury was reported
"""
        result = agent._parse_inconsistencies(response, "timeline")
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_check_numerical_exceeds_coverage(self, agent):
        claim_data = {
            "claim": {"amount": 200000},
            "medical": {"costs": 0},
            "policy": {"coverage_amount": 50000},
        }
        result = await agent._check_numerical_consistency(claim_data)
        assert len(result) >= 1
        assert result[0].type == "numerical"

    @pytest.mark.asyncio
    async def test_check_numerical_medical_exceeds_claim(self, agent):
        claim_data = {
            "claim": {"amount": 10000},
            "medical": {"costs": 15000},
            "policy": {"coverage_amount": 50000},
        }
        result = await agent._check_numerical_consistency(claim_data)
        assert any("Medical costs" in i.description for i in result)

    @pytest.mark.asyncio
    async def test_analyze_returns_structure(self, agent, claim_data, raw_text, patch_nim_client):
        patch_nim_client.chat.return_value = "No inconsistencies detected."
        result = await agent.analyze(claim_data, raw_text)
        assert "inconsistencies" in result
        assert "inconsistency_score" in result
        assert "summary" in result
