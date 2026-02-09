"""Tests for the ScoringAgent."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.scoring_agent import ScoringAgent, RiskFactor, FraudScore


@pytest.fixture
def scoring_agent(patch_nim_client):
    return ScoringAgent()


class TestRiskFactor:
    def test_to_dict(self):
        rf = RiskFactor(name="Test", score=50.0, weight=0.25,
                        description="desc", evidence=["a", "b"])
        d = rf.to_dict()
        assert d["name"] == "Test"
        assert d["weighted_score"] == 12.5
        assert len(d["evidence"]) == 2

    def test_weighted_score_calculation(self):
        rf = RiskFactor(name="X", score=80.0, weight=0.2, description="")
        assert rf.to_dict()["weighted_score"] == pytest.approx(16.0)


class TestScoringAgent:
    def test_risk_level_critical(self, scoring_agent):
        assert scoring_agent._get_risk_level(80) == "critical"

    def test_risk_level_high(self, scoring_agent):
        assert scoring_agent._get_risk_level(60) == "high"

    def test_risk_level_medium(self, scoring_agent):
        assert scoring_agent._get_risk_level(30) == "medium"

    def test_risk_level_low(self, scoring_agent):
        assert scoring_agent._get_risk_level(10) == "low"

    def test_recommendation_mapping(self, scoring_agent):
        assert "DENY" in scoring_agent._get_recommendation("critical")
        assert "INVESTIGATE" in scoring_agent._get_recommendation("high")
        assert "REVIEW" in scoring_agent._get_recommendation("medium")
        assert "APPROVE" in scoring_agent._get_recommendation("low")

    def test_confidence_calculation(self, scoring_agent):
        factors = [
            RiskFactor("A", 50, 0.25, ""),
            RiskFactor("B", 50, 0.25, ""),
        ]
        conf = scoring_agent._calculate_confidence(factors)
        # Identical scores -> high confidence
        assert conf >= 0.9

    def test_confidence_with_variance(self, scoring_agent):
        factors = [
            RiskFactor("A", 10, 0.25, ""),
            RiskFactor("B", 90, 0.25, ""),
        ]
        conf = scoring_agent._calculate_confidence(factors)
        # High variance -> lower confidence
        assert conf < 0.9

    @pytest.mark.asyncio
    async def test_score_claim_characteristics_high_amount(self, scoring_agent):
        claim_data = {"claim": {"amount": 200000}, "medical": {"injuries": []}}
        result = await scoring_agent._score_claim_characteristics(claim_data)
        assert result["score"] >= 20

    @pytest.mark.asyncio
    async def test_score_claim_characteristics_soft_tissue(self, scoring_agent):
        claim_data = {
            "claim": {"amount": 5000},
            "medical": {"injuries": ["whiplash", "bruising"]},
        }
        result = await scoring_agent._score_claim_characteristics(claim_data)
        assert result["score"] >= 15

    @pytest.mark.asyncio
    async def test_calculate_score_returns_all_fields(self, scoring_agent, claim_data, patch_nim_client):
        patch_nim_client.chat.return_value = "Moderate risk due to soft tissue injuries."

        inconsistency_results = {"inconsistency_score": 30, "inconsistencies": []}
        pattern_results = {"pattern_risk_score": 20, "matched_patterns": []}

        result = await scoring_agent.calculate_score(
            claim_data, inconsistency_results, pattern_results
        )

        assert "overall_score" in result
        assert "risk_level" in result
        assert "confidence" in result
        assert "risk_factors" in result
        assert "recommendation" in result
