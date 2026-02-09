"""Tests for the FraudLens orchestrator (fraudlens.py)."""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from fraudlens import FraudLensAI, FraudAnalysisResult


class TestFraudAnalysisResult:
    def test_to_dict(self):
        result = FraudAnalysisResult(
            claim_data={"claimant": {"name": "Test"}},
            fraud_score=65.0,
            risk_level="high",
            recommendation="INVESTIGATE",
            narrative="Test narrative",
            inconsistencies={"inconsistencies": [], "inconsistency_score": 0},
            pattern_matches={"matched_patterns": [], "pattern_risk_score": 0},
            network_analysis={"fraud_ring_detected": True},
            deepfake_analysis={},
        )
        d = result.to_dict()
        assert d["fraud_score"] == 65.0
        assert d["fraud_ring_detected"] is True

    def test_fraud_ring_not_detected(self):
        result = FraudAnalysisResult(
            claim_data={}, fraud_score=10, risk_level="low",
            recommendation="APPROVE", narrative="", inconsistencies={},
            pattern_matches={}, network_analysis={}, deepfake_analysis={},
        )
        assert result.fraud_ring_detected is False

    def test_fraud_ring_detected(self):
        result = FraudAnalysisResult(
            claim_data={}, fraud_score=80, risk_level="critical",
            recommendation="DENY", narrative="", inconsistencies={},
            pattern_matches={}, network_analysis={"fraud_ring_detected": True},
            deepfake_analysis={},
        )
        assert result.fraud_ring_detected is True


class TestFraudLensAI:
    @pytest.mark.asyncio
    async def test_analyze_json(self, sample_claim_path, patch_nim_client):
        patch_nim_client.chat.return_value = "No issues found. Low risk."

        # Patch PatternAgent.initialize to avoid Milvus
        with patch("agents.pattern_agent.PatternAgent.initialize", new_callable=AsyncMock):
            with patch("agents.pattern_agent.PatternAgent.analyze", new_callable=AsyncMock) as mock_pattern:
                mock_pattern.return_value = {
                    "status": "success", "matched_patterns": [],
                    "pattern_risk_score": 0, "summary": "No matches",
                }
                detector = FraudLensAI()
                result = await detector.analyze(sample_claim_path)

        assert isinstance(result, FraudAnalysisResult)
        assert result.fraud_score >= 0
        assert result.risk_level in ("low", "medium", "high", "critical", "unknown")

    @pytest.mark.asyncio
    async def test_run_analysis_without_network(self, patch_nim_client):
        patch_nim_client.chat.return_value = "Low risk claim."

        with patch("agents.pattern_agent.PatternAgent.analyze", new_callable=AsyncMock) as mock_pat:
            mock_pat.return_value = {
                "matched_patterns": [], "pattern_risk_score": 0, "summary": "",
            }
            detector = FraudLensAI()
            result = await detector._run_analysis(
                claim_data={"claim": {"amount": 1000}, "medical": {"injuries": []}},
                raw_text="test",
                include_network=False,
                include_deepfake=False,
            )

        assert result.network_analysis == {}
        assert result.deepfake_analysis == {}
