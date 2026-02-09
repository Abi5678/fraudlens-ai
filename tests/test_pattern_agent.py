"""Tests for the PatternAgent."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.pattern_agent import PatternAgent, PatternMatch


@pytest.fixture
def agent(patch_nim_client):
    return PatternAgent()


class TestPatternMatch:
    def test_to_dict(self):
        pm = PatternMatch(
            pattern_id="p1", pattern_name="staged_accident",
            description="desc", similarity_score=0.85,
            category="staged_accident", severity="high",
            matching_elements=["elem1"],
        )
        d = pm.to_dict()
        assert d["pattern_id"] == "p1"
        assert d["similarity_score"] == 0.85


class TestPatternAgent:
    def test_build_search_query(self, agent):
        claim_data = {
            "incident": {"description": "rear-ended at red light"},
            "claim": {"type": "auto", "amount": 45000},
            "medical": {"injuries": ["whiplash"]},
        }
        query = agent._build_search_query(claim_data, "some raw text")
        assert "rear-ended" in query
        assert "auto" in query
        assert "whiplash" in query

    def test_calculate_pattern_score_empty(self, agent):
        assert agent._calculate_pattern_score([]) == 0.0

    def test_calculate_pattern_score_with_matches(self, agent):
        matches = [
            PatternMatch("p1", "staged", "d", 0.9, "staged", "critical", ["e1", "e2"]),
        ]
        score = agent._calculate_pattern_score(matches)
        assert score > 0

    def test_calculate_pattern_score_capped(self, agent):
        matches = [
            PatternMatch(f"p{i}", "x", "d", 0.95, "x", "critical", ["e"] * 5)
            for i in range(10)
        ]
        score = agent._calculate_pattern_score(matches)
        assert score <= 100

    def test_generate_summary_no_matches(self, agent):
        summary = agent._generate_summary([])
        assert "No known fraud" in summary

    def test_generate_summary_critical_match(self, agent):
        matches = [
            PatternMatch("p1", "fraud_ring", "d", 0.9, "fraud_ring", "critical", []),
        ]
        summary = agent._generate_summary(matches)
        assert "CRITICAL" in summary

    @pytest.mark.asyncio
    async def test_analyze_returns_structure(self, agent, claim_data, raw_text, patch_nim_client):
        # Mock the vector store to skip initialization
        agent.vector_store = None
        patch_nim_client.chat.return_value = "- Matching Element 1: soft tissue injury pattern"

        # Patch initialize to skip Milvus
        async def mock_init():
            from unittest.mock import AsyncMock, MagicMock
            mock_store = MagicMock()
            mock_store.search = AsyncMock(return_value=[])
            agent.vector_store = mock_store

        agent.initialize = mock_init

        result = await agent.analyze(claim_data, raw_text)
        assert "matched_patterns" in result
        assert "pattern_risk_score" in result
