"""Tests for the NetworkAgent."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.network_agent import NetworkAgent, FraudConnection


@pytest.fixture
def agent(patch_nim_client):
    return NetworkAgent()


class TestFraudConnection:
    def test_fields(self):
        fc = FraudConnection("Alice", "Bob", "shared_address", 0.9, "Same address")
        assert fc.entity1 == "Alice"
        assert fc.strength == 0.9


class TestNetworkAgent:
    def test_extract_entities(self, agent):
        claim_data = {
            "claim_data": {
                "claimant": {"name": "John", "address": "123 Main St", "phone": "555-1234"},
                "policy": {"number": "POL-001"},
                "vehicle": {"vin": "VIN123"},
                "medical": {"providers": ["Dr. Smith"]},
                "incident": {"location": "Highway 1"},
            }
        }
        entities = agent._extract_entities(claim_data)
        assert entities["claimant_name"] == "John"
        assert entities["address"] == "123 Main St"

    def test_find_connections_shared_address(self, agent):
        entities = {
            "claimant_name": "Alice",
            "address": "123 Main St",
            "phone": "",
            "providers": [],
        }
        historical = [{
            "claim_data": {
                "claimant": {"name": "Bob", "address": "123 Main St", "phone": ""},
                "policy": {"number": "P2"},
                "vehicle": {"vin": ""},
                "medical": {"providers": []},
                "incident": {"location": ""},
            }
        }]
        connections = agent._find_connections(entities, historical)
        assert len(connections) == 1
        assert connections[0].connection_type == "shared_address"

    def test_detect_communities_cpu(self, agent):
        connections = [
            FraudConnection("A", "B", "addr", 0.9, "same"),
            FraudConnection("B", "C", "phone", 0.8, "same"),
        ]
        communities = agent._detect_communities_cpu(connections)
        assert len(communities) >= 1
        assert communities[0]["size"] >= 2

    def test_calculate_network_risk_empty(self, agent):
        assert agent._calculate_network_risk([], []) == 0

    def test_calculate_network_risk_with_connections(self, agent):
        connections = [
            FraudConnection("A", "B", "addr", 0.9, "same"),
            FraudConnection("B", "C", "phone", 0.85, "same"),
        ]
        communities = [{"id": 0, "members": ["A", "B", "C"], "size": 3}]
        score = agent._calculate_network_risk(connections, communities)
        assert score > 0

    def test_generate_summary_high(self, agent):
        connections = [FraudConnection("A", "B", "addr", 0.9, "same")] * 5
        communities = [{"id": 0, "members": ["A", "B", "C", "D"], "size": 4}]
        summary = agent._generate_summary(connections, communities, 80)
        assert "CRITICAL" in summary

    @pytest.mark.asyncio
    async def test_analyze_with_llm_fallback(self, agent, patch_nim_client):
        patch_nim_client.chat.return_value = "Low risk. No fraud ring indicators found."
        claim_data = {
            "claim_data": {
                "claimant": {"name": "Test", "address": "1 St", "phone": "555"},
                "policy": {"number": "P1"},
                "vehicle": {"vin": "V1"},
                "medical": {"providers": []},
                "incident": {"location": "Loc"},
            }
        }
        result = await agent.analyze(claim_data)
        assert "network_risk_score" in result
        assert result["network_risk_score"] <= 25
