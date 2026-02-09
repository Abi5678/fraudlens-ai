"""
FraudLens AI Agents
Multi-agent system for insurance fraud detection
"""

from agents.document_agent import DocumentAgent
from agents.inconsistency_agent import InconsistencyAgent
from agents.pattern_agent import PatternAgent
from agents.scoring_agent import ScoringAgent
from agents.narrative_agent import NarrativeAgent
from agents.network_agent import NetworkAgent
from agents.deepfake_agent import DeepfakeAgent

__all__ = [
    "DocumentAgent",
    "InconsistencyAgent", 
    "PatternAgent",
    "ScoringAgent",
    "NarrativeAgent",
    "NetworkAgent",
    "DeepfakeAgent",
]
