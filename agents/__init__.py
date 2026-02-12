"""
FraudLens AI Agents
Multi-agent system for document fraud detection across verticals
"""

from agents.document_agent import DocumentAgent
from agents.inconsistency_agent import InconsistencyAgent
from agents.pattern_agent import PatternAgent
from agents.scoring_agent import ScoringAgent
from agents.narrative_agent import NarrativeAgent
from agents.network_agent import NetworkAgent
from agents.deepfake_agent import DeepfakeAgent
from agents.income_verification_agent import IncomeVerificationAgent
from agents.property_valuation_agent import PropertyValuationAgent
from agents.template_match_agent import TemplateMatchAgent
from agents.metadata_agent import MetadataAgent
from agents.text_gen_detector import TextGenDetectorAgent
from agents.ai_image_detector import AIImageDetectorAgent

__all__ = [
    # Insurance fraud agents
    "DocumentAgent",
    "InconsistencyAgent",
    "PatternAgent",
    "ScoringAgent",
    "NarrativeAgent",
    "NetworkAgent",
    "DeepfakeAgent",
    # Mortgage agents
    "IncomeVerificationAgent",
    "PropertyValuationAgent",
    # Photo ID agents
    "TemplateMatchAgent",
    "MetadataAgent",
    # AI Content agents
    "TextGenDetectorAgent",
    "AIImageDetectorAgent",
]
