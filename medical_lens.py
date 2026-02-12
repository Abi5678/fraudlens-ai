"""
MedicalClaimLens AI - Medical Insurance Fraud Detection Orchestrator
Coordinates agents to detect upcoding, unbundling, duplicate billing,
and clinical inconsistencies in medical insurance claims.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from agents.document_agent import DocumentAgent
from agents.billing_integrity_agent import BillingIntegrityAgent
from agents.clinical_consistency_agent import ClinicalConsistencyAgent
from agents.eligibility_agent import EligibilityAgent
from agents.inconsistency_agent import InconsistencyAgent
from agents.scoring_agent import ScoringAgent
from agents.narrative_agent import NarrativeAgent


@dataclass
class MedicalAnalysisResult:
    """Complete medical claim fraud analysis result."""
    claim_data: Dict[str, Any]
    risk_score: float
    risk_level: str
    recommendation: str
    narrative: str
    billing_analysis: Dict[str, Any]
    clinical_analysis: Dict[str, Any]
    eligibility_analysis: Dict[str, Any]
    inconsistencies: Dict[str, Any]
    scoring_details: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_data": self.claim_data,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "narrative": self.narrative,
            "billing_analysis": self.billing_analysis,
            "clinical_analysis": self.clinical_analysis,
            "eligibility_analysis": self.eligibility_analysis,
            "inconsistencies": self.inconsistencies,
            "scoring_details": self.scoring_details,
        }


class MedicalClaimLensAI:
    """Orchestrator for medical insurance claim fraud detection."""

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key

        self.document_agent = DocumentAgent()
        self.billing_agent = BillingIntegrityAgent()
        self.clinical_agent = ClinicalConsistencyAgent()
        self.eligibility_agent = EligibilityAgent()
        self.inconsistency_agent = InconsistencyAgent()
        self.scoring_agent = ScoringAgent()
        self.narrative_agent = NarrativeAgent()

        logger.info("MedicalClaimLens AI initialized")

    async def analyze(self, document_path: str) -> MedicalAnalysisResult:
        """Analyze medical insurance claim documents for fraud."""
        logger.info(f"Starting medical claim analysis for: {document_path}")

        path = Path(document_path)

        # Phase 1: Document extraction
        if path.suffix.lower() == ".json":
            with open(document_path, "r") as f:
                data = json.load(f)
            claim_data = data.get("claim_data", data)
            raw_text = data.get("raw_text", json.dumps(claim_data, indent=2))
        else:
            doc_result = await self.document_agent.process(document_path)
            claim_data = doc_result.get("claim_data", {})
            raw_text = doc_result.get("raw_text", "")

        # Phase 2: Parallel analysis — all medical agents run concurrently
        tasks = [
            self.billing_agent.analyze(claim_data, raw_text),
            self.clinical_agent.analyze(claim_data, raw_text),
            self.eligibility_agent.analyze(claim_data, raw_text),
            self.inconsistency_agent.analyze(claim_data, raw_text),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        billing_result = (
            results[0] if not isinstance(results[0], Exception)
            else {"billing_verified": False, "risk_score": 0, "flags": [], "summary": "Analysis failed"}
        )
        clinical_result = (
            results[1] if not isinstance(results[1], Exception)
            else {"clinically_consistent": False, "risk_score": 0, "flags": [], "summary": "Analysis failed"}
        )
        eligibility_result = (
            results[2] if not isinstance(results[2], Exception)
            else {"eligible": False, "risk_score": 0, "flags": [], "summary": "Analysis failed"}
        )
        inconsistency_result = (
            results[3] if not isinstance(results[3], Exception)
            else {"inconsistencies": [], "inconsistency_score": 0}
        )

        # Phase 3: Scoring
        # Combine medical agent scores into pattern_result for scoring agent
        medical_risk = (
            billing_result.get("risk_score", 0) * 0.4
            + clinical_result.get("risk_score", 0) * 0.35
            + eligibility_result.get("risk_score", 0) * 0.25
        )
        pattern_result = {
            "matched_patterns": [],
            "pattern_risk_score": medical_risk,
        }

        score_result = await self.scoring_agent.calculate_score(
            claim_data, inconsistency_result, pattern_result, None, None,
            raw_text=raw_text,
        )

        # Phase 4: Reasoning + Narrative in parallel
        doc_result_ctx = {"claim_data": claim_data, "raw_text": raw_text}
        reasoning_task = self.scoring_agent.generate_reasoning(score_result)
        narrative_task = self.narrative_agent.generate(
            doc_result_ctx, score_result, inconsistency_result, pattern_result, None,
        )
        reasoning, narrative_result = await asyncio.gather(
            reasoning_task, narrative_task, return_exceptions=True
        )
        if isinstance(reasoning, Exception):
            reasoning = f"Claim scored {score_result.get('overall_score', 0):.0f}/100."
        if isinstance(narrative_result, Exception):
            narrative_result = {"status": "error", "full_narrative": ""}
        score_result["reasoning"] = reasoning

        logger.info(f"Medical analysis complete. Risk Score: {score_result.get('overall_score', 0)}/100")

        return MedicalAnalysisResult(
            claim_data=claim_data,
            risk_score=score_result.get("overall_score", 0),
            risk_level=score_result.get("risk_level", "unknown"),
            recommendation=score_result.get("recommendation", ""),
            narrative=narrative_result.get("full_narrative", ""),
            billing_analysis=billing_result,
            clinical_analysis=clinical_result,
            eligibility_analysis=eligibility_result,
            inconsistencies=inconsistency_result,
            scoring_details=score_result,
            raw_text=raw_text,
        )
