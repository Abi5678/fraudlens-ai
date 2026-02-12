"""
MortgageLens AI - Mortgage Loan Verification Orchestrator
Coordinates agents to verify mortgage application documents for fraud.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from agents.document_agent import DocumentAgent
from agents.inconsistency_agent import InconsistencyAgent
from agents.income_verification_agent import IncomeVerificationAgent
from agents.property_valuation_agent import PropertyValuationAgent
from agents.scoring_agent import ScoringAgent
from agents.narrative_agent import NarrativeAgent


@dataclass
class MortgageAnalysisResult:
    """Complete mortgage verification result."""
    application_data: Dict[str, Any]
    risk_score: float
    risk_level: str
    recommendation: str
    narrative: str
    inconsistencies: Dict[str, Any]
    income_analysis: Dict[str, Any]
    property_analysis: Dict[str, Any]
    scoring_details: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application_data": self.application_data,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "narrative": self.narrative,
            "inconsistencies": self.inconsistencies,
            "income_analysis": self.income_analysis,
            "property_analysis": self.property_analysis,
            "scoring_details": self.scoring_details,
        }


class MortgageLensAI:
    """Orchestrator for mortgage loan verification."""

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key

        self.document_agent = DocumentAgent()
        self.inconsistency_agent = InconsistencyAgent()
        self.income_agent = IncomeVerificationAgent()
        self.property_agent = PropertyValuationAgent()
        self.scoring_agent = ScoringAgent()
        self.narrative_agent = NarrativeAgent()

        logger.info("MortgageLens AI initialized")

    async def analyze(self, document_path: str) -> MortgageAnalysisResult:
        """Analyze mortgage application documents."""
        logger.info(f"Starting mortgage analysis for: {document_path}")

        path = Path(document_path)

        # Phase 1: Document extraction
        if path.suffix.lower() == ".json":
            with open(document_path, "r") as f:
                data = json.load(f)
            app_data = data.get("application_data", data.get("claim_data", data))
            raw_text = data.get("raw_text", json.dumps(app_data, indent=2))
        else:
            doc_result = await self.document_agent.process(document_path)
            app_data = doc_result.get("claim_data", {})
            raw_text = doc_result.get("raw_text", "")

        # Phase 2: Parallel analysis
        tasks = [
            self.inconsistency_agent.analyze(app_data, raw_text),
            self.income_agent.analyze(app_data, raw_text),
            self.property_agent.analyze(app_data, raw_text),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        inconsistency_result = (
            results[0] if not isinstance(results[0], Exception)
            else {"inconsistencies": [], "inconsistency_score": 0}
        )
        income_result = (
            results[1] if not isinstance(results[1], Exception)
            else {"risk_score": 0, "flags": [], "summary": "Analysis failed"}
        )
        property_result = (
            results[2] if not isinstance(results[2], Exception)
            else {"risk_score": 0, "flags": [], "summary": "Analysis failed"}
        )

        # Phase 3: Scoring (local — no LLM call)
        pattern_result = {
            "matched_patterns": [],
            "pattern_risk_score": (income_result.get("risk_score", 0) + property_result.get("risk_score", 0)) / 2,
        }

        score_result = await self.scoring_agent.calculate_score(
            app_data, inconsistency_result, pattern_result, None, None,
        )

        # Phase 4: Reasoning + Narrative in PARALLEL
        doc_result_ctx = {"claim_data": app_data, "raw_text": raw_text}
        reasoning_task = self.scoring_agent.generate_reasoning(score_result)
        narrative_task = self.narrative_agent.generate(
            doc_result_ctx, score_result, inconsistency_result, pattern_result, None,
        )
        reasoning, narrative_result = await asyncio.gather(
            reasoning_task, narrative_task, return_exceptions=True
        )
        if isinstance(reasoning, Exception):
            reasoning = f"Application scored {score_result.get('overall_score', 0):.0f}/100."
        if isinstance(narrative_result, Exception):
            narrative_result = {"status": "error", "full_narrative": ""}
        score_result["reasoning"] = reasoning

        logger.info(f"Mortgage analysis complete. Risk Score: {score_result.get('overall_score', 0)}/100")

        return MortgageAnalysisResult(
            application_data=app_data,
            risk_score=score_result.get("overall_score", 0),
            risk_level=score_result.get("risk_level", "unknown"),
            recommendation=score_result.get("recommendation", ""),
            narrative=narrative_result.get("full_narrative", ""),
            inconsistencies=inconsistency_result,
            income_analysis=income_result,
            property_analysis=property_result,
            scoring_details=score_result,
            raw_text=raw_text,
        )
