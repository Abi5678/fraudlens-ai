"""
IDVerify AI - Photo ID Verification Orchestrator
Coordinates agents to verify photo ID authenticity.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from agents.document_agent import DocumentAgent
from agents.deepfake_agent import DeepfakeAgent
from agents.template_match_agent import TemplateMatchAgent
from agents.metadata_agent import MetadataAgent
from agents.id_consistency_agent import IDConsistencyAgent
from agents.scoring_agent import ScoringAgent
from agents.narrative_agent import NarrativeAgent


@dataclass
class IDVerificationResult:
    """Complete ID verification result."""
    document_data: Dict[str, Any]
    authenticity_score: float
    risk_level: str
    recommendation: str
    narrative: str
    deepfake_analysis: Dict[str, Any]
    template_analysis: Dict[str, Any]
    metadata_analysis: Dict[str, Any]
    consistency_analysis: Dict[str, Any] = field(default_factory=dict)
    face_verification: Dict[str, Any] = field(default_factory=dict)
    scoring_details: Dict[str, Any] = field(default_factory=dict)
    image_paths: List[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_data": self.document_data,
            "authenticity_score": self.authenticity_score,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "narrative": self.narrative,
            "deepfake_analysis": self.deepfake_analysis,
            "template_analysis": self.template_analysis,
            "metadata_analysis": self.metadata_analysis,
            "consistency_analysis": self.consistency_analysis,
            "face_verification": self.face_verification,
            "scoring_details": self.scoring_details,
            "images_analyzed": len(self.image_paths),
        }


class IDVerifyAI:
    """Orchestrator for photo ID verification."""

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key

        self.document_agent = DocumentAgent()
        self.deepfake_agent = DeepfakeAgent()
        self.template_agent = TemplateMatchAgent()
        self.metadata_agent = MetadataAgent()
        self.consistency_agent = IDConsistencyAgent()
        self.scoring_agent = ScoringAgent()
        self.narrative_agent = NarrativeAgent()

        logger.info("IDVerify AI initialized")

    async def analyze(self, image_paths: List[str]) -> IDVerificationResult:
        """Analyze photo ID images for authenticity."""
        logger.info(f"Starting ID verification for {len(image_paths)} images")

        # Phase 1: Document extraction from first image (NeMo Retriever OCR + Nemotron Nano VL when available)
        doc_data = {}
        raw_text = ""
        if image_paths:
            try:
                doc_result = await self.document_agent.process_id_image(image_paths[0])
                doc_data = doc_result.get("claim_data", {})
                raw_text = doc_result.get("raw_text", "")
            except Exception as e:
                logger.warning(f"Document extraction failed: {e}")

        # Phase 2: Parallel analysis (including ID plausibility + AI-generated ID detection)
        tasks = [
            self.deepfake_agent.analyze(image_paths, context="id_document"),
            self.template_agent.analyze(doc_data, raw_text),
            self.metadata_agent.analyze(image_paths, doc_data),
            self.consistency_agent.analyze(doc_data, raw_text, image_paths),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        deepfake_result = (
            results[0] if not isinstance(results[0], Exception)
            else {"manipulation_score": 0, "status": "error"}
        )
        template_result = (
            results[1] if not isinstance(results[1], Exception)
            else {"template_match_score": 0, "risk_score": 0, "flags": []}
        )
        metadata_result = (
            results[2] if not isinstance(results[2], Exception)
            else {"risk_score": 0, "flags": []}
        )
        consistency_result = (
            results[3] if not isinstance(results[3], Exception)
            else {"risk_score": 0, "flags": [], "summary": ""}
        )

        # Optional: face verification when 2+ images (e.g. ID portrait + selfie)
        face_result = {}
        if len(image_paths) >= 2:
            try:
                from core.id_ocr_service import face_verify_nano_vl
                same_person, confidence, explanation = await face_verify_nano_vl(
                    image_paths[0], image_paths[1]
                )
                face_result = {
                    "performed": True,
                    "same_person": same_person,
                    "confidence": confidence,
                    "explanation": explanation,
                }
                if not same_person and confidence > 30:
                    consistency_result = dict(consistency_result)
                    consistency_result["risk_score"] = max(
                        consistency_result.get("risk_score", 0), 75
                    )
                    consistency_result["flags"] = list(
                        consistency_result.get("flags", [])
                    ) + [{
                        "type": "face_mismatch",
                        "severity": "critical",
                        "description": f"Face verification: photos do not appear to be the same person (confidence {confidence:.0f}%).",
                        "confidence": confidence / 100,
                    }]
            except Exception as e:
                logger.warning(f"Face verification failed: {e}")
                face_result = {"performed": False, "error": str(e)}

        # Phase 3: Scoring (local — no LLM call), with ID-specific weights and consistency
        incon_result = {
            "inconsistencies": template_result.get("flags", []),
            "inconsistency_score": template_result.get("risk_score", 0),
        }
        pattern_result = {
            "matched_patterns": metadata_result.get("flags", []),
            "pattern_risk_score": metadata_result.get("risk_score", 0),
        }

        score_result = await self.scoring_agent.calculate_score(
            doc_data, incon_result, pattern_result, None, deepfake_result,
            id_consistency_results=consistency_result,
        )

        # Phase 4: Reasoning + Narrative in PARALLEL
        doc_result_ctx = {"claim_data": doc_data, "raw_text": raw_text}
        reasoning_task = self.scoring_agent.generate_reasoning(score_result)
        narrative_task = self.narrative_agent.generate(
            doc_result_ctx, score_result, incon_result, pattern_result, None,
            report_type="id_verification",
        )
        reasoning, narrative_result = await asyncio.gather(
            reasoning_task, narrative_task, return_exceptions=True
        )
        if isinstance(reasoning, Exception):
            reasoning = f"Document scored {score_result.get('overall_score', 0):.0f}/100."
        if isinstance(narrative_result, Exception):
            narrative_result = {"status": "error", "full_narrative": ""}
        score_result["reasoning"] = reasoning

        authenticity_score = 100 - score_result.get("overall_score", 0)

        logger.info(f"ID verification complete. Authenticity: {authenticity_score}/100")

        return IDVerificationResult(
            document_data=doc_data,
            authenticity_score=score_result.get("overall_score", 0),
            risk_level=score_result.get("risk_level", "unknown"),
            recommendation=score_result.get("recommendation", ""),
            narrative=narrative_result.get("full_narrative", ""),
            deepfake_analysis=deepfake_result,
            template_analysis=template_result,
            metadata_analysis=metadata_result,
            consistency_analysis=consistency_result,
            face_verification=face_result,
            scoring_details=score_result,
            image_paths=image_paths,
            raw_text=raw_text,
        )
