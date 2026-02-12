"""
FraudLens AI - Main Orchestrator
Coordinates all agents to analyze insurance claims for fraud
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from agents import (
    DocumentAgent, InconsistencyAgent, PatternAgent,
    ScoringAgent, NarrativeAgent, NetworkAgent, DeepfakeAgent,
)


@dataclass
class FraudAnalysisResult:
    """Complete fraud analysis result"""
    claim_data: Dict[str, Any]
    fraud_score: float
    risk_level: str
    recommendation: str
    narrative: str
    inconsistencies: Dict[str, Any]
    pattern_matches: Dict[str, Any]
    network_analysis: Dict[str, Any]
    deepfake_analysis: Dict[str, Any]
    scoring_details: Dict[str, Any] = field(default_factory=dict)
    extracted_images: list = field(default_factory=list)
    raw_text: str = ""

    @property
    def fraud_ring_detected(self) -> bool:
        return self.network_analysis.get("fraud_ring_detected", False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_data": self.claim_data,
            "fraud_score": self.fraud_score,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "narrative": self.narrative,
            "fraud_ring_detected": self.fraud_ring_detected,
            "inconsistencies": self.inconsistencies,
            "pattern_matches": self.pattern_matches,
            "network_analysis": self.network_analysis,
            "deepfake_analysis": self.deepfake_analysis,
            "scoring_details": self.scoring_details,
            "extracted_images_count": len(self.extracted_images),
        }


class FraudLensAI:
    """Main orchestrator for FraudLens AI multi-agent system."""

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key

        self.document_agent = DocumentAgent()
        self.inconsistency_agent = InconsistencyAgent()
        self.pattern_agent = PatternAgent()
        self.scoring_agent = ScoringAgent()
        self.narrative_agent = NarrativeAgent()
        self.network_agent = NetworkAgent()
        self.deepfake_agent = DeepfakeAgent()

        logger.info("FraudLens AI initialized with NVIDIA tech stack")

    async def analyze_json(self, claim_json_path: str, image_paths: list = None,
                           include_network: bool = True,
                           include_deepfake: bool = True) -> "FraudAnalysisResult":
        """Analyze a claim from a JSON file (bypasses document extraction)."""
        logger.info(f"Starting fraud analysis from JSON: {claim_json_path}")
        with open(claim_json_path, "r") as f:
            data = json.load(f)

        claim_data = data.get("claim_data", data)
        raw_text = data.get("raw_text", json.dumps(claim_data, indent=2))

        return await self._run_analysis(
            claim_data, raw_text, image_paths or [], include_network, include_deepfake
        )

    async def analyze(self, document_path: str, image_paths: list = None,
                      include_network: bool = True,
                      include_deepfake: bool = True) -> "FraudAnalysisResult":
        """Analyze an insurance claim document for fraud."""
        logger.info(f"Starting fraud analysis for: {document_path}")

        path = Path(document_path)

        # Support direct JSON input
        if path.suffix.lower() == ".json":
            return await self.analyze_json(
                document_path, image_paths, include_network, include_deepfake
            )

        # Phase 1: Document Extraction (now also extracts images)
        doc_result = await self.document_agent.process(document_path)
        claim_data = doc_result.get("claim_data", {})
        raw_text = doc_result.get("raw_text", "")

        # Collect image paths: user-provided + extracted from PDF
        all_image_paths = list(image_paths or [])
        extracted_images = doc_result.get("extracted_images", [])
        all_image_paths.extend(extracted_images)

        logger.info(f"Total images for analysis: {len(all_image_paths)} ({len(extracted_images)} extracted from document)")

        result = await self._run_analysis(
            claim_data, raw_text, all_image_paths, include_network, include_deepfake
        )
        result.extracted_images = extracted_images
        result.raw_text = raw_text
        return result

    async def _run_analysis(self, claim_data: Dict, raw_text: str,
                            image_paths: list = None,
                            include_network: bool = True,
                            include_deepfake: bool = True) -> "FraudAnalysisResult":
        """Core analysis pipeline shared by all entry points."""
        # Phase 2: Parallel Analysis
        tasks = [
            self.inconsistency_agent.analyze(claim_data, raw_text),
            self.pattern_agent.analyze(claim_data, raw_text),
        ]
        if include_network:
            tasks.append(
                self.network_agent.analyze(
                    {"claim_data": claim_data, "raw_text": raw_text}
                )
            )
        # Deepfake runs on any available images (extracted from PDF or user-provided)
        if include_deepfake and image_paths:
            tasks.append(self.deepfake_agent.analyze(image_paths))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        inconsistency_result = (
            results[0] if not isinstance(results[0], Exception)
            else {"inconsistencies": [], "inconsistency_score": 0}
        )
        pattern_result = (
            results[1] if not isinstance(results[1], Exception)
            else {"matched_patterns": [], "pattern_risk_score": 0}
        )

        idx = 2
        network_result = None
        deepfake_result = None
        if include_network:
            network_result = (
                results[idx]
                if idx < len(results) and not isinstance(results[idx], Exception)
                else None
            )
            idx += 1
        if include_deepfake and image_paths:
            deepfake_result = (
                results[idx]
                if idx < len(results) and not isinstance(results[idx], Exception)
                else None
            )

        # Phase 3: Scoring (local — no LLM call)
        score_result = await self.scoring_agent.calculate_score(
            claim_data, inconsistency_result, pattern_result,
            network_result, deepfake_result,
        )

        # Phase 4: Reasoning + Narrative in PARALLEL (saves ~3-5s)
        doc_result = {"claim_data": claim_data, "raw_text": raw_text}
        reasoning_task = self.scoring_agent.generate_reasoning(score_result)
        narrative_task = self.narrative_agent.generate(
            doc_result, score_result, inconsistency_result,
            pattern_result, network_result,
        )
        reasoning, narrative_result = await asyncio.gather(
            reasoning_task, narrative_task, return_exceptions=True
        )

        # Handle exceptions from parallel tasks
        if isinstance(reasoning, Exception):
            reasoning = f"Claim scored {score_result.get('overall_score', 0):.0f}/100."
        if isinstance(narrative_result, Exception):
            narrative_result = {"status": "error", "full_narrative": ""}

        score_result["reasoning"] = reasoning

        logger.info(
            f"Analysis complete. Fraud Score: {score_result.get('overall_score', 0)}/100"
        )

        return FraudAnalysisResult(
            claim_data=claim_data,
            fraud_score=score_result.get("overall_score", 0),
            risk_level=score_result.get("risk_level", "unknown"),
            recommendation=score_result.get("recommendation", ""),
            narrative=narrative_result.get("full_narrative", ""),
            inconsistencies=inconsistency_result,
            pattern_matches=pattern_result,
            network_analysis=network_result or {},
            deepfake_analysis=deepfake_result or {},
            scoring_details=score_result,
            extracted_images=image_paths or [],
            raw_text=raw_text,
        )


async def analyze_claim(document_path: str, **kwargs) -> FraudAnalysisResult:
    """Quick function to analyze a single claim"""
    detector = FraudLensAI()
    return await detector.analyze(document_path, **kwargs)
