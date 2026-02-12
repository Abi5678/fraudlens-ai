"""
ContentScan AI - AI Content Detection Orchestrator
Detects AI-generated text and images in submitted content.
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from agents.text_gen_detector import TextGenDetectorAgent
from agents.ai_image_detector import AIImageDetectorAgent
from agents.metadata_agent import MetadataAgent


@dataclass
class AIContentResult:
    """Complete AI content detection result."""
    ai_probability: float
    risk_score: float
    risk_level: str
    recommendation: str
    text_analysis: Dict[str, Any]
    image_analysis: Dict[str, Any]
    metadata_analysis: Dict[str, Any]
    content_type: str = "mixed"  # text, image, mixed
    raw_text: str = ""
    image_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ai_probability": self.ai_probability,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "content_type": self.content_type,
            "text_analysis": self.text_analysis,
            "image_analysis": self.image_analysis,
            "metadata_analysis": self.metadata_analysis,
            "images_analyzed": len(self.image_paths),
        }


class ContentScanAI:
    """Orchestrator for AI content detection."""

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key

        self.text_detector = TextGenDetectorAgent()
        self.image_detector = AIImageDetectorAgent()
        self.metadata_agent = MetadataAgent()

        logger.info("ContentScan AI initialized")

    async def analyze(self, text: str = "", image_paths: List[str] = None) -> AIContentResult:
        """Analyze content for AI generation.

        Args:
            text: Text content to analyze.
            image_paths: Image file paths to analyze.

        Returns:
            AIContentResult with detection details.
        """
        image_paths = image_paths or []
        has_text = bool(text and text.strip())
        has_images = bool(image_paths)

        content_type = "mixed" if (has_text and has_images) else ("text" if has_text else "image")
        logger.info(f"Starting AI content detection (type={content_type})")

        tasks = []
        task_names = []

        if has_text:
            tasks.append(self.text_detector.analyze(text))
            task_names.append("text")

        if has_images:
            tasks.append(self.image_detector.analyze(image_paths))
            task_names.append("image")
            tasks.append(self.metadata_agent.analyze(image_paths))
            task_names.append("metadata")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse results
        text_result = {}
        image_result = {}
        metadata_result = {}

        for i, name in enumerate(task_names):
            r = results[i] if not isinstance(results[i], Exception) else {}
            if name == "text":
                text_result = r or {"ai_probability": 0, "risk_score": 0, "indicators": []}
            elif name == "image":
                image_result = r or {"overall_ai_probability": 0, "risk_score": 0, "individual_results": []}
            elif name == "metadata":
                metadata_result = r or {"risk_score": 0, "flags": []}

        # Calculate overall score
        scores = []
        if has_text:
            scores.append(text_result.get("ai_probability", 0) * 100)
        if has_images:
            scores.append(image_result.get("overall_ai_probability", 0) * 100)

        overall_score = sum(scores) / max(len(scores), 1)
        ai_probability = overall_score / 100

        # Determine risk level
        if overall_score >= 75:
            risk_level = "critical"
            recommendation = "Content is very likely AI-generated. Manual verification strongly recommended."
        elif overall_score >= 50:
            risk_level = "high"
            recommendation = "Content shows significant AI generation indicators. Further review needed."
        elif overall_score >= 25:
            risk_level = "medium"
            recommendation = "Some AI indicators detected. Consider additional verification."
        else:
            risk_level = "low"
            recommendation = "Content appears to be human-created. Low risk of AI generation."

        logger.info(f"AI content detection complete. Score: {overall_score:.0f}/100")

        return AIContentResult(
            ai_probability=ai_probability,
            risk_score=overall_score,
            risk_level=risk_level,
            recommendation=recommendation,
            text_analysis=text_result,
            image_analysis=image_result,
            metadata_analysis=metadata_result,
            content_type=content_type,
            raw_text=text,
            image_paths=image_paths,
        )
