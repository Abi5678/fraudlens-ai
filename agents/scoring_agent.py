"""
Scoring Agent
Calculates overall fraud risk score with explainable reasoning
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from loguru import logger

from core.nim_client import get_nim_client


@dataclass 
class RiskFactor:
    """Individual risk factor contributing to fraud score"""
    name: str
    score: float
    weight: float
    description: str
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "weighted_score": self.score * self.weight,
            "description": self.description,
            "evidence": self.evidence,
        }


@dataclass
class FraudScore:
    """Complete fraud score assessment"""
    overall_score: float
    risk_level: str
    confidence: float
    risk_factors: List[RiskFactor] = field(default_factory=list)
    recommendation: str = ""
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "risk_factors": [f.to_dict() for f in self.risk_factors],
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
        }


class ScoringAgent:
    """Calculates overall fraud risk score combining all agent outputs."""
    
    WEIGHTS = {
        "inconsistency": 0.25,
        "pattern_match": 0.25,
        "network_risk": 0.20,
        "document_quality": 0.10,
        "claim_characteristics": 0.15,
        "deepfake": 0.05,
    }
    
    def __init__(self):
        self.nim_client = get_nim_client()
        logger.info("ScoringAgent initialized")
    
    async def calculate_score(
        self,
        claim_data: Dict[str, Any],
        inconsistency_results: Dict[str, Any],
        pattern_results: Dict[str, Any],
        network_results: Dict[str, Any] = None,
        deepfake_results: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Calculate overall fraud score locally (no LLM call).

        The reasoning LLM call is separated out so orchestrators
        can run it in parallel with narrative generation.
        """
        logger.info("ScoringAgent calculating fraud score")

        try:
            risk_factors = []

            # Inconsistency Score
            risk_factors.append(RiskFactor(
                name="Inconsistencies",
                score=inconsistency_results.get("inconsistency_score", 0),
                weight=self.WEIGHTS["inconsistency"],
                description=inconsistency_results.get("summary", ""),
                evidence=[i.get("description", "") for i in inconsistency_results.get("inconsistencies", [])[:3]],
            ))

            # Pattern Match Score
            risk_factors.append(RiskFactor(
                name="Fraud Pattern Match",
                score=pattern_results.get("pattern_risk_score", 0),
                weight=self.WEIGHTS["pattern_match"],
                description=pattern_results.get("summary", ""),
                evidence=[p.get("pattern_name", "") for p in pattern_results.get("matched_patterns", [])[:3]],
            ))

            # Network Risk Score
            if network_results:
                risk_factors.append(RiskFactor(
                    name="Network/Ring Risk",
                    score=network_results.get("network_risk_score", 0),
                    weight=self.WEIGHTS["network_risk"],
                    description=network_results.get("summary", ""),
                    evidence=network_results.get("connections", [])[:3],
                ))

            # Deepfake / Image Authenticity Score
            if deepfake_results and deepfake_results.get("status") == "success":
                df_score = deepfake_results.get("manipulation_score", 0)
                df_detections = deepfake_results.get("detections", [])
                risk_factors.append(RiskFactor(
                    name="Image Authenticity",
                    score=df_score,
                    weight=self.WEIGHTS["deepfake"],
                    description=deepfake_results.get("summary", ""),
                    evidence=[d.replace("_", " ").title() for d in df_detections[:3]],
                ))

            # Claim Characteristics (local — no LLM)
            claim_score = self._score_claim_characteristics_local(claim_data)
            risk_factors.append(RiskFactor(
                name="Claim Characteristics",
                score=claim_score["score"],
                weight=self.WEIGHTS["claim_characteristics"],
                description=claim_score["description"],
                evidence=claim_score["flags"],
            ))

            # Calculate weighted overall score
            overall_score = sum(f.score * f.weight for f in risk_factors)
            risk_level = self._get_risk_level(overall_score)
            confidence = self._calculate_confidence(risk_factors)
            recommendation = self._get_recommendation(risk_level)

            return {
                "status": "success",
                "overall_score": round(overall_score, 1),
                "risk_level": risk_level,
                "confidence": round(confidence, 2),
                "risk_factors": [f.to_dict() for f in risk_factors],
                "recommendation": recommendation,
                "reasoning": "",
            }

        except Exception as e:
            logger.error(f"ScoringAgent error: {e}")
            return {"status": "error", "error": str(e), "overall_score": 0, "risk_level": "unknown"}
    
    def _score_claim_characteristics_local(self, claim_data: Dict) -> Dict[str, Any]:
        """Score claim based on inherent characteristics (no LLM)."""
        flags = []
        score = 0
        
        amount = claim_data.get("claim", {}).get("amount", 0)
        if amount > 100000:
            score += 20
            flags.append(f"High claim amount: ${amount:,.2f}")
        
        injuries = claim_data.get("medical", {}).get("injuries", [])
        soft_tissue = ["whiplash", "soft tissue", "strain", "sprain", "neck pain", "back pain"]
        for injury in injuries:
            if any(kw in injury.lower() for kw in soft_tissue):
                score += 15
                flags.append(f"Soft tissue injury: {injury}")
                break
        
        return {"score": min(score, 100), "description": f"{len(flags)} risk flags", "flags": flags}
    
    def _get_risk_level(self, score: float) -> str:
        if score >= 75: return "critical"
        elif score >= 50: return "high"
        elif score >= 25: return "medium"
        return "low"
    
    def _calculate_confidence(self, risk_factors: List[RiskFactor]) -> float:
        scores = [f.score for f in risk_factors if f.score > 0]
        if not scores: return 0.5
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        return max(0.5, min(0.95, 1 - (variance / 2500)))
    
    def _get_recommendation(self, risk_level: str) -> str:
        recommendations = {
            "critical": "DENY - Refer to SIU immediately",
            "high": "INVESTIGATE - Assign to fraud analyst",
            "medium": "REVIEW - Additional documentation required",
            "low": "APPROVE - Proceed with standard processing",
        }
        return recommendations.get(risk_level, "REVIEW")
    
    async def generate_reasoning(self, score_result: Dict[str, Any]) -> str:
        """Generate LLM reasoning for a score result. Can be called in parallel with narrative."""
        score = score_result.get("overall_score", 0)
        risk_level = score_result.get("risk_level", "unknown")
        risk_factors = score_result.get("risk_factors", [])
        factors_summary = "\n".join([f"- {f.get('name','?')}: {f.get('score',0):.0f}/100" for f in risk_factors])
        prompt = f"""Explain fraud risk assessment in 2 sentences.
SCORE: {score:.0f}/100, LEVEL: {risk_level.upper()}
FACTORS:
{factors_summary}
Be professional and specific."""

        try:
            return await self.nim_client.chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=150)
        except Exception:
            return f"Claim scored {score:.0f}/100 based on {len(risk_factors)} factors."
