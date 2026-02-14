"""
Scoring Agent
Calculates overall fraud risk score with explainable reasoning.
Supports RIGID_SCORING (env): stricter thresholds and conservative recommendations.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from core.nim_client import get_nim_client


def _is_rigid() -> bool:
    """Use strict thresholds and conservative recommendations when True (default: True)."""
    return os.environ.get("RIGID_SCORING", "true").strip().lower() in ("1", "true", "yes")


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
        "inconsistency": 0.30,
        "pattern_match": 0.25,
        "network_risk": 0.20,
        "claim_characteristics": 0.15,
        "deepfake": 0.10,
    }
    # Photo ID flow: heavier weight on consistency and deepfake
    WEIGHTS_ID = {
        "inconsistency": 0.20,
        "pattern_match": 0.20,
        "deepfake": 0.20,
        "id_consistency": 0.30,
        "claim_characteristics": 0.10,
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
        id_consistency_results: Dict[str, Any] = None,
        raw_text: str = "",
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Calculate overall fraud score locally (no LLM call).

        The reasoning LLM call is separated out so orchestrators
        can run it in parallel with narrative generation.
        When id_consistency_results is provided (Photo ID flow), uses WEIGHTS_ID.
        If weights is provided, uses it for keys present; missing keys use default.
        """
        logger.info("ScoringAgent calculating fraud score")
        use_id_weights = id_consistency_results is not None
        base_weights = self.WEIGHTS_ID if use_id_weights else self.WEIGHTS
        if weights is not None:
            weights = {k: weights.get(k, base_weights[k]) for k in base_weights}
        else:
            weights = base_weights
        # Support raw_text from param or from claim_data (e.g. medical flow)
        raw_text = raw_text or claim_data.get("_raw_text", "")

        try:
            risk_factors = []

            # Inconsistency Score
            risk_factors.append(RiskFactor(
                name="Inconsistencies",
                score=inconsistency_results.get("inconsistency_score", 0),
                weight=weights["inconsistency"],
                description=inconsistency_results.get("summary", ""),
                evidence=[i.get("description", "") for i in inconsistency_results.get("inconsistencies", [])[:3]],
            ))

            # Pattern Match Score
            risk_factors.append(RiskFactor(
                name="Fraud Pattern Match",
                score=pattern_results.get("pattern_risk_score", 0),
                weight=weights["pattern_match"],
                description=pattern_results.get("summary", ""),
                evidence=[p.get("pattern_name", "") for p in pattern_results.get("matched_patterns", [])[:3]],
            ))

            # Network Risk Score (not used in ID flow)
            if network_results and not use_id_weights:
                risk_factors.append(RiskFactor(
                    name="Network/Ring Risk",
                    score=network_results.get("network_risk_score", 0),
                    weight=weights["network_risk"],
                    description=network_results.get("summary", ""),
                    evidence=network_results.get("connections", [])[:3],
                ))

            # Deepfake / Image Authenticity Score (in ID flow, include AI-generated score).
            # When status is success: use real score. When skipped/error (e.g. NIM 404): add factor with score 0 so total weight and scale match previous behavior (score ~30 for same doc).
            if deepfake_results and not use_id_weights:
                df_status = deepfake_results.get("status", "")
                df_score = deepfake_results.get("manipulation_score", 0) if df_status == "success" else 0
                df_detections = deepfake_results.get("detections", []) if df_status == "success" else []
                df_summary = deepfake_results.get("summary", "") if df_status == "success" else (deepfake_results.get("message", "Skipped") or "Image model unavailable")
                risk_factors.append(RiskFactor(
                    name="Image Authenticity",
                    score=df_score,
                    weight=weights["deepfake"],
                    description=df_summary,
                    evidence=[d.replace("_", " ").title() for d in df_detections[:3]],
                ))
            elif deepfake_results and use_id_weights:
                df_status = deepfake_results.get("status", "")
                if df_status == "success":
                    df_score = deepfake_results.get("manipulation_score", 0)
                    if "ai_generated_score" in deepfake_results:
                        df_score = max(df_score, deepfake_results.get("ai_generated_score", 0))
                    df_detections = deepfake_results.get("detections", [])
                    if deepfake_results.get("ai_generated_detected"):
                        df_detections = ["ai_generated"] + [d for d in df_detections if d != "ai_generated"]
                else:
                    df_score = 0
                    df_detections = []
                risk_factors.append(RiskFactor(
                    name="Image Authenticity",
                    score=df_score,
                    weight=weights["deepfake"],
                    description=deepfake_results.get("summary", "") if df_status == "success" else "Skipped",
                    evidence=[d.replace("_", " ").title() for d in df_detections[:3]],
                ))

            # ID Consistency (Photo ID flow only)
            if id_consistency_results is not None:
                id_risk = id_consistency_results.get("risk_score", 0)
                id_flags = id_consistency_results.get("flags", [])
                risk_factors.append(RiskFactor(
                    name="ID Plausibility",
                    score=id_risk,
                    weight=weights["id_consistency"],
                    description=id_consistency_results.get("summary", ""),
                    evidence=[f.get("description", "")[:80] for f in id_flags[:3]],
                ))

            # Claim Characteristics (local — no LLM, uses raw_text when claim_data sparse)
            claim_score = self._score_claim_characteristics_local(claim_data, raw_text or "")
            risk_factors.append(RiskFactor(
                name="Claim Characteristics",
                score=claim_score["score"],
                weight=weights["claim_characteristics"],
                description=claim_score["description"],
                evidence=claim_score["flags"],
            ))

            # Renormalize: scale weights so present factors sum to 1.0.
            # This ensures the 0-100 scale is always fully utilized regardless
            # of which agents ran (e.g. no images → no deepfake factor).
            total_weight = sum(f.weight for f in risk_factors)
            if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
                scale = 1.0 / total_weight
                for f in risk_factors:
                    f.weight = f.weight * scale
            overall_score = sum(f.score * f.weight for f in risk_factors)
            # Rigid mode: small upward nudge for ID flow so borderline cases tip to next level
            if _is_rigid() and use_id_weights and overall_score > 0 and overall_score < 50:
                overall_score = min(100, overall_score + 5)
            risk_level = self._get_risk_level(overall_score, use_id_weights)
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
    
    def _score_claim_characteristics_local(self, claim_data: Dict, raw_text: str = "") -> Dict[str, Any]:
        """Score claim based on inherent characteristics (no LLM). Uses raw_text when claim_data is sparse."""
        flags = []
        score = 0

        amount = claim_data.get("claim", {}).get("amount", 0)
        if amount and amount > 0:
            if amount > 100000:
                score += 20
                flags.append(f"High claim amount: ${amount:,.2f}")
        elif raw_text:
            # Fallback: scan raw_text for amounts when claim_data is sparse
            import re
            amount_matches = re.findall(r'\$[\d,]+(?:\.\d{2})?', raw_text[:5000])
            for m in amount_matches:
                try:
                    val = float(m.replace("$", "").replace(",", ""))
                    if val > 100000:
                        score += 15
                        flags.append(f"High amount mentioned in document: ${val:,.2f}")
                        break
                except (ValueError, TypeError):
                    pass

        injuries = claim_data.get("medical", {}).get("injuries", []) or []
        if not isinstance(injuries, list):
            injuries = [injuries] if injuries else []
        soft_tissue = ["whiplash", "soft tissue", "strain", "sprain", "neck pain", "back pain"]
        for injury in injuries:
            if injury and any(kw in str(injury).lower() for kw in soft_tissue):
                score += 15
                flags.append(f"Soft tissue injury: {injury}")
                break
        if not flags and raw_text:
            txt_lower = raw_text[:4000].lower()
            if any(kw in txt_lower for kw in soft_tissue):
                score += 10
                flags.append("Soft tissue injury mentioned in document")

        return {"score": min(score, 100), "description": f"{len(flags)} risk flags" if flags else "No inherent risk flags", "flags": flags}
    
    def _get_risk_level(self, score: float, is_id_flow: bool = False) -> str:
        """Stricter bands when RIGID_SCORING: harder to get 'low', easier to get 'high'/'critical'."""
        if _is_rigid():
            if is_id_flow:
                # Photo ID: very strict — only very low scores get low risk
                if score >= 62: return "critical"
                if score >= 35: return "high"
                if score >= 12: return "medium"
                return "low"
            # General: stricter than default
            if score >= 68: return "critical"
            if score >= 42: return "high"
            if score >= 18: return "medium"
            return "low"
        # Default (non-rigid)
        if score >= 75: return "critical"
        elif score >= 50: return "high"
        elif score >= 25: return "medium"
        return "low"
    
    def _calculate_confidence(self, risk_factors: List[RiskFactor]) -> float:
        """Confidence in the assessment. Higher when factors agree; lower when they diverge."""
        all_scores = [f.score for f in risk_factors]
        if not all_scores:
            return 0.5
        # When all scores are 0 (or near 0), agents agree on low risk → high confidence
        active_scores = [s for s in all_scores if s > 0]
        if not active_scores:
            return 0.85  # All agents agree: no risk
        avg = sum(active_scores) / len(active_scores)
        variance = sum((s - avg) ** 2 for s in active_scores) / len(active_scores)
        return max(0.5, min(0.95, 1 - (variance / 2500)))
    
    def _get_recommendation(self, risk_level: str) -> str:
        if _is_rigid():
            recommendations = {
                "critical": "DENY - Refer to SIU immediately",
                "high": "INVESTIGATE - Assign to fraud analyst",
                "medium": "REVIEW - Additional documentation required",
                "low": "REVIEW - Verify before approval (rigid mode)",
            }
        else:
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
