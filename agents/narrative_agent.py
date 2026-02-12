"""
Narrative Agent - Generates comprehensive fraud investigation narratives
"""

from typing import Dict, Any
from loguru import logger
from core.nim_client import get_nim_client


class NarrativeAgent:
    """Generates AI-powered fraud investigation narratives."""
    
    def __init__(self):
        self.nim_client = get_nim_client()
    
    async def generate(self, claim_data: Dict, fraud_score: Dict, inconsistencies: Dict, pattern_matches: Dict, network_analysis: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive fraud investigation narrative."""
        try:
            prompt = self._build_prompt(claim_data, fraud_score, inconsistencies, pattern_matches, network_analysis)
            narrative = await self.nim_client.chat(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1000)
            
            return {"status": "success", "full_narrative": narrative, "sections": self._parse_sections(narrative)}
        except Exception as e:
            logger.error(f"NarrativeAgent error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _build_prompt(self, claim_data: Dict, fraud_score: Dict, inconsistencies: Dict, pattern_matches: Dict, network_analysis: Dict) -> str:
        claim = claim_data.get("claim_data", claim_data)
        return f"""Generate a professional fraud investigation report.

CLAIM: {claim.get('claimant', {}).get('name', 'Unknown')} - ${claim.get('claim', {}).get('amount', 0):,.2f}
FRAUD SCORE: {fraud_score.get('overall_score', 0)}/100 ({fraud_score.get('risk_level', 'unknown').upper()})
INCONSISTENCIES: {len(inconsistencies.get('inconsistencies', []))} detected
PATTERN MATCHES: {len(pattern_matches.get('matched_patterns', []))} found

Write sections: EXECUTIVE SUMMARY, KEY FINDINGS, RED FLAGS, RECOMMENDED ACTIONS."""
    
    def _parse_sections(self, narrative: str) -> Dict[str, str]:
        return {"full_text": narrative}
