"""
Narrative Agent - Generates comprehensive fraud investigation or ID verification narratives
"""

from typing import Dict, Any, Optional
from loguru import logger
from core.nim_client import get_nim_client


class NarrativeAgent:
    """Generates AI-powered fraud investigation or ID verification narratives."""
    
    def __init__(self):
        self.nim_client = get_nim_client()
    
    async def generate(
        self,
        claim_data: Dict,
        fraud_score: Dict,
        inconsistencies: Dict,
        pattern_matches: Dict,
        network_analysis: Dict = None,
        report_type: str = "fraud",
    ) -> Dict[str, Any]:
        """Generate narrative. Use report_type='id_verification' for Photo ID reports."""
        try:
            prompt = self._build_prompt(
                claim_data, fraud_score, inconsistencies, pattern_matches,
                network_analysis, report_type=report_type,
            )
            narrative = await self.nim_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1200,
            )
            return {"status": "success", "full_narrative": narrative, "sections": self._parse_sections(narrative)}
        except Exception as e:
            logger.error(f"NarrativeAgent error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _build_prompt(
        self,
        claim_data: Dict,
        fraud_score: Dict,
        inconsistencies: Dict,
        pattern_matches: Dict,
        network_analysis: Dict = None,
        report_type: str = "fraud",
    ) -> str:
        if report_type == "id_verification":
            return self._build_id_verification_prompt(
                claim_data, fraud_score, inconsistencies, pattern_matches,
            )
        claim = claim_data.get("claim_data", claim_data)
        return f"""Generate a professional fraud investigation report.

CLAIM: {claim.get('claimant', {}).get('name', 'Unknown')} - ${claim.get('claim', {}).get('amount', 0):,.2f}
FRAUD SCORE: {fraud_score.get('overall_score', 0)}/100 ({fraud_score.get('risk_level', 'unknown').upper()})
INCONSISTENCIES: {len(inconsistencies.get('inconsistencies', []))} detected
PATTERN MATCHES: {len(pattern_matches.get('matched_patterns', []))} found

Write sections: EXECUTIVE SUMMARY, KEY FINDINGS, RED FLAGS, RECOMMENDED ACTIONS."""

    def _build_id_verification_prompt(
        self,
        claim_data: Dict,
        fraud_score: Dict,
        inconsistencies: Dict,
        pattern_matches: Dict,
    ) -> str:
        doc = claim_data.get("claim_data", claim_data)
        claimant = doc.get("claimant") or {}
        name = claimant.get("name") if isinstance(claimant, dict) else doc.get("name") or "Unknown"
        if isinstance(name, dict):
            name = name.get("name", "Unknown")
        license_no = doc.get("license_number") or doc.get("id_number") or doc.get("document_number") or "—"
        doc_type = doc.get("document_type") or "ID document"
        jurisdiction = doc.get("issuing_jurisdiction") or "—"
        risk_level = (fraud_score.get("risk_level") or "unknown").upper()
        score = fraud_score.get("overall_score", 0)
        incon_list = inconsistencies.get("inconsistencies", [])
        pattern_list = pattern_matches.get("matched_patterns", [])
        incon_descs = [f.get("description", "")[:80] for f in incon_list[:5]]
        pattern_descs = [f.get("description", f.get("pattern_name", ""))[:80] for f in pattern_list[:5]]
        return f"""CRITICAL: You are writing an **ID VERIFICATION REPORT** for an identity document. This is NOT an insurance fraud report. You must NEVER use fraud/claim language.

BANNED WORDS/PHRASES: claimant (in fraud sense), claim amount, claim denial, SIU, fraud investigation, claim balance, corroborating evidence, transaction patterns, offshore account, policy number, coverage amount.

INPUT DATA:
- Document holder: {name}
- Document type: {doc_type}
- Jurisdiction: {jurisdiction}
- License/ID number: {license_no}
- Risk score: {score:.1f}/100 ({risk_level})
- Plausibility flags ({len(incon_list)}): {'; '.join(incon_descs) if incon_descs else 'None'}
- Template/metadata flags ({len(pattern_list)}): {'; '.join(pattern_descs) if pattern_descs else 'None'}

OUTPUT FORMAT (use exactly these headings):
# ID VERIFICATION REPORT

## Document Holder
{name} — {doc_type} ({jurisdiction})

## Risk Assessment
Score: {score:.1f}/100 — {risk_level}

## Executive Summary
[2-3 sentences: document type, holder, risk level, and whether ID appears authentic or should be rejected. Use ID/authenticity language only.]

## Document Findings
[Bullet list: layout, data consistency, suspicious ID numbers, dates, security features. No insurance terms.]

## Red Flags
[Bullet list of authenticity issues only: placeholder ID, expired, face mismatch, AI-generated signs, etc.]

## Recommendation
[One line: APPROVE / REVIEW / REJECT + brief reason based on ID risk.]"""

    def _parse_sections(self, narrative: str) -> Dict[str, str]:
        return {"full_text": narrative}
