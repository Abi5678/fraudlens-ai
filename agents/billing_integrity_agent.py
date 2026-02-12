"""
Billing Integrity Agent
Cross-verifies CPT/HCPCS/ICD-10 codes against clinical notes to detect
upcoding, unbundling, and services-not-rendered in medical insurance claims.
Powered by NVIDIA NIM.
"""

import json
from typing import Dict, Any, List
from dataclasses import dataclass, field
from loguru import logger

from core.nim_client import get_nim_client


@dataclass
class BillingFlag:
    """A single billing anomaly detected."""
    type: str       # upcoding, unbundling, services_not_rendered, duplicate, code_mismatch
    severity: str   # critical, high, medium, low
    description: str
    confidence: float
    codes_involved: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "description": self.description,
            "confidence": self.confidence,
            "codes_involved": self.codes_involved,
            "evidence": self.evidence,
        }


class BillingIntegrityAgent:
    """
    Detects billing fraud in medical insurance claims by cross-referencing
    CPT/HCPCS procedure codes, ICD-10 diagnosis codes, and clinical notes.

    Checks:
    - Upcoding: billing for more expensive service than provided
    - Unbundling: splitting bundled procedures into separate codes for higher payouts
    - Services not rendered: billing for procedures not documented in clinical notes
    - Duplicate billing: same service billed multiple times
    - Code mismatch: diagnosis codes that don't justify procedures billed
    """

    def __init__(self):
        self.nim_client = get_nim_client()
        logger.info("BillingIntegrityAgent initialized with NVIDIA NIM")

    async def analyze(self, claim_data: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
        """
        Analyze medical claim for billing integrity issues.

        Args:
            claim_data: Structured medical claim data
            raw_text: Raw text from claim documents / clinical notes

        Returns:
            Dict with billing flags, risk score, and summary
        """
        logger.info("BillingIntegrityAgent analyzing claim")

        prompt = f"""You are an expert medical billing fraud investigator. Analyze this medical insurance claim for billing integrity issues.

CLAIM DATA:
{json.dumps(claim_data, indent=2, default=str)[:4000]}

CLINICAL NOTES / DOCUMENT TEXT:
{raw_text[:5000]}

Perform these checks:

1. **UPCODING**: Are any CPT/HCPCS codes billed at a higher level than clinical documentation supports?
   - Example: Billing 99215 (high-complexity visit) when notes only support 99213 (moderate)
   - Look at E/M levels, procedure complexity, and documented medical necessity

2. **UNBUNDLING**: Are procedures billed separately that should be bundled under a single code?
   - Example: Billing individual lab tests separately instead of using a panel code
   - Check for CCI (Correct Coding Initiative) edit violations

3. **SERVICES NOT RENDERED**: Are there billed procedures with NO supporting clinical documentation?
   - Cross-reference every billed code against the clinical notes

4. **DUPLICATE BILLING**: Is the same service billed more than once for the same date/encounter?
   - Check for identical codes on the same date of service

5. **CODE MISMATCH**: Do the ICD-10 diagnosis codes medically justify the CPT procedure codes?
   - Example: Billing for cardiac catheterization with a diagnosis of common cold

Return a JSON object:
{{
    "billing_verified": true/false,
    "risk_score": 0-100,
    "flags": [
        {{
            "type": "upcoding|unbundling|services_not_rendered|duplicate|code_mismatch",
            "severity": "critical|high|medium|low",
            "description": "detailed explanation",
            "confidence": 0.0-1.0,
            "codes_involved": ["CPT/ICD codes"],
            "evidence": ["supporting evidence from notes"]
        }}
    ],
    "codes_analyzed": {{
        "cpt_codes": ["list of CPT codes found"],
        "icd10_codes": ["list of ICD-10 codes found"],
        "total_billed": "dollar amount or N/A"
    }},
    "summary": "brief narrative of findings"
}}"""

        try:
            response = await self.nim_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )

            try:
                text = response.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                result = json.loads(text)
            except (json.JSONDecodeError, IndexError):
                result = {
                    "billing_verified": False,
                    "risk_score": 50,
                    "flags": [],
                    "summary": response[:500],
                    "_raw_response": True,
                }

            logger.info(f"BillingIntegrity complete. Risk score: {result.get('risk_score', 0)}")
            return result

        except Exception as e:
            logger.error(f"BillingIntegrityAgent error: {e}")
            return {
                "billing_verified": False,
                "risk_score": 0,
                "flags": [],
                "summary": f"Analysis error: {str(e)}",
                "error": str(e),
            }
