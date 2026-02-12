"""
Eligibility & Duplicate Verification Agent
Flags overlapping claims for the same patient, date, or provider
to prevent double-billing and verify claim eligibility.
Powered by NVIDIA NIM.
"""

import json
from typing import Dict, Any
from loguru import logger

from core.nim_client import get_nim_client


class EligibilityAgent:
    """
    Verifies claim eligibility and detects duplicate / overlapping billing.

    Checks:
    - Duplicate claims: same patient + date + provider + procedure
    - Overlapping services: concurrent treatments that conflict
    - Eligibility gaps: patient not covered on date of service
    - Provider validity: billing provider credentials and enrollment status
    - Coordination of benefits: other insurance primary payer issues
    """

    def __init__(self):
        self.nim_client = get_nim_client()
        logger.info("EligibilityAgent initialized with NVIDIA NIM")

    async def analyze(self, claim_data: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
        """
        Analyze medical claim for eligibility and duplicate billing.

        Args:
            claim_data: Structured medical claim data
            raw_text: Raw text from claim documents

        Returns:
            Dict with eligibility flags, risk score, and summary
        """
        logger.info("EligibilityAgent analyzing claim")

        prompt = f"""You are an expert medical claims auditor specializing in eligibility verification and duplicate detection.

CLAIM DATA:
{json.dumps(claim_data, indent=2, default=str)[:4000]}

DOCUMENT TEXT:
{raw_text[:4000]}

Perform these checks:

1. **DUPLICATE CLAIMS**: Look for signs of the same service being billed multiple times:
   - Same CPT code on the same date of service
   - Very similar claims with minor variations (different modifiers, slightly different dates)
   - Multiple claims for the same encounter

2. **OVERLAPPING SERVICES**: Identify conflicting concurrent treatments:
   - Two procedures that cannot physically be performed at the same time
   - Inpatient and outpatient services on the same date
   - Contradictory treatment locations

3. **ELIGIBILITY CONCERNS**: Flag potential eligibility issues:
   - Dates of service outside policy coverage periods
   - Services not covered under the plan type
   - Pre-authorization requirements not met

4. **PROVIDER VALIDITY**: Check for provider-related red flags:
   - Rendering provider different from billing provider without explanation
   - Out-of-network billing for in-network required services
   - Unusual provider-patient geographic distance

5. **COORDINATION OF BENEFITS**: Check for multi-payer issues:
   - Signs that another insurer should be primary
   - Workers' comp or auto insurance should be primary payer

Return a JSON object:
{{
    "eligible": true/false,
    "risk_score": 0-100,
    "flags": [
        {{
            "type": "duplicate|overlap|eligibility_gap|provider_issue|coordination",
            "severity": "critical|high|medium|low",
            "description": "detailed explanation",
            "confidence": 0.0-1.0
        }}
    ],
    "duplicate_indicators": {{
        "potential_duplicates": 0,
        "details": "description if any"
    }},
    "summary": "brief narrative of findings"
}}"""

        try:
            response = await self.nim_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
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
                    "eligible": False,
                    "risk_score": 50,
                    "flags": [],
                    "summary": response[:500],
                    "_raw_response": True,
                }

            logger.info(f"Eligibility complete. Risk score: {result.get('risk_score', 0)}")
            return result

        except Exception as e:
            logger.error(f"EligibilityAgent error: {e}")
            return {
                "eligible": False,
                "risk_score": 0,
                "flags": [],
                "summary": f"Analysis error: {str(e)}",
                "error": str(e),
            }
