"""
Property Valuation Agent
Analyzes property appraisal documents for mortgage fraud indicators.
Uses NVIDIA NIM for intelligent cross-checking of property values.
"""

import json
from typing import Dict, Any
from loguru import logger


class PropertyValuationAgent:
    """Cross-checks property valuation claims against submitted appraisal documents."""

    def __init__(self):
        from core.nim_client import get_nim_client
        self.nim = get_nim_client()

    async def analyze(self, claim_data: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
        """Analyze property documents for valuation fraud.

        Args:
            claim_data: Structured mortgage application data.
            raw_text: Raw extracted text from documents.

        Returns:
            Dict with property valuation analysis results.
        """
        logger.info("Property Valuation Agent: starting analysis")

        prompt = f"""You are an expert mortgage fraud investigator specializing in property appraisal fraud.

Analyze the following mortgage application and appraisal data for property valuation fraud indicators.

APPLICATION DATA:
{json.dumps(claim_data, indent=2, default=str)[:3000]}

DOCUMENT TEXT:
{raw_text[:3000]}

Check for these fraud indicators:
1. **Inflated appraisal**: Property value significantly above market comparables
2. **Missing comparables**: Appraisal lacks proper comparable property analysis
3. **Suspicious timing**: Rapid property flips or recent purchases at much lower price
4. **Property description mismatch**: Listed features don't match actual property details
5. **Location discrepancies**: Address or neighborhood description inconsistencies
6. **LTV manipulation**: Loan-to-value ratio manipulated through inflated appraisal
7. **Appraiser concerns**: Signs of appraiser collusion or negligence

Return a JSON object:
{{
    "valuation_verified": true/false,
    "stated_value": "amount or N/A",
    "estimated_fair_value": "amount or N/A",
    "valuation_confidence": 0.0-1.0,
    "risk_score": 0-100,
    "ltv_ratio": "percentage or N/A",
    "flags": [
        {{
            "type": "inflated_appraisal|missing_comparables|suspicious_timing|description_mismatch|location_error|ltv_manipulation|appraiser_concern",
            "severity": "critical|high|medium|low",
            "description": "detailed explanation",
            "confidence": 0.0-1.0
        }}
    ],
    "comparables_analysis": {{
        "provided": 0,
        "adequate": true/false,
        "notes": "analysis of comparables"
    }},
    "summary": "brief narrative of findings"
}}"""

        try:
            response = await self.nim.chat(
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
                    "valuation_verified": False,
                    "risk_score": 50,
                    "flags": [],
                    "summary": response[:500],
                    "_raw_response": True,
                }

            logger.info(f"Property Valuation complete. Risk score: {result.get('risk_score', 0)}")
            return result

        except Exception as e:
            logger.error(f"Property Valuation Agent error: {e}")
            return {
                "valuation_verified": False,
                "risk_score": 0,
                "flags": [],
                "summary": f"Analysis error: {str(e)}",
                "error": str(e),
            }
