"""
Income Verification Agent
Analyzes income documents (pay stubs, W-2s, tax returns) for mortgage applications.
Uses NVIDIA NIM for intelligent document cross-referencing.
"""

import json
from typing import Dict, Any
from loguru import logger


class IncomeVerificationAgent:
    """Verifies income claims against submitted documentation."""

    def __init__(self):
        from core.nim_client import get_nim_client
        self.nim = get_nim_client()

    async def analyze(self, claim_data: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
        """Analyze income documents for inconsistencies and fraud indicators.

        Args:
            claim_data: Structured mortgage application data.
            raw_text: Raw extracted text from documents.

        Returns:
            Dict with income verification results.
        """
        logger.info("Income Verification Agent: starting analysis")

        prompt = f"""You are an expert mortgage fraud investigator specializing in income verification.

Analyze the following mortgage application data and documents for income-related fraud indicators.

APPLICATION DATA:
{json.dumps(claim_data, indent=2, default=str)[:3000]}

DOCUMENT TEXT:
{raw_text[:3000]}

Check for these fraud indicators:
1. **Income inflation**: Stated income significantly higher than industry averages for the stated occupation
2. **Employer mismatch**: Employer details don't match pay stub or W-2 information
3. **Date inconsistencies**: Pay periods, employment dates, or tax years that don't align
4. **Mathematical errors**: Gross pay, deductions, or net pay that don't add up
5. **Format anomalies**: Pay stubs or W-2s that look irregular (unusual formatting, missing standard fields)
6. **Employment gaps**: Unexplained gaps in employment history
7. **Multiple income sources**: Unreported or suspicious additional income streams

Return a JSON object with this structure:
{{
    "income_verified": true/false,
    "stated_income": "amount or N/A",
    "verified_income": "amount or N/A",
    "income_match_percentage": 0-100,
    "risk_score": 0-100,
    "flags": [
        {{
            "type": "income_inflation|employer_mismatch|date_error|math_error|format_anomaly|employment_gap",
            "severity": "critical|high|medium|low",
            "description": "detailed explanation",
            "confidence": 0.0-1.0
        }}
    ],
    "employer_verification": {{
        "name": "stated employer",
        "verified": true/false,
        "notes": "any discrepancies"
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
                # Try to extract JSON from response
                text = response.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                result = json.loads(text)
            except (json.JSONDecodeError, IndexError):
                result = {
                    "income_verified": False,
                    "risk_score": 50,
                    "income_match_percentage": 50,
                    "flags": [],
                    "summary": response[:500],
                    "_raw_response": True,
                }

            logger.info(f"Income Verification complete. Risk score: {result.get('risk_score', 0)}")
            return result

        except Exception as e:
            logger.error(f"Income Verification Agent error: {e}")
            return {
                "income_verified": False,
                "risk_score": 0,
                "income_match_percentage": 0,
                "flags": [],
                "summary": f"Analysis error: {str(e)}",
                "error": str(e),
            }
