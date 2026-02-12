"""
Template Match Agent
Analyzes photo IDs against known document templates (driver's license, passport, etc.).
Uses NVIDIA NIM for intelligent layout and format analysis.
"""

import json
from typing import Dict, Any, List
from loguru import logger


class TemplateMatchAgent:
    """Matches submitted ID documents against known valid templates."""

    def __init__(self):
        from core.nim_client import get_nim_client
        self.nim = get_nim_client()

    async def analyze(self, claim_data: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
        """Analyze ID document structure against known templates.

        Args:
            claim_data: Extracted document data.
            raw_text: Raw text from the ID document.

        Returns:
            Dict with template matching results.
        """
        logger.info("Template Match Agent: starting analysis")

        prompt = f"""You are an expert document forensics analyst specializing in identity document verification.

Analyze the following extracted ID document data for authenticity indicators.

EXTRACTED DATA:
{json.dumps(claim_data, indent=2, default=str)[:3000]}

DOCUMENT TEXT:
{raw_text[:2000]}

Check for these indicators:
1. **Template compliance**: Does the document follow the expected layout for its type (driver's license, passport, national ID)?
2. **Required fields**: Are all mandatory fields present (name, DOB, expiry, ID number, photo)?
3. **Format consistency**: Do field formats match expected patterns (date formats, ID number structure)?
4. **Font anomalies**: Any mentions of inconsistent text rendering or font issues?
5. **Security features**: Any indicators of missing security elements (hologram markers, watermarks)?
6. **Expiry status**: Is the document expired or suspiciously close to expiry/issuance?
7. **Issuing authority**: Does the issuing authority match the document type and jurisdiction?

Return a JSON object:
{{
    "document_type": "drivers_license|passport|national_id|other|unknown",
    "issuing_jurisdiction": "state/country or unknown",
    "template_match_score": 0-100,
    "risk_score": 0-100,
    "required_fields_present": true/false,
    "fields_checked": [
        {{
            "field": "name|dob|expiry|id_number|photo|address",
            "present": true/false,
            "format_valid": true/false,
            "notes": "any issues"
        }}
    ],
    "flags": [
        {{
            "type": "template_violation|missing_field|format_error|font_anomaly|security_feature|expired|authority_mismatch",
            "severity": "critical|high|medium|low",
            "description": "detailed explanation",
            "confidence": 0.0-1.0
        }}
    ],
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
                    "template_match_score": 50,
                    "risk_score": 50,
                    "flags": [],
                    "summary": response[:500],
                    "_raw_response": True,
                }

            logger.info(f"Template Match complete. Score: {result.get('template_match_score', 0)}")
            return result

        except Exception as e:
            logger.error(f"Template Match Agent error: {e}")
            return {
                "template_match_score": 0,
                "risk_score": 0,
                "flags": [],
                "summary": f"Analysis error: {str(e)}",
                "error": str(e),
            }
