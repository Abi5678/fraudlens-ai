"""
Clinical Consistency Agent
Uses NLP to verify that documented symptoms and diagnoses justify
the prescribed treatments, procedures, and medications.
Powered by NVIDIA NIM.
"""

import json
from typing import Dict, Any
from loguru import logger

from core.nim_client import get_nim_client


class ClinicalConsistencyAgent:
    """
    Verifies clinical consistency between diagnoses, symptoms,
    treatments, and procedures in medical insurance claims.

    Checks:
    - Symptom-diagnosis alignment: do documented symptoms match the diagnosis?
    - Diagnosis-treatment appropriateness: is the treatment medically justified?
    - Procedure medical necessity: does clinical evidence support the procedure?
    - Medication-diagnosis consistency: are prescribed meds consistent with diagnosis?
    - Timeline plausibility: treatment timeline makes clinical sense
    """

    def __init__(self):
        self.nim_client = get_nim_client()
        logger.info("ClinicalConsistencyAgent initialized with NVIDIA NIM")

    async def analyze(self, claim_data: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
        """
        Analyze medical claim for clinical consistency.

        Args:
            claim_data: Structured medical claim data
            raw_text: Raw text from clinical notes / EHR

        Returns:
            Dict with clinical consistency flags, risk score, and summary
        """
        logger.info("ClinicalConsistencyAgent analyzing claim")

        prompt = f"""You are an expert clinical auditor reviewing medical insurance claims for clinical consistency fraud.

CLAIM DATA:
{json.dumps(claim_data, indent=2, default=str)[:4000]}

CLINICAL NOTES / DOCUMENT TEXT:
{raw_text[:5000]}

Perform these clinical consistency checks:

1. **SYMPTOM-DIAGNOSIS ALIGNMENT**: Do the documented symptoms and patient complaints support the ICD-10 diagnosis codes?
   - Flag if diagnosis is not supported by documented symptoms (e.g., billing for pneumonia when notes show only mild cold symptoms)

2. **DIAGNOSIS-TREATMENT APPROPRIATENESS**: Is the treatment plan medically appropriate for the documented diagnosis?
   - Flag unnecessary or excessive treatments relative to the condition

3. **PROCEDURE MEDICAL NECESSITY**: Does the clinical documentation provide sufficient medical necessity for each procedure?
   - Flag procedures that lack clinical justification in the notes

4. **MEDICATION CONSISTENCY**: Are prescribed medications consistent with the documented diagnosis and standard of care?
   - Flag unusual drug combinations or medications that don't match the condition

5. **TIMELINE PLAUSIBILITY**: Does the treatment timeline make clinical sense?
   - Flag impossible recovery claims, suspiciously rapid treatment escalation, or treatment dates before diagnosis

6. **ROOT CAUSE DENIAL RISK**: Would this claim likely be denied upon audit? Why?
   - Identify specific coding or documentation gaps that would trigger denial

Return a JSON object:
{{
    "clinically_consistent": true/false,
    "risk_score": 0-100,
    "flags": [
        {{
            "type": "symptom_mismatch|treatment_inappropriate|no_medical_necessity|medication_mismatch|timeline_issue|denial_risk",
            "severity": "critical|high|medium|low",
            "description": "detailed explanation with clinical reasoning",
            "confidence": 0.0-1.0
        }}
    ],
    "clinical_summary": {{
        "primary_diagnosis": "main diagnosis from notes",
        "documented_symptoms": ["list of symptoms found"],
        "procedures_justified": true/false,
        "treatment_appropriate": true/false
    }},
    "denial_risk": {{
        "likely_denied": true/false,
        "denial_reasons": ["list of reasons"]
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
                    "clinically_consistent": False,
                    "risk_score": 50,
                    "flags": [],
                    "summary": response[:500],
                    "_raw_response": True,
                }

            logger.info(f"ClinicalConsistency complete. Risk score: {result.get('risk_score', 0)}")
            return result

        except Exception as e:
            logger.error(f"ClinicalConsistencyAgent error: {e}")
            return {
                "clinically_consistent": False,
                "risk_score": 0,
                "flags": [],
                "summary": f"Analysis error: {str(e)}",
                "error": str(e),
            }
