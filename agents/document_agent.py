"""
Document Agent
Extracts structured data from insurance claim documents using NVIDIA Nemotron-Parse
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

from core.document_processor import DocumentProcessor, ExtractedDocument
from core.nim_client import get_nim_client


@dataclass
class ClaimData:
    """Structured claim data extracted from document"""
    
    # Claimant Information
    claimant_name: str = ""
    claimant_address: str = ""
    claimant_phone: str = ""
    claimant_email: str = ""
    claimant_dob: str = ""
    claimant_ssn_last4: str = ""
    
    # Policy Information
    policy_number: str = ""
    policy_holder: str = ""
    policy_type: str = ""
    coverage_amount: float = 0.0
    
    # Incident Information
    incident_date: str = ""
    incident_time: str = ""
    incident_location: str = ""
    incident_description: str = ""
    
    # Claim Details
    claim_number: str = ""
    claim_date: str = ""
    claim_amount: float = 0.0
    claim_type: str = ""  # auto, property, health, life
    
    # Vehicle Information (if auto claim)
    vehicle_make: str = ""
    vehicle_model: str = ""
    vehicle_year: str = ""
    vehicle_vin: str = ""
    
    # Medical Information (if injury claim)
    injuries_reported: list = field(default_factory=list)
    treatment_providers: list = field(default_factory=list)
    medical_costs: float = 0.0
    
    # Witnesses
    witnesses: list = field(default_factory=list)
    
    # Raw extracted content
    raw_text: str = ""
    tables: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claimant": {
                "name": self.claimant_name,
                "address": self.claimant_address,
                "phone": self.claimant_phone,
                "email": self.claimant_email,
                "dob": self.claimant_dob,
            },
            "policy": {
                "number": self.policy_number,
                "holder": self.policy_holder,
                "type": self.policy_type,
                "coverage_amount": self.coverage_amount,
            },
            "incident": {
                "date": self.incident_date,
                "time": self.incident_time,
                "location": self.incident_location,
                "description": self.incident_description,
            },
            "claim": {
                "number": self.claim_number,
                "date": self.claim_date,
                "amount": self.claim_amount,
                "type": self.claim_type,
            },
            "vehicle": {
                "make": self.vehicle_make,
                "model": self.vehicle_model,
                "year": self.vehicle_year,
                "vin": self.vehicle_vin,
            },
            "medical": {
                "injuries": self.injuries_reported,
                "providers": self.treatment_providers,
                "costs": self.medical_costs,
            },
            "witnesses": self.witnesses,
        }


class DocumentAgent:
    """
    Document extraction agent using NVIDIA Nemotron-Parse.
    
    Extracts structured claim data from insurance documents including:
    - Claimant information
    - Policy details
    - Incident information
    - Claim amounts and types
    - Vehicle information (auto claims)
    - Medical information (injury claims)
    """
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.nim_client = get_nim_client()
        
        logger.info("DocumentAgent initialized with Nemotron-Parse")
    
    async def process(self, file_path: str) -> Dict[str, Any]:
        """
        Process a claim document and extract structured data.
        
        Args:
            file_path: Path to the claim document
            
        Returns:
            Dictionary with extracted claim data and metadata
        """
        file_path = Path(file_path)
        logger.info(f"DocumentAgent processing: {file_path.name}")
        
        try:
            # Step 1: Extract raw content using Nemotron-Parse
            extracted = await self.processor.process(
                file_path,
                extract_tables=True,
                extract_charts=True,
            )
            
            # Step 2: Use LLM to structure the extracted content
            claim_data = await self._structure_claim_data(extracted)
            
            # Step 3: Add raw content for other agents
            claim_data.raw_text = extracted.raw_text
            claim_data.tables = extracted.tables
            
            return {
                "status": "success",
                "claim_data": claim_data.to_dict(),
                "raw_text": extracted.raw_text,
                "markdown": extracted.markdown,
                "tables": extracted.tables,
                "metadata": extracted.metadata,
                "extracted_images": getattr(extracted, 'extracted_images', []),
            }
            
        except Exception as e:
            logger.error(f"DocumentAgent error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "claim_data": ClaimData().to_dict(),
            }

    async def process_id_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a photo ID image using NeMo Retriever OCR (if configured) and/or
        Nemotron Nano VL for text extraction and multimodal reasoning.
        Returns the same shape as process() for compatibility with ID verification pipeline.
        """
        image_path = Path(image_path)
        logger.info(f"DocumentAgent processing ID image: {image_path.name}")

        try:
            from core.id_ocr_service import (
                id_image_to_raw_text,
                id_multimodal_reasoning,
            )
            raw_text, processor_used = await id_image_to_raw_text(str(image_path))
            if not raw_text.strip():
                # Fallback to existing Nemotron-Parse image path
                extracted = await self.processor.process(image_path, extract_images=False)
                claim_data = await self._structure_claim_data(extracted)
                claim_data.raw_text = extracted.raw_text
                return {
                    "status": "success",
                    "claim_data": claim_data.to_dict(),
                    "raw_text": extracted.raw_text,
                    "metadata": {"processor": "nemotron-parse-fallback", **extracted.metadata},
                    "extracted_images": [],
                }

            # Optional: multimodal reasoning (layout, security features)
            multimodal_notes = await id_multimodal_reasoning(
                str(image_path), raw_text, self.nim_client
            )
            if multimodal_notes:
                raw_text = raw_text + "\n\n[MULTIMODAL REASONING]\n" + multimodal_notes

            # Structure ID fields into claim_data-like dict
            claim_data_dict = await self._structure_id_data(raw_text)
            return {
                "status": "success",
                "claim_data": claim_data_dict,
                "raw_text": raw_text,
                "metadata": {"processor": processor_used, "multimodal": bool(multimodal_notes)},
                "extracted_images": [],
            }
        except Exception as e:
            logger.error(f"DocumentAgent ID image error: {e}")
            # Fallback to standard image processing
            try:
                extracted = await self.processor.process(image_path, extract_images=False)
                claim_data = await self._structure_claim_data(extracted)
                return {
                    "status": "success",
                    "claim_data": claim_data.to_dict(),
                    "raw_text": extracted.raw_text,
                    "metadata": {"processor": "nemotron-parse-fallback", "error_fallback": str(e)},
                    "extracted_images": [],
                }
            except Exception as e2:
                logger.error(f"ID fallback also failed: {e2}")
                return {
                    "status": "error",
                    "error": str(e2),
                    "claim_data": {},
                    "raw_text": "",
                }

    async def _structure_id_data(self, raw_text: str) -> Dict[str, Any]:
        """Use LLM to structure extracted ID text into a consistent dict for downstream agents."""
        prompt = f"""You are an expert at parsing identity document text. Extract structured fields from this OCR/reasoning output.
Output ONLY a JSON object with these keys (use null if not found):
- document_type (e.g. "driver_license", "passport", "national_id")
- issuing_jurisdiction (e.g. "California", "USA")
- last_name, first_name (or single "name" if not split)
- license_number or id_number or document_number
- date_of_birth (DOB)
- expiration_date or expiry
- issue_date
- address
- hair (color), eyes (color)
- height, weight, sex
- class, endorsements, restrictions
- signature (present: true/false)

Raw text:
{raw_text[:6000]}

Return valid JSON only, no markdown code block."""

        try:
            response = await self.nim_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            import json
            text = (response or "").strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            data = json.loads(text)
            # Normalize to claim_data shape for compatibility
            name = data.get("name") or (f"{data.get('first_name', '')} {data.get('last_name', '')}".strip())
            return {
                "claimant": {
                    "name": name or data.get("first_name") or data.get("last_name") or "",
                    "address": data.get("address") or "",
                    "dob": data.get("date_of_birth") or data.get("DOB") or "",
                },
                "document_type": data.get("document_type"),
                "issuing_jurisdiction": data.get("issuing_jurisdiction"),
                "license_number": data.get("license_number") or data.get("id_number") or data.get("document_number"),
                "date_of_birth": data.get("date_of_birth") or data.get("DOB"),
                "expiration_date": data.get("expiration_date") or data.get("expiry"),
                "issue_date": data.get("issue_date"),
                "address": data.get("address"),
                "hair": data.get("hair"),
                "eyes": data.get("eyes"),
                "height": data.get("height"),
                "weight": data.get("weight"),
                "sex": data.get("sex"),
                "class": data.get("class"),
                "raw_text": raw_text[:3000],
            }
        except Exception as e:
            logger.warning(f"ID structure parse error: {e}")
            return {"claimant": {"name": "", "address": "", "dob": ""}, "raw_text": raw_text[:3000]}
    
    async def _structure_claim_data(self, extracted: ExtractedDocument) -> ClaimData:
        """Use LLM to extract structured fields from raw text"""
        
        prompt = f"""You are an expert insurance document analyst. Extract structured data from this claim document.
Be precise — copy values EXACTLY as they appear in the text (names, numbers, dates, VINs).

Document Content:
{extracted.raw_text[:10000]}

Extract these fields (write "N/A" if not found):

1. CLAIMANT INFORMATION:
   - Full Name: (the renter, claimant, or insured person's full name)
   - Address: (full mailing address)
   - Phone Number:
   - Email:
   - Date of Birth:

2. POLICY INFORMATION:
   - Policy Number: (or rental agreement number)
   - Policy Holder Name:
   - Policy Type: (auto/property/health/life)
   - Coverage Amount:

3. INCIDENT INFORMATION:
   - Date of Incident: (or Date of Loss)
   - Time of Incident:
   - Location:
   - Description: (brief description of what happened)

4. CLAIM DETAILS:
   - Claim Number:
   - Date Filed: (or Invoice Date)
   - Amount Claimed: (total claim balance, damage amount, or total amount owed — use the largest total)
   - Type of Claim:

5. VEHICLE INFORMATION (if auto claim):
   - Make: (e.g. Chevrolet, Toyota, Ford)
   - Model: (e.g. Suburban, Camry)
   - Year: (e.g. 2025)
   - VIN: (17-character Vehicle Identification Number)

6. MEDICAL INFORMATION (if injury claim):
   - Injuries Reported: (comma-separated list)
   - Treatment Providers: (comma-separated list)
   - Medical Costs:

7. WITNESSES (list names and contact info)

IMPORTANT: For the Amount, look for "Claim Balance", "Total Amount", "Amount Due", or "Damage Amount".
For vehicle info, look for VIN numbers (17 chars like 1GNXXXXXX), make/model/year anywhere in the document.
Format your response as a structured list with clear labels for each field."""

        try:
            response = await self.nim_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            
            # Parse LLM response into ClaimData
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"LLM structuring error: {e}")
            return ClaimData(raw_text=extracted.raw_text)
    
    def _parse_llm_response(self, response: str) -> ClaimData:
        """Parse LLM response into ClaimData structure"""
        claim = ClaimData()
        
        lines = response.split("\n")
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            line_lower = line.lower()
            if "claimant" in line_lower:
                current_section = "claimant"
            elif "policy" in line_lower:
                current_section = "policy"
            elif "incident" in line_lower:
                current_section = "incident"
            elif "claim" in line_lower and "claimant" not in line_lower:
                current_section = "claim"
            elif "vehicle" in line_lower:
                current_section = "vehicle"
            elif "medical" in line_lower:
                current_section = "medical"
            elif "witness" in line_lower:
                current_section = "witness"
            
            # Extract field values
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if value.lower() in ["n/a", "not found", "unknown", ""]:
                    continue
                
                # Map to ClaimData fields
                if current_section == "claimant":
                    if "name" in key:
                        claim.claimant_name = value
                    elif "address" in key:
                        claim.claimant_address = value
                    elif "phone" in key:
                        claim.claimant_phone = value
                    elif "email" in key:
                        claim.claimant_email = value
                    elif "birth" in key or "dob" in key:
                        claim.claimant_dob = value
                        
                elif current_section == "policy":
                    if "number" in key:
                        claim.policy_number = value
                    elif "holder" in key:
                        claim.policy_holder = value
                    elif "type" in key:
                        claim.policy_type = value
                    elif "coverage" in key or "amount" in key:
                        claim.coverage_amount = self._parse_amount(value)
                        
                elif current_section == "incident":
                    if "date" in key:
                        claim.incident_date = value
                    elif "time" in key:
                        claim.incident_time = value
                    elif "location" in key:
                        claim.incident_location = value
                    elif "description" in key:
                        claim.incident_description = value
                        
                elif current_section == "claim":
                    if "number" in key:
                        claim.claim_number = value
                    elif "date" in key:
                        claim.claim_date = value
                    elif "amount" in key:
                        claim.claim_amount = self._parse_amount(value)
                    elif "type" in key:
                        claim.claim_type = value
                        
                elif current_section == "vehicle":
                    if "make" in key:
                        claim.vehicle_make = value
                    elif "model" in key:
                        claim.vehicle_model = value
                    elif "year" in key:
                        claim.vehicle_year = value
                    elif "vin" in key:
                        claim.vehicle_vin = value
                        
                elif current_section == "medical":
                    if "injur" in key:
                        claim.injuries_reported = [i.strip() for i in value.split(",")]
                    elif "provider" in key:
                        claim.treatment_providers = [p.strip() for p in value.split(",")]
                    elif "cost" in key:
                        claim.medical_costs = self._parse_amount(value)
        
        return claim
    
    def _parse_amount(self, value: str) -> float:
        """Parse monetary amount from string (handles '$3,798.76', '$3798.76 (note)', etc.)"""
        import re
        try:
            # Find the first dollar amount pattern in the value
            match = re.search(r'\$?([\d,]+\.?\d*)', value.replace(" ", ""))
            if match:
                cleaned = match.group(1).replace(",", "")
                return float(cleaned)
            # Fallback: strip everything non-numeric
            cleaned = re.sub(r'[^\d.]', '', value)
            return float(cleaned) if cleaned else 0.0
        except (ValueError, AttributeError):
            return 0.0
