"""
Metadata Analysis Agent
Analyzes file metadata (EXIF, creation dates, editing traces) for document/image authenticity.
Uses NVIDIA NIM for intelligent metadata interpretation.
"""

import json
from typing import Dict, Any, List
from pathlib import Path
from loguru import logger


class MetadataAgent:
    """Analyzes image and document metadata for signs of tampering."""

    def __init__(self):
        from core.nim_client import get_nim_client
        self.nim = get_nim_client()

    def _extract_metadata(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Extract metadata from image files using PIL."""
        results = []
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
        except ImportError:
            return results

        for fp in file_paths[:10]:
            try:
                img = Image.open(fp)
                meta = {
                    "file": Path(fp).name,
                    "format": img.format,
                    "size": list(img.size),
                    "mode": img.mode,
                }
                exif = img.getexif()
                if exif:
                    exif_data = {}
                    for tag_id, val in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', errors='replace')[:100]
                        exif_data[str(tag)] = str(val)[:200]
                    meta["exif"] = exif_data
                else:
                    meta["exif"] = None
                results.append(meta)
            except Exception as e:
                results.append({"file": Path(fp).name, "error": str(e)})

        return results

    async def analyze(self, file_paths: List[str], claim_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze file metadata for tampering indicators.

        Args:
            file_paths: List of file paths to analyze.
            claim_data: Optional structured data for context.

        Returns:
            Dict with metadata analysis results.
        """
        logger.info(f"Metadata Agent: analyzing {len(file_paths)} files")

        metadata = self._extract_metadata(file_paths)

        prompt = f"""You are a digital forensics expert specializing in image and document metadata analysis.
**Be strict and conservative:** When in doubt, assign a higher risk_score and add a flag. Missing or ambiguous metadata should be treated as suspicious.

Analyze the following file metadata for signs of tampering or manipulation.

FILE METADATA:
{json.dumps(metadata, indent=2, default=str)[:4000]}

{f"DOCUMENT CONTEXT: {json.dumps(claim_data, indent=2, default=str)[:1000]}" if claim_data else ""}

Check for these indicators:
1. **Editing software traces**: Was the image created/modified by photo editing software (Photoshop, GIMP, etc.)?
2. **Creation date anomalies**: Does the creation date make sense given the document context?
3. **GPS inconsistencies**: Does GPS location data (if present) match the claimed document origin?
4. **Resolution anomalies**: Is the image resolution consistent or does it suggest compositing?
5. **Missing metadata**: Critical metadata fields stripped (could indicate tampering)?
6. **Camera/device info**: Does the device info match expectations?
7. **Modification timestamps**: Evidence of post-creation modification?

Return a JSON object:
{{
    "files_analyzed": 0,
    "risk_score": 0-100,
    "tampering_detected": true/false,
    "flags": [
        {{
            "file": "filename",
            "type": "editing_software|date_anomaly|gps_mismatch|resolution_anomaly|missing_metadata|device_mismatch|modification",
            "severity": "critical|high|medium|low",
            "description": "detailed explanation",
            "confidence": 0.0-1.0
        }}
    ],
    "per_file_results": [
        {{
            "file": "filename",
            "risk_score": 0-100,
            "has_exif": true/false,
            "editing_software_detected": true/false,
            "notes": "brief notes"
        }}
    ],
    "summary": "brief narrative of findings"
}}"""

        try:
            response = await self.nim.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
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
                    "files_analyzed": len(file_paths),
                    "risk_score": 30,
                    "tampering_detected": False,
                    "flags": [],
                    "summary": response[:500],
                    "_raw_response": True,
                }

            logger.info(f"Metadata analysis complete. Risk score: {result.get('risk_score', 0)}")
            return result

        except Exception as e:
            logger.error(f"Metadata Agent error: {e}")
            return {
                "files_analyzed": len(file_paths),
                "risk_score": 0,
                "tampering_detected": False,
                "flags": [],
                "summary": f"Analysis error: {str(e)}",
                "error": str(e),
            }
