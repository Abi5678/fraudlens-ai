"""
AI Image Generation Detector Agent
Detects AI-generated images (DALL-E, Midjourney, Stable Diffusion, etc.).
Uses NVIDIA NIM for intelligent visual analysis.
"""

import json
from typing import Dict, Any, List
from pathlib import Path
from loguru import logger


class AIImageDetectorAgent:
    """Detects AI-generated images and visual content."""

    def __init__(self):
        from core.nim_client import get_nim_client
        self.nim = get_nim_client()

    async def analyze(self, image_paths: List[str]) -> Dict[str, Any]:
        """Analyze images for AI generation indicators.

        Args:
            image_paths: List of image file paths.

        Returns:
            Dict with AI image detection results.
        """
        logger.info(f"AI Image Detector: analyzing {len(image_paths)} images")

        # Gather basic image info
        image_info = []
        try:
            from PIL import Image, ImageStat
            for fp in image_paths[:10]:
                try:
                    img = Image.open(fp)
                    stat = ImageStat.Stat(img)
                    info = {
                        "file": Path(fp).name,
                        "format": img.format,
                        "size": list(img.size),
                        "mode": img.mode,
                        "mean_brightness": sum(stat.mean) / len(stat.mean),
                        "stddev": sum(stat.stddev) / len(stat.stddev),
                    }
                    # Check for EXIF (AI images typically lack it)
                    exif = img.getexif()
                    info["has_exif"] = bool(exif)
                    info["exif_fields"] = len(exif) if exif else 0
                    image_info.append(info)
                except Exception as e:
                    image_info.append({"file": Path(fp).name, "error": str(e)})
        except ImportError:
            pass

        prompt = f"""You are an expert AI-generated image detection analyst. Your job is to determine whether images were created by AI (DALL-E, Midjourney, Stable Diffusion, etc.) or are genuine photographs/scans.

Analyze the following image data for AI generation indicators.

IMAGE DATA:
{json.dumps(image_info, indent=2, default=str)[:4000]}

Check for these AI image indicators:
1. **EXIF absence**: AI-generated images typically lack camera EXIF data
2. **Resolution patterns**: Unusual or perfectly square dimensions common in AI art
3. **Artifacts**: AI-specific visual artifacts (mangled text, irregular fingers, asymmetric features)
4. **Texture consistency**: Unnaturally smooth or repeating texture patterns
5. **Lighting anomalies**: Physically impossible lighting or shadow directions
6. **Generation fingerprints**: Known output sizes (512x512, 1024x1024, etc.)
7. **Metadata traces**: AI generation metadata or unusual software traces

Return a JSON object:
{{
    "images_analyzed": 0,
    "overall_ai_probability": 0.0-1.0,
    "risk_score": 0-100,
    "individual_results": [
        {{
            "file": "filename",
            "ai_probability": 0.0-1.0,
            "classification": "likely_ai|possibly_ai|uncertain|likely_genuine",
            "indicators": ["indicator1", "indicator2"],
            "notes": "brief analysis"
        }}
    ],
    "flags": [
        {{
            "file": "filename",
            "type": "exif_absence|resolution_pattern|artifact|texture|lighting|fingerprint|metadata",
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
                    "images_analyzed": len(image_paths),
                    "overall_ai_probability": 0.5,
                    "risk_score": 50,
                    "individual_results": [],
                    "flags": [],
                    "summary": response[:500],
                    "_raw_response": True,
                }

            logger.info(f"AI Image Detection complete. AI probability: {result.get('overall_ai_probability', 0)}")
            return result

        except Exception as e:
            logger.error(f"AI Image Detector error: {e}")
            return {
                "images_analyzed": len(image_paths),
                "overall_ai_probability": 0.0,
                "risk_score": 0,
                "individual_results": [],
                "flags": [],
                "summary": f"Analysis error: {str(e)}",
                "error": str(e),
            }
