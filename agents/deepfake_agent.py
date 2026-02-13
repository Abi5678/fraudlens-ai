"""
Deepfake Agent - Detects image manipulation and AI-generated content
Uses TensorRT for optimized inference when available
"""

from typing import Dict, Any, List
from pathlib import Path
import base64
from loguru import logger

from core.nim_client import get_nim_client


class DeepfakeAgent:
    """
    Detects image manipulation and deepfakes in claim photos.
    
    Detection methods:
    1. AI-generated image detection
    2. Digital manipulation traces (ELA)
    3. EXIF metadata analysis
    4. Lighting/physics consistency
    5. Compression artifact analysis
    """
    
    def __init__(self):
        self.nim_client = get_nim_client()
    
    async def analyze(
        self,
        image_paths: List[str] = None,
        image_data: bytes = None,
        context: str = None,
    ) -> Dict[str, Any]:
        """
        Analyze images for manipulation or AI generation.

        Args:
            image_paths: List of paths to images
            image_data: Raw image bytes (alternative to paths)
            context: Optional "id_document" for photo ID verification (uses AI-generated-ID-specific prompt)
        """
        logger.info("DeepfakeAgent analyzing images" + (f" (context={context})" if context else ""))
        
        try:
            if not image_paths and not image_data:
                return {"status": "skipped", "message": "No images provided", "manipulation_score": 0}
            
            results = []
            nim_unavailable = False  # set if first image gets 404 so we skip NIM for rest
            
            if image_paths:
                for path in image_paths:
                    result = await self._analyze_image(Path(path), context=context, skip_nim_if_unavailable=nim_unavailable)
                    results.append(result)
                    if result.get("nim_unavailable"):
                        nim_unavailable = True
            
            if image_data:
                result = await self._analyze_image_bytes(image_data, context=context)
                results.append(result)
            
            # Aggregate results
            if not results:
                return {"status": "success", "manipulation_score": 0, "images_analyzed": 0}
            
            # If NIM image model was unavailable (404), return skipped so scorer can renormalize weights
            if any(r.get("nim_unavailable") for r in results):
                return {"status": "skipped", "message": "Image model not available (404)", "manipulation_score": 0, "images_analyzed": len(results)}
            
            avg_score = sum(r.get("score", 0) for r in results) / len(results)
            detections = [d for r in results for d in r.get("detections", [])]
            # AI-generated ID: take max across images when context is id_document
            ai_generated_score = max(
                (r.get("ai_generated_score", 0) for r in results),
                default=0,
            )
            ai_generated_detected = any(r.get("ai_generated_detected", False) for r in results)
            
            out = {
                "status": "success",
                "manipulation_score": avg_score,
                "images_analyzed": len(results),
                "detections": detections,
                "individual_results": results,
                "summary": self._generate_summary(avg_score, detections),
            }
            if context == "id_document":
                out["ai_generated_score"] = ai_generated_score
                out["ai_generated_detected"] = ai_generated_detected
                if ai_generated_detected or ai_generated_score >= 50:
                    out["summary"] = (
                        f"AI-GENERATED ID RISK: Score {ai_generated_score:.0f}/100. "
                        + out["summary"]
                    )
            return out
            
        except Exception as e:
            logger.error(f"DeepfakeAgent error: {e}")
            return {"status": "error", "error": str(e), "manipulation_score": 0}
    
    async def _analyze_image(self, image_path: Path, context: str = None, skip_nim_if_unavailable: bool = False) -> Dict[str, Any]:
        """Analyze a single image file. If skip_nim_if_unavailable, only run basic checks (no NIM call)."""
        if not image_path.exists():
            return {"path": str(image_path), "error": "File not found", "score": 0}
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        if skip_nim_if_unavailable:
            result = await self._basic_checks(image_data)
            result["path"] = str(image_path)
            return result
        result = await self._analyze_image_bytes(image_data, context=context)
        result["path"] = str(image_path)
        return result
    
    async def _analyze_image_bytes(self, image_data: bytes, context: str = None) -> Dict[str, Any]:
        """Analyze image from bytes using multimodal LLM"""
        global _nim_vision_unavailable
        if _nim_vision_unavailable:
            return {"score": 0, "detections": [], "nim_unavailable": True}
        
        # Encode image for API
        encoded = base64.b64encode(image_data).decode()
        
        if context == "id_document":
            prompt = """This image is of an IDENTITY DOCUMENT (e.g. driver's license, ID card). Your task is to detect if the document or the PORTRAIT PHOTO on it is AI-GENERATED (synthetic).
**Be strict:** If you are uncertain whether the document or portrait is AI-generated, prefer AI_GENERATED: yes and a higher MANIPULATION_SCORE (at least 55). Only answer AI_GENERATED: no when you are confident the document and photo are genuine.

Also assess **security feature plausibility from the photo** (note: UV/raised text cannot be verified without specific capture):
- **Hologram/overlay**: Is there a visible hologram/ghost-image/overlay region where expected for the jurisdiction/type? If it looks flat/printed-only, flag it.
- **Microprint/fine-line patterns**: If the image is high-res, do you see fine-line guilloche patterns and microprint areas, or do they look blurred/smudged/AI-like?
- **UV elements**: You cannot see UV ink in normal light. If the user did not provide a UV photo, recommend requesting one.
- **Raised text**: You cannot feel raised text in a photo. If the user did not provide angled lighting close-ups, recommend requesting them.

**AI-generated ID / portrait indicators — look for:**
1. **Face**: Overly smooth or plastic skin; asymmetric or uncanny eyes/ears; hair that looks painted or has odd strands; teeth or smile that look artificial; face too symmetrical or "perfect".
2. **Portrait background**: Uniform color or obvious AI blur; sharp cutout edges around the head; background inconsistent with real ID photo booths.
3. **Document itself**: Card or text that looks digitally generated; text with minor artifacts (wrong kerning, floating pixels); hologram/security area that looks flat or fake; overall "too clean" or synthetic texture.
4. **Combined**: Portrait lighting doesn't match document lighting; resolution mismatch between face and rest of card; signs of a pasted or generated face onto a template.

**Output in this exact format:**
- First line: "MANIPULATION_SCORE: <0-100>" (overall manipulation/synthetic likelihood)
- Second line: "AI_GENERATED: yes" or "AI_GENERATED: no"
- Then briefly list 1–3 specific indicators you found (or "No AI-generated indicators")."""
        else:
            prompt = """Analyze this image for signs of manipulation or AI generation.

Check for:
1. AI generation artifacts (unusual textures, impossible geometry)
2. Digital manipulation (clone stamps, content-aware fill)
3. Inconsistent lighting or shadows
4. Compression artifacts suggesting re-saving
5. Inconsistencies in damage patterns (for accident photos)
6. Signs of photo staging

For each issue found, rate severity: low, medium, high.

Provide:
- Overall manipulation likelihood (0-100)
- List of specific issues found
- Confidence in assessment"""

        try:
            vision_model = getattr(self.nim_client.config, "vision_model", "meta/llama-3.2-11b-vision-instruct")
            response = await self.nim_client.chat(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                model=vision_model,
                max_tokens=600,
            )
            
            # Parse response
            score, detections = self._parse_analysis(response)
            ai_generated_score = 0
            ai_generated_detected = False
            
            if context == "id_document":
                ai_generated_score, ai_generated_detected = self._parse_ai_generated_id_response(response, score)
                if ai_generated_detected and "ai_generated" not in detections:
                    detections.append("ai_generated")
            
            result = {
                "score": score,
                "detections": detections,
                "raw_analysis": response,
            }
            if context == "id_document":
                result["ai_generated_score"] = ai_generated_score
                result["ai_generated_detected"] = ai_generated_detected
            return result
            
        except Exception as e:
            err_str = str(e).lower()
            # NIM 404 = image/multimodal model not available for this account; skip to avoid spam
            if "404" in err_str or "not found" in err_str:
                _nim_vision_unavailable = True
                logger.warning("Image model not available (404); skipping NIM vision for remaining images this process")
                return {"score": 0, "detections": [], "nim_unavailable": True}
            logger.warning(f"Image analysis error: {e}")
            return await self._basic_checks(image_data)
    
    def _parse_ai_generated_id_response(self, response: str, fallback_score: float) -> tuple:
        """Parse ID-specific response for AI_GENERATED and MANIPULATION_SCORE. Returns (ai_generated_score, ai_generated_detected)."""
        import re
        response_upper = (response or "").upper()
        ai_generated_detected = "AI_GENERATED: YES" in response_upper
        ai_generated_score = fallback_score
        
        # Prefer explicit MANIPULATION_SCORE line
        m = re.search(r"MANIPULATION_SCORE\s*:\s*(\d{1,3})", response, re.I)
        if m:
            n = int(m.group(1))
            if 0 <= n <= 100:
                ai_generated_score = n
        if ai_generated_detected and ai_generated_score < 60:
            ai_generated_score = max(ai_generated_score, 65)
        
        return ai_generated_score, ai_generated_detected

    def _parse_analysis(self, response: str) -> tuple:
        """Parse LLM response into score and detections"""
        import re
        score = 25  # Default
        detections = []
        
        response_lower = (response or "").lower()
        
        # ID-document format: MANIPULATION_SCORE: N
        m = re.search(r"MANIPULATION_SCORE\s*:\s*(\d{1,3})", response, re.I)
        if m:
            n = int(m.group(1))
            if 0 <= n <= 100:
                score = n
        elif "0-100" in response or "likelihood" in response_lower:
            numbers = re.findall(r'\b(\d{1,3})\b', response)
            for num in numbers:
                n = int(num)
                if 0 <= n <= 100:
                    score = n
                    break
        
        # Classify based on keywords
        if any(w in response_lower for w in ["highly likely", "definite", "clear manipulation", "ai_generated: yes"]):
            score = max(score, 80)
        elif any(w in response_lower for w in ["likely", "probable", "signs of", "synthetic", "ai-generated"]):
            score = max(score, 60)
        elif any(w in response_lower for w in ["possible", "minor", "subtle"]):
            score = max(score, 40)
        elif any(w in response_lower for w in ["unlikely", "authentic", "no signs", "no ai-generated"]):
            score = min(score, 20)
        
        # Extract detections
        detection_keywords = [
            ("ai_generated", ["ai generated", "artificial", "synthetic", "ai_generated: yes"]),
            ("cloning", ["clone", "duplicate", "copy"]),
            ("lighting", ["lighting", "shadow", "inconsistent"]),
            ("compression", ["compression", "artifact", "jpeg"]),
            ("staging", ["staged", "arranged", "positioned"]),
        ]
        
        for detection_type, keywords in detection_keywords:
            if any(kw in response_lower for kw in keywords):
                detections.append(detection_type)
        
        return score, detections
    
    async def _basic_checks(self, image_data: bytes) -> Dict[str, Any]:
        """Basic image checks without LLM"""
        detections = []
        score = 0
        
        try:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(image_data))
            
            # Check for EXIF data
            exif = img._getexif() if hasattr(img, '_getexif') else None
            if not exif:
                detections.append("missing_exif")
                score += 15
            
            # Check image size (very small or very large can be suspicious)
            width, height = img.size
            if width < 100 or height < 100:
                detections.append("suspicious_size")
                score += 10
            
            # Check format
            if img.format not in ["JPEG", "PNG", "TIFF"]:
                detections.append("unusual_format")
                score += 5
                
        except Exception as e:
            logger.warning(f"Basic check error: {e}")
        
        return {"score": score, "detections": detections, "method": "basic"}
    
    def _generate_summary(self, score: float, detections: List[str]) -> str:
        if score > 75:
            return f"HIGH RISK: Strong manipulation indicators ({', '.join(detections[:3])})"
        elif score > 50:
            return f"MODERATE: Some manipulation signs detected"
        elif score > 25:
            return f"LOW: Minor concerns ({len(detections)} flags)"
        return "CLEAN: No significant manipulation detected"
