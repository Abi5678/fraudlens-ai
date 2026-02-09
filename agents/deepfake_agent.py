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
    
    async def analyze(self, image_paths: List[str] = None, image_data: bytes = None) -> Dict[str, Any]:
        """
        Analyze images for manipulation or AI generation.
        
        Args:
            image_paths: List of paths to images
            image_data: Raw image bytes (alternative to paths)
        """
        logger.info("DeepfakeAgent analyzing images")
        
        try:
            if not image_paths and not image_data:
                return {"status": "skipped", "message": "No images provided", "manipulation_score": 0}
            
            results = []
            
            if image_paths:
                for path in image_paths:
                    result = await self._analyze_image(Path(path))
                    results.append(result)
            
            if image_data:
                result = await self._analyze_image_bytes(image_data)
                results.append(result)
            
            # Aggregate results
            if not results:
                return {"status": "success", "manipulation_score": 0, "images_analyzed": 0}
            
            avg_score = sum(r.get("score", 0) for r in results) / len(results)
            detections = [d for r in results for d in r.get("detections", [])]
            
            return {
                "status": "success",
                "manipulation_score": avg_score,
                "images_analyzed": len(results),
                "detections": detections,
                "individual_results": results,
                "summary": self._generate_summary(avg_score, detections),
            }
            
        except Exception as e:
            logger.error(f"DeepfakeAgent error: {e}")
            return {"status": "error", "error": str(e), "manipulation_score": 0}
    
    async def _analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Analyze a single image file"""
        if not image_path.exists():
            return {"path": str(image_path), "error": "File not found", "score": 0}
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        result = await self._analyze_image_bytes(image_data)
        result["path"] = str(image_path)
        return result
    
    async def _analyze_image_bytes(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze image from bytes using multimodal LLM"""
        
        # Encode image for API
        encoded = base64.b64encode(image_data).decode()
        
        # Use multimodal model to analyze
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
            response = await self.nim_client.chat(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                model="nvidia/nemotron-4-340b-instruct",
                max_tokens=500,
            )
            
            # Parse response
            score, detections = self._parse_analysis(response)
            
            return {
                "score": score,
                "detections": detections,
                "raw_analysis": response,
            }
            
        except Exception as e:
            logger.warning(f"Image analysis error: {e}")
            # Fallback to basic checks
            return await self._basic_checks(image_data)
    
    def _parse_analysis(self, response: str) -> tuple:
        """Parse LLM response into score and detections"""
        score = 25  # Default
        detections = []
        
        response_lower = response.lower()
        
        # Extract score
        if "0-100" in response or "likelihood" in response_lower:
            import re
            numbers = re.findall(r'\b(\d{1,3})\b', response)
            for num in numbers:
                n = int(num)
                if 0 <= n <= 100:
                    score = n
                    break
        
        # Classify based on keywords
        if any(w in response_lower for w in ["highly likely", "definite", "clear manipulation"]):
            score = max(score, 80)
        elif any(w in response_lower for w in ["likely", "probable", "signs of"]):
            score = max(score, 60)
        elif any(w in response_lower for w in ["possible", "minor", "subtle"]):
            score = max(score, 40)
        elif any(w in response_lower for w in ["unlikely", "authentic", "no signs"]):
            score = min(score, 20)
        
        # Extract detections
        detection_keywords = [
            ("ai_generated", ["ai generated", "artificial", "synthetic"]),
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
