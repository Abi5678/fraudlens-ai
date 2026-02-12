"""
ID document extraction using:
- NeMo Retriever OCR v1 — best for structured, noisy, real-world ID images (when NEMO_OCR_BASE_URL is set)
- Nemotron Nano VL — vision-language model for OCR fallback and multimodal reasoning
"""

import os
import base64
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

import httpx

# Model ID for Nemotron Nano 12B v2 VL (document intelligence, tops OCR benchmarks)
NEMOTRON_NANO_VL_MODEL = "nvidia/nemotron-nano-12b-v2-vl"


def _get_nim_headers() -> Dict[str, str]:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        try:
            from dotenv import load_dotenv
            from pathlib import Path as P
            load_dotenv(P(__file__).parent.parent / ".env")
            api_key = os.environ.get("NVIDIA_API_KEY", "")
        except Exception:
            pass
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def extract_text_nemo_ocr(
    image_path: str,
    merge_level: str = "paragraph",
    base_url: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """
    Extract text from an ID image using NeMo Retriever OCR v1.
    Returns (raw_text, text_detections, error_message).
    When base_url is not set or request fails, returns ("", [], error_msg).
    """
    base_url = (base_url or os.environ.get("NEMO_OCR_BASE_URL", "")).rstrip("/")
    if not base_url:
        return "", [], "NEMO_OCR_BASE_URL not set"

    path = Path(image_path)
    if not path.exists():
        return "", [], f"File not found: {image_path}"

    try:
        with open(path, "rb") as f:
            img_bytes = f.read()
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        b64 = base64.b64encode(img_bytes).decode()
        data_url = f"data:{mime};base64,{b64}"

        payload = {
            "input": [{"type": "image_url", "url": data_url}],
            "merge_levels": [merge_level],
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{base_url}/v1/infer",
                headers=_get_nim_headers(),
                json=payload,
            )

        if resp.status_code != 200:
            return "", [], f"NeMo OCR HTTP {resp.status_code}: {resp.text[:200]}"

        data = resp.json()
        text_parts: List[str] = []
        all_detections: List[Dict[str, Any]] = []

        for item in data.get("data", []):
            for det in item.get("text_detections", []):
                pred = det.get("text_prediction", {})
                text = pred.get("text", "").strip()
                if text:
                    text_parts.append(text)
                    all_detections.append({
                        "text": text,
                        "confidence": pred.get("confidence", 0),
                        "bounding_box": det.get("bounding_box"),
                    })

        raw_text = "\n".join(text_parts) if merge_level == "paragraph" else " ".join(text_parts)
        logger.info(f"NeMo Retriever OCR extracted {len(text_parts)} regions from {path.name}")
        return raw_text, all_detections, None

    except Exception as e:
        logger.warning(f"NeMo Retriever OCR error: {e}")
        return "", [], str(e)


async def extract_text_and_reason_nano_vl(
    image_path: str,
    prompt: str,
    nim_client: Any = None,
) -> str:
    """
    Use Nemotron Nano VL for text extraction or multimodal reasoning from an ID image.
    """
    path = Path(image_path)
    if not path.exists():
        return ""

    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"

        if nim_client is None:
            from core.nim_client import get_nim_client
            nim_client = get_nim_client()

        response = await nim_client.chat(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            model=NEMOTRON_NANO_VL_MODEL,
            temperature=0.0,
            max_tokens=2048,
        )
        return (response or "").strip()
    except Exception as e:
        logger.warning(f"Nemotron Nano VL error: {e}")
        return ""


async def id_image_to_raw_text(
    image_path: str,
    use_nemo_ocr_first: bool = True,
    nemo_ocr_base_url: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Extract text from an ID document image.
    Tries NeMo Retriever OCR first if use_nemo_ocr_first and NEMO_OCR_BASE_URL is set;
    otherwise uses Nemotron Nano VL for OCR-style extraction.
    Returns (raw_text, processor_used).
    """
    nemo_url = nemo_ocr_base_url or os.environ.get("NEMO_OCR_BASE_URL", "").rstrip("/")
    if use_nemo_ocr_first and nemo_url:
        raw_text, _, err = await extract_text_nemo_ocr(image_path, base_url=nemo_url)
        if not err and raw_text.strip():
            return raw_text, "nemo-retriever-ocr"
        logger.info(f"NeMo OCR unavailable or empty, falling back to Nano VL: {err or 'no text'}")

    prompt = """Extract ALL text visible on this identity document (driver's license, ID card, or passport) in reading order.
Include: document type, name (first and last), ID number, license number, date of birth, expiration date, issue date, address, 
physical description (height, weight, hair, eyes), class/restrictions, signature line, and any other printed text.
Output as plain text with one field per line where possible (e.g. "Name: John Doe"). Preserve exact spelling and numbers."""
    raw = await extract_text_and_reason_nano_vl(image_path, prompt)
    return raw or "", "nemotron-nano-vl"


async def id_multimodal_reasoning(
    image_path: str,
    extracted_text: str,
    nim_client: Any = None,
) -> str:
    """
    Use Nemotron Nano VL to combine visual and text context: layout, security features, consistency.
    """
    prompt = f"""You have an identity document image and the following text extracted from it:

EXTRACTED TEXT:
{extracted_text[:2000]}

Based on BOTH the image and the text:
1. Describe the document layout and any visible security features (hologram, watermark, UV pattern mentions).
2. Note if any text appears inconsistent with typical ID layout (e.g. misaligned fields, wrong fonts).
3. State whether the portrait area looks consistent with the rest of the card (lighting, resolution).
Keep the response under 300 words and focused on authenticity-related observations."""

    return await extract_text_and_reason_nano_vl(image_path, prompt, nim_client)


async def face_verify_nano_vl(
    image_path_1: str,
    image_path_2: str,
    nim_client: Any = None,
) -> Tuple[bool, float, str]:
    """
    Use Nemotron Nano VL to compare two face images (e.g. ID portrait vs selfie).
    Returns (same_person, confidence_0_100, explanation).
    """
    p1, p2 = Path(image_path_1), Path(image_path_2)
    if not p1.exists() or not p2.exists():
        return False, 0.0, "Missing image"

    try:
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            b64_1 = base64.b64encode(f1.read()).decode()
            b64_2 = base64.b64encode(f2.read()).decode()
        mime = "image/jpeg"

        if nim_client is None:
            from core.nim_client import get_nim_client
            nim_client = get_nim_client()

        prompt = """Are these two photos of the SAME person? Consider face shape, eyes, nose, mouth, and overall appearance.
Reply with exactly:
SAME: yes or no
CONFIDENCE: a number from 0 to 100
Then one short sentence explaining why."""

        response = await nim_client.chat(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_1}"}},
                    {"type": "text", "text": "First photo (e.g. ID portrait)."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_2}"}},
                    {"type": "text", "text": "Second photo (e.g. selfie)."},
                    {"type": "text", "text": prompt},
                ],
            }],
            model=NEMOTRON_NANO_VL_MODEL,
            temperature=0.0,
            max_tokens=150,
        )
        return _parse_face_verify_response(response or "")
    except Exception as e:
        logger.warning(f"Face verify Nano VL error: {e}")
        return False, 0.0, str(e)


def _parse_face_verify_response(response: str) -> Tuple[bool, float, str]:
    same = False
    confidence = 0.0
    response_lower = response.lower()
    if "same: yes" in response_lower or "same:yes" in response_lower:
        same = True
    import re
    for m in re.finditer(r"confidence\s*:\s*(\d{1,3})", response, re.I):
        try:
            confidence = min(100, max(0, float(m.group(1))))
            break
        except ValueError:
            pass
    if confidence == 0 and re.search(r"\b(\d{1,3})\s*%", response):
        m = re.search(r"\b(\d{1,3})\s*%", response)
        if m:
            confidence = min(100, max(0, float(m.group(1))))
    return same, confidence, response.strip()[:500]
