"""
ID Consistency Agent
Rule-based and heuristic checks for photo ID/document plausibility.
Detects placeholder IDs, expired docs, generic dates, and data consistency red flags
that catch fakes even when template/metadata/deepfake scores are low.
"""

import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger


# Common placeholder or obviously fake ID number patterns
PLACEHOLDER_ID_PATTERNS = [
    re.compile(r"^[A-Z]?1+$", re.I),           # G1111111, 1111111
    re.compile(r"^[A-Z]?0+$", re.I),           # 0000000
    re.compile(r"^12345678?9?0?$", re.I),      # 12345678, 123456789, 1234567890
    re.compile(r"^[A-Z]?(\d)\1{6,}$", re.I),   # Same digit repeated 7+ times
    re.compile(r"^(?:TEST|SAMPLE|FAKE|X+)", re.I),
    re.compile(r"^[A-Z]{1,2}\d{1,2}$", re.I),  # Too short: A1, AB12
]

# Generic/suspicious DOB or date patterns (first of month/year common in fakes)
GENERIC_DATE_PATTERNS = [
    "01/01/", "1/1/", "01-01-", "001/01/",
    "12/31/", "99/99/", "00/00/",
]

# US state abbreviations for basic jurisdiction check
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "USA",
}


def _normalize_text(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, dict):
        return " ".join(str(v) for v in s.values())
    return str(s).upper().strip()


def _parse_date(mo: str, day: str, yr: str, default_century: int = 2000) -> Optional[datetime]:
    try:
        y = int(yr)
        if y < 100:
            y = default_century + y if y < 50 else 1900 + y
        return datetime(y, int(mo), int(day))
    except (ValueError, IndexError):
        return None


def _extract_dates_from_text(raw: str) -> Dict[str, Optional[datetime]]:
    """Extract expiry, DOB, issue dates from raw text using common patterns."""
    out = {"expiry": None, "dob": None, "issue": None}
    raw_upper = raw.replace("\n", " ").replace("\r", " ")
    raw_upper_norm = raw_upper.upper()

    # EXP 03/28/2026 or EXPIRATION 03-28-2026 or EXP 01/01/2020
    for m in re.finditer(r"EXP(?:IRATION)?\s*[:\s]*(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", raw_upper_norm):
        d = _parse_date(m.group(1), m.group(2), m.group(3), 2000)
        if d:
            out["expiry"] = d
            break

    # DOB 03/29/1981 or DATE OF BIRTH or 01/01/1997
    for m in re.finditer(r"(?:DOB|DATE\s*OF\s*BIRTH|BIRTH)\s*[:\s]*(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", raw_upper_norm):
        d = _parse_date(m.group(1), m.group(2), m.group(3), 1900)
        if d:
            out["dob"] = d
            break
    # Fallback: 8-digit DOB 03291981 or 03291981
    if not out["dob"]:
        for m in re.finditer(r"(?:DOB|BIRTH)\s*[:\s]*(\d{8})", raw_upper_norm):
            s = m.group(1)
            d = _parse_date(s[:2], s[2:4], s[4:8], 1900)
            if d:
                out["dob"] = d
                break

    # ISS 06/14/2021 or ISSUE
    for m in re.finditer(r"(?:ISS|ISSUE)\s*[:\s]*(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", raw_upper_norm):
        d = _parse_date(m.group(1), m.group(2), m.group(3), 2000)
        if d:
            out["issue"] = d
            break

    return out


def _looks_like_dob(s: str) -> bool:
    """True if value is likely a date (8 digits) not a license number."""
    s = (s or "").replace(" ", "").replace("-", "").replace("/", "")
    if len(s) != 8 or not s.isdigit():
        return False
    # DDMMYYYY or MMDDYYYY
    return True


def _extract_id_number(claim_data: Dict, raw_text: str) -> Optional[str]:
    """Extract license/document number from raw text first (to get DL like G1111111), then claim_data."""
    raw = (raw_text or "").strip()
    raw_upper = raw.upper()

    # Prefer raw text so we capture "ID Number: G1111111" or "License Number: G1111111", not DOB
    candidates: List[str] = []
    # "ID Number: G1111111" or "License Number: G1111111" or "DL G1111111" or "DL: V6167147"
    for pattern in [
        r"(?:ID\s+Number|License\s+Number|Document\s+Number)\s*[#:]?\s*([A-Z0-9]{4,20})",
        r"\bDL\s*[#:]?\s*([A-Z0-9]{4,20})",
        r"(?:License|ID)\s*[#:]?\s*([A-Z0-9]{4,20})",
    ]:
        for m in re.finditer(pattern, raw_upper, re.I):
            val = m.group(1).strip()
            if val and not _looks_like_dob(val):
                candidates.append(val)
    # Prefer one that contains at least one letter (typical for US state DLs like G1111111, V6167147)
    for c in candidates:
        if any(ch.isalpha() for ch in c):
            return c
    if candidates:
        return candidates[0]

    # From structured data (skip values that look like DOB)
    for key in ("license_number", "dl_number", "id_number", "document_number", "license_no", "dl"):
        val = claim_data.get(key)
        if val and isinstance(val, str) and len(val) > 2 and not _looks_like_dob(val):
            return val.strip()
    # Nested
    for section in ("claimant", "document", "license", "id"):
        obj = claim_data.get(section)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "number" in k.lower() or "num" in k.lower() or k.lower() in ("dl", "license_no"):
                    if v and isinstance(v, str) and len(v) > 2 and not _looks_like_dob(v):
                        return str(v).strip()
    # Recursively search nested dicts (skip DOB-like values)
    def find_in(obj: Any, depth: int = 0) -> Optional[str]:
        if depth > 3:
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                klo = str(k).lower() if k else ""
                if klo and ("dl" in klo or "license" in klo or "id_no" in klo):
                    if "dob" in klo or "birth" in klo or "date" in klo:
                        continue
                    if isinstance(v, str) and 4 <= len(v) <= 20 and v.replace(" ", "").isalnum() and not _looks_like_dob(v):
                        return v.strip()
                r = find_in(v, depth + 1)
                if r:
                    return r
        return None
    found = find_in(claim_data)
    if found:
        return found
    # Last resort: any DL/ID pattern in raw text
    for m in re.finditer(r"(?:DL|LICENSE|ID)\s*[#:]?\s*([A-Z0-9]{4,20})", raw_upper):
        val = m.group(1).strip()
        if not _looks_like_dob(val):
            return val
    return None


def _extract_physical_description(claim_data: Dict, raw_text: str) -> Dict[str, str]:
    """Extract hair color, eye color from document text for consistency checks."""
    out = {}
    raw_upper = raw_text.upper()

    for label, key in [("HAIR", ["hair", "hair_color", "hair_color_opt"]), ("EYES", ["eyes", "eye", "eye_color", "eye_color_opt"])]:
        val = None
        for k in key:
            v = claim_data.get(k)
            if v and isinstance(v, str):
                val = v.strip()
                break
        if not val and key[0] in raw_upper:
            m = re.search(rf"{key[0].upper()}\s*[:\s]*([A-Za-z]+)", raw_upper)
            if m:
                val = m.group(1).strip()
        if val:
            out[key[0]] = val.upper()
    return out


def _check_placeholder_id(id_number: Optional[str]) -> Tuple[bool, float, str]:
    """Return (is_placeholder, risk_0_100, description)."""
    if not id_number or len(id_number) < 4:
        return False, 0, ""
    for pat in PLACEHOLDER_ID_PATTERNS:
        if pat.search(id_number):
            return True, 85, f"Placeholder or fake ID number pattern: '{id_number}'"
    # Same digit repeated
    if len(set(id_number.replace(" ", ""))) == 1 and len(id_number) >= 6:
        return True, 90, f"Repetitive ID number (all same digit): '{id_number}'"
    return False, 0, ""


def _check_expired(expiry: Optional[datetime]) -> Tuple[bool, float, str]:
    if not expiry:
        return False, 0, ""
    if expiry < datetime.now():
        return True, 70, f"Document expired on {expiry.strftime('%Y-%m-%d')}"
    return False, 0, ""


def _check_generic_dates(raw_text: str, claim_data: Dict) -> List[Tuple[float, str]]:
    """Check for generic/suspicious dates (01/01/1997, etc.)."""
    flags = []
    text = _normalize_text(claim_data) + " " + raw_text.upper()
    for pattern in GENERIC_DATE_PATTERNS:
        if pattern.replace("/", "").replace("-", "") in text.replace("/", "").replace("-", ""):
            flags.append((50, f"Generic or placeholder date pattern: {pattern}..."))
    return flags


def _check_issue_vs_dob(issue: Optional[datetime], dob: Optional[datetime]) -> Tuple[bool, float, str]:
    """Issue date should be after person is at least 14–16 (e.g. driver license)."""
    if not issue or not dob:
        return False, 0, ""
    age_at_issue = (issue - dob).days / 365.25
    if age_at_issue < 14:
        return True, 75, f"Issue date implies age {age_at_issue:.0f} at issuance (suspicious for ID)"
    return False, 0, ""


def _check_address_red_flags(raw_text: str, claim_data: Dict) -> List[Tuple[float, str]]:
    """Suite/STE in address can be red flag for non-residential on driver license."""
    flags = []
    text = _normalize_text(claim_data) + " " + raw_text.upper()
    if re.search(r"\b(?:STE|SUITE|UNIT|#)\s*\d+\b", text) and re.search(r"DRIVER|LICENSE|DL\b", text):
        flags.append((25, "Commercial-style address (suite/unit) on driver license"))
    return flags


class IDConsistencyAgent:
    """Rule-based agent for ID document plausibility and consistency checks."""

    def __init__(self):
        try:
            from core.nim_client import get_nim_client
            self._nim = get_nim_client()
        except Exception:
            self._nim = None

    async def _check_physical_vs_photo(
        self,
        image_paths: List[str],
        physical: Dict[str, str],
        raw_text: str,
    ) -> Optional[Dict]:
        """Use vision model to check if listed hair/eye matches person in photo. Returns flag dict if mismatch."""
        if not image_paths or not physical or not self._nim:
            return None
        import base64
        from pathlib import Path
        path = Path(image_paths[0])
        if not path.exists():
            return None
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        hair = physical.get("hair", "unknown")
        eyes = physical.get("eyes", "unknown")
        prompt = f"""Look at this ID document image. The document states: HAIR = {hair}, EYES = {eyes}.
Does the person visible in the photo clearly match this description?
- If the visible hair color clearly does NOT match {hair} (e.g. document says RED but photo shows brown/black), answer: MISMATCH
- If the visible eye color clearly does NOT match {eyes}, answer: MISMATCH
- If you cannot see the person clearly or colors match, answer: MATCH or UNKNOWN.
Reply with exactly one word: MISMATCH, MATCH, or UNKNOWN."""

        try:
            response = await self._nim.chat(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                model="nvidia/nemotron-4-340b-instruct",
                max_tokens=20,
            )
            reply = (response or "").strip().upper()
            if "MISMATCH" in reply:
                return {
                    "type": "physical_description_mismatch",
                    "severity": "critical",
                    "description": f"Listed physical description (hair: {hair}, eyes: {eyes}) does not match the person visible in the photo.",
                    "confidence": 0.9,
                }
        except Exception as e:
            logger.warning(f"Physical vs photo check failed: {e}")
        return None

    async def analyze(
        self,
        claim_data: Dict[str, Any],
        raw_text: str = "",
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run deterministic and heuristic checks on extracted ID data.

        Returns:
            risk_score 0-100, flags list, summary, and sub-scores for UI.
        """
        logger.info("ID Consistency Agent: running plausibility checks")

        raw_text = raw_text or _normalize_text(claim_data)
        flags: List[Dict[str, Any]] = []
        risk_score = 0
        sub_scores: Dict[str, float] = {}

        # 1) Placeholder / fake ID number
        id_number = _extract_id_number(claim_data, raw_text)
        is_placeholder, place_risk, place_desc = _check_placeholder_id(id_number)
        if is_placeholder:
            risk_score = max(risk_score, place_risk)
            sub_scores["id_number"] = place_risk
            flags.append({
                "type": "placeholder_id",
                "severity": "critical",
                "description": place_desc,
                "confidence": 0.95,
            })

        # 2) Expired document
        dates = _extract_dates_from_text(raw_text)
        exp_red, exp_risk, exp_desc = _check_expired(dates["expiry"])
        if exp_red:
            risk_score = max(risk_score, exp_risk)
            sub_scores["expired"] = exp_risk
            flags.append({
                "type": "expired_document",
                "severity": "high",
                "description": exp_desc,
                "confidence": 0.99,
            })

        # 3) Generic dates
        for score, desc in _check_generic_dates(raw_text, claim_data):
            risk_score = max(risk_score, score)
            flags.append({
                "type": "generic_date",
                "severity": "medium",
                "description": desc,
                "confidence": 0.8,
            })
        if flags and any(f.get("type") == "generic_date" for f in flags):
            sub_scores["generic_date"] = 50

        # 4) Issue vs DOB sanity
        issue_red, issue_risk, issue_desc = _check_issue_vs_dob(dates["issue"], dates["dob"])
        if issue_red:
            risk_score = max(risk_score, issue_risk)
            sub_scores["issue_dob"] = issue_risk
            flags.append({
                "type": "issue_dob_inconsistent",
                "severity": "high",
                "description": issue_desc,
                "confidence": 0.9,
            })

        # 5) Address red flags
        for score, desc in _check_address_red_flags(raw_text, claim_data):
            risk_score = min(100, risk_score + score)
            flags.append({
                "type": "address_red_flag",
                "severity": "low",
                "description": desc,
                "confidence": 0.6,
            })

        # 6) Physical description vs photo (vision check)
        physical = _extract_physical_description(claim_data, raw_text)
        if physical:
            sub_scores["physical_fields_present"] = 0
            photo_flag = await self._check_physical_vs_photo(
                image_paths or [], physical, raw_text
            )
            if photo_flag:
                risk_score = max(risk_score, 92)
                sub_scores["physical_photo_mismatch"] = 92
                flags.append(photo_flag)

        # Cap and build summary
        risk_score = min(100, round(risk_score, 1))
        if not flags:
            summary = "No consistency red flags detected."
        else:
            critical = sum(1 for f in flags if f.get("severity") == "critical")
            high = sum(1 for f in flags if f.get("severity") == "high")
            summary = f"{len(flags)} plausibility flag(s): {critical} critical, {high} high. "
            summary += " ".join(f.get("description", "")[:80] for f in flags[:2])

        return {
            "risk_score": risk_score,
            "flags": flags,
            "summary": summary,
            "sub_scores": sub_scores,
            "id_number_checked": id_number,
            "dates_checked": {
                "expiry": dates["expiry"].isoformat() if dates["expiry"] else None,
                "dob": dates["dob"].isoformat() if dates["dob"] else None,
                "issue": dates["issue"].isoformat() if dates["issue"] else None,
            },
            "physical_description_present": bool(physical),
        }
