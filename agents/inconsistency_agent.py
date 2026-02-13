"""
Inconsistency Agent
Detects logical contradictions and timeline impossibilities using NVIDIA NIM
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from loguru import logger

from core.nim_client import get_nim_client


@dataclass
class Inconsistency:
    """Represents a detected inconsistency"""
    type: str  # timeline, logical, factual, numerical
    description: str
    severity: str  # low, medium, high, critical
    confidence: float
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


class InconsistencyAgent:
    """
    Detects inconsistencies and contradictions in insurance claims.
    
    Analyzes claims for:
    - Timeline impossibilities (events happening in impossible order)
    - Logical contradictions (conflicting statements)
    - Factual discrepancies (mismatched data)
    - Numerical inconsistencies (amounts that don't add up)
    """
    
    def __init__(self):
        self.nim_client = get_nim_client()
        logger.info("InconsistencyAgent initialized with NVIDIA NIM")
    
    async def analyze(self, claim_data: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        """
        Analyze claim for inconsistencies.
        
        Args:
            claim_data: Structured claim data from DocumentAgent
            raw_text: Raw text from document
            
        Returns:
            Dictionary with detected inconsistencies
        """
        logger.info("InconsistencyAgent analyzing claim")
        
        try:
            # Run multiple analysis passes; don't let NIM failures kill the whole run
            timeline_issues = []
            logical_issues = []
            try:
                timeline_issues = await self._check_timeline(claim_data, raw_text)
            except Exception as e:
                logger.warning(f"InconsistencyAgent timeline check failed: {e}")
            try:
                logical_issues = await self._check_logical_consistency(claim_data, raw_text)
            except Exception as e:
                logger.warning(f"InconsistencyAgent logical check failed: {e}")
            numerical_issues = await self._check_numerical_consistency(claim_data)
            
            # Combine all inconsistencies
            all_inconsistencies = timeline_issues + logical_issues + numerical_issues
            
            # Calculate overall inconsistency score
            score = self._calculate_score(all_inconsistencies)
            
            return {
                "status": "success",
                "inconsistencies": [i.to_dict() for i in all_inconsistencies],
                "inconsistency_count": len(all_inconsistencies),
                "inconsistency_score": score,
                "summary": self._generate_summary(all_inconsistencies),
            }
            
        except Exception as e:
            logger.error(f"InconsistencyAgent error: {e}")
            # Still try numerical-only so we don't return 0 when only NIM failed
            try:
                numerical_issues = await self._check_numerical_consistency(claim_data)
                score = self._calculate_score(numerical_issues)
                if score == 0:
                    score = 10.0  # Small baseline so total score isn't killed when LLM is down
                return {
                    "status": "fallback",
                    "error": str(e),
                    "inconsistencies": [i.to_dict() for i in numerical_issues],
                    "inconsistency_count": len(numerical_issues),
                    "inconsistency_score": score,
                    "summary": self._generate_summary(numerical_issues) or "LLM checks unavailable; numerical check only.",
                }
            except Exception:
                return {
                    "status": "error",
                    "error": str(e),
                    "inconsistencies": [],
                    "inconsistency_score": 0,
                }
    
    async def _check_timeline(self, claim_data: Dict, raw_text: str) -> List[Inconsistency]:
        """Check for timeline impossibilities"""
        
        prompt = f"""Analyze this insurance claim for timeline inconsistencies.

CLAIM DATA:
- Incident Date: {claim_data.get('incident', {}).get('date', 'Unknown')}
- Incident Time: {claim_data.get('incident', {}).get('time', 'Unknown')}
- Claim Filed: {claim_data.get('claim', {}).get('date', 'Unknown')}
- Incident Location: {claim_data.get('incident', {}).get('location', 'Unknown')}

CLAIM TEXT:
{raw_text[:4000]}

Look for:
1. Events happening in impossible order
2. Dates that don't make sense (claim before incident, etc.)
3. Time gaps that are suspicious
4. Location/time conflicts (person in two places at once)
5. Treatment dates before injury dates

For each inconsistency found, provide:
- Type: timeline
- Description: What is inconsistent
- Severity: low/medium/high/critical
- Evidence: Specific text or data that shows the inconsistency

If no timeline inconsistencies found, state "No timeline inconsistencies detected."
"""

        response = await self.nim_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500,
        )
        
        return self._parse_inconsistencies(response, "timeline")
    
    async def _check_logical_consistency(self, claim_data: Dict, raw_text: str) -> List[Inconsistency]:
        """Check for logical contradictions"""
        
        prompt = f"""Analyze this insurance claim for logical contradictions.

CLAIM DATA:
{claim_data}

CLAIM TEXT:
{raw_text[:4000]}

Look for:
1. Contradictory statements (saying opposite things)
2. Impossible scenarios described
3. Witness statements that conflict
4. Description of damage inconsistent with accident type
5. Claimed injuries inconsistent with incident description

For each contradiction found, provide:
- Type: logical
- Description: What contradicts what
- Severity: low/medium/high/critical
- Evidence: Specific contradicting statements

If no logical contradictions found, state "No logical contradictions detected."
"""

        response = await self.nim_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500,
        )
        
        return self._parse_inconsistencies(response, "logical")
    
    async def _check_numerical_consistency(self, claim_data: Dict) -> List[Inconsistency]:
        """Check for numerical inconsistencies"""
        
        inconsistencies = []
        
        claim = claim_data.get("claim", {})
        medical = claim_data.get("medical", {})
        policy = claim_data.get("policy", {})
        
        # Check if claim exceeds coverage
        claim_amount = claim.get("amount", 0)
        coverage = policy.get("coverage_amount", 0)
        
        if claim_amount > 0 and coverage > 0:
            if claim_amount > coverage * 1.5:  # 50% over coverage is suspicious
                inconsistencies.append(Inconsistency(
                    type="numerical",
                    description=f"Claim amount (${claim_amount:,.2f}) significantly exceeds policy coverage (${coverage:,.2f})",
                    severity="high" if claim_amount > coverage * 2 else "medium",
                    confidence=0.9,
                    evidence=[f"Claim: ${claim_amount:,.2f}", f"Coverage: ${coverage:,.2f}"],
                ))
        
        # Check medical costs vs claim amount
        medical_costs = medical.get("costs", 0)
        if medical_costs > 0 and claim_amount > 0:
            if medical_costs > claim_amount:
                inconsistencies.append(Inconsistency(
                    type="numerical",
                    description=f"Medical costs (${medical_costs:,.2f}) exceed total claim amount (${claim_amount:,.2f})",
                    severity="medium",
                    confidence=0.85,
                    evidence=[f"Medical: ${medical_costs:,.2f}", f"Total Claim: ${claim_amount:,.2f}"],
                ))
        
        return inconsistencies
    
    def _parse_inconsistencies(self, response: str, inconsistency_type: str) -> List[Inconsistency]:
        """Parse LLM response into Inconsistency objects"""
        
        inconsistencies = []
        
        # Check if no inconsistencies found
        if "no" in response.lower() and ("inconsistenc" in response.lower() or "contradiction" in response.lower()):
            return []
        
        # Parse response for inconsistency patterns
        lines = response.split("\n")
        current = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Look for severity indicators
            if any(s in line_lower for s in ["critical", "high", "medium", "low"]):
                if "critical" in line_lower:
                    severity = "critical"
                elif "high" in line_lower:
                    severity = "high"
                elif "medium" in line_lower:
                    severity = "medium"
                else:
                    severity = "low"
                
                if current:
                    current.severity = severity
            
            # Look for description patterns
            elif "description" in line_lower and ":" in line:
                desc = line.split(":", 1)[1].strip()
                if current is None:
                    current = Inconsistency(
                        type=inconsistency_type,
                        description=desc,
                        severity="medium",
                        confidence=0.8,
                    )
                else:
                    current.description = desc
            
            # Look for evidence
            elif "evidence" in line_lower and ":" in line:
                evidence = line.split(":", 1)[1].strip()
                if current:
                    current.evidence.append(evidence)
            
            # Check for numbered items that might be inconsistencies
            elif line[0].isdigit() and "." in line[:3]:
                # Save previous if exists
                if current and current.description:
                    inconsistencies.append(current)
                
                # Start new inconsistency
                desc = line.split(".", 1)[1].strip() if "." in line else line
                current = Inconsistency(
                    type=inconsistency_type,
                    description=desc,
                    severity="medium",
                    confidence=0.75,
                )
        
        # Don't forget the last one
        if current and current.description:
            inconsistencies.append(current)
        
        return inconsistencies
    
    def _calculate_score(self, inconsistencies: List[Inconsistency]) -> float:
        """Calculate overall inconsistency score (0-100)"""
        if not inconsistencies:
            return 0.0
        
        severity_weights = {
            "critical": 30,
            "high": 20,
            "medium": 10,
            "low": 5,
        }
        
        score = 0
        for inc in inconsistencies:
            weight = severity_weights.get(inc.severity, 5)
            score += weight * inc.confidence
        
        # Cap at 100
        return min(100, score)
    
    def _generate_summary(self, inconsistencies: List[Inconsistency]) -> str:
        """Generate human-readable summary"""
        if not inconsistencies:
            return "No significant inconsistencies detected in the claim."
        
        critical = [i for i in inconsistencies if i.severity == "critical"]
        high = [i for i in inconsistencies if i.severity == "high"]
        
        summary_parts = []
        
        if critical:
            summary_parts.append(f"{len(critical)} critical inconsistencies found")
        if high:
            summary_parts.append(f"{len(high)} high-severity issues detected")
        
        summary_parts.append(f"Total of {len(inconsistencies)} inconsistencies identified")
        
        return ". ".join(summary_parts) + "."
