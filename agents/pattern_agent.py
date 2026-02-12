"""
Pattern Agent
Matches claims against known fraud patterns using NeMo Retriever RAG
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from loguru import logger

from core.nim_client import get_nim_client
from core.embedding_service import VectorStore, EmbeddingService, SearchResult


@dataclass
class PatternMatch:
    """A matched fraud pattern"""
    pattern_id: str
    pattern_name: str
    description: str
    similarity_score: float
    category: str
    severity: str
    matching_elements: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "description": self.description,
            "similarity_score": self.similarity_score,
            "category": self.category,
            "severity": self.severity,
            "matching_elements": self.matching_elements,
        }


class PatternAgent:
    """
    Matches insurance claims against known fraud patterns.
    
    Uses NVIDIA NeMo Retriever for semantic search against a database
    of known fraud patterns and historical cases.
    """
    
    def __init__(self, vector_store: VectorStore = None):
        self.nim_client = get_nim_client()
        self.embedding_service = EmbeddingService()
        self.vector_store = vector_store
        
        logger.info("PatternAgent initialized with NeMo Retriever")
    
    async def initialize(self):
        """Initialize vector store if not provided"""
        if self.vector_store is None:
            self.vector_store = VectorStore(
                collection_name="fraud_patterns",
                embedding_service=self.embedding_service,
            )
            await self.vector_store.initialize()
            
            # Load default patterns
            from core.embedding_service import initialize_fraud_patterns
            await initialize_fraud_patterns(self.vector_store)
    
    async def analyze(
        self,
        claim_data: Dict[str, Any],
        raw_text: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze claim against known fraud patterns.
        
        Args:
            claim_data: Structured claim data
            raw_text: Raw text from document
            top_k: Number of patterns to match
            
        Returns:
            Dictionary with matched patterns and risk assessment
        """
        logger.info("PatternAgent analyzing claim")
        
        try:
            # Ensure vector store is initialized
            if self.vector_store is None:
                await self.initialize()
            
            # Create search query from claim
            query = self._build_search_query(claim_data, raw_text)
            
            # Search for similar patterns
            results = await self.vector_store.search(
                query=query,
                top_k=top_k,
                rerank=True,
            )
            
            # Convert to PatternMatch objects
            matches = await self._process_matches(results, claim_data, raw_text)
            
            # Calculate pattern risk score
            pattern_score = self._calculate_pattern_score(matches)
            
            return {
                "status": "success",
                "matched_patterns": [m.to_dict() for m in matches],
                "pattern_count": len(matches),
                "pattern_risk_score": pattern_score,
                "top_pattern": matches[0].to_dict() if matches else None,
                "summary": self._generate_summary(matches),
            }
            
        except Exception as e:
            logger.error(f"PatternAgent error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "matched_patterns": [],
                "pattern_risk_score": 0,
            }
    
    def _build_search_query(self, claim_data: Dict, raw_text: str) -> str:
        """Build semantic search query from claim data"""
        
        query_parts = []
        
        # Add incident description
        incident = claim_data.get("incident", {}) or {}
        if incident.get("description"):
            query_parts.append(f"Incident: {incident['description']}")
        
        # Add claim type and amount
        claim = claim_data.get("claim", {}) or {}
        if claim.get("type"):
            query_parts.append(f"Claim type: {claim['type']}")
        if claim.get("amount"):
            query_parts.append(f"Claim amount: ${claim['amount']:,.2f}")
        
        # Add injury information
        medical = claim_data.get("medical", {}) or {}
        injuries = medical.get("injuries") or []
        if injuries:
            inj_str = injuries if isinstance(injuries, str) else ", ".join(str(x) for x in injuries)
            query_parts.append(f"Injuries: {inj_str}")
        
        # Add key phrases from raw text (first 1500 chars for better coverage)
        if raw_text and raw_text.strip():
            query_parts.append(f"Details: {raw_text[:1500].strip()}")
        
        query = " ".join(query_parts).strip()
        # Fallback: never return empty query — use raw text or generic placeholder
        if not query:
            query = raw_text[:2000].strip() if raw_text and raw_text.strip() else "insurance claim document"
        return query
    
    async def _process_matches(
        self,
        results: List[SearchResult],
        claim_data: Dict,
        raw_text: str,
    ) -> List[PatternMatch]:
        """Process search results into PatternMatch objects with analysis"""
        
        matches = []
        
        for result in results:
            # Skip low similarity results (0.2 threshold to avoid overly strict filtering)
            if result.score < 0.2:
                continue
            
            # Use LLM to analyze the match
            analysis = await self._analyze_match(result, claim_data, raw_text)
            
            match = PatternMatch(
                pattern_id=result.id,
                pattern_name=result.metadata.get("category", "Unknown Pattern"),
                description=result.text,
                similarity_score=result.score,
                category=result.metadata.get("category", "general"),
                severity=result.metadata.get("severity", "medium"),
                matching_elements=analysis.get("matching_elements", []),
            )
            
            matches.append(match)
        
        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return matches
    
    async def _analyze_match(
        self,
        result: SearchResult,
        claim_data: Dict,
        raw_text: str,
    ) -> Dict[str, Any]:
        """Use LLM to analyze why a pattern matches"""
        
        prompt = f"""Compare this insurance claim to a known fraud pattern.

KNOWN FRAUD PATTERN:
{result.text}

CLAIM SUMMARY:
- Type: {claim_data.get('claim', {}).get('type', 'Unknown')}
- Amount: ${claim_data.get('claim', {}).get('amount', 0):,.2f}
- Incident: {claim_data.get('incident', {}).get('description', 'Unknown')[:300]}

CLAIM TEXT (excerpt):
{raw_text[:1000]}

Identify specific elements in this claim that match the fraud pattern.
List each matching element as a bullet point.
If the match is weak or coincidental, state that.

Format:
- Matching Element 1: [description]
- Matching Element 2: [description]
...
"""

        try:
            response = await self.nim_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            
            # Parse matching elements
            elements = []
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("•"):
                    element = line.lstrip("-•").strip()
                    if element and len(element) > 10:
                        elements.append(element)
            
            return {"matching_elements": elements[:5]}  # Limit to top 5
            
        except Exception as e:
            logger.warning(f"Match analysis error: {e}")
            return {"matching_elements": []}
    
    def _calculate_pattern_score(self, matches: List[PatternMatch]) -> float:
        """Calculate overall pattern risk score (0-100)"""
        if not matches:
            return 0.0
        
        severity_multipliers = {
            "critical": 2.0,
            "high": 1.5,
            "medium": 1.0,
            "low": 0.5,
        }
        
        score = 0
        for match in matches:
            multiplier = severity_multipliers.get(match.severity, 1.0)
            # Weight by similarity and number of matching elements
            match_score = match.similarity_score * multiplier * (1 + len(match.matching_elements) * 0.1)
            score += match_score * 20  # Scale to 0-100 range
        
        # Cap at 100
        return min(100, score)
    
    def _generate_summary(self, matches: List[PatternMatch]) -> str:
        """Generate human-readable summary"""
        if not matches:
            return "No known fraud patterns matched this claim."
        
        critical = [m for m in matches if m.severity == "critical"]
        high = [m for m in matches if m.severity == "high"]
        
        summary_parts = []
        
        if critical:
            patterns = ", ".join([m.category for m in critical])
            summary_parts.append(f"CRITICAL: Matches {len(critical)} critical pattern(s): {patterns}")
        
        if high:
            patterns = ", ".join([m.category for m in high])
            summary_parts.append(f"HIGH: Matches {len(high)} high-risk pattern(s): {patterns}")
        
        if not summary_parts:
            top = matches[0]
            summary_parts.append(
                f"Claim shows similarity to {top.category} pattern "
                f"(score: {top.similarity_score:.2f})"
            )
        
        return " | ".join(summary_parts)
