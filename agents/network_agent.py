"""
Network Agent - Detects fraud rings using graph analysis
Uses NVIDIA cuGraph for GPU-accelerated graph analytics when available
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from loguru import logger

from core.nim_client import get_nim_client


@dataclass
class FraudConnection:
    """Connection in fraud network"""
    entity1: str
    entity2: str
    connection_type: str
    strength: float
    evidence: str


class NetworkAgent:
    """
    Analyzes cross-claim relationships to detect fraud rings.
    Uses graph analysis to find suspicious connection patterns.
    """
    
    def __init__(self):
        self.nim_client = get_nim_client()
        self._cugraph_available = self._check_cugraph()
    
    def _check_cugraph(self) -> bool:
        """Check if cuGraph is available for GPU acceleration"""
        try:
            import cugraph
            return True
        except ImportError:
            logger.info("cuGraph not available, using CPU-based analysis")
            return False
    
    async def analyze(self, claim_data: Dict, historical_claims: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze claim for fraud ring connections.
        
        Looks for:
        - Shared addresses across claims
        - Common phone numbers
        - Same providers/attorneys
        - Coordinated claim timing
        - Linked vehicles or policies
        """
        logger.info("NetworkAgent analyzing fraud network")
        
        try:
            # Extract entities from current claim
            entities = self._extract_entities(claim_data)
            
            # If no historical data, use LLM to analyze potential connections
            if not historical_claims:
                return await self._analyze_with_llm(claim_data, entities)
            
            # Build network graph
            connections = self._find_connections(entities, historical_claims)
            
            # Detect communities (potential fraud rings)
            if self._cugraph_available and len(connections) > 10:
                communities = self._detect_communities_gpu(connections)
            else:
                communities = self._detect_communities_cpu(connections)
            
            # Calculate network risk
            risk_score = self._calculate_network_risk(connections, communities)
            
            return {
                "status": "success",
                "fraud_ring_detected": risk_score > 50,
                "network_risk_score": risk_score,
                "connections_found": len(connections),
                "individuals_count": len(set([c.entity1 for c in connections] + [c.entity2 for c in connections])),
                "key_connections": [f"{c.entity1} <-> {c.entity2} ({c.connection_type})" for c in connections[:5]],
                "communities": communities,
                "summary": self._generate_summary(connections, communities, risk_score),
            }
            
        except Exception as e:
            logger.error(f"NetworkAgent error: {e}")
            return {"status": "error", "error": str(e), "network_risk_score": 0}
    
    def _extract_entities(self, claim_data: Dict) -> Dict[str, Any]:
        """Extract network-relevant entities from claim"""
        claim = claim_data.get("claim_data", claim_data)
        
        return {
            "claimant_name": claim.get("claimant", {}).get("name", ""),
            "address": claim.get("claimant", {}).get("address", ""),
            "phone": claim.get("claimant", {}).get("phone", ""),
            "policy_number": claim.get("policy", {}).get("number", ""),
            "vehicle_vin": claim.get("vehicle", {}).get("vin", ""),
            "providers": claim.get("medical", {}).get("providers", []),
            "incident_location": claim.get("incident", {}).get("location", ""),
        }
    
    async def _analyze_with_llm(self, claim_data: Dict, entities: Dict) -> Dict[str, Any]:
        """Use LLM to analyze potential fraud network indicators"""

        # Get the full claim text for richer analysis
        raw_text = claim_data.get("raw_text", "")
        claim_inner = claim_data.get("claim_data", claim_data)
        incident_desc = claim_inner.get("incident", {}).get("description", "")
        medical_costs = claim_inner.get("medical", {}).get("costs", 0)
        claim_amount = claim_inner.get("claim", {}).get("amount", 0)
        witnesses = claim_inner.get("witnesses", [])

        prompt = f"""You are an insurance fraud network analyst. Analyze this claim for fraud ring indicators and provide a STRUCTURED assessment.

CLAIM ENTITIES:
- Claimant: {entities.get('claimant_name')}
- Address: {entities.get('address')}
- Phone: {entities.get('phone')}
- Medical Providers: {', '.join(entities.get('providers', []))}
- Incident Location: {entities.get('incident_location')}
- Vehicle VIN: {entities.get('vehicle_vin')}
- Policy: {entities.get('policy_number')}
- Claim Amount: ${claim_amount:,.2f}
- Medical Costs: ${medical_costs:,.2f}
- Witnesses: {len(witnesses)} {'(none)' if not witnesses else ''}
- Incident: {incident_desc[:300]}

Analyze these SPECIFIC fraud network indicators and rate each 0-100:

1. PROVIDER NETWORK RISK: Are the medical providers associated with fraud patterns (pain clinics, excessive treatment)?
2. GEOGRAPHIC CLUSTERING: Is the location/address in a known fraud hotspot or suspicious pattern?
3. STAGED ACCIDENT INDICATORS: Does the incident description suggest coordination (late night, no witnesses, hit-and-run)?
4. CLAIM COORDINATION: Does the timing, amount, or structure suggest organized activity?
5. IDENTITY/POLICY FLAGS: Any suspicious patterns in the claimant's information?

For each indicator provide:
- Score (0-100)
- Brief explanation (1-2 sentences)

Then give an OVERALL NETWORK RISK SCORE (0-100) and a 2-3 sentence summary.

Format your response as:
PROVIDER_RISK: [score] - [explanation]
GEOGRAPHIC_RISK: [score] - [explanation]
STAGED_ACCIDENT: [score] - [explanation]
CLAIM_COORDINATION: [score] - [explanation]
IDENTITY_FLAGS: [score] - [explanation]
OVERALL_SCORE: [score]
SUMMARY: [2-3 sentence overall assessment]"""

        response = await self.nim_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )

        # Parse structured response
        risk_score, indicators = self._parse_network_response(response)

        return {
            "status": "success",
            "fraud_ring_detected": risk_score > 50,
            "network_risk_score": risk_score,
            "analysis": response,
            "indicators": indicators,
            "summary": self._build_network_summary(risk_score, indicators),
        }

    def _parse_network_response(self, response: str) -> tuple:
        """Parse the structured LLM response into score and indicators."""
        import re

        indicators = []
        overall_score = 25  # default

        indicator_names = {
            "PROVIDER_RISK": "Provider Network Risk",
            "GEOGRAPHIC_RISK": "Geographic Clustering",
            "STAGED_ACCIDENT": "Staged Accident Indicators",
            "CLAIM_COORDINATION": "Claim Coordination",
            "IDENTITY_FLAGS": "Identity / Policy Flags",
        }

        for key, display_name in indicator_names.items():
            pattern = rf"{key}:\s*(\d+)\s*[-–]\s*(.*?)(?=\n[A-Z_]+:|\nOVERALL|\nSUMMARY|$)"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                score = min(int(match.group(1)), 100)
                explanation = match.group(2).strip()
                indicators.append({
                    "name": display_name,
                    "score": score,
                    "explanation": explanation,
                })

        # Parse overall score
        overall_match = re.search(r"OVERALL_SCORE:\s*(\d+)", response)
        if overall_match:
            overall_score = min(int(overall_match.group(1)), 100)
        elif indicators:
            # Fallback: average of indicators
            overall_score = int(sum(i["score"] for i in indicators) / len(indicators))
        else:
            # Keyword fallback
            resp_lower = response.lower()
            if "critical" in resp_lower or "definite" in resp_lower:
                overall_score = 85
            elif "high" in resp_lower or "significant" in resp_lower:
                overall_score = 65
            elif "low" in resp_lower or "minimal" in resp_lower:
                overall_score = 15

        # Parse summary
        summary_match = re.search(r"SUMMARY:\s*(.*?)$", response, re.DOTALL)
        if summary_match:
            parsed_summary = summary_match.group(1).strip()
            # Attach parsed summary to the first indicator list item as a special entry
            indicators.append({
                "name": "_llm_summary",
                "score": overall_score,
                "explanation": parsed_summary,
            })

        return overall_score, indicators

    def _build_network_summary(self, risk_score: float, indicators: list) -> str:
        """Build a human-readable summary from indicators."""
        # Use LLM's own summary if available
        for ind in indicators:
            if ind["name"] == "_llm_summary":
                return ind["explanation"]

        # Fallback summary
        if risk_score > 75:
            return f"CRITICAL: High fraud ring risk ({risk_score}/100). Multiple network indicators triggered."
        elif risk_score > 50:
            return f"HIGH RISK: Significant network indicators detected ({risk_score}/100)."
        elif risk_score > 25:
            return f"MODERATE: Some network indicators flagged for review ({risk_score}/100)."
        return f"LOW: No significant fraud network indicators ({risk_score}/100)."
    
    def _find_connections(self, entities: Dict, historical_claims: List[Dict]) -> List[FraudConnection]:
        """Find connections between current claim and historical claims"""
        connections = []
        
        for hist_claim in historical_claims:
            hist_entities = self._extract_entities(hist_claim)
            
            # Check address match
            if entities.get("address") and entities["address"] == hist_entities.get("address"):
                connections.append(FraudConnection(
                    entity1=entities["claimant_name"],
                    entity2=hist_entities["claimant_name"],
                    connection_type="shared_address",
                    strength=0.9,
                    evidence=f"Same address: {entities['address']}"
                ))
            
            # Check phone match
            if entities.get("phone") and entities["phone"] == hist_entities.get("phone"):
                connections.append(FraudConnection(
                    entity1=entities["claimant_name"],
                    entity2=hist_entities["claimant_name"],
                    connection_type="shared_phone",
                    strength=0.85,
                    evidence=f"Same phone: {entities['phone']}"
                ))
            
            # Check provider match
            for provider in entities.get("providers", []):
                if provider in hist_entities.get("providers", []):
                    connections.append(FraudConnection(
                        entity1=entities["claimant_name"],
                        entity2=hist_entities["claimant_name"],
                        connection_type="shared_provider",
                        strength=0.7,
                        evidence=f"Same provider: {provider}"
                    ))
        
        return connections
    
    def _detect_communities_gpu(self, connections: List[FraudConnection]) -> List[Dict]:
        """Detect fraud communities using cuGraph"""
        import cudf
        import cugraph
        
        # Build edge list
        edges = [(c.entity1, c.entity2, c.strength) for c in connections]
        df = cudf.DataFrame(edges, columns=['src', 'dst', 'weight'])
        
        G = cugraph.Graph()
        G.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight')
        
        # Run Louvain community detection
        parts, modularity = cugraph.louvain(G)
        
        # Convert to community dict
        communities = parts.to_pandas().groupby('partition')['vertex'].apply(list).to_dict()
        
        return [{"id": k, "members": v, "size": len(v)} for k, v in communities.items()]
    
    def _detect_communities_cpu(self, connections: List[FraudConnection]) -> List[Dict]:
        """Simple community detection without GPU"""
        # Build adjacency
        nodes = set()
        adj = {}
        for c in connections:
            nodes.add(c.entity1)
            nodes.add(c.entity2)
            adj.setdefault(c.entity1, set()).add(c.entity2)
            adj.setdefault(c.entity2, set()).add(c.entity1)
        
        # Simple connected components
        visited = set()
        communities = []
        
        for node in nodes:
            if node in visited:
                continue
            community = []
            stack = [node]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                community.append(n)
                stack.extend(adj.get(n, []))
            if len(community) > 1:
                communities.append({"id": len(communities), "members": community, "size": len(community)})
        
        return communities
    
    def _calculate_network_risk(self, connections: List[FraudConnection], communities: List[Dict]) -> float:
        """Calculate overall network risk score"""
        if not connections:
            return 0
        
        base_score = min(len(connections) * 10, 50)
        
        # Add score for large communities
        for community in communities:
            if community["size"] > 3:
                base_score += 20
            elif community["size"] > 2:
                base_score += 10
        
        # Add score for strong connections
        strong = sum(1 for c in connections if c.strength > 0.8)
        base_score += strong * 5
        
        return min(base_score, 100)
    
    def _generate_summary(self, connections: List[FraudConnection], communities: List[Dict], risk_score: float) -> str:
        if risk_score > 75:
            return f"CRITICAL: Fraud ring detected with {len(connections)} connections across {len(communities)} communities"
        elif risk_score > 50:
            return f"HIGH RISK: Significant network connections found ({len(connections)} links)"
        elif risk_score > 25:
            return f"MODERATE: Some network connections identified for review"
        return "LOW: No significant fraud network indicators"
