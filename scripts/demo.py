#!/usr/bin/env python3
"""
FraudLens AI - Demo Script
Processes a sample claim and displays results.

Usage:
    python scripts/demo.py                         # Use sample_claim.json
    python scripts/demo.py path/to/claim.json      # Use custom claim
    NVIDIA_API_KEY=nvapi-... python scripts/demo.py  # With API key
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(text: str, char: str = "="):
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_section(title: str, content: str = ""):
    print(f"\n--- {title} ---")
    if content:
        print(content)


def print_risk_badge(level: str, score: float):
    badges = {
        "critical": "🔴 CRITICAL",
        "high": "🟠 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🟢 LOW",
    }
    badge = badges.get(level, f"⚪ {level.upper()}")
    print(f"\n  Fraud Score:  {score:.1f} / 100")
    print(f"  Risk Level:   {badge}")


async def run_demo(claim_path: str):
    from fraudlens import FraudLensAI

    print_header("FraudLens AI - Multi-Agent Fraud Detection Demo")
    print("  Powered by NVIDIA NIM | NeMo Retriever | cuGraph")
    print("=" * 70)

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("\n  NOTE: NVIDIA_API_KEY not set.")
        print("  The demo will attempt to run but API calls may fail.")
        print("  Set it with: export NVIDIA_API_KEY=nvapi-...")

    print(f"\n  Input: {claim_path}")

    # Load and preview the claim
    with open(claim_path, "r") as f:
        claim_data = json.load(f)

    cd = claim_data.get("claim_data", claim_data)
    claimant = cd.get("claimant", {}).get("name", "Unknown")
    amount = cd.get("claim", {}).get("amount", 0)
    claim_type = cd.get("claim", {}).get("type", "unknown")

    print(f"  Claimant:  {claimant}")
    print(f"  Amount:    ${amount:,.2f}")
    print(f"  Type:      {claim_type}")

    print_section("Phase 1: Initializing Multi-Agent System")
    detector = FraudLensAI(api_key=api_key)
    print("  [OK] 7 agents initialized")
    print("    - DocumentAgent (Nemotron-Parse)")
    print("    - InconsistencyAgent (NIM LLM)")
    print("    - PatternAgent (NeMo Retriever RAG)")
    print("    - ScoringAgent (weighted ensemble)")
    print("    - NarrativeAgent (NIM LLM)")
    print("    - NetworkAgent (cuGraph)")
    print("    - DeepfakeAgent (TensorRT)")

    print_section("Phase 2: Running Analysis Pipeline")
    print("  Extracting document data...")
    print("  Running parallel analysis (inconsistency + pattern + network)...")

    try:
        result = await detector.analyze(claim_path, include_network=True, include_deepfake=False)

        print_section("Phase 3: Results")
        print_risk_badge(result.risk_level, result.fraud_score)
        print(f"  Recommendation: {result.recommendation}")
        print(f"  Fraud Ring:     {'DETECTED' if result.fraud_ring_detected else 'Not detected'}")

        # Inconsistencies
        incs = result.inconsistencies.get("inconsistencies", [])
        print_section(f"Inconsistencies ({len(incs)} found)")
        for i, inc in enumerate(incs[:5], 1):
            print(f"  {i}. [{inc.get('severity', '?').upper()}] {inc.get('description', 'N/A')}")

        # Pattern matches
        patterns = result.pattern_matches.get("matched_patterns", [])
        print_section(f"Pattern Matches ({len(patterns)} found)")
        for i, pat in enumerate(patterns[:5], 1):
            print(f"  {i}. {pat.get('pattern_name', 'Unknown')} "
                  f"(similarity: {pat.get('similarity_score', 0):.2f}, "
                  f"severity: {pat.get('severity', '?')})")

        # Scoring breakdown
        if result.scoring_details:
            factors = result.scoring_details.get("risk_factors", [])
            print_section("Scoring Breakdown")
            for f in factors:
                bar_len = int(f.get("score", 0) / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  {f['name']:25s} {bar} {f['score']:.0f}/100 (x{f['weight']:.2f})")

        # Narrative
        print_section("Investigation Narrative")
        if result.narrative:
            # Print first 500 chars
            narrative = result.narrative
            if len(narrative) > 800:
                print(f"  {narrative[:800]}...")
                print(f"  [... {len(narrative) - 800} more characters]")
            else:
                print(f"  {narrative}")

        # Save report
        report_path = PROJECT_ROOT / "demo_report.json"
        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        print_header("Demo Complete", "=")
        print(f"  Full report saved to: {report_path}")
        print(f"  Run the UI: streamlit run ui/app.py")
        print("=" * 70)

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n  Make sure NVIDIA_API_KEY is set and valid.")
        print("  Get your key at: https://build.nvidia.com")
        sys.exit(1)


def main():
    if len(sys.argv) > 1:
        claim_path = sys.argv[1]
    else:
        claim_path = str(PROJECT_ROOT / "sample_claim.json")

    if not Path(claim_path).exists():
        print(f"Error: Claim file not found: {claim_path}")
        sys.exit(1)

    asyncio.run(run_demo(claim_path))


if __name__ == "__main__":
    main()
