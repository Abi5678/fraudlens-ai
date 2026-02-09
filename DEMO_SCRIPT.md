# FraudLens AI - 2-Minute Demo Video Script

## Setup (before recording)
- Have Streamlit UI running: `streamlit run ui/app.py`
- Have `sample_claim.json` ready
- Terminal open for CLI demo
- NVIDIA API key configured

---

## SCENE 1: Hook (0:00 - 0:15)

**[Screen: Title card with FraudLens AI logo]**

**Narration:**
"Insurance fraud costs the industry $308 billion annually. Investigations take 60+ days per claim. FraudLens AI uses NVIDIA's full AI stack to analyze claims in under 5 minutes — detecting not just individual fraud, but organized fraud rings."

---

## SCENE 2: Architecture Overview (0:15 - 0:35)

**[Screen: Architecture diagram from README]**

**Narration:**
"FraudLens AI is a multi-agent system built entirely on NVIDIA technology:

- **Nemotron-Parse** extracts structured data from any document
- **NVIDIA NIM** powers 7 specialized AI agents running in parallel
- **NeMo Retriever** matches claims against known fraud patterns using RAG
- **cuGraph** detects fraud rings through GPU-accelerated graph analytics
- **Milvus** stores and searches fraud pattern embeddings"

---

## SCENE 3: Live Demo - Upload (0:35 - 0:55)

**[Screen: Streamlit UI]**

**Narration:**
"Let me walk through a live analysis. Here's a suspicious auto insurance claim — a hit-and-run with no witnesses, no police report, and $47,500 in soft-tissue injury claims."

**Action:**
1. Show the Streamlit dashboard
2. Click "Use Sample Claim" tab
3. Click "Load Sample Claim"
4. Click "Analyze Claim"
5. Show the spinner: "7 agents working in parallel..."

---

## SCENE 4: Results Walkthrough (0:55 - 1:30)

**[Screen: Results dashboard]**

**Narration:**
"In under a minute, our agents return a comprehensive analysis."

**Action — show each section:**

1. **Fraud Score gauge** — "A score of [X]/100 — [risk level] risk"
2. **Risk factors chart** — "The weighted scoring breaks down each risk dimension"
3. **Inconsistencies tab** — "The inconsistency agent found [N] red flags:
   - Claim filed 2 weeks after incident
   - Medical treatment delayed 5 days
   - Claim amount is 95% of coverage limit"
4. **Pattern matches tab** — "NeMo Retriever matched this against staged accident and inflated claim patterns"
5. **Network analysis** — "The network agent checked for fraud ring connections"
6. **Investigation narrative** — "And here's the AI-generated investigative report — ready for an adjuster to review"

---

## SCENE 5: Technical Differentiators (1:30 - 1:50)

**[Screen: Split — terminal running tests + architecture]**

**Narration:**
"What makes FraudLens AI unique:

1. **7 specialized agents** working in parallel — not a single prompt
2. **RAG-powered pattern matching** with NeMo Retriever embeddings and reranking
3. **Graph-based fraud ring detection** — because fraud is organized, not isolated
4. **Explainable AI** — every score comes with evidence and reasoning
5. **61 passing tests** covering all agents end-to-end"

**Action:** Show `pytest tests/ -v` running with all green

---

## SCENE 6: Impact & Close (1:50 - 2:00)

**[Screen: Impact metrics]**

**Narration:**
"For a mid-size insurer processing 50,000 claims per year:
- Investigation time: 60 days down to 5 minutes
- Cost per investigation: $500 down to $0.10
- Projected savings: $45 million annually

FraudLens AI — detecting fraud in minutes, not months. Built 100% on NVIDIA."

**[Screen: End card with repo URL and tech stack badges]**

---

## Recording Tips
- Use 1920x1080 resolution
- Zoom in on key UI elements when discussing them
- Keep transitions smooth — no dead air
- Show real API responses (pre-run if needed for reliability)
- End with the GitHub URL visible
