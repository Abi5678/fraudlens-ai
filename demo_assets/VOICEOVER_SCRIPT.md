# FraudLens AI — End-to-End Voiceover Script (with visual cues)

Use this for recording: read the **Narration** and follow the **Visual** cues so the voiceover matches what’s on screen. Narration uses the actual results from the demo (Auto 61.4, 67% confidence, 23 images; MedClaim 37 medium, 95% confidence; Photo ID 70 critical).

---

## SCENE 1 — Hook (Title card)

**Visual:** Show title card with FraudLens AI logo.

**Narration:**  
"Insurance fraud costs the industry 308 billion dollars annually. Investigations take 60 plus days per claim. FraudLens AI uses NVIDIA's full AI stack to analyze claims in under 5 minutes — detecting not just individual fraud, but organized fraud rings."

---

## SCENE 2 — Architecture

**Visual:** Show architecture diagram (e.g. from README or demo_assets/architecture_slide.html). Highlight Nemotron, NIM, NeMo, cuGraph, Milvus as narrated.

**Narration:**  
"FraudLens AI is a multi-agent system built entirely on NVIDIA technology. Nemotron-Parse extracts structured data from any document. NVIDIA NIM powers 7 specialized AI agents running in parallel. NeMo Retriever matches claims against known fraud patterns using RAG. cuGraph detects fraud rings through GPU-accelerated graph analytics. Milvus stores and searches fraud pattern embeddings."

---

## SCENE 3 — Auto Insurance: upload and analyze

**Visual:** Show Streamlit dashboard. Select Auto Insurance. Click "Use sample claim" (or "Upload file" and load sample). Click "Analyze Claim". Show spinner / "7 agents working in parallel".

**Narration:**  
"Let me walk through a live analysis. Here's a suspicious auto insurance claim — a hit-and-run with no witnesses, no police report, and 47,500 dollars in soft-tissue injury claims. We'll use the sample claim and run the analysis."

---

## SCENE 4 — Auto Insurance: results

**Visual:** Show Auto results: Fraud Score 61.4, 67% confidence, 23 images; risk gauge, Risk Factor Breakdown (Inconsistencies, Fraud Pattern Match, Network/Ring). Highlight 7 inconsistencies, 5 patterns. Show recommendation "INVESTIGATE - Assign to fraud analyst". Optionally scroll to Key Findings and "Ask about this analysis" chat.

**Narration:**  
"In under a minute, our agents return a comprehensive analysis. For this claim we see a fraud score of 61.4 out of 100 — high risk. The system identified 7 inconsistencies and 5 pattern matches, and flagged potential fraud ring connections. It analyzed 23 images and reports 67% confidence in the findings. The risk breakdown shows high scores in inconsistencies, fraud pattern match, and network or ring risk. The recommendation is clear: investigate, and assign to a fraud analyst. Key findings include inconsistent documentation, unverifiable information, suspicious transaction patterns, and similarity to known fraud schemes. The AI-generated investigative report is ready for an adjuster to review."

---

## SCENE 5 — Medical Insurance (MedClaim)

**Visual:** Switch to Medical Insurance in sidebar. Click "Use sample claim". Click "Analyze Claim". Show results: Fraud Score 37, MEDIUM, 95% confidence. Highlight Billing (2), Clinical (2), Eligibility (2), Denial Risk No. Show risk breakdown: Inconsistencies 56, Fraud Pattern 36. Show 7 inconsistencies, 0 pattern matches. Show recommendation "REVIEW - Additional documentation required". Optionally show "Ask about this analysis".

**Narration:**  
"FraudLens also runs MedClaim AI for medical billing fraud — same NVIDIA stack, different vertical. We switch to Medical Insurance and use the sample claim. MedClaim analyzes billing codes, clinical consistency, and eligibility, and can process CMS-1500 forms, EOBs, clinical notes, and JSON with CPT and ICD-10 codes. For this claim the fraud score is 37 — medium risk — with 95% confidence. The breakdown shows scores of 2 each for billing, clinical, and eligibility, and the system has marked denial risk as no. The risk factors show 56 for inconsistencies and 36 for fraud pattern match, with 7 inconsistencies and 0 pattern matches detected. The recommendation is to review — additional documentation required. Analysts can dig deeper using the risk factor breakdown and the Ask about this analysis chat."

---

## SCENE 6 — Photo ID Check (IDVerify)

**Visual:** Switch to Photo ID Check. Click "Use sample ID". Click "Verify ID". Show results: Risk Score 70, CRITICAL. Highlight Deepfake 85, Template 80%, Metadata, Plausibility. Show 95% confidence and recommendation DENY / Refer to SIU. Optionally show AI Agents (NIM, Nemotron, NeMo, etc.) in sidebar.

**Narration:**  
"Now we switch to Photo ID Check. IDVerify AI authenticates every identity using the same NVIDIA stack. We use the sample ID image. The pipeline runs upload, scan, and verify. The results are critical: IDVerify assigns a risk score of 70 and flags the ID as critical. The deepfake score is 85 — high probability of digital manipulation. Template matching is 80%, suggesting inconsistencies with standard ID templates. Metadata and plausibility scores are also elevated, raising strong concerns about authenticity. All of this from a single image. The system reports 95% confidence in its findings. The recommendation is to deny the ID and refer to SIU — Special Investigations Unit — for further review. This verification is driven by NVIDIA NIM, Nemotron, NeMo, Guardrails, Milvus, and Curator — the same stack that powers Auto and MedClaim."

---

## SCENE 7 — Technical differentiators

**Visual:** Split screen: terminal running `pytest tests/ -v` (all green) and/or architecture slide. Highlight 7 agents, RAG, graph, explainable AI, tests.

**Narration:**  
"What makes FraudLens AI unique: 7 specialized agents working in parallel — not a single prompt. RAG-powered pattern matching with NeMo Retriever embeddings and reranking. Graph-based fraud ring detection — because fraud is organized, not isolated. Explainable AI — every score comes with evidence and reasoning. And 61 passing tests covering all agents end to end."

---

## SCENE 8 — Impact and close

**Visual:** Show impact metrics (60 days to 5 min, $500 to $0.10, $45M savings). Then end card with repo URL and tech stack badges (e.g. demo_assets/end_card.html).

**Narration:**  
"For a mid-size insurer processing 50,000 claims per year: investigation time drops from 60 days to 5 minutes. Cost per investigation from 500 dollars to 10 cents. Projected savings: 45 million dollars annually. FraudLens AI — detecting fraud in minutes, not months. Built 100% on NVIDIA."

---

## Recording checklist

| Scene | Focus           | Key numbers / cues                          |
|-------|-----------------|---------------------------------------------|
| 1     | Hook            | Title card                                  |
| 2     | Architecture    | Nemotron, NIM, NeMo, cuGraph, Milvus        |
| 3     | Auto upload     | Sample claim, Analyze                       |
| 4     | Auto results    | 61.4, 67% confidence, 23 images, 7 inconsistencies, 5 patterns, INVESTIGATE |
| 5     | MedClaim        | 37, medium, 95% confidence, REVIEW, 7 inconsistencies, 0 patterns |
| 6     | Photo ID        | 70 critical, Deepfake 85, DENY / SIU, 95%   |
| 7     | Differentiators | 7 agents, RAG, graph, 61 tests               |
| 8     | Impact          | 60 days to 5 min, $45M, end card           |
