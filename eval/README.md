# Evaluation Pipeline

Labeled evaluation dataset and scripts to measure **precision, recall, F1, AUC, and FPR** at operational thresholds; **calibration** (Platt scaling or isotonic regression) so scores map to meaningful probabilities; **threshold optimization** (maximize value: savings minus cost of review) under FPR or workload constraints; **robustness** (out-of-domain and adversarial); and **explainability** (per-factor breakdown, narrative, RAG rationales).

## Dataset schema (labeled_claims.jsonl)

One JSON object per line. Fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | string | Yes | Path to claim file (PDF, image, or JSON) relative to project root, or absolute path. |
| `vertical` | string | Yes | `"auto"` or `"medical"` — which orchestrator to run. |
| `label_fraud` | int | Yes | Binary 0/1 — ground truth fraud label for P/R/F1. |
| `label_risk` | string or int | No | Ordinal risk (e.g. 1–5 or "low"/"medium"/"high"/"critical") for rank correlation / calibration. |
| `domain` or `region` | string | No | For **robustness**: line of business or region; metrics are reported per domain and cross-domain when present. |
| `per_agent_labels` | object | No | Per-agent ground truth, e.g. `{"billing_issue": 1, "clinical_issue": 0}`. Used only when present for per-agent metrics. |

Example lines:

```json
{"input": "sample_claim.json", "vertical": "auto", "label_fraud": 1, "label_risk": "high"}
{"input": "path/to/medical_claim.pdf", "vertical": "medical", "label_fraud": 0, "per_agent_labels": {"billing_issue": 0}}
```

## Evaluation and training data sources

Use these two sources for **evaluation** and **training** (threshold/weight tuning):

- **Auto:** `eval/data/Auto Insurance Claim Metadata & Automation Service.zip`
- **Medical:** `eval/data/archive` (directory) or `eval/data/archive.zip`

Place the Auto zip and the medical archive (folder or zip) in `eval/data/`.

Then from the project root:

```bash
# Extract zips and build labeled_claims.jsonl (placeholder labels; edit before running full eval)
python -m eval.prep_data_from_zips
```

This extracts zips into `eval/data/auto_insurance/` and `eval/data/medical_claim/`, scans `eval/data/archive/` if present (medical), and writes `eval/data/labeled_claims.jsonl` with placeholder labels. **Apply real labels** from the Auto dataset metadata (if present):

```bash
python -m eval.apply_labels_from_metadata
```

This reads `eval/data/auto_insurance/claim_metadata.xlsx` and sets `label_fraud` (1 if `fraud_reported` is Y, else 0) and `label_risk` for each auto claim by matching `claim_id` in the filename (e.g. CL001.jpg). Medical and other rows are left unchanged. For those, edit `label_fraud` and `label_risk` manually if you have ground truth. Then run evaluation as below. Options:

- `--append` — merge with existing `labeled_claims.jsonl` instead of overwriting
- `--no-extract` — only scan already-extracted dirs (do not extract zips)
- `--output path` — write JSONL to a different path

## How to run evaluation

From project root:

```bash
# Single run (default threshold 50)
python -m eval.run_eval --dataset eval/data/labeled_claims.jsonl --output eval/results.json

# Filter by vertical
python -m eval.run_eval --dataset eval/data/labeled_claims.jsonl --vertical auto --output eval/results.json

# Custom threshold
python -m eval.run_eval --dataset eval/data/labeled_claims.jsonl --threshold 60 --output eval/results.json

# Threshold sweep (P/R/F1 at 20, 25, ..., 80)
python -m eval.run_eval --dataset eval/data/labeled_claims.jsonl --sweep-thresholds --output eval/sweep.json

# With custom scoring weights
python -m eval.run_eval --dataset eval/data/labeled_claims.jsonl --weights eval/weights.json --output eval/results.json

# (1) Full report: P/R/F1, AUC, FPR at operational thresholds (20–80)
python -m eval.run_eval --dataset eval/data/labeled_claims.jsonl --sweep-thresholds --output eval/results.json

# (2) Calibration — map score to risk probability (Platt or isotonic); save calibrator for production
python -m eval.run_eval --dataset eval/data/labeled_claims.jsonl --calibrate --calibration-method platt --calibrator-out eval/calibrator.json --output eval/results.json

# (3) Threshold optimization — maximize value (savings − cost of review), optional max FPR or workload
python -m eval.run_eval --dataset eval/data/labeled_claims.jsonl --optimize-threshold --savings-per-tp 1000 --cost-per-review 50 --max-fpr 0.05 --output eval/results.json

# (4) Robustness — add "domain" or "region" to JSONL rows; report includes by_domain metrics.
#    Adversarial rephrase (score stability to rewording):
python -m eval.robustness --dataset eval/data/labeled_claims.jsonl --sample 5 --threshold 50 --output eval/robustness.json

# (5) Explainability — per-factor breakdown and narrative in app; RAG rationales in pattern_matches (see below)
```

To choose a decision threshold: run with `--sweep-thresholds`, inspect the output (e.g. `eval/sweep.json` or stdout), then set `--threshold` for production or later runs (e.g. pick threshold that gives target 90% precision).

## Improving accuracy

1. **Run eval** and inspect `results.json`: overall P/R/F1 and, if available, per-agent metrics.
2. **Identify weak agents** (low precision or recall for that dimension).
3. **Edit prompts or patterns** in code, then re-run the same eval to compare.

Relevant files:

- **Auto / Pattern agent:** `agents/pattern_agent.py` — `_analyze_match` prompt; `core/embedding_service.py` — `FRAUD_PATTERNS` list (add or refine entries; re-run app or eval so vector store is re-initialized if needed).
- **Medical:** `agents/billing_integrity_agent.py`, `agents/clinical_consistency_agent.py`, `agents/eligibility_agent.py`, `agents/inconsistency_agent.py` — each agent’s `analyze` or internal LLM prompt. Edit those prompts and re-run eval.
- **Scoring weights:** Use `eval/weights.json` and `--weights` to try different weight sets without changing code. For learned weights, fit a logistic regression (or similar) on `risk_factors` vs `label_fraud` on the labeled set, then use coefficients as a weight proposal and plug into `--weights`.

## Extraction validation

Optional: run extraction-only eval to check document extraction quality (no full fraud analysis):

```bash
python -m eval.run_extraction_eval --dataset eval/data/golden_extraction.jsonl --output eval/extraction_results.json
```

Schema for `golden_extraction.jsonl`: `input` (path to PDF/image), and either `golden_claim_data` (object) or a minimal set of required fields to compare. See `eval/data/golden_extraction.jsonl` for an example.

## Explainability

- **Per-factor breakdown:** Each run returns `risk_factors` (name, score, weight, description, evidence) and an overall narrative.
- **RAG rationales:** For each matched fraud pattern, the Pattern Agent returns a `rationale` field (why this pattern hit), in addition to `matching_elements`. Use these for attention over RAG hits and auditor review.
- **Interpretability:** Scoring is a fixed weighted sum of agent scores (no black-box model); weights are in code or `eval/weights.json`. Any learned components (e.g. calibration) are simple (Platt/isotonic) and interpretable.

## Interpreting results and measuring improvement

**What the evaluation result contains**

- **`results.json`** (or `--output` path): `auc`, `n`, `n_positive`, `n_negative`, and `operational_thresholds` — for each threshold (e.g. 20, 25, …, 80) you get **precision**, **recall**, **F1**, **FPR**, and counts (tp, fp, fn, tn). Use these to see how the system behaves at different score cutoffs.

**Quick baseline**

- A small **labeled demo set** is in `eval/data/labeled_demo.jsonl` (3 rows, same claim with mixed labels). A **baseline** run is saved as `eval/results_baseline.json`. To re-run the same evaluation:
  ```bash
  python3 -m eval.run_eval --dataset eval/data/labeled_demo.jsonl --limit 3 --sweep-thresholds --no-network --no-deepfake --output eval/results_baseline.json
  ```

**Has the system got better?**

1. **Add real labels** — Edit `eval/data/labeled_claims.jsonl` (or your own JSONL) and set `label_fraud` (0/1) for as many claims as you have ground truth for.
2. **Run evaluation** — Use the same dataset and options each time (e.g. same `--dataset`, `--limit`, `--vertical`). Save outputs with distinct names, e.g. `eval/results_baseline.json` and `eval/results_after_change.json`.
3. **Compare** — Compare **precision**, **recall**, **F1**, and **FPR** at your chosen threshold (e.g. 50) between the two JSON files. Higher F1 (or higher recall at acceptable precision) = better. Lower FPR at similar recall = better. You can diff the two JSONs or use: `python3 -m eval.compare_results eval/results_baseline.json eval/results.json --threshold 50`.
4. **Subset runs** — Use `--limit N` to evaluate only the first N rows for faster iteration (e.g. `--limit 20`).

## Reproducibility

Results JSON includes dataset path, threshold, weights path (if any), `limit`, and options. Log NIM model names from env when feasible so runs are reproducible.
