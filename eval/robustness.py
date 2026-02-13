"""
Robustness: out-of-domain and adversarial evaluation.

(1) Out-of-domain: Add "domain" or "region" to labeled_claims.jsonl rows;
    run_eval reports metrics per domain (see run_eval --output and "by_domain" in results).

(2) Adversarial: Rephrase claim text (e.g. via LLM), re-run pipeline, measure score
    change and conclusion flip rate at a threshold.
"""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def rephrase_text(raw_text: str, nim_client: Any) -> str:
    """Use LLM to rephrase claim text (preserve meaning, change wording)."""
    prompt = """Rephrase the following insurance claim text. Keep the same facts, amounts, dates, and meaning. Only change wording and sentence structure. Output only the rephrased text, no preamble."""
    try:
        response = await nim_client.chat(
            messages=[{"role": "user", "content": f"{prompt}\n\n{raw_text[:4000]}"}],
            temperature=0.5,
            max_tokens=2000,
        )
        return response.strip() if response else raw_text
    except Exception:
        return raw_text


async def run_adversarial_rephrase(
    dataset_path: Path,
    project_root: Path,
    sample_size: int = 10,
    threshold: float = 50.0,
    vertical: str = "auto",
) -> Dict[str, Any]:
    """
    For a sample of JSON rows: rephrase raw_text, re-run pipeline, report score delta
    and conclusion flip rate (alert vs no-alert) at threshold.
    """
    rows = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Keep only JSON inputs for this test
    json_rows = [r for r in rows if (r.get("input") or "").lower().endswith(".json")][:sample_size]
    if not json_rows:
        return {"error": "No JSON rows in dataset", "n": 0}

    from core.nim_client import get_nim_client
    nim = get_nim_client()
    if vertical == "medical":
        from medical_lens import MedicalClaimLensAI
        lens = MedicalClaimLensAI()
        def get_score(res):
            return getattr(res, "risk_score", 0) or 0
    else:
        from fraudlens import FraudLensAI
        lens = FraudLensAI()
        def get_score(res):
            return getattr(res, "fraud_score", 0) or 0

    deltas = []
    flips = 0
    for row in json_rows:
        inp = row.get("input")
        path = project_root / inp if not Path(inp).is_absolute() else Path(inp)
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        claim_data = data.get("claim_data", data)
        raw_text = data.get("raw_text", json.dumps(claim_data, indent=2))
        # Original score
        try:
            if vertical == "medical":
                res_orig = await lens.analyze(str(path))
            else:
                res_orig = await lens.analyze_json(str(path))
            score_orig = get_score(res_orig)
        except Exception:
            score_orig = 0.0
        # Rephrase and re-run
        raw_rephrased = await rephrase_text(raw_text, nim)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump({"claim_data": claim_data, "raw_text": raw_rephrased}, tf)
            tf.flush()
            temp_path = tf.name
        try:
            if vertical == "medical":
                res_rephrased = await lens.analyze(temp_path)
            else:
                res_rephrased = await lens.analyze_json(temp_path)
            score_rephrased = get_score(res_rephrased)
        except Exception:
            score_rephrased = score_orig
        finally:
            Path(temp_path).unlink(missing_ok=True)
        delta = score_rephrased - score_orig
        deltas.append(delta)
        alert_orig = score_orig >= threshold
        alert_rephrased = score_rephrased >= threshold
        if alert_orig != alert_rephrased:
            flips += 1

    n = len(deltas)
    if n == 0:
        return {"error": "No successful pairs", "n": 0}
    mean_abs_delta = sum(abs(d) for d in deltas) / n
    mean_delta = sum(deltas) / n
    flip_rate = flips / n
    return {
        "n": n,
        "threshold": threshold,
        "mean_score_delta": round(mean_delta, 3),
        "mean_abs_score_delta": round(mean_abs_delta, 3),
        "conclusion_flip_rate": round(flip_rate, 4),
        "flips": flips,
    }


def main():
    parser = argparse.ArgumentParser(description="Adversarial rephrase robustness test")
    parser.add_argument("--dataset", default="eval/data/labeled_claims.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--vertical", default="auto", choices=["auto", "medical"])
    args = parser.parse_args()
    path = Path(args.dataset)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    result = asyncio.run(run_adversarial_rephrase(path, PROJECT_ROOT, args.sample, args.threshold, args.vertical))
    out = json.dumps(result, indent=2)
    if args.output:
        op = Path(args.output)
        if not op.is_absolute():
            op = PROJECT_ROOT / op
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            f.write(out)
        print(f"Wrote {op}")
    print(out)


if __name__ == "__main__":
    main()
