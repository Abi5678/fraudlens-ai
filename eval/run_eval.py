"""
Run the fraud detection pipeline on labeled_claims.jsonl and report metrics:
precision, recall, F1, AUC, and FPR at operational thresholds.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (
    operational_report,
    metrics_at_threshold,
    compute_auc,
    optimize_threshold,
)
from eval.calibration import calibration_metrics, save_calibrator


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_score_from_result(result: Any, vertical: str) -> float:
    """Extract 0–100 score from orchestrator result."""
    if vertical == "auto":
        return getattr(result, "fraud_score", 0) or 0
    return getattr(result, "risk_score", 0) or 0


async def run_one(
    row: Dict[str, Any],
    project_root: Path,
    weights: Optional[Dict[str, float]] = None,
    include_network: bool = False,
    include_deepfake: bool = False,
) -> Optional[float]:
    """Run pipeline for one row; return score or None if skip/fail."""
    input_path = row.get("input")
    vertical = (row.get("vertical") or "auto").lower()
    if not input_path:
        return None
    path = Path(input_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        return None
    # Skip CSV / unsupported for full pipeline (no document extraction for CSV in this script)
    suf = path.suffix.lower()
    if suf == ".csv":
        return None
    try:
        if vertical == "medical":
            from medical_lens import MedicalClaimLensAI
            lens = MedicalClaimLensAI()
            result = await lens.analyze(str(path), score_weights=weights)
        else:
            from fraudlens import FraudLensAI
            lens = FraudLensAI()
            result = await lens.analyze(
                str(path),
                include_network=include_network,
                include_deepfake=include_deepfake,
                score_weights=weights,
            )
        return get_score_from_result(result, vertical)
    except Exception as e:
        return None


async def run_all(
    rows: List[Dict[str, Any]],
    project_root: Path,
    weights: Optional[Dict[str, float]] = None,
    include_network: bool = False,
    include_deepfake: bool = False,
) -> List[Dict[str, Any]]:
    """Run pipeline for all rows; return list of {row, score or None}."""
    results = []
    for i, row in enumerate(rows):
        score = await run_one(row, project_root, weights, include_network, include_deepfake)
        results.append({"row": row, "score": score})
    return results


def main():
    parser = argparse.ArgumentParser(description="Run pipeline and report P/R/F1, AUC, FPR at thresholds")
    parser.add_argument("--dataset", default="eval/data/labeled_claims.jsonl", help="Labeled JSONL path")
    parser.add_argument("--output", default=None, help="Output JSON path (default: print only)")
    parser.add_argument("--vertical", default=None, choices=["auto", "medical"], help="Filter to one vertical")
    parser.add_argument("--threshold", type=float, default=50, help="Decision threshold (default 50)")
    parser.add_argument("--sweep-thresholds", action="store_true", help="Report at 20,25,...,80 instead of single threshold")
    parser.add_argument("--weights", default=None, help="Path to JSON file with scoring weights")
    parser.add_argument("--no-network", action="store_true", help="Disable network agent (faster)")
    parser.add_argument("--no-deepfake", action="store_true", help="Disable deepfake agent (faster)")
    parser.add_argument("--calibrate", action="store_true", help="Fit and report calibration (Platt/isotonic)")
    parser.add_argument("--calibration-method", default="platt", choices=["platt", "isotonic"])
    parser.add_argument("--calibrator-out", default=None, help="Save fitted calibrator to this path")
    parser.add_argument("--optimize-threshold", action="store_true", help="Report value-optimal threshold (savings - cost)")
    parser.add_argument("--savings-per-tp", type=float, default=1000.0, help="Savings per true positive (for --optimize-threshold)")
    parser.add_argument("--cost-per-review", type=float, default=50.0, help="Cost per review (for --optimize-threshold)")
    parser.add_argument("--max-fpr", type=float, default=None, help="Max FPR constraint for threshold optimization")
    parser.add_argument("--max-workload", type=int, default=None, help="Max number of reviews for threshold optimization")
    parser.add_argument("--limit", type=int, default=None, help="Max number of rows to evaluate (for quick subset runs)")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_dataset(dataset_path)
    if args.vertical:
        rows = [r for r in rows if (r.get("vertical") or "auto").lower() == args.vertical]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print("No rows to evaluate.", file=sys.stderr)
        sys.exit(1)

    weights = None
    if args.weights:
        wp = Path(args.weights)
        if not wp.is_absolute():
            wp = project_root / wp
        if wp.exists():
            with open(wp) as f:
                weights = json.load(f)

    results = asyncio.run(run_all(
        rows,
        project_root,
        weights=weights,
        include_network=not args.no_network,
        include_deepfake=not args.no_deepfake,
    ))

    # Collect (label_fraud, score) for rows that got a score
    y_true = []
    y_score = []
    for r in results:
        row, score = r["row"], r["score"]
        if score is None:
            continue
        y_true.append(int(row.get("label_fraud", 0)))
        y_score.append(float(score))

    if not y_true:
        print("No valid scores (all rows skipped or failed).", file=sys.stderr)
        sys.exit(1)

    # Build report
    thresholds = list(range(20, 85, 5)) if args.sweep_thresholds else [args.threshold]
    report = operational_report(y_true, y_score, thresholds=thresholds)
    report["dataset"] = str(dataset_path)
    report["vertical_filter"] = args.vertical
    report["limit"] = args.limit
    report["n_evaluated"] = len(y_true)
    report["n_skipped"] = len(rows) - len(y_true)
    report["weights_path"] = args.weights
    report["single_threshold"] = args.threshold
    if not args.sweep_thresholds:
        report["at_threshold"] = metrics_at_threshold(y_true, y_score, args.threshold)

    if args.calibrate:
        cal = calibration_metrics(y_true, y_score, method=args.calibration_method)
        report["calibration"] = {k: v for k, v in cal.items() if k != "calibrator"}
        if "calibrator" in cal and args.calibrator_out:
            out_path = Path(args.calibrator_out)
            if not out_path.is_absolute():
                out_path = project_root / out_path
            save_calibrator(cal["calibrator"], out_path)
            report["calibration"]["calibrator_saved"] = str(out_path)

    if args.optimize_threshold:
        opt = optimize_threshold(
            y_true,
            y_score,
            savings_per_tp=args.savings_per_tp,
            cost_per_review=args.cost_per_review,
            max_fpr=args.max_fpr,
            max_workload=args.max_workload,
        )
        report["threshold_optimization"] = opt

    # Robustness: per-domain report if domain/region present in data
    domains = {}
    for r in results:
        row, score = r["row"], r["score"]
        if score is None:
            continue
        d = row.get("domain") or row.get("region") or "default"
        domains.setdefault(d, {"y_true": [], "y_score": []})
        domains[d]["y_true"].append(int(row.get("label_fraud", 0)))
        domains[d]["y_score"].append(float(score))
    if len(domains) > 1 or (len(domains) == 1 and "default" not in domains):
        report["by_domain"] = {}
        for d, data in domains.items():
            report["by_domain"][d] = operational_report(
                data["y_true"], data["y_score"],
                thresholds=list(range(20, 85, 5)),
            )

    out = json.dumps(report, indent=2)
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(out)
        print(f"Wrote {out_path}")
    print(out)


if __name__ == "__main__":
    main()
