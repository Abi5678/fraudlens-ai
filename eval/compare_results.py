"""Compare two evaluation result JSON files. Usage: python -m eval.compare_results baseline.json current.json"""
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load(p):
    p = PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)
    with open(p) as f: return json.load(f)

def get_at_threshold(data, threshold):
    for t in data.get("operational_thresholds", []):
        if t.get("threshold") == threshold: return t
    return data.get("at_threshold")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline"); ap.add_argument("current"); ap.add_argument("--threshold", type=float, default=50)
    args = ap.parse_args()
    a, b = load(args.baseline), load(args.current)
    print("Comparison (baseline vs current)")
    print("  n_evaluated:", a.get("n_evaluated"), "vs", b.get("n_evaluated"))
    print("  AUC:", a.get("auc"), "vs", b.get("auc"))
    th = args.threshold
    at_a, at_b = get_at_threshold(a, th), get_at_threshold(b, th)
    if at_a and at_b:
        print("At threshold", th, ":")
        for k in ["precision", "recall", "f1", "fpr"]:
            va, vb = at_a.get(k), at_b.get(k)
            diff = (vb - va) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
            print("  ", k, ":", va, "vs", vb, "({:+.4f})".format(diff) if diff is not None and diff != 0 else "")
    return 0
if __name__ == "__main__": sys.exit(main())
