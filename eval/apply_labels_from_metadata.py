"""
Apply real labels from claim_metadata.xlsx to labeled_claims.jsonl.
Auto claims: claim_id (e.g. CL001) from filename; fraud_reported Y/N -> label_fraud 1/0.
Run from project root: python -m eval.apply_labels_from_metadata
"""
import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "eval" / "data"
AUTO_META = DATA_DIR / "auto_insurance" / "claim_metadata.xlsx"


def load_fraud_map():
    try:
        import pandas as pd
    except ImportError:
        return {}
    if not AUTO_META.exists():
        return {}
    df = pd.read_excel(AUTO_META, sheet_name="BigQueryData")
    if "claim_id" not in df.columns or "fraud_reported" not in df.columns:
        return {}
    return {
        str(row["claim_id"]).strip(): 1 if str(row["fraud_reported"]).strip().upper() == "Y" else 0
        for _, row in df.iterrows()
    }


def claim_id_from_path(input_path: str) -> Optional[str]:
    base = Path(input_path).stem
    if re.match(r"CL\d+", base, re.I):
        return base
    return None


def main():
    fraud_map = load_fraud_map()
    if not fraud_map:
        print("No claim_metadata.xlsx or fraud_reported column found.", file=sys.stderr)
        sys.exit(1)
    labeled_path = DATA_DIR / "labeled_claims.jsonl"
    out_path = DATA_DIR / "labeled_claims.jsonl"
    if not labeled_path.exists():
        print("Not found:", labeled_path, file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(labeled_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = claim_id_from_path(row.get("input", ""))
            if cid is not None and cid in fraud_map:
                row["label_fraud"] = fraud_map[cid]
                row["label_risk"] = "high" if fraud_map[cid] == 1 else "medium"
            rows.append(row)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_auto = sum(1 for r in rows if claim_id_from_path(r.get("input", "")) is not None)
    n_fraud = sum(1 for r in rows if r.get("label_fraud") == 1)
    print("Wrote", len(rows), "rows to", out_path)
    print("  Auto with metadata:", n_auto, ", fraud_reported=Y (label 1):", n_fraud)


if __name__ == "__main__":
    main()
