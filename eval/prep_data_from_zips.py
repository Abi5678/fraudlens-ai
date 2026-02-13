"""
Build labeled_claims.jsonl for evaluation and training from canonical data sources:

  - Auto:  eval/data/Auto Insurance Claim Metadata & Automation Service.zip
  - Medical: eval/data/archive (directory) or eval/data/archive.zip

Run from project root:

  python -m eval.prep_data_from_zips

Extracts zips to eval/data/auto_insurance/ and eval/data/medical_claim/. If eval/data/archive
is a directory, files inside it are included as medical. Writes eval/data/labeled_claims.jsonl
with placeholder labels for run_eval and training (threshold/weight tuning).
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import List

# Project root (parent of eval/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "eval" / "data"
ALLOWED_EXT = {".pdf", ".json", ".png", ".jpg", ".jpeg", ".csv"}


def infer_vertical(zip_path: Path) -> str:
    """Infer vertical from zip filename (auto vs medical)."""
    name = zip_path.stem.lower()
    if "medical" in name or "med" in name or "archive" in name:
        return "medical"
    if "auto" in name or "insurance" in name:
        return "auto"
    return "auto"  # default


def extract_zip(zip_path: Path, out_dir: Path) -> List[Path]:
    """Extract zip into out_dir; return list of extracted file paths (under out_dir) with allowed extensions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir() or "/__MACOSX" in info.filename or info.filename.startswith("__MACOSX"):
                continue
            # Avoid path traversal: extract under out_dir
            safe_name = Path(info.filename).name
            if not safe_name.strip():
                continue
            zf.extract(info, out_dir)
    return discover_files(out_dir)


def discover_files(dir_path: Path) -> List[Path]:
    """Recursively find PDF, JSON, PNG, JPG under dir_path."""
    found = []
    for ext in ALLOWED_EXT:
        found.extend(dir_path.rglob(f"*{ext}"))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description="Build labeled_claims.jsonl from zip files in eval/data")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: eval/data/labeled_claims.jsonl)")
    parser.add_argument("--append", action="store_true", help="Append to existing labeled_claims.jsonl instead of overwriting")
    parser.add_argument("--no-extract", action="store_true", help="Only scan existing eval/data/auto_insurance and eval/data/medical_claim dirs; do not extract zips")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else DATA_DIR / "labeled_claims.jsonl"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.append and output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    if not args.no_extract:
        # Find zips in eval/data
        for zip_path in sorted(DATA_DIR.glob("*.zip")):
            vertical = infer_vertical(zip_path)
            out_subdir = DATA_DIR / ("auto_insurance" if vertical == "auto" else "medical_claim")
            print(f"Extracting {zip_path.name} -> {out_subdir.relative_to(PROJECT_ROOT)}")
            try:
                extracted = extract_zip(zip_path, out_subdir)
                for fp in extracted:
                    rel = fp.relative_to(PROJECT_ROOT)
                    rows.append({
                        "input": str(rel).replace("\\", "/"),
                        "vertical": vertical,
                        "label_fraud": 0,
                        "label_risk": "medium",
                    })
                print(f"  Added {len(extracted)} files for vertical={vertical}")
            except Exception as e:
                print(f"  Error: {e}")

    # Discover from extracted dirs and from canonical archive path (dir or contents)
    # Canonical sources: Auto zip -> auto_insurance/; archive (dir or zip) -> medical_claim/ or archive/
    scan_dirs = [
        (DATA_DIR / "auto_insurance", "auto"),
        (DATA_DIR / "medical_claim", "medical"),
        (DATA_DIR / "archive", "medical"),  # eval/data/archive directory (medical training/eval data)
    ]
    for subdir, vertical in scan_dirs:
        if not subdir.exists():
            continue
        added = 0
        for fp in discover_files(subdir):
            rel = fp.relative_to(PROJECT_ROOT)
            rel_str = str(rel).replace("\\", "/")
            if not any(r.get("input") == rel_str for r in rows):
                rows.append({
                    "input": rel_str,
                    "vertical": vertical,
                    "label_fraud": 0,
                    "label_risk": "medium",
                })
                added += 1
        if added:
            print(f"  Scanned {subdir.relative_to(PROJECT_ROOT)} ({vertical}): {added} files")

    # Deduplicate by input path (keep first)
    seen = set()
    unique = []
    for r in rows:
        key = (r["input"], r.get("vertical"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    with open(output_path, "w") as f:
        for r in unique:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(unique)} rows to {output_path}")
    print("Edit label_fraud (and label_risk) in that file, then run: python -m eval.run_eval --dataset eval/data/labeled_claims.jsonl --output eval/results.json")


if __name__ == "__main__":
    main()
