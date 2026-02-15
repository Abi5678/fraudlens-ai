#!/usr/bin/env python3
"""
Generate a sample claim PDF for the demo: all text/identity from sample_claim.json,
images extracted from a source PDF (e.g. MB (2).PDF). No real identity in the output.
"""
from pathlib import Path
import json
import sys
import argparse
import tempfile
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_PDF = PROJECT_ROOT / "MB (2).PDF"
DEFAULT_CLAIM_JSON = PROJECT_ROOT / "sample_claim.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "demo_assets" / "sample_claim.pdf"

MIN_IMAGE_DIM = 200
PAGE_WIDTH = 612
PAGE_HEIGHT = 792
MARGIN = 36
FONT_SIZE = 11
TITLE_SIZE = 14


def load_claim_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("claim_data", data)


def build_claim_form_text(data: dict) -> str:
    c = data.get("claimant", {})
    p = data.get("policy", {})
    i = data.get("incident", {})
    cl = data.get("claim", {})
    v = data.get("vehicle", {})
    m = data.get("medical", {})
    lines = [
        "INSURANCE CLAIM FORM",
        "",
        f"Claimant: {c.get('name', '')}",
        f"Address: {c.get('address', '')}",
        f"Phone: {c.get('phone', '')}",
        f"Email: {c.get('email', '')}",
        f"DOB: {c.get('dob', '')}",
        "",
        f"Policy: {p.get('number', '')}  Type: {p.get('type', '')}  Coverage: ${p.get('coverage_amount', 0):,.0f}",
        f"Claim #: {cl.get('number', '')}  Date: {cl.get('date', '')}  Amount: ${cl.get('amount', 0):,.2f}  Type: {cl.get('type', '')}",
        "",
        f"Vehicle: {v.get('year', '')} {v.get('make', '')} {v.get('model', '')}  VIN: {v.get('vin', '')}",
        "",
        f"Date of Loss: {i.get('date', '')}  Time: {i.get('time', '')}",
        f"Location: {i.get('location', '')}",
        "",
        "DESCRIPTION OF INCIDENT:",
        (i.get("description") or "").strip(),
        "",
        "MEDICAL:",
        f"Injuries: {', '.join(m.get('injuries', []))}",
        f"Providers: {', '.join(m.get('providers', []))}",
        f"Costs: ${m.get('costs', 0):,.0f}",
        "",
        f"TOTAL CLAIM: ${cl.get('amount', 0):,.2f}",
        "",
        f"Signed: {c.get('name', '')}  Date: {cl.get('date', '')}",
    ]
    return "\n".join(lines)


def extract_images_from_pdf(source_pdf: Path) -> list:
    """Extract image (width, height, bytes, ext) from PDF; keep only those >= MIN_IMAGE_DIM."""
    import fitz
    out = []
    seen = set()
    doc = fitz.open(str(source_pdf))
    try:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    info = doc.extract_image(xref)
                    w, h = info["width"], info["height"]
                    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                        continue
                    out.append((w, h, info["image"], info.get("ext", "png")))
                except Exception:
                    continue
    finally:
        doc.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate sample claim PDF (fake identity + images from source PDF)")
    ap.add_argument("--source-pdf", default=None, help=f"Source PDF for images (default: MB (2).PDF)")
    ap.add_argument("--claim-json", default=None, help="Claim JSON for text (default: sample_claim.json)")
    ap.add_argument("--output", default=None, help="Output PDF path (default: demo_assets/sample_claim.pdf)")
    args = ap.parse_args()

    source_pdf = Path(args.source_pdf) if args.source_pdf else DEFAULT_SOURCE_PDF
    claim_json = Path(args.claim_json) if args.claim_json else DEFAULT_CLAIM_JSON
    output = Path(args.output) if args.output else DEFAULT_OUTPUT

    if not source_pdf.is_absolute():
        source_pdf = PROJECT_ROOT / source_pdf
    if not claim_json.is_absolute():
        claim_json = PROJECT_ROOT / claim_json
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    if not claim_json.exists():
        print(f"Error: Claim JSON not found: {claim_json}", file=sys.stderr)
        sys.exit(1)
    if not source_pdf.exists():
        print(f"Error: Source PDF not found: {source_pdf}", file=sys.stderr)
        sys.exit(1)

    data = load_claim_data(claim_json)
    form_text = build_claim_form_text(data)
    images = extract_images_from_pdf(source_pdf)
    print(f"Loaded claim data from {claim_json.name}, extracted {len(images)} images from {source_pdf.name}")

    import fitz
    doc = fitz.open()
    try:
        # Page 1: claim form text
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        rect = fitz.Rect(MARGIN, MARGIN, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN)
        # Title
        page.insert_text(fitz.Point(MARGIN, MARGIN + 16), "INSURANCE CLAIM FORM", fontsize=TITLE_SIZE, fontname="helv")
        text_rect = fitz.Rect(MARGIN, MARGIN + 28, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN)
        page.insert_textbox(text_rect, form_text, fontsize=FONT_SIZE, fontname="helv", align=fitz.TEXT_ALIGN_LEFT)
        # If text overflows we could add a second page for raw_text; for simplicity one page is enough

        # Following pages: one image per page, scaled to fit (aspect ratio preserved)
        usable_w = PAGE_WIDTH - 2 * MARGIN
        usable_h = PAGE_HEIGHT - 2 * MARGIN
        for w_px, h_px, img_bytes, ext in images:
            page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            if w_px and h_px:
                if usable_w * h_px <= usable_h * w_px:
                    rect_w, rect_h = usable_w, usable_w * h_px / w_px
                else:
                    rect_w, rect_h = usable_h * w_px / h_px, usable_h
            else:
                rect_w, rect_h = usable_w, usable_h
            x0 = MARGIN + (usable_w - rect_w) / 2
            y0 = MARGIN + (usable_h - rect_h) / 2
            img_rect = fitz.Rect(x0, y0, x0 + rect_w, y0 + rect_h)
            page.insert_image(img_rect, stream=img_bytes)

        output.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output), garbage=4, deflate=True)
        print(f"Saved: {output}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
