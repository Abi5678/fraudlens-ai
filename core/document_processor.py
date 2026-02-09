"""
Document Processor
Primary:  NVIDIA Nemotron-Parse VLM (page images -> structured markdown)
Fallback: pypdf local text extraction
Image extraction via PyMuPDF (fitz) with photo-vs-document heuristic
"""

import os
import io
import json
import base64
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
import httpx
from loguru import logger

from core.nim_client import get_nim_client, NIMClient


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    IMAGE = "image"
    SCANNED = "scanned"


@dataclass
class ExtractedElement:
    """Represents an extracted element from a document"""
    type: str  # text, table, chart, image, header, footer
    content: str
    confidence: float = 1.0
    bbox: Optional[Dict[str, float]] = None
    page_number: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """Complete extracted document"""
    raw_text: str
    markdown: str
    elements: List[ExtractedElement]
    metadata: Dict[str, Any]
    tables: List[Dict[str, Any]]
    page_count: int = 1
    extracted_images: List[str] = field(default_factory=list)

    def get_text_elements(self) -> List[ExtractedElement]:
        return [e for e in self.elements if e.type == "text"]

    def get_tables(self) -> List[ExtractedElement]:
        return [e for e in self.elements if e.type == "table"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "markdown": self.markdown,
            "elements": [
                {
                    "type": e.type,
                    "content": e.content,
                    "confidence": e.confidence,
                    "bbox": e.bbox,
                    "page_number": e.page_number,
                    "metadata": e.metadata,
                }
                for e in self.elements
            ],
            "metadata": self.metadata,
            "tables": self.tables,
            "page_count": self.page_count,
            "extracted_images": self.extracted_images,
        }


class DocumentProcessor:
    """
    Process documents using NVIDIA Nemotron-Parse VLM.
    Falls back to local pypdf when the API is unavailable.
    Uses PyMuPDF (fitz) for image extraction from PDFs.
    """

    def __init__(
        self,
        nim_client: Optional[NIMClient] = None,
        use_nv_ingest: bool = True,
    ):
        self.nim_client = nim_client or get_nim_client()
        self.use_nv_ingest = use_nv_ingest
        self.nv_ingest_url = os.environ.get(
            "NV_INGEST_URL",
            "https://integrate.api.nvidia.com/v1"
        )
        logger.info("DocumentProcessor initialized with Nemotron-Parse")

    async def process(
        self,
        file_path: Union[str, Path],
        extract_tables: bool = True,
        extract_charts: bool = True,
        extract_images: bool = True,
    ) -> ExtractedDocument:
        """
        Process a document and extract structured content + images.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            doc_type = DocumentType.PDF
        elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            doc_type = DocumentType.IMAGE
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        logger.info(f"Processing {doc_type.value} document: {file_path.name}")

        # Extract images from PDF
        extracted_image_paths: List[str] = []
        if doc_type == DocumentType.PDF and extract_images:
            extracted_image_paths = self._extract_pdf_images(file_path)
            logger.info(f"Extracted {len(extracted_image_paths)} photo images from PDF")

        # Parse document text — Nemotron-Parse primary, pypdf fallback
        if doc_type == DocumentType.PDF:
            doc = await self._process_with_nemotron_parse(file_path)
        else:
            doc = await self._process_image_document(file_path)

        doc.extracted_images = extracted_image_paths
        return doc

    # ------------------------------------------------------------------
    # Photo-vs-document heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def _is_likely_photo(img) -> bool:
        """
        Heuristic using PIL ImageStat.  Real photographs have high colour
        variance; document pages are mostly white with sparse text.
        """
        try:
            from PIL import ImageStat
            rgb = img.convert("RGB") if img.mode != "RGB" else img
            stat = ImageStat.Stat(rgb)
            avg_stddev = sum(stat.stddev) / 3.0
            avg_mean = sum(stat.mean) / 3.0

            # Documents: low stddev + high mean (white background)
            if avg_stddev < 35 and avg_mean > 230:
                return False
            if avg_stddev < 20:
                return False
            return True
        except Exception:
            return True

    # ------------------------------------------------------------------
    # PDF image extraction using PyMuPDF (fitz)
    # ------------------------------------------------------------------

    def _extract_pdf_images(self, file_path: Path) -> List[str]:
        """
        Extract photo/evidence images from a PDF using PyMuPDF.
        Uses two strategies:
          1. Extract embedded XObject images (JPEG/PNG photos)
          2. Render photo-heavy pages as images (for pages where photos
             are part of the content stream rather than separate objects)
        Applies the photo heuristic to skip document pages / logos.
        """
        extracted_paths: List[str] = []
        tmp_dir = tempfile.mkdtemp(prefix="fraudlens_imgs_")

        MIN_DIM = 200
        MIN_ASPECT = 0.25
        MAX_ASPECT = 4.5

        try:
            import fitz
            from PIL import Image, ImageStat

            doc = fitz.open(str(file_path))
            img_count = 0
            seen_xrefs: set = set()

            # --- Strategy 1: embedded XObject images ---
            for page_num in range(doc.page_count):
                page = doc[page_num]
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)

                    try:
                        base_img = doc.extract_image(xref)
                        w = base_img["width"]
                        h = base_img["height"]
                        ext = base_img["ext"]
                        data = base_img["image"]

                        if w < MIN_DIM or h < MIN_DIM:
                            continue

                        aspect = w / max(h, 1)
                        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                            continue

                        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
                        if not self._is_likely_photo(pil_img):
                            logger.debug(
                                f"Skipping doc-like embedded img p{page_num+1} "
                                f"xref={xref} ({w}x{h})"
                            )
                            continue

                        out_ext = "jpg" if ext in ("jpeg", "jpg") else "png"
                        img_path = os.path.join(
                            tmp_dir,
                            f"photo_p{page_num+1}_{img_count}.{out_ext}",
                        )
                        with open(img_path, "wb") as f:
                            f.write(data)
                        extracted_paths.append(img_path)
                        img_count += 1
                        logger.debug(
                            f"Extracted embedded photo p{page_num+1} "
                            f"xref={xref} ({w}x{h} {ext})"
                        )

                    except Exception as e:
                        logger.debug(f"Skip xref {xref}: {e}")

            # --- Strategy 2: render photo-heavy pages ---
            # If embedded extraction found few photos, render pages and
            # keep those whose pixel stats indicate real photo content.
            # Track which pages already have an embedded photo to avoid
            # duplicate renders.
            pages_with_embedded = set()
            for p in extracted_paths:
                fname = os.path.basename(p)
                # Extract page num from "photo_pN_..." pattern
                if fname.startswith("photo_p"):
                    try:
                        pn = int(fname.split("_")[1][1:])
                        pages_with_embedded.add(pn)
                    except (IndexError, ValueError):
                        pass

            if img_count < 3:
                logger.debug(
                    f"Only {img_count} embedded photos found, "
                    "trying page-render strategy"
                )
                for page_num in range(doc.page_count):
                    # Skip pages that already have embedded photo
                    if (page_num + 1) in pages_with_embedded:
                        continue

                    page = doc[page_num]
                    mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI
                    pix = page.get_pixmap(matrix=mat)
                    png_bytes = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

                    stat = ImageStat.Stat(pil_img)
                    avg_stddev = sum(stat.stddev) / 3.0
                    avg_mean = sum(stat.mean) / 3.0

                    # Keep pages that look photographic (high variance)
                    # Use a moderate threshold to catch pages with mixed
                    # photo+text content (like vehicle incident reports).
                    if avg_stddev > 40 and avg_mean < 220:
                        img_path = os.path.join(
                            tmp_dir,
                            f"page_render_p{page_num+1}_{img_count}.png",
                        )
                        pil_img.save(img_path, "PNG")
                        extracted_paths.append(img_path)
                        img_count += 1
                        logger.debug(
                            f"Rendered photo-page p{page_num+1} "
                            f"(stddev={avg_stddev:.1f}, mean={avg_mean:.1f})"
                        )

            doc.close()
            logger.info(
                f"Extracted {img_count} photo images from PDF "
                f"(skipped logos/scans/docs)"
            )

        except ImportError:
            logger.warning(
                "PyMuPDF (fitz) not installed — falling back to pypdf extraction"
            )
            return self._extract_pdf_images_pypdf(file_path, tmp_dir)
        except Exception as e:
            logger.warning(f"PDF image extraction failed: {e}")

        return extracted_paths

    # pypdf fallback for image extraction (kept for environments without fitz)
    def _extract_pdf_images_pypdf(
        self, file_path: Path, tmp_dir: str
    ) -> List[str]:
        """Fallback image extraction using pypdf."""
        extracted_paths: List[str] = []
        try:
            from pypdf import PdfReader
            from PIL import Image

            reader = PdfReader(str(file_path))
            img_count = 0

            for page_num, page in enumerate(reader.pages):
                resources = page.get("/Resources")
                if not resources or "/XObject" not in resources:
                    continue
                x_objects = resources["/XObject"].get_object()

                for obj_name in x_objects:
                    obj = x_objects[obj_name].get_object()
                    if obj.get("/Subtype") != "/Image":
                        continue
                    try:
                        w = int(obj.get("/Width", 0))
                        h = int(obj.get("/Height", 0))
                        if w < 200 or h < 200:
                            continue

                        cs = str(obj.get("/ColorSpace") or "")
                        if "Indexed" in cs:
                            continue

                        filters = str(obj.get("/Filter") or "")
                        data = obj.get_data()
                        if not data:
                            continue

                        if "DCTDecode" in filters:
                            img_path = os.path.join(
                                tmp_dir, f"photo_p{page_num+1}_{img_count}.jpg"
                            )
                            with open(img_path, "wb") as f:
                                f.write(data)

                            pil_img = Image.open(img_path).convert("RGB")
                            if self._is_likely_photo(pil_img):
                                extracted_paths.append(img_path)
                                img_count += 1
                            else:
                                os.remove(img_path)

                    except Exception:
                        continue

            logger.info(f"pypdf extracted {img_count} images")
        except Exception as e:
            logger.warning(f"pypdf image extraction failed: {e}")

        return extracted_paths

    # ------------------------------------------------------------------
    # Nemotron-Parse VLM document processing
    # ------------------------------------------------------------------

    async def _process_with_nemotron_parse(
        self, file_path: Path
    ) -> ExtractedDocument:
        """
        Primary parser: render each PDF page, send to Nemotron-Parse VLM,
        collect structured markdown.  Falls back to pypdf on error.
        """
        try:
            import fitz

            doc = fitz.open(str(file_path))
            page_count = doc.page_count
            all_text_parts: List[str] = []
            all_elements: List[ExtractedElement] = []
            all_tables: List[Dict[str, Any]] = []

            api_key = self.nim_client.config.api_key
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            for page_num in range(page_count):
                page = doc[page_num]

                # Render page at ~1400 wide for Nemotron-Parse input range
                scale = 1400 / max(page.rect.width, 1)
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat)
                png_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(png_bytes).decode()

                logger.debug(
                    f"Sending page {page_num+1}/{page_count} "
                    f"({pix.width}x{pix.height}) to Nemotron-Parse"
                )

                payload = {
                    "model": "nvidia/nemotron-parse",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    },
                                }
                            ],
                        }
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.0,
                }

                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.post(
                            "https://integrate.api.nvidia.com/v1/chat/completions",
                            headers=headers,
                            json=payload,
                        )

                    if resp.status_code != 200:
                        logger.warning(
                            f"Nemotron-Parse page {page_num+1} "
                            f"HTTP {resp.status_code}: {resp.text[:200]}"
                        )
                        continue

                    result = resp.json()
                    page_md = self._parse_nemotron_response(
                        result, page_num + 1
                    )
                    if page_md:
                        all_text_parts.append(page_md["text"])
                        all_elements.extend(page_md["elements"])
                        all_tables.extend(page_md["tables"])

                except Exception as e:
                    logger.warning(
                        f"Nemotron-Parse page {page_num+1} error: {e}"
                    )

            doc.close()

            raw_text = "\n\n".join(all_text_parts)

            if not raw_text.strip():
                logger.warning(
                    "Nemotron-Parse returned no text, falling back to pypdf"
                )
                return await self._process_with_pypdf(file_path)

            logger.info(
                f"Nemotron-Parse extracted {len(raw_text)} chars "
                f"from {page_count} pages"
            )

            return ExtractedDocument(
                raw_text=raw_text,
                markdown=raw_text,
                elements=all_elements,
                metadata={
                    "source": str(file_path),
                    "processor": "nemotron-parse",
                    "page_count": page_count,
                },
                tables=all_tables,
                page_count=page_count,
            )

        except ImportError:
            logger.warning("fitz not available, falling back to pypdf")
            return await self._process_with_pypdf(file_path)
        except Exception as e:
            logger.error(f"Nemotron-Parse processing error: {e}")
            logger.info("Falling back to local pypdf text extraction")
            return await self._process_with_pypdf(file_path)

    def _parse_nemotron_response(
        self, response: Dict[str, Any], page_num: int
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Nemotron-Parse API response.
        The model returns tool_calls with markdown_bbox function
        containing a JSON array of {bbox, text, type} objects.
        """
        try:
            choice = response["choices"][0]
            message = choice["message"]

            # Direct content (some model versions)
            content = message.get("content")
            if content:
                elements = self._parse_extracted_text(content, page_num)
                return {
                    "text": content,
                    "elements": elements,
                    "tables": self._extract_tables_from_text(content),
                }

            # Tool-call format (nemotron-parse typical response)
            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                return None

            for tc in tool_calls:
                func = tc.get("function", {})
                if func.get("name") == "markdown_bbox":
                    args_str = func.get("arguments", "[]")
                    try:
                        items = json.loads(args_str)
                    except json.JSONDecodeError:
                        # Sometimes wrapped in extra list
                        try:
                            items = json.loads(f"[{args_str}]")
                        except Exception:
                            items = []

                    # Flatten if nested list
                    if items and isinstance(items[0], list):
                        items = items[0]

                    text_parts = []
                    elements = []
                    tables = []

                    for item in items:
                        txt = item.get("text", "")
                        item_type = item.get("type", "Text")
                        bbox = item.get("bbox")

                        if not txt.strip():
                            continue

                        text_parts.append(txt)

                        el_type = "text"
                        if "header" in item_type.lower() or "title" in item_type.lower():
                            el_type = "header"
                        elif "table" in item_type.lower():
                            el_type = "table"
                            tables.append(
                                {"markdown": txt, "rows": [], "page": page_num}
                            )
                        elif "caption" in item_type.lower():
                            el_type = "text"
                        elif "image" in item_type.lower():
                            el_type = "image"

                        elements.append(
                            ExtractedElement(
                                type=el_type,
                                content=txt,
                                confidence=1.0,
                                bbox=bbox,
                                page_number=page_num,
                                metadata={"nemotron_type": item_type},
                            )
                        )

                    page_text = "\n".join(text_parts)
                    return {
                        "text": page_text,
                        "elements": elements,
                        "tables": tables,
                    }

            return None

        except Exception as e:
            logger.debug(f"Parse nemotron response error: {e}")
            return None

    # ------------------------------------------------------------------
    # Image document processing (single image, not PDF)
    # ------------------------------------------------------------------

    async def _process_image_document(
        self, file_path: Path
    ) -> ExtractedDocument:
        """Process a single image document using Nemotron-Parse."""
        try:
            with open(file_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            suffix = file_path.suffix.lower()
            mime = (
                "image/png"
                if suffix == ".png"
                else "image/jpeg"
                if suffix in (".jpg", ".jpeg")
                else "image/png"
            )

            api_key = self.nim_client.config.api_key
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "nvidia/nemotron-parse",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{img_b64}"
                                },
                            }
                        ],
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.0,
            }

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )

            if resp.status_code == 200:
                result = resp.json()
                parsed = self._parse_nemotron_response(result, 1)
                if parsed:
                    return ExtractedDocument(
                        raw_text=parsed["text"],
                        markdown=parsed["text"],
                        elements=parsed["elements"],
                        metadata={
                            "source": str(file_path),
                            "processor": "nemotron-parse",
                        },
                        tables=parsed["tables"],
                        page_count=1,
                    )

            # Fallback to LLM description
            return await self._process_image_with_llm(file_path)

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return await self._process_image_with_llm(file_path)

    async def _process_image_with_llm(
        self, file_path: Path
    ) -> ExtractedDocument:
        """Fallback: describe image using LLM."""
        try:
            with open(file_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            suffix = file_path.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"

            response = await self.nim_client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{img_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Extract all text, tables, and structured "
                                    "content from this document image. "
                                    "Format as markdown."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=8192,
            )

            elements = self._parse_extracted_text(response)
            return ExtractedDocument(
                raw_text=response,
                markdown=response,
                elements=elements,
                metadata={
                    "source": str(file_path),
                    "processor": "llm-vision-fallback",
                },
                tables=self._extract_tables_from_text(response),
                page_count=1,
            )

        except Exception as e:
            logger.error(f"Image LLM fallback failed: {e}")
            return ExtractedDocument(
                raw_text="",
                markdown="",
                elements=[],
                metadata={"source": str(file_path), "processor": "failed"},
                tables=[],
                page_count=0,
            )

    # ------------------------------------------------------------------
    # pypdf local fallback
    # ------------------------------------------------------------------

    async def _process_with_pypdf(self, file_path: Path) -> ExtractedDocument:
        """Local fallback: extract text from PDF using pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            page_texts = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    page_texts.append(text)

            raw_text = "\n\n".join(page_texts)
            if not raw_text.strip():
                raise ValueError("pypdf extracted no text")

            logger.info(
                f"pypdf extracted {len(raw_text)} chars "
                f"from {len(reader.pages)} pages"
            )

            elements = self._parse_extracted_text(raw_text)
            return ExtractedDocument(
                raw_text=raw_text,
                markdown=raw_text,
                elements=elements,
                metadata={
                    "source": str(file_path),
                    "processor": "pypdf-local",
                    "page_count": len(reader.pages),
                },
                tables=self._extract_tables_from_text(raw_text),
                page_count=len(reader.pages),
            )

        except Exception as e:
            logger.error(f"pypdf fallback also failed: {e}")
            return ExtractedDocument(
                raw_text="",
                markdown="",
                elements=[],
                metadata={"source": str(file_path), "processor": "failed"},
                tables=[],
                page_count=0,
            )

    # ------------------------------------------------------------------
    # Text parsing helpers
    # ------------------------------------------------------------------

    def _parse_extracted_text(
        self, text: str, page_num: int = 1
    ) -> List[ExtractedElement]:
        """Parse extracted text into structured elements."""
        elements = []
        lines = text.split("\n")
        current_section: List[str] = []
        current_type = "text"

        for line in lines:
            if line.startswith("#"):
                if current_section:
                    elements.append(
                        ExtractedElement(
                            type=current_type,
                            content="\n".join(current_section),
                            page_number=page_num,
                        )
                    )
                current_section = [line]
                current_type = "header"
            elif line.startswith("|") and "|" in line[1:]:
                if current_type != "table":
                    if current_section:
                        elements.append(
                            ExtractedElement(
                                type=current_type,
                                content="\n".join(current_section),
                                page_number=page_num,
                            )
                        )
                    current_section = []
                current_type = "table"
                current_section.append(line)
            else:
                if current_type in ("header", "table"):
                    if current_section:
                        elements.append(
                            ExtractedElement(
                                type=current_type,
                                content="\n".join(current_section),
                                page_number=page_num,
                            )
                        )
                    current_section = []
                current_type = "text"
                current_section.append(line)

        if current_section:
            elements.append(
                ExtractedElement(
                    type=current_type,
                    content="\n".join(current_section),
                    page_number=page_num,
                )
            )

        return elements

    def _extract_tables_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from markdown-formatted text."""
        tables = []
        lines = text.split("\n")
        current_table: List[str] = []
        in_table = False

        for line in lines:
            if line.startswith("|") and "|" in line[1:]:
                in_table = True
                current_table.append(line)
            else:
                if in_table and current_table:
                    tables.append(
                        {
                            "markdown": "\n".join(current_table),
                            "rows": self._parse_markdown_table(current_table),
                        }
                    )
                    current_table = []
                in_table = False

        if current_table:
            tables.append(
                {
                    "markdown": "\n".join(current_table),
                    "rows": self._parse_markdown_table(current_table),
                }
            )

        return tables

    def _parse_markdown_table(
        self, table_lines: List[str]
    ) -> List[List[str]]:
        """Parse markdown table into rows."""
        rows = []
        for line in table_lines:
            if line.replace("|", "").replace("-", "").replace(" ", "") == "":
                continue
            cells = [cell.strip() for cell in line.split("|")]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)
        return rows

    def _convert_to_markdown(
        self, elements: List[ExtractedElement]
    ) -> str:
        """Convert elements to markdown format."""
        sections = []
        for element in elements:
            if element.type in ("header", "table", "text"):
                sections.append(element.content)
            elif element.type == "chart":
                sections.append(
                    f"[Chart: {element.metadata.get('description', 'Chart')}]"
                )
        return "\n\n".join(sections)


# Factory function
async def process_document(
    file_path: Union[str, Path], **kwargs
) -> ExtractedDocument:
    """Convenience function to process a document."""
    processor = DocumentProcessor()
    return await processor.process(file_path, **kwargs)
