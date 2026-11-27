"""doc_to_markdown.py - convert any business document to LLM-ready Markdown.

    python doc_to_markdown.py <file> [-o out.md] [--chunks]

Input : one file path. PDF, Word, PowerPoint, Excel, CSV, text, image, a legacy
        Office / OpenDocument / RTF / HTML file (converted with LibreOffice first),
        or Azure Document Intelligence output (.md, or a saved result .json whose
        analyzeResult.content is used).
Output: one Markdown file, same structure for every format:
        YAML front matter, <!-- page N --> markers, pipe tables (merged cells
        repeated), [x] / [ ] checkboxes, **Figure:** blocks, optional .chunks.jsonl.

Routing
    .docx .pptx .xlsx .csv .txt/.md   parsed natively, exact, no OCR, no cost
    .doc .ppt .xls .rtf .odt ...       LibreOffice -> modern format -> native parser
    .json (ADI analyzeResult)          analyzeResult.content -> normalizer
    .pdf                               text pages: PyMuPDF text + tables
                                       scanned pages: rendered -> vision model
                                       (or Azure Document Intelligence when configured)
    images                             vision model (or ADI when configured)
    every picture / chart / scan       vision model, two passes:
                                       1. overview  - all images in one call
                                       2. zoom      - full view + magnified crops of
                                          regions the model flagged (one call)

Vision model: OpenAI GPT (default gpt-5.4) through LangChain v1 ChatOpenAI on the
Responses API, image detail "original", structured JSON output (strict schema).

Environment (a .env file next to this script or in the working directory is loaded)
    OPENAI_API_KEY                      required when the file has images or scans
    OPENAI_BASE_URL                     optional: OpenAI-compatible gateway
    AZURE_DI_ENDPOINT, AZURE_DI_KEY     optional: ADI for PDFs and image files

Install
    pip install python-pptx python-docx openpyxl pymupdf pymupdf4llm pillow pydantic \
                "langchain>=1.0" "langchain-openai>=1.0" python-dotenv
    pip install azure-ai-documentintelligence      # optional
    LibreOffice (soffice)                          # optional: legacy formats, EMF/WMF

Sections in this file
    1  Settings
    2  Markdown normalizer (ADI Markdown -> structured Markdown, chunking)
    3  Format detection and LibreOffice conversion
    4  Image registry and image preparation (sizing, upscaling, tiling, crops)
    5  Native converters: PPTX, DOCX, XLSX, CSV, text, ADI JSON
    6  PDF converter (PyMuPDF) and Azure Document Intelligence route
    7  Vision: overview pass + zoom pass (LangChain / OpenAI)
    8  Pipeline and CLI
    9  Tests (pytest doc_to_markdown.py; no network)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import csv
import datetime as dt
import hashlib
import html
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PIL import Image, ImageOps
from pydantic import BaseModel, Field

log = logging.getLogger("doc2md")
Image.MAX_IMAGE_PIXELS = 400_000_000  # big scans are legitimate input


# =========================================================================== #
# 1. Settings
# =========================================================================== #
@dataclass(frozen=True)
class Settings:
    # --- vision model (OpenAI through LangChain) ---
    model: str = "gpt-5.4"
    reasoning_effort: str = "medium"     # "high" helps dense charts; costs more
    verbosity: str = "high"              # keeps transcription literal (OpenAI guidance)
    detail: str = "original"             # OpenAI: use "original" for OCR / fine detail
    max_retries: int = 3
    timeout: float = 300.0
    vision_concurrency: int = 4          # parallel calls when a pass needs >1 batch
    max_images_per_call: int = 16        # images per request
    max_patches_per_call: int = 60_000   # sum of image patches per request
    # --- model image limits (GPT-5.4 family, detail="original") ---
    patch_px: int = 32
    max_patches_per_image: int = 10_000
    max_side_px: int = 6000
    # --- image preparation ---
    overview_patches: int = 6000         # pass-1 budget per image (~6 MP)
    zoom_patches: int = 2500             # budget per zoom crop
    max_upscale: float = 4.0             # never magnify more than this
    min_long_side: int = 1280            # small images are upscaled to this (pass 1)
    min_image_px: int = 48               # ignore icons / bullets smaller than this
    # --- zoom pass ---
    zoom_rounds: int = 1                 # 0 disables the zoom pass
    zoom_confidence: float = 0.9         # below this an image is always re-checked
    max_zooms_per_image: int = 6         # regions the model may flag per image
    max_auto_tiles: int = 12             # automatic tiles per image (full coverage)
    zoom_padding: float = 0.18           # padding around a flagged region (OpenAI cookbook)
    tile_overlap: float = 0.10
    # --- routing ---
    use_adi: str = "auto"                # auto | always | never  (PDF and image files)
    pdf_mode: str = "auto"               # auto | text | vision
    render_dpi: int = 300                # scanned pages / figures rendered at this DPI
    scan_text_chars: int = 80            # a PDF page with fewer text chars is a scan
    scan_image_cover: float = 0.9        # a page this covered by one image is a scan
    vector_figures: bool = True          # send vector charts in PDFs to the vision model
    skip_repeated_images: bool = True    # logos on most pages/slides
    include_hidden: bool = True          # hidden slides / sheets (marked as hidden)
    include_notes: bool = True           # speaker notes
    max_sheet_rows: int = 0              # 0 = no limit
    page_rule: bool = False
    keep_furniture: bool = False


# =========================================================================== #
# 2. Markdown normalizer
#    ADI-style Markdown -> structured Markdown. Every route produces ADI-style
#    Markdown (<!-- PageBreak -->, HTML tables allowed) and ends here.
# =========================================================================== #
CHECKED = re.compile(r"☒|☑|:selected:")
UNCHECKED = re.compile(r"☐|:unselected:")
FURNITURE = re.compile(r"^\s*<!--\s*Page(Header|Footer|Number)\s*=.*?-->\s*$")
PAGEBREAK = re.compile(r"^\s*<!--\s*PageBreak\s*-->\s*$")
PAGEBREAK_TAG = "<!-- PageBreak -->"


def checkboxes(text: str) -> str:
    return UNCHECKED.sub("[ ]", CHECKED.sub("[x]", text))


def has_marks(text: str) -> bool:
    return "[x]" in text or "[ ]" in text


def clean_inline(text: str) -> str:
    """Strip tags, unescape entities, collapse whitespace, convert checkboxes."""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", checkboxes(text)).strip()


class _TableParser(HTMLParser):
    """Collect rows of cells: {text, header, rowspan, colspan}."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.rows, self.caption = [], ""
        self._row = None
        self._cell = None
        self._in_caption = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            def span(k):
                try:
                    return max(1, int(a.get(k) or 1))
                except ValueError:
                    return 1
            self._cell = {"text": [], "header": tag == "th",
                          "rowspan": span("rowspan"), "colspan": span("colspan")}
        elif tag == "caption":
            self._in_caption = True
        elif tag == "br" and self._cell is not None:
            self._cell["text"].append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._cell["text"] = clean_inline("".join(self._cell["text"]))
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "caption":
            self._in_caption = False

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"].append(data)
        elif self._in_caption:
            self.caption += data

    def handle_entityref(self, name):
        self.handle_data(f"&{name};")

    def handle_charref(self, name):
        self.handle_data(f"&#{name};")


def table_to_pipe(table_html: str) -> str:
    """HTML table (rowspan/colspan) -> pipe table; merged cells repeated,
    multi-row headers joined as "Parent / Child"."""
    p = _TableParser()
    p.feed(table_html)
    if not p.rows:
        return ""
    grid: dict[tuple[int, int], str] = {}
    header_cells: set[tuple[int, int]] = set()
    n_cols = 0
    for r, row in enumerate(p.rows):
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    grid[(r + dr, c + dc)] = cell["text"]
                    if cell["header"]:
                        header_cells.add((r + dr, c + dc))
            c += cell["colspan"]
            n_cols = max(n_cols, c)
    if not grid:
        return ""
    n_rows = max(r for r, _ in grid) + 1

    def row_vals(r):
        return [grid.get((r, c), "") for c in range(n_cols)]

    # header rows = leading rows made only of <th>; if none, use the first row
    n_head = 0
    while n_head < n_rows and all((n_head, c) in header_cells or (n_head, c) not in grid
                                  for c in range(n_cols)) \
            and any((n_head, c) in header_cells for c in range(n_cols)):
        n_head += 1
    n_head = n_head or 1

    header = []
    for c in range(n_cols):
        parts = [grid.get((r, c), "") for r in range(n_head)]
        parts = [x for x in dict.fromkeys(parts) if x]          # dedupe repeated span text
        header.append(" / ".join(parts))
    for c in range(1, n_cols):                                 # blank header after a named one
        if not header[c] and header[c - 1]:
            header[c] = header[c - 1]

    def esc(x):
        return (x or " ").replace("|", "\\|")

    out = []
    caption = clean_inline(p.caption)
    if caption:
        out += [f"**Table:** {caption}", ""]
    out.append("| " + " | ".join(esc(h) for h in header) + " |")
    out.append("|" + "|".join(["---"] * n_cols) + "|")
    for r in range(n_head, n_rows):
        out.append("| " + " | ".join(esc(v) for v in row_vals(r)) + " |")
    return "\n".join(out)


def figure_to_md(fig_html: str) -> str:
    cap = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", fig_html, flags=re.S | re.I)
    caption = clean_inline(cap.group(1)) if cap else ""
    body = re.sub(r"<figcaption[^>]*>.*?</figcaption>", "", fig_html, flags=re.S | re.I)
    body = re.sub(r"</?figure[^>]*>", "", body, flags=re.I)
    text = clean_inline(body)
    lines = [f"**Figure:** {caption}" if caption else "**Figure:**"]
    if text:
        lines += ["", text]
    return "\n".join(lines)


def normalize(adi_md: str, source: str | None = None, page_rule: bool = False,
              keep_furniture: bool = False, bullets_for_checkboxes: bool = True,
              extra_front_matter: dict[str, Any] | None = None,
              fallback_title: str | None = None) -> str:
    """ADI-style Markdown -> structured Markdown with front matter and page markers."""
    blocks: dict[str, str] = {}

    def stash(kind, fn):
        def _sub(m):
            key = f"\x00{kind}{len(blocks)}\x00"
            blocks[key] = fn(m.group(0))
            return f"\n{key}\n"
        return _sub

    # 1. pull out tables and figures first (they can span many lines)
    text = re.sub(r"<table\b.*?</table>", stash("T", table_to_pipe), adi_md, flags=re.S | re.I)
    text = re.sub(r"<figure\b.*?</figure>", stash("F", figure_to_md), text, flags=re.S | re.I)

    # 2. line pass
    out, page, title = ["<!-- page 1 -->", ""], 1, None
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip() in blocks:
            out += ["", blocks[line.strip()], ""]
            continue
        if PAGEBREAK.match(line):
            page += 1
            if page_rule:
                out += ["", "-----"]
            out += ["", f"<!-- page {page} -->", ""]
            continue
        if FURNITURE.match(line):
            if keep_furniture:
                out.append(line)
            continue
        line = html.unescape(checkboxes(line))
        if title is None and line.startswith("# "):
            title = line[2:].strip()
        # checkbox lines -> list items:  - **Question:** [x] A [ ] B
        if (bullets_for_checkboxes and has_marks(line)
                and not line.lstrip().startswith(("-", "*", "|", "#"))):
            t = re.sub(r"\s+", " ", line.strip())
            t = re.sub(r"^(\d+)\\?\. ", r"\1\\. ", t)              # keep "1." literal in a list
            t = re.sub(r"^([^\[]+?):?\s+\[",
                       lambda m: f"**{m[1]}{'' if m[1].endswith('?') else ':'}** [",
                       t, count=1)
            out.append(f"- {t}")
            continue
        out.append(line)

    # 3. front matter
    title = title or (fallback_title or None)
    fm = ["---"]
    if title:
        fm.append(f"title: {json.dumps(title, ensure_ascii=False)}")
    if source:
        fm.append(f"source: {json.dumps(source, ensure_ascii=False)}")
    fm.append(f"pages: {page}")
    for k, v in (extra_front_matter or {}).items():
        fm.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    if any(has_marks(x) for x in out):
        fm.append('checkbox_legend: "[x] = selected, [ ] = not selected"')
    fm += ["---", ""]

    md = "\n".join(fm + out)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def normalize_fragment(md: str) -> str:
    """Light clean-up for model output inserted after normalize()."""
    md = re.sub(r"<table\b.*?</table>", lambda m: table_to_pipe(m.group(0)), md, flags=re.S | re.I)
    return checkboxes(md).strip()


def _split_table_rows(text: str, n: int) -> list[str]:
    """Split each pipe table into pieces of n body rows, repeating its header."""
    lines, out, cur, i = text.splitlines(), [], [], 0
    while i < len(lines):
        if (lines[i].startswith("|") and i + 1 < len(lines)
                and re.fullmatch(r"\|(?:-+\|)+", lines[i + 1])):
            head, sep, i = lines[i], lines[i + 1], i + 2
            body = []
            while i < len(lines) and lines[i].startswith("|"):
                body.append(lines[i])
                i += 1
            if len(body) <= n:
                cur += [head, sep] + body
                continue
            for k in range(0, len(body), n):
                part = body[k:k + n]
                out.append("\n".join(cur + [f"<!-- rows {k + 1}-{k + len(part)} of {len(body)} -->",
                                            head, sep] + part).strip())
                cur = []
            continue
        cur.append(lines[i])
        i += 1
    if "\n".join(cur).strip():
        out.append("\n".join(cur).strip())
    return out or [text]


def chunk_by_headings(md: str, max_level: int = 3, rows_per_chunk: int = 0) -> list[dict]:
    """One chunk per heading (up to max_level); each chunk repeats its heading path."""
    body = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)
    chunks, path, buf, page, start_page = [], {}, [], 1, 1

    def flush():
        text = "\n".join(buf).strip()
        if not text or re.fullmatch(r"(<!--.*?-->\s*|-----\s*)+", text):
            return
        heads = "\n".join(f"{'#' * k} {path[k]}" for k in sorted(path))
        for part in (_split_table_rows(text, rows_per_chunk) if rows_per_chunk else [text]):
            chunks.append({"heading_path": " > ".join(path[k] for k in sorted(path)),
                           "page": start_page,
                           "text": (heads + "\n\n" + part).strip()})

    for line in body.splitlines():
        if (m := re.match(r"<!-- page (\d+) -->", line)):
            page = int(m.group(1))
            if not "".join(buf).strip():
                start_page = page
        h = re.match(r"^(#{1,6}) (.+)$", line)
        if h and len(h.group(1)) <= max_level:
            flush()
            buf, start_page = [], page
            lvl = len(h.group(1))
            path = {k: v for k, v in path.items() if k < lvl}
            path[lvl] = h.group(2).strip()
        else:
            buf.append(line)
    flush()
    return chunks


def md_cell(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).replace("|", "\\|").strip()


def rows_to_pipe(rows: Sequence[Sequence[Any]]) -> str:
    """List of rows -> pipe table. First row is the header."""
    rows = [list(r) for r in rows if r is not None]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    if width == 0:
        return ""
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(md_cell(c) or " " for c in rows[0]) + " |",
           "|" + "|".join(["---"] * width) + "|"]
    out += ["| " + " | ".join(md_cell(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


# =========================================================================== #
# 3. Format detection and LibreOffice conversion
# =========================================================================== #
class ConversionError(Exception):
    """The file cannot be converted. The message is shown to the user."""


class EncryptedFileError(ConversionError):
    pass


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
TEXT_EXT = {".txt", ".md", ".markdown", ".text", ".log"}
CSV_EXT = {".csv", ".tsv"}
OOXML_EXT = {".docx": "docx", ".docm": "docx", ".dotx": "docx",
             ".pptx": "pptx", ".pptm": "pptx", ".ppsx": "pptx", ".potx": "pptx",
             ".xlsx": "xlsx", ".xlsm": "xlsx", ".xltx": "xlsx"}
_DOCX_FILTER = ("docx", "docx:MS Word 2007 XML")
_PPTX_FILTER = ("pptx", "pptx:Impress MS PowerPoint 2007 XML")
_XLSX_FILTER = ("xlsx", "xlsx:Calc MS Excel 2007 XML")
# extension -> (target kind, LibreOffice filter)
LEGACY_EXT = {
    ".doc": _DOCX_FILTER, ".dot": _DOCX_FILTER, ".rtf": _DOCX_FILTER, ".odt": _DOCX_FILTER,
    ".wpd": _DOCX_FILTER, ".htm": _DOCX_FILTER, ".html": _DOCX_FILTER, ".xml": _DOCX_FILTER,
    ".ppt": _PPTX_FILTER, ".pps": _PPTX_FILTER, ".odp": _PPTX_FILTER,
    ".xls": _XLSX_FILTER, ".xlsb": _XLSX_FILTER, ".ods": _XLSX_FILTER,
}


def detect_kind(path: Path) -> str:
    """Return pdf | docx | pptx | xlsx | legacy | image | csv | text | json.
    Content wins over extension."""
    ext = path.suffix.lower()
    with path.open("rb") as fh:
        head = fh.read(8)
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        except zipfile.BadZipFile as exc:
            raise ConversionError(f"{path.name}: damaged ZIP container ({exc})") from exc
        if any(n.startswith("word/") for n in names):
            return "docx"
        if any(n.startswith("ppt/") for n in names):
            return "pptx"
        if any(n.startswith("xl/") for n in names):
            return "xlsx"
        if "mimetype" in names:              # OpenDocument
            return "legacy"
        raise ConversionError(f"{path.name}: ZIP file that is not an Office document")
    if head.startswith(b"\xd0\xcf\x11\xe0"):  # OLE2: legacy Office, or encrypted OOXML
        if ext in OOXML_EXT:
            raise EncryptedFileError(f"{path.name}: the file is password-protected. "
                                     "Remove the password and try again.")
        return "legacy"
    if ext in IMAGE_EXT:
        return "image"
    try:
        with Image.open(path) as im:
            im.verify()
        return "image"
    except Exception:  # noqa: BLE001
        pass
    if ext == ".json":
        return "json"
    if ext in CSV_EXT:
        return "csv"
    if ext in TEXT_EXT:
        return "text"
    if ext in LEGACY_EXT:
        return "legacy"
    raise ConversionError(f"{path.name}: unsupported file type '{ext or 'no extension'}'")


def soffice_binary() -> Optional[str]:
    for name in ("soffice", "libreoffice"):
        if shutil.which(name):
            return shutil.which(name)
    mac = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    return mac if os.path.exists(mac) else None


def soffice_convert(src: Path, target_filter: str, out_dir: Path, timeout: int = 300) -> Path:
    """Convert with headless LibreOffice. target_filter like 'docx:MS Word 2007 XML' or 'png'."""
    binary = soffice_binary()
    if not binary:
        raise ConversionError(f"{src.name}: LibreOffice is required for this format. Install it (soffice).")
    profile = Path(tempfile.mkdtemp(prefix="lo-profile-"))  # isolated profile: safe for parallel runs
    cmd = [binary, f"-env:UserInstallation={profile.as_uri()}", "--headless", "--norestore",
           "--convert-to", target_filter, "--outdir", str(out_dir), str(src)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ConversionError(f"{src.name}: LibreOffice timed out") from exc
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    ext = target_filter.split(":")[0]
    out = out_dir / f"{src.stem}.{ext}"
    if not out.exists():
        raise ConversionError(f"{src.name}: LibreOffice could not convert the file "
                              "(damaged or password-protected?)")
    return out


# =========================================================================== #
# 4. Image registry and image preparation
# =========================================================================== #
def placeholder(img_id: str) -> str:
    return f"<<{img_id}>>"


PLACEHOLDER_RE = re.compile(r"<<(IMG_\d+)>>")


def to_rgb(im: Image.Image) -> Image.Image:
    """Apply EXIF rotation; composite transparency onto white (not black)."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P", "PA") or (im.mode == "RGB" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        return bg
    if im.mode not in ("RGB", "L"):          # CMYK, I;16, F, 1 ...
        return im.convert("RGB")
    return im


def patch_count(w: int, h: int, s: Settings) -> int:
    return math.ceil(w / s.patch_px) * math.ceil(h / s.patch_px)


def fit_size(w: int, h: int, max_patches: int, s: Settings, allow_upscale: float = 1.0) -> tuple[int, int]:
    """Largest aspect-preserving size within the patch budget, model max side and upscale cap.

    'Budget-filling' rule (Anthropic zoom-tool cookbook) adapted to OpenAI's 32 px
    patches: a crop is magnified until it fills its budget, so every character gets
    as many patches as possible.
    """
    scale = min(allow_upscale,
                s.max_side_px / max(w, h),
                math.sqrt(max_patches * s.patch_px ** 2 / (w * h)))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    while patch_count(nw, nh, s) > max_patches and nw > 1 and nh > 1:
        nw, nh = max(1, int(nw * 0.98)), max(1, int(nh * 0.98))
    return nw, nh


def resize(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    return im if im.size == size else im.resize(size, Image.LANCZOS)


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=False)
    return buf.getvalue()


def data_url(im: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(im)).decode()


@dataclass
class ImageItem:
    id: str
    kind: str                      # "figure" (picture inside a page) or "page" (whole page/scan)
    image: Image.Image             # full-resolution original, RGB
    units: list[int]               # page / slide / sheet numbers (1-based)
    context: str = ""              # where it is: "slide 3: Q3 results"
    ocr_text: str = ""             # exact OCR text when ADI read it
    caption: str = ""
    sha: str = ""


class ImageRegistry:
    """Unique images across the document, each with a placeholder in the Markdown."""

    def __init__(self, settings: Settings, work_dir: Path):
        self.s = settings
        self.work_dir = work_dir
        self.items: dict[str, ImageItem] = {}
        self._by_hash: dict[str, Optional[str]] = {}
        self._n = 0

    def add_blob(self, blob: bytes, unit: int, ext: str = "", context: str = "") -> Optional[str]:
        """Add an embedded picture. Returns a placeholder, a note, or None (skipped)."""
        sha = hashlib.sha1(blob).hexdigest()
        if sha in self._by_hash:
            img_id = self._by_hash[sha]
            if img_id:
                if unit not in self.items[img_id].units:
                    self.items[img_id].units.append(unit)
                return placeholder(img_id)
            return None
        im = self._open(blob, ext)
        if im is None:
            self._by_hash[sha] = None
            return f"_(picture in {ext or 'unknown'} format could not be read)_"
        img_id = self._add(im, "figure", unit, context, sha)
        self._by_hash[sha] = img_id
        return placeholder(img_id) if img_id else None

    def add_image(self, im: Image.Image, unit: int, kind: str, context: str = "",
                  ocr_text: str = "", caption: str = "") -> Optional[str]:
        img_id = self._add(to_rgb(im), kind, unit, context, "", ocr_text, caption)
        return placeholder(img_id) if img_id else None

    def _add(self, im, kind, unit, context, sha, ocr_text="", caption="") -> Optional[str]:
        if kind == "figure" and min(im.size) < self.s.min_image_px:
            return None
        self._n += 1                       # ids stay unique even after drop_repeated()
        img_id = f"IMG_{self._n}"
        self.items[img_id] = ImageItem(img_id, kind, im, [unit], context, ocr_text, caption, sha)
        return img_id

    def _open(self, blob: bytes, ext: str) -> Optional[Image.Image]:
        try:
            im = Image.open(io.BytesIO(blob))
            im.load()
            return to_rgb(im)
        except Exception:  # noqa: BLE001
            pass
        # EMF / WMF / SVG and other vector pictures: rasterize with LibreOffice
        ext = (ext or "bin").lstrip(".").lower()
        if ext not in ("emf", "wmf", "svg", "emz", "wmz", "eps", "pict", "tif", "tiff"):
            return None
        try:
            src = self.work_dir / f"vec_{hashlib.sha1(blob).hexdigest()[:12]}.{ext}"
            src.write_bytes(blob)
            out = soffice_convert(src, "png", self.work_dir, timeout=120)
            im = Image.open(out)
            im.load()
            return to_rgb(im)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not rasterize %s picture: %s", ext, exc)
            return None

    def drop_repeated(self, n_units: int) -> list[str]:
        """Remove figures that appear on more than half of the pages/slides (logos)."""
        if n_units < 3:
            return []
        gone = [i for i, it in self.items.items()
                if it.kind == "figure" and len(it.units) > n_units / 2]
        for i in gone:
            del self.items[i]
        return gone


def overview_image(item: ImageItem, s: Settings) -> tuple[Image.Image, float]:
    """Pass-1 view: native resolution, small images magnified, big images fitted to budget.

    Returns the image and its scale versus the original (scale < 0.75 means detail was
    lost, which triggers automatic tiles in the zoom pass).
    """
    w, h = item.image.size
    upscale = min(s.max_upscale, max(1.0, s.min_long_side / max(w, h)))
    size = fit_size(w, h, s.overview_patches, s, allow_upscale=upscale)
    return resize(item.image, size), size[0] / w


def crop_region(item: ImageItem, box: tuple[float, float, float, float], s: Settings,
                pad: Optional[float] = None) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop a region given in a 0..999 grid from the full-resolution original and magnify it."""
    w, h = item.image.size
    x0, y0, x1, y1 = [max(0.0, min(999.0, float(v))) for v in box]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    px0, px1 = x0 * (w - 1) / 999, x1 * (w - 1) / 999
    py0, py1 = y0 * (h - 1) / 999, y1 * (h - 1) / 999
    pad = s.zoom_padding if pad is None else pad
    bw, bh = max(px1 - px0, 8), max(py1 - py0, 8)
    left = int(max(0, px0 - pad * bw))
    top = int(max(0, py0 - pad * bh))
    right = int(min(w, max(px1 + pad * bw + 1, left + 8)))
    bottom = int(min(h, max(py1 + pad * bh + 1, top + 8)))
    crop = item.image.crop((left, top, right, bottom))
    size = fit_size(crop.width, crop.height, s.zoom_patches, s, allow_upscale=s.max_upscale)
    return resize(crop, size), (left, top, right, bottom)


def auto_tiles(item: ImageItem, s: Settings) -> list[tuple[float, float, float, float]]:
    """Overlapping grid (0..999 coords) covering the whole image.

    The tile count is what native resolution needs, capped at max_auto_tiles; the grid
    always covers the full image (tiles get larger, not dropped, when capped).
    """
    w, h = item.image.size
    need = min(max(1.0, patch_count(w, h, s) / s.zoom_patches), float(s.max_auto_tiles))
    cols = max(1, min(s.max_auto_tiles, round(math.sqrt(need * w / h)) or 1))
    rows = max(1, min(s.max_auto_tiles // cols or 1, math.ceil(need / cols)))
    boxes = []
    ox, oy = s.tile_overlap / cols, s.tile_overlap / rows
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c / cols, (c + 1) / cols
            y0, y1 = r / rows, (r + 1) / rows
            boxes.append((max(0, x0 - ox) * 999, max(0, y0 - oy) * 999,
                          min(1, x1 + ox) * 999, min(1, y1 + oy) * 999))
    return boxes


# =========================================================================== #
# 5. Native converters
#    Each returns ADI-style Markdown: units separated by <!-- PageBreak -->,
#    pictures as <<IMG_n>> placeholders on their own line.
# =========================================================================== #
@dataclass
class Converted:
    body: str
    units: int
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- PowerPoint
def convert_pptx(path: Path, reg: ImageRegistry, s: Settings) -> Converted:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
    from pptx.util import Emu

    ns_a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ns_dgm = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
    row_tol = Emu(150000)

    prs = Presentation(str(path))
    title_types = (PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE, PP_PLACEHOLDER.VERTICAL_TITLE)

    def reading_order(shapes):
        """Top-to-bottom rows (shapes whose tops are within row_tol), left-to-right inside."""
        shapes = sorted(shapes, key=lambda sh: ((sh.top or 0), (sh.left or 0)))
        rows, cur, top0 = [], [], None
        for sh in shapes:
            top = sh.top or 0
            if top0 is None or abs(top - top0) <= row_tol:
                cur.append(sh)
                top0 = top if top0 is None else top0
            else:
                rows.append(cur)
                cur, top0 = [sh], top
        if cur:
            rows.append(cur)
        return [sh for row in rows for sh in sorted(row, key=lambda x: x.left or 0)]

    def para_text(p) -> str:
        return "".join(r.text for r in p.runs)

    def text_frame_md(tf, is_title: bool) -> str:
        paras = [p for p in tf.paragraphs if para_text(p).strip()]
        if not paras:
            return ""
        if is_title:
            return "## " + " ".join(para_text(p).strip() for p in paras)
        bullet = len(paras) > 1
        lines = []
        for p in paras:
            t = "".join(f"**{r.text.strip()}**" if r.font.bold and r.text.strip() else r.text
                        for r in p.runs).strip()
            lines.append(("  " * p.level + "- " + t) if bullet else t)
        return "\n".join(lines)

    def table_md(tbl) -> str:
        n_rows, n_cols = len(tbl.rows), len(tbl.columns)
        grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        for r in range(n_rows):
            for c in range(n_cols):
                cell = tbl.cell(r, c)
                if cell.is_merge_origin:
                    for dr in range(cell.span_height):
                        for dc in range(cell.span_width):
                            if r + dr < n_rows and c + dc < n_cols:
                                grid[r + dr][c + dc] = cell.text
                elif not cell.is_spanned:
                    grid[r][c] = cell.text
        return rows_to_pipe(grid)

    def chart_md(chart) -> str:
        try:
            title = chart.chart_title.text_frame.text.strip() if chart.has_title else ""
            plot = chart.plots[0]
            cats = [str(c) for c in plot.categories]
            series = [sr for p in chart.plots for sr in p.series]
            rows = [["Category"] + [sr.name or f"Series {i + 1}" for i, sr in enumerate(series)]]
            vals = [list(sr.values) for sr in series]
            for i, cat in enumerate(cats):
                rows.append([cat] + ["" if i >= len(v) or v[i] is None else format(v[i], ".15g")
                                     for v in vals])
            return f"**Chart:** {title}\n\n{rows_to_pipe(rows)}".strip()
        except Exception as exc:  # noqa: BLE001
            return f"_(chart data could not be read: {exc})_"

    def smartart_md(shape, slide) -> str:
        try:
            rel = shape._element.find(f".//{{{ns_dgm}}}relIds")
            if rel is None:
                return ""
            part = slide.part.related_part(rel.get(f"{{{ns_r}}}dm"))
            from lxml import etree
            root = etree.fromstring(part.blob)
            texts = [t.text.strip() for t in root.iter(f"{{{ns_a}}}t") if t.text and t.text.strip()]
            return "\n".join(f"- {t}" for t in texts)
        except Exception:  # noqa: BLE001
            return ""

    def blips(shape, slide, unit, ctx) -> list[str]:
        out = []
        for blip in shape._element.iter(f"{{{ns_a}}}blip"):
            rid = blip.get(f"{{{ns_r}}}embed")
            if not rid:
                continue
            try:
                part = slide.part.related_part(rid)
            except KeyError:
                continue
            ph = reg.add_blob(part.blob, unit, Path(str(part.partname)).suffix, ctx)
            if ph:
                out.append(ph)
        return out

    def shapes_md(shapes, slide, unit, ctx) -> list[str]:
        blocks = []
        for sh in reading_order(list(shapes)):
            if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                blocks += shapes_md(sh.shapes, slide, unit, ctx)
                continue
            if getattr(sh, "has_table", False) and sh.has_table:
                blocks.append(table_md(sh.table))
                continue
            if getattr(sh, "has_chart", False) and sh.has_chart:
                blocks.append(chart_md(sh.chart))
                continue
            sa = smartart_md(sh, slide)
            if sa:
                blocks.append(sa)
                continue
            if sh.has_text_frame:
                is_title = sh.is_placeholder and sh.placeholder_format.type in title_types
                t = text_frame_md(sh.text_frame, is_title)
                if t:
                    blocks.append(t)
            blocks += blips(sh, slide, unit, ctx)  # pictures, picture fills, OLE previews
        return [b for b in blocks if b]

    pages = []
    for n, slide in enumerate(prs.slides, start=1):
        hidden = slide._element.get("show") in ("0", "false")
        if hidden and not s.include_hidden:
            pages.append(f"_(slide {n} is hidden and was skipped)_")
            continue
        title = ""
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            title = slide.shapes.title.text_frame.text.strip()
        ctx = f"slide {n}" + (f": {title}" if title else "")
        blocks = shapes_md(slide.shapes, slide, n, ctx)
        if hidden:
            blocks.insert(0, "_(hidden slide)_")
        if s.include_notes and slide.has_notes_slide:
            tf = slide.notes_slide.notes_text_frame
            notes = tf.text.strip() if tf is not None else ""
            if notes:
                blocks.append("> **Speaker notes:** " + notes.replace("\n", "\n> "))
        pages.append("\n\n".join(blocks))
    title = (prs.core_properties.title or "").strip()
    return Converted(f"\n\n{PAGEBREAK_TAG}\n\n".join(pages), len(pages),
                     {"format": "pptx", **({"doc_title": title} if title else {})})


# ---------------------------------------------------------------- Word
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
V = "urn:schemas-microsoft-com:vml"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
WINGDING_BOXES = {"F0FE": "☒", "F078": "☒", "F0FD": "☒", "F0A8": "☐", "F06F": "☐", "F071": "☐"}


def _q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


class _DocxWalker:
    """Walk a Word XML subtree in document order: text, checkboxes, pictures, page breaks."""

    SKIP = {_q(W, "del"), _q(W, "delText"), _q(MC, "Fallback"), _q(W, "instrText"),
            _q(W, "rPr"), _q(W, "pPr"), _q(W, "sectPr")}

    def __init__(self, part, reg: ImageRegistry, s: Settings):
        self.part, self.reg, self.s = part, reg, s
        self.unit = 1

    def walk(self, el, text: list[str], pics: list[str], ctx: str, top=None) -> bool:
        """Append text and picture placeholders. Returns True if a page break was found."""
        brk = False
        tag = el.tag
        if not isinstance(tag, str) or tag in self.SKIP:   # comments / processing instructions
            return False
        if tag == _q(W, "t"):
            text.append(el.text or "")
        elif tag == _q(W, "tab"):
            text.append(" ")
        elif tag in (_q(W, "br"), _q(W, "cr")):
            if el.get(_q(W, "type")) == "page":
                brk = True
            else:
                text.append(" ")
        elif tag == _q(W, "lastRenderedPageBreak"):
            brk = True
        elif tag == _q(W, "sym"):
            text.append(WINGDING_BOXES.get((el.get(_q(W, "char")) or "").upper(), ""))
        elif tag == _q(W, "checkBox"):              # legacy form field
            checked = el.find(_q(W, "checked"))
            default = el.find(_q(W, "default"))
            on = checked is not None and checked.get(_q(W, "val"), "1") not in ("0", "false")
            if checked is None and default is not None:
                on = default.get(_q(W, "val")) not in ("0", "false")
            text.append("☒ " if on else "☐ ")
        elif tag in (_q(A, "blip"), _q(V, "imagedata")):
            rid = el.get(_q(R, "embed")) or el.get(_q(R, "id"))
            if rid and rid in self.part.related_parts:
                p = self.part.related_parts[rid]
                ph = self.reg.add_blob(p.blob, self.unit, Path(str(p.partname)).suffix, ctx)
                if ph:
                    pics.append(ph)
        elif tag == _q(W, "p") and el is not top:
            text.append(" ")
        for child in el:
            brk = self.walk(child, text, pics, ctx, top) or brk
        return brk


def convert_docx(path: Path, reg: ImageRegistry, s: Settings) -> Converted:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    walker = _DocxWalker(d.part, reg, s)
    blocks: list[str] = []
    page = 1

    def para_md(p: Paragraph) -> tuple[str, list[str], bool]:
        text, pics = [], []
        brk = walker.walk(p._p, text, pics, f"page {page}", top=p._p)
        t = re.sub(r"\s+", " ", "".join(text)).strip()
        try:
            style = (p.style.name if p.style is not None else "") or ""
        except Exception:  # noqa: BLE001  (broken style references)
            style = ""
        if t:
            m = re.match(r"Heading (\d)", style, flags=re.I)
            if style == "Title":
                t = "# " + t
            elif m:
                t = "#" * min(6, int(m.group(1)) + 1) + " " + t
            else:
                num = p._p.find(f"{_q(W, 'pPr')}/{_q(W, 'numPr')}")
                if num is not None or style.startswith("List"):
                    lvl_el = num.find(_q(W, "ilvl")) if num is not None else None
                    lvl = int(lvl_el.get(_q(W, "val"), 0)) if lvl_el is not None else 0
                    marker = "1." if "Number" in style else "-"
                    t = "  " * lvl + f"{marker} {t}"
        return t, pics, brk

    def table_md(tbl: Table) -> tuple[str, list[str]]:
        rows, pics = [], []
        for row in tbl.rows:
            cells = []
            for cell in row.cells:                 # merged cells come back repeated
                text: list[str] = []
                walker.walk(cell._tc, text, pics, f"page {page} (table)", top=cell._tc)
                cells.append(re.sub(r"\s+", " ", "".join(text)).strip())
            rows.append(cells)
        return rows_to_pipe(rows), list(dict.fromkeys(pics))

    def block(el):
        nonlocal page
        walker.unit = page
        if el.tag == _q(W, "p"):
            t, pics, brk = para_md(Paragraph(el, d))
            if t:
                blocks.append(t)
            blocks.extend(pics)
            if brk:
                blocks.append(PAGEBREAK_TAG)
                page += 1
        elif el.tag == _q(W, "tbl"):
            t, pics = table_md(Table(el, d))
            if t:
                blocks.append(t)
            blocks.extend(pics)
        elif el.tag == _q(W, "sdt"):                # content control wrapping blocks
            content = el.find(_q(W, "sdtContent"))
            for child in (content if content is not None else []):
                block(child)

    for el in d.element.body.iterchildren():
        block(el)

    # footnotes and endnotes
    from lxml import etree
    for kind in ("footnotes", "endnotes"):
        part = next((p for p in d.part.package.parts if str(p.partname) == f"/word/{kind}.xml"), None)
        if part is None:
            continue
        root = etree.fromstring(part.blob)
        notes = []
        for note in root.iter(_q(W, kind[:-1])):
            nid = note.get(_q(W, "id"))
            if nid is None or int(nid) <= 0:     # separators
                continue
            t = re.sub(r"\s+", " ", "".join(x.text or "" for x in note.iter(_q(W, "t")))).strip()
            if t:
                notes.append(f"{nid}. {t}")
        if notes:
            blocks.append(f"## {kind.capitalize()}\n\n" + "\n".join(notes))

    body = "\n\n".join(b for b in blocks if b)
    title = (d.core_properties.title or "").strip()
    return Converted(body, page, {"format": "docx", **({"doc_title": title} if title else {})})


# ---------------------------------------------------------------- Excel
def _fmt_cell(value: Any, number_format: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time(0) else value.isoformat(sep=" ")
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return str(value)
    if isinstance(value, (int, float)):
        nf = number_format or "General"
        if "%" in nf:
            m = re.search(r"\.(0+)", nf)
            return f"{value * 100:.{len(m.group(1)) if m else 0}f}%"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    return str(value).strip()


def convert_xlsx(path: Path, reg: ImageRegistry, s: Settings) -> Converted:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True)          # cached values
    try:
        wbf = openpyxl.load_workbook(str(path), data_only=False)    # formulas (fallback)
    except Exception:  # noqa: BLE001
        wbf = None
    pages, uncached = [], 0

    for n, ws in enumerate(wb.worksheets, start=1):
        hidden = ws.sheet_state != "visible"
        if hidden and not s.include_hidden:
            continue
        unit = len(pages) + 1
        head = f"## Sheet: {ws.title}" + (" (hidden)" if hidden else "")
        wsf = wbf[ws.title] if wbf is not None and ws.title in wbf.sheetnames else None

        # merged ranges: repeat the top-left value in every cell
        merged: dict[tuple[int, int], tuple[int, int]] = {}
        for rng in ws.merged_cells.ranges:
            for r in range(rng.min_row, rng.max_row + 1):
                for c in range(rng.min_col, rng.max_col + 1):
                    merged[(r, c)] = (rng.min_row, rng.min_col)

        max_r, max_c = ws.max_row or 0, ws.max_column or 0
        if s.max_sheet_rows:
            max_r = min(max_r, s.max_sheet_rows)
        grid: list[list[str]] = []
        for r in range(1, max_r + 1):
            row = []
            for c in range(1, max_c + 1):
                src = merged.get((r, c), (r, c))
                cell = ws.cell(*src)
                v = cell.value
                if v is None and wsf is not None:
                    f = wsf.cell(*src).value
                    if isinstance(f, str) and f.startswith("="):
                        uncached += 1
                        row.append(f"`{f}`")
                        continue
                row.append(_fmt_cell(v, cell.number_format))
            grid.append(row)

        # split into blocks at empty rows; trim empty columns per block
        blocks_md, block = [], []
        for row in grid + [[]]:
            if any(x for x in row):
                block.append(row)
                continue
            if block:
                width = max(len(r) for r in block)
                cols = [c for c in range(width) if any(c < len(r) and r[c] for r in block)]
                rows = [[r[c] if c < len(r) else "" for c in cols] for r in block]
                first = {x for x in rows[0] if x}
                if len(rows) > 1 and len(first) == 1:       # title row (single or merged cell)
                    blocks_md.append(f"**{first.pop()}**")
                    rows = rows[1:]
                if len(rows) == 1 and len({x for x in rows[0] if x}) == 1:
                    blocks_md.append(next(x for x in rows[0] if x))
                else:
                    blocks_md.append(rows_to_pipe(rows))
                block = []
        if s.max_sheet_rows and (ws.max_row or 0) > s.max_sheet_rows:
            blocks_md.append(f"_(sheet truncated at {s.max_sheet_rows} of {ws.max_row} rows)_")

        for chart in getattr(ws, "_charts", []):
            try:
                refs = [sr.val.numRef.f for sr in chart.series if sr.val and sr.val.numRef]
                blocks_md.append(f"**Chart:** {type(chart).__name__} - data: {', '.join(refs)}")
            except Exception:  # noqa: BLE001
                blocks_md.append("**Chart:** (data is in the cells above)")
        for img in getattr(ws, "_images", []):
            try:
                ph = reg.add_blob(img._data(), unit, getattr(img, "format", "png"), f"sheet {ws.title}")
                if ph:
                    blocks_md.append(ph)
            except Exception as exc:  # noqa: BLE001
                log.debug("sheet image skipped: %s", exc)

        pages.append(head + "\n\n" + ("\n\n".join(blocks_md) or "_(empty sheet)_"))

    if uncached:
        log.warning("%d formula cells have no saved value (shown as `=FORMULA`). "
                    "Open and save the file in Excel to store the values.", uncached)
    title = (wb.properties.title or "").strip() if wb.properties else ""
    return Converted(f"\n\n{PAGEBREAK_TAG}\n\n".join(pages), len(pages),
                     {"format": "xlsx", "sheets": len(pages), **({"doc_title": title} if title else {})})


# ---------------------------------------------------------------- CSV / text / ADI JSON
def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def convert_csv(path: Path, reg: ImageRegistry, s: Settings) -> Converted:
    text = _read_text(path)
    try:
        dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    total, note = len(rows), ""
    if s.max_sheet_rows and total > s.max_sheet_rows:
        note = f"\n\n_(truncated at {s.max_sheet_rows} of {total} rows)_"
        rows = rows[: s.max_sheet_rows]
    return Converted(rows_to_pipe(rows) + note, 1, {"format": "csv", "rows": total})


def convert_text(path: Path, reg: ImageRegistry, s: Settings) -> Converted:
    text = _read_text(path)
    return Converted(text, text.count(PAGEBREAK_TAG) + 1,
                     {"format": path.suffix.lower().lstrip(".") or "text"})


def convert_json(path: Path, reg: ImageRegistry, s: Settings) -> Converted:
    """A saved ADI result (analyzeResult.content) -> its Markdown; other JSON -> code block."""
    raw = _read_text(path)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"{path.name}: invalid JSON ({exc})") from exc
    res = data.get("analyzeResult", data) if isinstance(data, dict) else None
    if isinstance(res, dict) and isinstance(res.get("content"), str):
        if res.get("contentFormat", "markdown") != "markdown":
            log.warning("%s: the result was not produced with outputContentFormat=markdown", path.name)
        content = res["content"]
        return Converted(content, content.count(PAGEBREAK_TAG) + 1,
                         {"format": "adi-json", "ocr": "azure-document-intelligence"})
    return Converted("```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```", 1,
                     {"format": "json"})


# =========================================================================== #
# 6. PDF converter (PyMuPDF) and Azure Document Intelligence route
# =========================================================================== #
def _render_clip(page, rect, dpi: int) -> Image.Image:
    pix = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _page_dpi_for_budget(page, s: Settings, rect=None) -> int:
    """Highest DPI <= render_dpi that keeps the region inside the per-image patch limit."""
    rect = rect or page.rect
    w_in, h_in = rect.width / 72, rect.height / 72
    dpi = s.render_dpi
    while dpi > 72 and (patch_count(int(w_in * dpi), int(h_in * dpi), s) > s.max_patches_per_image
                        or max(w_in, h_in) * dpi > s.max_side_px):
        dpi -= 10
    return dpi


def _pdf_markdown_fn():
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown
    except ImportError:
        log.warning("pymupdf4llm is not installed: PDF tables come out as plain text")
        return None


def convert_pdf(path: Path, reg: ImageRegistry, s: Settings) -> Converted:
    import pymupdf

    try:                                   # PyMuPDF / pymupdf4llm notices -> logging, not stdout
        pymupdf.set_messages(pylogging=True, pylogging_level=logging.DEBUG)
    except Exception:  # noqa: BLE001
        pass
    doc = pymupdf.open(str(path))
    if doc.needs_pass and not doc.authenticate(""):
        raise EncryptedFileError(f"{path.name}: the PDF is password-protected. Remove the password and try again.")
    rag_markdown = _pdf_markdown_fn()

    pages_md = []
    for i, page in enumerate(doc):
        n = i + 1
        area = abs(page.rect) or 1
        chars = len(page.get_text().strip())
        infos = [x for x in page.get_image_info(xrefs=True) if x.get("bbox")]
        cover = max((abs(pymupdf.Rect(x["bbox"]) & page.rect) / area for x in infos), default=0)
        scanned = s.pdf_mode == "vision" or (
            s.pdf_mode == "auto" and (chars < s.scan_text_chars or cover >= s.scan_image_cover))

        if scanned:
            dpi = _page_dpi_for_budget(page, s)
            ph = reg.add_image(_render_clip(page, page.rect, dpi), n, "page", f"page {n} (whole page)")
            pages_md.append(ph or "")
            continue

        figs, fig_rects = [], []
        # raster pictures: render the region (keeps overlaid labels) at native density
        for x in sorted(infos, key=lambda x: (x["bbox"][1], x["bbox"][0])):
            rect = pymupdf.Rect(x["bbox"]) & page.rect
            if rect.is_empty or min(rect.width, rect.height) < s.min_image_px * 72 / s.render_dpi:
                continue
            if any(abs(rect & r) > 0.8 * abs(rect) for r in fig_rects):   # same picture twice
                continue
            native_dpi = 72 * x.get("width", 0) / max(rect.width, 1)
            dpi = min(int(max(s.render_dpi, min(native_dpi, 600))), _page_dpi_for_budget(page, s, rect) * 2)
            ph = reg.add_blob(png_bytes(_render_clip(page, rect, max(72, dpi))), n, "png", f"page {n}")
            if ph:
                figs.append(ph)
                fig_rects.append(rect)
        # vector charts / diagrams: clusters of drawings that are not tables
        if s.vector_figures:
            try:
                tables = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
                for rect in page.cluster_drawings():
                    rect = pymupdf.Rect(rect) & page.rect
                    if abs(rect) < 0.04 * area or rect.height < 40:
                        continue
                    if any(abs(rect & tb) > 0.5 * abs(rect) for tb in tables):
                        continue
                    if any(abs(rect & fr) > 0.5 * abs(rect) for fr in fig_rects):
                        continue
                    # include labels just outside the drawing (titles, axis ticks)
                    rect = (rect + (-24, -24, 24, 24)) & page.rect
                    ph = reg.add_image(_render_clip(page, rect, s.render_dpi), n, "figure",
                                       f"page {n} (vector graphic)")
                    if ph:
                        figs.append(ph)
                        fig_rects.append(rect)
            except Exception as exc:  # noqa: BLE001
                log.debug("vector figure detection failed on page %d: %s", n, exc)

        # page text without the text inside figures (the vision model reads that)
        text_doc, text_page = doc, i
        if fig_rects:
            text_doc = pymupdf.open()
            text_doc.insert_pdf(doc, from_page=i, to_page=i)
            tp = text_doc[0]
            for rect in fig_rects:
                tp.add_redact_annot(rect)
            try:
                tp.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                    graphics=getattr(pymupdf, "PDF_REDACT_LINE_ART_NONE", 0))
            except TypeError:                                  # older PyMuPDF: no graphics arg
                tp.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)
            text_page = 0
        text = ""
        if rag_markdown is not None:
            try:
                text = rag_markdown(text_doc, pages=[text_page], ignore_images=True, show_progress=False)
            except Exception as exc:  # noqa: BLE001
                log.debug("pymupdf4llm failed on page %d: %s", n, exc)
                text = ""
        if not text.strip():
            text = text_doc[text_page].get_text()
        pages_md.append("\n\n".join(x for x in [text.strip(), *figs] if x))

    meta: dict[str, Any] = {"format": "pdf"}
    title = (doc.metadata or {}).get("title") or ""
    if title.strip():
        meta["doc_title"] = title.strip()
    return Converted(f"\n\n{PAGEBREAK_TAG}\n\n".join(pages_md), len(pages_md), meta)


def _image_frames(path: Path) -> list[Image.Image]:
    """Every page of a multi-page TIFF; the first frame of anything else (animated GIF/WebP)."""
    im = Image.open(path)
    n = getattr(im, "n_frames", 1) if (im.format or "").upper() in ("TIFF", "MPO") else 1
    frames = []
    for i in range(n):
        im.seek(i)
        frames.append(to_rgb(im.copy()))
    return frames


def convert_image(path: Path, reg: ImageRegistry, s: Settings) -> Converted:
    frames = [reg.add_image(f, i + 1, "page", f"page {i + 1}") or "" for i, f in enumerate(_image_frames(path))]
    return Converted(f"\n\n{PAGEBREAK_TAG}\n\n".join(frames), len(frames), {"format": "image"})


def adi_configured() -> bool:
    return bool(os.getenv("AZURE_DI_ENDPOINT") and os.getenv("AZURE_DI_KEY"))


def convert_with_adi(path: Path, reg: ImageRegistry, s: Settings, kind: str) -> Converted:
    """Azure Document Intelligence layout (exact OCR, tables, checkboxes) + vision for figures.

    Figures are cropped by us from the full-resolution source (PDF rendered at render_dpi,
    images at native pixels) instead of ADI's preview crops, and ADI's OCR text of each
    figure is sent to the vision model as grounding for exact digits.
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(os.environ["AZURE_DI_ENDPOINT"],
                                        AzureKeyCredential(os.environ["AZURE_DI_KEY"]))
    started = time.perf_counter()
    result = client.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=path.read_bytes()),
        output_content_format=DocumentContentFormat.MARKDOWN,
        string_index_type="unicodeCodePoint",      # span offsets match Python str indices
    ).result()
    log.info("ADI layout: %d pages in %.1fs", len(result.pages or []), time.perf_counter() - started)
    content = result.content or ""
    units = {p.page_number: (p.unit or "pixel") for p in result.pages or []}

    pdf = None
    src_frames: dict[int, Image.Image] = {}
    if kind == "pdf":
        import pymupdf
        pdf = pymupdf.open(str(path))
    else:
        src_frames = {i + 1: f for i, f in enumerate(_image_frames(path))}

    edits = []   # (start, end, replacement)
    for fig in result.figures or []:
        if not fig.spans or not fig.bounding_regions:
            continue
        region = fig.bounding_regions[0]
        poly = region.polygon or []
        xs, ys = poly[0::2], poly[1::2]
        if not xs:
            continue
        pn = region.page_number
        try:
            if pdf is not None:
                import pymupdf
                k = 72.0 if units.get(pn) == "inch" else 1.0
                rect = pymupdf.Rect(min(xs) * k, min(ys) * k, max(xs) * k, max(ys) * k)
                im = _render_clip(pdf[pn - 1], rect, _page_dpi_for_budget(pdf[pn - 1], s, rect))
            else:
                im = src_frames[pn].crop((int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1))
        except Exception as exc:  # noqa: BLE001
            log.debug("figure crop failed: %s", exc)
            continue
        start = fig.spans[0].offset
        end = fig.spans[-1].offset + fig.spans[-1].length
        ocr = clean_inline(re.sub(r"<figcaption.*?</figcaption>", "", content[start:end], flags=re.S))
        caption = fig.caption.content if getattr(fig, "caption", None) else ""
        ph = reg.add_image(im, pn, "figure", f"page {pn}", ocr_text=ocr, caption=caption)
        if ph:
            edits.append((start, end, f"\n\n{ph}\n\n"))   # "**Figure:** caption" added by the pipeline
    for start, end, rep in sorted(edits, reverse=True):
        content = content[:start] + rep + content[end:]
    return Converted(content, len(result.pages or []) or 1,
                     {"format": kind, "ocr": "azure-document-intelligence"})


# =========================================================================== #
# 7. Vision: overview pass + zoom pass
# =========================================================================== #
class ZoomRegion(BaseModel):
    x_min: int = Field(description="Left edge, 0..999 grid over the image as shown.")
    y_min: int = Field(description="Top edge, 0..999 grid.")
    x_max: int = Field(description="Right edge, 0..999 grid.")
    y_max: int = Field(description="Bottom edge, 0..999 grid.")
    reason: str = Field(description="What is too small or unclear in this region.")


class ImageResult(BaseModel):
    id: str = Field(description="The image id, for example IMG_3.")
    markdown: str = Field(description="All content of the image as Markdown.")
    confidence: float = Field(description="0 to 1. How sure you are that every character and number is correct.")
    zoom_regions: list[ZoomRegion] = Field(
        description="Regions to examine at higher magnification. Empty if all text is clear.")


class ImageBatch(BaseModel):
    images: list[ImageResult]


# Written in ASD-STE100 Simplified Technical English (same style as the extraction prompts).
OVERVIEW_PROMPT = """\
You change images from business documents into Markdown. Each image has a header line with its id, its kind and its position in the document.

For each image, write Markdown that contains all the information in the image.

Rules:
1. Copy all text, numbers, dates, signs, units, currency symbols and footnote marks exactly as they are shown. Do not correct them. Do not round numbers.
2. If the image shows a table or a financial statement, write a Markdown pipe table. Keep all rows, all columns and the column headers. Keep negative numbers as they are shown, for example (1,234) or -1,234.
3. If the image shows a chart, write the chart title. Then write a pipe table with each data label and each value that the chart shows. Then write one sentence about the trend. If a value is not printed and you read it from the axis, put "~" before it.
4. If the image shows a form, write each field as "Label: value". Write a selected checkbox as [x] and an empty checkbox as [ ].
5. If the image shows a diagram, write the text of each box and show the connections as a list.
6. If the image is a whole page, write the page in reading order. Use # headings for the headings on the page.
7. If the image is a photo, a logo or a decoration, write one short sentence in italics.
8. Do not add information that is not in the image. Do not summarize. Do not explain.
9. If OCR text is given for an image, use it as the correct spelling of the characters and numbers. Use the image for the structure.
10. In `confidence`, give a number from 0 to 1. Give a lower number if some text is small, blurred or cut.
11. In `zoom_regions`, give the regions where text or numbers are too small to read with full confidence. Use a 0..999 grid over the image as you see it. The origin (0,0) is the top-left corner. Give the smallest region that contains the item. Give at most {max_zooms} regions. If all text is clear, give an empty list.

Give one result for each image id, in the same order."""

ZOOM_PROMPT = """\
You correct Markdown transcriptions of images. For each image id you get:
- The full image (zoom out). Use it for the structure and the order.
- One or more magnified crops of regions of the same image (zoom in). Each crop has its region in a 0..999 grid over the full image. The crops come from the full-resolution original.
- The draft Markdown from the first pass.

For each image:
1. Read each crop carefully. The crops are the correct source for small text and numbers.
2. Compare the draft with the full image and the crops. Correct each wrong character, number, sign or label.
3. Add each item that the draft does not have.
4. Keep the rules: copy exactly, do not round, do not summarize, keep tables as pipe tables, write checkboxes as [x] and [ ].
5. Give the complete corrected Markdown for the whole image, not only the changes.
6. Give `confidence` from 0 to 1. Give an empty `zoom_regions` list.

Give one result for each image id, in the same order."""


def build_model(s: Settings):
    """ChatOpenAI on the Responses API. OPENAI_BASE_URL points it at a compatible gateway."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=s.model,
        use_responses_api=True,
        reasoning={"effort": s.reasoning_effort},
        verbosity=s.verbosity,
        max_retries=s.max_retries,
        timeout=s.timeout,
    )


def _image_block(im: Image.Image, s: Settings) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url(im), "detail": s.detail}}


def _batches(items: list[tuple[str, int]], max_items: int, max_patches: int) -> list[list[str]]:
    """Group (id, patches) into batches under both limits."""
    out, cur, cur_p = [], [], 0
    for img_id, p in items:
        if cur and (len(cur) >= max_items or cur_p + p > max_patches):
            out.append(cur)
            cur, cur_p = [], 0
        cur.append(img_id)
        cur_p += p
    if cur:
        out.append(cur)
    return out


@dataclass
class VisionStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0

    def add(self, raw: Any) -> None:
        self.calls += 1
        usage = getattr(raw, "usage_metadata", None) or {}
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)


async def _run_batches(model, prompt: str, payloads: list[list[dict]], s: Settings,
                       stats: VisionStats) -> list[Optional[ImageBatch]]:
    from langchain_core.messages import HumanMessage, SystemMessage

    structured = model.with_structured_output(ImageBatch, method="json_schema", strict=True, include_raw=True)
    inputs = [[SystemMessage(prompt), HumanMessage(content=p)] for p in payloads]
    results = await structured.abatch(inputs, config={"max_concurrency": s.vision_concurrency},
                                      return_exceptions=True)
    out: list[Optional[ImageBatch]] = []
    for r in results:
        if isinstance(r, BaseException):
            stats.errors += 1
            log.warning("vision call failed: %s: %s", type(r).__name__, r)
            out.append(None)
            continue
        stats.add(r.get("raw"))
        if r.get("parsing_error") or r.get("parsed") is None:
            stats.errors += 1
            log.warning("vision output could not be parsed: %s", r.get("parsing_error"))
            out.append(None)
            continue
        parsed = r["parsed"]
        out.append(parsed if isinstance(parsed, ImageBatch) else ImageBatch.model_validate(parsed))
    return out


def _header(item: ImageItem) -> str:
    h = f"--- Image id: {item.id} | kind: {'whole page' if item.kind == 'page' else 'picture'}"
    h += f" | position: {item.context or 'unknown'} | size: {item.image.width}x{item.image.height} px ---"
    if item.caption:
        h += f"\nCaption: {item.caption}"
    if item.ocr_text:
        h += f"\nOCR text (exact characters):\n{item.ocr_text[:6000]}"
    return h


async def transcribe_images(reg: ImageRegistry, s: Settings, model=None) -> tuple[dict[str, str], VisionStats]:
    """Pass 1: overview of every image. Pass 2: zoom where needed. Returns {id: markdown}."""
    stats = VisionStats()
    if not reg.items:
        return {}, stats
    model = model or build_model(s)
    items = reg.items

    # ---------------- pass 1: overview (zoom out), all images, as few calls as possible
    views = {i: overview_image(it, s) for i, it in items.items()}
    sizes = [(i, patch_count(*views[i][0].size, s)) for i in items]
    batches = _batches(sizes, s.max_images_per_call, s.max_patches_per_call)
    payloads = []
    for b in batches:
        content: list[dict] = []
        for i in b:
            content.append({"type": "text", "text": _header(items[i])})
            content.append(_image_block(views[i][0], s))
        payloads.append(content)
    prompt = OVERVIEW_PROMPT.format(max_zooms=s.max_zooms_per_image)
    started = time.perf_counter()
    results = await _run_batches(model, prompt, payloads, s, stats)
    log.info("vision pass 1: %d image(s) in %d call(s), %.1fs", len(items), len(payloads),
             time.perf_counter() - started)

    drafts: dict[str, ImageResult] = {}
    for b, res in zip(batches, results):
        got = {r.id: r for r in (res.images if res else [])}
        for i in b:
            if i in got:
                drafts[i] = got[i]

    # ---------------- pass 2: zoom in where the model asked, detail was lost, or confidence is low
    Box = tuple[float, float, float, float]
    for rnd in range(s.zoom_rounds):
        plan: dict[str, list[tuple[Box, str, Optional[float]]]] = {}
        for i, it in items.items():
            d = drafts.get(i)
            regions: list[tuple[Box, str, Optional[float]]] = [
                ((z.x_min, z.y_min, z.x_max, z.y_max), z.reason, None)
                for z in (d.zoom_regions if d else [])][: s.max_zooms_per_image]
            lost_detail = views[i][1] < 0.75                 # overview was downscaled a lot
            low_conf = d is None or d.confidence < s.zoom_confidence
            if not regions and (lost_detail or (low_conf and it.kind == "page")):
                regions = [(b, "automatic tile", 0.0) for b in auto_tiles(it, s)]
            elif not regions and low_conf:
                regions = [((0, 0, 999, 999), "low confidence: full image at maximum magnification", 0.0)]
            if regions:
                plan[i] = regions
        if not plan:
            break

        entries = []
        for i, regions in plan.items():
            crops = [(box, why, crop_region(items[i], box, s, pad)[0]) for box, why, pad in regions]
            p = patch_count(*views[i][0].size, s) + sum(patch_count(*c.size, s) for _, _, c in crops)
            entries.append((i, crops, p))
        groups = _batches([(i, p) for i, _, p in entries], s.max_images_per_call, s.max_patches_per_call)
        by_id = {i: crops for i, crops, _ in entries}
        payloads = []
        for g in groups:
            content = []
            for i in g:
                content.append({"type": "text", "text": _header(items[i]) + "\nFull image (zoom out):"})
                content.append(_image_block(views[i][0], s))
                for k, (box, why, crop) in enumerate(by_id[i], start=1):
                    box_s = ", ".join(str(int(v)) for v in box)
                    content.append({"type": "text", "text": f"{i} zoom {k}: region [{box_s}] ({why})"})
                    content.append(_image_block(crop, s))
                draft = drafts[i].markdown if i in drafts else "(no draft: the first pass failed)"
                content.append({"type": "text", "text": f"{i} draft Markdown:\n{draft}"})
            payloads.append(content)
        started = time.perf_counter()
        results = await _run_batches(model, ZOOM_PROMPT, payloads, s, stats)
        n_crops = sum(len(c) for c in by_id.values())
        log.info("vision pass 2 (zoom round %d): %d image(s), %d crop(s), %d call(s), %.1fs",
                 rnd + 1, len(plan), n_crops, len(payloads), time.perf_counter() - started)
        for g, res in zip(groups, results):
            for r in (res.images if res else []):
                if r.id in plan and r.markdown.strip():
                    r.zoom_regions = []      # one round of corrections per pass
                    drafts[r.id] = r

    out = {}
    for i, it in items.items():
        if i in drafts and drafts[i].markdown.strip():
            out[i] = normalize_fragment(drafts[i].markdown)
            if drafts[i].confidence < 0.5:
                out[i] += "\n\n_(low confidence: check this content against the source)_"
        elif it.ocr_text:
            out[i] = it.ocr_text + "\n\n_(vision model failed: OCR text only)_"
        else:
            out[i] = f"_(image {i} could not be read)_"
    return out, stats


def _run_async(coro):
    """asyncio.run, also when called from code that already has a running loop (Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


# =========================================================================== #
# 8. Pipeline and CLI
# =========================================================================== #
ROUTES: dict[str, Callable[[Path, ImageRegistry, Settings], Converted]] = {
    "pptx": convert_pptx, "docx": convert_docx, "xlsx": convert_xlsx, "csv": convert_csv,
    "text": convert_text, "json": convert_json, "pdf": convert_pdf, "image": convert_image,
}


def convert_file(path: str | Path, s: Settings = Settings(), model=None) -> str:
    """File path -> Markdown string. Raises ConversionError for files that cannot be read."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConversionError(f"{path}: file not found")
    if path.stat().st_size == 0:
        raise ConversionError(f"{path.name}: the file is empty")

    with tempfile.TemporaryDirectory(prefix="doc2md-") as tmp:
        work = Path(tmp)
        kind = detect_kind(path)
        src = path
        if kind == "legacy":
            target, filt = LEGACY_EXT.get(path.suffix.lower(), _DOCX_FILTER)
            log.info("%s: converting with LibreOffice -> %s", path.name, target)
            src = soffice_convert(path, filt, work)
            kind = target

        reg = ImageRegistry(s, work)
        if s.use_adi == "always" and kind in ("pdf", "image") and not adi_configured():
            raise ConversionError("use_adi=always but AZURE_DI_ENDPOINT / AZURE_DI_KEY are not set")
        use_adi = kind in ("pdf", "image") and (
            s.use_adi == "always" or (s.use_adi == "auto" and adi_configured()))
        log.info("%s: route=%s%s", path.name, kind, " + Azure Document Intelligence" if use_adi else "")

        try:
            conv = convert_with_adi(src, reg, s, kind) if use_adi else ROUTES[kind](src, reg, s)
        except ConversionError:
            raise
        except ImportError as exc:
            raise ConversionError(f"{path.name}: missing Python package for the {kind} route ({exc.name}). "
                                  "See the Install section at the top of this file.") from exc
        except Exception as exc:  # noqa: BLE001
            raise ConversionError(f"{path.name}: could not be read as {kind} "
                                  f"({type(exc).__name__}: {exc})") from exc

        if s.skip_repeated_images:
            gone = reg.drop_repeated(conv.units)
            if gone:
                log.info("skipped %d repeated picture(s) (logos): %s", len(gone), ", ".join(gone))
        body = conv.body
        for m in set(PLACEHOLDER_RE.findall(body)):
            if m not in reg.items:
                body = re.sub(rf"\n*{re.escape(placeholder(m))}\n*", "\n\n", body)

        log.info("%s: %d unit(s), %d picture(s)/scan(s) for the vision model",
                 path.name, conv.units, len(reg.items))
        descriptions: dict[str, str] = {}
        if reg.items:
            if model is None and not os.getenv("OPENAI_API_KEY"):
                raise ConversionError("OPENAI_API_KEY is not set (needed for pictures and scanned pages)")
            descriptions, stats = _run_async(transcribe_images(reg, s, model))
            log.info("vision: %d call(s), input=%d output=%d tokens, %d error(s)",
                     stats.calls, stats.input_tokens, stats.output_tokens, stats.errors)

        meta = dict(conv.meta)
        doc_title = meta.pop("doc_title", None)
        if reg.items:
            meta["vision_model"] = s.model
        md = normalize(body, source=path.name, page_rule=s.page_rule, keep_furniture=s.keep_furniture,
                       extra_front_matter=meta, fallback_title=doc_title)
        for img_id, text in descriptions.items():
            it = reg.items[img_id]
            label = "" if it.kind == "page" else ("**Figure:**" + (f" {it.caption}" if it.caption else "") + "\n\n")
            md = md.replace(placeholder(img_id), f"<!-- {img_id} -->\n{label}{text}")
        return re.sub(r"\n{3,}", "\n\n", md)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    d = Settings()
    p = argparse.ArgumentParser(description="Convert PDF / Word / PowerPoint / Excel / images to Markdown.")
    p.add_argument("input", help="path of the file to convert")
    p.add_argument("-o", "--out", help="output .md path (default: next to the input; '-' = stdout)")
    p.add_argument("--chunks", action="store_true", help="also write <out>.chunks.jsonl")
    p.add_argument("--rows-per-chunk", type=int, default=0,
                   help="split big tables into chunks of N rows (header repeated)")
    p.add_argument("--model", default=d.model)
    p.add_argument("--reasoning-effort", default=d.reasoning_effort,
                   choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    p.add_argument("--detail", default=d.detail, choices=["low", "high", "original", "auto"])
    p.add_argument("--zoom-rounds", type=int, default=d.zoom_rounds, help="0 disables the zoom pass")
    p.add_argument("--max-images-per-call", type=int, default=d.max_images_per_call)
    p.add_argument("--use-adi", default=d.use_adi, choices=["auto", "always", "never"])
    p.add_argument("--pdf-mode", default=d.pdf_mode, choices=["auto", "text", "vision"])
    p.add_argument("--render-dpi", type=int, default=d.render_dpi)
    p.add_argument("--keep-repeated-images", action="store_true")
    p.add_argument("--skip-hidden", action="store_true", help="skip hidden slides and sheets")
    p.add_argument("--no-notes", action="store_true", help="leave out speaker notes")
    p.add_argument("--max-sheet-rows", type=int, default=0)
    p.add_argument("--page-rule", action="store_true", help="add ----- at each page break")
    p.add_argument("--keep-furniture", action="store_true",
                   help="keep ADI PageHeader/PageFooter/PageNumber comments")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def settings_from(a: argparse.Namespace) -> Settings:
    return replace(
        Settings(), model=a.model, reasoning_effort=a.reasoning_effort, detail=a.detail,
        zoom_rounds=a.zoom_rounds, max_images_per_call=a.max_images_per_call, use_adi=a.use_adi,
        pdf_mode=a.pdf_mode, render_dpi=a.render_dpi, skip_repeated_images=not a.keep_repeated_images,
        include_hidden=not a.skip_hidden, include_notes=not a.no_notes,
        max_sheet_rows=a.max_sheet_rows, page_rule=a.page_rule, keep_furniture=a.keep_furniture,
    )


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).with_name(".env"))
    load_dotenv()                                   # working directory / parents


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    if not a.verbose:
        for noisy in ("httpx", "openai", "httpcore", "azure"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    _load_dotenv()
    started = time.perf_counter()
    try:
        md = convert_file(a.input, settings_from(a))
    except ConversionError as exc:
        log.error("%s", exc)
        return 2
    if a.out == "-":
        sys.stdout.write(md)
        return 0
    out = Path(a.out) if a.out else Path(a.input).with_suffix(".md")
    if out.resolve() == Path(a.input).resolve():
        out = out.with_name(out.stem + ".converted.md")
    out.write_text(md, encoding="utf-8")
    log.info("wrote %s (%.1fs)", out, time.perf_counter() - started)
    if a.chunks:
        cp = out.with_name(out.stem + ".chunks.jsonl")
        with cp.open("w", encoding="utf-8") as fh:
            for c in chunk_by_headings(md, rows_per_chunk=a.rows_per_chunk):
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        log.info("wrote %s", cp)
    return 0


# =========================================================================== #
# 9. Tests (no network):  pytest doc_to_markdown.py
# =========================================================================== #
class FakeVisionModel:
    """Stands in for ChatOpenAI. Records every request; answers with canned Markdown."""

    def __init__(self, answer: Callable[[str, int], str] | None = None, zoom_first: bool = False,
                 confidence: float = 0.95):
        self.requests: list[list[dict]] = []
        self.answer = answer or (lambda img_id, n: f"Transcribed {img_id}")
        self.zoom_first = zoom_first
        self.confidence = confidence

    def with_structured_output(self, *a, **k):
        return self

    async def abatch(self, inputs, config=None, return_exceptions=False):
        outs = []
        for system, human in inputs:
            self.requests.append(human.content)
            ids = [re.search(r"Image id: (IMG_\d+)", p["text"]).group(1)
                   for p in human.content if p["type"] == "text" and "Image id:" in p["text"]]
            zoom_pass = system.content.startswith("You correct")
            images = []
            for i in ids:
                regions = [] if zoom_pass or not self.zoom_first else [
                    ZoomRegion(x_min=0, y_min=0, x_max=500, y_max=500, reason="small digits")]
                images.append(ImageResult(id=i, markdown=self.answer(i, len(self.requests)),
                                          confidence=0.99 if zoom_pass else self.confidence,
                                          zoom_regions=regions))
            outs.append({"raw": None, "parsed": ImageBatch(images=images), "parsing_error": None})
        return outs


def _png(w=400, h=200, text="123.45") -> bytes:
    from PIL import ImageDraw
    im = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(im).text((20, 20), text, fill="black")
    return png_bytes(im)


def test_normalizer_tables_checkboxes_pages():
    md = normalize("# Title\n<table><tr><th>A</th><th>B</th></tr><tr><td colspan=2>x</td></tr></table>\n"
                   "Smoker: ☒ Yes ☐ No\n<!-- PageBreak -->\n<!-- PageFooter=\"p1\" -->\nPage two",
                   source="t.pdf")
    assert "| x | x |" in md and "- **Smoker:** [x] Yes [ ] No" in md and "<!-- page 2 -->" in md
    assert 'title: "Title"' in md and "pages: 2" in md and "PageFooter" not in md
    assert "checkbox_legend" in md


def test_multirow_header_and_chunks():
    t = ("<table><tr><th rowspan=2>Item</th><th colspan=2>FY</th></tr>"
         "<tr><th>2023</th><th>2024</th></tr><tr><td>Sales</td><td>1</td><td>2</td></tr></table>")
    md = normalize("# Doc\n## Part\n" + t)
    assert "| Item | FY / 2023 | FY / 2024 |" in md and "| Sales | 1 | 2 |" in md
    ch = chunk_by_headings(md)
    assert ch[-1]["heading_path"] == "Doc > Part" and "| Sales |" in ch[-1]["text"]


def test_sizing_rules():
    s = Settings()
    assert fit_size(100, 50, s.zoom_patches, s, allow_upscale=4.0) == (400, 200)   # magnified, capped 4x
    w, h = fit_size(8000, 6000, s.overview_patches, s)
    assert patch_count(w, h, s) <= s.overview_patches and max(w, h) <= s.max_side_px


def test_auto_tiles_cover_whole_image_and_are_capped():
    s = Settings()
    it = ImageItem("IMG_1", "page", Image.new("RGB", (12000, 16000), "white"), [1])
    tiles = auto_tiles(it, s)
    assert len(tiles) <= s.max_auto_tiles + 2
    assert min(t[0] for t in tiles) == 0 and min(t[1] for t in tiles) == 0
    assert max(t[2] for t in tiles) == 999 and max(t[3] for t in tiles) == 999


def test_docx_route(tmp_path):
    import docx
    d = docx.Document()
    d.core_properties.title = "Annual report"
    d.add_heading("Report", level=1)
    d.add_paragraph("Revenue grew.")
    d.add_paragraph("Smoker: ☒ Yes ☐ No")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text, t.cell(1, 0).text, t.cell(1, 1).text = "Item", "2024", "Sales", "1,200"
    (tmp_path / "p.png").write_bytes(_png())
    d.add_picture(str(tmp_path / "p.png"))
    f = tmp_path / "a.docx"
    d.save(f)
    fake = FakeVisionModel()
    md = convert_file(f, Settings(zoom_rounds=0), model=fake)
    assert "## Report" in md and "| Sales | 1,200 |" in md and "Transcribed IMG_1" in md
    assert 'title: "Annual report"' in md and "- **Smoker:** [x] Yes [ ] No" in md
    assert "**Figure:**" in md and len(fake.requests) == 1


def test_xlsx_route(tmp_path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "P&L"
    ws.append(["Income statement"])
    ws.append(["Item", "FY24", "Margin"])
    ws.append(["Revenue", 1200.5, 0.153])
    ws["C3"].number_format = "0.0%"
    ws.merge_cells("A1:C1")
    f = tmp_path / "a.xlsx"
    wb.save(f)
    md = convert_file(f, Settings(), model=FakeVisionModel())
    assert "## Sheet: P&L" in md and "| Revenue | 1200.5 | 15.3% |" in md
    assert "**Income statement**" in md and "| Item | FY24 | Margin |" in md


def test_pptx_route_single_call_and_zoom(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    for k in range(2):
        sl = prs.slides.add_slide(prs.slide_layouts[5])
        sl.shapes.title.text = f"Slide {k}"
        (tmp_path / f"i{k}.png").write_bytes(_png(text=f"value {k}"))
        sl.shapes.add_picture(str(tmp_path / f"i{k}.png"), Inches(1), Inches(2))
    f = tmp_path / "a.pptx"
    prs.save(f)
    fake = FakeVisionModel(zoom_first=True, answer=lambda i, n: f"{i} pass{n}")
    md = convert_file(f, Settings(), model=fake)
    assert len(fake.requests) == 2                       # 1 overview call + 1 zoom call
    assert sum(p["type"] == "image_url" for p in fake.requests[0]) == 2
    assert sum(p["type"] == "image_url" for p in fake.requests[1]) == 4   # 2 full views + 2 crops
    assert "IMG_1 pass2" in md and "## Slide 0" in md and "pages: 2" in md


def test_repeated_logo_dropped(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    (tmp_path / "logo.png").write_bytes(_png(text="LOGO"))
    for k in range(3):
        sl = prs.slides.add_slide(prs.slide_layouts[5])
        sl.shapes.title.text = f"Slide {k}"
        sl.shapes.add_picture(str(tmp_path / "logo.png"), Inches(1), Inches(2))
    f = tmp_path / "logo.pptx"
    prs.save(f)
    fake = FakeVisionModel()
    md = convert_file(f, Settings(), model=fake)
    assert not fake.requests and "<<IMG_" not in md


def test_pdf_scanned_and_text_pages(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Balance sheet text page. " * 5)
    p2 = doc.new_page()
    p2.insert_image(p2.rect, stream=_png(600, 800, "scanned 42"))
    f = tmp_path / "a.pdf"
    doc.save(f)
    fake = FakeVisionModel()
    md = convert_file(f, Settings(zoom_rounds=0, use_adi="never"), model=fake)
    assert "Balance sheet text page" in md and "Transcribed IMG_1" in md and "<!-- page 2 -->" in md


def test_low_confidence_page_gets_tiles(tmp_path):
    f = tmp_path / "scan.png"
    f.write_bytes(_png(2550, 3300, "scan"))
    fake = FakeVisionModel(confidence=0.5, answer=lambda i, n: f"{i} pass{n}")
    md = convert_file(f, Settings(use_adi="never"), model=fake)
    assert len(fake.requests) == 2 and sum(p["type"] == "image_url" for p in fake.requests[1]) >= 3
    assert "IMG_1 pass2" in md


def test_adi_json_route(tmp_path):
    f = tmp_path / "result.json"
    f.write_text(json.dumps({"analyzeResult": {"contentFormat": "markdown", "content":
                 "# ACORD 125\n:selected: New :unselected: Renewal\n<!-- PageBreak -->\nPage 2"}}))
    md = convert_file(f, Settings())
    assert 'title: "ACORD 125"' in md and "- [x] New [ ] Renewal" in md and "pages: 2" in md


def test_csv_route(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("name;amount\nA|B;1,5\n", encoding="utf-8")
    md = convert_file(f, Settings())
    assert "| name | amount |" in md and "| A\\|B | 1,5 |" in md


def test_encrypted_ooxml_detected(tmp_path):
    f = tmp_path / "locked.xlsx"
    f.write_bytes(b"\xd0\xcf\x11\xe0" + b"\0" * 600)
    try:
        detect_kind(f)
        assert False
    except EncryptedFileError:
        pass


if __name__ == "__main__":
    sys.exit(main())
