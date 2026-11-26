"""
Convert Azure Document Intelligence (ADI) Markdown output into structured,
LLM-ready Markdown. No Azure call, standard library only.

Input : the Markdown ADI returns with outputContentFormat=markdown
        (a .md file, or a saved result .json, whose analyzeResult.content is used)
Output: structured Markdown (+ optional heading-based chunks as .jsonl)

What it changes
  ADI Markdown                                  ->  structured Markdown
  --------------------------------------------      -----------------------------------------
  <table><tr><th>/<td> (rowspan/colspan)        ->  GitHub pipe tables (merged cells repeated,
                                                    multi-row headers as "Parent / Child")
  ☒ / ☐  (and :selected: / :unselected:)        ->  [x] / [ ]
  lines with checkboxes                         ->  list items: - **Question:** [x] A [ ] B
  <!-- PageBreak -->                            ->  <!-- page N -->   (+ ----- with --page-rule)
  <!-- PageHeader/PageFooter/PageNumber=... --> ->  removed (repeated noise); --keep-furniture keeps them
  <figure> ... <figcaption>                     ->  **Figure:** caption + its text
  HTML entities (&amp; ...)                     ->  plain characters
  #, ##, ### headings                           ->  kept as ADI produced them
  (top of file)                                 ->  YAML front matter: title, source, pages

Usage
  python adi_markdown_converter.py input.md
  python adi_markdown_converter.py input.md --out structured.md --chunks --rows-per-chunk 50
  python adi_markdown_converter.py result.json            # uses analyzeResult.content

As a library
  from adi_markdown_converter import convert, chunk_by_headings
  md = convert(adi_markdown, source="acord125.pdf")
  chunks = chunk_by_headings(md, rows_per_chunk=50)
"""
from __future__ import annotations

import argparse
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

CHECKED = re.compile(r"☒|:selected:")
UNCHECKED = re.compile(r"☐|:unselected:")
FURNITURE = re.compile(r"^\s*<!--\s*Page(Header|Footer|Number)\s*=.*?-->\s*$")
PAGEBREAK = re.compile(r"^\s*<!--\s*PageBreak\s*-->\s*$")


# --------------------------------------------------------------------------- text helpers
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


# --------------------------------------------------------------------------- HTML tables
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
            self._cell = {"text": [], "header": tag == "th",
                          "rowspan": int(a.get("rowspan") or 1),
                          "colspan": int(a.get("colspan") or 1)}
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
    p = _TableParser()
    p.feed(table_html)
    if not p.rows:
        return ""

    # place cells on a grid, honouring rowspan / colspan
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


# --------------------------------------------------------------------------- figures
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


# --------------------------------------------------------------------------- main converter
def convert(adi_md: str, source: str | None = None, page_rule: bool = False,
            keep_furniture: bool = False, bullets_for_checkboxes: bool = True) -> str:
    # 1. pull out tables and figures first (they can span many lines)
    blocks: dict[str, str] = {}

    def stash(kind, fn):
        def _sub(m):
            key = f"\x00{kind}{len(blocks)}\x00"
            blocks[key] = fn(m.group(0))
            return f"\n{key}\n"
        return _sub

    text = re.sub(r"<table\b.*?</table>", stash("T", table_to_pipe), adi_md, flags=re.S | re.I)
    text = re.sub(r"<figure\b.*?</figure>", stash("F", figure_to_md), text, flags=re.S | re.I)

    # 2. line pass
    out, page, title = [f"<!-- page 1 -->", ""], 1, None
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
    fm = ["---"]
    if title:
        fm.append(f"title: {json.dumps(title)}")
    if source:
        fm.append(f"source: {json.dumps(source)}")
    fm.append(f"pages: {page}")
    if any(has_marks(x) for x in out):
        fm.append('checkbox_legend: "[x] = selected, [ ] = not selected"')
    fm += ["---", ""]

    md = "\n".join(fm + out)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


# --------------------------------------------------------------------------- chunking
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


# --------------------------------------------------------------------------- CLI
def _read_input(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        data = data.get("analyzeResult", data)
        if data.get("contentFormat", "markdown") != "markdown":
            print("warning: result was not produced with outputContentFormat=markdown")
        return data["content"]
    return raw


def main():
    ap = argparse.ArgumentParser(description="ADI Markdown -> structured Markdown")
    ap.add_argument("input", help="ADI Markdown file (.md) or saved result (.json)")
    ap.add_argument("--out", help="output path (default: <input>.structured.md)")
    ap.add_argument("--source", help="source document name for front matter")
    ap.add_argument("--chunks", action="store_true", help="also write <out>.chunks.jsonl")
    ap.add_argument("--rows-per-chunk", type=int, default=0,
                    help="split big tables into chunks of N rows (header repeated)")
    ap.add_argument("--page-rule", action="store_true", help="add ----- at each page break")
    ap.add_argument("--keep-furniture", action="store_true",
                    help="keep PageHeader/PageFooter/PageNumber comments")
    a = ap.parse_args()

    src = Path(a.input)
    md = convert(_read_input(src), source=a.source or src.name,
                 page_rule=a.page_rule, keep_furniture=a.keep_furniture)
    out = Path(a.out) if a.out else src.with_name(src.stem + ".structured.md")
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")

    if a.chunks:
        cp = out.with_name(out.stem + ".chunks.jsonl")
        with cp.open("w", encoding="utf-8") as f:
            for c in chunk_by_headings(md, rows_per_chunk=a.rows_per_chunk):
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"wrote {cp}")


if __name__ == "__main__":
    main()
