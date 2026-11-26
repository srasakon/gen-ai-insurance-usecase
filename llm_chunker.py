"""Context-aware chunker for insurance financial documents (LangChain v1 + OpenAI GPT-5.4).

Inputs are plain variables in this file:
    SYSTEM_PROMPT       chunking rules + one worked example
    FINANCIAL_DOCUMENT  one financial document (markdown, with <!-- page N --> markers)

The LLM does the heavy lifting: chunk boundaries (line ranges), header breadcrumb, section,
scale, currency, reinsurance basis, periods, and evidence lines. It never returns chunk text.
Code only guards:
    1. cut chunk text from the ORIGINAL document (the LLM cannot change a number)
    2. validate coverage / overlaps / split tables -> send errors back for repair (max 2)
       -> deterministic header fallback if still invalid (report.degraded = True)
    3. evidence checks: unsupported scale / currency / basis / period date -> unknown + warning
    4. compute pages, inherited scale/currency, is_latest, is_full_year, safe splits

Prompt layout (what OpenAI caches):
    system message = SYSTEM_PROMPT + <document>numbered FINANCIAL_DOCUMENT</document>
                     -> stable prefix: CACHED (prompt_cache_key per document)
    user message   = task (line count, table spans, MAX_CHUNK_LINES); repair turns append
                     (assistant: previous JSON, user: errors) after it

Usage
-----
    export OPENAI_API_KEY=sk-...
    python context_chunker_runner.py        # writes results/chunks.jsonl, results/report.json

Tests (no network):
    pytest context_chunker_runner.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

log = logging.getLogger("chunker")  # never log document content at INFO or above

PROMPT_VERSION = "2026-11-26.1"


# =========================================================================== #
# Inputs (hardcoded variables)
# =========================================================================== #
SYSTEM_PROMPT = """\
You are a document-structure analyst for insurance financial documents (audited financial
statements, statutory filings, interim reports, loss runs, Schedule P exhibits, MD&A).

The document is below, between the <document> tags. Every line starts with its line number,
for example `L0007 `. The line number is not part of the document. A line like
`<!-- page 3 -->` is a page marker.

Your task: split the document into chunks and describe each chunk with structured metadata.
Give line ranges and metadata only. Do not give chunk text. Do not copy numbers.

BOUNDARY RULES
1. Cover every line exactly once. Give the chunks in document order. Chunk 1 starts at line 1.
   Each chunk starts at the previous end_line + 1. The last chunk ends at the last line.
   Page markers and blank lines also belong to a chunk.
2. Do not cut inside a table. The user message lists the table spans. A boundary can be before
   the first line of a span or after its last line. It cannot be between them. A table that
   continues after a page marker is one span.
3. Start a chunk at a header, or at a page marker or blank line just before a header. Do not
   start a chunk in the middle of a section. Keep a header with the content below it.
4. Keep a chunk to about MAX_CHUNK_LINES lines or fewer. Go over only for one table that you
   cannot split. Prefer one chunk for each statement, schedule or note.
5. If there is content before the first header (entity name, policy number, carrier,
   valuation date, "amounts in thousands" note), put that content in its own chunk with
   section "preamble".
6. If a parent header is directly followed by a child header, keep the parent header in the
   chunk of the first child, and include it in header_path.

METADATA RULES
7. header_path: the headers above and at the chunk start, outermost first, joined by " > ".
   Include parent headers that have no content of their own. Write the header text without
   `#` marks. For the preamble chunk, write "Preamble".
8. Use only the allowed enum values. If you are not sure, use "unknown" or null. A missing
   value is much better than a wrong value.
9. document: facts for the whole document. Set default_scale only if the document states it
   (for example "in thousands of Canadian dollars"), and give that line in
   default_scale_evidence_line. default_currency is an ISO 4217 code (USD, CAD, EUR, GBP).
10. For each chunk:
    - scale and currency: what applies to the numbers in this chunk.
      scale_source = "explicit_in_chunk" if the chunk states it,
      "inherited_from_doc" if an earlier note for the whole document states it,
      "inferred" only if strong evidence shows it, otherwise "unknown".
    - reinsurance_basis: "gross", "ceded", "assumed" or "net" if all figures in the chunk
      are on one basis. "mixed" if the chunk shows more than one (for example Gross / Ceded /
      Net columns). "not_applicable" for text without amounts (auditor report, preamble,
      accounting policies). Otherwise "unknown".
    - periods: one entry for each period COLUMN in the chunk's tables, or the period that a
      text chunk describes. Copy column_label exactly. Give facts only:
      period_end (YYYY-MM-DD), period_months (12, 9, 6, 3; null for a balance-sheet date),
      period_type, audited (true, false or null), restated.
      period_type: "annual" (12-month fiscal year), "interim_ytd" (for example nine months
      ended), "quarter", "ttm", "budget", "projection", "pro_forma", "as_of_date" (a
      balance-sheet or valuation date), "unknown".
      Do not say which period is the latest or a full year. Code calculates that.
      Each column_label must be unique in a chunk.
    - evidence: a list of {field, line} for each of section, scale, currency and
      reinsurance_basis that is not unknown, null or not_applicable. The line must show the
      fact as text (for example the line with "in thousands", "Net" or "CAD").
      For each period, give in evidence_line the line that shows its column label or date.

WORKED EXAMPLE
Input:
L0001 ACME INSURANCE LTD.
L0002 Amounts in thousands of Canadian dollars
L0003
L0004 <!-- page 1 -->
L0005 # Financial Statements
L0006 ## Statement of Income
L0007 | Item | 2025 | 2024 |
L0008 |---|---|---|
L0009 | Net premiums earned | 1,200 | 1,100 |
Output:
{"document": {"entity_name": "ACME INSURANCE LTD.", "document_type": "financial statements",
  "basis": "unknown", "default_currency": "CAD", "default_scale": "thousands",
  "default_scale_evidence_line": 2, "fiscal_year_end": null},
 "chunks": [
  {"start_line": 1, "end_line": 3, "header_path": "Preamble", "section": "preamble",
   "reinsurance_basis": "not_applicable", "currency": "CAD", "scale": "thousands",
   "scale_source": "explicit_in_chunk", "periods": [],
   "evidence": [{"field": "scale", "line": 2}, {"field": "currency", "line": 2}]},
  {"start_line": 4, "end_line": 9,
   "header_path": "Financial Statements > Statement of Income", "section": "income_statement",
   "reinsurance_basis": "net", "currency": "CAD", "scale": "thousands",
   "scale_source": "inherited_from_doc",
   "periods": [
     {"column_label": "2025", "period_end": null, "period_months": 12, "period_type": "annual",
      "audited": null, "restated": false, "evidence_line": 7},
     {"column_label": "2024", "period_end": null, "period_months": 12, "period_type": "annual",
      "audited": null, "restated": false, "evidence_line": 7}],
   "evidence": [{"field": "section", "line": 6}, {"field": "reinsurance_basis", "line": 9},
                {"field": "scale", "line": 2}, {"field": "currency", "line": 2}]}]}
(period_end is null because the example does not state the fiscal year-end date.)
"""

FINANCIAL_DOCUMENT_NAME = "northwind_mutual_fy2025"
FINANCIAL_DOCUMENT = """\
NORTHWIND MUTUAL INSURANCE COMPANY (synthetic example)
Audited Financial Statements
For the years ended December 31, 2025 and 2024
(in thousands of Canadian dollars)

<!-- page 1 -->
# Independent Auditor's Report

To the Board of Directors of Northwind Mutual Insurance Company.
We have audited the financial statements, which comprise the statement of financial
position as at December 31, 2025 and the statements of income for the year then ended.
In our opinion, the financial statements present fairly, in all material respects, the
financial position of the Company in accordance with IFRS.

<!-- page 2 -->
# Financial Statements
## Statement of Income
### Underwriting Results

| Item | Gross 2025 | Ceded 2025 | Net 2025 | Net 2024 |
|---|---|---|---|---|
| Premiums written | 52,400 | 8,100 | 44,300 | 41,050 |
| Premiums earned | 50,900 | 7,800 | 43,100 | 40,200 |
| Claims incurred | 31,200 | 4,900 | 26,300 | 25,750 |
| Underwriting income | 4,150 | 610 | 3,540 | 2,980 |

## Statement of Financial Position

| Item | Dec 31, 2025 | Dec 31, 2024 |
|---|---|---|
| Cash and investments | 188,400 | 171,900 |
| Unpaid claims (net) | 96,300 | 90,100 |
| Total equity | 61,700 | 55,200 |

<!-- page 3 -->
# Loss Run

Valuation date: March 31, 2026. Amounts in Canadian dollars (units).

| Accident Year | # of claims | Paid | Reserve | Incurred |
|---|---|---|---|---|
| 2016 | 12 | 410,000 | 0 | 410,000 |
| 2017 | 9 | 288,500 | 0 | 288,500 |
| 2018 | 14 | 512,300 | 4,000 | 516,300 |
| 2019 | 11 | 377,900 | 9,500 | 387,400 |
| 2020 | 7 | 199,000 | 22,000 | 221,000 |

<!-- page 4 -->

| 2021 | 10 | 305,600 | 41,000 | 346,600 |
| 2022 | 13 | 288,000 | 97,500 | 385,500 |
| 2023 | 15 | 201,400 | 188,000 | 389,400 |
| 2024 | 8 | 96,000 | 210,300 | 306,300 |
| 2025 | 6 | 22,500 | 251,000 | 273,500 |

# Management Outlook

| Item | 2025 Actual | 2026 Budget |
|---|---|---|
| Net premiums earned | 43,100 | 46,000 |
| Combined ratio | 94.1% | 93.0% |
"""

# The run iterates over these. Add entries to chunk more documents.
DOCUMENTS: dict[str, str] = {FINANCIAL_DOCUMENT_NAME: FINANCIAL_DOCUMENT}

OUT_DIR = Path(__file__).resolve().parent / "results"


# =========================================================================== #
# Settings
# =========================================================================== #
@dataclass(frozen=True)
class Settings:
    model: str = "gpt-5.4"
    reasoning_effort: str = "low"
    cache_retention: str = "24h"
    max_chunk_lines: int = 60
    page_marker_regex: str = r"^\s*<!--\s*page\s+(\d+)\s*-->\s*$"
    max_repairs: int = 2             # repair turns after a response fails validation
    max_retries: int = 3             # transport retries per call (ModelRetryMiddleware)
    timeout: float = 300.0
    doc_concurrency: int = 3
    response_cache_dir: Optional[str] = str(Path(__file__).resolve().parent / ".chunker_cache")


# =========================================================================== #
# Schema
#   LLM*  = what the LLM returns. Every field required (nullable where needed) so the
#           schema is valid for OpenAI strict structured output. No text, no derived flags.
#   Chunk = what we emit. Code fills text, pages and derived fields.
# =========================================================================== #
Section = Literal[
    "preamble", "auditor_report", "income_statement", "balance_sheet",
    "cash_flow", "equity", "comprehensive_income", "notes", "mdna",
    "loss_run", "schedule_p", "underwriting_exhibit", "investment_schedule",
    "reinsurance_schedule", "other",
]
Scale = Literal["units", "thousands", "millions", "billions", "unknown"]
Basis = Literal["gaap", "ifrs", "statutory", "unknown"]
ReinsBasis = Literal["gross", "ceded", "assumed", "net", "mixed", "not_applicable", "unknown"]
PeriodType = Literal[
    "annual", "interim_ytd", "quarter", "ttm", "budget",
    "projection", "pro_forma", "as_of_date", "unknown",
]
ScaleSource = Literal["explicit_in_chunk", "inherited_from_doc", "inferred", "unknown"]
NON_ACTUAL = frozenset({"budget", "projection", "pro_forma"})


class _LLMBase(BaseModel):
    model_config = ConfigDict(extra="ignore")  # stray keys ("text", "is_latest", ...) are dropped


class LLMPeriod(_LLMBase):
    column_label: str = Field(description="Column header exactly as printed, e.g. '2025' or '9M Sep-25'.")
    period_end: Optional[str] = Field(description="YYYY-MM-DD, or null if the date is not stated.")
    period_months: Optional[int] = Field(description="12, 9, 6, 3; null for a point-in-time date.")
    period_type: PeriodType
    audited: Optional[bool]
    restated: bool
    evidence_line: Optional[int] = Field(description="Line number that shows this column label or date.")


class LLMEvidence(_LLMBase):
    field: Literal["section", "scale", "currency", "reinsurance_basis"]
    line: int


class LLMChunk(_LLMBase):
    start_line: int
    end_line: int
    header_path: str = Field(description="Breadcrumb 'A > B > C', including empty parent headers.")
    section: Section
    reinsurance_basis: ReinsBasis
    currency: Optional[str] = Field(description="ISO 4217 code, or null.")
    scale: Scale
    scale_source: ScaleSource
    periods: list[LLMPeriod]
    evidence: list[LLMEvidence]


class DocumentContext(_LLMBase):
    entity_name: Optional[str]
    document_type: Optional[str]
    basis: Basis
    default_currency: Optional[str] = Field(description="ISO 4217 code, or null.")
    default_scale: Scale
    default_scale_evidence_line: Optional[int]
    fiscal_year_end: Optional[str] = Field(description="'MM-DD', or null.")


class LLMResponse(_LLMBase):
    """Chunk boundaries and metadata for one document. No chunk text."""

    document: DocumentContext
    chunks: list[LLMChunk]


class Period(BaseModel):
    column_label: str
    period_end: Optional[date] = None
    period_months: Optional[int] = None
    period_type: PeriodType = "unknown"
    audited: Optional[bool] = None
    restated: bool = False
    evidence_line: Optional[int] = None
    is_latest: Optional[bool] = None       # computed in code
    is_full_year: Optional[bool] = None    # computed in code


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    start_line: int
    end_line: int
    header_path: str
    section: Section
    reinsurance_basis: ReinsBasis = "unknown"
    currency: Optional[str] = None
    scale: Scale = "unknown"
    scale_source: ScaleSource = "unknown"
    periods: list[Period] = Field(default_factory=list)
    evidence: dict[str, int] = Field(default_factory=dict)
    text: str                              # sliced from the ORIGINAL document, never from the LLM
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    contains_table: bool = False
    warnings: list[str] = Field(default_factory=list)


class RunReport(BaseModel):
    document: str
    model: str
    prompt_version: str = PROMPT_VERSION
    retries: int = 0
    degraded: bool = False
    response_cache_hit: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latest_annual_end: Optional[date] = None
    latest_interim_end: Optional[date] = None
    coverage_pct: float = 0.0
    reconstruction_ok: bool = False
    numbers_preserved: bool = False
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class ChunkResult(BaseModel):
    document: DocumentContext
    chunks: list[Chunk]
    report: RunReport


# =========================================================================== #
# Source document (thin plumbing; the input is well-formed markdown)
# Line numbers are 1-based. ``lines`` keep their line endings: "".join(lines) == markdown.
# =========================================================================== #
@dataclass
class SourceDoc:
    markdown: str
    page_marker_regex: str
    lines: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.lines = self.markdown.splitlines(keepends=True)
        self._marker = re.compile(self.page_marker_regex, re.IGNORECASE)

    @property
    def n(self) -> int:
        return len(self.lines)

    def line(self, i: int) -> str:
        return self.lines[i - 1].rstrip("\r\n")

    def slice(self, start: int, end: int) -> str:
        return "".join(self.lines[start - 1 : end])

    def numbered(self) -> str:
        width = max(4, len(str(self.n)))
        return "\n".join(f"L{i:0{width}d} {self.line(i)}" for i in range(1, self.n + 1))

    def page_marker(self, i: int) -> int | None:
        m = self._marker.match(self.line(i))
        return int(m.group(1)) if m else None

    @cached_property
    def pages(self) -> list[int | None]:
        """pages[i-1] = page of line i = the last marker at or before it (None before any)."""
        out: list[int | None] = []
        current: int | None = None
        for i in range(1, self.n + 1):
            current = self.page_marker(i) or current
            out.append(current)
        return out

    def page_range(self, start: int, end: int) -> tuple[int | None, int | None]:
        """Pages of the content lines; markers and blank lines count only if that is all."""
        content = [self.pages[i - 1] for i in range(start, end + 1)
                   if self.line(i).strip() and self.page_marker(i) is None]
        pool = [p for p in content if p is not None] or [p for p in self.pages[start - 1 : end] if p is not None]
        return (min(pool), max(pool)) if pool else (None, None)

    @cached_property
    def fence_lines(self) -> set[int]:
        inside, out = False, set()
        for i in range(1, self.n + 1):
            if self.line(i).lstrip().startswith(("```", "~~~")):
                out.add(i)
                inside = not inside
            elif inside:
                out.add(i)
        return out

    def is_table_row(self, i: int) -> bool:
        return i not in self.fence_lines and self.line(i).lstrip().startswith("|")

    def is_header(self, i: int) -> bool:
        return i not in self.fence_lines and re.match(r"^\s{0,3}#{1,6}\s", self.line(i)) is not None

    @cached_property
    def table_spans(self) -> list[tuple[int, int]]:
        """Runs of table rows. Runs separated only by page markers / blank lines (a table that
        continues on the next page) are one span."""
        runs: list[list[int]] = []
        for i in range(1, self.n + 1):
            if self.is_table_row(i):
                if runs and runs[-1][1] == i - 1:
                    runs[-1][1] = i
                else:
                    runs.append([i, i])
        merged: list[list[int]] = []
        for run in runs:
            if merged:
                gap = range(merged[-1][1] + 1, run[0])
                if any(self.page_marker(g) is not None for g in gap) and all(
                    self.page_marker(g) is not None or not self.line(g).strip() for g in gap
                ):
                    merged[-1][1] = run[1]
                    continue
            merged.append(run)
        return [(a, b) for a, b in merged]

    def tables_in(self, start: int, end: int) -> list[tuple[int, int]]:
        return [(a, b) for a, b in self.table_spans if a <= end and b >= start]


def doc_id_for(name: str, markdown: str) -> str:
    """First 16 hex chars of sha1(name + content). Stable across runs, so the cache key is too."""
    return hashlib.sha1((name + markdown).encode("utf-8")).hexdigest()[:16]


# =========================================================================== #
# Prompt layout
# =========================================================================== #
def build_system_prompt(system_prompt: str, doc: SourceDoc) -> str:
    """Stable, cacheable prefix: rules + numbered document. No timestamps or IDs."""
    return f"{system_prompt}\n<document>\n{doc.numbered()}\n</document>"


def task_message(doc: SourceDoc, max_chunk_lines: int) -> dict[str, str]:
    spans = ", ".join(f"L{a}-L{b}" for a, b in doc.table_spans) or "none"
    return {"role": "user", "content": (
        f"The document has {doc.n} lines (L1 to L{doc.n}).\n"
        f"Table spans (do not cut inside): {spans}\n"
        f"MAX_CHUNK_LINES = {max_chunk_lines}\n\n"
        "Give the document context and the full chunk list in the output format."
    )}


def repair_message(errors: list[str]) -> dict[str, str]:
    bullets = "\n".join(f"- {e}" for e in errors)
    return {"role": "user", "content": (
        f"Your previous response failed validation:\n{bullets}\n\n"
        "Give a corrected, complete response. Change only the chunks that you must change to "
        "fix these errors. Keep all other chunks and metadata the same. Cover every line exactly "
        "once, in order, and do not cut inside a table span."
    )}


# =========================================================================== #
# Model / runnable
# =========================================================================== #
def build_model(doc_id: str, settings: Settings):  # noqa: ANN201 - langchain type
    """ChatOpenAI with cache routing set on the model (invoke() does not forward it)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model,
        use_responses_api=True,
        reasoning={"effort": settings.reasoning_effort},  # same on every call -> same prefix
        max_retries=0,  # retries live in ModelRetryMiddleware
        timeout=settings.timeout,
        model_kwargs={
            "prompt_cache_key": f"chunker-{doc_id}",  # one key per document
            "prompt_cache_retention": settings.cache_retention,
        },
    )


def build_agent(system_prompt: str, doc_id: str, settings: Settings):  # noqa: ANN201
    """One runnable per document; its system prompt holds that document."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
    from langchain.agents.structured_output import ProviderStrategy

    return create_agent(
        model=build_model(doc_id, settings),
        tools=[],
        system_prompt=system_prompt,
        response_format=ProviderStrategy(LLMResponse, strict=True),
        middleware=[
            ModelRetryMiddleware(
                max_retries=settings.max_retries, initial_delay=2.0, max_delay=60.0, on_failure="error"
            ),
            ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),
        ],
    )


def usage_of(result: Any) -> tuple[int, int, int]:
    """(input, cached, output) tokens summed over every AI message in one result."""
    if not isinstance(result, dict):
        return 0, 0, 0
    total_in = total_cached = total_out = 0
    for msg in result.get("messages", []):
        if getattr(msg, "type", None) != "ai":
            continue
        usage = getattr(msg, "usage_metadata", None) or {}
        total_in += int(usage.get("input_tokens") or 0)
        total_out += int(usage.get("output_tokens") or 0)
        total_cached += int((usage.get("input_token_details") or {}).get("cache_read") or 0)
    return total_in, total_cached, total_out


async def call_llm(
    agent: Any, system_prompt: str, messages: list[dict[str, str]], settings: Settings, report: RunReport
) -> dict[str, Any]:
    """One LLM call -> the response as a plain dict. A file cache keyed by
    sha256(system + messages + model + prompt version) makes reruns deterministic."""
    cache_path: Path | None = None
    if settings.response_cache_dir:
        key = hashlib.sha256(json.dumps(
            {"s": system_prompt, "m": messages, "model": settings.model,
             "effort": settings.reasoning_effort, "pv": PROMPT_VERSION}, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        cache_path = Path(settings.response_cache_dir) / f"{key}.json"
        if cache_path.exists():
            report.response_cache_hit = True
            return json.loads(cache_path.read_text(encoding="utf-8"))

    result = await agent.ainvoke({"messages": messages})
    tin, tcached, tout = usage_of(result)
    report.input_tokens += tin
    report.cached_tokens += tcached
    report.output_tokens += tout
    parsed = result.get("structured_response") if isinstance(result, dict) else None
    if parsed is None:
        raise RuntimeError("no structured_response (refusal or model call limit reached)")
    data = parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else dict(parsed)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


# =========================================================================== #
# Validation: structural errors trigger a repair turn
# =========================================================================== #
def check_schema(data: Any) -> tuple[LLMResponse | None, list[str]]:
    try:
        return LLMResponse.model_validate(data), []
    except ValidationError as e:
        return None, [f"schema: {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:20]]


def check_ranges(chunks: list[LLMChunk], n: int) -> list[str]:
    errs = []
    for k, c in enumerate(chunks, 1):
        if c.start_line > c.end_line:
            errs.append(f"chunk {k}: start_line {c.start_line} > end_line {c.end_line}")
        if c.start_line < 1 or c.end_line > n:
            errs.append(f"chunk {k}: range L{c.start_line}-L{c.end_line} is outside L1-L{n}")
    return errs


def check_coverage(chunks: list[LLMChunk], n: int) -> list[str]:
    if not chunks:
        return ["no chunks returned"]
    errs = []
    ordered = sorted(chunks, key=lambda c: (c.start_line, c.end_line))
    if [(c.start_line, c.end_line) for c in ordered] != [(c.start_line, c.end_line) for c in chunks]:
        errs.append("chunks are not in document order")
    if ordered[0].start_line != 1:
        errs.append(f"lines L1-L{ordered[0].start_line - 1} are not covered (first chunk must start at L1)")
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.start_line > prev.end_line + 1:
            errs.append(f"gap: lines L{prev.end_line + 1}-L{cur.start_line - 1} are not covered")
        elif cur.start_line <= prev.end_line:
            errs.append(f"overlap: chunks L{prev.start_line}-L{prev.end_line} and "
                        f"L{cur.start_line}-L{cur.end_line} share lines")
    if ordered[-1].end_line != n:
        errs.append(f"last chunk ends at L{ordered[-1].end_line}; the document ends at L{n}")
    return errs


def check_tables(chunks: list[LLMChunk], doc: SourceDoc) -> list[str]:
    errs = set()
    for c in chunks:
        for a, b in doc.table_spans:
            if a < c.start_line <= b:
                errs.add(f"chunk starting at L{c.start_line} cuts table L{a}-L{b}; start at L{a} or L{b + 1}")
            if a <= c.end_line < b:
                errs.add(f"chunk ending at L{c.end_line} cuts table L{a}-L{b}; end at L{a - 1} or L{b}")
    return sorted(errs)


def structural_errors(resp: LLMResponse, doc: SourceDoc) -> list[str]:
    errs = check_ranges(resp.chunks, doc.n)
    return errs or check_coverage(resp.chunks, doc.n) + check_tables(resp.chunks, doc)


# =========================================================================== #
# Evidence checks: downgrade to unknown + warning, never a retry
# =========================================================================== #
SCALE_WORDS = {
    "thousands": ["thousand", "000s", "000's", "'000", "(000)", "$000", "in 000"],
    "millions": ["million", "mln", "(mm", "mm)", "in mm", "000,000"],
    "billions": ["billion", "bn"],
}
REINS_WORDS = {
    "gross": ["gross", "direct"],
    "net": ["net"],
    "ceded": ["ceded", "cede", "reinsurers"],
    "assumed": ["assumed", "assume"],
}
CURRENCY_WORDS = {
    "USD": ["usd", "us$", "u.s. dollar", "$"],
    "CAD": ["cad", "c$", "canadian"],
    "EUR": ["eur", "€", "euro"],
    "GBP": ["gbp", "£", "sterling", "pound"],
    "JPY": ["jpy", "¥", "yen"],
    "CHF": ["chf", "swiss franc"],
    "AUD": ["aud", "a$", "australian"],
}


def _text(doc: SourceDoc, line: int | None) -> str | None:
    return doc.line(line).lower() if line is not None and 1 <= line <= doc.n else None


def scale_ok(doc: SourceDoc, scale: str, line: int | None) -> bool:
    if scale in ("unknown", "units"):
        return True
    t = _text(doc, line)
    return t is not None and any(w in t for w in SCALE_WORDS[scale])


def currency_ok(doc: SourceDoc, ccy: str | None, line: int | None) -> bool:
    if not ccy:
        return True
    t = _text(doc, line)
    return t is not None and any(w in t for w in CURRENCY_WORDS.get(ccy.upper(), [ccy.lower()]))


def reins_ok(doc: SourceDoc, basis: str, line: int | None) -> bool:
    if basis in ("unknown", "not_applicable"):
        return True
    t = _text(doc, line)
    if t is None:
        return False
    hits = {b for b, ws in REINS_WORDS.items() if any(re.search(rf"\b{re.escape(w)}", t) for w in ws)}
    return len(hits) >= 2 if basis == "mixed" else basis in hits


def check_document_evidence(ctx: DocumentContext, doc: SourceDoc) -> list[str]:
    if scale_ok(doc, ctx.default_scale, ctx.default_scale_evidence_line):
        return []
    msg = (f"document default_scale '{ctx.default_scale}' not supported by line "
           f"{ctx.default_scale_evidence_line}; set to unknown")
    ctx.default_scale, ctx.default_scale_evidence_line = "unknown", None
    return [msg]


def check_chunk_evidence(chunk: Chunk, doc: SourceDoc, ctx: DocumentContext) -> None:
    ev = chunk.evidence
    for name, line in list(ev.items()):
        if not 1 <= line <= doc.n:
            chunk.warnings.append(f"evidence for {name} points outside the document (L{line}); dropped")
            del ev[name]

    if chunk.scale not in ("unknown", "units") and not scale_ok(doc, chunk.scale, ev.get("scale")):
        if ctx.default_scale == chunk.scale and ctx.default_scale_evidence_line is not None:
            chunk.scale_source = "inherited_from_doc"
            ev["scale"] = ctx.default_scale_evidence_line
        else:
            chunk.warnings.append(f"scale '{chunk.scale}' not supported by evidence line {ev.get('scale')}; set to unknown")
            chunk.scale, chunk.scale_source = "unknown", "unknown"
            ev.pop("scale", None)

    if chunk.currency and not currency_ok(doc, chunk.currency, ev.get("currency")):
        if not (ctx.default_currency and ctx.default_currency.upper() == chunk.currency.upper()):
            chunk.warnings.append(f"currency '{chunk.currency}' not supported by evidence line {ev.get('currency')}; set to null")
            chunk.currency = None
        ev.pop("currency", None)

    if not reins_ok(doc, chunk.reinsurance_basis, ev.get("reinsurance_basis")):
        chunk.warnings.append(f"reinsurance_basis '{chunk.reinsurance_basis}' not supported by evidence line "
                              f"{ev.get('reinsurance_basis')}; set to unknown")
        chunk.reinsurance_basis = "unknown"
        ev.pop("reinsurance_basis", None)

    for p in chunk.periods:
        if p.evidence_line is None:
            continue
        t = _text(doc, p.evidence_line)
        if t is None:
            chunk.warnings.append(f"period '{p.column_label}': evidence line {p.evidence_line} out of range; dropped")
            p.evidence_line = None
            continue
        norm = re.sub(r"\s+", " ", t)
        label_ok = re.sub(r"\s+", " ", p.column_label.lower()).strip() in norm
        year_ok = p.period_end is not None and str(p.period_end.year) in norm
        if not (label_ok or year_ok):
            chunk.warnings.append(f"period '{p.column_label}': evidence line {p.evidence_line} shows neither "
                                  "the label nor the year; period_end set to null")
            p.period_end = None

    seen: set[str] = set()
    unique = []
    for p in chunk.periods:
        if p.column_label in seen:
            chunk.warnings.append(f"duplicate period column_label '{p.column_label}' removed")
        else:
            seen.add(p.column_label)
            unique.append(p)
    chunk.periods = unique


# =========================================================================== #
# Fallback: split at ATX headers, tables whole, parent header kept with child
# =========================================================================== #
def fallback_chunks(doc: SourceDoc) -> list[LLMChunk]:
    if doc.n == 0:
        return []

    def only_headers(s: int, e: int) -> bool:
        return all(doc.is_header(i) or not doc.line(i).strip() or doc.page_marker(i) is not None
                   for i in range(s, e))

    starts = [1]
    for i in range(2, doc.n + 1):
        if doc.is_header(i) and not any(a < i <= b for a, b in doc.table_spans):
            if only_headers(starts[-1], i) and any(doc.is_header(j) for j in range(starts[-1], i)):
                continue  # parent header directly followed by child header: same chunk
            starts.append(i)
    bounds = list(zip(starts, [s - 1 for s in starts[1:]] + [doc.n]))

    stack: list[tuple[int, str]] = []
    out = []
    for s, e in bounds:
        for i in range(s, e + 1):
            if doc.is_header(i):
                raw = doc.line(i).lstrip()
                level = len(raw) - len(raw.lstrip("#"))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, raw.lstrip("#").strip()))
            elif doc.line(i).strip() and doc.page_marker(i) is None:
                break
        path = " > ".join(t for _, t in stack)
        out.append(LLMChunk(
            start_line=s, end_line=e, header_path=path or "Preamble",
            section="preamble" if not path else "other", reinsurance_basis="unknown",
            currency=None, scale="unknown", scale_source="unknown", periods=[], evidence=[],
        ))
    return out


# =========================================================================== #
# Post-processing
# =========================================================================== #
def to_chunk(c: LLMChunk, doc: SourceDoc, doc_id: str) -> Chunk:
    warnings: list[str] = []
    periods = []
    for p in c.periods:
        end: date | None = None
        if p.period_end:
            try:
                end = date.fromisoformat(p.period_end)
            except ValueError:
                warnings.append(f"period '{p.column_label}': period_end '{p.period_end}' is not YYYY-MM-DD; set to null")
        periods.append(Period(
            column_label=p.column_label, period_end=end, period_months=p.period_months,
            period_type=p.period_type, audited=p.audited, restated=p.restated,
            evidence_line=p.evidence_line,
        ))
    return Chunk(
        chunk_id=f"{doc_id}:{c.start_line}-{c.end_line}", doc_id=doc_id,
        start_line=c.start_line, end_line=c.end_line, header_path=c.header_path,
        section=c.section, reinsurance_basis=c.reinsurance_basis, currency=c.currency,
        scale=c.scale, scale_source=c.scale_source, periods=periods,
        evidence={e.field: e.line for e in c.evidence},
        text=doc.slice(c.start_line, c.end_line),  # from the source, never from the LLM
        warnings=warnings,
    )


def inherit_defaults(chunk: Chunk, ctx: DocumentContext, doc: SourceDoc) -> None:
    has_table = bool(doc.tables_in(chunk.start_line, chunk.end_line))
    if chunk.reinsurance_basis == "not_applicable" and not has_table:
        return  # text without amounts gets no scale / currency
    if chunk.scale == "unknown" and ctx.default_scale != "unknown" and chunk.section != "preamble":
        chunk.scale, chunk.scale_source = ctx.default_scale, "inherited_from_doc"
        if ctx.default_scale_evidence_line is not None:
            chunk.evidence.setdefault("scale", ctx.default_scale_evidence_line)
    if not chunk.currency and ctx.default_currency:
        chunk.currency = ctx.default_currency
    if chunk.currency:
        chunk.currency = chunk.currency.upper()


def full_year(p: Period) -> bool | None:
    if p.period_months == 12 and p.period_type in ("annual", "ttm"):
        return True
    if p.period_months is not None and p.period_months < 12:
        return False
    return None


def _period_group(c: Chunk, p: Period) -> str:
    """annual / interim / point:<section>. Point-in-time dates compare only within a section,
    so a loss-run valuation date does not make the balance-sheet year end 'not latest'."""
    if p.period_type == "as_of_date" or (p.period_months is None and p.period_type == "unknown"):
        return f"point:{c.section}"
    if p.period_type in ("interim_ytd", "quarter") or (p.period_months is not None and p.period_months < 12):
        return "interim"
    return "annual"


def compute_period_flags(chunks: list[Chunk]) -> dict[str, date | None]:
    """Sets is_full_year and is_latest on every period (LLM values never survive).
    Latest = max period_end among ACTUAL periods of the same group, across the document."""
    pairs = [(_period_group(c, p), p) for c in chunks for p in c.periods]
    actual = [(g, p) for g, p in pairs if p.period_type not in NON_ACTUAL and p.period_end is not None]
    latest: dict[str, date | None] = {"annual": None, "interim": None}
    for g, p in actual:
        cur = latest.get(g)
        latest[g] = p.period_end if cur is None or (p.period_end and p.period_end > cur) else cur
    for g, p in pairs:
        p.is_full_year = full_year(p)
        is_actual = p.period_type not in NON_ACTUAL and p.period_end is not None
        p.is_latest = (p.period_end == latest.get(g)) if is_actual else None
    return latest


def _clone(chunk: Chunk, start: int, end: int, doc: SourceDoc, suffix: str, text: str | None = None) -> Chunk:
    c = chunk.model_copy(deep=True)
    c.start_line, c.end_line = start, end
    c.chunk_id = f"{c.doc_id}:{start}-{end}"
    c.text = doc.slice(start, end) if text is None else text
    if suffix:
        c.header_path = f"{chunk.header_path} > {suffix}"
    return c


def _split_table(chunk: Chunk, doc: SourceDoc, max_lines: int) -> list[Chunk]:
    """One table too big on its own: split by rows, repeat header row + separator on later parts."""
    a, b = doc.tables_in(chunk.start_line, chunk.end_line)[0]
    has_header = a + 1 <= b and set(doc.line(a + 1).replace("|", "").strip()) <= set("-: ")
    header = doc.slice(a, a + 1) if has_header else ""
    rows_per = max(1, max_lines - 2)
    pieces, s, i = [], chunk.start_line, (a + 2 if has_header else a)
    while i <= b:
        e = min(i + rows_per - 1, b)
        e = chunk.end_line if e == b else e
        pieces.append((s, e))
        s = i = e + 1
    if len(pieces) == 1:
        chunk.warnings.append(f"chunk has {chunk.end_line - chunk.start_line + 1} lines (> {max_lines}); table kept whole")
        return [chunk]
    out = []
    for k, (s, e) in enumerate(pieces, 1):
        prefix = header if k > 1 else ""
        c = _clone(chunk, s, e, doc, f"part {k}/{len(pieces)}", prefix + doc.slice(s, e))
        c.warnings.append(f"oversized table split by rows (part {k}/{len(pieces)})"
                          + (f"; header row L{a}-L{a + 1} repeated at top of text" if prefix else ""))
        out.append(c)
    return out


def split_oversized(chunk: Chunk, doc: SourceDoc, max_lines: int) -> list[Chunk]:
    """Safety net (the LLM already sizes chunks): split at sub-headers / blank lines between
    blocks, never inside a table; a single oversized table is split by rows."""
    length = chunk.end_line - chunk.start_line + 1
    if length <= max_lines:
        return [chunk]
    tables = doc.tables_in(chunk.start_line, chunk.end_line)
    pts = [i for i in range(chunk.start_line + 1, chunk.end_line + 1)
           if not any(a < i <= b for a, b in tables)
           and (doc.is_header(i) or (not doc.line(i - 1).strip() and doc.line(i).strip()))]
    bounds, s = [], chunk.start_line
    while pts and chunk.end_line - s + 1 > max_lines:
        cands = [p for p in pts if s < p <= s + max_lines]
        cut = max(cands) if cands else min((p for p in pts if p > s), default=None)
        if cut is None:
            break
        bounds.append((s, cut - 1))
        s = cut
    bounds.append((s, chunk.end_line))
    if len(bounds) > 1:
        out = []
        for k, (s, e) in enumerate(bounds, 1):
            suffix = "" if k == 1 else (doc.line(s).lstrip("# ").strip() if doc.is_header(s) else f"part {k}")
            c = _clone(chunk, s, e, doc, suffix)
            c.warnings.append(f"oversized chunk ({length} lines) split at a block boundary")
            out.extend(split_oversized(c, doc, max_lines))
        return out
    if len(tables) == 1:
        return _split_table(chunk, doc, max_lines)
    chunk.warnings.append(f"chunk has {length} lines (> {max_lines}) and no safe split point")
    return [chunk]


# =========================================================================== #
# Orchestration
# =========================================================================== #
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def empty_context() -> DocumentContext:
    return DocumentContext(entity_name=None, document_type=None, basis="unknown", default_currency=None,
                           default_scale="unknown", default_scale_evidence_line=None, fiscal_year_end=None)


async def ask_with_repair(doc: SourceDoc, doc_id: str, settings: Settings, report: RunReport) -> LLMResponse | None:
    """First call + up to settings.max_repairs repair turns. None -> caller uses the fallback."""
    system_prompt = build_system_prompt(SYSTEM_PROMPT, doc)
    agent = build_agent(system_prompt, doc_id, settings)
    messages = [task_message(doc, settings.max_chunk_lines)]
    for attempt in range(settings.max_repairs + 1):
        try:
            data: Any = await call_llm(agent, system_prompt, messages, settings, report)
        except Exception as exc:  # noqa: BLE001 - includes StructuredOutputValidationError
            data, errors = None, [f"response could not be parsed: {type(exc).__name__}: {exc}"]
        else:
            resp, errors = check_schema(data)
            if resp is not None:
                errors = structural_errors(resp, doc)
                if not errors:
                    report.validation_errors = []
                    return resp
        report.validation_errors = errors
        log.info("%s: attempt %d failed validation (%d error(s))", doc_id, attempt + 1, len(errors))
        for e in errors:
            log.debug("%s: %s", doc_id, e)
        if attempt == settings.max_repairs:
            return None
        report.retries += 1
        if data is not None:
            messages = messages + [{"role": "assistant", "content": json.dumps(data)}]
        messages = messages + [repair_message(errors)]
    return None


async def chunk_document(label: str, markdown: str, settings: Settings) -> ChunkResult:
    started = time.perf_counter()
    doc = SourceDoc(markdown, settings.page_marker_regex)
    doc_id = doc_id_for(label, markdown)
    report = RunReport(document=label, model=settings.model)

    resp = await ask_with_repair(doc, doc_id, settings, report) if doc.n else None
    if resp is None:
        report.degraded = doc.n > 0
        if report.degraded:
            report.warnings.append(f"LLM output failed validation after {report.retries} repair(s); "
                                   "used the header fallback")
        llm_chunks, ctx = fallback_chunks(doc), empty_context()
    else:
        llm_chunks, ctx = resp.chunks, resp.document

    report.warnings += check_document_evidence(ctx, doc)
    chunks: list[Chunk] = []
    for lc in llm_chunks:
        c = to_chunk(lc, doc, doc_id)
        check_chunk_evidence(c, doc, ctx)
        inherit_defaults(c, ctx, doc)
        chunks.extend(split_oversized(c, doc, settings.max_chunk_lines))
    for c in chunks:
        c.page_start, c.page_end = doc.page_range(c.start_line, c.end_line)
        c.contains_table = bool(doc.tables_in(c.start_line, c.end_line))
        report.warnings += [f"{c.chunk_id}: {w}" for w in c.warnings]

    latest = compute_period_flags(chunks)
    report.latest_annual_end, report.latest_interim_end = latest["annual"], latest["interim"]

    covered = sum(c.end_line - c.start_line + 1 for c in chunks)
    report.coverage_pct = round(100.0 * covered / doc.n, 2) if doc.n else 100.0
    report.reconstruction_ok = "".join(doc.slice(c.start_line, c.end_line) for c in chunks) == markdown
    # split tables repeat their header row on purpose: count numbers on the source slice for those
    emitted = "".join(doc.slice(c.start_line, c.end_line) if any("repeated" in w for w in c.warnings) else c.text
                      for c in chunks)
    report.numbers_preserved = Counter(NUMBER.findall(emitted)) == Counter(NUMBER.findall(markdown))
    report.latency_s = round(time.perf_counter() - started, 3)
    log.info("%s: %d chunks | repairs=%d degraded=%s warnings=%d | %.1fs | input=%d cached=%d output=%d",
             label, len(chunks), report.retries, report.degraded, len(report.warnings), report.latency_s,
             report.input_tokens, report.cached_tokens, report.output_tokens)
    return ChunkResult(document=ctx, chunks=chunks, report=report)


def write_outputs(results: list[ChunkResult], out_dir: str | Path) -> None:
    """chunks.jsonl (one chunk per line) and report.json (document context + run report per doc)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            for c in r.chunks:
                fh.write(json.dumps({"document": r.report.document, **c.model_dump(mode="json")},
                                    ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(
        [{"document": r.document.model_dump(mode="json"), "report": r.report.model_dump(mode="json")}
         for r in results], indent=2), encoding="utf-8")


async def run(settings: Settings, out_dir: str | Path) -> list[ChunkResult]:
    """Chunk every document in DOCUMENTS, write outputs. One bad document does not stop the run."""
    semaphore = asyncio.Semaphore(settings.doc_concurrency)

    async def one(label: str, md: str) -> ChunkResult | None:
        async with semaphore:
            try:
                return await chunk_document(label, md, settings)
            except Exception as exc:  # noqa: BLE001
                log.error("%s failed: %s: %s", label, type(exc).__name__, exc)
                return None

    results = [r for r in await asyncio.gather(*(one(k, v) for k, v in DOCUMENTS.items())) if r]
    write_outputs(results, out_dir)
    log.info("wrote %s", Path(out_dir).resolve())
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY is not set")
        return 2
    asyncio.run(run(Settings(), OUT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())


# =========================================================================== #
# Tests (no network): pytest context_chunker_runner.py
# =========================================================================== #
def _doc() -> SourceDoc:
    return SourceDoc(FINANCIAL_DOCUMENT, Settings().page_marker_regex)


def _P(label: str, end: str | None, months: int | None, ptype: str, line: int) -> dict[str, Any]:
    return {"column_label": label, "period_end": end, "period_months": months, "period_type": ptype,
            "audited": True, "restated": False, "evidence_line": line}


def _good_response() -> dict[str, Any]:
    """A correct response for FINANCIAL_DOCUMENT (61 lines; tables L20-25, L29-33, L40-54, L58-61)."""
    return {
        "document": {"entity_name": "NORTHWIND MUTUAL INSURANCE COMPANY", "document_type": "audited financial statements",
                     "basis": "ifrs", "default_currency": "CAD", "default_scale": "thousands",
                     "default_scale_evidence_line": 4, "fiscal_year_end": "12-31"},
        "chunks": [
            {"start_line": 1, "end_line": 5, "header_path": "Preamble", "section": "preamble",
             "reinsurance_basis": "not_applicable", "currency": "CAD", "scale": "thousands",
             "scale_source": "explicit_in_chunk", "periods": [],
             "evidence": [{"field": "scale", "line": 4}, {"field": "currency", "line": 4}]},
            {"start_line": 6, "end_line": 14, "header_path": "Independent Auditor's Report",
             "section": "auditor_report", "reinsurance_basis": "not_applicable", "currency": None,
             "scale": "unknown", "scale_source": "unknown", "periods": [],
             "evidence": [{"field": "section", "line": 7}]},
            {"start_line": 15, "end_line": 26,
             "header_path": "Financial Statements > Statement of Income > Underwriting Results",
             "section": "income_statement", "reinsurance_basis": "mixed", "currency": "CAD",
             "scale": "thousands", "scale_source": "inherited_from_doc",
             "periods": [_P("Gross 2025", "2025-12-31", 12, "annual", 20), _P("Ceded 2025", "2025-12-31", 12, "annual", 20),
                         _P("Net 2025", "2025-12-31", 12, "annual", 20), _P("Net 2024", "2024-12-31", 12, "annual", 20)],
             "evidence": [{"field": "section", "line": 17}, {"field": "reinsurance_basis", "line": 20},
                          {"field": "scale", "line": 4}]},
            {"start_line": 27, "end_line": 34, "header_path": "Financial Statements > Statement of Financial Position",
             "section": "balance_sheet", "reinsurance_basis": "unknown", "currency": "CAD", "scale": "thousands",
             "scale_source": "inherited_from_doc",
             "periods": [_P("Dec 31, 2025", "2025-12-31", None, "as_of_date", 29),
                         _P("Dec 31, 2024", "2024-12-31", None, "as_of_date", 29)],
             "evidence": [{"field": "section", "line": 27}]},
            {"start_line": 35, "end_line": 55, "header_path": "Loss Run", "section": "loss_run",
             "reinsurance_basis": "unknown", "currency": "CAD", "scale": "units", "scale_source": "explicit_in_chunk",
             "periods": [_P("Valuation date", "2026-03-31", None, "as_of_date", 38)],
             "evidence": [{"field": "section", "line": 36}, {"field": "currency", "line": 38}]},
            {"start_line": 56, "end_line": 61, "header_path": "Management Outlook", "section": "mdna",
             "reinsurance_basis": "net", "currency": "CAD", "scale": "thousands", "scale_source": "inherited_from_doc",
             "periods": [_P("2025 Actual", "2025-12-31", 12, "annual", 58), _P("2026 Budget", "2026-12-31", 12, "budget", 58)],
             "evidence": [{"field": "reinsurance_basis", "line": 60}]},
        ],
    }


class _FakeAgent:
    """Stands in for the compiled runnable: returns canned responses in order.
    A dict becomes structured_response; an Exception is raised (e.g. unparseable output)."""

    def __init__(self, responses: list[Any], calls: list[Any]) -> None:
        self.responses, self.calls = responses, calls

    async def ainvoke(self, inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
        from langchain_core.messages import AIMessage

        self.calls.append(inp["messages"])
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        usage = {"input_tokens": 3000, "output_tokens": 500, "total_tokens": 3500,
                 "input_token_details": {"cache_read": 2800}}
        return {"messages": [AIMessage("{}", usage_metadata=usage)], "structured_response": item}


def _run_fake(monkeypatch: Any, responses: list[Any], **overrides: Any) -> tuple[ChunkResult, list[Any]]:
    calls: list[Any] = []
    built: list[str] = []

    def fake_build_agent(system_prompt: str, doc_id: str, settings: Settings) -> _FakeAgent:
        built.append(system_prompt)
        return _FakeAgent(list(responses), calls)

    monkeypatch.setattr(sys.modules[__name__], "build_agent", fake_build_agent)
    settings = Settings(response_cache_dir=None, **overrides)
    result = asyncio.run(chunk_document(FINANCIAL_DOCUMENT_NAME, FINANCIAL_DOCUMENT, settings))
    assert len(built) == 1 and built[0].startswith(SYSTEM_PROMPT) and "L0061 | Combined ratio" in built[0]
    return result, calls


def _assert_invariants(result: ChunkResult) -> None:
    r = result.report
    assert r.coverage_pct == 100.0 and r.reconstruction_ok and r.numbers_preserved
    assert "".join(c.text for c in result.chunks) == FINANCIAL_DOCUMENT
    doc = _doc()
    for c in result.chunks:
        for a, b in doc.table_spans:
            assert not (a < c.start_line <= b), f"table L{a}-L{b} split at {c.start_line}"


def test_source_doc_numbering_pages_and_tables() -> None:
    doc = _doc()
    assert doc.n == 61 and "".join(doc.lines) == FINANCIAL_DOCUMENT
    assert doc.numbered().splitlines()[6] == "L0007 # Independent Auditor's Report"
    assert doc.table_spans == [(20, 25), (29, 33), (40, 54), (58, 61)]  # loss run crosses page 3 -> 4
    assert doc.page_range(1, 5) == (None, None) and doc.page_range(35, 55) == (3, 4)


def test_good_response_end_to_end(monkeypatch: Any) -> None:
    result, calls = _run_fake(monkeypatch, [_good_response()])
    _assert_invariants(result)
    r = result.report
    assert len(calls) == 1 and r.retries == 0 and not r.degraded
    assert (r.input_tokens, r.cached_tokens) == (3000, 2800)
    by = {c.section: c for c in result.chunks}
    assert by["preamble"].start_line == 1 and by["preamble"].page_start is None
    assert by["auditor_report"].scale == "unknown" and by["auditor_report"].currency is None
    assert by["income_statement"].reinsurance_basis == "mixed" and by["income_statement"].page_start == 2
    assert by["loss_run"].scale == "units" and (by["loss_run"].page_start, by["loss_run"].page_end) == (3, 4)
    flags = {p.column_label: (p.is_latest, p.is_full_year) for c in result.chunks for p in c.periods}
    assert flags["Net 2025"] == (True, True) and flags["Net 2024"] == (False, True)
    assert flags["Dec 31, 2025"] == (True, None)       # not beaten by the loss-run valuation date
    assert flags["2026 Budget"] == (None, None)        # budget: no is_latest; full year only for annual/ttm
    assert r.latest_annual_end == date(2025, 12, 31) and r.latest_interim_end is None


def test_llm_text_and_derived_flags_are_ignored(monkeypatch: Any) -> None:
    bad = _good_response()
    bad["chunks"][2]["text"] = "TAMPERED 99,999"
    bad["chunks"][2]["periods"][3]["is_latest"] = True   # Net 2024 is not latest
    result, _ = _run_fake(monkeypatch, [bad])
    _assert_invariants(result)
    assert all("TAMPERED" not in c.text for c in result.chunks)
    net24 = next(p for c in result.chunks for p in c.periods if p.column_label == "Net 2024")
    assert net24.is_latest is False


def test_repair_loop_fixes_gap_overlap_and_split_table(monkeypatch: Any) -> None:
    gap, overlap, split = _good_response(), _good_response(), _good_response()
    gap["chunks"][1]["end_line"] = 12                       # L13-L14 uncovered
    overlap["chunks"][3]["start_line"] = 25                 # overlaps and cuts table L20-25
    split["chunks"][4]["end_line"] = 46                     # cuts loss run L40-54
    split["chunks"][5]["start_line"] = 47
    for bad, expect in ((gap, "gap"), (overlap, "overlap"), (split, "cuts table")):
        result, calls = _run_fake(monkeypatch, [bad, _good_response()])
        _assert_invariants(result)
        assert result.report.retries == 1 and not result.report.degraded
        repair = calls[1]
        assert repair[1]["role"] == "assistant" and expect in repair[2]["content"]


def test_unparseable_then_fallback(monkeypatch: Any) -> None:
    err = RuntimeError("StructuredOutputValidationError: invalid JSON")
    result, calls = _run_fake(monkeypatch, [err, {"chunks": "nope"}, err])
    _assert_invariants(result)
    r = result.report
    assert len(calls) == 3 and r.retries == 2 and r.degraded
    assert result.chunks[0].section == "preamble" and result.chunks[0].end_line == 6
    paths = [c.header_path for c in result.chunks]
    assert "Financial Statements > Statement of Income > Underwriting Results" in paths  # parent kept
    assert all(c.section in ("preamble", "other") for c in result.chunks)


def test_evidence_failures_downgrade_without_retry(monkeypatch: Any) -> None:
    resp = _good_response()
    resp["chunks"][3]["reinsurance_basis"] = "net"
    resp["chunks"][3]["evidence"].append({"field": "reinsurance_basis", "line": 30})   # separator row
    resp["chunks"][2]["periods"][0]["evidence_line"] = 9                              # auditor text
    resp["document"]["default_scale_evidence_line"] = 1                               # no "thousand"
    result, calls = _run_fake(monkeypatch, [resp])
    assert len(calls) == 1
    bs = next(c for c in result.chunks if c.section == "balance_sheet")
    assert bs.reinsurance_basis == "unknown" and any("reinsurance_basis" in w for w in bs.warnings)
    assert next(p for c in result.chunks for p in c.periods if p.column_label == "Gross 2025").period_end is None
    assert result.document.default_scale == "unknown"


def test_oversized_table_split_repeats_header(monkeypatch: Any) -> None:
    result, _ = _run_fake(monkeypatch, [_good_response()], max_chunk_lines=8)
    r = result.report
    assert r.coverage_pct == 100.0 and r.reconstruction_ok and r.numbers_preserved
    parts = [c for c in result.chunks if c.section == "loss_run"]
    assert len(parts) > 2
    repeated = [c for c in parts if any("repeated" in w for w in c.warnings)]
    assert repeated and all(c.text.startswith("| Accident Year |") for c in repeated)


def test_response_cache_makes_reruns_deterministic(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(sys.modules[__name__], "build_agent",
                        lambda s, d, st: _FakeAgent([_good_response()], calls))
    settings = Settings(response_cache_dir=str(tmp_path))
    first = asyncio.run(chunk_document(FINANCIAL_DOCUMENT_NAME, FINANCIAL_DOCUMENT, settings))
    second = asyncio.run(chunk_document(FINANCIAL_DOCUMENT_NAME, FINANCIAL_DOCUMENT, settings))
    assert len(calls) == 1 and second.report.response_cache_hit
    assert [c.model_dump() for c in first.chunks] == [c.model_dump() for c in second.chunks]


def test_run_writes_outputs(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(sys.modules[__name__], "build_agent",
                        lambda s, d, st: _FakeAgent([_good_response()], []))
    asyncio.run(run(Settings(response_cache_dir=None), tmp_path / "out"))
    rows = [json.loads(line) for line in (tmp_path / "out" / "chunks.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and rows[0]["document"] == FINANCIAL_DOCUMENT_NAME
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report[0]["report"]["reconstruction_ok"] is True


def test_model_request_carries_cache_settings_and_strict_schema(monkeypatch: Any) -> None:
    """Offline check that cache key/retention and the strict schema reach the request payload."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline")
    from langchain.agents.structured_output import ProviderStrategy
    from langchain_core.messages import HumanMessage, SystemMessage

    model = build_model("abc123", Settings())
    kwargs = ProviderStrategy(LLMResponse, strict=True).to_model_kwargs()
    payload = model._get_request_payload([SystemMessage("sys"), HumanMessage("hi")], **kwargs)
    assert payload["prompt_cache_key"] == "chunker-abc123"
    assert payload["prompt_cache_retention"] == "24h"
    assert payload["reasoning"] == {"effort": "low"}
    fmt = payload["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert set(fmt["schema"]["required"]) == set(LLMResponse.model_fields)
