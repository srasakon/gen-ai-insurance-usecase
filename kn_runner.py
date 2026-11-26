"""Knowledge node runner (LangChain v1 + OpenAI GPT-5.4).

Inputs are plain variables in this file:
    SYSTEM_PROMPT       generic rules (no document type or field knowledge)
    FINANCIAL_DOCUMENT  one financial statement (markdown)
    KNOWLEDGE_NODE      one field requirement (markdown knowledge node)

Prompt layout (what OpenAI caches):
    system message  = SYSTEM_PROMPT + <document>FINANCIAL_DOCUMENT</document>
                      -> stable, byte-identical prefix: CACHED
    user message    = <knowledge_node field="...">KNOWLEDGE_NODE</knowledge_node> + instruction
                      -> the only part that varies, placed LAST

The cached prefix is (output schema + reasoning settings + system message). It is
written by the first call and read by every later call for the same document
within the retention window (default 24h), so a second run, or more knowledge
nodes added to KNOWLEDGE_NODES, pay for the document once.

Per document:
    1. ``await agent.ainvoke(first_node)``                         -> writes the cache
    2. ``await agent.abatch(rest, max_concurrency=FIELD_CONCURRENCY)`` -> reads it
Across documents: ``asyncio.gather`` bounded by ``Semaphore(DOC_CONCURRENCY)``.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python knowledge_node_runner.py         # writes results/ ; add -v for per-call logs

Tests (no network):
    pytest knowledge_node_runner.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field

log = logging.getLogger("finx")


# =========================================================================== #
# Inputs (hardcoded variables)
# =========================================================================== #
# Generic: no document type or field knowledge here. That belongs in the knowledge node.
# Written in ASD-STE100 Simplified Technical English: short sentences, one instruction
# per sentence, active voice, the condition before the instruction.
SYSTEM_PROMPT = """\
You find data in documents and copy it into a structured output.

The document is below, between the <document> tags.

Each user message has one knowledge node. The knowledge node gives the requirements for one field. \
Find that field in the document. Then give the result in the output format.

If a knowledge node rule is different from a rule in this list, use the knowledge node rule.

Rules:
1. Use only the data in the document. Do not use other data.
2. Do not guess a value.
3. Do not calculate a value. Calculate a value only if the knowledge node tells you to calculate it.
4. Put each item that you find in `values`. Use the item names that the knowledge node gives.
5. If the knowledge node asks for values by date, write one entry for each item and each date.
6. Put the date or the period of each value in `as_of`.
7. If an item is not in the document, include the item. Set its `value` to null.
8. If the field is not in the document, set `found` to false. Make `values` an empty list.
9. Write numbers without thousands separators. Write numbers without currency symbols.
10. If the document shows a number in parentheses, the number is negative. Write a minus sign before it.
11. Do not change the scale of a number. Write the printed unit and scale in `unit`.
12. In `source_quote`, copy the text or the table row that shows the value. Copy it exactly as it is written.
13. Before you answer, examine the evidence that the knowledge node describes.
14. Before you answer, do the checks in the knowledge node.
15. In `reasoning`, write one or two sentences.
16. If a label is not clear, give a lower `confidence`.
17. If more than one value is possible, give a lower `confidence`."""

FINANCIAL_DOCUMENT_NAME = "harbor_point_fy2024"
FINANCIAL_DOCUMENT = """\
# Harbor Point Fabrication, Inc.

## Consolidated Financial Statements

For the fiscal years ended December 31, 2024 and 2023
(All amounts in USD thousands, except per-share data)

*This is a fictional sample prepared for testing an extraction pipeline.*

---

## Consolidated Statements of Operations

| | 2024 | 2023 |
|---|---:|---:|
| Product sales | 412,850 | 376,204 |
| Service and maintenance revenue | 58,316 | 51,930 |
| **Total revenue** | **471,166** | **428,134** |
| Cost of product sales | (281,407) | (262,118) |
| Cost of services | (33,962) | (30,845) |
| **Gross profit** | **155,797** | **135,171** |
| Selling, general and administrative expenses | (71,240) | (66,512) |
| Research and development | (18,905) | (16,330) |
| Restructuring charges | (4,120) | — |
| Impairment of goodwill | — | (9,800) |
| **Operating income** | **61,532** | **42,529** |
| Interest expense, net | (8,714) | (9,960) |
| Other income (expense), net | 1,237 | (642) |
| **Income before income taxes** | **54,055** | **31,927** |
| Provision for income taxes | (12,433) | (8,302) |
| **Net income** | **41,622** | **23,625** |
| Less: net income attributable to non-controlling interests | (1,146) | (988) |
| **Net income attributable to Harbor Point Fabrication, Inc.** | **40,476** | **22,637** |
| | | |
| Earnings per share — basic | 2.71 | 1.52 |
| Earnings per share — diluted | 2.66 | 1.49 |
| Weighted-average shares outstanding — basic | 14,936 | 14,893 |
| Weighted-average shares outstanding — diluted | 15,217 | 15,192 |

---

## Consolidated Balance Sheets

As of December 31

| | 2024 | 2023 |
|---|---:|---:|
| **Assets** | | |
| Cash and cash equivalents | 38,412 | 29,075 |
| Accounts receivable, net of allowance of 1,840 and 1,615 | 72,960 | 66,218 |
| Inventories | 88,304 | 81,770 |
| Prepaid expenses and other current assets | 9,615 | 8,442 |
| **Total current assets** | **209,291** | **185,505** |
| Property, plant and equipment, net | 164,880 | 158,326 |
| Operating lease right-of-use assets | 21,407 | 23,961 |
| Goodwill | 47,200 | 47,200 |
| Intangible assets, net | 18,533 | 21,904 |
| Deferred tax assets | 3,118 | 4,027 |
| Other non-current assets | 5,226 | 4,883 |
| **Total assets** | **469,655** | **445,806** |
| **Liabilities and equity** | | |
| Accounts payable | 41,872 | 39,605 |
| Accrued compensation and benefits | 14,330 | 12,914 |
| Current portion of long-term debt | 12,500 | 12,500 |
| Current operating lease liabilities | 4,906 | 4,780 |
| Other current liabilities | 11,287 | 10,066 |
| **Total current liabilities** | **84,895** | **79,865** |
| Long-term debt, net of current portion | 118,750 | 143,750 |
| Non-current operating lease liabilities | 17,930 | 20,412 |
| Deferred tax liabilities | 9,842 | 8,617 |
| Other non-current liabilities | 6,214 | 6,903 |
| **Total liabilities** | **237,631** | **259,547** |
| Common stock and additional paid-in capital | 96,418 | 94,203 |
| Retained earnings | 131,987 | 99,423 |
| Accumulated other comprehensive loss | (4,212) | (3,051) |
| **Total Harbor Point Fabrication, Inc. shareholders' equity** | **224,193** | **190,575** |
| Non-controlling interests | 7,831 | (4,316) |
| **Total equity** | **232,024** | **186,259** |
| **Total liabilities and equity** | **469,655** | **445,806** |

---

## Consolidated Statements of Cash Flows

| | 2024 | 2023 |
|---|---:|---:|
| **Operating activities** | | |
| Net income | 41,622 | 23,625 |
| Depreciation and amortization | 24,618 | 23,077 |
| Impairment of goodwill | — | 9,800 |
| Share-based compensation | 3,904 | 3,512 |
| Deferred income taxes | 2,134 | (1,226) |
| Changes in accounts receivable | (6,742) | (4,388) |
| Changes in inventories | (6,534) | (9,915) |
| Changes in accounts payable and accrued liabilities | 3,683 | 2,940 |
| Other operating activities, net | (1,147) | 806 |
| **Net cash provided by operating activities** | **61,538** | **48,231** |
| **Investing activities** | | |
| Capital expenditures | (27,806) | (31,448) |
| Proceeds from sale of equipment | 1,212 | 640 |
| **Net cash used in investing activities** | **(26,594)** | **(30,808)** |
| **Financing activities** | | |
| Repayments of long-term debt | (25,000) | (12,500) |
| Dividends paid | (7,912) | (7,446) |
| Proceeds from employee stock plans | 1,689 | 1,322 |
| Distributions to non-controlling interests | (412) | (390) |
| **Net cash used in financing activities** | **(31,635)** | **(19,014)** |
| Effect of exchange rate changes on cash | 6,028 | (1,180) |
| **Net increase (decrease) in cash and cash equivalents** | **9,337** | **(2,771)** |
| Cash and cash equivalents, beginning of year | 29,075 | 31,846 |
| **Cash and cash equivalents, end of year** | **38,412** | **29,075** |

---

## Selected Notes to the Consolidated Financial Statements

### Note 1 — Basis of presentation

The consolidated financial statements include the accounts of Harbor Point Fabrication, Inc. and its \
majority-owned subsidiaries. The fiscal year ends on December 31. Amounts are presented in thousands of \
U.S. dollars unless otherwise noted.

### Note 4 — Revenue

Revenue from product sales is recognized when control transfers to the customer, generally upon shipment. \
Service and maintenance revenue is recognized over the contract term. Revenue by segment for 2024 was \
Industrial Systems 298,410, Marine Components 126,904 and Aftermarket Services 45,852 \
(2023: 271,662; 113,240; 43,232).

### Note 9 — Debt

The company has a term loan with an original principal of 175,000, repayable in quarterly instalments of \
3,125, maturing in 2028. During 2024 the company made a voluntary prepayment of 12,500. The weighted-average \
interest rate for 2024 was 5.9% (2023: 6.3%).

### Note 15 — Subsequent events

On February 14, 2025 the Board declared a quarterly dividend of 0.14 per share, payable on March 20, 2025.
"""

KNOWLEDGE_NODE_NAME = "current_assets_and_current_liabilities"
KNOWLEDGE_NODE = """\
# Field: `current_assets_and_current_liabilities`

- **Node ID:** FIN_07
- **Scope:** [Balance sheet, Audited or reviewed financial statements]
- **Product line:** [Casualty, Inland Marine, Life Sciences, Property, Railroad, Transportation, Wholesale Fin Lines, Small Fin Lines]

## Definition
This field is the total current assets and the total current liabilities on each balance sheet date. \
These two values show the short-term liquidity of the company.

## Is
- The line with the label "Total current assets".
- The line with the label "Total current liabilities".

## Is not
- The line "Total assets".
- The line "Total liabilities".
- A line for one current item, for example cash, receivables or payables.
- A working capital value. Do not calculate working capital.

## Rules
- Give one object for each balance sheet date.
- Each object has three items: `as_of`, `current_assets` and `current_liabilities`.
- If a subtotal is not printed, set its value to null.
- An unclassified balance sheet does not have current subtotals. If the balance sheet is unclassified, set `found` to false.
- Use a unit note only if the statements print it. Do not add a unit that is not printed.

## Evidence to weigh
For each line that you find, write these four items:
- The label.
- The amount.
- The column.
- Keep or reject. Give the reason.

## Verify before answering
- Make sure that each value has a "Total current" label.
- Make sure that you did not calculate a value.

## Examples
### Example 1
- **Situation:** A classified balance sheet shows the two subtotals.
- **Method:** Copy the two values for each date.
- **Result:** One object for each date.

### Example 2
- **Situation:** The balance sheet shows only the "Total current assets" subtotal.
- **Method:** Copy the current assets value. Set the current liabilities value to null.
- **Result:** `current_liabilities` is null.

### Example 3
- **Situation:** The balance sheet is unclassified. It does not show current subtotals.
- **Method:** Do not use other lines to find the values.
- **Result:** `found` is false.
"""

# The run iterates over these. Add entries to process more documents / nodes;
# every node for a document reuses that document's cached prefix.
DOCUMENTS: dict[str, str] = {FINANCIAL_DOCUMENT_NAME: FINANCIAL_DOCUMENT}
KNOWLEDGE_NODES: dict[str, str] = {KNOWLEDGE_NODE_NAME: KNOWLEDGE_NODE}


# =========================================================================== #
# Shared output schema. ONE schema for every node: it is part of the cached prefix.
# Generic `values` list so nodes that return several items / dates (like FIN_07)
# fit without a per-field schema.
# =========================================================================== #
class ExtractedValue(BaseModel):
    """One extracted item at one date or period."""

    name: str = Field(description="The item name that the knowledge node gives.")
    value: Optional[str] = Field(
        description="The number or the text. Write numbers without thousands separators and without currency "
        "symbols. Write a minus sign for a negative number. Null if the item is not printed."
    )
    unit: Optional[str] = Field(description='The printed unit and scale. For example: "USD thousands". Null if not printed.')
    as_of: Optional[str] = Field(description="The date or the period of the value. Null if the value has no date.")
    source_quote: Optional[str] = Field(
        description="The text or table row that shows the value, exactly as it is written. Null if not printed."
    )


class Extraction(BaseModel):
    """Result for one knowledge node."""

    field: str = Field(description="The field name from the knowledge node.")
    found: bool = Field(description="False if the field is not in the document.")
    values: list[ExtractedValue] = Field(description="The items that you found. An empty list if found is false.")
    reasoning: str = Field(description="One or two sentences. Tell how you found the values.")
    confidence: float = Field(description="A number from 0 to 1.")


RESULT_COLUMNS = [
    "document", "field", "status", "found", "values", "reasoning", "confidence", "error",
    "input_tokens", "cached_tokens", "output_tokens",
]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    prompt: str


@dataclass(frozen=True)
class Doc:
    label: str
    doc_id: str
    markdown: str


@dataclass(frozen=True)
class Settings:
    model: str = "gpt-5.4"
    reasoning_effort: str = "low"
    cache_retention: str = "24h"
    field_concurrency: int = 10
    doc_concurrency: int = 3
    max_retries: int = 3
    timeout: float = 120.0


# =========================================================================== #
# Inputs -> objects
# =========================================================================== #
def doc_id_for(name: str, markdown: str) -> str:
    """First 16 hex chars of sha1(name + content). Stable across runs, so the cache key is too."""
    return hashlib.sha1((name + markdown).encode("utf-8")).hexdigest()[:16]


def load_documents(documents: dict[str, str]) -> list[Doc]:
    return [Doc(label=name, doc_id=doc_id_for(name, md), markdown=md) for name, md in documents.items()]


def load_fields(nodes: dict[str, str]) -> list[FieldSpec]:
    fields = [FieldSpec(name=name.strip(), prompt=node.strip()) for name, node in nodes.items()]
    if not fields or any(not f.name or not f.prompt for f in fields):
        raise ValueError("every knowledge node needs a non-empty name and text")
    return fields


# =========================================================================== #
# Prompt layout
# =========================================================================== #
def build_system_prompt(system_prompt: str, markdown: str) -> str:
    """Stable, cacheable prefix: system prompt + document. No timestamps, IDs or field text."""
    return f"{system_prompt}\n\n<document>\n{markdown}\n</document>"


def field_message(spec: FieldSpec) -> dict[str, Any]:
    """Input for one knowledge node: the only part that differs between calls."""
    content = (
        f'<knowledge_node field="{spec.name}">\n{spec.prompt}\n</knowledge_node>\n\n'
        f"Find the field `{spec.name}` in the document above. Give the result in the output format."
    )
    return {"messages": [{"role": "user", "content": content}]}


# =========================================================================== #
# Model / runnable
# =========================================================================== #
def build_model(doc_id: str, settings: Settings):
    """ChatOpenAI with cache routing set on the model (invoke() does not forward it)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model,
        use_responses_api=True,
        reasoning={"effort": settings.reasoning_effort},  # same on every call -> same prefix
        max_retries=0,  # retries live in ModelRetryMiddleware
        timeout=settings.timeout,
        model_kwargs={
            "prompt_cache_key": f"finx-{doc_id}",  # one key per document
            "prompt_cache_retention": settings.cache_retention,
        },
    )


def build_agent(system_prompt: str, doc_id: str, settings: Settings):
    """One runnable per document; its system prompt holds that document."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
    from langchain.agents.structured_output import ProviderStrategy

    return create_agent(
        model=build_model(doc_id, settings),
        tools=[],
        system_prompt=system_prompt,
        response_format=ProviderStrategy(Extraction, strict=True),
        middleware=[
            # on_failure="error": raise after the last retry so the real error lands in the row
            # (the default "continue" swallows it into an AIMessage).
            ModelRetryMiddleware(
                max_retries=settings.max_retries, initial_delay=2.0, max_delay=60.0, on_failure="error"
            ),
            ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),
        ],
    )


# =========================================================================== #
# Results
# =========================================================================== #
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


def to_row(doc: Doc, spec: FieldSpec, result: Any) -> dict[str, Any]:
    """Flatten one result (or exception) into an output row. Never raises."""
    row: dict[str, Any] = {c: None for c in RESULT_COLUMNS}
    row.update(document=doc.label, field=spec.name)
    row["input_tokens"], row["cached_tokens"], row["output_tokens"] = usage_of(result)

    if isinstance(result, BaseException):
        # Includes StructuredOutputValidationError, which ProviderStrategy raises on refusal.
        row.update(status="error", error=f"{type(result).__name__}: {result}")
        return row

    extraction = result.get("structured_response") if isinstance(result, dict) else None
    if extraction is None:
        row.update(status="error", error="no structured_response (refusal or model call limit reached)")
        return row
    if isinstance(extraction, dict):
        extraction = Extraction.model_validate(extraction)

    row.update(extraction.model_dump())
    row.update(field=spec.name, status="ok")  # our name, not the model's echo
    return row


def compact_values(values: Sequence[dict[str, Any]] | None) -> str:
    """'current_assets[December 31, 2024]=209291; ...' for the wide table."""
    parts = []
    for v in values or []:
        key = f"{v['name']}[{v['as_of']}]" if v.get("as_of") else v["name"]
        parts.append(f"{key}={'' if v.get('value') is None else v['value']}")
    return "; ".join(parts)


# =========================================================================== #
# Orchestration
# =========================================================================== #
async def extract_document(
    doc: Doc,
    fields: Sequence[FieldSpec],
    system_prompt: str,
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Warm the cache with node 1, then batch the rest. A failed node becomes an error row."""
    async with semaphore:
        started = time.perf_counter()
        results: list[Any]
        try:
            agent = build_agent(build_system_prompt(system_prompt, doc.markdown), doc.doc_id, settings)
        except Exception as exc:  # noqa: BLE001 - one bad doc must not stop the run
            results = [exc] * len(fields)
        else:
            try:
                first = await agent.ainvoke(field_message(fields[0]))
            except Exception as exc:  # noqa: BLE001
                first = exc
            results = [first]
            if len(fields) > 1:
                results += await agent.abatch(
                    [field_message(f) for f in fields[1:]],
                    config={"max_concurrency": settings.field_concurrency},
                    return_exceptions=True,
                )

        rows = [to_row(doc, spec, res) for spec, res in zip(fields, results)]
        for r in rows:
            if r["status"] == "error":
                log.warning("%s / %s failed: %s", doc.label, r["field"], r["error"])
            else:
                log.debug("%s / %s: input=%s cached=%s", doc.label, r["field"], r["input_tokens"], r["cached_tokens"])
        _log_summary(doc.label, rows, time.perf_counter() - started)
        return rows


def _log_summary(label: str, rows: Sequence[dict[str, Any]], seconds: float) -> None:
    ok = sum(r["status"] == "ok" for r in rows)
    tokens_in = sum(r["input_tokens"] or 0 for r in rows)
    cached = sum(r["cached_tokens"] or 0 for r in rows)
    pct = 100.0 * cached / tokens_in if tokens_in else 0.0
    log.info(
        "%s: %d/%d fields ok | %.1fs | input=%d cached=%d (%.1f%% cache hit)",
        label, ok, len(rows), seconds, tokens_in, cached, pct,
    )


def write_outputs(rows: Sequence[dict[str, Any]], fields: Sequence[FieldSpec], out_dir: str | Path) -> None:
    """Write results.jsonl, results.csv and wide.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({c: r.get(c) for c in RESULT_COLUMNS}, ensure_ascii=False) + "\n")

    with (out / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            flat = dict(r)
            flat["values"] = json.dumps(r.get("values"), ensure_ascii=False) if r.get("values") is not None else ""
            writer.writerow(flat)

    wide: dict[str, dict[str, Any]] = {}
    for r in rows:
        line = wide.setdefault(r["document"], {"document": r["document"]})
        line[r["field"]] = "#ERROR" if r["status"] == "error" else compact_values(r.get("values"))
    with (out / "wide.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["document", *(f.name for f in fields)])
        writer.writeheader()
        writer.writerows(wide.values())


async def run(settings: Settings, out_dir: str | Path) -> list[dict[str, Any]]:
    """Run every knowledge node on every document, write outputs, log totals."""
    docs = load_documents(DOCUMENTS)
    fields = load_fields(KNOWLEDGE_NODES)
    log.info("%d document(s) x %d knowledge node(s) = %d calls", len(docs), len(fields), len(docs) * len(fields))

    started = time.perf_counter()
    semaphore = asyncio.Semaphore(settings.doc_concurrency)
    per_doc = await asyncio.gather(
        *(extract_document(d, fields, SYSTEM_PROMPT, settings, semaphore) for d in docs)
    )
    rows = [r for doc_rows in per_doc for r in doc_rows]

    write_outputs(rows, fields, out_dir)
    _log_summary("TOTAL", rows, time.perf_counter() - started)
    log.info("wrote %s", Path(out_dir).resolve())
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the hardcoded knowledge node(s) on the hardcoded document(s).")
    p.add_argument("--out", default="results/")
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--reasoning-effort", default="low", choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    p.add_argument("--cache-retention", default="24h", choices=["in_memory", "24h"])
    p.add_argument("--field-concurrency", type=int, default=10)
    p.add_argument("--doc-concurrency", type=int, default=3)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.field_concurrency < 1 or args.doc_concurrency < 1:
        p.error("concurrency values must be >= 1")
    return args


def settings_from(args: argparse.Namespace) -> Settings:
    return Settings(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        cache_retention=args.cache_retention,
        field_concurrency=args.field_concurrency,
        doc_concurrency=args.doc_concurrency,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.verbose:  # keep HTTP client chatter out of normal runs
        for noisy in ("httpx", "openai", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY is not set")
        return 2
    asyncio.run(run(settings_from(args), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

