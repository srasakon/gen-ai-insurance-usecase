"""GEPA optimizer for one knowledge node (gepa >= 0.1.4, ``optimize_anything`` API).

Inputs are plain variables in this file:
    DOCUMENT_MARKDOWN   the document the node failed on
    CURRENT_PROMPT      the knowledge node that produced the wrong answer
    EXPECTED_VALUE      the correct answer (gold)
    ACTUAL_VALUE        the wrong answer the node produced
    REVIEWER_DIAGNOSIS  another agent's reasoning on what is wrong and what to change

The candidate is scored by running it through the SAME production path as
``knowledge_node_runner.py`` (same system prompt, same cached document prefix,
same model, same strict schema). Only the knowledge node text changes.

Anti-overfitting design (one document is a tiny dataset, so this matters most):
    1. Held-out selection. GEPA reflects on a small feedback set and SELECTS on a
       separate validation set. Both are built from label-preserving variants
       of the document: amounts scaled by one factor, years shifted, table
       columns reversed. A node that memorises numbers, dates or column
       positions fails the variants. Add real labelled documents to
       EXTRA_LABELED_EXAMPLES for true generalisation; they go to the valset.
    2. Hard guards in the score (field-agnostic, deterministic):
       - copies a number from any evaluated document or gold value -> score 0
       - more than 500 words (knowledge node rule)                   -> score 0
       - template headings or metadata lines removed or changed      -> x0.5
       - copies a long text span from an evaluated document          -> x0.7
       - small penalties for sentences over 20 words (ASD-STE100) and growth.
    3. A reflection prompt that tells the reflection model to fix the root
       cause with a general rule, to treat the reviewer diagnosis as a
       hypothesis, and to make the smallest change.
    4. Final acceptance gate: the best candidate replaces the current prompt
       only if it beats the seed on the validation set by MIN_GAIN and passes
       every hard guard. Otherwise the current prompt is kept.

GEPA best practices applied: rich actionable side information (expected vs
actual diff, model reasoning, source quotes, guard issues) instead of a bare
score; multi-objective ``scores`` for the Pareto frontier; Pareto candidate
selection; small reflection minibatch; evaluation cache; skip perfect scores;
budget + no-improvement stopper; fixed seed; resumable run_dir; strong
reflection model separate from the task model.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python gepa_optimize_node.py              # writes gepa_runs/<field>/<timestamp>/

Tests (no network):
    pytest gepa_optimize_node.py
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import knowledge_node_runner as runner  # noqa: E402  (production path: prompts, agent, schema)

log = logging.getLogger("gepa_node")


# =========================================================================== #
# Inputs (hardcoded variables). Replace the example values with a real failure.
# =========================================================================== #
DOCUMENT_NAME = runner.FINANCIAL_DOCUMENT_NAME
DOCUMENT_MARKDOWN = runner.FINANCIAL_DOCUMENT

FIELD_NAME = runner.KNOWLEDGE_NODE_NAME
CURRENT_PROMPT = runner.KNOWLEDGE_NODE

# Gold answer. Same shape as the runner output: found + values[name, as_of, value].
EXPECTED_VALUE: dict[str, Any] = {
    "found": True,
    "values": [
        {"name": "current_assets", "as_of": "December 31, 2024", "value": "209291"},
        {"name": "current_liabilities", "as_of": "December 31, 2024", "value": "84895"},
        {"name": "current_assets", "as_of": "December 31, 2023", "value": "185505"},
        {"name": "current_liabilities", "as_of": "December 31, 2023", "value": "79865"},
    ],
}

# EXAMPLE failure: replace with the wrong answer the node actually produced.
ACTUAL_VALUE: dict[str, Any] = {
    "found": True,
    "values": [
        {"name": "current_assets", "as_of": "December 31, 2024", "value": "209291"},
        {"name": "current_liabilities", "as_of": "December 31, 2024", "value": "237631"},
    ],
}

# EXAMPLE diagnosis: replace with the other agent's reasoning.
REVIEWER_DIAGNOSIS = """\
The output has two errors. First, current_liabilities for the latest date is the "Total liabilities" \
line, not the "Total current liabilities" line. The node says "Is not: total liabilities", but the \
rule is not near the instruction that selects the line, so the model did not use it. Second, the \
output has only the latest balance sheet date. The node says "one object for each balance sheet date", \
but it does not say that every date column in the balance sheet must be returned. A possible change: \
tell the model to read each date column and to match the exact subtotal label before it copies a value."""

# Optional real labelled documents for true generalisation (always added to the valset):
#   [{"name": "acme_fy2023", "markdown": "...", "expected": {"found": True, "values": [...]}}]
EXTRA_LABELED_EXAMPLES: list[dict[str, Any]] = []

# =========================================================================== #
# Optimisation settings
# =========================================================================== #
TASK_SETTINGS = runner.Settings(max_retries=2)  # same model / effort as production
REFLECTION_LM = os.environ.get("GEPA_REFLECTION_LM", "openai/gpt-5.4")  # litellm model name
REFLECTION_LM_KWARGS: dict[str, Any] = {"reasoning_effort": "high"}
MAX_METRIC_CALLS = 60          # total (candidate, example) evaluations
REFLECTION_MINIBATCH_SIZE = 2  # small minibatches -> focused, cheap reflections
NO_IMPROVEMENT_PATIENCE = 6    # stop after this many proposals without a new best
MIN_GAIN = 0.01                # the best candidate must beat the seed by this much
TASK_TRIALS = 1                # >1 averages repeated runs to reduce model noise
MAX_WORKERS = 4
RANDOM_SEED = 0
OUT_ROOT = Path(__file__).resolve().parent / "gepa_runs"

MAX_NODE_WORDS = 500           # knowledge node rule
MAX_STE_SENTENCE_WORDS = 20    # ASD-STE100 procedural sentence limit


# =========================================================================== #
# Normalisation and scoring
# =========================================================================== #
_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%d %B %Y", "%d %b %Y", "%B %Y", "%Y")


def norm_value(value: Any) -> Optional[str]:
    """'(4,212)' -> '-4212', '$1,000.50' -> '1000.5'. Text is lower-cased and trimmed."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "null", "none", "n/a"):
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[\s$€£¥,()]", "", text)
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if negative and not text.startswith("-"):
            text = "-" + text
    return text


def norm_date(value: Any) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    text = re.sub(r"\s+", " ", str(value).strip().replace("Sept", "Sep"))
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d" if "%d" in fmt else "%Y")
        except ValueError:
            continue
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_answer(answer: Any) -> dict[str, Any]:
    """Accept {'found', 'values'} or a plain scalar string."""
    if isinstance(answer, (str, int, float)):
        return {"found": True, "values": [{"name": FIELD_NAME, "as_of": None, "value": str(answer)}]}
    return {"found": bool(answer.get("found", True)), "values": list(answer.get("values") or [])}


def _keyed(values: list[dict[str, Any]], use_dates: bool) -> dict[tuple, Optional[str]]:
    keyed = {}
    for v in values:
        key = (str(v.get("name", "")).strip().lower(), norm_date(v.get("as_of")) if use_dates else None)
        keyed[key] = norm_value(v.get("value"))
    return keyed


def score_answer(expected: dict[str, Any], predicted: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """F1 over (item, date) -> value pairs; found-flag mismatch scores 0. Returns (score, diff)."""
    expected, predicted = normalize_answer(expected), normalize_answer(predicted)
    if not expected["found"]:
        ok = not predicted["found"] or not predicted["values"]
        return (1.0 if ok else 0.0), {"expected_found": False, "predicted_found": predicted["found"]}
    if not predicted["found"]:
        return 0.0, {"expected_found": True, "predicted_found": False, "missing": expected["values"]}

    use_dates = any(v.get("as_of") for v in expected["values"])
    scalar = all(v.get("name") == FIELD_NAME and not v.get("as_of") for v in expected["values"])
    if scalar:  # compare values only
        exp_vals = [norm_value(v.get("value")) for v in expected["values"]]
        pred_vals = [norm_value(v.get("value")) for v in predicted["values"]]
        correct = sum(min(exp_vals.count(x), pred_vals.count(x)) for x in set(exp_vals))
        diff = {"expected": exp_vals, "predicted": pred_vals}
        n_exp, n_pred = len(exp_vals), len(pred_vals)
    else:
        exp, pred = _keyed(expected["values"], use_dates), _keyed(predicted["values"], use_dates)
        correct = sum(1 for k, v in exp.items() if k in pred and pred[k] == v)
        diff = {
            "missing": [f"{k[0]} @ {k[1]}: expected {v}" for k, v in exp.items() if k not in pred],
            "wrong": [f"{k[0]} @ {k[1]}: expected {v}, got {pred[k]}" for k, v in exp.items()
                      if k in pred and pred[k] != v],
            "extra": [f"{k[0]} @ {k[1]}: {v}" for k, v in pred.items() if k not in exp],
        }
        n_exp, n_pred = len(exp), len(pred)

    if correct == 0:
        return 0.0, diff
    precision, recall = correct / n_pred, correct / n_exp
    return 2 * precision * recall / (precision + recall), diff


# =========================================================================== #
# Label-preserving document variants (field-agnostic)
# =========================================================================== #
_AMOUNT_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d{4,})(?![\w]|[.,]\d)")
_YEAR_RE = re.compile(r"(?<![\d,.])((?:19|20)\d{2})(?![\d]|,\d{3})")


def _is_year(token: str) -> bool:
    return "," not in token and len(token) == 4 and 1900 <= int(token) <= 2099


def _expected_digits(expected: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for v in normalize_answer(expected)["values"]:
        out.update(t for t in re.findall(r"\d{4,}", str(norm_value(v.get("value")) or "")) if not _is_year(t))
    return out


def scale_amounts(markdown: str, expected: dict[str, Any], factor: float) -> Optional[tuple[str, dict[str, Any]]]:
    """Multiply every amount (4+ digits, not a year) by one factor, in the document and the gold.

    One factor keeps totals consistent (up to rounding). Returns None when a gold
    number is not copied from the document (computed field): the label would break.
    """
    mapping: dict[str, str] = {}
    for token in _AMOUNT_RE.findall(markdown):
        digits = token.replace(",", "")
        if _is_year(token) or digits in mapping:
            continue
        new = max(1, round(int(digits) * factor))
        if "," not in token and 1900 <= new <= 2099:
            new += 200  # never create a fake year
        mapping[digits] = str(new)
    if not _expected_digits(expected) <= set(mapping):
        return None

    def doc_sub(m: re.Match) -> str:
        token = m.group(1)
        digits = token.replace(",", "")
        if digits not in mapping:
            return token
        return f"{int(mapping[digits]):,}" if "," in token else mapping[digits]

    def gold_sub(text: Any) -> Any:
        if text is None:
            return None
        return re.sub(r"\d{4,}", lambda m: mapping.get(m.group(0), m.group(0)), str(norm_value(text)))

    new_expected = normalize_answer(expected)
    new_expected["values"] = [{**v, "value": gold_sub(v.get("value"))} for v in new_expected["values"]]
    return _AMOUNT_RE.sub(doc_sub, markdown), new_expected


def shift_years(markdown: str, expected: dict[str, Any], delta: int) -> tuple[str, dict[str, Any]]:
    """Shift every year by delta in the document and in the gold dates/values."""
    def sub(text: Any) -> Any:
        if text is None:
            return None
        return _YEAR_RE.sub(lambda m: str(int(m.group(1)) + delta), str(text))

    new_expected = normalize_answer(expected)
    new_expected["values"] = [{**v, "as_of": sub(v.get("as_of")), "value": sub(v.get("value"))}
                              for v in new_expected["values"]]
    return sub(markdown), new_expected


def reverse_table_columns(markdown: str) -> str:
    """Reverse the data columns of every markdown table (the label column stays first)."""
    out = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = stripped[1:-1].split("|")
            if len(cells) >= 3:
                line = "|" + "|".join([cells[0], *cells[1:][::-1]]) + "|"
        out.append(line)
    return "\n".join(out) + ("\n" if markdown.endswith("\n") else "")


@dataclass(frozen=True)
class Example:
    name: str
    description: str
    markdown: str
    expected: dict[str, Any] = field(hash=False, compare=False)


def build_examples(
    markdown: str, expected: dict[str, Any], extra: list[dict[str, Any]], seed: int = RANDOM_SEED
) -> tuple[list[Example], list[Example]]:
    """Feedback set (reflection) and validation set (selection) from one labelled document."""
    rng = random.Random(seed)
    original = Example(DOCUMENT_NAME, "original document (the reported failure)", markdown, normalize_answer(expected))
    feedback, validation = [original], [original]

    scaled = scale_amounts(markdown, expected, round(rng.uniform(1.3, 1.9), 3))
    if scaled:
        feedback.append(Example(f"{DOCUMENT_NAME}__amounts_scaled", "variant: every amount scaled by one factor", *scaled))
    else:
        log.warning("gold values are not copied from the document; amount-scaling variants skipped")

    validation.append(Example(f"{DOCUMENT_NAME}__columns_reversed", "variant: table date columns reversed",
                              reverse_table_columns(markdown), normalize_answer(expected)))
    shifted_md, shifted_gold = shift_years(markdown, expected, -rng.randint(3, 9))
    validation.append(Example(f"{DOCUMENT_NAME}__years_shifted_columns_reversed",
                              "variant: years shifted and table columns reversed",
                              reverse_table_columns(shifted_md), shifted_gold))
    combo = scale_amounts(shifted_md, shifted_gold, round(rng.uniform(2.1, 2.9), 3))
    if combo:
        validation.append(Example(f"{DOCUMENT_NAME}__all_changes", "variant: amounts scaled, years shifted",
                                  combo[0], combo[1]))
    for item in extra:
        validation.append(Example(item["name"], "extra labelled document", item["markdown"],
                                  normalize_answer(item["expected"])))
    return feedback, validation


# =========================================================================== #
# Generalisation guards (deterministic, field-agnostic)
# =========================================================================== #
def _protected_lines(seed: str) -> list[str]:
    """Headings and metadata lines of the template; they must stay exactly as they are."""
    return [ln.rstrip() for ln in seed.splitlines() if re.match(r"^(#{1,6} |- \*\*[^*]+:\*\*)", ln)]


def _numbers(text: str) -> set[str]:
    return {t.replace(",", "") for t in re.findall(r"\d[\d,]*\d|\d", text) if len(t.replace(",", "")) >= 3}


def _spans(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _long_sentences(text: str) -> list[str]:
    body = re.sub(r"[`*#>|]", " ", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?:])\s+|\n", body) if s.strip()]
    return [s for s in sentences if len(s.split()) > MAX_STE_SENTENCE_WORDS]


def guard_candidate(candidate: str, seed: str, corpus: list[str]) -> tuple[float, float, list[str]]:
    """Return (multiplier, penalty, issues). Only text NEW relative to the seed is judged."""
    multiplier, penalty, issues = 1.0, 0.0, []
    words = len(candidate.split())
    if words > MAX_NODE_WORDS:
        multiplier = 0.0
        issues.append(f"HARD: {words} words; a knowledge node must have at most {MAX_NODE_WORDS} words.")

    corpus_numbers = set().union(*(_numbers(t) for t in corpus)) if corpus else set()
    leaked = sorted((_numbers(candidate) - _numbers(seed)) & corpus_numbers)
    if leaked:
        multiplier = 0.0
        issues.append(f"HARD: copies numbers from the evaluated documents or gold values: {leaked[:8]}. "
                      "A knowledge node must not contain document values.")

    missing = [ln for ln in _protected_lines(seed) if ln not in {c.rstrip() for c in candidate.splitlines()}]
    if missing:
        multiplier *= 0.5
        issues.append(f"Template headings or metadata changed or removed: {missing[:6]}. Keep them unchanged.")

    corpus_spans = set().union(*(_spans(t) for t in corpus)) if corpus else set()
    copied = sorted((_spans(candidate) - _spans(seed)) & corpus_spans)
    if copied:
        multiplier *= 0.7
        issues.append(f"Copies text spans from the evaluated documents: {copied[:4]}. Write general rules.")

    long_ones = [s for s in _long_sentences(candidate) if s not in _long_sentences(seed)]
    if long_ones:
        penalty += min(0.05, 0.01 * len(long_ones))
        issues.append(f"{len(long_ones)} new sentence(s) over {MAX_STE_SENTENCE_WORDS} words (ASD-STE100): "
                      f"{[s[:80] for s in long_ones[:3]]}")

    growth = words - len(seed.split())
    if growth > 75:
        penalty += min(0.05, 0.0005 * (growth - 75))
        issues.append(f"The node grew by {growth} words. Prefer the smallest change that fixes the cause.")
    return multiplier, penalty, issues


# =========================================================================== #
# Task run (production path) and the GEPA evaluator
# =========================================================================== #
def clean_candidate(candidate: str) -> str:
    """Drop a stray language tag line that a reflection model may leave after ``` ."""
    lines = candidate.strip("\n").splitlines()
    if lines and re.fullmatch(r"(markdown|md|text)", lines[0].strip(), re.I):
        lines = lines[1:]
    return "\n".join(lines).strip() + "\n"


def run_node(node_text: str, example: Example) -> dict[str, Any]:
    """Run one candidate node on one document exactly as production does. Returns a runner row."""
    doc = runner.Doc(label=example.name, doc_id=runner.doc_id_for(example.name, example.markdown),
                     markdown=example.markdown)
    spec = runner.FieldSpec(name=FIELD_NAME, prompt=node_text.strip())
    agent = runner.build_agent(runner.build_system_prompt(runner.SYSTEM_PROMPT, doc.markdown), doc.doc_id, TASK_SETTINGS)
    try:
        result: Any = asyncio.run(agent.ainvoke(runner.field_message(spec)))
    except Exception as exc:  # noqa: BLE001 - becomes a 0 score with the error as feedback
        result = exc
    return runner.to_row(doc, spec, result)


def make_evaluator(seed: str, corpus: list[str]) -> Callable[[str, Example], tuple[float, dict[str, Any]]]:
    def evaluate(candidate: str, example: Example) -> tuple[float, dict[str, Any]]:
        node = clean_candidate(candidate)
        multiplier, penalty, issues = guard_candidate(node, seed, corpus)
        if multiplier == 0.0:  # do not spend a model call on a candidate that cannot be accepted
            return 0.0, {"example": example.description, "guard_issues": issues,
                         "scores": {"task": 0.0, "generalization": 0.0}}

        task_scores, rows, diffs = [], [], []
        for _ in range(max(1, TASK_TRIALS)):
            row = run_node(node, example)
            rows.append(row)
            if row["status"] != "ok":
                task_scores.append(0.0)
                diffs.append({"error": row["error"]})
                continue
            s, d = score_answer(example.expected, {"found": row["found"], "values": row["values"]})
            task_scores.append(s)
            diffs.append(d)
        task = sum(task_scores) / len(task_scores)
        generalization = max(0.0, multiplier - penalty)
        score = max(0.0, min(1.0, task * multiplier - penalty))
        last = rows[-1]
        side_info = {
            "example": example.description,
            "scores": {"task": task, "generalization": generalization},
            "expected": example.expected,
            "actual": {"found": last.get("found"), "values": last.get("values")},
            "diff": diffs[-1],
            "model_reasoning": last.get("reasoning"),
            "source_quotes": [v.get("source_quote") for v in (last.get("values") or [])],
            "guard_issues": issues or ["none"],
            "cached_tokens": last.get("cached_tokens"),
        }
        return score, side_info

    return evaluate


# =========================================================================== #
# Reflection prompt (anti-overfitting instructions)
# =========================================================================== #
def build_reflection_template(seed: str) -> str:
    original_failure = json.dumps({"expected": normalize_answer(EXPECTED_VALUE),
                                   "actual": normalize_answer(ACTUAL_VALUE)}, indent=2)
    protected = "\n".join(f"  {ln}" for ln in _protected_lines(seed))
    return f"""You improve a knowledge node. A knowledge node is a markdown instruction that tells a \
language model how to find ONE field (`{FIELD_NAME}`) in a document. The same knowledge node is used on \
MANY documents of this type: other companies, other values, other dates, other layouts, other units. \
Your change must make the node better for all of those documents, not only for the documents in the data.

The current knowledge node:
```
<curr_param>
```

Evaluation data. Each case shows the expected answer, the actual answer, the difference, the model \
reasoning, the source quotes, and guard issues. Some cases are variants of one document with changed \
amounts, years or column order. They exist to detect overfitting:
```
<side_info>
```

The reported failure that started this optimisation:
```
{original_failure}
```

A reviewer agent diagnosed the failure. Treat it as a HYPOTHESIS. Accept a point only if the evaluation \
data supports it:
```
{REVIEWER_DIAGNOSIS.strip()}
```

How to change the node:
1. Find the ROOT CAUSE of the wrong answers. Fix the cause with a GENERAL rule that is true for every \
document of this type.
2. Make the smallest change that fixes the cause. Keep every rule that is correct. Do not remove a rule \
to fix one case.
3. Do not overfit. Never write values, amounts, dates, years, company names, column positions or text \
copied from the evaluated documents. Never write a rule that is true only for these documents. A \
candidate that contains document values gets a score of 0.
4. If you add an example to the Examples section, describe a general situation. Do not describe an \
evaluated document.
5. If the node is already correct and the error comes from the document or the model, make no change \
or only a short clarification.
6. Keep these lines exactly as they are (template headings and metadata):
{protected}
7. Keep the node at or under {MAX_NODE_WORDS} words. Keep it about one field only. Do not add roles or \
designations.
8. Write in ASD-STE100 Simplified Technical English: one instruction in each sentence, active voice, the \
condition before the instruction, at most {MAX_STE_SENTENCE_WORDS} words in each sentence.

Before you write the new node, think about this question: "Will this rule also be correct for a \
different document of this type that I have not seen?" If the answer is no, do not write the rule.

Write the complete new knowledge node in one ``` block. Do not put a language tag after the opening \
backticks. Do not use backticks inside the node except for inline code."""


# =========================================================================== #
# Run
# =========================================================================== #
def optimize(
    reflection_lm: Any = None, run_dir: Optional[Path] = None, max_metric_calls: int = MAX_METRIC_CALLS
) -> dict[str, Any]:
    """Run GEPA, apply the acceptance gate, write the report. Returns the report dict."""
    from gepa import NoImprovementStopper
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    seed = clean_candidate(CURRENT_PROMPT)
    feedback, validation = build_examples(DOCUMENT_MARKDOWN, EXPECTED_VALUE, EXTRA_LABELED_EXAMPLES)
    corpus = [e.markdown for e in feedback + validation] + [json.dumps(e.expected) for e in feedback + validation]
    run_dir = run_dir or OUT_ROOT / FIELD_NAME / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("feedback set: %s", [e.name for e in feedback])
    log.info("validation set: %s", [e.name for e in validation])

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir / "gepa_state"),  # resumable
            seed=RANDOM_SEED,
            max_metric_calls=max_metric_calls,
            candidate_selection_strategy="pareto",
            frontier_type="hybrid",
            acceptance_criterion="strict_improvement",
            cache_evaluation=True,
            parallel=MAX_WORKERS > 1,
            max_workers=MAX_WORKERS,
            raise_on_exception=False,
            track_best_outputs=True,
            display_progress_bar=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm or REFLECTION_LM,
            reflection_lm_kwargs=None if reflection_lm else REFLECTION_LM_KWARGS,
            reflection_minibatch_size=REFLECTION_MINIBATCH_SIZE,
            skip_perfect_score=True,
            perfect_score=1.0,
            reflection_prompt_template=build_reflection_template(seed),
        ),
        stop_callbacks=[NoImprovementStopper(NO_IMPROVEMENT_PATIENCE)],
    )
    result = optimize_anything(
        seed_candidate=seed,
        evaluator=make_evaluator(seed, corpus),
        dataset=feedback,
        valset=validation,
        config=config,
    )

    scores = list(result.val_aggregate_scores)
    best_idx = result.best_idx
    best = clean_candidate(result.best_candidate)
    multiplier, penalty, issues = guard_candidate(best, seed, corpus)
    gain = scores[best_idx] - scores[0]
    accepted = best_idx != 0 and gain >= MIN_GAIN and multiplier == 1.0
    if best_idx == 0:
        reason = "no candidate beat the current prompt on the validation set"
    elif gain < MIN_GAIN:
        reason = f"gain {gain:.3f} is below MIN_GAIN {MIN_GAIN}"
    elif multiplier != 1.0:
        reason = f"best candidate fails a guard: {issues}"
    else:
        reason = f"validation score {scores[0]:.3f} -> {scores[best_idx]:.3f}"
    final = best if accepted else seed

    names = [e.name for e in validation]
    report = {
        "field": FIELD_NAME,
        "accepted": accepted,
        "reason": reason,
        "seed_val_score": scores[0],
        "best_val_score": scores[best_idx],
        "per_example": {
            "seed": dict(zip(names, (result.val_subscores[0] or {}).values()))
            if isinstance(result.val_subscores[0], dict) else result.val_subscores[0],
            "best": dict(zip(names, (result.val_subscores[best_idx] or {}).values()))
            if isinstance(result.val_subscores[best_idx], dict) else result.val_subscores[best_idx],
        },
        "guard_issues_best": issues,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "feedback_set": [e.name for e in feedback],
        "validation_set": names,
    }
    (run_dir / "optimized_node.md").write_text(final, encoding="utf-8")
    (run_dir / "node.diff").write_text("".join(difflib.unified_diff(
        seed.splitlines(True), final.splitlines(True), "current_prompt.md", "optimized_node.md")), encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("%s | %s | wrote %s", "ACCEPTED" if accepted else "KEPT CURRENT PROMPT", reason, run_dir)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "openai", "httpcore", "LiteLLM", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY is not set")
        return 2
    optimize()
    return 0


if __name__ == "__main__":
    sys.exit(main())


# =========================================================================== #
# Tests (no network): pytest gepa_optimize_node.py
# =========================================================================== #
def test_normalisation_and_scoring() -> None:
    assert norm_value("(4,212)") == "-4212" and norm_value("$1,000.50") == "1000.5"
    assert norm_date("December 31, 2024") == norm_date("2024-12-31") == "2024-12-31"
    exact, _ = score_answer(EXPECTED_VALUE, EXPECTED_VALUE)
    partial, diff = score_answer(EXPECTED_VALUE, ACTUAL_VALUE)
    assert exact == 1.0 and 0 < partial < 1
    assert diff["wrong"] and len(diff["missing"]) == 2
    assert score_answer({"found": False, "values": []}, {"found": False, "values": []})[0] == 1.0
    assert score_answer({"found": False, "values": []}, EXPECTED_VALUE)[0] == 0.0
    assert score_answer("41622", {"found": True, "values": [{"name": "x", "value": "41,622"}]})[0] == 1.0


def test_variants_preserve_labels() -> None:
    scaled = scale_amounts(DOCUMENT_MARKDOWN, EXPECTED_VALUE, 1.5)
    assert scaled is not None
    md, gold = scaled
    for v in gold["values"]:  # every scaled gold value is printed in the scaled document
        assert f"{int(v['value']):,}" in md
        assert v["value"] not in {x["value"] for x in EXPECTED_VALUE["values"]}
    assert "2024" in md  # years are not scaled

    shifted, sgold = shift_years(DOCUMENT_MARKDOWN, EXPECTED_VALUE, -5)
    assert "2019" in shifted and "2024" not in shifted
    assert {v["as_of"] for v in sgold["values"]} == {"December 31, 2019", "December 31, 2018"}

    reversed_md = reverse_table_columns(DOCUMENT_MARKDOWN)
    assert "| | 2023 | 2024 |" in reversed_md and "| **Total current assets** | **185,505** | **209,291** |" in reversed_md

    computed = {"found": True, "values": [{"name": "working_capital", "value": "124396"}]}
    assert scale_amounts(DOCUMENT_MARKDOWN, computed, 1.5) is None  # not copied -> unsafe label

    feedback, validation = build_examples(DOCUMENT_MARKDOWN, EXPECTED_VALUE, [])
    assert len(feedback) == 2 and len(validation) == 4
    assert not {e.name for e in feedback} - {DOCUMENT_NAME} & {e.name for e in validation}  # variants held out


def test_guards_block_overfitting() -> None:
    corpus = [DOCUMENT_MARKDOWN, json.dumps(EXPECTED_VALUE)]
    seed = clean_candidate(CURRENT_PROMPT)
    assert guard_candidate(seed, seed, corpus) == (1.0, 0.0, [])

    leaky = seed + "\n- If current liabilities is 84,895, use it.\n"
    assert guard_candidate(leaky, seed, corpus)[0] == 0.0

    no_heading = seed.replace("## Rules\n", "")
    assert guard_candidate(no_heading, seed, corpus)[0] == 0.5

    copied = seed + "\n- Use the line accrued compensation and benefits current portion of long term debt.\n"
    assert guard_candidate(copied, seed, corpus)[0] == 0.7

    too_long = seed + "\n" + "word " * 600
    assert guard_candidate(too_long, seed, corpus)[0] == 0.0

    general = seed.replace("## Rules\n", "## Rules\n- Read every date column of the balance sheet.\n")
    assert guard_candidate(general, seed, corpus)[0] == 1.0


def test_gepa_end_to_end_with_fake_models(tmp_path: Path, monkeypatch: Any) -> None:
    """Real GEPA loop; fake task model and fake reflection model. The leaky proposal must lose."""
    fix = "- Read every date column. Copy the subtotal that has the exact label."

    def fake_run_node(node_text: str, example: Example) -> dict[str, Any]:
        good = fix in node_text
        values = example.expected["values"] if good else example.expected["values"][:1]
        return {"status": "ok", "found": True, "values": values, "reasoning": "fake",
                "error": None, "cached_tokens": 0}

    proposals = iter([
        "LEAK",  # first proposal copies a gold value
        "GOOD",
    ])

    def fake_reflection_lm(prompt: Any) -> str:
        text = prompt if isinstance(prompt, str) else json.dumps(prompt)
        assert "HYPOTHESIS" in text and "Do not overfit" in text
        current = text.split("```\n", 1)[1].split("\n```", 1)[0]
        kind = next(proposals, "GOOD")
        extra = "- Use the value 84,895." if kind == "LEAK" else fix
        return "```\n" + current.replace("## Rules\n", f"## Rules\n{extra}\n", 1) + "\n```"

    monkeypatch.setattr(sys.modules[__name__], "run_node", fake_run_node)
    monkeypatch.setattr(sys.modules[__name__], "MAX_WORKERS", 1)
    report = optimize(reflection_lm=fake_reflection_lm, run_dir=tmp_path / "run", max_metric_calls=30)

    assert report["accepted"], report
    assert report["best_val_score"] == 1.0 and report["seed_val_score"] < 1.0
    final = (tmp_path / "run" / "optimized_node.md").read_text()
    assert fix in final and "84,895" not in final
    assert "## Rules" in final and "- **Node ID:** FIN_07" in final
    assert (tmp_path / "run" / "node.diff").read_text().count("+" + fix) == 1
