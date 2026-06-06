"""
Gap-report schema, helpers, and persistence.

Centralizes:
  - GapResolution, KnowledgeGapReport dataclasses
  - extract_target_predicates, find_predicate_arities, kb_declares_predicate
  - verify_gap_resolution (shallow schema-presence check, arity-drift reporting)
  - load_gap_report / write_gap_report (JSON persistence)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

_PRED_ARITY_RE = re.compile(r'\b([a-z][a-zA-Z0-9_]*)/([0-9]+)\b')
_HEAD_RE = re.compile(
    r'^\s*([a-z][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*\??\.?\s*$',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Gap-report schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GapResolution:
    """
    Result of verify_gap_resolution. Records which targeted predicates
    were found in the augmented KB and which remain missing.

    related_predicates_found maps a requested pred/arity to a list of
    same-name predicates with differing arities found in the KB.
    E.g. {"profit/2": ["profit/3"]}
    Empty dict means no arity drift was detected.
    """
    predicates_checked: List[str]
    predicates_found: List[str]
    predicates_still_missing: List[str]
    predicates_dropped: List[str]
    related_predicates_found: Dict[str, List[str]]
    all_resolved: bool


@dataclass
class KnowledgeGapReport:
    """
    DSL-MAP: ORCHESTRATOR-ARTIFACT-KNOWLEDGE-GAP-REPORT

    Orchestrator-level artifact produced by ask --gap-report and consumed/
    updated by learn --fill-gap. NOT a flow artifact — does not participate
    in flow-step routing or ArtifactRegistry.

    Fields:
      question          — the question that was asked
      committed_answer  — the final committed answer
      gap_rationale     — human-readable text from the answer's rationale
                          describing what KB support is missing
      target_predicates — structured list of "pred/arity" strings extracted
                          from gap_rationale. This is the canonical form
                          used by verify_gap_resolution and gap_context.
      target_review     — optional dict for human correction without
                          rewriting target_predicates.
                          {"drop": ["pred/arity", ...]}
      status            — "open" (gaps identified) or "closed" (all
                          non-dropped targeted predicates now appear)
      resolution        — populated by learn --fill-gap after integration
    """
    question: str
    committed_answer: str
    gap_rationale: str
    target_predicates: List[str]
    status: Literal["open", "closed"]
    target_review: Optional[dict] = None
    resolution: Optional[GapResolution] = None


# ---------------------------------------------------------------------------
# Predicate-scan helpers
# ---------------------------------------------------------------------------

def extract_target_predicates(gap_rationale: str) -> List[str]:
    """
    DSL-MAP: ORCHESTRATOR-UTILITY-EXTRACT-TARGET-PREDICATES

    Extract "pred/arity" strings from freeform prose text.
    Used as fallback when target_predicates is empty or for
    backward compatibility with older gap files.
    """
    seen: set[str] = set()
    result: List[str] = []
    for name, arity in _PRED_ARITY_RE.findall(gap_rationale or ""):
        key = f"{name}/{arity}"
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _head_arity(line: str, name: str) -> Optional[int]:
    """
    Given an ASP source line and a predicate name, return the arity
    if the line's head matches that name, else None.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("%") or stripped.startswith("#"):
        return None
    if " :-" in stripped or ":- " in stripped:
        head = stripped.split(":-", 1)[0].strip()
    else:
        head = stripped.rstrip(".").strip()
    m = re.match(r"^([a-z][a-zA-Z0-9_]*)\s*\((.*)\)$", head)
    if not m:
        return None
    if m.group(1) != name:
        return None
    args_str = m.group(2).strip()
    arity = len([a.strip() for a in args_str.split(",") if a.strip()]) if args_str else 0
    return arity


def find_predicate_arities(kb_program: str, name: str) -> set:
    """
    Return the set of all arities at which a predicate `name` appears
    as a head-of-rule or fact in the KB.
    """
    arities: set = set()
    for line in kb_program.splitlines():
        a = _head_arity(line, name)
        if a is not None:
            arities.add(a)
    return arities


def kb_declares_predicate(kb_program: str, pred_sig: str) -> bool:
    """
    DSL-MAP: ORCHESTRATOR-UTILITY-KB-DECLARES-PREDICATE

    Check whether a predicate signature "name/arity" appears as a fact
    or rule head anywhere in the KB ASP program at the correct arity.

    Deterministic KB-text scan. Not LLM-backed, not Clingo-solver-backed.
    """
    name, arity_s = pred_sig.split("/", 1)
    target_arity = int(arity_s)
    for line in kb_program.splitlines():
        a = _head_arity(line, name)
        if a is not None and a == target_arity:
            return True
    return False


def verify_gap_resolution(
    report: KnowledgeGapReport,
    kb_program: str,
) -> GapResolution:
    """
    DSL-MAP: ORCHESTRATOR-UTILITY-VERIFY-GAP-RESOLUTION

    Deterministic predicate-presence check with arity-drift reporting
    and target_review.drop support.

    Predicates listed in target_review.drop are excluded from
    resolution checks. The original target_predicates list is
    preserved for traceability; dropped targets are recorded in
    predicates_dropped and not treated as unresolved.

    Returns a GapResolution with the results. Does NOT modify the
    KnowledgeGapReport itself — the caller updates and persists it.
    """
    predicates = report.target_predicates or extract_target_predicates(report.gap_rationale)
    dropped_set: set = set()
    if report.target_review and isinstance(report.target_review.get("drop"), list):
        dropped_set = set(report.target_review["drop"])
    found: List[str] = []
    missing: List[str] = []
    dropped: List[str] = []
    related: Dict[str, List[str]] = {}
    for pred in predicates:
        if pred in dropped_set:
            dropped.append(pred)
            continue
        name, arity_s = pred.split("/", 1)
        target_arity = int(arity_s)
        if kb_declares_predicate(kb_program, pred):
            found.append(pred)
        else:
            missing.append(pred)
            # Check for same-name predicates with different arities
            other_arities = find_predicate_arities(kb_program, name)
            drift = [f"{name}/{a}" for a in sorted(other_arities) if a != target_arity]
            if drift:
                related[pred] = drift
    return GapResolution(
        predicates_checked=predicates,
        predicates_found=found,
        predicates_still_missing=missing,
        predicates_dropped=dropped,
        related_predicates_found=related,
        all_resolved=len(missing) == 0,
    )


# ---------------------------------------------------------------------------
# Gap-report persistence
# ---------------------------------------------------------------------------

_GAP_RESOLUTION_FIELDS = (
    "predicates_checked",
    "predicates_found",
    "predicates_still_missing",
    "predicates_dropped",
    "related_predicates_found",
    "all_resolved",
)


def load_gap_report(path: str) -> KnowledgeGapReport:
    """Load and validate a KnowledgeGapReport from a JSON file on disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    res_data = data.get("resolution")
    resolution = None
    if res_data:
        resolution = GapResolution(
            predicates_checked=res_data.get("predicates_checked", []),
            predicates_found=res_data.get("predicates_found", []),
            predicates_still_missing=res_data.get("predicates_still_missing", []),
            predicates_dropped=res_data.get("predicates_dropped", []),
            related_predicates_found=res_data.get("related_predicates_found", {}),
            all_resolved=res_data.get("all_resolved", False),
        )
    return KnowledgeGapReport(
        question=data.get("question", ""),
        committed_answer=data.get("committed_answer", ""),
        gap_rationale=data.get("gap_rationale", ""),
        target_predicates=data.get("target_predicates", []),
        target_review=data.get("target_review"),
        status=data.get("status", "open"),
        resolution=resolution,
    )


def write_gap_report(path: str, report: KnowledgeGapReport) -> None:
    """Serialize and write a KnowledgeGapReport to a JSON file on disk."""
    data: dict = {
        "question": report.question,
        "committed_answer": report.committed_answer,
        "gap_rationale": report.gap_rationale,
        "target_predicates": report.target_predicates,
        "status": report.status,
    }
    if report.target_review is not None:
        data["target_review"] = report.target_review
    if report.resolution:
        data["resolution"] = {
            field: getattr(report.resolution, field)
            for field in _GAP_RESOLUTION_FIELDS
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
