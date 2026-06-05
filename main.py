#!/usr/bin/env python3
"""
LiveKnowledge
main.py — 1:1 Python translation of system_model.yaml v2.1.

v2.1 changes from v0.2:
- verify_candidate_knowledge now checks MERGED coherence (kb ⊕ candidate)
  instead of local satisfiability of the candidate in isolation. A candidate
  that is locally satisfiable but contradicts the existing KB is rejected.
  This is the key semantic fix; the rest of the integrate_knowledge flow
  (abduce / revise loop) handles the rejection unchanged.
- revise is split into revise_answer and revise_knowledge as separate
  primitives (was already separate in code; the v0.2 YAML under-specified).
- integrate is sharpened: it applies a verified candidate to the KB and
  produces a new immutable KnowledgeBase snapshot, with provenance metadata
  recorded in the new KB.

What this file implements (v2.1):
- Artifact dataclasses (Question, KnowledgeBase, CandidateAnswer, CandidateKnowledge,
  Critique, VerificationReport, AbductiveHypothesis, FinalAnswer, IntegrationResult,
  LoopStatus) — all frozen, kw_only, with field(default_factory=...) for mutables.
- Per-run ArtifactRegistry with collision-raise.
- Deterministic id_factory (Python-assigned IDs; LLM returns content only).
- LLM-backed primitives: generate_answer, verify_candidate_answer, abduce_answer,
  abduce_knowledge, revise_answer, revise_knowledge.
- Clingo-backed primitive: verify_candidate_knowledge with MERGED-coherence semantics.
- Verification failure classifier (classify_verification_failure / is_recoverable_failure).
- Hard-coded flows: answer_question, integrate_knowledge, with max_iterations + FlowFailed.
- Thin Orchestrator façade + CLI.

DSL-map conventions follow main.py so future cross-references remain stable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional, Protocol, Tuple

# ---------------------------------------------------------------------------
# Gap report schema (orchestrator artifact, not a flow artifact)
# ---------------------------------------------------------------------------


@dataclass
class GapResolution:
    """
    DSL-MAP: ORCHESTRATOR-ARTIFACT-GAP-RESOLUTION

    Result of verify_gap_resolution. Records which targeted predicates
    were found in the augmented KB and which remain missing.

    related_predicates_found maps a requested pred/arity to a list of
    same-name predicates with differing arities found in the KB.
    E.g. {"profit/2": ["profit/3"]}
    Empty dict means no arity drift was detected.
    """
    predicates_checked: list[str]
    predicates_found: list[str]
    predicates_still_missing: list[str]
    predicates_dropped: list[str]
    related_predicates_found: dict[str, list[str]]
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
    target_predicates: list[str]
    status: Literal["open", "closed"]
    target_review: Optional[dict] = None
    resolution: Optional[GapResolution] = None


# Regex for extracting pred/n patterns from prose text (fallback)
_PRED_ARITY_RE = re.compile(r'\b([a-z][a-zA-Z0-9_]*)/([0-9]+)\b')
# Regex for matching ASP head atoms: pred(arg, ...) at start of line/head
_HEAD_RE = re.compile(
    r'^\s*([a-z][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*\??\.?\s*$',
    re.MULTILINE,
)

import clingo
from dotenv import load_dotenv
import openai
from openai import OpenAI
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

# ---------------------------------------------------------------------------
# Environment / Logging
# ---------------------------------------------------------------------------

load_dotenv()

MODEL = os.getenv("LLM_MODEL")
BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
MAX_JSON_RETRIES = int(os.getenv("MAX_JSON_RETRIES", "2"))
MAX_MODELS = int(os.getenv("MAX_MODELS", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "0"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_RAW_LLM = os.getenv("LOG_RAW_LLM", "0") == "1"
DEFAULT_MAX_ITERATIONS = int(os.getenv("DEFAULT_MAX_ITERATIONS", "3"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("asplearning.main2")


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------


class FlowFailed(Exception):
    """
    DSL-MAP: FAIL-FLOW-EXCEPTION

    Raised by fail_flow() and by flow loops when termination conditions
    (iterations_exceeded, unrecoverable_verification_failure) trigger.
    Encodes I-QUESTION-TERMINATES as an exception-driven escape.
    """


class LLMArtifactError(Exception):
    """
    DSL-MAP: LLM-ARTIFACT-ERROR

    Raised when an LLM-produced artifact cannot be parsed, validated, or
    recovered after the configured retry budget. Stores the offending raw
    text so callers can log or surface it.
    """

    def __init__(self, message: str, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


# ---------------------------------------------------------------------------
# HasId protocol + ArtifactRegistry
# ---------------------------------------------------------------------------


class HasId(Protocol):
    """Anything with a string id is registrable."""
    id: str


@dataclass
class ArtifactRegistry:
    """
    DSL-MAP: ARTIFACT-REGISTRY

    Per-run registry of artifacts. Raises ValueError on id collision to catch
    bugs early; revisions always produce new IDs so collisions indicate logic
    errors.
    """

    _artifacts: dict[str, HasId] = field(default_factory=dict)

    def register(self, artifact: HasId) -> None:
        if artifact.id in self._artifacts:
            raise ValueError(
                f"Artifact with id '{artifact.id}' already registered"
            )
        self._artifacts[artifact.id] = artifact
        logger.debug(
            "Registered artifact id=%s type=%s",
            artifact.id,
            type(artifact).__name__,
        )

    def get(self, artifact_id: str) -> HasId:
        if artifact_id not in self._artifacts:
            raise ValueError(
                f"Artifact '{artifact_id}' does not exist in registry"
            )
        return self._artifacts[artifact_id]

    def has(self, artifact_id: str) -> bool:
        return artifact_id in self._artifacts

    def __len__(self) -> int:
        return len(self._artifacts)

    def ids(self) -> List[str]:
        return list(self._artifacts.keys())


def make_id_factory(prefix: str) -> Callable[[], str]:
    """
    DSL-MAP: ID-FACTORY

    Deterministic, monotonic id generator with a per-prefix counter. The LLM
    is never trusted to assign ids; Python assigns them after validation.
    """
    counter = {"n": 0}

    def factory() -> str:
        counter["n"] += 1
        return f"{prefix}-{counter['n']:03d}"

    return factory


# ---------------------------------------------------------------------------
# Artifact dataclasses (kw_only, frozen, factory defaults)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Question:
    """
    DSL-MAP: ARTIFACT-QUESTION

    The user query driving the answer_question flow.
    """
    id: str
    text: str


@dataclass(frozen=True, kw_only=True)
class KnowledgeBase:
    """
    DSL-MAP: ARTIFACT-KNOWLEDGE-BASE

    Real object with provenance (source_text, metadata), not just a string.
    asp_program is the cumulative Clingo program the KB represents.
    """
    id: str
    source_text: str
    asp_program: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class CandidateAnswer:
    """
    DSL-MAP: ARTIFACT-CANDIDATE-ANSWER

    A proposed answer produced by generate_answer or revise_answer. Every
    revision is a new immutable snapshot with a new id.
    """
    id: str
    question_id: str
    text: str
    rationale: str = ""


@dataclass(frozen=True, kw_only=True)
class CandidateKnowledge:
    """
    DSL-MAP: ARTIFACT-CANDIDATE-KNOWLEDGE (v2.1)

    A proposed ADDITIVE ASP fragment. v2.1: candidate_knowledge is strictly
    additive — it is an ASP program to be merged with the existing KB. There
    is no native retraction / deletion semantic; contradictions are handled
    by the verify_knowledge step rejecting the candidate and the abduce /
    revise loop rewriting the fragment to drop or fix the offending facts.
    """
    id: str
    asp_program: str
    notes: str = ""


@dataclass(frozen=True, kw_only=True)
class Critique:
    """
    DSL-MAP: ARTIFACT-CRITIQUE

    Produced alongside an abductive_hypothesis. The target_id must refer
    to an artifact already in the registry (I-CRITIQUE-MUST-TARGET-EXISTING-ARTIFACT).
    """
    id: str
    target_id: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class VerificationReport:
    """
    DSL-MAP: ARTIFACT-VERIFICATION-REPORT

    status:
        - "verified" : claim accepted
        - "rejected" : claim semantically rejected (eligible for revision)
        - "failed"   : verifier transport/runtime error (classify further)
    verifier_kind:
        - "llm"    : produced by verify_candidate_answer
        - "clingo" : produced by verify_candidate_knowledge
    """
    id: str
    status: Literal["verified", "rejected", "failed"]
    reason: str
    evidence: str = ""
    raw_output: str = ""
    verifier_kind: Literal["llm", "clingo"] = "llm"


@dataclass(frozen=True, kw_only=True)
class AbductiveHypothesis:
    """
    DSL-MAP: ARTIFACT-ABDUCTIVE-HYPOTHESIS

    A best-guess explanation of why the candidate was rejected/failed, plus
    a concrete repair plan the next revision should follow.
    """
    id: str
    explanation: str
    repair_plan: str


@dataclass(frozen=True, kw_only=True)
class FinalAnswer:
    """
    DSL-MAP: ARTIFACT-FINAL-ANSWER

    Produced by commit_answer, which enforces I-COMMIT-ANSWER-ONLY-IF-VERIFIED.
    """
    id: str
    question_id: str
    answer_text: str


@dataclass(frozen=True, kw_only=True)
class IntegrationResult:
    """
    DSL-MAP: ARTIFACT-INTEGRATION-RESULT

    Wraps the outcome of merging verified knowledge into a KB. updated_program
    is the new full program string (or the unchanged one on failure).
    """
    id: str
    success: bool
    message: str
    updated_program: str


@dataclass(frozen=True, kw_only=True)
class LoopStatus:
    """
    DSL-MAP: ARTIFACT-LOOP-STATUS

    Snapshot of loop state. bounded is always True in main2 (hard-coded
    max_iterations). escaped is True if the loop exited via FlowFailed or
    commit/integrate before iteration cap.
    """
    flow_id: str
    iteration: int
    max_iterations: int
    bounded: bool
    escaped: bool


# ---------------------------------------------------------------------------
# Gap report helpers
# ---------------------------------------------------------------------------


def extract_target_predicates(gap_rationale: str) -> list[str]:
    """
    DSL-MAP: ORCHESTRATOR-UTILITY-EXTRACT-TARGET-PREDICATES

    Extract "pred/arity" strings from freeform prose text.
    Used as fallback when target_predicates is empty or for
    backward compatibility with older gap files.
    """
    seen: set[str] = set()
    result: list[str] = []
    for name, arity in _PRED_ARITY_RE.findall(gap_rationale or ""):
        key = f"{name}/{arity}"
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _head_arity(line: str, name: str) -> int | None:
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


def find_predicate_arities(kb_program: str, name: str) -> set[int]:
    """
    Return the set of all arities at which a predicate `name` appears
    as a head-of-rule or fact in the KB.
    """
    arities: set[int] = set()
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
    dropped_set: set[str] = set()
    if report.target_review and isinstance(report.target_review.get("drop"), list):
        dropped_set = set(report.target_review["drop"])
    found: list[str] = []
    missing: list[str] = []
    dropped: list[str] = []
    related: dict[str, list[str]] = {}
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
    data: dict[str, Any] = {
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
            "predicates_checked": report.resolution.predicates_checked,
            "predicates_found": report.resolution.predicates_found,
            "predicates_still_missing": report.resolution.predicates_still_missing,
            "predicates_dropped": report.resolution.predicates_dropped,
            "related_predicates_found": report.resolution.related_predicates_found,
            "all_resolved": report.resolution.all_resolved,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def get_client() -> OpenAI:
    """
    DSL-MAP: ROLE-ACCESS-INFRASTRUCTURE

    Returns the LLM client. Mirrors main.py initialization so we reuse the
    same .env contract (LLM_API_KEY, LLM_MODEL, LLM_BASE_URL).
    """
    if not LLM_API_KEY:
        raise RuntimeError("Missing LLM_API_KEY environment variable.")
    kwargs: dict[str, Any] = {"api_key": LLM_API_KEY}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    logger.info(
        "Initializing LLM client model=%s base_url=%s",
        MODEL,
        BASE_URL or "<default>",
    )
    return OpenAI(**kwargs)


# --- Robust JSON extraction ------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def extract_json(text: str) -> dict:
    """
    DSL-MAP: INTERMEDIATE-ARTIFACT-NORMALIZATION

    Best-effort extraction of a JSON object from LLM output. Tries:
      1. Direct json.loads
      2. Strip ```json fences
      3. Slice between first '{' and last '}'
    Raises LLMArtifactError on total failure.
    """
    text = text.strip()
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    m = _FENCE_RE.search(text)
    if m:
        snippet = m.group(1).strip()
        try:
            loaded = json.loads(snippet)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            loaded = json.loads(snippet)
            if isinstance(loaded, dict):
                logger.warning("JSON recovered via {…} slice only")
                return loaded
        except json.JSONDecodeError as e2:
            raise LLMArtifactError(
                f"JSON parse failed after slicing: {e2}", raw_text=text
            ) from e2

    raise LLMArtifactError(f"JSON parse failed: no object in text", raw_text=text)


# --- Pydantic schemas for the LLM boundary ---------------------------------


class _CandidateAnswerContent(BaseModel):
    question_id: str = ""
    text: str
    rationale: str = ""


class _AnswerVerifyContent(BaseModel):
    status: Literal["verified", "rejected", "failed"]
    reason: str
    evidence: str = ""


class _CritiqueContent(BaseModel):
    target_id: str
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class _AbductionContent(BaseModel):
    explanation: str
    repair_plan: str
    critique: _CritiqueContent


class _ReviseAnswerContent(BaseModel):
    question_id: str = ""
    text: str
    rationale: str = ""


class _CandidateKnowledgeContent(BaseModel):
    asp_program: str
    notes: str = ""


# TypeAdapters (validate model dicts at the LLM boundary).
_TA_CAND_ANSWER = TypeAdapter(_CandidateAnswerContent)
_TA_ANSWER_VERIFY = TypeAdapter(_AnswerVerifyContent)
_TA_ABDUCTION = TypeAdapter(_AbductionContent)
_TA_REVISE_ANSWER = TypeAdapter(_ReviseAnswerContent)
_TA_CAND_KNOWLEDGE = TypeAdapter(_CandidateKnowledgeContent)


# --- LLM call helpers ------------------------------------------------------


def _call_llm_raw(
    client: OpenAI, system_prompt: str, user_prompt: str, label: str
) -> str:
    """Thin transport wrapper; returns the raw text output."""
    logger.info(
        "Calling LLM label=%s model=%s prompt_chars=%d",
        label,
        MODEL,
        len(system_prompt) + len(user_prompt),
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except openai.APIError:
        logger.exception("LLM API error label=%s", label)
        raise
    except Exception:
        logger.exception("LLM request failed label=%s", label)
        raise

    # chat.completions: response.choices[0].message.content
    raw_text = (response.choices[0].message.content or "") if response.choices else ""
    if LOG_RAW_LLM:
        logger.debug("Raw LLM output label=%s:\n%s", label, raw_text)
    return raw_text


def _call_llm_json(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    adapter: TypeAdapter,
    label: str,
) -> BaseModel:
    """
    Call the LLM, extract JSON, and validate it through the supplied Pydantic
    TypeAdapter. Retries malformed JSON up to MAX_JSON_RETRIES times; raises
    LLMArtifactError on total failure.
    """
    last_error: Optional[Exception] = None
    last_raw: str = ""
    for attempt in range(1, MAX_JSON_RETRIES + 2):  # 1 initial + N retries
        try:
            raw = _call_llm_raw(client, system_prompt, user_prompt, label=label)
            last_raw = raw
            data = extract_json(raw)
            return adapter.validate_python(data)
        except (LLMArtifactError, ValidationError) as e:
            last_error = e
            logger.warning(
                "LLM JSON validation failed label=%s attempt=%d error=%s",
                label,
                attempt,
                e,
            )
            if attempt <= MAX_JSON_RETRIES:
                if RETRY_DELAY > 0:
                    time.sleep(RETRY_DELAY)
                continue
    raise LLMArtifactError(
        f"LLM JSON could not be recovered after {MAX_JSON_RETRIES + 1} attempts: {last_error}",
        raw_text=last_raw,
    )


# ---------------------------------------------------------------------------
# System prompts (artifacts: content only, no id field)
# ---------------------------------------------------------------------------

GENERATE_ANSWER_SYSTEM = """
You propose an answer to a user's question using a provided knowledge base.
You return ONLY a JSON object with this exact shape and NO other text:

{
  "question_id": "string (echoed or empty)",
  "text": "string — the proposed answer",
  "rationale": "string — justification grounded in the KB, plus any KB facts
                or predicate domains that are MISSING and would improve this
                answer. Name missing predicates as `pred/arity` or topic
                areas the KB does not cover."
}

Rules:
- Do NOT include any "id" field. The host will assign the artifact id.
- Ground the answer in the provided knowledge base when possible.
- If the knowledge base is insufficient, say so honestly in text and rationale.
  List the specific predicates or topics that are absent and would help.
- No markdown fences, no prose outside JSON.
"""

VERIFY_ANSWER_SYSTEM = """
You verify a proposed answer against a knowledge base.

Return ONLY a JSON object with this exact shape and NO other text:

{
  "status": "verified" | "rejected" | "failed",
  "reason": "string — concise explanation",
  "evidence": "string — quoted facts/rules from the KB that support your call"
}

Rules:
- "verified"  : the answer is supported by the KB and is internally consistent.
- "rejected"  : the answer contradicts the KB or makes unsupported claims.
                (An empty KB is also a "rejected" — there is no evidence to
                support the answer, not a verifier failure.)
- "failed"    : you could not evaluate the answer (e.g., the question is
                nonsensical, the answer text is malformed/garbled, or the
                KB is so corrupted that no semantic check is possible).
- Do NOT include any "id" field. The host will assign the artifact id.
- No markdown fences, no prose outside JSON.
"""

ABDUCE_SYSTEM = """
You play the abductive revisor. Given a rejected or failed candidate and the
verification report, you produce a best-guess explanation of the failure and
a concrete repair plan. You also produce a critique that targets the existing
candidate artifact.

Return ONLY a JSON object with this exact shape and NO other text:

{
  "explanation": "string — why the candidate was rejected/failed",
  "repair_plan": "string — concrete steps the next revision must take",
  "critique": {
    "target_id": "string — id of the candidate artifact being critiqued",
    "issues":    ["string", ...],
    "suggestions": ["string", ...]
  }
}

Rules:
- Do NOT include any top-level "id" fields. The host will assign ids.
- target_id must be the id of an existing candidate artifact (provided to you).
- No markdown fences, no prose outside JSON.
"""

REVISE_ANSWER_SYSTEM = """
You revise a previously rejected/failed candidate answer using the abductive
hypothesis and critique.

Return ONLY a JSON object with this exact shape and NO other text:

{
  "question_id": "string (echoed or empty)",
  "text": "string — the revised answer",
  "rationale": "string — how this revision addresses the critique"
}

Rules:
- Do NOT include any "id" field. The host will assign a new artifact id.
- Address every issue raised in the critique.
- Stay grounded in the knowledge base.
- No markdown fences, no prose outside JSON.
"""

REVISE_KNOWLEDGE_SYSTEM = """
You revise a Clingo ASP program using the abductive hypothesis and critique.

Return ONLY a JSON object with this exact shape and NO other text:

{
  "asp_program": "string — the full revised clingo program",
  "notes":       "string — modeling notes / uncertainty"
}

Rules:
- Do NOT include any "id" field. The host will assign a new artifact id.
- The asp_program MUST be valid clingo syntax (facts and rules only).
- Do NOT include #show directives — they are query-specific and added
  per-run, not stored in the KB.
- Do NOT restate existing KB facts or rules. Output only the NEW or
  CHANGED fragment.
- Address every issue raised in the critique.
- No markdown fences, no prose outside JSON.
"""

GENERATE_KNOWLEDGE_SYSTEM = """
You extract a small, ADDITIVE ASP program fragment from unstructured source text.
The fragment will be merged with an existing knowledge base, so it must not
restate or contradict the KB; it must be coherent under the closed-world
assumption.

Return ONLY a JSON object with this exact shape and NO other text:

{
  "asp_program": "string — the additive ASP fragment",
  "notes":       "string — modeling notes / uncertainty"
}

Rules:
- Do NOT include any "id" field. The host will assign a new artifact id.
- Output ONLY new facts and rules. Do NOT include #show directives — they
  are query-specific and added per-run, not stored in the KB.
- Do NOT restate the contents of the knowledge base, even as comments.
  Output only ADDITIVE facts and rules that are NOT already in the KB.
- Prefer reusing predicates and constants already present in the knowledge
  base. Only invent new predicates when the text clearly requires them.
- Every variable in a rule must be safe (bound by a positive atom in the body).
- Use only valid clingo syntax (facts `pred(args).`, rules `head :- body.`).
- Avoid integrity constraints (`:- body.`) unless the text explicitly forbids
  some combination, and even then keep them minimal and well-grounded.
- Keep the fragment small and grounded in the source text. Do not speculate
  beyond what is stated or strongly implied.
- notes should briefly explain modeling choices, vocabulary reuse, or
  uncertainty.
- No markdown fences, no prose outside JSON.
"""

# ---------------------------------------------------------------------------
# Clingo adapter
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    """
    DSL-MAP: CLINGO-SOLVE-RESULT

    Low-level solver outcome. verify_candidate_knowledge maps this into a
    VerificationReport (verifier_kind="clingo").
    """
    ok: bool
    satisfiable: Optional[bool]
    optimal: Optional[bool]
    models: List[List[str]]
    cost: List[int]
    error: str
    program: str


def clingo_solve(asp_program: str, max_models: int = MAX_MODELS) -> SolveResult:
    """
    DSL-MAP: CLINGO-ADAPTER

    Loads, grounds, and solves a clingo program. Returns a SolveResult with
    solver details; the caller maps this to a VerificationReport.
    """
    logger.info("Solving ASP program max_models=%d chars=%d", max_models, len(asp_program))
    try:
        ctl = clingo.Control(["-n", str(max_models)])
        ctl.add("base", [], asp_program)
        ctl.ground([("base", [])])

        models: List[List[str]] = []
        last_cost: List[int] = []
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                shown = sorted(str(sym) for sym in model.symbols(shown=True))
                models.append(shown)
                last_cost = list(model.cost)
            raw_result = handle.get()

        satisfiable = bool(raw_result.satisfiable)
        optimal = getattr(raw_result, "optimality_proven", None) if satisfiable else None
        logger.info(
            "Solve finished satisfiable=%s optimal=%s models=%d",
            satisfiable,
            optimal,
            len(models),
        )
        return SolveResult(
            ok=True,
            satisfiable=satisfiable,
            optimal=optimal,
            models=models,
            cost=last_cost,
            error="",
            program=asp_program,
        )
    except Exception as e:  # clingo raises RuntimeError on parse/ground failures
        logger.exception("Clingo solve failed")
        return SolveResult(
            ok=False,
            satisfiable=None,
            optimal=None,
            models=[],
            cost=[],
            error=str(e),
            program=asp_program,
        )


# ---------------------------------------------------------------------------
# Verification failure classifier
# ---------------------------------------------------------------------------


def classify_verification_failure(
    report: VerificationReport,
) -> Literal["recoverable", "unrecoverable"]:
    """
    Classify a 'failed' verification report as recoverable or unrecoverable.

    Checks `reason` before `raw_output` to avoid false matches from noisy
    solver/transport dumps. Only meaningful when report.status == "failed".

    LLM failures:
      - recoverable   : timeout, rate limit, connection, temporary
      - unrecoverable : api key, authentication, invalid model, not found
      - default       : UNRECOVERABLE (conservative)

    Clingo failures:
      - recoverable   : syntax error, unsafe, grounding, parse
      - unrecoverable : memory, internal error, segfault
      - default       : RECOVERABLE (typically fixable via revision)
    """
    if report.status != "failed":
        return "unrecoverable"

    text = (report.reason or report.raw_output or "").lower()

    if report.verifier_kind == "llm":
        if any(p in text for p in ["timeout", "rate limit", "connection", "temporary"]):
            return "recoverable"
        if any(p in text for p in ["api key", "authentication", "invalid model", "not found"]):
            return "unrecoverable"
        return "unrecoverable"

    # clingo
    if any(p in text for p in ["syntax error", "unsafe", "grounding", "parse"]):
        return "recoverable"
    if any(p in text for p in ["memory", "internal error", "segfault"]):
        return "unrecoverable"
    return "recoverable"


def is_recoverable_failure(report: VerificationReport) -> bool:
    """Convenience: True iff the report is a 'failed' and classifier says recoverable."""
    return (
        report.status == "failed"
        and classify_verification_failure(report) == "recoverable"
    )


# ---------------------------------------------------------------------------
# Primitive functions
# ---------------------------------------------------------------------------


def generate_answer(
    question: Question,
    kb: KnowledgeBase,
    client: OpenAI,
    id_factory: Callable[[], str],
) -> CandidateAnswer:
    """
    DSL-MAP: PRIMITIVE-GENERATE-ANSWER

    LLM-backed answer proposal. Python assigns the artifact id after the
    content is parsed and validated.
    """
    user_prompt = (
        f"Question (id={question.id}):\n{question.text}\n\n"
        f"Knowledge base (id={kb.id}):\n{kb.asp_program}\n\n"
        f"Knowledge base notes: {kb.metadata.get('notes', '')}\n\n"
        "Propose an answer grounded in the knowledge base."
    )
    parsed = _call_llm_json(
        client, GENERATE_ANSWER_SYSTEM, user_prompt,
        _TA_CAND_ANSWER, label="generate_answer",
    )
    return CandidateAnswer(
        id=id_factory(),
        question_id=question.id,
        text=parsed.text,
        rationale=parsed.rationale,
    )


def verify_candidate_answer(
    answer: CandidateAnswer,
    kb: KnowledgeBase,
    client: OpenAI,
    id_factory: Callable[[], str],
) -> VerificationReport:
    """
    DSL-MAP: PRIMITIVE-VERIFY-ANSWER

    LLM-backed text verification. Returns a VerificationReport with
    verifier_kind="llm". Python assigns the artifact id.
    """
    user_prompt = (
        f"Question (id={answer.question_id}):\n"
        f"---\n"
        f"Candidate answer (id={answer.id}):\n{answer.text}\n"
        f"Rationale: {answer.rationale}\n\n"
        f"Knowledge base (id={kb.id}):\n{kb.asp_program}\n\n"
        "Verify the candidate answer against the knowledge base. "
        "Quote supporting rules/facts in 'evidence'."
    )
    parsed = _call_llm_json(
        client, VERIFY_ANSWER_SYSTEM, user_prompt,
        _TA_ANSWER_VERIFY, label="verify_answer",
    )
    return VerificationReport(
        id=id_factory(),
        status=parsed.status,
        reason=parsed.reason,
        evidence=parsed.evidence,
        raw_output="",
        verifier_kind="llm",
    )


def verify_candidate_knowledge(
    ck: CandidateKnowledge,
    kb: KnowledgeBase,
    id_factory: Callable[[], str],
) -> VerificationReport:
    """
    DSL-MAP: PRIMITIVE-VERIFY-KNOWLEDGE (v2.1 merged-coherence semantics)

    Clingo-backed ASP verification. v2.1 change: this primitive verifies the
    MERGED program (knowledge_base ⊕ candidate_knowledge), not the candidate
    in isolation. A candidate that is locally satisfiable but contradicts
    existing KB facts is rejected because the merged program is unsatisfiable.

    Status mapping:
      - candidate sat AND merged sat        -> "verified"
      - candidate unsat (self-contradict)   -> "rejected" (candidate is broken on its own)
      - candidate sat AND merged unsat      -> "rejected" (candidate contradicts existing KB)
      - clingo transport/runtime error      -> "failed"
    """
    # Step 1: solve the candidate alone to detect self-contradiction early.
    # If the candidate references KB-only predicates, solo grounding may
    # fail — treat that as expected, not a failure.
    solo = clingo_solve(ck.asp_program)
    if not solo.ok:
        # Grounding failure likely means candidate references KB predicates.
        # Skip to merged solve rather than returning failed.
        pass
    elif not solo.satisfiable:
        return VerificationReport(
            id=id_factory(),
            status="rejected",
            reason=(
                "Candidate is self-contradictory: clingo reports UNSAT on the "
                "candidate program alone (before merging with the KB)."
            ),
            evidence="",
            raw_output=json.dumps({"models": solo.models, "cost": solo.cost}),
            verifier_kind="clingo",
        )

    # Step 2: solve the MERGED program (kb + candidate). This is the v2.1
    # semantic fix — a candidate can be locally satisfiable and still
    # contradict existing KB facts.
    merged_program = kb.asp_program.rstrip() + "\n\n" + ck.asp_program.strip() + "\n"
    merged = clingo_solve(merged_program)
    if not merged.ok:
        return VerificationReport(
            id=id_factory(),
            status="failed",
            reason=merged.error or "Clingo solve error on merged program",
            evidence="",
            raw_output=merged.error,
            verifier_kind="clingo",
        )
    if not merged.satisfiable:
        return VerificationReport(
            id=id_factory(),
            status="rejected",
            reason=(
                "Merged KB + candidate is UNSATISFIABLE — the candidate "
                "contradicts the existing knowledge_base. The clingo error "
                "in raw_output points at the conflicting fragment; the LLM "
                "revisor will rewrite the candidate to drop or fix it."
            ),
            evidence="",
            raw_output=json.dumps(
                {
                    "clingo_error": merged.error,
                    "candidate_chars": len(ck.asp_program),
                    "kb_chars": len(kb.asp_program),
                }
            ),
            verifier_kind="clingo",
        )

    # Both sat: verified. Models come from the merged program.
    return VerificationReport(
        id=id_factory(),
        status="verified",
        reason=(
            f"Clingo: merged KB + candidate is satisfiable; produced "
            f"{len(merged.models)} model(s) (showing up to 3)."
        ),
        evidence="\n".join(", ".join(m) for m in merged.models[:3]),
        raw_output=json.dumps(
            {
                "models": merged.models,
                "cost": merged.cost,
                "optimal": merged.optimal,
                "kb_chars": len(kb.asp_program),
                "candidate_chars": len(ck.asp_program),
            }
        ),
        verifier_kind="clingo",
    )


def abduce_answer(
    question: Question,
    kb: KnowledgeBase,
    candidate: CandidateAnswer,
    report: VerificationReport,
    registry: ArtifactRegistry,
    client: OpenAI,
    id_factory: Callable[[], str],
) -> Tuple[AbductiveHypothesis, Critique]:
    """
    DSL-MAP: PRIMITIVE-ABDUCE-ANSWER

    LLM-backed abductive revision. Produces an AbductiveHypothesis + a
    Critique that targets the existing candidate (I-CRITIQUE-MUST-TARGET-EXISTING-ARTIFACT).
    """
    user_prompt = (
        f"Question (id={question.id}):\n{question.text}\n\n"
        f"Knowledge base (id={kb.id}):\n{kb.asp_program}\n\n"
        f"Candidate answer (id={candidate.id}):\n{candidate.text}\n"
        f"Rationale: {candidate.rationale}\n\n"
        f"Verification report (id={report.id}, verifier={report.verifier_kind}):\n"
        f"  status:  {report.status}\n"
        f"  reason:  {report.reason}\n"
        f"  evidence:{report.evidence}\n\n"
        f"Produce an abductive hypothesis. The critique.target_id must be: "
        f"\"{candidate.id}\"."
    )
    parsed = _call_llm_json(
        client, ABDUCE_SYSTEM, user_prompt,
        _TA_ABDUCTION, label="abduce_answer",
    )
    # Invariant: critique.target_id must point to an existing artifact.
    if not registry.has(parsed.critique.target_id):
        raise ValueError(
            f"I-CRITIQUE-MUST-TARGET-EXISTING-ARTIFACT violated: "
            f"critique.target_id={parsed.critique.target_id!r} not in registry"
        )
    hypothesis = AbductiveHypothesis(
        id=id_factory(),
        explanation=parsed.explanation,
        repair_plan=parsed.repair_plan,
    )
    critique = Critique(
        id=id_factory(),
        target_id=parsed.critique.target_id,
        issues=list(parsed.critique.issues),
        suggestions=list(parsed.critique.suggestions),
    )
    return hypothesis, critique


def abduce_knowledge(
    kb: KnowledgeBase,
    candidate: CandidateKnowledge,
    report: VerificationReport,
    registry: ArtifactRegistry,
    client: OpenAI,
    id_factory: Callable[[], str],
) -> Tuple[AbductiveHypothesis, Critique]:
    """
    DSL-MAP: PRIMITIVE-ABDUCE-KNOWLEDGE

    LLM-backed abductive revision for an ASP candidate. Enforces
    I-CRITIQUE-MUST-TARGET-EXISTING-ARTIFACT.
    """
    user_prompt = (
        f"Knowledge base (id={kb.id}):\n{kb.asp_program}\n\n"
        f"Candidate knowledge (id={candidate.id}):\n{candidate.asp_program}\n"
        f"Notes: {candidate.notes}\n\n"
        f"Verification report (id={report.id}, verifier={report.verifier_kind}):\n"
        f"  status:  {report.status}\n"
        f"  reason:  {report.reason}\n"
        f"  evidence:{report.evidence}\n\n"
        f"Produce an abductive hypothesis. The critique.target_id must be: "
        f"\"{candidate.id}\"."
    )
    parsed = _call_llm_json(
        client, ABDUCE_SYSTEM, user_prompt,
        _TA_ABDUCTION, label="abduce_knowledge",
    )
    if not registry.has(parsed.critique.target_id):
        raise ValueError(
            f"I-CRITIQUE-MUST-TARGET-EXISTING-ARTIFACT violated: "
            f"critique.target_id={parsed.critique.target_id!r} not in registry"
        )
    hypothesis = AbductiveHypothesis(
        id=id_factory(),
        explanation=parsed.explanation,
        repair_plan=parsed.repair_plan,
    )
    critique = Critique(
        id=id_factory(),
        target_id=parsed.critique.target_id,
        issues=list(parsed.critique.issues),
        suggestions=list(parsed.critique.suggestions),
    )
    return hypothesis, critique


def revise_answer(
    answer: CandidateAnswer,
    hypothesis: AbductiveHypothesis,
    critique: Critique,
    client: OpenAI,
    id_factory: Callable[[], str],
) -> CandidateAnswer:
    """
    DSL-MAP: PRIMITIVE-REVISE-ANSWER

    LLM-backed revision. Always produces a NEW CandidateAnswer with a new
    artifact id (immutable snapshot).
    """
    user_prompt = (
        f"Original question (id={answer.question_id}):\n"
        f"Previous candidate (id={answer.id}):\n{answer.text}\n"
        f"Previous rationale: {answer.rationale}\n\n"
        f"Abductive hypothesis (id={hypothesis.id}):\n{hypothesis.explanation}\n"
        f"Repair plan:\n{hypothesis.repair_plan}\n\n"
        f"Critique (id={critique.id}, target_id={critique.target_id}):\n"
        f"  issues:      {critique.issues}\n"
        f"  suggestions: {critique.suggestions}\n\n"
        "Revise the candidate to address the critique."
    )
    parsed = _call_llm_json(
        client, REVISE_ANSWER_SYSTEM, user_prompt,
        _TA_REVISE_ANSWER, label="revise_answer",
    )
    return CandidateAnswer(
        id=id_factory(),
        question_id=answer.question_id,
        text=parsed.text,
        rationale=parsed.rationale,
    )


def revise_knowledge(
    ck: CandidateKnowledge,
    hypothesis: AbductiveHypothesis,
    critique: Critique,
    client: OpenAI,
    id_factory: Callable[[], str],
) -> CandidateKnowledge:
    """
    DSL-MAP: PRIMITIVE-REVISE-KNOWLEDGE

    LLM-backed revision of an ASP candidate. Always produces a NEW
    CandidateKnowledge with a new artifact id.
    """
    user_prompt = (
        f"Previous candidate knowledge (id={ck.id}):\n{ck.asp_program}\n"
        f"Notes: {ck.notes}\n\n"
        f"Abductive hypothesis (id={hypothesis.id}):\n{hypothesis.explanation}\n"
        f"Repair plan:\n{hypothesis.repair_plan}\n\n"
        f"Critique (id={critique.id}, target_id={critique.target_id}):\n"
        f"  issues:      {critique.issues}\n"
        f"  suggestions: {critique.suggestions}\n\n"
        "Revise the ASP program to address the critique. Return the FULL program."
    )
    parsed = _call_llm_json(
        client, REVISE_KNOWLEDGE_SYSTEM, user_prompt,
        _TA_CAND_KNOWLEDGE, label="revise_knowledge",
    )
    return CandidateKnowledge(
        id=id_factory(),
        asp_program=parsed.asp_program,
        notes=parsed.notes,
    )


def generate_candidate_knowledge(
    source_text: str,
    kb: KnowledgeBase,
    client: OpenAI,
    id_factory: Callable[[], str],
    gap_context: str = "",
) -> CandidateKnowledge:
    """
    DSL-MAP: PRIMITIVE-GENERATE-KNOWLEDGE (v2.1)

    LLM-backed extraction of an ADDITIVE ASP fragment from unstructured
    source text. v2.1: the candidate is strictly additive; contradictions
    with the existing KB are not surfaced here but will be caught downstream
    by verify_candidate_knowledge's merged-coherence check (status="rejected"
    with reason pointing at the merged unsat), and resolved by the abduce
    / revise loop.

    When gap_context is provided (from a previous ask --gap-report), it is
    appended as a targeting signal so the LLM knows which predicates or
    topics are specifically needed.

    The prompt explicitly tells the LLM to:
      - reuse KB predicates and constants as vocabulary when possible;
      - NOT restate the KB;
      - NOT invent unrelated predicates;
      - keep the fragment small and grounded.
    """
    gap_section = (
        f"\n\nKnowledge gap targeting:\n{gap_context}\n\n"
        "The above gap report was produced while answering a question. "
        "Prioritise extracting facts and rules that fill these gaps. "
        "If the source text provides information matching the missing "
        "predicates or topics, extract it using the existing KB predicate "
        "vocabulary where possible."
        if gap_context else ""
    )
    user_prompt = (
        f"Existing knowledge base (id={kb.id}):\n{kb.asp_program}\n\n"
        f"Source text to extract knowledge from:\n{source_text}\n\n"
        f"Produce a small, additive ASP fragment that captures the new facts "
        f"and rules grounded in the source text. Reuse KB predicates and "
        f"constants where they fit. Do not restate the KB."
        f"{gap_section}"
    )
    parsed = _call_llm_json(
        client, GENERATE_KNOWLEDGE_SYSTEM, user_prompt,
        _TA_CAND_KNOWLEDGE, label="generate_knowledge",
    )
    return CandidateKnowledge(
        id=id_factory(),
        asp_program=parsed.asp_program,
        notes=parsed.notes,
    )


def merge_verified_knowledge(
    ck: CandidateKnowledge,
    report: VerificationReport,
    kb: KnowledgeBase,
    id_factory: Callable[[], str],
) -> Tuple[IntegrationResult, KnowledgeBase]:
    """
    DSL-MAP: PRIMITIVE-INTEGRATE (v2.1 snapshot semantics)

    Apply a verified CandidateKnowledge to the existing KnowledgeBase and
    produce a NEW immutable KnowledgeBase snapshot. The prior KB is
    preserved. The new KB records last_integrated_candidate (and any
    other provenance metadata) so the integration is auditable.

    v2.1 invariant enforcement: report.status must be "verified". Anything
    else (rejected, failed) returns a failed IntegrationResult and the
    unchanged KB. The actual *correctness* of the integration is ensured
    upstream by verify_candidate_knowledge (v2.1 merged-coherence check).
    """
    if report.status != "verified":
        return IntegrationResult(
            id=id_factory(),
            success=False,
            message=(
                f"I-INTEGRATE-KNOWLEDGE-ONLY-IF-VERIFIED violated: "
                f"report.status={report.status!r}"
            ),
            updated_program=kb.asp_program,
        ), kb

    # Check for redundancy: if every non-empty, non-comment line in the
    # candidate already exists verbatim in the KB, skip the merge.
    candidate_lines = set(
        line.strip() for line in ck.asp_program.splitlines()
        if line.strip() and not line.strip().startswith("%")
        and not line.strip().startswith("#show")
    )
    kb_lines = set(
        line.strip() for line in kb.asp_program.splitlines()
        if line.strip() and not line.strip().startswith("%")
    )
    if candidate_lines and candidate_lines.issubset(kb_lines):
        logger.info(
            "Candidate %s is fully redundant — all facts/rules already in KB.",
            ck.id,
        )
        return IntegrationResult(
            id=id_factory(),
            success=True,
            message=(
                f"Candidate {ck.id} is redundant: all facts and rules "
                f"already exist in the KB. Nothing added."
            ),
            updated_program=kb.asp_program,
        ), kb

    merged_program = (kb.asp_program.rstrip() + "\n\n" + ck.asp_program.strip() + "\n").strip() + "\n"
    # Deliberate design: the integrated KB keeps the same id as the input KB.
    # It is treated as an updated version of the same logical store, not a
    # new artifact. Callers that need versioned KB identities should layer
    # their own versioning on top. (This also keeps the per-run
    # ArtifactRegistry collision-safe: new_kb is a flow output and is not
    # registered in the flow's intermediate-artifact registry.)
    new_kb = KnowledgeBase(
        id=kb.id,
        source_text=kb.source_text,
        asp_program=merged_program,
        metadata={**kb.metadata, "last_integrated_candidate": ck.id},
    )
    result = IntegrationResult(
        id=id_factory(),
        success=True,
        message=f"Integrated {ck.id} into {kb.id}.",
        updated_program=merged_program,
    )
    return result, new_kb


def commit_answer(
    answer: CandidateAnswer,
    report: VerificationReport,
    id_factory: Callable[[], str],
) -> FinalAnswer:
    """
    DSL-MAP: PRIMITIVE-COMMIT

    Promote a verified candidate answer to a final answer. Enforces
    I-COMMIT-ANSWER-ONLY-IF-VERIFIED.
    """
    if report.status != "verified":
        raise ValueError(
            f"I-COMMIT-ANSWER-ONLY-IF-VERIFIED violated: "
            f"report.status={report.status!r}"
        )
    return FinalAnswer(
        id=id_factory(),
        question_id=answer.question_id,
        answer_text=answer.text,
    )


def fail_flow(flow_id: str, reason: str) -> None:
    """
    DSL-MAP: PRIMITIVE-FAIL

    Raises FlowFailed with a descriptive message. Flow loops call this on
    iterations_exceeded or unrecoverable_verification_failure.
    """
    raise FlowFailed(f"flow={flow_id} failed: {reason}")


# ---------------------------------------------------------------------------
# Flows (hard-coded, direct translation of YAML v0.2)
# ---------------------------------------------------------------------------


def answer_question(
    question: Question,
    kb: KnowledgeBase,
    registry: ArtifactRegistry,
    client: OpenAI,
    id_factory: Callable[[], str],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> FinalAnswer:
    """
    DSL-MAP: FLOW-ANSWER-QUESTION

    Direct translation of system_model.yaml flows.answer_question:
      T-ANSWER-GENERATE -> T-ANSWER-VERIFY -> D-ANSWER-VERIFIED
        verified                   -> T-ANSWER-COMMIT
        rejected                   -> T-ANSWER-ABDUCE
        failed + recoverable       -> T-ANSWER-ABDUCE
        failed + unrecoverable     -> F-ANSWER-FAILED
      T-ANSWER-ABDUCE -> T-ANSWER-REVISE -> (loop to T-ANSWER-VERIFY)
      T-ANSWER-COMMIT -> FinalAnswer
    """
    # T-ANSWER-GENERATE
    candidate = generate_answer(question, kb, client, id_factory)
    registry.register(candidate)
    logger.info("T-ANSWER-GENERATE produced candidate id=%s", candidate.id)

    # Iteration counter counts (verify) attempts, including the first one.
    # The initial generate happens outside the loop. So with max_iterations=N
    # we permit up to N verify+commit/abduce cycles. A candidate that is
    # rejected on iteration 1 may still be revised on iteration 2 (within
    # the cap). This is intentional: a single generate+verify+commit costs
    # iteration 1, and any revision cycle is iteration 2..N.
    iteration = 0
    while True:
        iteration += 1
        if iteration > max_iterations:
            fail_flow(
                "answer_question",
                f"iterations_exceeded (max={max_iterations})",
            )

        # C-ANSWER-VERIFY
        report = verify_candidate_answer(candidate, kb, client, id_factory)
        registry.register(report)
        logger.info(
            "C-ANSWER-VERIFY iteration=%d report id=%s status=%s verifier=%s",
            iteration,
            report.id,
            report.status,
            report.verifier_kind,
        )

        # D-ANSWER-VERIFIED — four-way branch
        if report.status == "verified":
            # T-ANSWER-COMMIT
            final = commit_answer(candidate, report, id_factory)
            registry.register(final)
            logger.info("T-ANSWER-COMMIT produced final id=%s", final.id)
            return final

        if report.status == "rejected":
            pass  # fall through to abduce
        elif report.status == "failed":
            if is_recoverable_failure(report):
                logger.info("Verification failed but recoverable; entering abduce.")
            else:
                fail_flow(
                    "answer_question",
                    f"unrecoverable_verification_failure: {report.reason}",
                )
        else:  # pragma: no cover — Literal is enforced at construction
            fail_flow("answer_question", f"unknown status: {report.status}")

        # T-ANSWER-ABDUCE
        hypothesis, critique = abduce_answer(
            question, kb, candidate, report, registry, client, id_factory
        )
        registry.register(hypothesis)
        registry.register(critique)
        logger.info(
            "T-ANSWER-ABDUCE iteration=%d hypothesis id=%s critique id=%s target=%s",
            iteration,
            hypothesis.id,
            critique.id,
            critique.target_id,
        )

        # T-ANSWER-REVISE — produces a new candidate (immutable snapshot)
        candidate = revise_answer(
            candidate, hypothesis, critique, client, id_factory
        )
        registry.register(candidate)
        logger.info(
            "T-ANSWER-REVISE iteration=%d new candidate id=%s",
            iteration,
            candidate.id,
        )
        # loop back to C-ANSWER-VERIFY


def integrate_knowledge(
    candidate_knowledge: CandidateKnowledge,
    kb: KnowledgeBase,
    registry: ArtifactRegistry,
    client: OpenAI,
    id_factory: Callable[[], str],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    auto_revise: bool = True,
) -> Tuple[IntegrationResult, KnowledgeBase]:
    """
    DSL-MAP: FLOW-INTEGRATE-KNOWLEDGE

    Direct translation of system_model.yaml flows.integrate_knowledge.
    Returns (IntegrationResult, new_kb) where new_kb is the updated KB on
    success or the unchanged KB on failure.
    """
    iteration = 0
    candidate = candidate_knowledge
    # Re-assign id from factory to avoid collision with reports
    if candidate.id and candidate.id.startswith("ck-") and len(candidate.id) <= 6:
        candidate = CandidateKnowledge(
            id=id_factory(),
            asp_program=candidate.asp_program,
            notes=candidate.notes,
        )
    registry.register(candidate)
    while True:
        iteration += 1
        if iteration > max_iterations:
            fail_flow(
                "integrate_knowledge",
                f"iterations_exceeded (max={max_iterations})",
            )

        # T-KNOWLEDGE-VERIFY (Clingo, v2.1 merged-coherence: kb ⊕ candidate)
        report = verify_candidate_knowledge(candidate, kb, id_factory)
        registry.register(report)
        logger.info(
            "T-KNOWLEDGE-VERIFY iteration=%d report id=%s status=%s",
            iteration,
            report.id,
            report.status,
        )

        # D-KNOWLEDGE-VERIFIED — four-way branch
        if report.status == "verified":
            # T-KNOWLEDGE-INTEGRATE
            result, new_kb = merge_verified_knowledge(
                candidate, report, kb, id_factory
            )
            registry.register(result)
            # Note: new_kb is a flow OUTPUT (returned to caller), not an
            # intermediate artifact, and it intentionally retains the same
            # id as the input kb. Do NOT register it — that would collide
            # with any future re-registration of the same KB identity.
            logger.info(
                "T-KNOWLEDGE-INTEGRATE success=%s result id=%s new_kb id=%s",
                result.success,
                result.id,
                new_kb.id,
            )
            return result, new_kb

        if report.status == "rejected":
            if not auto_revise:
                fail_flow(
                    "integrate_knowledge",
                    f"CONFLICT: candidate contradicts existing KB. "
                    f"Verification rejected. Reason: {report.reason}. "
                    f"Set auto_revise=True or resolve the conflict manually before retrying.",
                )
            pass
        elif report.status == "failed":
            if is_recoverable_failure(report):
                logger.info("Knowledge verification failed but recoverable; entering abduce.")
            else:
                fail_flow(
                    "integrate_knowledge",
                    f"unrecoverable_verification_failure: {report.reason}",
                )
        else:  # pragma: no cover
            fail_flow("integrate_knowledge", f"unknown status: {report.status}")

        # T-KNOWLEDGE-ABDUCE
        hypothesis, critique = abduce_knowledge(
            kb, candidate, report, registry, client, id_factory
        )
        registry.register(hypothesis)
        registry.register(critique)
        logger.info(
            "T-KNOWLEDGE-ABDUCE iteration=%d hypothesis id=%s critique id=%s target=%s",
            iteration,
            hypothesis.id,
            critique.id,
            critique.target_id,
        )

        # T-KNOWLEDGE-REVISE — new candidate
        candidate = revise_knowledge(
            candidate, hypothesis, critique, client, id_factory
        )
        registry.register(candidate)
        logger.info(
            "T-KNOWLEDGE-REVISE iteration=%d new candidate id=%s",
            iteration,
            candidate.id,
        )
        # loop back to T-KNOWLEDGE-VERIFY


# ---------------------------------------------------------------------------
# Orchestrator (thin façade)
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    DSL-MAP: ORCHESTRATOR-FAÇADE

    Wires dependencies (LLM client, registry, id factory) and dispatches to
    the hard-coded flow functions. Holds no flow state itself.
    """

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        self.client: OpenAI = client or get_client()

    def run_answer_question(
        self,
        question: Question,
        kb: KnowledgeBase,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> Tuple[FinalAnswer, ArtifactRegistry]:
        registry = ArtifactRegistry()
        id_factory = make_id_factory("ca")  # candidate_answer ids
        final = answer_question(
            question=question,
            kb=kb,
            registry=registry,
            client=self.client,
            id_factory=id_factory,
            max_iterations=max_iterations,
        )
        return final, registry

    def run_integrate_knowledge(
        self,
        candidate_knowledge: CandidateKnowledge,
        kb: KnowledgeBase,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        auto_revise: bool = True,
    ) -> Tuple[IntegrationResult, KnowledgeBase, ArtifactRegistry]:
        registry = ArtifactRegistry()
        id_factory = make_id_factory("ck")  # candidate_knowledge ids
        result, new_kb = integrate_knowledge(
            candidate_knowledge=candidate_knowledge,
            kb=kb,
            registry=registry,
            client=self.client,
            id_factory=id_factory,
            max_iterations=max_iterations,
            auto_revise=auto_revise,
        )
        return result, new_kb, registry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_kb(path: str) -> KnowledgeBase:
    """Load an ASP knowledge base from a .lp file (metadata = file path)."""
    with open(path, "r", encoding="utf-8") as f:
        program = f.read()
    return KnowledgeBase(
        id=f"kb-{os.path.basename(path)}",
        source_text=program,
        asp_program=program,
        metadata={"path": path, "notes": f"Loaded from {path}"},
    )


def _load_question(args: argparse.Namespace) -> Question:
    if args.question:
        text = args.question.strip()
        qid = args.question_id or f"q-cli"
    else:
        path = args.question_file
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        qid = args.question_id or f"q-{os.path.basename(path)}"
    return Question(id=qid, text=text)


_H1_LINE_RE = re.compile(r"^\s*#\s+(.+?)\s*$")


def extract_first_h1_question(text: str) -> Optional[str]:
    """
    DSL-MAP: QUESTION-EXTRACTOR-FROM-MARKDOWN

    Return the text of the first markdown H1 (`# ...`) in `text`, or None
    if no H1 is present. Used by the `learn` CLI subcommand as a fallback
    when the user does not pass --question explicitly.

    This is intentionally minimal: we trust the user's file structure when
    it exists, and fall back to "no question" (which means the caller should
    skip the re-ask step) when it does not.
    """
    for line in text.splitlines():
        m = _H1_LINE_RE.match(line)
        if m:
            return m.group(1).strip()
    return None


def _load_source_text(path: str) -> str:
    """Load a source text file (UTF-8)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    """
    DSL-MAP: ENTRYPOINT-ADAPTER

    CLI adapter. Two subcommands:
      ask        : run answer_question flow
      integrate  : run integrate_knowledge flow
    """
    parser = argparse.ArgumentParser(
        description="main2.py — 1:1 implementation of system_model.yaml v2.1",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="Answer a question against a knowledge base")
    p_ask.add_argument("--kb", default="kb.lp", help="Path to .lp knowledge base file")
    p_ask.add_argument("--question", help="Question text (overrides --question-file)")
    p_ask.add_argument("--question-file", help="Path to file with the question text")
    p_ask.add_argument("--question-id", help="Override the question artifact id")
    p_ask.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Flow bound (default {DEFAULT_MAX_ITERATIONS})",
    )
    p_ask.add_argument(
        "--gap-report", default=None, nargs="?", const="-",
        help="Path to write a gap signal JSON file that learn --fill-gap can "
             "consume. Pass no argument to print to stdout (terminal view) or "
             "a filename like gaps.json to persist for a follow-up learn call. "
             "The file contains the committed answer's gap rationale.",
    )

    p_int = sub.add_parser("integrate", help="Integrate a candidate ASP program into a KB")
    p_int.add_argument("--kb", required=True, help="Path to existing .lp knowledge base")
    p_int.add_argument(
        "--program-file", required=True,
        help="Path to file containing the candidate ASP program",
    )
    p_int.add_argument("--notes", default="", help="Optional modeling notes")
    p_int.add_argument("--candidate-id", help="Override the candidate artifact id")
    p_int.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Flow bound (default {DEFAULT_MAX_ITERATIONS})",
    )

    p_learn = sub.add_parser(
        "learn",
        help="Learn from a source text: extract candidate knowledge, integrate "
             "it into the KB (with v2.1 merged-coherence verification), and "
             "optionally re-answer the question against the augmented KB",
    )
    p_learn.add_argument("--kb", default="kb.lp", help="Path to .lp knowledge base file")
    p_learn.add_argument(
        "--unstructured", default=None,
        help="Path to source text file. If omitted, uses --text or falls "
             "back to unstructured.txt.",
    )
    p_learn.add_argument(
        "--text", default=None,
        help="Raw text string to extract knowledge from (alternative to "
             "--unstructured file). Use this to pass pasted content "
             "directly without creating a file.",
    )
    p_learn.add_argument(
        "--fill-gap", default=None,
        help="Path to a gap signal JSON file (produced by ask --gap-report). "
             "When set, learn reads the gap summary and uses it to contextualise "
             "extraction even when --unstructured is also given. Optional.",
    )
    p_learn.add_argument(
        "--out-kb", default=None,
        help="Where to write the augmented KB. If omitted, the source KB "
             "(--kb) is overwritten in place. Pass an explicit path to keep "
             "the source KB and write the augmented copy elsewhere.",
    )
    p_learn.add_argument(
        "--question", default=None,
        help="Question to re-ask against the augmented KB. If omitted, the "
             "first H1 (# ...) in the source text is used. If neither is "
             "present, the re-ask step is skipped.",
    )
    p_learn.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Flow bound for both integrate and re-ask (default {DEFAULT_MAX_ITERATIONS})",
    )
    p_learn.add_argument(
        "--skip-reask", action="store_true",
        help="Integrate only; do not re-run answer_question against the augmented KB.",
    )
    p_learn.add_argument(
        "--question-id", default=None,
        help="Override the question artifact id used in the re-ask step.",
    )
    p_learn.add_argument(
        "--gap-report", default=None, nargs="?", const="-",
        help="Path to write a gap signal JSON file from the re-ask step. "
             "Pass no argument to print to stdout. "
             "The file contains missing predicates/topics from the re-asked answer.",
    )

    args = parser.parse_args()
    orch = Orchestrator()

    if args.cmd == "ask":
        kb = _load_kb(args.kb)
        question = _load_question(args)
        try:
            final, registry = orch.run_answer_question(
                question=question,
                kb=kb,
                max_iterations=args.max_iterations,
            )
        except FlowFailed as e:
            logger.error("answer_question failed: %s", e)
            raise SystemExit(1)
        print("\n=== FINAL ANSWER ===\n")
        print(final.answer_text)
        print(f"\n(final_id={final.id} question_id={final.question_id})")
        print(f"(registry contains {len(registry)} artifacts: {registry.ids()})")

        if args.gap_report is not None:
            # The last committed CandidateAnswer is the one that was
            # promoted to FinalAnswer. Its rationale contains the gap
            # signal (missing predicates/topics named by the new
            # GENERATE_ANSWER_SYSTEM prompt). Retrieve it from registry.
            gap_artifact = None
            for rid in reversed(registry.ids()):
                a = registry.get(rid)
                if isinstance(a, CandidateAnswer):
                    gap_artifact = a
                    break
            if gap_artifact and gap_artifact.rationale.strip():
                gap_text = gap_artifact.rationale
                target_preds = extract_target_predicates(gap_text)
                report = KnowledgeGapReport(
                    question=question.text,
                    committed_answer=final.answer_text,
                    gap_rationale=gap_text,
                    target_predicates=target_preds,
                    status="open",
                )
                if args.gap_report == "-":
                    print("\n=== KNOWLEDGE GAP REPORT ===")
                    print(json.dumps({
                        "question": report.question,
                        "target_predicates": report.target_predicates,
                        "gap_rationale": report.gap_rationale,
                        "status": report.status,
                    }, indent=2))
                else:
                    gap_path = args.gap_report
                    write_gap_report(gap_path, report)
                    print(f"\nGap report written to: {gap_path}")
                    if report.target_predicates:
                        print(f"Target predicates: {report.target_predicates}")
                    print("You can now use it with: "
                          f"python main2.py learn --fill-gap {gap_path} ...")
            else:
                if args.gap_report == "-":
                    print("\n(no gap report: rationale was empty)")

        else:
            # Auto-surface: even without --gap-report, briefly note any gaps
            gap_artifact = None
            for rid in reversed(registry.ids()):
                a = registry.get(rid)
                if isinstance(a, CandidateAnswer):
                    gap_artifact = a
                    break
            if gap_artifact and gap_artifact.rationale.strip():
                target_preds = extract_target_predicates(gap_artifact.rationale)
                if target_preds:
                    print(f"\nℹ KB may need predicates: {target_preds}")
                    print(f"  To generate a gap report: re-run with --gap-report gaps.json")

    elif args.cmd == "integrate":
        kb = _load_kb(args.kb)
        with open(args.program_file, "r", encoding="utf-8") as f:
            program = f.read()
        # The CLI does NOT own the id factory for candidate_knowledge;
        # Orchestrator.run_integrate_knowledge creates the factory used by
        # the revision loop. We only need a stable id for the initial
        # candidate, which the caller may override via --candidate-id.
        candidate = CandidateKnowledge(
            id=args.candidate_id or "ck-cli",
            asp_program=program,
            notes=args.notes,
        )
        try:
            result, new_kb, registry = orch.run_integrate_knowledge(
                candidate_knowledge=candidate,
                kb=kb,
                max_iterations=args.max_iterations,
                auto_revise=False,
            )
        except FlowFailed as e:
            logger.error("integrate_knowledge failed: %s", e)
            raise SystemExit(1)
        print("\n=== INTEGRATION RESULT ===\n")
        print(f"success: {result.success}")
        print(f"message: {result.message}")
        print(f"result_id: {result.id}")
        print(f"new_kb_id: {new_kb.id}")
        print(f"updated_program_chars: {len(new_kb.asp_program)}")
        print(f"registry contains {len(registry)} artifacts")

        if result.success:
            # Resolve output path first (needed by both conflict surface and persist)
            try:
                out_kb_path = args.out_kb or args.kb
            except AttributeError:
                out_kb_path = args.kb

            # Check registry for any rejections — the revised candidate
            # succeeded, but the original was rejected. Surface this.
            rejected_ids = []
            for rid in registry.ids():
                a = registry.get(rid)
                if isinstance(a, VerificationReport) and a.status == "rejected":
                    rejected_ids.append(rid)
            if rejected_ids:
                print(f"\n⚠ CONFLICT DETECTED AND RESOLVED")
                print(f"  Original candidate was rejected ({len(rejected_ids)} report(s)):")
                report = registry.get(rejected_ids[-1])
                print(f"  Reason: {report.reason[:200]}")
                print(f"  The system auto-revised and integrated a non-conflicting version.")
                print(f"  Original intent may have been altered — review final KB at {out_kb_path}")
            else:
                print(f"  No conflicts detected — candidate integrated directly.")

            # Persist the augmented KB to disk
            with open(out_kb_path, "w", encoding="utf-8") as f:
                f.write(new_kb.asp_program)
            print(f"\naugmented KB written to: {out_kb_path}")

    elif args.cmd == "learn":
        # Chain: load text -> generate candidate knowledge -> integrate ->
        # optionally re-ask against the augmented KB.
        kb = _load_kb(args.kb)

        # Resolve unstructured source. If --fill-gap is given and no
        # explicit --unstructured, read the gap file for the question
        # context and use unstructured.jsonl or fallback.
        src_path = args.unstructured
        gap_report_obj = None
        gap_context = ""
        gap_file_path = None
        if args.fill_gap:
            gap_file_path = args.fill_gap
            gap_report_obj = load_gap_report(args.fill_gap)
            # Build gap_context from target_predicates (primary) with
            # gap_rationale as descriptive context (fallback).
            if gap_report_obj.target_predicates:
                gap_context = (
                    "Target predicates to fill: "
                    + ", ".join(gap_report_obj.target_predicates)
                )
                if gap_report_obj.gap_rationale:
                    gap_context += (
                        "\n\nContext: " + gap_report_obj.gap_rationale
                    )
            elif gap_report_obj.gap_rationale:
                gap_context = gap_report_obj.gap_rationale
            print(f"\nLoaded gap signal from: {args.fill_gap}")
            print(f"Status: {gap_report_obj.status}")
            if gap_report_obj.target_predicates:
                print(f"Target predicates: {gap_report_obj.target_predicates}")
            elif gap_context:
                print(f"Targeting gaps: {gap_context[:200]}...")
        if not src_path and not args.text:
            src_path = "unstructured.txt"
        if src_path:
            source_text = _load_source_text(src_path)
        elif args.text:
            source_text = args.text
        if not source_text.strip():
            raise SystemExit("Empty source text.")

        # Step 1: generate a CandidateKnowledge from the source text.
        ck_factory = make_id_factory("ck")
        try:
            candidate = generate_candidate_knowledge(
                source_text=source_text,
                kb=kb,
                client=orch.client,
                id_factory=ck_factory,
                gap_context=gap_context,
            )
        except LLMArtifactError as e:
            logger.error("generate_knowledge failed: %s", e)
            raise SystemExit(1)
        print("\n=== GENERATED CANDIDATE KNOWLEDGE ===\n")
        print(candidate.asp_program)
        if candidate.notes:
            print(f"\n--- notes ---\n{candidate.notes}")
        print(f"\n(candidate id={candidate.id})")

        # Step 2: integrate via the v2.1 flow (merged-coherence verify,
        # abduce/revise loop on rejection, snapshot KB on success).
        try:
            result, new_kb, int_registry = orch.run_integrate_knowledge(
                candidate_knowledge=candidate,
                kb=kb,
                max_iterations=args.max_iterations,
            )
        except FlowFailed as e:
            logger.error("integrate_knowledge failed: %s", e)
            raise SystemExit(1)

        print("\n=== INTEGRATION RESULT ===\n")
        print(f"success: {result.success}")
        print(f"message: {result.message}")
        print(f"result_id: {result.id}")
        print(f"new_kb_id: {new_kb.id}")
        print(f"updated_program_chars: {len(new_kb.asp_program)}")
        print(f"integrate registry size: {len(int_registry)}")

        if not result.success:
            # The flow returned an unsuccessful integration; surface and stop.
            raise SystemExit(1)

        # Step 3: write the augmented KB to disk. By default, we overwrite
        # the source KB in place so the working KB is always the latest
        # augmented one. Pass --out-kb to keep the source and write the
        # augmented copy elsewhere. The write only happens after a
        # successful integration (an unsuccessful one already exited above).
        out_kb_path = args.out_kb or args.kb
        with open(out_kb_path, "w", encoding="utf-8") as f:
            f.write(new_kb.asp_program)
        print(f"\naugmented KB written to: {out_kb_path}")
        if out_kb_path == args.kb:
            print("(source KB overwritten in place; git diff will show the new facts)")

        # Step 3.5: gap-resolution check (T-LEARN-RESOLVE-GAP).
        # After a successful integration, verify whether the targeted
        # predicates now appear in the augmented KB. Only runs when
        # --fill-gap was provided and the gap report is still "open".
        if gap_report_obj is not None and gap_report_obj.status == "open":
            resolution = verify_gap_resolution(
                gap_report_obj,
                new_kb.asp_program,
            )
            gap_report_obj.resolution = resolution
            if resolution.all_resolved and not resolution.predicates_still_missing:
                gap_report_obj.status = "closed"
                print(f"\n=== GAP RESOLUTION: CLOSED ===")
            else:
                print(f"\n=== GAP RESOLUTION: PARTIAL ===")
            for pred in resolution.predicates_checked:
                if pred in resolution.predicates_dropped:
                    print(f"  {pred} — excluded by review (target_review.drop)")
                elif pred in resolution.predicates_found:
                    print(f"  {pred} — resolved")
                else:
                    print(f"  {pred} — exact target missing")
                    related = resolution.related_predicates_found.get(pred)
                    if related:
                        for r in related:
                            print(f"    related predicate found with different arity: {r}")
            if not resolution.predicates_found and not resolution.related_predicates_found:
                print(f"  (no matching predicates found in KB at all)")
            # Derive output path: same file as input, or
            # gaps-resolved.json if input was stdin.
            if gap_file_path == "-":
                # Can't write back to stdin; emit a note
                print("  (gap file was stdin; resolution not persisted)")
            else:
                write_gap_report(gap_file_path, gap_report_obj)
                print(f"  gap file updated: {gap_file_path}")
        elif gap_report_obj is not None and gap_report_obj.status == "closed":
            print(f"\n(gap report already closed; skipping resolution check)")

        # Step 4: optionally re-ask the question against the augmented KB.
        if args.skip_reask:
            print("\n(re-ask skipped via --skip-reask)")
        else:
            question_text = args.question or extract_first_h1_question(source_text)
            if not question_text:
                print("\n(no question found in source text and --question not given; skipping re-ask)")
            else:
                print(f"\n=== RE-ASK AGAINST AUGMENTED KB ===\n")
                print(f"question: {question_text}")
                question = Question(
                    id=args.question_id or f"q-{os.path.basename(args.unstructured)}",
                    text=question_text,
                )
                try:
                    final, ask_registry = orch.run_answer_question(
                        question=question,
                        kb=new_kb,
                        max_iterations=args.max_iterations,
                    )
                except FlowFailed as e:
                    logger.error("answer_question (re-ask) failed: %s", e)
                    raise SystemExit(1)
                print(f"\n--- final answer ---\n{final.answer_text}")
                print(f"\n(final_id={final.id} question_id={final.question_id})")
                print(f"(ask registry size: {len(ask_registry)})")

                # Optional gap report from re-ask step
                if args.gap_report is not None:
                    gap_artifact = None
                    for rid in reversed(ask_registry.ids()):
                        a = ask_registry.get(rid)
                        if isinstance(a, CandidateAnswer):
                            gap_artifact = a
                            break
                    if gap_artifact and gap_artifact.rationale.strip():
                        gap_text = gap_artifact.rationale
                        target_preds = extract_target_predicates(gap_text)
                        report = KnowledgeGapReport(
                            question=question_text,
                            committed_answer=final.answer_text,
                            gap_rationale=gap_text,
                            target_predicates=target_preds,
                            status="open",
                        )
                        if args.gap_report == "-":
                            print("\n=== KNOWLEDGE GAP REPORT (re-ask) ===")
                            print(json.dumps({
                                "question": report.question,
                                "target_predicates": report.target_predicates,
                                "gap_rationale": report.gap_rationale,
                                "status": report.status,
                            }, indent=2))
                        else:
                            write_gap_report(args.gap_report, report)
                            print(f"\nGap report written to: {args.gap_report}")
                            if report.target_predicates:
                                print(f"Target predicates: {report.target_predicates}")


if __name__ == "__main__":
    main()
