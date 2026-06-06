"""
Artifact dataclasses, registry, id factory, and shared exception types.

DSL-MAP alignment:
- KnowledgeBase fields: sourcetext, aspprogram (compact YAML names retained in
  Python as source_text / asp_program mapped in __init__).
- CandidateAnswer: questionid → question_id
- CandidateKnowledge: aspprogram → asp_program
- Critique: targetid → target_id
- VerificationReport: verifierkind, rawoutput
- AbductiveHypothesis: repairplan → repair_plan
- FinalAnswer: questionid, answertext
- IntegrationResult: updatedprogram → updated_program
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FlowFailed(Exception):
    """
    Raised by fail_flow() and by flow loops when termination conditions
    (iterations_exceeded, unrecoverable_verification_failure) trigger.
    Encodes I-QUESTION-TERMINATES as an exception-driven escape.
    """

class LLMArtifactError(Exception):
    """
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
    Per-run registry of artifacts. Raises ValueError on id collision to catch
    bugs early; revisions always produce new IDs so collisions indicate logic
    errors.
    """
    _artifacts: Dict[str, HasId] = field(default_factory=dict)

    def register(self, artifact: HasId) -> None:
        if artifact.id in self._artifacts:
            raise ValueError(
                f"Artifact with id '{artifact.id}' already registered"
            )
        self._artifacts[artifact.id] = artifact

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
    Deterministic, monotonic id generator with a per-prefix counter. The LLM
    is never trusted to assign ids; Python assigns them after validation.
    """
    counter = {"n": 0}

    def factory() -> str:
        counter["n"] += 1
        return f"{prefix}-{counter['n']:03d}"

    return factory


# ---------------------------------------------------------------------------
# Artifact dataclasses
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

    The YAML uses compact field names: sourcetext, aspprogram.
    We preserve them as Python keyword arguments via aliases in __init__
    so both `KnowledgeBase(id=..., source_text=...)` and
    `KnowledgeBase(id=..., sourcetext=...)` work identically.
    """
    id: str
    source_text: str = ""
    asp_program: str = ""
    metadata: dict = field(default_factory=dict)

    def __init__(
        self,
        id: str,
        source_text: str = "",
        asp_program: str = "",
        metadata: Optional[dict] = None,
        sourcetext: str = "",
        aspprogram: str = "",
    ) -> None:
        # Accept both compact YAML-style and Pythonic names.
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "source_text", source_text or sourcetext)
        object.__setattr__(self, "asp_program", asp_program or aspprogram)
        object.__setattr__(self, "metadata", metadata if metadata is not None else {})


@dataclass(frozen=True, kw_only=True)
class CandidateAnswer:
    """
    DSL-MAP: ARTIFACT-CANDIDATE-ANSWER

    A proposed answer produced by generate_answer or revise_answer. Every
    revision is a new immutable snapshot with a new id.

    YAML field: questionid → Python kw: question_id (alias supported).
    """
    id: str
    text: str
    rationale: str = ""
    question_id: str = ""

    def __init__(
        self,
        id: str,
        text: str,
        rationale: str = "",
        question_id: str = "",
        questionid: str = "",
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "question_id", question_id or questionid)


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
    asp_program: str = ""
    notes: str = ""

    def __init__(
        self,
        id: str,
        asp_program: str = "",
        notes: str = "",
        aspprogram: str = "",
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "asp_program", asp_program or aspprogram)
        object.__setattr__(self, "notes", notes)


@dataclass(frozen=True, kw_only=True)
class Critique:
    """
    DSL-MAP: ARTIFACT-CRITIQUE

    Produced alongside an abductive_hypothesis. The target_id must refer
    to an artifact already in the registry (I-CRITIQUE-MUST-TARGET-EXISTING-ARTIFACT).

    YAML field: targetid → Python kw: target_id (alias supported).
    """
    id: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    target_id: str = ""

    def __init__(
        self,
        id: str,
        target_id: str = "",
        issues: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
        targetid: str = "",
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "target_id", target_id or targetid)
        object.__setattr__(self, "issues", issues if issues is not None else [])
        object.__setattr__(self, "suggestions", suggestions if suggestions is not None else [])


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

    YAML fields: verifierkind, rawoutput → Python: verifier_kind, raw_output.
    """
    id: str
    status: Literal["verified", "rejected", "failed"] = "failed"
    reason: str = ""
    evidence: str = ""
    raw_output: str = ""
    verifier_kind: Literal["llm", "clingo"] = "llm"

    def __init__(
        self,
        id: str,
        status: Literal["verified", "rejected", "failed"] = "failed",
        reason: str = "",
        evidence: str = "",
        raw_output: str = "",
        rawoutput: str = "",
        verifier_kind: Literal["llm", "clingo"] = "llm",
        verifierkind: str = "llm",
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "evidence", evidence)
        # Accept both names; prefer explicit raw_output/verifier_kind.
        object.__setattr__(self, "raw_output", raw_output or rawoutput)
        vk = verifier_kind or verifierkind
        if vk not in ("llm", "clingo"):
            raise ValueError(f"verifier_kind must be 'llm' or 'clingo', got {vk!r}")
        object.__setattr__(self, "verifier_kind", vk)


@dataclass(frozen=True, kw_only=True)
class AbductiveHypothesis:
    """
    DSL-MAP: ARTIFACT-ABDUCTIVE-HYPOTHESIS

    A best-guess explanation of why the candidate was rejected/failed, plus
    a concrete repair plan the next revision should follow.

    YAML field: repairplan → Python kw: repair_plan (alias supported).
    """
    id: str
    explanation: str = ""
    repair_plan: str = ""

    def __init__(
        self,
        id: str,
        explanation: str = "",
        repair_plan: str = "",
        repairplan: str = "",
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "explanation", explanation)
        object.__setattr__(self, "repair_plan", repair_plan or repairplan)


@dataclass(frozen=True, kw_only=True)
class FinalAnswer:
    """
    DSL-MAP: ARTIFACT-FINAL-ANSWER

    Produced by commit_answer, which enforces I-COMMIT-ANSWER-ONLY-IF-VERIFIED.

    YAML fields: questionid, answertext → Python: question_id, answer_text.
    """
    id: str
    question_id: str = ""
    answer_text: str = ""

    def __init__(
        self,
        id: str,
        question_id: str = "",
        answer_text: str = "",
        questionid: str = "",
        answertext: str = "",
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "question_id", question_id or questionid)
        object.__setattr__(self, "answer_text", answer_text or answertext)


@dataclass(frozen=True, kw_only=True)
class IntegrationResult:
    """
    DSL-MAP: ARTIFACT-INTEGRATION-RESULT

    Wraps the outcome of merging verified knowledge into a KB. updated_program
    is the new full program string (or the unchanged one on failure).

    YAML field: updatedprogram → Python kw: updated_program (alias supported).
    """
    id: str
    success: bool = False
    message: str = ""
    updated_program: str = ""

    def __init__(
        self,
        id: str,
        success: bool = False,
        message: str = "",
        updated_program: str = "",
        updatedprogram: str = "",
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "updated_program", updated_program or updatedprogram)


@dataclass(frozen=True, kw_only=True)
class LoopStatus:
    """
    DSL-MAP: ARTIFACT-LOOP-STATUS

    Snapshot of loop state. bounded is always True in main2 (hard-coded
    max_iterations). escaped is True if the loop exited via FlowFailed or
    commit/integrate before iteration cap.

    Note on field names: as encoded in system_model.yaml, loop_status includes
    `id` as model-level normalization for artifact uniformity. The Python
    dataclass does not carry an id field here because LoopStatus is an internal
    flow-state value, not an artifact registered in the ArtifactRegistry.
    Callers that need to track individual loops can name them via flow_id.
    """
    flow_id: str
    iteration: int
    max_iterations: int
    bounded: bool
    escaped: bool
