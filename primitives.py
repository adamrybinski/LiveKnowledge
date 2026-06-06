"""
Primitive functions: LLM-backed generation, verification, abduce, revise,
merged-coherence knowledge verification, integration, routing, and CLI.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any, Callable, List, Literal, Optional, Tuple

import artifacts
import gap_report
import llm
import prompts
import solver

logger = logging.getLogger("asplearning.main2")

# Re-export facilitate imports from other modules (e.g., CLI in main.py).
FlowFailed = artifacts.FlowFailed
LLMArtifactError = artifacts.LLMArtifactError
ArtifactRegistry = artifacts.ArtifactRegistry
make_id_factory = artifacts.make_id_factory

Question = artifacts.Question
KnowledgeBase = artifacts.KnowledgeBase
CandidateAnswer = artifacts.CandidateAnswer
CandidateKnowledge = artifacts.CandidateKnowledge
Critique = artifacts.Critique
VerificationReport = artifacts.VerificationReport
AbductiveHypothesis = artifacts.AbductiveHypothesis
FinalAnswer = artifacts.FinalAnswer
IntegrationResult = artifacts.IntegrationResult


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

    text = (report.reason or "").lower()
    raw = (report.raw_output or "").lower()

    if report.verifier_kind == "llm":
        if any(p in text for p in ["timeout", "rate limit", "connection", "temporary"]):
            return "recoverable"
        if any(p in text for p in ["api key", "authentication", "invalid model", "not found"]):
            return "unrecoverable"
        return "unrecoverable"

    # clingo
    check = text if text else raw
    if any(p in check for p in ["syntax error", "unsafe", "grounding", "parse"]):
        return "recoverable"
    if any(p in check for p in ["memory", "internal error", "segfault"]):
        return "unrecoverable"
    return "recoverable"


def is_recoverable_failure(report: VerificationReport) -> bool:
    """Convenience: True iff the report is a 'failed' and classifier says recoverable."""
    return (
        report.status == "failed"
        and classify_verification_failure(report) == "recoverable"
    )


# ---------------------------------------------------------------------------
# LLM primitives
# ---------------------------------------------------------------------------

def generate_answer(
    question: Question,
    kb: KnowledgeBase,
    client,
    id_factory: Callable[[], str],
) -> CandidateAnswer:
    """
    DSL-MAP: PRIMITIVE-GENERATE-ANSWER

    LLM-backed answer proposal. Python assigns the artifact id after the
    content is parsed and validated.
    """
    logger.info("PRIMITIVE start label=generate_answer question_id=%s", question.id)
    parsed = llm.call_llm_json(
        client, prompts.GENERATE_ANSWER_SYSTEM, user_prompt,
        llm._TA_CAND_ANSWER, label="generate_answer",
    )
    logger.info(
        "PRIMITIVE end label=generate_answer candidate_id=%s question_id=%s chars=%d",
        candidate.id,
        candidate.question_id,
        len(candidate.text),
    )
    return candidate


def verify_candidate_answer(
    answer: CandidateAnswer,
    kb: KnowledgeBase,
    client,
    id_factory: Callable[[], str],
) -> VerificationReport:
    """
    DSL-MAP: PRIMITIVE-VERIFY-ANSWER

    LLM-backed text verification. Returns a VerificationReport with
    verifier_kind="llm". Python assigns the artifact id.
    """
    logger.info(
        "PRIMITIVE start label=verify_candidate_answer candidate_id=%s verifier=llm",
        answer.id,
    )
    parsed = llm.call_llm_json(
        client, prompts.VERIFY_ANSWER_SYSTEM, user_prompt,
        llm._TA_ANSWER_VERIFY, label="verify_answer",
    )
    logger.info(
        "PRIMITIVE end label=verify_candidate_answer report_id=%s status=%s",
        report.id,
        report.status,
    )
    return report


def verify_candidate_knowledge(
    ck: CandidateKnowledge,
    kb: KnowledgeBase,
    id_factory: Callable[[], str],
) -> VerificationReport:
    """
    DSL-MAP: PRIMITIVE-VERIFY-KNOWLEDGE (v2.1 merged-coherence semantics)
    """
    logger.info(
        "PRIMITIVE start label=verify_candidate_knowledge candidate_id=%s verifier=clingo kb_id=%s",
        ck.id,
        kb.id,
    )
    solo = solver.clingo_solve(ck.asp_program)
    if not solo.ok:
        logger.info(
            "PRIMITIVE end label=verify_candidate_knowledge status=skipped_solo reason=solo_grounding_failure"
        )
    elif not solo.satisfiable:
        report = VerificationReport(
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
        logger.info(
            "PRIMITIVE end label=verify_candidate_knowledge report_id=%s status=%s reason=%s",
            report.id,
            report.status,
            (report.reason or "")[:120],
        )
        return report

    merged_program = kb.asp_program.rstrip() + "\n\n" + ck.asp_program.strip() + "\n"
    merged = solver.clingo_solve(merged_program)
    if not merged.ok:
        report = VerificationReport(
            id=id_factory(),
            status="failed",
            reason=merged.error or "Clingo solve error on merged program",
            evidence="",
            raw_output=merged.error,
            verifier_kind="clingo",
        )
        logger.info(
            "PRIMITIVE end label=verify_candidate_knowledge report_id=%s status=%s reason=%s",
            report.id,
            report.status,
            (report.reason or "")[:120],
        )
        return report
    if not merged.satisfiable:
        report = VerificationReport(
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
        logger.info(
            "PRIMITIVE end label=verify_candidate_knowledge report_id=%s status=%s reason=%s",
            report.id,
            report.status,
            (report.reason or "")[:120],
        )
        return report

    report = VerificationReport(
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
    logger.info(
        "PRIMITIVE end label=verify_candidate_knowledge report_id=%s status=%s reason=%s",
        report.id,
        report.status,
        (report.reason or "")[:120],
    )
    return report


def abduce_answer(
    question: Question,
    kb: KnowledgeBase,
    candidate: CandidateAnswer,
    report: VerificationReport,
    registry: ArtifactRegistry,
    client,
    id_factory: Callable[[], str],
) -> Tuple[AbductiveHypothesis, Critique]:
    """
    DSL-MAP: PRIMITIVE-ABDUCE-ANSWER

    LLM-backed abductive revision. Produces an AbductiveHypothesis + a
    Critique that targets the existing candidate (I-CRITIQUE-MUST-TARGET-EXISTING-ARTIFACT).
    """
    logger.info(
        "PRIMITIVE start label=abduce_answer candidate_id=%s report_id=%s",
        candidate.id,
        report.id,
    )
    parsed = llm.call_llm_json(
        client, prompts.ABDUCE_SYSTEM, user_prompt,
        llm._TA_ABDUCTION, label="abduce_answer",
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


def abduce_knowledge(
    kb: KnowledgeBase,
    candidate: CandidateKnowledge,
    report: VerificationReport,
    registry: ArtifactRegistry,
    client,
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
    parsed = llm.call_llm_json(
        client, prompts.ABDUCE_SYSTEM, user_prompt,
        llm._TA_ABDUCTION, label="abduce_knowledge",
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
    client,
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
    parsed = llm.call_llm_json(
        client, prompts.REVISE_ANSWER_SYSTEM, user_prompt,
        llm._TA_REVISE_ANSWER, label="revise_answer",
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
    client,
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
    parsed = llm.call_llm_json(
        client, prompts.REVISE_KNOWLEDGE_SYSTEM, user_prompt,
        llm._TA_CAND_KNOWLEDGE, label="revise_knowledge",
    )
    return CandidateKnowledge(
        id=id_factory(),
        asp_program=parsed.asp_program,
        notes=parsed.notes,
    )


def generate_candidate_knowledge(
    source_text: str,
    kb: KnowledgeBase,
    client,
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
    parsed = llm.call_llm_json(
        client, prompts.GENERATE_KNOWLEDGE_SYSTEM, user_prompt,
        llm._TA_CAND_KNOWLEDGE, label="generate_knowledge",
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

    # Check for redundancy.
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
    Raise FlowFailed with a descriptive message.
    """
    raise FlowFailed(f"flow={flow_id} failed: {reason}")
