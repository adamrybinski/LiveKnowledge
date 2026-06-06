"""
LiveKnowledge flows: answer_question, integrate_knowledge, and the Orchestrator
façade.

DSL-MAP:
- FLOW-ANSWER-QUESTION
- FLOW-INTEGRATE-KNOWLEDGE
- ORCHESTRATOR-FACADE
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import artifacts
import gap_report
import llm
import primitives
import prompts

logger = logging.getLogger("asplearning.main2")

# Re-export for convenience.
Question = primitives.Question
KnowledgeBase = primitives.KnowledgeBase
CandidateAnswer = primitives.CandidateAnswer
CandidateKnowledge = primitives.CandidateKnowledge
VerificationReport = primitives.VerificationReport
AbductiveHypothesis = primitives.AbductiveHypothesis
IntegrationResult = primitives.IntegrationResult
FinalAnswer = primitives.FinalAnswer
FlowFailed = primitives.FlowFailed


# ---------------------------------------------------------------------------
# Hard-coded flows
# ---------------------------------------------------------------------------

def answer_question(
    question: Question,
    kb: KnowledgeBase,
    registry: artifacts.ArtifactRegistry,
    client,
    id_factory,
    max_iterations: int = 3,
) -> FinalAnswer:
    """
    DSL-MAP: FLOW-ANSWER-QUESTION

    Direct translation of system_model.yaml flows.answer_question.
    """
    # T-ANSWER-GENERATE
    candidate = primitives.generate_answer(question, kb, client, id_factory)
    registry.register(candidate)
    logger.info("T-ANSWER-GENERATE produced candidate id=%s", candidate.id)

    iteration = 0
    while True:
        iteration += 1
        if iteration > max_iterations:
            primitives.fail_flow(
                "answer_question",
                f"iterations_exceeded (max={max_iterations})",
            )

        # C-ANSWER-VERIFY
        report = primitives.verify_candidate_answer(candidate, kb, client, id_factory)
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
            final = primitives.commit_answer(candidate, report, id_factory)
            registry.register(final)
            logger.info("T-ANSWER-COMMIT produced final id=%s", final.id)
            return final

        if report.status == "rejected":
            pass  # fall through to abduce
        elif report.status == "failed":
            if primitives.is_recoverable_failure(report):
                logger.info("Verification failed but recoverable; entering abduce.")
            else:
                primitives.fail_flow(
                    "answer_question",
                    f"unrecoverable_verification_failure: {report.reason}",
                )
        else:  # pragma: no cover
            primitives.fail_flow("answer_question", f"unknown status: {report.status}")

        # T-ANSWER-ABDUCE
        hypothesis, critique = primitives.abduce_answer(
            question, kb, candidate, report, registry, client, id_factory
        )
        registry.register(hypothesis)
        registry.register(critique)

        # T-ANSWER-REVISE — produces a new candidate (immutable snapshot)
        candidate = primitives.revise_answer(
            candidate, hypothesis, critique, client, id_factory
        )
        registry.register(candidate)
        logger.info("T-ANSWER-REVISE iteration=%d new candidate id=%s", iteration, candidate.id)


def integrate_knowledge(
    candidate_knowledge: CandidateKnowledge,
    kb: KnowledgeBase,
    registry: artifacts.ArtifactRegistry,
    client,
    id_factory,
    max_iterations: int = 3,
    auto_revise: bool = True,
) -> Tuple[artifacts.IntegrationResult, KnowledgeBase]:
    """
    DSL-MAP: FLOW-INTEGRATE-KNOWLEDGE

    Direct translation of system_model.yaml flows.integrate_knowledge.
    Returns (IntegrationResult, new_kb).
    """
    iteration = 0
    candidate = candidate_knowledge

    # Re-assign id from factory to avoid collision with reports.
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
            primitives.fail_flow(
                "integrate_knowledge",
                f"iterations_exceeded (max={max_iterations})",
            )

        # T-KNOWLEDGE-VERIFY
        report = primitives.verify_candidate_knowledge(candidate, kb, id_factory)
        registry.register(report)
        logger.info(
            "T-KNOWLEDGE-VERIFY iteration=%d report id=%s status=%s",
            iteration,
            report.id,
            report.status,
        )

        # D-KNOWLEDGE-VERIFIED — four-way branch
        if report.status == "verified":
            result, new_kb = primitives.merge_verified_knowledge(
                candidate, report, kb, id_factory
            )
            registry.register(result)
            logger.info(
                "T-KNOWLEDGE-INTEGRATE success=%s result id=%s new_kb id=%s",
                result.success,
                result.id,
                new_kb.id,
            )
            return result, new_kb

        if report.status == "rejected":
            if not auto_revise:
                primitives.fail_flow(
                    "integrate_knowledge",
                    f"CONFLICT: candidate contradicts existing KB. "
                    f"Verification rejected. Reason: {report.reason}. "
                    f"Set auto_revise=True or resolve the conflict manually before retrying.",
                )
            pass
        elif report.status == "failed":
            if primitives.is_recoverable_failure(report):
                logger.info("Knowledge verification failed but recoverable; entering abduce.")
            else:
                primitives.fail_flow(
                    "integrate_knowledge",
                    f"unrecoverable_verification_failure: {report.reason}",
                )
        else:  # pragma: no cover
            primitives.fail_flow("integrate_knowledge", f"unknown status: {report.status}")

        # T-KNOWLEDGE-ABDUCE
        hypothesis, critique = primitives.abduce_knowledge(
            kb, candidate, report, registry, client, id_factory
        )
        registry.register(hypothesis)
        registry.register(critique)

        # T-KNOWLEDGE-REVISE
        candidate = primitives.revise_knowledge(
            candidate, hypothesis, critique, client, id_factory
        )
        registry.register(candidate)
        logger.info(
            "T-KNOWLEDGE-REVISE iteration=%d new candidate id=%s",
            iteration,
            candidate.id,
        )


# ---------------------------------------------------------------------------
# Orchestrator façade
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    DSL-MAP: ORCHESTRATOR-FACADE

    Wires dependencies (LLM client, registry, id factory) and dispatches to
    the hard-coded flow functions.
    """

    def __init__(self, client=None) -> None:
        self.client = client or llm.get_client()

    def run_answer_question(
        self,
        question: Question,
        kb: KnowledgeBase,
        max_iterations: int = 3,
    ) -> Tuple[FinalAnswer, artifacts.ArtifactRegistry]:
        registry = artifacts.ArtifactRegistry()
        id_factory = artifacts.make_id_factory("ca")
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
        max_iterations: int = 3,
        auto_revise: bool = True,
    ) -> Tuple[artifacts.IntegrationResult, KnowledgeBase, artifacts.ArtifactRegistry]:
        registry = artifacts.ArtifactRegistry()
        id_factory = artifacts.make_id_factory("ck")
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
