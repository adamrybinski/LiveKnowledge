"""
Tests for LiveKnowledge v2.1 primitives and gap-report utilities.

Coverage targets (from requirements):
  - verify_gap_resolution: exact arity match vs different arity
  - integrate_knowledge: auto_revise=False rejection
  - merge_verified_knowledge: redundancy skip
  - classify_verification_failure: recoverable / unrecoverable behavior

Run with:
  python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

# Make sure the repo root is importable when running pytest from inside tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import artifacts
import gap_report
import primitives
import flows


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def dummy_client():
    """A stub OpenAI client that returns canned responses."""

    class _StubCompletions:
        def create(self, **kwargs):
            class _Msg:
                content = json.dumps({
                    "question_id": "q-1",
                    "text": "Stub answer",
                    "rationale": "",
                })

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _StubChat:
        completions = _StubCompletions()

    return types.SimpleNamespace(chat=mock.MagicMock(completions=_StubCompletions()))


def make_kb(text: str = "a(1).", id_: str = "kb-1") -> artifacts.KnowledgeBase:
    return artifacts.KnowledgeBase(
        id=id_,
        source_text=text,
        asp_program=text,
        metadata={},
    )


def make_candidate(program: str = "b(2).") -> artifacts.CandidateKnowledge:
    return artifacts.CandidateKnowledge(id="ck-1", asp_program=program)


# ---------------------------------------------------------------------------
# Gap-report tests
# ---------------------------------------------------------------------------

class TestVerifyGapResolution:
    """Exact arity vs arity-drift reporting."""

    def test_exact_match_found(self):
        kb = "target(1,2). other(3)."
        gr = gap_report.KnowledgeGapReport(
            question="q",
            committed_answer="a",
            gap_rationale="missing target/2",
            target_predicates=["target/2"],
            status="open",
        )
        res = gap_report.verify_gap_resolution(gr, kb)
        assert "target/2" in res.predicates_found
        assert not res.predicates_still_missing
        assert res.all_resolved is True
        assert res.related_predicates_found == {}

    def test_exact_match_missing(self):
        kb = "other(3)."
        gr = gap_report.KnowledgeGapReport(
            question="q",
            committed_answer="a",
            gap_rationale="missing target/2",
            target_predicates=["target/2"],
            status="open",
        )
        res = gap_report.verify_gap_resolution(gr, kb)
        assert "target/2" in res.predicates_still_missing
        assert "target/2" not in res.predicates_found
        assert res.all_resolved is False

    def test_arity_drift_reported(self):
        kb = "target(1,2,3)."
        gr = gap_report.KnowledgeGapReport(
            question="q",
            committed_answer="a",
            gap_rationale="missing target/2",
            target_predicates=["target/2"],
            status="open",
        )
        res = gap_report.verify_gap_resolution(gr, kb)
        assert "target/2" in res.predicates_still_missing
        assert res.related_predicates_found.get("target/2") == ["target/3"]

    def test_target_review_drop_excluded(self):
        kb = "target(1,2)."
        gr = gap_report.KnowledgeGapReport(
            question="q",
            committed_answer="a",
            gap_rationale="missing target/2",
            target_predicates=["target/2"],
            target_review={"drop": ["target/2"]},
            status="open",
        )
        res = gap_report.verify_gap_resolution(gr, kb)
        assert "target/2" in res.predicates_dropped
        assert not res.predicates_still_missing
        assert res.all_resolved is True


# ---------------------------------------------------------------------------
# Solver / verification tests
# ---------------------------------------------------------------------------

class TestVerifyCandidateKnowledge:
    """Merged-coherence semantics and auto_revise=False rejection."""

    def test_merge_satisfiable_returns_verified(self):
        kb = make_kb("a(1).")
        ck = make_candidate("b(2).")
        id_factory = artifacts.make_id_factory("ck")
        report = primitives.verify_candidate_knowledge(ck, kb, id_factory)
        # a(1). + b(2). is satisfiable → verified
        assert report.status == "verified"
        assert report.verifier_kind == "clingo"

    def test_merge_unsat_returns_rejected(self):
        kb = make_kb("p(1). :- p(1), q(1).")
        ck = make_candidate("q(1).")
        id_factory = artifacts.make_id_factory("ck")
        # Merged: p(1). q(1). :- p(1), q(1). => UNSAT
        report = primitives.verify_candidate_knowledge(ck, kb, id_factory)
        assert report.status == "rejected"

    def test_self_contradicting_candidate_rejected(self):
        # candidate alone is UNSAT: p. and :- p.
        kb = make_kb("")
        ck = make_candidate("p.\n:- p.")
        id_factory = artifacts.make_id_factory("ck")
        report = primitives.verify_candidate_knowledge(ck, kb, id_factory)
        assert report.status == "rejected"
        assert "self-contradictory" in report.reason.lower()


class TestMergeVerifiedKnowledgeRedundancy:
    """Redundant candidate fragments are skipped."""

    def test_redundant_candidate_not_integrated(self):
        kb = make_kb("a(1).\nb(2).\n")
        ck = make_candidate("a(1).\nb(2).\n")
        id_factory = artifacts.make_id_factory("ck")
        report = artifacts.VerificationReport(
            id="vr-1",
            status="verified",
            reason="ok",
            verifier_kind="clingo",
        )
        result, new_kb = primitives.merge_verified_knowledge(ck, report, kb, id_factory)
        assert result.success is True
        assert "redundant" in result.message.lower()
        # Program unchanged.
        assert new_kb.asp_program == kb.asp_program

    def test_non_redundant_candidate_integrated(self):
        kb = make_kb("a(1).\n")
        ck = make_candidate("b(2).\n")
        id_factory = artifacts.make_id_factory("ck")
        report = artifacts.VerificationReport(
            id="vr-1",
            status="verified",
            reason="ok",
            verifier_kind="clingo",
        )
        result, new_kb = primitives.merge_verified_knowledge(ck, report, kb, id_factory)
        assert result.success is True
        assert "integrated" in result.message.lower()
        assert "b(2)." in new_kb.asp_program
        # Provenance metadata recorded.
        assert new_kb.metadata.get("last_integrated_candidate") == "ck-1"


# ---------------------------------------------------------------------------
# Failure classifier tests
# ---------------------------------------------------------------------------

class TestClassifyVerificationFailure:
    """recoverable vs unrecoverable on failed reports."""

    @pytest.mark.parametrize("reason,expected", [
        ("timeout after 30s", "recoverable"),
        ("rate limit exceeded", "recoverable"),
        ("connection reset", "recoverable"),
        ("temporary service error", "recoverable"),
        ("invalid api key", "unrecoverable"),
        ("authentication failed", "unrecoverable"),
        ("model not found", "unrecoverable"),
        ("unknown transport failure", "unrecoverable"),  # default for LLM is unrecoverable
    ])
    def test_llm_failure(self, reason, expected):
        report = artifacts.VerificationReport(
            id="vr-llm",
            status="failed",
            reason=reason,
            verifier_kind="llm",
        )
        assert primitives.classify_verification_failure(report) == expected

    @pytest.mark.parametrize("reason,expected", [
        ("syntax error: unexpected token", "recoverable"),
        ("unsafe variable in rule", "recoverable"),
        ("grounding failed for atom", "recoverable"),
        ("parse error near line 3", "recoverable"),
        ("out of memory during solve", "unrecoverable"),
        ("segfault in solver", "unrecoverable"),
        ("segfault detected", "unrecoverable"),
        ("some other solver error", "recoverable"),  # default for clingo is recoverable
    ])
    def test_clingo_failure(self, reason, expected):
        report = artifacts.VerificationReport(
            id="vr-cl",
            status="failed",
            reason=reason,
            verifier_kind="clingo",
        )
        assert primitives.classify_verification_failure(report) == expected

    def test_non_failed_status_is_unrecoverable(self):
        report = artifacts.VerificationReport(
            id="vr-ok",
            status="verified",
            reason="ok",
            verifier_kind="llm",
        )
        assert primitives.classify_verification_failure(report) == "unrecoverable"


# ---------------------------------------------------------------------------
# Flow-level tests
# ---------------------------------------------------------------------------

class TestIntegrateKnowledgeAutoReviseFalse:
    """auto_revise=False must still fail immediately on rejected candidate."""

    def test_rejected_candidate_fails_without_revision(self):
        kb = make_kb("p(1). :- p(1), q(1).")
        # candidate introduces q(1), making merged UNSAT.
        ck = artifacts.CandidateKnowledge(id="ck-bad", asp_program="q(1).")
        registry = artifacts.ArtifactRegistry()
        id_factory = artifacts.make_id_factory("ck")
        fake_client = mock.MagicMock()

        with pytest.raises(artifacts.FlowFailed):
            flows.integrate_knowledge(
                candidate_knowledge=ck,
                kb=kb,
                registry=registry,
                client=fake_client,
                id_factory=id_factory,
                max_iterations=2,
                auto_revise=False,
            )
