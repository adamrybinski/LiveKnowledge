"""
LiveKnowledge — thin CLI entrypoint.

All business logic lives in flows.py, primitives.py, gap_report.py, and
the supporting modules. This file only:
  - parses CLI args
  - loads/saves files
  - calls Orchestrator.run_answer_question / run_integrate_knowledge
  - prints human-readable output

Behavioral contract is identical to the original single-file main.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Optional

import artifacts
import flows
import gap_report
import primitives

logger = logging.getLogger("asplearning.main2")


# ---------------------------------------------------------------------------
# KB / question / source-text loaders
# ---------------------------------------------------------------------------

def _load_kb(path: str) -> artifacts.KnowledgeBase:
    """Load an ASP knowledge base from a .lp file."""
    with open(path, "r", encoding="utf-8") as f:
        program = f.read()
    return artifacts.KnowledgeBase(
        id=f"kb-{os.path.basename(path)}",
        source_text=program,
        asp_program=program,
        metadata={"path": path, "notes": f"Loaded from {path}"},
    )


def _load_question(args: argparse.Namespace) -> artifacts.Question:
    if args.question:
        text = args.question.strip()
        qid = args.question_id or "q-cli"
    else:
        path = args.question_file
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        qid = args.question_id or f"q-{os.path.basename(path)}"
    return artifacts.Question(id=qid, text=text)


_H1_LINE_RE = re.compile(r"^\s*#\s+(.+?)\s*$")


def extract_first_h1_question(text: str) -> Optional[str]:
    """Return the text of the first markdown H1 (`# ...`) in `text`, or None."""
    for line in text.splitlines():
        m = _H1_LINE_RE.match(line)
        if m:
            return m.group(1).strip()
    return None


def _load_source_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Gap-report helpers (thin wrappers around gap_report module)
# ---------------------------------------------------------------------------

def _maybe_write_gap_report(path: Optional[str], report: gap_report.KnowledgeGapReport) -> None:
    if path is None:
        return
    if path == "-":
        print("\n=== KNOWLEDGE GAP REPORT ===")
        print(json.dumps({
            "question": report.question,
            "target_predicates": report.target_predicates,
            "gap_rationale": report.gap_rationale,
            "status": report.status,
        }, indent=2))
    else:
        gap_report.write_gap_report(path, report)
        print(f"\nGap report written to: {path}")
        if report.target_predicates:
            print(f"Target predicates: {report.target_predicates}")


def _surface_gaps_from_registry(registry: artifacts.ArtifactRegistry) -> None:
    """Print any KB gap predicates named in the latest CandidateAnswer rationale."""
    gap_artifact = None
    for rid in reversed(registry.ids()):
        a = registry.get(rid)
        if isinstance(a, artifacts.CandidateAnswer):
            gap_artifact = a
            break
    if gap_artifact and gap_artifact.rationale.strip():
        target_preds = gap_report.extract_target_predicates(gap_artifact.rationale)
        if target_preds:
            print(f"\nℹ KB may need predicates: {target_preds}")
            print("  To generate a gap report: re-run with --gap-report gaps.json")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_ask(args: argparse.Namespace, orch: flows.Orchestrator) -> int:
    kb = _load_kb(args.kb)
    question = _load_question(args)
    try:
        final, registry = orch.run_answer_question(
            question=question,
            kb=kb,
            max_iterations=args.max_iterations,
        )
    except primitives.FlowFailed as e:
        logger.error("answer_question failed: %s", e)
        return 1

    print("\n=== FINAL ANSWER ===\n")
    print(final.answer_text)
    print(f"\n(final_id={final.id} question_id={final.question_id})")
    print(f"(registry contains {len(registry)} artifacts: {registry.ids()})")

    # Optional gap report from the committed CandidateAnswer.
    gap_artifact = None
    for rid in reversed(registry.ids()):
        a = registry.get(rid)
        if isinstance(a, artifacts.CandidateAnswer):
            gap_artifact = a
            break
    if gap_artifact and gap_artifact.rationale.strip():
        gap_report_obj = gap_report.KnowledgeGapReport(
            question=question.text,
            committed_answer=final.answer_text,
            gap_rationale=gap_artifact.rationale,
            target_predicates=gap_report.extract_target_predicates(gap_artifact.rationale),
            status="open",
        )
        _maybe_write_gap_report(args.gap_report, gap_report_obj)
    else:
        if args.gap_report == "-":
            print("\n(no gap report: rationale was empty)")

    # Even without --gap-report, optionally surface gap predicates.
    if args.gap_report is None:
        _surface_gaps_from_registry(registry)

    return 0


def cmd_integrate(args: argparse.Namespace, orch: flows.Orchestrator) -> int:
    kb = _load_kb(args.kb)
    with open(args.program_file, "r", encoding="utf-8") as f:
        program = f.read()

    candidate = artifacts.CandidateKnowledge(
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
    except primitives.FlowFailed as e:
        logger.error("integrate_knowledge failed: %s", e)
        return 1

    print("\n=== INTEGRATION RESULT ===\n")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"result_id: {result.id}")
    print(f"new_kb_id: {new_kb.id}")
    print(f"updated_program_chars: {len(new_kb.asp_program)}")
    print(f"registry contains {len(registry)} artifacts")

    out_kb_path = args.out_kb or args.kb

    if result.success:
        # Surface any rejections that were auto-revised away.
        rejected_ids = [
            rid for rid in registry.ids()
            if isinstance(registry.get(rid), artifacts.VerificationReport)
            and registry.get(rid).status == "rejected"
        ]
        if rejected_ids:
            print(f"\n⚠ CONFLICT DETECTED AND RESOLVED")
            print(f"  Original candidate was rejected ({len(rejected_ids)} report(s)):")
            report = registry.get(rejected_ids[-1])
            print(f"  Reason: {report.reason[:200]}")
            print(f"  The system auto-revised and integrated a non-conflicting version.")
            print(f"  Original intent may have been altered — review final KB at {out_kb_path}")
        else:
            print("  No conflicts detected — candidate integrated directly.")

        with open(out_kb_path, "w", encoding="utf-8") as f:
            f.write(new_kb.asp_program)
        print(f"\naugmented KB written to: {out_kb_path}")
    else:
        return 1

    return 0


def cmd_learn(args: argparse.Namespace, orch: flows.Orchestrator) -> int:
    kb = primitives._load_kb(args.kb)

    # Resolve source text + gap context.
    src_path = args.unstructured
    gap_report_obj = None
    gap_context = ""
    gap_file_path = None
    if args.fill_gap:
        gap_file_path = args.fill_gap
        gap_report_obj = gap_report.load_gap_report(args.fill_gap)
        if gap_report_obj.target_predicates:
            gap_context = "Target predicates to fill: " + ", ".join(gap_report_obj.target_predicates)
            if gap_report_obj.gap_rationale:
                gap_context += "\n\nContext: " + gap_report_obj.gap_rationale
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

    # Step 1: generate CandidateKnowledge.
    ck_factory = artifacts.make_id_factory("ck")
    try:
        candidate = primitives.generate_candidate_knowledge(
            source_text=source_text,
            kb=kb,
            client=orch.client,
            id_factory=ck_factory,
            gap_context=gap_context,
        )
    except primitives.LLMArtifactError as e:
        logger.error("generate_knowledge failed: %s", e)
        return 1

    print("\n=== GENERATED CANDIDATE KNOWLEDGE ===\n")
    print(candidate.asp_program)
    if candidate.notes:
        print(f"\n--- notes ---\n{candidate.notes}")
    print(f"\n(candidate id={candidate.id})")

    # Step 2: integrate.
    try:
        result, new_kb, int_registry = orch.run_integrate_knowledge(
            candidate_knowledge=candidate,
            kb=kb,
            max_iterations=args.max_iterations,
        )
    except primitives.FlowFailed as e:
        logger.error("integrate_knowledge failed: %s", e)
        return 1

    print("\n=== INTEGRATION RESULT ===\n")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"result_id: {result.id}")
    print(f"new_kb_id: {new_kb.id}")
    print(f"updated_program_chars: {len(new_kb.asp_program)}")
    print(f"integrate registry size: {len(int_registry)}")

    if not result.success:
        return 1

    # Step 3: persist augmented KB.
    out_kb_path = args.out_kb or args.kb
    with open(out_kb_path, "w", encoding="utf-8") as f:
        f.write(new_kb.asp_program)
    print(f"\naugmented KB written to: {out_kb_path}")
    if out_kb_path == args.kb:
        print("(source KB overwritten in place; git diff will show the new facts)")

    # Step 3.5: gap-resolution check.
    if gap_report_obj is not None and gap_report_obj.status == "open":
        resolution = gap_report.verify_gap_resolution(gap_report_obj, new_kb.asp_program)
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
        if gap_file_path != "-":
            gap_report.write_gap_report(gap_file_path, gap_report_obj)
            print(f"  gap file updated: {gap_file_path}")
        else:
            print("  (gap file was stdin; resolution not persisted)")
    elif gap_report_obj is not None and gap_report_obj.status == "closed":
        print(f"\n(gap report already closed; skipping resolution check)")

    # Step 4: optionally re-ask.
    if args.skip_reask:
        print("\n(re-ask skipped via --skip-reask)")
        return 0

    question_text = args.question or extract_first_h1_question(source_text)
    if not question_text:
        print("\n(no question found in source text and --question not given; skipping re-ask)")
        return 0

    print(f"\n=== RE-ASK AGAINST AUGMENTED KB ===\n")
    print(f"question: {question_text}")
    question = artifacts.Question(
        id=args.question_id or f"q-{os.path.basename(args.unstructured or 'source')}",
        text=question_text,
    )
    try:
        final, ask_registry = orch.run_answer_question(
            question=question,
            kb=new_kb,
            max_iterations=args.max_iterations,
        )
    except primitives.FlowFailed as e:
        logger.error("answer_question (re-ask) failed: %s", e)
        return 1

    print(f"\n--- final answer ---\n{final.answer_text}")
    print(f"\n(final_id={final.id} question_id={final.question_id})")
    print(f"(ask registry size: {len(ask_registry)})")

    # Optional gap report from re-ask step.
    if args.gap_report is not None:
        gap_artifact = None
        for rid in reversed(ask_registry.ids()):
            a = ask_registry.get(rid)
            if isinstance(a, artifacts.CandidateAnswer):
                gap_artifact = a
                break
        if gap_artifact and gap_artifact.rationale.strip():
            report = gap_report.KnowledgeGapReport(
                question=question_text,
                committed_answer=final.answer_text,
                gap_rationale=gap_artifact.rationale,
                target_predicates=gap_report.extract_target_predicates(gap_artifact.rationale),
                status="open",
            )
            _maybe_write_gap_report(args.gap_report, report)

    return 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiveKnowledge — v2.1 implementation",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="Answer a question against a knowledge base")
    p_ask.add_argument("--kb", default="kb.lp", help="Path to .lp knowledge base file")
    p_ask.add_argument("--question", help="Question text")
    p_ask.add_argument("--question-file", help="Path to file with the question text")
    p_ask.add_argument("--question-id", help="Override the question artifact id")
    p_ask.add_argument(
        "--max-iterations", type=int, default=3,
        help="Flow bound (default 3)",
    )
    p_ask.add_argument(
        "--gap-report", default=None, nargs="?", const="-",
        help="Path to write a gap signal JSON file (default: print to stdout with '-')",
    )

    p_int = sub.add_parser("integrate", help="Integrate a candidate ASP program into a KB")
    p_int.add_argument("--kb", required=True, help="Path to existing .lp knowledge base")
    p_int.add_argument("--program-file", required=True, help="Path to candidate ASP program")
    p_int.add_argument("--notes", default="", help="Optional modeling notes")
    p_int.add_argument("--candidate-id", help="Override candidate artifact id")
    p_int.add_argument(
        "--max-iterations", type=int, default=3,
        help="Flow bound (default 3)",
    )

    p_learn = sub.add_parser(
        "learn",
        help="Learn from source text: extract knowledge, integrate, optionally re-ask",
    )
    p_learn.add_argument("--kb", default="kb.lp", help="Path to .lp knowledge base file")
    p_learn.add_argument("--unstructured", default=None, help="Path to source text file")
    p_learn.add_argument("--text", default=None, help="Raw text to extract knowledge from")
    p_learn.add_argument("--fill-gap", default=None, help="Path to gap signal JSON file")
    p_learn.add_argument("--out-kb", default=None, help="Where to write augmented KB")
    p_learn.add_argument("--question", default=None, help="Question for re-ask step")
    p_learn.add_argument(
        "--max-iterations", type=int, default=3,
        help="Flow bound for integrate and re-ask (default 3)",
    )
    p_learn.add_argument(
        "--skip-reask", action="store_true",
        help="Integrate only; skip re-ask step",
    )
    p_learn.add_argument("--question-id", default=None, help="Override question id for re-ask")
    p_learn.add_argument(
        "--gap-report", default=None, nargs="?", const="-",
        help="Path to write gap signal JSON from re-ask step",
    )

    args = parser.parse_args()
    orch = flows.Orchestrator()

    if args.cmd == "ask":
        raise SystemExit(cmd_ask(args, orch))
    elif args.cmd == "integrate":
        raise SystemExit(cmd_integrate(args, orch))
    elif args.cmd == "learn":
        raise SystemExit(cmd_learn(args, orch))


if __name__ == "__main__":
    main()
