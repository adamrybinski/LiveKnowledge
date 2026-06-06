"""
Clingo solver adapter and solver-specific verification helpers.

DSL-MAP:
- DSL-MAP: CLINGO-SOLVE-RESULT (SolveResult)
- DSL-MAP: CLINGO-ADAPTER (clingo_solve)
"""

from __future__ import annotations

import clingo
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import artifacts

MAX_MODELS = int(__import__("os").getenv("MAX_MODELS", "3"))

import logging
logger = logging.getLogger("asplearning.main2")


@dataclass
class SolveResult:
    """
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
