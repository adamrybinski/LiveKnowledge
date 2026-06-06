"""
LLM client initialization, JSON extraction, typed parsing, and Pydantic
boundary models.

This module owns the OpenAI client contract and everything that touches raw
LLM output. Flow logic lives in primitives.py and flows.py.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional, Type

import clingo
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

import artifacts
import prompts

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

import logging as _logging
logger = _logging.getLogger("asplearning.main2")


def get_client() -> OpenAI:
    """
    DSL-MAP: ROLE-ACCESS-INFRASTRUCTURE

    Returns the LLM client. Mirrors main.py initialization so we reuse the
    same .env contract (LLM_API_KEY, LLM_MODEL, LLM_BASE_URL).
    """
    if not LLM_API_KEY:
        raise RuntimeError("Missing LLM_API_KEY environment variable.")
    kwargs: Dict[str, Any] = {"api_key": LLM_API_KEY}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    logger.info(
        "Initializing LLM client model=%s base_url=%s",
        MODEL,
        BASE_URL or "<default>",
    )
    return OpenAI(**kwargs)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def extract_json(text: str) -> dict:
    """
    DSL-MAP: INTERMEDIATE-ARTIFACT-NORMALIZATION

    Best-effort extraction of a JSON object from LLM output. Tries:
      1. Direct json.loads
      2. Strip ```json fences
      3. Slice between first '{' and last '}'
    Raises artifacts.LLMArtifactError on total failure.
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
            raise artifacts.LLMArtifactError(
                f"JSON parse failed after slicing: {e2}", raw_text=text
            ) from e2

    raise artifacts.LLMArtifactError(
        f"JSON parse failed: no object in text", raw_text=text
    )


# ---------------------------------------------------------------------------
# Pydantic boundary models
# ---------------------------------------------------------------------------

class _CandidateAnswerContent(BaseModel):
    question_id: str = ""
    text: str
    rationale: str = ""


class _AnswerVerifyContent(BaseModel):
    status: Literal["verified", "rejected", "failed"]
    reason: str
    evidence: str = ""


class _CritiqueContent(BaseModel):
    target_id: str = ""
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class _AbductionContent(BaseModel):
    explanation: str
    repair_plan: str = ""
    critique: _CritiqueContent


class _ReviseAnswerContent(BaseModel):
    question_id: str = ""
    text: str
    rationale: str = ""


class _CandidateKnowledgeContent(BaseModel):
    asp_program: str = ""
    notes: str = ""


_TA_CAND_ANSWER = TypeAdapter(_CandidateAnswerContent)
_TA_ANSWER_VERIFY = TypeAdapter(_AnswerVerifyContent)
_TA_ABDUCTION = TypeAdapter(_AbductionContent)
_TA_REVISE_ANSWER = TypeAdapter(_ReviseAnswerContent)
_TA_CAND_KNOWLEDGE = TypeAdapter(_CandidateKnowledgeContent)


# ---------------------------------------------------------------------------
# LLM transport + retry
# ---------------------------------------------------------------------------

def _call_llm_raw(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    label: str,
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
    except Exception:
        logger.exception("LLM request failed label=%s", label)
        raise

    raw_text = (response.choices[0].message.content or "") if response.choices else ""
    if LOG_RAW_LLM:
        logger.debug("Raw LLM output label=%s:\n%s", label, raw_text)
    return raw_text


def call_llm_json(
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

    This is the single shared JSON extraction + retry path used by all
    primitives that call the LLM. Primitive functions in primitives.py build
    the prompts and pass them here; this helper owns transport + parsing.
    """
    last_error: Optional[Exception] = None
    last_raw: str = ""
    for attempt in range(1, MAX_JSON_RETRIES + 2):  # 1 initial + N retries
        try:
            raw = _call_llm_raw(client, system_prompt, user_prompt, label=label)
            last_raw = raw
            data = extract_json(raw)
            return adapter.validate_python(data)
        except (artifacts.LLMArtifactError, ValidationError) as e:
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
            raise
    raise artifacts.LLMArtifactError(
        f"LLM JSON could not be recovered after {MAX_JSON_RETRIES + 1} attempts: {last_error}",
        raw_text=last_raw,
    )
