"""
LLM client initialization, JSON extraction, typed parsing, and Pydantic boundary models.

This module owns the OpenAI client contract and everything that touches raw LLM output.
Flow logic lives in primitives.py and flows.py.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

import dotenv
from dotenv import load_dotenv

import artifacts
import prompts


load_dotenv()

MODEL = os.getenv("LLM_MODEL")
BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

MAX_JSON_RETRIES = int(os.getenv("MAX_JSON_RETRIES", "2"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "0"))
LOG_RAW_LLM = os.getenv("LOG_RAW_LLM", "0") == "1"

import logging
logger = logging.getLogger("asplearning.llm")


def get_client() -> OpenAI:
    if not LLM_API_KEY:
        raise RuntimeError("Missing LLM_API_KEY environment variable.")
    kwargs: dict[str, Any] = {"api_key": LLM_API_KEY}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    logger.info("Initializing LLM client model=%s base_url=%s", MODEL, BASE_URL or "<default>")
    return OpenAI(**kwargs)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def extract_json(text: str) -> dict:
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

    raise artifacts.LLMArtifactError("JSON parse failed: no object in text", raw_text=text)


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
    explanation: str = ""
    repair_plan: str = ""
    critique: '_CritiqueContent'


class _ReviseAnswerContent(BaseModel):
    question_id: str = ""
    text: str
    rationale: str = ""


class _CandidateKnowledgeContent(BaseModel):
    asp_program: str = ""
    notes: str = ""


_TA_CAND_ANSWER = TypeAdapter(_CandidateAnswerContent)
_TA_ANSWER_VERIFY = TypeAdapter(_AnswerVerifyContent)
_TA_REVISE_ANSWER = TypeAdapter(_ReviseAnswerContent)
_TA_CAND_KNOWLEDGE = TypeAdapter(_CandidateKnowledgeContent)
_AbductionContent.model_rebuild()
_TA_ABDUCTION = TypeAdapter(_AbductionContent)


def _call_llm_raw(client: OpenAI, system_prompt: str, user_prompt: str, label: str) -> str:
    logger.info(
        "LLM call label=%s model=%s system_chars=%d user_chars=%d",
        label,
        MODEL,
        len(system_prompt),
        len(user_prompt),
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
    last_error: Optional[Exception] = None
    last_raw: str = ""
    for attempt in range(1, MAX_JSON_RETRIES + 2):
        try:
            raw_text = _call_llm_raw(client, system_prompt, user_prompt, label=label)
            last_raw = raw_text
            logger.info("LLM raw_length=%d label=%s attempt=%d", len(raw_text), label, attempt)
            data = extract_json(raw_text)
            model = adapter.validate_python(data)
        except (artifacts.LLMArtifactError, ValidationError) as e:
            last_error = e
            logger.warning(
                "LLM JSON validation failed label=%s attempt=%d error=%s",
                label,
                attempt,
                e,
            )
            if LOG_RAW_LLM:
                logger.debug("Raw LLM output label=%s attempt=%d:\n%s", label, attempt, last_raw)
            if attempt <= MAX_JSON_RETRIES:
                if RETRY_DELAY > 0:
                    time.sleep(RETRY_DELAY)
                continue
            raise artifacts.LLMArtifactError(
                f"LLM JSON could not be recovered after {MAX_JSON_RETRIES + 1} attempts: {last_error}",
                raw_text=last_raw,
            ) from last_error
        return model
    raise RuntimeError("unreachable")
