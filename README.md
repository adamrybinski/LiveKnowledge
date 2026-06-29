# LiveKnowledge

A **knowledge engineering** CLI tool that combines LLM reasoning with Answer Set Programming (Clingo) to build, verify, and query structured knowledge from unstructured text.

---

## Overview

LiveKnowledge implements the `system_model.yaml` v2.1 architecture — a **closed-loop knowledge lifecycle**:

```
ask ──► answer question from KB
 │        │
 │    [gap detected]          learn ──► extract knowledge from text
 │        │                                │
 │        ▼                                ▼
 │    gaps.json ◄────── verify_gap_resolution
 │        │                                │
 │        └────────── fill-gap ────────────┘
```

**Three CLI subcommands**:

| Command | Purpose |
|---------|---------|
| `ask` | Answer a question using the knowledge base |
| `learn` | Extract structured facts from unstructured text and merge them into the KB |
| `integrate` | Directly merge a hand-written or pre-existing ASP program into the KB |

---

## Setup

```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install clingo openai python-dotenv pydantic

# Configure LLM (create .env with your API key)
echo "LLM_API_KEY=sk-..." >> .env
echo "LLM_MODEL=openrouter/..." >> .env
echo "LLM_BASE_URL=https://openrouter.ai/api/v1" >> .env

# Optional tuning
echo "LOG_LEVEL=INFO" >> .env       # DEBUG for verbose LLM output
echo "MAX_RETRIES=4" >> .env       # max revision attempts
echo "DEFAULT_MAX_ITERATIONS=3" >> .env
```

---

## Usage

## Website / Pages deploy (adamrybinski-com)

The static site lives in `blog-site/` and is deployed to **Cloudflare Pages** under the project **`adamrybinski-com`**.

```bash
cd blog-site
wrangler pages deploy . --project-name adamrybinski-com
```

The Pages settings are documented in `blog-site/wrangler.toml`.

### `ask` — Answer a question against the knowledge base

```bash
# Basic question
python main.py ask --kb /path/to/knowledge/kb.lp --question "What factors affect profit margins?"

# Question from a file
python main.py ask --kb /path/to/knowledge/kb.lp --question-file my-question.txt

# Generate a gap report (identifies missing predicates)
python main.py ask --kb /path/to/knowledge/kb.lp \
  --question "What are the most profitable crafted items?" \
  --gap-report /path/to/knowledge/gaps.json
```

The `--gap-report` flag produces a structured JSON file listing:
- **`target_predicates`** — predicate signatures (`pred/arity`) the KB is missing
- **`gap_rationale`** — human-readable description of what would improve the answer
- **`status: "open"`** — ready to be consumed by `learn --fill-gap`

Print to stdout (no file written):
```bash
python main.py ask --kb /path/to/knowledge/kb.lp \
  --question "..." --gap-report
```

**Optional**: after inspection, you can manually add a `target_review` field to the gap report to exclude known-bad targets:
```json
{
  "target_review": { "drop": ["garbage_predicate/1"] }
}
```

---

### `learn` — Extract knowledge from unstructured text

```bash
# Basic — extract from text and merge into KB
python main.py learn \
  --kb /path/to/knowledge/kb.lp \
  --unstructured /path/to/knowledge/source.txt \
  --question "What factors affect profit margins?"

# Targeted extraction — fill gaps identified by ask
python main.py learn \
  --kb /path/to/knowledge/kb.lp \
  --unstructured /path/to/knowledge/source.txt \
  --fill-gap /path/to/knowledge/gaps.json \
  --question "What are the most profitable crafted items?"
```

The `--fill-gap` flag does three things:

1. **Prioritises extraction** — the LLM targets the missing predicates listed in the gap report
2. **Verifies resolution** — after integration, checks whether each target predicate now appears in the augmented KB
3. **Updates the gap file** — records which predicates were found, which remain missing, and any arity mismatches

Output example:
```
  recipe/2 — resolved
  profit/2 — exact target missing
    related predicate found with different arity: profit/3
  garbage_pred/1 — excluded by review (target_review.drop)
```

**Other flags**:

| Flag | Purpose |
|------|---------|
| `--out-kb <path>` | Write augmented KB elsewhere instead of overwriting in place |
| `--skip-reask` | Integrate only, skip the re-answer step |
| `--max-iterations N` | Override the default revision loop bound |

---

### `integrate` — Directly merge an ASP program

For hand-written or external ASP fragments that bypass the extraction step:

```bash
python main.py integrate \
  --kb /path/to/knowledge/kb.lp \
  --program-file my-rules.lp \
  --notes "Hand-written rules"
```

The fragment goes through the same merged-coherence verification (Clingo checks that the candidate doesn't contradict existing KB facts). On success it's merged into the KB.

---

## Gap report lifecycle

The gap report is a structured orchestrator artifact that bridges `ask` and `learn`:

```
state: open                    state: open (updated)     state: closed
┌─────────────────┐            ┌─────────────────────┐   ┌─────────────────────┐
│ ask produces it  │   learn    │ resolution populated│   │ all targets found   │
│ target_predicates│ ─────►    │ some predicates     │   │ or reviewed out     │
│ status=open      │           │ still missing        │   │ status=closed       │
│ resolution=null  │           │ status=open          │   │                     │
└─────────────────┘            └─────────────────────┘   └─────────────────────┘
         │                             │
         │ reuse with different        │ no — still missing
         │ source text                 ▼
         │                     find better source, run again
         ▼
```

### Gap resolution states

| CLI output | Meaning |
|------------|---------|
| `recipe/2 — resolved` | Predicate exists at exact arity in KB |
| `profit/2 — exact target missing` | Predicate name not found at requested arity |
| `→ related: profit/3` | Same predicate name exists at a different arity (schema evolution) |
| `garbage_pred/1 — excluded by review` | Target was manually dropped via `target_review.drop` |

---

## Architecture summary

```
                      answer_question flow
                      ┌──────────────────────────┐
       question ─────►│  generate (LLM)           │
                      │  verify (LLM)             │
                      │  abduce / revise (loop)   │
                      │  commit                   │────► final_answer
                      └──────────────────────────┘
                               │
                      gap_rationale (in rationale)
                               │
                               ▼
                      knowledge_gap_report (JSON)
                      ┌──────────────────────────┐
                      │  target_predicates        │
                      │  status: open             │
                      │  target_review (optional) │
                      └──────────────────────────┘
                               │
                               ▼
                    learn composition
                    ┌──────────────────────────────────┐
     source text ──►│  generate_knowledge (LLM,        │
                    │    with gap_context targeting)    │
                    │  integrate_knowledge (Clingo      │
                    │    merged-coherence verify)       │
                    │  persist KB to disk               │
                    │  verify_gap_resolution (text scan)│
                    │  update gap file                  │
                    │  re-ask (optional)                │
                    └──────────────────────────────────┘
```

### Key concepts

- **Knowledge Base (`kb.lp`)** — pure ASP facts and rules. No `#show` directives (projection is a per-run concern). Lives separately from the engine code.
- **Candidate knowledge** — extracted ASP fragments that are additive and merged into the KB after verification.
- **Merged-coherence verification** — Clingo solves the combined KB + candidate program. If unsatisfiable, the candidate is rejected as contradictory.
- **Abductive revision** — when a candidate is rejected, the LLM generates a hypothesis about the failure and a repair plan, then revises the candidate.
- **Gap report** — structured orchestrator artifact. Bridges the answer and learning flows. Not a flow artifact — lives outside the registry machinery.

---

## Requirements

- Python 3.11+
- [Clingo](https://potassco.org/clingo/) — answer set solver
- OpenAI-compatible API (OpenAI, OpenRouter, etc.)
- `python-dotenv` for environment configuration
