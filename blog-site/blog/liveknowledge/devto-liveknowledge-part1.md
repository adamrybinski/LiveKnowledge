---
title: "LiveKnowledge: Engineering Verifiable Knowledge (Part 1)"
published: false
description: "Closed-loop knowledge engineering with LLMs + Answer Set Programming (ASP). Part 1 covers the v2.1 architecture: LLM-Modulo framework, CEGIS, the hermeneutic gap lifecycle, and arity-drift as schema evolution."
tags: [ai, asp, nlp, neurosymbolic, python]
canonical_url: https://adamrybinski.com/blog/liveknowledge/
cover_image: https://adamrybinski.com/blog/liveknowledge/cover.png
series: LiveKnowledge
---

*Part 1 of a series. Part 2 coming soon — covers semantic gap-completeness, non-monotonic retractions, and multi-shot reasoning.*

**LiveKnowledge** is a CLI tool that closes the loop between unstructured text and structured, verifiable knowledge. It combines large language models with Answer Set Programming (via [Clingo](https://potassco.org/clingo/)) to extract, verify, and query facts in an evolving knowledge base. This post covers the v2.1 architecture — what it does, why it works, and where it sits in the broader neuro-symbolic landscape.

---

## The Problem

Most knowledge tools treat knowledge as vectors in a database. You can retrieve them, but you cannot *check* them. If two facts contradict each other, the vector store won't tell you. RAG pipelines can ground generation in retrieved documents, but they have no way of knowing whether the retrieved facts form a consistent logical theory.

LiveKnowledge treats knowledge as **logic programs** — ASP facts and rules that a solver can verify for consistency. Every integration goes through a **merged-coherence check**: Clingo solves the combined KB + candidate program. If unsatisfiable, the candidate is rejected as contradictory. This is the core invariant: *no contradiction ever enters the knowledge base*.

---

## Architecture: The LLM-Modulo Framework

The architecture follows what the literature calls the **LLM-Modulo framework** [[1]](#ref-shetye)[[2]](#ref-ishay). An LLM proposes candidate knowledge (or candidate answers), but a separate, verifiable module — here, Clingo — acts as an unforgiving mathematical gatekeeper. If the solver finds a contradiction, the candidate is rejected and the LLM must revise.

This is a direct application of **Counterexample-Guided Inductive Synthesis (CEGIS)** [[3]](#ref-orvalho). The solver produces a counterexample (the unsatisfiable core of the merged program), and the LLM uses that failure trace as a signal for abductive revision. The system doesn't just accept or reject — it explains *why* and rewrites.

```
                       answer_question flow

      question ──► generate (LLM) ──► verify (LLM) ──► D ──►
                                            ▲           │
                                            │      [rejected]
                                            │           │
                                            │     abduce / revise
                                            │◄──────── loop ──────
                                                       │
                                                  [verified]
                                                       │
                                                    commit ──►
                                                       final
                                                      answer
                           │
                           │ gap_rationale (missing predicates)
                           ▼
                  knowledge_gap_report (JSON)
                   ┌──────────────────────┐
                   │ target_predicates     │
                   │ status: "open"       │──── human review ──► drop
                   │ resolution: null     │
                   └──────────────────────┘
                           │
                           ▼
                       learn composition

      source text ──► generate_knowledge (LLM + gap_context)
                           │
                           ▼
                 integrate_knowledge (Clingo verify)
                           │
                           ▼
                    persist KB to disk
                           │
                           ▼
            verify_gap_resolution (deterministic scan)
                           │
                           ▼
                 update gap file (open/closed)
                           │
                           ▼
               re-ask against augmented KB (optional)
```

### The Loop in Detail

1. **Ask** — a question against the KB. The LLM proposes an answer grounded in the facts.
2. **Verify** — the answer is verified against the KB (LLM-backed for textual answers, Clingo-backed for knowledge coherence).
3. **Abduce & Revise** — if verification fails, the `abductive_revisor` generates a hypothesis about the failure and a repair plan, then the candidate is rewritten. This is CEGIS in action [[3]](#ref-orvalho).
4. **Gap report** — the committed answer's rationale lists specific predicate signatures (`profit/2`, `demand_cycle/1`) that the KB is missing. This heuristic form of abductive knowledge induction is related to **Learning from Answer Sets (LAS)** [[4]](#ref-borroto): the system identifies *what is missing to complete a logical theory*.
5. **Learn** — a source text is fed to the LLM, which extracts facts targeting the exact missing predicates. The gap report is a **prioritisation signal**, not a hard constraint — the LLM must still remain grounded in the source text.
6. **Verify coherence** — Clingo checks the merged program (KB ⊕ candidate). If unsatisfiable, the candidate is rejected.
7. **Resolve** — a deterministic scan checks whether the targeted predicates now exist in the KB at the correct arity.

---

## Gap Lifecycle: From Heuristics to Resolution

The gap report is the bridge between the answer flow and the learning flow. It moves through three layers of interpretation:

| Layer | Meaning | Confidence |
|-------|---------|------------|
| Answer gap signal | "This answer would improve if the KB had X" | Heuristic, LLM-based |
| Learn targeting | "Prefer extracting facts about X from this source" | Heuristic but useful |
| Resolution check | "Predicate name X now appears in KB text at the correct arity" | Deterministic, shallow |

### The Hermeneutic Review

Not all gap targets are correct. An LLM might hallucinate a predicate name, or the gap might ask for `profit/2` when the source text actually supports `profit/3` (a richer schema). In v2.1, human review is surfaced via `target_review.drop` — a simple JSON field that lets a human say "this target was wrong, exclude it from resolution."

I call this the **Hermeneutic Loop**. Clingo can verify whether `profit/3` is mathematically consistent with the rest of the KB, but only a human knows whether `profit/3` makes sense for a specific business domain. As Dreyfus argued [[5]](#ref-dreyfus), machines lack the embodied, contextual understanding that humans bring to real-world reasoning. The `target_review` mechanism is where the *Live* in LiveKnowledge happens — the machine handles the syntax, the human provides the meaning.

### Three States of Resolution

The resolution check produces three outcomes for any target predicate:

| State | CLI Output | Meaning |
|-------|------------|---------|
| Exact match found | `recipe/2 — resolved` | Predicate exists at exact arity in KB |
| Exact match missing | `profit/2 — exact target missing` + `related predicate found with different arity: profit/3` | Name found, but at a different arity — schema evolution |
| Dropped by review | `garbage/1 — excluded by review (target_review.drop)` | Human excluded it; not treated as unresolved |

---

## Arity-Drift as Schema Evolution

One of the most interesting emergent properties of the system is **arity-drift detection**. When the resolution check finds `profit/3` but the gap asked for `profit/2`, it reports the mismatch rather than silently accepting or rejecting.

This is not a bug — it's a form of **predicate invention** [[6]](#ref-dumancic). The LLM, presented with a source text that describes profit as a range ("200–500 gold"), naturally extracts the richer `profit/3` schema (item, low, high) instead of the narrower `profit/2` the gap report expected. The system surfaces this as a related predicate, giving the human a clear choice:

- **Accept the drift** — drop the old `profit/2` target and update the question to use `profit/3`
- **Reject the drift** — keep `profit/2` as unresolved, find a source that matches the expected schema

This separation of concerns — the LLM proposes extended schemas, the solver verifies consistency, the human decides on adoption — is exactly what a robust neuro-symbolic pipeline should do.

---

## Non-Monotonicity and Why It Matters

Vector stores are monotonic: adding more documents can only expand what a retrieval system can find. But knowledge is **non-monotonic** — new facts can invalidate old conclusions. If you learn that a previously trusted supplier is unreliable, the inference "buy from them because they're cheap" should retract, not just be buried under competing evidence.

Clingo, as an answer set solver, natively supports non-monotonic reasoning via negation-as-failure and **strong negation**. LiveKnowledge's additive-only v2.1 doesn't yet exploit this fully — contradictions cause rejection rather than retraction — but the solver is designed for it. The KB structure is ready for non-monotonic updates: every integration produces a new immutable snapshot, so reverting a change is always possible by loading the prior snapshot.

The literature on **theory revision** [[7]](#ref-dai) shows that abductive knowledge induction from raw data requires exactly this capability — the ability to revise a logical theory when new data contradicts earlier conclusions. This is a v2.2 goal, and the architecture is designed for it.

---

## How It Maps to the Literature

LiveKnowledge v2.1 sits at the intersection of several research threads:

- **LLM-Modulo Framework** [[1]](#ref-shetye)[[2]](#ref-ishay) — the LLM proposes, the solver verifies. Our `abductive_revisor` is a direct implementation of CEGIS [[3]](#ref-orvalho), using solver failure traces as revision signals.
- **Learning from Answer Sets (LAS)** [[4]](#ref-borroto) — the gap report is a heuristic form of abductive knowledge induction, identifying missing predicates needed to complete a logical theory.
- **Abductive Knowledge Induction** [[7]](#ref-dai) — the combined generate→verify→abduce→revise loop mirrors Meta-Interpretive Learning frameworks, with the LLM acting as a learned meta-interpreter.
- **Predicate Invention** [[6]](#ref-dumancic) — arity-drift detection surfaces cases where the LLM has invented a richer predicate schema than what the gap report expected.
- **Multi-Shot ASP Solving** [[8]](#ref-gebser) — the architecture is designed to adopt Clingo's multi-shot mode for streaming, stateful reasoning in future versions.

---

## Future Steps (LiveKnowledge v2.2+)

v2.1 is an additive, shallow-resolution architecture. Several deferred features are planned:

### Semantic Gap-Completeness
Currently, a gap is "closed" if the predicate name and arity appear anywhere in the augmented KB. Future versions will evaluate **example-instance thresholds**, inspect **minimal models**, and automatically re-answer the original question to prove the gap was actually resolved.

### Non-Monotonic Retractions
Additive-only is a safe start, but real knowledge evolves. Future work will implement **theory revision** via minimal unsatisfiable core analysis, safely deleting or overriding outdated ASP fragments without breaking the rest of the knowledge graph [[7]](#ref-dai).

### Multi-Shot Stateful Reasoning
Right now, every successful integration overwrites the prior snapshot. For real-time streaming agents, the system will adopt Clingo's **multi-shot solving** [[8]](#ref-gebser), maintaining a persistent, stateful reasoning stream that dynamically injects and toggles `#program` scenarios without ever re-grounding the entire KB.

---

## References

[1] S. Shetye. *"An LLM-Modulo Framework for Automated PDDL Domain Model Generation."* Master Thesis, Universität Stuttgart. [PDF](https://elib.uni-stuttgart.de/server/api/core/bitstreams/40946f84-8ab6-4934-8648-7d4b59103b7a/content)

[2] A. Ishay, J. Lee. *"LLMs as ASP Programmers: Self-Correction Enables Task-Agnostic Nonmonotonic Reasoning."* [arXiv:2604.27960](https://arxiv.org/html/2604.27960v1)

[3] P. Orvalho, M. Janota, V. Manquinho. *"Counterexample Guided Program Repair Using Zero-Shot Learning and MaxSAT-based Fault Localization."* AAAI 2025. [PDF](https://web.ist.utl.pt/pmorvalho/papers/aaai25-LLM-CEGIS-Repair.pdf)

[4] M. Borroto, K. Gallagher, A. Ielo, I. Kareem, F. Ricca, A. Russo. *"Question Answering with LLMs and Learning from Answer Sets."* [arXiv:2509.16590](https://arxiv.org/pdf/2509.16590)

[5] H. Dreyfus. *What Computers Still Can't Do: A Critique of Artificial Reason.* MIT Press, 1992.

[6] S. Dumančić, W. Meert, H. Blockeel. *"Theory Reconstruction: A Representation Learning View on Predicate Invention."* [arXiv:1606.08660](https://arxiv.org/pdf/1606.08660)

[7] W-Z. Dai, S.H. Muggleton. *"Abductive Knowledge Induction From Raw Data."* [PDF](https://www.doc.ic.ac.uk/~shm/Papers/abdmetaraw.pdf)

[8] M. Gebser, R. Kaminski, B. Kaufmann, T. Schaub. *"Multi-shot ASP solving with Clingo."* Theory and Practice of Logic Programming, 2019. [arXiv:1705.09811](https://arxiv.org/pdf/1705.09811)

---

*Part 1 of a series. **Part 2 coming soon** — covers semantic gap-completeness, non-monotonic retractions, and multi-shot reasoning. Follow me on [dev.to](https://dev.to/adamrybinski) or visit [adamrybinski.com](https://adamrybinski.com/).*
