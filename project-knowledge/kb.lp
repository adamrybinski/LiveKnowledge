% v2.1 knowledge flow: additive-only candidate integration
% Candidate knowledge must be merged-coherent with existing KB.
% Contradictions cause rejection, not overwrite.

% Roles
role(proposer).
role(verifier).
role(abductive_revisor).
role(clingo).
role(knowledge_store).
role(orchestrator).

% Artifacts
artifact(question).
artifact(knowledge_base).
artifact(candidate_answer).
artifact(candidate_knowledge).
artifact(external_source_text).
artifact(critique).
artifact(verification_report).
artifact(abductive_hypothesis).
artifact(final_answer).
artifact(integration_result).
artifact(loop_status).
artifact(knowledge_gap_report).

% Primitives
primitive(generate).
primitive(generate_knowledge).
primitive(critique).
primitive(verify_answer).
primitive(verify_knowledge).
primitive(abduce_answer).
primitive(abduce_knowledge).
primitive(revise_answer).
primitive(revise_knowledge).
primitive(integrate).
primitive(commit).
primitive(fail).
primitive(answer).

% Verification report statuses
verification_status(verified).
verification_status(rejected).
verification_status(failed).

% Verifier kinds
verifier_kind(llm).
verifier_kind(clingo).

% Gap report statuses
gap_status(open).
gap_status(closed).

% Invariant: committed final answer implies verified candidate answer
:- committed(FinalAnswer), final_answer(FinalAnswer), not verified(CandidateAnswer), candidate_answer(CandidateAnswer).

% Invariant: integrated candidate knowledge implies verified candidate knowledge
:- integrated(CandidateKnowledge), candidate_knowledge(CandidateKnowledge), not verified(CandidateKnowledge).

% Invariant: critique must target an existing artifact
:- critique(Critique), targets(Critique, Target), not artifact(Target).

% Invariant: active loop must be bounded or escaped
:- active(Loop), loop_status(Loop), not bounded(Loop), not escaped(Loop).

% Guard: recoverable failure
is_recoverable_failure(VerificationReport) :- verification_report(VerificationReport), status(VerificationReport, failed), recoverable(VerificationReport).

% Flow: answer_question
flow(answer_question).
flow_input(answer_question, question).
flow_input(answer_question, knowledge_base).
flow_output(answer_question, final_answer).
flow_max_iterations(answer_question, 3).

% Flow: integrate_knowledge
flow(integrate_knowledge).
flow_input(integrate_knowledge, candidate_knowledge).
flow_input(integrate_knowledge, knowledge_base).
flow_output(integrate_knowledge, integration_result).
flow_output(integrate_knowledge, knowledge_base).
flow_max_iterations(integrate_knowledge, 3).

% Flow composition: learn
flow_composition(learn).
learn_input(knowledge_base).
learn_input(external_source_text).
learn_output(knowledge_base).

% Knowledge gap report structure
gap_report_field(question).
gap_report_field(gap_rationale).
gap_report_field(target_predicates).
gap_report_field(target_review).
gap_report_field(status).
gap_report_field(resolution).

% Resolution fields
resolution_field(predicates_checked).
resolution_field(predicates_found).
resolution_field(predicates_still_missing).
resolution_field(predicates_dropped).
resolution_field(related_predicates_found).
resolution_field(all_resolved).

% Target review supports drop list
target_review_action(drop).

% Candidate knowledge constraints
candidate_knowledge_no_show_directives :- candidate_knowledge(C), not has_show_directives(C).

% Integration produces new immutable KB snapshot
immutable_snapshot(KB2) :- integrate(CandidateKnowledge, KB1, KB2), candidate_knowledge(CandidateKnowledge), knowledge_base(KB1), knowledge_base(KB2), KB1 != KB2.

% Redundancy check: skip merge if candidate already fully in KB
redundant_candidate(CandidateKnowledge) :- candidate_knowledge(CandidateKnowledge), knowledge_base(KB), all_facts_in_kb(CandidateKnowledge, KB).

% Merged coherence verification: candidate + KB must be satisfiable
merged_coherent(CandidateKnowledge, KB) :- candidate_knowledge(CandidateKnowledge), knowledge_base(KB), satisfiable(merge(CandidateKnowledge, KB)).

% Reject candidate if merged program is unsatisfiable
rejected_due_to_contradiction(CandidateKnowledge, KB) :- candidate_knowledge(CandidateKnowledge), knowledge_base(KB), not satisfiable(merge(CandidateKnowledge, KB)).

% Provenance: new KB records last integrated candidate
provenance(KB, last_integrated_candidate, CandidateKnowledge) :- integrate(CandidateKnowledge, _, KB), candidate_knowledge(CandidateKnowledge), knowledge_base(KB).

% Gap resolution: predicate found at exact arity
predicate_found_in_kb(PredArity, KB) :- knowledge_base(KB), pred_arity(PredArity), head_atom_in_kb(PredArity, KB).

% Gap resolution: related predicate with different arity
related_predicate_found(PredArity, RelatedArity, KB) :- knowledge_base(KB), pred_arity(PredArity), pred_arity(RelatedArity), same_name_different_arity(PredArity, RelatedArity), head_atom_in_kb(RelatedArity, KB).

% Gap resolution: all resolved when no still-missing predicates (excluding dropped)
all_resolved(GapReport, KB) :- knowledge_gap_report(GapReport), knowledge_base(KB), not has_still_missing(GapReport, KB).

% Gap report closed when all resolved
gap_closed(GapReport) :- knowledge_gap_report(GapReport), all_resolved(GapReport, _).

% auto_revise parameter for integrate_knowledge flow
auto_revise_default(true).
auto_revise_override(false).

% When auto_revise=false, rejected candidate triggers immediate failure
immediate_failure_on_rejection(CandidateKnowledge, KB) :- candidate_knowledge(CandidateKnowledge), knowledge_base(KB), rejected_due_to_contradiction(CandidateKnowledge, KB), auto_revise_override(false).

% Revised artifacts are immutable snapshots with new IDs
immutable_snapshot(NewCandidate) :- revise_answer(_, _, NewCandidate), candidate_answer(NewCandidate).
immutable_snapshot(NewCandidate) :- revise_knowledge(_, _, NewCandidate), candidate_knowledge(NewCandidate).

% generate_knowledge optionally accepts gap_context for targeted extraction
gap_context_from_report(GapReport, GapContext) :- knowledge_gap_report(GapReport), target_predicates(GapReport, TargetPredicates), gap_context(GapContext), contains_predicates(GapContext, TargetPredicates).

% verify_knowledge uses merged program (KB ⊕ candidate)
verify_merged(CandidateKnowledge, KB, VerificationReport) :- candidate_knowledge(CandidateKnowledge), knowledge_base(KB), verification_report(VerificationReport), merged_coherent(CandidateKnowledge, KB), status(VerificationReport, verified).
verify_merged(CandidateKnowledge, KB, VerificationReport) :- candidate_knowledge(CandidateKnowledge), knowledge_base(KB), verification_report(VerificationReport), rejected_due_to_contradiction(CandidateKnowledge, KB), status(VerificationReport, rejected).

% Solo solve grounding failure is expected and does not block merged verification
expected_grounding_failure(CandidateKnowledge) :- candidate_knowledge(CandidateKnowledge), references_kb_predicates(CandidateKnowledge).

% Answer flow is closed-world with respect to supplied KB
closed_world_answer_flow :- flow(answer_question).

% Knowledge flow reasons over KB state transitions
knowledge_flow_state_transitions :- flow(integrate_knowledge).

% Candidate knowledge is additive only (no retractions in v2.1)
additive_only_candidate(CandidateKnowledge) :- candidate_knowledge(CandidateKnowledge), not has_retractions(CandidateKnowledge).

% Contradictions handled by rejecting candidate, never overwriting prior KB
no_overwrite_on_contradiction(CandidateKnowledge, KB) :- candidate_knowledge(CandidateKnowledge), knowledge_base(KB), rejected_due_to_contradiction(CandidateKnowledge, KB), not integrated(CandidateKnowledge).

% Integration failure does not write to disk (CLI-level guarantee)
no_disk_write_on_failure :- integration_failed.

% CLI default: overwrite source KB file in place on successful integration
cli_default_overwrite :- successful_integration.

% --out-kb flag keeps source and writes augmented copy elsewhere
cli_out_kb_flag :- use_out_kb_flag.

% Gap report lifecycle: open -> closed when resolved
gap_report_lifecycle(open, closed) :- gap_closed(_).

% Closed gap report is archived, not deleted
archived_gap_report(GapReport) :- knowledge_gap_report(GapReport), status(GapReport, closed).

% Open gap report can be reused with different source text
reusable_gap_report(GapReport) :- knowledge_gap_report(GapReport), status(GapReport, open).

% v2.1 changes from v0.2
v2_1_change(merged_coherence_verification).
v2_1_change(split_revise_primitives).
v2_1_change(redundancy_check).
v2_1_change(learn_composition).
v2_1_change(gap_signal_in_generate_rationale).
v2_1_change(auto_revise_parameter).
v2_1_change(no_show_directives_in_candidate).

% v2.1 deferred to v2.2+
deferred(retractions_deletions).
deferred(knowledge_delta_artifact).
deferred(pre_post_kb_invariants).
deferred(minimal_unsatisfiable_core).
deferred(standalone_analyse_gaps_primitive).
deferred(semantic_gap_completeness_verification).
deferred(example_instance_thresholds).
deferred(question_reanswer_evaluation).
deferred(minimal_model_inspection).

% main2.py v2.1 — Python implementation facts

% Roles implemented in Python
python_role(orchestrator).
python_role(cli).

% Exception / error types
exception_type(flow_failed).
exception_type(llm_artifact_error).

% Primitive implementations (Python-backed)
python_primitive(generate_answer).
python_primitive(verify_candidate_answer).
python_primitive(abduce_answer).
python_primitive(revise_answer).
python_primitive(generate_candidate_knowledge).
python_primitive(abduce_knowledge).
python_primitive(revise_knowledge).
python_primitive(commit_answer).
python_primitive(verify_candidate_knowledge).
python_primitive(merge_verified_knowledge).

% Gap resolution utility predicates
gap_resolution_field(predicates_checked).
gap_resolution_field(predicates_found).
gap_resolution_field(predicates_still_missing).
gap_resolution_field(predicates_dropped).
gap_resolution_field(related_predicates_found).
gap_resolution_field(all_resolved).

% Artifact types specific to main2.py
artifact(question).
artifact(candidate_answer).
artifact(candidate_knowledge).
artifact(verification_report).
artifact(abductive_hypothesis).
artifact(critique).
artifact(final_answer).
artifact(integration_result).
artifact(knowledge_gap_report).
artifact(gap_resolution).
artifact(artifact_registry).
artifact(solve_result).
artifact(loop_status).

% Registry invariants
registry_no_collision(Registry) :- artifact_registry(Registry), not has_duplicate_ids(Registry).

% Flow implementations
implemented_flow(answer_question).
implemented_flow(integrate_knowledge).
implemented_flow(learn).

% Verification failure classifier
failure_classification(recoverable).
failure_classification(unrecoverable).

% CLI subcommands
cli_command(ask).
cli_command(integrate).
cli_command(learn).

% learn flow composition: generate_knowledge -> integrate_knowledge -> answer_question
learn_flow_step(generate_knowledge).
learn_flow_step(integrate_knowledge).
learn_flow_step(answer_question).

% CLI argument facts from main2.py
cli_flag(ask, kb).
cli_flag(ask, question).
cli_flag(ask, question_file).
cli_flag(ask, question_id).
cli_flag(ask, max_iterations).
cli_flag(ask, gap_report).
cli_flag(integrate, kb).
cli_flag(integrate, program_file).
cli_flag(integrate, notes).
cli_flag(integrate, candidate_id).
cli_flag(integrate, max_iterations).
cli_flag(learn, kb).
cli_flag(learn, unstructured).
cli_flag(learn, text).
cli_flag(learn, fill_gap).
cli_flag(learn, out_kb).
cli_flag(learn, question).
cli_flag(learn, max_iterations).
cli_flag(learn, skip_reask).
cli_flag(learn, question_id).
cli_flag(learn, gap_report).

% Default max iterations constant
max_iterations_default(3).

% GapResolution artifact fields
gap_resolution_field(predicates_checked).
gap_resolution_field(predicates_found).
gap_resolution_field(predicates_still_missing).
gap_resolution_field(predicates_dropped).
gap_resolution_field(related_predicates_found).
gap_resolution_field(all_resolved).

% KnowledgeGapReport is an orchestrator-level artifact, not a flow artifact
orchestrator_artifact(knowledge_gap_report).
not_flow_artifact(knowledge_gap_report).

% Gap report produced by ask --gap-report, consumed by learn --fill-gap
gap_report_producer(ask).
gap_report_consumer(learn).

% learn --fill-gap reads gap report to contextualise extraction
fill_gap_contextualises_extraction :- cli_flag(learn, fill_gap).

% learn --out-kb controls whether source KB is overwritten or copy written elsewhere
learn_out_kb_overwrite :- learn(L), not has_flag(L, out_kb).
learn_out_kb_copy :- learn(L), has_flag(L, out_kb).

% learn --skip-reask skips the re-answer step
learn_skip_reask(L) :- learn(L), has_flag(L, skip_reask).

% GapResolution: related_predicates_found maps requested pred/arity to same-name diff-arity predicates
related_predicate_found(PredArity, RelatedArity, KB) :- gap_resolution(GapRes), related_predicates_found(GapRes, PredArity, RelatedArity), knowledge_base(KB).

% Gap report status transitions
GapReport = open :- knowledge_gap_report(GapReport), not gap_closed(GapReport).

% answer_question flow: iteration counter semantics
% Iteration counts verify attempts; initial generate is outside loop.
% max_iterations=N permits up to N verify+commit/abduce cycles.
answer_question_iteration_count(verify_attempts).
answer_question_generate_outside_loop.

% answer_question: four-way branch on verification status
answer_question_branch(verified, commit).
answer_question_branch(rejected, abduce).
answer_question_branch(failed_recoverable, abduce).
answer_question_branch(failed_unrecoverable, fail).

% answer_question: abduce produces hypothesis and critique
abduce_answer_produces(hypothesis).
abduce_answer_produces(critique).

% answer_question: revise produces new immutable candidate snapshot
revise_answer_produces_new_candidate.

% integrate_knowledge: candidate ID collision avoidance
integrate_reassign_candidate_id(CandidateKnowledge) :- candidate_knowledge(CandidateKnowledge), short_id(CandidateKnowledge).

% integrate_knowledge: new_kb is flow output, not intermediate artifact
flow_output_not_intermediate(new_kb).

% integrate_knowledge: four-way branch mirrors answer_question
integrate_knowledge_branch(verified, integrate).
integrate_knowledge_branch(rejected_no_autorevise, fail).
integrate_knowledge_branch(rejected_autorevise, abduce).
integrate_knowledge_branch(failed_recoverable, abduce).
integrate_knowledge_branch(failed_unrecoverable, fail).

% merge_verified_knowledge: only proceeds if report status is verified
merge_requires_verified_report :- merge_verified_knowledge(_, Report, _, _), verification_report(Report), status(Report, verified).

% merge_verified_knowledge: non-verified report returns failed IntegrationResult and unchanged KB
integration_result_failed_status(CandidateKnowledge, KB) :- candidate_knowledge(CandidateKnowledge), knowledge_base(KB), verification_report(Report), status(Report, rejected), not merge_requires_verified_report.
integration_result_failed_status(CandidateKnowledge, KB) :- candidate_knowledge(CandidateKnowledge), knowledge_base(KB), verification_report(Report), status(Report, failed), not merge_requires_verified_report.

% merge_verified_knowledge: redundancy check — skip merge if candidate fully in KB
redundant_candidate(CandidateKnowledge) :- candidate_knowledge(CandidateKnowledge), knowledge_base(KB), candidate_lines_subset(CandidateKnowledge, KB).

% verify_gap_resolution: deterministic predicate-presence check
gap_resolution_deterministic.
gap_resolution_checks_predicate_presence.
gap_resolution_reports_arity_drift.

% verify_gap_resolution: target_review.drop support
gap_resolution_supports_drop :- gap_resolution(GapRes), predicates_dropped(GapRes, _).

% verify_gap_resolution: dropped predicates excluded from resolution checks
gap_resolution_excludes_dropped.

% verify_gap_resolution: does not modify KnowledgeGapReport
gap_resolution_readonly_report.

% GapResolution: related_predicates_found maps requested pred/arity to same-name diff-arity
gap_resolution_related_arity_drift(PredArity, RelatedArity) :- gap_resolution(GapRes), related_predicates_found(GapRes, PredArity, RelatedArity).

% GapResolution: all_resolved when no predicates_still_missing
gap_resolution_all_resolved(GapRes) :- gap_resolution(GapRes), all_resolved(GapRes), not has_still_missing_resolution(GapRes).

% load_gap_report: loads from JSON file on disk
gap_report_persisted_to_disk.

% write_gap_report: serializes to JSON file on disk
gap_report_serializable_to_json.

% Gap report JSON fields
gap_report_json_field(question).
gap_report_json_field(committed_answer).
gap_report_json_field(gap_rationale).
gap_report_json_field(target_predicates).
gap_report_json_field(status).
gap_report_json_field(target_review).
gap_report_json_field(resolution).

% IntegrationResult fields
integration_result_field(id).
integration_result_field(success).
integration_result_field(message).
integration_result_field(updated_program).

% answer_question flow step ordering
aq_step(t_answer_generate).
aq_step(c_answer_verify).
aq_step(d_answer_verified).
aq_step(t_answer_abduce).
aq_step(t_answer_revise).
aq_step(t_answer_commit).
aq_step(f_answer_failed).

aq_order(t_answer_generate, c_answer_verify).
aq_order(c_answer_verify, d_answer_verified).
aq_order(d_answer_verified, t_answer_commit).
aq_order(d_answer_verified, t_answer_abduce).
aq_order(d_answer_verified, f_answer_failed).
aq_order(t_answer_abduce, t_answer_revise).
aq_order(t_answer_revise, c_answer_verify).

% integrate_knowledge flow step ordering
ik_step(t_knowledge_verify).
ik_step(d_knowledge_verified).
ik_step(t_knowledge_abduce).
ik_step(t_knowledge_revise).
ik_step(t_knowledge_integrate).
ik_step(f_knowledge_failed).

ik_order(t_knowledge_verify, d_knowledge_verified).
ik_order(d_knowledge_verified, t_knowledge_integrate).
ik_order(d_knowledge_verified, t_knowledge_abduce).
ik_order(d_knowledge_verified, f_knowledge_failed).
ik_order(t_knowledge_abduce, t_knowledge_revise).
ik_order(t_knowledge_revise, t_knowledge_verify).

% learn composition step ordering
learn_step(t_learn_generate).
learn_step(t_learn_integrate).
learn_step(t_learn_persist).
learn_step(t_learn_resolve_gap).
learn_step(t_learn_reask).

learn_order(t_learn_generate, t_learn_integrate).
learn_order(t_learn_integrate, t_learn_persist).
learn_order(t_learn_persist, t_learn_resolve_gap).
learn_order(t_learn_resolve_gap, t_learn_reask).

% Failure conditions
aq_failure_reason(iterations_exceeded).
aq_failure_reason(unrecoverable).

% I-QUESTION-TERMINATES: answer_question always reaches commit or failed
aq_terminates(Loop) :- active(Loop), loop_status(Loop), flow(Loop), loop_flow(Loop, answer_question), reached(Loop, t_answer_commit).
aq_terminates(Loop) :- active(Loop), loop_status(Loop), flow(Loop), loop_flow(Loop, answer_question), reached(Loop, f_answer_failed).

% Reviewed candidate fragment extracted from README probe.
% Promotion path: learn probe -> manual review -> integrate CLI.

gap_report_json_field(committed_answer).

integration_result_field(id).
integration_result_field(success).
integration_result_field(message).
integration_result_field(updated_program).

gap_resolution_status(resolved).
gap_resolution_status(exact_target_missing).
gap_resolution_status(related_predicate_found).
gap_resolution_status(excluded_by_review).

gap_report_persistence(gaps_json).
gap_report_persistence_serialized_to_json.

fill_gap_prioritises_extraction.
fill_gap_verifies_resolution.
fill_gap_updates_gap_file.

learn_consume_fill_gap :- cli_command(learn), has_flag(learn, fill_gap).
learn_integrate_skipped_reask :- cli_command(learn), has_flag(learn, skip_reask).

related_predicate_arity_drift(PredArity, RelatedArity) :- gap_resolution(GapRes), related_predicates_found(GapRes, PredArity, RelatedArity).
