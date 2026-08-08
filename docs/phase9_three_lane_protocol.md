# Phase 9 三条独立任务签名与结果门

- Task：`T9.1.1`
- 协议 verdict：`PASS_PHASE9_THREE_INDEPENDENT_LANE_PROTOCOL_FROZEN`
- seal：`SEALED_PRE_OUTCOME`；本文件不是性能实验结果。
- 当前 LER、lifetime、HIL speed 的 `result_verdict` 均为 `null`；协议通过不得写成性能 GO。

## 三条互不补门的 lane

| Lane | 当前状态 | 主指标 / 主边界 | Signature SHA-256 |
| --- | --- | --- | --- |
| `ROUND_LER_SINGLE_MODE` | `NOT_EVALUATED_NULL` / `null` | p_L, p_X, p_Y, p_Z, logical_PTM | `984a832fec0c9f423b594a057dc05af98c9876ec19fb026cd1d1bb972b2b0ac2` |
| `SIX_STATE_LOGICAL_LIFETIME` | `NOT_EVALUATED_NULL` / `null` | T_X, T_Y, T_Z, T_ch, area, e_folding, survival, fit_diagnostics | `29003e03986be1cbc6e90ab216133e79e829dc2f7d921f9743846e72d36665ac` |
| `RAW_IQ_DIGITAL_HIL` | `MISSING_BOARD` / `null` | p50, p95, p99, max, WCET, II, deadline_miss, mismatch, undefined_action, silent_overflow, resource, power | `4965da60999451c61a4d5d6c4a0b67d722bb848ea012a4536eb9a29ba77a8756` |

单轮 LER 不能由 lifetime 补门；lifetime 不能由单轮 LER 推导；HIL speed 的主边界固定为 `raw_iq_source_to_trigger`，不能用 6-cycle core、CXXRTL、P&R 或 host timing 补值。三条 lane 禁止 weighted score、winner count 或全局榜单。

## 24-field task signature

- `code_family`
- `state_family`
- `physical_backend_contract`
- `decision_target`
- `observation_schema`
- `history_horizon`
- `action_set`
- `action_cost_contract`
- `noise_drift_family`
- `observability`
- `online_privilege`
- `cycle_time_contract`
- `primary_estimand`
- `denominator_contract`
- `postselection_policy`
- `baseline_eligibility_contract`
- `compute_budget_contract`
- `wall_clock_budget_contract`
- `precision_contract`
- `split_contract`
- `statistical_unit`
- `multiplicity_contract`
- `missingness_contract`
- `evidence_grade_contract`

### `ROUND_LER_SINGLE_MODE`

- `code_family`：`single_mode_square_approximate_gkp`
- `state_family`：`balanced_plus_minus_X_Y_Z_logical_eigenstates`
- `physical_backend_contract`：`same_registered_action_conditioned_twin_with_independent_backend_reevaluation`
- `decision_target`：`per_round_logical_Pauli_recovery_outcome`
- `observation_schema`：`frozen_same_representation_stratum_of_raw_or_recorded_IQ_LLR_confidence_timestamp_reset_ack_and_past_actions`
- `history_horizon`：`causal_prefix_o_<=t_a_<t_no_future_suffix`
- `action_set`：`trusted_codebook_id_bounded_residual_phase_frame_reset_leakage_FSM`
- `action_cost_contract`：`all_reset_control_fallback_and_idle_costs_retained`
- `noise_drift_family`：`preregistered_stationary_smooth_step_telegraph_burst_compound_action_conditioned_families`
- `observability`：`observed_only_hidden_truth_never_deployable`
- `online_privilege`：`same_update_cadence_state_memory_and_deadline`
- `cycle_time_contract`：`one_complete_sBs_round_including_measurement_readout_reset_control_and_idle`
- `primary_estimand`：`per_round_p_L_p_X_p_Y_p_Z_and_logical_PTM`
- `denominator_contract`：`all_registered_state_trajectory_round_opportunities_with_timeout_fallback_reset_leakage_and_control_retained`
- `postselection_policy`：`PROHIBITED_ZERO_REJECTION_PRIMARY`
- `baseline_eligibility_contract`：`MATCHED_DEPLOYABLE_RANKED_exact_signature_and_pre_pilot_registry`
- `compute_budget_contract`：`matched_tokens_samples_memory_MAC_FLOP_CPU_GPU_and_restart_ledger`
- `wall_clock_budget_contract`：`matched_inference_update_deadline_and_timeout_accounting`
- `precision_contract`：`same_frozen_IQ_LLR_parameter_and_action_precision_stratum`
- `split_contract`：`train_calibration_single_pass_pilot_untouched_formal`
- `statistical_unit`：`paired_independent_device_scenario_trajectory_cluster_not_round`
- `multiplicity_contract`：`paired_cluster_maxT_95pct_all_baselines_endpoints_families`
- `missingness_contract`：`no_silent_drop_mandatory_failure_closes_SOTA`
- `evidence_grade_contract`：`UNTOUCHED_FORMAL_SIMULATION_plus_INDEPENDENT_BACKEND_REEVALUATION`

Result gate：

- `min_relative_improvement_point_each_baseline` >= `0.15`
- `min_simultaneous_relative_lcb_each_baseline` >= `0.1`
- `min_simultaneous_absolute_lcb_each_baseline` > `0.0`
- `stationary_degradation_ucb` <= `0.02`
- `max_ood_family_degradation_ucb` <= `0.05`
- `calibration_worst_window_improvement_lcb` > `0.0`
- `telegraph_cvar_improvement_lcb` > `0.0`

Claim ladder：

- `LER-C0-PROTOCOL`：`SUPPORTED_PROTOCOL_ONLY`；required=PROTOCOL_ONLY；allowed=lane-local wording for LER-C0-PROTOCOL at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `LER-C1-DEVELOPMENT`：`CLOSED`；required=PROJECT_NATIVE_DEVELOPMENT_SIMULATION；allowed=lane-local wording for LER-C1-DEVELOPMENT at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `LER-C2-TASK-LOCAL-FORMAL`：`CLOSED`；required=UNTOUCHED_FORMAL_SIMULATION,INDEPENDENT_BACKEND_REEVALUATION；allowed=lane-local wording for LER-C2-TASK-LOCAL-FORMAL at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `LER-C3-BEST-REGISTERED`：`CLOSED`；required=UNTOUCHED_FORMAL_SIMULATION,INDEPENDENT_BACKEND_REEVALUATION；allowed=lane-local wording for LER-C3-BEST-REGISTERED at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `LER-C4-SOTA`：`CLOSED`；required=UNTOUCHED_FORMAL_SIMULATION,INDEPENDENT_BACKEND_REEVALUATION；allowed=SOTA on the frozen single-mode round-LER task after GO_LER_SOTA；forbidden=lifetime, tail-only, safety, or RTL evidence as an LER substitute。

### `SIX_STATE_LOGICAL_LIFETIME`

- `code_family`：`single_mode_square_approximate_gkp`
- `state_family`：`balanced_plus_minus_X_Y_Z_logical_eigenstates_long_sequence`
- `physical_backend_contract`：`same_registered_action_conditioned_twin_with_independent_backend_reevaluation`
- `decision_target`：`six_state_logical_channel_lifetime_over_at_least_1e4_cycles`
- `observation_schema`：`frozen_same_representation_stratum_of_raw_or_recorded_IQ_LLR_confidence_timestamp_reset_ack_and_past_actions`
- `history_horizon`：`causal_prefix_through_each_cycle_no_smoothing_or_future`
- `action_set`：`trusted_codebook_id_bounded_residual_phase_frame_reset_leakage_FSM`
- `action_cost_contract`：`all_reset_control_fallback_and_idle_costs_retained`
- `noise_drift_family`：`preregistered_stationary_smooth_step_telegraph_burst_compound_action_conditioned_families`
- `observability`：`observed_only_hidden_truth_never_deployable`
- `online_privilege`：`same_update_cadence_state_memory_and_deadline`
- `cycle_time_contract`：`physical_cycle_and_simulated_cycle_both_recorded_no_unmapped_physical_time_claim`
- `primary_estimand`：`T_X_T_Y_T_Z_T_ch_area_efolding_survival_and_fit_diagnostics`
- `denominator_contract`：`all_six_state_trajectories_from_t0_with_registered_censoring_and_no_accepted_only_subset`
- `postselection_policy`：`PROHIBITED_ZERO_REJECTION_PRIMARY`
- `baseline_eligibility_contract`：`same_task_same_cycle_action_reset_control_and_registered_matched_deployable`
- `compute_budget_contract`：`matched_controller_budget_and_complete_simulation_cost_ledger`
- `wall_clock_budget_contract`：`physical_protocol_time_separate_from_simulator_runtime_both_reported`
- `precision_contract`：`same_frozen_IQ_LLR_parameter_and_action_precision_stratum`
- `split_contract`：`same_untouched_formal_trajectories_or_preregistered_disjoint_lifetime_formal_split`
- `statistical_unit`：`paired_independent_device_scenario_trajectory_cluster_not_cycle`
- `multiplicity_contract`：`paired_cluster_maxT_95pct_all_six_states_lifetimes_baselines`
- `missingness_contract`：`registered_censoring_is_observation_other_failures_retained_and_close_gate`
- `evidence_grade_contract`：`UNTOUCHED_FORMAL_SIMULATION_plus_INDEPENDENT_BACKEND_REEVALUATION_simulated_only`

Result gate：

- `minimum_sequence_cycles` >= `10000`
- `min_six_state_relative_gain_point` >= `0.15`
- `min_simultaneous_lifetime_gain_lcb` > `0.0`
- `reset_control_burden_margin_ucb` <= `0.0`

Claim ladder：

- `LIFE-C0-PROTOCOL`：`SUPPORTED_PROTOCOL_ONLY`；required=PROTOCOL_ONLY；allowed=lane-local wording for LIFE-C0-PROTOCOL at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `LIFE-C1-DEVELOPMENT`：`CLOSED`；required=PROJECT_NATIVE_DEVELOPMENT_SIMULATION；allowed=lane-local wording for LIFE-C1-DEVELOPMENT at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `LIFE-C2-DUAL-BACKEND-SIMULATED`：`CLOSED`；required=UNTOUCHED_FORMAL_SIMULATION,INDEPENDENT_BACKEND_REEVALUATION；allowed=lane-local wording for LIFE-C2-DUAL-BACKEND-SIMULATED at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `LIFE-C3-BEST-REGISTERED`：`CLOSED`；required=PAPER_CONSTRAINED_REIMPLEMENTATION,UNTOUCHED_FORMAL_SIMULATION,INDEPENDENT_BACKEND_REEVALUATION；allowed=lane-local wording for LIFE-C3-BEST-REGISTERED at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `LIFE-C4-PUVIANI-SURPASS`：`BLOCKED_NULL`；required=OFFICIAL_EXACT_REPRODUCTION；allowed=surpasses official Puviani NMF only after official-exact same-signature qualification；forbidden=paper-constrained or project-native evidence described as official Puviani surpass。
- `LIFE-C5-PHYSICAL-BREAK-EVEN`：`BLOCKED_NULL`；required=QPU_MEASURED；allowed=physical break-even measured on a protocol-matched QPU；forbidden=simulator lifetime, matched-idle crossover, or accepted-only curve as physical break-even。

### `RAW_IQ_DIGITAL_HIL`

- `code_family`：`single_mode_square_gkp_integrated_digital_control_chain`
- `state_family`：`registered_raw_and_recorded_IQ_transaction_traces`
- `physical_backend_contract`：`real_board_same_source_bitstream_recorded_and_raw_IQ_HIL`
- `decision_target`：`trusted_action_and_trigger_from_raw_IQ_source`
- `observation_schema`：`raw_or_recorded_complex_IQ_through_discriminator_with_timestamp_queue_and_backpressure`
- `history_horizon`：`causal_stream_and_bounded_registered_controller_state`
- `action_set`：`same_trusted_codebook_residual_frame_reset_leakage_trigger_actions`
- `action_cost_contract`：`transport_CDC_queue_discriminator_action_trigger_and_backpressure_included`
- `noise_drift_family`：`registered_trace_families_and_fault_injection_not_simulator_gain`
- `observability`：`wire_observed_only_no_hidden_truth_in_action_path`
- `online_privilege`：`same_clock_precision_queue_deadline_and_resource_boundary`
- `cycle_time_contract`：`board_clock_plus_ADC_sample_window_and_all_four_timestamped_boundaries`
- `primary_estimand`：`raw_IQ_source_to_trigger_p50_p95_p99_max_WCET_II_deadline_miss`
- `denominator_contract`：`all_registered_source_transactions_including_queue_backpressure_after_frozen_warmup`
- `postselection_policy`：`PROHIBITED_NO_COMPLETED_ONLY_LATENCY_DENOMINATOR`
- `baseline_eligibility_contract`：`same_task_code_observation_action_precision_boundary_statistic_and_measured_grade`
- `compute_budget_contract`：`same_target_resource_clock_precision_and_queue_envelope`
- `wall_clock_budget_contract`：`measured_timestamp_boundary_with_frozen_deadline_and_no_host_only_substitution`
- `precision_contract`：`bit_exact_discriminator_fixed_point_action_and_trigger_formats`
- `split_contract`：`frozen_calibration_trace_then_disjoint_validation_and_formal_HIL_traces`
- `statistical_unit`：`implementation_seed_trace_seed_run_cluster_not_transaction`
- `multiplicity_contract`：`paired_build_run_cluster_simultaneous_95pct_primary_boundary_contrasts`
- `missingness_contract`：`all_dropped_timeout_corrupt_and_deadline_transactions_retained`
- `evidence_grade_contract`：`RECORDED_IQ_HIL_MEASURED_plus_RAW_IQ_HIL_MEASURED`

Result gate：

- `implementation_seed_count` >= `3`
- `transaction_count` >= `1000000`
- `mismatch_count` == `0`
- `undefined_action_count` == `0`
- `silent_overflow_count` == `0`
- `deadline_miss_count` == `0`
- `initiation_interval_cycles` == `1`
- `wcet_minus_deadline_ns` <= `0.0`
- `primary_speed_contrast_simultaneous_lcb_ns` > `0.0`

Claim ladder：

- `HIL-C0-PROTOCOL`：`SUPPORTED_PROTOCOL_ONLY`；required=PROTOCOL_ONLY；allowed=lane-local wording for HIL-C0-PROTOCOL at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `HIL-C1-FIXED-POINT-CXXRTL`：`PARENT_RESTRICTED_ONLY`；required=FIXED_POINT_REFERENCE,CXXRTL_PREBOARD,RTL_PROPERTY_PROOF；allowed=lane-local wording for HIL-C1-FIXED-POINT-CXXRTL at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `HIL-C2-POST-ROUTE`：`PARENT_RESTRICTED_ONLY`；required=POST_ROUTE_ESTIMATE；allowed=lane-local wording for HIL-C2-POST-ROUTE at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `HIL-C3-MEASURED-INTEGRATED-CHAIN`：`BLOCKED_NULL`；required=RECORDED_IQ_HIL_MEASURED,RAW_IQ_HIL_MEASURED；allowed=lane-local wording for HIL-C3-MEASURED-INTEGRATED-CHAIN at its stated evidence grade；forbidden=cross-lane or unqualified evidence promotion。
- `HIL-C4-SAME-TASK-SPEED`：`BLOCKED_NULL`；required=RAW_IQ_HIL_MEASURED；allowed=faster raw-IQ-source-to-trigger HIL under a same-task measured comparison；forbidden=core cycles, CXXRTL, P&R, host timing, or cross-code nanoseconds as measured speed。

## Evidence grade（集合资格，不是全局线性等级）

- `PROTOCOL_ONLY`：protocol_frozen
- `LITERATURE_ONLY`：external_context_only
- `OFFICIAL_SOURCE_PINNED`：source_intake_only
- `OFFICIAL_EXACT_REPRODUCTION`：official_exact_reproduction
- `PAPER_CONSTRAINED_REIMPLEMENTATION`：registered_reimplementation_only
- `PROJECT_NATIVE_DEVELOPMENT_SIMULATION`：development_diagnostic_only
- `UNTOUCHED_FORMAL_SIMULATION`：task_local_formal_simulation
- `INDEPENDENT_BACKEND_REEVALUATION`：task_local_cross_backend_formal
- `FIXED_POINT_REFERENCE`：integer_reference_only
- `CXXRTL_PREBOARD`：preboard_replay_only
- `RTL_PROPERTY_PROOF`：rtl_property_only
- `POST_ROUTE_ESTIMATE`：tool_device_specific_estimate_only
- `RECORDED_IQ_HIL_MEASURED`：recorded_iq_board_hil
- `RAW_IQ_HIL_MEASURED`：raw_iq_source_to_trigger_board_hil
- `QPU_MEASURED`：physical_qec_and_break_even

## Baseline class

- `MATCHED_DEPLOYABLE_RANKED`：ranked=`true`，may_support_sota=`true`
- `CAPACITY_CEILING_NONRANKING`：ranked=`false`，may_support_sota=`false`
- `PRIVILEGED_UPPER_BOUND_NONRANKING`：ranked=`false`，may_support_sota=`false`
- `PROTOCOL_ANCHOR_NONRANKING`：ranked=`false`，may_support_sota=`false`

## 外部/null claim slot

- `OFFICIAL_PUVIANI_EXACT`：state=`MISSING_EXTERNAL_ASSET`，value=`null`
- `PUVIANI_NMF_SURPASS`：state=`MISSING_EXTERNAL_ASSET`，value=`null`
- `PHYSICAL_BREAK_EVEN`：state=`NOT_EVALUATED_NULL`，value=`null`
- `RAW_IQ_HIL_SPEED`：state=`MISSING_BOARD`，value=`null`

## 禁止证据迁移

- `FT-LIFETIME-TO-LER`：`SIX_STATE_LOGICAL_LIFETIME` → `GO_LER_SOTA`，拒绝码 `CROSS_LANE_SUBSTITUTION`
- `FT-LER-TO-LIFETIME`：`ROUND_LER_SINGLE_MODE` → `GO_LIFETIME`，拒绝码 `CROSS_LANE_SUBSTITUTION`
- `FT-CORE-TO-RAW-IQ`：`decoder_core` → `raw_iq_source_to_trigger`，拒绝码 `TIMING_BOUNDARY_SUBSTITUTION`
- `FT-PREBOARD-TO-MEASURED`：`CXXRTL_PREBOARD_or_POST_ROUTE_ESTIMATE` → `RAW_IQ_HIL_MEASURED`，拒绝码 `EVIDENCE_GRADE_PROMOTION`
- `FT-SIM-TO-PHYSICAL-BREAK-EVEN`：`simulation_gain` → `PHYSICAL_BREAK_EVEN`，拒绝码 `PHYSICAL_EVIDENCE_MISSING`
- `FT-PAPER-CONSTRAINED-TO-OFFICIAL`：`PAPER_CONSTRAINED_REIMPLEMENTATION` → `OFFICIAL_EXACT_REPRODUCTION`，拒绝码 `PROVENANCE_NAMESPACE_SUBSTITUTION`
- `FT-HIDDEN-TO-DEPLOYABLE`：`hidden_teacher_or_oracle` → `MATCHED_DEPLOYABLE_RANKED`，拒绝码 `PRIVILEGE_SUBSTITUTION`
- `FT-CEILING-TO-RANKED`：`CAPACITY_CEILING_NONRANKING` → `MATCHED_DEPLOYABLE_RANKED`，拒绝码 `BUDGET_SUBSTITUTION`
- `FT-CROSS-LANE-SCORE`：`three_lane_outcomes` → `weighted_score_or_win_count`，拒绝码 `GLOBAL_SCORE_PROHIBITED`
- `FT-POSTSELECTED-TO-FULL`：`accepted_only_or_postselected` → `full_denominator`，拒绝码 `DENOMINATOR_SUBSTITUTION`
- `FT-SAFETY-TO-PERFORMANCE`：`deterministic_atomic_fail_closed` → `LER_or_lifetime_SOTA`，拒绝码 `SYSTEM_SAFETY_NOT_PERFORMANCE`
- `FT-CROSS-CODE-LATENCY`：`different_code_family_or_problem` → `same_task_speed_rank`，拒绝码 `TASK_SIGNATURE_MISMATCH`
- `FT-MISSING-AS-ZERO`：`MISSING_BOARD_or_MISSING_EXTERNAL_ASSET` → `numeric_zero`，拒绝码 `NULL_SEMANTICS_VIOLATION`

## 二值结果门的合成逻辑夹具

- `ROUND_LER_SINGLE_MODE:unopened`：state=`NOT_EVALUATED_NULL`，verdict=`None`；仅验证 gate logic，不是实验结果。
- `ROUND_LER_SINGLE_MODE:incomplete`：state=`INCOMPLETE`，verdict=`None`；仅验证 gate logic，不是实验结果。
- `ROUND_LER_SINGLE_MODE:complete_no_go`：state=`COMPLETE`，verdict=`NO_GO_LER_SOTA`；仅验证 gate logic，不是实验结果。
- `ROUND_LER_SINGLE_MODE:complete_go`：state=`COMPLETE`，verdict=`GO_LER_SOTA`；仅验证 gate logic，不是实验结果。
- `SIX_STATE_LOGICAL_LIFETIME:unopened`：state=`NOT_EVALUATED_NULL`，verdict=`None`；仅验证 gate logic，不是实验结果。
- `SIX_STATE_LOGICAL_LIFETIME:incomplete`：state=`INCOMPLETE`，verdict=`None`；仅验证 gate logic，不是实验结果。
- `SIX_STATE_LOGICAL_LIFETIME:complete_no_go`：state=`COMPLETE`，verdict=`NO_GO_LIFETIME`；仅验证 gate logic，不是实验结果。
- `SIX_STATE_LOGICAL_LIFETIME:complete_go`：state=`COMPLETE`，verdict=`GO_LIFETIME`；仅验证 gate logic，不是实验结果。
- `RAW_IQ_DIGITAL_HIL:unopened`：state=`NOT_EVALUATED_NULL`，verdict=`None`；仅验证 gate logic，不是实验结果。
- `RAW_IQ_DIGITAL_HIL:incomplete`：state=`INCOMPLETE`，verdict=`None`；仅验证 gate logic，不是实验结果。
- `RAW_IQ_DIGITAL_HIL:complete_no_go`：state=`COMPLETE`，verdict=`NO_GO_HIL_SPEED`；仅验证 gate logic，不是实验结果。
- `RAW_IQ_DIGITAL_HIL:complete_go`：state=`COMPLETE`，verdict=`GO_HIL_SPEED`；仅验证 gate logic，不是实验结果。
- `RAW_IQ_DIGITAL_HIL:engineering_only_no_comparator`：state=`INCOMPLETE`，verdict=`None`；仅验证 gate logic，不是实验结果。

## Fail-closed gates

- `G01_identity_and_preoutcome_seal`
- `G02_exactly_three_independent_namespaces`
- `G03_signature_schema_has_24_frozen_fields`
- `G04_signatures_are_complete_nonempty_and_distinct`
- `G05_ler_code_state_metrics_are_six_state_single_mode`
- `G06_ler_observation_action_and_denominator_are_causal_full`
- `G07_lifetime_metrics_horizon_and_six_state_aggregation_are_complete`
- `G08_lifetime_inherits_ler_physics_observation_action_and_cost`
- `G09_algorithm_lanes_prohibit_postselection_and_accepted_only_denominators`
- `G10_hil_has_four_boundaries_and_raw_iq_primary`
- `G11_hil_statistics_denominator_and_hardware_cost_are_complete`
- `G12_baseline_classes_keep_only_matched_deployable_ranked`
- `G13_matched_baseline_predicate_is_exact_and_fail_closed`
- `G14_split_is_single_pass_pilot_then_untouched_formal`
- `G15_observed_only_contract_rejects_future_truth_and_scenario_privilege`
- `G16_compute_precision_wallclock_and_deadline_fields_are_nonempty`
- `G17_multiplicity_is_cluster_level_simultaneous_and_closed_family`
- `G18_missingness_retains_failures_and_never_imputes_null_as_zero`
- `G19_evidence_grades_are_scope_sets_not_a_global_rank`
- `G20_puviani_official_and_surpass_slots_remain_local_nulls`
- `G21_physical_break_even_and_raw_iq_speed_remain_null_without_grade`
- `G22_ler_gate_freezes_each_baseline_and_tail_safety_thresholds`
- `G23_lifetime_gate_freezes_six_state_gain_cost_and_horizon`
- `G24_hil_gate_requires_board_chain_three_seeds_million_transactions_and_comparator`
- `G25_future_evaluator_fixtures_cover_go_no_go_incomplete_and_unopened`
- `G26_claim_ladders_have_wording_grades_and_revocation`
- `G27_forbidden_transfer_registry_is_complete`
- `G28_global_score_winner_count_and_cross_lane_rescue_are_prohibited`
- `G29_source_contracts_are_semantically_or_exactly_live`
- `G30_current_performance_results_are_null_not_fake_go_or_no_go`
- `G31_lifetime_does_not_promote_puviani_or_physical_claims`
- `G32_preboard_rtl_does_not_promote_measured_hil_or_speed`
- `G33_independent_backends_each_pass_without_averaging`
- `G34_result_state_machine_is_null_incomplete_then_binary_complete`
- `G35_source_data_and_human_contract_are_lossless_and_live`
- `G36_one_substantive_mutation_per_gate_fails_closed`

## 解释边界

Puviani official asset 缺失只让 `OFFICIAL_PUVIANI_EXACT` / `PUVIANI_NMF_SURPASS` 保持 null，不阻塞 paper-constrained、数字孪生、codebook、model tournament 或 project-native formal。模拟 lifetime 无论多好都不能填 `PHYSICAL_BREAK_EVEN`；pre-board RTL 无论多确定都不能填 `RAW_IQ_HIL_SPEED`。
