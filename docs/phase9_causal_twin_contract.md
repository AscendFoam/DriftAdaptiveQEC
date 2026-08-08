# T9.2.1 因果数字孪生接口与有限总递推合同

## 结论

- verdict：`PASS_T9_2_1_CAUSAL_TWIN_CONTRACT_FROZEN`；本 verdict 仅冻结接口合同，不是物理 backend、codebook、frontend、性能或硬件资格。
- parent：T9.1.1 `c88110375c358794339e72d672e4624871425fe480e5da091ddd1d6595255e18`；T9.1.5 `4623492481ca83cb4dddf571ffd934b940846c9fff87331bfb5b600500ac5941`。
- 五个 namespace：`BACKEND_LATENT`, `DEPLOYABLE_OBSERVED`, `CONTROLLER_MEMORY`, `EVALUATOR_TRUTH`, `PROVENANCE`；`ACTION_WORD` 是独立输出，不构成第六输入 namespace。
- totality：nominal N=`1024`，transition T=`131072`，lossless composition quotient=`196608`，覆盖 raw Cartesian keys=`16777216`。
- action 为 80 bit；`discriminator-out -> action` 的未实测目标为 6 cycles、II=1。raw-IQ/frontend/trigger 均排除在该边界之外，`latency_measured=null`。
- phase-frame 为 2-bit 四态 `(q_byte,p_byte)∈{0,128}²`；当前 two-uint8 RTL adapter 未资格。FSM 为 3-bit mode + 5-bit active-dwell counter 的新 Markov safety state；当前 six-counter FSM adapter 未资格。
- FSM 的 68-state reachability 与 192-state reset/max-3 witness 只属于完整 syntactic T domain（含 reserved/未做 previous-receipt gate 的 event），不是 deployable-causal reachability 结论。
- gates/mutations：`40/40`、`40/40`；Source Data `246` rows。

## 因果与权限边界

- deployable 输入只能来自 `DEPLOYABLE_OBSERVED` 与 `CONTROLLER_MEMORY` 的白名单；`BACKEND_LATENT`、`EVALUATOR_TRUTH`、`PROVENANCE`、future suffix 与 hidden teacher 均结构性拒绝。
- slow path 只能提名预编译、完整、version/CRC/provenance-bound package；禁止逐周期 action、单 entry patch、partial visibility、host callback 与 free-form waveform。
- base lane residual 在序列化的 80-bit action word 内恒为 bit-exact zero；非零 residual 需要独立 amendment 和完整资格链。
- previous action receipt 必须携带 canonical prior K，并由同一个 `F(K)` 重算后逐字段相等；CRC 正确但 recurrence 不可达的 receipt 会 fail closed。
- T9.2.1 只冻结 I/Q 的非空、等长、最大 frame 与 signed-int64 container；sample rate、窗口、Q-format、rounding/saturation 等保持 null，由 T9.2.6 冻结。

## 有限因子化总递推

- composite key 逻辑视图为 `(bank_id, discriminator word, phase-frame, event class, leakage/reset FSM state)`；实现分成 nominal `N(bank, word)` 与 transition `T(phase, event, FSM, nominal-action-index)`。
- 每个 legal cell 均有唯一 action/next-state/reason/error。invalid/OOD/CRC/version/stale/partial/deadline 等故障按冻结优先级闭合到 LKG hold 或 reset。
- composition 与实际 `total_recurrence` 共用唯一 action 组装函数；1,024 个 raw nominal keys 到 12 个 signatures 的 class sizes、representatives 与 hashes 均写入 totality manifest。
- LKG 恢复分为 fast hold/reset 与异步完整 image republish；republish 使用更高版本，禁止版本号倒退。

## Causal graph 原子 ID

- nodes：`LATENT_T`, `PREVIOUS_ACTION_RECEIPT_T`, `RAW_RECORDED_IQ_T`, `MATCHED_FILTER_LLR_T`, `DISCRIMINATOR_T`, `RESET_ACK_T`, `MEMORY_T`, `COMPOSITE_KEY_T`, `ACTION_WORD_T`, `LATENT_T_PLUS_1`, `MEMORY_T_PLUS_1`, `ACTION_RECEIPT_T_PLUS_1`, `EVALUATOR_TRUTH_T`, `PROVENANCE_AUDIT`。
- edges：`CE01_LATENT_TO_IQ`, `CE02_IQ_TO_FILTER`, `CE03_FILTER_TO_DISCRIMINATOR`, `CE04_PREVIOUS_ACTION_TO_MEMORY`, `CE05_DISCRIMINATOR_TO_KEY`, `CE06_RESET_ACK_TO_KEY`, `CE07_MEMORY_TO_KEY`, `CE08_KEY_TO_ACTION`, `CE09_ACTION_TO_NEXT_LATENT`, `CE10_LATENT_TO_NEXT_LATENT`, `CE11_ACTION_TO_NEXT_MEMORY`, `CE12_MEMORY_TO_NEXT_MEMORY`, `CE13_ACTION_TO_NEXT_RECEIPT`, `CE14_LATENT_TO_EVALUATOR`, `CE15_LATENT_TO_PROVENANCE`, `CE16_ACTION_TO_PROVENANCE`。

## Representative probes（非 codebook）

- `P01_IDLE`：neutral idle；probe_only=`true`，codebook_candidate=`false`。
- `P02_Q_POS`：positive q correction；probe_only=`true`，codebook_candidate=`false`。
- `P03_Q_NEG`：negative q correction；probe_only=`true`，codebook_candidate=`false`。
- `P04_P_POS`：positive p correction；probe_only=`true`，codebook_candidate=`false`。
- `P05_P_NEG`：negative p correction；probe_only=`true`，codebook_candidate=`false`。
- `P06_ALTERNATE`：alternating axes；probe_only=`true`，codebook_candidate=`false`。
- `P07_BOUNDARY`：quantizer boundary；probe_only=`true`，codebook_candidate=`false`。
- `P08_PHASE`：phase-frame recurrence；probe_only=`true`，codebook_candidate=`false`。
- `P09_LEAK_RESET`：persistent leakage；probe_only=`true`，codebook_candidate=`false`。
- `P10_RESET_OK`：observed reset success；probe_only=`true`，codebook_candidate=`false`。
- `P11_RESET_FAIL`：observed reset failure；probe_only=`true`，codebook_candidate=`false`。
- `P12_BAD_CRC`：integrity rejection；probe_only=`true`，codebook_candidate=`false`。
- `P13_STALE`：stale/version rejection；probe_only=`true`，codebook_candidate=`false`。
- `P14_OOD`：low-confidence hold then OOD abstention；probe_only=`true`，codebook_candidate=`false`。
- `P15_DEADLINE`：transport/deadline rejection；probe_only=`true`，codebook_candidate=`false`。
- `P16_LKG_RECOVERY`：LKG trusted-bank hold；probe_only=`true`，codebook_candidate=`false`。

## Gate 与 mutation

- `G01_identity_and_protocol_only_scope_are_exact` = `true`；`M01_change_protocol_id` detected=`true`。
- `G02_t9_1_1_parent_is_live_semantic_and_byte_exact` = `true`；`M02_change_t9_1_1_analysis` detected=`true`。
- `G03_t9_1_5_release_pin_is_live_semantic_and_byte_exact` = `true`；`M03_change_t9_1_5_pin_payload` detected=`true`。
- `G04_config_generator_and_physics_implementations_are_live` = `true`；`M04_change_physics_hash` detected=`true`。
- `G05_exactly_five_namespaces_are_frozen_and_action_is_separate` = `true`；`M05_remove_namespace` detected=`true`。
- `G06_namespace_field_schemas_match_the_runtime_contract_exactly` = `true`；`M06_change_namespace_schema` detected=`true`。
- `G07_deployable_input_uses_observed_and_memory_allowlists_only` = `true`；`M07_allow_latent_deployable_input` detected=`true`。
- `G08_latent_evaluator_future_and_provenance_are_recursively_denied` = `true`；`M08_remove_truth_deny_token` detected=`true`。
- `G09_causal_nodes_are_complete_namespace_bound_and_time_indexed` = `true`；`M09_remove_causal_node` detected=`true`。
- `G10_causal_edges_and_intervention_points_are_exact` = `true`；`M10_reverse_causal_edge` detected=`true`。
- `G11_forbidden_future_truth_and_reverse_causal_edges_are_absent` = `true`；`M11_allow_future_truth_edge` detected=`true`。
- `G12_six_cycle_ii1_timing_and_old_or_new_sampling_are_frozen` = `true`；`M12_change_six_cycle_boundary` detected=`true`。
- `G13_composite_key_and_factorized_n_t_domains_are_exact` = `true`；`M13_remove_key_field` detected=`true`。
- `G14_finite_cardinalities_and_state_invariants_are_exact` = `true`；`M14_reduce_nominal_cardinality` detected=`true`。
- `G15_nominal_n_map_is_total_deterministic_and_enumerated` = `true`；`M15_mark_nominal_partial` detected=`true`。
- `G16_transition_t_map_is_total_deterministic_and_enumerated` = `true`；`M16_drop_transition_cell` detected=`true`。
- `G17_totality_fingerprints_and_repeated_enumeration_are_stable` = `true`；`M17_forge_repeat_fingerprint` detected=`true`。
- `G18_factorized_recurrence_is_not_a_partial_or_host_callback_map` = `true`；`M18_allow_host_callback` detected=`true`。
- `G19_fault_priority_order_is_exact_unique_and_fail_closed` = `true`；`M19_swap_fault_priority` detected=`true`。
- `G20_invalid_and_integrity_faults_close_to_lkg_hold_or_reset` = `true`；`M20_allow_undefined_fault_action` detected=`true`。
- `G21_crc_version_stale_partial_ood_deadline_faults_are_covered` = `true`；`M21_remove_crc_fault` detected=`true`。
- `G22_lkg_republish_is_monotonic_and_never_version_decrement` = `true`；`M22_allow_version_decrement` detected=`true`。
- `G23_action_word_is_exactly_80_bits_with_exact_layout` = `true`；`M23_shorten_action_word` detected=`true`。
- `G24_reserved_codes_bounds_and_reason_error_outputs_are_total` = `true`；`M24_allow_reserved_output` detected=`true`。
- `G25_base_lane_residual_is_structurally_bit_exact_zero` = `true`；`M25_enable_nonzero_residual` detected=`true`。
- `G26_slow_path_can_nominate_complete_precompiled_packages_only` = `true`；`M26_give_slow_path_action_authority` detected=`true`。
- `G27_entry_patch_per_cycle_action_and_freeform_waveform_are_denied` = `true`；`M27_allow_entry_patch` detected=`true`。
- `G28_package_commit_is_complete_atomic_versioned_crc_bound` = `true`；`M28_allow_partial_visibility` detected=`true`。
- `G29_hidden_teacher_is_training_only_and_not_deployable` = `true`；`M29_deploy_hidden_teacher` detected=`true`。
- `G30_future_suffix_invariance_and_observed_only_validation_are_frozen` = `true`；`M30_drop_future_invariance` detected=`true`。
- `G31_provenance_is_audit_only_and_cannot_enter_policy_inputs` = `true`；`M31_feed_provenance_to_policy` detected=`true`。
- `G32_exactly_16_representative_probes_are_frozen_before_codebook` = `true`；`M32_remove_probe` detected=`true`。
- `G33_representative_probes_are_noncodebook_nonranking_nonperformance` = `true`；`M33_promote_probe_to_codebook` detected=`true`。
- `G34_probes_cover_nominal_boundary_reset_leakage_and_fault_interventions` = `true`；`M34_remove_fault_probe_coverage` detected=`true`。
- `G35_iq_reset_ack_and_action_conditioning_have_physical_causal_semantics` = `true`；`M35_source_reset_ack_from_truth` detected=`true`。
- `G36_all_physics_performance_codebook_frontend_claim_rank_fields_are_null` = `true`；`M36_fill_claim_null` detected=`true`。
- `G37_factorized_totality_manifest_is_canonical_live_and_exact` = `true`；`M37_change_totality_manifest_path` detected=`true`。
- `G38_source_data_reconstructs_the_full_analysis_losslessly` = `true`；`M38_change_source_row_count` detected=`true`。
- `G39_markdown_is_canonical_exact_and_contains_all_atomic_ids` = `true`；`M39_change_markdown_path` detected=`true`。
- `G40_one_substantive_mutation_per_gate_is_replayed_and_rejected` = `true`；`M40_forge_mutation_count` detected=`true`。

## Typed-null 结果边界

- `physics`：`physics.backend_a_qualified`, `physics.backend_b_qualified`, `physics.dual_backend_agreement`, `physics.physical_lifetime`, `physics.qpu_measurement`（全部 `null`）。
- `performance`：`performance.round_ler`, `performance.six_state_lifetime`, `performance.worst_window_ler`, `performance.latency_measured`, `performance.power_measured`（全部 `null`）。
- `codebook`：`codebook.codebook_id`, `codebook.codebook_sha256`, `codebook.optimized_action_map`, `codebook.quantization_result`, `codebook.coverage_result`（全部 `null`）。
- `frontend`：`frontend.frontend_profile`, `frontend.raw_iq_qualification`, `frontend.recorded_iq_qualification`, `frontend.rtl_frontend_result`, `frontend.board_frontend_result`（全部 `null`）。
- `claim`：`claim.registered_best`, `claim.external_sota`, `claim.official_puviani_exact`, `claim.puviani_nmf_surpass`, `claim.physical_break_even`, `claim.hil_integrated`, `claim.hil_external_speed`（全部 `null`）。
- `rank`：`rank.registered_rank`, `rank.external_rank`, `rank.hardware_speed_rank`, `rank.global_rank`（全部 `null`）。

## 后续消费

- T9.2.2/T9.2.3 实现两个独立 physics backend；T9.2.4 才可做双后端资格对拍；T9.3.3/T9.3.4 才可生成并枚举最终 trusted codebook。
- 下游必须从 canonical release pin 接收 expected analysis SHA，先 live verify，再消费；报告自选路径、seal-only acceptance 和跨 lane promotion 均被禁止。


## Causal node/edge atomic IDs

- node `LATENT_T`: namespace=`BACKEND_LATENT`, time=`t`.
- node `PREVIOUS_ACTION_RECEIPT_T`: namespace=`DEPLOYABLE_OBSERVED`, time=`t`.
- node `RAW_RECORDED_IQ_T`: namespace=`DEPLOYABLE_OBSERVED`, time=`t`.
- node `MATCHED_FILTER_LLR_T`: namespace=`DEPLOYABLE_OBSERVED`, time=`t`.
- node `DISCRIMINATOR_T`: namespace=`DEPLOYABLE_OBSERVED`, time=`t`.
- node `RESET_ACK_T`: namespace=`DEPLOYABLE_OBSERVED`, time=`t`.
- node `MEMORY_T`: namespace=`CONTROLLER_MEMORY`, time=`t`.
- node `COMPOSITE_KEY_T`: namespace=`CONTROLLER_MEMORY`, time=`t`.
- node `ACTION_WORD_T`: namespace=`ACTION_WORD`, time=`t`.
- node `LATENT_T_PLUS_1`: namespace=`BACKEND_LATENT`, time=`t+1`.
- node `MEMORY_T_PLUS_1`: namespace=`CONTROLLER_MEMORY`, time=`t+1`.
- node `ACTION_RECEIPT_T_PLUS_1`: namespace=`DEPLOYABLE_OBSERVED`, time=`t+1`.
- node `EVALUATOR_TRUTH_T`: namespace=`EVALUATOR_TRUTH`, time=`t`.
- node `PROVENANCE_AUDIT`: namespace=`PROVENANCE`, time=`audit-only`.
- edge `CE01_LATENT_TO_IQ`: `LATENT_T` -> `RAW_RECORDED_IQ_T`.
- edge `CE02_IQ_TO_FILTER`: `RAW_RECORDED_IQ_T` -> `MATCHED_FILTER_LLR_T`.
- edge `CE03_FILTER_TO_DISCRIMINATOR`: `MATCHED_FILTER_LLR_T` -> `DISCRIMINATOR_T`.
- edge `CE04_PREVIOUS_ACTION_TO_MEMORY`: `PREVIOUS_ACTION_RECEIPT_T` -> `MEMORY_T`.
- edge `CE05_DISCRIMINATOR_TO_KEY`: `DISCRIMINATOR_T` -> `COMPOSITE_KEY_T`.
- edge `CE06_RESET_ACK_TO_KEY`: `RESET_ACK_T` -> `COMPOSITE_KEY_T`.
- edge `CE07_MEMORY_TO_KEY`: `MEMORY_T` -> `COMPOSITE_KEY_T`.
- edge `CE08_KEY_TO_ACTION`: `COMPOSITE_KEY_T` -> `ACTION_WORD_T`.
- edge `CE09_ACTION_TO_NEXT_LATENT`: `ACTION_WORD_T` -> `LATENT_T_PLUS_1`.
- edge `CE10_LATENT_TO_NEXT_LATENT`: `LATENT_T` -> `LATENT_T_PLUS_1`.
- edge `CE11_ACTION_TO_NEXT_MEMORY`: `ACTION_WORD_T` -> `MEMORY_T_PLUS_1`.
- edge `CE12_MEMORY_TO_NEXT_MEMORY`: `MEMORY_T` -> `MEMORY_T_PLUS_1`.
- edge `CE13_ACTION_TO_NEXT_RECEIPT`: `ACTION_WORD_T` -> `ACTION_RECEIPT_T_PLUS_1`.
- edge `CE14_LATENT_TO_EVALUATOR`: `LATENT_T` -> `EVALUATOR_TRUTH_T`.
- edge `CE15_LATENT_TO_PROVENANCE`: `LATENT_T` -> `PROVENANCE_AUDIT`.
- edge `CE16_ACTION_TO_PROVENANCE`: `ACTION_WORD_T` -> `PROVENANCE_AUDIT`.
