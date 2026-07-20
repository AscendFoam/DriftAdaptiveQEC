# T6.17.3 learned model eligibility 与只读 replay

- verdict：`PASS_READONLY_LEARNED_ELIGIBILITY_NO_SAME_TASK_CHECKPOINT`
- candidate families：16；same-task eligible=0；diagnostic replay=1
- legacy CNN replay：206 samples，MSE=2.41445285e-06，parent max-abs diff=0
- gates / mutations：16/16 / 16/16；Source Data=434 rows

## Eligibility 结论

| candidate | category | native lane | mismatched signature fields | replay |
| --- | --- | --- | --- | --- |
| `legacy_residual_tinycnn` | `legacy_cnn_parameter_estimator` | `single_mode_decoder` | decision_target, input_semantics, history_horizon, output_action, noise_model, time_basis, compute_budget, precision, evidence_level | `DIAGNOSTIC_REPLAY_EXACT_NOT_RANKED` |
| `legacy_static_theta_tinycnn` | `legacy_cnn_noise_parameter_estimator` | `single_mode_decoder` | decision_target, input_semantics, history_horizon, output_action, noise_model, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `t411_causal_tcn` | `causal_adaptive_nn_regime_estimator` | `single_mode_decoder` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `t411_small_gru` | `causal_adaptive_nn_regime_estimator` | `single_mode_decoder` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `t327_latest_outcome_fnn` | `latest_outcome_neural_controller` | `controller_rl_nmf` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `t3210_exponential_recurrence` | `causal_adaptive_control_student` | `controller_rl_nmf` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `t415_distilled_recurrence_student` | `causal_adaptive_control_student` | `controller_rl_nmf` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `t441_bounded_residual_gru_teacher` | `offline_teacher` | `controller_rl_nmf` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision | `NOT_REPLAYED_INELIGIBLE` |
| `t443_distilled_state4_student` | `causal_adaptive_control_student` | `controller_rl_nmf` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `t545_horizon_student_family` | `causal_adaptive_control_student` | `controller_rl_nmf` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `t554_quantized_gru_shadow` | `quantized_offline_teacher_shadow` | `controller_rl_nmf` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `t237_project_nmf_controller` | `model_based_nmf_controller` | `controller_rl_nmf` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `gqf_official_nmf_controller` | `official_source_controller` | `controller_rl_nmf` | decision_target, input_semantics, history_horizon, output_action, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `wang2022_direct_nn` | `external_direct_nn` | `surface_gkp_gate_outer_code` | code_family, modes_or_distance, decision_target, input_semantics, history_horizon, output_action, noise_model, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `sivak2023_rl_controller` | `external_rl_controller` | `controller_rl_nmf` | code_family, decision_target, input_semantics, history_horizon, output_action, noise_model, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |
| `sivak2026_rl_drift` | `external_rl_controller` | `controller_rl_nmf` | code_family, modes_or_distance, decision_target, input_semantics, history_horizon, output_action, noise_model, time_basis, compute_budget, precision, evidence_level | `NOT_REPLAYED_INELIGIBLE` |

没有 checkpoint 同时匹配 syndrome/action、observed-only history、cadence/warm-up、production fixed-point、MAC/state/workspace/wall-clock budget 和 parent trace。因而本 task 不产生 learned-decoder `p_L/p_X/p_Y/p_Z/average_ler/latency_ns` 排名；这些字段对全部 ineligible rows 都是 null。

## Legacy CNN diagnostic replay

保留模型 `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz` 在冻结的 206-sample test split 上重推理 5 次，输出 hash 完全一致，并与 T5.4.3 保存的逐样本预测 bit-exact。该模型输入为 21-channel、5-window histograms 与 teacher 参数/差分，输出为连续 `b_q/b_p` residual；它证明 artifact 可重放，不证明 logical decoding、drift-control gain 或 latency advantage。host batch median=0.398428 s，也不转换为 `latency_ns`。

## 方法边界

Wang 2022 是 surface-GKP direct decoder，但无 exact public checkpoint 且 code/task 不同；Sivak 2023/2026 是 experiment-in-loop controller；Puviani NMF 与项目 teacher/student 输出 15 个物理控制参数。它们分别留在 surface-GKP 或 controller lane，不与 single-mode Pauli decoder 合并。T6.15.5 后没有训练、超参搜索、checkpoint 重选或新 checkpoint 写入。

## 产物

- report：`docs/t6_17_3_learned_model_eligibility_replay.json`
- Source Data：`docs/t6_17_3_learned_model_eligibility_replay_source_data.csv`
- implementation：`cnn_fpga/benchmark/learned_model_eligibility_replay.py`
