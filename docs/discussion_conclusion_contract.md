# T7.2.4 Discussion/Conclusion 证据合同

- verdict：`PASS_DISCUSSION_LIMITATIONS_COST_AND_PHYSICAL_TRANSITION_BOUNDARIES`
- gates：`22/22`
- semantic mutations：`22/22`
- discussion rows：`27`
- board measured fields：`0/42` non-null
- V5：`20` dropped，`0` downstream outputs

## 结果状态

| ID | 主题 | 状态 | 边界 |
| --- | --- | --- | --- |
| DC-001 | restricted_contribution | `INTERPRETATION_ESTABLISHED` | Integration and falsification, not a winning decoder |
| DC-002 | role_separation | `INTERPRETATION_ESTABLISHED` | MAP, safety FSM, fast path, and learning modules own different claims |
| DC-003 | locked_ewma_positive | `INTERPRETATION_ESTABLISHED` | Positive only against the preregistered EWMA contrast |
| DC-004 | static_window_negative | `LIMITATION_REQUIRED` | Static and Window prevent a best-decoder claim |
| DC-005 | v5_causal_stop | `LIMITATION_REQUIRED` | Observed-only causal/action headroom failed before V5 implementation |
| DC-006 | tail_noninferiority | `INTERPRETATION_ESTABLISHED` | Safety non-inferiority is not broad tail superiority |
| DC-007 | intervention_occupancy | `COST_REQUIRED` | Fallback and unnecessary fallback can dominate tail intervals |
| DC-008 | host_update_cost | `COST_REQUIRED` | Slow-loop cadence and compute budget are separate from fast-path cycles |
| DC-009 | fast_path_cost | `COST_REQUIRED` | Six-cycle II=1 and P&R are pre-board evidence |
| DC-010 | offline_learning_only | `LIMITATION_REQUIRED` | No board-resident or on-board training |
| DC-011 | board_measurement_null | `LIMITATION_REQUIRED` | All 42 physical-board fields remain null |
| DC-012 | single_mode_model_scope | `EXTERNAL_VALIDITY_REQUIRED` | Single-mode square-lattice syndrome/effective model |
| DC-013 | no_calibrated_cavity_transmon | `EXTERNAL_VALIDITY_REQUIRED` | No calibrated cavity/transmon pulse-level device model or experiment |
| DC-014 | synthetic_drift_scope | `EXTERNAL_VALIDITY_REQUIRED` | Held-out synthetic generators do not establish real drift prevalence |
| DC-015 | no_physical_lifetime | `LIMITATION_REQUIRED` | Per-round simulated LER is not a physical lifetime |
| DC-016 | no_outer_threshold | `LIMITATION_REQUIRED` | No surface-code or fault-tolerant threshold from single-mode data |
| DC-017 | task_signature_comparison | `INTERPRETATION_ESTABLISHED` | CI/ML/NN/AQEC/CPD/NMF/FPGA lanes have no global leaderboard |
| DC-018 | phase6c_separation | `LIMITATION_REQUIRED` | Secondary CPD/CNOT/AQEC evidence cannot rescue V5 |
| DC-019 | identifiability_first | `INTERPRETATION_ESTABLISHED` | Require prospective causal information before estimator expansion |
| DC-020 | action_value_first | `INTERPRETATION_ESTABLISHED` | Require realizable action-value headroom before compiler expansion |
| DC-021 | real_data_intake | `FUTURE_GATE_ONLY` | Immutable metadata, units, labels, permission, and chronological splits |
| DC-022 | shadow_mode | `FUTURE_GATE_ONLY` | Prospective output logging without actuation |
| DC-023 | board_hil | `FUTURE_GATE_ONLY` | Named-board streaming HIL populates timing, power, and 42 fields |
| DC-024 | guarded_qpu | `FUTURE_GATE_ONLY` | Frame first; displacement only after separate authorization |
| DC-025 | physical_effectiveness | `FUTURE_GATE_ONLY` | Matched corrected and best-physical channels on the same device |
| DC-026 | break_even_gate | `FUTURE_GATE_ONLY` | Decay-rate ratio with simultaneous lower confidence bound above one |
| DC-027 | balanced_conclusion | `CONCLUSION_BOUNDARY` | Conclusion retains positives, negatives, nulls, and prohibited upgrades |

本合同只允许 restricted simulator/pre-board 结论。真实 cavity/transmon、物理 lifetime/beyond-break-even、板上训练、板测 speed/power 与闭环 QPU 均必须经独立 future gate。
