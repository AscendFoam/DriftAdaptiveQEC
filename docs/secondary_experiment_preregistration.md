# T6.16.3 Phase 6C 二级实验预注册与只读边界

- verdict：`PASS_PHASE6C_READONLY_SECONDARY_PREREGISTRATION`
- downstream preregistrations：`9`；gates/mutations：`17/17`、`17/17`
- secondary seed namespace：`phase6c-secondary-v1`
- Phase 6B byte hash 会因任务板元数据重建而变化，因此同时保存 initial artifact hash 与 scientific semantic hash；后续只允许元数据重建，不允许改变 verdict/gates/claims/dropped/absence。

## 只读锁

- V5 LER gate 固定 `10%`，incremental action-space gate 固定 `12%`；comparator=`NONE_EARLY_STOP`，formal split=`NOT_CREATED`。
- step-calibration/telegraph worst-window endpoints 固定为 `NOT_RUN_EARLY_STOP`，不得用 Phase 6C secondary 数值代替。
- T6.10.2--T6.15.4 共 20 项保持 Dropped；T6.13.3/T6.14.*/T6.15.1--4 output absence proof 为真。
- Phase 6C 不得改变 T6.15.5、挽救 V5 或解锁 T6.9.2。

## 二级实验清单

| task | lane | execution | split/seeds | runtime cap | failure branches |
| --- | --- | --- | --- | ---: | --- |
| `T6.17.1` | `single_mode_decoder` | `PROJECT_NATIVE_MATCHED` | secondary_correctness / 1 seeds | 7200 s | NEGATIVE_EQUIVALENCE_FAILED, PARTIAL_RUNTIME_BUDGET_EXCEEDED |
| `T6.17.2` | `surface_gkp_gate_outer_code` | `PROJECT_NATIVE_MATCHED_OR_BLOCKED` | secondary_gate_reproduction / 32 seeds | 28800 s | BLOCKED_SOURCE_INCOMPLETE, NEGATIVE_ANCHOR_TOLERANCE_FAIL, PARTIAL_NMAX_OR_RUNTIME |
| `T6.17.3` | `single_mode_decoder` | `READONLY_CHECKPOINT_REPLAY` | existing_parent_trace_replay / 0 seeds | 7200 s | INELIGIBLE_TASK_SIGNATURE, NEGATIVE_REPLAY, PARTIAL_REPLAY |
| `T6.18.1` | `aqec_wallclock` | `PROJECT_NATIVE_MATCHED` | secondary_aqec_independent / 24 seeds | 14400 s | NEGATIVE_NO_LIFETIME_GAIN, PARTIAL_RUNTIME, BLOCKED_OFFICIAL_PROTOCOL_REPRODUCTION |
| `T6.18.2` | `multimode_structured_lattice_cpd` | `OFFICIAL_CODE_REPRODUCTION` | secondary_cpd_stationary / 32 seeds | 28800 s | BLOCKED_TOOLCHAIN, PARTIAL_UPSTREAM_OR_MODEL_MISMATCH, NEGATIVE_THRESHOLD_TOLERANCE_FAIL, PARTIAL_RUNTIME |
| `T6.18.3` | `multimode_structured_lattice_cpd` | `CONDITIONAL_PROJECT_NATIVE_MATCHED` | secondary_multimode_drift / 32 seeds | 28800 s | NOT_RUN_SCOPE_GATE, NEGATIVE_NO_DRIFT_GAIN, PARTIAL_RUNTIME |
| `T6.19.1` | `fpga_implementation` | `PROJECT_NATIVE_MATCHED` | secondary_preboard_hardware / 3 seeds | 14400 s | N_A_NO_RTL, NEGATIVE_ACTION_MISMATCH, PARTIAL_TOOL_RUNTIME |
| `T6.19.2` | `fpga_implementation` | `LITERATURE_REFRESH` | secondary_literature_refresh / 0 seeds | 7200 s | PARTIAL_SEARCH, INELIGIBLE_TASK_SIGNATURE, NEGATIVE_ZERO_SAME_TASK_COMPARATOR |
| `T6.19.3` | `fpga_implementation` | `READONLY_INTEGRITY_GATE` | no_new_data / 0 seeds | 7200 s | FAIL_INTEGRITY_HASH, FAIL_INTEGRITY_CROSS_LANE, FAIL_INTEGRITY_EVIDENCE_PROMOTION, FAIL_INTEGRITY_INCOMPLETE |

## 统计与停止规则

- paired bootstrap=20,000；threshold bootstrap=2,000；resampling unit 是 seed cluster/trajectory，不是相关 round。
- Holm 只在单 task 预声明 endpoint family 内执行，不跨 lane 汇总 p-value、胜场或总分。
- 保存 raw counts/denominator/seed/config/hash/all attempted cells；零事件报告 exact one-sided 95% upper bound。
- 只允许 correctness/source/tool/runtime/entry gate 提前停止；不得因结果有利或不利而改候选、容差、size、sigma grid 或 endpoint。

## 当前工具状态不是结果

- Julia：`None`；Yosys：`None`；nextpnr-gowin：`None`。缺失工具在对应 task 使用预注册 BLOCKED/PARTIAL 分支，不能用自写替代实现追求正值。

## 产物

- `configs/literature/t6_16_3_secondary_preregistration.json`
- `docs/t6_16_3_secondary_preregistration.json`
- `docs/t6_16_3_secondary_preregistration_source_data.csv`（158 rows）
