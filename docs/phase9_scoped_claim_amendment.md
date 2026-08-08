# Phase 9 scoped claim amendment（T9.1.5）

> 本文档是 parent-bound、pre-outcome 的 claim 协议 child seal，不是解码性能、外部 SOTA、真实板延迟或物理寿命结果。

## 1. 不可变 parent

- T9.1.1 `analysis_sha256`：`c88110375c358794339e72d672e4624871425fe480e5da091ddd1d6595255e18`；
- T9.1.4 `analysis_sha256`：`d6c5ac4fd9587854cd6fec7d390c1fd2ddd5300bbe069ff2db1e16950aa21b7d`；
- 文献截止日：`2026-07-25T01:37:24+08:00`；
- 当前 external same-task eligible：`0`。

## 2. 当前 scoped states

- `GO_LER_REGISTERED_BEST`：`NOT_EVALUATED_NULL`，value/verdict=`null`；原因 `NO_PHASE9_FORMAL_OUTCOME`。
- `GO_LER_EXTERNAL_SOTA`：`MISSING_EXTERNAL_COMPARATOR_NULL`，value/verdict=`null`；原因 `ZERO_EXTERNAL_SAME_TASK_ELIGIBLE_AND_NO_FORMAL_OUTCOME`。
- `GO_LIFETIME_PROJECT_NATIVE`：`NOT_EVALUATED_NULL`，value/verdict=`null`；原因 `LIFETIME_POWER_AND_FORMAL_OUTCOME_NOT_YET_FROZEN`。
- `GO_LIFETIME_EXTERNAL_SOTA`：`MISSING_EXTERNAL_COMPARATOR_NULL`，value/verdict=`null`；原因 `ZERO_EXTERNAL_SAME_TASK_ELIGIBLE_AND_NO_PROJECT_NATIVE_LIFETIME`。
- `OFFICIAL_PUVIANI_EXACT`：`MISSING_EXTERNAL_ASSET_NULL`，value/verdict=`null`；原因 `OFFICIAL_CHECKPOINT_SEEDS_SELECTION_LEDGER_AND_SIX_STATE_EVALUATOR_MISSING`。
- `GO_PUVIANI_NMF_SURPASS`：`MISSING_EXTERNAL_ASSET_NULL`，value/verdict=`null`；原因 `OFFICIAL_PUVIANI_EXACT_IS_NULL`。
- `GO_PHYSICAL_LIFETIME`：`MISSING_QPU_NULL`，value/verdict=`null`；原因 `NO_QPU_OR_REAL_GKP_MEASUREMENT`。
- `GO_HIL_INTEGRATED`：`MISSING_BOARD_NULL`，value/verdict=`null`；原因 `HIGH_SPEED_BOARD_NOT_AVAILABLE`。
- `GO_HIL_EXTERNAL_SPEED`：`MISSING_BOARD_NULL`，value/verdict=`null`；原因 `HIL_INTEGRATED_AND_EXTERNAL_SAME_BOUNDARY_COMPARATOR_ARE_NULL`。

## 3. state 证据与撤销边界

- `GO_LER_REGISTERED_BEST`：scope=`PROJECT_NATIVE_REGISTERED_MATCHED_DEPLOYABLE`，14 个原子 predicate，5 个 state-specific revocation triggers；允许措辞：best result among all fully qualified methods in the frozen Phase-9 matched deployable registry。
- `GO_LER_EXTERNAL_SOTA`：scope=`EXTERNAL_SAME_TASK_LITERATURE_CUTOFF_BOUND`，12 个原子 predicate，5 个 state-specific revocation triggers；允许措辞：external same-task LER SOTA as of the bound literature cutoff。
- `GO_LIFETIME_PROJECT_NATIVE`：scope=`PROJECT_NATIVE_SIX_STATE_SIMULATION`，13 个原子 predicate，4 个 state-specific revocation triggers；允许措辞：project-native six-state simulated logical-lifetime advantage under the frozen task。
- `GO_LIFETIME_EXTERNAL_SOTA`：scope=`EXTERNAL_SAME_TASK_LIFETIME_CUTOFF_BOUND`，12 个原子 predicate，4 个 state-specific revocation triggers；允许措辞：external same-task six-state logical-lifetime SOTA as of the bound literature cutoff。
- `OFFICIAL_PUVIANI_EXACT`：scope=`OFFICIAL_SOURCE_EXACT_REPRODUCTION`，10 个原子 predicate，4 个 state-specific revocation triggers；允许措辞：official-source exact Puviani NMF reproduction under the separately frozen evaluator。
- `GO_PUVIANI_NMF_SURPASS`：scope=`OFFICIAL_EXACT_SAME_TASK_SURPASS`，10 个原子 predicate，4 个 state-specific revocation triggers；允许措辞：surpasses the independently reproduced official Puviani NMF on the same frozen six-state task。
- `GO_PHYSICAL_LIFETIME`：scope=`QPU_REAL_GKP_MEASURED`，16 个原子 predicate，5 个 state-specific revocation triggers；允许措辞：measured physical logical-lifetime advantage on a real-GKP QPU under the stated physical break-even definition。
- `GO_HIL_INTEGRATED`：scope=`REAL_BOARD_END_TO_END_INTEGRATION`，20 个原子 predicate，5 个 state-specific revocation triggers；允许措辞：integrated measured raw/recorded-IQ to trigger HIL operation on the identified board and bitstream。
- `GO_HIL_EXTERNAL_SPEED`：scope=`EXTERNAL_SAME_BOUNDARY_MEASURED_SPEED`，13 个原子 predicate，5 个 state-specific revocation triggers；允许措辞：faster measured HIL latency than every eligible external decoder on the same frozen boundary/task/platform contract。

## 4. legacy migration（不自动迁移）

- `MIG-LER-GO-SPLIT`：`ROUND_LER_SINGLE_MODE/GO_LER_SOTA` → candidate `GO_LER_REGISTERED_BEST`；auto=`false`；当前 mapped value=`null`。
- `MIG-LER-NOGO-LOCAL`：`ROUND_LER_SINGLE_MODE/NO_GO_LER_SOTA` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LIFETIME-GO-SPLIT`：`SIX_STATE_LOGICAL_LIFETIME/GO_LIFETIME` → candidate `GO_LIFETIME_PROJECT_NATIVE`；auto=`false`；当前 mapped value=`null`。
- `MIG-LIFETIME-NOGO-LOCAL`：`SIX_STATE_LOGICAL_LIFETIME/NO_GO_LIFETIME` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-HIL-GO-SPLIT`：`RAW_IQ_DIGITAL_HIL/GO_HIL_SPEED` → candidate `GO_HIL_INTEGRATED`；auto=`false`；当前 mapped value=`null`。
- `MIG-HIL-NOGO-LOCAL`：`RAW_IQ_DIGITAL_HIL/NO_GO_HIL_SPEED` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LER-NULL`：`ROUND_LER_SINGLE_MODE/NOT_EVALUATED_NULL` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LER-INCOMPLETE-INTEGRITY`：`ROUND_LER_SINGLE_MODE/INCOMPLETE_LER_INTEGRITY` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LER-INCOMPLETE-BASELINE`：`ROUND_LER_SINGLE_MODE/INCOMPLETE_LER_BASELINE` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LER-INCOMPLETE-EVIDENCE`：`ROUND_LER_SINGLE_MODE/INCOMPLETE_LER_EVIDENCE` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LER-NOGO-TAIL`：`ROUND_LER_SINGLE_MODE/NO_GO_LER_SOTA_TAIL_ONLY` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LER-NOGO-ROBUSTNESS`：`ROUND_LER_SINGLE_MODE/NO_GO_LER_SOTA_ROBUSTNESS_ONLY` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LER-NOGO-NONINFERIOR`：`ROUND_LER_SINGLE_MODE/NO_GO_LER_SOTA_NONINFERIOR` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LER-NOGO-NEGATIVE`：`ROUND_LER_SINGLE_MODE/NO_GO_LER_SOTA_NEGATIVE` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LIFETIME-NULL`：`SIX_STATE_LOGICAL_LIFETIME/NOT_EVALUATED_NULL` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LIFETIME-INCOMPLETE-INTEGRITY`：`SIX_STATE_LOGICAL_LIFETIME/INCOMPLETE_LIFETIME_INTEGRITY` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LIFETIME-INCOMPLETE-BASELINE`：`SIX_STATE_LOGICAL_LIFETIME/INCOMPLETE_LIFETIME_BASELINE` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LIFETIME-INCOMPLETE-EVIDENCE`：`SIX_STATE_LOGICAL_LIFETIME/INCOMPLETE_LIFETIME_EVIDENCE` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LIFETIME-NOGO-FIT`：`SIX_STATE_LOGICAL_LIFETIME/NO_GO_LIFETIME_FIT` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LIFETIME-NOGO-COST`：`SIX_STATE_LOGICAL_LIFETIME/NO_GO_LIFETIME_COST` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LIFETIME-NOGO-NONINFERIOR`：`SIX_STATE_LOGICAL_LIFETIME/NO_GO_LIFETIME_NONINFERIOR` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-LIFETIME-NOGO-NEGATIVE`：`SIX_STATE_LOGICAL_LIFETIME/NO_GO_LIFETIME_NEGATIVE` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-HIL-MISSING-BOARD`：`RAW_IQ_DIGITAL_HIL/MISSING_BOARD_NULL` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-HIL-INCOMPLETE-INTEGRITY`：`RAW_IQ_DIGITAL_HIL/INCOMPLETE_HIL_SPEED_INTEGRITY` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-HIL-INCOMPLETE-EVIDENCE`：`RAW_IQ_DIGITAL_HIL/INCOMPLETE_HIL_SPEED_EVIDENCE` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-HIL-INCOMPLETE-COMPARATOR`：`RAW_IQ_DIGITAL_HIL/INCOMPLETE_HIL_SPEED_NO_COMPARATOR` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-HIL-NOGO-DEADLINE`：`RAW_IQ_DIGITAL_HIL/NO_GO_HIL_SPEED_DEADLINE` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-HIL-NOGO-NEGATIVE`：`RAW_IQ_DIGITAL_HIL/NO_GO_HIL_SPEED_NEGATIVE` → candidate `null`；auto=`false`；当前 mapped value=`null`。
- `MIG-HIL-ENGINEERING-NONRANKING`：`RAW_IQ_DIGITAL_HIL/GO_HIL_ENGINEERING_NONRANKING` → candidate `null`；auto=`false`；当前 mapped value=`null`。

## 5. 禁止证据迁移

- `FT-V2-REGISTERED-TO-EXTERNAL-LER`：`GO_LER_REGISTERED_BEST` ↛ `GO_LER_EXTERNAL_SOTA`（`MISSING_EXTERNAL_SAME_TASK_LEDGER_AND_AUDIT`）。
- `FT-V2-PROJECT-TO-EXTERNAL-LIFETIME`：`GO_LIFETIME_PROJECT_NATIVE` ↛ `GO_LIFETIME_EXTERNAL_SOTA`（`MISSING_EXTERNAL_SAME_TASK_LEDGER_AND_AUDIT`）。
- `FT-V2-PROJECT-TO-PHYSICAL`：`GO_LIFETIME_PROJECT_NATIVE` ↛ `GO_PHYSICAL_LIFETIME`（`SIMULATION_IS_NOT_QPU_MEASUREMENT`）。
- `FT-V2-INTEGRATED-TO-EXTERNAL-SPEED`：`GO_HIL_INTEGRATED` ↛ `GO_HIL_EXTERNAL_SPEED`（`MISSING_EXTERNAL_SAME_BOUNDARY_COMPARATOR`）。
- `FT-V2-RECORDED-TO-LIVE-HIL`：`BOARD_RECORDED_IQ_REPLAY` ↛ `BOARD_LIVE_RAW_IQ_HIL`（`RECORDED_REPLAY_IS_NOT_LIVE_ADC_RAW_IQ`）。
- `FT-V2-PREBOARD-TO-INTEGRATED`：`CXXRTL_OR_SYNTHESIS` ↛ `GO_HIL_INTEGRATED`（`ESTIMATE_IS_NOT_REAL_BOARD_HIL`）。
- `FT-V2-SIX-CYCLE-TO-END-TO-END`：`DISCRIMINATOR_OUT_TO_ACTION_6_CYCLES` ↛ `ADC_OR_REPLAY_TO_TRIGGER_LATENCY`（`LATENCY_BOUNDARY_MISMATCH`）。
- `FT-V2-PAPER-CONSTRAINED-TO-OFFICIAL`：`PAPER_CONSTRAINED_REIMPLEMENTATION` ↛ `OFFICIAL_PUVIANI_EXACT`（`MISSING_OFFICIAL_ASSETS`）。
- `FT-V2-PAPER-CONSTRAINED-TO-SURPASS`：`T9.1.3_SHORT_HORIZON_RESULT` ↛ `GO_PUVIANI_NMF_SURPASS`（`OFFICIAL_EXACT_AND_SAME_TASK_REQUIRED`）。
- `FT-V2-LIFETIME-TO-LER`：`SIX_STATE_LOGICAL_LIFETIME` ↛ `GO_LER_REGISTERED_BEST`（`CROSS_LANE_SUBSTITUTION`）。
- `FT-V2-LER-TO-LIFETIME`：`ROUND_LER_SINGLE_MODE` ↛ `GO_LIFETIME_PROJECT_NATIVE`（`CROSS_LANE_SUBSTITUTION`）。
- `FT-V2-HIL-TO-ALGORITHM`：`RAW_IQ_DIGITAL_HIL` ↛ `GO_LER_REGISTERED_BEST`（`HARDWARE_SAFETY_OR_SPEED_IS_NOT_LER`）。
- `FT-V2-CROSS-TASK-EXTERNAL`：`CROSS_CODE_OR_CROSS_BOUNDARY_RESULT` ↛ `ANY_EXTERNAL_SOTA`（`SAME_TASK_ELIGIBILITY_REQUIRED`）。
- `FT-V2-MISSING-TO-ZERO`：`MISSING_OR_UNAVAILABLE_EVIDENCE` ↛ `ZERO_OR_PASS`（`TYPED_NULL_REQUIRED`）。
- `FT-V2-LEGACY-AUTO-MAP`：`GO_LER_SOTA_OR_GO_LIFETIME_OR_GO_HIL_SPEED` ↛ `ANY_V2_GO_STATE`（`FRESH_SCOPED_REEVALUATION_REQUIRED`）。
- `FT-V2-POINT-TO-PROMOTION`：`POINT_ESTIMATE_ONLY` ↛ `ANY_V2_GO_STATE`（`SIMULTANEOUS_CI_AND_INDEPENDENT_AUDIT_REQUIRED`）。
- `FT-V2-NULLABLE-HIL-BLOCKS-ALGORITHM`：`T9.7.4_TERMINAL_BLOCKED_NULL` ↛ `BLOCK_ALGORITHM_ONLY_OR_SPLIT_DECISION`（`NULLABLE_HIL_DEPENDENCY_MUST_NOT_BLOCK_SOFTWARE_VERDICTS`）。

## 6. nullable terminal

- T9.1.2 缺 official assets 只保持 official exact / Puviani surpass 为 null；
- T9.7.4 可为 `Done` 或 terminal `Blocked/null`，无板只保持 HIL states 为 null；
- 无 Phase-8 QPU/real-GKP 证据只保持 physical lifetime 为 null；
- 上述局部 null 均不得阻塞 algorithm-only、split 或诚实 NO-GO 决策。

## 7. 反简化证据

- 9 个 state 各有 complete/incomplete/revoked 三类 synthetic schema fixtures，共 `27` 条；value/verdict 均为 null，generic evaluator 也永不签发 GO；
- `4` 条 executable revocation fixtures验证 exact prerequisite closure、sequence/previous-entry hash chain、externally pinned snapshot anchor 和证据 hash 保留；
- legacy migration `29` 条，forbidden transfers `17` 条；
- gates `36/36`，mutations `36/36`。

## 8. 复现

```powershell
python -m cnn_fpga.benchmark.phase9_scoped_claim_amendment
python -m cnn_fpga.benchmark.phase9_scoped_claim_amendment --verify
python -m pytest -q tests/test_phase9_scoped_claim_amendment.py
```
