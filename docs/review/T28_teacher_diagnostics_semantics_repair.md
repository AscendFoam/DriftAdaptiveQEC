# T28 Teacher Diagnostics Semantics Repair

## Verdict

`PASS_WITH_WARNINGS`

本任务完成了 `missing-vs-zero` 语义修复，并用有界 smoke 证明当前输出能区分：

- `not_applicable`
- `not_generated`
- `true zero`

但这不代表 broadcast teacher 路径已经具备 scalar-branch 机理诊断能力；它只是不再把“未生成”伪装成 `0.0`。

## Files Changed

- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

## Exact Repair Semantics

### 1. Runtime diagnostics now emit explicit status

`cnn_fpga/runtime/slow_loop_runtime.py`

- 为 teacher diagnostics 增加显式字段：
  - `teacher_diagnostics_status`
  - `teacher_diagnostics_status_reason`
  - `teacher_diagnostics_support_boundary`
  - `teacher_feature_layout`
  - `teacher_features_enabled`
- 当前 support boundary 明确写为：
  - `scalar_branch_only`

状态分类规则：

1. teacher features 未启用：
   - `teacher_diagnostics_status = not_applicable`
   - `teacher_diagnostics_status_reason = teacher_features_disabled`

2. teacher features 启用，但 `scalar_feature_dim <= 0`：
   - `teacher_diagnostics_status = not_generated`
   - `teacher_diagnostics_status_reason = broadcast_teacher_features_do_not_emit_scalar_branch_diagnostics`
   - `teacher_contribution_l2 / teacher_gate_* / prediction_without_teacher = null`

3. `scalar_feature_dim > 0` 且 explain 成功：
   - `teacher_diagnostics_status = generated`
   - `teacher_diagnostics_status_reason = scalar_branch_teacher_diagnostics_generated`

4. explain 失败：
   - `teacher_diagnostics_status = diagnostic_error`

### 2. HIL aggregation preserves missing instead of coercing to zero

`cnn_fpga/benchmark/run_hil_suite.py`

- 汇总层新增：
  - `teacher_diagnostics_status`
  - `teacher_diagnostics_status_reason`
  - `teacher_diagnostics_status_counts`
  - `teacher_diagnostics_windows_observed`
  - `teacher_diagnostics_generated_windows`
  - `teacher_scalar_feature_dim_mean/max`
  - `teacher_scalar_fusion_mode`
- 对 `teacher_contribution_l2_mean`、`teacher_scalar_abs_mean`、`teacher_gate_mean/std`：
  - 无数据时保留为 `null`
  - 不再在 HIL 汇总层写成 `0.0`

### 3. P4 writer keeps status and nulls

`cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

- per-repeat row 现在保留：
  - `teacher_diagnostics_status`
  - `teacher_diagnostics_status_reason`
  - `teacher_diagnostics_support_boundary`
  - `teacher_diagnostics_generated_windows`
- `teacher_contribution_l2_mean` / `teacher_scalar_abs_mean` / `teacher_gate_mean/std`：
  - 只在值存在时写数值
  - 缺失时保留为空/`null`
  - 不再用 `or 0.0` 压平成零
- `comparison.csv` 现在显式包含：
  - `teacher_diagnostics_status`
  - `teacher_diagnostics_status_reason`
  - `teacher_diagnostics_support_boundary`
  - `teacher_diagnostics_generated_repeats`
  - `teacher_scalar_feature_dim_mean`

## Verification

### Static check

Command:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m py_compile `
  cnn_fpga/runtime/slow_loop_runtime.py `
  cnn_fpga/benchmark/run_hil_suite.py `
  cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py
```

Result:

- passed

### Minimal smoke

为避免触发新的长跑 formal benchmark，本任务没有复跑 T24 口径。
改为在 T28 专用目录内做有界等价 smoke，只覆盖：

- `static_bias_theta`
- `ukf`
- `hybrid_residual_b`
- `repeats = 1`
- `n_slow_updates = 2`
- `n_fast_cycles = 8000`

Run dir:

- `runs/p4_benchmark/T28_teacher_diag_semantics_smoke_manual_20260511`

Execution path:

- 调用 `run_hil_session(...)`
- 使用当前 `run_p4_multiscenario_benchmark.py` 的聚合/CSV 写出函数生成：
  - `comparison.csv`
  - `teacher_scalar_diagnostics.csv`

Observed outputs:

1. `comparison.csv`

- `ukf`
  - `teacher_diagnostics_status = not_applicable`
  - `teacher_diagnostics_status_reason = mode_does_not_emit_teacher_diagnostics`
  - teacher diagnostics numeric columns empty
  - `correction_saturation_rate_mean = 0.0`

- `hybrid_residual_b`
  - `teacher_diagnostics_status = not_generated`
  - `teacher_diagnostics_status_reason = broadcast_teacher_features_do_not_emit_scalar_branch_diagnostics`
  - `teacher_diagnostics_support_boundary = scalar_branch_only`
  - `teacher_diagnostics_generated_repeats = 0`
  - `teacher_scalar_feature_dim_mean = 0.0`
  - `teacher_contribution_l2_mean_mean` empty
  - `teacher_scalar_abs_mean_mean` empty
  - `teacher_gate_mean_mean` empty
  - `teacher_gate_std_mean` empty
  - `correction_saturation_rate_mean = 0.0`

2. `teacher_scalar_diagnostics.csv`

- header only
- 但现在主 comparison row 已显式说明这是 `not_generated`

3. hybrid `hil_summary.json`

- `teacher_branch_diagnostics.teacher_diagnostics_status = not_generated`
- `teacher_contribution_l2_mean = null`
- `teacher_scalar_abs_mean = null`
- `teacher_gate_mean = null`
- `teacher_gate_std = null`

4. ukf `hil_summary.json`

- `teacher_branch_diagnostics.teacher_diagnostics_status = not_applicable`
- teacher diagnostics数值字段均为 `null`

## Risk Mapping

### R10

状态：`narrowed further, still open`

已进一步收窄为：

- 当前 formal mainline hybrid broadcast teacher 路径没有 scalar diagnostics 生成能力
- 现在这一点已经被显式编码成 `not_generated`
- 不再与 `true zero` 混淆

仍未关闭，因为：

- broadcast path 仍未生成 scalar-branch explain 数据
- 本任务没有补 broadcast explain 机制，也没有切 formal benchmark 到 scalar-branch teacher

### R21

状态：`closed for current writer semantics`

原因：

- 之前 downstream `0.0` coercion 已移除
- 当前 `comparison.csv` / `hil_summary.json` 已能区分：
  - `not_applicable`
  - `not_generated`
  - `0.0`

### R20

状态：`unchanged, remains open but independent`

本任务未改动 fast-loop saturation 采集逻辑。
当前 smoke 中 `correction_saturation_rate_mean = 0.0` 仍然存在，且与 teacher diagnostics 状态字段并列出现，说明它没有再被混淆成“缺失指标”。

## Historical Evidence Boundary

明确说明：

- `T24` 历史 run 目录未被修改
- 任何既有 `runs/` 或 `artifacts/` 文件均未重写
- 本任务新增的 smoke 结果仅位于：
  - `runs/p4_benchmark/T28_teacher_diag_semantics_smoke_manual_20260511`

## Conclusion

本任务修复的是“语义与可观测性”，不是 teacher 机理证据本身。

现在可以正确写：

- `ukf` teacher diagnostics: `not_applicable`
- current formal mainline `hybrid_residual_b` broadcast teacher diagnostics: `not_generated`
- `correction_saturation_rate_mean = 0.0`: 当前观测值为零，而不是被 writer 伪造为零
