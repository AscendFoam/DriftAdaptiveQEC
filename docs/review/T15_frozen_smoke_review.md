# T15 Review: P4 Multi-Scenario Frozen Baseline Bounded Smoke

**Reviewer**: Claude Code (normal review)
**Date**: 2026-05-09
**Task package**: `docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md`

---

## Verdict: **PASS_WITH_WARNINGS**

---

## 1. Blocking Issues

None.

---

## 2. Non-blocking Issues

### N1: handoff.md 多个历史/状态节未同步更新

Worker 更新了 handoff 的 Section 6（当前任务摘要）和 Section 7（下一步建议），但以下节未更新：

| 节 | 问题 |
|----|------|
| Section 1 | 仍写 `当前唯一任务：T15`，但 T15 已完成 |
| Section 2 "本轮已完成" | 未追加 T14/T15 条目 |
| Section 4 item 12 | 仍写"T15 是否运行多场景 bounded smoke，必须等待 T14 产出明确 run matrix 后再执行"——已过时 |
| Section 5 "已完成任务包" | 未追加 T14/T15 条目及关键产出 |

这不是功能性问题（Section 6 已正确记录 T15 output status），但会造成后续 Captain 或新 session 读取时状态不一致。建议 Captain 整合时一次性校正。

### N2: hybrid_residual_b 的 teacher diagnostics 全部为零

`summary.json` 中所有 10 个 comparison_rows 的 teacher 指标均为零：

- `teacher_contribution_l2_mean = 0.0`
- `teacher_scalar_abs_mean = 0.0`
- `teacher_gate_mean = 0.0`
- `teacher_gate_std = 0.0`
- `teacher_per_scalar = {}`

对 `ekf / ukf / constant_residual_mu / rls_residual_b` 这些经典基线，零值合理——它们不使用 CNN teacher 机制。但 `hybrid_residual_b` 模式明确依赖 `window_variance` teacher 的预测 + CNN 残差修正，teacher diagnostics 应为非零。

可能原因：
1. benchmark runner 对 `hybrid_residual_b` 模式未收集 teacher diagnostic 指标（仅对 gated-teacher 等变体收集）
2. 指标收集路径存在 bug

不影响 LER 结果的可信度（artifact path 正确、overflow 来源正确、correction_saturation = 0），但 teacher diagnostics 空缺会削弱后续 T16 对 hybrid residual-b 机制分析的深度。建议 T16 或后续任务排查。

### N3: delta_rows 全部为 null

`summary.json` 的 `delta_rows` 中，`delta_ler_vs_static_linear` 和 `delta_ler_vs_cnn_fpga` 全部为 null。这是预期行为——`p4_multiscenario_strong_baselines.yaml` 的 `frozen_baseline_set` 不包含 `static_linear` 和 `cnn_fpga`，所以 delta 无法计算。不影响结论，但建议 T16 reviewer 知道这个设计后果。

---

## 3. Missing Tests

无额外测试需求。本任务的核心验证对象是 run 输出，已通过直接读取 `summary.json` 完成交叉验证（见 Section 6）。

---

## 4. Suspicious Implementation Details

除了 N2 的 teacher diagnostics 全零问题外，无其他可疑点。

Run 数据内部一致性检查通过：

| 检查项 | 结果 |
|--------|------|
| `summary.json` 的 `scenario_winners` 与 `comparison_rows` 数值一致 | 通过 |
| `raw_rows` 的 20 行（2 scenario × 5 mode × 2 repeat）全部存在 | 通过 |
| `paired_seeds` 的 seed stride 正确（static: 20260403/20260404, linear: 20261403/20261404, stride=1000） | 通过 |
| `protocol_id = p4_strong_baselines_v1` 与 config 一致 | 通过 |
| `n_commits_applied` 非零且合理（899-900） | 通过 |
| `slow_update_violation_rate = 0.0` 对所有行 | 通过 |

---

## 5. Scope Compliance

### 5.1 Forbidden scope — 全部合规

| 禁止项 | 是否违反 |
|--------|----------|
| 不改 benchmark runner 代码 | 未违反（`cnn_fpga/` 无 diff） |
| 不改 frozen baseline set | 未违反（config 未改） |
| 不改场景定义 | 未违反 |
| 不改 ParamMapper | 未违反 |
| 不改训练 artifact | 未违反 |
| 不运行超出 T14 matrix 的长跑 | 未违反（matrix 完全匹配） |
| 不把结果写成正式 paper 结论 | 未违反（文档多处标注 "development bounded run"） |
| 不改 `docs/04_task_board.md` | 未违反（等 Captain 整合） |

### 5.2 Matrix 匹配验证

T14 protocol Section 6.2 定义：

| 参数 | T14 要求 | T15 实际执行 | 匹配 |
|------|----------|-------------|------|
| scenarios | `static_bias_theta`, `linear_ramp` | `static_bias_theta`, `linear_ramp` | OK |
| modes | `ekf, ukf, constant_residual_mu, rls_residual_b, hybrid_residual_b` | 完全一致 | OK |
| repeats | `2` | `2` | OK |
| paired seeds | `true` | `paired_seeds: true` | OK |
| interpreter | `C:\ProgramData\anaconda3\python.exe` | 一致 | OK |
| config | `p4_multiscenario_strong_baselines.yaml` | `protocol_id: p4_strong_baselines_v1` | OK |

---

## 6. Evidence Cross-Verification

以下 Worker 声明已通过直接读取 `summary.json` 和 `hil_summary.json` 交叉验证：

| Worker 声明 | 验证来源 | 结果 |
|-------------|----------|------|
| `missing_runs = []` | `summary.json` line 576 | 确认 |
| 全部 coverage = 1.0 | 10 个 `comparison_rows` 均为 `coverage: 1.0` | 确认 |
| `static_bias_theta` winner: hybrid_residual_b, LER=0.8109 | `scenario_winners[0]` + `comparison_rows[4]` | 确认 |
| `linear_ramp` winner: hybrid_residual_b, LER=0.7878 | `scenario_winners[1]` + `comparison_rows[9]` | 确认 |
| runner-up gap (static): 0.01447 | `scenario_winners[0].runner_up_gap` | 确认 |
| runner-up gap (linear): 0.02345 | `scenario_winners[1].runner_up_gap` | 确认 |
| `backend = mock` | `hil_summary.json` 两份抽查 | 确认 |
| `inference_service_mode = inproc` | `hil_summary.json` 两份抽查 | 确认 |
| non-learned: `artifact_path = null` | UKF hil_summary + summary.json | 确认 |
| hybrid: artifact = `runtime_b_residual_v1/...npz` | hybrid hil_summary + summary.json | 确认 |
| `correction_saturation_rate_mean = 0.0` | 全部 10 行 | 确认 |
| `aggressive_param_rate_mean = 0.0` | 全部 10 行 | 确认 |
| `dominant_overflow_source = histogram_input` | 全部 10 行 | 确认 |
| resumable continuation on same run_dir | `summary.json` `filters.resume_only = false` + 完整 coverage | 确认 |

---

## 7. Recommended Next Action

1. **Captain 整合**：Captain 审查本 review 后应：
   - 更新 `docs/04_task_board.md`：标记 T14 `[x]`、T15 `[x]`，切换 Current Unique Task 至 `T16`
   - 同步校正 `docs/07_handoff.md` 的 Sections 1/2/4/5，补齐 T14/T15 历史记录
2. **T16 gate review**：T16 应重点判断：
   - 当前双场景 bounded evidence 是否足够支撑论文用途
   - 是否需要补 `step_sigma_theta / periodic_drift` 后才能讨论 formal 恢复
   - N2（hybrid_residual_b teacher diagnostics 全零）是否需要排查
3. **Teacher diagnostics 排查**（可选，可在 T16 或后续任务中处理）：确认 `hybrid_residual_b` 模式下 teacher_contribution_l2 等指标是否本就不应在此 config 下收集，还是 runner 有 bug。这不影响当前 T15 的 LER 结论有效性，但会影响机制分析深度。
