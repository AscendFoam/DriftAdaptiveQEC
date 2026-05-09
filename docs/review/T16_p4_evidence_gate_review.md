# T16 Review: P4 Evidence Gate Review

**Reviewer**: Codex Worker (milestone review)
**Date**: 2026-05-09
**Task package**: `docs/tasks/Phase2/T16_p4_evidence_gate_review.md`

---

## Verdict: **Conditional**

---

## 1. Blocking Issues

None.

---

## 2. Non-blocking Issues

### N1: `T15` 仍然只是 bounded development evidence，不是正式四场景 frozen benchmark

当前证据足以支持继续受控开发，但还不足以把 `T15` 升级写成正式四场景 benchmark 已恢复。

### N2: `hybrid_residual_b` 的 teacher diagnostics 全零

这更像是指标收集缺口、配置边界或 runner 指标路径问题，而不是“teacher contribution 不存在”的充分证据。它不阻塞当前 LER 排序证据，但会限制机制分析深度。

### N3: `delta_rows = null` 在本次 strong-baseline bounded run 中是预期后果

`p4_multiscenario_strong_baselines.yaml` 的 baseline 集不包含 `static_linear` / `cnn_fpga`，因此这里的 `delta_rows` 为空不应被判定为 missing run 或报表失败。

---

## 3. Evidence Assessment

当前 `T14 + T15` 已提供以下足够稳定的 gate 证据：

1. `T15` matrix 与 `T14` protocol 保持一致：
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`
   - repeats: `2`
   - seed policy: `paired`
2. `runs/p4_benchmark/p4multis_20260508_221718_b82874_48280` 已完成：
   - `missing_runs = []`
   - `coverage = 1.0`
   - `raw_rows = 20`
3. 两个已覆盖场景里，winner 都是 `hybrid_residual_b`，runner-up 都是 `ukf`，方向上与历史 strong-baseline 主结论一致。
4. 边界仍然诚实：
   - `mock-backed P4 wrapper over software HIL`
   - 不是 `real_board`
   - 不是 `.tflite` runtime
   - 不是正式四场景 benchmark 已恢复

---

## 4. Gate Decision

本次 gate 结论为 `Conditional`：

1. 允许项目继续 Phase 2 的受控开发。
2. 不建议在本轮直接继续扩大 P4 benchmark 到剩余场景。
3. 更合适的下一方向是转向独立环境 manifest / boundary 任务，如 `T17` / `T18`。
4. 在 `teacher diagnostics` 问题未澄清前，不把该指标写入机制性结论。

---

## 5. Recommended Next Action

1. 优先转向 `T17` 或 `T18` 这类独立 manifest / boundary 任务。
2. 保持 `R10` 为非阻塞风险，后续单独澄清 `teacher diagnostics` 路径。
3. 若未来重开 `step_sigma_theta` / `periodic_drift`，必须通过新的任务包进入，不应把它当作 `T16` 的隐含后续动作。
