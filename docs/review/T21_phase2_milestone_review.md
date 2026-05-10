# T21 Review: Phase 2 Milestone Review And Next-Phase Decision

**Task ID**: `T21`
**Reviewer**: Codex Worker (milestone review)
**Date**: 2026-05-10
**Task package**: `docs/tasks/Phase2/T21_phase2_milestone_review.md`

---

## Verdict: `Conditional`

---

## 1. Milestone Completion Summary

`T14` 至 `T20` 已覆盖 Phase 2 原计划中的五类受控任务：

1. `T14-T16`
   - P4 benchmark evidence hardening
   - 已从 recovery smoke 扩到 bounded development evidence
   - 但未恢复 formal four-scenario frozen benchmark
2. `T17-T18`
   - 训练链与 `.tflite` 路径已独立收口为 bootstrap / manifest
   - 但都没有被升级为跨机器完整环境保证
3. `T19`
   - tracked cache cleanup 已形成 manifest
   - 但未执行物理 cleanup
4. `T20`
   - real-board HIL readiness checklist 已形成
   - 但仍是 readiness / acceptance criteria，不是真板验证

结论：

- Phase 2 当前任务队列已完成一轮 bounded documentation / evidence hardening / readiness 收口。
- 当前仓库状态适合继续受控开发。
- 当前仓库状态不适合把任何 readiness / development evidence 升级为“已正式恢复”。

## 2. Evidence Level Assessment

### 2.1 P4 benchmark line

- `T14`: protocol audit
- `T15`: `development_smoke`
- `T16`: gate verdict = `Conditional`

当前最准确口径：

- `T15` 是 bounded multi-scenario development evidence
- 不是 formal frozen benchmark revalidation

### 2.2 Training chain

- `T17` 已形成 bootstrap 文档
- 当前更接近 `code_exists + bootstrap_confirmed`
- 不是跨机器完整依赖锁定

### 2.3 `.tflite` line

- `T18` 已形成 boundary smoke plan
- 当前更接近 `code_exists + boundary_audited`
- 真实 runtime 仍未达到 `recovery_smoke`

### 2.4 Repo hygiene

- `T19` 已形成 cleanup manifest
- 当前更接近 `plan_ready`
- 未达到“physical cleanup executed”

### 2.5 Real-board HIL

- `T20` 已形成 readiness checklist
- 当前更接近 `readiness_defined`
- 明确未达到 `hardware_validated`

## 3. Remaining Blockers And Warnings

当前不适合直接进入更强结论的四类主要阻塞仍然存在：

1. P4 formal benchmark blocker
   - `T15` 只覆盖 `static_bias_theta + linear_ramp`
   - 仍缺 `step_sigma_theta + periodic_drift`
   - 当前仍不能写成 formal four-scenario frozen benchmark 已恢复
2. `.tflite` runtime blocker
   - 本机 `tensorflow = False`
   - 本机 `tflite_runtime = False`
   - 只能确认 `tflite_stub_v1` 边界，不是真实 runtime 恢复
3. Physical cleanup blocker
   - `T19` 只产出 manifest
   - `116` 个 tracked `.pyc` 仍未 untrack
   - `runs/` / `artifacts/` 仍未进入独立 cleanup 执行任务
4. Real-board blocker
   - `board_backend.py` 仍是 placeholder
   - `T20` 只定义了 readiness / acceptance criteria
   - 真板 smoke 仍缺设备、权限、地址表来源、量化阈值与平台确认

非阻塞但必须继续保留的 warning：

1. `R10`
   - `hybrid_residual_b` teacher diagnostics 全零
   - 当前不影响 bounded LER 排序证据，但影响机制分析深度
2. `R11`
   - 训练链仍依赖本机 `DLEnv` 与 dev torch build
3. `R14`
   - 真板 readiness checklist 里的寄存器来源、阈值与权限模型仍需在后续执行任务中具体化

## 4. Gate Decision

本次 `T21` gate 结论为 `Conditional`。

含义：

1. 允许项目继续停留在 `Phase 2: Controlled Development`。
2. 允许继续开 bounded 下一任务。
3. 不允许把当前阶段升级写成：
   - formal P4 benchmark 已恢复
   - 真实 `.tflite` runtime 已恢复
   - physical cleanup 已完成
   - real-board HIL 已验证

这不是 `Block`，因为：

- `T14-T20` 都已按边界完成，且没有 blocking review failure；
- 当前文档、风险和 handoff 已具备继续接力的稳定性。

这也不是 `Allow`，因为：

- 几条高风险路径都还停在 development / bootstrap / manifest / readiness 层；
- 若直接跨到真板 smoke、formal benchmark 或物理 cleanup，容易打破当前边界诚实。

## 5. Recommended Next Unique Task

推荐下一唯一任务：

- `T22: Real-board smoke execution plan with platform / AXI-map audit and quantitative acceptance thresholds`

推荐理由：

1. `T20` 已把真板 readiness checklist 固定下来，但 `R13/R14` 仍是当前最集中的高风险边界。
2. 真板路径目前缺的是“可执行计划与量化阈值”，而不是继续扩写 readiness。
3. 这个下一任务仍可保持为文档/计划类 bounded task，不必直接调用硬件。
4. 相比立刻重开 formal P4 benchmark、真实 `.tflite` runtime 或 physical cleanup，真板 smoke execution plan 更能收口后续高风险任务的进入条件。

低优先级候选，但不建议优先：

1. tracked cache cleanup execution task
2. `.tflite` runtime environment execution task
3. full four-scenario P4 formal benchmark reopen task

## 6. Review Boundary Reminder

本 review 明确不声称：

1. `T15` development run = formal benchmark
2. `T20` readiness checklist = real-board validation
3. `T19` manifest = physical cleanup complete
4. `T18` bootstrap = true `.tflite` runtime restored
