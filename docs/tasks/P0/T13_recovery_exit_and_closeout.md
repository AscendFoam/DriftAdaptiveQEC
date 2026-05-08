# Task Package: T13

Task ID:
`T13`

Goal:
对 `Phase 1: Recovery` 做正式退出评审与治理收尾；如果当前证据已经满足工作流中的 `Recovery-Ready` 条件，就把仓库状态从 `Phase 1: Recovery / Repair` 切换到“可继续开发，但仍受边界约束”的下一阶段，并把这一结论同步回治理文件。

Why now:
`T11` 已补齐 recovery-scoped manifest，`T12` 已把 bounded software HIL recovery smoke 收口到逐字一致复验。恢复期前两轮 gate review 中最关键的三个缺口里，已有两个被收口，当前需要把“是否可以退出 Recovery”这件事正式写成仓库事实，而不是继续停留在口头判断。

Allowed files:
- `README.md`
- `AGENTS.md`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/tasks/P0/T13_recovery_exit_and_closeout.md`
- `docs/review/T13_recovery_exit_review.md`

Forbidden scope:
- 不改任何 benchmark 口径、baseline 集合、ParamMapper 语义
- 不新增训练、长跑 benchmark、teacher-representation 扩展
- 不把 `real_board`、真实 `.tflite` runtime 或正式多场景 P4 benchmark 写成已恢复
- 不做代码重构、依赖安装、`runs/`/`artifacts/`/`__pycache__/` 清理

Inputs to read:
- `README.md`
- `AGENTS.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/02_experiment_plan.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/reference/AI_coding_workflow.md`
- `docs/review/T8_gate_review.md`
- `docs/review/T10_gate_review.md`
- `docs/tasks/P0/T11_recovery_dependency_manifest.md`
- `docs/tasks/P0/T12_software_hil_determinism_recovery.md`
- `docs/P0_smoke_bootstrap.md`
- `docs/P3_software_hil_bootstrap.md`
- `docs/P4_benchmark_recovery_bootstrap.md`

Expected output:
1. 一份明确的 recovery exit review 结论
2. 一次正式的阶段/决策状态切换，或明确写出的“不允许切换”理由
3. 更新后的 README / AGENTS / task board / handoff / decision log / risks / audit
4. 明确的下一阶段边界与下一唯一任务状态

Verification:
- `Get-Content -Encoding utf8 README.md`
- `Get-Content -Encoding utf8 AGENTS.md`
- `Get-Content -Encoding utf8 docs/04_task_board.md`
- `Get-Content -Encoding utf8 docs/07_handoff.md`
- `Get-Content -Encoding utf8 docs/08_risks_and_open_questions.md`
- `Get-Content -Encoding utf8 docs/review/T13_recovery_exit_review.md`

Docs to update:
- `README.md`
- `AGENTS.md`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/review/T13_recovery_exit_review.md`

Reviewer type:
`milestone`
