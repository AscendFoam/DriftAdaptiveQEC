# T13 Recovery Exit Review

Date:
`2026-05-08`

Verdict:
`Allow`

Decision:
当前仓库已经完成 `Phase 1: Recovery` 的第一轮收尾，可以退出恢复期，进入“受控继续开发”阶段；对应决策状态可从 `Repair` 提升为 `Go`，但这个 `Go` 只代表“允许继续开发”，不代表所有历史路径都已完全恢复。

## Evidence Reviewed

1. `docs/reference/AI_coding_workflow.md`
2. `docs/02_experiment_plan.md`
3. `docs/04_task_board.md`
4. `docs/07_handoff.md`
5. `docs/08_risks_and_open_questions.md`
6. `docs/recovery_bootstrap/P0_smoke_bootstrap.md`
7. `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
8. `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
9. `docs/tasks/P0/T11_recovery_dependency_manifest.md`
10. `docs/tasks/P0/T12_software_hil_determinism_recovery.md`
11. `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104/hil_summary.json`
12. `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104/hil_summary.json`
13. `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/summary.json`
14. `docs/review/T8_gate_review.md`
15. `docs/review/T10_gate_review.md`

## Recovery Exit Criteria Check

按 `AI_coding_workflow.md` 第 4 节迁移/恢复工作流，当前已满足：

1. 文档给出的最小路径可以真实运行：
   - P0 smoke 已复验
   - P3 bounded software HIL recovery smoke 已复验
   - P4 bounded recovery smoke 已复验
2. MVP 与边界清楚：
   - `mock / stub / placeholder` 已明示
   - `real_board` 与 `.tflite` 未被误写为已完成
3. task board / handoff / risks 已形成稳定接力面
4. recovery-scoped manifest 已补齐：
   - `requirements-recovery.txt`
5. bounded software HIL recovery path 已完成逐字一致复验：
   - `hil_summary.json` 一致
   - `hil_events.json` 一致
6. 后续工作已不再依赖旧长 session 才能解释仓库状态

## Why Allow Go Now

1. 之前阻塞 `Go` 的三个关键缺口中：
   - recovery-scoped manifest 已由 `T11` 收口
   - software HIL 确定性已由 `T12` 收口
   - 剩余的 P4 缺口已经从“最小路径不存在”降为“正式多场景证据仍不足”
2. 这个剩余缺口已经不再阻止仓库进入“受控继续开发”：
   - 它会影响“正式 benchmark 是否恢复”
   - 但不再影响“仓库是否具备可信接力面”

## What This Does Not Mean

1. 不代表 `real_board` 已恢复
2. 不代表真实 `.tflite` runtime 已恢复
3. 不代表正式多场景 frozen benchmark 已恢复
4. 不代表可以绕过任务包、验证和边界说明直接自由扩功能

## Practical Conclusion

当前最合理的阶段判断是：

1. `Phase 1: Recovery` 已完成第一轮收尾
2. 仓库进入 `Phase 2: Controlled Development`
3. 决策状态切换为 `Go`
4. 下一唯一任务应重新定义为一个 bounded 开发/验证任务，而不是继续把整个仓库当成恢复期废墟
