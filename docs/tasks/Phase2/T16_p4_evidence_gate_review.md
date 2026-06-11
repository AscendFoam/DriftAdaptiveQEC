# T16: P4 Evidence Gate Review

Task ID: `T16`

Goal: 对 `T14 + T15` 的 P4 benchmark 证据做 gate review，决定是否允许继续扩大 benchmark、转向环境 manifest，或暂停。

Why now: P4 benchmark 证据会影响后续是否进入更正式的多场景复验，因此必须先做 reviewer-style gate，而不是连续扩大运行规模。

Allowed files:

- `docs/tasks/Phase2/T16_p4_evidence_gate_review.md`
- `docs/review/T16_p4_evidence_gate_review.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/05_decision_log.md`

Forbidden scope:

- 不运行新的 benchmark
- 不修改代码或 config
- 不把单次 smoke 结果升级为正式论文结论

Inputs to read:

- `docs/review/T15_frozen_smoke_review.md`
- `docs/protocols/benchmark/P4_benchmark_development_protocol.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/08_risks_and_open_questions.md`
- `docs/reference/AI_coding_workflow.md`

Expected output:

- `docs/review/T16_p4_evidence_gate_review.md`
- 结论必须是：
  - `Allow`: 允许下一步扩大 P4 benchmark
  - `Conditional`: 允许但附条件
  - `Block`: 暂停 P4 扩展，转入其他任务
- 必须显式处理 `T15` review warning：
  - N2: `hybrid_residual_b` teacher diagnostics 全零，判断是否影响 gate
  - N3: `delta_rows` 为 null，说明这是 strong-baseline config 不含 `static_linear` / `cnn_fpga` 的预期后果还是需要后续补报表

Verification:

- 只读审查，无新运行。
- 检查 gate review 是否明确列出 blocking / non-blocking issues 与下一建议任务。

Docs to update:

- `docs/review/T16_p4_evidence_gate_review.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/05_decision_log.md`（若改变决策状态）

Reviewer type: `milestone`

## Worker Output Summary

- Review result: `Conditional`
- No new benchmark run executed
- No code or config changes made
- Updated docs:
  - `docs/review/T16_p4_evidence_gate_review.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
