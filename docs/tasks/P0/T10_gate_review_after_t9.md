# Task Package: T10

Task ID:
`T10`

Goal:
基于 `T8 + T9` 已经形成的恢复期证据，重新做一次 `Go / Repair` gate review，明确判断项目当前是否已经达到进入受控正常开发的门槛，并把 verdict、剩余缺口与下一唯一任务写回治理文档。

Why now:
`T8` 已经在 `T7` 的“两模式最小 P4 recovery smoke”证据下给出过一次 `Continue Repair`。`T9` 又把 `P4 frozen baseline` 的单场景证据扩到了四个正式 baseline。现在需要正式判断：这组增强后的证据是否足以把仓库从 `Repair` 切到 `Go`，还是仍然应继续恢复期。

Allowed files:
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/review/T10_gate_review.md`
- `docs/tasks/P0/T10_gate_review_after_t9.md`

Forbidden scope:
- `cnn_fpga/`、`physics/` 核心逻辑修改
- 正式 benchmark 主线语义改写
- 真板设备联调
- `.tflite` 真导出主线扩写
- 新训练、新 benchmark 长跑、teacher-representation 新扩展
- `runs/`、`artifacts/`、`__pycache__/` 的清理或 untrack

Inputs to read:
- `README.md`
- `AGENTS.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/02_experiment_plan.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/06_repo_noise_governance.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/review/T8_gate_review.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`
- `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/summary.json`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/summary.json`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/comparison.csv`

Expected output:
1. 一份新的 gate review 结论文档
2. 明确写出的 `Go` 或 `Repair` 判断及其依据
3. 若继续 `Repair`，一个新的唯一且有界的后续任务
4. 更新后的 task board、decision log、handoff、legacy audit 与 risk 文档

Verification:
- `Get-Content -Raw -Encoding UTF8 "docs/review/T10_gate_review.md"`
- `Get-Content -Raw -Encoding UTF8 "docs/04_task_board.md"`
- `Get-Content -Raw -Encoding UTF8 "docs/07_handoff.md"`
- `Get-Content -Raw -Encoding UTF8 "docs/08_risks_and_open_questions.md"`
- `Get-ChildItem -Name`

Docs to update:
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/review/T10_gate_review.md`

Reviewer type:
`milestone`
