# Task Package: T8

Task ID:
`T8`

Goal:
基于 `T6 + T7` 的最小复验证据，对当前项目做一次恢复期 gate review，明确判断项目现在应继续 `Repair` 还是进入 `Go`，并把决策理由、剩余缺口和下一唯一任务写回治理文档。

Why now:
`T6` 已经把最小 software HIL 路径重新验收到“可复验”状态，`T7` 又把最小 P4 benchmark 路径恢复到了同一套 `mock + model_artifact + artifact_npz + inproc` 口径。现在需要做一次正式收口，避免项目一边继续按恢复期硬规则执行，一边又在文档里默认自己已经进入正常开发状态。

Allowed files:
- `AGENTS.md`
- `README.md`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/review/T8_gate_review.md`
- `docs/tasks/P0/T8_gate_review_and_phase_decision.md`

Forbidden scope:
- `cnn_fpga/`、`physics/` 核心逻辑修改
- 正式 benchmark 口径扩写
- 真板设备联调
- `.tflite` 真导出主线扩写
- teacher-representation 新长跑
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
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/summary.json`
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/comparison.csv`

Expected output:
1. 一份明确的 gate review 结论
2. 明确写出的 `Go` 或 `Repair` 判断及其依据
3. 若继续 `Repair`，一个新的唯一任务与理由
4. 更新后的 task board、decision log、handoff、legacy audit、risk、README 与 AGENTS 状态

Verification:
- `Get-Content -Raw -Encoding UTF8 "docs/review/T8_gate_review.md"`
- `Get-Content -Raw -Encoding UTF8 "docs/04_task_board.md"`
- `Get-Content -Raw -Encoding UTF8 "docs/07_handoff.md"`
- `Get-Content -Raw -Encoding UTF8 "docs/08_risks_and_open_questions.md"`

Docs to update:
- `AGENTS.md`
- `README.md`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/review/T8_gate_review.md`

Reviewer type:
`milestone`
