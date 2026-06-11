# Task Package: T12

Task ID:
`T12`

Goal:
收敛 `hardware_hil_recovery_smoke` 这条 software HIL recovery smoke 的随机源与确定性表述，明确当前 run-to-run 差异来自哪里；如果可以在不改 benchmark 主线语义的前提下做最小修复，就把这条 bounded recovery path 提升到更强的可复验结论。

Why now:
`T11` 已经把 recovery 期最小依赖 manifest 收口到可接力状态，但 `T6/T10` 仍保留一个关键缺口：同一条 `hardware_hil_recovery_smoke` 路径的 control-plane 字段一致，`final_ler` / `overflow_rate` 却仍有小幅 run-to-run 差异。如果这一点不收口，仓库虽然“能跑”，却还没有真正恢复到更强的可复现状态。

Allowed files:
- `physics/syndrome_measurement.py`
- `cnn_fpga/runtime/fast_loop_emulator.py`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/tasks/P0/T12_software_hil_determinism_recovery.md`

Forbidden scope:
- 正式 benchmark 主线语义改写
- `cnn_fpga/decoder/`、ParamMapper、baseline 集合或统计口径变更
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
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/review/T10_gate_review.md`
- `docs/tasks/P0/T11_recovery_dependency_manifest.md`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/runtime/fast_loop_emulator.py`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/runtime/latency_injector.py`
- `cnn_fpga/hwio/mock_fpga.py`
- `physics/syndrome_measurement.py`
- `runs/hil_suite/hardware_hil_recovery_smoke_20260506_021326_3ae9f9176104/hil_summary.json`
- `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`

Expected output:
1. 一份明确写出的随机源链路说明
2. 若可行，一处最小而有界的 seed / RNG 收口修复
3. 两次连续 `hardware_hil_recovery_smoke` 复验的对比证据
4. 更新后的 task board、decision log、handoff、legacy audit、risk 与 P3 bootstrap 文档

Verification:
- `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- 再次执行同一命令一遍
- 对两次新 run 的：
  - `hil_summary.json`
  - `hil_events.json`
  计算哈希或逐字比对
- `Get-Content -Raw -Encoding UTF8 "docs/recovery_bootstrap/P3_software_hil_bootstrap.md"`

Docs to update:
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`

Reviewer type:
`adversarial`
