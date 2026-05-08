# Task Package: T6

Task ID:
`T6`

Goal:
重新验收一条软件 HIL 最小路径，确认恢复期最小 bootstrap 仍可复用，并把实际复验结果、backend、artifact type 与解释器口径写回治理文档。

Why now:
`T4` 已经恢复了最小 software HIL bootstrap，`T5` 又把仓库噪声治理边界固定好了。现在需要做一次新的最小复验，确认这条路径不是一次性偶然成功，也不是被历史噪声掩盖的假阳性。

Allowed files:
- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/P3_software_hil_bootstrap.md`
- `docs/tasks/P0/T6_software_hil_reverification.md`

Forbidden scope:
- `runs/`、`artifacts/`、`__pycache__/` 的清理或 untrack
- `cnn_fpga/`、`physics/` 核心逻辑修改
- 真板设备联调
- `.tflite` 真导出主线扩写
- teacher-representation 新分支扩展

Inputs to read:
- `docs/P3_software_hil_bootstrap.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/06_repo_noise_governance.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- `runs/hil_suite/hardware_hil_recovery_smoke_20260506_021326_3ae9f9176104/hil_summary.json`

Expected output:
1. 一次新的 software HIL 最小复验结果
2. 明确写出的 backend / slow-loop mode / inference mode / artifact type / artifact path
3. 更新后的 task board、decision log、handoff、legacy audit、risk 口径
4. 若复验失败，输出可复现的阻塞证据

Verification:
- `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- `Get-Content -Raw -Encoding UTF8 "<run_dir>/hil_summary.json"`
- `Get-Content -Raw -Encoding UTF8 "<run_dir>/hil_events.json"`

Docs to update:
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/P3_software_hil_bootstrap.md`

Reviewer type:
`milestone`
