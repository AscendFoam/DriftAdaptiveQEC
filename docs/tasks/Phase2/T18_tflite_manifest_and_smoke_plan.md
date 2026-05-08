# T18: TFLite Manifest And Smoke Plan

Task ID: `T18`

Goal: 为 `.tflite` export/runtime 路径补独立 manifest 与 smoke plan，继续严格区分真实 `.tflite` 与 `tflite_stub_v1`。

Why now: `.tflite` 路径已有代码与历史证据，但当前恢复期未复验真实 runtime；后续若要推进部署路径，必须先固定环境和边界。

Allowed files:

- `docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`
- `docs/TFLite_runtime_bootstrap.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Forbidden scope:

- 不改 `cnn_fpga/model/export.py`
- 不改 `cnn_fpga/runtime/inference_service.py`
- 不把 `.tflite.json` stub manifest 写成真实 `.tflite` runtime
- 不改 HIL benchmark 口径

Inputs to read:

- `docs/03_hil_p4_boundary_audit.md`
- `cnn_fpga/model/export.py`
- `cnn_fpga/runtime/inference_service.py`
- `cnn_fpga/model/evaluate_tflite.py`
- `cnn_fpga/model/validate_export.py`

Expected output:

- `docs/TFLite_runtime_bootstrap.md`
- 明确：
  - 真实 `.tflite` export/runtime 的依赖
  - stub manifest 的边界
  - 可运行 smoke 命令
  - 当前无法验证时的阻塞项

Verification:

- 只读代码审计加环境探测。
- 如环境具备，可运行最小 `--help` 或 import smoke；不得强行改代码绕过依赖。

Docs to update:

- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/04_task_board.md`（Captain 整合阶段）

Reviewer type: `normal`

