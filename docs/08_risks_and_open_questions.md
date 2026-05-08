# Risks And Open Questions

## 风险清单

| ID | Risk | Level | Evidence | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | 默认运行环境不可直接执行最小 benchmark | 中 | 默认 `python 3.13.7` 仍缺 `numpy`，但 `C:\ProgramData\anaconda3\python.exe` 已可跑通 P0 smoke | 后续所有治理文档继续显式指定推荐解释器 |
| R2 | 根目录虽已补 recovery-scoped manifest，但完整训练链、`.tflite` 与真板环境仍无统一依赖说明 | 中 | `requirements-recovery.txt` 只覆盖 `P0/P3/P4 recovery smoke`，且显式不含 `torch`、`tensorflow`、`tflite-runtime` | 继续保持作用域诚实；若后续确需训练或 `.tflite` 路径，再单开有界 manifest 任务 |
| R3 | 软件 HIL 与真板 HIL 边界容易被误写 | 高 | `cnn_fpga/hwio/board_backend.py` 仍是 placeholder 风格；`docs/03_hil_p4_boundary_audit.md` 已完成边界澄清 | 后续所有文档、复验与报告都必须引用 `docs/03_hil_p4_boundary_audit.md` 的统一口径 |
| R4 | 仓库中已有大量缓存与生成物噪声 | 中 | `.gitignore` 已忽略 `__pycache__/`、`runs/`、`artifacts/`，但 Git 中仍有 `116` 个已跟踪缓存/字节码文件、`1841` 个已跟踪 `runs/` 文件、`110` 个已跟踪 `artifacts/` 文件 | 已补 `docs/06_repo_noise_governance.md` 固定“先治理后清理”；后续需单开 cleanup 任务执行物理移除 |
| R5 | P4 目前虽已完成 frozen baseline 单场景全模式 smoke 复验，但仍没有恢复正式多场景 frozen benchmark | 中高 | `T9` 已复验 `static_bias_theta + static_linear/window_variance/ekf/cnn_fpga + repeats=1`，run dir 为 `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`；`T10` 已明确这仍不足以直接进入 `Go` | 基于 `T12` 已完成的确定性收口，再决定是继续补多场景证据还是先补其他环境说明 |
| R6 | `.tflite` 真导出与 stub 回退容易混淆 | 中高 | `cnn_fpga/model/export.py` 与 `cnn_fpga/runtime/inference_service.py` 同时支持两种路径，且 runtime 输出不同 `source`；`T4/T7` 当前都刻意未走 `.tflite` 路径 | 文档与日志必须显式标注 `artifact type`，并区分 `tflite_service` 与 `tflite_stub_service` |
| R7 | 虽然 `T5` 已立治理口径，但具体 cleanup 执行窗口与归档方式仍未决定 | 中 | `docs/06_repo_noise_governance.md` 只固定了分类与阶段策略，尚未执行物理 cleanup | 在后续单开有界 cleanup 任务，显式列出 manifest、回滚方式与验收标准 |
| R8 | 最小 software HIL 路径虽然已在 bounded recovery path 上完成逐字一致复验，但该结论容易被误外推到真板、`.tflite` 或正式 benchmark | 中 | `T12` 已确认 `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104` 与 `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104` 的 `hil_summary.json` / `hil_events.json` 哈希一致；但路径仍固定为 `mock + model_artifact + artifact_npz + inproc` | 后续文档必须继续写清结论边界，不把 bounded recovery smoke 扩写成真板或正式 benchmark 已恢复 |
| R9 | Phase 2 若直接运行 P4 多场景 benchmark，可能隐式扩大计算范围或混淆 recovery/development/formal 口径 | 中高 | `docs/02_experiment_plan.md` 明确禁止无准备长跑；`T9` 只覆盖 `single-scenario + four-mode + repeats=1` | 当前唯一任务设为 `T14`，先做 protocol audit 与 bounded run plan，再决定是否进入 `T15` |

## 当前开放问题

1. 当前项目在这台机器上实际可用的 Python 环境是哪一个？
   - 当前答案：
     - P0/P3/P4 recovery smoke: `C:\ProgramData\anaconda3\python.exe`
     - torch 训练候选: `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
2. 历史文档中引用的 `.venvs/tf311` 是否在本工作区外部，还是已经失效？
   - 当前已知：工作区内未找到该路径
3. `T4/T6/T7` 的最小 recovery 复验路径，默认应该先选哪条组合？
   - 当前答案：
     - software HIL: `hil.backend=mock` + `model_artifact` + `artifact_npz` + `inproc`
     - P4 benchmark 最小路径: `p4_multiscenario_recovery_smoke.yaml` + `static_bias_theta` + `static_linear/cnn_fpga` + `paired_seeds`
     - P4 frozen baseline smoke: `p4_multiscenario_recovery_smoke.yaml` + `static_bias_theta` + `static_linear/window_variance/ekf/cnn_fpga` + `paired_seeds`
4. `T9` 的 `single-scenario / four-mode / repeats=1` 证据，是否已经足以支撑项目从 `Repair` 进入 `Go`？
   - 当前答案：在 `T10` 时点是否；但结合 `T11 + T12 + T13` 后，答案是可以进入“受控 `Go`”
5. 最小 software HIL bounded recovery path 是否已经收口到更严格的确定性复现？
   - 当前答案：是。`T12` 已完成，且两次新 run 的 `hil_summary.json` / `hil_events.json` 已逐字一致
6. 训练与 recovery benchmark 当前分别依赖哪些最小包集？
   - 当前答案：
     - recovery smoke root manifest: `numpy + PyYAML`
     - 训练链仍单独依赖 `DLEnv / torch`
7. 是否需要再为训练链、`.tflite` 或真板路径补独立 manifest？
   - 当前答案：可能需要，但不在当前恢复优先级
8. 已跟踪的 `.pyc` / `__pycache__/`、`runs/`、`artifacts/` 何时启动有界 cleanup，并如何拆分“bootstrap 必需”与“历史归档”？
9. 下一张继续开发任务包应该优先选哪一类？
   - 当前答案：
     - 已选 `T14: P4 frozen benchmark protocol audit and bounded run plan`
     - 后续候选按 `docs/04_task_board.md` 排队：`T15` P4 bounded smoke、`T17` training manifest、`T18` `.tflite` manifest、`T19` cleanup manifest、`T20` real-board readiness
10. `T15` 是否应直接运行多场景 P4 smoke？
   - 当前答案：否。必须先完成 `T14`，明确 run matrix、解释器、配置、scenario/mode/repeat 与边界。

## 暂缓事项

以下事项重要，但在新的任务包明确前暂缓：

1. `noise_channels -> effective parameters` 桥接
2. load-aware latency injector
3. stateful fault injector
4. bit-accurate control pipeline
5. teacher-representation 新分支扩展
6. 未经 `T14` 审计的 P4 长跑或正式 benchmark
