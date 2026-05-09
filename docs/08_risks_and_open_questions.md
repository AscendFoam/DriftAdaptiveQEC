# Risks And Open Questions

## 风险清单

| ID | Risk | Level | Evidence | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | 默认运行环境不可直接执行最小 benchmark | 中 | 默认 `python 3.13.7` 仍缺 `numpy`，但 `C:\ProgramData\anaconda3\python.exe` 已可跑通 P0 smoke | 后续所有治理文档继续显式指定推荐解释器 |
| R2 | 根目录虽已补 recovery-scoped manifest，但完整训练链、`.tflite` 与真板环境仍无统一依赖说明 | 中 | `requirements-recovery.txt` 只覆盖 `P0/P3/P4 recovery smoke`，且显式不含 `torch`、`tensorflow`、`tflite-runtime`；`docs/training_chain_bootstrap.md` 已补训练链 bootstrap，但还不是跨机器完整依赖锁定 | 继续保持作用域诚实；训练链已独立说明，`.tflite` 与真板路径仍需单开有界 manifest / bootstrap 任务 |
| R3 | 软件 HIL 与真板 HIL 边界容易被误写 | 高 | `cnn_fpga/hwio/board_backend.py` 仍是 placeholder 风格；`docs/03_hil_p4_boundary_audit.md` 已完成边界澄清 | 后续所有文档、复验与报告都必须引用 `docs/03_hil_p4_boundary_audit.md` 的统一口径 |
| R4 | 仓库中已有大量缓存与生成物噪声 | 中 | `.gitignore` 已忽略 `__pycache__/`、`runs/`、`artifacts/`，但 Git 中仍有 `116` 个已跟踪缓存/字节码文件、`1841` 个已跟踪 `runs/` 文件、`110` 个已跟踪 `artifacts/` 文件 | 已补 `docs/06_repo_noise_governance.md` 固定“先治理后清理”；后续需单开 cleanup 任务执行物理移除 |
| R5 | P4 目前已完成双场景、五模式、`repeats=2` 的 development bounded run，但仍没有恢复正式四场景 frozen benchmark | 中高 | 新 run 为 `runs/p4_benchmark/p4multis_20260508_221718_b82874_48280`；覆盖 `static_bias_theta + linear_ramp` 与 `ekf/ukf/constant_residual_mu/rls_residual_b/hybrid_residual_b`；两场景 winner 均为 `hybrid_residual_b` | `T16` 已判定当前证据足以支持继续受控开发，但不足以升级为正式四场景结论；如要补 `step_sigma_theta / periodic_drift`，必须新开任务包 |
| R6 | `.tflite` 真导出与 stub 回退容易混淆 | 中高 | `cnn_fpga/model/export.py` 与 `cnn_fpga/runtime/inference_service.py` 同时支持两种路径，且 runtime 输出不同 `source`；`T4/T7` 当前都刻意未走 `.tflite` 路径 | 文档与日志必须显式标注 `artifact type`，并区分 `tflite_service` 与 `tflite_stub_service` |
| R7 | 虽然 `T5` 已立治理口径，但具体 cleanup 执行窗口与归档方式仍未决定 | 中 | `docs/06_repo_noise_governance.md` 只固定了分类与阶段策略，尚未执行物理 cleanup | 在后续单开有界 cleanup 任务，显式列出 manifest、回滚方式与验收标准 |
| R8 | 最小 software HIL 路径虽然已在 bounded recovery path 上完成逐字一致复验，但该结论容易被误外推到真板、`.tflite` 或正式 benchmark | 中 | `T12` 已确认 `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104` 与 `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104` 的 `hil_summary.json` / `hil_events.json` 哈希一致；但路径仍固定为 `mock + model_artifact + artifact_npz + inproc` | 后续文档必须继续写清结论边界，不把 bounded recovery smoke 扩写成真板或正式 benchmark 已恢复 |
| R9 | 在 `T15` 已完成后，若继续直接扩大到剩余场景或更长 repeat，仍可能隐式越过 bounded/development/formal 边界 | 中高 | `T15` 已按协议跑完 `static_bias_theta + linear_ramp`、五模式、`repeats=2`；`docs/02_experiment_plan.md` 仍禁止无准备长跑 | `T16` 后仍不应自动追加 `step_sigma_theta`、`periodic_drift` 或更大 repeat；任何进一步 P4 扩展都必须新开任务包 |
| R10 | `hybrid_residual_b` 的 teacher diagnostics 在 T15 summary 中全零，可能影响机制分析深度 | 中 | `docs/review/T15_frozen_smoke_review.md` N2 指出所有 10 个 comparison rows 的 `teacher_contribution_l2_mean`、`teacher_scalar_abs_mean`、`teacher_gate_mean`、`teacher_gate_std` 均为 0，且 `teacher_per_scalar = {}` | `T16` 已将其判为非阻塞风险：在路径未澄清前，不把 teacher diagnostics 用作机制结论；当前优先转向 manifest / boundary 任务，而不是为了该指标直接重开 benchmark |
| R11 | 训练链 bootstrap 记录了本机 `torch` dev build，但尚未形成可移植依赖锁定 | 中 | `docs/review/T17_review.md` N1/N2 指出 `torch = 2.8.0.dev20250405+cu128` 是 dev build，且本轮未产出 `requirements-train.txt`；`docs/training_chain_bootstrap.md` 只承诺本机 `DLEnv` 探测结果 | 不把 `DLEnv` 或 dev torch 写成跨机器保证；若后续需要训练链可移植性，单开 `requirements-train.txt` / lockfile 任务，并显式说明 dev build 渠道限制 |

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
     - 训练链当前单独记录在 `docs/training_chain_bootstrap.md`，推荐解释器为本机 `DLEnv`
7. 是否需要再为训练链、`.tflite` 或真板路径补独立 manifest？
   - 当前答案：训练链 bootstrap 已补；`.tflite` 与真板路径仍需要后续独立任务
8. 已跟踪的 `.pyc` / `__pycache__/`、`runs/`、`artifacts/` 何时启动有界 cleanup，并如何拆分“bootstrap 必需”与“历史归档”？
9. 下一张继续开发任务包应该优先选哪一类？
   - 当前答案：
     - `T17` 已完成并通过 review。
     - 当前唯一任务已切到 `T18` `.tflite` manifest / boundary smoke plan。
10. `T15` 是否应直接运行多场景 P4 smoke？
   - 当前答案：已执行完成。
     - run dir: `runs/p4_benchmark/p4multis_20260508_221718_b82874_48280`
     - matrix:
       - `static_bias_theta + linear_ramp`
       - `ekf / ukf / constant_residual_mu / rls_residual_b / hybrid_residual_b`
       - `--paired-seeds`
       - `--repeats 2`
       - `C:\ProgramData\anaconda3\python.exe`
       - `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
     - two scenario winners:
       - `hybrid_residual_b`
       - `hybrid_residual_b`
11. `T15` 的 review warning 如何处理？
   - 当前答案：
     - N1 handoff 状态不一致：`accepted`，Captain 已修正 04/07 文档状态。
     - N2 `hybrid_residual_b` teacher diagnostics 全零：`T16` 已判定为非阻塞风险，继续保留在 R10。
     - N3 `delta_rows` 为 null：`accepted`，这是 strong-baseline config 不包含 `static_linear` / `cnn_fpga` 的预期后果，不应误判为缺失结果。
12. `T17` 的 review warning 如何处理？
   - 当前答案：
     - Verdict：`PASS`。
     - N1 `torch` dev build：`accepted`，只作为本机环境事实记录，不写成跨机器保证，风险保留到 R11。
     - N2 未产出 `requirements-train.txt`：`accepted`，因为任务允许用 `docs/training_chain_bootstrap.md` 收口；训练链可移植性后续单开任务。

## 暂缓事项

以下事项重要，但在新的任务包明确前暂缓：

1. `noise_channels -> effective parameters` 桥接
2. load-aware latency injector
3. stateful fault injector
4. bit-accurate control pipeline
5. teacher-representation 新分支扩展
6. 未经 `T14` 审计的 P4 长跑或正式 benchmark
7. 未经新任务包批准的 P4 剩余场景补跑
