# Risks And Open Questions

## 风险清单

| ID | Risk | Level | Evidence | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | 默认运行环境不可直接执行最小 benchmark | 中 | 默认 `python 3.13.7` 仍缺 `numpy`，但 `C:\ProgramData\anaconda3\python.exe` 已可跑通 P0 smoke | 后续所有治理文档继续显式指定推荐解释器 |
| R2 | 根目录缺少统一依赖说明文件 | 中高 | 无 `requirements.txt`、`pyproject.toml`、`environment.yml` | 已补 `docs/P0_smoke_bootstrap.md`、`docs/P3_software_hil_bootstrap.md` 与 `docs/P4_benchmark_recovery_bootstrap.md`，后续再决定是否补等价依赖文件 |
| R3 | 软件 HIL 与真板 HIL 边界容易被误写 | 高 | `cnn_fpga/hwio/board_backend.py` 仍是 placeholder 风格；`docs/03_hil_p4_boundary_audit.md` 已完成边界澄清 | 后续所有文档、复验与报告都必须引用 `docs/03_hil_p4_boundary_audit.md` 的统一口径 |
| R4 | 仓库中已有大量缓存与生成物噪声 | 中 | `.gitignore` 已忽略 `__pycache__/`、`runs/`、`artifacts/`，但 Git 中仍有 `116` 个已跟踪缓存/字节码文件、`1841` 个已跟踪 `runs/` 文件、`110` 个已跟踪 `artifacts/` 文件 | 已补 `docs/06_repo_noise_governance.md` 固定“先治理后清理”；后续需单开 cleanup 任务执行物理移除 |
| R5 | P4 目前虽已完成 frozen baseline 单场景全模式 smoke 复验，但仍没有恢复正式多场景 frozen benchmark | 中高 | `T9` 已复验 `static_bias_theta + static_linear/window_variance/ekf/cnn_fpga + repeats=1`，run dir 为 `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`；但当前仍只有单场景、单 repeat 证据 | 先执行 `T10` 重新做 `Go / Repair` gate review；若仍是 `Repair`，再决定是继续补多场景证据还是优先收口依赖 manifest / 确定性复现 |
| R6 | `.tflite` 真导出与 stub 回退容易混淆 | 中高 | `cnn_fpga/model/export.py` 与 `cnn_fpga/runtime/inference_service.py` 同时支持两种路径，且 runtime 输出不同 `source`；`T4/T7` 当前都刻意未走 `.tflite` 路径 | 文档与日志必须显式标注 `artifact type`，并区分 `tflite_service` 与 `tflite_stub_service` |
| R7 | 虽然 `T5` 已立治理口径，但具体 cleanup 执行窗口与归档方式仍未决定 | 中 | `docs/06_repo_noise_governance.md` 只固定了分类与阶段策略，尚未执行物理 cleanup | 在 `T8` 之后单开有界 cleanup 任务，显式列出 manifest、回滚方式与验收标准 |
| R8 | 最小 software HIL 路径当前更接近“可复验”而非“逐字确定性复现” | 中 | 两次 `hardware_hil_recovery_smoke` run 的 control-plane 字段一致，但 `final_ler` / `overflow_rate` 存在小幅差异；`physics/syndrome_measurement.py` 仍使用全局 `np.random` 路径 | 先保持 `T6` 的“可复验”表述；后续若需要更严格复现性，再单开任务统一随机源控制 |

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
   - 当前答案：尚未重新 gate review；下一任务是 `T10`
5. 两次最小 software HIL 复验的 `final_ler` / `overflow_rate` 小幅差异，是否需要专门收敛到更严格的确定性复现？
6. 训练与 benchmark 当前分别依赖哪些最小包集？
7. 是否需要为恢复期补一个最小 `requirements-recovery.txt` 或等价依赖文件？
8. 已跟踪的 `.pyc` / `__pycache__/`、`runs/`、`artifacts/` 何时启动有界 cleanup，并如何拆分“bootstrap 必需”与“历史归档”？

## 暂缓事项

以下事项重要，但在 `T10` 给出新的 gate review 结论前暂缓：

1. `noise_channels -> effective parameters` 桥接
2. load-aware latency injector
3. stateful fault injector
4. bit-accurate control pipeline
5. teacher-representation 新分支扩展
