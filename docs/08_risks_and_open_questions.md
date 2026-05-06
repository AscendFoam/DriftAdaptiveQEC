# Risks And Open Questions

## 风险清单

| ID | Risk | Level | Evidence | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | 默认运行环境不可直接执行最小 benchmark | 中 | 默认 `python 3.13.7` 仍缺 `numpy`，但 `C:\ProgramData\anaconda3\python.exe` 已可跑通 P0 smoke | 后续所有治理文档继续显式指定推荐解释器 |
| R2 | 根目录缺少统一依赖说明文件 | 中高 | 无 `requirements.txt`、`pyproject.toml`、`environment.yml` | 已补 `docs/P0_smoke_bootstrap.md` 与 `docs/P3_software_hil_bootstrap.md`，后续再决定是否补等价依赖文件 |
| R3 | 软件 HIL 与真板 HIL 边界容易被误写 | 高 | `cnn_fpga/hwio/board_backend.py` 仍是 placeholder 风格；`docs/03_hil_p4_boundary_audit.md` 已完成边界澄清 | 后续所有文档、复验与报告都必须引用 `docs/03_hil_p4_boundary_audit.md` 的统一口径 |
| R4 | 仓库中已有大量缓存与生成物噪声 | 中 | 已跟踪 `__pycache__/`、`.pyc`，且 `runs/` 中有大量生成配置 | 后续专门做 `T5`，先立规则再清理 |
| R5 | `run_hil_suite.py` 与 `run_p4_multiscenario_benchmark.py` 还没有完成恢复期之后的正式最小复验 | 中高 | `T4` 已跑通 bootstrap-level software HIL smoke，但还不是 `T6/T7` 的正式最小复验 | 保留 `T4` 最小路径，后续按 `T6/T7` 继续提升验证强度 |
| R6 | `.tflite` 真导出与 stub 回退容易混淆 | 中高 | `cnn_fpga/model/export.py` 与 `cnn_fpga/runtime/inference_service.py` 同时支持两种路径，且 runtime 输出不同 `source`；`T4` 当前刻意未走 `.tflite` 路径 | 文档与日志必须显式标注 `artifact type`，并区分 `tflite_service` 与 `tflite_stub_service` |
| R7 | 当前未跟踪文件会继续放大工作树混乱 | 中 | 恢复开始时已观察到未跟踪 `docs/reference/AI_coding_workflow.md` | 后续统一决定纳入或忽略策略 |

## 当前开放问题

1. 当前项目在这台机器上实际可用的 Python 环境是哪一个？
   - 当前初步答案：
     - P0 smoke: `C:\ProgramData\anaconda3\python.exe`
     - torch 训练候选: `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
2. 历史文档中引用的 `.venvs/tf311` 是否在本工作区外部，还是已经失效？
   - 当前已知：工作区内未找到该路径
3. `T4/T6` 的最小 software HIL 路径，默认应该先选哪条组合？
   - 当前答案：`hil.backend=mock` + `model_artifact` + `artifact_npz` + `inproc`
4. 训练与 benchmark 当前分别依赖哪些最小包集？
5. 是否需要为恢复期补一个最小 `requirements-recovery.txt` 或等价依赖文件？
6. 已跟踪的 `.pyc` 与 `__pycache__/` 后续是仅标记，还是清理出版本库？

## 暂缓事项

以下事项重要，但在软件 HIL / P4 最小复验闭环恢复前暂缓：

1. `noise_channels -> effective parameters` 桥接
2. load-aware latency injector
3. stateful fault injector
4. bit-accurate control pipeline
5. teacher-representation 新分支扩展
