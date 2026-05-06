# Legacy Audit

## 1. 审计目的

本轮审计只做只读核查，目标是回答：

1. 这个项目到底已经有什么
2. 哪些是代码已实现
3. 哪些只是未来扩展或 placeholder
4. 当前唯一恢复任务应该是什么

## 2. 总体结论

### 2.1 结论摘要

- 项目主体代码真实存在，且明显不是最小脚手架
- P0/P2/P3/P4 的 benchmark 入口与对应配置存在
- 软件 HIL 主线存在，但真板 backend 仍未到“已完成”状态
- 默认环境不可直接复现实验，当前最明显阻塞是缺依赖与入口说明
- `.tflite` 部署链路同时存在真实导出/runtime 与 stub 回退路径
- 仓库此前缺少治理文件，导致“代码真实状态”和“开发流程状态”没有被固定

### 2.2 恢复推进状态

- 初始恢复任务建议是 `T1: 确认依赖矩阵与最小入口`
- 截至 `2026-05-06` 已完成：
  - `T1`
  - `T2`
  - `T3`
  - `T4`
- 当前下一唯一任务建议为：
  - `T5: 清点并处理仓库中的缓存/生成物噪声治理策略`

说明：

- `T3` 已把 `mock` / `stub` / `placeholder` 边界固定到 `docs/03_hil_p4_boundary_audit.md`
- `T4` 已把恢复期最小 software HIL 路径固定到 `docs/P3_software_hil_bootstrap.md`
- 后续 P3/P4 文档与复验结果都应沿用该口径

## 3. Feature Reality Matrix

| Feature | Claimed status | Evidence path | Verified? | Risk |
| --- | --- | --- | --- | --- |
| P0 full vs simplified 基线脚本 | 已存在最小对比脚本 | `benchmark/compare_full_vs_simplified_ler.py` | 部分验证 | 默认环境缺 `numpy`，当前无法在系统 Python 下直接运行 |
| P1 数据与训练链 | 已有数据集构建、训练、评估、量化入口 | `cnn_fpga/data/dataset_builder.py`, `cnn_fpga/model/train.py`, `cnn_fpga/model/evaluate.py`, `cnn_fpga/model/quantize.py` | 代码存在 | 依赖矩阵未确认，当前未复跑 |
| P2 行为级硬件仿真 | 已有硬件行为仿真与模式 benchmark | `cnn_fpga/benchmark/run_hardware_emulation.py`, `cnn_fpga/benchmark/run_p2_mode_benchmark.py` | 代码存在 | 当前环境未复验 |
| P3 软件 HIL | 已有 HIL 主流程、mock backend、驱动抽象、推理服务 | `cnn_fpga/benchmark/run_hil_suite.py`, `cnn_fpga/hwio/mock_fpga.py`, `cnn_fpga/runtime/inference_service.py`, `docs/03_hil_p4_boundary_audit.md`, `docs/P3_software_hil_bootstrap.md` | 最小路径已重验 | bootstrap-level 路径已恢复，但仍不等于正式 P3 验收 |
| P3 真板 HIL | 当前是 placeholder real-board backend | `cnn_fpga/hwio/board_backend.py`, `docs/CNN_FPGA_GKP_阶段结论.md`, `docs/03_hil_p4_boundary_audit.md` | 是 | 真实设备节点和地址表缺失，不能写成已完成 |
| P4 多场景 benchmark | 已有统一 benchmark 汇总脚本，但实际复用 HIL 会话 | `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`, `docs/03_hil_p4_boundary_audit.md` | 边界已审计 | 真实性继承自 HIL backend / artifact |
| teacher-representation 多版本分支 | 已有 v2-v9 配置与配对 benchmark 入口 | `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v*.yaml`, `cnn_fpga/benchmark/run_p4_teacher_representation_paired.py` | 代码存在 | 当前不应继续扩分支，先恢复可信度 |
| 真板 backend 语义 | placeholder/骨架状态 | `cnn_fpga/hwio/board_backend.py` | 是 | 若表述不严谨，极易误导项目完成度判断 |
| `.tflite` 真导出路径 | 代码支持真导出与 stub 回退双路径 | `cnn_fpga/model/export.py`, `cnn_fpga/runtime/inference_service.py`, `docs/03_hil_p4_boundary_audit.md` | 边界已审计 | 必须明确区分真实 `.tflite`、artifact 与 stub manifest |
| 根级治理文件 | 恢复前缺失 | 根目录与 `docs/` | 是 | 高，直接影响后续接力与审查 |

## 4. 关键证据

### 4.1 代码主干不是空壳

关键证据：

- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/model/train.py`
- `physics/logical_tracking.py`

### 4.2 真板路径尚未完成

关键证据：

- `cnn_fpga/hwio/board_backend.py`
  - 文件顶层注释直接标注为 placeholder real-board backend
- `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md`
  - 明确写到 board backend 仍是 placeholder 级

### 4.3 `.tflite` 路径不能默认视为真实部署

关键证据：

- `cnn_fpga/model/export.py`
  - 优先导出真实 `.tflite`
  - 失败时回退为 `tflite_stub_v1`
- `cnn_fpga/runtime/inference_service.py`
  - 真实路径使用 `tflite_service`
  - stub manifest 路径使用 `tflite_stub_service`

### 4.4 默认环境不可信

关键证据：

- 根目录没有：
  - `requirements.txt`
  - `pyproject.toml`
  - `environment.yml`
- 最小 benchmark 在默认 `python 3.13.7` 下因缺少 `numpy` 失败

### 4.5 T3 边界澄清已完成

关键证据：

- `docs/03_hil_p4_boundary_audit.md`
  - 已把以下边界固定写清：
    - `software_hil_orchestrator`
    - `mock_backend`
    - `placeholder_real_board_backend`
    - `true_tflite_or_stub_export`
    - `true_tflite_or_stub_runtime`

### 4.6 T4 最小 software HIL 路径已恢复

关键证据：

- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
  - 显式固定 `hil.backend=mock`
  - 显式固定 `inference_service.mode=inproc`
  - 显式固定 `inference_service.backend=artifact_npz`
- `docs/P3_software_hil_bootstrap.md`
  - 固定恢复期最小 software HIL 复用命令
- `runs/hil_suite/hardware_hil_recovery_smoke_20260506_021326_3ae9f9176104/hil_summary.json`
  - `backend = mock`
  - `n_slow_updates_finished = 2`
  - `n_commits_applied = 2`
  - `artifact_path` 指向 `static_theta_v2` 下的 `.npz`

## 5. 疑似需要后续标记或治理的问题

1. 仓库中存在大量已跟踪的 `__pycache__/` 与 `.pyc`
2. `runs/` 目录中混有许多历史生成配置，容易被误读为源码一部分
3. `cnn_fpga/model/export.py` 同时支持真实 `.tflite` 与 stub 回退，后续文档必须持续严区分
4. `docs/reference/AI_coding_workflow.md` 当前为未跟踪文件，需要后续明确其纳入方式

## 6. 审计建议

建议按 `Repair` 路线推进，而不是 `Salvage`：

- 原因 1：核心算法、runtime、benchmark 资产都已经存在
- 原因 2：当前混乱主要集中在环境、治理和复验层
- 原因 3：直接弃用现仓库会丢失大量可核查实验资产

后续优先级建议：

1. 先做 `T5`，把缓存/生成物噪声的治理策略写清
2. 再做 `T6`，在 `T4` 最小路径基础上重新验收一条 mock-backed software HIL 最小路径
3. 最后才做 `T7`，重新验收一条 P4 最小路径
