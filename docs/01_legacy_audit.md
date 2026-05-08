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
- 截至 `2026-05-08` 已完成：
  - `T1`
  - `T2`
  - `T3`
  - `T4`
  - `T5`
  - `T6`
  - `T7`
  - `T8`
  - `T9`
  - `T10`
- 当前下一唯一任务建议为：
  - `T11: 补一份恢复期最小依赖 manifest（优先覆盖 P0/P3/P4 recovery smoke）`

说明：

- `T3` 已把 `mock` / `stub` / `placeholder` 边界固定到 `docs/03_hil_p4_boundary_audit.md`
- `T4` 已把恢复期最小 software HIL 路径固定到 `docs/P3_software_hil_bootstrap.md`
- `T5` 已把缓存/生成物噪声治理口径固定到 `docs/06_repo_noise_governance.md`
- `T6` 已对最小 software HIL 路径完成新的复验，并再次确认 `mock + model_artifact + artifact_npz + inproc`
- `T7` 已对最小 P4 benchmark 路径完成新的复验，并把 recovery 级 P4 配置固定到 `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
- `T8` 已完成 gate review，并明确结论为 `Continue Repair`
- `T9` 已把 `P4 frozen baseline` recovery 证据从两种 mode 扩到四种正式 baseline，但仍限制在 `single-scenario + repeats=1`
- `T10` 已完成二次 gate review，并继续给出 `Continue Repair`
- 后续 P3/P4 文档与复验结果都应沿用同一套 backend / artifact type 表述口径

## 3. Feature Reality Matrix

| Feature | Claimed status | Evidence path | Verified? | Risk |
| --- | --- | --- | --- | --- |
| P0 full vs simplified 基线脚本 | 已存在最小对比脚本 | `benchmark/compare_full_vs_simplified_ler.py` | 部分验证 | 默认环境缺 `numpy`，当前无法在系统 Python 下直接运行 |
| P1 数据与训练链 | 已有数据集构建、训练、评估、量化入口 | `cnn_fpga/data/dataset_builder.py`, `cnn_fpga/model/train.py`, `cnn_fpga/model/evaluate.py`, `cnn_fpga/model/quantize.py` | 代码存在 | 依赖矩阵未确认，当前未复跑 |
| P2 行为级硬件仿真 | 已有硬件行为仿真与模式 benchmark | `cnn_fpga/benchmark/run_hardware_emulation.py`, `cnn_fpga/benchmark/run_p2_mode_benchmark.py` | 代码存在 | 当前环境未复验 |
| P3 软件 HIL | 已有 HIL 主流程、mock backend、驱动抽象、推理服务 | `cnn_fpga/benchmark/run_hil_suite.py`, `cnn_fpga/hwio/mock_fpga.py`, `cnn_fpga/runtime/inference_service.py`, `docs/03_hil_p4_boundary_audit.md`, `docs/P3_software_hil_bootstrap.md`, `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json` | 最小路径已二次复验 | control-plane 已稳定，但数值结果仍存在小幅 run-to-run 波动 |
| P3 真板 HIL | 当前是 placeholder real-board backend | `cnn_fpga/hwio/board_backend.py`, `docs/CNN_FPGA_GKP_阶段结论.md`, `docs/03_hil_p4_boundary_audit.md` | 是 | 真实设备节点和地址表缺失，不能写成已完成 |
| P4 多场景 benchmark | 已有统一 benchmark 汇总脚本，且最小 recovery path 与 frozen baseline 单场景全模式 smoke 都已复验 | `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`, `docs/P4_benchmark_recovery_bootstrap.md`, `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/summary.json`, `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/summary.json` | 两级 recovery 证据已复验 | 当前仍只覆盖 `single-scenario + four-mode + repeats=1` smoke，不等于正式多场景 frozen benchmark 已恢复 |
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

### 4.6 T4 / T6 最小 software HIL 路径已恢复并二次复验

关键证据：

- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
  - 显式固定 `hil.backend=mock`
  - 显式固定 `inference_service.mode=inproc`
  - 显式固定 `inference_service.backend=artifact_npz`
- `docs/P3_software_hil_bootstrap.md`
  - 固定恢复期最小 software HIL 复用命令
- `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`
  - `backend = mock`
  - `n_windows_ready = 2`
  - `n_slow_updates_finished = 2`
  - `n_commits_applied = 2`
  - `artifact_path` 指向 `static_theta_v2` 下的 `.npz`
  - `inference_service_mode = inproc`

### 4.7 T7 最小 P4 benchmark 路径已复验

关键证据：

- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
  - 显式固定 `hil.backend=mock`
  - 显式固定 `slow_loop.inference_service.mode=inproc`
  - 显式固定 `slow_loop.inference_service.backend=artifact_npz`
  - 显式固定 `slow_loop.model_artifact.path`
- `docs/P4_benchmark_recovery_bootstrap.md`
  - 固定恢复期最小 P4 benchmark 复用命令与过滤条件
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/summary.json`
  - `protocol_id = p4_hil_recovery_smoke_v1`
  - `scenario = static_bias_theta`
  - `modes = static_linear, cnn_fpga`
  - `seed_pairing = paired`
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/comparison.csv`
  - `static_linear final_ler = 1.00890625`
  - `cnn_fpga final_ler = 0.72109375`
  - `cnn_fpga artifact_path = ...static_theta_v2...npz`
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/static/cnnfpg/repeat_00/hil_summary.json`
  - `backend = mock`
  - `n_slow_updates_finished = 8`
  - `n_commits_applied = 8`
  - `inference_service_mode = inproc`
  - `artifact_path = ...static_theta_v2...npz`

### 4.8 T9 单场景全模式 frozen baseline smoke 已复验

关键证据：

- `docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`
  - 已把 `T9` 的目标、边界、验证命令与文档更新范围固定
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/summary.json`
  - `protocol_id = p4_hil_recovery_smoke_v1`
  - `scenario = static_bias_theta`
  - `modes = static_linear, window_variance, ekf, cnn_fpga`
  - `seed_pairing = paired`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/comparison.csv`
  - `Static Linear final_ler = 0.99575`
  - `Window Variance final_ler = 0.57440625`
  - `EKF final_ler = 0.6795`
  - `CNN-FPGA final_ler = 0.7248125`
  - scenario winner: `window_variance`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/static/static/repeat_00/hil_summary.json`
  - `backend = mock`
  - `artifact_path = null`
  - `inference_service_mode = inproc`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/static/window/repeat_00/hil_summary.json`
  - `backend = mock`
  - `artifact_path = null`
  - `inference_service_mode = inproc`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/static/ekf/repeat_00/hil_summary.json`
  - `backend = mock`
  - `artifact_path = null`
  - `inference_service_mode = inproc`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/static/cnnfpg/repeat_00/hil_summary.json`
  - `backend = mock`
  - `artifact_path = ...static_theta_v2...npz`
  - `inference_service_mode = inproc`

### 4.9 T10 gate review 已完成

关键证据：

- `docs/review/T10_gate_review.md`
  - 已明确给出 verdict：
    - `Continue Repair`
  - 已明确下一唯一任务：
    - `T11: 补一份恢复期最小依赖 manifest（优先覆盖 P0/P3/P4 recovery smoke）`
- 根目录当前仍无：
  - `requirements.txt`
  - `pyproject.toml`
  - `environment.yml`
- `T6` 的“可复验而非逐字确定性复现”观察仍成立：
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`
- `T9` 的 P4 证据虽然增强，但仍只覆盖：
  - `single-scenario`
  - `four-mode`
  - `repeats = 1`

## 5. 疑似需要后续标记或治理的问题

1. `docs/06_repo_noise_governance.md` 已确认仓库中存在 `116` 个已跟踪缓存/字节码文件；物理 cleanup 仍未执行
2. `runs/` 当前有 `1841` 个已跟踪文件，只能暂作历史证据，不能自动视作当前事实来源
3. `artifacts/` 当前有 `110` 个已跟踪文件，且 `T4/T6/T7` 的最小复验路径仍依赖其中的 `static_theta_v2` `.npz`
4. `T6` 已确认 software HIL control-plane 复验成功，但 `final_ler` 与 `overflow_rate` 在两次 run 之间存在小幅差异，说明当前更接近“可复验”而非“逐字确定性复现”
5. `T9` 已确认 P4 recovery smoke 可以把 frozen baseline 集扩到四种正式 baseline，但目前还不等于正式多场景 frozen benchmark 已恢复
6. `cnn_fpga/model/export.py` 同时支持真实 `.tflite` 与 stub 回退，后续文档必须持续严区分

## 6. 审计建议

建议继续按 `Repair` 路线推进，当前不直接宣布 `Go`：

- 原因 1：核心算法、runtime、benchmark 资产都已经存在
- 原因 2：当前最小 P3/P4 路径都已恢复到“可复验”状态
- 原因 3：`T9` 已把 P4 recovery 证据扩到 `single-scenario + four-mode + repeats=1`，但仍不是正式多场景 frozen benchmark
- 原因 4：`T10` gate review 已再次确认依赖 manifest 与确定性复现缺口仍未收口

后续优先级建议：

1. 先做 `T11`，补一份 recovery 期最小依赖 manifest，优先覆盖 `P0/P3/P4 recovery smoke`
2. 再基于 `T11` 的结果，决定是继续收口“确定性复现”，还是再补更强的 P4 多场景证据
3. 其后再考虑单开 cleanup 任务处理 repo noise 的物理移除
