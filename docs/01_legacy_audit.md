# Feasibility And Legacy Audit

## 1. 审计目的

本文件对应 `docs/reference/AI_coding_workflow.md` 中 `01_feasibility_report.md` 的角色，并保留 legacy audit 结果。它回答两个问题：

1. 项目是否值得继续
2. 继续时哪些能力是真实代码、哪些只是 `mock` / `stub` / `placeholder`

本轮审计默认只读核查，目标是回答：

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
- 截至 `2026-05-08`，第一轮恢复期收尾已完成，仓库可退出 `Phase 1: Recovery`，进入受控继续开发

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
  - `T11`
  - `T12`
  - `T13`
- 当前下一唯一任务建议为：
  - `T38: seed=20260429 single-seed trace-export probe, bounded unchanged-semantics rerun`

说明：

- `T3` 已把 `mock` / `stub` / `placeholder` 边界固定到 `docs/03_hil_p4_boundary_audit.md`
- `T4` 已把恢复期最小 software HIL 路径固定到 `docs/P3_software_hil_bootstrap.md`
- `T5` 已把缓存/生成物噪声治理口径固定到 `docs/06_repo_noise_governance.md`
- `T6` 已对最小 software HIL 路径完成新的复验，并再次确认 `mock + model_artifact + artifact_npz + inproc`
- `T7` 已对最小 P4 benchmark 路径完成新的复验，并把 recovery 级 P4 配置固定到 `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
- `T8` 已完成 gate review，并明确结论为 `Continue Repair`
- `T9` 已把 `P4 frozen baseline` recovery 证据从两种 mode 扩到四种正式 baseline，但仍限制在 `single-scenario + repeats=1`
- `T10` 已完成二次 gate review，并继续给出 `Continue Repair`
- `T11` 已在根目录补入 `requirements-recovery.txt`，并把它接入 `P0/P3/P4 recovery smoke` 的 bootstrap 文档
- `T12` 已把最小 software HIL recovery smoke 的随机源链路收口到逐字一致复验
- `T13` 已通过 recovery exit review，项目进入 `Phase 2: Controlled Development`
- `T24` 已完成 frozen-set formal software revalidation，并由 Captain 接受为 `PASS_WITH_WARNINGS`
- `T25` 已完成 result-boundary gate review，并确认 T24 只能作为 `mock-backed` software HIL formal software revalidation
- 后续 P3/P4 文档与复验结果都应沿用同一套 backend / artifact type 表述口径

## 3. 可行性判断

### 3.1 问题定义

项目目标是在 GKP 纠错中验证一种工程可落地的快慢回路架构：FPGA 侧 fast loop 执行低延迟线性控制，CPU/CNN 侧 slow loop 根据窗口统计更新控制参数。

### 3.2 可差异化点

1. 不是纯离线 decoder accuracy 项目，而是带 HIL / latency / commit / overflow 指标的闭环工程实验。
2. learned module 的角色被限制为 residual / calibration，而不是完全替代解码器。
3. 当前文档已经明确区分 mock-backed HIL、真实 `.tflite`、stub manifest 与真板 placeholder。

### 3.3 MVP 实验

当前 Phase 2 MVP 不是重新训练模型，而是增强 P4 evidence：

1. 审计 P4 frozen benchmark protocol。
2. 在 bounded matrix 下扩展 P4 多场景 smoke。
3. 经 gate review 决定是否进入更正式 benchmark。

### 3.4 主要风险

1. 把 recovery smoke 误写成正式 benchmark。
2. 把 `board_backend.py` placeholder 误写成真板完成。
3. 把 `.tflite.json` stub manifest 误写成真实 `.tflite` runtime。
4. 在未固定环境和 run matrix 前启动长跑。

### 3.5 Go / No-Go 判断

当前判断：`Go`，但只允许 bounded development。

理由：

1. Recovery exit review 已给出 `Allow`。
2. 最小 P3 software HIL path 已逐字一致复验。
3. P4 recovery smoke 已覆盖单场景四模式。
4. 仍有明确边界和风险文档，适合继续受控推进。

## 4. Feature Reality Matrix

| Feature | Claimed status | Evidence path | Verified? | Risk |
| --- | --- | --- | --- | --- |
| P0 full vs simplified 基线脚本 | 已存在最小对比脚本 | `benchmark/compare_full_vs_simplified_ler.py` | 部分验证 | 默认环境缺 `numpy`，当前无法在系统 Python 下直接运行 |
| P1 数据与训练链 | 已有数据集构建、训练、评估、量化入口 | `cnn_fpga/data/dataset_builder.py`, `cnn_fpga/model/train.py`, `cnn_fpga/model/evaluate.py`, `cnn_fpga/model/quantize.py` | 代码存在 | 依赖矩阵未确认，当前未复跑 |
| P2 行为级硬件仿真 | 已有硬件行为仿真与模式 benchmark | `cnn_fpga/benchmark/run_hardware_emulation.py`, `cnn_fpga/benchmark/run_p2_mode_benchmark.py` | 代码存在 | 当前环境未复验 |
| P3 软件 HIL | 已有 HIL 主流程、mock backend、驱动抽象、推理服务 | `cnn_fpga/benchmark/run_hil_suite.py`, `cnn_fpga/hwio/mock_fpga.py`, `cnn_fpga/runtime/inference_service.py`, `docs/03_hil_p4_boundary_audit.md`, `docs/P3_software_hil_bootstrap.md`, `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104/hil_summary.json`, `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104/hil_summary.json` | bounded 最小路径已逐字一致复验 | 结论仅限 `mock + model_artifact + artifact_npz + inproc`，不等于真板或 `.tflite` 路径已恢复 |
| P3 真板 HIL | 当前是 placeholder real-board backend | `cnn_fpga/hwio/board_backend.py`, `docs/CNN_FPGA_GKP_阶段结论.md`, `docs/03_hil_p4_boundary_audit.md` | 是 | 真实设备节点和地址表缺失，不能写成已完成 |
| P4 多场景 benchmark | 已有统一 benchmark 汇总脚本；最小 recovery path、frozen baseline 单场景全模式 smoke、以及 T24 frozen-set formal software revalidation 都已复验 | `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`, `docs/P4_benchmark_recovery_bootstrap.md`, `docs/P4_benchmark_formal_protocol.md`, `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json` | `mock-backed` software HIL formal revalidation 已完成 | 当前证据仍不是 `.tflite` runtime、真板验证或 paper-grade expanded benchmark；T28 已修复 teacher diagnostics missing-vs-zero writer 语义，但 R10 机制证据仍未完全修复 |
| teacher-representation 多版本分支 | 已有 v2-v9 配置与配对 benchmark 入口 | `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v*.yaml`, `cnn_fpga/benchmark/run_p4_teacher_representation_paired.py` | 代码存在 | 当前不应继续扩分支，先恢复可信度 |
| 真板 backend 语义 | placeholder/骨架状态 | `cnn_fpga/hwio/board_backend.py` | 是 | 若表述不严谨，极易误导项目完成度判断 |
| `.tflite` 真导出路径 | 代码支持真导出与 stub 回退双路径 | `cnn_fpga/model/export.py`, `cnn_fpga/runtime/inference_service.py`, `docs/03_hil_p4_boundary_audit.md` | 边界已审计 | 必须明确区分真实 `.tflite`、artifact 与 stub manifest |
| recovery 期根级依赖 manifest | 已新增 recovery-scoped 最小 manifest | `requirements-recovery.txt`, `docs/P0_smoke_bootstrap.md`, `docs/P3_software_hil_bootstrap.md`, `docs/P4_benchmark_recovery_bootstrap.md` | 是 | 只覆盖 `P0/P3/P4 recovery smoke`，不等于完整训练链、`.tflite` 或真板环境 |
| 根级治理文件 | 恢复前缺失 | 根目录与 `docs/` | 是 | 高，直接影响后续接力与审查 |

## 5. 关键证据

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

- 根目录现已新增：
  - `requirements-recovery.txt`
- 但它只覆盖：
  - `P0/P3/P4 recovery smoke`
- 根目录仍没有完整仓库环境文件：
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
- 这是 `T10` 时点的历史背景：
  - 当时 `T6` 的“可复验而非逐字确定性复现”观察仍成立
  - 对应 run 为 `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`
- `T9` 的 P4 证据虽然增强，但仍只覆盖：
  - `single-scenario`
  - `four-mode`
  - `repeats = 1`

### 4.10 T11 recovery 期最小依赖 manifest 已完成

关键证据：

- `docs/tasks/P0/T11_recovery_dependency_manifest.md`
  - 已把 `T11` 的作用域、验证命令与文档更新范围固定
- `requirements-recovery.txt`
  - 当前 manifest 仅包含：
    - `numpy`
    - `PyYAML`
  - 已明确覆盖：
    - `benchmark/compare_full_vs_simplified_ler.py --no-plot`
    - `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
    - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml ...`
  - 已明确不覆盖：
    - `torch` 训练链
    - `tensorflow` / `tflite-runtime`
    - `.tflite` export/runtime
    - `real_board` HIL backend
- `README.md`
  - 已改为显式引用 `requirements-recovery.txt`
- `docs/P0_smoke_bootstrap.md`、`docs/P3_software_hil_bootstrap.md`、`docs/P4_benchmark_recovery_bootstrap.md`
  - 已改为显式引用同一份 root manifest

### 4.11 T12 software HIL recovery smoke 确定性收口已完成

关键证据：

- `docs/tasks/P0/T12_software_hil_determinism_recovery.md`
  - 已把 `T12` 的作用域、验证命令与文档更新范围固定
- `physics/syndrome_measurement.py`
  - `RealisticSyndromeMeasurement` 已支持显式 `rng`
  - recovery 路径的测量噪声不再依赖全局 `np.random`
- `cnn_fpga/runtime/fast_loop_emulator.py`
  - 已把快回路误差 RNG 与测量噪声 RNG 分离
- `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104/hil_summary.json`
- `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104/hil_summary.json`
  - `final_ler = 0.454375`
  - `overflow_rate = 0.002`
- 两次 run 的文件级对比
  - `hil_summary.json` SHA256 一致
  - `hil_events.json` SHA256 一致

## 6. 疑似需要后续标记或治理的问题

1. `docs/06_repo_noise_governance.md` 已确认仓库中存在 `116` 个已跟踪缓存/字节码文件；物理 cleanup 仍未执行
2. `runs/` 当前有 `1841` 个已跟踪文件，只能暂作历史证据，不能自动视作当前事实来源
3. `artifacts/` 当前有 `110` 个已跟踪文件，且 `T4/T6/T7` 的最小复验路径仍依赖其中的 `static_theta_v2` `.npz`
4. `T12` 已把 bounded software HIL recovery smoke 收口到逐字一致复验，但这不外推到 `real_board` 或 `.tflite`
5. `T9` 已确认 P4 recovery smoke 可以把 frozen baseline 集扩到四种正式 baseline，但目前还不等于正式多场景 frozen benchmark 已恢复
6. `cnn_fpga/model/export.py` 同时支持真实 `.tflite` 与 stub 回退，后续文档必须持续严区分
7. `requirements-recovery.txt` 已经补齐 recovery 期最小 manifest，但完整训练链、`.tflite` 与真板路径仍没有统一根级环境文件

## 7. 审计建议

基于 `T13` recovery exit review，当前更合理的建议已从 `Repair` 更新为“受控 `Go`”：

- 原因 1：核心算法、runtime、benchmark 资产都已经存在
- 原因 2：当前最小 P3/P4 路径都已恢复，其中 software HIL bounded path 已做到逐字一致复验
- 原因 3：`T9` 已把 P4 recovery 证据扩到 `single-scenario + four-mode + repeats=1`，但仍不是正式多场景 frozen benchmark
- 原因 4：`T11/T12` 已分别收口 recovery 期 manifest 与 software HIL 确定性，剩余缺口已经从“阻止接力”降为“下一阶段的 bounded 开发任务”

后续优先级建议：

1. `T14` 至 `T30`、`T26` 以及 `T36` 已完成；当前下一唯一任务为 `T38`，只做 `seed=20260429` 的 single-seed trace-export probe
2. 继续保持 `mock` / `.tflite` / `real_board` 边界表述诚实
3. `T26` gate 结论为 `CONDITIONAL_GO`，且 `T30` 已把 statcalib 收紧为 interface-only separate comparator contract；后续仍不得把 statcalib 静默并入 T24 frozen benchmark set，不得扩展 formal benchmark、baseline/scenario、`.tflite` 或真板范围。
4. `T36` 已把 `seed=20260429` 诊断缩窄到 residual-amplitude / teacher-delta regime instability hypothesis；`T38` 不得启动新的 teacher-representation 长跑或新分支，只能在 unchanged semantics 下导出 per-window trace，验证 T36 留下的因果缺口。
