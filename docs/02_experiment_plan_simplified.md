# DriftAdaptiveQEC — 项目集成文档与后续开发计划

**文档角色：** 本文件是项目唯一的核心上下文文档。任何新 Captain AI、Worker AI 或 Reviewer AI 在参与本项目前必须首先阅读本文件。它合并了此前分散在多个文档中的项目背景、阶段历史、实验结果、工程边界、论文路线和开发计划。

**最后更新：** 2026-05-08
**当前阶段：** Phase 2: Controlled Development
**当前唯一任务：** 待定义
**决策状态：** Go（受控继续开发）

---

## 目录

1. [Quick Start（Captain AI 5 分钟速览）](#1-quick-startcaptain-ai-5-分钟速览)
2. [项目身份与核心问题](#2-项目身份与核心问题)
3. [代码仓库结构](#3-代码仓库结构)
4. [阶段历史与关键结果（P0 → P4）](#4-阶段历史与关键结果p0--p4)
5. [恢复期 Phase 0–1 详细记录（T1–T13）](#5-恢复期-phase-01-详细记录t1t13)
6. [当前环境与依赖矩阵](#6-当前环境与依赖矩阵)
7. [工程真实性边界表](#7-工程真实性边界表)
8. [Teacher-Representation 分支谱系（v2–v9）](#8-teacher-representation-分支谱系v2v9)
9. [稳定结论清单](#9-稳定结论清单)
10. [论文撰写与投稿路线](#10-论文撰写与投稿路线)
11. [后续开发优先级与候选任务包](#11-后续开发优先级与候选任务包)
12. [治理规范：AI Coding 工作流](#12-治理规范ai-coding-工作流)
13. [风险与待解决问题](#13-风险与待解决问题)
14. [附录：文件路径索引与常用命令](#14-附录文件路径索引与常用命令)

---

## 1. Quick Start（Captain AI 5 分钟速览）

### 1.1 这项目是做什么的

面向**漂移自适应 GKP 量子纠错**的 **CNN-FPGA 双回路解码系统**。

- **快回路（FPGA 侧）**：每周期 5μs 级，执行线性解码 `Δ = K @ s + b`，同时累积 32×32 syndrome 直方图
- **慢回路（ARM/CNN 侧）**：每 10–100ms 级，从直方图窗口估计噪声漂移参数，更新快回路解码器参数 `(K, b)`
- **核心创新**：不让 CNN 直接替代经典估计器全部工作，而是用"经典 teacher + CNN residual-b 修正"的混合方案

### 1.2 项目做到哪了

| 阶段 | 状态 | 关键事实 |
|------|------|----------|
| P0 物理基线 | 完成 | `full_qec` vs `simplified` LER gap 确认存在 |
| P1 CNN 训练 | 完成 | 浮点 R² > 0.99，int8 退化 < 1% |
| P2 行为仿真 | 完成 | 公平基线下的自适应闭环已跑通 |
| P3 软件 HIL | 完成 | `mock + artifact_npz + inproc` 路径逐字一致复验 |
| P3 真板 HIL | 未完成 | `board_backend.py` 仍是 placeholder |
| P4 多场景 benchmark | 部分完成 | 正式强 baseline 对比已完成；recovery 级 smoke 已复验 |

### 1.3 当前最重要的 5 个状态事实

1. **环境**：恢复期 smoke 推荐 `C:\ProgramData\anaconda3\python.exe`；训练候选 `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
2. **依赖**：根目录有 `requirements-recovery.txt`（仅含 `numpy + PyYAML`，只覆盖 P0/P3/P4 recovery smoke）
3. **确定性**：`mock + model_artifact + artifact_npz + inproc` 路径已做到逐字一致复验
4. **主线模型**：当前正式方案是 `Hybrid Residual-B`（teacher-guided residual-b），最强候选是 `Gated v5`
5. **真板**：`board_backend.py` 仍是 placeholder，不要写"真板 HIL 已通过"

### 1.4 不该做什么

- 继续微调 `gated v10/v11` 这类 gate/clip/scale 超参——边际收益已极低
- 做大规模"删/不删 teacher params"长跑——结论已充分
- 把 `PB Bound / PB ST` 扩写成论文主线——仅作辅助支线
- 把 bounded recovery smoke 扩写成"真板已恢复"或"正式多场景 benchmark 已恢复"

### 1.5 最值得做什么

1. **第一优先**：做 `seed=20260429` 的失败机理诊断（为什么 Gated v5 在该 seed 上不稳）
2. **第二优先**：落地 paper-inspired 分支（在 Gated v5 基础上加统计聚合 + 闭环一致 loss）
3. **第三优先**：补 P4 多场景正式证据，或补训练/.tflite 独立 manifest

---

## 2. 项目身份与核心问题

### 2.1 项目名称与定位

- **仓库名**：`DriftAdaptiveQEC`
- **正式项目名**：CNN-FPGA 协同 GKP 漂移自适应解码系统
- **定位**：工程系统型量子纠错研究项目，兼具方法机制贡献
- **当前分支**：`main`
- **Git 用户**：AscendFoam

### 2.2 核心物理问题

在连续变量 GKP 编码的量子纠错中，噪声参数会随时间漂移（sigma 幅值变化、位移均值偏置、协方差旋转角变化）。固定参数的线性解码器在漂移场景下会逐渐失配，导致逻辑错误率（LER）上升。

解决方案是**双回路在线自适应解码**：
- 快回路执行超低时延的确定性线性解码
- 慢回路周期性从 syndrome 统计中估计漂移，更新快回路参数

### 2.3 核心工程问题

实时硬件约束下，慢回路不能是任意复杂的模型。需要通过软件 HIL 验证：模型精度、延迟预算、参数切换原子性、量化部署一致性。

### 2.4 核心方法论贡献

当前主方案 `Hybrid Residual-B` 的方法论贡献是：

> 在实时硬件约束下，保留稳定的经典 teacher（如 Window Variance），让轻量 CNN 只学习对运行时控制偏置 `b` 的残差修正，而不是让 CNN 独立承担全部漂移参数估计。

这比"CNN 直接回归全部绝对物理参数"更有效、更稳定、更可解释。

---

## 3. 代码仓库结构

### 3.1 顶层目录

```text
DriftAdaptiveQEC/
├── AGENTS.md                         # AI agent 治理文件
├── CLAUDE.md                         # Claude Code 审查指令
├── README.md                         # 项目入口说明（引用 requirements-recovery.txt）
├── requirements-recovery.txt         # 恢复期最小依赖 manifest（numpy + PyYAML）
├── physics/                          # 现有物理仿真模块
├── cnn_fpga/                         # 工程主模块（数据/模型/运行时/HIL/benchmark）
├── fpga/                             # FPGA RTL/HLS 实现目录（规划中）
├── benchmark/                        # 最小 P0 对比脚本
├── docs/                             # 项目文档
├── runs/                             # 历史运行输出（大量已跟踪文件）
└── artifacts/                        # 模型/数据集/报告产物
```

### 3.2 physics/ 模块

| 文件 | 功能 | 状态 |
|------|------|------|
| `gkp_state.py` | GKP 态定义 | 可用 |
| `noise_channels.py` | 玻色量子噪声通道（光子损失/热噪声/相位噪声/位移噪声） | 已实现，但未接入 P2/P3/P4 主线 |
| `syndrome_measurement.py` | 综合征测量模型（含有限压缩噪声、测量效率、ancilla 错误、shot noise） | T12 已支持显式 rng 注入 |
| `error_correction.py` | 线性解码参数映射 | 可用 |
| `logical_tracking.py` | 逻辑错误判定（GKP 决策边界 `±sqrt(2pi)/2`） | 有效模型口径，非完整电路级 |

### 3.3 cnn_fpga/ 模块（工程核心）

#### cnn_fpga/data/ — 数据流水线

| 文件 | 功能 |
|------|------|
| `schema.py` | 数据结构定义（syndrome/histogram/labels） |
| `dataset_builder.py` | 数据集生成（含各向异性高斯、椭圆高斯等模式） |
| `histogram.py` | 直方图映射与归一化 |
| `split.py` | train/val/test 划分 |

#### cnn_fpga/model/ — 模型与训练

| 文件 | 功能 |
|------|------|
| `tiny_cnn.py` | CNN 模型定义（手写 NumPy，当前主线；PyTorch 后端可选） |
| `train.py` | 训练入口 |
| `evaluate.py` | 回归评估（MSE/MAE/R²） |
| `evaluate_tflite.py` | 独立 TFLite 精度评估 |
| `quantize.py` | QAT/PTQ + int8 导出 |
| `export.py` | ONNX/TFLite 导出（真导出 + stub 回退双路径） |
| `validate_export.py` | artifact 与 TFLite 一致性验收 |

#### cnn_fpga/decoder/ — 解码器

| 文件 | 功能 |
|------|------|
| `param_mapper.py` | (σ, μ_q, μ_p, θ) → (K, b) 协方差一致映射 |
| `linear_runtime.py` | 快回路等价软件实现（固定点运算 + 饱和处理） |
| `mwpm_stub.py` | MWPM 接口占位 |
| `ekf_baseline.py` | EKF 基线实现 |

#### cnn_fpga/runtime/ — 双回路运行时

| 文件 | 功能 |
|------|------|
| `fast_loop_emulator.py` | 快回路仿真（含周期约束、固定点、overflow 三类拆分统计） |
| `slow_loop_runtime.py` | 慢回路 CNN 推理与参数更新 |
| `scheduler.py` | 双回路调度控制 |
| `param_bank.py` | 双缓冲参数切换 |
| `latency_injector.py` | DMA/AXI/推理延迟注入（当前基于独立高斯/常数抽样） |
| `inference_service.py` | 进程内/子进程推理服务（含 tflite_service / tflite_stub_service 双路径） |
| `inference_worker.py` | 独立推理 worker |
| `feature_builder.py` | 慢回路输入特征构建（containing histogram delta, teacher prediction, teacher params, teacher deltas） |
| `noise_bridge.py` | 物理噪声桥接（已移除重型未使用导入以解决 Windows 启动阻塞） |

#### cnn_fpga/hwio/ — 硬件 I/O 抽象

| 文件 | 功能 | 边界标签 |
|------|------|----------|
| `axi_map.py` | AXI-Lite 寄存器映射定义 | — |
| `dma_client.py` | DMA 读写接口 | — |
| `mock_fpga.py` | 无板卡时的 event-driven 行为仿真后端 | `mock_backend` |
| `fpga_driver.py` | HIL 驱动封装（含 board/real 后端为 future integration） | — |
| `board_backend.py` | 真板卡后端骨架 | `placeholder_real_board_backend` |

#### cnn_fpga/benchmark/ — 实验入口

| 文件 | 功能 |
|------|------|
| `run_hil_suite.py` | P3 软件 HIL 会话入口（通过 `hil.backend` 选择 backend） |
| `run_p4_multiscenario_benchmark.py` | P4 多场景 benchmark（核心：调用 `run_hil_session()`） |
| `run_p2_mode_benchmark.py` | P2 行为仿真模式对比 |
| `run_p3_param_sweep.py` | P3 参数清扫 |
| `run_p3_histogram_tuning.py` | P3 直方图范围调参 |
| `run_p4_teacher_representation_paired.py` | P4 teacher-representation paired benchmark |
| `run_p4_hybrid_vs_ukf_ablation.py` | P4 Hybrid vs UKF ablation |
| `run_p4_no_teacher_params_stability.py` | No TeacherParams 稳定性复查 |
| `run_p4_teacher_params_reencoding_controlled.py` | Teacher params 重编码对照 |
| `summarize_p4_features_ablation.py` | P4 features 消融汇总 |
| `run_drift_suite.py` | P0 漂移场景对比 |
| `run_hardware_emulation.py` | P2 硬件行为仿真 |
| `run_hil_mode_benchmark.py` | P3 HIL 模式 benchmark |

### 3.4 fpga/ 目录（规划中）

当前存在目录结构但内容为初始/RTL 级规划。包含 `rtl/`（解码器/直方图/参数bank Verilog）、`hls/`、`sim/`、`constraints/`、`vivado/`。

### 3.5 仓库噪声情况

- `.gitignore` 已忽略 `__pycache__/`、`runs/`、`artifacts/`
- 但 Git 历史中仍有大量已跟踪噪声：116 个缓存/字节码文件、1841 个 `runs/` 文件、110 个 `artifacts/` 文件
- `T5` 已固定治理口径：恢复期先治理不清理，后续需单开有界 cleanup 任务
- 当前工作区可见 9 个 `__pycache__` 目录，133 个 `.pyc` 文件

---

## 4. 阶段历史与关键结果（P0 → P4）

### 4.1 P0：物理仿真基线确认

**目标**：确认 `full_qec` 与 `simplified` 物理模型的 LER 差异。

**运行配置**：
- 场景：`linear_low / step_mid / sinusoidal / random_walk`
- 每场景 `n_rounds = 2000`，`repeats = 10`

**关键结果**：
| 场景 | full_qec LER | simplified LER | gap |
|------|-------------|----------------|-----|
| `linear_low` | 0.4237 | 0.0205 | 0.4032 |
| `step_mid` | 0.42665 | 0.01855 | 0.40810 |
| `sinusoidal` | 0.42530 | 0.02405 | 0.40125 |
| `random_walk` | 0.41200 | 0.01565 | 0.39635 |

**意义**：`simplified` 模型明显低估逻辑错误率。后续工程验证不能以简化物理模型为可信主结论来源。

**配置与入口**：
- `benchmark/compare_full_vs_simplified_ler.py`
- `cnn_fpga/config/experiment_drift.yaml`

### 4.2 P1：CNN 数据与训练闭环

**目标**：完成从 syndrome 直方图到 `(σ, μ_q, μ_p, θ)` 的回归模型。

**关键修正**：
1. 数据生成从"各向同性高斯"改为"各向异性高斯"（`sigma_ratio_p = 0.55`），使 `theta_deg` 可辨识
2. 训练目标中对 `theta_deg` 引入更高的损失权重

**正式模型**：`static_theta_v2` Tiny-CNN（`artifacts/models/static_theta_v2/`）

**浮点模型 test 集指标**：
| 标签 | R² |
|------|-----|
| `sigma` | 0.997613 |
| `mu_q` | 0.996473 |
| `mu_p` | 0.998459 |
| `theta_deg` | 0.984862 |
| **平均** | **0.994352** |

**int8 模型 test 集指标**：
| 标签 | R² |
|------|-----|
| `sigma` | 0.997730 |
| `mu_q` | 0.996324 |
| `mu_p` | 0.998160 |
| `theta_deg` | 0.984634 |
| **平均** | **0.994212** |

**验收结论**：P1 通过。所有标签 R² 均超过阈值，int8 退化 < 1%。

**关键 artifact**：
- 浮点模型：`tiny_cnn_20260319_151717_b87c6c227b57.npz`
- int8 模型：`tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz`
- 浮点 .tflite：`tiny_cnn_20260319_151717_b87c6c227b57_tflite_20260328_012736.tflite`
- int8 .tflite：`tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_tflite_20260328_012736.tflite`
- 配置：`cnn_fpga/config/experiment_static_theta_v2.yaml`

### 4.3 P2：硬件行为仿真（无板卡）

**目标**：在软件里模拟硬件行为（固定点、延迟、参数切换），公平基线对比。

**修正**：
1. `ParamMapper` 从 `K = gain * rotation, b = -mu` 改为协方差驱动的对称增益矩阵 + 偏置 `b = (I-K)mu`
2. 新增公平基线：`fixed_baseline`（不自适应）和 `oracle_delayed`（延迟一窗真值）

**正式 P2 模式**：`fixed_baseline / oracle_delayed / model_artifact / int8_artifact`

**关键结果（model_artifact vs fixed_baseline）**：
| 场景 | fixed_baseline | model_artifact | Δ |
|------|---------------|----------------|-----|
| `linear_med` | 0.816067 | 0.696206 | -0.119861 |
| `step_large` | 0.937822 | 0.731906 | -0.205917 |
| `sinusoidal_mid` | 1.019033 | 0.759289 | -0.259744 |

float 与 int8 差异 < 0.006，所有模式 `fast_cycle_violation_rate = 0`、`slow_update_violation_rate = 0`。

**验收结论**：P2 通过。自适应模型在公平基线下确实改善 LER；量化可用；调度可用。

**运行目录**：`runs/p2_mode_benchmark/hardware_emulation_v1_20260319_160130_20670d1c0d1f`

### 4.4 P3：软件 HIL 主线与真实 .tflite

**目标**：端到端软件 HIL 链路验证，真实 .tflite 导出/评测，overflow 来源拆分。

**软件 HIL 路径打通**（2026-03-28）：
- `fixed_baseline_mock` vs `float_artifact_mock` vs `int8_artifact_mock` vs `real_board`
- real_board 跳过（`skipped_unavailable`，缺 `/dev/uio0,/dev/uio1`）

**Overflow 来源拆分**（2026-03-31）：
三类 overflow 诊断：`histogram_input_saturation_rate / correction_saturation_rate / aggressive_param_rate`。
主导来源一致为 `histogram_input`（其他两项均为 0），说明瓶颈不在参数映射或控制激进。

**输入范围调参**（2026-03-31）：
- 新默认值：`syndrome_limit = 1.441311257912825`，`histogram_range_limit = 1.8799712059732503`，`sigma_measurement = 0.03`
- LER 从 1.069503 降到 1.046678
- `histogram_input_saturation_rate` 从 0.387092 降到 0.022181

**真实 .tflite 独立验收**（2026-03-29/30）：
- float .tflite: MSE = 0.292359, R² = 0.994359, export consistency ok
- int8 .tflite: MSE = 0.297316, R² = 0.994192, export consistency ok

**验收结论**：P3 软件 HIL 通过。P3 真板 HIL 未完成。

### 4.5 P4：多基线统计对比与正式 benchmark

#### 4.5.1 首轮正式结果（2026-04-01）

方案从"直接回归绝对参数"切换到 `teacher + residual-b`。

跨 4 场景平均 LER（首轮）：
| 模式 | 平均 LER |
|------|----------|
| Hybrid Residual-B | **0.850799** |
| Constant Residual-Mu | 0.855549 |
| EKF | 0.855779 |
| Window Variance | 0.857016 |
| CNN-FPGA（旧） | 0.954315 |

Hybrid Residual-B 在 4 个场景中全部最优。

#### 4.5.2 长配置复验（2026-04-02）

输入范围提升为新默认值：
- `syndrome_limit = 1.566643`
- `histogram_range_limit = 2.255965971`

更长配置下 4 场景平均 LER：
| 模式 | 平均 LER |
|------|----------|
| Hybrid Residual-B | **0.798807** |
| Constant Residual-Mu | 0.826193 |
| EKF | 0.828108 |

优势扩大到 0.023~0.032。`histogram_input_saturation_rate` 降到 ~0.00254~0.00258。

#### 4.5.3 强 baseline 扩展（2026-04-03）

加入 `UKF`（修正后——保留 full covariance + 对称化/正定稳定化）和 `RLS Residual-B`。

4 场景平均 LER：
| 模式 | 平均 LER |
|------|----------|
| Hybrid Residual-B | **0.798332** |
| UKF | 0.817974 |
| Constant Residual-Mu | 0.825719 |
| RLS Residual-B | 0.827908 |
| EKF | 0.828369 |

**关键结论**：UKF 是当前最强经典 baseline，但 Hybrid Residual-B 仍保持 ~0.019642 优势。优势不是来自更激进控制（`correction_saturation_rate = 0, aggressive_param_rate = 0`）。

#### 4.5.4 Features 正式 ablation（2026-04-04/05）

跨 4 场景平均 LER：
| 变体 | 平均 LER |
|------|----------|
| Hybrid Full | 0.798355 |
| Hybrid No HistDelta | 0.826422 |
| Hybrid No TeacherPred | 0.807556 |
| Hybrid No TeacherParams | 0.749436 |
| Hybrid No TeacherDelta | 0.800473 |
| UKF | 0.818081 |

**关键发现**：
- `histogram delta` 是关键通道（去掉后退化到 vs UKF 以下）
- `teacher prediction` 有价值但不是唯一关键
- `No TeacherParams` 出现异常优势（0.749436），触发后续机制复查

#### 4.5.5 Physical Bridge 支线（2026-04-26）

`PB Bound` 和 `PB ST` 作为场景特定辅助支线：
- PB Bound 在 `periodic_drift / step_sigma_theta` 动态场景较好
- PB ST 仅在 `periodic_drift` 平滑动态场景更好
- 两者均不能替代 Full 为统一主线

**结论**：`physical_bridge` 仅作为场景特定机制分析支线，不进入主论文排序。

---

## 5. 恢复期 Phase 0–1 详细记录（T1–T13）

### 5.1 背景

项目在积累了较完整的 P0-P4 代码路径与实验结果后，仓库缺少统一治理层。默认环境（`python 3.13.7`）缺 `numpy`，最小 benchmark 无法运行。2026-05-05 启动恢复期，按 `docs/reference/AI_coding_workflow.md` 第 4 节执行。

### 5.2 Phase 0: Stabilization（T0–T5）

| 任务 | 内容 | 产出 |
|------|------|------|
| T0 | 冻结 legacy 状态，只读审计 | `docs/00_project_snapshot.md` |
| T1 | 确认依赖矩阵与最小入口 | 固定解释器：`C:\ProgramData\anaconda3\python.exe` |
| T2 | 跑通最小 P0 smoke benchmark | `docs/P0_smoke_bootstrap.md`，首次 P0 smoke 成功 |
| T3 | 审计 HIL/P4 链路 mock/stub/placeholder 边界 | `docs/03_hil_p4_boundary_audit.md` |
| T4 | 补软件 HIL 最小 bootstrap/smoke test | `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`，`docs/P3_software_hil_bootstrap.md` |
| T5 | 清点并治理仓库噪声 | `docs/06_repo_noise_governance.md`，固定分类策略，不做破坏性清理 |

### 5.3 Phase 1: Recovery（T6–T13）

| 任务 | 内容 | 关键产出 |
|------|------|----------|
| T6 | 重新验收 software HIL 最小路径 | 二次复验：`runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`。control-plane 字段一致，LER/overflow 有小幅 run-to-run 差异 |
| T7 | 重新验收 P4 benchmark 最小路径 | `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`。复验 `single-scenario + two-mode + repeats=1`：CNN-FPGA LER=0.721，Static Linear LER=1.009 |
| T8 | 基于 T6+T7 做 Go/Repair gate review | 决定 Continue Repair（T7 只覆盖 single-scenario+two-mode，缺 manifest） |
| T9 | 扩 P4 frozen baseline 到四模式单场景 | 复验 `static_bias_theta + four-mode + repeats=1`：WV LER=0.574, EKF=0.680, CNN-FPGA=0.725, SL=0.996 |
| T10 | 基于 T8+T9 二次 gate review | 继续 Repair（仍缺 manifest，P4 非正式，HIL 非确定性） |
| T11 | 补 recovery 期最小依赖 manifest | `requirements-recovery.txt`（numpy + PyYAML）。只覆盖 P0/P3/P4 recovery smoke |
| T12 | 收敛 software HIL 随机源与确定性 | 修复 `physics/syndrome_measurement.py` 显式 rng 注入，分离快回路/测量噪声 RNG。两次连续复验的 `hil_summary.json` 和 `hil_events.json` SHA256 完全一致 |
| T13 | Recovery exit review 并收尾 | `docs/review/T13_recovery_exit_review.md` verdict = **Allow**。项目从 Repair 切换为 Go，退出 Phase 1，进入 Phase 2 |

### 5.4 恢复期核心成果

1. **确定性的 bounded software HIL 路径**：`mock + model_artifact + artifact_npz + inproc`，逐字一致复验
2. **边界清晰化**：`docs/03_hil_p4_boundary_audit.md` 固定了 6 种边界标签
3. **依赖 manifest**：`requirements-recovery.txt`，作用域诚实
4. **治理文件**：`04_task_board.md`、`05_decision_log.md`、`07_handoff.md`、`08_risks_and_open_questions.md` 均已建立
5. **三条 recovery bootstrap 文档**：`docs/P0_smoke_bootstrap.md`、`docs/P3_software_hil_bootstrap.md`、`docs/P4_benchmark_recovery_bootstrap.md`

### 5.5 恢复期的边界纪律

所有恢复期的操作均遵循以下纪律：
- 不改 benchmark 主线语义
- 不把 `board_backend.py` 的 placeholder 写成真板完成
- 不借恢复任务扩写 .tflite、teacher-representation 或真板功能
- 不做 `runs/`、`artifacts/`、`__pycache__/` 的大规模清理

---

## 6. 当前环境与依赖矩阵

### 6.1 可用解释器

| 路径 | 可用包 | 角色 |
|------|--------|------|
| `C:\ProgramData\anaconda3\python.exe` | `numpy + PyYAML` | P0/P3/P4 recovery smoke |
| `C:\ProgramData\anaconda3\envs\DLEnv\python.exe` | `numpy + PyYAML + torch` | 训练候选 |
| `C:\Python313\python.exe` | 无 numpy | **不可用于项目** |

### 6.2 依赖矩阵

**Recovery smoke（`requirements-recovery.txt`）**：
- `numpy` — 物理仿真、模型推理、HIL 模拟
- `PyYAML` — 配置文件解析
- 覆盖范围：`benchmark/compare_full_vs_simplified_ler.py --no-plot`、`run_hil_suite`、`run_p4_multiscenario_benchmark`
- 不覆盖：`matplotlib`（去掉 `--no-plot` 时触发）、`torch`（训练链）、`tensorflow`/`tflite-runtime`（.tflite 路径）、`real_board` HIL backend

**训练链**：依赖 `DLEnv` 环境，含 `torch`。当前 Tiny-CNN 训练支持 NumPy 和 PyTorch/CUDA 双后端。

**.tflite 路径**：历史文档引用的 `.venvs/tf311` 在当前 Windows 工作区不存在。`.tflite` 独立验收在当前环境下不可运行。

### 6.3 最小运行命令

```powershell
# P0 smoke（~1 分钟）
& 'C:\ProgramData\anaconda3\python.exe' benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test

# P3 software HIL recovery smoke（~秒级）
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml

# P4 recovery smoke（~分钟级）
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds
```

### 6.4 恢复期 Recovery Smoke 口径

所有 recovery smoke 的固定口径为：
- `hil.backend = mock`
- `slow_loop.mode = model_artifact`
- `inference_service.mode = inproc`
- `inference_service.backend = artifact_npz`
- `model_artifact.path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`

---

## 7. 工程真实性边界表

以下边界标签是 T3（`docs/03_hil_p4_boundary_audit.md`）固定下来的。后续所有文档、报告、论文必须使用这些标签。

### 7.1 主边界矩阵

| 组件 | 边界标签 | 当前真实状态 | 禁止表述 |
|------|----------|-------------|----------|
| `run_hil_suite.py` + `hil.backend=mock` | `software_hil_orchestrator` | 真实可运行的软件 HIL orchestration。通过 `hil.backend` 选择 backend，产出 `hil_events.json` / `hil_summary.json` | — |
| `mock_fpga.py` | `mock_backend` | event-driven FPGA 行为仿真。维护 DMA/param-bank 语义，`metadata={"backend": "mock_fpga"}` | "真板运行结果" |
| `board_backend.py` | `placeholder_real_board_backend` | placeholder 真板骨架。`schedule_commit()` 返回大量 `None`，`step()` 返回空事件 | "真板 HIL 已完成"、"真板 backend 已验收" |
| `run_p4_multiscenario_benchmark.py` | `p4_wrapper_over_hil` | 直接调用 `run_hil_session()`，不绕开 HIL backend 边界。P4 的真实性继承自 HIL 链路 | "P4 有独立于 HIL 的更高真实性" |
| `export.py` (TFLite 导出) | `true_tflite_or_stub_export` | 优先真 `.tflite` 导出，失败回退 `tflite_stub_v1` | "TFLite 导出已完成"（不声明是真/stub） |
| `inference_service.py` (TFLite 推理) | `true_tflite_or_stub_runtime` | stub 路径 `source="tflite_stub_service"`；真路径 `source="tflite_service"` | 不区分 source 的"TFLite 已部署" |

### 7.2 推荐表述规则

1. 可以说："P3 software HIL 主链存在"
   - 必须同时标注：`hil.backend` 和 inference artifact type
2. 不能说："real-board HIL 已完成"
   - 除非有独立真板证据覆盖 `board_backend.py` 的当前占位状态
3. 不能说："P4 benchmark 是独立于 HIL 的更真实执行链"
   - 它只是同一 HIL 会话的批量包装
4. 不能说："TFLite 已部署"
   - 必须区分 `tflite_service` vs `tflite_stub_service`

### 7.3 工程仿真补强缺口总览

以下缺口不影响当前结论成立，但补上后将增强论文说服力（引自 `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md`）：

| 方向 | 当前状态 | 优先级 |
|------|----------|--------|
| 多类物理噪声接入（`noise_channels.py`） | 已实现但未接入 P2/P3/P4 主线 | 高 |
| 延迟模型负载耦合（`latency_injector.py`） | 独立高斯/常数抽样，无负载条件偏置 | 高 |
| 慢回路故障模型细化 | 独立伯努利注错，无状态机/恢复/重试 | 高 |
| 快回路控制路径 bit-accurate 固定点 | 接近硬件近似，非逐级位宽精确 | 高 |
| syndrome 读出链（ADC/AFE 模型） | 统计测量模型，非完整模拟/数字读出链 | 中高 |
| 逻辑错误定义扩展 | 有效模型口径，非完整电路级容错 | 中 |
| 板级 I/O 语义补全 | `board_backend.py` 仍是 placeholder | 中（条件性） |

---

## 8. Teacher-Representation 分支谱系（v2–v9）

### 8.1 分支演进总览

Teacher-representation 分支的起点是发现 `No TeacherParams` 在离线训练中表现异常好（2026-04-05），但 formal HIL 中结论随 seed 翻转。此后不再讨论"删不删 teacher params"，转为研究"teacher 信息应如何编码进闭环模型"。

### 8.2 Gated v2/v3/v4（早期试探）

- **v2**：首版 scalar-branch + gated 注入。单轮动态场景有正信号，但不完整
- **v3**：调整 scalar 归一化与 gate 结构
- **v4**：仅小幅优于 Full。`aggressive_param_rate` 偏高

### 8.3 Gated v5（当前最强候选）

**设计**：
- 保留 histogram 主干 + scalar branch + gated 注入
- 仅保留 4 个与 `residual-b` 最直接相关的 teacher 标量：
  - `teacher_b_q`
  - `teacher_b_p`
  - `teacher_delta_b_q`
  - `teacher_delta_b_p`
- 不恢复整包 teacher params broadcast

**三 seed / 四场景 paired benchmark 汇总（2026-04-27）**：
| Seed | Full LER | Gated v5 LER | Gap |
|------|----------|-------------|-----|
| 20260427 | 0.779861 | 0.547688 | -0.232173 |
| 20260428 | 0.798706 | 0.710131 | -0.088574 |
| 20260429 | 0.688990 | 0.674559 | -0.014432 |

按场景跨 seed 汇总（3 seed）：
| 场景 | Full | Gated v5 |
|------|------|----------|
| static_bias_theta | 0.751062 | 0.637209 |
| linear_ramp | 0.764402 | 0.631810 |
| step_sigma_theta | 0.759205 | 0.638351 |
| periodic_drift | 0.748741 | 0.669133 |

**chunked pair 复验（2026-04-28）**：3 seed × 4 scenario × 2 repeats，coverage=100%，0 hil_errors。
- 跨 seed 均值：`Full = 0.758829`，`Gated v5 = 0.618195`，gap = -0.140634
- Gated v5 在 12 个 seed-scenario 中赢 9 个

**定位**：当前最强 teacher-representation 候选。尚未完全替代 Full（seed=20260429 持平/略差），但已足够说明"少数关键 teacher 标量 + 低维 gated 注入"方向正确。

**Teacher scalar 诊断**：`teacher_b_p` 平均贡献最大（ablation_l2=0.137, gate_delta_l2=4.05），`teacher_b_q` 次之（ablation_l2=0.067, gate_delta_l2=3.09），两个 delta 项贡献较小。

### 8.4 Gated v6/v7（gated 微调）

- **v6**：在 v5 基础上调 gate init、scalar clipping。收益不明显
- **v7**：调 residual scale/clip。有改进但未越过 v5 最优 seed

### 8.5 Gated v8/v9（过冲控制与失败）

**v8** 目标：保持 `aggressive_param_rate = 0`，增强 `b_q / b_p` 主支路贡献。
- 对 v7 有改进，但未稳定超过 Full
- 不同 seed 间不够稳（20260429 上 LER=0.766638，Full=0.532874）

**v9** 目标：抑制 v8 的过冲风险，缓解 20260429。
- 实际结果相反——三 seed 汇总平均 `v9 = 0.820259`，`Full = 0.732550`，退化明显
- 原因：teacher 分支被压得过弱，收益流失

**v8/v9 的共同教训**：
- 继续微调 `scalar_gate_init_bias / scalar_norm_clip / residual_clip_b / residual_scale_b` 很难再带来实质性提升
- teacher 分支更激进 → 部分 seed 翻车；更保守 → 收益消失
- 当前瓶颈不是"超参还没拧对"，而是"表征方式和闭环目标本身不够对"

### 8.6 关于 No TeacherParams 的最终判断（2026-04-17）

经过离线训练多 seed 复查 + formal HIL benchmark-only 多 seed 复查后：

| 证据 | 结论 |
|------|------|
| 离线训练重训（3 seed） | No TeacherParams 稳定更好（ΔR² = +0.08576） |
| formal HIL seed=20260405 | Hybrid Full 更好 |
| formal HIL seed=20260406 | No TeacherParams 更好 |
| formal HIL seed=20260407 | Hybrid Full 更好 |

**最终判断**：`No TeacherParams` 的 formal HIL 优势不稳定且会随 seed 翻转。不能作为稳定更优正式主线。真正需要解决的不是"删不删 teacher params"，而是"teacher params 应如何编码"。

### 8.7 不再建议继续投入的方向

以下方向已被多轮实验基本证伪或边际收益极低：
1. 继续做 `gated v10 → v11` 这类 gate/clip/scale 微调
2. 继续做"删/不删 teacher params"大规模长跑
3. 继续扩 `PB Bound / PB ST` 为论文主线
4. 继续把 `No TeacherParams` 当成主叙事

---

## 9. 稳定结论清单

以下是经过多轮实验验证、多 seed 复查后可以安全写进论文和文档的结论。排序从最稳定到需谨慎使用。

### 9.1 安全可写入的结论

1. **Hybrid Residual-B 是当前正式主线方案**，它稳定优于 EKF 和 UKF
2. **UKF 是当前最强经典 baseline**（修正后：full covariance + 对称化/正定稳定化）
3. **优势不是来自更激进控制**：`correction_saturation_rate = 0`，`aggressive_param_rate = 0`
4. **主导 overflow 来源是 `histogram_input`**，不是控制参数或校正饱和
5. **输入统计范围是真实有效旋钮**：放宽输入范围不放不恶化 LER，反而显著压低 overflow
6. **float/int8 差异在各阶段都极小**，量化不是当前瓶颈
7. **离线训练改善 ≠ formal HIL 改善**：已被 No TeacherParams 和多个 gated 版本反复验证
8. **histogram delta 是关键输入通道**：去掉后 formal HIL 明显退化
9. **teacher params 的核心问题是编码方式**：不是数值坏掉，而是整平面广播 + 高冗余 + 闭环语义失配
10. **Gated v5 方向正确**：少数关键 teacher 标量 + 低维 gated scalar branch + 去冗余

### 9.2 需附限定条件的结论

1. "Hybrid Residual-B 优于 UKF"——结论在正式 4 场景下成立，但优势幅度 seed 依赖
2. "Gated v5 优于 Full"——在 3 seed 中 2 seed 显著更好、1 seed 持平（20260429）
3. "No TeacherParams 离线更好"——离线训练指标确实稳定更好，但 formal HIL 结论翻转
4. "当前 overflow 主导来自 histogram_input"——结论基于当前默认输入范围，在极端噪声下可能改变

### 9.3 不能写的结论

1. "CNN 全面优于所有经典解码器"
2. "项目提出了通用于所有量子纠错码的统一最优方法"
3. "已完成完整真实 FPGA 部署并实现工业级可用"
4. "No TeacherParams 是稳定更优正式主线"

---

## 10. 论文撰写与投稿路线

### 10.1 推荐标题方向

**最稳妥的工作标题**：
> A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction

备选方向：
- 工程系统型：`A Runtime-Consistent CNN-FPGA Adaptive Decoder for Drift-Aware GKP Error Correction`
- 方法机制型：`Teacher-Guided Residual Adaptive Decoding for GKP Error Correction Under Real-Time Hardware Constraints`

### 10.2 核心论文主张

> 在实时硬件约束下，保留稳定的经典 teacher，并让轻量 CNN 仅学习对运行时控制偏置有用的残差修正。在运行时一致的数据构造、双回路调度和软件 HIL 约束下，该 teacher-guided residual-b 方案能够稳定超过当前最强经典自适应 baseline UKF。

### 10.3 贡献点（三点式）

1. **双回路实时解码框架**：将快回路低时延解码、窗口统计累积、参数 bank 切换、慢回路推理和 HIL 验证组织为同一条可复现实验链路
2. **Teacher-guided residual-b 方案**：相比直接回归绝对参数更贴近在线解码控制语义；在正式 benchmark 中稳定优于 EKF、UKF、RLS 等经典基线
3. **系统化工程验证**：包括 float/int8/TFLite artifact 验证、overflow 来源定位、teacher/context/features ablation、软件 HIL 闭环验证

### 10.4 投稿目标排序

| 优先级 | Venue | 类型 | 匹配度 | 需补内容 |
|--------|-------|------|--------|----------|
| 1 | **QCE**（IEEE Quantum Week） | 会议 | 高 | 基本可投，补 features 正式结果 + 机制分析 |
| 2 | **TQE**（IEEE Trans. Quantum Engineering） | 期刊 | 高 | 补更完整 engineering cost 分析 |
| 3 | **EPJ Quantum Technology** | 期刊 | 中高 | 补更完整 ablation + 机制解释 |
| 4 | **QST**（Quantum Science and Technology） | 期刊 | 中 | 补更强 baseline + 真板/跨码扩展 |
| 5 | **npj Quantum Information / ACM TQC** | 期刊 | 中低 | 需显著补强后冲 |
| 6 | **FCCM / ACM FPGA / ICCAD / DATE** | 会议 | 低（当前） | 需真实 FPGA 综合/资源报告 |

### 10.5 论文结构提纲

1. **Introduction**：GKP 漂移自适应需求 + 离线 ≠ 在线 + residual corrector 定位
2. **Background**：GKP syndrome、快慢回路时间尺度、`(K, b)` 是运行时真正执行的目标
3. **Dual-Loop Framework**：Fast loop / Slow loop / Parameter update protocol / HIL execution
4. **Runtime-Consistent Learning**：为何 absolute regression 不够 + teacher-guided residual + 输入构成
5. **Experimental Protocol**：场景集/baseline 集/评价指标
6. **Main Results**：强 baseline 排序（Hybrid > UKF > EKF/RLS/Constant）
7. **Mechanism Analysis & Ablation**：teacher/context/features 消融
8. **Engineering Considerations**：量化/延迟/budget/overflow/board backend 现状与限制
9. **Discussion**：为何更像是 residual correction、未来扩展到 concatenated GKP-surface
10. **Conclusion**

### 10.6 投稿前建议补齐的证据

1. **features 正式 ablation 回填**：已有一轮结果但种子覆盖尚不完整
2. **机制解释链书面化**：argue 清楚 absolute regression → residual-b → Gated v5 的递进逻辑
3. **工程代价分析**：推理延迟、commit 成功率、参数更新频率、float/int8 差异
4. **至少一个场景的多 seed 稳定性复验**
5. **论文级图表**：双回路架构图、基线排序柱状图、ablation 表

---

## 11. 后续开发优先级与候选任务包

以下优先级排序基于"对当前 P4 结论可信度的增益"与"不依赖外部硬件条件"两个原则。

### 11.1 第一优先级：失败机理诊断

**问题**：为什么 Gated v5 在多数 seed 上明显更好，但在 `20260429` 上不稳？

**建议产出**：
1. `Full vs Gated v5` 的逐 window / 逐 commit 时间序列对照
2. 关键量可视化（`teacher_b_q/b_p`、预测 residual、commit 后实际 b、overflow、LER 恶化区间）
3. 翻车机制判断：符号偏移 / 幅度过冲 / 响应滞后 / teacher 本身不稳

**价值**：直接决定后续应该重写 loss、换 teacher 表征、做更强约束、还是补 teacher 估计质量。

**建议配置**：单 seed（20260429）、全 4 场景、paired benchmark-only。不扩大为新长跑。

### 11.2 第二优先级：Paper-Inspired 分支

**已有设计草案**：[docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md](docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md)

**核心改动**（建立在 Gated v5 之上）：
1. **输入侧**：新增 compact histogram/teacher stability summary 标量（histogram 总能量/质心漂移/各向异性；teacher b 变化量均值/std）
2. **模型侧**：新增 light stat summary branch（小 MLP），用 dual-gate 与主干融合
3. **Loss 侧**：主损失保持 `delta_b` MSE + λ1·最终 b_next 对齐损失 + λ2·轻量稳定性约束

**分支名**：`paper_inspired_statcalib_v1`

**推荐 benchmark 顺序**：
- 第一轮：2 seeds × 2 dynamic scenarios × repeats=2（对标 Gated v5）
- 第二轮：3 seeds × 4 scenarios × repeats=2（若第一轮方向正确）
- 第三轮：中间长度 paired benchmark（若第二轮也成立）

**原则**：优先验证方法论对不对，不是堆模型容量。

### 11.3 第三优先级：工程仿真补强

按性价比排序：
1. **noise_channels → effective parameters 离线桥接**（`physics/noise_effective_mapper.py`）
2. **load-aware latency injector**（条件偏置：pending_windows → 均值/方差抬高）
3. **stateful fault injector**（normal → retrying → degraded_hold_last → dropped_update）
4. **bit-accurate control pipeline**（先出位宽规范，再实现逐级 shadow pipeline）
5. **ADC/AFE 轻量读出链**（`physics/readout_chain.py`：gain/offset/ADC bits/full_scale）

前三项都可在软件 HIL 主线内完成。后两项偏中长期。`board_backend` 补全在真板条件具备前不建议重点投入。

### 11.4 第四优先级：文档与治理

1. 把 teacher-representation 结论回写到阶段结论文档和本集成文档
2. 补训练链独立 manifest（`requirements-train.txt`）
3. 补 .tflite 路径独立 manifest / smoke（前提：恢复 .tflite 运行环境）
4. 单开 cleanup 任务处理 `__pycache__/` / `.pyc` 的物理移除

### 11.5 当前不应启动的任务

- 新的 teacher-representation 长跑（在机理诊断完成前）
- 长时间 P4 正式多场景 frozen benchmark（在 Gated v5 稳定性确认前）
- 真板 backend 能力扩写
- 大规模 repo cleanup
- 论文正文正式写作（在关键证据补齐前）

---

## 12. 治理规范：AI Coding 工作流

本项目遵循 [docs/reference/AI_coding_workflow.md](docs/reference/AI_coding_workflow.md) 中定义的 AI Coding 工作流。以下为与 Captain AI 最相关的关键要点。

### 12.1 项目状态机

本项目的决策状态只能是以下之一：
- **Go**：允许继续做 bounded 开发任务（当前状态）
- **Narrow**：缩小范围
- **Pause**：暂停
- **Archive**：归档
- **Repair**：恢复期（当前已退出）

### 12.2 Captain 职责

Captain 是本项目的开发主控 AI 会话。职责：
1. 阅读本文档（`docs/02_experiment_plan.md`）及关键治理文件
2. 维护 `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md`
3. 把任务拆成 worker 可执行的任务包
4. 整合 worker 结果和 reviewer 意见
5. **不直接进行大规模实现**

Captain 每轮必须输出：
```text
1. 当前唯一任务
2. 为什么现在做它
3. Worker 任务包（Task ID / Goal / Allowed files / Forbidden scope / Verification）
4. 允许修改的文件范围
5. 禁止做的事
6. 验证命令或验收标准
7. 完成后需要更新的治理文件
```

### 12.3 单任务开发循环

每个任务都应按：**Captain 生成任务包 → Worker 执行 → Reviewer 审查 → Captain 整合** 的顺序执行。

Task package 模板：
```text
Task ID:
Goal:
Why now:
Allowed files:
Forbidden scope:
Inputs to read:
Expected output:
Verification:
Docs to update:
Reviewer type: normal / adversarial / milestone
```

### 12.4 里程碑闸门

每个 milestone 结束后：
- 暂停开发，做里程碑审查
- 输出 `docs/review/<TaskID>_milestone_review.md`
- 结论只能是：Allow / Conditional / Block

### 12.5 关键约束

- 仓库文件是主状态，AI 会话不是主状态
- 每轮只推进一个当前唯一任务
- 不让两个 agent 同时修改同一批文件
- 不把计划、mock、stub、未来能力写成已完成事实
- 每个任务完成后必须有可验证结果、风险记录和下一步唯一任务
- 默认顺序执行；并行仅在文件不重叠时允许

---

## 13. 风险与待解决问题

### 13.1 当前风险清单

| ID | 风险 | 等级 | 缓解措施 |
|------|------|------|----------|
| R1 | 默认运行环境不可直接执行最小 benchmark | 中 | 所有文档显式指定推荐解释器 |
| R2 | 完整训练链、.tflite 与真板环境仍无统一依赖说明 | 中 | `requirements-recovery.txt` 作用域诚实；后续单开 manifest 任务 |
| R3 | 软件 HIL 与真板 HIL 边界易被误写 | 高 | 所有文档引用 `docs/03_hil_p4_boundary_audit.md` 统一口径 |
| R4 | 仓库中已有大量缓存与生成物噪声 | 中 | `docs/06_repo_noise_governance.md` 已固定分类策略；后续单开 cleanup |
| R5 | P4 只有 recovery smoke 级证据，非正式多场景 frozen benchmark | 中高 | T9 已完成四模式单场景 smoke；正式多场景仍需后续补 |
| R6 | .tflite 真导出与 stub 回退易混淆 | 中高 | 文档与日志必须显式标注 artifact type |
| R7 | 具体 cleanup 执行窗口与归档方式未定 | 中 | 后续单开有界 cleanup 任务 |
| R8 | bounded recovery smoke 结论易被误外推到真板 | 中 | 持续写清结论边界 |

### 13.2 当前开放问题

1. **下一张 bounded 开发任务包应该优先选哪一类？**
   - 候选人：失败机理诊断 / paper-inspired 分支 / P4 多场景证据补全 / 训练 manifest

2. **历史文档中引用的 `.venvs/tf311` 不可用，如何恢复 .tflite 验收能力？**
   - 当前状态：Windows 工作区内不存在该路径
   - 后续若需 .tflite 路径，需先建立 TensorFlow/TFLite 运行环境

3. **训练链需要什么级别的独立 manifest？**
   - 当前训练可通过 `DLEnv` 的 `torch` 后端运行
   - 是否需要独立 `requirements-train.txt` / `pyproject.toml`

4. **正式 P4 多场景 frozen benchmark 何时恢复？**
   - 建议在机理诊断或 paper-inspired 分支方向确认后再决定

5. **已跟踪的 `.pyc`、`runs/`、`artifacts/` 何时启动有界 cleanup？**
   - 建议在下一 milestone 稳定后单开 cleanup 任务

6. **board_backend placeholder 是否需要现在补强，还是继续延期？**
   - 当前建议：在真板条件具备前不投入，继续作为条件性扩展

### 13.3 暂缓事项

在下一任务包明确前暂缓：
1. `noise_channels → effective parameters` 桥接
2. load-aware latency injector
3. stateful fault injector
4. bit-accurate control pipeline
5. teacher-representation 新分支扩展
6. 论文正文正式写作

---

## 14. 附录：文件路径索引与常用命令

### 14.1 关键文件快速索引

**治理与计划**：
- [CLAUDE.md](CLAUDE.md) — Claude Code 审查指令
- [AGENTS.md](AGENTS.md) — AI agent 治理文件
- [docs/02_experiment_plan.md](docs/02_experiment_plan.md) — 本文件
- [docs/04_task_board.md](docs/04_task_board.md) — 任务板
- [docs/05_decision_log.md](docs/05_decision_log.md) — 决策日志
- [docs/07_handoff.md](docs/07_handoff.md) — 交接文档
- [docs/08_risks_and_open_questions.md](docs/08_risks_and_open_questions.md) — 风险清单
- [docs/reference/AI_coding_workflow.md](docs/reference/AI_coding_workflow.md) — AI 开发工作流
- [requirements-recovery.txt](requirements-recovery.txt) — 恢复期最小依赖

**项目方案与结论**：
- [docs/CNN_FPGA_GKP_工程化实验方案.md](docs/CNN_FPGA_GKP_工程化实验方案.md) — 工程方案全文
- [docs/CNN_FPGA_GKP_阶段结论.md](docs/CNN_FPGA_GKP_阶段结论.md) — 阶段结论全文
- [docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md](docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md) — 7 项仿真补强
- [docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md](docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md) — paper-inspired 分支设计
- [docs/CNN_FPGA_GKP_论文提纲_摘要_贡献点草稿.md](docs/CNN_FPGA_GKP_论文提纲_摘要_贡献点草稿.md) — 论文草稿
- [docs/CNN_FPGA_GKP_项目完成目标与投稿路线报告.md](docs/CNN_FPGA_GKP_项目完成目标与投稿路线报告.md) — 投稿路线

**恢复期专项**：
- [docs/00_project_snapshot.md](docs/00_project_snapshot.md) — 恢复期起始快照
- [docs/01_legacy_audit.md](docs/01_legacy_audit.md) — legacy 审计报告
- [docs/03_hil_p4_boundary_audit.md](docs/03_hil_p4_boundary_audit.md) — HIL/P4 边界审计
- [docs/06_repo_noise_governance.md](docs/06_repo_noise_governance.md) — 仓库噪声治理
- [docs/P0_smoke_bootstrap.md](docs/P0_smoke_bootstrap.md) — P0 smoke 复用说明
- [docs/P3_software_hil_bootstrap.md](docs/P3_software_hil_bootstrap.md) — P3 software HIL 复用说明
- [docs/P4_benchmark_recovery_bootstrap.md](docs/P4_benchmark_recovery_bootstrap.md) — P4 recovery bootstrap 说明
- [docs/legacy_context/2026-05-06_CNN_FPGA_GKP_legacy_handoff.md](docs/legacy_context/2026-05-06_CNN_FPGA_GKP_legacy_handoff.md) — 2026-05-06 交接文档

**核心代码入口**：
- [cnn_fpga/benchmark/run_hil_suite.py](cnn_fpga/benchmark/run_hil_suite.py) — P3 HIL 主入口
- [cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py](cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py) — P4 benchmark 主入口
- [cnn_fpga/benchmark/run_p4_teacher_representation_paired.py](cnn_fpga/benchmark/run_p4_teacher_representation_paired.py) — teacher-representation paired benchmark
- [cnn_fpga/model/tiny_cnn.py](cnn_fpga/model/tiny_cnn.py) — CNN 模型定义
- [cnn_fpga/model/train.py](cnn_fpga/model/train.py) — 训练入口
- [cnn_fpga/model/export.py](cnn_fpga/model/export.py) — .tflite 导出
- [cnn_fpga/decoder/param_mapper.py](cnn_fpga/decoder/param_mapper.py) — 参数映射
- [cnn_fpga/runtime/fast_loop_emulator.py](cnn_fpga/runtime/fast_loop_emulator.py) — 快回路仿真
- [cnn_fpga/runtime/slow_loop_runtime.py](cnn_fpga/runtime/slow_loop_runtime.py) — 慢回路运行时
- [cnn_fpga/hwio/board_backend.py](cnn_fpga/hwio/board_backend.py) — 真板 backend（placeholder）
- [cnn_fpga/hwio/mock_fpga.py](cnn_fpga/hwio/mock_fpga.py) — mock FPGA backend
- [benchmark/compare_full_vs_simplified_ler.py](benchmark/compare_full_vs_simplified_ler.py) — P0 对比脚本

**关键配置**：
- [cnn_fpga/config/hardware_hil_recovery_smoke.yaml](cnn_fpga/config/hardware_hil_recovery_smoke.yaml) — recovery smoke HIL 配置
- [cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml](cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml) — recovery smoke P4 配置
- [cnn_fpga/config/experiment_static_theta_v2.yaml](cnn_fpga/config/experiment_static_theta_v2.yaml) — P1 主模型训练配置
- [cnn_fpga/config/experiment_runtime_b_residual.yaml](cnn_fpga/config/experiment_runtime_b_residual.yaml) — Full residual-b 训练配置
- [cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml](cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml) — Gated v5 训练配置
- [cnn_fpga/config/p4_multiscenario_strong_baselines.yaml](cnn_fpga/config/p4_multiscenario_strong_baselines.yaml) — 强 baseline P4 配置

**关键 artifact**：
- `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz` — P1 浮点主模型
- `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz` — P1 int8 模型
- `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz` — residual-b v1 模型

**最新 recovery smoke 证据**：
- `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104/hil_summary.json` — T12 确定性复验 run 1
- `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104/hil_summary.json` — T12 确定性复验 run 2
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/summary.json` — T9 四模式单场景 smoke

### 14.2 常用运行命令

```powershell
# === 恢复期 smoke（推荐在 AConda 下运行）===

# P0: full_qec vs simplified 最小对比
& 'C:\ProgramData\anaconda3\python.exe' benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test

# P3: software HIL 最小 smoke（秒级，逐字一致复验）
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml

# P4: 单场景单模式 benchmark smoke
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds

# P4: frozen baseline 四模式单场景 smoke
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds

# === 正式开发级命令（需要更完整环境）===

# P4: 强 baseline 多场景对比（需要 torch 环境或更完整配置）
# 在 DLEnv 或其他完整环境下运行
# python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml ...
```

### 14.3 本文档的更新规则

1. 每次里程碑闸门后更新本文件
2. 每当稳定结论清单发生变化时更新第 9 节
3. 每当工程边界标签发生变化时更新第 7 节
4. 每当环境/依赖信息发生变化时更新第 6 节
5. 更新后同步修改 `docs/07_handoff.md` 中的"当前状态"

---

**文档结束。** 本文件是项目 Captain AI 的唯一切入点。新会话启动时只需阅读本文件即可获得继续开发的全部必要上下文。
