# DriftAdaptiveQEC — 项目集成文档与后续开发计划

**文档角色：** 本文件是项目唯一的核心上下文文档。合并了分散在 6 份主要文档、9 份治理文件、1 份 legacy handoff 中的所有关键信息。任何新 Captain AI、Worker AI 或 Reviewer AI 在参与本项目前必须首先阅读本文件。

**源文档：**
- `docs/CNN_FPGA_GKP_工程化实验方案.md` — 工程方案全文（双回路架构、时序预算、实验矩阵）
- `docs/CNN_FPGA_GKP_阶段结论.md` — 阶段结论全文（P0-P4 所有正式结果、参数调优、overflow 拆分）
- `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md` — 7 大仿真补强方向及优先级
- `docs/CNN_FPGA_GKP_论文提纲_摘要_贡献点草稿.md` — 论文草稿（标题、摘要、贡献点、提纲）
- `docs/CNN_FPGA_GKP_项目完成目标与投稿路线报告.md` — 完成标准、投稿目标、baseline 建议
- `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md` — paper-inspired statcalib 分支设计
- `docs/legacy_context/2026-05-06_CNN_FPGA_GKP_legacy_handoff.md` — 最新工程交接（v8/v9 失败分析）
- `docs/00_project_snapshot.md` ~ `docs/08_risks_and_open_questions.md` — 恢复期治理文件（T1-T13）

**最后更新：** 2026-05-08
**当前阶段：** Phase 2: Controlled Development
**当前唯一任务：** 待定义
**决策状态：** Go（受控继续开发）
**当前分支：** `main`

---

## 目录

1. [Quick Start（Captain AI 5 分钟速览）](#1-quick-startcaptain-ai-5-分钟速览)
2. [项目身份与核心问题](#2-项目身份与核心问题)
3. [代码仓库结构](#3-代码仓库结构)
4. [阶段历史与关键结果（P0 → P4）](#4-阶段历史与关键结果p0--p4)
5. [恢复期 Phase 0–1 详细记录（T1–T13）](#5-恢复期-phase-01-详细记录t1t13)
6. [当前环境与依赖矩阵](#6-当前环境与依赖矩阵)
7. [工程真实性边界表](#7-工程真实性边界表)
8. [Teacher-Representation 分支谱系（v1–v9）](#8-teacher-representation-分支谱系v1v9)
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

**关键物理量**：
- 快回路输入：syndrome `(s_q, s_p)`，浮点 `[float32×2]`，FPGA 内固定点 `Q4.20`
- 直方图：32×32 bin，窗口 W=2048 周期（T_window=10.24ms）
- 慢回路输出：噪声参数 `(σ, μ_q, μ_p, θ)` → 参数映射 → 解码器参数 `(K 2×2, b 2×1)`
- 参数切换：双缓冲 bank_A/bank_B + epoch_id + 周期边界原子切换

### 1.2 项目做到哪了

| 阶段 | 状态 | 关键事实 | 证据 |
|------|------|----------|------|
| P0 物理基线 | **完成** | `full_qec` vs `simplified` LER gap 确认存在 (gap ~0.40) | `runs/drift_suite/drift_v1_20260317_154905_87cc72d5c4de/` |
| P1 CNN 训练 | **完成** | 浮点 R² > 0.99，int8 退化 < 1%，全部标签达标 | `artifacts/models/static_theta_v2/` |
| P2 行为仿真 | **完成** | 公平基线下的自适应闭环已跑通，float/int8 一致 | `runs/p2_mode_benchmark/hardware_emulation_v1_20260319_160130_20670d1c0d1f` |
| P3 软件 HIL | **完成** | `mock + artifact_npz + inproc` 路径逐字一致复验 | T12 两次 run SHA256 一致 |
| P3 真板 HIL | **未完成** | `board_backend.py` 仍是 placeholder，缺 `/dev/uio*` | — |
| P4 多场景 benchmark | **部分** | 强 baseline 对比已完成；recovery smoke 已复验 | `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/` |

### 1.3 当前最重要的 7 个状态事实

1. **环境**：恢复期 smoke 推荐 `C:\ProgramData\anaconda3\python.exe`（numpy+PyYAML）；训练候选 `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`（+torch）
2. **依赖**：根目录 `requirements-recovery.txt` 仅含 `numpy + PyYAML`，故意不命名为 `requirements.txt`，只覆盖 P0/P3/P4 recovery smoke
3. **确定性**：`mock + model_artifact + artifact_npz + inproc` 路径已做到逐字一致复验（`hil_summary.json` 和 `hil_events.json` SHA256 一致，LER=0.454375, overflow_rate=0.002）
4. **主线模型**：正式方案是 `Hybrid Residual-B`（teacher-guided residual-b），最强候选是 `Gated v5`（仅 4 个 teacher 标量，低维 gated scalar branch）
5. **最强 baseline**：UKF（修正后：full covariance + 对称化/正定稳定化），但仍弱于 Hybrid Residual-B ~0.0196
6. **主导瓶颈**：overflow 来源统一为 `histogram_input`（非 correction_saturation 或 aggressive_param）
7. **真板**：`board_backend.py` 仍是 placeholder，不要写"真板 HIL 已通过"

### 1.4 不该做什么

- 继续微调 `gated v10/v11/v12` 这类 gate/clip/scale 超参——v8/v9 已证明边际收益极低，甚至会退化
- 做大规模"删/不删 teacher params"长跑——No TeacherParams 的 formal HIL 结果已明确会随 seed 翻转
- 把 `PB Bound / PB ST` 扩写成论文主线——仅作场景特定的辅助机制分析支线
- 把 bounded recovery smoke 扩写成"真板已恢复"或"正式多场景 frozen benchmark 已恢复"
- 继续围绕 EKF 做小幅改进——UKF 已是更强且经过验证的经典 baseline

### 1.5 最值得做什么（按优先级）

1. **第一优先**：做 `seed=20260429` 的失败机理诊断——解释为什么 Gated v5 在多数 seed 上明显更好，但 20260429 上不稳
2. **第二优先**：落地 paper-inspired 分支（`paper_inspired_statcalib_v1`）——在 Gated v5 基础上加统计聚合标量 + 闭环一致 loss
3. **第三优先**：补 P4 多场景正式 frozen benchmark 证据，或补训练/.tflite 独立 manifest

---

## 2. 项目身份与核心问题

### 2.1 项目基本信息

- **仓库名**：`DriftAdaptiveQEC`
- **正式项目名**：CNN-FPGA 协同 GKP 漂移自适应解码系统
- **定位**：工程系统型量子纠错研究项目，兼具方法机制贡献
- **当前分支**：`main`
- **Git 用户**：AscendFoam
- **代码版本**：`cnn_fpga.__version__ = "0.1.0"`
- **最近提交**：`35e32e1` (docs: 更新决策日志、任务板和交接文档以反映T10完成)

### 2.2 核心物理问题

在连续变量 GKP 编码的量子纠错中，噪声参数会随时间漂移：
- **sigma 漂移**：误差幅值随时间变化（线性增加、阶跃跳变、周期振荡、随机游走）
- **mu 偏置漂移**：位移均值偏离零点
- **theta 旋转漂移**：协方差主轴旋转角变化

固定参数的线性解码器 `Δ = K @ s + b` 在漂移场景下会逐渐失配，导致逻辑错误率（LER）上升。

**解决方案：双回路在线自适应解码**
- 快回路（FPGA）：超低时延（≤5μs）的确定性线性解码
- 慢回路（ARM/CNN）：周期性从 syndrome 统计中估计漂移，更新 K, b

### 2.3 核心工程问题

实时硬件约束下，慢回路不能是任意复杂的模型：
1. **延迟预算**：慢回路单次更新 < 5ms（保守 < 20ms）
2. **参数更新**：必须支持原子切换（双缓冲 + epoch_id + 周期边界）
3. **数值精度**：FPGA 侧固定点 Q4.20，量化误差必须在可控范围
4. **部署一致性**：训练/导出/HIL 推理必须在统一语义下对齐
5. **鲁棒性**：DMA 超时、推理失败、参数越界等异常必须有降级策略

### 2.4 核心方法论贡献

当前主方案 `Hybrid Residual-B` 的方法论贡献是：

> 在实时硬件约束下，保留稳定的经典 teacher（如 Window Variance），让轻量 CNN 只学习对运行时控制偏置 `b` 的残差修正，而不是让 CNN 独立承担全部漂移参数估计。

这澄清了一个重要设计原则：

- **不做的**：端到端让 CNN 直接回归 `(σ, μ_q, μ_p, θ)` 全部绝对参数
  - 问题：离线预测目标与运行时控制语义错位（离线看的是参数拟合 MSE，在线用的是 K,b 控制效果）
- **做的**：Teacher 给出稳定一阶估计 → CNN 输出 `delta_b` → `b_next = teacher_b + delta_b` → ParamMapper 严格映射为 `(K, b)`
  - 优势：teacher 保证底线不崩溃，CNN 只做小幅、局部、时间一致的残差修正

### 2.5 技术路线核心参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| T_fast | 5 μs | 快回路周期目标 |
| T_slow | 20 ms | 慢回路更新周期 |
| W (窗口长度) | 2048 周期 | T_window = 10.24 ms |
| 直方图网格 | 32×32 | bin 计数 uint16/uint32 |
| 固定点格式 | Q4.20 | syndrome 输入 + K,b 参数 |
| 内部累加 | Q8.24 → Q4.20 | 乘加后回落 |
| 饱和策略 | 钳位到最大可表示值 | 不做截断回绕 |
| 参数平滑 | EMA β=0.1~0.3 | 防抖动 |
| 双缓冲 | bank_A/bank_B | 运行中/写入中，周期边界切换 |

### 2.6 参数映射规则（已冻结，不得隐式修改）

1. **误差协方差**：`theta_hat` clip 到 `[-20°, +20°]`，构造旋转矩阵，主轴系 `C_principal = diag(sigma_q², sigma_p²)`，实验室系 `C = R·C_principal·Rᵀ`
2. **测量噪声协方差**：`R_meas = (sigma_meas² + delta_eff²)·I`
3. **线性增益矩阵**：`K_raw = C·(C + R_meas)⁻¹`，然后特征值裁剪 clip(lambda_raw, [g_min, g_max])，再可选 gain_scale 缩放
4. **偏置项**：`b_target = alpha·(I - K_target)·mu`（不能用 `-alpha·mu`）
5. **参数平滑**：`K_next = (1-β)·K_prev + β·K_target`，`b_next = (1-β)·b_prev + β·b_target`

### 2.7 直方图边界策略（必须显式实现）

- `s_q/s_p` 超出映射范围：先 clip 到 `[s_min, s_max]` 再入 bin
- 累计 `overflow_counter` 作为诊断指标
- 每次慢回路推理输出 `overflow_ratio = overflow_counter / W`
- 若超过阈值（如 5%）触发告警
- 溢出分三类诊断：`histogram_input_saturation_rate` / `correction_saturation_rate` / `aggressive_param_rate`

---

## 3. 代码仓库结构

### 3.1 顶层目录

```text
DriftAdaptiveQEC/
├── AGENTS.md                         # AI agent 治理文件
├── CLAUDE.md                         # Claude Code 审查指令（默认角色：只读 reviewer）
├── README.md                         # 项目入口说明（引用 requirements-recovery.txt）
├── requirements-recovery.txt         # 恢复期最小依赖 manifest（仅 numpy + PyYAML）
├── physics/                          # 物理仿真模块
│   ├── gkp_state.py                  # GKP 态定义与演化
│   ├── noise_channels.py             # 玻色量子噪声通道（未接入 P2/P3/P4 主线）
│   ├── syndrome_measurement.py       # 综合征测量（T12 后支持显式 rng）
│   ├── error_correction.py           # 线性解码参数映射
│   └── logical_tracking.py           # 逻辑错误判定（有效模型口径）
├── cnn_fpga/                         # 工程主模块
│   ├── __init__.py                   # version = "0.1.0"
│   ├── config/                       # 配置文件（~50 个 YAML）
│   ├── data/                         # 数据流水线（schema/dataset/histogram/split）
│   ├── model/                        # 模型训练/评估/量化/导出
│   ├── decoder/                      # 解码器（param_mapper/linear_runtime/EKF）
│   ├── runtime/                      # 双回路运行时（10 个文件）
│   ├── hwio/                         # 硬件 I/O 抽象（5 个文件）
│   ├── benchmark/                    # 实验入口（15 个文件）
│   └── report/                       # 报告生成（metrics/plots/markdown）
├── fpga/                             # FPGA RTL/HLS 实现（规划中）
│   ├── rtl/                          # Verilog（linear_decoder/histogram_accum/param_bank）
│   ├── hls/                          # HLS C++
│   ├── sim/                          # Testbench
│   ├── constraints/                  # 时序约束
│   └── vivado/                       # 项目脚本
├── benchmark/                        # 最小 P0 对比脚本
│   └── compare_full_vs_simplified_ler.py
├── docs/                             # 项目文档（治理 + 方案 + 结论 + 草稿）
├── runs/                             # 历史运行输出（1841 个已跟踪文件）
├── artifacts/                        # 模型/数据集/报告产物（110 个已跟踪文件）
└── relative_papers/                  # 参考论文
```

### 3.2 physics/ 模块详细

| 文件 | 功能 | 状态 | 接入主线? |
|------|------|------|-----------|
| `gkp_state.py` | GKP 态定义、Wigner 函数、位移操作 | 可用 | 是 |
| `noise_channels.py` | 光子损失/热噪声/位移噪声/相位噪声通道 | 已实现 | **否**（P2/P3/P4 用有效参数模型，非底层通道） |
| `syndrome_measurement.py` | 有限压缩噪声 + 测量效率 + ancilla error + shot noise | T12 后支持显式 rng | 是 |
| `error_correction.py` | 线性解码 `Δ = K@s + b` 及参数映射 | 可用 | 是 |
| `logical_tracking.py` | 累计残差越 GKP 决策边界 `±√(2π)/2` 则记逻辑错误 | 可用 | 是（有效模型口径，非完整电路级） |

**重要说明**：`noise_channels.py` 的底层物理噪声通道虽然已实现，但 P2/P3/P4 主线实际使用的是"有效噪声参数 + 场景化漂移 envelope"（`sigma/mu_q/mu_p/theta_deg` + `linear/periodic/step/random_walk`）。两者是不同层次的建模，不应混淆。后续仿真补强计划中建议做"两层噪声模型"桥接（物理通道层 → 有效参数层），而不是直接替换。

### 3.3 cnn_fpga/ 模块详细

#### cnn_fpga/data/ — 数据流水线

| 文件 | 功能 | 关键特性 |
|------|------|----------|
| `schema.py` | 数据结构定义（syndrome/histogram/labels） | 固定数据契约 |
| `dataset_builder.py` | 数据集生成 | 支持 `anisotropic_gaussian`（椭圆高斯，使 theta 可辨识）和 `isotropic_gaussian` |
| `histogram.py` | 直方图映射、归一化（含 `log1p(alpha*H)` 可选） | 32×32 网格，支持 overflow 计数 |
| `split.py` | train/val/test 划分 | 分层划分，保证标签分布一致 |

#### cnn_fpga/model/ — 模型与训练

| 文件 | 功能 | 关键特性 |
|------|------|----------|
| `tiny_cnn.py` | CNN 模型定义 | **双后端**：主路径手写 NumPy（`TinyCNN`）+ PyTorch/CUDA 可选（`TinyCNNTorch`）。支持 Conv2d + ReLU + AvgPool + Flatten + FC。输出支持 `absolute_params`(4) 或 `residual_b`(2) |
| `train.py` | 训练入口 | 支持 `--backend numpy|torch`，支持 `--device cuda` |
| `evaluate.py` | 回归评估 | 输出 MSE/MAE/R²/per_label |
| `evaluate_tflite.py` | 独立 TFLite 精度评估 | 需 TensorFlow/TFLite 运行环境（当前 Windows 环境未恢复） |
| `quantize.py` | QAT/PTQ + int8 导出 | 对称 int8 量化，支持 per-tensor/per-channel |
| `export.py` | ONNX/TFLite 导出 | **双路径**：真 `.tflite` 导出 + `tflite_stub_v1` 回退（manifest 格式 `.tflite.json`） |
| `validate_export.py` | artifact 与 TFLite 一致性验收 | 逐样本输出差异对比；未通过则不允许进入 HIL benchmark |

**训练细节**：
- 当前 P1 主模型：`static_theta_v2`，训练集 13107 样本，test 集 1639 样本
- 损失函数：MSE with loss weights (sigma=1.0, mu_q=1.0, mu_p=1.0, theta_deg=2.0)
- 学习率：0.001，batch_size：64，epochs：200
- 输入：5 窗口 × 32×32 histogram = (5, 32, 32) 单通道

#### cnn_fpga/decoder/ — 解码器

| 文件 | 功能 | 关键特性 |
|------|------|----------|
| `param_mapper.py` | `(σ, μ_q, μ_p, θ)` → `(K 2×2, b 2×1)` | 协方差一致映射：`K = C·(C+R)⁻¹`（特征值裁剪），`b = α·(I-K)·μ`。**已冻结公式，不得隐式修改** |
| `linear_runtime.py` | 快回路等价软件实现 | 固定点 Q4.20 运算 + 饱和处理 + correction 输出 |
| `ekf_baseline.py` | EKF 基线 | 经典递推滤波，作为 baseline 对照 |
| `mwpm_stub.py` | MWPM 接口占位 | 后续扩展用 |

**param_mapper 默认参数（当前主线）**：
- `sigma_ratio_p = 0.55`
- `gain_clip = [0.10, 1.20]`
- `beta_smoothing = 0.20`
- `alpha_bias = 0.90`
- `gain_scale = 1.0`
- `sigma_measurement = 0.03`

#### cnn_fpga/runtime/ — 双回路运行时

| 文件 | 功能 | 关键特性 |
|------|------|----------|
| `fast_loop_emulator.py` | 快回路仿真 | 固定点 Q4.20 + 周期约束（1.5μs 关键路径）+ 三类 overflow 拆分统计。T12 后快回路噪声 RNG 与测量噪声 RNG 分离 |
| `slow_loop_runtime.py` | 慢回路 CNN 推理与参数更新 | 支持多种模式：`fixed_baseline` / `oracle_delayed` / `model_artifact` / `window_variance` / `ekf` / `ukf` / `cnn_fpga` / `hybrid_residual_b` / `hybrid_residual_mu` 等 |
| `scheduler.py` | 双回路调度控制 | stage/commit 状态机、窗口就绪检测、慢回路触发、参数提交 |
| `param_bank.py` | 双缓冲参数切换 | bank_A/bank_B + epoch_id + commit_epoch 原子切换 + 回显确认 |
| `latency_injector.py` | DMA/AXI/推理延迟注入 | 当前基于独立高斯/常数抽样（非负载耦合） |
| `inference_service.py` | 推理服务抽象 | `inproc`（进程内 NumPy 推理）/ `tflite`（子进程 TFLite 推理）。tflite 路径区分 `tflite_service` 和 `tflite_stub_service` |
| `inference_worker.py` | 独立推理 worker | 子进程 TFLite 推理 worker |
| `feature_builder.py` | 慢回路输入特征构建 | 多窗口 histogram + histogram deltas + teacher prediction/params/deltas + context 窗口 |
| `noise_bridge.py` | 物理噪声桥接 | 已移除重型未使用的物理噪声顶层导入（解决 Windows benchmark 启动阻塞） |

#### cnn_fpga/hwio/ — 硬件 I/O 抽象

| 文件 | 功能 | 边界标签 | 说明 |
|------|------|----------|------|
| `axi_map.py` | AXI-Lite 寄存器映射定义 | — | CTRL/STATUS/K/b/ACTIVE_BANK/EPOCH_ID 地址表 |
| `dma_client.py` | DMA 读写接口 | — | histogram buffer 双缓冲 DMA（1024 bins × 4B = 4096B） |
| `mock_fpga.py` | 无板卡时的 event-driven 行为仿真 | **`mock_backend`** | 维护 DMA/param-bank 语义，产出 `window_ready`/`commit_applied`/`commit_ack_asserted` 事件。`metadata={"backend": "mock_fpga"}` |
| `fpga_driver.py` | HIL 驱动封装 | — | `board/real` backend 标为 reserved for future real-board integration |
| `board_backend.py` | 真板卡 MMIO/DMA backend 骨架 | **`placeholder_real_board_backend`** | `schedule_commit()` 返回大量 `None`；`step()` 返回空事件。文件顶层注释 `Placeholder real-board backend` |

**AXI-Lite 地址建议**：
| 地址 | 寄存器 | 位定义 |
|------|--------|--------|
| `0x00` | CTRL | bit0 start, bit1 reset_hist, bit2 commit_bank |
| `0x04` | STATUS | bit0 ready, bit1 hist_ready, bit2 commit_ack |
| `0x10–0x24` | K11, K12, K21, K22, b1, b2 | Q4.20 固定点 |
| `0x30` | ACTIVE_BANK | 当前运行 bank |
| `0x34` | EPOCH_ID | 当前 epoch |

#### cnn_fpga/benchmark/ — 实验入口

| 文件 | 功能 | 使用阶段 |
|------|------|----------|
| `run_hil_suite.py` | **P3 软件 HIL 单会话入口**。通过 `hil.backend` 选择 backend，产出 `hil_events.json` / `hil_summary.json` | P3 |
| `run_p4_multiscenario_benchmark.py` | **P4 多场景 benchmark 核心**。批量调用 `run_hil_session()`，支持 paired seeds、多 mode、多 scenario、repeat 分块、resume | P4 |
| `run_p4_teacher_representation_paired.py` | **Teacher-representation paired benchmark**。Full vs Gated v5/v8/v9 等对照。支持 `--detach`、`--detach-log-dir`、分块 repeat | P4 |
| `run_p2_mode_benchmark.py` | P2 行为仿真模式对比 | P2 |
| `run_p3_param_sweep.py` | P3 参数清扫（gain_clip/beta/alpha_bias/gain_scale） | P3 |
| `run_p3_histogram_tuning.py` | P3 直方图范围调参（syndrome_limit/histogram_range_limit） | P3 |
| `run_p4_hybrid_vs_ukf_ablation.py` | P4 teacher/context/features ablation | P4 |
| `run_p4_no_teacher_params_stability.py` | No TeacherParams 稳定性复查（裁剪版 formal HIL） | P4 |
| `run_p4_teacher_params_reencoding_controlled.py` | Teacher params 重编码对照（Gated/Selective/Minimal/Reencoded） | P4 |
| `summarize_p4_features_ablation.py` | P4 features 消融自动汇总 | P4 |
| `run_drift_suite.py` | P0 漂移场景对比（full_qec vs simplified vs EKF vs CNN-FPGA） | P0 |
| `run_hardware_emulation.py` | P2 硬件行为仿真单场景 | P2 |
| `run_hil_mode_benchmark.py` | P3 多 mode HIL 对比 | P3 |
| `run_p4_gap_diagnostic.py` | P4 gap 诊断工具 | P4 |

### 3.4 配置文件全景

#### 训练/数据配置
| 文件 | 用途 |
|------|------|
| `experiment_static_theta_v2.yaml` | **P1 主模型训练配置**（anisotropic_gaussian, sigma_ratio_p=0.55） |
| `experiment_runtime_b_residual.yaml` | **Full residual-b 训练配置** |
| `experiment_runtime_mu_residual.yaml` | residual-mu 训练配置 |
| `experiment_runtime_b_residual_ctx1/ctx3.yaml` | context=1/3 窗口训练配置 |
| `experiment_runtime_b_residual_no_hist_deltas.yaml` | 去掉 histogram delta 的训练配置 |
| `experiment_runtime_b_residual_no_teacher_prediction.yaml` | 去掉 teacher prediction 的训练配置 |
| `experiment_runtime_b_residual_no_teacher_params.yaml` | 去掉 teacher params 的训练配置 |
| `experiment_runtime_b_residual_no_teacher_deltas.yaml` | 去掉 teacher deltas 的训练配置 |
| `experiment_runtime_b_residual_teacher_ukf.yaml` | teacher=UKF 的训练配置 |

#### Teacher-representation 配置（gated 系列）
| 文件 | 版本 | 状态 |
|------|------|------|
| `experiment_runtime_b_residual_norm_gated_teacher.yaml` | v1 | 早期 |
| `experiment_runtime_b_residual_norm_gated_teacher_v2.yaml` | v2 | 早期 |
| `experiment_runtime_b_residual_norm_gated_teacher_v3.yaml` | v3 | 早期 |
| `experiment_runtime_b_residual_norm_gated_teacher_v4.yaml` | v4 | 早期 |
| `experiment_runtime_b_residual_norm_gated_teacher_v5.yaml` | **v5** | **当前最强候选** |
| `experiment_runtime_b_residual_norm_gated_teacher_v6.yaml` | v6 | 微调 |
| `experiment_runtime_b_residual_norm_gated_teacher_v7.yaml` | v7 | 微调 |
| `experiment_runtime_b_residual_norm_gated_teacher_v8.yaml` | v8 | 过冲控制（不够稳） |
| `experiment_runtime_b_residual_norm_gated_teacher_v9.yaml` | v9 | 过度保守（退化明显） |

#### P4 benchmark 配置
| 文件 | 用途 |
|------|------|
| `p4_multiscenario.yaml` | 早期 P4 多场景配置 |
| `p4_multiscenario_hybrid_b.yaml` | Hybrid Residual-B 正式 P4 配置 |
| `p4_multiscenario_hybrid_b_long.yaml` | Hybrid Residual-B 长配置 P4 |
| `p4_multiscenario_strong_baselines.yaml` | **强 baseline 正式 P4 配置**（EKF/UKF/RLS/Constant/Hybrid） |
| `p4_hybrid_vs_ukf_ablation_teacher.yaml` | teacher ablation 正式配置 |
| `p4_hybrid_vs_ukf_ablation_context.yaml` | context ablation 配置 |
| `p4_hybrid_vs_ukf_ablation_features.yaml` | features ablation 正式配置 |
| `p4_teacher_repr_mid.yaml` | teacher-representation 中间长度配置 |

#### Recovery 期专用配置
| 文件 | 用途 |
|------|------|
| `hardware_hil_recovery_smoke.yaml` | **P3 recovery smoke**：`mock + model_artifact + artifact_npz + inproc` |
| `p4_multiscenario_recovery_smoke.yaml` | **P4 recovery smoke**：同上口径，2 windows，单场景 |

### 3.5 仓库噪声情况

- `.gitignore` 已正确忽略 `__pycache__/`、`runs/`、`artifacts/`
- 但 Git 历史中仍有大量已跟踪噪声文件：
  - 已跟踪缓存/字节码：**116** 个（9 个 `__pycache__` 目录）
  - 当前工作区 `.pyc`：**133** 个
  - 已跟踪 `runs/`：**1841** 个
  - 已跟踪 `artifacts/`：**110** 个
- T5 已固定治理口径：恢复期先分类治理，不执行破坏性清理
- `runs/` 和 `artifacts/` 暂作为历史证据保留，后续需拆分"bootstrap 必需"与"历史归档"
- 物理 cleanup 需后续单开有界任务

---

## 4. 阶段历史与关键结果（P0 → P4）

### 4.1 P0：物理仿真基线确认

**目标**：确认 `full_qec` 与 `simplified` 物理模型在漂移场景下的 LER 差异，建立后续评价基线。

**运行配置**：
- 入口：`benchmark/compare_full_vs_simplified_ler.py`
- 配置：`cnn_fpga/config/experiment_drift.yaml`
- 场景：`linear_low / step_mid / sinusoidal / random_walk`（4 组漂移）
- 每场景：`n_rounds = 2000`，`repeats = 10`

**关键结果**（`runs/drift_suite/drift_v1_20260317_154905_87cc72d5c4de/summary.json`）：

| 场景 | full_qec LER | simplified LER | gap |
|------|-------------|----------------|-----|
| `linear_low` | 0.4237 | 0.0205 | 0.4032 |
| `step_mid` | 0.42665 | 0.01855 | 0.40810 |
| `sinusoidal` | 0.42530 | 0.02405 | 0.40125 |
| `random_walk` | 0.41200 | 0.01565 | 0.39635 |

**意义**：
1. `simplified` 模型平均 LER ~0.02，`full_qec` 平均 ~0.42，gap ~0.40
2. 简化物理模型明显低估逻辑错误率，不能作为工程验证的可信主结论来源
3. 后续所有工程判断必须以更严格的物理口径为准

**验收**：P0 通过。`final_gap_mean > 0` 且跨重复实验标准差可控。

### 4.2 P1：CNN 数据与训练闭环

**目标**：完成从 syndrome 直方图到噪声参数 `(σ, μ_q, μ_p, θ)` 的回归模型训练与量化。

**关键修正（从早期 static_v1 到 static_theta_v2）**：

1. **数据生成修正**（`cnn_fpga/data/dataset_builder.py`）：
   - 早期 `static_v1` 使用各向同性高斯 → `theta_deg` 在统计上几乎不可辨识
   - 改为各向异性高斯：`distribution: anisotropic_gaussian`，`sigma_ratio_p: 0.55`
   - 椭圆高斯噪声点云使得旋转角变得可辨识

2. **训练目标修正**（`cnn_fpga/model/tiny_cnn.py`）：
   - 对 `theta_deg` 引入更高的损失权重（×2.0 vs 其他标签 ×1.0）
   - 避免被 `sigma / mu_q / mu_p` 这些更容易拟合的标签主导训练

**正式模型**：`static_theta_v2` Tiny-CNN
- 浮点模型：`artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- int8 模型：`artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz`
- 配置：`cnn_fpga/config/experiment_static_theta_v2.yaml`

**浮点模型 test 集指标**（`artifacts/reports/static_theta_v2/eval_test_20260319_151756.json`）：

| 标签 | MSE | MAE | R² |
|------|-----|-----|-----|
| `sigma` | — | — | **0.997613** |
| `mu_q` | — | — | **0.996473** |
| `mu_p` | — | — | **0.998459** |
| `theta_deg` | — | — | **0.984862** |
| **全局** | **0.293336** | **0.220503** | **0.994352** |

**int8 模型 test 集指标**（`artifacts/reports/static_theta_v2/eval_test_20260319_151817.json`）：

| 标签 | R² |
|------|-----|
| `sigma` | 0.997730 |
| `mu_q` | 0.996324 |
| `mu_p` | 0.998160 |
| `theta_deg` | 0.984634 |
| **全局** | **0.994212** |

**对照验收阈值**：

| 指标 | 阈值 | 实际（浮点） | 实际（int8） | 通过? |
|------|------|-------------|-------------|-------|
| R²(sigma) | > 0.90 | 0.9976 | 0.9977 | ✓ |
| R²(mu_q, mu_p) | > 0.85 | 0.9965/0.9985 | 0.9963/0.9982 | ✓ |
| R²(theta) | > 0.80 | 0.9849 | 0.9846 | ✓ |
| int8 退化 | < 10% | — | ΔR² = -0.00014 | ✓ |

**验收结论**：**P1 通过**。所有标签 R² 均远超过阈值。int8 与 float 几乎重合。

**TFLite 部署产物**：
- Float `.tflite`：`tiny_cnn_20260319_151717_b87c6c227b57_tflite_20260328_012736.tflite`
- Int8 `.tflite`：`tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_tflite_20260328_012736.tflite`
- 独立验收结果（需 `.venvs/tf311`，当前 Windows 环境不可用）：
  - float .tflite：MSE=0.292359, R²=0.994359, export consistency ok
  - int8 .tflite：MSE=0.297316, R²=0.994192, export consistency ok

### 4.3 P2：硬件行为仿真（无板卡）

**目标**：在软件里模拟硬件行为（固定点量化误差、寄存器延迟、DMA 传输延迟、参数切换原子性），完成公平基线对比。

**关键修正（从早期不公平对比到公平基线）**：

**问题诊断**：早期 P2 对比中 `model_artifact` LER 明显差于旧 `mock`。原因是：
1. 旧 `mock` 直接读取当前窗口的 `target_params`（相当于 oracle + 小噪声），是不公平的"作弊"基线
2. `ParamMapper` 物理含义不合理：旧版 `K = gain * rotation, b = -mu` 有方向错误

**修正措施**：

1. **公平基线重定义**（`cnn_fpga/runtime/slow_loop_runtime.py`, `hardware_emulation.yaml`）：
   - `fixed_baseline`：始终使用固定参数，代表"不自适应"的工程下界
   - `oracle_delayed`：使用延迟一窗的 target 参数，作为公平上界参考

2. **ParamMapper 修正**（`cnn_fpga/decoder/param_mapper.py`）：
   - `K` 改用协方差驱动的对称增益矩阵：`K = C·(C+R)⁻¹`（特征值裁剪）
   - `b` 改用后验线性估计形式：`b = α·(I-K)·μ`
   - `sigma_ratio_p = 0.55` 同步到 P2 物理侧

**正式 P2 对比模式**：`fixed_baseline / oracle_delayed / model_artifact / int8_artifact`

**运行配置**：
- 入口：`cnn_fpga/benchmark/run_p2_mode_benchmark.py`
- 配置：`cnn_fpga/config/hardware_emulation.yaml`
- 场景：`linear_med / step_large / sinusoidal_mid`
- 运行目录：`runs/p2_mode_benchmark/hardware_emulation_v1_20260319_160130_20670d1c0d1f`

**正式 P2 结果**：

| 场景 | fixed_baseline | oracle_delayed | model_artifact | int8_artifact |
|------|---------------|----------------|----------------|---------------|
| `linear_med` | 0.816067 | 0.764239 | **0.696206** | 0.701372 |
| `step_large` | 0.937822 | 0.926650 | **0.731906** | 0.732072 |
| `sinusoidal_mid` | 1.019033 | 1.020556 | **0.759289** | 0.761878 |

**model_artifact 相对公平基线的 LER 下降**：
| 场景 | vs fixed_baseline | vs oracle_delayed |
|------|-------------------|-------------------|
| `linear_med` | **-0.119861** | **-0.068033** |
| `step_large` | **-0.205917** | **-0.194744** |
| `sinusoidal_mid` | **-0.259744** | **-0.261267** |

**float vs int8 差异**：
| 场景 | Δ(int8 - float) |
|------|-----------------|
| `linear_med` | +0.005167 |
| `step_large` | +0.000167 |
| `sinusoidal_mid` | +0.002589 |

**运行统计**（所有模式、所有场景）：
- `commit_count_mean = 7.0`
- `fast_cycle_violation_rate_mean = 0.0`
- `slow_update_violation_rate_mean = 0.0`
- 各模式间 `overflow` 没有异常级放大

**验收结论**：**P2 通过**。
1. 双回路调度正常，stage/commit 无毛刺
2. 时序预算正常，无快/慢回路违约
3. 自适应模型在三组场景下均显著优于 `fixed_baseline` 和 `oracle_delayed`
4. int8 与 float 基本重合

### 4.4 P3：软件 HIL 主线与真实 .tflite

**目标**：端到端软件 HIL 链路验证，真实 .tflite 导出/评测/一致性验收，overflow 来源拆分与输入范围调参。

**P3 软件 HIL 首次打通**（2026-03-28）：
- 入口：`cnn_fpga/benchmark/run_hil_mode_benchmark.py`
- 配置：`cnn_fpga/config/hardware_hil.yaml`
- 对比模式：`fixed_baseline_mock / float_artifact_mock / int8_artifact_mock / real_board`
- 运行目录：`runs/hil_mode_benchmark/hardware_hil_v1_20260328_012839_7b58cdf75d7a`

**首轮 HIL 结果**（旧默认值）：

| 模式 | LER | overflow_rate | n_commits |
|------|-----|---------------|-----------|
| `fixed_baseline_mock` | 1.198702 | 0.297288 | 1500 |
| `float_artifact_mock` | 1.140575 | 0.393561 | 1500 |
| `int8_artifact_mock` | 1.140785 | 0.393397 | 1500 |
| `real_board` | skipped_unavailable | — | — |

**关键发现**：
1. 真实 `.tflite` 路径已打通：`float` 与 `int8` 的 LER 差异约 2.09e-4，overflow 差异约 1.63e-4
2. 自适应模型降低 LER ~0.058, 但 `overflow_rate` 也上升了 ~0.096（参数映射仍偏激进）
3. 调度正常：`slow_update_violation_rate = 0.0`，`fast_cycle_violation_rate = 1.75e-05`

#### 4.4.1 Overflow 来源拆分（2026-03-31）

三类 overflow 诊断被实现到以下文件：
- `cnn_fpga/runtime/fast_loop_emulator.py` — `histogram_input_saturation_rate`
- `cnn_fpga/decoder/linear_runtime.py` — `correction_saturation_rate`
- `cnn_fpga/decoder/param_mapper.py` — `aggressive_param_rate`

首版 smoke（`runs/tmp_smoke_overflow_breakdown_v2/`）：主导来源一致为 `histogram_input`（0.269625），其余两项为 0。

**后续所有正式 benchmark 均确认同一结论**：overflow 主要来自 histogram/syndrome 输入超范围，不是校正量硬饱和或参数过激进。

#### 4.4.2 参数调优（三轮 sweep）

**第一轮 sweep**（gain_clip 上界 0.8/0.75/0.70）：
- 运行目录：`runs/p3_param_sweep/hardware_hil_overflow_tuning_v1_20260320_233531_583cb3f5b61c`
- 结论：单纯压低 gain_clip 上界不能稳定降低 overflow_rate。问题主矛盾不在大增益上限过高

**第二轮 sweep**（gain_clip + beta_smoothing + alpha_bias）：
- 运行目录：`runs/p3_param_sweep/hardware_hil_overflow_tuning_v2_20260320_235250_dfb3515b0016`
- 最优候选：`gain_clip = [0.10, 1.20]`, `beta_smoothing = 0.20`, `alpha_bias = 0.90`
- LER 从 1.069886 降到 1.069234（微降），overflow 从 0.387596 降到 0.386844（微降）
- 虽改进幅度不大，但满足两个条件：(a) 没用更高 LER 换更低 overflow；(b) 改动方向有清晰物理解释

**第三轮 sweep**（新增 gain_scale 旋钮）：
- 运行目录：`runs/p3_param_sweep/hardware_hil_overflow_tuning_v3_20260329_235435_b867c5a238a8`
- `gain_scale` 是真实有效的旋钮，但表现为"越保守 overflow 越低、LER 越差"的单调 tradeoff
- 长确认：`gain_scale=0.97` → LER=1.078498（差于 baseline 的 1.069064）；`gain_scale=0.95` → LER=1.084769
- 结论：当前主线默认值不应改成更小的 gain_scale，保持 `gain_scale = 1.0`

**最终主线默认值**（回写到 `hardware_hil.yaml`）：
```yaml
gain_clip: [0.10, 1.20]
beta_smoothing: 0.20
alpha_bias: 0.90
gain_scale: 1.0
```

#### 4.4.3 输入范围侧定向调参（2026-03-31）

- 运行目录：`runs/p3_histogram_tuning/hardware_hil_histogram_tuning_20260331_001959_bfea868ecd4f`
- 最优组合：`syndrome_limit = 1.441311257912825`, `histogram_range_limit = 1.8799712059732503`, `sigma_measurement = 0.03`
- 相对旧默认 baseline：LER 从 1.069503 降到 1.046678，`histogram_input_saturation_rate` 从 0.387092 降到 0.022181

**更新默认值后的 mode benchmark 复跑**（`runs/hil_mode_benchmark/hardware_hil_v1_20260331_010329_c128fa34262e`）：

| 模式 | LER | overflow_rate | 主导来源 |
|------|-----|---------------|----------|
| `fixed_baseline_mock` | 1.199330 | 0.016157 | histogram_input |
| `float_artifact_mock` | 1.123811 | 0.022764 | histogram_input |
| `int8_artifact_mock` | 1.124699 | 0.022638 | histogram_input |

#### 4.4.4 真实 .tflite 独立验收（2026-03-29/30）

在 `.venvs/tf311` 环境（macOS，当前 Windows 不可用）中完成：
- Float .tflite：MSE=0.292359, R²=0.994359, max_abs_diff=0.119338, export consistency **ok**
- Int8 .tflite：MSE=0.297316, R²=0.994192, max_abs_diff=0.124383, export consistency **ok**

**验收结论**：
- **P3 软件 HIL：通过**（双回路事件推进接通、artifact/int8/真实 .tflite 路径接通、overflow 来源拆分完成、输入范围调参有效、确定性复验完成）
- **P3 真板 HIL：未完成**（缺 `/dev/uio0,/dev/uio1` + 最终 RTL 地址表）

### 4.5 P4：多基线统计对比与正式 benchmark

#### 4.5.1 方案路线切换（2026-04-01）

从"直接回归绝对参数 `(sigma, mu_q, mu_p, theta)`"切换到 `teacher + residual-b`：
- `Window Variance` 或 EKF 先给出稳定 teacher（稳态一阶估计）
- CNN 不再独立承担全部参数辨识
- CNN 只学习对运行时控制偏置 `b` 的小幅残差修正 `delta_b`
- 输入显式包含时间上下文、histogram 差分、teacher prediction 和 teacher 参数

**首轮正式 P4 benchmark**（`runs/p4_benchmark/p4_multiscenario_hybrid_b_v1_20260401_083649_41775bdd90b1_11082/`）：
- 配置：`cnn_fpga/config/p4_multiscenario_hybrid_b.yaml`
- 场景：`static_bias_theta / linear_ramp / step_sigma_theta / periodic_drift`
- 模式：`Window Variance / EKF / Constant Residual-Mu / CNN-FPGA / Hybrid Residual-B`

跨 4 场景平均 LER：

| 模式 | 平均 LER |
|------|----------|
| **Hybrid Residual-B** | **0.850799** |
| Constant Residual-Mu | 0.855549 |
| EKF | 0.855779 |
| Window Variance | 0.857016 |
| CNN-FPGA（旧） | 0.954315 |

Hybrid Residual-B 在 4 个场景中全部最优，且各模式均满足：
- `n_commits_applied` 基本为 600
- `slow_update_violation_rate = 0`
- `correction_saturation_rate = 0`
- `aggressive_param_rate = 0`
- 主导 overflow 来源一致为 `histogram_input`

**风险点**：`static_bias_theta` 场景 Hybrid Residual-B 跨 seed 波动偏大（LER std = 0.005309），需后续复查。

#### 4.5.2 输入范围再提升 + 更长配置复验（2026-04-02）

在冻结 Hybrid Residual-B 模型前提下，进一步放宽输入统计范围：
- `syndrome_limit = 1.566643`
- `histogram_range_limit = 2.255965971`

调参结果（`runs/p3_histogram_tuning/hybrid_b_histogram_tuning_20260401_144804_19d9812a12db/`）：
- LER：1.388708 → 1.375373
- `histogram_input_saturation_rate`：0.021543 → 0.002756

**更长配置正式 P4 复验**（`runs/p4_benchmark/p4_multiscenario_hybrid_b_long_v1_20260402_145451_94eb56a87b59_38723/`）：
- 只比较 3 条主线：`EKF / Constant Residual-Mu / Hybrid Residual-B`

跨 4 场景平均 LER：

| 模式 | 平均 LER |
|------|----------|
| **Hybrid Residual-B** | **0.798807** |
| Constant Residual-Mu | 0.826193 |
| EKF | 0.828108 |

相对次优模式优势扩大到 0.027~0.029。逐场景全部由 Hybrid Residual-B 最优：

| 场景 | Hybrid LER | 领先次优 | histogram_input_sat |
|------|-----------|----------|---------------------|
| `static_bias_theta` | 0.812559 | 0.024062 | 0.002541 (avg) |
| `linear_ramp` | 0.787913 | 0.029804 | — |
| `step_sigma_theta` | 0.787615 | 0.032185 | — |
| `periodic_drift` | 0.807143 | 0.023492 | — |

`static_bias_theta` 的 LER std 从 0.005021 收敛到 **0.000629**（下降了 87.5%），说明之前的波动主要来自输入统计范围过窄导致的窗口观测饱和敏感性。

#### 4.5.3 强 baseline 扩展（2026-04-03）

加入修正后的 UKF 和 RLS Residual-B。

**UKF 修正**（区别于首版 smoke）：
1. 保留 full covariance，不退化为对角协方差
2. 每步更新后做对称化与正定稳定化

结果（`runs/p4_benchmark/p4_multiscenario_strong_baselines_v1_20260403_145747_b82874392710_86447/`）：

| 模式 | 平均 LER |
|------|----------|
| **Hybrid Residual-B** | **0.798332** |
| UKF | 0.817974 |
| Constant Residual-Mu | 0.825719 |
| RLS Residual-B | 0.827908 |
| EKF | 0.828369 |

**关键结论**：
1. 修正后的 UKF 在 4 个正式场景中全部优于 EKF，成为当前最强经典 baseline
2. Hybrid Residual-B 仍保持约 **0.0196** 的平均 LER 优势
3. 各模式 `correction_saturation_rate = 0`，`aggressive_param_rate = 0`，主导 overflow 仍是 `histogram_input`
4. Hybrid 的优势来自对慢回路漂移参数更有效的估计，不是通过更激进控制换来的

#### 4.5.4 Features 正式 ablation（2026-04-04/05，2026-04-07 Windows 迁移后续跑）

结果（`runs/p4_features_summary/features_summary_20260405_145948/`）：

| 变体 | 移除内容 | 平均 LER |
|------|----------|----------|
| **Hybrid Full** | 无 | **0.798355** |
| Hybrid No HistDelta | histogram delta | 0.826422 |
| Hybrid No TeacherPred | teacher prediction | 0.807556 |
| Hybrid No TeacherParams | teacher params | 0.749436 |
| Hybrid No TeacherDelta | teacher deltas | 0.800473 |
| UKF | — | 0.818081 |

**解读**：
1. `histogram delta` 是**关键通道**：去掉后 LER 0.826422，已劣于 UKF (0.818081)
2. `teacher prediction` 有价值但不是唯一关键：去掉后 0.807556，仍优于 UKF
3. `No TeacherParams` 出现**异常优势**（0.749436）：触发后续机制复查路线

#### 4.5.5 No TeacherParams 机制复查（2026-04-05~17）

**离线训练多 seed 复查**（`artifacts/reports/no_teacher_params_training_seed_recheck_20260405.json`）：
- 3 seed 配对重训：No TeacherParams 在 test split 上 MSE 3/3 更好，R² 3/3 更好
- 平均 ΔR² = +0.08576

**Teacher params 通道分析**（`artifacts/reports/teacher_params_coupling_analysis_20260405.json`）：
- 归一化正常，非"数值坏掉"
- `teacher_b_q` 和 `teacher_b_p` 原始标准差很小（~5.48e-4, ~1.02e-3）
- 与 teacher prediction/deltas 高度冗余（可被线性预测到 R²=0.76~0.86）
- 解释：低方差 + 高冗余耦合 → 整平面广播编码与闭环控制语义不匹配

**Formal HIL 多 seed 复查**（裁剪版 benchmark-only, 3 seeds）：

| Seed | Hybrid Full LER | No TeacherParams LER | UKF LER | 胜者 |
|------|----------------|---------------------|---------|------|
| 20260405 | **0.544676** | 0.667064 | 0.829061 | Full |
| 20260406 | 0.828538 | **0.748563** | 0.827337 | NoTP |
| 20260407 | **0.673350** | 0.786860 | 0.826962 | Full |

**最终判断**：No TeacherParams 的 formal HIL 优势不稳定且会随 seed 翻转。离线训练改善 ≠ formal HIL 改善。从此不再讨论"删/不删 teacher params"，转为研究"teacher params 应如何编码"。

#### 4.5.6 Physical Bridge 支线（2026-04-26）

`PB Bound`（物理约束映射）和 `PB ST`（软阈值映射）作为场景特定辅助支线。

第一轮 `Full vs PB Bound` 四场景对照：
- PB Bound 在 `periodic_drift`(0.701 vs 0.718) 和 `step_sigma_theta`(0.727 vs 0.741) 更好
- PB Bound 在 `linear_ramp`(0.778 vs 0.706) 和 `static_bias_theta`(0.752 vs 0.710) 更差

第二轮 `Full vs PB Bound vs PB ST` 动态场景辅助：
- `periodic_drift`：PB ST 最好 (0.682)
- `step_sigma_theta`：PB Bound 最好 (0.716)，PB ST 最差 (0.793)

**结论**：两者都不是跨场景统一更优的表示。`physical_bridge` 仅作为场景特定机制分析支线，不进入主论文排序主线。

---

## 5. 恢复期 Phase 0–1 详细记录（T1–T13）

### 5.1 背景与决策 D-2026-05-05-01

2026-05-05，项目在积累了较完整的 P0-P4 代码路径与实验结果后，仓库存在以下问题：
- 治理文件缺失（无 README/AGENTS/CLAUDE/task_board/handoff/risks）
- 默认 `python 3.13.7` 下缺少 `numpy`，最小 benchmark 无法运行
- 大量历史运行结果和缓存文件混在 Git 中，但缺少 bootstrap 文档

**决策**：项目进入 `Repair`，优先恢复治理、依赖、入口和最小验证。暂不新增实验主线。

### 5.2 Phase 0: Stabilization（T0–T5）

#### T0：冻结 legacy 状态并完成只读审计
- 产出：`docs/00_project_snapshot.md`
- 固定仓库基线事实：主代码存在、治理缺失、环境不可直接运行

#### T1：确认依赖矩阵与最小入口
- 决策 D-2026-05-06-01：固定恢复期解释器为 `C:\ProgramData\anaconda3\python.exe`
  - 确认有 `numpy + yaml`
  - 成功跑通 P0 smoke（`benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot`）
- 发现 `C:\ProgramData\anaconda3\envs\DLEnv\python.exe` 有 `torch`，作为后续训练候选
- 确认 `C:\Python313\python.exe` 不可用（无 numpy）
- 确认工作区内不存在 `.venvs/tf311`

#### T2：跑通最小 P0 smoke benchmark
- 决策 D-2026-05-06-02：固定恢复期 P0 smoke 交接口径
- 产出：`docs/P0_smoke_bootstrap.md`

#### T3：审计 HIL/P4 链路 mock/stub/placeholder 边界
- 决策 D-2026-05-06-03：固定 6 种边界标签
- 产出：`docs/03_hil_p4_boundary_audit.md`、`docs/tasks/P0/T3_hil_p4_boundary_audit.md`
- 统一口径规则：必须显式标注 backend 和 artifact type

#### T4：补软件 HIL 最小 bootstrap/smoke test
- 决策 D-2026-05-06-04：固定恢复期 software HIL 口径为 `mock + model_artifact + artifact_npz + inproc`
- 产出：`cnn_fpga/config/hardware_hil_recovery_smoke.yaml`、`docs/P3_software_hil_bootstrap.md`、`docs/tasks/P0/T4_software_hil_bootstrap_and_smoke.md`
- 首次成功运行：`runs/hil_suite/hardware_hil_recovery_smoke_20260506_021326_3ae9f9176104`
- 关键结果：backend=mock, n_slow_updates_finished=2, n_commits_applied=2, artifact_path=...static_theta_v2...npz

#### T5：清点并处理仓库噪声治理策略
- 决策 D-2026-05-07-01：先分类治理，暂不破坏性清理
- 产出：`docs/06_repo_noise_governance.md`、`docs/tasks/P0/T5_repo_noise_governance.md`
- 噪声分类：A（缓存/字节码）B（runs/）C（artifacts/）D（临时文档文件）
- `.gitignore` 增补 `*.drawio.dtmp`
- 不执行物理清理

### 5.3 Phase 1: Recovery（T6–T13）

#### T6：重新验收 software HIL 最小路径
- 决策 D-2026-05-07-02：最小路径提升为"可复验"
- 使用同一命令二次运行：`runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`
- control-plane 字段一致（backend/artifact_path/n_windows/n_commits）
- LER 和 overflow_rate 存在小幅 run-to-run 差异（随机源未完全控制）
- 表述为"可复验"而非"逐字确定性复现"

#### T7：重新验收 P4 benchmark 最小路径
- 决策 D-2026-05-08-01：固定 P4 recovery smoke 口径
- 产出：`cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`、`docs/P4_benchmark_recovery_bootstrap.md`
- 运行命令：`--scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds`
- 运行目录：`runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308`
- 结果：Static Linear LER=1.009, CNN-FPGA LER=0.721（scenario winner: cnn_fpga）
- 确认 P4 benchmark 显式复用 T6 的 HIL 主链

#### T8：基于 T6+T7 做 Go/Repair gate review
- 决策 D-2026-05-08-02：**Continue Repair**
- 理由：T7 只覆盖 single-scenario+two-mode+repeats=1；根目录缺 manifest；HIL 非确定性
- 产出：`docs/review/T8_gate_review.md`、`docs/tasks/P0/T8_gate_review_and_phase_decision.md`

#### T9：扩 P4 frozen baseline 到四模式单场景
- 决策 D-2026-05-08-03：P4 recovery 证据增强
- 运行命令：`--mode static_linear --mode window_variance --mode ekf --mode cnn_fpga`
- 运行目录：`runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`
- 结果：WV=0.574, EKF=0.680, CNN-FPGA=0.725, SL=0.996（winner: window_variance）
- 四个 mode 都确认 `backend=mock, inference_service_mode=inproc`

#### T10：基于 T8+T9 二次 gate review
- 决策 D-2026-05-08-04：**Continue Repair**
- 理由：仍缺 manifest；HIL 仍非确定性；P4 仍是 smoke 而非正式
- 产出：`docs/review/T10_gate_review.md`、`docs/tasks/P0/T10_gate_review_after_t9.md`

#### T11：补 recovery 期最小依赖 manifest
- 决策 D-2026-05-08-05：新增 `requirements-recovery.txt`
- 内容：`numpy` + `PyYAML`
- 覆盖范围：P0/P3/P4 recovery smoke
- 不覆盖：torch/tensorflow/tflite-runtime/matplotlib
- 同步更新：README.md, P0/P3/P4 bootstrap 文档

#### T12：收敛 software HIL 随机源与确定性表述
- 决策 D-2026-05-08-06：确定性复验
- 代码修改：
  - `physics/syndrome_measurement.py`：`RealisticSyndromeMeasurement` 支持显式 `rng`
  - `cnn_fpga/runtime/fast_loop_emulator.py`：快回路误差 RNG 与测量噪声 RNG 分离（seed + 1）
- 两次连续复验：
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104`
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104`
- 文件级 SHA256 对比：`hil_summary.json` 一致，`hil_events.json` 一致
- 共同结果：LER=0.454375, overflow_rate=0.002
- 表述升级为"逐字一致复验"

#### T13：Recovery exit review 并完成阶段收尾
- 决策 D-2026-05-08-07：**Allow**（退出恢复期）
- 产出：`docs/review/T13_recovery_exit_review.md`、`docs/tasks/P0/T13_recovery_exit_and_closeout.md`
- 项目从 `Repair` 切换为 `Go`，退出 Phase 1，进入 Phase 2
- 关键判断：这里的 Go 只代表"允许继续做 bounded 开发任务"，不代表真板/.tflite/正式多场景 benchmark 已恢复

### 5.4 恢复期核心成果总结

| 成果类型 | 具体内容 |
|----------|----------|
| 确定性复验 | `mock + model_artifact + artifact_npz + inproc` 路径逐字一致，两次 run SHA256 一致 |
| 边界清晰 | 6 种边界标签固定（software_hil_orchestrator/mock_backend/placeholder_real_board_backend/p4_wrapper_over_hil/true_tflite_or_stub_export/true_tflite_or_stub_runtime） |
| 依赖 manifest | `requirements-recovery.txt`（numpy+PyYAML），作用域诚实 |
| 治理文件 | task_board/decision_log/handoff/risks 均建立并维护 |
| Bootstrap 文档 | P0/P3/P4 三条 recovery smoke bootstrap 均可直接复用 |
| 种子链 | recovery 路径的完整 RNG seed 链已明确（noise provider: seed+17, driver: seed+7, slow loop: seed+31, latency: seed+43 等） |

---

## 6. 当前环境与依赖矩阵

### 6.1 可用解释器分工

| 路径 | 关键包 | 角色 | 验证状态 |
|------|--------|------|----------|
| `C:\ProgramData\anaconda3\python.exe` | `numpy`, `PyYAML` | **P0/P3/P4 recovery smoke** | T12 逐字一致复验通过 |
| `C:\ProgramData\anaconda3\envs\DLEnv\python.exe` | `numpy`, `PyYAML`, `torch` | **训练候选**（PyTorch/CUDA 后端） | T1 确认可用 |
| `C:\Python313\python.exe` | 无 numpy | **不可用** | T1 已排除 |

### 6.2 依赖矩阵详情

**Recovery smoke（`requirements-recovery.txt`）**：
```
numpy
PyYAML
```
- 覆盖范围：`benchmark/compare_full_vs_simplified_ler.py --no-plot`、`run_hil_suite`（inproc + artifact_npz）、`run_p4_multiscenario_benchmark`（inproc + artifact_npz）
- 故意命名为 `requirements-recovery.txt` 而非 `requirements.txt`：避免被误读为"完整仓库环境已恢复"
- 不覆盖：
  - `matplotlib`（仅在去掉 `--no-plot` 时触发）
  - `torch`（训练链，依赖 DLEnv）
  - `tensorflow` / `tflite-runtime`（.tflite 导出/推理）
  - `real_board` HIL backend（需 `/dev/uio*` 设备节点 + RTL 地址表）

**训练链**：依赖 `DLEnv` 环境（`C:\ProgramData\anaconda3\envs\DLEnv\python.exe`），含 `torch`。当前 `tiny_cnn.py` 支持 `--backend numpy|torch`，`--device cuda`。

**.tflite 路径**：
- 代码支持：`cnn_fpga/model/export.py`（真导出 + stub 回退）、`cnn_fpga/model/evaluate_tflite.py`（独立评测）、`cnn_fpga/model/validate_export.py`（一致性验收）
- 历史运行环境：`.venvs/tf311`（macOS, python=3.11.15, tensorflow=2.21.0）
- 当前 Windows 环境：**不可用**（`.venvs/tf311` 在工作区不存在）。若要恢复 .tflite 验收能力，需先建立 TensorFlow/TFLite 运行环境

### 6.3 最小运行命令

```powershell
# === 恢复期 smoke（必须在 AConda 下运行）===

# P0: full_qec vs simplified 最小对比（~1 分钟）
& 'C:\ProgramData\anaconda3\python.exe' benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test

# P3: software HIL 最小 smoke（~秒级，逐字一致复验已确认）
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml

# P4: frozen baseline 单场景四模式 smoke（~分钟级）
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds

# === 正式开发级命令（需要 DLEnv 或更完整环境）===

# P4: 强 baseline 多场景对比（需要 torch + 完整训练 artifact）
# python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ekf --mode ukf --mode constant_residual_mu --mode rls_residual_b --mode hybrid_residual_b --paired-seeds --repeats 2

# Teacher-representation paired benchmark
# python -m cnn_fpga.benchmark.run_p4_teacher_representation_paired --config cnn_fpga/config/p4_teacher_repr_mid.yaml --v1-config .../v5.yaml --v2-config .../Full.yaml --scenario linear_ramp --scenario periodic_drift --repeats 2 --seed 20260427 --seed 20260428
```

### 6.4 Recovery Smoke 固定口径

| 参数 | 固定值 |
|------|--------|
| `hil.backend` | `mock` |
| `slow_loop.mode` | `model_artifact` |
| `inference_service.mode` | `inproc` |
| `inference_service.backend` | `artifact_npz` |
| `model_artifact.path` | `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz` |
| 解释器 | `C:\ProgramData\anaconda3\python.exe` |
| HIL 配置 | `cnn_fpga/config/hardware_hil_recovery_smoke.yaml` |
| P4 配置 | `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml` |

---

## 7. 工程真实性边界表

### 7.1 主边界矩阵

| 组件 | 边界标签 | 当前真实状态 | 禁止表述 |
|------|----------|-------------|----------|
| `run_hil_suite.py` + `hil.backend=mock` | **`software_hil_orchestrator`** | 真实可运行的 software HIL orchestration。通过 `hil.backend` 选择 backend，产出 `hil_events.json`/`hil_summary.json` | — |
| `mock_fpga.py` | **`mock_backend`** | event-driven FPGA 行为仿真。维护 DMA/param-bank 语义，`metadata={"backend": "mock_fpga"}` | "真板运行结果"、"FPGA 实现已完成" |
| `board_backend.py` + `fpga_driver.py` | **`placeholder_real_board_backend`** | placeholder 真板骨架。`schedule_commit()` 返回大量 `None`；`step()` 返回空事件。`fpga_driver.py` 中 `board/real` 标为 reserved for future integration | "真板 HIL 已完成"、"真板 backend 已验收"、"板级 I/O 已验证" |
| `run_p4_multiscenario_benchmark.py` | **`p4_wrapper_over_hil`** | 直接调用 `run_hil_session()`，不绕开 HIL backend 边界。P4 的真实性继承自 HIL 链路 | "P4 有独立于 HIL 的更高真实性" |
| `export.py` (TFLite 导出) | **`true_tflite_or_stub_export`** | 优先真 `.tflite` 导出，失败回退 `tflite_stub_v1`（manifest 格式 `.tflite.json`） | "TFLite 导出已完成"（不声明真/stub） |
| `inference_service.py` (TFLite 推理) | **`true_tflite_or_stub_runtime`** | stub 路径 `source="tflite_stub_service"`；真路径 `source="tflite_service"` | "TFLite 已部署"（不区分 source） |
| `cnn_fpga/config/hardware_hil_recovery_smoke.yaml` | **`bounded_recovery_smoke`** | T12 逐字一致复验的 bounded 路径 | "这条路径等于正式 HIL" |
| `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml` | **`bounded_recovery_smoke`** | T9 四模式单场景 smoke | "正式 P4 frozen benchmark 已恢复" |

### 7.2 统一口径规则

1. 可以说"P3 software HIL 主链存在"，但必须同时标注 `hil.backend` 和 inference artifact type
2. 不能写"real-board HIL 已完成"或"真板 backend 已验收"，除非后续有独立真板证据覆盖 `board_backend.py` 当前占位状态
3. 不能把"P4 benchmark"写成比 `run_hil_session()` 更真实的一条独立执行链
4. 不能因为配置里写了 `backend=tflite` 就默认它是真实 TFLite 部署；必须区分 `tflite_service` vs `tflite_stub_service`
5. recovery smoke 的结论严格限定在 `mock + model_artifact + artifact_npz + inproc` 边界内，不外推

### 7.3 工程仿真补强缺口总览

| 方向 | 当前状态 | 主要代码入口 | 优先级 | 说明 |
|------|----------|-------------|--------|------|
| 多类物理噪声接入 | `noise_channels.py` 已实现但未接入 P2/P3/P4 主线 | `physics/noise_channels.py`、`cnn_fpga/benchmark/run_hil_suite.py` | **高** | 建议做"两层噪声模型"桥接而非直接替换 |
| 延迟模型负载耦合 | 独立高斯/常数抽样，无负载条件偏置 | `cnn_fpga/runtime/latency_injector.py`、`cnn_fpga/runtime/scheduler.py` | **高** | pending_windows → 均值/方差条件偏置 |
| 慢回路故障模型细化 | 独立伯努利注错，无状态机/恢复/重试 | `cnn_fpga/benchmark/run_hil_suite.py`、`cnn_fpga/runtime/slow_loop_runtime.py` | **高** | normal→retrying→degraded→dropped |
| 快回路 bit-accurate 固定点 | 接近硬件近似，非逐级位宽精确 | `cnn_fpga/runtime/fast_loop_emulator.py`、`cnn_fpga/model/quantize.py` | **高** | 先出位宽规范，再逐级实现 |
| syndrome 读出链 ADC/AFE | 统计化测量模型，非完整模拟/数字读出链 | `physics/syndrome_measurement.py` | **中高** | 轻量 ADC bits/full_scale/gain/offset |
| 逻辑错误定义扩展 | 有效模型口径，非完整电路级容错 | `physics/logical_tracking.py` | **中** | 先抽样双口径对照 |
| 板级 I/O 语义补全 | `board_backend.py` 仍是 placeholder | `cnn_fpga/hwio/board_backend.py`、`cnn_fpga/hwio/fpga_driver.py` | **中（条件性）** | 真板条件具备前不重点投入 |

---

## 8. Teacher-Representation 分支谱系（v1–v9）

### 8.1 起点：从 AbsReg 到 Residual-B

- **早期（P1-P2）**：CNN 直接回归 `(sigma, mu_q, mu_p, theta)` 绝对物理参数 → `ParamMapper` 映射为 `(K, b)`
  - 问题：离线预测目标与运行时控制语义错位
- **Residual-Mu 版本**：CNN 输出 `delta_mu` → `mu_next = teacher_mu + delta_mu`
  - 已有改善，但离线指标塌缩（输出接近常数）
- **Residual-B 版本（Full / Hybrid Residual-B）**：CNN 输出 `delta_b` → `b_next = teacher_b + delta_b`
  - 更贴近闭环控制语义
  - 输入：5 窗口 histogram + histogram deltas + teacher prediction/params/deltas（整平面广播）

### 8.2 branch 起点：No TeacherParams 异常信号

2026-04-05 的 features ablation 中发现 `Hybrid No TeacherParams` LER=0.749436，显著优于 `Hybrid Full` LER=0.798355。这触发了后续 teacher-representation 重编码路线。

### 8.3 早期 gated 系列（v1–v4）

- **v1**：首版 scalar-branch + gated 注入。保留整包 teacher 标量，但改为低维注入而非平面广播
- **v2**：调整 scalar 归一化与 gate 结构。单轮动态场景有正信号但 coverage 不完整
- **v3**：调 gate init 和 scalar feature weights
- **v4**：仅小幅优于 Full。`aggressive_param_rate` 偏高

### 8.4 Gated v5（当前最强候选）

**设计特征**：
- 仅保留 4 个与 `residual-b` 最直接相关的 teacher 标量：
  - `teacher_b_q`、`teacher_b_p`
  - `teacher_delta_b_q`、`teacher_delta_b_p`
- 通过 `scalar_branch + gated` 低维注入主干
- 不恢复整包 teacher params broadcast
- 输出：`delta_b_q`, `delta_b_p`（2 维）

**三 seed / 四场景 paired benchmark（2026-04-27）**：

| Seed | Full LER | Gated v5 LER | Gap | 胜者 |
|------|----------|-------------|-----|------|
| 20260427 | 0.779861 | **0.547688** | -0.232173 | Gated v5 |
| 20260428 | 0.798706 | **0.710131** | -0.088574 | Gated v5 |
| 20260429 | **0.688990** | 0.674559 | -0.014432 | Full（Gated v5 接近持平） |

按场景跨 seed 汇总：

| 场景 | Full LER | Gated v5 LER | 胜者 |
|------|----------|-------------|------|
| `static_bias_theta` | 0.751062 | **0.637209** | Gated v5 |
| `linear_ramp` | 0.764402 | **0.631810** | Gated v5 |
| `step_sigma_theta` | 0.759205 | **0.638351** | Gated v5 |
| `periodic_drift` | 0.748741 | **0.669133** | Gated v5 |

**Chunked pair 复验（2026-04-28，新 runner，3 seed × 4 scenario × 2 repeats）**：
- Coverage = 100%，hil_errors = 0
- 跨 seed 均值：`Full = 0.758829`, `Gated v5 = 0.618195`, gap = **-0.140634**
- Gated v5 在 12 个 seed-scenario 对照中赢 **9 个**
- seed=20260429 四场景均值持平/略差（Full=0.637, Gated v5=0.640, gap=+0.002）

**Teacher scalar 诊断**（来自 teacher_scalar_diagnostics.csv）：
- `teacher_b_p`：平均贡献最大（ablation_l2=0.137, gate_delta_l2=4.05）
- `teacher_b_q`：次之（ablation_l2=0.067, gate_delta_l2=3.09）
- `teacher_delta_b_q`：较小（ablation_l2=0.023）
- `teacher_delta_b_p`：最小（ablation_l2=0.015）

**定位**：当前最强 teacher-representation 候选。已足够说明"少数关键 teacher 标量 + 低维 gated 注入"方向正确。尚未完全替代 Full（seed=20260429 持平），但论文中可作为 teacher-representation 重编码的主分析版本。

### 8.5 Gated v6/v7（gated 微调期）

- **v6**：在 v5 基础上调 gate init bias、scalar clipping。收益不明显
- **v7**：调 residual scale、residual clip。对 v6 有改进但未超过 v5 的最优 seed

### 8.6 Gated v8/v9（过冲控制尝试 → 失败）

**v8 设计目标**：
- 保持 `aggressive_param_rate = 0`
- 比 v7 增强 `b_q / b_p` 主支路有效贡献
- 抑制 20260429 上的翻车

结果（`runs/teachrepr_v8_bench_pair/paired_20260502_014626/summary.csv`）：
- 对 v7 有改进，但没有稳定超过 Full
- 不同 seed 间不够稳

**v9 设计目标**：
- 进一步抑制 v8 的过冲风险
- 更保守的 gate 和 residual 约束

结果（`runs/teachrepr_v9_pair/paired_20260502_115405/summary.csv`）：

| Seed | Full | v8 | v9 |
|------|------|----|----|
| 20260427 | 0.836762 | **0.737844** | 0.806988 |
| 20260428 | 0.828015 | **0.738263** | 0.819800 |
| 20260429 | **0.532874** | 0.766638 | 0.833991 |
| **平均** | **0.732550** | **0.747582** | **0.820259** |

v9 三 seed 均值退化到 0.820，甚至不如 Full (0.733)。teacher 分支被压得过弱，收益流失。

**v8/v9 的共同教训**：
- 单纯微调 `scalar_gate_init_bias / scalar_norm_clip / scalar_feature_weights / residual_clip_b / residual_scale_b` → 边际收益极低
- Teacher 分支更激进 → 部分 seed 翻车；更保守 → 收益消失
- 当前瓶颈已不是"超参还没拧对"，而是"表征方式和闭环目标本身不够对"

### 8.7 No TeacherParams 的最终判断（2026-04-17）

**离线训练**：3 seed 配对重训中 No TeacherParams 稳定更好（ΔR² = +0.086）
**Formal HIL**：3 seed 中结论随 seed 翻转（1 seed Full 胜，1 seed NoTP 胜，1 seed Full 胜）

**最终判断**：No TeacherParams 的 formal HIL 优势不稳定且会随 seed 翻转。不能作为稳定更优正式主线。真正需要解决的**不是"删不删 teacher params"，而是"teacher params 应如何编码进闭环模型"**。

### 8.8 不再建议继续投入的方向

以下是已被多轮实验基本证伪或边际收益极低的方向：

1. **继续做 gated v10/v11/v12**：v8→v9 已经证明继续微调 gate/clip/scale 很难再带来实质性提升
2. **继续做"删/不删 teacher params"大规模长跑**：3 seed formal HIL 已确认结论会翻转
3. **继续扩 PB Bound / PB ST 为论文主线**：两种映射都不是跨场景统一更优的表示
4. **继续把 No TeacherParams 当成主叙事**：formal HIL 证据不支持

---

## 9. 稳定结论清单

以下按可信度从最稳到需谨慎使用排列。

### 9.1 安全可写进论文和文档的结论

| # | 结论 | 证据强度 | 适用边界 |
|---|------|----------|----------|
| 1 | **Hybrid Residual-B 是当前正式主线方案**，稳定优于 EKF 和 UKF | 强 | 当前 4 个正式场景、长配置、多 seed |
| 2 | **UKF 是当前最强经典 baseline**（修正后：full covariance + 对称化/正定稳定化） | 强 | 4 场景全面优于 EKF |
| 3 | **优势不是来自更激进控制**：correction_saturation_rate=0, aggressive_param_rate=0 | 强 | 所有正式 benchmark 一致 |
| 4 | **主导 overflow 是 histogram_input**，非控制参数饱和或校正饱和 | 强 | 所有正式 benchmark 一致，被 overflow 拆分明确证实 |
| 5 | **输入统计范围是真实有效旋钮**：放宽不恶化 LER，显著压低 overflow | 强 | 多轮调参 + 正式复验验证 |
| 6 | **float/int8 差异在各阶段都极小**（ΔLER ≤ 0.006，ΔR² ≤ 0.0002） | 强 | P1 模型评估 + P2/P3 HIL |
| 7 | **离线训练改善 ≠ formal HIL 改善** | 强 | No TeacherParams + 多个 gated 版本反复验证 |
| 8 | **histogram delta 是关键输入通道** | 强 | 去掉后 LER 劣于 UKF |
| 9 | **teacher params 的核心问题不是数值坏掉，而是编码方式**：低方差 + 高冗余 + 整平面广播 | 强 | 通道分析 + Gated v5 对比 |
| 10 | **Gated v5 方向正确**：少数关键 teacher 标量 + 低维 gated scalar branch | 中强 | 3 seed 中 2 seed 显著更好 |

### 9.2 需附限定条件的结论

| # | 结论 | 限定条件 |
|---|------|----------|
| 1 | "Hybrid Residual-B 优于 UKF ~0.0196" | 在正式 4 场景 + 当前默认参数下成立；优势幅度 seed 依赖 |
| 2 | "Gated v5 优于 Full" | 3 seed 中 2 seed 显著更好，1 seed (20260429) 持平/略差 |
| 3 | "No TeacherParams 离线更好" | 离线训练指标确实稳定更好；但这一优势不能外推到 formal HIL |
| 4 | "当前 overflow 主导来自 histogram_input" | 基于当前默认输入范围；在极端噪声或不同范围设置下可能改变 |

### 9.3 当前不能写的结论

1. ❌ "CNN 全面优于所有经典解码器"
2. ❌ "项目提出了通用于所有量子纠错码的统一最优方法"
3. ❌ "已完成完整真实 FPGA 部署并实现工业级可用"
4. ❌ "No TeacherParams 是稳定更优正式主线"
5. ❌ "teacher params 对模型有害，应该全部删除"

---

## 10. 论文撰写与投稿路线

### 10.1 推荐标题方向

**最稳妥的工作标题**（折中系统 + 方法）：
> A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction

**备选方向**：
- 工程系统型：`A Runtime-Consistent CNN-FPGA Adaptive Decoder for Drift-Aware GKP Error Correction`
- 方法机制型：`Teacher-Guided Residual Correction for Runtime-Consistent Adaptive GKP Decoding`

### 10.2 核心论文主张

> 在实时硬件约束下，保留稳定的经典 teacher，并让轻量 CNN 仅学习对运行时控制偏置有用的残差修正。在运行时一致的数据构造、双回路调度和软件 HIL 约束下，该 teacher-guided residual-b 方案能够稳定超过当前最强经典自适应 baseline UKF。

### 10.3 贡献点（三点式，当前最稳）

1. **双回路实时解码框架**：将快回路低时延解码、窗口统计累积、参数 bank 切换、慢回路推理和 HIL 验证组织为同一条可复现实验链路。使研究问题从"离线模型精度"提升为"部署语义一致的在线解码系统"。

2. **Teacher-guided residual-b 学习方案**：相比直接回归绝对物理参数，更贴近在线解码控制语义。在当前正式多场景 benchmark 中稳定优于 EKF、UKF、RLS 和常数残差补偿等经典基线。在正式口径下 Hybrid Residual-B 平均 LER=0.798332，最强经典 baseline UKF=0.817974。

3. **系统化工程验证闭环**：包括 float/int8/TFLite artifact 验证、overflow 来源定位与输入统计范围修正、teacher/context/features ablation、软件 HIL 闭环验证。说明该方法并非单一模型技巧，而是一套可落地的工程方法。

### 10.4 投稿目标优先排序

| 优先级 | Venue | 类型 | 当前匹配度 | 投稿前需补内容 |
|--------|-------|------|-----------|---------------|
| **1** | **QCE** (IEEE Quantum Week) | 会议 | **高** | features 正式结果回填 + 机制分析 + 工程代价分析 |
| **2** | **TQE** (IEEE Trans. Quantum Engineering) | 期刊 | **高** | 更完整 engineering cost + 多 seed 稳定性复验 |
| **3** | **EPJ Quantum Technology** | 期刊 | **中高** | 更完整 ablation + 更扎实机制解释 |
| 4 | **QST** (Quantum Science and Technology) | 期刊 | 中 | 更强 baseline + 真板/跨码扩展 |
| 5 | **npj Quantum Information** / **ACM TQC** | 期刊 | 中低 | 需显著补强后冲 |
| 6 | **FCCM** / **ACM FPGA** / **ICCAD** | 会议 | 低（当前） | 需真实 FPGA 综合/资源/时延报告 |

### 10.5 论文结构提纲

1. **Introduction**（~2 页）
   - GKP 在线纠错为什么需要漂移自适应？
   - 为什么单纯离线回归精度不足以说明在线可用性？
   - 实时硬件约束下，什么才是合理的解码问题定义？
   - 核心主张：学习模型不应替代 classical teacher，而应作为满足运行时语义的 residual corrector

2. **Background and Problem Formulation**（~3 页）
   - GKP syndrome 与漂移参数背景
   - 快回路与慢回路的时间尺度差异
   - 在线解码的部署约束（延迟/参数切换/量化）
   - 为什么 `(K, b)` 才是运行时真正执行的目标（非 `σ,μ,θ`）

3. **Dual-Loop Runtime-Consistent Decoding Framework**（~4 页）
   - Fast loop：线性解码 + 直方图累积 + fixed-point/cycle budget
   - Slow loop：histogram window + teacher estimation + residual model inference + param mapping
   - Parameter update protocol：param bank + stage/commit + atomic switch + latency/scheduler
   - HIL-consistent execution：mock FPGA + software HIL + artifact runtime

4. **Runtime-Consistent Learning Formulation**（~3 页）
   - 为什么 absolute parameter regression 不够理想
   - 为什么 target 从 `delta_mu` 切到 `delta_b`
   - 为什么 teacher-guided residual 更符合部署语义
   - 输入构成：histogram history + histogram deltas + teacher prediction + teacher params + teacher deltas
   - 建议配一张结构图

5. **Experimental Protocol**（~2 页）
   - 正式场景集：`static_bias_theta / linear_ramp / step_sigma_theta / periodic_drift`
   - Baseline 集合：EKF / UKF / Constant Residual-Mu / RLS Residual-B / Hybrid Residual-B
   - 评价指标：LER / overflow_rate / histogram_input_saturation / commit / violation rates
   - repeats / seed / 固定协议

6. **Main Results**（~4 页）
   - 强 baseline 主结果表（Hybrid > UKF > EKF/RLS/Constant）
   - 优势不是 saturation 换来的
   - 分场景结果表
   - overflow 来源分析

7. **Mechanism Analysis and Ablation**（~4 页）
   - Teacher ablation：WV vs UKF teacher
   - Context ablation：Ctx1/3/5
   - Features ablation：No HistDelta / No TeacherPred / No TeacherParams / No TeacherDelta
   - Gated v5 teacher-representation re-encoding

8. **Engineering Considerations**（~3 页）
   - float / int8 / TFLite artifact 一致性
   - latency budget 与 violation rates
   - stage/commit 行为统计
   - overflow source breakdown
   - board backend 当前状态与限制

9. **Discussion**（~2 页）
   - 为什么更像是"对 teacher 偏差的残差修正"而不是"全面替代经典解码器"
   - 未来扩展到 concatenated GKP-surface
   - 与 MWPM / Blossom 的关系
   - 当前限制

10. **Conclusion**（~1 页）

### 10.6 论文图表建议

| 图表 | 内容 |
|------|------|
| 图 1 | 双回路架构图（fast loop + slow loop + param bank + HIL） |
| 图 2 | Runtime-consistent learning pipeline 图 |
| 图 3 | 主结果柱状图（5 模式平均 LER + 误差棒） |
| 表 1 | 分场景结果表（5 mode × 4 scenario） |
| 表 2 | Features ablation 表 |
| 表 3 | Teacher ablation 表 |
| 表 4 | 工程开销表（float/int8/延迟/overflow/commit） |

### 10.7 投稿前建议补齐的关键证据

1. **features ablation 正式结果**：已有一轮（2026-04-05），但建议补到覆盖更多 seed
2. **机制解释链书面化**：argue 清楚 absolute regression → residual-b → Gated v5 的递进逻辑
3. **工程代价分析**：推理延迟分布、commit 成功率、参数更新频率、float/int8 精度
4. **至少一个关键场景的多 seed (≥5) 稳定性复验**
5. **Gated v5 vs Full 的逐 window 对比**（用于机制解释）

---

## 11. 后续开发优先级与候选任务包

### 11.1 第一优先级：失败机理诊断

**问题**：Gated v5 在多数 seed（20260427/20260428）上大幅优于 Full（gap 0.09~0.23），但在 20260429 上接近持平（gap 0.014）甚至 chunked pair 复验中略差（gap +0.002）。为什么？

**建议产出**：
1. `Full vs Gated v5` 逐 window / 逐 commit 时间序列对照（4 场景 × seed=20260429）
2. 关键量可视化脚本：
   - `teacher_b_q / teacher_b_p` 时间序列
   - CNN 预测的 `delta_b` 时间序列
   - commit 后实际使用的 `b_q / b_p` 时间序列（teacher_b + delta_b）
   - correction 幅度时间序列
   - overflow / saturation 时间序列
   - LER 累积曲线
3. 翻车机制判断矩阵：符号偏移 / 幅度过冲 / 响应滞后 / teacher 本身不稳 / gated 分支过保守
4. 诊断报告文档

**执行方式**：
- 单 seed（20260429）、全 4 场景
- 使用已有结果文件（不需要重跑 benchmark）
- 新增诊断分析脚本（只读已有 JSON，做可视化与统计分析）
- 不在该任务中改模型或重跑长 benchmark

**价值**：直接决定后续应该重写 loss、换 teacher 表征、做更强约束、还是补 teacher 估计质量。

### 11.2 第二优先级：Paper-Inspired 分支

**设计草案**：[docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md](docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md)

**分支名**：`paper_inspired_statcalib_v1`

**核心改动**（建立在 Gated v5 之上）：

1. **输入侧**：在 Gated v5 的 scalar_branch 中新增 compact summary 标量
   - Histogram summary（~6 个）：当前窗口能量、5 窗能量均值/std、质心漂移 q/p、各向异性强度
   - Teacher stability summary（~3 个）：`||delta_b||`、5 窗 b 变化量均值/std
   - 不恢复整包 teacher params broadcast

2. **模型侧**：新增 light stat summary branch（小 MLP），用 dual-gate 融合
   - spatial branch (histogram) + teacher scalar branch (gated) + stat summary branch (小 MLP)
   - 融合：`hidden = relu(base_hidden * teacher_gate * stat_gate + teacher_shift + stat_shift)`

3. **Loss 侧**：三项组合
   - 主损失：`L_residual = MSE(delta_b_pred, delta_b_target)`
   - 闭环目标：`L_bnext = MSE(teacher_b + delta_b_pred, target_b)`（权重 λ1，较小）
   - 风险代理：`L_risk = soft_clip_penalty + smooth_penalty`（权重 λ2，更小）

**推荐 benchmark 顺序**：
- 第一轮：2 seeds × 2 dynamic scenarios（linear_ramp, periodic_drift）× repeats=2，对标 Gated v5
- 第二轮（若第一轮方向正确）：3 seeds × 4 scenarios × repeats=2
- 第三轮（若第二轮仍成立）：中间长度 paired benchmark-only

**原则**：
- 优先验证方法论对不对，不是堆模型容量
- 不改 teacher 模式、benchmark 场景集合、ParamMapper 口径、快回路控制语义
- 不与 Gated v5 同时做大规模对照——先单独验证新分支

### 11.3 第三优先级：工程仿真补强

按性价比排序的 5 个可独立推进的子任务：

#### E1: noise_channels → effective parameters 离线桥接
- 新增：`physics/noise_effective_mapper.py`
- 新增：`cnn_fpga/data/build_physical_augmented_dataset.py`
- 产出：`physical_noise_lookup.json` + `physical_noise_samples.npz`
- 不改当前主线数据生成方式——新增 `distribution: physical_mixed_channels` 分支

#### E2: load-aware latency injector v1
- 简单条件偏置：`pending_windows > 0` → DMA/inference/writeback 均值抬高
- 连续多窗积压 → 方差也抬高
- timeout/retry → 下次写回和 commit 延迟增加
- 输出：轻载/重载/拥塞三种状态统计

#### E3: stateful fault injector v1
- 新增：`cnn_fpga/runtime/fault_injector.py`
- 状态机：normal → retrying → degraded_hold_last → dropped_update → service_stalled
- 新指标：`stale_param_window_count` / `retry_count` / `consecutive_fail_max` / `fallback_ratio`

#### E4: bit-accurate control pipeline
- 先出位宽规范文档：`docs/CNN_FPGA_GKP_固定点与位宽规范.md`
- 再实现 shadow pipeline：`cnn_fpga/runtime/fixed_point_pipeline.py`
- 逐级比较 float reference / approximate fixed-point / bit-accurate fixed-point

#### E5: ADC/AFE 轻量读出链
- 新增：`physics/readout_chain.py`
- 最小版本：gain / offset / analog noise / ADC bits / full_scale / quantization + saturation
- 接入 fast loop 输入侧，比较当前统计模型 vs ADC/AFE 模型

### 11.4 第四优先级：文档与治理

1. 把 teacher-representation 结论系统回写到阶段结论文档
2. 补训练链独立 manifest（`requirements-train.txt` 或 `pyproject.toml`）
3. 补 .tflite 路径独立 manifest / smoke（前提：先恢复 .tflite 运行环境）
4. 单开有界 cleanup 任务处理 `__pycache__/` / `.pyc` 的物理移除（列出 manifest、回滚方式、验收标准）

### 11.5 当前不应启动的任务

- ❌ 新的 teacher-representation 长跑（在机理诊断完成前）
- ❌ 长时间 P4 正式多场景 frozen benchmark（在 Gated v5 稳定性确认前）
- ❌ 真板 backend 能力扩写
- ❌ 大规模 repo cleanup（需先定义 bootstrap 必需 vs 历史归档）
- ❌ 论文正文正式写作（在关键证据补齐前）
- ❌ 继续微调 gated v10+

---

## 12. 治理规范：AI Coding 工作流

本项目遵循 [docs/reference/AI_coding_workflow.md](docs/reference/AI_coding_workflow.md) 中定义的 AI Coding 工作流。

### 12.1 项目决策状态机

本项目的决策状态只能是以下之一：
- **Go**：允许继续做 bounded 开发任务（**当前状态**，自 T13 起）
- **Narrow**：缩小范围（出现超出预期的重大风险时）
- **Pause**：暂停（环境/依赖/硬件条件不具备时）
- **Archive**：归档（项目目标已达成或方向放弃）
- **Repair**：恢复期（已完成，2026-05-05 至 2026-05-08）

状态切换必须通过正式的 gate review 或 recovery exit review。

### 12.2 角色分工

| 角色 | 工具 | 职责 |
|------|------|------|
| **Project Manager** | 人类 | 最终裁决者。批准立项、选择任务、判断 review 意见、决定状态切换 |
| **Codex Captain** | AI（如本会话的接任者） | 项目开发主控。维护任务板/交接/风险文档、拆解任务、整合结果、**不直接做大规模实现** |
| **Codex Worker** | AI | 单任务实现。只做 Captain 指定的任务、只改允许的文件、运行验证、汇报结果 |
| **Claude Code Reviewer** | AI | 只读审查。检查伪实现/mock/stub/hardcode/过度工程。输出 PASS/PASS_WITH_WARNINGS/BLOCK |

### 12.3 Captain 每轮必须输出

```text
1. 当前唯一任务（Task ID + Goal）
2. 为什么现在做它（Why now）
3. Worker 任务包（包含以下字段）
4. 允许修改的文件范围（Allowed files）
5. 禁止做的事（Forbidden scope）
6. 验证命令或验收标准（Verification）
7. 完成后需要更新的治理文件（Docs to update）
```

### 12.4 Worker 任务包模板

```text
Task ID:       T<编号>
Goal:          一句话目标
Why now:       为什么当前时机做（而不是别的任务）
Allowed files: 具体文件路径列表
Forbidden scope: 明确不能碰的文件/模块/语义
Inputs to read:  必须在执行前阅读的文件
Expected output: 可验证的产出物
Verification:    验证命令或标准
Docs to update:  完成后需更新的文档列表
Reviewer type:   normal / adversarial / milestone
```

### 12.5 单任务开发循环

```
Captain 生成任务包 → Worker 执行 → Reviewer 审查 → Captain 整合
```

- **PASS**：标记任务完成，更新 handoff，推荐下一任务但不执行
- **PASS_WITH_WARNINGS**：warning 分类为 accepted / deferred / rejected；deferred 写入 risks
- **BLOCK**：Worker 只修 blocking issue。同一任务最多自动复审一次。第二次仍 BLOCK → 停止，交给人类裁决

### 12.6 里程碑闸门

每个 milestone 结束后：
- 暂停开发，做里程碑审查
- 输出：`docs/review/<TaskID>_milestone_review.md`
- 审查问题：功能是否真的完成？是否从干净环境可运行？是否有伪完成？是否允许进入下一里程碑？
- 结论只能为：**Allow / Conditional / Block**

### 12.7 关键约束

1. **仓库文件是主状态**，AI 会话不是主状态
2. **每轮只推进一个当前唯一任务**
3. 不让两个 agent 同时修改同一批文件
4. **不把计划、mock、stub、未来能力写成已完成事实**
5. 每个任务完成后必须有可验证结果、风险记录和下一步唯一任务
6. 默认顺序执行；并行仅在修改文件不重叠时允许
7. 分支命名：`agent/codex-T<编号>-<简短描述>`

### 12.8 项目文件维护要求

每次任务完成时 Captain 必须维护以下文件：

| 文件 | 更新内容 |
|------|----------|
| `docs/04_task_board.md` | 标记已完成任务、更新 Current Unique Task |
| `docs/07_handoff.md` | 更新当前状态、已完成/已验证/下一任务 |
| `docs/08_risks_and_open_questions.md` | 新增/关闭/更新风险项 |
| `docs/05_decision_log.md` | 若有新的关键决策则追加 |
| `docs/02_experiment_plan.md`（本文件） | 若稳定结论、工程边界或环境信息变化则更新 |

---

## 13. 风险与待解决问题

### 13.1 当前风险清单

| ID | 风险 | 等级 | 缓解措施 | 最后更新 |
|------|------|------|----------|----------|
| R1 | 默认运行环境不可直接执行最小 benchmark | **中** | 所有文档显式指定推荐解释器；`requirements-recovery.txt` 已补齐 | T12 |
| R2 | 完整训练链、.tflite 与真板环境仍无统一依赖说明 | **中** | `requirements-recovery.txt` 作用域诚实；后续单开 manifest 任务 | T11 |
| R3 | 软件 HIL 与真板 HIL 边界易被误写 | **高** | 所有文档引用 `docs/03_hil_p4_boundary_audit.md` 统一口径 | T3 |
| R4 | 仓库中已有大量缓存与生成物噪声 | **中** | `docs/06_repo_noise_governance.md` 已固定分类策略；后续单开有界 cleanup | T5 |
| R5 | P4 只有 recovery smoke 级证据，非正式多场景 frozen benchmark | **中高** | T9 四模式单场景 smoke 已复验；正式多场景需后续补 | T9 |
| R6 | .tflite 真导出与 stub 回退易混淆 | **中高** | 文档与日志显式标注 artifact type（tflite_service vs tflite_stub_service） | T3 |
| R7 | 具体 cleanup 执行窗口与归档方式未定 | **中** | 后续单开有界 cleanup 任务（含 manifest + 回滚方式） | T5 |
| R8 | bounded recovery smoke 结论易被误外推到真板/.tflite/正式 benchmark | **中** | 持续写清结论边界；本文件第 7 节固定所有边界标签 | T12 |

### 13.2 当前开放问题

1. **下一张 bounded 开发任务包应该优先选哪一类？**
   - 候选：失败机理诊断 / paper-inspired 分支 / P4 多场景证据补全 / 训练 manifest
   - 建议：先做机理诊断（性价比最高、不依赖环境补全、直接指导后续方向）

2. **`.venvs/tf311` 不可用，如何恢复 .tflite 验收能力？**
   - 当前状态：Windows 工作区内不存在该路径（原始环境在 macOS 上）
   - 需决策：是否在 Windows 上新建 TensorFlow/TFLite 环境，还是等到有 macOS/Linux 机器再补

3. **训练链需要什么级别的独立 manifest？**
   - `DLEnv` 含 `torch`，可以通过 `pip freeze` 导出
   - 是否还需要 `requirements-train.txt` / `pyproject.toml`？

4. **正式 P4 多场景 frozen benchmark 何时恢复？**
   - 建议在机理诊断或 paper-inspired 分支方向确认后再决定是否投入长跑算力

5. **已跟踪的 `.pyc` / `runs/` / `artifacts/` 何时启动有界 cleanup？**
   - 建议在下一 milestone 稳定后单开 cleanup 任务
   - 必须先拆分"bootstrap 必需"（如 `static_theta_v2` .npz）vs"历史归档"

6. **board_backend placeholder 是否需要现在补强？**
   - 当前建议：在真板条件（设备节点 + RTL 地址表）具备前不投入
   - 继续作为条件性扩展，不阻塞主线实验

### 13.3 暂缓事项

以下事项重要但在下一唯一任务明确前暂缓：

1. `noise_channels → effective parameters` 桥接（E1）
2. load-aware latency injector（E2）
3. stateful fault injector（E3）
4. bit-accurate control pipeline（E4）
5. ADC/AFE 轻量读出链（E5）
6. teacher-representation 新分支扩展
7. 论文正文正式写作

---

## 14. 附录：文件路径索引与常用命令

### 14.1 关键文件快速索引

**治理与计划**：
- [CLAUDE.md](CLAUDE.md) — Claude Code 审查指令（默认角色：只读 reviewer）
- [AGENTS.md](AGENTS.md) — AI agent 治理文件
- [README.md](README.md) — 项目入口说明
- [docs/02_experiment_plan.md](docs/02_experiment_plan.md) — **本文件（Captain 唯一切入点）**
- [docs/04_task_board.md](docs/04_task_board.md) — 任务板（当前唯一任务）
- [docs/05_decision_log.md](docs/05_decision_log.md) — 决策日志（7 条正式决策）
- [docs/07_handoff.md](docs/07_handoff.md) — 交接文档
- [docs/08_risks_and_open_questions.md](docs/08_risks_and_open_questions.md) — 风险清单
- [docs/reference/AI_coding_workflow.md](docs/reference/AI_coding_workflow.md) — AI 开发工作流（角色/任务包/里程碑闸门）
- [requirements-recovery.txt](requirements-recovery.txt) — 恢复期最小依赖（numpy + PyYAML）

**项目方案与结论**：
- [docs/CNN_FPGA_GKP_工程化实验方案.md](docs/CNN_FPGA_GKP_工程化实验方案.md) — 工程方案全文（双回路架构/时序预算/参数映射规则/实验矩阵）
- [docs/CNN_FPGA_GKP_阶段结论.md](docs/CNN_FPGA_GKP_阶段结论.md) — 阶段结论全文（P0-P4 所有正式结果）
- [docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md](docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md) — 7 大仿真补强方向及优先级
- [docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md](docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md) — paper-inspired statcalib 分支设计
- [docs/CNN_FPGA_GKP_论文提纲_摘要_贡献点草稿.md](docs/CNN_FPGA_GKP_论文提纲_摘要_贡献点草稿.md) — 论文草稿
- [docs/CNN_FPGA_GKP_项目完成目标与投稿路线报告.md](docs/CNN_FPGA_GKP_项目完成目标与投稿路线报告.md) — 投稿路线
- [docs/P4_UKF正式结果与Hybrid机制分析.md](docs/P4_UKF正式结果与Hybrid机制分析.md) — UKF 修正与 Hybrid 机制分析
- [docs/legacy_context/2026-05-06_CNN_FPGA_GKP_legacy_handoff.md](docs/legacy_context/2026-05-06_CNN_FPGA_GKP_legacy_handoff.md) — 最新工程交接

**恢复期专项**：
- [docs/00_project_snapshot.md](docs/00_project_snapshot.md) — 恢复起始快照（2026-05-05）
- [docs/01_legacy_audit.md](docs/01_legacy_audit.md) — legacy 审计报告（含 Feature Reality Matrix）
- [docs/03_hil_p4_boundary_audit.md](docs/03_hil_p4_boundary_audit.md) — HIL/P4 边界审计（6 种边界标签）
- [docs/06_repo_noise_governance.md](docs/06_repo_noise_governance.md) — 仓库噪声治理
- [docs/P0_smoke_bootstrap.md](docs/P0_smoke_bootstrap.md) — P0 smoke 复用说明
- [docs/P3_software_hil_bootstrap.md](docs/P3_software_hil_bootstrap.md) — P3 software HIL 复用说明
- [docs/P4_benchmark_recovery_bootstrap.md](docs/P4_benchmark_recovery_bootstrap.md) — P4 recovery bootstrap 说明

**核心代码入口**：
- [cnn_fpga/benchmark/run_hil_suite.py](cnn_fpga/benchmark/run_hil_suite.py) — **P3 HIL 主入口**（单会话）
- [cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py](cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py) — **P4 benchmark 主入口**（多场景批量）
- [cnn_fpga/benchmark/run_p4_teacher_representation_paired.py](cnn_fpga/benchmark/run_p4_teacher_representation_paired.py) — **Teacher-representation paired benchmark**
- [cnn_fpga/model/tiny_cnn.py](cnn_fpga/model/tiny_cnn.py) — CNN 模型定义（NumPy + PyTorch 双后端）
- [cnn_fpga/model/train.py](cnn_fpga/model/train.py) — 训练入口
- [cnn_fpga/model/export.py](cnn_fpga/model/export.py) — .tflite 导出（真 + stub 双路径）
- [cnn_fpga/decoder/param_mapper.py](cnn_fpga/decoder/param_mapper.py) — 参数映射 `(σ,μ,θ) → (K,b)`（已冻结）
- [cnn_fpga/runtime/fast_loop_emulator.py](cnn_fpga/runtime/fast_loop_emulator.py) — 快回路仿真（含 overflow 拆分）
- [cnn_fpga/runtime/slow_loop_runtime.py](cnn_fpga/runtime/slow_loop_runtime.py) — 慢回路运行时（多 mode）
- [cnn_fpga/runtime/feature_builder.py](cnn_fpga/runtime/feature_builder.py) — 慢回路输入特征构建
- [cnn_fpga/hwio/board_backend.py](cnn_fpga/hwio/board_backend.py) — 真板 backend（**placeholder**）
- [cnn_fpga/hwio/mock_fpga.py](cnn_fpga/hwio/mock_fpga.py) — Mock FPGA backend（**mock_backend**）
- [benchmark/compare_full_vs_simplified_ler.py](benchmark/compare_full_vs_simplified_ler.py) — P0 对比脚本

**关键配置**：
- [cnn_fpga/config/hardware_hil_recovery_smoke.yaml](cnn_fpga/config/hardware_hil_recovery_smoke.yaml) — Recovery smoke HIL 配置
- [cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml](cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml) — Recovery smoke P4 配置
- [cnn_fpga/config/experiment_static_theta_v2.yaml](cnn_fpga/config/experiment_static_theta_v2.yaml) — P1 主模型训练配置
- [cnn_fpga/config/experiment_runtime_b_residual.yaml](cnn_fpga/config/experiment_runtime_b_residual.yaml) — Full residual-b 训练配置
- [cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml](cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml) — **Gated v5 训练配置**
- [cnn_fpga/config/p4_multiscenario_strong_baselines.yaml](cnn_fpga/config/p4_multiscenario_strong_baselines.yaml) — 强 baseline P4 配置

**关键 artifact**：
- `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz` — P1 浮点主模型
- `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz` — P1 int8 模型
- `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz` — residual-b v1 模型

**最新 recovery smoke 证据（T12）**：
- `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104/hil_summary.json` — 确定性复验 run 1
- `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104/hil_summary.json` — 确定性复验 run 2
- 两次 `hil_summary.json` SHA256 一致，`hil_events.json` SHA256 一致

### 14.2 常用运行命令

```powershell
# ============================================
# 恢复期 smoke（在 AConda 下运行即可，秒~分钟级）
# ============================================

# P0: full_qec vs simplified 最小对比
& 'C:\ProgramData\anaconda3\python.exe' benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test

# P3: software HIL 最小 smoke（逐字一致复验已确认）
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml

# P4: 单场景两模式 benchmark smoke
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds

# P4: frozen baseline 四模式单场景 smoke
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds

# ============================================
# 正式开发级命令（需要 DLEnv 或更完整环境）
# ============================================

# P4: 强 baseline 多场景全部对比
# python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark \
#   --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml \
#   --scenario static_bias_theta --scenario linear_ramp \
#   --scenario step_sigma_theta --scenario periodic_drift \
#   --mode ekf --mode ukf --mode constant_residual_mu \
#   --mode rls_residual_b --mode hybrid_residual_b \
#   --paired-seeds --repeats 2

# Teacher-representation paired benchmark
# python -m cnn_fpga.benchmark.run_p4_teacher_representation_paired \
#   --config cnn_fpga/config/p4_teacher_repr_mid.yaml \
#   --v1-config cnn_fpga/config/experiment_runtime_b_residual.yaml \
#   --v2-config cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml \
#   --scenario linear_ramp --scenario periodic_drift \
#   --repeats 2 --seed 20260427 --seed 20260428
```

### 14.3 本文档更新规则

1. **每次里程碑闸门后**：更新本文件全部相关章节
2. **稳定结论清单变化时**：更新第 9 节
3. **工程边界标签变化时**：更新第 7 节
4. **环境/依赖信息变化时**：更新第 6 节
5. **同步维护**：更新后同步 `docs/07_handoff.md` 的"当前状态"、"已完成"、"当前判断"
6. **版本记录**：建议在更新时在文件顶部更新"最后更新"日期

---

**文档结束。** 本文件是项目 Captain AI 的唯一切入点。新 Captain 会话启动时只需完整阅读本文件，即可获得继续开发所需的全部上下文、约束、优先级和历史判断。如有不明确之处，对应的原始详情可从第 14.1 节索引的源文档中查找。

建议新 Captain 会话的第一条指令：
> 请作为 Codex Captain 工作。先完整阅读 docs/02_experiment_plan.md。不要直接实现。请根据当前状态推荐下一唯一任务，并生成一个 worker 任务包。
