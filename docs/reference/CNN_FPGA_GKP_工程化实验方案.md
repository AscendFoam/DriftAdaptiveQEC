# CNN-FPGA 协同 GKP 解码工程化实验方案（可落地实施版）

## 0. 文档目标

本文档用于指导下一阶段 CNN-FPGA 协同 GKP 解码实验的工程化实施，目标是：

1. 在保留原《CNN_FPGA_GKP_实验设计》研究目标的前提下，补齐可执行的项目结构与模块边界。
2. 将“物理仿真 -> 硬件行为仿真 -> 硬件在环(HIL)”串成同一套可验证流程。
3. 明确每个阶段的输入输出、接口协议、延迟预算、验证指标和验收门槛。

该方案强调工程可行性，优先采用可逐步替换、可回退、可量化验收的实现路径。

---

## 1. 总体技术路线

### 1.1 双回路控制保持不变

- 快回路（FPGA，周期级）
  - 输入：每轮 syndrome `(s_q, s_p)`
  - 计算：线性解码 `Δ = K @ s + b`
  - 输出：校正位移指令
  - 并行任务：32x32 直方图在线累积

- 慢回路（ARM/CNN，窗口级）
  - 输入：快回路累积的直方图窗口
  - 计算：CNN 预测噪声参数 `(σ, μ_q, μ_p, θ)`
  - 输出：更新解码器参数 `(K, b)`

### 1.2 三层验证闭环

1. 物理层仿真：基于 `physics` 模块生成 syndrome 与逻辑错误统计。
2. 硬件行为仿真：加入固定点量化、寄存器/总线延迟、DMA 读写延迟、参数切换保护。
3. 硬件在环：FPGA 快回路 + ARM 慢回路真实运行，软件负责驱动和记录。

---

## 2. 建议项目结构（新增模块）

```text
DriftAdaptiveQEC/
├── physics/                               # 现有物理仿真
│   ├── gkp_state.py
│   ├── noise_channels.py
│   ├── syndrome_measurement.py
│   ├── error_correction.py
│   └── logical_tracking.py
│
├── cnn_fpga/                              # 新增：工程主模块
│   ├── __init__.py
│   ├── config/
│   │   ├── experiment_static.yaml
│   │   ├── experiment_drift.yaml
│   │   ├── hardware_emulation.yaml
│   │   └── hardware_hil.yaml
│   ├── data/
│   │   ├── schema.py                      # 数据结构定义（syndrome/hist/labels）
│   │   ├── dataset_builder.py             # 数据集生成
│   │   ├── histogram.py                   # 直方图映射和归一化
│   │   └── split.py                       # train/val/test 划分
│   ├── model/
│   │   ├── tiny_cnn.py                    # CNN 模型定义
│   │   ├── train.py                       # 训练入口
│   │   ├── evaluate.py                    # 回归评估
│   │   ├── evaluate_tflite.py             # 独立 TFLite 精度评估
│   │   ├── quantize.py                    # QAT/PTQ + int8 导出
│   │   ├── export.py                      # ONNX/TFLite 导出
│   │   └── validate_export.py             # artifact 与 TFLite 一致性验收
│   ├── decoder/
│   │   ├── param_mapper.py                # (σ,μ,θ) -> (K,b)
│   │   ├── linear_runtime.py              # 快回路等价软件实现
│   │   ├── mwpm_stub.py                   # MWPM 接口占位（后续实现）
│   │   └── ekf_baseline.py                # EKF 基线
│   ├── runtime/
│   │   ├── fast_loop_emulator.py          # 带周期约束的快回路仿真
│   │   ├── slow_loop_runtime.py           # CNN 推理与参数更新
│   │   ├── scheduler.py                   # 双回路调度
│   │   ├── param_bank.py                  # 双缓冲参数切换
│   │   ├── latency_injector.py            # DMA/AXI/推理延迟注入
│   │   ├── inference_service.py           # 进程内/子进程推理服务
│   │   └── inference_worker.py            # 独立推理 worker
│   ├── hwio/
│   │   ├── axi_map.py                     # 寄存器映射定义
│   │   ├── dma_client.py                  # DMA 读写接口
│   │   ├── fpga_driver.py                 # HIL 驱动封装
│   │   ├── mock_fpga.py                   # 无板卡时模拟后端
│   │   └── board_backend.py               # 真板卡后端骨架
│   ├── benchmark/
│   │   ├── run_static_suite.py
│   │   ├── run_drift_suite.py
│   │   ├── run_hardware_emulation.py
│   │   ├── run_hil_suite.py
│   │   ├── run_hil_mode_benchmark.py
│   │   ├── run_p2_mode_benchmark.py
│   │   ├── run_p3_param_sweep.py
│   │   └── summarize.py
│   └── report/
│       ├── metrics.py
│       ├── plots.py
│       └── export_markdown.py
│
├── fpga/                                  # 新增：FPGA 实现目录
│   ├── rtl/
│   │   ├── linear_decoder.v
│   │   ├── histogram_accum.v
│   │   ├── param_bank.v
│   │   └── axi_lite_regs.v
│   ├── hls/
│   │   └── cnn_preprocess_hls.cpp         # 可选
│   ├── sim/
│   │   ├── tb_linear_decoder.sv
│   │   └── tb_histogram.sv
│   ├── constraints/
│   │   └── timing.xdc
│   └── vivado/
│       └── project.tcl
│
├── benchmark/
│   └── compare_full_vs_simplified_ler.py  # 已有最小对比脚本
│
└── docs/
    ├── CNN_FPGA_GKP_实验设计.md
    └── CNN_FPGA_GKP_工程化实验方案.md
```

---

## 3. 模块接口与数据契约

### 3.1 syndrome 数据格式

- `float32[2]`：`[s_q, s_p]`
- 快回路内部固定点：`Q4.20`（建议）
- 量程建议：`[-lambda/2, +lambda/2]`，超限做饱和而非截断回绕

### 3.2 直方图格式

- 网格：`32 x 32`
- bin 计数：`uint16` 或 `uint32`（根据窗口大小选择）
- 窗口长度：建议 `W=1024` 或 `W=2048` 周期
- 归一化策略：
  1. `H = H / sum(H)`
  2. 可选 `log1p(alpha*H)` 以增强弱信号分辨率

### 3.3 参数更新格式

- CNN 输出：`[σ, μ_q, μ_p, θ]`
- 映射输出：`K(2x2) + b(2,)`，共 6 个参数
- FPGA 寄存器写入固定点：`Q4.20`
- 参数切换策略：双缓冲
  - `bank_A` 运行中
  - `bank_B` 写入中
  - `commit_epoch` 到达后在周期边界原子切换

### 3.4 AXI-Lite 地址建议

- `0x00` `CTRL`：bit0 start, bit1 reset_hist, bit2 commit_bank
- `0x04` `STATUS`：bit0 ready, bit1 hist_ready, bit2 commit_ack
- `0x10~0x24`：`K11,K12,K21,K22,b1,b2`
- `0x30`：`ACTIVE_BANK`
- `0x34`：`EPOCH_ID`

### 3.5 DMA 缓冲建议

- `histogram_buffer`: 1024 bins x 4B = 4096B
- 双缓冲 DMA：`buf0/buf1` 轮转
- ARM 端读取目标：每 10~50 ms 一次

### 3.6 参数映射与稳定化规则（必须固定，避免训练-部署语义漂移）

给定 CNN 预测输出 `(sigma_hat, mu_q_hat, mu_p_hat, theta_hat)`，当前工程主线已经固定为“协方差一致”的确定性映射，不能再使用早期的 `g * R(theta)` 简化式。建议采用以下规则：

1. 预测误差协方差  
   先将 `theta_hat` clip 到 `[-20°, +20°]`，构造旋转矩阵  
   `R(theta_hat) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]`  
   设 `sigma_q = sigma_hat`，`sigma_p = sigma_ratio_p * sigma_hat`，则主轴系误差协方差为  
   `C_principal = diag(sigma_q^2, sigma_p^2)`  
   实验室坐标系协方差为  
   `C = R(theta_hat) * C_principal * R(theta_hat)^T`
2. 测量噪声协方差  
   用测量效率和有限能量参数折算出等效测量噪声  
   `R_meas = (sigma_meas^2 + delta_eff^2) * I`
3. 线性增益矩阵  
   使用 Wiener / Kalman 风格矩阵形式  
   `K_raw = C * (C + R_meas)^(-1)`  
   然后对 `K_raw` 的特征值做裁剪，而不是对单一标量增益裁剪。当前工程实现已经进一步加入“保守缩放”旋钮：  
   `K_raw = U * diag(lambda_raw) * U^T`  
   `lambda_clip = clip(lambda_raw, [g_min, g_max])`  
   `lambda_scaled = clip(lambda_clip * gain_scale, [0, g_max])`  
   `K_target = U * diag(lambda_scaled) * U^T`  
   其中 `gain_scale` 只作为工程稳定性调参项，不改变物理映射主线语义；若没有同口径长时 benchmark 支撑，不应随意修改默认值
4. 偏置项  
   先验均值 `mu = [mu_q_hat, mu_p_hat]^T` 对应的偏置应写成  
   `b_target = alpha * (I - K_target) * mu`  
   不能再写成 `-alpha * mu`，否则在非零均值漂移下会把校正方向推错
5. 参数平滑（防止抖动）  
   `K_next = (1-beta) * K_prev + beta * K_target`  
   `b_next = (1-beta) * b_prev + beta * b_target`，推荐 `beta=0.1~0.3`

说明：参数映射一旦在实验主线确定，不得在不同基线间隐式修改，否则会破坏公平对比。

### 3.7 直方图边界策略（必须显式实现）

- `s_q/s_p` 超出映射范围时，不允许静默丢弃。
- 推荐策略：先 clip 到 `[s_min, s_max]` 再入 bin，并累计 `overflow_counter` 作为诊断指标。
- 每次慢回路推理时输出 `overflow_ratio = overflow_counter / W`，若超过阈值（如 5%）触发告警。

---

## 4. 时序预算与可行性约束

### 4.1 目标周期

- QEC 快回路周期目标：`T_fast = 5 us`（可根据设备调到 1~10 us）
- 慢回路更新周期：`T_slow = 10~100 ms`

### 4.2 快回路预算（每周期）

- syndrome 输入与寄存：`<= 0.5 us`
- 线性解码（固定点矩阵运算）：`<= 0.5 us`
- 校正输出准备：`<= 0.5 us`
- 直方图累积：并行，不占关键路径

总计关键路径目标：`<= 1.5 us`，保留裕量给 ADC/控制链路。

### 4.3 慢回路预算（每次更新）

- DMA 读出：`5~20 us`
- 预处理：`20~100 us`
- CNN int8 推理：`0.1~2 ms`（取决于部署目标）
- 参数映射与写回：`10~100 us`

总计目标：`< 5 ms`（保守可接受 `< 20 ms`）。

### 4.4 窗口长度与更新周期耦合约束

为避免“训练设定”和“在线推理节奏”不一致，必须满足：

- `T_window = W * T_fast`
- `T_slow_update >= T_window` 时，采用“非重叠窗口”更新；
- `T_slow_update < T_window` 时，采用“滑动窗口”更新，并固定 stride（建议 `stride = W/4`）。

建议默认：

- `T_fast = 5 us`
- `W = 2048`（`T_window = 10.24 ms`）
- `T_slow_update = 20 ms`（非重叠 + 一次缓冲冗余）

该配置能兼顾统计稳定性与慢回路实时性，避免过于频繁更新导致参数抖动。

---

## 5. 实验阶段规划

### 5.1 阶段 P0：仿真基线确认（已可开始）

目标：确认 `full_qec` 与 `simplified` 差异，建立后续评价基线。

- 使用脚本：`benchmark/compare_full_vs_simplified_ler.py`
- 输出：`LER 曲线 + summary.json`
- 验收：同参数下 `final_gap_mean > 0` 且跨重复稳定

### 5.2 阶段 P1：CNN 数据与训练闭环

目标：完成从 syndrome 直方图到 `(σ, μ_q, μ_p, θ)` 的回归模型。

步骤：
1. 新增数据生成器 `cnn_fpga/data/dataset_builder.py`
2. 生成静态+漂移混合数据集
3. 训练 Tiny-CNN（浮点）
4. 评估回归指标（MSE/MAE/R²）
5. 导出 int8 模型并做精度回归

验收阈值建议：
- `R²(σ) > 0.9`
- `R²(μ_q, μ_p) > 0.85`
- `R²(θ) > 0.8`
- int8 精度退化 `< 10%`

### 5.3 阶段 P2：硬件行为仿真（无板卡）

目标：在软件里尽量逼近硬件行为。

需要引入：
- 固定点量化误差（Q4.20）
- 寄存器写入延迟/抖动
- DMA 传输延迟/偶发阻塞
- 参数切换原子性检查

关键测试：
- 参数切换点不导致解码突变（glitch-free）
- 慢回路掉帧时快回路稳定运行
- 极端噪声下无数值溢出

### 5.4 阶段 P3：软件 HIL 主线与真板扩展

目标：先在软件 HIL 中完成端到端链路、部署产物和运行时诊断验证；若未来具备板卡条件，再把相同链路迁移到真板。

软件 HIL 主线步骤：
1. 以 `mock FPGA backend` 驱动快回路事件推进、直方图窗口、参数 bank 和 commit 时序
2. 用真实 `.tflite` 或 artifact 子进程模拟 ARM 侧推理服务
3. 输出周期级/窗口级/运行级诊断，重点跟踪 `LER / commit / overflow breakdown`
4. 在软件 HIL 口径下完成参数调优、baseline 冻结和统计复现准备

真板扩展步骤（条件具备时再执行）：
1. 部署 FPGA 快回路 IP（线性解码 + 直方图 + 参数bank）
2. ARM 运行 TFLite 推理服务
3. Linux 用户态驱动完成 DMA + AXI 写回
4. 同步记录周期级日志和系统级指标

验收阈值建议：
- 参数更新成功率 `> 99.9%`
- 快回路周期违约率 `< 10^-6`
- HIL 相对行为仿真 LER 偏差 `< 15%`

当前代码进度对照（截至 2026-04-01）：

- 软件侧 HIL 双回路已经打通，正式结果位于：
  - [hardware_hil_v1_20260328_012839_7b58cdf75d7a](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/hil_mode_benchmark/hardware_hil_v1_20260328_012839_7b58cdf75d7a)
- 新统计口径已经补上 `overflow breakdown`，并完成了正式长跑复核：
  - [hardware_hil_v1_20260330_233853_c9ed542a0cfa](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/hil_mode_benchmark/hardware_hil_v1_20260330_233853_c9ed542a0cfa)
  - 该轮结果表明，长配置下主导 overflow 来源不是 `correction_saturation`，也不是 `aggressive_param`，而是 `histogram_input`
- 因此又补做了输入范围侧定向调参，并完成长确认：
  - [hardware_hil_histogram_tuning_20260331_001959_bfea868ecd4f](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_histogram_tuning/hardware_hil_histogram_tuning_20260331_001959_bfea868ecd4f)
  - 最优长确认组合：
    - `syndrome_limit = 1.441311257912825`
    - `histogram_range_limit = 1.8799712059732503`
    - `sigma_measurement = 0.03`
  - 相对旧默认值 baseline，`LER` 从 `1.069503` 降到 `1.046678`，`histogram_input_saturation_rate` 从 `0.387092` 降到 `0.022181`
- 基于上述结果，正式 HIL 默认配置已切换到新输入范围组合，并完成正式 mode benchmark 复跑：
  - [hardware_hil_v1_20260331_010329_c128fa34262e](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/hil_mode_benchmark/hardware_hil_v1_20260331_010329_c128fa34262e)
- 修正 `.tflite` 导出语义后，`float_artifact_mock` 与 `int8_artifact_mock` 已重新对齐：
  - 旧默认值下：
    - `fixed_baseline_mock`: `LER = 1.1987018`，`overflow_rate = 0.2972882`
    - `float_artifact_mock`: `LER = 1.1405753`，`overflow_rate = 0.3935607`
    - `int8_artifact_mock`: `LER = 1.1407845`，`overflow_rate = 0.3933973`
  - 新默认值下：
    - `fixed_baseline_mock`: `LER = 1.1993300`，`overflow_rate = 0.0161568`
    - `float_artifact_mock`: `LER = 1.1238110`，`overflow_rate = 0.0227638`
    - `int8_artifact_mock`: `LER = 1.1246990`，`overflow_rate = 0.0226377`
  - 三种可运行模式在新默认值下仍全部由 `histogram_input` 主导，且 `correction_saturation_rate = 0`、`aggressive_param_rate = 0`
- 真实 `.tflite` 导出物已在 `/.venvs/tf311` 中独立验证：
  - float `.tflite`: `MSE = 0.292359`，`R2_mean = 0.994359`
  - int8 `.tflite`: `MSE = 0.297316`，`R2_mean = 0.994192`
- `real_board` 当前仍因缺少 `/dev/uio0,/dev/uio1` 被 `skipped_unavailable`
- 因此，当前更准确的工程判断应是：
  - `P3-软件HIL`：已通过
  - `P3-真板HIL`：待板卡与地址表条件具备后继续
- 如果未来较长时间都无法获得板卡，则项目主线不应停在这里，而应继续推进：
  - `overflow` 来源拆分与参数收敛
  - baseline 冻结与多场景复现
  - 多场景统计复现与报告

截至 `2026-04-01`，在上述软件 HIL 主线之上又完成了一轮正式 P4 多场景对比，且主方案已从“直接回归绝对参数”演进到“teacher + residual-b”：

- runtime-consistent 数据集、训练和运行时链路已经打通：
  - [manifest.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/datasets/runtime_b_residual_v1/manifest.json)
  - [tiny_cnn_20260401_083648_2fc740424c0d.npz](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz)
  - [eval_test_20260401_083649.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/runtime_b_residual_v1/eval_test_20260401_083649.json)
- 对应正式 benchmark 已完成：
  - [p4_multiscenario_hybrid_b_v1_20260401_083649_41775bdd90b1_11082](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_hybrid_b_v1_20260401_083649_41775bdd90b1_11082)
- 当前 4 个正式场景上的平均 `LER` 为：
  - `Hybrid Residual-B = 0.850799`
  - `Constant Residual-Mu = 0.855549`
  - `EKF = 0.855779`
  - `Window Variance = 0.857016`
  - `CNN-FPGA = 0.954315`
- `Hybrid Residual-B` 在 `linear_ramp / periodic_drift / static_bias_theta / step_sigma_theta` 四个场景下均取得最优 `LER`
- 本轮正式对比同时表明：
  - `commit` 与调度链路稳定，`n_commits_applied` 基本均为 `600`
  - `slow_update_violation_rate = 0`
  - `correction_saturation_rate = 0`
  - `aggressive_param_rate = 0`
  - 当前主导瓶颈依然是 `histogram_input overflow`
- 但 `static_bias_theta` 场景下 `Hybrid Residual-B` 的跨 seed 波动偏大，需后续做额外 repeat 复查

截至 `2026-04-02`，在冻结模型的前提下，又完成了一轮输入统计范围定向调参与更长配置正式复验：

- 定向调参结果：
  - [hybrid_b_histogram_tuning_20260401_144804_19d9812a12db](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_histogram_tuning/hybrid_b_histogram_tuning_20260401_144804_19d9812a12db)
- 当前主线默认输入范围已更新为：
  - `syndrome_limit = 1.566643`
  - `histogram_range_limit = 2.255965971`
- 更长配置正式复验结果：
  - [p4_multiscenario_hybrid_b_long_v1_20260402_145451_94eb56a87b59_38723](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_hybrid_b_long_v1_20260402_145451_94eb56a87b59_38723)
- 本轮长配置只比较：
  - `EKF`
  - `Constant Residual-Mu`
  - `Hybrid Residual-B`
- 4 个正式场景上的平均 `LER` 为：
  - `Hybrid Residual-B = 0.798807`
  - `Constant Residual-Mu = 0.826193`
  - `EKF = 0.828108`
- `Hybrid Residual-B` 在四个场景下继续全部最优，且领先次优模式的幅度扩大到 `0.023~0.032`
- 三条主线的跨场景平均 `histogram_input_saturation_rate` 已降到约 `0.00254~0.00258`
- `static_bias_theta` 场景下 `Hybrid Residual-B` 的 `LER std` 也从此前 seed 复查中的 `0.005021` 收敛到 `0.000629`

### 5.5 阶段 P4：多基线统计对比与报告

输出：
- 静态噪声对比
- 漂移噪声追踪
- 非高斯鲁棒性
- 资源、延迟、功耗综合报告

进入条件调整为：

- `P3-软件HIL` 已通过
- `.tflite` 部署产物已独立验收
- `overflow` 来源统计链路已接通
- baseline 集合与统计口径已冻结

说明：

- `P4` 不再以“必须先拿到真板”为前置条件
- 若真板长期不可得，则 `P4` 的主结论先以“软件 HIL + 真实 `.tflite` + 严格物理仿真回看”的组合口径完成
- 真板联调转为后续增强验证，而不是阻塞条件
- 截至 `2026-04-02`，软件 HIL 主线不仅满足进入 `P4` 的条件，而且已经完成“首轮正式结果 + 新默认值长配置复验”的两轮验证

### 5.5.1 当前 P4 正式口径与工程含义

当前正式 P4 结果对应的核心工程选择如下：

1. 正式对比不再只看旧 `CNN-FPGA`

- 旧 `CNN-FPGA` 仍保留
- 但它现在主要承担“绝对参数回归基线”的角色
- 当前主方案已经切到 `Hybrid Residual-B`

2. 当前 CNN 的职责被重新定义为“残差补偿器”

- `Window Variance` 或其它 teacher 先给出稳定的一阶估计
- CNN 不再独立承担全部参数辨识
- CNN 只学习对运行时 `b` 的小幅修正

3. 当前最重要的工程瓶颈不是模型量化，而是输入统计范围

- 本轮正式对比里，`float/int8` 不再是主要矛盾
- 主要 overflow 来源一致落在 `histogram_input`
- 因而后续优先级应放在窗口统计范围和饱和控制，而不是继续盲目压 gain 或频繁改主模型

4. 当前主线结论已经升级为“优势在更长配置下成立”

- 经过 `2026-04-02` 的更长配置复验后，`Hybrid Residual-B` 对 `EKF / Constant Residual-Mu` 的优势不但保留，而且明显扩大
- 因此后续主线可以从“继续确认该方向是否有效”，转为“补更强 baseline 与论文级对比”

### 5.6 分阶段硬门槛（Go/No-Go）

| 阶段 | 必达条件 | 未达处理 |
|------|----------|----------|
| P0 | `final_gap_mean > 0` 且重复实验标准差可控 | 先修正物理建模与统计代码，不进入 P1 |
| P1 | 参数回归指标达阈值，int8 退化 < 10% | 扩展数据集并重训，不进入 P2 |
| P2 | 周期仿真无越界/无切换毛刺，延迟预算达标 | 修复运行时与参数bank，不进入 P3 |
| P3-软件HIL | 双回路事件推进、artifact/int8/真实 `.tflite` 模式对比可复现，预算/commit 正常 | 修复运行时、导出链路或参数映射，不进入真板联调 |
| P3-真板HIL | HIL 稳定运行 24h，无严重通信错误 | 驱动和RTL整改后复测 |
| P4 | 主结论在 2 组以上噪声场景复现，baseline 与统计口径冻结 | 增加重复实验与敏感性分析 |

补充说明：

- `P3-真板HIL` 是强增强项，但在板卡长期不可得时，不再阻塞 `P4`
- 主线 go/no-go 判断以 `P3-软件HIL -> P4` 为主，真板验证作为条件具备后的追加里程碑

---

## 6. 实验矩阵（建议）

### 6.1 静态场景

- `sigma_error`: 0.10 ~ 0.50（步长 0.05）
- `theta`: `0°, 5°, 10°, 15°`
- `delta`: `0.2, 0.3, 0.4`
- 每点重复：`N=20`
- 每次轮数：`T=10,000`

### 6.2 漂移场景

- 线性漂移：`σ(t)=σ0+αt`，`α` 分三档
- 阶跃漂移：`Δσ` 三档，`t_jump` 三档
- 周期漂移：`A,f` 组合网格
- 随机游走：`σ_d` 三档

### 6.3 非高斯与相关场景

- t 分布（自由度 `ν=3,5,8`）
- 双峰混合高斯（模拟 TLS 跳变）
- 协方差旋转噪声（`ρ=0.2,0.5,0.8`）

---

## 7. 评价指标与统计方法

### 7.1 主指标

- `LER`
- `X/Z` 分量错误率
- 参数回归 `MSE/MAE/R²`
- 更新延迟、推理延迟、端到端延迟
- 资源利用率（DSP/BRAM/LUT/FF）
- 功耗与性能功耗比

### 7.2 统计方法

- 以“独立 run 的汇总统计”作为 t 检验样本
- 对时序曲线使用 block bootstrap 估计置信区间
- 同时报告效应量（Cohen's d）

### 7.3 基线集合（统一对比口径，2026-03-31 冻结）

每个实验场景建议至少比较以下 4 条曲线：

1. 固定参数线性解码（Static Linear）
2. 窗口方差估计更新（Window Variance）
3. EKF 参数追踪（EKF）
4. CNN-FPGA 协同（本文方案）

MWPM 可作为扩展基线，但不应阻塞主实验流水线。

冻结说明：

- 自 `2026-03-31` 起，`P4` 主线统计对比默认冻结以上 4 条基线
- `MWPM` 保持为可选增强项，不作为 `P4` 启动前置条件
- `fixed_baseline / oracle_delayed / fixed_baseline_mock` 继续保留为 `P2/P3` 内部工程校验基线，但不替代 `P4` 的正式对比集合

### 7.3.1 2026-04-02 当前正式运行集合

在正式进入 runtime-consistent `residual-b` 主线后，当前实际运行的正式对比集合扩展为：

1. `Window Variance`
2. `EKF`
3. `Constant Residual-Mu`
4. `CNN-FPGA`
5. `Hybrid Residual-B`

补充说明：

- `Constant Residual-Mu` 的作用是提供一个公平、低复杂度、可解释的常数残差基线
- `Hybrid Residual-B` 是当前主线方案，用于回答“在 teacher 已经稳定的前提下，CNN 做残差修正是否还能继续带来收益”
- `Static Linear` 仍保留为工程锚点基线，但本轮 `hybrid_b` 正式对比的核心竞争关系已经转为上述 5 模式集合
- 后续若不发生严重反例，不应频繁再改这套正式比较口径

下一步强 baseline 扩展建议：

- 在上述正式集合之外，新增 `RLS Residual-B / UKF / Particle Filter` 作为更强的经典自适应对照
- 其中 `RLS Residual-B` 最容易与当前 `teacher + residual-b` 的运行时语义保持一致，可优先落地
- `UKF / Particle Filter` 则适合作为后续论文中“比 EKF 更强的经典滤波”补充对比

---

## 8. 关键工程细节（保证可实现）

### 8.1 固定点策略

- syndrome 输入：`Q4.20`
- K,b 参数：`Q4.20`
- 乘加结果：内部 `Q8.24` 累加后回落到 `Q4.20`
- 饱和处理：越界直接钳位到最大可表示值

### 8.2 参数切换保护

- 双 bank + epoch id
- 仅在周期边界切换
- 切换后读取回显寄存器确认

### 8.3 异常降级策略

- CNN 推理失败：保持上一版本参数
- DMA 超时：重试 1 次，失败则跳过本周期慢回路更新
- 参数非法（NaN/Inf/越界）：丢弃并报警

### 8.4 日志与追踪

至少记录：
- `timestamp`
- `epoch_id`
- `K,b`
- `predicted σ,μ,θ`
- 当前窗口 `LER`
- 慢回路耗时

### 8.5 可重复性与追溯字段（必须入日志）

每次 run 至少记录：

- `git_commit`
- `config_hash`
- `seed_data / seed_train / seed_runtime`
- `model_version`（含量化版本）
- `fpga_bitstream_version`（HIL 阶段）

缺失以上字段的结果不得进入最终对比统计。

### 8.6 仿真-硬件一致性校准（Sim2Real）

为避免离线结果与 HIL 偏差过大，建议执行两步校准：

1. 延迟校准：测量 DMA、AXI、推理真实延迟分布，回写到 `latency_injector`。
2. 数值校准：对比浮点与固定点 `K,b` 下的单步 correction 误差分布，更新量化 scale。

校准后再运行 P3/P4 正式实验，降低“仿真乐观偏差”风险。

### 8.7 `.tflite` 部署产物准入规则

任何准备进入 `P3/P4` 的 `.tflite` 模型，至少要经过两步独立验收：

1. 独立精度评测

- 使用 [evaluate_tflite.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/evaluate_tflite.py)
- 在 `/.venvs/tf311` 或其它真实 TensorFlow/TFLite 环境中直接执行
- 记录 `MSE / MAE / R2_mean / per_label`

2. 导出一致性检查

- 使用 [validate_export.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/validate_export.py)
- 对比源 artifact 与导出 `.tflite` 的逐样本输出差异
- 若该检查未通过，不允许进入 HIL benchmark

说明：

- 这样做的目的，是把“模型训练正确”和“部署产物语义正确”分开验证
- 尤其要防止归一化、通道顺序、量化 scale 等导出阶段问题污染后续 P3/P4 结果

### 8.8 长期无板卡时的主线调整

如果未来较长时间内无法获得真实板卡、地址表或寄存器说明，则建议把主线实验顺序调整为：

1. 优先完成 `P3-软件HIL` 内部收敛

- 继续拆解 `overflow` 来源
- 固定默认参数与主模型
- 清理所有会破坏横向可比性的隐式配置漂移

2. 提前进入 `P4` 的统计对比准备

- 冻结 baseline 集合
- 冻结场景集、seed、运行轮数和报告模板
- 在软件 HIL 口径下先形成一版完整主结论

3. 真板联调转为条件性扩展

- 只有在真实板卡和地址表具备时才重启
- 目的从“阻塞主线”改为“增强验证和工程落地补完”

这样调整后的核心原则是：

- 不因为外部硬件条件长期缺失而让项目主线停滞
- 先把软件 HIL、物理仿真和部署产物三条线收敛到可复现实验结论
- 真板验证放到条件成熟时补齐

---

## 9. 与现有 physics 库的集成策略

### 9.1 复用点

- 噪声与测量模型：`physics/noise_channels.py`, `physics/syndrome_measurement.py`
- 线性解码参数映射：`physics/error_correction.py`
- 逻辑错误评估：`physics/logical_tracking.py`

### 9.2 扩展点

- 新增 `cnn_fpga/decoder/mwpm_stub.py` 与统一解码接口
- 新增 `cnn_fpga/runtime/` 模块模拟硬件调度行为
- 新增 `fpga/` RTL/HLS 工程目录

---

## 10. 里程碑与时间建议

- 第 1 周：P0 收敛 + P1 数据流水线
- 第 2 周：P1 模型训练 + int8 / `.tflite` 导出
- 第 3 周：P2 行为仿真与调度器联调
- 第 4 周：P3 软件 HIL 打通 + `overflow` 来源拆分
- 第 5 周：baseline 冻结 + P4 多场景统计对比
- 第 6 周：若板卡条件具备，再补 P3 真板联调；若条件不具备，则继续扩展 P4 报告

---

## 11. 交付物清单

1. 可运行代码目录 `cnn_fpga/` 与 `fpga/`
2. 配置文件（静态、漂移、HIL）
3. 训练好的 float/int8 模型与版本记录
4. benchmark 输出（CSV/JSON/图）
5. HIL 实验日志与资源时序报告
6. 最终技术报告（性能、风险、限制、下一步）

---

## 12. 最小执行命令模板（后续实现后）

```bash
# 1) 生成训练数据
python -m cnn_fpga.data.dataset_builder --config cnn_fpga/config/experiment_static.yaml

# 2) 训练与评估
python -m cnn_fpga.model.train --config cnn_fpga/config/experiment_static.yaml
python -m cnn_fpga.model.evaluate --config cnn_fpga/config/experiment_static.yaml

# 3) 量化导出
python -m cnn_fpga.model.quantize --config cnn_fpga/config/hardware_emulation.yaml

# 4) 硬件行为仿真
python -m cnn_fpga.benchmark.run_drift_suite --config cnn_fpga/config/hardware_emulation.yaml

# 5) HIL 测试
python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil.yaml
```

建议在每条命令补充显式随机种子（如 `--seed 1234`）和输出目录（如 `--output runs/2026-...`），保证结果可追溯。

---

## 13. 风险与应对

- 风险1：CNN 回归精度不足
  - 应对：增加混合噪声数据覆盖；采用多头损失函数；引入轻量 Transformer 作为备选

- 风险2：量化后精度明显下降
  - 应对：QAT 替代 PTQ；对输入做离线标定；分层敏感度分析

- 风险3：HIL 延迟超预算
  - 应对：减小窗口、降低更新频率、前移预处理到 FPGA/HLS

- 风险4：参数切换造成短时误码尖峰
  - 应对：双bank + 周期边界切换 + 切换后 1 个窗口观测保护

---

## 14. 与原实验文档的关系

- 原文档继续承担“研究问题与整体对比框架”角色。
- 本文档承担“工程实施与执行细节”角色。
- 两者共同用于后续论文与系统实现：
  - 原文档：为什么做
  - 本文档：怎么做、做到什么程度算通过

---

## 15. 2026-04-03 强 baseline 正式更新补充

在 `2026-04-02` 的长配置复验之后，又完成了 `RLS Residual-B + UKF` 的正式强 baseline 扩展：

- 结果目录：
  - [p4_multiscenario_strong_baselines_v1_20260403_145747_b82874392710_86447](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_strong_baselines_v1_20260403_145747_b82874392710_86447)
- 4 个正式场景平均 `LER`：
  - `Hybrid Residual-B = 0.798332`
  - `UKF = 0.817974`
  - `Constant Residual-Mu = 0.825719`
  - `RLS Residual-B = 0.827908`
  - `EKF = 0.828369`

本轮结果说明：

1. 修正后的 `UKF` 已经成为当前最强经典 baseline

- 它在 `static_bias_theta / linear_ramp / step_sigma_theta / periodic_drift` 四个正式场景中均优于 `EKF`
- 说明前一阶段对 `UKF` 的数值稳定化修正是有效的

2. `Hybrid Residual-B` 的优势依旧成立

- 虽然 `UKF` 已经显著抬高了经典对照上限
- 但 `Hybrid Residual-B` 仍保持约 `0.0196` 的平均 `LER` 优势

3. 下一档更值得补的 baseline 是 `Particle Filter`

- 不是继续围绕 `EKF` 做小修小补
- 也不是继续长时间微调 `UKF`
- 而是新增一个对非线性、非高斯假设更宽松的经典滤波对照

因此，当前工程主线可以更新为：

1. 保持 `Hybrid Residual-B` 为主方案
2. 保持 `UKF` 为当前最强经典对照
3. 继续补 `Particle Filter`，以增强论文级对比说服力
