# CNN-FPGA-GKP 阶段结论

## 1. 文档目的

本文件用于记录当前阶段已经完成的工程工作、关键实验结果、是否达到阶段门槛，以及后续应继续推进的内容。

该结论文档对应：

- [CNN_FPGA_GKP_工程化实验方案.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/docs/CNN_FPGA_GKP_工程化实验方案.md)
- 当前代码版本中的 `cnn_fpga/` 与 `physics/` 运行结果

---

## 2. 当前总体判断

截至本次阶段性验证，项目状态可归纳为：

- `P0`：完成
- `P1`：完成，并达到文档验收阈值
- `P2`：完成首轮公平基线对比，并通过当前行为仿真验收
- `P3`：已完成软件侧 HIL 工程链路、真实 `.tflite` 导出/独立评测/子进程推理验证，但真实板卡仍未联调验收
- `P4`：已完成首轮正式多场景 benchmark，当前主线由“绝对参数回归 CNN”切到“teacher + residual-b 混合方案”

因此，当前项目状态可视为：

- `P2` 已完成并验收通过
- `P3` 软件链路已完成正式模式对比，并完成真实 `.tflite` 产物验收
- `P4` 已经开始产生正式结论，且当前最优模式为 `hybrid_residual_b`
- 但还不能称为“整体工程基本完成”，因为 `histogram_input overflow` 仍是主导瓶颈，且 `static_bias_theta` 场景下还存在跨 seed 波动待复查，真板验证也尚未补齐

---

## 3. P1 结论

### 3.1 问题定位

早期 `static_v1` 数据集中的噪声点云生成方式为“各向同性高斯”。

这会带来一个根本问题：

- 若分布是圆对称的，那么旋转角 `theta_deg` 在统计上几乎不可辨识
- 因此，CNN 即使容量足够，也难以学习到稳定的 `theta_deg`

这也是早期版本中：

- `sigma / mu_q / mu_p` 学得很好
- `theta_deg` 长期不过验收阈值

的根本原因。

### 3.2 修正措施

已完成两项关键修正：

1. 数据生成修正

- 文件：[dataset_builder.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/data/dataset_builder.py)
- 新增椭圆高斯数据生成模式
- 在配置中使用：
  - `distribution: anisotropic_gaussian`
  - `sigma_ratio_p: 0.55`

2. 训练目标修正

- 文件：[tiny_cnn.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/tiny_cnn.py)
- 对 `theta_deg` 引入更高的损失权重
- 避免被 `sigma / mu_q / mu_p` 这些更容易的标签主导训练

新增配置：

- [experiment_static_theta_v2.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/experiment_static_theta_v2.yaml)

### 3.3 P1 正式模型与结果

浮点模型：

- [tiny_cnn_20260319_151717_b87c6c227b57.npz](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz)

浮点评估：

- [eval_test_20260319_151756.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/eval_test_20260319_151756.json)

int8 模型：

- [tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz)

int8 评估：

- [eval_test_20260319_151817.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/eval_test_20260319_151817.json)

量化报告：

- [tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_quant_report.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_quant_report.json)

### 3.4 P1 指标

浮点模型 test 集：

- `MSE = 0.293336`
- `MAE = 0.220503`
- `R2_mean = 0.994352`

逐标签：

- `sigma`: `R² = 0.997613`
- `mu_q`: `R² = 0.996473`
- `mu_p`: `R² = 0.998459`
- `theta_deg`: `R² = 0.984862`

int8 模型 test 集：

- `MSE = 0.297742`
- `MAE = 0.221944`
- `R2_mean = 0.994212`

逐标签：

- `sigma`: `R² = 0.997730`
- `mu_q`: `R² = 0.996324`
- `mu_p`: `R² = 0.998160`
- `theta_deg`: `R² = 0.984634`

### 3.5 P1 验收判断

对照工程方案中的建议阈值：

- `R²(sigma) > 0.9`：通过
- `R²(mu_q, mu_p) > 0.85`：通过
- `R²(theta) > 0.8`：通过
- `int8 退化 < 10%`：通过

结论：

- `P1 通过`

---

## 4. P2 问题与修正

### 4.1 初始异常

在早期 P2 对比中，出现了一个明显异常：

- `model_artifact` 的 `LER` 明显差于旧 `mock`
- 但 `float artifact` 与 `int8 artifact` 几乎重合

这说明问题不在量化，而在以下两点：

1. `mock` 基线不公平

- 旧 `mock` 直接读取当前窗口的 `target_params`
- 这相当于“当前窗口真值 + 小噪声”
- 它更接近 oracle，而不是可部署基线

2. `ParamMapper` 物理含义不合理

旧版本使用：

- `K = gain * rotation`
- `b = -mu`

这有两个问题：

- 对 syndrome 再做一次显式旋转，容易把校正方向推错
- 在非零均值情况下，`b = -mu` 的符号与后验线性估计不一致

### 4.2 修正措施

已完成两类修正。

#### 4.2.1 公平基线重定义

文件：

- [slow_loop_runtime.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/runtime/slow_loop_runtime.py)
- [hardware_emulation.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/hardware_emulation.yaml)

新增两种公平基线：

1. `fixed_baseline`

- 始终使用固定参数
- 用于表示“不自适应”的静态基线

2. `oracle_delayed`

- 使用“延迟一窗”的 target 参数
- 不再直接读取当前窗口真值
- 作为公平上界参考，而不是作弊型 mock

#### 4.2.2 ParamMapper 修正

文件：

- [param_mapper.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/decoder/param_mapper.py)

关键变化：

1. `K` 改成协方差驱动的对称增益矩阵

- 使用实验室坐标系下的误差协方差
- 使用 `K = C (C + R)^-1`
- 不再简单使用 `gain * rotation`

2. `b` 改成 `(I-K)mu`

- 更接近后验线性估计
- 修复旧版 `-mu` 导致的方向错误

3. 与训练集各向异性保持一致

- `sigma_ratio_p` 进入映射配置
- P2 物理侧也同步设置为 `0.55`

---

## 5. P2 正式对比结果

### 5.1 运行目录

完整 benchmark：

- [hardware_emulation_v1_20260319_160130_20670d1c0d1f](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p2_mode_benchmark/hardware_emulation_v1_20260319_160130_20670d1c0d1f)

核心输出：

- [mode_comparison.csv](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p2_mode_benchmark/hardware_emulation_v1_20260319_160130_20670d1c0d1f/mode_comparison.csv)
- [mode_delta.csv](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p2_mode_benchmark/hardware_emulation_v1_20260319_160130_20670d1c0d1f/mode_delta.csv)
- [report.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p2_mode_benchmark/hardware_emulation_v1_20260319_160130_20670d1c0d1f/report.md)

### 5.2 模式定义

本次对比四种模式：

1. `fixed_baseline`
2. `oracle_delayed`
3. `model_artifact`（float）
4. `int8_artifact`

### 5.3 核心结果

#### `linear_med`

- `fixed_baseline`: `LER = 0.816067`
- `oracle_delayed`: `LER = 0.764239`
- `model_artifact`: `LER = 0.696206`
- `int8_artifact`: `LER = 0.701372`

#### `step_large`

- `fixed_baseline`: `LER = 0.937822`
- `oracle_delayed`: `LER = 0.926650`
- `model_artifact`: `LER = 0.731906`
- `int8_artifact`: `LER = 0.732072`

#### `sinusoidal_mid`

- `fixed_baseline`: `LER = 1.019033`
- `oracle_delayed`: `LER = 1.020556`
- `model_artifact`: `LER = 0.759289`
- `int8_artifact`: `LER = 0.761878`

### 5.4 相对公平基线的改进

相对 `fixed_baseline`：

- `linear_med`: `-0.119861`
- `step_large`: `-0.205917`
- `sinusoidal_mid`: `-0.259744`

相对 `oracle_delayed`：

- `linear_med`: `-0.068033`
- `step_large`: `-0.194744`
- `sinusoidal_mid`: `-0.261267`

这里负号表示 `LER` 下降，即性能更好。

### 5.5 float 与 int8 一致性

相对 float：

- `linear_med`: `+0.005167`
- `step_large`: `+0.000167`
- `sinusoidal_mid`: `+0.002589`

说明：

- int8 与 float 基本重合
- P1 的量化结论在 P2 闭环里依然成立

### 5.6 commit / overflow / 预算

所有模式、所有场景下：

- `commit_count_mean = 7.0`
- `fast_cycle_violation_rate_mean = 0.0`
- `slow_update_violation_rate_mean = 0.0`

`overflow` 在不同模式间差异极小，数量级始终接近：

- `linear_med`: `~0.254 - 0.259`
- `step_large`: `~0.306 - 0.311`
- `sinusoidal_mid`: `~0.339 - 0.342`

说明：

- P2 的核心改善来自参数控制效果，而不是通过放宽数值范围“换来”的假优势

### 5.7 P2 验收判断

基于当前行为仿真结果，可以给出以下判断：

1. 双回路调度正常

- stage / commit 无毛刺迹象
- 参数 bank 切换稳定

2. 时序预算正常

- 无快回路预算违约
- 无慢回路预算违约

3. 自适应模型优于公平基线

- `model_artifact` 在三组场景下均显著优于 `fixed_baseline`
- 也显著优于 `oracle_delayed`

4. int8 可用于后续工程阶段

- 与 float 几乎重合

结论：

- `P2 通过`

---

## 6. 当前正式主模型

当前建议作为 P2 正式主模型使用的产物：

- Float：
  - [tiny_cnn_20260319_151717_b87c6c227b57.npz](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz)
- Int8：
  - [tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz)

对应配置：

- [experiment_static_theta_v2.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/experiment_static_theta_v2.yaml)
- [hardware_emulation.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/hardware_emulation.yaml)

对应真实部署导出产物：

- Float `.tflite`：
  - [tiny_cnn_20260319_151717_b87c6c227b57_tflite_20260328_012736.tflite](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_tflite_20260328_012736.tflite)
- Int8 `.tflite`：
  - [tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_tflite_20260328_012736.tflite](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_tflite_20260328_012736.tflite)

部署与独立验收入口：

- TensorFlow 运行环境：
  - [/.venvs/tf311](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/.venvs/tf311)
- 独立评测脚本：
  - [evaluate_tflite.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/evaluate_tflite.py)
- 导出一致性验收脚本：
  - [validate_export.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/validate_export.py)

---

## 7. P3 初版结论

### 7.1 本轮新增内容

与上一版阶段结论相比，当前已补齐以下 `P3` 关键工程模块：

1. `hwio/` 首版

- [axi_map.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/hwio/axi_map.py)
- [dma_client.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/hwio/dma_client.py)
- [fpga_driver.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/hwio/fpga_driver.py)
- [mock_fpga.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/hwio/mock_fpga.py)
- [board_backend.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/hwio/board_backend.py)

2. 慢回路独立推理服务

- [inference_service.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/runtime/inference_service.py)
- [inference_worker.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/runtime/inference_worker.py)

3. 部署导出与独立验收入口

- [export.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/export.py)
- [evaluate_tflite.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/evaluate_tflite.py)
- [validate_export.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/validate_export.py)

4. 独立 TensorFlow/TFLite 运行环境

- [/.venvs/tf311](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/.venvs/tf311)
- `python = 3.11.15`
- `tensorflow = 2.21.0`

5. P3 正式 benchmark 入口

- [run_hil_suite.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/run_hil_suite.py)
- [run_hil_mode_benchmark.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/run_hil_mode_benchmark.py)

### 7.2 P3 正式运行目录

修正 `.tflite` 导出语义并重新跑完后的正式 `P3` 结果位于：

- [hardware_hil_v1_20260328_012839_7b58cdf75d7a](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/hil_mode_benchmark/hardware_hil_v1_20260328_012839_7b58cdf75d7a)

核心输出：

- [summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/hil_mode_benchmark/hardware_hil_v1_20260328_012839_7b58cdf75d7a/summary.json)
- [report.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/hil_mode_benchmark/hardware_hil_v1_20260328_012839_7b58cdf75d7a/report.md)
- [mode_comparison.csv](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/hil_mode_benchmark/hardware_hil_v1_20260328_012839_7b58cdf75d7a/mode_comparison.csv)

### 7.3 P3 模式定义

本轮比较 4 种模式：

1. `fixed_baseline_mock`
2. `float_artifact_mock`
3. `int8_artifact_mock`
4. `real_board`

其中：

- `fixed_baseline_mock` 表示固定参数、不自适应
- `float_artifact_mock` 表示使用 P1 通过验收的浮点 Tiny-CNN 作为慢回路模型
- `int8_artifact_mock` 表示使用 int8 `.tflite` 产物，并通过 `backend=tflite + subprocess + .venvs/tf311/bin/python` 路径执行
- `real_board` 表示真实板卡入口，但当前机器上无对应设备节点，因此本轮只验证入口与跳过逻辑

### 7.4 P3 核心结果

1. `fixed_baseline_mock`

- `LER = 1.1987018`
- `overflow_rate = 0.2972882`
- `n_commits_applied = 1500`

2. `float_artifact_mock`

- `LER = 1.1405753`
- `overflow_rate = 0.3935607`
- `n_commits_applied = 1500`

3. `int8_artifact_mock`

- `LER = 1.1407845`
- `overflow_rate = 0.3933973`
- `n_commits_applied = 1500`

4. `real_board`

- `status = skipped_unavailable`
- `error = board_device_missing:/dev/uio0,/dev/uio1`

### 7.5 结果解读

可以明确得出 4 点：

1. 真实 `.tflite` 路径已经打通，且 `float` 与 `int8` 几乎重合

- 二者 `LER` 差异约为 `2.09e-04`
- 二者 `overflow_rate` 差异约为 `1.63e-04`
- 说明 P1 的量化结论在 P3 HIL mock 闭环中依然成立
- 也说明当前 `backend=tflite` 路径不再是 stub，而是真实部署产物在运行

2. 自适应模型相对固定基线能降低 `LER`

- `float` 相对 `fixed_baseline`：`LER` 下降约 `0.058127`
- `int8` 相对 `fixed_baseline`：`LER` 下降约 `0.057917`

3. 当前自适应参数更激进，导致 `overflow_rate` 上升

- `float` 相对 `fixed_baseline`：`overflow_rate` 上升约 `0.096273`
- `int8` 相对 `fixed_baseline`：`overflow_rate` 上升约 `0.096109`

这说明当前 `P3` 的主要剩余问题不是“模型无效”，而是：

- 自适应确实在改善纠错
- 但参数映射仍偏激进，需要继续调 `gain_clip / beta / alpha_bias / gain_scale`

4. 调度与预算本身没有异常

- `slow_update_violation_rate = 0.0`
- `fast_cycle_violation_rate = 1.75e-05`
- `n_commits_applied = 1500`
- 未出现 `dma_timeout / axi_write_failure / cnn_infer_failure`

结论是：

- 当前异常主要来自控制参数幅度，而不是调度链路本身失效

补充两组独立 `.tflite` 验收结果：

1. Float `.tflite`

- 评测报告：
  - [eval_tflite_test_20260330_001725.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/eval_tflite_test_20260330_001725.json)
- 导出一致性报告：
  - [validate_export_tiny_cnn_20260319_151717_b87c6c227b57_20260330_001725.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/validate_export_tiny_cnn_20260319_151717_b87c6c227b57_20260330_001725.json)
- 指标：
  - `MSE = 0.292359`
  - `MAE = 0.220085`
  - `R2_mean = 0.994359`
  - `max_abs_diff = 0.119338`
  - `mean_abs_diff = 0.007563`
  - `status = ok`

2. Int8 `.tflite`

- 评测报告：
  - [eval_tflite_test_20260329_235412.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/eval_tflite_test_20260329_235412.json)
- 导出一致性报告：
  - [validate_export_tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_20260329_235412.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/validate_export_tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_20260329_235412.json)
- 指标：
  - `MSE = 0.297316`
  - `MAE = 0.221375`
  - `R2_mean = 0.994192`
  - `max_abs_diff = 0.124383`
  - `mean_abs_diff = 0.008229`
  - `status = ok`

### 7.6 当前限制

虽然 `P3` 软件链路已经打通，但仍存在 3 个必须明确的限制。

1. 真实板卡未验证

- 当前环境没有 `/dev/uio0` 与 `/dev/uio1`
- 因此 `real_board` 只验证了入口结构与失败保护逻辑
- 还不能声称“真实板卡 HIL 已通过”

2. `board_backend.py` 仍是“真实接口骨架”

- 已支持 `/dev/uio + mmap + DMA buffer` 的软件结构
- 但因为缺少最终 RTL 地址表和设备节点，当前仍未完成真实寄存器逐项对齐

3. `overflow_rate` 仍明显高于固定基线

- `.tflite` 路径已验证通过，并不构成当前瓶颈
- 当前主要剩余问题是参数映射在长时间闭环下仍偏激进
- 因此后续优化重点应转向更细粒度的控制参数与数值范围诊断，而不是重新质疑模型导出链路

### 7.7 P3 当前验收判断

如果把 `P3` 拆成“工程链路打通”和“真实板卡联调”两个层次，则当前判断应写成：

1. `P3-软件链路`：通过

- 双回路事件推进已经接通
- 慢回路模型 artifact / int8 / 真实 `.tflite` 路径已经接通
- `run_hil_mode_benchmark` 已能给出可复现结果
- `evaluate_tflite.py / validate_export.py` 已完成独立验收

2. `P3-真实板卡联调`：未通过，且尚未开始正式验收

- 原因不是代码入口缺失
- 而是本机没有实际板卡设备，且缺少最终 RTL 地址表

所以当前更准确的阶段表述应为：

- `P3 软件 HIL 完成`
- `P3 真板联调待完成`

---

### 7.8 P3 参数调优跟进

在完成修正后的正式 `P3` 模式对比后，又继续做了三轮面向 `overflow_rate` 的参数调优。

第一轮 sweep：

- [hardware_hil_overflow_tuning_v1_20260320_233531_583cb3f5b61c](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_param_sweep/hardware_hil_overflow_tuning_v1_20260320_233531_583cb3f5b61c)

结论：

- 单纯压低 `gain_clip` 上界到 `0.8 / 0.75 / 0.70`
- 并不能稳定降低 `overflow_rate`
- 说明问题主矛盾不在“大增益上限过高”

第二轮 sweep：

- [hardware_hil_overflow_tuning_v2_20260320_235250_dfb3515b0016](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_param_sweep/hardware_hil_overflow_tuning_v2_20260320_235250_dfb3515b0016)

第二轮最优长确认候选为：

- `gain_clip = [0.10, 1.20]`
- `beta_smoothing = 0.20`
- `alpha_bias = 0.90`

对应结果：

- `LER = 1.069234`
- `overflow_rate = 0.386844`

对比第二轮长确认基线：

- 基线：`LER = 1.069886`，`overflow_rate = 0.387596`
- 调优后：`LER` 下降约 `0.000651`
- 调优后：`overflow_rate` 下降约 `0.000753`

虽然改进幅度不算大，但它满足两个条件：

1. 没有用更高 `LER` 去换更低 `overflow`
2. 改动方向具有清晰物理解释

- 降低 `gain_clip` 下界，相当于减弱低噪声阶段的最小校正强度
- 降低 `alpha_bias`，相当于减弱先验均值对偏置项的注入幅度

因此当前建议将 [hardware_hil.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/hardware_hil.yaml) 的默认映射参数更新为：

- `gain_clip: [0.10, 1.20]`
- `beta_smoothing: 0.20`
- `alpha_bias: 0.90`

第三轮 sweep：

- [hardware_hil_overflow_tuning_v3_20260329_235435_b867c5a238a8](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_param_sweep/hardware_hil_overflow_tuning_v3_20260329_235435_b867c5a238a8)

第三轮的关键变化是：不再只改 `gain_clip`，而是新增整体增益缩放 `gain_scale`，直接收紧 `K` 的谱半径。

短 sweep 结果显示：

- `gain_scale = 0.97`：`LER = 1.067802`，`overflow_rate = 0.369854`
- `gain_scale = 0.95`：`LER = 1.068545`，`overflow_rate = 0.368538`
- `gain_scale = 0.93`：`LER = 1.070314`，`overflow_rate = 0.367250`
- `gain_scale = 0.90`：`LER = 1.075140`，`overflow_rate = 0.365824`

这说明：

- `gain_scale` 确实是有效旋钮
- 但它表现为“越保守，overflow 越低，LER 也越差”的单调 tradeoff

由于原始 sweep 脚本的长确认没有自动带上 baseline，本轮又额外补了同口径长确认：

- 候选长确认：
  - [confirm_20260330_001435/summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_param_sweep/manual_gain_scale_confirm/confirm_20260330_001435/summary.json)
- baseline 长确认：
  - [baseline_compare_20260330_002145/summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_param_sweep/manual_gain_scale_confirm/baseline_compare_20260330_002145/summary.json)

同口径长确认对比如下：

- `baseline_current (gain_scale=1.00)`：`LER = 1.069064`，`overflow_rate = 0.387018`
- `gain_scale = 0.97`：`LER = 1.078498`，`overflow_rate = 0.387071`
- `gain_scale = 0.95`：`LER = 1.084769`，`overflow_rate = 0.386074`

据此可以得出更可靠的结论：

1. `gain_scale` 是真实有效的调参旋钮
2. 但当前测试到的 `0.97 / 0.95` 并没有在长确认里优于默认值
3. 因此当前主线默认值不应改成更小的 `gain_scale`
4. [hardware_hil.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/hardware_hil.yaml) 仍保持：

- `gain_scale: 1.0`
- `gain_clip: [0.10, 1.20]`
- `beta_smoothing: 0.20`
- `alpha_bias: 0.90`

同时，本轮还顺手修正了 sweep 工具本身：

- [run_p3_param_sweep.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/run_p3_param_sweep.py)

修正后长确认会强制包含 baseline，并显式输出相对 confirm baseline 的差值，避免后续再出现“候选都确认了，但默认值没进长跑”的比较口径问题。

---

### 7.9 P3 Overflow 来源拆分首版

为避免后续继续只盯着单一 `overflow_rate`，当前已经把 `P3` 的 overflow 诊断拆成 3 条统计链路：

1. 直方图输入饱和
2. 校正量硬饱和
3. 预测参数过激进（启发式：高增益、高偏置或接近校正上限）

相关实现已补到：

- [fast_loop_emulator.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/runtime/fast_loop_emulator.py)
- [linear_runtime.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/decoder/linear_runtime.py)
- [param_mapper.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/decoder/param_mapper.py)
- [run_hil_suite.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/run_hil_suite.py)
- [run_hil_mode_benchmark.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/run_hil_mode_benchmark.py)

首版 smoke 验证：

- [tmp_smoke_overflow_breakdown_v2/hil_summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/tmp_smoke_overflow_breakdown_v2/hil_summary.json)

在该小规模 smoke 中观察到：

- `overflow_rate = 0.269625`
- `histogram_input_saturation_rate = 0.269625`
- `correction_saturation_rate = 0.0`
- `aggressive_param_rate = 0.0`
- `dominant_overflow_source = histogram_input`

这说明至少在当前这组快速检查里：

- overflow 主要来自 syndrome / histogram 输入超范围
- 不是校正量硬饱和
- 也没有观测到明显“参数过激进”信号

需要强调：

- 这只是首版 smoke 结果，不应替代正式长跑 benchmark
- 但它已经说明“overflow 来源拆分”这条诊断链路可以工作
- 后续正式 P3 调参将不再只看单个 `overflow_rate`，而是同时看三类来源的变化

---

## 8. 下一阶段工作

虽然 `P2` 已通过，但后续仍有明确工作尚未完成。

### 8.1 P3 之后优先要补的内容

1. `overflow_rate` 收紧

- 当前不建议继续简单压小 `gain_scale`
- `overflow` 来源拆分已经完成，正式长跑确认主导项是 `histogram_input`
- 更合理的下一步不再是继续盲调主线默认值，而是在冻结口径后进入 `P4` 多场景复现

2. P4 大规模实验准备

- 把当前已验证通过的 float/int8/`.tflite` 口径整理成统一实验模板
- 冻结 baseline 集合、场景集和统计口径
- 在更多漂移场景和重复实验下确认主结论稳定性

3. 真板联调（条件性扩展）

- 获取实际地址表 / 设备节点 / bitstream 说明
- 将 [board_backend.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/hwio/board_backend.py) 逐项对齐到真实寄存器语义
- 在板卡长期不可得的情况下，这一项不再阻塞主线实验推进

### 8.2 下一阶段目标

下一阶段的主要目标应为：

- 以当前已冻结的软件 HIL 口径推进 `P4` 的多场景统计复现
- 实现 `Static Linear / Window Variance / EKF / CNN-FPGA` 四类正式 baseline 对比入口
- 在具备板卡条件后，再将 `real_board` 从“可跳过入口”推进到“真实寄存器联调”

---

## 9. 最终阶段性结论

截至本次结论更新，可以明确写出：

- `P1 已通过`
- `P2 已通过`
- `P3 软件 HIL 已通过`
- `P4 baseline` 已冻结，当前项目的下一目标是 `P4` 多场景统计复现

换句话说，当前工程已经不再处于“只是脚手架”的状态，而是已经完成了：

- 可训练主模型
- 可量化主模型
- 可导出真实 `.tflite` 部署产物
- 公平基线定义
- 双回路闭环行为仿真
- P2 多模式正式对比验证
- P3 float/int8/真实 `.tflite` 正式对比验证

但同时也必须明确：

- 真实板级 HIL 还没联调通过
- `overflow_rate` 虽已大幅压低，但主导来源仍是 `histogram_input`
- 因此项目尚未进入“整体工程完成”状态

当前最准确的定位是：

- `P3 软件链路完成且口径已稳定，主线正式进入 P4 多场景统计对比；真板验证转为条件性扩展`

---

## 10. 阶段定位与 Baseline 安排补充说明（2026-03-30 追加）

这一节专门回答 3 个问题：

1. 参考工程化实验方案，当前到底做到哪一阶段
2. 到目前为止已经拿到了哪些结果，这些结果是怎么来的、有什么意义
3. 后续正式 baseline 对比应在什么时候冻结和展开

### 10.1 当前做到哪一阶段

对照 [CNN_FPGA_GKP_工程化实验方案.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/docs/CNN_FPGA_GKP_工程化实验方案.md) 第 5 节的阶段规划，当前状态应写成：

1. `P0`：完成

- 已完成 `full_qec` 与 `simplified` 的对比
- 已确认两者在相同漂移场景下存在稳定且显著的 `LER gap`
- 因此“物理侧不能继续依赖过度简化模型”这一前提已经建立

2. `P1`：完成

- 已完成数据生成、Tiny-CNN 训练、test 集评估、int8 量化
- 已达到工程方案中对 `sigma / mu_q / mu_p / theta_deg` 的回归阈值要求
- 因此“CNN 可作为慢回路参数估计器”这一前提已经建立

3. `P2`：完成

- 已完成行为级双回路闭环
- 已完成 `fixed_baseline / oracle_delayed / model_artifact / int8_artifact` 的公平对比
- 已确认调度、commit、固定点与延迟仿真链路基本成立

4. `P3-软件HIL`：完成

- 已完成 `mock backend + artifact/int8/真实 .tflite` 的软件侧 HIL 模式对比
- 已完成真实 `.tflite` 导出、独立评测、导出一致性验证
- 已完成 `overflow breakdown`、输入范围调参和正式长跑复核
- 已确认当前主导来源是 `histogram_input`，而不是模型训练或导出语义错误

5. `P3-真板HIL`：未完成

- 原因不是工程入口缺失
- 而是当前没有 `/dev/uio0,/dev/uio1` 与最终 RTL 地址表

6. `P4`：已可开始

- 软件 HIL 主线的默认参数、统计字段和正式 baseline 集合已经满足冻结条件
- 后续工作重点应转向“多基线、多噪声场景、资源/功耗/报告”这一最终对比阶段

所以一句话概括当前阶段：

- 当前已经走到 `P3-软件HIL通过、P4 可正式展开、P3-真板HIL待条件补齐`

### 10.2 已有结果、来源与意义

下面按阶段汇总当前已经拿到的主要结果。

#### 10.2.1 P0：物理仿真基线已经建立

结果来源：

- [drift_v1_20260317_154905_87cc72d5c4de/summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/drift_suite/drift_v1_20260317_154905_87cc72d5c4de/summary.json)

结果怎么来的：

- 在 `linear_low / step_mid / sinusoidal / random_walk` 4 组漂移场景上
- 每组 `n_rounds = 2000`
- 每组 `repeats = 10`
- 对比 `full_qec` 与 `simplified` 两种物理模型下的最终 `LER`

关键结果：

- `linear_low`：`full = 0.4237`，`simplified = 0.0205`，`gap = 0.4032`
- `step_mid`：`full = 0.42665`，`simplified = 0.01855`，`gap = 0.40810`
- `sinusoidal`：`full = 0.42530`，`simplified = 0.02405`，`gap = 0.40125`
- `random_walk`：`full = 0.41200`，`simplified = 0.01565`，`gap = 0.39635`

结果意义：

1. `simplified` 明显低估逻辑错误率
2. 这说明后续工程验证不能把简化物理模型当成可信主结论来源
3. 它为后面的 `P1/P2/P3` 建立了一个很重要的约束：

- 我们可以在工程上做近似
- 但最终对“纠错是否真的变好”的判断，必须回到更严格的物理口径上

#### 10.2.2 P1：慢回路主模型已经训练完成并通过量化验收

结果来源：

- Float test 报告：
  - [eval_test_20260319_151756.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/eval_test_20260319_151756.json)
- Int8 test 报告：
  - [eval_test_20260319_151817.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/eval_test_20260319_151817.json)

结果怎么来的：

- 先修正数据集，使样本从“各向同性高斯”改成“可辨识 `theta_deg` 的各向异性高斯”
- 再训练 `static_theta_v2` 版 Tiny-CNN
- 最后在 test 集 `1639` 个样本上分别评估 float 和 int8 版本

关键结果：

1. Float 模型

- `MSE = 0.293336`
- `MAE = 0.220503`
- `R2_mean = 0.994352`
- 逐标签：
  - `sigma`: `R² = 0.997613`
  - `mu_q`: `R² = 0.996473`
  - `mu_p`: `R² = 0.998459`
  - `theta_deg`: `R² = 0.984862`

2. Int8 模型

- `MSE = 0.297742`
- `MAE = 0.221944`
- `R2_mean = 0.994212`
- 逐标签：
  - `sigma`: `R² = 0.997730`
  - `mu_q`: `R² = 0.996324`
  - `mu_p`: `R² = 0.998160`
  - `theta_deg`: `R² = 0.984634`

结果意义：

1. 当前主模型已经不仅能学 `sigma / mu_q / mu_p`，也能稳定学到 `theta_deg`
2. Int8 与 float 几乎重合，说明量化不会破坏主模型语义
3. 这使得后续 `P2/P3` 的慢回路不再只是“脚手架 mock”，而是可以挂真实部署候选模型

#### 10.2.3 P2：行为级双回路与公平基线对比已经完成

结果来源：

- [hardware_emulation_v1_20260319_160130_20670d1c0d1f/report.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p2_mode_benchmark/hardware_emulation_v1_20260319_160130_20670d1c0d1f/report.md)

结果怎么来的：

- 在软件里模拟快回路固定点、窗口采样、参数 bank、延迟注入和 stage/commit
- 在 `linear_med / step_large / sinusoidal_mid` 三类漂移场景上
- 对比 4 种模式：
  - `fixed_baseline`
  - `oracle_delayed`
  - `model_artifact`
  - `int8_artifact`

关键结果：

1. 相对 `fixed_baseline`，`model_artifact` 在 3 个场景都明显降低 `LER`

- `linear_med`：`0.816067 -> 0.696206`
- `step_large`：`0.937822 -> 0.731906`
- `sinusoidal_mid`：`1.019033 -> 0.759289`

2. `int8_artifact` 与 float 基本重合

- `linear_med`：float/int8 差 `0.005167`
- `step_large`：float/int8 差 `0.000167`
- `sinusoidal_mid`：float/int8 差 `0.002589`

3. 调度与运行时统计正常

- `commit_count_mean = 7.0`
- `fast_cycle_violation_rate_mean = 0.0`
- `slow_update_violation_rate_mean = 0.0`
- 各模式间 `overflow` 没有出现异常级别的放大

结果意义：

1. 当前 `ParamMapper + 双回路调度 + 参数 bank` 的主线逻辑是成立的
2. 自适应模型不是“看起来在工作”，而是在公平基线下确实改善了行为级 `LER`
3. P2 的结论为进入 P3 提供了足够强的工程依据：

- 模型有效
- 量化可用
- 调度可用

#### 10.2.4 P3：软件侧 HIL 与真实 `.tflite` 已经打通

结果来源：

- HIL 模式对比：
  - [hardware_hil_v1_20260328_012839_7b58cdf75d7a/report.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/hil_mode_benchmark/hardware_hil_v1_20260328_012839_7b58cdf75d7a/report.md)
- Float `.tflite` 独立评测：
  - [eval_tflite_test_20260330_001725.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/eval_tflite_test_20260330_001725.json)
- Int8 `.tflite` 独立评测：
  - [eval_tflite_test_20260329_235412.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/eval_tflite_test_20260329_235412.json)
- Float 导出一致性：
  - [validate_export_tiny_cnn_20260319_151717_b87c6c227b57_20260330_001725.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/validate_export_tiny_cnn_20260319_151717_b87c6c227b57_20260330_001725.json)
- Int8 导出一致性：
  - [validate_export_tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_20260329_235412.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/static_theta_v2/validate_export_tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_20260329_235412.json)

结果怎么来的：

- 在 `mock FPGA backend` 下跑双回路事件推进
- 用 `subprocess + .venvs/tf311/bin/python` 调用真实 TensorFlow/TFLite 解释器
- 对比 `fixed_baseline_mock / float_artifact_mock / int8_artifact_mock / real_board`
- 同时把导出的 `.tflite` 单独拿出来做 test 集评测和 artifact 对齐验证

关键结果：

1. HIL 模式对比

- `fixed_baseline_mock`: `LER = 1.1987018`，`overflow_rate = 0.2972882`
- `float_artifact_mock`: `LER = 1.1405753`，`overflow_rate = 0.3935607`
- `int8_artifact_mock`: `LER = 1.1407845`，`overflow_rate = 0.3933973`
- `real_board`: `skipped_unavailable`

2. Float `.tflite`

- `MSE = 0.292359`
- `MAE = 0.220085`
- `R2_mean = 0.994359`
- `max_abs_diff = 0.119338`
- `mean_abs_diff = 0.007563`
- `status = ok`

3. Int8 `.tflite`

- `MSE = 0.297316`
- `MAE = 0.221375`
- `R2_mean = 0.994192`
- `max_abs_diff = 0.124383`
- `mean_abs_diff = 0.008229`
- `status = ok`

结果意义：

1. 真实 `.tflite` 路径已经不是概念验证，而是实际跑通并验收过
2. float 与 int8 在 P3 闭环里几乎重合，说明部署产物没有明显语义漂移
3. 现在 P3 的主要矛盾已经被定位得很清楚：

- 不是训练不行
- 不是量化不行
- 不是 `.tflite` 导出不行
- 而是 `histogram_input` 在旧默认值下统计范围过窄，导致 `overflow_rate` 偏高

#### 10.2.5 P3 补充：第三轮 overflow 调参结果

结果来源：

- 第三轮 sweep：
  - [hardware_hil_overflow_tuning_v3_20260329_235435_b867c5a238a8](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_param_sweep/hardware_hil_overflow_tuning_v3_20260329_235435_b867c5a238a8)
- 补充长确认：
  - [confirm_20260330_001435/summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_param_sweep/manual_gain_scale_confirm/confirm_20260330_001435/summary.json)
- baseline 长确认：
  - [baseline_compare_20260330_002145/summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_param_sweep/manual_gain_scale_confirm/baseline_compare_20260330_002145/summary.json)

结果怎么来的：

- 在前两轮发现 `gain_clip / alpha_bias` 调整幅度有限后
- 第三轮新增 `gain_scale`，直接缩放 `K` 的谱半径
- 先短时筛选，再补同口径长确认

关键结果：

- `baseline_current (gain_scale = 1.00)`：`LER = 1.069064`，`overflow = 0.387018`
- `gain_scale = 0.97`：`LER = 1.078498`，`overflow = 0.387071`
- `gain_scale = 0.95`：`LER = 1.084769`，`overflow = 0.386074`

结果意义：

1. `gain_scale` 是真有效的工程旋钮
2. 但当前测到的 `0.97 / 0.95` 都没有形成优于默认值的长时 tradeoff
3. 因此当前主线默认值仍应保持 `gain_scale = 1.0`
4. 后续优化不能只靠“把增益整体压小”，而要继续拆解 overflow 的来源

#### 10.2.6 P3 补充：overflow 来源拆分、输入范围调参与正式复跑

结果来源：

- 旧默认值下的新统计口径正式长跑：
  - [hardware_hil_v1_20260330_233853_c9ed542a0cfa](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/hil_mode_benchmark/hardware_hil_v1_20260330_233853_c9ed542a0cfa)
- 定向输入范围调参：
  - [hardware_hil_histogram_tuning_20260331_001959_bfea868ecd4f](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_histogram_tuning/hardware_hil_histogram_tuning_20260331_001959_bfea868ecd4f)
- 更新正式默认值后的 mode benchmark 复跑：
  - [hardware_hil_v1_20260331_010329_c128fa34262e](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/hil_mode_benchmark/hardware_hil_v1_20260331_010329_c128fa34262e)

结果怎么来的：

- 先在 `fast_loop_emulator` 中把 overflow 拆成三类：
  - `histogram_input_saturation_rate`
  - `correction_saturation_rate`
  - `aggressive_param_rate`
- 再用正式长配置确认旧默认值下到底是哪一类 overflow 主导
- 如果主导项仍是 `histogram_input`，则不再盲目压 `gain_scale`
- 而是转向调 `syndrome_limit / histogram_range_limit / sigma_measurement`
- 最后把最优组合写回正式配置，再做一次完整 `mode benchmark`

关键结果：

1. 旧默认值下的新统计口径正式长跑

- `fixed_baseline_mock`: `overflow_rate = 0.2972187`，主导来源 `histogram_input`
- `float_artifact_mock`: `overflow_rate = 0.3940017`，主导来源 `histogram_input`
- `int8_artifact_mock`: `overflow_rate = 0.3936963`，主导来源 `histogram_input`
- 三种模式均满足：
  - `correction_saturation_rate = 0`
  - `aggressive_param_rate = 0`

2. 输入范围侧长确认最优组合

- `syndrome_limit = 1.441311257912825`
- `histogram_range_limit = 1.8799712059732503`
- `sigma_measurement = 0.03`
- 相对旧默认 baseline：
  - `LER`: `1.069503 -> 1.046678`
  - `histogram_input_saturation_rate`: `0.387092 -> 0.022181`

3. 更新默认值后的正式 mode benchmark

- `fixed_baseline_mock`: `LER = 1.1993300`，`overflow_rate = 0.0161568`
- `float_artifact_mock`: `LER = 1.1238110`，`overflow_rate = 0.0227638`
- `int8_artifact_mock`: `LER = 1.1246990`，`overflow_rate = 0.0226377`
- `real_board`: `skipped_unavailable`
- 三种可运行模式仍然一致满足：
  - `dominant_overflow_source = histogram_input`
  - `correction_saturation_rate = 0`
  - `aggressive_param_rate = 0`

结果意义：

1. `overflow` 的主导机制已经稳定识别出来，结论不是偶然短跑波动
2. 旧默认值的核心问题不是 `gain_scale` 太大，而是输入统计范围过窄
3. 当前正式默认值已经把长跑 overflow 从三四成量级压到约 `1.6% ~ 2.3%`
4. `float` 与 `int8` 在新默认值下仍然高度重合，说明部署链路结论稳定
5. 这意味着软件 HIL 主线的运行口径已经足够稳定，可以冻结 `P4` baseline 集合并开始多场景对比

### 10.3 后续什么时候设置 baseline 进行对比

这个问题要分“当前已经设置过的 baseline”和“后续正式大对比的 baseline”两层来回答。

#### 10.3.1 当前其实已经设置了 baseline

1. `P0` 的 baseline

- `full_qec` 对比 `simplified`
- 用来确认物理主口径必须以更严格模型为准

2. `P2` 的 baseline

- `fixed_baseline`
- `oracle_delayed`

这两个 baseline 的作用分别是：

- `fixed_baseline`：代表“不自适应，只用固定参数”的工程下界
- `oracle_delayed`：代表“延迟一窗的公平上界参考”

3. `P3` 的 baseline

- `fixed_baseline_mock`

它的作用是：

- 在软件 HIL 口径下，提供一个不依赖慢回路模型的统一参照
- 用来判断 float/int8/`.tflite` 版本到底有没有真正带来收益

所以从工程推进角度看：

- baseline 不是“后面才要开始设”
- 而是 `P0/P2/P3` 已经分别设过，并已经用于当前所有关键结论

#### 10.3.2 后续正式 baseline 对比应在什么时候冻结

后续真正需要“冻结一整套 baseline 并做系统化对比”的时点，是进入 `P4` 之前。

更准确地说，应在满足下面两个条件后立即冻结：

1. `P3-软件HIL` 参数口径稳定

- 主模型固定
- `ParamMapper` 公式固定
- 默认运行参数固定
- `overflow_rate` 不再处于明显异常波动阶段

2. 统计口径固定

- 统一场景集
- 统一 seed 策略
- 统一运行轮数
- 统一日志与汇总字段

截至 `2026-03-31`，软件 HIL 主线已经基本满足这两个条件：

- 主模型已固定到 `static_theta_v2` 的 Tiny-CNN 主模型
- `ParamMapper` 公式和默认运行参数已按长确认结果回写
- overflow 主导来源已稳定识别为 `histogram_input`
- 新正式默认值下 `float/int8` 结果继续对齐
- 因而 `P4` 的 baseline 集合可以正式冻结，不需要再等真板

只有在这两个条件满足后，才适合进入 `P4` 做大规模基线对比；否则会出现一个问题：

- baseline 还没冻结
- 主模型和运行参数却还在变化
- 那么后面的对比结论就容易失去可比性

#### 10.3.3 P4 正式对比时建议冻结的 baseline 集合

参考工程化方案第 7 节，进入 `P4` 后建议固定以下 baseline 集合：

1. `Static Linear`

- 固定参数线性解码
- 这是最核心、也最不能缺的工程 baseline

2. `Window Variance`

- 用简单窗口统计更新参数
- 用来回答“是否真的需要 CNN，而不是简单统计就够了”

3. `EKF`

- 作为经典模型驱动追踪基线
- 用来回答“CNN 学习型慢回路是否优于传统滤波器”

4. `CNN-FPGA`

- 当前主方案
- 即最终要证明的目标方法

5. `MWPM`（可选扩展）

- 可作为更强参考
- 但不应阻塞主线实验

#### 10.3.4 因此后续 baseline 的时间安排应写成

1. 现在到 `P3` 收敛前

- 继续使用当前已经建立的 `fixed_baseline / oracle_delayed / fixed_baseline_mock`
- 这些 baseline 足以支撑参数调优、导出验收和软件 HIL 判断

2. `P3` 软件口径收敛后

- 冻结 `P4` 使用的 baseline 集合与统计口径
- `2026-03-31` 已完成这一步
- 这一步是“正式大对比开始前”的分界线

3. `P4` 开始后

- 不再频繁改 baseline 定义
- 重点转向多噪声场景、重复实验、资源/延迟/功耗和真板复现

换句话说：

- 当前 baseline 已经足够支持“工程链路是否成立”的判断
- 后续要做的是在 `P3` 稳定后，把 baseline 集合正式冻结，然后进入 `P4` 做系统化对比

---

## 11. 2026-04-01 正式 P4 更新

### 11.1 当前主线已经从“准备进入 P4”推进到“P4 首轮正式结果已产出”

本轮新增了 runtime-consistent 的 `residual_b` 数据链路，并将慢回路主方案从“直接回归 `(sigma, mu_q, mu_p, theta)`”切到：

- `Window Variance / EKF` 负责给出稳定 teacher
- CNN 只学习对运行时 `b` 的残差修正
- 输入显式包含时间上下文、histogram 差分、teacher 预测和 teacher 参数

对应正式数据、模型与报告路径为：

- 数据集：
  - [manifest.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/datasets/runtime_b_residual_v1/manifest.json)
- 模型：
  - [tiny_cnn_20260401_083648_2fc740424c0d.npz](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz)
- test 评估：
  - [eval_test_20260401_083649.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/reports/runtime_b_residual_v1/eval_test_20260401_083649.json)
- 正式多场景 benchmark：
  - [summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_hybrid_b_v1_20260401_083649_41775bdd90b1_11082/summary.json)
  - [comparison.csv](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_hybrid_b_v1_20260401_083649_41775bdd90b1_11082/comparison.csv)
  - [report.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_hybrid_b_v1_20260401_083649_41775bdd90b1_11082/report.md)

### 11.2 residual-b 模型本身的离线表现

在 test 集上：

- `MSE = 2.41445e-06`
- `MAE = 1.32358e-03`
- `R2_mean = 0.09905`

逐标签：

- `b_q`: `R² = 0.01346`
- `b_p`: `R² = 0.18465`

这说明：

- 模型已经不再是早期 `delta_mu` 版本那种“几乎只输出常数”的塌缩状态
- 但它仍明显欠拟合，尤其是 `b_q`
- 当前优势主要来自“teacher 基线已经足够稳，CNN 只做小幅、局部、时间一致的残差修正”

因此当前更准确的判断不是“CNN 已经学得很好”，而是：

- `CNN + teacher residual-b` 这条工程路线已经被验证为有效
- 但 CNN 本体还没有达到完全令人满意的辨识能力

### 11.3 P4 正式 benchmark 结果

本轮正式对比模式为：

1. `Window Variance`
2. `EKF`
3. `Constant Residual-Mu`
4. `CNN-FPGA`（旧的绝对参数回归主模型）
5. `Hybrid Residual-B`

跨 4 个正式场景的平均 `LER` 为：

- `Hybrid Residual-B = 0.850799`
- `Constant Residual-Mu = 0.855549`
- `EKF = 0.855779`
- `Window Variance = 0.857016`
- `CNN-FPGA = 0.954315`

逐场景最佳模式全部为 `Hybrid Residual-B`：

- `linear_ramp`: `0.851224`，相对次优 `Constant Residual-Mu` 再降 `0.000769`
- `periodic_drift`: `0.848230`，相对次优 `EKF` 再降 `0.007000`
- `static_bias_theta`: `0.859707`，相对次优 `Constant Residual-Mu` 再降 `0.002810`
- `step_sigma_theta`: `0.844036`，相对次优 `EKF` 再降 `0.007136`

这意味着：

- 从正式统计口径看，当前主线方案已经从“落后于简单统计基线”修正为“稳定优于已冻结正式基线集合中的其余模式”
- 从工程逻辑看，真正有效的不是“直接让 CNN 取代一切”，而是“teacher 提供稳态，CNN 负责小残差补偿”

### 11.4 当前风险点

虽然本轮主结论成立，但仍有两个不能忽略的风险：

1. `static_bias_theta` 存在跨 seed 波动

- `Hybrid Residual-B` 在该场景两次正式 repeat 的 `LER` 分别为：
  - `0.854399`
  - `0.865016`
- 对应 `final_ler_std = 0.005309`

这个波动量级明显高于其它场景，因此需要单独做更多 repeats 检查它是否只是 seed 差异，还是该场景本身对当前 residual-b 路线更敏感。

2. overflow 主导来源仍是 `histogram_input`

本轮所有正式模式均满足：

- `n_commits_applied` 基本为 `600`
- `slow_update_violation_rate = 0`
- `correction_saturation_rate = 0`
- `aggressive_param_rate = 0`

但主导 overflow 来源仍一致为：

- `histogram_input`

这说明当前瓶颈已经很明确：

- 不是参数映射过激进
- 不是 commit / 调度异常
- 而是输入统计范围仍会让窗口直方图在进入慢回路前发生饱和

### 11.5 因此下一步应如何推进

在冻结当前 `hybrid_residual_b` 模型的前提下，下一步最合理的顺序是：

1. 先定向复查 `static_bias_theta` 的跨 seed 波动
2. 只调输入统计范围，继续压 `histogram_input overflow`
3. 在新输入范围下做更长配置复验
4. 若优势仍保持，再把这一版口径视为当前正式主线

结论更新为：

- `P3-软件HIL`：通过
- `P4-首轮正式多场景 benchmark`：通过，并已得到当前最优方案
- 当前最重要的后续工作不再是大改模型结构，而是先把统计稳定性和输入范围问题继续收敛

## 12. 2026-04-02 长配置复验更新

### 12.1 输入统计范围已提升为当前正式默认值

在完成 `static_bias_theta` 的跨 seed 复查后，确认旧主线中最主要的问题仍然是：

- `histogram_input overflow`

因此又做了一轮冻结模型前提下的输入范围定向长确认，结果位于：

- [summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p3_histogram_tuning/hybrid_b_histogram_tuning_20260401_144804_19d9812a12db/summary.json)

最终将以下参数提升为当前正式默认值：

- `syndrome_limit = 1.566643`
- `histogram_range_limit = 2.255965971`

对应长确认中最优候选 `syndrome_125_hist_180` 相对旧 baseline：

- `LER`: `1.388708 -> 1.375373`
- `histogram_input_saturation_rate`: `0.021543 -> 0.002756`

这说明：

- 继续放宽输入统计范围不但没有恶化 `LER`
- 反而显著压低了窗口输入饱和

### 12.2 更长配置正式 P4 复验结果

基于新默认值，又完成了一轮更长配置的正式复验：

- 配置：[p4_multiscenario_hybrid_b_long.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/p4_multiscenario_hybrid_b_long.yaml)
- 结果：
  - [summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_hybrid_b_long_v1_20260402_145451_94eb56a87b59_38723/summary.json)
  - [comparison.csv](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_hybrid_b_long_v1_20260402_145451_94eb56a87b59_38723/comparison.csv)

本轮只比较三条主线：

1. `EKF`
2. `Constant Residual-Mu`
3. `Hybrid Residual-B`

跨 4 个正式场景的平均 `LER` 为：

- `Hybrid Residual-B = 0.798807`
- `Constant Residual-Mu = 0.826193`
- `EKF = 0.828108`

相对次优模式的平均优势已经扩大到：

- 对 `Constant Residual-Mu`：约 `0.027386`
- 对 `EKF`：约 `0.029300`

逐场景仍然全部由 `Hybrid Residual-B` 取得最优：

- `static_bias_theta`: `0.812559`，领先次优 `0.024062`
- `linear_ramp`: `0.787913`，领先次优 `0.029804`
- `step_sigma_theta`: `0.787615`，领先次优 `0.032185`
- `periodic_drift`: `0.807143`，领先次优 `0.023492`

### 12.3 这轮结果的真正意义

与 `2026-04-01` 的首轮正式 P4 相比，这轮结果说明了两件事：

1. `Hybrid Residual-B` 的优势不是偶然 seed 现象

- 在更长配置、更多重复下，它不仅继续保持最优
- 而且领先幅度已经从“略优”扩大为“稳定且清楚的优势”

2. 之前限制主线发挥的关键瓶颈确实是输入统计饱和

旧正式口径中，各模式的 `histogram_input_saturation_rate` 大约还在 `0.019~0.020` 量级；  
本轮长配置里，三条主线的跨场景平均值已经下降到：

- `Hybrid Residual-B = 0.002541`
- `EKF = 0.002564`
- `Constant Residual-Mu = 0.002576`

这说明：

- 当前主瓶颈判断是正确的
- 输入范围调优属于“有效修正”，不是偶然噪声

### 12.4 `static_bias_theta` 波动已显著收敛

此前对 `static_bias_theta` 的 8-seed 复查显示：

- `Hybrid Residual-B` 的 `LER std = 0.005021`

本轮新默认值下的长配置复验中：

- `Hybrid Residual-B` 在该场景的 `LER std = 0.000629`

即标准差下降约：

- `87.5%`

因此，先前该场景中的明显跨 seed 波动，现在已经基本可以解释为：

- 输入统计范围导致的窗口观测饱和敏感性

而不是：

- 参数映射方向错误
- commit / 调度异常
- 校正量饱和

### 12.5 当前阶段判断更新

截至 `2026-04-02`，可以把当前主线判断更新为：

- `Hybrid Residual-B` 已经稳定超过 `EKF / Constant Residual-Mu`
- 这种优势在更长配置下仍成立
- 当前正式主线可以冻结为“新输入范围默认值 + Hybrid Residual-B”

因此后续更值得推进的方向，不再是继续盲调主模型默认值，而是：

1. 补更强的经典 baseline
2. 补机制分析和 ablation
3. 为论文准备更有说服力的对比与叙事

## 13. 2026-04-03 强 baseline 正式扩展更新

### 13.1 修正后的 `UKF` 已成为当前最强经典 baseline

在补入 `RLS Residual-B` 后，又继续完成了 `UKF` 的正式接入和长配置多场景 benchmark：

- 配置：[p4_multiscenario_strong_baselines.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/p4_multiscenario_strong_baselines.yaml)
- 结果：
  - [summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_strong_baselines_v1_20260403_145747_b82874392710_86447/summary.json)
  - [comparison.csv](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_strong_baselines_v1_20260403_145747_b82874392710_86447/comparison.csv)

本轮正式对比集合为：

1. `EKF`
2. `UKF`
3. `Constant Residual-Mu`
4. `RLS Residual-B`
5. `Hybrid Residual-B`

4 个正式场景上的平均 `LER` 为：

- `Hybrid Residual-B = 0.798332`
- `UKF = 0.817974`
- `Constant Residual-Mu = 0.825719`
- `RLS Residual-B = 0.827908`
- `EKF = 0.828369`

这说明：

- 修正后的 `UKF` 已经超过 `EKF / RLS Residual-B / Constant Residual-Mu`
- 它是当前最强的经典 baseline
- 但仍落后于 `Hybrid Residual-B` 约 `0.019642`

### 13.2 `UKF` 提升的来源

`UKF` 首版 smoke 只表现出“略好于 EKF”的趋势；在保持公平观测口径不变的前提下，又进行了两项关键修正：

1. 保留 full covariance，而不再退化成对角协方差
2. 每步更新后做对称化与正定稳定化，避免长配置下协方差数值漂移

正式结果表明，这两项修正是有效的：

- `static_bias_theta`: `UKF = 0.826767`，优于 `EKF = 0.838399`
- `linear_ramp`: `UKF = 0.810263`，优于 `EKF = 0.820161`
- `step_sigma_theta`: `UKF = 0.812898`，优于 `EKF = 0.821733`
- `periodic_drift`: `UKF = 0.821967`，优于 `EKF = 0.833184`

即 `UKF` 在 4 个正式场景中全部优于 `EKF`。

### 13.3 当前强 baseline 结论更新

截至 `2026-04-03`，当前正式强 baseline 排序可以更新为：

1. `Hybrid Residual-B`
2. `UKF`
3. `Constant Residual-Mu`
4. `RLS Residual-B`
5. `EKF`

同时需要强调：

- 各模式 `correction_saturation_rate = 0`
- 各模式 `aggressive_param_rate = 0`
- 主导 overflow 来源仍然统一落在 `histogram_input`

因此，`Hybrid Residual-B` 当前的优势仍然主要来自：

- 对慢回路漂移参数的更有效估计

而不是：

- 通过更激进的 `K,b` 把系统推到饱和边缘

### 13.4 后续主线更新

由于 `UKF` 已经完成“更强经典滤波”这一级别的验证，下一步更值得补入的经典对照不再是继续微调 `UKF`，而是：

1. `Particle Filter`
2. 更强的 residual 线性自适应模型（若有必要）
3. 机制分析 / ablation / 论文叙事整理

## 14. 2026-04-05 Features 正式结果与迁移建议更新

### 14.1 `features` 正式 ablation 已完成

虽然一次意外重启清空了本机后台进程，但 `features` 正式 benchmark 的结果文件并未丢失，已确认完整落盘：

- 原始结果：
  - [comparison.csv](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_hybrid_vs_ukf_ablation_features_v1_20260404_211945_a7d2984ad50e_23637/comparison.csv)
  - [summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_hybrid_vs_ukf_ablation_features_v1_20260404_211945_a7d2984ad50e_23637/summary.json)
- 自动汇总：
  - [report.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_features_summary/features_summary_20260405_145948/report.md)
  - [table.csv](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_features_summary/features_summary_20260405_145948/table.csv)
  - [summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_features_summary/features_summary_20260405_145948/summary.json)

跨 4 个正式场景的平均 `LER` 为：

- `UKF = 0.818081`
- `Hybrid Full = 0.798355`
- `Hybrid No HistDelta = 0.826422`
- `Hybrid No TeacherPred = 0.807556`
- `Hybrid No TeacherParams = 0.749436`
- `Hybrid No TeacherDelta = 0.800473`

### 14.2 当前对 `features` 结果的初步解读

从工程角度看，当前 `features` 结果可以先得出两条可信结论：

1. `histogram delta` 通道是重要的

- 去掉后，`LER` 从 `0.798355` 明显恶化到 `0.826422`
- 甚至已经劣于 `UKF = 0.818081`

这说明：

- 当前主线性能的一部分确实来自“窗口级变化趋势”信息
- 不是单帧统计就足够

2. `teacher prediction` 通道有价值，但不是唯一关键来源

- 去掉后，`LER` 从 `0.798355` 上升到 `0.807556`
- 性能变差，但仍优于 `UKF`

这说明：

- learner 不只是单纯复制 teacher prediction
- 但 teacher 的显式预测值仍然提供了有效条件信息

### 14.3 当前存在的异常点

`features` 结果中最需要后续复查的是：

- `Hybrid No TeacherParams = 0.749436`

它不仅优于 `Hybrid Full`，甚至大幅优于所有其他模式。

这与直觉上的标准 ablation 预期不一致，因此当前不应直接把它解释为：

- “teacher params 本身有害”

更合理的工程判断是：

1. 当前 `teacher params` 通道的构造、缩放或归一化可能存在冗余或泄漏式干扰
2. 也可能是当前 `No TeacherParams` 模型偶然更契合这组配置，需要进一步多 seed 复查
3. 在确认之前，不能把这个结果直接写成最终论文结论

因此，`features` 组虽然已经跑完，但仍需要下一步专门复查：

1. `No TeacherParams` 跨 seed 稳定性
2. `teacher params` 通道的数值尺度与特征耦合关系
3. 是否存在“teacher prediction` + `teacher deltas` 已足够，而 `teacher params` 造成冗余输入污染”的现象

### 14.4 是否建议迁移到 Windows + RTX4070 笔记本继续运行

结论先写在前面：

- 可以迁移
- 但当前代码主线不会直接吃到 `RTX 4070` 的 GPU 加速
- 真正可能带来速度提升的，主要是 Windows 笔记本的 CPU、散热和持续功耗能力，而不是显卡本身

原因如下：

1. 当前主模型训练是纯 `NumPy`

- 文件：[tiny_cnn.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/tiny_cnn.py)
- 当前 `Tiny-CNN` 是手写 `NumPy` 实现，不依赖 `PyTorch` 或 `TensorFlow` 的 CUDA 训练后端

这意味着：

- 即使换到 `RTX 4070 Laptop`，当前训练不会自动走 GPU
- 是否更快，主要取决于新机器的 CPU 和 BLAS 实现，而不是显卡

2. 当前大部分 benchmark 也是 CPU 主导

包括：

- 物理仿真
- runtime dataset builder
- `EKF / UKF / RLS / PF`
- HIL mock
- 多场景 benchmark

这些流程大量是：

- Python 控制逻辑
- `NumPy`
- 小矩阵递推
- 窗口级循环

因此它们更偏 CPU / 内存 / 散热，而不是 GPU。

3. 若迁移后仍沿用当前实现，速度提升通常是“中等”，而不是“数量级”

更现实的判断是：

- 如果新 Windows 笔记本的 CPU 持续性能明显强于当前无独显 MacBook Air
- 那么 dataset build、正式 benchmark、长时间多场景对比通常会更快
- 但大概率是 `1.2x ~ 3x` 这类提升，而不是 `10x` 以上的 GPU 级提升

4. 若未来把训练主模型改成 `PyTorch/CUDA` 或 `TensorFlow GPU`，RTX 4070 才会明显发挥作用

在那种情况下：

- CNN 训练和批量评估确实可能显著提速
- 但当前代码主线还没改到这一步

### 14.5 迁移建议

如果迁移到 Windows 继续做后续工作，我的建议是：

1. 可以迁移，并优先把它作为长跑 benchmark 机器

- 更适合跑：
  - 多 seed 正式对比
  - 更长配置 benchmark
  - `No TeacherParams` 复查

2. 不要把迁移收益预期建立在 GPU 上

- 当前收益更可能来自：
  - 更强 CPU
  - 更大功耗墙
  - 更好散热
  - 长时间跑任务不降频

3. 若想真正利用 `RTX 4070`，后续应把训练主路径改到 GPU 框架

- 例如：
  - `PyTorch`
  - 或真正可用的 `TensorFlow GPU`

但这属于新的工程任务，不能和当前结果直接混为一谈

### 14.6 当前迁移前的状态总结

截至本节更新时：

1. `teacher` 正式 ablation：完成
2. `context` smoke：完成
3. `features` 正式 ablation：完成
4. `features` 自动汇总脚本：完成
5. 论文提纲 / 摘要 / 贡献点草稿：完成

当前最值得迁移后继续推进的任务已经更新为：

1. 复查 `No TeacherParams` 异常优势是否稳定
2. 回写 `features` 结果到阶段结论、工程化实验方案和论文草稿
3. 继续做论文表格、机制分析与图表整理
