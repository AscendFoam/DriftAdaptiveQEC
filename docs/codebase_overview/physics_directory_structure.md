# physics 目录代码功能与调用关系详解

本文档目标：
1. 给出 `physics/` 目录内每个代码文件的功能定位。
2. 逐个说明类与函数的输入、输出、内部逻辑与副作用。
3. 梳理跨文件调用链与典型执行路径。
4. 帮你在阅读源码时快速定位“从哪里来、到哪里去、为什么这样写”。

适用代码版本：
- 目录：`physics/`
- 覆盖文件：
  - `physics/__init__.py`
  - `physics/gkp_state.py`
  - `physics/noise_channels.py`
  - `physics/syndrome_measurement.py`
  - `physics/error_correction.py`
  - `physics/logical_tracking.py`

阅读建议：
1. 先看“第 1 章 总体架构”和“第 2 章 依赖图”。
2. 再按“状态 -> 噪声 -> 测量 -> 解码 -> 逻辑错误统计”顺序看每个文件。
3. 最后看“第 9 章 典型执行流程”，把函数串起来。

---

## 0. 目录结构速览

`physics/` 目录树：
- `__init__.py`：统一导出 API。
- `gkp_state.py`：GKP 态构造与 Wigner 近似。
- `noise_channels.py`：噪声通道建模（光子损失、热噪声、位移噪声、相位噪声）。
- `syndrome_measurement.py`：综合征测量模型（理想与真实）。
- `error_correction.py`：线性解码器、QEC 单轮和多轮仿真。
- `logical_tracking.py`：逻辑错误跟踪、滑窗统计、实验级统计与误差累积仿真入口。

定位一句话：
- 这个目录本质上是“物理过程 + 解码过程 + 统计过程”的一条完整仿真链。

---

## 1. 总体架构与职责分层

分层视角：
1. `gkp_state.py`：提供“状态表示基底”和晶格常量。
2. `noise_channels.py`：对相空间分布施加噪声演化。
3. `syndrome_measurement.py`：将真实误差映射成可观测 syndrome。
4. `error_correction.py`：根据 syndrome 做线性纠错并形成残差。
5. `logical_tracking.py`：把“残差随轮次演化”转为逻辑错误率（LER）统计。

这套链路服务两类任务：
1. 研究任务：分析在不同噪声/漂移参数下 LER 怎么变化。
2. 工程任务：给 `cnn_fpga` 的训练、benchmark、HIL 脚手架提供可对照的物理仿真后端。

系统核心共享常量：
- `LATTICE_CONST = sqrt(2*pi)`，定义 GKP 晶格尺度。

系统核心状态对象：
1. `ApproximateGKPState`：近似 GKP 态。
2. `NoiseParameters`：噪声参数集合。
3. `MeasurementConfig`：测量模型参数集合。
4. `DecoderParameters`：线性解码参数集合。
5. `LogicalErrorTracker`：逻辑错误统计状态机。

---

## 2. 跨文件依赖图（静态）

从 import 关系看：
- `physics/__init__.py`
  - 导出：`gkp_state`、`noise_channels`、`syndrome_measurement`、`error_correction`、`logical_tracking` 的关键类。

- `physics/gkp_state.py`
  - 不依赖其他 physics 文件。
  - 被其他模块通过 `LATTICE_CONST` 间接依赖。

- `physics/noise_channels.py`
  - 依赖：`gkp_state.LATTICE_CONST`、`ApproximateGKPState`（当前未实质使用）。

- `physics/syndrome_measurement.py`
  - 依赖：`gkp_state.LATTICE_CONST`。

- `physics/error_correction.py`
  - 依赖：`gkp_state.LATTICE_CONST`。
  - 依赖：`syndrome_measurement.MeasurementConfig`、`RealisticSyndromeMeasurement`。
  - 依赖：`noise_channels.CombinedNoiseModel`（当前属性持有，多轮主路径未调用）。

- `physics/logical_tracking.py`
  - 依赖：`gkp_state.LATTICE_CONST`。
  - 依赖：`error_correction.LinearDecoder`。
  - 依赖：`syndrome_measurement.MeasurementConfig`、`RealisticSyndromeMeasurement`。

外部调用关系（目录外）：
- `benchmark/compare_full_vs_simplified_ler.py`
  - 调用：`logical_tracking.simulate_error_accumulation`。

- `cnn_fpga/benchmark/run_drift_suite.py`
  - 调用：`LinearDecoder`、`LogicalErrorTracker`、`MeasurementConfig`、`RealisticSyndromeMeasurement`。

---

## 3. 文件详解：`physics/__init__.py`

文件职责：
- 作为 `physics` 包统一入口。
- 收敛常用类，避免上层脚本写太长 import 路径。

核心内容：
1. 导出 `ApproximateGKPState`、`GKPStateFactory`。
2. 导出噪声类 `QuantumNoiseChannel`、`PhotonLossChannel`、`ThermalNoiseChannel`。
3. 导出测量类 `SyndromeMeasurement`、`RealisticSyndromeMeasurement`。
4. 导出纠错类 `GKPErrorCorrector`、`LinearDecoder`。
5. 导出统计类 `LogicalErrorTracker`。

`__all__` 的意义：
1. 约束 `from physics import *` 的导出符号。
2. 对上层阅读者形成“主 API 白名单”。

调用关系：
- 本文件不提供运算函数。
- 仅提供符号重导出。

阅读建议：
- 把它当“目录索引页”。
- 如果不确定某个类在哪个文件，先看这里。

---

## 4. 文件详解：`physics/gkp_state.py`

### 4.1 文件定位

角色：
- GKP 态建模入口。
- 定义相空间晶格常量。
- 提供两条状态生成路径：
  1. Strawberry Fields 精确仿真路径。
  2. 解析近似路径（SF 不可用或失败时）。

该文件对全目录最重要的贡献：
- `LATTICE_CONST`。

### 4.2 顶层常量与运行时标志

常量：
- `LATTICE_CONST = sqrt(2*pi)`。

运行时标志：
- `HAS_STRAWBERRYFIELDS`。
- 由 `try import strawberryfields` 决定。

行为影响：
1. SF 可用时：可尝试 `_prepare_state_sf()`。
2. SF 不可用时：强制走 `_compute_wigner_analytical()` 近似。

### 4.3 数据类 `GKPParameters`

字段：
1. `delta`：有限能量参数。
2. `logical_state`：`'0'|'1'|'+'|'-'`。
3. `cutoff`：Fock 截断维度。

用途：
- 作为工厂 `create_from_params` 的输入容器。

调用方：
- `GKPStateFactory.create_from_params`。

### 4.4 类 `ApproximateGKPState` 总览

对象职责：
1. 持有 GKP 态基础参数。
2. 生成 Wigner 分布。
3. 提供简单位移操作接口。
4. 提供平均光子数估计。

关键内部状态：
- `self.delta`
- `self.logical_state`
- `self.cutoff`
- `self.lattice`
- `self.squeezing_db`
- `self._wigner_cache`（当前未真正使用缓存逻辑）
- `self._sf_state`
- `self._use_analytical`

### 4.5 方法详解：`__init__`

输入：
1. `delta`
2. `logical_state`
3. `cutoff`
4. `use_strawberryfields`

主要逻辑：
1. 记录参数。
2. 计算 `squeezing_db = -10*log10(2*delta^2)`。
3. 初始化内部缓存/状态句柄。
4. 根据 `use_strawberryfields` 与 `HAS_STRAWBERRYFIELDS` 决定是否尝试 SF 路径。

输出：
- 初始化完成的状态对象。

副作用：
- 可能调用 `_prepare_state_sf`（依赖外部包）。

### 4.6 方法详解：`_prepare_state_sf`

输入：
- 无显式输入（使用对象参数）。

内部步骤：
1. 创建 `sf.Program(1)`。
2. 根据 `logical_state` 查表决定 `GKP` 操作参数。
3. 对 `+/-` 态附加 `Rgate(pi/2)` 旋转。
4. 用 `sf.Engine("fock")` 执行。
5. 成功则保存 `self._sf_state`，并置 `self._use_analytical=False`。
6. 异常则回退解析模式。

潜在问题点：
1. SF 后端计算较慢，且依赖环境。
2. except 中使用 `print`，不是统一日志系统。

被调用关系：
- 仅在 `__init__` 中调用。

### 4.7 方法详解：`get_wigner`

输入：
1. 网格点数 `q_points/p_points`。
2. 网格范围 `q_range/p_range`。

内部流程：
1. 生成 `q_vec/p_vec`。
2. 若 `self._sf_state` 可用，则调用 SF 的 `wigner`。
3. SF 路径失败则回退解析路径。
4. 返回 `self._compute_wigner_analytical(...)`。

输出：
- 2D `numpy.ndarray`。

依赖调用：
- `_compute_wigner_analytical`。

### 4.8 方法详解：`_compute_wigner_analytical`

输入：
- `q_vec`, `p_vec`。

关键计算：
1. 构建网格 `Q,P`。
2. 根据范围估计求和截断 `n_max`。
3. 双重循环遍历晶格点 `(nq,np_)`。
4. 每个点计算 envelope 与 gaussian 峰。
5. 根据逻辑态奇偶性给定 `sign`。
6. 累加后做幅值归一化。

输出：
- 近似 Wigner 函数。

复杂度特点：
- 与 `n_max^2 * grid_size^2` 相关。
- 在大网格与小 `delta` 场景会偏慢。

### 4.9 方法详解：`apply_displacement`

输入：
- `alpha: complex`。

行为：
1. 创建新 `ApproximateGKPState`（强制不用 SF）。
2. 复制参数。
3. 给新对象附加/累加 `_displacement` 属性。

输出：
- 新状态对象。

注意事项：
1. 该位移是“属性追踪式”，不是对 Wigner 立即重算。
2. 下游若不读取 `_displacement`，该信息不会自动生效。

### 4.10 属性方法详解：`mean_photon_number`

逻辑：
1. 若 SF 状态可用，调用 `self._sf_state.mean_photon`。
2. 否则用近似公式 `1/(2*delta^2)`。

用途：
- 评估状态能量尺度。

### 4.11 类 `GKPStateFactory`

职责：
- 用统一配置快速创建常见逻辑态。

字段：
1. `default_cutoff`
2. `use_sf`（与 `HAS_STRAWBERRYFIELDS` 联合决定）

方法：
1. `create_logical_zero`
2. `create_logical_one`
3. `create_logical_plus`
4. `create_from_params`

公共特性：
- 都返回 `ApproximateGKPState`。

### 4.12 工具函数

函数：`delta_to_squeezing_db`
- 输入：`delta`
- 输出：`dB`

函数：`squeezing_db_to_delta`
- 输入：`dB`
- 输出：`delta`

二者关系：
- 数学互逆（理想浮点误差外）。

### 4.13 与其他文件关系

被依赖符号：
1. `LATTICE_CONST` 被 `syndrome_measurement.py`、`error_correction.py`、`logical_tracking.py`、`noise_channels.py` 使用。
2. `ApproximateGKPState` 在其他文件目前主要用于类型层面的耦合准备，未深度参与在线主流程。

---

## 5. 文件详解：`physics/syndrome_measurement.py`

### 5.1 文件定位

角色：
- 定义从真实位移误差到 syndrome 的观测模型。

分层：
1. `SyndromeMeasurement`：理想测量。
2. `RealisticSyndromeMeasurement`：带噪声测量。
3. `AdaptiveSyndromeMeasurement`：在真实测量基础上做增益自适应。

### 5.2 数据类 `MeasurementConfig`

字段解释：
1. `delta`：有限能量参数，决定测量噪声基线。
2. `measurement_efficiency`：探测效率。
3. `ancilla_error_rate`：辅助模错误概率。
4. `add_shot_noise`：是否叠加 shot noise。

调用方：
1. `RealisticSyndromeMeasurement.__init__`
2. `logical_tracking.simulate_error_accumulation`
3. `cnn_fpga/benchmark/run_drift_suite.py`

### 5.3 类 `SyndromeMeasurement`（理想版）

职责：
- 做“取模回晶胞”的理想 syndrome 测量。

方法：`measure`
- 输入：`displacement=[dq,dp]`
- 公式：`mod(displacement + lattice/2, lattice) - lattice/2`
- 输出：`syndrome=[sq,sp]`

方法：`get_correction`
- 输入：`syndrome`
- 输出：`-syndrome`

应用场景：
- 理论基线。
- 单元测试。

### 5.4 类 `RealisticSyndromeMeasurement` 总览

职责：
1. 建模有限能量导致的测量方差。
2. 建模探测效率不足导致的真空噪声。
3. 可选 shot noise。
4. 可选 ancilla 随机错位事件。

关键字段：
- `self.config`
- `self.lattice`
- `self.sigma_meas`

### 5.5 方法：`__init__`

行为：
1. 若无配置则创建默认 `MeasurementConfig()`。
2. 初始化晶格常量。
3. 立刻调用 `_compute_noise_variance()`。

### 5.6 方法：`_compute_noise_variance`

计算逻辑：
1. `var_squeezing = delta^2`
2. `var_inefficiency = (1-eta)/(2*eta)`（eta>0）
3. `sigma_meas = sqrt(var_squeezing + var_inefficiency)`

设计意义：
- 用一个标量把主要测量噪声合并。

### 5.7 方法：`measure`

输入：
1. `true_displacement`
2. `add_noise`

流程：
1. 先计算理想 syndrome。
2. 若 `add_noise=False`，直接返回理想值。
3. 采样 `measurement_noise ~ N(0, sigma_meas)`。
4. 若配置允许，叠加 `shot_noise ~ N(0,0.1)`。
5. 按概率触发 ancilla 错误，随机在 q 或 p 方向加半晶格偏置。

输出：
- 含噪 syndrome。

边界行为：
- ancilla 错误触发后，返回值可能远离晶胞中心。

### 5.8 方法：`get_correction`

行为：
- 返回 `-gain * syndrome`。

用途：
- 给简单控制策略快速形成 correction。

### 5.9 方法：`get_optimal_gain`

公式：
1. `var_signal = (lattice/2)^2 / 3`
2. `var_noise = sigma_meas^2`
3. `gain = var_signal / (var_signal + var_noise)`

作用：
- Wiener 风格标量增益近似。

### 5.10 方法：`get_measurement_covariance`

行为：
1. 基础方差 `sigma_meas^2`。
2. 若开 shot noise 再加 `0.01`。
3. 返回 `diag([var,var])`。

### 5.11 方法：`update_delta`

作用：
- 在线更新 `delta` 后重算噪声方差。

依赖：
- `_compute_noise_variance`。

### 5.12 子类 `AdaptiveSyndromeMeasurement`

设计：
- 在真实测量基础上再加一个“可更新 gain”。

新增字段：
- `self.gain`。

方法：`measure_and_correct`
1. 调 `self.measure(...)`
2. 调 `self.get_correction(..., self.gain)`
3. 返回 `(syndrome, correction)`

方法：`update_gain`
- 做 `[0.1, 1.5]` 裁剪。

方法：`adapt_to_noise`
1. 更新 `sigma_meas = sqrt(delta^2 + estimated_sigma^2)`。
2. 用 `get_optimal_gain` 重估增益。

### 5.13 函数：`simulate_measurement_statistics`

用途：
- 统计学分析入口。

流程：
1. 创建配置与 `RealisticSyndromeMeasurement`。
2. 采样 `n_samples` 个真实误差。
3. 调 `measurement.measure` 生成 syndromes。
4. 汇总均值、标准差、相关系数、`measurement_sigma`、`optimal_gain`。

返回：
- `dict`。

### 5.14 与其他文件调用关系

被调用方：
1. `error_correction.GKPErrorCorrector`
2. `logical_tracking.simulate_error_accumulation`
3. `cnn_fpga/benchmark/run_drift_suite.py`

输出价值：
- 为解码器提供更接近真实硬件的观测输入。

---

## 6. 文件详解：`physics/noise_channels.py`

### 6.1 文件定位

角色：
- 相空间噪声通道集合。

重点能力：
1. 多噪声顺序叠加。
2. 单噪声通道类封装。
3. 可提取“等效总噪声 sigma”。

### 6.2 数据类 `NoiseParameters`

字段：
1. `gamma`：光子损失强度。
2. `n_bar`：热光子数。
3. `sigma_displacement`：位移噪声标准差。
4. `sigma_phase`：相位噪声标准差。

用途：
- 作为 `QuantumNoiseChannel.apply_all` 的统一参数对象。

### 6.3 类 `QuantumNoiseChannel` 总览

职责：
- 承担“组合噪声施加引擎”。

字段：
1. `cutoff`
2. `use_sf`

当前主路径说明：
- 该类主要通过数值近似（卷积/旋转）直接作用在 Wigner 网格。

### 6.4 方法：`apply_all`

输入：
1. `wigner`
2. `params`
3. `grid_range`

执行顺序：
1. `_apply_photon_loss_wigner`
2. `_apply_thermal_noise_wigner`
3. `_apply_displacement_noise_wigner`
4. 若 `sigma_phase > 0.01` 再 `_apply_phase_noise_wigner`

输出：
- 新 `W`。

顺序意义：
- 先做衰减收缩，再做扩散类噪声，最后可选角向弥散。

### 6.5 方法：`_apply_photon_loss_wigner`

输入：
- `W`, `gamma`, `grid_range`

核心步骤：
1. `eta = exp(-gamma)`。
2. 若 `eta > 0.999` 近似无损直接返回。
3. 通过 `zoom` + `pad` 做收缩近似。
4. 用 `sigma_eta = sqrt((1-eta)/2)` 换算像素尺度。
5. `gaussian_filter` 做扩散。
6. 幅值重标定。

注意：
1. 使用 `scipy.ndimage.zoom` 与 `gaussian_filter`。
2. 是数值近似，不是严格 Kraus 级模拟。

### 6.6 方法：`_apply_thermal_noise_wigner`

逻辑：
1. `n_bar` 太小直接返回。
2. `sigma_thermal = sqrt(n_bar)`。
3. 转像素单位后高斯滤波。

### 6.7 方法：`_apply_displacement_noise_wigner`

逻辑：
1. `sigma_disp` 太小直接返回。
2. 转像素单位。
3. 高斯滤波。

### 6.8 方法：`_apply_phase_noise_wigner`

逻辑：
1. 采样 `n_samples=5` 个相位偏移。
2. 用 `scipy.ndimage.rotate` 旋转。
3. 平均得到角向模糊结果。

特点：
- 是 Monte Carlo 近似。

### 6.9 类 `PhotonLossChannel`

职责：
- 对光子损失提供专门包装。

字段：
1. `gamma`
2. `eta = exp(-gamma)`

方法：`apply_to_wigner`
- 内部复用 `QuantumNoiseChannel._apply_photon_loss_wigner`。

方法：`from_t1_and_time`
- 输入 `T1`, `t`
- 计算 `gamma=t/T1`
- 返回实例。

### 6.10 类 `ThermalNoiseChannel`

职责：
- 热噪声包装。

方法：`apply_to_wigner`
- 复用 `_apply_thermal_noise_wigner`。

方法：`from_temperature`
1. 用物理常数估算 `n_bar`。
2. 返回实例。

### 6.11 类 `DisplacementNoiseChannel`

职责：
- 位移噪声包装。

字段：
1. `sigma_q`
2. `sigma_p`（默认同 `sigma_q`）

方法：`sample_displacement`
- 随机采样 `(dq,dp)`。

方法：`apply_to_wigner`
- 用均值 sigma 近似调用 `_apply_displacement_noise_wigner`。

### 6.12 类 `CombinedNoiseModel`

职责：
1. 持有统一噪声参数。
2. 对外暴露 `apply`。
3. 支持在线参数更新。
4. 支持等效 sigma 估算。

构造逻辑：
1. 生成 `NoiseParameters`。
2. 创建 `QuantumNoiseChannel`。

方法：`apply`
- 调 `self.channel.apply_all(...)`。

方法：`update_params`
- 遍历 kwargs，匹配字段后更新。

方法：`get_effective_sigma`
1. 光子损失折算项 `sqrt(gamma/2)`。
2. 热噪声折算项 `sqrt(n_bar)`。
3. 与位移噪声二范合成。

### 6.13 与其他文件关系

实际耦合现状：
1. `error_correction.QECSimulator` 持有 `CombinedNoiseModel`。
2. 但 `simulate_multiple_rounds` 目前没有调用 `noise_model.apply`。
3. 因此当前主 LER 路径更多是“误差随机注入 + 测量 + 解码 + 追踪”模式。

可改进空间：
- 若要强化物理一致性，可让 `QECSimulator` 真正通过 `noise_model` 生成每轮噪声。

---

## 7. 文件详解：`physics/error_correction.py`

### 7.1 文件定位

角色：
- 纠错流程核心。

提供能力：
1. 线性解码参数容器。
2. 线性解码器。
3. 近似最优参数计算。
4. 单轮纠错执行。
5. 多轮纠错仿真。
6. 漂移条件下周期重标定。

### 7.2 数据类 `DecoderParameters`

字段：
1. `K`：`2x2` 增益矩阵。
2. `b`：`2` 维偏置。

方法：`to_flat`
- 输出 `[K11,K12,K21,K22,b1,b2]`。

方法：`from_flat`
- 从长度 6 向量还原对象。

使用意义：
- 与硬件寄存器写入格式相容。

### 7.3 类 `LinearDecoder`

职责：
- 实现公式 `Δ = K @ s + b`。

字段：
1. `self.K`
2. `self.b`
3. `self.lattice`

方法：`decode`
- 输入 syndrome。
- 输出 correction。

方法：`update`
- 直接更新 `K,b`。

方法：`update_from_flat`
- 从长度 6 向量更新。

方法：`get_params`
- 返回 `DecoderParameters` 拷贝。

方法：`get_flat_params`
- 返回平坦向量。

外部调用方：
1. `GKPErrorCorrector.run_qec_round`
2. `logical_tracking.simulate_error_accumulation`（full_qec 分支）
3. `cnn_fpga/benchmark/run_drift_suite.py`

### 7.4 函数：`compute_optimal_decoder_params`

输入：
1. `sigma`
2. `delta`
3. `theta`
4. `meas_efficiency`

步骤：
1. 计算 `var_signal`（晶胞均匀分布近似）。
2. 计算位移/GKP/测量噪声方差。
3. 求 Wiener 增益 `gain`。
4. 构造旋转矩阵 `R(theta)`。
5. `K = gain * R`。
6. `b = 0`。

输出：
- `DecoderParameters`。

被调用方：
1. `GKPErrorCorrector.__init__`（默认解码器）。
2. `QECSimulator.run_with_drift`（重标定）。

### 7.5 类 `GKPErrorCorrector`

职责：
- 把“测量 + 解码 + 残差 + 成功判定”组织成单轮流程。

构造流程：
1. 设置 `delta/lattice`。
2. 准备测量对象 `RealisticSyndromeMeasurement`。
3. 若无 decoder，按默认 `sigma=0.3` 调 `compute_optimal_decoder_params`。

方法：`run_qec_round`

输入：
1. `error=[eq,ep]`
2. `add_measurement_noise`

执行：
1. `measurement.measure(error)`。
2. `decoder.decode(syndrome)`。
3. `residual = error - correction`。
4. 成功判定：`|residual_q|<lattice/2 且 |residual_p|<lattice/2`。

输出字典字段：
1. `syndrome`
2. `correction`
3. `residual`
4. `success`
5. `error`

方法：`evaluate_performance`

输入：
1. `n_samples`
2. `error_sigma`

执行：
1. 循环采样随机误差。
2. 调 `run_qec_round`。
3. 统计成功率与残差统计量。

输出：
- 包含 `logical_error_rate`、`success_rate`、残差均值/方差。

方法：`update_decoder`
- 代理到 `LinearDecoder.update`。

方法：`update_decoder_from_fno`
- 代理到 `LinearDecoder.update_from_flat`。
- 注：函数名提到 FNO，但当前代码中没有 FNO 实现，属于接口兼容预留。

### 7.6 类 `QECSimulator`

职责：
- 多轮仿真控制器。

字段：
1. `delta`
2. `lattice`
3. `noise_model`
4. `corrector`

构造默认行为：
1. 默认 `noise_model=CombinedNoiseModel(gamma=0.05, n_bar=0.01, sigma_disp=0.1)`。
2. 默认 `corrector=GKPErrorCorrector(delta)`。

方法：`simulate_multiple_rounds`

输入：
1. `n_rounds`
2. `error_sigma`

流程：
1. 初始化 `cumulative_error=0`。
2. 每轮采样 `new_error`。
3. `total_error = cumulative_error + new_error`。
4. 调 `corrector.run_qec_round(total_error)`。
5. 把 `residual` 写回 `cumulative_error`。
6. 统计每轮 success。

输出：
- 总轮数、成功数、LER、每轮结果列表。

关键观察：
- 此方法没有使用 `self.noise_model` 生成噪声，是简化注入路径。

方法：`run_with_drift`

输入：
1. `n_timesteps`
2. `drift_model(t)->(sigma,delta,theta)`
3. `recalibrate_every`

流程：
1. 每个 t 获取漂移参数。
2. 调 `corrector.evaluate_performance(n_samples=1000,error_sigma=sigma_t)`。
3. 周期触发重标定：`compute_optimal_decoder_params` + `update_decoder`。

输出：
- `error_rates` 曲线与均值。

### 7.7 与其他文件关系

上游依赖：
1. `syndrome_measurement` 提供观测模型。
2. `gkp_state` 提供晶格常量。
3. `noise_channels` 提供噪声模型容器。

下游使用：
1. `logical_tracking` 中只直接用到 `LinearDecoder`。
2. `cnn_fpga` benchmark 直接复用 `LinearDecoder`。

---

## 8. 文件详解：`physics/logical_tracking.py`

### 8.1 文件定位

角色：
- 逻辑错误统计与误差累积仿真总入口。

核心价值：
1. 把逐轮残差映射为逻辑 X/Z 错误事件。
2. 提供滑窗与多配置实验统计封装。
3. 提供 `simulate_error_accumulation` 统一接口，支持 full 与 simplified 两套模型。

### 8.2 数据类 `LogicalErrorEvent`

字段：
1. `timestep`
2. `error_type`
3. `accumulated_q`
4. `accumulated_p`

用途：
- 记录具体逻辑错误发生时刻与状态。

### 8.3 类 `LogicalErrorTracker` 总览

内部状态：
1. `accumulated_q`
2. `accumulated_p`
3. `logical_x_errors`
4. `logical_z_errors`
5. `total_rounds`
6. `error_history`

状态机思想：
- 每轮把 residual 累加。
- 若超出 `±lattice/2` 则记一次逻辑错，并包裹回晶胞。

### 8.4 方法：`reset`

效果：
- 清空所有累计量和历史。

调用场景：
1. 初始化。
2. 滑窗或实验切换。

### 8.5 方法：`update`

输入：
1. `error_q,error_p`（校正前真实误差）
2. `correction_q,correction_p`

步骤：
1. `residual = error - correction`。
2. 累加到 `accumulated_q/p`。
3. 检查 q 方向是否越界，越界则记 X 错并 wrap。
4. 检查 p 方向是否越界，越界则记 Z 错并 wrap。

输出：
- `(x_error,z_error)`。

细节：
- q 与 p 检查相互独立，同一轮理论上可同时触发 X/Z。

### 8.6 方法：`update_from_qec_result`

输入：
- `qec_result`（来自 `GKPErrorCorrector.run_qec_round`）。

行为：
- 解包 `error/correction` 后调用 `update`。

### 8.7 统计 getter 方法组

函数列表：
1. `get_total_logical_errors`
2. `get_logical_error_rate`
3. `get_x_error_rate`
4. `get_z_error_rate`
5. `get_statistics`
6. `get_error_times`

共同特征：
- 不改内部状态。
- 输出汇总统计。

### 8.8 类 `WindowedErrorTracker`

用途：
- 追踪最近窗口内 LER，用于漂移恶化检测。

字段：
1. `window_size`
2. `tracker`（`LogicalErrorTracker`）
3. `window_errors`（0/1 列表）

方法：`update`
1. 调 `tracker.update(...)`。
2. 记录本轮是否出错。
3. 保持滑窗长度。
4. 返回 `get_windowed_error_rate()`。

方法：`is_performance_degraded`
- 比较窗口 LER 是否超过阈值。

### 8.9 类 `ExperimentErrorTracker`

用途：
- 跨多个配置记录结果。

字段：
1. `results`
2. `current_tracker`
3. `current_config`（运行时动态设置）

方法：`start_configuration`
- 新建 `LogicalErrorTracker` 并记录配置元信息。

方法：`update`
- 更新当前配置追踪器。

方法：`end_configuration`
- 把当前配置统计写入 `results`。

方法：`get_all_results`
- 返回所有配置结果。

方法：`get_summary`
- 汇总平均/标准差/最佳/最差配置。

### 8.10 函数：`simulate_error_accumulation` 概览

这是本文件最重要入口。

输入参数大类：
1. 轮次与噪声：`n_rounds`、`sigma_error`、`sigma_error_p`。
2. 测量额外噪声：`sigma_measurement`。
3. 解码增益与旋转：`gain`、`theta`。
4. 测量模型：`delta`、`measurement_efficiency`、`ancilla_error_rate`、`add_shot_noise`。
5. 偏置与模式：`error_bias`、`use_full_qec_model`。
6. 结果控制：`return_history`、`seed`。

返回：
- 统计字典。
- 若 `return_history=True` 包含逐轮记录。

### 8.11 输入校验逻辑

校验点：
1. `n_rounds > 0`。
2. `sigma_error` 和 `sigma_measurement` 非负。
3. `sigma_error_p` 非负。
4. full 模型时要求 `measurement_efficiency in (0,1]`。
5. full 模型时要求 `ancilla_error_rate in [0,1]`。
6. `error_bias` 形状必须 `(2,)`。

意义：
- 提前阻断无效参数导致的“静默错误”。

### 8.12 分支 A：`use_full_qec_model=False`（simplified）

流程：
1. 每轮独立采样 `error_q/error_p`。
2. 各轴做 wrap 后叠加高斯测量噪声。
3. correction 采用标量增益 `gain*syndrome`。
4. 调 `tracker.update`。
5. 可选记录 history。

特征：
1. 不显式引入 `MeasurementConfig`。
2. 不使用 `RealisticSyndromeMeasurement`。
3. 不携带“上一轮残差到下一轮总误差”的闭环测量细节。
4. 保留历史兼容性。

返回附加字段：
- `model='simplified'`。

### 8.13 分支 B：`use_full_qec_model=True`（full_qec）

流程：
1. 创建 `MeasurementConfig`。
2. 创建 `RealisticSyndromeMeasurement`。
3. 根据 `theta` 构造旋转矩阵。
4. 创建 `LinearDecoder(K=gain*rotation,b=0)`。
5. 初始化 `cumulative_residual=0`。
6. 每轮采样各向异性噪声并旋转。
7. `new_error = bias + rotation @ base_noise`。
8. `total_error = cumulative_residual + new_error`。
9. syndrome 通过 `measurement.measure(total_error, add_noise=True)`。
10. 可选再叠加 `sigma_measurement` 额外噪声。
11. `correction = decoder.decode(syndrome)`。
12. `residual = total_error - correction`。
13. 调 `tracker.update(total_error, correction)`。
14. 下轮残差使用 tracker 的 wrapped 状态。
15. 可选记录 history。

特征：
1. 体现闭环：跨轮残差持续影响后续。
2. 使用真实测量模型参数。
3. 可模拟偏置、旋转、各向异性与 ancilla 事件。

返回附加字段：
1. `model='full_qec'`。
2. `measurement_sigma_model`。
3. `measurement_sigma_extra`。

### 8.14 为何同时保留两条分支

原因：
1. 兼容历史结果。
2. 给新旧模型提供直接对照。
3. benchmark 可量化“简化误差”。

外部使用示例：
- `benchmark/compare_full_vs_simplified_ler.py` 就是专门对比两分支。

### 8.15 与 `cnn_fpga` 的关系

虽然 `cnn_fpga/benchmark/run_drift_suite.py` 没直接调用本函数，
但它复制了类似的 full/simplified 双分支思想：
1. full 分支：`MeasurementConfig + RealisticSyndromeMeasurement + LinearDecoder + LogicalErrorTracker`。
2. simplified 分支：wrap + scalar gain + tracker。

---

## 9. `physics` 目录内调用关系矩阵（函数级）

说明：
- 左列是调用者。
- 右列是其直接调用（同目录内）。

`ApproximateGKPState.__init__`
- 可能调用：`ApproximateGKPState._prepare_state_sf`

`ApproximateGKPState.get_wigner`
- 调用：`ApproximateGKPState._compute_wigner_analytical`（回退路径）

`ApproximateGKPState.apply_displacement`
- 调用：`ApproximateGKPState(...)` 构造函数

`GKPStateFactory.create_logical_zero`
- 调用：`ApproximateGKPState(...)`

`GKPStateFactory.create_logical_one`
- 调用：`ApproximateGKPState(...)`

`GKPStateFactory.create_logical_plus`
- 调用：`ApproximateGKPState(...)`

`GKPStateFactory.create_from_params`
- 调用：`ApproximateGKPState(...)`

`RealisticSyndromeMeasurement.__init__`
- 调用：`RealisticSyndromeMeasurement._compute_noise_variance`

`RealisticSyndromeMeasurement.update_delta`
- 调用：`RealisticSyndromeMeasurement._compute_noise_variance`

`AdaptiveSyndromeMeasurement.__init__`
- 调用：`RealisticSyndromeMeasurement.__init__`
- 调用：`RealisticSyndromeMeasurement.get_optimal_gain`

`AdaptiveSyndromeMeasurement.measure_and_correct`
- 调用：`RealisticSyndromeMeasurement.measure`
- 调用：`RealisticSyndromeMeasurement.get_correction`

`AdaptiveSyndromeMeasurement.adapt_to_noise`
- 调用：`RealisticSyndromeMeasurement.get_optimal_gain`

`simulate_measurement_statistics`
- 调用：`MeasurementConfig(...)`
- 调用：`RealisticSyndromeMeasurement(...)`
- 调用：`RealisticSyndromeMeasurement.measure`
- 调用：`RealisticSyndromeMeasurement.get_optimal_gain`

`QuantumNoiseChannel.apply_all`
- 调用：`_apply_photon_loss_wigner`
- 调用：`_apply_thermal_noise_wigner`
- 调用：`_apply_displacement_noise_wigner`
- 条件调用：`_apply_phase_noise_wigner`

`PhotonLossChannel.apply_to_wigner`
- 调用：`QuantumNoiseChannel._apply_photon_loss_wigner`

`PhotonLossChannel.from_t1_and_time`
- 调用：`PhotonLossChannel(...)`

`ThermalNoiseChannel.apply_to_wigner`
- 调用：`QuantumNoiseChannel._apply_thermal_noise_wigner`

`ThermalNoiseChannel.from_temperature`
- 调用：`ThermalNoiseChannel(...)`

`DisplacementNoiseChannel.apply_to_wigner`
- 调用：`QuantumNoiseChannel._apply_displacement_noise_wigner`

`CombinedNoiseModel.apply`
- 调用：`QuantumNoiseChannel.apply_all`

`LinearDecoder.update_from_flat`
- 不调用同目录其他函数（内部 reshape）

`LinearDecoder.get_params`
- 调用：`DecoderParameters(...)`

`compute_optimal_decoder_params`
- 调用：`DecoderParameters(...)`

`GKPErrorCorrector.__init__`
- 调用：`MeasurementConfig(...)`（默认）
- 调用：`RealisticSyndromeMeasurement(...)`
- 条件调用：`compute_optimal_decoder_params`
- 条件调用：`LinearDecoder(...)`

`GKPErrorCorrector.run_qec_round`
- 调用：`RealisticSyndromeMeasurement.measure`
- 调用：`LinearDecoder.decode`

`GKPErrorCorrector.evaluate_performance`
- 调用：`GKPErrorCorrector.run_qec_round`

`GKPErrorCorrector.update_decoder`
- 调用：`LinearDecoder.update`

`GKPErrorCorrector.update_decoder_from_fno`
- 调用：`LinearDecoder.update_from_flat`

`QECSimulator.__init__`
- 条件调用：`CombinedNoiseModel(...)`
- 条件调用：`GKPErrorCorrector(...)`

`QECSimulator.simulate_multiple_rounds`
- 调用：`GKPErrorCorrector.run_qec_round`

`QECSimulator.run_with_drift`
- 调用：`GKPErrorCorrector.evaluate_performance`
- 条件调用：`compute_optimal_decoder_params`
- 条件调用：`GKPErrorCorrector.update_decoder`

`LogicalErrorTracker.__init__`
- 调用：`LogicalErrorTracker.reset`

`LogicalErrorTracker.update_from_qec_result`
- 调用：`LogicalErrorTracker.update`

`LogicalErrorTracker.get_total_logical_errors`
- 无内部调用

`LogicalErrorTracker.get_logical_error_rate`
- 调用：`LogicalErrorTracker.get_total_logical_errors`

`LogicalErrorTracker.get_x_error_rate`
- 无内部调用

`LogicalErrorTracker.get_z_error_rate`
- 无内部调用

`LogicalErrorTracker.get_statistics`
- 调用：`get_total_logical_errors`
- 调用：`get_x_error_rate`
- 调用：`get_z_error_rate`
- 调用：`get_logical_error_rate`

`LogicalErrorTracker.get_error_times`
- 无内部调用

`WindowedErrorTracker.__init__`
- 调用：`LogicalErrorTracker(...)`

`WindowedErrorTracker.reset`
- 调用：`LogicalErrorTracker.reset`

`WindowedErrorTracker.update`
- 调用：`LogicalErrorTracker.update`
- 调用：`WindowedErrorTracker.get_windowed_error_rate`

`WindowedErrorTracker.get_windowed_error_rate`
- 无内部调用

`WindowedErrorTracker.is_performance_degraded`
- 调用：`WindowedErrorTracker.get_windowed_error_rate`

`ExperimentErrorTracker.start_configuration`
- 调用：`LogicalErrorTracker(...)`

`ExperimentErrorTracker.update`
- 调用：`LogicalErrorTracker.update`

`ExperimentErrorTracker.end_configuration`
- 调用：`LogicalErrorTracker.get_statistics`

`ExperimentErrorTracker.get_summary`
- 无同目录函数调用（内部 numpy 汇总）

`simulate_error_accumulation`
- 调用：`LogicalErrorTracker(...)`
- full 分支调用：`MeasurementConfig(...)`
- full 分支调用：`RealisticSyndromeMeasurement(...)`
- full 分支调用：`LinearDecoder(...)`
- full/simplified 均调用：`LogicalErrorTracker.update`
- 结束调用：`LogicalErrorTracker.get_statistics`

---

## 10. 目录外对 `physics` 的调用关系

### 10.1 `benchmark/compare_full_vs_simplified_ler.py`

入口函数：`main`

调用链：
1. `main` -> `_run_mode(...,True)`。
2. `_run_mode` -> `simulate_error_accumulation(use_full_qec_model=True)`。
3. `main` -> `_run_mode(...,False)`。
4. `_run_mode` -> `simulate_error_accumulation(use_full_qec_model=False)`。
5. 再把两条曲线写 CSV/JSON/可选 PNG。

意义：
- 是检验 full vs simplified 差异的最直接脚本。

### 10.2 `cnn_fpga/benchmark/run_drift_suite.py`

调用 `physics` 组件：
1. `LinearDecoder`
2. `LogicalErrorTracker`
3. `MeasurementConfig`
4. `RealisticSyndromeMeasurement`

其 `_simulate_trace` 与 `simulate_error_accumulation` 的关系：
- 思路一致。
- 但实现独立，便于和 `cnn_fpga` 配置系统直接耦合。

---

## 11. 数据流与状态流（按一轮 QEC）

标准 full 模型一轮：
1. 输入：上一轮 wrapped residual `r_prev`。
2. 采样：本轮噪声 `n_t`。
3. 合成：`total_error = r_prev + n_t (+ bias/rotation)`。
4. 测量：`syndrome = measurement.measure(total_error)`。
5. 解码：`correction = decoder.decode(syndrome)`。
6. 残差：`residual = total_error - correction`。
7. 跟踪：`tracker.update(total_error, correction)`。
8. wrap：内部把超晶胞部分折返，形成下一轮 `r_next`。

输出：
- 每轮错误事件。
- 累积 LER。

---

## 12. 关键参数如何影响结果

`delta`：
- 进入 `MeasurementConfig`，影响 `sigma_meas`。
- `delta` 越大，测量噪声基线通常更高。

`measurement_efficiency`：
- 直接影响 `var_inefficiency`。
- 效率越低，观测噪声越大。

`ancilla_error_rate`：
- 控制罕见的大偏移事件概率。

`sigma_error/sigma_error_p`：
- 控制每轮注入误差规模。
- 各向异性时两个轴可不同。

`theta`：
- 同时影响噪声旋转与解码矩阵旋转。

`gain`：
- 决定 correction 幅度。
- 过大可能过补偿，过小可能欠补偿。

`sigma_measurement`：
- 在 full 分支中作为“额外读出噪声”叠加于 syndrome。

`error_bias`：
- 注入系统偏置漂移。

---

## 13. 你在阅读时最容易混淆的点

混淆点 1：`logical_tracking.update` 参数名中的 `error_q/error_p`。
- 这里应理解为“校正前真实误差”，不一定是原始新噪声。

混淆点 2：`simulate_error_accumulation` full 分支里的 `residual` 与 `cumulative_residual`。
- 每轮先算未 wrap 的 `residual`。
- 但下轮使用的是 tracker 内部 wrap 后状态。

混淆点 3：`QECSimulator.noise_model` 的存在感。
- 对象层面已接入。
- 当前 `simulate_multiple_rounds` 未真正用它施加噪声。

混淆点 4：`SyndromeMeasurement` 与 `RealisticSyndromeMeasurement`。
- 前者理想取模。
- 后者是工程主线模型。

混淆点 5：`sigma_measurement` 与 `measurement.sigma_meas`。
- 前者是调用层额外加噪。
- 后者是测量模型内部噪声。

---

## 14. 工程视角下的可改进点（阅读辅助）

1. 随机数体系统一：
- 目前有 `np.random.seed` + `np.random.default_rng` 混用。
- 可以统一传入 `Generator`，提升可复现性可控度。

2. 日志体系统一：
- `gkp_state._prepare_state_sf` 里直接 `print`。
- 可改为 logging。

3. 噪声模型接入一致性：
- `QECSimulator` 持有 `noise_model` 但未在主多轮路径使用。

4. 类型注解完善：
- 多个函数返回 `Dict` 可细化为 TypedDict/dataclass。

5. 统计对象可序列化：
- `error_history` 可补充导出接口，便于统一写盘。

---

## 15. 实操导航：按问题反查代码

问题：想看晶格常量定义在哪里。
- 看：`gkp_state.py` 顶部 `LATTICE_CONST`。

问题：想看真实测量噪声怎么算。
- 看：`RealisticSyndromeMeasurement._compute_noise_variance`。

问题：想看解码核心公式。
- 看：`LinearDecoder.decode`。

问题：想看逻辑错误如何判定。
- 看：`LogicalErrorTracker.update`。

问题：想看 full 与 simplified 分歧点。
- 看：`simulate_error_accumulation` 两个分支。

问题：想看跨轮残差是否保留。
- 看：full 分支 `cumulative_residual` 更新。

问题：想看 benchmark 是怎么触发比较的。
- 看：`benchmark/compare_full_vs_simplified_ler.py` 的 `_run_mode`。

问题：想看 `cnn_fpga` 怎么复用 physics。
- 看：`cnn_fpga/benchmark/run_drift_suite.py` 的 `_simulate_trace`。

---

## 16. 总结（作为你的记忆锚点）

一句话总结各文件：
1. `gkp_state.py`：定义状态与晶格。
2. `noise_channels.py`：定义噪声怎么改 Wigner。
3. `syndrome_measurement.py`：定义怎么“看见”误差。
4. `error_correction.py`：定义怎么“改正”误差。
5. `logical_tracking.py`：定义怎么“统计是否越界成逻辑错”。

一句话总结 full/simplified：
- simplified：快、旧、对照用。
- full_qec：带测量模型和跨轮闭环，更接近真实实验逻辑。

一句话总结你后续阅读主线：
- 先读 `simulate_error_accumulation`，再反向跳到 `measurement`、`decoder`、`tracker` 三块。

---

## 17. 附录 A：符号索引（按字母）

`AdaptiveSyndromeMeasurement`
- 文件：`syndrome_measurement.py`
- 作用：自适应增益测量器。

`ApproximateGKPState`
- 文件：`gkp_state.py`
- 作用：近似 GKP 态对象。

`CombinedNoiseModel`
- 文件：`noise_channels.py`
- 作用：组合噪声模型容器。

`compute_optimal_decoder_params`
- 文件：`error_correction.py`
- 作用：计算线性解码最优近似参数。

`DecoderParameters`
- 文件：`error_correction.py`
- 作用：解码参数数据类。

`DisplacementNoiseChannel`
- 文件：`noise_channels.py`
- 作用：位移噪声封装。

`ExperimentErrorTracker`
- 文件：`logical_tracking.py`
- 作用：多配置实验统计。

`GKPErrorCorrector`
- 文件：`error_correction.py`
- 作用：单轮纠错执行器。

`GKPParameters`
- 文件：`gkp_state.py`
- 作用：GKP 构造参数数据类。

`GKPStateFactory`
- 文件：`gkp_state.py`
- 作用：快捷构造 GKP 态。

`LATTICE_CONST`
- 文件：`gkp_state.py`
- 作用：晶格常量。

`LinearDecoder`
- 文件：`error_correction.py`
- 作用：线性解码器。

`LogicalErrorEvent`
- 文件：`logical_tracking.py`
- 作用：单次逻辑错事件记录。

`LogicalErrorTracker`
- 文件：`logical_tracking.py`
- 作用：逻辑错计数与状态跟踪。

`MeasurementConfig`
- 文件：`syndrome_measurement.py`
- 作用：测量参数数据类。

`NoiseParameters`
- 文件：`noise_channels.py`
- 作用：噪声参数数据类。

`PhotonLossChannel`
- 文件：`noise_channels.py`
- 作用：光子损失封装。

`QECSimulator`
- 文件：`error_correction.py`
- 作用：多轮仿真器。

`QuantumNoiseChannel`
- 文件：`noise_channels.py`
- 作用：组合噪声计算引擎。

`RealisticSyndromeMeasurement`
- 文件：`syndrome_measurement.py`
- 作用：真实测量模型。

`simulate_error_accumulation`
- 文件：`logical_tracking.py`
- 作用：逻辑错误累积仿真统一入口。

`simulate_measurement_statistics`
- 文件：`syndrome_measurement.py`
- 作用：测量统计分析函数。

`SyndromeMeasurement`
- 文件：`syndrome_measurement.py`
- 作用：理想测量模型。

`ThermalNoiseChannel`
- 文件：`noise_channels.py`
- 作用：热噪声封装。

`WindowedErrorTracker`
- 文件：`logical_tracking.py`
- 作用：滑窗 LER 跟踪。

---

## 18. 附录 B：参数-函数映射速查

`delta`
- 使用位置：
  - `MeasurementConfig`
  - `RealisticSyndromeMeasurement._compute_noise_variance`
  - `compute_optimal_decoder_params`
  - `simulate_error_accumulation`

`measurement_efficiency`
- 使用位置：
  - `MeasurementConfig`
  - `RealisticSyndromeMeasurement._compute_noise_variance`
  - `simulate_error_accumulation` 参数校验

`ancilla_error_rate`
- 使用位置：
  - `MeasurementConfig`
  - `RealisticSyndromeMeasurement.measure`
  - `simulate_error_accumulation` 参数校验

`gain`
- 使用位置：
  - `RealisticSyndromeMeasurement.get_correction`
  - `LinearDecoder(K=gain*rotation)`
  - simplified 分支 `correction = gain * syndrome`

`theta`
- 使用位置：
  - `compute_optimal_decoder_params`（旋转矩阵）
  - `simulate_error_accumulation`（噪声旋转 + 解码旋转）

`sigma_error`
- 使用位置：
  - `simulate_error_accumulation`
  - `GKPErrorCorrector.evaluate_performance`
  - `QECSimulator.simulate_multiple_rounds`

`sigma_measurement`
- 使用位置：
  - `simulate_error_accumulation` full/simplified 两分支都可能叠加

`error_bias`
- 使用位置：
  - `simulate_error_accumulation` 中 `new_error` 构造

`use_full_qec_model`
- 使用位置：
  - `simulate_error_accumulation` 分支开关

---

## 19. 附录 C：最小阅读路径（30 分钟版）

第 1 步（5 分钟）：
- 读 `logical_tracking.py` 的 `simulate_error_accumulation` 参数与两分支。

第 2 步（8 分钟）：
- 跳到 `syndrome_measurement.py` 看 `RealisticSyndromeMeasurement.measure`。

第 3 步（5 分钟）：
- 跳到 `error_correction.py` 看 `LinearDecoder.decode`。

第 4 步（6 分钟）：
- 回到 `LogicalErrorTracker.update` 看越界判定与 wrap。

第 5 步（6 分钟）：
- 看 `benchmark/compare_full_vs_simplified_ler.py`，理解外部如何使用入口函数做对照。

---

## 20. 文档维护建议

1. 每次新增 `physics` 函数时，先更新第 9 章调用矩阵。
2. 每次改动 full/simplified 逻辑时，先更新第 8.12/8.13 节。
3. 每次新增跨目录调用时，更新第 10 章。
4. 若 `QECSimulator` 开始真实调用 `noise_model`，更新第 6.13 与第 7.6 的说明。

