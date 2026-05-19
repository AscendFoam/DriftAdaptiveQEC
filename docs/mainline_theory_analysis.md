# 本项目主线理论分析：从近似 GKP 码到双回路纠错闭环

## 1. 范围与非结论

这份文档是给人看的主线理论说明书，不是论文结果节，也不是新的实验报告。

它的目标是做三件事：

1. 从公式层面解释本项目当前主线为什么成立。
2. 把理论对象、代码实现、运行时 I/O 契约对应起来。
3. 用项目里已经存在的结果数字解释“我们观察到了什么现象”，但不把这些数字升级成新的事实等级。

这份文档不会做以下事情：

1. 不把 `mock-backed software HIL` 写成真板完成。
2. 不把历史 `.tflite` 文档写成当前机器已恢复真 `.tflite` runtime。
3. 不把历史长文档中的较强表述自动升级成当前 recovery 后的主线事实。
4. 不把 `T44` 之后仍然 blocked 的论文级缺口写成已经闭合。

因此，阅读时请始终区分四层：

1. 理论上应该是什么。
2. 仓库当前主线代码实现了什么。
3. 仓库当前已有证据支持到什么程度。
4. 哪些只是未来可扩展方向。

---

## 2. 近似 GKP 码的定义与相空间图像

### 2.1 理想 GKP 码字

GKP 码把一个逻辑量子比特编码到单个谐振子的连续变量相空间里。理想情况下，逻辑码字可看成在位置或动量方向上的无限梳状结构。仓库中的基础晶格常数固定为

\[
\lambda = \sqrt{2\pi}.
\]

代码里这一常数对应 `physics.constants.LATTICE_CONST`，在 [gkp_state.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/physics/gkp_state.py) 和 [physics/README.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/physics/README.md) 中都被作为后续测量、解码、逻辑错误判定的统一尺度。

理想化地说，逻辑零态可以写成位置表象中的梳状叠加

\[
|\bar 0\rangle \propto \sum_{n\in\mathbb Z} |n\lambda\rangle_q.
\]

这不是一个物理可归一化的有限能量态，因此项目主线从一开始就不是“理想 GKP”，而是“近似 GKP”。

### 2.2 近似 GKP 态

仓库当前采用的是有限能量近似。`ApproximateGKPState` 的说明中给出

\[
|{\rm GKP}_\Delta\rangle \propto \sum_{n\in\mathbb Z} e^{-\Delta^2 n^2}\,|n\lambda\rangle_q,
\]

其中 \(\Delta\) 控制包络宽度。它越小，越接近理想 GKP，但平均能量越高。代码实现见 [gkp_state.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/physics/gkp_state.py)。

这个定义的物理含义很重要：项目里很多后续“噪声方差”“测量方差”“最优增益”公式，其实都默认了一个有限能量、有限测量效率、有限窗口统计的近似世界，而不是理想无噪声 stabilizer 世界。

### 2.3 为什么相空间图像有用

如果从 Wigner 函数或 syndrome 分布看，近似 GKP 态在相空间中会呈现周期性峰结构；加入噪声后，这些峰会展宽、偏移、倾斜、旋转。项目的慢回路输入并不直接使用完整量子态，而是使用快回路累积出的 syndrome 直方图窗口。这个设计的核心理由是：

1. 快回路天然持续产生 syndrome。
2. syndrome 的窗口统计正好压缩了当前噪声场的有效信息。
3. 32×32 的二维直方图既保留几何结构，又足够轻量，可以喂给小 CNN。

这也是本项目与“直接让神经网络替代整个 decoder”的一个根本区别：它学习的是窗口统计下的有效控制修正，而不是从头求解完整量子后验。

---

## 3. 编码信息、综合征与模晶格测量

### 3.1 模晶格综合征

设一次快回路之前的数据模态累积位移误差为

\[
e_t = \begin{bmatrix} e_{q,t} \\ e_{p,t}\end{bmatrix}.
\]

GKP 综合征测量本质上测的是误差对晶格常数取模后的结果。理想综合征可写成

\[
s_t = e_t \bmod \lambda,
\]

并映射到对称基本区间

\[
s_{q,t}, s_{p,t} \in \left[-\frac{\lambda}{2}, \frac{\lambda}{2}\right).
\]

在代码里，这个映射就是 [syndrome_measurement.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/physics/syndrome_measurement.py) 中

\[
{\rm syndrome} = \operatorname{mod}(e+\lambda/2,\lambda)-\lambda/2.
\]

因此，快回路每一拍真正看到的不是绝对误差，而是“落在基本晶胞中的剩余位移”。

### 3.2 真实测量不是理想取模

项目默认主线不是 `SyndromeMeasurement`，而是 `RealisticSyndromeMeasurement`。这意味着测量值实际上更接近

\[
\tilde s_t = s_t + n_{{\rm squeeze},t} + n_{{\rm ineff},t} + n_{{\rm shot},t} + n_{{\rm ancilla},t}.
\]

其中有限能量与探测效率损失被折算成等效测量噪声标准差

\[
\sigma_{\rm meas}=\sqrt{\Delta^2+\frac{1-\eta}{2\eta}},
\]

这正是 [syndrome_measurement.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/physics/syndrome_measurement.py) 和 [physics/README.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/physics/README.md) 里写出的公式。

这里的理论含义是：快回路解码面对的是“模晶格后的真实误差 + 测量噪声”，所以最优校正不应该永远是简单的 \(-s_t\)，而是与当前噪声水平相关的线性估计。

### 3.3 逻辑错误何时发生

项目里逻辑错误不是看单次校正是否偏了一点，而是看多轮校正后残差是否越过逻辑边界。`LogicalErrorTracker` 采用的规则是：

\[
|r_{q,t}| > \frac{\lambda}{2}\Rightarrow X_L\ {\rm error},
\qquad
|r_{p,t}| > \frac{\lambda}{2}\Rightarrow Z_L\ {\rm error}.
\]

也就是说，真正关心的是“未被快回路完全消掉的残差如何跨轮积累”。这也是为什么项目主线最终评价指标不是单轮参数拟合 MSE，而是 HIL 闭环中的 `LER`。

---

## 4. 误差模型：为什么会需要自适应

### 4.1 当前主线处理的是有效噪声参数，而不是完整底层通道

虽然 `physics/noise_channels.py` 中存在更底层的物理通道抽象，但当前 `P2/P3/P4` 主线并不直接在完整光子损耗或热噪声通道上做端到端硬件评价。更贴近当前主线的对象是一个窗口级、运行时级的有效噪声参数：

\[
\theta_t^{\rm noise} = (\sigma_t,\mu_{q,t},\mu_{p,t},\theta_t).
\]

它们分别对应：

1. 总体误差尺度 \(\sigma_t\)。
2. q/p 方向均值偏置 \(\mu_{q,t}, \mu_{p,t}\)。
3. 协方差主轴方向 \(\theta_t\)。

在快回路里，这些量不直接出现；它们先通过慢回路估计，再被映射成真正执行的运行时参数 \((K_t,b_t)\)。

### 4.2 当前主线中的漂移来源

在运行时一侧，噪声可由 [noise_bridge.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/runtime/noise_bridge.py) 桥接为有效量。桥接层把物理量如 \(\gamma, n_{\bar{}}, \sigma_{\rm displacement}, \sigma_{\rm phase}\) 压缩到

\[
\sigma_t,\mu_{q,t},\mu_{p,t},\theta_t.
\]

因此，主线闭环里真正“慢慢漂”的不是某个抽象神经网络 hidden state，而是这个低维有效噪声描述。

### 4.3 为什么固定参数会失配

如果快回路始终用固定参数 \((K_0,b_0)\)，那么它隐含假设

\[
\theta_t^{\rm noise}\approx \theta_0^{\rm noise}.
\]

一旦窗口统计明显变化，这个假设就失效：原来合适的增益矩阵和偏置不再合适，残差积累就会增加，最终 `LER` 上升。

这也是项目从“静态参数”逐步走向“窗口方差”“EKF”“UKF”“teacher-guided residual-b”的原因。它们本质上都在回答同一个问题：怎样根据窗口统计重新估计当前的有效噪声状态，并把它转成更合适的 \((K,b)\)。

---

## 5. 为什么主线快回路是线性解码 `Δ = K s + b`

### 5.1 线性解码的角色

本项目快回路的核心公式是

\[
\Delta_t = K_t s_t + b_t.
\]

这里：

1. \(s_t\) 是当前周期的 2 维 syndrome。
2. \(K_t\in\mathbb R^{2\times 2}\) 是线性增益矩阵。
3. \(b_t\in\mathbb R^2\) 是偏置项。
4. \(\Delta_t\) 是要施加的校正位移。

这一公式同时出现在 [physics/error_correction.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/physics/error_correction.py)、[decoder/linear_runtime.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/decoder/linear_runtime.py)、[runtime/README.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/runtime/README.md) 中，是整个工程栈共同遵守的运行时 contract。

### 5.2 线性近似为什么合理

如果把 syndrome 和真实误差在局部高斯近似下看作联合随机变量，那么最优线性估计器就是一个仿射映射。项目早期 `physics/error_correction.py` 中的 `compute_optimal_decoder_params()` 用的是一个 Wiener 风格近似：测量噪声越大，增益越该小；如果有相空间旋转，增益方向也该随之转动。

更重要的是，当前主线在 [param_mapper.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/decoder/param_mapper.py) 中已经从“简单 gain × rotation”更新为更合理的协方差形式：

\[
C = R(\theta)\,
\begin{bmatrix}
\sigma_q^2 & 0\\
0 & \sigma_p^2
\end{bmatrix}
R(\theta)^\top,
\]

\[
R_{\rm meas} = (\sigma_{\rm meas}^2+\Delta_{\rm eff}^2)I,
\]

\[
K_{\rm raw}=C(C+R_{\rm meas})^{-1}.
\]

然后再对特征值做裁剪和整体缩放，得到工程上可用的 \(K_t\)。

### 5.3 为什么要有偏置项 \(b\)

如果噪声均值不为零，仅靠线性增益不足以表达最优校正。当前主线采用的是

\[
b_{\rm target}=\alpha (I-K_{\rm target})\mu,
\qquad
\mu = \begin{bmatrix}\mu_q\\ \mu_p\end{bmatrix}.
\]

这在 [param_mapper.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/decoder/param_mapper.py) 中有明确注释：旧的简单负号方向是错的，当前实现把偏置写成 \((I-K)\mu\) 的形式，更符合后验线性估计的语义。

这一点对理解 `residual-b` 尤其关键：主线最终不是让 CNN 直接控制整个 \(K\)，而是先保留 teacher 给出的 \(K\) 与基础 \(b\)，然后只对 \(b\) 做轻量残差修正。

---

## 6. 从噪声参数到运行时参数：`ParamMapper` 的公式语义

### 6.1 输入

`ParamMapper` 的输入是

\[
\hat\theta_t^{\rm noise}=(\hat\sigma_t,\hat\mu_{q,t},\hat\mu_{p,t},\hat\theta_t),
\]

对应代码里的 `NoisePrediction`。

### 6.2 协方差构造

当前主线先构造主轴系协方差

\[
C_{\rm principal}=
\begin{bmatrix}
\sigma_q^2 & 0\\
0 & \sigma_p^2
\end{bmatrix},
\qquad
\sigma_p = \sigma_q\cdot {\rm sigma\_ratio\_p},
\]

再旋转回实验室坐标：

\[
C = R(\theta)C_{\rm principal}R(\theta)^\top.
\]

这意味着 \(\theta\) 不是一个“附加标签”，而是真的改变了 syndrome 到 correction 的最佳线性方向。

### 6.3 增益裁剪与工程保守性

理论上的 \(K_{\rm raw}\) 还会经过两层工程约束：

1. 特征值裁剪 `gain_clip`，防止增益过激。
2. `gain_scale` 统一缩放，保留方向结构但整体更保守。

因此主线不是“纯理论最优”，而是“受硬件和稳定性约束的受限最优”。这点很重要，因为快回路要在固定点、裁剪、原子切换和延迟限制下工作。

### 6.4 指数平滑

若上一轮参数为 \((K_{t-1},b_{t-1})\)，当前目标参数为 \((K_{\rm target},b_{\rm target})\)，当前主线默认还会做平滑：

\[
K_t = (1-\beta)K_{t-1}+\beta K_{\rm target},
\]

\[
b_t = (1-\beta)b_{t-1}+\beta b_{\rm target}.
\]

这层平滑在 `ParamMapper` 内部完成，而 `hybrid_residual_b` 路线对最终 \(b\) 还可能再加一层针对 \(b\) 的 EMA。理论上，它们都在处理同一个问题：慢回路每个窗口都在更新，但快回路不能因为一次估计抖动就大幅跳参。

---

## 7. teacher：为什么主线要先有一个经典锚点

### 7.1 teacher 不是论文修辞，而是运行时对象

在当前主线里，teacher 不是抽象概念，而是 `SlowLoopRuntime` 里真实可调用的一组基线模式：

1. `window_variance`
2. `ekf`
3. `ukf`
4. `particle_filter`

具体见 [slow_loop_runtime.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/runtime/slow_loop_runtime.py) 的 `_predict_teacher()`。

### 7.2 最朴素的 teacher：窗口方差法

`WindowVarianceBaseline` 做的事情其实很清楚：从 32×32 直方图重建一阶、二阶矩，然后给出

\[
(\hat\sigma,\hat\mu_q,\hat\mu_p,\hat\theta).
\]

其核心步骤是：

1. 对直方图做归一化。
2. 估计均值 \((\hat\mu_q,\hat\mu_p)\)。
3. 估计协方差矩阵。
4. 用主轴方向估计 \(\hat\theta\)。
5. 用去除测量底噪后的迹估计 \(\hat\sigma\)。

这对应 [window_baseline.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/decoder/window_baseline.py)。

### 7.3 更强 teacher：EKF/UKF/Particle Filter

这些方法的理论角色是“在单窗口估计之外再加时序状态模型”。它们不是直接输出 correction，而是输出当前有效噪声参数，再交给 `ParamMapper` 变成快回路参数。

就主线叙事而言，可以把它们理解为

\[
\hat\theta_t^{\rm teacher} = {\rm Teacher}(\mathcal H_{1:t}),
\]

其中 \(\mathcal H_{1:t}\) 表示截至当前窗口的统计历史。

### 7.4 为什么最终主线不是 CNN-only

项目中一个非常关键的稳定结论是：离线训练指标更好，不等于 formal HIL 更好。文档里已经多次记录 `No TeacherParams` 或某些 gated 版本在离线 \(R^2\) 上更好，但在正式闭环 benchmark 中并不稳定。

因此，当前主线的理论判断不是“teacher 是累赘”，而是：

1. teacher 提供一个稳定的、物理上可解释的底座。
2. learned module 最好只修正 teacher 的系统性残差。
3. 这样更符合真正的部署语义，因为快回路执行的是 \((K,b)\)，不是离线标签本身。

---

## 8. CNN：输入是什么，输出是什么，为什么是 runtime-consistent

### 8.1 从静态绝对参数回归到 runtime-consistent 学习

项目里存在两种重要数据构造路径：

1. `dataset_builder.py`：静态参数采样，标签是 \((\sigma,\mu_q,\mu_p,\theta)\)。
2. `runtime_dataset_builder.py`：真实快慢回路仿真采窗，标签是 `residual_mu` 或 `residual_b`。

当前主线属于后者，即“先跑真实闭环，再从窗口中提取训练样本”。这就是 runtime-consistent 的含义：训练样本和运行时看到的是同一种窗口对象，而不是一个脱离调度和 commit 语义的离线代理任务。

### 8.2 CNN 的空间输入

对主线 `residual-b` 数据集，空间主干输入来自窗口直方图历史：

\[
X_t^{\rm hist} = [H_{t-c+1},\ldots,H_t,\Delta H_{t-c+2},\ldots,\Delta H_t].
\]

其中 \(H_t\in\mathbb R^{32\times 32}\)，当前默认主线配置 [experiment_runtime_b_residual.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/experiment_runtime_b_residual.yaml) 使用：

1. `context_windows = 5`
2. `include_histogram_deltas = true`

这意味着模型不仅看到“当前窗口长什么样”，还看到“窗口统计是怎样变化过来的”。

### 8.3 teacher 条件输入

`feature_builder.py` 支持三类 teacher 侧输入：

1. `teacher_prediction`
2. `teacher_params`
3. `teacher_deltas`

它们可以用两种方式注入：

1. `broadcast`：把标量铺成整张常数平面，拼到卷积通道里。
2. `scalar_branch`：保留为低维向量，走单独标量支路。

这在理论上意味着模型输入不是单纯的图像，而是

\[
X_t = \big(X_t^{\rm hist}, z_t^{\rm teacher}\big).
\]

### 8.4 主线 residual-b 的输出

当前主线 `hybrid_residual_b` 的 artifact 输出不是完整 \((\sigma,\mu_q,\mu_p,\theta)\)，而是一个 2 维残差：

\[
\widehat{\delta b}_t =
\begin{bmatrix}
\widehat{\delta b}_{q,t}\\
\widehat{\delta b}_{p,t}
\end{bmatrix}.
\]

在 [slow_loop_runtime.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/runtime/slow_loop_runtime.py) 中，它从 artifact 的 `raw_prediction` 取出 `b_q, b_p`，然后做

\[
\delta b_t = {\rm clip}\!\left(s_b \cdot \widehat{\delta b}_t,\,-b_{\max},\,b_{\max}\right).
\]

当前主线模板 [experiment_runtime_b_residual.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/experiment_runtime_b_residual.yaml) 中，正式主线使用：

1. `residual_scale_b = 1.0`
2. `residual_clip_b = 0.12`

这个 `0.12` 后面会在 `Gated v5` 诊断里再次出现。

### 8.5 最终组合公式

teacher 给出

\[
(K_t^{\rm teacher}, b_t^{\rm teacher}),
\]

CNN 给出

\[
\delta b_t,
\]

然后主线组合成

\[
K_t = K_t^{\rm teacher},
\qquad
b_t = b_t^{\rm teacher} + \delta b_t,
\]

并可选再做一层 \(b\) 上的 EMA 平滑。

这就是当前主线 `Hybrid Residual-B` 的真正数学语义。它不是“teacher 和 CNN 各做一半解码”，而是“teacher 负责稳定的主估计，CNN 只纠偏置残差”。

---

## 9. 为什么主线最终落在 `residual-b`

### 9.1 因为快回路真正执行的是 `(K, b)`

很多离线回归任务默认把 \((\sigma,\mu_q,\mu_p,\theta)\) 当目标，但快回路并不直接执行它们。快回路读的是当前 active bank 中的 \(K\) 和 \(b\)。因此，真正对控制效果负责的量不是标签本身，而是标签经过 `ParamMapper` 之后的运行时参数。

### 9.2 `residual-b` 比绝对回归更贴近控制语义

从控制角度看，teacher 已经提供了一个一阶可用估计。如果 teacher 的主要误差体现在偏置项，那么让 CNN 只学习

\[
\delta b_t = b_t^\star - b_t^{\rm teacher}
\]

往往比让 CNN 从头输出整套 \((\sigma,\mu_q,\mu_p,\theta)\) 更直接，也更稳定。

### 9.3 项目里已有的现象支持这一点

当前文档里已经冻结的几个观察点与这个解释一致：

1. `P1` 静态参数回归上，float 模型全局 \(R^2 \approx 0.994352\)，各标签 `R²` 都很高，说明“看静态图像预测参数”本身并不难。
2. 但这并没有自动变成最强在线控制主线，说明离线 \(R^2\) 与闭环 `LER` 并不等价。
3. `runtime_b_residual` 早期离线 `b` 标签的 `R²` 并不高，文档中甚至明确写过 `b_q`、`b_p` 的离线 `R²` 很低，但闭环主线依然有效。

这正好说明：主线有效性的核心不在于“把一个标签拟合得极准”，而在于“在 teacher 已有的基础上，给快回路一个更合适的运行时修正”。

---

## 10. `Gated v5`：从广播 teacher 到低维标量支路

### 10.1 当前代码里真正实现了什么

在 [tiny_cnn.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/model/tiny_cnn.py) 中，标量特征融合支持两种模式：

1. `concat`
2. `gated`

若使用 `gated`，网络会生成一个 gate 和一个 shift：

\[
g_t = \sigma(z_t W_g + b_g),
\qquad
u_t = z_t W_s + b_s,
\]

然后把卷积主干的 hidden pre-activation 调制为

\[
h_t^{\rm pre} = (h_t^{\rm conv} W + b)\odot g_t + u_t.
\]

这里不必把它神化为某种复杂 attention；从实现上看，它就是“让少量 teacher 标量通过门控和偏移作用到隐藏层”。

### 10.2 为什么 `Gated v5` 值得单独讲

当前配置 [experiment_runtime_b_residual_norm_gated_teacher_v5.yaml](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml) 把 teacher 条件输入收窄到 4 个量：

1. `teacher_b_q`
2. `teacher_b_p`
3. `teacher_delta_b_q`
4. `teacher_delta_b_p`

并把它们改走 `scalar_branch`。这背后的理论直觉是：问题不一定是 teacher 信息本身有害，而可能是“整包 teacher 标量 + 广播平面注入”引入了冗余和耦合噪声。

### 10.3 现有单 seed 诊断说明了什么

在 `seed=20260429` 的 trace 诊断里，`Gated v5` 的最大残差范数反复撞到

\[
\max \|\delta b\|_2 = 0.169705627 = \sqrt{2}\cdot 0.12.
\]

这并不是巧合，而是两个分量同时触碰 `±0.12` clip 边界时的几何结果。项目里的解释是：某些窗口下，`Gated v5` 会进入高幅残差 regime，这能在部分场景带来收益，也能在某些 seed 下导致不稳定。

因此目前最安全的理论解读是：

1. 低维 gated 标量支路方向是合理的。
2. 但它当前仍有幅度/稳定性问题，尤其在特定 seed 和场景组合下。
3. 所以它是有价值的机制分支，不是当前正式排序主线。

---

## 11. 快回路：固定点、裁剪、AXI 与参数双缓冲

### 11.1 固定点 contract

当前快回路软件等价实现由 [linear_runtime.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/decoder/linear_runtime.py) 给出。默认固定点格式是

\[
Q4.20.
\]

这表示：

1. syndrome 输入按 `Q4.20` 量化。
2. \(K\)、\(b\) 也按 `Q4.20` 量化。
3. 输出 correction 在裁剪后再次量化回 `Q4.20`。

项目文档中长期使用的工程目标是：

1. `T_fast = 5 us`
2. `window_size = 2048`
3. `T_window = 10.24 ms`
4. 推荐 `T_slow_update = 20 ms`

这些数字来自 [docs/02_experiment_plan.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/docs/02_experiment_plan.md) 与 [docs/reference/CNN_FPGA_GKP_工程化实验方案.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/docs/reference/CNN_FPGA_GKP_工程化实验方案.md) 中的冻结工程口径。

### 11.2 快回路单拍公式

在软件实现里，一次快回路周期近似做的是：

\[
s_t^{\rm clip} = {\rm clip}(s_t, -s_{\max}, s_{\max}),
\]

\[
\Delta_t^{\rm raw} = K_t s_t^{\rm clip} + b_t,
\]

\[
\Delta_t^{\rm applied} = Q\!\left({\rm clip}(\Delta_t^{\rm raw}, -\Delta_{\max}, \Delta_{\max})\right).
\]

这里 \(Q(\cdot)\) 表示按 `Q4.20` 量化。

因此，快回路不是一个“数学上完全连续”的解码器，而是一个“裁剪 + 量化 + 饱和监控”的实时控制器。

### 11.3 AXI 寄存器 I/O

AXI 寄存器契约在 [axi_map.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/hwio/axi_map.py) 里是明确的。关键地址包括：

1. `0x10–0x1C`: `K11, K12, K21, K22`
2. `0x20–0x24`: `b1, b2`
3. `0x30`: `active_bank`
4. `0x34`: `epoch_id`
5. `0x38`: `commit_epoch`
6. `0x3C`: `hist_seq`

理论上，这意味着慢回路并不是把某个高层 Python 对象“交给 FPGA”，而是把量化后的 \(K,b\) 通过固定寄存器格式写入硬件契约。

### 11.4 参数双缓冲

`ParamBank` 是快慢回路解耦的关键。其核心思想是：

1. 快回路只读 `active_bank`。
2. 慢回路把新参数写到 `staging_bank`。
3. 到指定 `commit_epoch` 后，再原子切换。

若把活动参数记为 \((K_t^{A}, b_t^{A})\)，待切换参数记为 \((K_t^{B}, b_t^{B})\)，那么切换不是连续插值，而是某个安全边界上的离散更新：

\[
(K_t,b_t)=
\begin{cases}
(K^{A},b^{A}), & t < t_{\rm commit}\\
(K^{B},b^{B}), & t \ge t_{\rm commit}
\end{cases}
\]

这正是项目强调 `stage-then-commit`、`no glitch`、`atomic switch` 的原因。

---

## 12. 完整纠错大回路：从噪声到逻辑错误率

现在把整个主线闭环连起来。

### 12.1 快回路单周期

在 [fast_loop_emulator.py](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/cnn_fpga/runtime/fast_loop_emulator.py) 中，一次快回路周期可概括为：

1. 采样当前有效噪声状态

\[
\theta_t^{\rm noise} = (\sigma_t,\mu_{q,t},\mu_{p,t},\theta_t).
\]

2. 采样本轮新增误差 \(n_t\)，叠加到累积残差上：

\[
e_t =
\begin{cases}
r_{t-1}+n_t, & {\rm full\_qec}\\
n_t, & {\rm simplified}
\end{cases}
\]

3. 测量带噪 syndrome \(\tilde s_t\)。
4. 读取 active bank 中的 \((K_t,b_t)\)，执行线性解码。
5. 应用校正，得到新残差。
6. 更新逻辑错误追踪器。
7. 把 syndrome 投到 32×32 直方图窗口中。

### 12.2 窗口形成

当累积满 `window_size = 2048` 个周期后，快回路向慢回路发射一个窗口对象。窗口里最重要的内容有三类：

1. `histogram`
2. `target_params`
3. `diagnostics`

其中 `diagnostics` 包括：

1. `overflow_ratio`
2. `correction_saturation_ratio`
3. `aggressive_param_ratio`
4. `window_ler`
5. `mean_correction_utilization`

这些量就是后续机制分析和公平比较的基础。

### 12.3 慢回路更新

慢回路拿到窗口后，大致执行：

\[
H_t \rightarrow {\rm Teacher}(H_{1:t}) \rightarrow \hat\theta_t^{\rm teacher}
\]

\[
(\hat\theta_t^{\rm teacher}, H_{t-c+1:t}, \Delta H, b_{t-1}^{\rm teacher}, \Delta b_{t-1}^{\rm teacher})
\rightarrow {\rm CNN}
\rightarrow \delta b_t
\]

\[
(\hat\theta_t^{\rm teacher}, \delta b_t)
\rightarrow (K_t, b_t)
\rightarrow {\rm stage}
\rightarrow {\rm commit}.
\]

对于当前正式主线 `Hybrid Residual-B`，更具体地说：

1. teacher 先给 `NoisePrediction`
2. `ParamMapper` 把它映射成 `teacher_params = (K_t^{\rm teacher}, b_t^{\rm teacher})`
3. CNN 输出 \(\delta b_t\)
4. 最终使用

\[
K_t = K_t^{\rm teacher},
\qquad
b_t = {\rm EMA}\big(b_t^{\rm teacher}+\delta b_t\big).
\]

### 12.4 这条回路为什么叫“完整纠错大回路”

因为逻辑错误率最终由整个闭环共同决定：

1. 物理噪声决定窗口统计如何变化。
2. 窗口统计决定 teacher 和 CNN 看到什么。
3. teacher/CNN 决定提交什么 \(K,b\)。
4. 提交时序决定快回路何时切换参数。
5. 快回路参数决定每一拍 correction。
6. correction 决定残差如何积累。
7. 残差累积决定 `LER`。

所以从理论上讲，本项目真正研究的不是“一个 CNN 回归器”，而是“一个受实时约束的闭环控制系统”。

---

## 13. 如何用已有实验数据理解这条理论链

这一节只用仓库里已经文档化的数字做解释锚点，不把它们当作本轮新验证。

### 13.1 P1 说明“图像到参数”的静态识别能力是足够的

`static_theta_v2` 上，float 模型全局 `R² ≈ 0.994352`，各标签 `R²` 分别约为：

1. `sigma = 0.997613`
2. `mu_q = 0.996473`
3. `mu_p = 0.998459`
4. `theta_deg = 0.984862`

这说明：如果任务只是“从静态合成直方图回归绝对参数”，小 CNN 的表达能力是够的。

但这并不自动回答在线控制问题。

### 13.2 P3 说明“mock-backed software HIL”闭环是可重复的

恢复期当前明确验收过的最小软件 HIL 路径，关键数字是：

1. `final_ler = 0.454375`
2. `overflow_rate = 0.002`

它说明：在 `mock + model_artifact + artifact_npz + inproc` 的受限路径上，快慢回路闭环本身能确定性运行。

### 13.3 P4 说明“最好主线”不等于“最好离线回归器”

当前正式主线文档中，强 baseline 结果写得很清楚：

1. `Hybrid Residual-B = 0.798332`
2. `UKF = 0.817974`
3. gap \(\approx 0.019642\)

同时还明确写到：

1. `correction_saturation_rate = 0`
2. `aggressive_param_rate = 0`
3. 主导 overflow 仍是 `histogram_input`

这三点放在一起的理论含义是：`Hybrid Residual-B` 的收益并不是通过更激进控制“硬压出来”的，而更像是 teacher 残差修正在控制语义上确实更准。

### 13.4 `seed=20260429` 说明“更强 representation”仍可能在闭环里翻车

`Gated v5` 单 seed trace 诊断中，有两个非常关键的现象：

1. `static_bias_theta` 场景下 `Gated v5` 可略优于 `Full`
2. `step_sigma_theta` 场景下则明显更差

而且它反复打到

\[
\max \|\delta b\|_2 = \sqrt{2}\cdot 0.12.
\]

这表明：即便一种 teacher 表征在某些 seed 上更强，它仍可能因为残差幅度、clip 和时序互动，在闭环里表现不稳。

这与项目当前的稳定结论完全一致：离线改进不等于 formal HIL 改进。

---

## 14. 当前主线理论图景的最简总结

如果把整个项目压缩成一组最核心的公式，可以写成：

### 14.1 物理层

\[
e_t = r_{t-1} + n_t,\qquad
s_t = {\rm mod}(e_t,\lambda) + {\rm measurement\ noise}.
\]

### 14.2 快回路

\[
\Delta_t = K_t s_t + b_t,
\qquad
r_t = {\rm wrap}(e_t-\Delta_t).
\]

### 14.3 慢回路 teacher

\[
\hat\theta_t^{\rm teacher} = {\rm Teacher}(H_{1:t}),
\qquad
(K_t^{\rm teacher},b_t^{\rm teacher}) = {\rm ParamMapper}(\hat\theta_t^{\rm teacher}).
\]

### 14.4 慢回路 CNN

\[
\delta b_t = f_\phi(H_{t-c+1:t},\Delta H,z_t^{\rm teacher}),
\]

\[
K_t = K_t^{\rm teacher},
\qquad
b_t = {\rm EMA}(b_t^{\rm teacher}+\delta b_t).
\]

### 14.5 提交与执行

\[
(K_t,b_t)\xrightarrow{\rm stage/commit} {\rm active\ bank} \xrightarrow{\rm fast\ loop} \Delta_t.
\]

从这个角度看，本项目的主张其实非常窄也非常清楚：

不是“神经网络取代 GKP 解码”，  
而是“在实时硬件约束下，让经典 teacher 先给稳定估计，再让轻量 CNN 学习对快回路真正有用的 residual-b 修正”。

---

## 15. 边界、缺口与安全阅读方式

### 15.1 这份理论解释能支持什么

它能支持：

1. 你从公式上理解当前项目为何采用双回路。
2. 你理解 `teacher-guided residual-b` 为什么比绝对参数回归更贴近部署语义。
3. 你把物理层、数据层、模型层、运行时层和 FPGA I/O 层串起来。

### 15.2 它不能替代什么

它不能替代：

1. 真 `.tflite` runtime 验证
2. 真板验证
3. 多 seed 机制闭环证据
4. 更宽 benchmark 扩展
5. 论文级完整结果包

### 15.3 目前最值得你带着这份文档继续看的问题

如果后续要继续推进项目，最关键的问题已经不是“这条路线是否有理论合理性”，而是：

1. `residual-b` 为什么在大多数场景够用，在哪些场景不够用？
2. `Gated v5` 这类低维 teacher 标量门控，如何在不进入 clip-regime 的情况下稳定发挥作用？
3. `teacher + residual-b` 的闭环优势，究竟能否在多 seed 机制证据和更强 benchmark 中继续成立？

这些问题超出了本说明文档的范围，但正是它后面的下一层研究问题。
