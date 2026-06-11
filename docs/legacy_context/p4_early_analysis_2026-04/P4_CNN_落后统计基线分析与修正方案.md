# P4 CNN 落后统计基线分析与修正方案

## 1. 结论先行

截至 2026-03-31，P4 正式多场景对比已经说明一件事：

- 当前 `CNN-FPGA` 不是只在个别场景里偶然落后。
- 在冻结协议下，`Window Variance` 和 `EKF` 都稳定优于当前 Tiny-CNN。
- 其中最终胜者不是 `Window Variance`，而是 `EKF`。

正式结果目录：

- `runs/p4_benchmark/p4_multiscenario_v1_20260331_140222_6489e1886350`

同窗诊断目录：

- `runs/p4_diagnostic/p4_gap_diag_20260331_232738_6489e1886350`

这意味着后续主线不能再把问题表述成“CNN 还需要再调一点点”。
更准确的表述是：

- 当前 CNN 训练目标、训练数据分布、运行时输入分布、闭环控制目标之间存在系统性错配。

---

## 2. 正式 P4 结果解读

正式冻结模式：

- `Static Linear`
- `Window Variance`
- `EKF`
- `CNN-FPGA`

正式冻结场景：

- `static_bias_theta`
- `linear_ramp`
- `step_sigma_theta`
- `periodic_drift`

### 2.1 四场景平均 LER

- `Static Linear`: `1.111963`
- `Window Variance`: `0.857269`
- `EKF`: `0.855608`
- `CNN-FPGA`: `0.954301`

### 2.2 相对当前 CNN 的平均优势

- `Window Variance` 相对 `CNN-FPGA` 平均 `dLER = -0.097032`
- `EKF` 相对 `CNN-FPGA` 平均 `dLER = -0.098693`

这不是统计噪声级别的小波动，而是四个场景都同方向、且幅度接近的稳定差距。

### 2.3 正式结果的额外含义

正式 P4 里还可以看到：

- `dominant_overflow_source` 始终是 `histogram_input`
- `correction_saturation_rate = 0`
- `aggressive_param_rate = 0`

因此当前 CNN 落后，并不是“它把参数打得太猛导致溢出”。
主矛盾不是快回路爆炸，而是慢回路给出的参数虽然没有触发激进告警，却没有给出更好的控制效果。

---

## 3. 为什么需要“同窗诊断”

正式 P4 适合做最终排名，但不适合直接解释病因，原因有两点：

- 不同 mode 使用不同 seed。
- 不同 mode 的闭环参数会反过来改变后续窗口分布。

因此我们新增了：

- `cnn_fpga/benchmark/run_p4_gap_diagnostic.py`

它的做法是：

1. 先用一个参考闭环模式采集一批原始窗口。
2. 保存同一批窗口的：
   - 原始 histogram
   - `target_params`
   - 窗口诊断量
3. 再在完全相同的窗口序列上，离线重放：
   - `window_variance`
   - `ekf`
   - `cnn_fpga`

这样可以把“同一批窗口上的预测差异”单独拿出来看。

本次同窗诊断使用：

- `reference_mode = static_linear`
- `n_windows_per_scenario = 64`

---

## 4. 同窗诊断的关键发现

### 4.1 发现一：CNN 在 `sigma/theta` 上不差，甚至更接近窗口真值

在 4 个场景中，当前 CNN 的表现有一个很反直觉的点：

- `sigma MAE` 往往优于 `Window Variance / EKF`
- `theta MAE` 也往往优于 `Window Variance / EKF`

例如：

#### `static_bias_theta`

- `window_variance`: `sigma MAE = 0.630000`, `theta MAE = 7.397921`
- `ekf`: `sigma MAE = 0.609558`, `theta MAE = 3.077603`
- `cnn_fpga`: `sigma MAE = 0.470396`, `theta MAE = 2.618085`

#### `linear_ramp`

- `window_variance`: `sigma MAE = 0.697298`, `theta MAE = 9.874839`
- `ekf`: `sigma MAE = 0.676856`, `theta MAE = 2.958445`
- `cnn_fpga`: `sigma MAE = 0.542287`, `theta MAE = 1.697519`

这说明当前 Tiny-CNN 不是“什么都没学到”。

它学到了一部分与形状相关的静态信息。

---

### 4.2 发现二：CNN 在 `mu_q / mu_p` 上明显更差，而且带有稳定方向性偏置

这才是当前最关键的病因信号。

在 3 个非零均值场景里，CNN 的 `mu_q / mu_p` 明显劣于统计基线。

例如：

#### `static_bias_theta`

- `window_variance`: `mu_q MAE = 0.036493`, `mu_p MAE = 0.026677`
- `ekf`: `mu_q MAE = 0.037451`, `mu_p MAE = 0.026253`
- `cnn_fpga`: `mu_q MAE = 0.075729`, `mu_p MAE = 0.036755`

#### `linear_ramp`

- `window_variance`: `mu_q MAE = 0.015421`, `mu_p MAE = 0.011716`
- `ekf`: `mu_q MAE = 0.012117`, `mu_p MAE = 0.009115`
- `cnn_fpga`: `mu_q MAE = 0.053240`, `mu_p MAE = 0.019100`

#### `periodic_drift`

- `window_variance`: `mu_q MAE = 0.017556`, `mu_p MAE = 0.015625`
- `ekf`: `mu_q MAE = 0.014953`, `mu_p MAE = 0.014770`
- `cnn_fpga`: `mu_q MAE = 0.055004`, `mu_p MAE = 0.026728`

更重要的是，CNN 不是随机波动，而是表现出稳定偏置：

- `mean pred mu_q` 约为 `-0.04`
- `mean pred mu_p` 约为 `+0.01`

这个偏置在多个场景中几乎不变。

这说明当前 CNN 很可能学成了一个“固定方向偏置器”，而不是一个真正跟随窗口均值漂移的估计器。

---

### 4.3 发现三：CNN 的 `mu` 符号错判率极高

这比 MAE 更说明问题，因为 `mu` 在映射里直接进入偏置项 `b`，方向错了会直接把校正推反。

#### `static_bias_theta`

- `cnn_fpga`: `mu_q sign mismatch = 1.000000`
- `cnn_fpga`: `mu_p sign mismatch = 0.953125`

#### `linear_ramp`

- `cnn_fpga`: `mu_q sign mismatch = 1.000000`
- `cnn_fpga`: `mu_p sign mismatch = 0.906250`

#### `periodic_drift`

- `cnn_fpga`: `mu_q sign mismatch = 1.000000`
- `cnn_fpga`: `mu_p sign mismatch = 0.937500`

对比来看，`Window Variance / EKF` 虽然也不完美，但错判率通常在 `0.3 ~ 0.7` 区间，不会像 CNN 一样几乎始终朝错方向输出。

因此，当前 CNN 的核心问题不是“幅值略有偏差”，而是：

- `mu_q / mu_p` 的方向性估计存在系统性偏置。

---

### 4.4 发现四：即使按 `target_params` 衡量，统计基线的 `sigma/theta` 也并不准，但它们依然赢了 LER

这件事非常关键。

例如在 `static_bias_theta` 的同窗诊断中：

- `window_variance` 的平均 `pred_sigma = 0.85`
- `ekf` 的平均 `pred_sigma = 0.829558`
- 真值 `target_sigma = 0.22`

也就是说，统计基线在“回归窗口真值参数”这个口径下，并不准确。
但它们依然在正式 P4 的闭环 LER 上胜出。

这说明：

- 当前慢回路任务并不等价于“精确回归物理真值参数”。
- 现在的 `target_params = (sigma, mu_q, mu_p, theta_deg)` 更像是一个物理描述标签，
  但不一定是控制最优标签。

换句话说：

- 训练时的监督目标，与闭环时真正需要优化的对象，不是完全同一个量。

---

## 5. 根因分析

### 5.1 根因一：训练数据分布与运行时分布严重不一致

当前训练数据来自：

- `cnn_fpga/data/dataset_builder.py`

它生成的是：

- 椭圆高斯样本
- 旋转
- 平移
- wrap 到基本晶胞
- 直接做 histogram

但 P4 运行时窗口来自：

- `cnn_fpga/runtime/fast_loop_emulator.py`

运行时窗口包含了训练集没有覆盖的内容：

- 真实测量模型 `RealisticSyndromeMeasurement`
- `use_full_qec_model=True` 下的残余误差累积
- 已经施加过的快回路校正反馈
- 窗口平均 target，而不是单次静态样本标签
- 时变漂移轨迹
- 直方图范围截断和慢回路节拍效应

因此当前 CNN 训练时看到的输入分布，与 P4 真正部署时看到的窗口分布不是同一个问题。

这会直接导致：

- 离线 test 很好
- 闭环部署却明显掉队

这也解释了为什么：

- `static_theta_v2` test 上 `R²≈0.994`
- 但 P4 里仍系统性输给 `Window Variance / EKF`

---

### 5.2 根因二：当前 CNN 是“单窗口静态回归”，而 EKF 是“递推估计器”

当前 Tiny-CNN 的工作方式是：

- 输入一个窗口
- 独立输出一组参数

它没有显式使用：

- 前一窗口预测
- 前一窗口 active 参数
- 时间递推状态

而 `EKF` 明确利用了：

- 观测值
- 过程噪声
- 状态递推
- 平滑后的历史信息

因此在 `linear_ramp / step / periodic` 这类时变场景中，EKF 具备当前 CNN 没有的结构优势。

这一点不是简单多训几轮就会自动消失的。

---

### 5.3 根因三：当前监督标签对控制目标不够直接

当前训练目标是：

- `sigma`
- `mu_q`
- `mu_p`
- `theta_deg`

但闭环真正执行的是：

- `ParamMapper(prediction) -> K, b`

控制效果最后由：

- `K`
- `b`
- 快回路测量残差
- 物理反馈

共同决定。

所以目前存在一个结构性问题：

- “回归物理参数做得更准”，不一定等价于“LER 更低”。

从同窗诊断看，这个问题已经被实验直接证实了。

---

## 6. 修正方案

下面给出的不是泛泛建议，而是建议按优先级执行的工程路线。

### 6.1 第一优先级：先修 `mu_q / mu_p`，不要继续把重点放在 `theta`

原因：

- 当前 CNN 的主要闭环弱点不是 `theta`
- 而是 `mu_q / mu_p` 的幅值与方向
- 特别是方向错判率过高

建议：

1. 重新设计损失权重，优先提高 `mu_q / mu_p`
2. 对 `mu_q / mu_p` 增加符号一致性惩罚
3. 在验收里新增：
   - `mu_q sign mismatch rate`
   - `mu_p sign mismatch rate`

只有当 `mu` 的方向性问题解决后，再讨论 `theta` 的进一步提升是否值得。

---

### 6.2 第二优先级：把训练数据改成“运行时一致窗口”

建议新增一套 runtime-consistent 数据集，而不是继续只用 `dataset_builder.py` 的静态合成样本。

新数据应来自真实运行链路：

1. 用 `FastLoopEmulator` 生成窗口
2. 打开与 P4 一致的：
   - `use_full_qec_model`
   - measurement noise
   - histogram range
   - window size / stride
3. 用冻结场景集或其混合分布生成训练窗口

这样训练集和部署集才在同一个任务定义上。

最低要求：

- 训练窗口必须来自与 P4 相同的 histogram 构造逻辑
- 不能再用“静态单样本 wrap 后直接 histogram”代替部署窗口

---

### 6.3 第三优先级：把目标从“绝对物理参数回归”改成“基线残差回归”

当前最值得尝试的结构不是“纯 CNN 替代统计基线”，而是：

- `Window Variance / EKF` 作为稳定 teacher
- CNN 学 teacher 之上的残差或修正量

推荐两条路线：

#### 路线 A：`Window Variance + CNN residual`

- 先用 `Window Variance` 给出基础 `(sigma, mu_q, mu_p, theta_deg)`
- CNN 只输出修正量：
  - `delta_sigma`
  - `delta_mu_q`
  - `delta_mu_p`
  - `delta_theta`

优点：

- 输入分布更稳定
- 学习任务更小
- 不容易出现当前这种固定方向偏置

#### 路线 B：直接学 `delta_b` 或 `b`

由于当前主问题集中在 `mu -> b` 这条链上，可以考虑：

- 保留统计基线给 `K`
- CNN 重点修正 `b`

如果实验资源有限，优先做这条路线。

因为现在真正伤害 LER 的，更像是偏置项方向问题，而不是增益矩阵方向问题。

---

### 6.4 第四优先级：给 CNN 增加最小时间上下文

EKF 能赢，不只是因为它是“经典方法”，而是因为它真的用了时间信息。

因此后续 CNN 不应该继续保持“纯单窗独立回归”。

最小可行方案：

1. 输入堆叠最近 `N=2~4` 个窗口 histogram
2. 或者在输入里拼接：
   - 上一拍预测
   - 上一拍 active `K,b`
   - 上一拍窗口统计量

更保守的方案：

- CNN 仍输出当前窗预测
- 再在输出端接一个轻量 EMA/EKF 后处理

如果只允许做最小改动，优先这个“CNN + 递推平滑”方案。

---

### 6.5 第五优先级：评估指标改为“控制相关指标优先”

后续模型验收不应再只看：

- test MSE
- per-label R²

建议新增三层门槛：

#### 第一层：同窗预测门槛

- `mu_q / mu_p` MAE 不劣于 `Window Variance`
- `mu_q / mu_p` sign mismatch 显著低于当前 CNN

#### 第二层：映射后门槛

- `b` 相关误差优先于 `K` 相关误差
- 尤其关注 `b` 的方向和均值偏置

#### 第三层：闭环门槛

- 在 paired same-seed 或 frozen P4 上
- 至少不劣于 `Window Variance`
- 最终目标是接近或超过 `EKF`

---

## 7. 推荐的下一步实验顺序

### Step 1

基于现有诊断入口，冻结一版新的验收口径：

- 必看 `mu_q / mu_p` 的 MAE 和 sign mismatch
- 不再只看 `theta`

### Step 2

新增 runtime-consistent 数据集生成入口：

- 用 `FastLoopEmulator` 直接产出训练窗口
- 标签先保留原始 `(sigma, mu_q, mu_p, theta_deg)`
- 同时额外存 teacher 预测与 `K,b`

### Step 3

先做最保守、最可能成功的主模型：

- `Window Variance + CNN residual(mu/b)` 或
- `EKF teacher + CNN residual`

### Step 4

如果 Step 3 仍不能超过 `Window Variance`，再做：

- 多窗口输入
- 轻量递推头
- 直接回归 `b` 或 `delta_b`

---

## 8. 当前不建议做的事

### 8.1 不建议继续只围绕 `theta` 做定向增强

因为本轮证据已经显示：

- 当前 CNN 的 `theta` 不是最主要短板
- 真正更伤闭环的是 `mu` 偏置

### 8.2 不建议继续只调 `gain_scale / alpha_bias / beta`

这些参数当然还能调，但它们不能解决：

- CNN 输出方向错的问题
- 数据分布错配问题
- 单窗口模型缺少时间信息的问题

也就是说，继续纯调 mapper 只能缓解，不会根治。

### 8.3 不建议直接把“物理真值参数回归更准”当成唯一目标

因为这件事已经被本轮诊断否定：

- 统计基线并不一定更接近 `target_params`
- 但它们仍然能获得更低 LER

所以今后必须把控制有效性一起纳入监督和验收。

---

## 9. 一句话版本

当前 CNN 落后统计基线，不是因为它完全不会识别窗口形状，而是因为它在 runtime 分布下对 `mu_q / mu_p` 存在稳定方向偏置，同时训练目标又没有直接对齐闭环控制目标；后续主线应转向“runtime-consistent 数据 + teacher residual + 时间上下文”，而不是继续围绕静态标签或单纯调 mapper 打转。
