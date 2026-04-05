# P4 UKF 正式结果与 Hybrid 机制分析

## 1. 文档目的

本文档用于单独整理以下两部分内容：

1. `UKF` 作为强经典 baseline 的正式结果
2. `Hybrid Residual-B` 相对 `UKF` 的机制性差异分析与后续 ablation 建议

该文档不替代阶段结论，而是作为论文准备和组内讨论的专项分析附件。

---

## 2. 结果来源

正式结果来自：

- [summary.json](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_strong_baselines_v1_20260403_145747_b82874392710_86447/summary.json)
- [comparison.csv](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_strong_baselines_v1_20260403_145747_b82874392710_86447/comparison.csv)
- [report.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_multiscenario_strong_baselines_v1_20260403_145747_b82874392710_86447/report.md)

正式比较集合为：

1. `EKF`
2. `UKF`
3. `Constant Residual-Mu`
4. `RLS Residual-B`
5. `Hybrid Residual-B`

场景集为：

1. `static_bias_theta`
2. `linear_ramp`
3. `step_sigma_theta`
4. `periodic_drift`

每个场景重复 `4` 次。

---

## 3. 正式结果摘要

### 3.1 跨场景平均 LER

- `Hybrid Residual-B = 0.798332`
- `UKF = 0.817974`
- `Constant Residual-Mu = 0.825719`
- `RLS Residual-B = 0.827908`
- `EKF = 0.828369`

当前正式排序为：

1. `Hybrid Residual-B`
2. `UKF`
3. `Constant Residual-Mu`
4. `RLS Residual-B`
5. `EKF`

这说明：

- 修正后的 `UKF` 已经成为当前最强经典 baseline
- 但 `Hybrid Residual-B` 仍稳定领先 `UKF`

`UKF` 相对 `Hybrid Residual-B` 的平均 gap 为：

- `0.817974 - 0.798332 = 0.019642`

---

## 4. 分场景比较

### 4.1 `linear_ramp`

- `Hybrid Residual-B = 0.787960`
- `UKF = 0.810263`
- gap = `0.022304`

解释：

- 该场景是线性漂移，理论上更接近 `UKF` 的“连续状态追踪”假设
- 即便如此，`Hybrid Residual-B` 仍保持超过 `0.02` 的优势
- 这说明当前 `Hybrid` 的收益不只是“比 EKF/UKF 更会做状态平滑”

### 4.2 `periodic_drift`

- `Hybrid Residual-B = 0.806678`
- `UKF = 0.821967`
- gap = `0.015289`

解释：

- 周期漂移对纯状态模型提出更高要求
- `UKF` 已经优于 `EKF`
- 但 `Hybrid` 仍进一步降低了 LER

### 4.3 `static_bias_theta`

- `Hybrid Residual-B = 0.811530`
- `UKF = 0.826767`
- gap = `0.015237`

解释：

- 该场景不是强时间变化场景，更接近“静态偏置 + 姿态误差”
- `Hybrid` 在这里仍领先，说明其优势不只来自动态漂移追踪
- 更可能还包含对 teacher 偏差、映射误差或窗口统计偏差的补偿

### 4.4 `step_sigma_theta`

- `Hybrid Residual-B = 0.787161`
- `UKF = 0.812898`
- gap = `0.025737`

解释：

- 这是 `Hybrid` 相对 `UKF` 优势最大的场景
- 该现象支持一个重要判断：
  - `Hybrid` 对突变型 drift 更有优势
  - `UKF` 即使数值稳定，也仍受到递推状态模型惯性的限制

---

## 5. 为什么可以认为 Hybrid 的优势是“真实机制差异”，而不是统计假象

### 5.1 不是 overflow 差异导致

跨场景平均 overflow：

- `Hybrid Residual-B = 0.002535`
- `UKF = 0.002557`
- `EKF = 0.002567`

`Hybrid` 和 `UKF` 的 overflow 差距非常小，仅约 `2.2e-05` 量级。

这意味着：

- `Hybrid` 领先不是因为它显著减少了输入饱和
- 输入统计范围固然重要，但不足以解释 `0.0196` 的 LER 差距

### 5.2 不是更激进的参数导致

正式结果中：

- `correction_saturation_rate = 0`
- `aggressive_param_rate = 0`

对 `Hybrid / UKF / EKF / RLS` 都成立。

这意味着：

- `Hybrid` 并不是通过更激进的 `K,b` 换取更低 LER
- 当前优势更可能来自“预测方向更准”，而不是“动作更猛”

### 5.3 不是偶然 seed 波动

本轮正式结果使用 4 次重复，且：

- `UKF` 在 4 个场景中全部优于 `EKF`
- `Hybrid` 在 4 个场景中全部优于 `UKF`

因此，现阶段至少可以认为：

- `UKF > EKF` 是稳定结论
- `Hybrid > UKF` 也是稳定结论

---

## 6. 机制分析

以下结论属于基于当前结果的工程推断，不是数学证明。

### 6.1 `UKF` 的作用

`UKF` 的提升主要来自两点：

1. 比 `EKF` 更完整地保留状态协方差结构
2. 数值稳定化后，递推不再因为协方差退化而失真

因此它把“纯经典递推滤波”这一条线抬到了当前最强水平。

### 6.2 `Hybrid` 为什么还能继续赢

当前更合理的解释是：

1. `Hybrid` 不是直接替代 teacher

- 它保留了 teacher 提供的稳定一阶估计
- 不承担全部绝对参数辨识

2. `Hybrid` 学的是运行时真正有用的残差量

- 目标不是直接输出完整 `(sigma, mu_q, mu_p, theta_deg)`
- 而是输出对 `b` 的小幅残差修正
- 这比“绝对参数回归”更接近实际部署链路

3. `Hybrid` 具备短时上下文和 teacher 条件信息

- 当前主模型输入不仅有窗口直方图
- 还有历史窗口、teacher 预测和 teacher 参数
- 因而它有能力学习“teacher 在什么情况下会系统性偏一点”

4. `Hybrid` 对非平稳窗口误差更敏感

- 在 `step_sigma_theta` 和 `linear_ramp` 上，`Hybrid` 的优势更大
- 说明它对 drift 突变和连续漂移都能利用额外上下文做修正

### 6.3 当前最可能的机制图景

一个与结果一致的工程图景是：

1. `UKF` 负责把“经典滤波上限”推高
2. `Hybrid` 在 teacher 已经不差的基础上，继续学习 teacher 的残余偏差
3. 这种残余偏差并不表现为极端参数，而表现为：
   - 更准确的 `b` 修正方向
   - 更好的短时场景适配
   - 更少的 teacher 系统偏差残留

---

## 7. 对论文叙事的意义

当前结果已经支持一个更清晰的叙事：

1. 单纯把 CNN 当作“绝对参数回归器”并不是最佳方案
2. 纯经典滤波中，`UKF` 已经是明显比 `EKF` 更强的对照
3. 即便在这种更强的经典对照下，`Hybrid Residual-B` 仍稳定更优
4. 因此项目的创新点更适合表述为：
   - 在双回路 runtime 语义一致的前提下
   - 用轻量学习模块做 teacher-guided residual correction
   - 而不是简单宣称“CNN 直接优于一切经典方法”

这种表述会更稳，也更容易经受审稿质疑。

---

## 8. 建议优先做的 ablation

当前最值得做的 ablation 不是再去和更弱 baseline 反复比较，而是围绕 `Hybrid vs UKF` 解释“为什么会赢”。

建议优先级如下：

### 8.1 `teacher_mode` ablation

目标：

- 验证 `Hybrid` 的优势有多少来自 teacher 质量

建议比较：

1. `UKF`
2. `Hybrid Residual-B (teacher = window_variance)`
3. `Hybrid Residual-B (teacher = ukf)`

解释：

- 如果 `teacher = ukf` 后 `Hybrid` 还能继续提升，就说明 residual learner 的价值独立存在
- 如果提升明显缩小，则说明当前收益的一部分来自“teacher 本身还不够强”

### 8.2 输入通道 ablation

目标：

- 区分 `Hybrid` 的收益到底来自哪里

建议拆掉以下输入：

1. 去掉 `teacher_params`
2. 去掉 `teacher_prediction`
3. 去掉 `teacher_deltas`
4. 去掉 `histogram_deltas`

解释：

- 这能回答“到底是 teacher 条件信息重要，还是时间差分信息重要”

### 8.3 时间上下文 ablation

目标：

- 判断收益是否主要来自上下文长度

建议比较：

1. `context_windows = 1`
2. `context_windows = 3`
3. `context_windows = 5`

解释：

- 如果 gap 主要在 `context=1 -> 3` 产生，说明短时上下文已足够
- 如果 `3 -> 5` 还有明显收益，说明 runtime-consistent temporal modeling 仍有空间

### 8.4 目标定义 ablation

目标：

- 验证“学 `delta_b`”是否确实优于“学绝对参数”或“学 `delta_mu`”

建议比较：

1. `CNN-FPGA` 绝对参数回归
2. `Hybrid Residual-Mu`
3. `Hybrid Residual-B`

解释：

- 这是当前主线最关键的一组方法学 ablation

---

## 9. Particle Filter 当前状态

`Particle Filter` 已经开始重构为更贴近当前运行时语义的：

- `teacher-guided residual PF`

即：

- 不再直接估计绝对 `(sigma, mu_q, mu_p, theta_deg)`
- 而是围绕 teacher 的 `b` 追踪 `delta_b`

当前判断：

- 这是正确方向
- 但在代码刚完成重构后，暂不继续跑正式 benchmark
- 先把 `UKF` 正式结论和 `Hybrid vs UKF` 机制分析整理清楚，更符合当前论文准备优先级

---

## 10. 当前可直接引用的结论

如果只保留一句最核心的正式结论，可以写成：

> 在当前固定场景集和长配置正式 benchmark 中，修正后的 UKF 已成为最强经典 baseline，但 Hybrid Residual-B 仍在四个场景上全部保持更低 LER，说明 teacher-guided residual correction 的收益并不等同于单纯更强的经典滤波。
