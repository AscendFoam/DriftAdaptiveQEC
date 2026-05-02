# CNN-FPGA-GKP paper-inspired 分支实验设计草案

## 1. 文档目的

本文档用于把参考论文  
[Fast and accurate AI-based pre-decoders for surface codes.md](../relative_papers/Fast%20and%20accurate%20AI-based%20pre-decoders%20for%20surface%20codes.md)  
带来的启发，进一步细化成一条可直接落地的实验分支。

这条分支的目标不是替代当前正式主线，也不是推翻 `Gated v5`，而是回答一个更具体的问题：

- 在保持当前 `teacher-guided residual-b` 路线不变的前提下，是否可以借鉴论文的“统计聚合 + 下游一致训练”思想，把 `Gated v5` 再往前推进一步？

因此，这份草案的定位是：

1. 保持与当前 `Gated v5` 高可比性；
2. 只引入少量、可解释、可单独验证的新改动；
3. 先做小规模 paired benchmark，不直接上最长正式长跑。

建议把该分支临时命名为：

- `paper_inspired_statcalib_v1`

---

## 2. 当前基线：`Gated v5`

当前最强的 teacher-representation 候选是 `Gated v5`，其关键特征是：

1. 仍然使用 `residual_b` 标签；
2. 仍然以多窗口 histogram 为主干输入；
3. 不再把整包 teacher 信息 broadcast 成整张平面；
4. 只保留 4 个 teacher 标量：
   - `teacher_b_q`
   - `teacher_b_p`
   - `teacher_delta_b_q`
   - `teacher_delta_b_p`
5. 这 4 个标量通过 `scalar_branch + gated` 方式注入主干。

因此，paper-inspired 分支不应回退到更早的“整包 teacher + broadcast”路线，而应建立在 `Gated v5` 之上继续增强。

---

## 3. 分支目标

这条分支要验证的不是“能否再造一个全新模型”，而是下面这 3 个更聚焦的问题：

1. 当前 `Gated v5` 的收益，有多少来自更合理的 teacher 表征？
2. 当前 `Gated v5` 的稳定性，能否通过模型内统计聚合再提升，而不是过度依赖外部 EMA？
3. 把 loss 再往闭环有效量推进一步后，是否能让离线训练与 formal HIL 的一致性更强？

因此，本分支只允许改动这 3 类东西：

1. 输入张量的统计表达；
2. 模型中的轻量聚合结构；
3. loss 的闭环一致附加项。

不建议在这条分支里同时改：

1. teacher 模式；
2. benchmark 场景集合；
3. ParamMapper 口径；
4. 快回路控制语义。

否则无法判断收益到底来自哪里。

---

## 4. 输入张量怎么改

### 4.1 保持不变的部分

为了与 `Gated v5` 保持可比性，以下输入不改：

1. 主干仍使用当前 runtime-consistent 多窗口 histogram；
2. 保留 `context_windows = 5`；
3. 保留 `include_histogram_deltas = true`；
4. 保留 `teacher_prediction_layout / teacher_params_layout / teacher_deltas_layout = scalar_branch`；
5. 保留 4 个 teacher 标量：
   - `teacher_b_q`
   - `teacher_b_p`
   - `teacher_delta_b_q`
   - `teacher_delta_b_p`

### 4.2 新增的输入表达

论文最值得借的地方，不是额外加更多 teacher 量，而是对“统计状态”的表达更清楚。  
因此建议在不增加太多维度的前提下，给当前输入补两类轻量统计量：

#### A. 窗口级 histogram summary 标量

在 `scalar_branch` 中新增少量从当前多窗口直方图直接提取的 summary 标量，例如：

1. 当前窗口 histogram 总能量
2. 最近 5 窗 histogram 总能量均值
3. 最近 5 窗 histogram 总能量标准差
4. 最近 5 窗 histogram 质心在 `q / p` 方向的漂移量
5. 最近 5 窗 histogram 各向异性强度摘要

这些量的作用是：

1. 让模型显式看到“当前统计状态稳不稳”；
2. 减轻网络只能靠卷积自己隐式压缩统计量的负担；
3. 更贴近论文中“全局统计量通过低维路径注入”的思想。

建议命名可统一为：

- `hist_stat_energy_t`
- `hist_stat_energy_mean`
- `hist_stat_energy_std`
- `hist_stat_centroid_shift_q`
- `hist_stat_centroid_shift_p`
- `hist_stat_anisotropy`

#### B. teacher 可信度 summary 标量

新增一组很轻量的“teacher 状态稳定性”标量，而不是新增更多 teacher 绝对值：

1. `||delta_b||`
2. 最近 5 窗 `b` 变化量均值
3. 最近 5 窗 `b` 变化量标准差

这些量的作用是：

1. 帮助模型判断“teacher 当前是稳定慢漂移，还是进入了突变区”；
2. 比单独重复塞更多 `teacher prediction / teacher delta` 更紧凑；
3. 更容易和后续 gate 或 loss 做对应分析。

### 4.3 输入改动的边界

这条分支里明确不建议：

1. 不恢复 `teacher_sigma / mu_q / mu_p / theta_deg` 的整包输入；
2. 不重新引入 broadcast teacher 平面；
3. 不增加大量 hand-crafted feature，避免把变量一次改太多。

### 4.4 推荐版本

因此，paper-inspired v1 的输入建议是：

1. 主干空间输入：
   - 与 `Gated v5` 完全相同的 histogram history + histogram deltas
2. scalar branch：
   - 原有 4 个 teacher 标量
   - 5 到 8 个轻量 histogram / teacher stability summary 标量

也就是说，本质上是：

- `Gated v5 + compact statistical summaries`

---

## 5. 模型怎么改

### 5.1 保持不变的部分

为保证实现成本可控，以下结构建议保持不变：

1. 仍然使用当前 `tiny_cnn` 路线；
2. 主干仍然是轻量 `Conv2d + ReLU + AvgPool + Flatten`；
3. teacher 标量仍然走 `scalar_branch + gated`；
4. 输出维度仍为 2，对应：
   - `delta_b_q`
   - `delta_b_p`

### 5.2 新增的模型结构

建议只加一个很轻量的“统计聚合头”，不要一步跳到重 2D/3D 网络。

#### 方案：dual-branch stat-calib head

把当前 `tiny_cnn` 结构轻量扩展成 3 支路：

1. `spatial branch`
   - 处理 histogram history / delta 的空间张量
   - 继续负责空间图样识别

2. `teacher scalar branch`
   - 处理 `teacher_b / delta_b`
   - 继续负责 gated 注入

3. `stat summary branch`
   - 处理新增的 compact summary 标量
   - 只经过一层小 MLP

然后在 hidden 层进行融合：

1. `base_hidden = fc1(spatial_features)`
2. `teacher_gate = sigmoid(W_teacher * teacher_scalar + b_teacher)`
3. `stat_gate = sigmoid(W_stat * stat_summary + b_stat)`
4. `hidden = relu(base_hidden * teacher_gate * stat_gate + teacher_shift + stat_shift)`

这样设计的好处是：

1. 仍保持轻量；
2. 可以单独分析 teacher 与统计 summary 各自的门控作用；
3. 很适合做解释型诊断。

### 5.3 为什么不建议直接上 attention / transformer

当前阶段不建议直接引入更重的时间注意力或 transformer，原因是：

1. 当前样本规模和 benchmark 规模还不足以支撑复杂模型自由度；
2. 部署语义上不如当前轻量模型清楚；
3. 一旦结果变好，很难判断收益来自“更合理的统计表达”还是“模型单纯更大”。

因此，paper-inspired v1 要坚持一个原则：

- 优先验证“方法论对不对”，而不是优先堆模型容量。

### 5.4 推荐版本

模型建议命名为：

- `tiny_cnn_gated_statcalib_v1`

其本质是：

1. `Gated v5` 的主干不动；
2. 新增一条 compact summary 标量支路；
3. 用双门控或双 shift 的方式与主干融合。

---

## 6. loss 怎么改

### 6.1 当前 `Gated v5` 的 loss

当前主线的核心还是监督 `delta_b_q / delta_b_p`，也就是：

- 目标是让模型输出的残差尽量接近 target residual-b

这已经比早期绝对参数回归更贴近下游闭环，但仍然主要是“标签数值一致”。

### 6.2 paper-inspired v1 的 loss 设计原则

借鉴论文后，loss 的改动不应过重，而应遵循：

1. 主损失仍保留 `delta_b` 监督；
2. 只增加少量闭环一致附加项；
3. 每个附加项都必须能解释它与最终 LER 的关系。

### 6.3 推荐 loss 结构

建议采用三项组合：

#### A. 主损失：残差监督损失

仍然保留：

- `L_residual = MSE(delta_b_pred, delta_b_target)`

这是主项，权重最大。

#### B. 闭环目标损失：补偿后 `b_next` 对齐损失

定义：

- `b_next_pred = teacher_b + delta_b_pred`
- `b_target = target_runtime_b`

然后增加：

- `L_bnext = MSE(b_next_pred, b_target)`

它的意义是：

1. 允许模型不只看“残差像不像”；
2. 还看“补偿后的最终控制量对不对”。

这比单纯残差监督更贴近闭环语义。

#### C. 风险代理损失：轻量稳定性约束

加入一个很轻量的 smooth penalty，例如：

1. 超过 `residual_clip_b` 附近时增加软惩罚；
2. 对非常激进的 `delta_b` 增加小权重约束；
3. 可选地对短窗口内输出跳变过大做平滑约束。

可以写成：

- `L_risk = soft_clip_penalty(delta_b_pred) + smooth_penalty`

它的意义是：

1. 让训练时就对闭环风险有一点敏感性；
2. 不必等到 formal HIL 才发现某些输出虽然拟合对了，但运行时很激进。

### 6.4 推荐总损失

建议总损失先用一个保守版本：

- `L = L_residual + λ1 * L_bnext + λ2 * L_risk`

其中：

1. `λ1` 设为较小但非零，先确保不会压过主损失；
2. `λ2` 更小，只作为稳定性正则。

推荐第一版不要加入：

1. surrogate LER 直接近似；
2. 多步 rollout loss；
3. 太复杂的 differentiable runtime penalty

因为这些会显著增加实现复杂度，也会让定位问题变困难。

### 6.5 第一版 loss 的目标

paper-inspired v1 的 loss 改动只想验证一件事：

- 当训练目标从“只拟合残差标签”稍微推进到“兼顾最终控制量与轻量闭环风险”后，formal HIL 是否更稳。

---

## 7. 跑哪一轮 paired benchmark 最合适

### 7.1 不建议直接上的 benchmark

第一轮不建议直接上：

1. 最长正式长跑；
2. 全 seed、大 repeats、全 scenario 一次打满；
3. `Full / Gated v5 / paper-inspired` 三者一起大规模全量跑。

原因是：

1. 当前长进程托管本身不稳定；
2. 这条分支是结构增强试探，不是正式主线替换；
3. 先判断方向值不值得继续，再决定是否扩大。

### 7.2 最适合的第一轮 benchmark

最适合的第一轮是：

- 直接对标 `Gated v5`
- 用 paired benchmark-only
- 先打动态主场景

推荐配置：

1. 模式
   - `Hybrid Full`
   - `Gated v5`
   - `paper_inspired_statcalib_v1`

2. 场景
   - `linear_ramp`
   - `periodic_drift`

3. repeats
   - `2`

4. seeds
   - 先用 `20260427 / 20260428`

理由是：

1. `linear_ramp / periodic_drift` 是当前最能体现“统计聚合是否有用”的场景；
2. 它们正好对应慢漂移与周期漂移；
3. 这两个场景也是之前 `Gated v5` 最早给出强正信号的地方。

### 7.3 第二轮 benchmark

如果第一轮结果方向正确，再补第二轮：

1. 加入 `static_bias_theta`
2. 加入 `step_sigma_theta`
3. 补 `seed=20260429`

也就是变成：

- `4 scenario × 3 seeds × repeats=2`

这时才适合判断：

1. 它是否只是动态场景特化；
2. 还是能在四场景下整体逼近或超过 `Gated v5`。

### 7.4 第三轮 benchmark

只有当第二轮也站得住，才建议再上“长一点但可控”的中间长度 paired benchmark，例如：

1. 使用当前 `p4_teacher_repr_mid.yaml` 或等价中间时长口径；
2. 只跑：
   - `Full`
   - `Gated v5`
   - `paper_inspired_statcalib_v1`

目标是验证：

- 这条分支的收益，是否在更长运行时长下仍能保住

### 7.5 推荐的 benchmark 顺序总结

建议顺序如下：

1. 第一轮：`2 seeds × 2 dynamic scenarios × repeats=2`
2. 第二轮：`3 seeds × 4 scenarios × repeats=2`
3. 第三轮：中间长度 paired benchmark-only

这样能兼顾：

1. 结果可比性
2. 长进程风险控制
3. 计算量可控

---

## 8. 推荐的具体落地顺序

如果下一步真的要实现，我建议按下面顺序推进：

1. 先扩 `runtime_dataset_builder.py`
   - 加 compact histogram / teacher summary 标量
   - 不改主干张量口径

2. 再扩 `tiny_cnn.py`
   - 新增 `stat_summary_branch`
   - 保留 `Gated v5` 原有 gated 路线

3. 再加 loss 配置项
   - `lambda_bnext`
   - `lambda_risk`

4. 先跑单 seed 离线训练与 test 指标
   - 只确认训练流程通、数值稳定、诊断字段完整

5. 再跑第一轮小规模 paired benchmark
   - `linear_ramp + periodic_drift`
   - `Full / Gated v5 / paper_inspired_statcalib_v1`

---

## 9. 最终建议

这条 paper-inspired 分支最合理的第一版，不应该是“另起一个大模型”，而应该是：

- `Gated v5 + compact statistical summaries + light in-model aggregation + small closed-loop-consistency loss`

它的核心目标不是追求一下子大幅刷新结果，而是回答：

1. `Gated v5` 还缺的，是不是“模型内统计聚合”这一层？
2. 当前离线训练与 formal HIL 的差距，能不能通过更下游一致的 loss 再缩小？

如果这条线有效，那么它会自然成为：

- `teacher-representation` 主线的下一代候选

如果这条线无效，也依然有价值，因为它能帮我们排除一种很合理、但未必必要的增强方向。
