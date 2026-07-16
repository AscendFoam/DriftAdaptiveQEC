# T3.2.10 PRL-inspired 指数递推 baseline

## 1. 结论

本任务实现了可解释因果递推

\[
\pi_{t+1}=a_m\pi_t+(1-a_m)\pi_m^\infty,
\qquad m\in\{g,e,\mathrm{leakage}\},
\]

而不是用隐藏大网络模拟递推。物理 lane 在 two-level differentiable sBs 模型内 exact 枚举
两个 full cycles 的 16 条 `g/e` 轨迹，训练 15 维 initial state、`g/e` saturation 与 decay，
共 75 个 trainable scalars；另存固定但未物理校准的 leakage saturation/decay，共 105 个
stored scalars。事件 lane 复用同一标量核，消费 observed X/Z `g/e/leakage` 与 health flags，
并使用真实 `ParamBank`。两条 lane 的 fidelity 和 abstract event cost 不合并排名。

## 2. 一手来源与实现合同

Puviani 等 Supplement 观察到 sBs 参数对连续 `g/e` outcome 呈不同 saturation level 和
decay rate；outcome 切换时以上一递推状态为新分支初值。当前实现直接冻结这一式，动作 at
half-cycle `j` 只读 `[0,j)` prefix。它是 paper-inspired repository-local optimization，
不是 Fig. S14 数值复现。

物理策略的 15 个 raw residual 分量分别拥有 `initial`, `g_inf`, `a_g`, `e_inf`, `a_e`。
所有学得 decay 落在 `0.517601--0.947588`，最大 raw stored value 为 `2.973718`，没有发散；
selected restart 为 seed 307，refinement 最后 25 点 gain 仅 `2.24e-6`。全部 3 个 phase-one
和 3 个 refinement run 均覆盖并改变全部 75 个参数。

## 3. Exact 物理 fidelity lane

| 策略 | cutoff 12 fidelity | cutoff 16 frozen fidelity | cutoff 12 `p(g)` |
| --- | ---: | ---: | ---: |
| standard | 0.396787 | 0.559221 | 0.780881 |
| exponential recurrence | 0.784921 | 0.773930 | 0.984661 |
| Q-fixed recurrence | 0.784921 | 0.773932 | 0.984662 |
| T3.2.9 lookup reference | 0.815799 | 0.638688 | 0.680229 |

cutoff 12 下 recurrence 比 standard 高 `0.388134`，比 lookup 低 `0.030878`；software
integer recurrence 的 fidelity 仅低 `7.62e-8`。cutoff 16 不重训，recurrence 比 standard
高 `0.214709`，却比旧 lookup 高 `0.135242`。后者不是“超越 oracle”：recurrence policy
可嵌入 lookup tree，但 T3.2.9 的非凸 multi-start 表只在 cutoff 12 优化，冻结迁移后不是
全局或跨 cutoff 上界。该反转是 optimizer/transfer counterevidence，必须保留。

recurrence 将 outcome distribution 推到 `p(g)≈0.98466`，与 lookup 的 `0.68023` 明显不同；
因此 fidelity gain 同时包含 measurement-branch reshaping，不能只解释为相同分支上的恢复改进。

## 4. Event-cost lane 与 run-length FSM

事件 lane 只回答 synthetic observed-event action 问题，不是物理 fidelity/LER。training 使用
3 seeds×4 scenarios×4,096 cycles；72 个 recurrence candidates 与 24 个 FSM candidates
只用 training truth 计算选择目标。evaluation 为独立 8 seeds×4 scenarios×12,000 cycles，
共 384,000 cycles。

| 控制器 | event+write cost | action accuracy | false intervention |
| --- | ---: | ---: | ---: |
| static safe | 0.604539 | 0.400034 | 0 |
| memoryless | 0.022917 | 0.955440 | 0.047494 |
| run-length FSM | 0.202829 | 0.793974 | 0.007888 |
| exponential recurrence | 0.073618 | 0.849286 | 0.315915 |
| truth evaluator lower reference | 0.000628 | 1 | 0 |

recurrence 相对 run-length cost 降低 `0.129211`，seed-cluster 95% CI
`[0.128225,0.130196]`；但仍比 memoryless 高 `0.050700`，且 false intervention 为
`31.59%`。所以结果支持“平滑记忆比连续 run threshold 更适合此 cost matrix”，不支持
“记忆总是优于 latest observation”。

selected scalar kernel 为 `a_g/a_e/a_leak=0.45/0.55/0.40`，recovery enter/exit
`0.45/0.25`，leakage enter/exit `0.60/0.15`。Q4.16 state、Q2.18 decay 的 384k-cycle
mode parity 为 `99.9716%`，109 次 mismatch、最大 state error `1.56e-5`，无 bank conflict。
这是 software integer evidence，不是 RTL、综合、Fmax 或板测。

## 5. 非 demo 审计

- 物理目标 exact sum 全 16 branches，保留 policy 对 branch probability 的梯度；
- 三 independent restarts 均 `300+250` epochs，不按 evaluation/cutoff16 选模；
- cutoff16、checkpoint reload、hash、probability normalization、trace/Hermiticity/PSD 全部门禁；
- closed-form repeated branch、outcome switch、suffix causality、全部 75 梯度、integer signed rounding
  与长 mixed trace 均有直接测试；
- event 比较使用 T3.2.5 同 trace、同 cost matrix、同 ParamBank write cost 与重新 training-only
  选择的 FSM，不把不同物理量拼成一张“总优越性”表；
- 1,888 行 Source Data 保存 optimization curve、terminal branches、event training grid 和 32 条
  evaluation traces。

## 6. Claim 边界

允许：finite-cutoff/two-level/short-horizon assumed-model 下的 causal interpretable recurrence，
cutoff transfer、software fixed-point mirror 与 synthetic paired event-cost comparison。

禁止：全局最优、跨 cutoff lookup 上界、paper-number reproduction、物理 leakage calibration、
event cost 等同 LER、pulse/multilevel/device 模型、RTL/综合/Fmax 或 target-board claim。

