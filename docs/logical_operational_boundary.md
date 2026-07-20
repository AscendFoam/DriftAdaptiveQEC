# T5.3.3 simulated operational boundary

## 1. 结论

T5.3.3 在 T5.3.1/T5.3.2 的同一 finite-cutoff CPTNI channel 上建立了一个严格限定的
`simulation-derived wall-clock operational boundary`。cutoff 36 与 40 的三种噪声 profile 都满足：

- 主动通道在经历早期劣势后，从某个 `10 us` 采样点起对 matched idle 的 leakage-inclusive `F_avg`
  持续非劣，直到 `300 us` horizon 结束；
- 主动通道随后偿还早期累计 fidelity deficit，使 `integral(F_on-F_off) dt` 从某个采样点起持续非负；
- 36/40 两个 terminal cutoffs 的上述边界完全重复，采样边界 spread 为 0。

正式 artifact 为 `PASS`：12 个 cutoff×noise comparisons、416 行 Source Data、25/25 gates。这个 PASS
不等于论文意义的 coherence gain 或 beyond break-even。更强的三个结论继续为 `NOT_ESTABLISHED`：

1. paper-defined simulation-derived coherence gain：主动短时率不合格，且 qec-off 不是最佳被动物理编码；
2. full-cost operational boundary：active pulse/reset/classical/latency 成本待 T5.3.4；
3. experimental/physical-memory break-even：当前没有量子硬件或 best-passive physical reference。

## 2. 为什么不用终点或单一寿命

cutoff 40 下，主动通道第一周期的 `F_avg` 比 passive 低 `0.269--0.307`，并有下降—回升瞬态；只看
30-cycle 终点会隐藏启动代价，只取 T5.3.2 的 `1/Gamma` 又会把不稳定初始有限差分伪装成寿命。因此本任务
使用完整 31 点曲线，冻结两个互补边界：

```text
sustained boundary:
  最后一个 F_on - F_off < 0 的样本之后的首个样本，且此后全部样本非负

cumulative payback boundary:
  最后一个 integral_0^t(F_on-F_off) dt < 0 的样本之后的首个样本，
  且此后累计优势全部非负
```

积分使用原始 `10 us` grid 上的 trapezoidal difference，报告的是有单位的 fidelity·μs 差，不生成 area
ratio。相邻样本间线性零点仅作诊断，不能替代采样边界或宣称 sub-grid 精度。

## 3. 正式边界结果

| cutoff | noise | sustained boundary (`us`) | cumulative payback (`us`) | final `F_avg` advantage | terminal integrated advantage (`F_avg*us`) | qualified |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 12 | high | null | null | -0.043415 | -46.959 | no |
| 12 | medium | null | null | -0.120614 | -85.428 | no |
| 12 | low | null | null | -0.164587 | -99.467 | no |
| 24 | high | 40 | 70 | +0.298723 | +66.053 | yes |
| 24 | medium | 60 | 110 | +0.358693 | +57.171 | yes |
| 24 | low | 60 | 130 | +0.347684 | +49.419 | yes |
| 36 | high | 40 | 60 | +0.362101 | +79.250 | yes |
| 36 | medium | 40 | 90 | +0.436579 | +72.508 | yes |
| 36 | low | 60 | 110 | +0.425957 | +64.785 | yes |
| 40 | high | 40 | 60 | +0.365057 | +79.783 | yes |
| 40 | medium | 40 | 90 | +0.440077 | +73.114 | yes |
| 40 | low | 60 | 110 | +0.429304 | +65.370 | yes |

cutoff 40 的 sample-to-sample sign reversals 为 high `3`、medium `1`、low `1`，均被保留。cutoff 12 的
三条曲线在整个 horizon 内没有边界，证明“只报告高 cutoff 有利方向”会构成选择偏差。正式 verdict 只依赖
36/40 terminal repeat，不把 cutoff 24 当作收敛证明，也不删除 cutoff 12 反例。

## 4. baseline 与成本口径

active/passive 每一对共享 code basis、cutoff、noise profile、初态、`10 us` cycle、31 点时间网格和
`300 us` horizon，唯一 channel intervention 是 fixed nominal sBs 对 matched idle。因而当前可比较的是同一
encoded state 的 wall-clock channel performance。

但 qec-off 是 `matched_uncorrected_grid_code`，不是论文中的 best passive physical qubit，例如
`{|0>,|1>}` encoding。与此同时，active sBs 的 pulse、measurement、reset、squeezing、classical resource 和
latency 尚未在本 artifact 内统一定价。因此：

- `wall_clock_matched = true`；
- `full_cost_matched = false`；
- `coherence_gain_qualified = false`；
- `coherence_gain_value = null`。

论文以 `G=Gamma_best-passive/Gamma_active` 定义 break-even；本任务不能用 matched idle、终点优势或累计面积
偷偷替换该定义。

## 5. 数值敏感度与非 demo 审计

- 36/40 的 sustained/cumulative sampled boundaries 在三种 noise 下均完全一致；这只是 deterministic
  terminal-cutoff repeat，不是统计 CI 或 infinite-cutoff theorem；
- exact curves 没有随机 seed population，statistical SE/CI 保持 null；
- validator 从 T5.3.2 全部 31 点 `F_avg` 重算 12 个 comparisons、cumulative curve、terminal table 与 verdict；
- 17 类 semantic mutations 覆盖 parent/paper hash、baseline 冒充、裁剪曲线、移动两个边界、隐藏初始损失、
  cutoff12 强制通过、terminal 表篡改、主动 rate 强制合格、伪 full cost、ratio、指数拟合、coherence gain、
  experimental break-even 与 fake CI；
- 12 个 core edge/failure tests 另覆盖多次反转、累计偿还、无边界、uniform grid、范围和对齐错误。

## 6. Claim 边界

允许：在当前 finite-cutoff nominal-sBs vs matched encoded-idle model、相同 wall-clock grid 和 300 μs horizon
内，报告完整曲线的 simulation-derived wall-clock operational boundary。

禁止：paper-defined coherence gain、terminal-only/single-point break-even、area ratio、单指数寿命、full-cost
boundary、best-passive physical benchmark、physical-memory/experimental beyond break-even、device/QPU/FPGA
性能。T5.3.4 必须显式核算 active 与 post-selection 成本后再决定 full-cost verdict。

