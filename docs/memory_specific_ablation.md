# T3.2.11 memory-specific 消融

## 1. 结论

本任务没有得到跨 cutoff 稳健的“full history 机制增益”。预注册规则要求 full-history NMF
相对四个对照——同预算重训 latest-only、冻结 parent latest-only、三组 history shuffle 均值、
periodic reset R2——在 logical-Z effective lifetime 上的 paired 95% CI 下界，必须在 cutoff
12 和 16 同时大于零。cutoff 12 与 cutoff 16 均未通过，最终结论为
`cross_cutoff_memory_mechanism_not_supported`。

这不等于“memory 永远无用”。它只否定当前 finite-cutoff、two-level、10-cycle、五 agent、
当前训练 protocol 下的稳健机制 claim。

## 2. 消融合同

所有 frozen intervention 复用 T2.3.7 的五个 `GRU10-DENSE256-DENSE256-OUT15` parent
checkpoint，权重 bit-identical：

- history shuffle：为每个绝对历史位置生成固定随机优先级，只排序已观测 prefix；保留 token
  multiset、破坏顺序、相对 shuffle 顺序随 prefix 增长保持一致，且不读 future；
- history truncation：从零 hidden state 只重放最后 `L=1,2,4,8` 个 observed tokens；
- periodic hidden reset：按固定 half-cycle block reset，报告 `R=1,2,4,8`；R1 等价冻结
  latest-only，但 R>1 与 sliding truncation 不同；
- last-outcome-only：同时报告 frozen parent GRU latest-token view，以及 T3.2.7 已在相同
  train/validation split 独立训练的 72,853-param/72,266-MAC stateless FNN。

三组 shuffle seed 不当作 15 个独立 agent；先在每个 parent agent 内求 replicate mean，再对
五个训练 seed 做 paired bootstrap。

## 3. 主要数值

### cutoff 12

| 变体 | logical-Z lifetime |
| --- | ---: |
| full history | 6.740785 |
| retrained exact-budget latest-only | 6.888249 |
| frozen parent latest-only | 6.031675 |
| truncation L2 / L4 / L8 | 6.433236 / 6.685419 / 6.736147 |
| reset R2 / R4 / R8 | 6.240261 / 6.445210 / 6.624258 |
| prefix-consistent shuffle mean | 6.672168 |

在 cutoff 12，增加可用 history 通常改善冻结 parent；full 相对 frozen latest-only 为
`+0.709110 [0.568837,0.894025]`，相对 shuffle 为
`+0.068617 [0.050734,0.084185]`。但同预算重训 latest-only 比 full 高 `0.147464`，full-minus-
latest CI 为 `[-0.386866,0.147532]`，不支持 capacity-controlled memory gain。

### cutoff 16 frozen confirmation

| 变体 | logical-Z lifetime |
| --- | ---: |
| full history | 7.708351 |
| retrained exact-budget latest-only | 7.168269 |
| frozen parent latest-only | 8.271987 |
| truncation L2 / L4 / L8 | 8.082278 / 7.841547 / 7.719854 |
| reset R2 / R4 / R8 | 8.254680 / 8.096502 / 7.840701 |
| prefix-consistent shuffle mean | 7.661793 |

cutoff 16 的方向相反：full 显著高于 independently retrained latest-only
`+0.540082 [0.231972,0.785521]`，却显著低于 frozen parent latest-only
`-0.563636 [-0.665556,-0.461717]` 和 reset R2
`-0.546329 [-0.679249,-0.399115]`。shuffle 差 `+0.046558` 的 CI
`[-0.014033,0.107150]` 跨零。truncation/reset 越短反而越好，构成清晰 counterevidence。

## 4. 非 demo 审计

- 5 个 parent 与 5 个 independently retrained comparator checkpoint 均 live hash 验证；
- full-view 对 parent 128×20-history action bit-exact；10 类 intervention 对每个 agent 都改变动作；
- parent 权重在全部物理重放前后 hash 不变；
- 每个变体重新闭环推进 trajectory，未把同一 frozen outcome 序列当成全部 policy 的结果；
- primary 8 held-out seeds×64 trajectories，confirmation 4 seeds×32 trajectories；所有 density
  trace/Hermiticity/PSD gate 通过；
- 28,230 行 Source Data 保存每 agent/seed/cycle 曲线、辅助指标和 action-distance audit；
- 15/15 machine gates 通过，方向不作为 PASS gate。

## 5. Claim 边界

允许：报告当前 registered model 中的 cutoff-dependent signed memory interventions，并写“缺少
跨 cutoff 稳健机制证据”。

禁止：普遍声称 long memory 有益或无益、复现论文 1000-cycle memory mechanism、物理
multilevel leakage robustness、optimizer optimality、device/RTL/FPGA/board 结果。

