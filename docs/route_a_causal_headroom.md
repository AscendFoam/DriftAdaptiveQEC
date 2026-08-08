# T6.10.1：Route-A causal headroom、expert disagreement 与 action-value audit

## 结论

T6.10.1 得到诚实的 `NO_GO_V5_INSUFFICIENT_ACTION_SPACE_HEADROOM`。旧五专家只改 router 的严格因果 held-out headroom 为 `-0.2322%`，低于 `10%` 入口门；posterior-mixture 相对既有 hard-action decision oracle 的纯动作空间增量只有 9 个错误，即相对最强 nested baseline 的 `0.02549%`，远低于 `12%`。因此不得进入当前 Phase 6B V5 实现/新 formal，也不得以增加 seed、放宽门槛或把 truth-privileged decision oracle 包装成 mixture gain 挽救。

## 证据边界

- T6.7.1/T6.7.2 的 1,464 条已打开 V4 formal trajectory 只作诊断；未用于 selector 拟合、超参数选择或 V5 确认性结论。
- selector 只在全新 `v5_headroom_development` 上拟合：31 个 off-grid cells × 6 个全新 seed clusters = 186 条 trajectory；旧 calibration/pilot/formal seed 与 development seed 零交集。
- 外层为 leave-one-seed-cluster-out；每个外层训练集内部再以 leave-one-training-seed-cluster-out 选择正则化 `C`。held-out seed 不参与 baseline、mixture 或 selector 选择。
- 在线特征只复用上一 1,024-decision Window sufficient state、当前五专家 causal state 和上一完整 2,000-decision activation period 的 hard-action disagreement counters；不读取 family、cell、truth、onset 或 future suffix，也不重复执行 MAP。
- family/cell/activation/decision oracle 和 expanded candidate-set oracle 均为 truth-privileged 诊断对象，不是可部署方法。

## V4 formal 独立复算

五种 deployable experts 为 standard binning、static joint MAP、Window MAP、EWMA adaptive MAP 和 Kalman adaptive MAP。每条 trajectory 都从相同 10-bit syndrome residual 重放 predict-then-update cadence，并逐项验证：

- observed input SHA-256；
- truth SHA-256；
- 每个方法的 96×4 Pauli window counts；
- 每个方法的完整 scored error-class trace SHA-256。

共 `1,464/1,464` 条 exact match，合计 71,958,528 scored decisions。全局最强单专家是 EWMA，LER 为 `0.01154953`。非因果诊断上界如下：

| 诊断对象 | LER | 相对 EWMA headroom |
| --- | ---: | ---: |
| family oracle | 0.01126382 | 2.4737% |
| cell oracle | 0.01106694 | 4.1784% |
| 2,000-decision activation oracle | 0.00962479 | 16.6650% |
| per-decision hard-expert oracle | 0.00813979 | 29.5228% |

这表明专家在短时间尺度确有互补，但 family/cell 级静态路由空间很小；较大的 oracle 数值依赖不可获得的未来 truth，不能直接转化为 causal router claim。

## 全新 development nested audit

全新 split 共评估 4,571,136 scored decisions 和 2,418 个实际 activation units。每个外层 fold 的 strongest baseline 都仅在训练 seeds 上选择，六个 fold 均选出 Window MAP。

| 对象 | errors | LER | 相对 strongest baseline |
| --- | ---: | ---: | ---: |
| nested strongest baseline | 35,314 | 0.00772543 | 0 |
| strict-causal nested selector | 35,396 | 0.00774337 | -0.2322% |
| activation-period oracle | 33,922 | 0.00742091 | 3.9413%（不可部署） |
| per-decision hard-expert oracle | 29,105 | 0.00636713 | 17.5823%（不可部署） |
| held-out fixed posterior mixture | 35,152 | 0.00768999 | 0.4587% |
| expanded candidate-set oracle | 29,096 | 0.00636516 | overall 17.6077%（不可部署） |

最重要的反简化检查是把 expanded oracle 再拆开：其 overall `17.6077%` 中，已有 hard experts 的 truth-privileged per-decision switching 已贡献几乎全部空间；36 个固定 one-hot/pair/uniform posterior-mixture candidates 只比 hard decision oracle 多避免 9 个错误。纯 action-space 增量为：

\[
\frac{29105-29096}{35314}=0.000254856=0.02549\%.
\]

因此以 overall expanded-oracle headroom 通过 `12%` 门属于错误归因；机器 gate 使用上述纯增量，而不是 overall truth-oracle 数值。

## regret 分解与不可辨识区

按同一 nested selector 到 expanded action oracle 的差距分解：

- selection regret LER：`3.22458e-4`；
- estimation/activation-granularity regret LER：`1.053786e-3`；
- incremental action-space regret LER：`1.968876e-6`；
- 三项之和：`1.378213e-3`，与总差精确闭合。

selection posterior 的 top probability `<0.55` 或 top-two margin `<0.10` 定义为 operational non-identifiable region。该区域覆盖约 39% decisions，说明现有观测在许多 activation boundary 无法可靠识别短时最优专家；但这不能通过 truth label 在线补足。

## 预算、因果和 fail-closed 检查

- 五专家 shadow update：7,385 MAC/update；selector：120 MAC/update；总计 7,505 `<8,192`。
- private state：2,608 B `<8,192`；workspace：2,048 B `<8,192`。
- selector 特征复用 Window sufficient state 和上一周期 hard actions，零重复 MAP decode；该账本是 host/NumPy derived matched-budget，不是 FPGA 实测资源。
- future-suffix、scenario metadata 和 truth/error-label mutation 后历史 selector output 不变。
- source/cache hash mutation fail closed。
- 五项 focused tests 全通过；artifact validator 对 Source Data、parent SHA、manifest、gate recomputation 和 analysis hash 全通过。

## 对后续路线的影响

当前 Phase 6B V5 的入口条件未满足。合理结论不是继续扩大 IMM/BOCPD/typed-expert 工程量，而是停止当前候选：

1. 只改现有专家 router 明确否决；
2. 当前简单 posterior convex mixture 也没有足够的独立动作空间；
3. 如果未来提出 V6，必须先从新的物理 likelihood、非凸/结构化 action、不同 observation 或更短而真实可执行的 activation mechanism 证明独立 development headroom，再建立全新 protocol；
4. 已打开 V4 formal 仍不得成为新方法的确认性证据；
5. Phase 6C 的异构 secondary comparison 可以在 Phase 6B NO-GO 收口后继续，但不得回写或挽救本主结论。

机器产物：`docs/t6_10_1_causal_headroom.json`；Source Data：`docs/t6_10_1_causal_headroom_source_data.csv`。
