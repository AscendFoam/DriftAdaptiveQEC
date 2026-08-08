# T5.4.3 因果消融与负结果表

## 结论

本任务完成了 history、CNN residual、regime state、run-length、parameter update 与 fallback 六项
`mechanism off` 对照，但没有把六项伪装成同一个“端到端系统”的开关。它们来自五条不同的原生证据
lane，metric 分别是逻辑寿命、残差参数 MSE、regime NLL、软件事件代价和 syndrome logical-class
failure；因此正式产物禁止跨 lane 总分或全局排名。

结果并非“六项全部有效”：history 在 cutoff 12/16 方向反转，run-length 在全部 32 个 matched cells
均劣于 memoryless；fallback 只在注册 OOD mixture 聚合后微弱为正，在 compound 场景显著为负；
regime state 改善 proper score 但增加检测延迟；CNN residual 只在一个 legacy held-out split 上改善参数
MSE，已撤销性能分支；parameter update 只建立软件事件代价的组件结果。

## 原生 lane 与 intervention

统一约定：对“越大越好”的指标使用 `active - off`，对“越小越好”的指标使用 `off - active`；正值
表示 active 较好。所有 active/off 对照只在同一原生 lane 内解释。

| 机制 | active | off intervention | 原生 metric | 结果 | claim 决定 |
| --- | --- | --- | --- | --- | --- |
| history | frozen full-history NMF | 同权重、同 trace，只保留 latest outcome | logical-Z effective lifetime | cutoff 12 正、cutoff 16 负 | memory mechanism 不成立 |
| CNN residual | legacy TinyCNN residual-b inference | residual 精确置 `[0,0]` | residual-b MSE | 参数 MSE 改善；无 seed-cluster CI | 只保留方法细节，移除性能 claim |
| regime state | causal four-state HMM | 同 emission、移除 temporal state | NLL | proper score 改善，检测更慢 | 只保留 estimator proper-score claim |
| run-length | 非退化 run-length FSM | same-trace memoryless event controller | event+write cost | active 显著更差 | 降级 run-length 性能 claim |
| parameter update | run-length decision 真实写 ParamBank | 保持 last-known-good normal image | event+write cost | active 组件代价较低 | 仅保留软件 actuation 组件结果 |
| fallback | uncertainty-gated EWMA→static | 始终用 frozen EWMA | logical-class failure | OOD aggregate 正、场景不普适 | 仅保留 mixture-qualified aggregate |

## 六项结果

### History

五个 paired agents 在 cutoff 12 的 full-history-minus-latest-only lifetime 为
`0.709110 [0.568837, 0.894025]` cycles；cutoff 16 则为
`-0.563636 [-0.665556, -0.461717]`。两处干预均实际改变动作，且 exact-budget retrained latest-only
capacity control 也跨 cutoff 反向。结论是当前 finite-cutoff、two-level、ten-cycle 证据不能支持稳健
memory mechanism，而不是“history 已被证明无用”这一更宽的命题。

### CNN residual

对保存的 206-sample legacy test split 重新加载 checkpoint 实际推理，并将 off residual 精确置零。
active/off MSE 为 `2.41445e-6 / 8.03405e-6`，差值 `5.61959e-6`；四个 legacy scenario 的点估计
均为正。该 split 没有独立 seed clusters 或正式 uncertainty interval，且 T5.1.4 已撤销 learned decoder
performance branch，所以这里只能说明旧 residual-parameter prediction 比零残差更准，不能升级为 LER、
控制增益或主算法成功。

### Regime state

八个 evaluation seeds 上，同一 fitted emission 的 HMM 相对 memoryless NLL benefit 为
`0.401514 [0.366352, 0.436676]`。与此同时 HMM-minus-memoryless detection-delay cost 为
`1.228754 [1.111948, 1.345561]` windows。两项必须同时报告：temporal state 改善 proper score，但
不是无代价的更快检测器，也没有建立 logical/control gain。

### Run-length 与 parameter update

四场景×八 seeds、384,000 cycles 的同 trace event ledger 中，run-length 相对 memoryless 的
`off - active` cost benefit 为 `-0.179911 [-0.180782, -0.179041]`；32/32 cells 均为负，正式保留为
失败结果。

在同一批 cells 上，禁用 normal parameter-bank update、保持 last-known-good image 后，active update 的
组件 cost benefit 为 `0.401710 [0.399014, 0.404407]`。该 off lane 仍保留 local health fallback，
因此不是“完全关闭安全机制”，也不是 decoder 或 physical-memory 增益；它只回答 parameter write
actuation 在当前软件事件目标中的作用。

### Fallback

T5.4.2 的 12 fresh confirmation clusters 上，OOD aggregate logical-class failure reduction 为
`0.00107490 [0.00001950, 0.00227615]`；但 compound range-extrapolation 为
`-0.00488536 [-0.00557454, -0.00418854]`，nominal 点估计也为 `-0.00001272`。实际有 6,170 次
avoided 和 4,902 次 induced failures，并有 7,093 次 unnecessary fallback。故只能保留当前注册 mixture
的聚合证据，不能写成普适安全机制。

## 防简化实现与可追溯性

- 五个 parent artifacts、四个 parent Source Data、六个 implementation files 与四个 legacy CNN assets
  均 SHA-256 绑定；parent implementation composite 也重新核对；
- history、regime、event/update、fallback 直接从已验证的 native same-trace campaigns 重算；CNN lane
  重新加载真实 checkpoint 和 `test.npz` 推理，不使用手填历史标量；
- 保存 206 个 CNN sample rows、10 个 history agent rows、8 个 regime seed rows、32 个 event cells、
  12 个 fallback seed rows与逐场景结果；
- 338-row Source Data 同时绑定 canonical row hash 与 CSV byte hash；
- 18 个 machine gates 会重算全部 lane、符号、代价、parent/asset freshness、nonmixing 与 claim boundary；
- mutation tests 在重算顶层 contract hash 后，仍会拒绝隐藏 history reversal、CNN 非零 off、regime delay、
  run-length harm、fallback harmful scenario、伪造跨 lane 总分和 device claim。

## Claim 边界

允许：逐 lane 写明 mechanism-off intervention、估计量、正负结果、代价和 claim 降级。

禁止：六机制端到端联合归因、跨 metric 排名、普遍机制收益、physical-memory LER、device calibration、
RTL/FPGA/board 或实验结论。
