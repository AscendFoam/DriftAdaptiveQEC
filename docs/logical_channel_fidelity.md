# T5.3.2 logical-channel fidelity 与短时率

## 1. 结论

T5.3.2 从 T5.3.1 的六态、未归一化 code-space outputs 和完整 PTM 重新计算
leakage-inclusive `F_e`、`F_avg` 与短时有效退极化率；没有把单态保真度、conditional postselection 或
单指数拟合改名为平均通道指标。正式产物包含 24 条 matched finite-cutoff lanes、5,294 行 Source Data 和
23/23 个机器门，状态为 `PASS`。

但 `PASS` 只表示定义、来源、重算、数值敏感度和 claim 边界完整，不表示主动纠错已经通过 simulated
break-even。正式结果反而揭示一个必须保留的失败分支：cutoff 40 的 `qec_on` 在 30 cycles 末端有更高
`F_avg`，但最初几个 10 us 周期存在明显下降—回升瞬态，1/3/4-point 初始率估计互不一致。因此主动通道
只报告 raw `Gamma`，不授予短时寿命；T5.3.3 必须用完整 matched curve 与统一成本口径定义 operational
boundary，不能使用这个不合格的初始率倒数。

## 2. CPTNI fidelity 定义

T5.3.1 重构的是 code projector 内的 completely positive trace-nonincreasing map。令其 Pauli transfer
matrix 为 `R`，则相对于 identity target：

```text
F_e   = Tr(R) / 4
F_avg = (2 F_e + R_II) / 3
```

`R_II` 是六态平均 code survival。这里的 `F_avg` 是未归一化输出与目标态 overlap 的 Haar 平均；把
leakage 权重送入与 code 正交的 erasure flag 得到同一数值，因此该指标显式惩罚 leakage。

常见的 trace-preserving 公式 `(2 F_e + 1)/3` 在这里不适用，其高估量恰为
`(1-R_II)/3`。正式 cutoff 40 终点的最大高估为 `0.227211`。六个输入态的 direct mean overlap 与 PTM
公式逐点一致；先按各态 survival 归一化再平均的 conditional fidelity 仅作诊断，因为 state-dependent
survival 会使这种后处理不再代表同一个线性 channel。

## 3. cutoff 40 终点结果

| noise | mode | `F_e` | `F_avg` | survival | conditional diagnostic | TP 公式高估 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| high | qec_off | 0.135184 | 0.201892 | 0.335307 | 0.587593 | 0.221564 |
| high | qec_on | 0.524685 | 0.566949 | 0.651477 | 0.870253 | 0.116174 |
| medium | qec_off | 0.246495 | 0.270452 | 0.318366 | 0.847876 | 0.227211 |
| medium | qec_on | 0.696250 | 0.710529 | 0.739089 | 0.961359 | 0.086970 |
| low | qec_off | 0.295916 | 0.310279 | 0.339004 | 0.914780 | 0.220332 |
| low | qec_on | 0.729908 | 0.739583 | 0.758933 | 0.974504 | 0.080356 |

cutoff 40 的 `qec_on-qec_off` 末端 `F_avg` 差为 high `+0.365057`、medium `+0.440077`、low
`+0.429304`。这些只是固定 30-cycle horizon 的 raw matched differences；没有转换成 ratio、gain 或
break-even。cutoff 12 的相同差分别为 `-0.043415/-0.120614/-0.164587`，方向全部反转，证明低截断
结果不能被选择性丢弃。

## 4. 短时有效退极化率

沿用本地 beyond-break-even 论文的 leading-order channel-fidelity 口径，但不套单指数：

```text
Gamma = -2 dF_avg/dt |_(t=0)
```

primary estimator 是均匀 `10 us` grid 上的三点二阶 forward difference；一点评估和四点三阶 forward
difference 只用于离散化敏感度。只有前 3 点非增、`Gamma>0` 且三种估计的相对 spread 不超过 25% 时，
才报告 `1/Gamma` 为 qualified discrete short-time proxy lifetime。

| noise | mode | raw `Gamma` (`us^-1`) | estimator spread (`us^-1`) | 状态 | qualified lifetime (`us`) |
| --- | --- | ---: | ---: | --- | ---: |
| high | qec_off | 0.0306565 | 0.0034061 | reliable discrete proxy | 32.6195 |
| high | qec_on | 0.139511 | 0.116511 | unreliable cycle-scale transient | null |
| medium | qec_off | 0.0155676 | 0.0009285 | reliable discrete proxy | 64.2360 |
| medium | qec_on | 0.132832 | 0.118747 | unreliable cycle-scale transient | null |
| low | qec_off | 0.0125313 | 0.0006085 | reliable discrete proxy | 79.8002 |
| low | qec_on | 0.131386 | 0.119256 | unreliable cycle-scale transient | null |

cutoff 40 的主动 lane 在 high/medium/low 下相对 spread 为 `0.835/0.894/0.908`。其 raw
`qec_on-qec_off Gamma` 为 `+0.108854/+0.117265/+0.118855 us^-1`，与 30-cycle 末端 `F_avg` 改善方向
相反。这不是矛盾：前者描述粗粒度初始瞬态，后者描述固定长 horizon 的累计 channel。任何只选择其中
一个时间尺度的“纠错改善”叙述都不完整。

## 5. 不确定度与数值稳定性

本任务的 channel 是确定性 exact finite-matrix propagation，没有独立随机 seed clusters。因此 statistical
standard error 和 confidence interval 均保持 `null`，不伪造 sampling CI。报告两类系统敏感度：

- cutoff 36/40 deterministic terminal interval；它不是置信区间，也不是 infinite-cutoff extrapolation；
- 1/3/4-point rate envelope；它是时间离散化敏感度，不是统计误差条。

36→40 六条 matched lanes 的最大 absolute spread 为：终点 `F_avg 0.003475`、`F_e 0.004861`、survival
`0.000734`、primary `Gamma 0.000222 us^-1`、rate-envelope width `0.000354 us^-1`。这些数值说明 formal
terminal pair 在当前 finite-cutoff model 内稳定，但不能消除 code-basis、nominal fixed-control、analytic
noise、10 us sampling 和 30-cycle horizon 的模型不确定性。

## 6. 可追溯性与非 demo 审计

- parent artifact、parent implementation、当前 fidelity implementation 与本地论文 source 均作 SHA/fragment
  绑定；
- 每个 cycle 从 T5.3.1 六个 raw `2x2` outputs 重新计算 direct overlaps、survival、PTM `F_e/F_avg`；
- validator 重建全部 24 lanes、30 terminal intervals、3 matched differences 和 cutoff-direction audit；
- semantic mutations 覆盖 parent/paper hash、TP 公式、conditional 指标升级、伪统计 CI、单指数替代、
  主动寿命强行合格、删除 cutoff reversal、发明 gain 与 terminal table 篡改；
- formal Source Data 为 5,294 行，保留 cycle metrics、rate components、cutoff intervals、matched differences
  和 claim-boundary records。

## 7. Claim 边界

允许表述为：finite-cutoff matched CPTNI code-subchannel 的 leakage-inclusive `F_e/F_avg`、固定 horizon
raw differences、qualified passive short-time proxy 以及 deterministic cutoff/time-grid sensitivity。

禁止表述为：experimental process tomography、multilevel/device-calibrated leakage、无限维收敛、主动短时
lifetime、single-exponential decay、physical-memory LER、simulated/experimental break-even、universal QEC gain、
真实 FPGA/QPU 结果。T5.3.3 只能在同一模型和同一成本口径下建立 operational boundary。

