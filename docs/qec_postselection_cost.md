# T5.3.4 真实纠错与 post-selection 成本账本

## 1. 结论

T5.3.4 已把在线 nominal-sBs channel、offline post-selection 诊断、software safety campaign 和未测量字段拆成
四条不可混排的成本证据。正式 artifact 含 6 条 terminal-cutoff online rows、8 条 post-selection target
rows、12 个显式 missing fields、94 行 Source Data 和 27/27 gates，状态为 `PASS`。

成本审计没有把 T5.3.3 的 wall-clock boundary 升级为 full-cost break-even：

- online QEC：300 μs 内 30 full cycles、60 half cycles、60 measurements、60 resets、540 active gates；
- finite-energy：`Delta=0.34`，按仓库约定等效 `6.360122 dB`；
- classical resource：fixed nominal controller 为 15 个 stored scalars、0 persistent state、0 online policy MAC；
- achieved channel：报告 leakage-inclusive `F_avg`、code-space survival 与 T5.3.3 boundary，LER 保持 null；
- latency/pulse energy：matched controller、target-board、physical frontend、pulse duration/energy 均未测，保持 null；
- post-selection：8/8 target 的 conditional error 更低，但单位 rejection penalty 下 8/8 total cost 都高于 raw；
- final verdict：full-cost operational boundary、paper-defined coherence gain、postselected break-even 均
  `NOT_ESTABLISHED`。

## 2. Online QEC 原生成本

T5.3.1 的每条 qec-on lane 已保存 `event_accounting`；本任务直接读取这些计数，并与 T5.1.5 standard
measurement-feedback 的每周期 `2 measurement + 2 reset + 18 active gates` 独立交叉核对。qec-off 的三类
操作计数均为 0。没有用 controller cutoff12/16 的性能替换 channel cutoff36/40 的 fidelity。

| cutoff | noise | cycles | measurement | reset | active gates | `F_avg` at 300 us | code survival | sustained / cumulative boundary (`us`) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 36 | high | 30 | 60 | 60 | 540 | 0.564008 | 0.651097 | 40 / 60 |
| 36 | medium | 30 | 60 | 60 | 540 | 0.707055 | 0.738387 | 40 / 90 |
| 36 | low | 30 | 60 | 60 | 540 | 0.736262 | 0.758199 | 60 / 110 |
| 40 | high | 30 | 60 | 60 | 540 | 0.566949 | 0.651477 | 40 / 60 |
| 40 | medium | 30 | 60 | 60 | 540 | 0.710529 | 0.739089 | 40 / 90 |
| 40 | low | 30 | 60 | 60 | 540 | 0.739583 | 0.758933 | 60 / 110 |

这里 `code survival` 是未归一化 CPTNI map 留在 finite-cutoff code projector 内的权重，不是 trajectory
acceptance。online rows 不使用 post-selection，所有输入均保留；因此 `postselection acceptance=1` 与
`code-space survival<1` 必须分列。

`540` 是 active gate applications 的软件协议计数，不是微波 pulse energy、pulse duration、FPGA 指令数或
board power。15 scalars/0 MAC 是 fixed nominal classical policy 的解析资源，不包含物理演化、矩阵仿真、
packing/routing/RTL 或 target-board latency。

## 3. Post-selection 诊断成本

T3.2.4 是独立的 synthetic wrapped-Gaussian decoder decision lane。threshold 只由 training split 校准；下表
的 CI 以 evaluation seed 为 cluster。conditional error 不能与 CPTNI `F_avg` 或 physical-memory LER 合并。

成本定义为：

```text
accepted failures per input = acceptance * conditional error
total cost(lambda) = accepted failures per input + lambda * rejection
```

| target | realized acceptance | rejection | raw error | conditional error | total cost at `lambda=1` | break-even rejection penalty |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.995 | 0.994845 | 0.005155 | 0.013785 | 0.011438 | 0.016534 | 0.466735 |
| 0.990 | 0.989833 | 0.010167 | 0.013785 | 0.009545 | 0.019615 | 0.426600 |
| 0.980 | 0.979901 | 0.020099 | 0.013785 | 0.006816 | 0.026778 | 0.353573 |
| 0.950 | 0.949589 | 0.050411 | 0.013785 | 0.002966 | 0.053227 | 0.217587 |
| 0.900 | 0.899108 | 0.100892 | 0.013785 | 0.001242 | 0.102009 | 0.125560 |
| 0.800 | 0.798551 | 0.201449 | 0.013785 | 0.000406 | 0.201773 | 0.066821 |
| 0.700 | 0.699035 | 0.300965 | 0.013785 | 0.000180 | 0.301091 | 0.045384 |
| 0.500 | 0.499534 | 0.500466 | 0.013785 | 0.000060 | 0.500496 | 0.027484 |

越严格的 post-selection 令 conditional error 越小，却拒绝更多输入。`lambda=1` 时所有 target 的 total cost
都比 raw error 差；这正是“条件结果看起来很好”不等于在线纠错或免费增益的反例。truth-only upper bound
只供 evaluator，未被升级为 deployable score。

## 4. 分开的 safety burden

T5.1.6 的定向 software campaign 有 767,872 cycles、11,552 fallback cycles、4 reset requests、0 observed
unsafe/undefined actions。它是 fault-coverage evidence，不是 iid device population；statistical upper bound 为
null。该 row 不与 300 μs channel 或 post-selection 决策成本相加。

## 5. 显式缺失字段

以下 12 项保持 null：matched controller latency；active pulse duration/energy；device-calibrated reset
fidelity/energy；matched physical-memory LER；best-passive physical reference；target-board core/transport/end-to-end
latency；quantum measurement、high-speed ADC、AWG/DAC output 和 physical action latency。

配置中的 `1/995 us`、host estimator timing 或别的 controller profile 均未转移填补这些字段。缺失字段使
`full_cost_complete=false`，但不妨碍“成本账本已完整地说明知道什么和不知道什么”这一 task-local PASS。

## 6. 非 demo 审计

- 六个 parent artifacts/implementations 与 T3.2.4 的 256-row raw source 均作 hash/live validation；
- online event counts 同时从 T5.3.1 native ledger 与 T5.1.5 protocol reference 复核；
- 所有 `F_avg`、survival 与 boundary 从 T5.3.2/T5.3.3 重算，不从 controller lifetime 拼接；
- 8 个 post-selection rows 全部重算 acceptance/rejection、accepted failures、4 档 penalty cost 与 break-even
  penalty，并保留 cluster CI；
- 20 类 semantic mutations 覆盖 parent hash、active/passive counts、squeezing、survival/acceptance 混名、
  LER/latency/pulse duration 填值、full-cost 升级、post-selection identity、truth 部署、safety 拼接、missing
  field 填零、global score 与 break-even claim。

## 7. Claim 边界

允许：分别报告 simulation-derived online event/resource counts、CPTNI fidelity/survival、offline
post-selection acceptance/rejection/penalty cost、software safety burden 和明确 null 字段。

禁止：conditional post-selection 作为在线增益、postselected/full-cost break-even、code survival 当 acceptance、
解析 MAC/scalar 当 RTL/board resource、配置 latency 当实测、跨 lane 总分、physical-memory LER、device reset
rate 或实验结果。

