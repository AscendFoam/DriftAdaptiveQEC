# T5.4.4 multi-agent / seed 选择偏差审计

## 结论

本任务对当前 learned evidence chain 的 6 个 selection episodes 做只读重构：T2.3.7 MF/NMF agents、
T4.1.1 slow-loop family/restarts、T4.1.5 legacy student、T4.4.1 fresh teacher restarts、T4.4.3
low-dimensional student candidates，以及 T4.4.4 frozen gain-retention。正式产物保留 255 个 evaluation
units、39 组 median/IQR/worst-quartile 分布和 420-row Source Data。

结论为 `PASS_WITH_WARNINGS`：当前 active selection 全部由 validation 决定，独立 test 没有改变选中的
agent/model；T2.3.7 与 T4.4.4 保留全部五个 agents。与此同时，teacher 的 validation-selected restart 0
在 primary/confirmation test 上均不是事后最优 restart；旧 T4.1.5 又未保留非选中 restart 的 test 指标。
这两点被作为负证据/缺口保留，禁止把 audit PASS 写成“没有 selection bias 风险”或 optimizer optimality。

## 冻结审计口径

- selection 只允许使用 training/validation；test 只用于独立评估和事后偏差诊断；
- 对越大越好的指标，worst quartile 取最低 `ceil(n/4)`；对越小越好的指标取最高
  `ceil(n/4)`；
- quartile 使用线性 `Q1/median/Q3`，`IQR=Q3-Q1`；
- agent、restart、family、dimension 与 evaluation seed 的注册单位全部进入 Source Data；
- “若看过 test 后选最优会得到多少”只作 hindsight diagnostic，不回写 parent selection；
- agent-seed 联合分布标为 descriptive，不能把同一 agent 内 seeds 当独立 agent 重复。

## Selection census

| lane | candidates / units | parent selection | test coverage | 审计结论 |
| --- | --- | --- | --- | --- |
| T2.3.7 NMF directional | 5 MF + 5 paired NMF | 每个 agent 内 validation 选 checkpoint；不选 agent | 5 pairs × 8 primary + 4 confirmation seeds | 全体报告，无 best-agent selection |
| T4.1.1 slow loop | 6 families；TCN/GRU 各 5 restarts | validation NLL | 6 families × 8 seeds | HMM validation/test 排名一致 |
| T4.1.5 legacy student | 3 restarts | validation MSE | 仅 selected restart | 非选中 test 缺失，降为 superseded warning |
| T4.4.1 fresh teacher | 3 fresh restarts | validation score | 3 × 8 primary + 4 confirmation seeds | test-best 为 restart 2，但仍冻结 restart 0 |
| T4.4.3 student | 1/2/4-state × 3 restarts | validation restart + 最小 eligible dimension | 三个 best-per-dimension 均评估 | 4-state restart 0 在 validation/test 一致 |
| T4.4.4 retention | standard、5 MF、teacher、recurrence、student | parent 已冻结，本 lane 不选模 | 9 strategies × 8 + 4 seeds | 全策略、全 MF agents、全 seeds 报告 |

## 全 agent 分布与 best-of-N 诊断

T2.3.7 的主 estimand 是每个 paired agent 的 NMF-minus-MF logical-Z effective lifetime。结果不是只报
最佳 agent：

| split | n | minimum | median | IQR | maximum | worst quartile |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cutoff12 primary | 5 | 0.022475 | 0.257219 | 0.242015 | 0.390225 | agent 3、agent 0 |
| cutoff16 confirmation | 5 | 0.281037 | 0.386384 | 0.175685 | 0.780587 | agent 3、agent 2 |

若违规地在 test 后只选 agent 4，primary/confirmation 相对全体 median 会分别夸大 `0.133006` 和
`0.394202` cycles。正式 claim 始终使用五 agent 分布；该 best-agent 数值只说明 best-of-N 包装会产生
多大乐观偏差。

T4.4.4 的独立 retention lane 也保留五个 MF agents。primary MF logical-Z lifetime 的
median/IQR 为 `6.790720/0.238626`，worst quartile 是 agent 307/401；confirmation 为
`7.510979/0.186115`，worst quartile 是 agent 401/307。没有用 test 重新挑 MF comparator。

## Validation 选择与 test hindsight

### Slow-loop family

六个 family 全部在八个 evaluation seeds 上报告。validation 选中的 Gaussian HMM 仍是 evaluation NLL
第一名，因此 test-best 与 parent selection 一致。其八 seed NLL median/IQR 为
`0.453580/0.078631`；worst quartile seeds 为 `20261238` 与 `20261236`。这不消除 T4.1.1 已登记的
detection-delay、synthetic-generator 和 richer-input 外推限制。

### Fresh teacher restart

validation score 选择 restart 0。三个 restart 的 primary test score median/IQR 为
`0.563544/0.004940`，confirmation 为 `0.597035/0.003310`。test hindsight 的 restart 2 分别达到
`0.567672` 和 `0.598480`，相对冻结 restart 0 的乐观差为 `0.004127` 和 `0.001445`。

这是本任务最重要的选择反例：若根据 test 改选 restart 2，数值会更好，但会破坏独立评估。因此正式
candidate 仍是 restart 0；两个 disagreement 均进入 machine-readable diagnostic，不能被“test 也更好”
改写成重新选模理由。

### Low-dimensional student

九个 validation candidates 全部保留；validation MSE median/IQR 为
`1.19819e-5/3.49569e-6`，worst quartile 正好是三个 1-state restarts。按 frozen tolerance 选择的
4-state restart 0 也是三个 best-per-dimension 中 evaluation MSE 最低者 (`6.08314e-6`)，所以 hindsight
optimism 为 0。这个一致性是诊断结果，不反向授权 test 参与 selection。

### Legacy T4.1.5

三个 restart 的 validation MSE 都进入分布，median/IQR 为 `1.32256e-6/4.77462e-10`；但 parent 只保存
selected restart 的 evaluation metrics，无法复算“test 后最优 legacy restart”。本任务没有补造缺失
counterfactual，而是将 T4.1.5 标成 superseded predecessor；当前 active student 由 T4.4.3 的九候选
validation-only 流程承接。

## 防简化实现与验证

- 7 个 parent artifacts、7 个 parent Source Data 与 7 个 implementation files 全部 SHA-256 绑定；
- parent implementation composite 逐项重算，T5.1.4 learned-decoder performance branch 仍为 revoked；
- 420-row Source Data 保存 candidate、agent、restart、seed、distribution、hindsight 与 missing-evidence rows；
- 39 个 distribution contract 均重算 count、Q1/median/Q3/IQR 和 direction-aware worst quartile；
- 23 个 machine gates 重算全部 selection、candidate/seed census、parent freshness 与 claim boundary；
- semantic mutation tests 在重算顶层 hash 后，仍拒绝删除弱 agent、test 后重选、隐藏 worst quartile、
  伪造 legacy evaluation completeness 或升级 device claim。

## Claim 边界

允许：validation-only selection、全 agent/restart/seed 分布、worst quartile、hindsight selection-bias
diagnostic 和明确的历史 coverage warning。

禁止：best-of-N test selection、selected-agent-only 主结果、optimizer/global optimum、universal memory benefit、
physical-memory LER、device calibration、RTL/FPGA/board 或实验 claim。
