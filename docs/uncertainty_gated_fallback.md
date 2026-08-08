# T5.4.2 uncertainty-gated fallback 验证

## 结论

在预注册的 matched syndrome-decision 对照中，观测量驱动的 uncertainty gate 相对“始终使用
frozen EWMA MAP”的 no-fallback 系统，降低了三个 OOD drift family 聚合后的 logical-class failure：
绝对下降 `0.00107490`，12 个全新 base-seed clusters 的 95% bootstrap CI 为
`[0.00001950, 0.00227615]`。该结论很弱且不普适：telegraph family 显著受益，compound
range-extrapolation family 显著受损，sinusoidal family 的 CI 跨 0；in-distribution nominal
negative control 也出现很小的点估计代价。因此本 task 只建立“当前 synthetic OOD mixture 下的
聚合 syndrome-decision 证据”，不建立 universal OOD safety、physical-memory LER 或 device fallback
结论。

## 冻结对照与因果口径

- primary / no-fallback：T5.1.2 frozen EWMA periodic MAP；
- fallback / last-known-good：T5.1.2 frozen static MAP；
- 所有策略消费同一 residual、同一 posterior 组和同一 hidden evaluator truth；
- 当前 window 的 score/action 在 predictor 用本窗 observation 更新前完成；
- hidden truth 只用于离线计分，不进入 score 或动作；
- catastrophic failure 定义为 decoded logical class 与同一样本 hidden evaluator class 不同，
  不是 uncertainty、OOD flag 或 health proxy。

观测分数为三个 `[0,1]` 分量的最大值：

\[
s=\max\left(1-\max_c p_{\rm EWMA}(c),\;
\frac{1}{\log 4}\operatorname{JS}(p_{\rm static},p_{\rm window},p_{\rm Kalman}),\;
\frac{1}{3}\sum_j\mathbf{1}[\hat c_j\ne \hat c_{\rm EWMA}]\right).
\]

当 `s >= threshold` 时使用 static MAP，否则使用 EWMA。由于 static 可对也可错，gate 既可能
避免 primary error，也可能诱发新 error，收益不是由定义强制得到。

## 严格 split 与阈值选择

阈值只在已完成 T5.4.1 的 8 个 development seed clusters 与三个 drift families 上选择；41 点网格
为 `0, 0.025, ..., 1.0`。目标是最大化按 base seed 聚合的 `primary_failure - gated_failure`，并依次
用更低 fallback rate、更低 induced-failure rate、更高阈值破同值。选中阈值为 `0.45`：

| development 指标 | 点估计 | 95% Student-t seed-cluster CI |
| --- | ---: | ---: |
| absolute catastrophic reduction | 0.00219218 | [0.00019906, 0.00418530] |
| fallback rate | 0.01632055 | [0.01437446, 0.01826663] |
| induced-failure rate | 0.00416056 | [0.00387997, 0.00444115] |

随后才生成 12 个 parent-disjoint confirmation clusters `202607154501--202607154512`。确认数据不参与
阈值或方法选择；CI 用 20,000 次 whole-base-seed cluster bootstrap。

## 独立 OOD 确认

三个 family 共 36 cells、`1,179,648` 个 matched decisions：

| 指标 | 计数 | rate / 点估计 | 95% cluster bootstrap CI |
| --- | ---: | ---: | ---: |
| primary failure | 55,666 | 0.04718865 | [0.04449628, 0.05002427] |
| gated failure | 54,398 | 0.04611376 | [0.04435641, 0.04778718] |
| always-static failure | 89,492 | 0.07586331 | [0.07475535, 0.07702978] |
| absolute reduction | 1,268 net | 0.00107490 | [0.00001950, 0.00227615] |
| fallback | 17,788 | 0.01507907 | [0.01385835, 0.01623792] |
| avoided failure | 6,170 | 0.00523037 | [0.00421651, 0.00630527] |
| induced failure | 4,902 | 0.00415548 | [0.00390286, 0.00439199] |
| unnecessary fallback | 7,093 | 0.00601281 | [0.00567034, 0.00636970] |
| selected without benefit | 11,618 | 0.00984870 | [0.00949521, 0.01024291] |

计数恒等式逐 window、逐 threshold、逐 cell 和 aggregate 重算：

\[
N_{\rm primary}-N_{\rm gated}=N_{\rm avoided}-N_{\rm induced}=1268.
\]

只有约 `33.68%` 的 fallback actions 实际避免了 primary failure；不必要和诱发代价没有被净收益隐藏。

## 场景异质性与负结果

| OOD family | primary | gated | absolute reduction | 95% cluster bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| joint sinusoidal rotation | 0.00699615 | 0.00695546 | 0.00004069 | [-0.00008138, 0.00015259] |
| stochastic telegraph | 0.03235881 | 0.02428945 | 0.00806936 | [0.00506585, 0.01131185] |
| compound range extrapolation | 0.10221100 | 0.10709635 | -0.00488536 | [-0.00557454, -0.00418854] |

聚合 PASS 主要由 telegraph lane 驱动；compound lane 的 gate 更频繁地把正确 EWMA 决策换成错误
static 决策。`scenario_universal_benefit=NOT_ESTABLISHED` 是正式 machine claim boundary，不可把聚合
结果写成每个 OOD family 均获益。

## Nominal negative control

12 个 nominal clusters、393,216 decisions 中，primary/gated failure 为 `440/445`，fallback `101`
次，其中避免 `19`、诱发 `24`、不必要 `57`。absolute reduction 为 `-0.00001272`，95% CI
`[-0.00003052, 0.00000254]`。CI 跨 0，但点估计方向和实际诱发计数表明 fallback 不是免费保护。

## 防简化实现与可追溯性

- 5 个 parent artifacts 和 5 个 implementation files 均 SHA-256 绑定；
- development/confirmation split、41 点阈值网格、选择目标和 resampling unit 均冻结；
- 24 个 calibration cells 的全部 41 阈值和 64 windows、36 个 OOD confirmation cells、12 个 nominal
  cells 均保存原始计数；
- 21 个 machine gates 会重算 parent freshness、truth isolation、trace uniqueness、所有局部计数/率、
  aggregate 和逐场景结论；
- 517-row Source Data 同时绑定 canonical row hash 与 CSV byte hash；
- 即使攻击者重算顶层 contract hash，阈值漂移、局部计数破坏、隐藏 compound 负结果或 truth 输入仍会
  fail closed；
- always-static 只作 ungated comparator；T4.1.4 的 all-fallback 负证据未被改名复用。

## Claim 边界

允许：在冻结 synthetic drift families 和 matched syndrome decisions 上，`threshold=0.45` 的 observed-only
EWMA-to-static gate 得到很小的聚合 OOD error reduction，同时具有明确的 fallback burden 和场景反转。

禁止：physical-memory LER、device catastrophic-failure rate、controller/RTL/board fallback、普适 OOD
safety、每场景均获益，或把本 task 与 T4.1.4 的 future-horizon controller score 当成同一 gate。

