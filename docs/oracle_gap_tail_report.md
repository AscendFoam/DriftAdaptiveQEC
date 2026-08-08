# T5.1.3 average、tail 与双 oracle-gap 报告

## 结论

T5.1.3 已完成 15/15 个 reporting gates。runner 按 T5.1.2 的 RNG 顺序重新执行 36 个 scenario-seed
clusters，保存 1,152 条逐 window records；全部 trace hash 与 seed aggregate 均逐位复现 T5.1.2。每个场景
以 6 个 evaluation seeds 为独立 cluster，执行 20,000 次 paired bootstrap，报告平均 `P_L`、window LER
的 95th percentile、observed worst window 和 decoder-oracle gap。

24 个预注册比较使用 seed-level two-sided exact sign-flip test，并在完整 family 上做 Holm-Bonferroni 校正。
校正后正式发现为 **0 个**：最小 raw `p=0.03125`，最小 Holm-adjusted `p=0.75`。因此本 task 的 `PASS`
只表示报告协议、tail 数据、双 oracle、bootstrap 和多重比较完整，**不是 T5.1.4 的算法成功判定**。

机器产物：

- `docs/t5_1_3_oracle_gap_tail_report.json`；
- `docs/t5_1_3_oracle_gap_tail_source_data.csv`，7,139 rows；
- decoder lane 与 exact two-cycle control-oracle lane 严格分开，不做跨 lane 排名。

## 统计协议

### 独立单位与 bootstrap

- 同一 scenario-seed-window 内，standard、static、latest-window、EWMA、Kalman 与 decoder oracle 消费同一
  displacement/residual/truth trace；
- 512 个 window 内样本只用于估计该 window 的 LER，不能被当作 512 个独立部署环境；
- bootstrap 每次重采样 6 条完整 seed trajectories，再重算平均 `P_L` 和 192 windows 的 empirical p95；
- global maximum 只报告 observed worst。另对“每 seed 的 worst”求均值及 cluster-bootstrap CI，不对 192 个
  相关 windows 使用伪 iid maximum CI；
- oracle-gap ratio 仅在 bootstrap 中 `static - oracle > 0` 时定义；六场景、六方法的 valid fraction 均为 1.0。

### 多重比较

family 在看 evaluation 结果前由 schema 固定为：4 个可部署 challenger（standard、latest-window、EWMA、
Kalman）相对 static × 6 个场景，共 24 个 hypotheses。每项以 6 个 paired seed effect 做全部 `2^6=64`
sign flips，双侧检验后统一 Holm-Bonferroni 控制 family-wise error rate `0.05`。Oracle 不参与显著性排名，
只作为 nondeployable denominator/reference。

样本量只有 6 个独立 seeds 时，双侧 exact test 的最小 raw p-value 是 `2/64=0.03125`；在 24 项 Holm
family 中单靠全同号不足以显著。这一 power limitation 是正式结果的一部分，不能改用 1,152 个 windows
做伪重复来获得更小 p-value。

## Average 与 tail

下表每格为 `平均 P_L / p95 window LER / observed worst window LER`。

| 场景 | Static | Latest window | EWMA | Kalman | Oracle |
| --- | --- | --- | --- | --- | --- |
| static Gaussian | .001750 / .003906 / .009766 | .001231 / .003906 / .005859 | .001231 / .003906 / .005859 | .001190 / .003906 / .005859 | .001149 / .003906 / .005859 |
| mean drift | .006846 / .020410 / .029297 | .001017 / .003906 / .005859 | .001027 / .003906 / .005859 | .000977 / .003906 / .005859 | .000956 / .003906 / .007812 |
| variance drift | .009054 / .029297 / .041016 | .007253 / .023438 / .039062 | .007090 / .022363 / .037109 | .007202 / .022363 / .039062 | .006805 / .021484 / .037109 |
| correlation drift | .006826 / .015625 / .017578 | .003113 / .009766 / .017578 | .003082 / .009766 / .017578 | .002909 / .007812 / .017578 | .002797 / .007812 / .015625 |
| burst/outlier | .019246 / .076172 / .091797 | .018728 / .074219 / .091797 | .018748 / .074219 / .091797 | .018707 / .074219 / .091797 | .018575 / .073145 / .087891 |
| calibration shift | .021637 / .054688 / .072266 | .006114 / .014551 / .107422 | .006012 / .012598 / .107422 | .005941 / .013672 / .107422 | .003977 / .010645 / .019531 |

最重要的 tail counterevidence 出现在 calibration shift：三种 adaptive 方法显著降低 average 和 p95，但
observed worst 都是 `55/512=0.107422`，高于 static 的 `37/512=0.072266`。这是 abrupt shift 后的一窗
causal transient，不能被平均值覆盖。burst/outlier 中 adaptive 对 worst 也没有改善。T5.1.4 必须同时看
static nondegradation、average/p95 与 transient worst，不能只挑平均优势。

## Decoder-oracle gap

定义：

`static-oracle gap = P_L(static) - P_L(oracle)`；

`gap closed(method) = [P_L(static) - P_L(method)] / [P_L(static) - P_L(oracle)]`。

ratio 保留 signed 与超出 `[0,1]` 的结果，不裁剪。下表给出 Kalman point 与 95% paired-seed bootstrap CI：

| 场景 | static-oracle gap | Kalman gap closed | 95% CI |
| --- | ---: | ---: | ---: |
| static Gaussian | 0.000600 | 0.932 | [0.745, 1.140] |
| mean drift | 0.005890 | 0.997 | [0.979, 1.017] |
| variance drift | 0.002248 | 0.824 | [0.686, 0.988] |
| correlation drift | 0.004028 | 0.972 | [0.911, 1.028] |
| burst/outlier | 0.000671 | 0.803 | [0.513, 1.351] |
| calibration shift | 0.017660 | 0.889 | [0.863, 0.913] |

point ratios 较高，但 static/burst 的 denominator 小而 CI 宽，且 24 项 Holm family 无显著发现。它们可作为
effect-size/reporting evidence，不能绕过 T5.1.4 的成功/证否门。

## 多重比较完整结果摘要

- latest-window/EWMA/Kalman 相对 static 在 6 个场景的 paired bootstrap effect CI 均为正；
- 每项 exact sign-flip raw p 均为 `0.03125`；
- Holm-adjusted p 均为 `0.75`，所以 18 项均不拒绝零效应；
- standard 相对 static 的方向随场景变化，其中 correlation/calibration 明确更差；完整 family 仍无 adjusted
  discovery；
- 正 bootstrap CI 与 Holm exact-test 的结论不同，是因为前者估计在已观察 seed population 上的 effect
  uncertainty，后者又承担 24 项 family-wise correction 与仅 6 cluster 的离散 power。两者必须同时报告。

## Short-horizon control-oracle gap

control lane 从 T4.4.4 hash-bound artifact 读取 cutoff 12/16 的全部 16 terminal branches，horizon 固定为
exact two-cycle。每个策略的 expected metric 已由 branch probabilities 精确求和，因此这里不伪造 bootstrap CI：
把 16 个 policy-dependent branches 当独立样本，或把 optimization restarts 当实验 seeds，
都会产生错误 uncertainty。

gap 定义为 `control reference - method`，指标越高越好。该“control oracle”只是注册的 finite multistart、
bounded causal lookup reference，不是 globally certified optimum，也不能外推 10 cycles。

| cutoff | 方法 | selection-score gap | terminal-fidelity gap | fidelity-lifetime gap | logical-Z-lifetime gap |
| ---: | --- | ---: | ---: | ---: | ---: |
| 12 | standard | +0.039211 | +0.419012 | +0.229432 | +0.214614 |
| 12 | teacher | -0.077327 | +0.219799 | -0.595095 | -0.574475 |
| 12 | handcrafted | -0.083251 | +0.030878 | -0.620718 | -0.659905 |
| 12 | student | -0.076008 | +0.220408 | -0.582678 | -0.562522 |
| 16 | standard | -0.082202 | +0.079467 | -0.522636 | -0.523259 |
| 16 | teacher | -0.163540 | -0.018511 | -1.345844 | -1.313291 |
| 16 | handcrafted | -0.112389 | -0.135242 | -0.756241 | -0.798000 |
| 16 | student | -0.162027 | -0.016654 | -1.326456 | -1.294400 |

负 gap 必须保留。其原因不是“oracle 定理被违反”，而是该 reference 只优化有限 ansatz/目标，cutoff 16 又是
frozen transfer；不同 metric 也不是它的统一优化目标。例如 cutoff 12 reference 的 terminal fidelity 高于
handcrafted，但 selection score 与 lifetime 更低；cutoff 16 的 handcrafted terminal fidelity 也超过 frozen
reference。故后续只能称 `finite-horizon control reference gap`，不能称普适上界。

## 非 demo 与 claim 边界

本任务补回 T5.1.2 缺失的逐 window 粒度，并用 trace hash 和六种 seed rates 双重验证重放完全一致；没有从
seed 平均值推算 tail。7,139-row Source Data 保存 6,912 条 method-window rows、108 条 summary、24 条
multiplicity、80 条 control gaps 和 15 条 gates。validator 会拒绝 window 缺失、trace mismatch、ratio 分母
不可靠、未校正多重比较、control horizon 扩到 10 cycles、stale artifact 或 fake CI。

允许结论：冻结的 syndrome-level scenarios 上，可报告 average/p95/worst、paired decoder-oracle effect size
和 uncertainty；独立 exact two-cycle lane 上可报告 matched-model control-reference gap。禁止把 windows 当独立
seeds、把 negative gap 删除、把 exact expectation 包装成 sampling CI、跨 decoder/control lanes 排名，或把
本报告写成 finite-energy/device 性能与 T5.1.4 成功结论。
