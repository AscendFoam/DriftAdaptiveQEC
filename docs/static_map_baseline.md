# T3.1.2 training-average static MAP baseline

## 1. 结论

当前主要 decoder comparison 已把旧 `static_calibration_map` 替换为正式
`static_training_average_map`。它只在 evaluation 前由独立 training-state sequence 冻结一次
参数，evaluation 期间只读取 centered syndrome，不更新、不读取 truth/state/history。

8 个 evaluation seeds、576,000 个 paired samples 上，formal static MAP 均显著优于
standard binning；同时默认 trace 上预注册 EKF 不再显著优于 formal static。这是一次必要的
baseline-strengthening 结论：T1.3.4 的旧 72.1% gap-closure 只能保留为弱 calibration-static
anchor 下的历史结果，不能继续当作当前强 baseline 结论。

## 2. Training/evaluation contract

默认 training sequence 与 evaluation sequence 分离：

| 项目 | Training | Evaluation |
| --- | --- | --- |
| seed | `20260312` | `20260721`--`20260728` 八个独立 seeds |
| windows | 24 | 每 seed 24 |
| step index | 8 | 8 |
| 输入 | training `DriftState` mean + mixture covariance + weight | centered modular syndrome q/p |
| truth/state | 仅用于离线 training metadata 与 evaluator | decoder signature 中不存在 |
| update | fit 一次后冻结 | 无 |

training state hash 为
`239b57432a85586fd09babc18f942e14bdd4c6528c52d2fff0a871d423188031`；8 个
evaluation traces 各有独立 SHA-256。当前 training 不是装置 calibration，而是 synthetic
scenario distribution 的 state-metadata average；该边界由 R-N048 跟踪。

## 3. 拟合公式与冻结参数

对于 training state `j` 的均值 `mu_j`、同均值 core/outlier mixture covariance
`Sigma_j` 和归一化权重 `w_j`，使用边际矩匹配：

\[
\bar\mu=\sum_j w_j\mu_j,\qquad
\bar\Sigma=\sum_jw_j\left[\Sigma_j+
(\mu_j-\bar\mu)(\mu_j-\bar\mu)^\mathsf T\right].
\]

这包含 drift means 的 between-window covariance，不是只平均每窗 covariance 的简化实现。
冻结值为：

\[
\bar\mu=(0.3342171,-0.2506628)
=(0.133333,-0.100000)\lambda,
\]

\[
\bar\Sigma=
\begin{pmatrix}
0.31697375 & -0.00653556\\
-0.00653556 & 0.12274380
\end{pmatrix}.
\]

同均值 outlier mixture 使用 exact second-moment factor 后再做 Gaussian moment match；非零
`loss_gamma` 会显式拒绝，绝不静默折成 additive displacement variance。

## 4. 多 seed 结果

| Decoder | Aggregate LER |
| --- | ---: |
| Standard binning | 0.058870 |
| Static training-average MAP | 0.024498 |
| Full-state model oracle MAP | 0.011340 |

Standard minus static 的 paired difference 为 `0.034372`，95% CI
`[0.033798,0.034946]`；standard-only/static-only discordant failures 为
`24,468/4,670`。每个 seed 的 paired CI 下界都为正，不是单 seed 或 aggregate cancellation。

默认 T1.3.4 trace 的当前五行结果为：

| Decoder | LER |
| --- | ---: |
| Standard binning | 0.060417 |
| Static training-average MAP | 0.024792 |
| Window Variance + MAP | 0.022639 |
| EKF + MAP（历史预注册 primary） | 0.025319 |
| Full-state model oracle MAP | 0.011389 |

Formal static minus EKF 为 `-0.000528`，95% CI `[-0.001440,0.000385]`；因此 EKF
没有显著优于 formal static，`primary_alignment_gate_passed=False`。Window point estimate 略优于
static，但 T3.1.2 不提前替 T3.2 做完整 adaptive 多 seed 验收。

## 5. Comparison schema 与 provenance

当前 active T1.3.4 和 future T5 schemas 都显式选择
`static_anchor_method_id=static_training_average_map`，且恰好包含一次：

- `standard_binning`；
- `static_training_average_map`。

漏项、重复或把 sensitivity/legacy P4 冒充 decoder table 均 fail closed。production JSON 直接保存
descriptor、training/evaluation config、完整 frozen mean/covariance、training hash、per-seed Source
Data、聚合 paired CI、源码 hash 和 claim boundary。

产物：

- `docs/t3_1_2_static_map_validation.json`；
- `docs/t3_1_2_static_map_source_data.csv`。

T3.2.1 集成复核修正了旧 validator 的过宽假设：required standard-binning 不代表必须使用本任务
的 training-average static。memory comparison 合法地声明 `final_outcome_static_periodic_bayes`
为自己的 static anchor；本模块只验证明确选择 `STATIC_MAP_ID` 的 schemas。artifact schema 已升级为
`t3.1.2-static-map-v2`，gate 改名为 `formal_static_map_present_in_declared_schemas`。

## 6. 反简化验证

- 手算验证 law of total covariance，并证明结果不同于 naive covariance average；
- weighted fit、outlier mixture moment、training hash sensitivity 和 non-SPD 参数失败分支；
- loss 显式拒绝；非法 weights/shape/seed/split/chunk 均负测；
- chunked decoder 与 `map_decode_2d` reference bit-for-bit 一致；
- 两个 evaluation seeds 使用完全相同 frozen parameters/hash、不同 trace hash；
- 8 seeds × 72k、9/9 machine gates；focused+adjacent `119 passed`。

## 7. Claim 边界

允许：在注册的 synthetic Gaussian step distribution 上，evaluation-independent
training-state-moment-matched static periodic MAP 显著优于 standard binning。

禁止：称其为 universal/finite-energy/protocol-aware optimal decoder；称 training metadata 为真实
装置 calibration；声称处理 physical loss/leakage；外推 FPGA 或量子硬件结果。
