# T3.2.6 HMM / regime posterior baseline

## 结论

本任务实现了一个四态 host regime estimator：`normal / burst / leakage / calibration_shift`。
它使用 supervised Gaussian emissions 和在线 causal forward recursion，逐 32-cycle observed window 输出
归一化 posterior；不使用 Viterbi、backward smoothing、future window 或 evaluation truth。

在 8 条独立 evaluation trajectories、4,096 windows / 131,072 cycles 上，causal HMM accuracy 为
`0.846191`，同一 emission/同一 raw input 的 memoryless classifier 为 `0.660889`。HMM 明显降低 NLL、
Brier 和 false switches，但平均 transition detection delay 从 `0.574` 增至 `1.802` windows。结论仅适用
于登记的 synthetic semi-Markov regime process，不能写成装置状态识别或 logical/control gain。

## 输入、输出与 truth 隔离

每个非重叠 window 含 32 cycles、8 个 raw observed fields：

1. q/p residual；
2. observed X/Z `e` indicators；
3. any-leakage indicator；
4. quadrature phase bit；
5. valid 与 deadline flags。

固定 featurizer 将其变成 14 个 summary：均值、方差、协方差、绝对残差、tail fraction、X/Z e-rate、
leakage rate、health rates 和 phase-selected mean。在线 `RegimeObservationWindow` 结构没有 regime、label、
hidden state 或 truth 字段。

hidden label 只在 training/validation 中用于 supervised emission/transition fit，在 evaluation 中只进入
metric evaluator 和单独 truth SHA256。每条 trajectory 同时保存 deployable-trace hash 与 truth hash，
二者不共用 channel。

## 模型与 comparator

- emission：14 维 standardized full-covariance Gaussian，每态独立 mean/covariance；
- transition：四态 row-stochastic matrix，Dirichlet smoothing；
- online filter：上一步 posterior → transition prediction → 当前 emission → normalized posterior；
- calibration：只在 validation seeds 上选择 posterior temperature；calibrated output 不反馈改变内部
  uncalibrated forward state；
- `static_prior`：训练 occupancy prior；
- `memoryless_emission`：与 HMM 使用完全相同的 selected Gaussian emissions、raw window 和独立
  validation temperature，但不使用 transition/history；
- `causal_hmm`：唯一差别是 causal temporal prior，因此可隔离 history 的贡献。

## 数据与选择协议

- training / validation / evaluation seeds：`3 / 3 / 8`，两两不重叠；
- 每条 trajectory：512 windows ×32 cycles；完整 simulator workload 为 229,376 cycles；
- hyperparameter grid：6 个 covariance regularization ×9 个 transition smoothing =54 组；每组扫描
  10 个 temperature；
- 选择只看 validation NLL，evaluation truth 不参与；最终选择：covariance regularization `1.0`、
  transition smoothing `10.0`、HMM temperature `0.8`、memoryless temperature `0.6`；四项均不在搜索边界；
- evaluation class counts：normal `927`、burst `935`、leakage `1171`、calibration shift `1063`。

## 生产结果

| Estimator | Accuracy | Macro-F1 | NLL | Brier | ECE | Transition delay (window) | False switch rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static prior | 0.228271 | 0.092491 | 1.396805 | 0.754991 | 0.050703 | 11.998277 | 0 |
| memoryless emission | 0.660889 | 0.658755 | 0.847796 | 0.456635 | 0.048101 | 0.573579 | 0.501690 |
| causal HMM | 0.846191 | 0.842271 | 0.446282 | 0.233027 | 0.037404 | 1.802333 | 0.108816 |

HMM per-class recall：normal `0.845630`、burst `0.807395`、leakage `0.853709`、calibration shift
`0.866280`。

以 evaluation seed 为 cluster 的 paired Student-t intervals：

- memoryless minus HMM NLL：`0.401514 [0.366352, 0.436676]`；
- memoryless minus HMM Brier：`0.223608 [0.202000, 0.245216]`；
- HMM minus memoryless accuracy：`0.185303 [0.173322, 0.197284]`。

## 与未来 CNN 的预算合同

T3.2.6 只冻结未来 comparator 必须共享的 budget，不声称 CNN 已实现：

| 项 | 合同/结果 |
| --- | ---: |
| raw input | `32 × 8` |
| update cadence | 每 32 cycles 一次 |
| reserved MAC upper budget | 4096 / update |
| reserved float32-state upper budget | 4096 bytes |
| HMM full-precision stored values | 896 |
| HMM float32 storage proxy | 3584 bytes |
| HMM MAC proxy | 800 / update |
| 本机 host median | 约 68.1 us / update，20 repeats |

host timing 会随机器/负载漂移，且模型当前以 NumPy float64 运行；float32 bytes/MAC 只是 representation/
operation proxy，不是 bit-accurate、RTL、综合、Fmax 或板测。T4.1 的 CNN/TCN/GRU 必须真正使用同一
raw input/cadence/budget 后才能称 matched comparison。

## 反简化验证

- 20 项 core tests 覆盖 raw schema、summary exactness、full covariance/precision、row-stochastic
  transitions、initial vs occupancy prior 分离、posterior normalization、prefix causality、copy isolation、
  malformed/underidentified training 和预算；
- 17 项 artifact/benchmark tests 覆盖 seed split、54-grid interior selection、4,096-row Source Data、8+8
  unique hashes、四类 support/recall、paired CI、profile 和 claim boundary；合计 focused `37 passed`；
- 15/15 machine gates，implementation SHA256 绑定 HMM、benchmark 和 syndrome stream 源码；
- prefix test 对 1/17/129/512 windows 重算，前缀 posterior 与 full-run prefix bit-for-bit 相同。

## Claim 边界

允许写：在登记 synthetic four-regime process 上，observed-window causal HMM 相对 same-emission
memoryless estimator 改善 posterior score/accuracy 并抑制 false switches，同时产生可量化 detection lag。

禁止写：device-calibrated state identification、logical decoder/control gain、CNN 已完成的 latency parity、
bit-accurate/fixed-point、RTL/synthesis/FPGA latency 或真实量子实验。

