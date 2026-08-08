# T5.4.1：held-out/OOD 验证

## 结论

T5.4.1 以**访问结果前冻结场景、冻结 parent 模型、四组全新且互斥 seed** 的方式完成 4 条原生 lane：

1. frozen T5.1.2 decoder 的 unseen drift family/range；
2. sBs protocol-native measurement confusion；
3. persistent leakage/reset kernel 的 unseen leakage rate；
4. dual-loop scheduler 的 unseen communication disturbance。

机器产物 20/20 gates 通过，包含 104 个正式 seed cells、280-row Source Data。`PASS` 仅表示这些
OOD cells 已真实执行且 split、provenance、数值与 fail-closed 语义完整；它不表示统一系统稳健、所有方法
优于 baseline、uncertainty fallback 有效或目标板已验证。

## 预注册与数据隔离

- drift、measurement、leakage、communication 各 8 个 evaluation seed，共 32 个 seed cluster；四组两两
  不相交，也不与 8 个 parent artifacts 中递归提取的 train/validation/evaluation/bootstrap seeds 相交；
- T5.1.2 的 static MAP 与 EWMA/Kalman hyperparameters 从 parent artifact 按 hash 恢复，OOD 结果不参与
  refit、threshold 或 method selection；
- measurement matrices、leakage rates 和三类 communication pattern 都在 runner 常量与 frozen config
  中固定；artifact validator 会从 raw rows 重算 gates，而不是信任存储的布尔值；
- 四条 lane 保持自己的 observable，不生成 cross-lane score 或 universal rank。

## unseen drift family/range

三类 drift 各跑 8 seed×64 windows×512 decisions，共 786,432 个 paired decisions；每个 cell 内
standard/static/window/EWMA/Kalman/oracle 消费完全相同的 displacement/residual truth：

| OOD 场景 | static error | window | EWMA | Kalman | oracle | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| joint sinusoidal rotation | 0.016491 | 0.006256 | 0.007053 | 0.008865 | 0.003616 | unseen joint periodic family；window 点估计优于递推 |
| stochastic telegraph | 0.035587 | 0.043941 | 0.041214 | 0.045292 | 0.003021 | abrupt family 下 adaptive 点估计反而更差；static-minus-EWMA CI 跨 0 |
| compound range extrapolation | 0.172523 | 0.101925 | 0.102779 | 0.104141 | 0.083954 | 超出 parent 的 mean/sigma/rho/outlier envelope；各方法均明显远离 oracle |

compound lane 超过 8 项 parent envelope：`|mu_q|`、`|mu_p|`、`sigma_q max`、`sigma_p min/max`、
`|rho|`、`p_outlier` 与 `outlier_scale`。telegraph 的 negative point estimate 不被包装成显著失败：
static-minus-EWMA 为 `-0.005627`，95% seed-cluster CI `[-0.018982, 0.007729]`；
static-minus-Kalman 为 `-0.009705`，CI `[-0.024188, 0.004779]`。

## measurement confusion OOD

三张 4×3 row-stochastic confusion matrix 均不同于 T5.2.2 的 `0--0.08` symmetric rate grid：

| 场景 | target g→e / e→g | empirical g→e / e→g | aggregate mismatch |
| --- | ---: | ---: | ---: |
| asymmetric g→e | 0.15 / 0.03 | 0.148974 / 0.031040 | 0.089058 |
| asymmetric e→g | 0.04 / 0.18 | 0.040044 / 0.180536 | 0.117714 |
| high symmetric | 0.18 / 0.18 | 0.178809 / 0.180536 | 0.179764 |

每个场景使用 8×8,192 sBs cycles，共 393,216 cycles/786,432 constituent observations。所有解析
confusion rate 均落在预注册 five-sigma+finite-count tolerance 内；ancilla bit/phase event 与 ideal-label
change 始终为 0，避免把多通道 fault 混入 measurement-only lane。truth 仅用于评分，deployable schema
仍是 observed syndrome/reset/run fields。

## leakage-rate OOD

T5.2.3 parent grid 最大为 `0.004`；本任务固定 `0.003/0.006/0.012`，同时覆盖 unseen interpolation 与
extrapolation。每档运行 8 seed×256 trajectories×512 cycles：

| injected rate | empirical rate | hidden occupancy | episode detection | unsafe declared available | reset failures / 1000 cycles |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.003 | 0.002960 | 0.030600 | 1.000000 | 0.001545 | 52.3634 |
| 0.006 | 0.005959 | 0.059610 | 0.999917 | 0.002974 | 102.0126 |
| 0.012 | 0.012020 | 0.114154 | 0.999956 | 0.005731 | 195.3201 |

这里的 episode detection 接近 1 是 persistent episode 内可多步观察的结果，不是 per-step classifier 的
总体保证；classifier sensitivity、false-alarm 和 reset law 仍是 effective assumptions，不能外推 device。

## communication-disturbance OOD

reference 与三类新模式各跑 8×24,000 scheduler cycles，共 768,000 cycles。所有 pause start/end 都在每个
seed 被检测，version 单调、最大 version step≤1、数组有限、无 hidden-truth estimator 和 external conflicting
update：

| 场景 | paired ΔLER 95% bootstrap CI | paired Δavailability 95% CI | 解释 |
| --- | ---: | ---: | --- |
| periodic micro-outages | 0 `[0,0]` | 0 `[0,0]` | 10 次短 pause 被检测，但未命中改变当前 window outcome 的时点；保留 null |
| increasing-duration flaps | 0.012391 `[0.011898,0.012836]` | -0.387125 `[-0.390953,-0.383344]` | 长 outage 显著破坏 freshness/availability |
| communication+jitter+burst | 0.042977 `[0.041234,0.044992]` | -0.867977 `[-0.870414,-0.865336]` | pause、deadline、FIFO/drop 同时发生，保留复合负结果 |

短 pause null 不证明真实链路免疫；它只说明当前 scheduler 的 window cadence 与这些离线 pause 对齐方式下，
reference metric 未变化。T5.4.2 才会在同一 fault population 上比较 gated fallback 与 no-fallback。

## 产物与复现

- `cnn_fpga/benchmark/held_out_ood_validation.py`
- `tests/test_held_out_ood_validation.py`
- `docs/t5_4_1_held_out_ood_validation.json`
- `docs/t5_4_1_held_out_ood_source_data.csv`

```powershell
$env:PYTHONPATH='.'
python -m cnn_fpga.benchmark.held_out_ood_validation
python -m pytest -q tests/test_held_out_ood_validation.py
```

## Claim 边界

允许：预注册 synthetic OOD coverage、frozen decoder 的 paired degradation、protocol-native confusion/leakage
sensitivity，以及 software scheduler communication stress。

禁止：跨 lane 系统稳健性、全分布保证、显著 adaptive failure（telegraph CI 跨 0）、uncertainty-gated fallback
收益、device-calibrated leakage/readout、physical-memory LER、RTL、transport 或目标板结果。
