# T5.1.2 混合 noise/regime scenario matrix

## 结论

T5.1.2 已实际运行 10 类场景，并通过 15/15 个覆盖、因果、统计、物理与 provenance gate。这里的 `PASS` 表示矩阵完整且每条车道的原生检查通过，**不代表算法优势**，也不是全局排行榜。

核心机器产物：

- `docs/t5_1_2_mixed_scenario_matrix.json`；
- `docs/t5_1_2_mixed_scenario_matrix_source_data.csv`，共 116 行；
- 6 个 syndrome-decoder 场景共 36 个 seed-cluster，每个 cluster 32 windows、每窗 512 个 held-out decisions，共 589,824 个 paired decoder decisions；
- 所有 decoder 方法在同一 scenario-seed-window 内消费同一 displacement/residual/truth trace，observation 只在当前窗解码完成后更新 predictor；
- loss、readout/ancilla、large-error recovery、leakage 保留各自物理量，不与 decoder LER 混排。

## 为什么不是一张总表

十类场景对应四种不同的决策对象：

| lane | 覆盖场景 | 原生输出 | 是否进入 decoder 排名 |
| --- | --- | --- | --- |
| `decoder_syndrome_level_paired` | static Gaussian、mean/variance/correlation drift、burst/outlier、calibration shift | syndrome-level error rate、NLL、Brier、tracking、固定 signed contrasts | 只在本 lane 内比较 |
| `loss_noise_transfer` | loss | attenuation bias、decision covariance、alias-jump probability、validity | 否 |
| `protocol_readout_ancilla_fault_drift` | readout/ancilla drift | stage-resolved bit/phase fault、readout mismatch、logical backaction | 否 |
| component lanes | large-error recovery、leakage | recovery-run trend、occupancy、tail correlation | `component-only` |

No correction、measurement-feedback/autonomous sBs、top-K、Bayesian、training-selected sliding window、HMM、MF/FNN、teacher/student 和 control oracle 没有被伪记为失败；它们需要不同的 state、horizon、information set 或 protocol-native metric，留给后续 matched lane。theory-only Steane、Knill 和 P-Steane 仍不进入 sBs 主排名。

## Decoder lane

### 冻结协议

- training seeds：3 个；evaluation seeds：6 个，集合严格不相交；
- EWMA/Kalman 超参数只在原 T3.2.2 training scenarios 上选择，evaluation 之前冻结；本次选择为 `alpha=0.85`、Kalman process scale `1.5`、measurement scale `0.75`；
- frozen static MAP 同样只由 training states 拟合；
- 每个 scenario 的 6 个 seed 是 cluster unit，报告 Student-t cluster CI；
- oracle 只作 hidden-state reference，不是 deployable comparator；
- acceptance gate 不包含 `wins`、`best` 或 expected-direction 条件，因而不会用预期排序筛掉反例。

### 平均 error rate

| 场景 | Standard | Static MAP | Latest window | EWMA | Kalman | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static Gaussian | 0.001780 | 0.001750 | 0.001231 | 0.001231 | 0.001190 | 0.001149 |
| mean drift | 0.007009 | 0.006846 | 0.001017 | 0.001027 | 0.000977 | 0.000956 |
| variance drift | 0.008921 | 0.009054 | 0.007253 | 0.007090 | 0.007202 | 0.006805 |
| correlation drift | 0.007243 | 0.006826 | 0.003113 | 0.003082 | 0.002909 | 0.002797 |
| burst/outlier | 0.019114 | 0.019246 | 0.018728 | 0.018748 | 0.018707 | 0.018575 |
| calibration shift | 0.031108 | 0.021637 | 0.006114 | 0.006012 | 0.005941 | 0.003977 |

这些数值是 T5.1.2 的 raw lane-local 结果，不在本 task 宣布 winner。特别地，variance drift 中 static MAP 比 standard 略差，burst/outlier 中 adaptive 与 oracle 的差异很小；这些方向性反例均原样保留。正式 average/tail/oracle-gap、bootstrap 和多重比较属于 T5.1.3。

### 场景不是参数改名

- static Gaussian 的 `mu/sigma/rho/outlier` 在 32 windows 内严格不变；
- mean、variance、correlation 分别只改变目标分量，避免把不同 drift 混成一条曲线；
- burst/outlier 有两个独立事件段，事件内 `p_outlier=0.10`、scale `4.5`，事件外仍保留 `p_outlier=0.01`；
- calibration shift 在中点同时改变 mean、variance 与 correlation，并记录 event ID；
- 每个场景有 6 个唯一 trace hash，防止伪重复或把逐 window 当独立 seed。

## Loss lane

loss sweep 固定 10 dB resource squeezing、measurement efficiency `0.97`，只改变 transmissivity `1.00/0.98/0.94/0.88`。为隔离 attenuation-induced bias，输入使用非零 lattice index、零 signal offset；calibration offset 由独立 calibration-shift 场景承担。

| transmissivity | bias norm | decision-covariance trace | any jump probability | validity |
| ---: | ---: | ---: | ---: | --- |
| 1.00 | 0.000000 | 0.461856 | 0.018127 | localized |
| 0.98 | 0.035628 | 0.497856 | 0.024080 | localized |
| 0.94 | 0.107992 | 0.569856 | 0.039794 | clipping dominated |
| 0.88 | 0.219490 | 0.677856 | 0.073412 | clipping dominated |

两轴使用 diagonal covariance，因此联合 jump probability 由 axis independence 严格定义。若使用相关 covariance，surrogate 只允许报告 Fréchet bounds 和 `None`，不得把缺失的联合概率填成数字。本次开发首轮正是因此 fail closed；随后把 loss isolation 与 correlation scenario 分开，而不是放宽 validator。

## Readout/ancilla drift lane

四个 drift levels 分别使用 big-CD bit rate `0/0.01/0.02/0.04` 和 readout mismatch `0/0.005/0.01/0.02`，每点 20,000 cycles、fresh seed。所有 bit/readout/logical-backaction 频率均在期望值 5 sigma 内，observed bit/readout rate 随注册 drift level 严格增加。

此外另跑 60,000-sample protocol-native endpoint，sBs 与 sharpen--trim 的全部既有 checks 均通过。fault probabilities 仍是显式项目假设，不是 device calibration；theory-only Steane 与 secondary Knill/P-Steane 三条 schema 均保持 `executable=false`。

## Large-error 与 leakage component lanes

Large-error recovery 使用 4,096 shots × 9 amplitudes × 20 cycles 的 fresh-seed sweep。恢复 run 在 `l_S/4` 达峰，左右 Spearman 分别为 `+1/-1`，midpoint 相对 endpoints 的 CI-separated margin 为 `4.7358`，unaffected-axis negative control 为 `0.02539`，10/10 native gates 通过。

Leakage 使用 600 trajectories × 1,200 cycles。hidden occupancy 为 `0.81328 [0.81150, 0.81497]`，syndrome estimator 为 `0.81322`；移除 observed leakage run 后保留 `85.83%` trajectories，tail-correlation shrink ratio `2.4458`，paired difference `0.002588 [0.001420, 0.003845]`，11/11 native gates 通过。该动作是离线 post-selection，不是在线 leakage controller，也不进入 decoder leaderboard。

## 深度检查与边界

本 task 的非 demo 证据包括：

1. 36 个独立 seed-cluster 和 589,824 个 paired decisions，而不是单 seed 演示；
2. train/eval seed isolation、evaluation-before-update causality 与逐 trace SHA-256；
3. burst、abrupt shift、loss、fault-rate drift、large displacement、persistent leakage 六类不同失败机制；
4. 6 个 parent artifact hashes、8 个 implementation hashes、116-row Source Data；
5. 对 scenario deletion、seed overlap、component ranking promotion、stale hash、global leaderboard 注入和 gate failure 的 fail-closed mutation tests；
6. 不以“adaptive 应该更好”作为 task PASS 条件，并保留 static/standard 或 adaptive 间的反方向结果。

允许的结论仅是：10 类注册场景已在各自原生 lane 真实执行，并具有共享输入、统计复核、物理自检和可追溯产物。禁止把 syndrome-level、noise-transfer、protocol-fault、component-only 指标混成一个分数，也禁止外推到 finite-energy protocol lifetime、真实设备 fault rate、FPGA 实测或 universal adaptive/NMF/CNN superiority。
