# T5.2.1 displacement / large-distance 因果故障注入报告

## 结论

独立正式 campaign 为 `PASS`：17 个预注册 `epsilon/l_S` 幅度、8 个互不重叠的 evaluation seed clusters、
每点每 seed 4,096 shots、20 cycles，共形成 136 个 recovery seed rows 与 272 个 logical seed rows。
20/20 gates 和 1,863-row Source Data 通过。

本任务补上了 T2.0.5/T5.1.2 没有的 seed-cluster 复现与逻辑 estimand，但证据仍是 protocol-aligned
coarse model，不是 Fock-coherent injection、physical-memory LER、装置标定、真实 QPU 或 FPGA 实测。

## 因果设计

- 唯一跨 paired runs 改变的通道是 nominal displacement amplitude；
- recovery transition、readout confusion、reset kernel、fault quadrature、horizon 与 shots 全部冻结；
- 每个 seed cluster 内，17 个幅度使用共同随机数；8 个 seeds 才是独立统计单位；
- 旧 T2.0.5/T5.1.2 calibration/pilot seeds 与本次 evaluation seeds 完全不重叠；
- controller 只产生/消费同象限 `g/e` trajectory；actual parity 和 nominal logical target 仅供 evaluator 使用。

恢复与逻辑指标分成两条不可混用 lanes：

1. recovery lane 复用注册的 sBs error-space + observation/reset model，报告 initial recovery depth、same-quadrature
   e-run、restricted recovery time、horizon recovery 与 unaffected-axis negative control；
2. logical assay 在冻结的 injection-jitter profile 下，用 `l_S/2` logical-operation spacing 判定实际 parity，既报告
   相对最近 nominal logical operation 的 misclassification，也报告相对 identity frame 的 logical flip。

logical failure 不是“20 cycles 内未 recovery”重命名；后者单列且本次所有点的 observed recovered fraction 为 1。

## recovery-depth 与 e-run

| epsilon/l_S | 最近逻辑操作距离 | mean initial depth | observed same-axis max e-run | recovered fraction | unaffected max P(e) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00000 | 0.000 | 0.00000 | 0.09616 | 1.000000 | 0.007111 |
| 0.03125 | 0.125 | 0.75156 | 0.76013 | 1.000000 | 0.015381 |
| 0.06250 | 0.250 | 1.49744 | 1.40341 | 1.000000 | 0.020325 |
| 0.09375 | 0.375 | 2.24631 | 2.03348 | 1.000000 | 0.022156 |
| 0.12500 | 0.500 | 3.00064 | 2.64496 | 1.000000 | 0.022888 |
| 0.15625 | 0.625 | 3.75369 | 3.23901 | 1.000000 | 0.023773 |
| 0.18750 | 0.750 | 4.50256 | 3.80399 | 1.000000 | 0.024231 |
| 0.21875 | 0.875 | 5.24844 | 4.33719 | 1.000000 | 0.024872 |
| 0.25000 | 1.000 | 6.00000 | 4.84512 | 1.000000 | 0.025421 |
| 0.28125 | 0.875 | 5.24844 | 4.33719 | 1.000000 | 0.024872 |
| 0.31250 | 0.750 | 4.50256 | 3.80399 | 1.000000 | 0.024231 |
| 0.34375 | 0.625 | 3.75369 | 3.23901 | 1.000000 | 0.023773 |
| 0.37500 | 0.500 | 3.00064 | 2.64496 | 1.000000 | 0.022888 |
| 0.40625 | 0.375 | 2.24631 | 2.03348 | 1.000000 | 0.022156 |
| 0.43750 | 0.250 | 1.49744 | 1.40341 | 1.000000 | 0.020325 |
| 0.46875 | 0.125 | 0.75156 | 0.76013 | 1.000000 | 0.015381 |
| 0.50000 | 0.000 | 0.00000 | 0.09616 | 1.000000 | 0.007111 |

depth 与 e-run 在 `l_S/4` 严格达峰，并在两侧随到最近 logical operation 的距离单调变化。镜像点使用相同
severity 和 common random numbers，因此精确镜像；这验证当前 coarse model 的 causal mapping，不构成独立
物理定律发现。unaffected-axis 95% CI 上界始终小于冻结的 0.06 negative-control 门。

## 两种 logical rate 不得互相替代

| epsilon/l_S | nearest-op failure, sigma=0.040 | identity flip, sigma=0.040 | nearest-op failure, sigma=0.025 | identity flip, sigma=0.025 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.03125 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.06250 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.09375 | 0.000092 | 0.000092 | 0.000000 | 0.000000 |
| 0.12500 | 0.001099 | 0.001099 | 0.000000 | 0.000000 |
| 0.15625 | 0.010345 | 0.010345 | 0.000092 | 0.000092 |
| 0.18750 | 0.060211 | 0.060211 | 0.005493 | 0.005493 |
| 0.21875 | 0.219330 | 0.219330 | 0.105927 | 0.105927 |
| 0.25000 | 0.495300 | 0.504700 | 0.497864 | 0.502136 |
| 0.28125 | 0.215057 | 0.784943 | 0.102692 | 0.897308 |
| 0.31250 | 0.056152 | 0.943848 | 0.005707 | 0.994293 |
| 0.34375 | 0.008820 | 0.991180 | 0.000122 | 0.999878 |
| 0.37500 | 0.000732 | 0.999268 | 0.000000 | 1.000000 |
| 0.40625 | 0.000061 | 0.999939 | 0.000000 | 1.000000 |
| 0.43750 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.46875 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |

nearest-operation-relative rate 在 midpoint 约 0.5 并向两端下降；identity-reference rate 则从 0 单调升至 1，
明确暴露 `epsilon/l_S=0.5` 本身是 logical flip，而不是用“最近目标正确”把它隐藏。两个 profile 的 Monte Carlo
与独立 Gaussian boundary-crossing 公式全网格最大绝对差 `<0.005`，低于冻结门 0.012。

## 统计与非 demo 检查

- 所有 CI 以 8 条 whole-seed trajectories 为 bootstrap unit，20,000 次重采样；不把 4,096 shots 当独立实验；
- 34 个逐点 CI 不被误称 simultaneous band；解析一致性采用全网格最大绝对误差，两个 midpoint 的 0.5 均被 CI 覆盖；
- semantic validator 会拒绝 seed/grid 改写、同时篡改 seed rows 与 summary、logical/recovery 混名、truth 入 controller、
  parent/source/code hash 漂移、branch 改写和 device/physical-LER 升级；
- direct/mutation `43 passed`，T2.0.5/T5.1.2/T2.3.3/T5.0.2/T5.1.6 邻接 `137 passed`。

## 产物

- `cnn_fpga/benchmark/displacement_large_error_causal.py`
- `tests/test_displacement_large_error_causal.py`
- `docs/t5_2_1_displacement_large_error_causal.json`
- `docs/t5_2_1_displacement_large_error_source_data.csv`

可写“independent protocol-aligned causal displacement campaign reproduces midpoint-peaked recovery/e-run and
nearest-operation-relative logical classification difficulty”。不可写 quantitative Fig. 4(c) reproduction、physical-memory
LER、coherent/Fock displacement recovery、device calibration 或真实 hardware fault injection。
