# T5.1.6 实验可行性约束报告

## 结论

本任务的机器结论是 `PASS`，含义仅是可行性约束、缺失字段和软件安全观测已完整报告；总体
`deployment_readiness=NOT_ESTABLISHED`。当前唯一算法方向仍为
`event_aware_adaptive_map_fpga_codesign`，不构成 learned-decoder、RTL、实板或装置可部署性结论。

报告只读绑定 T4.4.4、T5.1.5、T4.3.3、T4.2.3、T4.4.3 和 T5.1.4 的 machine gates、文件 hash
与实现 hash，没有新增物理采样，也没有把不同模型层的 burden 拼成设备统计量。

## matched controller 可行性表

所有行共享 10 cycles/100 μs、20 measurements、20 resets 和 180 active gates。`p(g)`/`p(e)` 是
two-level matched simulator 的 outcome occupancy；不是装置 occupancy。multilevel leakage occupancy/events、
parameter saturation rate 和 matched classical latency 均未测，保持 `null`。

| cutoff | strategy | p(g) | p(e) | residual RMS | slew RMS | fidelity lifetime (μs) | logical-Z lifetime (μs) | scalars / MAC | lane peak |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 12 | standard | 0.667578 | 0.332422 | 0.000000 | 0.000000 | 35.706232 | 27.722165 | 15 / 0 | no |
| 12 | exact-budget MF | 0.822793 | 0.177207 | 0.167239 | 0.047723 | 86.227870 | 67.955500 | 72853 / 72266 | yes |
| 12 | fresh GRU teacher | 0.863965 | 0.136035 | 0.166539 | 0.023351 | 84.399248 | 67.569559 | 72853 / 72266 | no |
| 12 | handcrafted recurrence | 0.979102 | 0.020898 | 0.675113 | 0.033249 | 66.424408 | 64.328829 | 105 / 45 | no |
| 12 | distilled student | 0.864258 | 0.135742 | 0.166448 | 0.024415 | 84.271071 | 67.432372 | 95 / 87 | no |
| 16 | standard | 0.774219 | 0.225781 | 0.000000 | 0.000000 | 59.936245 | 51.351069 | 15 / 0 | no |
| 16 | exact-budget MF | 0.825234 | 0.174766 | 0.167205 | 0.047751 | 91.557119 | 75.076382 | 72853 / 72266 | no |
| 16 | fresh GRU teacher | 0.875781 | 0.124219 | 0.168122 | 0.022955 | 95.253959 | 79.608144 | 72853 / 72266 | yes |
| 16 | handcrafted recurrence | 0.978906 | 0.021094 | 0.675158 | 0.033267 | 63.806560 | 62.680715 | 105 / 45 | no |
| 16 | distilled student | 0.875781 | 0.124219 | 0.168083 | 0.024142 | 94.893946 | 79.084183 | 95 / 87 | no |

cutoff12 峰值属于 exact-budget MF，cutoff16 峰值属于 teacher；两者均为 72,853 scalars、72,266
MAC/half-cycle，且 latency 未测。峰值 lifetime 因而不能覆盖算法成本、复位、泄漏、饱和、硬件与物理前端缺口。
hard residual bounds 通过只能说明已记录动作未越冻结边界，不能替代 bound-hit/saturation rate。

## 软件故障恢复观测

T4.3.3 的 8 个定向场景各含 4 runs、95,984 cycles，总计 767,872 cycles。观测到 11,552 个 fallback
cycles（1.504417%）和 4 个 reset-request cycles（0.000521%）；unsafe action 与 undefined action 均为 0。
主要成本为 host-timeout 的 11,232 fallback cycles；另有 burst 260、leakage-reset 16、corrupt-transfer 32、
post-commit guard republish 12。通信暂停场景保留 1,596 个 ack-timeout 和 1,604 个 awaiting-readback cycles。

这些是预定向、确定性软件 campaign 的 observed rate，不是从声明的 device-fault population 独立同分布抽样；
因此不构造总体上界或设备安全率。T4.2.3 的 4,096-cycle component taxonomy 也单独保留：healthy 2050、
degraded 64、fallback 1296、recovering 622、reset-required 64，不能与 controller lifetime 合成总分。

## fail-closed 与缺失证据

4-state student 的失败路径仍是“reset state + exact zero physical residual”；leakage 重置内部状态，target latency、
RTL 与 board measurement 均为空/false。以下七项必须由后续任务补齐，当前不得填 0：

1. controller multilevel leakage occupancy/events；
2. controller parameter bound-hit/saturation rate；
3. matched controller classical latency；
4. target-board core/transport/end-to-end latency；
5. physical measurement/ADC/AWG/action latency；
6. device-calibrated reset fidelity/reset-storm burden；
7. 同一 matched finite-energy closed loop 中联合 lifetime 与 fault rates 的证据。

## 证据与验收

- 机器产物：`docs/t5_1_6_experimental_feasibility.json`；
- Source Data：`docs/t5_1_6_experimental_feasibility_source_data.csv`，408 rows；
- 21/21 contract gates；
- 34 项 direct/mutation tests，覆盖 leakage/saturation/latency 填零、复位与 fallback 隐藏、错误 peak、
  伪 population bound、student baseline 改写、stale/missing parent；
- 94 项父任务邻接回归通过。

因此 T5.1.6 完成的是“完整暴露可行性约束和不可用字段”，不是“证明实验可部署”。
