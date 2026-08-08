# T5.5.3 精度—资源—性能 Pareto 冻结

## 结论

唯一进入部署冻结的 source-bound 点为：

```text
MAP-LUT precision = ADC10 / address8 / signed Q9.12 (22 bit)
top-K reference   = K=4（只作离线/软件 comparator，不在 FPGA datapath）
student           = frozen 4-state exponential recurrence
student arithmetic= signed Q3.14, one time-multiplexed 18x18 multiplier
```

正式 verdict 是
`SELECT_P10_A8_Q9_12_K4_REFERENCE_STATE4_SERIAL_DSP_POST_ROUTE_PASS_NOT_BOARD_MEASURED`。
它来自 108 点联合矩阵、六个 hash-bound parent artifacts、student CXXRTL 对拍，以及最终点三 seed
目标器件 P&R；不是把各任务的 resource proxy 相加后自称“已综合”。

## 四条候选轴

### 1. MAP-LUT precision

精度数据直接消费 T4.2.4 的 matched 8-bank×4-seed paired trace。预注册门要求 exhaustive hard
mismatch 为 0、action disagreement `<=1e-4`、paired ΔLER 95% CI 两端绝对值 `<=1e-3`。

| profile | ADC/address/LLR | quantized LER | action disagreement | quality | 8-mirror BSRAM packing |
| --- | --- | ---: | ---: | --- | ---: |
| low | 6/4/Q5.6 | 0.0397644 | 1.1139e-3 | fail | 8 |
| medium | 8/6/Q7.10 | 0.0395660 | 2.1362e-4 | fail | 8 |
| selected | 10/8/Q9.12 | 0.0396271 | 9.1553e-5 | pass | 8 |
| dense | 12/10/Q10.14 | 0.0395966 | 0 | pass | 16 |

selected 是满足门的最小 profile；dense 没有 LER 优势 claim，却把八个 mirrored physical tables 的
最低 BSRAM packing 从 8 提到 16。只有 selected profile 具有本任务的 integrated actual synthesis；
其余 profile 的 BSRAM 是精确 packing 算术，LUT/DFF/Fmax 仍不得冒充工具报告。

### 2. top-K

T3.1.5 的六个 correlated-Gaussian 场景给出：

| K | 六场景全部收敛 | 最大 |ΔLER| | 最大 hard disagreement | 最大 LLR p99 error |
| ---: | --- | ---: | ---: | ---: |
| 1 | 否 | 2.50e-4 | 2.4792e-3 | 0.6472 |
| 2 | 否 | 2.0833e-5 | 2.0417e-3 | 0.07186 |
| 4 | 是 | 6.25e-5 | 8.3333e-5 | 3.8824e-4 |

因此 `K=4` 是冻结场景下最小通过点。但当前 integrated RTL 是 1D parameterized MAP-LUT fast path，
没有在线 2D Gaussian alias scorer/top-K accumulator。K=4 仅作为离线/软件 reference；它的资源没有
被静默计入 FPGA 数字，`online_topk_rtl=false`。

### 3. student state dimension

| dimension | validation MSE | evaluation MSE | stored scalars | analytic MAC | parent eligible |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.2974e-5 | 1.4071e-5 | 35 | 33 | 否 |
| 2 | 1.0833e-5 | 1.1716e-5 | 55 | 51 | 否 |
| 4 | 5.6485e-6 | 6.0831e-6 | 95 | 87 | 是 |

T4.4.3 的 evaluation-blind rule 只允许 4-state；T4.4.4 对该点的六个 retention gates 给出最低 point
`0.981457`、最低 CI lower `0.944501`。1/2-state 没有 physical gain-retention 证据，不能因资源小而
被本任务重新选中。

### 4. multiplier parallelism

实际 RTL 用代数等价的 `z_inf + a*(z-z_inf)`，以一个 DSP 串行完成 4 次 state update 与
15×4 次 output MAC，共 64 cycles。P=2/4 的 operation-count extrapolation 为 32/16 cycles，
但没有独立 RTL/P&R，故不能作为 measured hardware。串行点在 27 MHz 是 `2.37037 us < 5 us`
project-model slot，已经满足 deadline；按最小实际资源规则不增加 DSP。

## student RTL 非 demo 验证

- 来源：冻结的 `docs/t4_4_3_low_dimensional_student.json`；
- 定点：signed Q3.14，五组 coefficient memories 全部绑定源 SHA256；
- datapath：真实 recurrence、ties-to-even product、signed saturation、15-output hard bounds；
- failure：`health_ok=false` 取消 busy、重置 state、输出零 residual；
- CXXRTL：512 operations、507 healthy updates、5 forced resets、7,680 output codes、完整 72-bit
  state，0 mismatch；
- fixed-vs-float 最大输出差：`1.46038e-4`；
- 修复：terminal output 原先在未采用分支仍形成 `bias_mem[15]` 组合读，CXXRTL 越界断言后改为
  `next_output_cursor` 钳位；Windows helper 也设置 no-GP-fault error box，避免断言弹窗。

## 最终点三 seed P&R

| seed | Fmax (MHz) | critical period (ns) | logic (ns) | routing (ns) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 40.5351 | 24.6700 | 11.9100 | 12.4940 |
| 7 | 40.3226 | 24.8000 | 10.7980 | 13.7360 |
| 19 | 39.5726 | 25.2700 | 10.7400 | 14.2640 |

三个 critical paths 仍落在 core state 到 core activity fold，不在 student；不能把 Fmax 较 T5.5.2
偶然更高解释成 student 改善时序。报告使用 minimum/median/maximum
`39.5726/40.3226/40.5351 MHz`，全部通过 27 MHz。该组结果是在 core harness 配置地址
modulo-257 修复后重新综合/P&R 得到，旧报告未复用。

| resource | integrated max | available | 相对 T5.5.2 max 增量 |
| --- | ---: | ---: | ---: |
| LUT4 | 3,802 | 20,736 | +440 |
| DFF | 1,022 | 15,552 | +157 |
| BSRAM | 8 | 46 | 0 |
| MULT18X18 | 2 | 48 | +1 |
| MULT9X9 | 1 | 96 | 0 |
| ALU | 616 | 15,552 | +276 |
| IOB | 18 | 384 | 0 |

fast core latency 仍是 6 cycles=`222.222 ns`@27MHz；student 与 core 并行，完整 15-output update
是 64 cycles=`2.37037 us`。后者不是 transport 或 quantum-cycle latency。

## 108 点矩阵与证据等级

矩阵为 4 precision×3 K×3 dimension×3 parallelism。只有与实际 integrated RTL 一致的
selected precision、4-state、P=1 rows 带 `actual_three_seed_integrated_post_route`；其它 rows 只有
`calibrated_estimate_not_synthesis`。K 不在 datapath，因此所有 rows 的
`topk_hardware_resources_included=false`。质量门再把 K=1/2 排除，最终只有一个 eligible row。

该设计防止三类 shortcut：挑最佳 seed、把 107 个 estimate 写成实测、把离线 top-K 资源悄悄并入或
忽略。16/16 gates、10/10 semantic mutations 和 25 direct tests 通过；父链组合回归 184 passed。

## 正式产物与边界

- runner：`cnn_fpga/benchmark/precision_resource_pareto.py`；
- student equivalence：`cnn_fpga/benchmark/student_rtl_equivalence.py`；
- RTL：`cnn_fpga/rtl/low_dimensional_student_kernel.sv`、integrated synthesis top 与生成 memories；
- machine artifacts：`docs/t5_5_3_precision_resource_pareto.json`、Source Data、
  `docs/t5_5_3_student_rtl_equivalence.json`；
- raw tools：`docs/t5_5_3_yosys_integrated.log`、三 seed nextpnr report/log。

已建立：有限父证据下的唯一 deployment point、student fixed RTL equivalence、目标器件 integrated
open-source post-route estimate。

未建立：在线 top-K RTL、P=2/4 RTL、完整/量化 GRU 可行性、vendor signoff、bitstream、transport、
真板 latency/power/throughput 或 quantum hardware result。下一任务 T5.5.4 必须独立比较 GRU 与 student；
如其量化/gain gate 失败，应撤销 student，回退已综合的 MAP-LUT core。
