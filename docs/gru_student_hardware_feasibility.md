# T5.5.4 完整 GRU、量化 GRU 与蒸馏 student 硬件可行性

## 结论

默认硬件路线保持 4-state signed-Q3.14 蒸馏 student；完整 GRU 只保留为离线 teacher，optional
quantized-GRU enhanced route 明确 `Dropped`。正式 verdict 为：

```text
DISTILLED_STUDENT_ONLY_QUANTIZED_GRU_DROPPED_FULL_GRU_OFFLINE_TEACHER
```

这个结论不是因为量化 GRU “完全装不下”，而是更严格的三重否决：

1. 它的完整参数 ROM 与一个真实串行 MAC 的 integrated lower-bound 设计虽能装入，但已用
   `41/46` BSRAM；
2. 逐项消费全部权重/偏置的乐观下界也要 72,854 cycles，即 `2698.30 us`@27 MHz；
3. 被综合的 RTL 故意省略 GRU gate dependency、activation buffer 和 nonlinearities，不是 functional
   GRU；fake-quantized action 误差也没有重跑物理 gain-retention，不能用来补齐该缺口。

## 同口径比较

| candidate | 参数存储 | MAC/更新 | integrated BSRAM | Fmax 最差 | latency | physical gain retention | 决策 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| full GRU float64 | 4,662,592 bit | 72,266 | 至少 261/46（含 core 8） | 未综合 | 最乐观 753 cycles 仍是下界 | teacher reference=1，不是部署结果 | offline only |
| full GRU float32 | 2,331,296 bit | 72,266 | 至少 135/46（含 core 8） | 未综合 | 最乐观 753 cycles 仍是下界 | teacher reference=1，不是部署结果 | offline only |
| int8/Q3.14 GRU workload | 588,694 bit | 72,266 | 41/46 actual | 39.1527 MHz | 72,854 cycles lower bound | `null` | Dropped |
| 4-state Q3.14 student | 1,710 bit | parent 87 / RTL 64 multiplications | 8/46 actual | 39.5726 MHz | 64 cycles=`2.37037 us`@27 MHz | point min 0.981457；CI lower min 0.944501 | selected |

完整 GRU 的 float32/float64 参数分别至少需要 127/253 个 18-Kbit blocks，连同 fast core 的 8 个
blocks 是 135/261；这还没计 hidden/activation buffer、非线性函数和算术。因此不需要再用一个删去真实
结构的“float GRU demo top”制造无意义综合数字。96 个 MULT9X9 全并行的 `ceil(72266/96)=753`
cycles 只是极乐观运算数下界，并不代表浮点 datapath。

## 量化 GRU 参数与数值 shadow

选中的 T4.4.1 checkpoint 是 `GRU10-DENSE256-DENSE256-OUT15`：72,266 个 weight scalars、587
个 biases，共 72,853 参数。五组权重做 per-tensor signed-int8 power-of-two scale，fractional bits 为
`8/8/10/10/12`；bias 用 signed Q3.14。生成文件覆盖所有参数，未裁剪 layer 或只取稀疏样例。

为避免把错误的 GRU 公式也量化得“很接近”，先用 manual `r/z/n` gate equation 对 PyTorch
`GRUCell` 做 64-row 随机检查，最大差 `5.55e-17`。再执行 functional fake-quantized shadow：

| 数据 | action comparisons | action RMSE | action max error | action p99 error | hidden max error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 256 个 length-8 histories、所有 prefixes | 30,720 | 1.3213e-4 | 5.3794e-4 | 3.7884e-4 | 1.9397e-3 |
| 128×256 fresh random sequences | 491,520 | 1.4222e-4 | 5.5199e-4 | 4.0138e-4 | 2.0457e-3 |

全部值有限、15 个 action 全在 hard bounds 内。这里建立的只是量化数值接近性；T4.4.4 没有用该
shadow 重跑 matched physical benchmark，所以 `physical_gain_retention=null`，不能从 action MSE 外推。

## 量化 GRU lower-bound RTL 非 demo 检查

`quantized_gru_workload_kernel.sv` 保存全部 72,266×8-bit weights 与 587×18-bit biases，并用一个真实
signed multiplier 流过全部权重。它刻意不实现功能 GRU，因此只有一种合法用途：若这个乐观下界都超
deadline，则完整实现必然不能被当前证据放行；反之即使它通过，也不能证明功能实现可行。

CXXRTL 从 reset/start 跑到 done，直接验证：

- `weight_macs_completed=72,266`；
- `biases_consumed=587`；
- `cycles_after_start=72,854`；
- done 时 `busy=0`，并有非零 data-dependent signature；
- 独立 bit-vector reference 对全部 72,266 weights 和 587 biases 的顺序、18/26/40-bit wrap、LFSR
  重算，signature `730990968` 与 CXXRTL 完全一致；
- Yosys `check` 为 0 structural problems。

这项顺序对拍在复核中发现并修复了两个旧门禁看不见的问题：bias phase 起始时 `bias[0]` 被消费两次、
`bias[586]` 被漏掉；修正预取后，最后周期同步 ROM 又会组合读取越界地址 587。最终实现以 address-1
prefetch 加 terminal clamp 同时保证顺序 `0..586` 和全程合法地址。修复后重新运行了 CXXRTL、综合和
全部三个 P&R seeds；没有沿用旧 netlist/report。

权重/偏置 ROM 映射为 33 个 `SPX9`，fast core 保持 8 个 `SDPX9B`。这防止综合器把未被观测的参数
memory 优化掉；最终 P&R 的 BSRAM=41 与该结构一致。

## 三 seed P&R

| seed | Fmax (MHz) | BSRAM | LUT4 | DFF |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 40.2625 | 41 | 3,895 | 1,011 |
| 7 | 39.1527 | 41 | 3,904 | 1,011 |
| 19 | 40.6835 | 41 | 3,899 | 1,011 |

三 seed 都通过 27 MHz，最大资源还包括 MULT18X18=1、MULT9X9=3、ALU=518、IOB=18。报告使用
最差 Fmax，不挑最佳 seed。即使用最差可达 Fmax 而不是 27 MHz，72,854-cycle lower bound 仍为
`1860.76 us`，是 5 us slot 的 372 倍以上。真实功能 GRU 还要加入依赖、非线性和 activation traffic，
所以 worst-case latency 保持 `null`，不能把 lower bound 冒充完整延迟。

## student 选择和反简化审计

student 不是本任务新写的代理。它消费 T5.5.3 已完成的完整 CXXRTL 与 integrated 三 seed P&R：
7,680 output codes 和 72-bit state 0 mismatch，真实 64-cycle update，8 BSRAM、2×MULT18X18、
1×MULT9X9，harness 地址修复后最差 Fmax `39.5726 MHz`。T4.4.4 matched physical retention 的六项门也全部高于冻结的
0.90 point/CI threshold。

正式 report 有 16/16 gates、12/12 shortcut mutations rejected；13 个 focused tests 直接重算 gate、
参数覆盖、GRU equation、storage failure、三 seed、CXXRTL counters、evidence boundary 和 CSV/hash。
这些门会拒绝：少读一个权重、伪造 100-cycle latency、把 41-BRAM fit 当成 enhanced-route 通过、给
quantized shadow 编造 gain、或把 post-route 写成 board measurement。

跨 T5.5.1--T5.5.4 和协议层的定向回归为 `138 passed`。仓库全量有 2,469 tests；30 分钟执行到约
20% 后 timeout，已执行部分未出现 failure，但该结果严格标为 incomplete，不写成 full-suite PASS。

## 产物和证据边界

- runner：`cnn_fpga/benchmark/gru_student_hardware_feasibility.py`；
- memories：`cnn_fpga/rtl/generated/t5_5_4_quantized_gru_*`；
- lower-bound RTL：`cnn_fpga/rtl/quantized_gru_workload_kernel.sv` 与 integrated top；
- CXXRTL driver：`cnn_fpga/rtl/quantized_gru_workload_cxxrtl_driver.cc`；
- machine report：`docs/t5_5_4_gru_student_hardware_feasibility.json` 与 Source Data；
- raw evidence：CXXRTL trace/Yosys log、integrated Yosys log、三 seed nextpnr report/log。
- provenance：当前 RTL/manifest/SDC/CST hashes、9.88 MB uncompressed netlist hash、361 KB deterministic
  gzip netlist 与三 seed report/log 全部绑定；源码改变后旧工具报告会 fail closed。

已建立：精确参数/MAC/存储账、量化 functional shadow、完整参数 lower-bound RTL 的 CXXRTL、目标器件
三 seed post-route、student 的既有功能/物理收益/资源/deadline 闭环。

未建立：functional quantized-GRU RTL、其 physical gain retention、full-GRU target synthesis、vendor
signoff、bitstream、transport、真板 latency/power/throughput 或 quantum-hardware measurement。
