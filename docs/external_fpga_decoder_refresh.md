# T6.19.2 外部 FPGA QEC decoder 规范化刷新

冻结检索日：`2026-07-20`。共纳入 18 个外部实现/profile，其中继承并实时核验 T6.8.6 的 8 行，新增 10 行。

## 结论

本项目的 exact same-task 外部 comparator 仍为 **0**。因此不形成 raw-ns 排名，不声称比已有 FPGA decoder 更快，也不声称 fastest/SOTA。当前允许措辞仍是：已通过 CXXRTL 与目标器件 P&R estimate 的 single-mode GKP 六周期、II=1 确定性预板 fast path；真实 source-to-action、jitter、deadline 与 power 等待 T6.9.2。

## Source-normalized 外部条目（描述性、非总榜）

| row | family / size | boundary + statistic | latency | device / evidence | 关键边界 |
| --- | --- | --- | ---: | --- | --- |
| `lilliput_d5_m2` | rotated surface code; d=5, m=2; one logical qubit; 24-bit LUT address and 37-bit entry | syndrome-arrival to error assignment inside decoder; fixed seven-cycle decoder latency; external-memory allowance included | 42 ns | Intel Arria V; `POST_IMPLEMENTATION_ESTIMATE_NOT_QPU_CLOSED_LOOP` | different code family |
| `helios_d21` | rotated surface code; d=21; one logical qubit | average decoder execution time divided by d measurement rounds; mean over 1,000,000 synthetic trials; long tail exists | 11.5 ns | AMD/Xilinx VCU129 (VU29P); `FPGA_SYNTHETIC_INPUT_MEAN_PER_MEASUREMENT_ROUND` | different code family |
| `collision_clustering_d21` | rotated planar surface code; d=21; 881 physical qubits | decoder execution time divided by d syndrome-measurement rounds; mean over 100,000 synthetic shots | 810 ns | AMD/Xilinx UltraScale+ XCVU3P; `FPGA_SYNTHETIC_INPUT_MEAN_NORMALIZED_PER_ROUND` | different code family |
| `local_clustering_d17_adaptive_hl` | rotated surface code; d=17 | reported decoder time per measurement round; paper table execution time for adaptive high-level configuration | 676 ns | AMD/Xilinx UltraScale+ XCVU19P; `FPGA_IMPLEMENTATION_EXECUTION_TIME_NOT_QPU_CLOSED_LOOP` | different code family |
| `overwater_nn_d5` | rotated surface code; d=5; hidden layers 64 and 64 | decoder combinational/post-implementation inference delay; post-implementation timing estimate | 87.6 ns | AMD/Xilinx Artix-7; `POST_IMPLEMENTATION_TIMING_ESTIMATE` | different code family |
| `caune_stability8_9round_feedback` | 8-qubit surface-code stability experiment; 8-qubit stability experiment; 9 measurement rounds | final data-extraction round to application of conditional logical gate; measured full decoding response for nine rounds | 9600 ns | Rigetti control-system FPGA; exact part not reported; `REAL_QPU_FULL_DECODING_RESPONSE` | different code family and QPU |
| `maurer_gross_int4_x` | bivariate bicycle qLDPC gross code; [[144,12,12]] gross code; X decoder; 12-cycle window | average Relay-BP core completion for a 12-cycle detector window; about 20 iterations on average at p=0.1%; per-iteration time is deterministic | 480 ns | AMD/Xilinx UltraScale+ XCVU19P; `FPGA_SYNTHETIC_INPUT_AVERAGE_CONVERGENCE` | different qLDPC code family |
| `yang_nn_d3_closed_loop` | distance-3 rotated surface code; d=3; 17-qubit patch; one logical qubit | end of readout pulse to start of feedback pulse; deterministic closed-loop latency; deterministic NN core latency | 550 ns | AMD/Xilinx XC7K410T; `REAL_QPU_DETERMINISTIC_CLOSED_LOOP` | different code family and syndrome semantics |
| `micro_blossom_d13` | rotated surface code; d=13; one logical qubit; streamed d-round circuit-level decoding graph | syndrome ready after last round to correction bit available, including PS-PL I/O; measured mean on physical FPGA prototype | 800 ns | Xilinx Versal VMK180 evaluation board; `PHYSICAL_FPGA_SYNTHETIC_INPUT_MEASURED` | not a QPU closed loop |
| `gnn_d7_max_latency` | rotated surface code; d=7; input graph capped at N=30 nodes | one bounded graph inference through all GNN layers; synthesis cycle count at N=30 worst supported graph size | 988.8 ns | AMD/Xilinx Alveo U250 xcu250-figd2104-2L-e; `FPGA_SYNTHESIS_ESTIMATE_NOT_BOARD_MEASURED` | input-graph filtering discards graphs above N=30 and is part of the task definition |
| `gnn_d7_average_latency` | rotated surface code; d=7; graph-size distribution after input-graph filtering | one bounded graph inference through all GNN layers; mean over the paper graph-size distribution | 846 ns | AMD/Xilinx Alveo U250 xcu250-figd2104-2L-e; `FPGA_SYNTHESIS_ESTIMATE_NOT_BOARD_MEASURED` | mean latency is not a maximum-latency guarantee |
| `rethink_tcn_d9_hls` | rotated surface code; d=9; causal sliding window r=d; incremental one-frame update | incremental Conv2D encoder plus Conv1D temporal stage plus readout; HLS synthesis total; module table is primary cycle locator | 770 ns | AMD Versal Premium VP1902; `HLS_SYNTHESIS_ESTIMATE_NOT_PLACE_ROUTE_OR_BOARD` | HLS synthesis is not place-and-route or physical-board evidence |
| `bp_osd_surface_d9` | surface code detector-error-model graph; d=9; d syndrome rounds | OSD worst case when all columns are processed, for the complete d-round graph; worst-case implementation latency; OSD invoked only after BP failure | 134000 ns | AMD/Xilinx VCU129; `FPGA_IMPLEMENTATION_WITH_HARDWARE_EMULATOR_NOT_QPU` | worst-case OSD-only latency is not average full BP+OSD latency |
| `bp_osd_bicycle_d12` | bivariate bicycle qLDPC detector-error-model graph; d=12; one detector type as scoped by the paper | OSD worst case when all columns are processed; worst-case implementation latency; OSD invoked only after BP failure | 84000 ns | AMD/Xilinx VCU129; `FPGA_IMPLEMENTATION_WITH_HARDWARE_EMULATOR_NOT_QPU` | worst-case OSD-only latency is not average full BP+OSD latency |
| `deconet_100logical_d5` | surface code with lattice-surgery network integration; 100 logical qubits, d=5, five VMK180 FPGAs | one 100-logical-qubit decoding task; mean task latency; inverse throughput reported separately per measurement round | 2400 ns | five Xilinx Versal VMK180 boards; `PHYSICAL_MULTI_FPGA_SYNTHETIC_INPUT_MEASURED` | 2.40 us mean latency and 0.84 us inverse throughput are different metrics |
| `ced_d9_tail` | rotated surface code; d=9; full d-round task; K=24 ensemble candidates | complete d-round decoding task; optimized latency distribution p95 and p99 | p95 650 ns; p99 900 ns | AMD/Xilinx Virtex UltraScale+ VU19P; `FPGA_TOOL_REPORT_AND_CYCLE_SIMULATOR_NOT_BOARD_MEASURED` | 108k LUT, 43k FF and 252 BRAM are reported at d=15 and are not copied into this d=9 row |
| `gari24_gross_d12` | [[144,12,12]] bivariate bicycle qLDPC gross code; d=12; 24 NMS decoders; up to 400 iterations each | per decoding round; mean; 99.99% of instances below 1 us | 273 ns | one AMD/Xilinx VU19P per ensemble decoder in preliminary implementation; `PRELIMINARY_FPGA_IMPLEMENTATION_NOT_QPU` | preliminary implementation wording is preserved |
| `gari3_gross_d12` | [[144,12,12]] bivariate bicycle qLDPC gross code; d=12; ensemble of three decoder cores | per decoding round; mean for the three-core architecture | 596 ns | AMD/Xilinx VU19P; `FPGA_IMPLEMENTATION_REPORTED_NOT_QPU` | the source claims six-times lower resources than the earlier GARI proposal but exact absolute resources are not imputed from the ratio |

## 未进入数值行的已检索候选

- `QASBA` — `EXCLUDED_NO_ABSOLUTE_TASK_BOUNDARY_EXTRACTED`：Primary abstract supports up to 25.05x speedup and 304.16x energy-efficiency gain versus its software baseline, but no source-backed absolute latency row with exact problem size/boundary was extracted; relative values are not converted into fabricated nanoseconds.
- `QUEKUF` — `EXCLUDED_NO_PRIMARY_ABSOLUTE_LATENCY_ROW_EXTRACTED`：The public paper/repository was found, but numerical resource values seen in CED are secondary and normalized; they are not promoted to a primary-source row.
- `SOFT_SYNDROME_QLDPC` — `EXCLUDED_FAMILY_SUMMARY_NOT_ROW_SPECIFIC`：The paper reports about 600 ns for 30 iterations across five QLDPC codes and provides implementation tables, but the frozen extraction did not obtain a row-specific absolute table cell; the family summary is retained as coverage rather than copied to five false-exact rows.
- `DIVERSITY_METHODS_EMULATOR` — `EXCLUDED_EMULATOR_THROUGHPUT_NOT_DECODER_BOUNDARY`：The FPGA is used as a high-throughput error-pattern emulator for decoder research; the source does not define one comparable fixed decoder source-to-action row for this atlas.
- `CED_D15_RESOURCE_ONLY` — `EXCLUDED_FROM_D9_ROW_DIFFERENT_PROBLEM_SIZE`：The source reports d=15 resources (108k LUT, 43k FF, 252 BRAM) but evaluates latency only through d=11; these values are documented but not merged into the d=9 tail-latency row.
- `GKP_SPECIFIC_FPGA` — `NO_QUALIFYING_PRIMARY_SOURCE_IDENTIFIED_BY_FROZEN_SEARCH`：No concrete GKP FPGA implementation with an absolute timing boundary was identified. This negative search result does not establish nonexistence.

## 不能从本表推出的结论

- surface-code、qLDPC 与 single-mode GKP 的纳秒数不能直接排序。
- synthesis/HLS/P&R estimate 不能和 physical FPGA/QPU closed-loop measurement 混排。
- mean、p95/p99、worst、inverse throughput、II、per-round amortization 与 source-to-action 不能互换。
- CED 的 1.2 W 只是 24 个 EFE branch 的动态功耗项，不是整机总功耗；d=15 资源不能填进 d=9 tail row。
- Rethink TCN 的正文 271 cycles 与附录表 267 cycles 冲突保持显式，不以 0.77 us 反推并假装一致。

## 机器验证

- gates：`18/18`；mutations：`18/18`。
- same-task external comparator：`0`。
- verdict：`PASS_EXTERNAL_FPGA_REFRESH_ZERO_SAME_TASK_NO_SPEED_CLAIM`。
