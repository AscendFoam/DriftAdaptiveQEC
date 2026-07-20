# T6.8.6：FPGA QEC decoder 任务、时延与资源规范化

## 结论

截至 2026-07-18，规范化表中有 8 个外部具体实现、2 个本项目证据层级行，但 **0 个外部行满足与本项目直接比较速度的条件**。原因不是外部工作不够快，而是 code family、输入/动作语义、问题规模、时延边界、统计量和硬件证据等级均未同时对齐。

因此当前只允许写：本项目 Route-A 在预板 CXXRTL 中具有确定性的 6-cycle source-to-action 架构。不得写 `fastest`、`SOTA`、`faster than existing FPGA decoders`；真实 source-to-action 数值仍等待 T6.9.1 integrated P&R 和 T6.9.2 板测。

## 规范化规则

每行对应一个具体实现和问题规模，不把同一论文的不同尺寸挑成一个“最佳数字”。所有数值均绑定一手论文的表、正文或本地 hash-bound 报告；未报告、边界不等价或无法无歧义换算的字段均为 JSON `null`。

速度只有在下列字段全部一致时才允许排序：

1. code family 与 logical task；
2. syndrome 输入和 action 输出语义；
3. problem size、window/round 数与噪声模型；
4. core、average-per-round、source-to-action 或 closed-loop 边界；
5. deterministic、mean、worst 或 post-implementation estimate 统计口径；
6. precision、clock、资源和 FPGA/ASIC/仿真/实机证据等级。

## 主表

| 行 | 任务与规模 | 主要时延 | 时延证据 | 资源摘要 | QPU闭环 | 对本项目直接可比 |
| --- | --- | --- | --- | --- | --- | --- |
| LILLIPUT d5,m2 | surface code LUT，2-round window | 42 ns core，7 cycles，232.9 MHz | post-implementation estimate | 246 logic units，486 FF；baseline 148 MB external | 否 | 否 |
| Helios d21 | distributed UF surface-code decoder | 11.5 ns **mean per measurement round**，100 MHz | VCU129，1e6 synthetic trials | 898,715 LUT，238,939 registers，0 BRAM/DSP | 否 | 否 |
| Collision Clustering d21 | surface code，881 qubits | 810 ns **mean normalized per round**，405 MHz | FPGA synthetic shots | 17,237 LUT，11,957 FF，2 RAMB36 + 2 RAMB18 | 否 | 否 |
| Local Clustering d17 adaptive-HL | adaptive surface-code clustering | 676 ns reported per round，285 MHz | implementation execution-time table | 251,963 LUT，252,736 FF | 否 | 否 |
| Overwater NN d5 | 64/64 FCNN + pure-error decoder | 87.6 ns core estimate | Artix-7 post-implementation estimate | 44,670 LUT，132 mW estimate，4 bit | 否 | 否 |
| Caune stability-8 | 9-round real-time feedback | 9.6 us full response = 6.5 us decode + 3.1 us communication/control | real Ankaa-2 QPU | 未报告，均为 `null` | 是 | 否 |
| Maurer gross-code X | [[144,12,12]] Relay-BP, 12-cycle window | 24 ns/iteration；480 ns mean window；40 ns mean/cycle | FPGA synthetic stream + verification harness | 2,106,738 LUT，540,767 FF，29.5 BRAM，58.025 W | 否 | 否 |
| Yang NN d3 | 32-unit LSTM real-time surface code | 124 ns NN；184 ns throughput period；550 ns deterministic closed loop | real superconducting QPU | per X/Z decoder: 5.63% LUT，4.56% FF，399 DSP/25.91%，6 bit | 是 | 否 |
| Project T5 core | single-mode GKP MAP/event fast path | 6 cycles，222.222 ns @ 27 MHz，II 37.037 ns | 3-seed preboard P&R estimate | seed-1 3,357 LUT4，865 DFF，8 BSRAM | 否 | 项目参考行 |
| Project integrated Route-A | GKP regime-aware safe adaptive path | 6 cycles；所有 ns/Fmax/resource/power 为 `null` | 1e6-cycle CXXRTL + structural synthesis only | 待 T6.9.1 | 否 | 项目参考行 |

## 时延边界为何不能混排

- Helios 的 11.5 ns 是总执行时间除以 21 个 measurement rounds 的均值，论文同时说明分布有长尾；它不是 11.5 ns source-to-action。
- LILLIPUT 的 42 ns 是 syndrome arrival 到 error assignment 的固定 decoder path，未包含真实量子控制链。
- Maurer 的 24 ns 是一次 BP iteration；480 ns 才是其 p=0.1% 下约 20 次收敛的平均 12-cycle window 时间。
- Caune 的 9.6 us 与 Yang 的 550 ns 才包含真实 QPU/control feedback 边界，但二者分别是 stability-8 九轮和 d3 surface-code 单轮，不是 single-mode GKP 的同任务结果。
- 本项目 T5 的 222.222 ns 排除了 ADC、transport、CDC 和 actuation；T6.7.3 只验证 6-cycle 对齐，没有 integrated P&R，因此不能用 host/CXXRTL wall-clock 换算 FPGA ns。

## 反简化审计

- source ledger 固定 10 个一手来源/本地原始报告，前四项正式版本/DOI 已在冻稿日刷新；
- 23 个数值字段逐单元检查：非 `null` 必有 source locator，`null` 不允许附带伪 locator；
- 13/13 integrity gates 与 13/13 target-specific semantic mutations 通过；
- mutation 覆盖去掉 source locator、把 `null` 改成 `NR`、把 synthetic 标成 QPU、把跨 code 行设成 comparable、伪造本项目 integrated Fmax、直接声称 fastest 等捷径；
- 5 个 focused tests 通过；Source Data 为 10 行 CSV 并由 SHA-256 绑定。

## 一手来源

- [LILLIPUT, ASPLOS 2022](https://doi.org/10.1145/3503222.3507707)
- [Helios, FCCM 2023](https://doi.org/10.1109/FCCM57271.2023.00045)
- [Collision Clustering, Nature Electronics](https://doi.org/10.1038/s41928-024-01319-5)
- [Local Clustering, Nature Communications](https://doi.org/10.1038/s41467-025-66773-x)
- [Overwater et al., arXiv:2202.05741](https://arxiv.org/abs/2202.05741)
- [Caune et al., arXiv:2410.05202](https://arxiv.org/abs/2410.05202)
- [Maurer et al., arXiv:2510.21600](https://arxiv.org/abs/2510.21600)
- [Yang et al., arXiv:2605.04892](https://arxiv.org/abs/2605.04892)

## 机器产物

- `configs/literature/t6_8_6_fpga_decoder_sources.json`
- `cnn_fpga/benchmark/fpga_decoder_normalization.py`
- `docs/t6_8_6_fpga_decoder_normalization.json`
- `docs/t6_8_6_fpga_decoder_normalization_source_data.csv`
- `tests/test_fpga_decoder_normalization.py`

