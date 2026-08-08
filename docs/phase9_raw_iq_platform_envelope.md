# T9.2.6 raw-IQ 前端与候选平台交集 envelope

- 状态：`PASS_T9_2_6_RAW_IQ_PLATFORM_ENVELOPE_FROZEN`（仅协议/接口冻结 PASS）
- analysis：`b10a035f8d5134a2699b3611d28dea184dd797917d0d510f899547103a233b8d`
- T9.2.4 双后端结论仍是 `NO_GO_TWIN_QUALIFICATION`；本任务没有使用失败 twin 的性能值选阈值。
- `threshold_values`、frontend ROC/LER、recorded/live IQ、真板 latency/resource/power 与外部 SOTA 全部保持 `null`。

## 冻结边界

- 输入：after vendor RFDC DDC/decimation or bit-exact replay adapter, before project matched filter
- 输出：legacy 58-bit CRC input of gkp_fast_path_core
- 主 AXI4-Stream：32-bit `TDATA`（I16/Q16, Q1.15）+ 128-bit `TUSER`，250 MHz 只是实现目标，不是已达时序。
- rate family：125 MS/s × 64 与 250 MS/s × 128，均为 512 ns integration window。
- CDC 至少容纳两个最大完整窗口（256 beats）；8-bit reset epoch + stateful retired-window receipt 拒绝跨 reset 旧 beat、重放和跨窗乱序。
- 首个 TVALID 后 192 个 ACLK 内必须退休 TLAST；timeout/overflow/reset/序列/version/CRC 错误均只产生 fail-closed record，禁止静默丢样和 postselection。

## 定点链

| stage | bits | Q-format |
| --- | ---: | --- |
| `input_iq` | 16 | `Q1.15` |
| `complex_matched_filter_coefficient` | 18 | `Q1.17` |
| `real_scalar_product` | 34 | `Q2.32` |
| `complex_multiply_component_sum` | 35 | `Q3.32` |
| `matched_filter_accumulator` | 48 | `Q16.32` |
| `calibration_matrix` | 18 | `Q2.16` |
| `calibration_product` | 66 | `Q18.48` |
| `calibration_component_sum_with_aligned_offset` | 67 | `Q19.48` |
| `calibration_offset` | 24 | `Q8.16` |
| `calibrated_iq_llr_threshold_hysteresis` | 24 | `Q8.16` |
| `ood_score` | 8 | `UQ0.8` |
| `legacy_fast_path_map_llr` | 22 | `implementation-defined inherited LUT code` |

所有窄化使用 round-to-nearest ties-to-even；所有溢出使用 signed saturation + sticky fault，禁止 wraparound。matched filter 结构为 `sum(conj(h[n])*x[n])`，calibration 为 versioned 2×2 affine package。阈值寄存器格式已冻结为 signed Q8.16，但数值未资格化。

## 四个不可混排 latency boundary

| boundary | cycles | II | 状态 | measured ns |
| --- | ---: | ---: | --- | ---: |
| `FAST_PATH_CORE_INPUT_TO_ACTION` | 6 | 1 | `INHERITED_PREBOARD_RTL_CYCLE_CONTRACT` | null |
| `DISCRIMINATOR_OUT_TO_ACTION` | 6 | 1 | `FROZEN_TARGET_REQUIRES_T9_2_7_AND_T9_7_1_REQUALIFICATION` | null |
| `ADC_LAST_SAMPLE_TO_TRIGGER` | null | null | `NOT_IMPLEMENTED_NOT_MEASURED_NULL` | null |
| `RAW_IQ_SOURCE_TO_TRIGGER` | null | null | `MISSING_HIGH_SPEED_BOARD_NULL` | null |

六周期/II=1 只绑定既有 fast core 与待 T9.2.7 复证的 `discriminator-out -> action`；不得迁移到 ADC/raw-IQ/trigger。

## 候选平台交集

- ZCU111/XCZU28DR 与 ZCU216/XCZU49DR 仅为 vendor-source-confirmed candidate，尚未选择、综合、P&R 或上板。
- Tang Nano 20K/GW2AR 只保留为低速数字控制参考，明确不属于 raw-IQ platform intersection。
- 冻结 budget：250 MHz target、32-bit TDATA、128-bit TUSER、≤32 DSP、≤12 BRAM36、≤25k LUT、≤30k FF；这些是设计上限，不是资源结果。

## 可执行反简化证据

- signed I/Q code roundtrip：131,072
- TUSER boundary roundtrip：18
- ties-even exhaustive conversions：262,144
- matched-filter independent arithmetic cases：516
- 66/67-bit calibration arithmetic cases：729
- nominal rate×domain windows：6/6
- explicit error flags：16/16 全部 fail closed
- structural adversarial cases：8/8 全部拒绝
- stateful freshness/reset/timeout：5 + mixed-epoch + deadline 全部拒绝
- timeout quarantine transaction：10/10，poison count=1
- A/B activation：成功 0，拒绝 unsafe case 9；trusted receipt 保持 null
- strict bool/int alias：16/16 全部拒绝
- legacy 58-bit adapter roundtrip：18/18
- semantic gates/mutations：36/36

## 下游

`T9.2.7` 仍被 T9.2.4 NO-GO 阻塞；下一项只释放 `T-RISK-20260726-01` fresh twin IQ/likelihood 修复与重新资格化。旧 NO-GO 不回写。
