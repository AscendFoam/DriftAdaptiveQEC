# T6.16.2 comparison-lane / metric / timing-resource ontology

- verdict：`PASS_FAIL_CLOSED_COMPARISON_ONTOLOGY`
- lanes/metrics/timing/resources/states：`6/46/6/5/8`
- T6.16.1 metric crosswalk：`30` rows；只有 Wang 未定义 denominator 的 decoding-rate prose 被显式拒绝。
- 核心原则：只有同 lane、同 metric、同 denominator/statistic/timing boundary、完全相同 13-field task signature 且状态可排名时，才允许 raw comparison。

## 六条 lane

| lane | decision object | ranking unit |
| --- | --- | --- |
| `single_mode_decoder` | observed syndrome/history to logical coset, correction and safety action | same trace/seed, same syndrome/action, same observability, same fixed-point and compute budget |
| `surface_gkp_gate_outer_code` | error-corrected GKP gate failure or concatenated outer-code recovery | same circuit/code family, distance set, noise law, decoder history and threshold estimator |
| `multimode_structured_lattice_cpd` | multidimensional closest lattice point or logical coset | same lattice family/dimension/noise/precision/hardware and exactness tolerance |
| `controller_rl_nmf` | history/performance to physical-control parameter update | same physical simulator/apparatus, history/action, training/search budget, horizon and selection rule |
| `aqec_wallclock` | physical autonomous/measurement-feedback memory preservation | same apparatus/model, wall-clock horizon, noise, cutoff, duty, event and control budget |
| `fpga_implementation` | concrete code/task input to concrete decoder/control output | same code/input/action/size/precision/device class/boundary/statistic/hardware evidence |

## 状态语义

`NULL_NOT_REPORTED` 是适用但没有值；`N_A_NOT_APPLICABLE` 是不适用；`FAILED` 是已执行但未过门；`NEGATIVE` 是有效 NO-GO。四者 value 必须为 null、不可排名，也不能填 0。literature/reproduced/estimate/measured 仍需分开。

## timing/resource 边界

- `decoder_core`、`update_compute`、`transport`、`source_to_action`、`closed_loop`、`initiation_interval` 分开；II 是吞吐，不是 latency。
- 允许声明 `source_to_action` 可由明确的 transport+core 路径构成、closed-loop 可包含 source-to-action，但禁止在缺少边界事件和组件测量时自行相加。
- LUT/FF/BRAM/DSP 必须带 device/primitive/tool/stage/seed-profile；power 还必须带 voltage/clock/activity/method。

## fail-closed 自检

wrong-lane、null/N/A/failed 填零、无 boundary latency、无一手依据定性复杂度、跨 denominator、跨 family、core-vs-closed-loop、null 排名均被拒绝；模块没有 global-score API。

## 产物

- `configs/literature/t6_16_2_comparison_ontology.json`
- `docs/t6_16_2_comparison_ontology.json`
- `docs/t6_16_2_comparison_ontology_source_data.csv`（101 rows）
