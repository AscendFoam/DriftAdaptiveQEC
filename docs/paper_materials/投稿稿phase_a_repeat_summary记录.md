# 投稿稿 Phase A repeat summary 记录

日期：2026-07-06

本文档汇总 `runs/paper_submission_phase_a` 下已经完成的 Phase A repeat runs。它只做 source-data 汇总和边界登记，不运行 benchmark，不报告 CI / p-value，也不把 smoke-length rows 升级成主文性能证据。

## 生成文件

- `docs\paper_materials\submission_draft_phase_a_repeat_summary.csv`
- `docs\paper_materials\submission_draft_phase_a_repeat_summary.json`

## Completed rows

| Row type | Lane | Scenario | Paired repeats | Expected repeats | Selected range | Mean delta | Positive pairs | Boundary |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| `cumulative_by_scenario_lane` | `formal_length_phase_a_candidate` | `linear_ramp` | 12 | 12 | `aggregate` | 0.022416805556 | 12/12 | cumulative formal-length source data only; not repeat-expanded evidence until all expected pairs complete and separate paired interval analysis passes |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `0-1` | 0.020794722222 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `1-2` | 0.022285555556 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `2-3` | 0.022004722222 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `3-4` | 0.024568055556 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `4-5` | 0.024946388889 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `5-6` | 0.019950555556 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `6-7` | 0.024233055556 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `7-8` | 0.022746388889 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `8-9` | 0.022298333333 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `9-10` | 0.020249444444 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `10-11` | 0.024840000000 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `linear_ramp` | 1 | 12 | `11-12` | 0.020084444444 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `cumulative_by_scenario_lane` | `formal_length_phase_a_candidate` | `static_bias_theta` | 12 | 12 | `aggregate` | 0.015563263889 | 12/12 | cumulative formal-length source data only; not repeat-expanded evidence until all expected pairs complete and separate paired interval analysis passes |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `0-1` | 0.013953333333 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `1-2` | 0.014984444444 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `2-3` | 0.015146944444 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `3-4` | 0.015711944444 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `4-5` | 0.015884444444 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `5-6` | 0.016308611111 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `6-7` | 0.018355000000 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `7-8` | 0.014111666667 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `8-9` | 0.014656944444 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `9-10` | 0.016391388889 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `10-11` | 0.014732222222 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `per_run` | `formal_length_phase_a_candidate` | `static_bias_theta` | 1 | 12 | `11-12` | 0.016522222222 | 1/1 | candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording |
| `cumulative_by_scenario_lane` | `smoke_length_feasibility` | `static_bias_theta` | 12 | 12 | `aggregate` | 0.024392534722 | 12/12 | cumulative smoke-length source data only; not manuscript performance evidence |
| `per_run` | `smoke_length_feasibility` | `static_bias_theta` | 12 | 12 | `0-12` | 0.024392534722 | 12/12 | smoke-length feasibility only; not manuscript performance evidence |

## 可写边界

- 可以写：collector 会把已完成 Phase A rows 汇总为 scenario-level paired deltas，并保留 source summary hash。
- 可以写：smoke-length rows 只证明 command shape、missing-row accounting 和 source-data collector 可工作。
- 不能写：smoke-length rows 是主文性能证据、expanded benchmark、robustness proof、statistical significance 或硬件证据。
