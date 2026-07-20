# T6.18.1 AQEC/autonomous 共同 wall-clock replay

- verdict：`PASS_PROJECT_NATIVE_AQEC_WALLCLOCK_WITH_OFFICIAL_PROTOCOL_BLOCKED`
- cells / seed clusters：6 / 144
- cycle-vs-wall-clock ordering-reversal cells：6/6
- gates / mutations：16/16 / 16/16

## Project-native matched replay

| cell | feedback / idle lifetime | autonomous / idle lifetime | autonomous / feedback (us) | reversal seeds |
| --- | ---: | ---: | ---: | ---: |
| `cutoff12_high` | 0.2604 [0.2589, 0.2619] | 0.2458 [0.2449, 0.2466] | 0.9438 [0.9418, 0.9458] | 24/24 |
| `cutoff12_medium` | 0.2397 [0.2379, 0.2416] | 0.2170 [0.2152, 0.2188] | 0.9051 [0.9045, 0.9057] | 24/24 |
| `cutoff12_low` | 0.2181 [0.2155, 0.2205] | 0.1961 [0.1937, 0.1984] | 0.8992 [0.8986, 0.8997] | 24/24 |
| `cutoff16_high` | 0.4766 [0.4738, 0.4792] | 0.4072 [0.4058, 0.4085] | 0.8546 [0.8525, 0.8567] | 24/24 |
| `cutoff16_medium` | 0.4465 [0.4426, 0.4500] | 0.3631 [0.3596, 0.3662] | 0.8131 [0.8123, 0.8138] | 24/24 |
| `cutoff16_low` | 0.4086 [0.4046, 0.4127] | 0.3294 [0.3259, 0.3329] | 0.8061 [0.8055, 0.8067] | 24/24 |

六个 cell 中 measurement/reset 与 autonomous/reset 相对 idle 的 lifetime 95% CI 上界都低于 1，故项目原生结果是明确负结果；这只说明当前 fixed-nominal-control finite-cutoff 模型不产生 AQEC lifetime gain，不反驳论文装置的 reservoir-engineered experimental result。autonomous 每 100 us 避免 20 次 measurement，但 reset 从 20 增至 28.5714、active gates 从 180 增至 257.143。

主 lifetime 使用完整曲线的 area-equivalent 定义。全体 raw trace 的 logical-Z exponential-fit R² 为 0.181--0.954，code-survival R² 为 0.047--0.685，并存在非单调点；这些 fit 只作为诊断保留，未挑选时间窗或替换主指标。

每个 seed 是在查看 T6.18.1 结果前固定的 5% log-scale、mean-preserving quasi-static lifetime realization；idle、measurement/reset 与 autonomous/reset 共用同一 realization。置信区间只表示该项目 nuisance model 下的 24-cluster paired bootstrap，不是装置误差条。

## Evidence boundary

现有 simulator 使用 fixed nominal controls、instantaneous gates、analytic idle CPTP maps 与 trace-reset；它不是 Lachance 2024 的 dissipative transmon/reservoir Method A/B。论文的 1.14(18)/1.14(16) lifetime gains 保持 `LITERATURE_ONLY`，official protocol reproduction 为 `BLOCKED`。classical decoder latency 对 AQEC 是 N/A 而非 0；pulse energy 与 control-duty 未建模，保持 null。

## Artifacts

- `docs/t6_18_1_aqec_common_wallclock_replay.json`
- `docs/t6_18_1_aqec_common_wallclock_source_data.csv`
- `cnn_fpga/benchmark/aqec_secondary_wallclock_replay.py`
