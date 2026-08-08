# Integrated Route-A 真板前三 seed P&R / Pareto

T6.9.1 将 T6.7.3 已做百万周期 bit-exact 资格验证的 `gkp_fast_path_core + route_a_policy_overlay` 接入同一个小引脚 synthesis harness，并在 GOWIN `GW2AR-LV18QN88C8/I7` 上对两个真实 elaboration profile 各执行 seeds `1/7/19` 的 open-source synthesis/P&R。

这不是资源相加代理：两个 profile 从同一 parameterized top 分别产生网表；报告保存并哈希绑定两份 8.9/10.9 MB 综合网表、两份 synthesis log 和六组 nextpnr report/log。结构审计要求 policy/core hierarchy、8 个 A/B MAP BSRAM 和 optional student DSP 均真实存活。

## 三 seed 结果

| Profile | Fmax min / median / max (MHz) | 最大 LUT4 | 最大 DFF | BSRAM | MULT18 / MULT9 | ALU | 六周期 @27 MHz | 1.5 us margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_a_core_no_student` | 39.137 / 40.264 / 40.402 | 3859 (18.61%) | 1069 (6.87%) | 8 (17.39%) | 1 / 1 | 448 | 222.222 ns | 1.27778 us |
| `route_a_plus_student_sidecar` | 38.506 / 39.348 / 39.474 | 4889 (23.58%) | 1210 (7.78%) | 8 (17.39%) | 2 / 1 | 718 | 222.222 ns | 1.27778 us |

六个 P&R run 全部达到 27 MHz。每条关键路径都从 `integrated.core.*` 状态寄存器到注册的 `fold*` 可观测性端点，因而 Fmax 是包含 core telemetry/state CRC 与 small-pin fold 包装的保守 full-observability estimate；它不是实物板 source-to-action 测量，也不能给出 deadline miss 或 jitter。

## Optional student 的角色

打开 student sidecar 后，网表 cell 数从 6431 增至 8316，LUT4 增加 1030、DFF 增加 141、MULT18 增加 1，证明真实 student kernel 没有被综合器删掉。student 的完整 update 为 64 cycles，即 27 MHz 下约 2.370 us；它不驱动六周期 fast action，健康异常时由 Route-A action 关闭/复位。因此：

- 论文主硬件点选择 `route_a_core_no_student`，对应 MAP-based fast path + contract/FSM safety；
- student profile 只作为可替换 learning sidecar / ablation 的真实资源证据；
- 不能把 student 资源结果当成 CNN primary evidence，也不能把 64-cycle sidecar latency 混入六周期 action latency。

## 动态功耗敏感性，而非 power signoff

nextpnr/Gowin open-source flow 不提供厂商校准 power signoff。报告使用明确公式

\[
P_{\mathrm{dyn}}[\mathrm{mW}]
=C_{\mathrm{sw}}[\mathrm{pF}]\,V^2\,f[\mathrm{MHz}]\times 10^{-3}
\]

和逐资源有效电容、clock-tree 电容、activity factor、capacitance scale 假设，给出宽敏感性区间：

- no-student：`2.04 / 12.25 / 48.98 mW`（low / nominal / high）；
- student sidecar：`2.39 / 14.33 / 57.31 mW`。

这些值只用于设计敏感性。`static_power_mw`、`vendor_power_mw`、`board_measured_power_mw` 全部为 `null`，不得与外部论文功耗排名，也不得写成设备总功耗。

## 证据边界与下一门

- 已支持：synthesizable integrated top、six-cycle architecture、three-seed open-source P&R、resource/Fmax estimate、analytic power sensitivity。
- 未支持：vendor timing signoff、bitstream、transport、板上 latency/jitter/deadline/resource readback/power、FPGA speed advantage。
- T6.9.2 必须在真实板上用同一 source/bitstream hash 做至少 `10^6` cycles，并把 core/transport/source-to-action/end-to-end 分层；否则所有 measured 字段继续为 `null`。

## 复核

```powershell
python -m cnn_fpga.benchmark.route_a_hardware_pareto --verify docs/t6_9_1_route_a_hardware_pareto.json
python -m pytest tests/test_route_a_hardware_pareto.py -q
```

当前 verdict 为 `PASS_ROUTE_A_INTEGRATED_THREE_SEED_PR_ESTIMATE_NOT_BOARD_MEASURED`；15/15 gates、15/15 semantic mutations、6 focused tests 通过。
