# T6.19.3 六条 lane 非主排名对比图谱

- 完整性结论：`PASS_AUX_COMPARISON_INTEGRITY`
- 定位：只读、辅助、within-lane；没有全局分数或跨 lane 冠军。
- Phase 6B：保持 `NO_GO_V5_EARLY_HEADROOM_STOP`；T6.9.2 仍等待真板。

## 图谱合同

每个单元都绑定 source report、原始 Source Data 与冻结配置的 SHA-256；数值、CI、threshold、timing 和计数均从原始数据重算。

| lane | cells | 结论边界 |
|---|---:|---|
| `single_mode_decoder` | 5 | Route A 未胜 static MAP；oracle 仅上界 |
| `surface_gkp_gate_outer_code` | 7 | 同任务 Noh CNOT 复现中 ML 低于 CI |
| `multimode_structured_lattice_cpd` | 7 | official threshold 与 project drift 为不同 signature；均只在各自 signature 内解释 |
| `controller_rl_nmf` | 3 | 16 个 learned family 无同任务可排名项；GQF exact blocked |
| `aqec_wallclock` | 13 | 六个 common-wall-clock cell 的 active-QEC lifetime ratio 均低于 1 |
| `fpga_implementation` | 171 | 六周期/II=1 与资源均为仿真/布局布线估计；18 个外部实现中同任务为 0 |

## 统计与图形 QA

- Noh CNOT：paired Monte Carlo；每点 Wilson 95% CI；trial 数见 Source Data。
- Multimode：32 个 seed cluster；aggregate p_L 由 error/cycle 原始计数重算。
- AQEC：每 cell 24 个 seed cluster；20,000 次 cluster bootstrap 95% CI。
- FPGA：3 个 P&R seed；27 MHz 下 6-cycle latency 与 II=1 换算；非板测。
- Python/matplotlib 单后端；SVG 保留 editable text，同时导出 PDF/TIFF/PNG。

## 产物

- `docs/t6_19_3_secondary_evidence_integrity_source_data.csv`
- `docs/figures/t6_19_3_secondary_comparison_atlas.svg`
- `docs/figures/t6_19_3_secondary_comparison_atlas.pdf`
- `docs/figures/t6_19_3_secondary_comparison_atlas.tiff`
- `docs/figures/t6_19_3_secondary_comparison_atlas.png`

不得将该 atlas 用作跨 code、跨任务、跨 latency boundary 或 estimate-vs-measured 的速度/性能排名。
