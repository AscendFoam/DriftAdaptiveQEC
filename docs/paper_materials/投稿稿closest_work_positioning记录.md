# 投稿稿 closest-work positioning 记录

日期：2026-07-06

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 中的最近邻工作定位表。

## 生成文件

- `docs\paper_materials\submission_draft_closest_work_positioning.csv`
- `docs\paper_materials\submission_draft_closest_work_positioning.json`

## 边界

- 可以写：本稿相对 analog GKP、calibration-aware learned decoders、logical-channel analyses 和 real-time FPGA decoders 的定位差异。
- 不能写：这些外部指标是本稿复现实验结果，或本稿已完成 real-device learned-decoder、finite-energy tomography、FPGA synthesis、resource/power、real-board latency 或 source-vs-board validation。

## 行摘要

| Row | Boundary |
| --- | --- |
| `analog_surface_gkp` | not an end-to-end surface-GKP threshold or overhead result |
| `calibration_learned_decoders` | not a real-processor learned-decoder experiment |
| `logical_channel_gkp` | not calibrated finite-energy logical-channel fidelity or process tomography |
| `runtime_predecoders` | not measured neural-decoder or FPGA runtime |
| `real_time_fpga_decoders` | not FPGA synthesis, resource use, power, real-board latency or source-vs-board agreement |
