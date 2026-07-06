# 投稿稿 row-level provenance 补强记录

## 目的

本记录为当前投稿稿主 benchmark 行补充 row-level source trace。它只读取既有 T24 software-HIL 运行的 summary、launch plan、comparison.csv、配置和 runner 文件；不重新运行 benchmark，也不补硬件或统计推断证据。

## 输出

- `docs/paper_materials/submission_draft_row_provenance_manifest.csv`
- `docs/paper_materials/submission_draft_row_provenance_manifest.json`

## 覆盖范围

- row_count = 40
- scenarios = `static_bias_theta`, `linear_ramp`, `step_sigma_theta`, `periodic_drift`
- modes = `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`
- repeats = `0`, `1`

## 不可外推边界

- not new benchmark evidence
- not CI, p-value, standard error or robustness evidence
- not holdout-drift validation
- not training reproducibility closure
- not tflite portability or HIL closure
- not real-board execution, source-vs-board agreement, latency, resource or FPGA evidence
- not statistical-calibration comparator promotion

## 明确缺口

- 本 manifest 不递归 hash `hil_events.json`，也不是 historical run directory 的完整 hash closure。
- 本 manifest 不包含 board log、bitstream/RTL hash、DMA/MMIO trace、source-vs-board vector、latency/resource/power measurement。
