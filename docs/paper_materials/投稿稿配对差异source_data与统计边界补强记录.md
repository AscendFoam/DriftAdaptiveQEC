# 投稿稿配对差异 source-data 与统计边界补强记录

## 作用域

本文档记录 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的一次有界补强：从既有
`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`
中的 `raw_rows` 派生 UKF 与 Hybrid-\(b\) 的逐 repeat 配对差异，并把该信息登记到投稿稿主结果段落、source-data CSV、figure manifest 和机械审计中。

本记录不是新实验，不运行 benchmark，不补 confidence interval、\(p\)-value、holdout drift family、`.tflite`、HIL 或 real-board 证据。

## 本次修改

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`
  - 主结果段落新增“paired raw rows directionally consistent”的保守表述。
  - 新增 `tab:paired-deltas`，列出四个场景中 UKF minus Hybrid-\(b\) 的 mean paired delta、relative reduction 和 minimum paired delta。
  - Fig. 2 图注改为说明 paired deltas 已在表格和 source data 中提供，但仍不构成 inferential CI 或假设检验。

- `docs/figure_assets/submission_draft_python_figures/build_submission_draft_figures.py`
  - 新增 `source_data_fig02_paired_deltas.csv` 生成逻辑。
  - 从 T24 `summary.json/raw_rows` 读取同 scenario、同 repeat、同 seed 的 UKF 与 Hybrid-\(b\) 行。
  - 图件架构文字将硬件结果表述为 planned measurements，避免把硬件面写成已完成结果。
  - Fig. 2 标题改为 predeclared simulation ranking。

- `docs/figure_assets/submission_draft_python_figures/audit_submission_draft_source_data.py`
  - 审计范围从 `submission_draft_source_data_audit_v1` 扩展到 `v2`。
  - 新增 `source_data_fig02_paired_deltas.csv` 与 `summary.json/raw_rows` 的逐 repeat 机械一致性检查。
  - 新增 `tab:paired-deltas` 与 paired source CSV 的机械一致性检查。
  - 修复 LaTeX 行注释解析，避免把 `\%` 误当作注释起点。

## 配对差异结论

| Scenario | Mean \(\Delta\) final_ler vs UKF | Relative reduction | Minimum paired \(\Delta\) |
| --- | ---: | ---: | ---: |
| `static_bias_theta` | 0.014469 | 1.75% | 0.013953 |
| `linear_ramp` | 0.023446 | 2.89% | 0.023017 |
| `step_sigma_theta` | 0.022748 | 2.80% | 0.022440 |
| `periodic_drift` | 0.015166 | 1.85% | 0.013571 |

解释方式：

- \(\Delta\) final_ler = UKF final_ler - Hybrid-\(b\) final_ler。
- 四个场景的两个 paired repeats 均为正值，说明现有 paired source rows 在方向上支持 Hybrid-\(b\) 低于 UKF。
- 由于每个 scenario/mode 仍只有 `n=2` repeats，该结果只能写成 paired descriptive improvement。

## 不能外推的边界

- 不能写成 statistical significance、confidence interval、standard error 或 \(p\)-value。
- 不能写成 unseen drift robustness、expanded benchmark 或 universal GKP decoder advantage。
- 不能写成 hardware logical error rate、real-board success、deployment closure 或 `.tflite` runtime 结论。
- 不能把 supplementary statistical calibration 升级成 mature main comparator。

## 验证

已运行：

```powershell
python docs\figure_assets\submission_draft_python_figures\build_submission_draft_figures.py
python docs\figure_assets\submission_draft_python_figures\audit_submission_draft_source_data.py
python docs\figure_assets\submission_draft_python_figures\audit_submission_draft_symbol_boundary.py
```

结果：

- `audit_submission_draft_source_data.py`: `PASS_WITH_LIMITATIONS`
- `audit_submission_draft_symbol_boundary.py`: `PASS_WITH_LIMITATIONS`

这两项验证只证明当前投稿稿 TeX 表格、source-data、figure manifest、T24 CSV/JSON 和被覆盖的符号边界在机械层面一致，不证明完整复现、强统计或硬件证据。
