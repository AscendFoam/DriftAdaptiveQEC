# 投稿稿 metric triangulation 主文补强记录

日期：2026-07-06

## 变更对象

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

## 变更内容

- 在 `Metric-readiness matrix` 后补充 metric triangulation 读法：
  - `final_ler_mean` 检查 drift 条件下 protocol-defined boundary-crossing rate；
  - residual-boundary surrogate 只把同一 q/p event surface 转成 channel language，不升级为 finite-energy logical-channel fidelity；
  - operation-count / fixed-point parity 检查 affine correction rule 是否具有 future real-time datapath 的尺寸和数值稳定性。
- 在 Discussion 的 comparative-advantage 段落中把优势主张改成三条 non-substitutable tests：
  - software benchmark 的 drift-adaptive error-proxy advantage；
  - residual-boundary channel-language bridge；
  - analytical / software implementation-feasibility checks。

## 可写边界

- 可以写成：本文用 `final_ler_mean`、residual-boundary surrogate 和 fast-path feasibility checks 三条互补证据链来支撑 interface-level advantage。
- 不可以写成：surrogate fidelity 等于 finite-energy logical-channel fidelity。
- 不可以写成：analytical operation count 或 Q4.20 software parity 等于 FPGA 实测延迟、resource、power 或 source-vs-board agreement。
- 不可以写成：这次补丁新增了 benchmark、CI、p-value、holdout robustness、hardware validation 或 full reproducibility。

## 验证计划

- 重新生成 `submission_draft_source_data_manifest.csv/json`。
- 运行 `audit_submission_draft_source_data.py`。
- 重新编译 `CNN_FPGA_GKP_submission_draft.tex`。
- 扫描内部项目语、hardware/statistical/logical-channel overclaim 语和 LaTeX log。
