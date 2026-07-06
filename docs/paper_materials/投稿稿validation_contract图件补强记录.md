# 投稿稿 validation-contract 图件补强记录

## 目的

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。本次补强把已经存在的三类软件证据汇总为一张正文图件 `fig05_validation_contract.pdf`，帮助审稿人快速看到 affine fast path 的三个当前支撑面：

- analytical operation count：低复杂度 fast-path 设计理由；
- Q4.20 software fixed-point parity：固定点数值退化边界；
- software-in-the-loop runtime counters：stage-and-commit 软件协议的可观测 counters。

## 修改内容

- 更新 `docs/figure_assets/submission_draft_python_figures/build_submission_draft_figures.py`，新增 `fig05_validation_contract()`。
- 新增/生成 `docs/figure_assets/submission_draft_python_figures/source_data_fig05_validation_contract.csv`。
- 更新 `figure_manifest.json` 和 `figure_source_map.csv`，把 Fig. 5 纳入图件包。
- 更新 `docs/figure_assets/submission_draft_python_figures/audit_submission_draft_source_data.py`，使 source-data audit 覆盖 Fig. 5 的 source-data manifest。
- 在投稿稿正文 runtime-discipline 表后插入 Fig. 5。
- 在投稿稿稿末 `Figure Source Data and Status` 表中补登 Fig. 5 的 source-data 与软件验证边界。

## Source Data

Fig. 5 不运行新实验，只汇总已有 paper-material CSV：

- `docs/paper_materials/submission_draft_fast_path_cost_model.csv`
- `docs/paper_materials/submission_draft_fixed_point_parity_analysis.csv`
- `docs/paper_materials/submission_draft_runtime_discipline_summary.csv`

派生图件 source-data：

- `docs/figure_assets/submission_draft_python_figures/source_data_fig05_validation_contract.csv`

## 可写边界

可以写：

- affine fast path 在 analytical count 中比 one-step wrapped-Gaussian references 更轻；
- Q4.20 software emulation 在受控样本内没有改变 residual-boundary crossing rate；
- 当前 software-in-the-loop protocol 暴露了 commit、overflow、cycle-violation 和 saturation counters。

不能写：

- 已完成 FPGA synthesis、timing closure、resource/power measurement；
- 已完成 source-vs-board agreement；
- 已测得 board commit latency 或 hardware reliability；
- Fig. 5 提供新的 benchmark、CI、p-value 或 hardware validation。

## 验证

本记录要求后续 verification 至少包含：

- 运行 `python docs\figure_assets\submission_draft_python_figures\build_submission_draft_figures.py` 重新生成图件和 manifest；
- 运行 `python docs\figure_assets\submission_draft_python_figures\audit_submission_draft_source_data.py`；
- 运行 `python docs\figure_assets\submission_draft_python_figures\audit_submission_draft_symbol_boundary.py`；
- 编译 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。
