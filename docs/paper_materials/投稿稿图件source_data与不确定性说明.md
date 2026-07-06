# 投稿稿图件 source-data 与不确定性说明

## 目的

本文件记录 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 当前独立投稿稿的图件、source-data 与不确定性边界。它只服务投稿稿整理，不运行新实验，也不改变 `T24`、FR6/FR7、FR8/statcalib、`.tflite` 或 real-board 的证据等级。

## 当前图件包

- 图件目录：`docs/figure_assets/submission_draft_python_figures/`
- 生成脚本：`build_submission_draft_figures.py`
- 后端：Python / Matplotlib
- 图件输出：`outputs/*.pdf` 与 `outputs/*.png`
- 图件清单：`figure_manifest.json`
- 来源映射：`figure_source_map.csv`

## Source-data 文件

| 文件 | 对应图 | 来源 | 当前可写边界 |
| --- | --- | --- | --- |
| `source_data_fig02_main_results.csv` | Fig. 2 main software-HIL results | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` | `final_ler_mean` + descriptive SD；`n=2`，不是 CI 或显著性检验 |
| `source_data_fig02_paired_deltas.csv` | Fig. 2 paired UKF-vs-Hybrid deltas | T24 `summary.json/raw_rows` paired repeats | paired delta source data；只能写 directional consistency，不能写成 CI |
| `source_data_fig03_ablation_mechanism.csv` | Fig. 3 ablation / mechanism | 当前投稿稿中的 ablation 与 mechanism 表格 | 只能写 feature sensitivity 和 descriptive intervention；不能写 causal closure |
| `source_data_fig04_statcalib.csv` | Fig. 4 statcalib extension lane | 当前投稿稿中的 statcalib 表格 | 只能写 extension lane / no-promotion；不能写 mature comparator |
| `submission_draft_paired_uncertainty_analysis.csv` | Table `tab:paired-uncertainty` | `source_data_fig02_paired_deltas.csv` 派生 | non-inferential paired-bootstrap envelope；不是 confidence interval、standard error、p-value 或 robustness proof |

## 当前图号约定

- Fig. 1：dual-loop architecture schematic；无数值结果，只是方法与边界示意。
- Fig. 2：main software-HIL results；对应 `source_data_fig02_main_results.csv`。
- Fig. 3：ablation and mechanism；对应 `source_data_fig03_ablation_mechanism.csv`。
- Fig. 4：statistical-calibration extension lane；对应 `source_data_fig04_statcalib.csv`。
- Fig. 5：hardware-facing validation placeholder；当前没有 source data，不能写成 measured figure。

## 不确定性口径

当前 T24 主结果每个 scenario/mode 只有 `completed_repeats = 2`。因此：

- 可以画 `mean +/- descriptive SD`；
- 可以报告 UKF-minus-Hybrid paired deltas 和 `n=2` non-inferential paired-bootstrap envelope；
- 可以说 “the ranking is consistent across the frozen set”；
- 不应说 “statistically significant”；
- 不应把 SD 写成 confidence interval；
- 不应把当前图件写成 broader drift-family robustness；
- 若投稿前要强化可信度，需要额外任务把 descriptive paired envelope 升级为预声明 inferential uncertainty protocol，并补更多 repeats、holdout drift family 或更强 theoretical/oracle baseline。

## 仍缺材料

1. inferential paired CI / bootstrap/paired-difference 设计和更多 repeats；当前只有 descriptive paired envelope；
2. 更高 repeat 数或预注册 stopping rule；
3. unseen drift family / holdout protocol；
4. hardware timing、fixed-point degradation、resource report 和 source-vs-board vectors；
5. final BibTeX / DOI / arXiv 元数据核对。

## 禁止外推

- 不把 Fig. 2 写成 paper-grade expanded benchmark；
- 不把 Fig. 3 写成 teacher necessity 或 causal mechanism proof；
- 不把 Fig. 4 写成 statcalib promotion；
- 不把 hardware placeholder 写成 real-board result；
- 不把 source-data 文件本身写成新实验。
