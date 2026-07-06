# 投稿稿 Python 图件包

## 用途

本目录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的当前投稿稿版本，用 Python/Matplotlib 把已经写入稿件的表格和方法契约转换为初版可插入图件。

## 边界

- 不运行新 benchmark、训练、`.tflite`、HIL 或 real-board 实验。
- 不把软件 HIL 均值图升级为扩展 benchmark、部署或硬件证据。
- 不把消融/机制图写成因果闭环、teacher necessity 或干预成功。
- 不把 supplementary statistical calibration 写成 main comparator、expanded benchmark 或 mature comparator。
- 不把 validation-contract 汇总图写成 FPGA synthesis、timing closure、resource/power、source-vs-board agreement 或 board commit latency。
- 硬件相关图件仍只能作为 planned measurement protocol，不作为当前结果。

## 文件

| 文件 | 作用 |
| --- | --- |
| `build_submission_draft_figures.py` | 生成图件、manifest 和 source map 的脚本 |
| `audit_submission_draft_source_data.py` | 审计当前 TeX 表格、figure source CSV、T24 `comparison.csv` / `summary.json`、controlled oracle-affine / wrapped-Gaussian CSV、sequence-controlled baseline CSV、fast-path cost CSV、metric-readiness CSV、Phase A completed-scenario paired-interval rows、source-data manifest 与图件 manifest 的机械一致性；不运行正式 benchmark |
| `audit_submission_draft_symbol_boundary.py` | 审计当前 TeX 符号、`physics/`、fast-loop、mock drift 字段与 Fig. 2 统计边界的机械一致性；不运行新实验 |
| `figure_manifest.json` | 图件包清单，运行脚本后生成 |
| `figure_source_map.csv` | 图件到数值来源和边界的追溯表，运行脚本后生成 |
| `source_data_fig02_main_results.csv` | Fig. 2 的 machine-readable source data，来自 T24 `comparison.csv` |
| `source_data_fig02_paired_deltas.csv` | Fig. 2 / `tab:paired-deltas` 的 paired descriptive source data，来自 T24 `summary.json/raw_rows` |
| `source_data_fig03_ablation_mechanism.csv` | Fig. 3 的 machine-readable source data，来自当前稿件表格 |
| `source_data_fig04_statcalib.csv` | Fig. 4 的 machine-readable source data，来自当前稿件 statcalib 表 |
| `source_data_fig05_validation_contract.csv` | Fig. 5 的 machine-readable source data，来自 fast-path cost、fixed-point parity 和 runtime-discipline CSV |
| `outputs/fig01_dual_loop_architecture.pdf` | 双环 affine calibration 架构示意 |
| `outputs/fig02_main_software_hil_results.pdf` | 预声明四场景 simulation 主结果图 |
| `outputs/fig03_ablation_and_mechanism.pdf` | 消融与机制/干预边界图 |
| `outputs/fig04_statcalib_extension_lane.pdf` | supplementary statistical-calibration 边界图 |
| `outputs/fig05_validation_contract.pdf` | software validation-contract 汇总图；只汇总已有 source CSV，不是硬件结果 |

## 复现

在仓库根目录运行：

```powershell
python docs\figure_assets\submission_draft_python_figures\build_submission_draft_figures.py
```

生成图只重排当前稿件表格中的数值和已有证据边界，不产生新实验事实。

Fig. 2 使用 T24 `comparison.csv` 中的 `final_ler_mean` 与 `final_ler_std`。`source_data_fig02_paired_deltas.csv` 另从 T24 `summary.json/raw_rows` 派生 UKF 与 Hybrid-\(b\) 的同 seed/repeat 配对差异。由于 T24 每个 scenario/mode 只有 `n=2` repeats，误差条和 paired deltas 都只能解释为 descriptive uncertainty / descriptive improvement，不能写成 inferential confidence interval、p-value 或显著性检验。

## Source-data 审计

在仓库根目录运行：

```powershell
python docs\figure_assets\submission_draft_python_figures\audit_submission_draft_source_data.py
```

该 helper 只做当前投稿稿的表格/source-data 机械一致性检查。除 Fig. 2-5
source data 外，它还核对 `tab:controlled-oracle-affine` 与
`docs/paper_materials/submission_draft_controlled_oracle_affine_analysis.csv`
的 oracle-affine / wrapped-Gaussian 显示数值一致性，以及
`tab:sequence-controlled-baselines` 与
`docs/paper_materials/submission_draft_sequence_controlled_baseline_analysis.csv`
的短序列受控 baseline 数值一致性，以及
`tab:fast-path-cost-model` 与
`docs/paper_materials/submission_draft_fast_path_cost_model.csv` 的 analytical
count 一致性，并核对 `tab:metric-readiness` 与
`docs/paper_materials/submission_draft_metric_readiness_matrix.csv` 的 metric
axis 覆盖一致性；2026-07-06 起还会逐 scenario 核对
`tab:phase-a-paired-interval` 与
`docs/paper_materials/submission_draft_phase_a_paired_interval_analysis.csv`
中的 completed formal scenario interval rows。输出：
此外，它会检查 `docs/paper_materials/submission_draft_source_data_manifest.csv`
与 `docs/paper_materials/submission_draft_source_data_manifest.json` 的行数、
source path 和 SHA-256 是否与当前 checkout 一致。输出：

- `docs/paper_materials/submission_draft_source_data_audit.json`
- `docs/paper_materials/投稿稿source_data机械审计报告.md`

它不是 full reproducibility proof，不补 CI / p-value，不证明 fallback-free runtime，也不把 controlled oracle-affine / wrapped-Gaussian CSV、sequence-controlled baseline CSV、fast-path cost CSV、metric-readiness CSV 或 source-data manifest 升级为 formal benchmark、fidelity estimate、historical run recursive hash closure、hardware、`.tflite` 或 statcalib 证据。

## 符号边界审计

在仓库根目录运行：

```powershell
python docs\figure_assets\submission_draft_python_figures\audit_submission_draft_symbol_boundary.py
```

该 helper 只做当前投稿稿的 TeX/code/source-data 符号边界一致性检查，输出：

- `docs/paper_materials/submission_draft_symbol_boundary_audit.json`
- `docs/paper_materials/投稿稿符号边界机械审计报告.md`

它不是完整物理噪声模型验证，不导入 runtime，不运行 benchmark，不补 holdout / CI / p-value，也不改变 hardware、`.tflite`、HIL 或 statcalib 证据等级。
