# 投稿稿 source-data manifest 补强记录

## 目的

本记录说明 `CNN_FPGA_GKP_submission_draft.tex` 的 source-data / provenance 口径补强。

新增或同步文件：

- `build_submission_draft_source_manifest.py`
- `submission_draft_source_data_manifest.csv`
- `submission_draft_source_data_manifest.json`
- `build_submission_draft_row_provenance_manifest.py`
- `submission_draft_row_provenance_manifest.csv`
- `submission_draft_row_provenance_manifest.json`
- `投稿稿row_level_provenance补强记录.md`

## 补强内容

`build_submission_draft_source_manifest.py` 对当前投稿稿使用的 manuscript-facing source CSV/JSON、figure manifest、figure source-map、row-level provenance 输出和生成/审计脚本计算 SHA-256 与文件大小，并生成机器可读 CSV/JSON manifest。

该 manifest 覆盖：

- Fig. 2--5 的 source CSV；
- paired uncertainty、controlled oracle-affine、sequence-controlled baseline、holdout drift stress、fast-path cost、fixed-point parity、logical-channel surrogate、runtime discipline、GKP boundary sensitivity、metric readiness、CNN branch details 等投稿稿表格/分析 source files；
- literature metric crosswalk、benchmark expansion protocol、runner smoke pair 与 row-level provenance 这四类投稿稿定位/规划/provenance source files；
- Phase A repeat plan、repeat summary、paired-interval analysis、upgrade gate 及其生成脚本；
- figure manifest / source-map；
- 投稿稿图件生成脚本、source-data 审计脚本、符号边界审计脚本和各项 controlled-analysis 生成脚本。

## 可写边界

可以写：

- 当前投稿稿已有文件级 source-data manifest 和主 benchmark row-level provenance manifest；
- manuscript-facing source CSV/JSON 与生成/审计脚本具有 SHA-256 文件级哈希；
- 主 benchmark 的 scenario/mode/repeat/seed/source path/config/runner/selected repeat-summary hash 已有 row-level source trace；
- 该 manifest 提升了表格、图件、source-data 文件和主 benchmark 行级来源的可审查性。

不能写：

- 这不是 historical `runs/` 目录的递归 hash closure；
- 这不是 full training reproducibility；
- 这不是 hardware provenance；
- 这不补 CI、p-value、standard error 或 inferential statistics；
- 这不补 real-board timing、resource、power、bitstream、DMA/MMIO log 或 source-vs-board vector。

## 正文同步

同步位置：

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

同步口径：

- `Availability of supporting materials` 后的补充段落说明当前稿件已有 manuscript-facing source files 的 file-level hash manifest 和主 benchmark row-level provenance manifest；
- `Scope of Validation` 中的 validation gaps 改为“已有文件级 manifest 和主 benchmark row-level provenance，但不是 historical run recursive hash closure”；
- `Statistical Treatment and Reproducibility` 表格中的 source-data 行加入 file-level source-data manifest 与 row-level provenance manifest，并保留 whole-manuscript / recursive historical run hash closure 缺口。

## 验证

已运行：

```powershell
python docs\paper_materials\build_submission_draft_row_provenance_manifest.py
python docs\paper_materials\build_submission_draft_source_manifest.py
```

输出：

- `row_provenance status = ok`
- `row_provenance rows = 40`
- `source_manifest status = ok`
- `source_manifest rows = 97`
- `docs/paper_materials/submission_draft_row_provenance_manifest.csv`
- `docs/paper_materials/submission_draft_row_provenance_manifest.json`
- `docs/paper_materials/submission_draft_source_data_manifest.csv`
- `docs/paper_materials/submission_draft_source_data_manifest.json`

2026-07-06 复跑补充：

- `python docs\paper_materials\build_submission_draft_source_manifest.py`
- `source_manifest status = ok`
- `source_manifest rows = 97`
- 新 manifest 覆盖当前 Phase A repeat summary、paired interval、upgrade gate、source-data coverage matrix、closest-work positioning 等新增投稿稿 source-data / boundary files 的文件级 hash。

后续仍需通过投稿稿 source-data 机械审计、符号边界审计、禁词扫描和 LaTeX 编译确认整体一致性。
