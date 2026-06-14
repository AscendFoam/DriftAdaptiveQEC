# T82 Worker Summary

## 改了什么

- 新增了 `docs/paper_materials/paper_supporting_material_closeout_pack.md`，把当前 supporting surface 统一整理成 `main text / appendix / supplement / blocked` 四层路由，并显式列出 evidence anchors 与 forbidden claims。
- 新增了 `docs/paper_materials/paper_manuscript_closeout_readiness_matrix.md`，把当前 manuscript-facing surface 按 `ready / support-only / blocked` 分类，并补了 blocker type 与 next bounded action。
- 回写了 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 中 4 处 supporting-boundary 片段，并加入 4 条 `% T82-SUPPORT: ...` 标记：
  - `Runtime, quantization, and fixed-point degradation`
  - `Embedded runtime and board-level validation`
  - `Discussion` 中的 deployment/support boundary 段落
  - `Conclusion` 中的 remaining technical gap 段落
- 更新了：
  - `docs/paper_notes/README.md`
  - `docs/paper_materials/README.md`
  - `docs/review/T82_review.md`
  - `docs/for_human/T82_explanation.md`
- 重新编译并刷新了 note 的 LaTeX 产物。

## 如何验证

- `latex_doctor.py --json` 在 `PYTHONUTF8=1` 下检测为：
  - detector status = `existing-usable`
  - `latexmk` smoke = `passed`
  - bundled `tectonic` 可见，但 smoke 仍因 `os error 5` 失败
- 使用 `TeX Live 2024 + latexmk` 重新编译了 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`：
  - compile exit code = `0`
  - PDF 已刷新
- `.log` 关键字扫描未发现：
  - `Underfull`
  - `Overfull`
  - `LaTeX Warning`
  - `undefined`
  - `Citation`
- note 源码中已出现 4 条 `% T82-SUPPORT` 标记。
- `T81` 的 4 条 `% T81-CALIBRATION` 与 `T80` 的 8 条 `% T80-REOPEN` 标记仍全部保留。
- `paper_supporting_material_closeout_pack.md` 已包含：
  - supporting surface
  - placement
  - evidence anchors
  - forbidden claims
- `paper_manuscript_closeout_readiness_matrix.md` 已包含：
  - surface/section
  - readiness status
  - blocker type
  - next bounded action
- `git diff --name-only` 的 tracked 改动落在允许路径内；`git status --short --untracked-files=all -- ...` 也已覆盖本轮新增的 allowed files。

## 剩余风险

- `T82` 完成后，当前材料栈仍然不是 full-manuscript closeout；它只代表 supporting-material route 已经被更清楚地收口。
- `statcalib` 仍然只能写成 extension lane / no-promotion / no unique clean threshold，不能回述成成熟 comparator。
- training/material 仍然只能写成 canonical chain intact + one clean CPU-only bounded rerun，不能回述成 full reproducibility。
- `.tflite` 仍然只能写成 isolated current-host true runtime，不能回述成 default-env、HIL 或 deployment closure。
- real-board 仍然只能写成 read-only gate / regeneration / provenance with current-host `NO_GO`；任何依赖 `Linux + FPGA` 宿主和 board timing/resource 的 surface 仍然是 blocked。
