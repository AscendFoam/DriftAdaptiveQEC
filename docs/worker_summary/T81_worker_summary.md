# T81 Worker Summary

## 改了什么

- 重写了 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 中 4 个 target sections：
  - `Summary of Contributions`
  - `Brief Review of the GKP Code`
  - `Noise and Drift Model`
  - `Model Architecture`
- 在这 4 个 section 入口增加了 `% T81-CALIBRATION: ...` 标记，便于后续 scope 审计。
- 把 `Summary of Contributions` 压回到当前最强可支撑口径：`T24` 仍是主线 frozen-set anchor，`FR6/FR7` 仍是 descriptive support，`FR8/statcalib` 仍是 extension lane / no-promotion / no unique clean threshold，training/material、`.tflite`、real-board 仍是 layered boundary evidence。
- 把三章 methods 压回到“理论背景 / effective model / mainline architecture”的角色，明确：
  - `Brief Review of the GKP Code` 只解释 GKP 物理层与局部 affine 近似；
  - `Noise and Drift Model` 是 effective model，不是 full circuit-level closure；
  - `Model Architecture` 的 mainline 是 teacher-anchored residual path，`statcalib` 只是 separately labeled FR8 extension lane。
- 更新了：
  - `docs/paper_notes/README.md`
  - `docs/paper_materials/README.md`
  - `docs/paper_materials/paper_methods_and_contribution_calibration_manifest.md`
  - `docs/review/T81_review.md`
  - `docs/for_human/T81_explanation.md`

## 如何验证

- 重新编译了 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`：
  - toolchain = `TeX Live 2024 + latexmk`
  - compile exit code = `0`
  - PDF 已刷新
- `latex_doctor.py --json` 在 `PYTHONUTF8=1` 下检测为：
  - detector status = `existing-usable`
  - `latexmk` smoke = `passed`
  - bundled `tectonic` 可见，但 smoke 仍因 `os error 5` 失败
- `.log` 关键字扫描未发现：
  - `Underfull`
  - `Overfull`
  - `LaTeX Warning`
  - `undefined`
  - `Citation`
- `git status --short --untracked-files=all -- ...` 显示本轮变更只落在任务包允许的 note / paper-material / review / for_human / worker_summary 文件，以及允许保留的 LaTeX 编译产物。
- note 源码中已出现 4 条 `% T81-CALIBRATION` 标记。
- `T80` 的 8 条 `% T80-REOPEN` 标记仍保留，用于证明本轮没有把 `T81` 扩成对 ready sections 的再次大改。

## 剩余风险

- `T81` 完成后，当前 note 仍然不是 full-manuscript reopen；它只代表 contribution/methods 的有界校准完成。
- `statcalib` 仍然只能写成 extension lane / no-promotion / no unique clean threshold，不能回述成成熟 comparator。
- `.tflite` 仍然只能写成 isolated current-host true runtime，不能回述成 default-env 或 deployment closure。
- real-board 仍然只能写成 read-only gate / regeneration / provenance with current-host `NO_GO`，不能回述成 execution success。
