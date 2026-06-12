# T78 Worker Summary

## 改了什么

1. 校准了 note 的非结果层 wording：
   - 修改标题，去掉把 `statcalib` 写成并列主线的暗示；
   - 更新 `Abstract`、`Introduction`、`Summary of Contributions`、`Relationship to Existing Work`、`Discussion`、`Conclusion`；
   - 在这些 section 旁加入 `% T78-SCOPE: ...` 注释。
2. 降低了 `statcalib` 在 note 内的视觉层级：
   - 增加 supplement-side bridge 句；
   - 把三个 `statcalib` 标题从 `subsection` 降为 `subsubsection`；
   - 保持 `extension lane / no promotion / persistent tie / no unique clean threshold` 不变。
3. 收了一轮 LaTeX warning：
   - `Metric-level advantages` 表改成 `raggedright` 列格式；
   - 调整 `Discussion` 中一条长句的断行；
   - 重新编译 note，刷新 `pdf/aux/fdb_latexmk/fls/log/out/synctex.gz/toc`。
4. 同步了 note 入口和收口文档：
   - 更新 `docs/paper_notes/README.md`
   - 更新 `docs/paper_materials/README.md`
   - 更新 `docs/paper_materials/paper_note_results_sync_manifest.md`
   - 更新 `docs/paper_materials/paper_results_section_assembly_pack.md`
   - 新增 `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`

## 如何验证

- `rg -n "T78-SCOPE" docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - 可覆盖本轮改过的非结果层 section：`Title`、`Abstract`、`Introduction`、`Summary of Contributions`、`Relationship to Existing Work`、`Discussion`、`Conclusion`
- `git diff --unified=0 -- docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - 本轮 diff 落点只在任务包允许的标题、非结果层校准区，以及 `Numerical Results` 中允许的 `statcalib` hierarchy 部分
- `git status --short --untracked-files=all`
  - 新增文件只包括：
    - `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`
    - `docs/review/T78_review.md`
    - `docs/for_human/T78_explanation.md`
    - `docs/worker_summary/T78_worker_summary.md`
  - 其余改动文件也都落在任务包允许列表内
- LaTeX 工具链与编译
  - `python scripts/latex_doctor.py --json`
  - `python scripts/compile_latex.py D:\Codes\Quantum\DriftAdaptiveQEC\docs\paper_notes\CNN_FPGA_GKP_theory_note_draft.tex --json`
  - `TeX Live 2024` 可用，note 编译成功
- warning before / after
  - `git show HEAD:docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log | Select-String 'Underfull \\hbox'`
  - `Select-String -LiteralPath docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log -Pattern 'Underfull \\hbox'`
  - `Underfull \hbox` 从 `32` 降到 `0`
- 边界 wording 检查
  - `statcalib` 仍保留 supplement-side extension lane / no promotion / persistent tie / no unique clean threshold
  - `T48` 仍保留 isolated current-host true runtime only
  - `T49/T71/T72` 仍保留 read-only gate / provenance / `NO_GO`
- git scope 检查
  - `git diff --name-only -- runs` 为空
  - `git diff --name-only -- artifacts` 为空
  - `git diff --name-only -- cnn_fpga physics benchmark tests` 为空
  - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/02_experiment_plan.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md` 为空

## 剩余风险

1. `T78` 只完成了任务包允许的 note 收口，不等于 full-manuscript reopen；`Brief Review of the GKP Code`、`Noise and Drift Model`、`Model Architecture` 等 section 仍保持原状。
2. `statcalib` 的层级已经降权，但这仍然只是 note 内部展示层面的收口，不改变 `T70` 的 no-promotion gate。
3. 当前 `.log` 已无 `Underfull \hbox` / `Overfull \hbox` / `pdfTeX warning`，但这不等于全文排版和投稿质量已经最终锁定；后续如果继续改 note，仍可能重新引入新 warning。
