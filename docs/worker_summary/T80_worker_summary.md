# T80 Worker Summary

## 改了什么

1. 重写了 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 中允许范围内的 8 个 ready sections，并在每个 section 入口补了 `% T80-REOPEN: ...` 标记：
   - `Title`
   - `Abstract`
   - `Introduction`
   - `Relationship to Existing Work`
   - `Experimental Setup`
   - `Numerical Results and Benchmark Plan`
   - `Discussion`
   - `Conclusion`
2. 保持 `Summary of Contributions` 与三章 methods (`Brief Review of the GKP Code`、`Noise and Drift Model`、`Model Architecture`) untouched，没有借 prose reopen 名义扩 scope。
3. 新增 `docs/paper_materials/paper_bounded_prose_reopen_manifest.md`，逐节登记：
   - 哪些 section 被改了；
   - 每节绑定哪些 evidence anchors；
   - 每节保留了哪些 guardrails；
   - compile 状态是什么。
4. 更新了 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，登记 `T80` 新入口与 `% T80-REOPEN` / manifest 的阅读规则。
5. 新增了 `docs/review/T80_review.md` 与 `docs/for_human/T80_explanation.md`。
6. 在本机可用工具链下重新编译了 note，并按当前编译结果保留对应 `.aux/.fdb_latexmk/.fls/.log/.pdf/.synctex.gz/.toc` 等辅助/输出文件；实际产生 diff 的编译产物以下方范围检查为准。

## 如何验证

- 文档与 section scope
  - 用 `rg -n "T80-REOPEN|Summary of Contributions|Brief Review of the GKP Code|Noise and Drift Model|Model Architecture" docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 检查：
    - `% T80-REOPEN` 只落在 8 个允许 section；
    - `Summary of Contributions` 和三章 methods 没被纳入本轮改写标记。
  - 用 `paper_bounded_prose_reopen_manifest.md` 逐节核对 changed ledger、evidence anchors 和 guardrails。
- 边界口径
  - 对照 `paper_claim_evidence_ledger.md`、`paper_claim_risk_table.md`、`paper_results_callout_sheet.md`、`paper_maintext_results_authoring_pack.md`、`paper_note_results_sync_manifest.md`，确认：
    - `T24` 仍是 frozen-set main anchor；
    - `FR6/FR7` 仍是 descriptive-only；
    - `FR8` 仍是 extension-lane / no-promotion / no unique clean threshold；
    - training/material、`.tflite`、real-board 仍保持 supporting / layered boundary。
- LaTeX 编译
  - 用 bundled LaTeX skill 对应脚本做 doctor：
    - `latex_doctor.py --json` 返回 `existing-usable`
  - 用 bundled LaTeX skill 对应脚本编译：
    - `compile_latex.py D:\Codes\Quantum\DriftAdaptiveQEC\docs\paper_notes\CNN_FPGA_GKP_theory_note_draft.tex --json`
    - 结果：`exitCode = 0`，`pdfExists = true`
  - 对 `.log` 做关键词扫描：
    - 未检出 `Underfull`
    - 未检出 `Overfull`
    - 未检出 `LaTeX Warning`
    - 未检出 `undefined` / `Citation`
- 范围检查
  - `git diff --name-only -- <T80 Allowed Files>` 当前只包含：
    - `docs/paper_notes/README.md`
    - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
    - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.aux`
    - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
    - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fls`
    - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log`
    - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf`
    - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.synctex.gz`
    - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.toc`
    - `docs/paper_materials/README.md`
    - `docs/paper_materials/paper_bounded_prose_reopen_manifest.md`
    - `docs/review/T80_review.md`
    - `docs/for_human/T80_explanation.md`
    - `docs/worker_summary/T80_worker_summary.md`

## 剩余风险

1. 本轮完成的是 section-bounded prose reopen，不是 full-manuscript reopen；methods calibration 仍未开始。
2. `T24` expanded benchmark、mechanism closure、default-env `.tflite`、real-board success、`statcalib` promotion 这些更强叙事仍然被现有 evidence 阻塞。
3. 整棵 worktree 里仍共存着 `T79` 及更早 paper-material 相关 diff，因此后续如果要提交，必须继续做 path-isolated 或 hunk-isolated staging，不能直接整树 `git add .`。
