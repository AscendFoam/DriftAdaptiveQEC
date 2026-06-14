# T86 Worker Summary

## 改了什么

- 新增 4 份 submission-facing 装配台账：
  - `docs/paper_materials/paper_submission_pack_assembly_manifest.md`
  - `docs/paper_materials/paper_submission_surface_route_map.md`
  - `docs/paper_materials/paper_submission_exclusion_register.md`
  - `docs/paper_materials/paper_submission_author_handoff.md`
- 更新 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`，只对 `Numerical Results`、`Discussion`、`Conclusion` 做最小 route / exclusion 导向刷新，并补上：
  - `% T86-ASSEMBLY: Numerical Results`
  - `% T86-ASSEMBLY: Discussion`
  - `% T86-ASSEMBLY: Conclusion`
- 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，登记 `T86` 新入口与 `% T86-ASSEMBLY` 使用规则。
- 新增 `docs/review/T86_review.md` 与 `docs/for_human/T86_explanation.md`，分别沉淀审查结论和作者向解释材料。
- 刷新当前主机上的 note 编译产物：
  - `CNN_FPGA_GKP_theory_note_draft.aux`
  - `CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
  - `CNN_FPGA_GKP_theory_note_draft.fls`
  - `CNN_FPGA_GKP_theory_note_draft.log`
  - `CNN_FPGA_GKP_theory_note_draft.out`
  - `CNN_FPGA_GKP_theory_note_draft.pdf`
  - `CNN_FPGA_GKP_theory_note_draft.synctex.gz`
  - `CNN_FPGA_GKP_theory_note_draft.toc`

## 如何验证

- 使用 allowlist-scoped `git status --short --untracked-files=all -- ...`，只核对 `T86` 允许路径内的改动。
- 检查主 note 标记链，确认 `% T80-REOPEN`、`% T81-CALIBRATION`、`% T82-SUPPORT`、`% T83-CLOSEOUT`、`% T84-POLISH`、`% T85-PREFLIGHT` 仍保留，且 `% T86-ASSEMBLY` 已覆盖本轮实际修改的 3 个 section。
- 检查 3 份新台账列头，确认分别包含任务包要求的字段：
  - `surface_id / surface_role / included_source / evidence_anchor / author_action`
  - `claim_or_section / main_text_route / appendix_route / supplement_route / exclusion_note / source_anchor`
  - `exclusion_id / blocked_surface / why_excluded_now / do_not_claim_wording / future_unblock_task`
- 检查 `paper_submission_author_handoff.md`，确认仍显式排除：
  - `real-board execution / timing / resource`
  - `default-env / cross-host .tflite portability`
  - `full training reproducibility`
  - `statcalib mature comparator promotion`
- 在 `docs/paper_notes/` 下执行：
  - `latexmk -g -pdf -synctex=1 -interaction=nonstopmode -halt-on-error CNN_FPGA_GKP_theory_note_draft.tex`
- 对 allowlist 路径执行 `git diff --check`；清掉 `.log` 中唯一一处生成尾随空白后，已无内容级报错，剩余仅为当前 Windows working-copy 的 `LF -> CRLF` 提示。
- 编译环境与结果：
  - 工具链：`TeX Live 2024 + latexmk`
  - 目标：`CNN_FPGA_GKP_theory_note_draft.tex`
  - 产物：`.aux/.fdb_latexmk/.fls/.log/.out/.pdf/.synctex.gz/.toc`
  - `.log` 关键字扫描：未检出 `Underfull`、`Overfull`、`LaTeX Warning`、`undefined`、`Citation`

## 剩余风险

- 当前 worktree 在进入 `T86` 前已经有与本轮无关的脏状态，后续人工审查仍应继续使用 allowlist-scoped status / diff，而不是把整仓 diff 直接当作 `T86` 输出。
- 当前主机的 Git / 行尾设置会在 `status` / `diff --check` 中继续出现 `C:\\Users\\26410/.config/git/ignore` 访问告警和 `LF -> CRLF` 提示；它们是宿主机噪声，不是 `T86` 内容错误，但人工审查时需要区分。
- `T86` 只完成 submission-facing assembly / exclusion 收口，不等于：
  - submission-ready pack 已完成；
  - real-board execution 已完成；
  - default-env / cross-host `.tflite` portability 已闭环；
  - full training reproducibility 已闭环；
  - `statcalib` 已升级为成熟主线 comparator。
- 当前 note 编译成功只代表这台主机上的 `TeX Live 2024 + latexmk` 可用，不代表其他宿主环境或后续期刊模板链路自动成立。
