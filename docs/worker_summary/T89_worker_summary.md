# T89 Worker Summary

## 改了什么

- 新增 4 份 `T89` frozen-mainline handoff / change-control 文档：
  - `docs/paper_materials/paper_frozen_mainline_handoff_packet.md`
  - `docs/paper_materials/paper_frozen_mainline_source_of_truth_map.md`
  - `docs/paper_materials/paper_postfreeze_change_control.md`
  - `docs/paper_materials/paper_blocked_surface_reentry_conditions.md`
- 更新 `docs/paper_materials/README.md`，登记 `T89` 的 handoff packet、source-of-truth map、post-freeze change-control 与 blocked-surface re-entry 入口，并明确这些文档不升级证据等级。
- 更新 `docs/paper_notes/README.md`，把 `T89` 四份文档加入 note 外部阅读链路，并明确它们只服务 handoff / change-control，不授权直接改写 note 或编译产物。
- 新增 `docs/review/T89_review.md` 与 `docs/for_human/T89_explanation.md`。
- 本轮没有修改：
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - 任意 note 编译产物
  - 任意 `runs/` / `artifacts/` / `docs/evidence_packs/` 文件
  - 任意治理文档或源码目录

## 如何验证

- 使用 allowlist-scoped `git status --short --untracked-files=all -- ...` 确认 `T89` 只触碰允许路径。
- 使用 allowlist-scoped `git diff --check -- ...` 验证本轮改动没有内容级格式错误；若仍出现 Windows 的 `LF -> CRLF` 提示，仅视为工作副本噪声，不当作内容缺陷。
- 检查 `paper_frozen_mainline_handoff_packet.md`，确认唯一 verdict 仍是 `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`，且没有把它改写成 submission-ready completion。
- 检查 `paper_frozen_mainline_source_of_truth_map.md`，确认至少覆盖 `T88` freeze manifest 的全部 `FZ01-FZ05` surfaces，并把 `BD01-BD06` blocked surfaces 回链到 authoritative source。
- 检查 `paper_postfreeze_change_control.md`，确认给出 `L0/L1/L2/L3` 四层分级与至少 8 条具体 change-control 规则。
- 检查 `paper_blocked_surface_reentry_conditions.md`，确认至少覆盖：
  - real-board execution / timing / resource
  - default-env / cross-host `.tflite` portability
  - full training reproducibility
  - `FR8/statcalib` mature comparator / unique clean threshold
  - expanded benchmark / stronger oracle baseline
  - theory-branch content mergeback into main
- 检查两个 README，确认都已登记 `T89` 四份新文档，并明确“只做 handoff / change-control，不升级证据等级”。
- 再次核对 `git status --short --untracked-files=all --` 对以下禁改路径的结果，确认本轮未触碰：
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.*`
  - `runs/`
  - `artifacts/`
  - `docs/evidence_packs/`

## 剩余风险

- `T89` 只是把 frozen-mainline handoff 与 post-freeze change-control 写清楚，不提供任何新实验或新证据；因此 blocked surface 依旧全部 blocked。
- 当前 worktree 不是全新干净状态，后续人工审查仍应继续采用 allowlist-scoped diff，而不是把 whole-repo diff 当作 `T89` 独有改动。
- `paper_postfreeze_change_control.md` 已经定义了 `L0/L1/L2/L3`，但这套纪律只有在后续维护者真的遵守时才有效；最大的真实风险仍是有人绕过任务包，直接手改 note 或弱化 disclaimer。
- theory 分支与 current main 仍隔离；如果未来有人想把 theory 内容并回 main，必须单独开 integration/evidence task，不能借 `T89` handoff 包直接回写。
