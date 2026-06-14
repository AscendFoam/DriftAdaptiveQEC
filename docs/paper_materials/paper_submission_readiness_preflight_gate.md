# T85 Submission Readiness Preflight Gate

## 1. Gate Inputs

本轮 preflight 只基于当前主线 paper-facing 材料做判断，输入包括：

- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/review/T84_review.md`
- `docs/paper_materials/paper_bounded_final_polish_change_map.md`
- `docs/paper_materials/paper_reader_facing_term_translation_table.md`
- `docs/paper_materials/paper_appendix_supplement_reader_assembly_map.md`
- `docs/paper_materials/paper_manuscript_closeout_readiness_matrix.md`
- `docs/paper_materials/paper_closeout_gate_and_blocker_register.md`
- `docs/paper_materials/paper_submission_material_gap_checklist.md`
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/paper_materials/paper_claim_risk_table.md`
- `docs/paper_materials/paper_residual_state_lag_sweep.md`

本轮不新增实验、不重算 benchmark、不改写历史 run/artifact，也不进入 submission-pack 实装。

## 2. Gate Verdict

- gate_verdict: `GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY`

## 3. Why This Is `GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY`

1. `T74` 的 submission-material gap checklist 已经证明当前仓库存在一条诚实的 simulation/material-first 提交路径，并不依赖 real-board 主结果才能形成主文/附录/补充的分层材料组合。
2. `T82`、`T83`、`T84` 已把 supporting-boundary、全文一致性、reader-facing translation / assembly 压回到当前 strongest supported truth；`T85` 又处理了 `Discussion` / `Conclusion` 中残留的 state-lag wording，因此当前 note 已不再把本轮已完成的 reader-facing polish 写成未来待办。
3. 当前真正剩下的是“一张 bounded submission-pack assembly 任务是否值得打开”的问题，而不是“note 本身还没自洽”或“还需要再补一轮新实验”。
4. 这个 `GO` 只表示：如果下一步继续推进，可以开一张受边界约束的 submission-facing assembly 任务；它不表示：
   - submission-ready pack 已完成；
   - blocked surface 已解除；
   - `.tflite` portability、real-board execution、expanded benchmark 或 full reproducibility 已补齐；
   - 任何 claim 可以越过当前 evidence layer。

## 4. What The Next Bounded Assembly May And May Not Do

### 4.1 May do

- 在现有 main text / appendix / supplement / blocked 分层上组织 submission-facing 材料包；
- 压缩作者说明、排除项说明和 supporting-surface route，使其更适合投稿前人工终修；
- 保持当前 claim/evidence/risk 三本账同步。

### 4.2 Must not do

- 不得顺手重开 benchmark、训练、`.tflite` portability 或 real-board execution；
- 不得把 support-only / blocked surface 升级成主结果或完成态；
- 不得把本轮 `GO` 回述成 submission-ready pack 已完成。

## 5. Explicit Blockers That Still Stay Outside The Assembly Boundary

下一步即使进入 bounded submission-pack assembly，也必须继续把以下 surface 视为 blocker / exclusion，而不是 silently absorb 进去：

- `paper_submission_blocker_matrix.md` 中的 `SB01` 到 `SB06`
- 尤其是：
  - board-level execution / timing / resource rows
  - default-env `.tflite` / deployment portability
  - full training reproducibility
  - promoted `statcalib` comparator retelling
  - broader expanded benchmark story

## 6. Operational Reading Of The Verdict

`GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY` 的实际含义是：

1. 当前 mainline note / paper-material 栈已经达到“可以进入一张 bounded submission-facing assembly 任务”的最小诚实状态；
2. 该下一任务必须继续是 docs-only、mainline-only、assembly-only；
3. 该下一任务不是 submission-ready completion，也不允许借装配名义推进 claim promotion 或 blocked surface 解锁。
