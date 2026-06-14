# T83 Mainline Note Full Consistency Crosswalk

## 1. Scope Verdict

- verdict: `FULLNOTE_CONSISTENCY_SWEEP_COMPLETED`
- 本轮目标不是继续扩实验、扩 claim 或宣称 full-manuscript closeout，而是把当前主线 note 的全文口径逐 section 对齐到“当前最强可支持事实”。
- 本轮全文 sweep 覆盖：
  - `Title`
  - `Abstract`
  - `Summary of Contributions`
  - `Introduction`
  - `Relationship to Existing Work`
  - `Brief Review of the GKP Code`
  - `Noise and Drift Model`
  - `Model Architecture`
  - `Experimental Setup`
  - `Numerical Results`
  - `Bounded follow-up lanes outside the accepted result layer`
  - `Discussion`
  - `Conclusion`

## 2. T83 Touched Sections

以下条目在 note 源文件中已留有对应的 `% T83-CLOSEOUT: ...` 注释，且本表中的 `touched_in_t83=yes` 与注释一一对应：

- `Numerical Results`
- `Bounded follow-up lanes outside the accepted result layer`
- `Discussion`
- `Conclusion`

## 3. Section-To-Evidence Crosswalk

| section_or_surface | touched_in_t83 | strongest_supported_truth | primary_evidence_anchors | forbidden_retelling | next_bounded_action |
| --- | --- | --- | --- | --- | --- |
| `Title` | `no` | 主线是 dual-loop、teacher-anchored、affine calibration；不把 `statcalib`、`.tflite` 或 real-board 写成并列主线 | `paper_bounded_prose_reopen_manifest.md`; `paper_claim_evidence_ledger.md` `C2/C3/C11` | 不写成 mature comparator closure、deployment closure 或 board-ready title | 无；仅在最终 polish 中做读者友好术语收口 |
| `Abstract` | `no` | 最强摘要事实仍是 `T24` 冻结主线 ranking，`FR6/FR7` 只是 descriptive support，`FR8` 只是 supplement-side extension lane | `paper_bounded_prose_reopen_manifest.md`; `paper_claim_evidence_ledger.md` `C3/C4/C11`; `paper_claim_risk_table.md` `PR2/PR9` | 不写 causal proof、不写 promoted `statcalib`、不写 deployment closure | 无；保持当前保守边界 |
| `Summary of Contributions` | `no` | 六条贡献均已压回分层事实：主线 frozen benchmark、support-only runtime chain、supplement-only board/statcalib boundary | `paper_methods_and_contribution_calibration_manifest.md`; `paper_claim_evidence_ledger.md` `C2/C5/C7/C9/C11`; `paper_manuscript_closeout_readiness_matrix.md` | 不把 support-only/supplement-only 写成 coequal main results；不写 full reproducibility 或 real-board success | 后续只做 reader-facing 术语和组织 polish |
| `Introduction` | `no` | 当前引言最强事实是“drift-adaptive GKP 被组织为 bounded affine calibration 问题”，并显式保留 evidence hierarchy | `paper_bounded_prose_reopen_manifest.md`; `paper_claim_evidence_ledger.md`; `paper_manuscript_closeout_readiness_matrix.md` | 不把 supporting layers 写成主结果；不把硬件边界写成已闭环 | 无；保持 hierarchy 明示 |
| `Relationship to Existing Work` | `no` | 这是 architecture/positioning 章节，不是 deployment-completion 章节 | `paper_note_alignment_and_layout_closeout.md`; `paper_bounded_prose_reopen_manifest.md`; `paper_claim_risk_table.md` `PR1/PR9/PR10` | 不写 completed board deployment、不写 promoted comparator closure | 若最终投稿化，需要把内部 boundary 语言再压成读者友好版本 |
| `Brief Review of the GKP Code` | `no` | 这里只支持 local affine approximation 的物理背景，不支持 exact decoder closure | `paper_methods_and_contribution_calibration_manifest.md` | 不把 affine fast path 写成 wrapped-posterior exact solution | 无；如要扩理论，只能另开理论任务 |
| `Noise and Drift Model` | `no` | 当前模型是 control-oriented effective model，四场景只覆盖当前 frozen benchmark 所需 drift family | `paper_methods_and_contribution_calibration_manifest.md`; `paper_claim_evidence_ledger.md` `C2/C12` | 不写 full circuit-level closure；不写 exhaustive drift coverage | 若要升级，只能新开 benchmark-expansion / noise-model task |
| `Model Architecture` | `no` | 主线架构仍是 teacher-anchored residual branch；`statcalib` 是 extension lane；runtime/board 仍是 boundary evidence | `paper_methods_and_contribution_calibration_manifest.md`; `paper_claim_evidence_ledger.md` `C7/C9/C11`; `paper_claim_risk_table.md` `PR4/PR8/PR9` | 不写 promoted comparator、不写 board-level closure、不写 default-env `.tflite` closure | 无；保持分层 |
| `Experimental Setup` | `no` | 当前 setup 只支持 `T24` locked mock-backed software-HIL protocol | `paper_bounded_prose_reopen_manifest.md`; `docs/review/T24_review.md`; `docs/review/T25_p4_formal_evidence_gate_review.md` | 不写 board timing/resource benchmark；不写 `.tflite` / real-board ranking | 如需升级，只能先开 protocol-then-execution task |
| `Numerical Results` | `yes` | 主结果层仍是 frozen five-mode ranking；ablation/mechanism/statcalib 只作为 bounded support or supplement-side lane | `paper_bounded_prose_reopen_manifest.md`; `paper_note_results_sync_manifest.md`; `paper_supporting_material_closeout_pack.md`; `paper_claim_evidence_ledger.md` `C3/C4/C11` | 不写 expanded benchmark、不写 causal closure、不写 promoted `statcalib` comparator | 下一步只允许 final-polish 级别的结构压缩与读者术语收口 |
| `Bounded follow-up lanes outside the accepted result layer` | `yes` | 该 subsection 现在被显式标成 follow-up/boundary register，而不是新的 Results pillar | `paper_manuscript_closeout_readiness_matrix.md`; `paper_claim_risk_table.md` `PR10`; `paper_reopen_gap_matrix.md` | 不暗示这些 lanes 已进入 accepted result layer；不把计划写成事实 | 若继续推进，只能逐条开新的 bounded task，不得直接并回主结果层 |
| `Discussion` | `yes` | Discussion 现在清楚区分 `main text / appendix / supplement / blocked` 四层 manuscript route | `paper_supporting_material_closeout_pack.md`; `paper_manuscript_closeout_readiness_matrix.md`; `paper_claim_evidence_ledger.md` `C5/C7/C9/C11` | 不把 layered route flatten 成单一 deployment story；不把 support-only surface 写成 ranking evidence | 下一步只做 route cleanup / condensation，不做新 evidence promotion |
| `Conclusion` | `yes` | 当前最强结论是“simulation/material-first manuscript route + bounded final polish”，而不是 deployment closure 或 hardware-ready finalization | `paper_manuscript_closeout_readiness_matrix.md`; `paper_claim_evidence_ledger.md` `C3/C11`; `paper_claim_risk_table.md` `PR8/PR10` | 不写 full-manuscript evidence closure；不写硬件已 ready | 只允许 final-polish 任务；若要升级 blocked surface，需新开 future-host 或 portability task |

## 4. Sweep Outcome

- 全文已能逐 section 回链到当前 strongest supported truth。
- `ready`、`support-only`、`blocked` 三类 surface 的主次层级已不再互相打架。
- 本轮仍未把以下内容写成已完成事实：
  - expanded benchmark
  - causal mechanism closure
  - full training reproducibility
  - default-env `.tflite` closure
  - real-board execution success
  - promoted `statcalib` comparator
- 因此，`T83` 的 sweep 结论是“当前 note 已形成可审计的一致性版本”，而不是“已经自动等于 full-manuscript closeout”。
