## 2026-06-15 Captain Final Supersession (T89 closeout)

- Current unique task: `T90: 训练链 clean-CPU 同机 repeated-run 一致性证据包`
- Task package: `docs/tasks/Phase2/T90_training_same_host_repeated_run_consistency_pack.md`
- `T89` has been judged `PASS`.
- `T89` completed one honest frozen-mainline handoff / source-of-truth / change-control closeout and opened no deferred/rejected warning-derived risk.
- The T89 review's non-blocking notes are accepted as operational reminders only: allowlist-scoped review discipline inside a dirty worktree, host-side git/CRLF noise separation, and the requirement to keep frozen-mainline handoff narrower than submission-ready completion or blocked-surface unlock.
- `R11/R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T90` is next because the mainline note is now intentionally frozen, `T51/T52` broader prose reopen is still premature, real-board remains hardware-blocked, and the highest-value feasible non-hardware evidence gap is to strengthen `R11` from one bounded rerun (`T50`) into one same-host repeated-run consistency pack.
- `T90` must stay mainline-only, clean-CPU-only, reproducibility-bounded, and theory-branch-isolated; it must not widen into note rewrite, benchmark/HIL reruns, `.tflite` portability, real-board execution, sidecar promotion, venue-template adaptation, theory-branch mergeback, or submission-ready completion claims.

## 2026-06-14 Captain Final Supersession (T87 closeout)

- Current unique task: `T88: 主线 bounded manual finish 执行与 surface freeze 收口包`
- Task package: `docs/tasks/Phase2/T88_mainline_bounded_manual_finish_and_surface_freeze.md`
- `T87` has been judged `PASS`.
- `T87` completed one honest docs-only author-final QA / pre-submission gate step and opened no deferred/rejected warning-derived risk.
- The T87 review's non-blocking notes are accepted as operational reminders only: CRLF working-copy noise, current-host git-ignore warning noise, and `GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY` must not be retold as submission-ready completion.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T88` is next because the remaining mainline gap is no longer whether another QA gate is needed, but whether the already approved `MF01-MF05` bounded manual-finish actions can be executed and then frozen into one auditable mainline surface answer.
- `T88` must stay docs-only, mainline-only, manual-finish-only, and theory-branch-isolated; it must not widen into benchmark/HIL reruns, `.tflite` portability, real-board execution, sidecar promotion, venue-template adaptation, or submission-ready completion claims.

## 2026-06-14 Captain Final Supersession (T86 closeout)

- Current unique task: `T87: 主线作者终检与 pre-submission QA 收口包`
- Task package: `docs/tasks/Phase2/T87_mainline_author_final_qa_and_presubmission_gate.md`
- `T86` has been judged `PASS`.
- `T86` completed one honest docs-only submission-facing assembly / exclusion closeout and opened no deferred/rejected warning-derived risk.
- The four T86 non-blocking notes are accepted as operational reminders only: allowlist-scoped diff discipline, no overclaim from assembly docs, current-host-only compile scope, and host-noise separation for CRLF / git-ignore warnings.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T87` is next because the remaining mainline gap is no longer assembly itself, but one stricter author-final QA / pre-submission regression gate over the assembled note/material package.
- `T87` must stay docs-only, mainline-only, QA-only, and theory-branch-isolated; it must not widen into benchmark/HIL reruns, `.tflite` portability, real-board execution, sidecar promotion, venue-template adaptation, or submission-ready completion claims.

## 2026-06-14 Captain Final Supersession (T85 closeout)

- Current unique task: `T86: 主线 bounded submission-pack assembly 与显式 exclusion route 收口`
- Task package: `docs/tasks/Phase2/T86_mainline_bounded_submission_pack_assembly.md`
- `T85` has been judged `PASS`.
- `T85` completed one honest docs-only submission-readiness preflight / blocker-gate step and closed `R36`; its non-blocking notes are accepted as operational reminders rather than new risks.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T86` is next because preflight already completed; the remaining mainline gap is one bounded assembly of existing mainline note/material surfaces with explicit inclusion/exclusion routing.
- `T86` must stay docs-only, mainline-only, assembly-only, and theory-branch-isolated; it must not widen into benchmark/HIL reruns, `.tflite` portability, real-board execution, sidecar promotion, or submission-ready completion claims.

## 2026-06-14 Captain Final Supersession (T84 closeout)

- Current unique task: `T85: 主线 submission-readiness preflight gate 与残余状态滞后清扫`
- Task package: `docs/tasks/Phase2/T85_mainline_submission_readiness_preflight_gate.md`
- `T84` has been judged `PASS_WITH_WARNINGS`.
- Warning classification:
  - `N1` `Conclusion` 残留一处把本轮已完成 reader-facing polish 写成未来工作的状态滞后句 = `deferred -> R36`
  - `N2` allowlist-scoped diff / precise staging discipline = `accepted`
  - `N3` compile 结论仅限当前宿主 `TeX Live 2024 + latexmk` = `accepted`
- `R13/R14/R32/R33/R36` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T85` is the next bounded mainline task because the repo no longer缺 reader-facing assembly，而是缺一个不扩范围的 submission-readiness preflight gate 与残余状态滞后清扫。
- `T85` must stay docs-only, mainline-only, and must not widen into benchmark/HIL reruns, `.tflite` portability, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or direct submission-pack completion.

## 2026-06-14 Captain Final Supersession (T83 closeout)

- Current unique task: `T84: 主线 note 有界 final polish 与读者化装配包`
- Task package: `docs/tasks/Phase2/T84_mainline_bounded_final_polish_and_reader_facing_assembly.md`
- `T83` has been judged `PASS`.
- `T83` completed one docs-only full-note consistency sweep plus one explicit closeout gate honestly and opened no deferred/rejected warning.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T84` is the next bounded mainline task because the repo no longer lacks section-local calibration, supporting-boundary route integration, or a full-note closeout gate; it now needs one reader-facing final-polish and assembly pass before any later submission-readiness decision.
- `T84` must stay docs-only, mainline-only, and must not widen into benchmark/HIL reruns, `.tflite` portability, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or direct submission-pack completion.

## 2026-06-12 Captain Final Supersession (T80 closeout)

- Current unique task: `T81: Summary of Contributions 与 methods-only calibration pack`
- Task package: `docs/tasks/Phase2/T81_summary_and_methods_calibration_pack.md`
- `T80` has been judged `PASS`.
- `T80` completed one docs-only section-bounded prose reopen honestly and kept `Summary of Contributions` plus the three methods chapters untouched on purpose.
- `T80` introduces no deferred/rejected warning and opens no new risk.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T81` is the next bounded mainline task because the repo no longer needs another ready-section prose wave; it now needs one controlled calibration pass over `Summary of Contributions` and the three methods chapters only.
- `T81` must stay docs-only, mainline-only, and must not widen into benchmark/HIL reruns, `.tflite` execution, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or direct full-manuscript expansion.

## 2026-06-12 Captain Final Supersession (T79 closeout)

- Current unique task: `T80: 主线校准段落的 bounded prose reopen`
- Task package: `docs/tasks/Phase2/T80_mainline_calibrated_sections_bounded_prose_reopen.md`
- `T79` has been judged `PASS`.
- `T79` completed one docs-only reopen gate honestly and fixed the current mainline answer to `GO_FOR_BOUNDED_PROSE_REOPEN`.
- `T79` introduces no deferred/rejected warning and opens no new risk.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T80` is the next bounded mainline task because the repo no longer needs another readiness gate; it now needs one section-bounded prose reopen on the already-ready areas only.
- `T80` must stay docs-only, mainline-only, and must not widen into methods chapters, benchmark/HIL reruns, `.tflite` execution, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or direct full-manuscript expansion.

## 2026-06-12 Captain Final Supersession (T78 closeout)

- Current unique task: `T79: 论文材料 reopen gate 与 bounded prose 扩写就绪性评审`
- Task package: `docs/tasks/Phase2/T79_paper_reopen_gate_and_prose_readiness_review.md`
- `T78` has been judged `PASS`.
- `T78` completed the bounded note non-results alignment, `statcalib` hierarchy de-emphasis, layout warning closeout, and scope-bounded note calibration record honestly.
- `T78` introduces no deferred/rejected warning and opens no new risk.
- `R35` is closed by `T78`; `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T79` is the next bounded mainline task because the repo now needs one explicit gate to judge whether the current note/results/claim-evidence/risk stack is already sufficient for bounded prose reopen, rather than reopening the manuscript directly.
- `T79` must stay docs-only, mainline-only, and must not widen into benchmark/HIL reruns, `.tflite` execution, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or direct full-manuscript expansion.

## 2026-06-12 Captain Final Supersession (T77 closeout)

- Current unique task: `T78: 论文 note-draft 非结果层校准、statcalib 层级降权与排版 warning 收口`
- Task package: `docs/tasks/Phase2/T78_paper_note_alignment_statcalib_hierarchy_and_layout_closeout.md`
- `T77` has been judged `PASS_WITH_WARNINGS`.
- `T77` completed the bounded note results-layer sync, T76 preview-source / stable-ID traceability hardening, local note compile refresh, and exact-path cleanup of temporary render residue honestly.
- `T77` warning classification:
  - `N1` whole-file `.tex` still contains unsynchronized non-results legacy hunks = `deferred -> R35`
  - `N2` `statcalib` still sits visually too close to the main results layer inside `Numerical Results` = `deferred -> R35`
  - `N3` note `.log` still contains `Underfull \hbox` layout warnings = `deferred -> R35`
  - `N4` section-scope proof still relies on manifest / `% T77-SOURCE` comments rather than a more mechanical audit = `deferred -> R35`
- `R34` is closed by `T77`; `R35` is new; `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T78` is the next bounded mainline task because the repo now needs note non-results calibration, `statcalib` hierarchy de-emphasis, and layout closeout before any paper reopen gate, not another experiment or a premature manuscript expansion.
- `T78` must stay docs-only, mainline-only, and must not widen into benchmark/HIL reruns, `.tflite` execution, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or full-manuscript reopen.

## 2026-06-12 Captain Final Supersession (T76 closeout)

- Current unique task: `T77: 论文 note-draft 结果层同步与 T76 traceability hardening`
- Task package: `docs/tasks/Phase2/T77_paper_note_results_sync_and_traceability_hardening.md`
- `T76` has been judged `PASS_WITH_WARNINGS`.
- `T76` completed the bounded rendered-QA / Results-assembly step honestly: the repo now has reviewed preview PNGs, contact sheet, PDF bundle, rendered-QA notes, callout sheet, and section-assembly materials under the already locked `T75` asset boundary.
- `T76` warning classification:
  - `N1` preview-source 聚合行字段语义复用 = `deferred -> R34`
  - `N2` `.tmp_t76_*` 探针/缓存残留 = `accepted`
  - `N3` 逐图 QA 结论未内联完整上游 `T74-*` stable ID = `deferred -> R34`
- `R34` is new; `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T77` is the next bounded mainline task because the repo now needs note-draft results-layer sync plus traceability hardening, not another experiment or a premature full-manuscript reopen.
- `T77` must stay docs-only, mainline-only, and must not widen into benchmark/HIL reruns, `.tflite` execution, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or full-manuscript reopen.

## 2026-06-12 Captain Final Supersession (T75 closeout)

- Current unique task: `T76: Rendered figure QA and results-section assembly pack`
- Task package: `docs/tasks/Phase2/T76_rendered_figure_qa_and_results_section_assembly_pack.md`
- `T75` has been judged `PASS`.
- `T75` completed the bounded authoring step honestly: the repo now has locked main-text Results prose, caption/placement notes, appendix bridge notes, do-not-write guardrails, and three publication-facing `T75-FIG-*` assets that trace back to `T74` stable IDs.
- `T75` has no blocking issue and no deferred/rejected warning; this closeout opens no new risk item.
- The carry-forward notes are operational rather than blocking: the current worktree still requires precise staging discipline, and rendered preview QA should now be handled by one bounded follow-up task instead of silently widening `T75`.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T76` is the next bounded mainline task because the repo now needs rendered figure QA plus manuscript-facing Results-section assembly under the already locked `T75` asset and wording boundary.
- `T76` must stay docs-only, mainline-only, and must not widen into benchmark/HIL reruns, `.tflite` execution, real-board execution, theory-branch work, sidecar promotion, or full-manuscript reopen.

## 2026-06-12 Captain Sidecar Supersession（PSE1）

- 当前 sidecar 治理入口已切换为 `docs/sidecar/` 下的 PSE1 精简治理文档。
- 任务包：`docs/tasks/Phase2/PSE1_sidecar_main_controlled_governance_refresh.md`
- 旧 `.wt/tcn`、`.wt/teach`、`.wt/bank`、`.wt/ctrl` 不再作为必须同步的长期分支；它们退役为 read-only reference。
- 旧 S0 思路已收编到 `docs/sidecar/lane_plans/`。
- 后续 sidecar 默认在 main 当前代码基础上做新增-only helper / standalone module / task-scoped config；需要并行或长跑隔离时，再从当前 main 稳定点新开短生命周期 worktree 或 clean clone。
- sidecar 结果仍必须写入 `runs/sidecar/<lane_id>/<run_id>/`，不得改写主线历史 run root，不得进入主线事实口径，除非后续 Captain promotion gate 明确批准。
- 本 supersession 不改变当前唯一主线任务 `T76`，不授权运行 sidecar 实验，不授权创建 `runs/sidecar`。

## 2026-06-12 Captain Final Supersession (T74 closeout)

- Current unique task: `T75: Main-text results prose and final figure authoring pack`
- Task package: `docs/tasks/Phase2/T75_maintext_results_prose_and_final_figure_authoring_pack.md`
- `T74` has been judged `PASS`.
- `T74` completed the paper-ready simulation/material packaging step honestly: stable-ID 表/图/补充说明、caption pack、insertion map、traceability assets 和 submission-material gap checklist 已全部落地，且没有源码、测试、`runs/`、`artifacts/` 或治理文档漂移。
- `T74` has no blocking issue and no deferred/rejected warning; this closeout opens no new risk item.
- The only non-blocking carry-forward note is commit-time staging discipline: the current worktree contains coexisting captain-side governance diffs, so future commits should use precise staging.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T75` is the next bounded mainline task because the repo now needs one stronger authoring layer that converts the `T74` stable-ID route into main-text Results prose, final figure assets, caption locks and appendix bridges.
- `T75` must stay docs-only, mainline-only, and must not widen into benchmark/HIL reruns, `.tflite` execution, real-board execution, theory-branch work, sidecar promotion, or full-manuscript reopen.

## 2026-06-08 Captain 并行 Sidecar 治理设置

- `PSE0：并行 sidecar 扩展实验治理设置` 已作为 docs-only Captain 设置任务加入。
- 任务包：`docs/tasks/Phase2/PSE0_parallel_sidecar_extension_governance_setup.md`
- 治理规则：`docs/sidecar/parallel_sidecar_extension_governance.md`
- worktree 计划：`docs/sidecar/parallel_sidecar_worktree_plan.md`
- `PSE0` 不替代也不执行主线当前唯一任务；在当时时点，主线任务已切换为 `T71`。
- `PSE0` 不创建 worktree、branch、run root、experiment、`.tflite` smoke、real-board smoke 或 benchmark output。
- PSE1 已 supersede 旧“必须使用 `codex/sidecar-*` 长期分支和隔离 worktree”的执行组织方式；后续默认由 main 控制台治理，必要时再开短生命周期 worktree / clean clone。
- 后续 sidecar 工作仍必须使用 `runs/sidecar/<lane_id>/...` run root。
- 后续 sidecar 输出在后续 Captain promotion gate 批准主线任务包之前，只能保持为 sidecar candidate。

## 2026-06-08 Captain Wave A Sidecar Worktree Setup

- 已创建 Wave A 四个隔离 worktree：
  - `.wt/tcn` -> `codex/sidecar-temporal-tcn-residual`
  - `.wt/teach` -> `codex/sidecar-adaptive-teacher-replay`
  - `.wt/bank` -> `codex/sidecar-gain-scheduled-bank-sim`
  - `.wt/ctrl` -> `codex/sidecar-atomic-commit-rollback`
- 已在各自 worktree 中写入中文 `S0_design` 任务包。
- 本轮未运行 sidecar 实验，未创建 `runs/sidecar`，未启动 benchmark、训练、`.tflite` smoke 或 real-board smoke。
- 在当时时点，main 分支当前唯一主线任务已切换为 `T71`；main 分支主线工作与四个 sidecar worktree 继续保持独立。
- 路径说明：使用 `.wt/<short>` 是为了规避 `.worktrees/<long-name>` 在 Windows 完整 checkout 时触发的 `Filename too long`。

## 2026-06-05 Captain Final Supersession

- Current unique task: `T68: FR8 statcalib generated-only robustness bounded benchmark`
- Task package: `docs/tasks/Phase2/T68_fr8_statcalib_generated_only_robustness_bounded_benchmark.md`
- `T67` has been judged `PASS_WITH_WARNINGS`.
- T67 warning classification:
  - `N1` source-worktree scope-external PDF diff but clean-clone launch preserved provenance = `accepted`
  - `N2` equal-mean tie is not represented explicitly in `better_parameter_point_by_mean_ler` = `accepted`
  - `N3` two comparison rows remain `mixed` = `deferred -> R24`
- `T67` closes the gross teacher-anchor dependence question, but it does not close `R24`.
- `T24` remains the authoritative historical frozen ranked table; `T64/T65/T66/T67` must not be used to rewrite it.
- `T68` is a bounded generated-only robustness benchmark only. It must not change statcalib/runtime semantics, widen into `.tflite` or real-board, or mix mainline experiment work with theory-only branch materials.

# Handoff

## 1. 当前状态

Authoritative status note (`2026-06-15`, Captain closeout):

- 当前唯一任务：`T90: 训练链 clean-CPU 同机 repeated-run 一致性证据包`
- 任务包：`docs/tasks/Phase2/T90_training_same_host_repeated_run_consistency_pack.md`
- 当前主线优先级：优先在不改写 frozen mainline、不碰 theory 分支、也不碰 hardware 的前提下，把 `T50` 的单次 bounded train+eval rerun 强化为一份 same-host repeated-run consistency pack，而不是恢复 prose 扩写或把仓库写成 submission-ready completed。
- `T89` 已完成并被 Captain 接受为 `PASS`；其 non-blocking notes 全部按 accepted operational reminder 处理，未新开 deferred/rejected 风险。
- 如下方旧状态行仍提到 `T89` 或更早任务，以本条 authoritative note 为准。

- 日期：`2026-06-14`
- 阶段：`Phase 2: Controlled Development`
- 决策：`Go`
- 当前子模式：`Research Reality Recovery Mode`
- 当前唯一任务：`T90: 训练链 clean-CPU 同机 repeated-run 一致性证据包`
- 任务包：`docs/tasks/Phase2/T90_training_same_host_repeated_run_consistency_pack.md`
- 当前主线优先级：优先在 clean CPU-only lane 上补 same-host repeated-run consistency，而不是继续改写 note、提前写成 submission-ready completed；real-board execution 因当前暂无 `Linux + FPGA` 硬件宿主而继续维持最低优先级 backlog

Captain continuity note:

- The authoritative current task for all new worker action is `T90: 训练链 clean-CPU 同机 repeated-run 一致性证据包`.
- Authoritative task package: `docs/tasks/Phase2/T90_training_same_host_repeated_run_consistency_pack.md`.
- `T89` is complete and accepted as `PASS`.
- `T89` non-blocking notes are all accepted as operational reminders; no new deferred/rejected warning-derived risk is opened by this closeout.
- `T90` is the single recommended next task after `T89`.
- `T90` is a same-host training reproducibility hardening task only. It is not permission to widen into note rewrite, benchmark/HIL/runtime semantics changes, `.tflite` portability, real-board success claims, theory-branch mergeback, sidecar-promotion scope, venue-template adaptation, or submission-ready completion.
- `T37` / real-board execution remains `blocked + lowest-priority backlog`; it is not the next mainline step while hardware host conditions remain unavailable.
- The theory branch remains isolated from this mainline task.
- If any older line below still mentions `T87` or earlier tasks as current, treat it as historical carry-forward text only.

## 2026-06-05 Captain Update (T67 closeout)

- `T67` review accepted by Captain as `PASS_WITH_WARNINGS`.
- Blocking issues: none.
- Warning classification:
  - `N1` source-worktree scope-external PDF diff but clean-clone launch preserved provenance = `accepted`
  - `N2` equal-mean tie is not represented explicitly in `better_parameter_point_by_mean_ler` = `accepted`
  - `N3` two comparison rows remain `mixed` = `deferred -> R24`
- `T67` is complete. It closes the gross teacher-anchor dependence question for the bounded statcalib lane: non-`ukf` teachers remain competitive and can still beat both frozen anchors across all four locked scenarios.
- `T67` does not close `R24`, does not validate `.tflite`, does not validate real-board behavior, and does not upgrade statcalib into a mature calibration comparator.
- Current unique task is now `T68: FR8 statcalib generated-only robustness bounded benchmark`.
- `T68` must keep the T24 frozen table authoritative, keep statcalib as a separately labeled extension lane, test generated-only robustness under the strongest non-`ukf` teachers only, and remain separate from theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T68_fr8_statcalib_generated_only_robustness_bounded_benchmark.md`.

## 2026-06-08 Captain Update (T68 closeout)

- `T68` review accepted by Captain as `PASS_WITH_WARNINGS`.
- Blocking issues: none.
- Warning classification:
  - `N1` full generated-only winner set remains a tie, not a unique final threshold = `deferred -> R24`
  - `N2` some predeclared candidates remain `mixed` even though the bounded existence question is closed = `deferred -> R24`
  - `N3` clean short-path clone launch boundary must remain visible in downstream retellings = `accepted`
- `T68` is complete. It closes the bounded generated-only existence question for the statcalib extension lane: there are now four full generated-only winners in the predeclared grid, and the strongest clean answer is the tied `window_variance_t001 = t003 = t005` set.
- `T68` does not close `R24`, does not validate `.tflite`, does not validate real-board behavior, and does not upgrade statcalib into a mature calibration comparator.
- Current unique task is now `T69: FR8 statcalib clean-winner tie-break bounded benchmark`.
- `T69` must keep `T24` authoritative, keep statcalib as a separately labeled extension lane, stay inside the clean-winner candidate set plus frozen anchors, and remain separate from theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T69_fr8_statcalib_clean_winner_tiebreak_bounded_benchmark.md`.

## 2026-06-01 Captain Update (T66 closeout)

- `T66` review accepted by Captain as `PASS_WITH_WARNINGS`.
- Blocking issues: none.
- Warning classification:
  - `N1` duplicate-running progress-log artifact after same-run-root timeout relaunch = `accepted`
  - `N2` aggregate-best vs stability-best split = `deferred -> R24`
  - `N3` `static_bias_theta / statcalib_high_threshold` best row still carries aggregate `statcalib_status = mixed` = `deferred -> R24`
- `T66` is complete. It closes one bounded local-grid robustness gap for the statcalib extension lane under clean provenance.
- `T66` does not close `R24`, does not validate `.tflite`, does not validate real-board behavior, and does not upgrade statcalib into a mature calibration comparator.
- Current unique task is now `T67: FR8 statcalib teacher-anchor dependence bounded benchmark`.
- `T67` must keep the T24 frozen table authoritative, keep statcalib as a separately labeled extension lane, test teacher-anchor dependence only, and remain separate from theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T67_fr8_statcalib_teacher_anchor_dependence_bounded_benchmark.md`.

## 2026-05-29 Captain Update (T65 closeout)

- `T65` review accepted by Captain as `PASS_WITH_WARNINGS`.
- Blocking issues: none.
- Warning classification:
  - `N1` mixed-diff scope acceptance depends on explicit user/captain clarification = `accepted`
  - `N2` T64-specific audit helper is intentionally narrow, not generic FR8 framework = `accepted`
  - `N3` review wording should have stated the clarification dependency more explicitly = `accepted`
- `T65` is complete. It closes the T64 report/artifact consistency gap and makes the T64 result pack self-audited through code, tests, and an explicit audit doc.
- `R28` should now be treated as closed by `T65`.
- `T65` does not close `R24`, does not validate `.tflite`, does not validate real-board behavior, and does not upgrade statcalib into a mature calibration comparator.
- Current unique task is now `T66: FR8 statcalib sensitivity bounded benchmark`.
- `T66` must keep the T24 frozen table authoritative, rerun only bounded anchor-plus-variant lanes, preserve clean provenance, and remain separate from theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T66_fr8_statcalib_sensitivity_bounded_benchmark.md`.

## 2026-05-29 Captain Update (T64 closeout)

- `T64` review accepted by Captain as `PASS_WITH_WARNINGS`.
- Blocking issues: none.
- Warning classification:
  - `N1` execution-shape wording drift in `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md` = `deferred` -> `R28`
  - `N2` finish-timestamp provenance wording drift in `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md` = `deferred` -> `R28`
  - `N3` extension-lane over-interpretation boundary = `deferred` -> `R24`
- `T64` is complete. It closes one bounded FR8 extension-lane benchmark with clean provenance and exact frozen-subset preservation against `T24`.
- `T64` does not close `R24`, does not validate `.tflite`, does not validate real-board behavior, and does not authorize a rewrite of the historical `T24` frozen ranked table.
- Current unique task is now `T65: FR8 extension-lane consistency guard and report closeout`.
- `T65` must harden report/artifact consistency and regression guards only. It must create no run root, modify no historical benchmark artifact, and remain separate from theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T65_fr8_extension_lane_consistency_guard_and_closeout.md`.

2026-05-24 Captain supersession:

- Current unique task: `T47: Paper ablation result-pack and material ledger`
- Task package: `docs/tasks/Phase2/T47_paper_ablation_result_pack_and_material_ledger.md`
- `T56` is complete and accepted as `PASS`.
- T56 warning classification: all non-blocking items are `accepted`; there are no `deferred` or `rejected` items.
- Mechanism claims are now explicitly reframed into retain / weaken / retire / reframe / still-open.
- The pure I1 lower-clip intervention remains mixed and mostly harmful across the locked 6-seed pack; the simple harmful-instability framing is not supported as a general mechanism explanation, and `C4` remains `partial`.
- Any second intervention lane remains `deferred pending better question`.
- `T47` may proceed, but only as a docs-only hedge-conditioned paper-material lane. Do not treat it as unconditional paper expansion or mechanism closure.

## 2026-05-24 Captain Update (T47 closeout)

- `T47` review accepted by Captain as `PASS`.
- Blocking issues: none.
- Warning classification for T47: all non-blocking items `accepted`; there are no `deferred` or `rejected` warning items from this review.
- `T47` is complete. The paper-facing ablation/material ledger is frozen honestly, and `FR7` remains the largest missing ablation item.
- Current unique task is now `T57: FR7 feature/teacher ablation re-execution under locked T24 protocol`.
- `T57` is bounded to the frozen four scenarios, the fixed six-mode feature-ablation set, and `repeats=2`. It must not retrain, touch source-tree code/config, or reopen `.tflite`, real-board, cleanup, benchmark expansion, comparator expansion, or intervention scope.
- Next worker-facing task package: `docs/tasks/Phase2/T57_fr7_feature_teacher_ablation_reexecution.md`.

## 2026-05-26 Captain Update (T57 closeout)

- `T57` review accepted by Captain as `PASS`.
- Blocking issues: none.
- `T57` review does not introduce any new `deferred` or `rejected` warning item.
- `T57` is complete. `FR7` is now a bounded frozen-set ready result table, but the mechanism interpretation remains explicitly non-causal.
- The strongest paper-facing caution from `T57` is that `hybrid_no_teacher_params` wins all 4 scenarios, so simple teacher-necessity attribution remains unsafe.
- Current unique task is now `T58: FR6 multi-seed mechanism/intervention figure pack`.
- `T58` is docs-only. It must reuse existing `T54/T55/T56` evidence, must not run new benchmark or intervention work, and must not touch theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T58_fr6_multi_seed_mechanism_intervention_figure_pack.md`.

## 2026-05-26 Captain Update (T58 closeout)

- `T58` review accepted by Captain as `PASS_WITH_WARNINGS`.
- Blocking issues: none.
- T58 warning classification: `N1 accepted`, `N2 accepted`, `N3 accepted`, `N4 accepted`.
- `T58` review introduces no `deferred` or `rejected` warning item, so no new warning-derived risk is opened.
- `T58` is complete. `FR6` is now closed as a bounded descriptive figure pack built from existing `T54/T55/T56` evidence only.
- `T58` does not close `R10`, does not upgrade `C4` beyond `partial`, and does not upgrade `.tflite`, real-board, or expanded benchmark evidence.
- Current unique task is now `T59: Statcalib separate comparator lane integration and bounded smoke`.
- `T59` is a mainline experiment-branch bounded integration/smoke task. It must keep `statcalib` as a separate comparator lane, must not rewrite frozen `T24` semantics, and must not touch theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T59_statcalib_comparator_lane_integration_and_smoke.md`.

## 2026-05-26 Captain Update (T59 closeout)

- `T59` review accepted by Captain as `PASS_WITH_WARNINGS`.
- Blocking issues: none.
- Warning classification:
  - `W1` cross-mode `teacher_mode` fallback coupling = `deferred`
  - `W2` smoke-doc key-name mismatch = `accepted`
  - `W3` dirty-worktree smoke provenance weakness = `deferred`
- Deferred items from `T59` are now written into risks and remain open before any `FR8` task.
- `T59` is complete. It closes separate-lane integration, status propagation, and one bounded smoke only.
- `T59` does not open `FR8`, does not close `R24`, and does not upgrade the evidence to formal comparator ranking.
- Current unique task is now `T60: Statcalib lane isolation and regression hardening`.
- `T60` is a mainline experiment-branch code/test hardening task. It must isolate `statcalib.teacher_mode`, add regression coverage, create no new run root, and remain separate from theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T60_statcalib_lane_isolation_and_regression_hardening.md`.

## 2026-05-27 Captain Update (T60 closeout)

- `T60` review accepted by Captain as `PASS`.
- Blocking issues: none.
- `T60` review introduces no new warning item that needs `accepted / deferred / rejected` handling.
- `T60` closes the T59 cross-mode semantics blocker: `W1` is resolved and `R26` should now be treated as closed.
- `R27` remains open, but narrower: T60 closes the regression-coverage gap and leaves only the clean-provenance / fairness-sanity blocker before any `FR8` task.
- `T60` is complete. It hardens semantics and tests only; it does not rerun benchmark evidence and does not upgrade the lane to formal comparator ranking.
- Current unique task is now `T61: Statcalib clean-provenance fairness sanity rerun`.
- `T61` is a mainline experiment-branch bounded rerun/audit task. It must start from a clean committed worktree, reuse the existing T59 smoke matrix with `repeats=2`, create exactly one T61-scoped run root, and remain separate from theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T61_statcalib_clean_provenance_fairness_sanity.md`.

## 2026-05-27 Captain Update (T61 closeout)

- `T61` review accepted by Captain as `BLOCK`.
- Blocking issues:
  - launch clean `HEAD=9174065`, but final `summary.json git_commit=6058f42`
  - mid-run branch movement means the rerun does not have a single defensible commit identity
- `T61` is not complete. It preserved the bounded fairness signal, but it did not repair the provenance blocker.
- Current unique task is now `T62: Statcalib provenance-isolated fairness rerun`.
- `T62` is blocking-only. It must rerun the exact same bounded matrix on clean committed `main`, in one uninterrupted invocation, with no same-run resume and no theory-branch mixing.
- Next worker-facing task package: `docs/tasks/Phase2/T62_statcalib_provenance_isolated_fairness_rerun.md`.

## 2026-05-27 Captain Update (T62 closeout)

- `T62` review accepted by Captain as `PASS`.
- Blocking issues: none.
- `T62` review introduces no new warning item that needs `accepted / deferred / rejected` handling.
- `T62` closes the specific provenance blocker that caused `T61` to fail: launch / finish / `summary.json` commit identity now matches across one clean `main` invocation.
- `R27` should now be treated as closed by `T62`.
- `T62` still does not open `FR8`, does not close `R24`, and does not upgrade the evidence beyond mock-backed software-HIL bounded sanity evidence.
- Current unique task is now `T63: FR8 statcalib comparator gate review`.
- `T63` is docs-only. It must decide whether a bounded FR8 task should exist at all, using only already existing repository evidence and without touching theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T63_fr8_statcalib_comparator_gate_review.md`.

## 2026-05-27 Captain Update (T63 closeout)

- `T63` review accepted by Captain as `PASS`.
- Blocking issues: none.
- `T63` review introduces no new warning item that needs `accepted / deferred / rejected` handling.
- `T63` closes the pre-FR8 gate-discussion lane only. It is not FR8 evidence and it does not close `R24`.
- `R27` remains closed by `T62`.
- Current unique task is now `T64: FR8 statcalib extension-lane bounded benchmark`.
- `T64` must preserve the historical frozen five-mode table, add `statcalib` only as a separately labeled extension lane, keep clean provenance, and remain isolated from theory-only branch materials.
- Next worker-facing task package: `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md`.

Captain closeout note after T56:

- `T56` is complete. `docs/review/T56_review.md` verdict = `PASS`; blocking issues = none.
- T56 warnings are all classified as `accepted`; there are no `deferred` or `rejected` warning items from this review.
- T56 freezes the current mechanism-claim boundary after `T55`: some claims are retained, some weakened, some retired, one reframed, and some remain explicitly open.
- T56 also changes the interpretation of the next step: the project does not auto-open a second intervention lane, and `T47` is only allowed as a downstream paper-material ledger under explicit mechanism-hedge wording.
- The active next task is `T47`, but only in that restricted docs-only form.

Captain closeout note after T55:

- `T55` is complete. `docs/review/T55_review.md` verdict = `PASS`; blocking issues = none.
- T55 adds the first bounded targeted intervention evidence on the same 6-seed pack and frozen four scenarios.
- T55 warnings are all classified as `accepted`; there are no `deferred` or `rejected` warning items from this review.
- T55 concludes that the pure I1 lower-clip intervention is mixed and mostly harmful (harms 4/6, helps 2/6), so the earlier simple harmful-instability framing is not supported as a general explanation.
- The active next task is `T56`, not `T47`, because the project now needs a mechanism-claim reframing gate before any paper-material packaging or second intervention decision.

Captain closeout note after T54:

- `T54` is complete. `docs/review/T54_review.md` verdict = `PASS`; blocking issues = none.
- T54 upgrades the mechanism story from single-seed diagnostic evidence to bounded multi-seed diagnostic generalization across the locked 6-seed pack.
- T54 warnings are all classified as `accepted`; there are no `deferred` or `rejected` warning items from this review.
- T54 refines the observed pattern into quiet / classic / universal categories, so `C4` remains `partial` and the project still lacks bounded intervention evidence.
- The active next task is `T55`, not `T47`, because the next smallest unresolved gap is the pure I1 intervention question rather than paper-material freezing.

Captain closeout note after T46:

- `T46` is complete. `docs/review/T46_review.md` verdict = `PASS`; blocking issues = none.
- T46 correctly freezes the mechanism-evidence plan without upgrading any single-seed result into multi-seed confirmation or causal proof.
- T46 non-blocking comments are all treated as `accepted`; there are no `deferred` or `rejected` warning items from this review.
- The active next task recorded at the end of T46 was `T54`, but that recommendation is now superseded by T54 closeout and the current-task switch to `T55`.

Captain closeout note after T45:

- `T45` is complete. `docs/review/T45_review.md` verdict = `PASS`; blocking issues = none.
- T45 freezes the benchmark-expansion protocol at the policy level without changing any benchmark code, config, runtime path, or deployment boundary.
- T45 correctly keeps the T24 frozen set separate from any future expansion lane and does not upgrade reference ideas into current evidence.
- T45 warnings are all classified as `accepted`; there are no `deferred` or `rejected` warning items from this review.
- The active next task is `T46`, not `T45`, because the next tightest paper-facing gap is still mechanism evidence rather than benchmark-policy wording.

Captain closeout note after T53:

- `T53` is complete. `docs/review/T53_review.md` verdict = `PASS`; blocking issues = none.
- T53 adds a bounded mainline theory document for personal understanding and later paper support, without changing any code, benchmark, runtime, or deployment boundary.
- T53 correctly explains the repository’s current narrow claim: teacher-anchored residual-`b` correction inside a dual-loop, runtime-constrained linear fast path.
- T53 does not upgrade true `.tflite` runtime, real-board HIL, or paper-grade benchmark breadth.
- The active next task is `T45`, not `T53`, because the next unresolved paper-facing question is whether the frozen-set benchmark story is sufficient or whether a controlled expansion lane must be formalized first.

Captain closeout note after T44:

- `T44` is complete. `docs/review/T44_review.md` verdict = `PASS`; blocking issues = none.
- T44 correctly freezes the paper-facing truth boundary without upgrading any evidence level.
- T44 outputs now exist for freeze snapshot, claim/evidence table, code truth audit, reproducibility audit, figure/result ledger, paper-risk table, and human brief.
- T44 concludes that current evidence supports a bounded paper plan, not a strong full-paper submission package, and recommends the next bounded lanes `T45-T47`.
- The active next task is `T53`, not `T44`, because the user then requested a mainline theory-analysis document aimed at personal understanding rather than further paper expansion or new experiments.

Captain closeout note after T43:

- `T43` is complete. `docs/review/T43_review.md` verdict = `PASS`; blocking issues = none.
- T43 comments are non-blocking and were handled as `accepted`: subsection-6 neutrality, placeholder citation markers, drafting annotations, and inline claim-reference formatting all remain later cleanup items, not evidence blockers.
- T43 does not change any risk status, evidence level, hardware status, `.tflite` status, reproducibility status, or repo-noise fact boundary.
- After T43, the project does not continue into more paper prose by default. The active next task is `T44`, which explicitly switches the project into recovery-first claim/evidence/material audit mode.

Captain closeout note after T41:

- `T41` is complete. `docs/review/T41_review.md` verdict = `PASS`; blocking issues = none.
- T41 comments are non-blocking and were handled as minor document corrections; there are no `deferred` warnings from this review.
- T41 does not change any risk status, evidence level, or repo-noise fact boundary. It closes Milestone 2K and sets the paper-positioning gate outcome only.
- The active next task is `T42`, not `T41`, `T35`, `T34`, `T33`, `T40`, `T39`, `T31`, or `T38`. Any older T41/T35-next wording later in this handoff is superseded by this status block and `docs/04_task_board.md`.

Captain closeout note after T42:

- `T42` is complete. `docs/review/T42_review.md` verdict = `PASS`; blocking issues = none.
- T42 comments are non-blocking and were handled as accepted framing guidance: subsection-6 wording was softened to neutral survey language, and the method-forward title remains only a working recommendation pending later human/Captain override.
- T42 does not change any risk status, evidence level, hardware status, `.tflite` status, or repo-noise fact boundary. It extends the paper scaffold only.
- The active next task is `T43`, not `T42`, `T41`, `T35`, `T34`, `T33`, `T40`, `T39`, `T31`, or `T38`. Any older T42-next wording later in this handoff is superseded by this status block and `docs/04_task_board.md`.

## 2. 本轮已完成

1. 完成了 `T1`，固定恢复期解释器分工，并跑通最小 P0 smoke
2. 完成了 `T2`，补充了 `docs/recovery_bootstrap/P0_smoke_bootstrap.md`
3. 完成了 `T3`，补充了：
   - `docs/tasks/P0/T3_hil_p4_boundary_audit.md`
   - `docs/03_hil_p4_boundary_audit.md`
4. 完成了 `T4`，补充了：
   - `docs/tasks/P0/T4_software_hil_bootstrap_and_smoke.md`
   - `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
5. 完成了 `T5`，补充了：
   - `docs/tasks/P0/T5_repo_noise_governance.md`
   - `docs/06_repo_noise_governance.md`
6. 完成了 `T6`，补充了：
   - `docs/tasks/P0/T6_software_hil_reverification.md`
   - `docs/recovery_bootstrap/P3_software_hil_bootstrap.md` 的二次复验证据
7. 完成了 `T7`，补充了：
   - `docs/tasks/P0/T7_p4_benchmark_reverification.md`
   - `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
   - `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
8. 完成了 `T8`，补充了：
   - `docs/tasks/P0/T8_gate_review_and_phase_decision.md`
   - `docs/review/T8_gate_review.md`
9. 完成了 `T9`，补充了：
   - `docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`
   - `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md` 的四模式复验证据
10. 完成了 `T10`，补充了：
   - `docs/tasks/P0/T10_gate_review_after_t9.md`
   - `docs/review/T10_gate_review.md`
11. 完成了 `T11`，补充了：
   - `docs/tasks/P0/T11_recovery_dependency_manifest.md`
   - `requirements-recovery.txt`
   - `docs/recovery_bootstrap/P0_smoke_bootstrap.md`、`docs/recovery_bootstrap/P3_software_hil_bootstrap.md`、`docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md` 中对 root manifest 的统一引用
12. 完成了 `T12`，补充了：
   - `docs/tasks/P0/T12_software_hil_determinism_recovery.md`
   - `physics/syndrome_measurement.py`
   - `cnn_fpga/runtime/fast_loop_emulator.py`
   - `docs/recovery_bootstrap/P3_software_hil_bootstrap.md` 的确定性复验证据
13. 完成了 `T13`，补充了：
   - `docs/tasks/P0/T13_recovery_exit_and_closeout.md`
   - `docs/review/T13_recovery_exit_review.md`
   - recovery exit 的阶段/状态切换
14. 同步更新了治理文档中的 task board、decision log、legacy audit 与风险口径
15. 作为 Phase 2 Captain 初始化，按 `docs/reference/AI_coding_workflow.md` 校正了 00~08 治理文档，并建立 Phase 2 任务包队列：
   - `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`
   - `docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md`
   - `docs/tasks/Phase2/T16_p4_evidence_gate_review.md`
   - `docs/tasks/Phase2/T17_training_manifest_bootstrap.md`
   - `docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`
   - `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`
   - `docs/tasks/Phase2/T20_real_board_readiness_checklist.md`
16. 完成了 `T14`，补充了：
   - `docs/protocols/benchmark/P4_benchmark_development_protocol.md`
   - `docs/review/T14_protocol_audit_review.md`
17. 完成了 `T15`，补充了：
   - `docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md` 的 Worker output
   - `docs/protocols/benchmark/P4_benchmark_development_protocol.md` 的 T15 execution record
   - `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md` 的 T15 关系说明
   - `docs/review/T15_frozen_smoke_review.md`
   - 新 run dir：`runs/p4_benchmark/p4multis_20260508_221718_b82874_48280`
18. Captain 已按 `PASS_WITH_WARNINGS` 处理 `T15` review：
   - N1 accepted：handoff / task board 状态由 Captain 修正
   - N2 deferred：`hybrid_residual_b` teacher diagnostics 全零交给 `T16` gate review 判断
   - N3 accepted：strong-baseline config 不含 `static_linear` / `cnn_fpga`，所以 delta rows 为 null 是预期设计后果
19. 完成了 `T16`，补充了：
   - `docs/review/T16_p4_evidence_gate_review.md`
   - `docs/tasks/Phase2/T16_p4_evidence_gate_review.md` 的 Worker output
20. `T16` gate review verdict = `Conditional`：
   - 允许继续 Phase 2 受控开发
   - 不把 `T15` 升级为正式四场景 frozen benchmark 已恢复
   - 当前更适合优先转向 `T17 / T18` 这类独立 manifest / boundary 任务
   - `hybrid_residual_b` teacher diagnostics 全零保留为非阻塞风险
21. 完成了 `T17`，补充了：
   - `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md`
   - `docs/tasks/Phase2/T17_training_manifest_bootstrap.md` 的 Worker output
22. `T17` 将训练链环境说明与 recovery smoke 依赖说明显式拆开：
   - `requirements-recovery.txt` 继续只覆盖 `P0/P3/P4 recovery smoke`
   - `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md` 单独记录训练链推荐解释器、训练入口、双后端边界与未覆盖项
   - 本轮没有启动训练长跑，也没有把 `DLEnv` 写成跨机器保证
23. Captain 已按 `PASS` 处理 `T17` review：
   - N1 accepted：`torch = 2.8.0.dev20250405+cu128` 是本机 dev build 事实，不能写成跨机器保证
   - N2 accepted：本任务允许用 `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md` 替代 `requirements-train.txt`；训练链可移植性如需增强，后续单开任务
24. 当前唯一任务已切换为 `T18`：
   - 目标是为 `.tflite` export/runtime 路径补独立 manifest 与 boundary smoke plan
   - 必须区分真实 `.tflite` 与 `.tflite.json` / `tflite_stub_v1`
25. 完成了 `T18`，补充了：
   - `docs/evidence_packs/deployment_boundary/TFLite_runtime_bootstrap.md`
   - `docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md` 的 Worker output
26. `T18` 将 `.tflite` 路径的真实 runtime 依赖与 stub 边界显式拆开：
   - 当前机器未安装 `tensorflow` / `tflite_runtime`
   - `export.py`、`evaluate_tflite.py`、`validate_export.py` 入口存在，但真实 runtime 不能写成已恢复事实
   - `tflite_stub_v1` 仅是可追溯回退，不等于真实部署
27. Captain 已按 `PASS` 处理 `T18` review：
   - Blocking issues: none
   - N1 accepted：推荐表述中的 Markdown 引号嵌套只是排版提醒，不影响结论，也不写入 risks
28. `T19` 已完成并通过 review：
   - `docs/review/T19_review.md` verdict = `PASS`
   - `docs/evidence_packs/repo_hygiene/cleanup_tracked_cache_manifest.md` 已固定 tracked cache cleanup 的 9 个目录、命令草案、回滚方案与验收标准
   - tracked `.pyc` 文件共 `116` 个，全部位于 `9` 个 `__pycache__` 目录中
   - 未执行任何物理 cleanup，`runs/` 与 `artifacts/` 仍保持不触碰
29. `T20` 已完成并通过 adversarial review：
   - `docs/review/T20_review.md` verdict = `PASS`
   - `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md` 已形成真板 readiness checklist、前置条件与最小 smoke 验收标准
   - 产物仍只是 readiness / acceptance criteria，不是真板验证
30. 当前唯一任务已切换为 `T21`：
   - 目标是做 Phase 2 milestone review 和 next-phase decision
   - 任务只做只读 review，不运行 benchmark、不执行 cleanup、不调用硬件
31. `T20` 的只读 Worker 输出已就位：
   - 新增 `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
   - 固定了 `board_backend.py` / `fpga_driver.py` 的 placeholder 证据点
   - 固定了真板前置条件、最小 smoke 验收标准与禁止表述
   - 未调用硬件，未修改真板代码，`T20` 已收口
32. `T21` 已完成并由 Captain 接受为 `PASS_WITH_WARNINGS`：
   - 新增 `docs/review/T21_phase2_milestone_review.md`
   - gate decision = `Conditional`
   - 推荐下一唯一任务为 `T22: Real-board smoke execution plan with platform / AXI-map audit and quantitative acceptance thresholds`
   - 本轮未运行 benchmark、未执行 cleanup、未调用硬件
   - `Conditional` gate 被接受，但 formal P4 benchmark、真实 `.tflite` runtime、physical cleanup 与 real-board validation 仍保持 deferred 风险
33. 当前唯一任务已切换为 `T22`：
   - 目标是为后续真板 smoke 制定 execution plan
   - 任务仍不调用硬件、不实现真板 backend、不运行 `backend=board` HIL
34. `T22` 的 Worker 计划层输出已就位，并在随后进入 reviewer / Captain 收口：
   - 新增 `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`
   - 已补 target platform decision points：Linux / Windows / WSL / remote board host
   - 已补 AXI/register map 审计清单，直接对应 `axi_map.py`
   - 已补 DMA buffer 审计清单，直接对应 `dma_client.py`
   - 已补 Layer A-D 量化阈值草案、fail-fast budget、future evidence pack 与 prohibited wording
   - 本轮仍未调用硬件、未运行 `backend=board` HIL、未将产物写成 real-board validation
35. `T22` 已完成并由 Captain 接受为 `PASS_WITH_WARNINGS`：
   - `docs/review/T22_review.md` verdict = `PASS_WITH_WARNINGS`
   - Blocking issues: none
   - N1 out-of-scope governance files：`accepted`，归属 Captain 整合阶段，不归为 Worker 越界
   - N2 preflight `AXI_REGISTER_MAP` repr：`deferred`，后续硬件执行任务需格式化地址表输出
   - N3 `byte_count = 4096` 假设：`deferred`，后续硬件执行任务需用实际 bitstream / DMA contract 确认
36. 当前唯一任务已切换为 `T23`：
   - 原先曾短暂安排为 paper claims roadmap，但 Project Manager 已澄清论文是远期目标，不应作为最近任务一步到位推进
   - 当前 T23 已改为 P4 formal benchmark protocol lock and evidence gap audit
   - 不运行 benchmark、不训练、不调用硬件、不执行 cleanup
   - 任务产物不得把 protocol lock 写成 formal benchmark 结果
37. `T23` 已完成并由 Captain 接受为 `PASS_WITH_WARNINGS`：
   - `docs/review/T23_review.md` verdict = `PASS_WITH_WARNINGS`
   - Blocking issues: none
   - N1 out-of-scope governance sync：`accepted`
   - N2 exact CLI shape：`deferred`，写入 R19，并在 T24 任务包中固定
   - N3/N4 requested metric availability：`deferred`，写入 R19，T24 必须报告实际可用字段与缺失字段
   - `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` 明确写出 `T23 did not run benchmark`
38. 当前唯一任务已切换为 `T24`：
   - 目标是执行 `4 scenarios x 5 modes x repeats=2` 的 bounded formal software revalidation
   - 任务仍固定为 `mock-backed` software HIL，不是 `.tflite` runtime，不是真板验证
   - `statcalib`、soft-information comparator、额外 drift family、CI-driven stopping、true `.tflite` runtime 与真板 smoke 都不得塞进 T24
39. `T24` Worker 已完成执行：
   - Run dir: `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
   - 执行方式：repeat-chunked（`--repeat-start 0 --repeat-stop 1`，然后 `--repeat-start 1 --repeat-stop 2`，最后 `--resume-only`）
   - `missing_runs = []`，20/20 scenario/mode pairs `coverage = 1.0`，40 repeat-runs
   - 四场景 winner 均为 `hybrid_residual_b`，runner-up 均为 `ukf`
   - 请求的统计字段全部存在于 `comparison.csv`
   - `correction_saturation_rate_mean` 全为 0.0，`teacher_scalar_diagnostics.csv` 仅有 header 行
   - Mock-backed software HIL only，不是 `.tflite` runtime、不是 `real_board`、不是 paper-grade expanded benchmark
   - `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` 已更新 T24 execution record (Section 15)
   - `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md` 已更新 Worker output
   - `docs/review/T24_review.md` verdict = `PASS_WITH_WARNINGS`，blocking issues = none
   - Captain 已接受 T24 为 `PASS_WITH_WARNINGS`
   - Warning 分类：
     - N1 `correction_saturation_rate_mean` structural zero：`deferred`，写入 risks，后续机制审计需判断 metric collection bug / genuine zero / not applicable
     - N2 `docs/04_task_board.md` 中环境提示越出 T24 执行结果口径：`accepted`，按 Captain 治理同步说明处理，不影响 T24 verdict
     - N3 `teacher_scalar_diagnostics.csv` header-only / teacher diagnostics 全零：`deferred`，写入 risks，并要求 T25 后优先安排机制证据审计
40. `T25` 已完成并由 Captain 接受为 `PASS_WITH_WARNINGS`：
   - `docs/review/T25_p4_formal_evidence_gate_review.md` verdict = `PASS_WITH_WARNINGS`
   - T24 可作为 completed frozen-set formal software revalidation，但仅限 `mock-backed` software HIL
   - N1 `correction_saturation_rate_mean` structural zero = `deferred` / R20
   - N2 T24 task-board environment-note warning = `accepted`
   - N3 `teacher_scalar_diagnostics.csv` header-only / teacher diagnostics 全零 = `deferred` / R10
   - T25 本身是 review 任务，本轮未启用重复 Claude review
41. 当前唯一任务已切换为 `T27`：
   - 目标是只读追踪 teacher diagnostics 生成、聚合、写出路径，并形成机制证据修复计划
   - 可相邻检查 `correction_saturation_rate_mean` structural zero 的指标路径
   - 不运行新 benchmark、不改源码、不改 config、不执行 cleanup、不调用硬件、不新增 run dir
   - 任务包：`docs/tasks/Phase2/T27_teacher_diagnostics_path_audit.md`
42. `T27` 已完成并由 Captain 接受为 `PASS_WITH_WARNINGS`：
   - `docs/review/T27_teacher_diagnostics_path_audit.md` verdict = `PASS_WITH_WARNINGS`
   - R10 已缩窄为：当前 `hybrid_residual_b` 使用 broadcast teacher features，而 explain 机制只在 `scalar_feature_dim > 0` 时产出 scalar teacher diagnostics，因此当前 hybrid path 是 `data not generated`
   - downstream aggregation / CSV 写出会把部分缺失 diagnostics 压成 `0.0`，形成 missing-vs-zero 语义风险
   - R20 已缩窄为独立 fast-loop saturation counter 路径；当前 T24 零值不再按 teacher diagnostics dead path 处理，但不关闭全局 stress/edge 触发性问题
43. 当前唯一任务已切换为 `T28`：
   - 目标是最小修复 teacher diagnostics missing-vs-zero 输出语义，并用最小 smoke 验证
   - 不扩 formal benchmark、baseline/scenario、statcalib、`.tflite` runtime 或真板范围
   - 任务包：`docs/tasks/Phase2/T28_teacher_diagnostics_semantics_repair.md`

## 3. T45 后的拟议路线图（非当前任务）

`T45`、`T46` 和 `T53` 已完成；下面这些是 recovery 结论导出的下一轮 bounded 路线图。当前被激活的是 `T54`，而不是直接进入旧版 `T47`：

1. `T54`：Phase A multi-seed trace-only generalization probe
   - 先判断 committed-`b` instability 是否在 `20260429` 之外复现
2. `T47`：paper ablation result-pack and material ledger
   - 只在更清楚的机制证据基础上再冻结论文图表、表格和 ablation 缺口

后续 milestone 的粗略方向：

- `Milestone 2P`: mainline evidence hardening
- `Milestone 2Q`: deployment boundary boosters
- `Milestone 2R`: reproducibility and material pack
- `Milestone 2S`: paper re-open gate

这些只表示“下一步应该长什么样”，不表示现在可以执行。
44. `T28` 已完成并由 Captain 接受为 `PASS_WITH_WARNINGS`：
   - `docs/review/T28_review.md` verdict = `PASS_WITH_WARNINGS`
   - T28 输出现在显式区分 `not_applicable` / `not_generated` / observed zero
   - R21 对当前 writer 语义可关闭；R10 进一步缩窄但不关闭
   - N1 duplicate markdown report header row = `deferred`，进入 T29
   - N2 tracked `.pyc` side-effect = `rejected as technical signal`，不应作为有意义改动提交
   - N3 comparison column order change = `accepted`
   - Missing focused tests = `deferred`，记录为后续 aggregation/report writer 风险
45. 当前唯一任务已切换为 `T29`：
   - 目标是修复 `run_p4_multiscenario_benchmark.py::_write_report()` 的重复 markdown header
   - 不运行 benchmark、不新增 run dir、不改变 teacher diagnostics 语义或 benchmark 口径
   - 任务包：`docs/tasks/Phase2/T29_p4_report_header_cleanup.md`
46. `T29` 已完成并由 Captain 接受为 `PASS`：
   - `docs/review/T29_review.md` verdict = `PASS`
   - 删除了 `_write_report()` 中旧的 11-column markdown header
   - 验证结果：`py_compile` passed；静态 `_write_report()` shape check 得到 `header_rows=1`、`column_counts=[12, 12, 12]`
   - 未运行 benchmark、未新增 run dir、未改变 benchmark 语义
   - N1 tracked `.pyc` side-effect = accepted known repo-noise / rejected as technical signal，不作为技术改动提交
47. 当前唯一任务已切换为 `T26`：
   - 目标是做 calibration/statcalib baseline feasibility gate 和最小设计计划
   - 不实现 comparator、不运行 benchmark、不新增 run dir、不改 formal benchmark protocol
   - 任务包：`docs/tasks/Phase2/T26_statcalib_feasibility_gate.md`
48. `T26` 已完成并由 Captain 接受为 `PASS`：
   - `docs/review/T26_review.md` verdict = `PASS`
   - `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md` gate verdict = `CONDITIONAL_GO`
   - statcalib 仍未实现、未验证；只能作为 separate comparator lane 后续推进
   - 未修改 source/config/run/artifact，未运行 benchmark，未新增 run dir
   - non-blocking comments accepted as follow-up constraints：后续 implementation task 必须给出更强 audit trail、exact typed `StatCalibInput` / `StatCalibOutput` 和清晰人读解释
49. `T30` 已切换并完成：
   - 目标是把 T26 的 conceptual comparator lane 收紧为 concrete typed interface contract 和 bounded implementation package
   - 不运行 benchmark、不扩 formal set、不改 existing `ParamMapper` 主线语义、不触碰 `.tflite` 或真板范围
   - 任务包：`docs/tasks/Phase2/T30_statcalib_interface_contract.md`
   - Captain verdict：`PASS`
50. 当前唯一任务已切换为 `T36`：
   - 目标是对既有 `seed=20260429` teacher-representation 结果做 bounded failure-mechanism diagnosis
   - 不重跑 benchmark、不扩新分支、不改模型、不改 formal benchmark 或部署边界
   - 任务包：`docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md`
51. `T36` 已完成并由 Captain 接受为 `PASS`：
   - `docs/review/T36_review.md` verdict = `PASS`
   - Blocking issues: none
   - 诊断报告：`docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md`
   - 小型只读脚本：`cnn_fpga/benchmark/analyze_seed20260429_failure.py`
   - 结论：`20260429` 的收益收缩更像 residual-amplitude / teacher-delta regime instability；response lag、overflow/correction saturation、dead teacher branch 不受当前 artifacts 支持
   - 边界：现有 artifacts 缺 per-window / per-commit trace，因此 sign offset、overshoot chronology、teacher-vs-CNN attribution 仍不能定因果
52. 当前唯一任务已切换为 `T38`：
   - 目标是为 `seed=20260429` 做单 seed trace-export probe
   - 允许一个 T38-scoped bounded rerun，但必须保持 benchmark 语义、baseline/scenario、seed/repeat policy 不变
   - 任务包：`docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md`

## 3. 已验证事实

### 3.1 环境与 P0 smoke

- 默认 `python 3.13.7` 跑最小 benchmark 仍会因缺少 `numpy` 失败
- `C:\ProgramData\anaconda3\python.exe` 已成功跑通：
  - `benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda`
- 根目录现已新增：
  - `requirements-recovery.txt`
  - 当前覆盖包集：`numpy + PyYAML`
  - 当前覆盖范围：`P0/P3/P4 recovery smoke`
  - 当前不覆盖：`DLEnv` 训练链、`.tflite` runtime / export、`real_board` HIL backend
- 当前恢复期解释器分工：
  - 最小 smoke：`C:\ProgramData\anaconda3\python.exe`
  - 训练候选：`C:\ProgramData\anaconda3\envs\DLEnv\python.exe`

### 3.2 HIL / P4 边界

- `run_hil_suite.py`
  - 是 software HIL orchestration 入口
  - 通过 `hil.backend` 选 backend
  - `mock` 路径会构造 mock noise provider，并写出 `hil_events.json` / `hil_summary.json`
- `run_p4_multiscenario_benchmark.py`
  - 直接调用 `run_hil_session(...)`
  - P4 benchmark 的真实性继承自同一条 HIL backend / artifact 链路
- `board_backend.py`
  - 文件顶层直接写明 `Placeholder real-board backend`
  - `schedule_commit(...)` 返回占位元信息
  - `step(...)` 返回空事件列表
- `export.py` + `inference_service.py`
  - 真实 `.tflite` 与 `.tflite.json` stub manifest 两条路径并存
  - runtime 输出会区分 `tflite_service` 与 `tflite_stub_service`

### 3.3 已恢复的最小 software HIL 路径

- 命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- 最新复验运行目录：
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104`
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104`
- 固定口径：
  - backend: `mock`
  - slow-loop mode: `model_artifact`
  - inference service mode: `inproc`
  - inference backend: `artifact_npz`
  - artifact path: `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- 最新关键结果：
  - `n_windows_ready = 2`
  - `n_slow_updates_finished = 2`
  - `n_commits_applied = 2`
  - `fast_budget_violation = 1`
  - `final_ler = 0.454375`
  - `overflow_rate = 0.002`
- 当前表述边界：
  - 该路径已完成逐字一致复验
  - 仍不应写成 `real_board` 或正式多场景 benchmark 已恢复

### 3.4 T12 确定性复验结果

- 两次连续复验命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- 对比 run dir：
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104`
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104`
- 对比结果：
  - `hil_summary.json` 的 SHA256 一致
  - `hil_events.json` 的 SHA256 一致
- 最小修复说明：
  - `RealisticSyndromeMeasurement` 现在支持注入显式 `rng`
  - `FastLoopEmulator` 将快回路噪声 RNG 与测量噪声 RNG 分开，并沿 seed 链显式传递
  - recovery 路径已不再依赖综合征测量中的全局 `np.random`

### 3.5 仓库噪声治理现状

- `.gitignore` 已忽略：
  - `__pycache__/`
  - `runs/`
  - `artifacts/`
- 但当前 Git 历史中仍存在大量已跟踪噪声：
  - 已跟踪缓存/字节码文件：`116`
  - `__pycache__` 目录数：`9`
  - 当前工作区 `.pyc` 总数：`133`
  - 已跟踪 `runs/` 文件：`1841`
  - 已跟踪 `artifacts/` 文件：`110`
- `T5` 已固定恢复期口径：
  - 先治理，后清理
  - `runs/` / `artifacts/` 在恢复期只视作历史证据
  - `__pycache__/` / `.pyc` 需要后续有界 cleanup，但不在当前轮次执行

### 3.6 T7 最小 P4 benchmark 复验结果

- 命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds`
- 新运行目录：
  - `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308`
- 新 protocol / filter 关键结果：
  - `protocol_id = p4_hil_recovery_smoke_v1`
  - `repeats = 1`
  - `seed_pairing = paired`
  - `scenario = static_bias_theta`
  - `modes = static_linear, cnn_fpga`
- 新 comparison 关键结果：
  - `Static Linear final_ler = 1.00890625`
  - `Static Linear overflow_rate = 0.0020625`
  - `CNN-FPGA final_ler = 0.72109375`
  - `CNN-FPGA overflow_rate = 0.002375`
  - scenario winner: `cnn_fpga`
  - `runner_up_gap = 0.2878125`
- 新 repeat HIL summary 关键结果：
  - 两个 mode 的 repeat summary 都是 `backend = mock`
  - `cnn_fpga` repeat 中：
    - `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
    - `inference_service_mode = inproc`
    - `n_slow_updates_finished = 8`
    - `n_commits_applied = 8`
- 当前表述边界：
  - 该路径是 `mock-backed P4 recovery smoke`
  - 不是 `real_board`
  - 不是 `.tflite` runtime 验收
  - 不是正式四场景四模式 frozen benchmark 已恢复

### 3.7 T8 gate review 结论

- gate review 文档：
  - `docs/review/T8_gate_review.md`
- 结论：
  - `Continue Repair`
- 当前不进入 `Go` 的主要原因：
  - `T7` 仍只覆盖 `single-scenario + two-mode + repeats=1`
  - 根目录仍缺少最小依赖 manifest
  - 当时 software HIL 仍是“可复验”而非“逐字确定性复现”
- 当前可以确认的积极结论：
  - 最小 P3/P4 recovery path 都已经重新变成可接力的事实

### 3.8 T9 frozen baseline 单场景全模式 smoke 结果

- 命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds`
- 新运行目录：
  - `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`
- 新 protocol / filter 关键结果：
  - `protocol_id = p4_hil_recovery_smoke_v1`
  - `repeats = 1`
  - `seed_pairing = paired`
  - `scenario = static_bias_theta`
  - `modes = static_linear, window_variance, ekf, cnn_fpga`
- 新 comparison 关键结果：
  - `Static Linear final_ler = 0.99575`
  - `Static Linear overflow_rate = 0.00246875`
  - `Window Variance final_ler = 0.57440625`
  - `Window Variance overflow_rate = 0.00221875`
  - `EKF final_ler = 0.6795`
  - `EKF overflow_rate = 0.0019375`
  - `CNN-FPGA final_ler = 0.7248125`
  - `CNN-FPGA overflow_rate = 0.00290625`
  - scenario winner: `window_variance`
  - `runner_up_gap = 0.10509375`
- 新 repeat HIL summary 关键结果：
  - 四个 mode 的 repeat summary 都是 `backend = mock`
  - 四个 mode 的 repeat summary 都是 `inference_service_mode = inproc`
  - 四个 mode 的 repeat summary 都有：
    - `n_slow_updates_finished = 8`
    - `n_commits_applied = 8`
  - `static_linear / window_variance / ekf` repeat 中：
    - `artifact_path = null`
  - `cnn_fpga` repeat 中：
    - `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- 当前表述边界：
  - 该路径是 `mock-backed P4 recovery smoke`
  - 不是 `real_board`
  - 不是 `.tflite` runtime 验收
  - 不是正式多场景 frozen benchmark 已恢复
  - 当前仍只是 `single-scenario + four-mode + repeats=1`

### 3.9 T10 gate review 结论

- gate review 文档：
  - `docs/review/T10_gate_review.md`
- 结论：
  - `Continue Repair`
- 当前不进入 `Go` 的主要原因：
  - 根目录仍缺少最小依赖 manifest
  - 当时 software HIL 仍是“可复验”而非“逐字确定性复现”
  - `T9` 仍只覆盖 `single-scenario + four-mode + repeats=1`
- 当前可以确认的积极结论：
  - `T9` 已经把 P4 recovery 证据增强到“冻结 baseline 四模式单场景 smoke”
  - 当前仓库更适合先补环境可移植性，而不是继续扩 benchmark 长跑

### 3.10 T11 recovery 期最小依赖 manifest 结果

- task package：
  - `docs/tasks/P0/T11_recovery_dependency_manifest.md`
- 根目录 manifest：
  - `requirements-recovery.txt`
- manifest 当前包含：
  - `numpy`
  - `PyYAML`
- manifest 当前覆盖：
  - `benchmark/compare_full_vs_simplified_ler.py --no-plot`
  - `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
  - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml ...`
- manifest 当前不覆盖：
  - `torch` 训练链 / `DLEnv`
  - `tensorflow` / `tflite-runtime`
  - `.tflite` export/runtime
  - `real_board` HIL backend
  - 去掉 `--no-plot` 后的 `matplotlib`
- 文档同步结果：
  - `README.md`
  - `docs/recovery_bootstrap/P0_smoke_bootstrap.md`
  - `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
  - `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
  都已改为显式引用 `requirements-recovery.txt`

### 3.11 T15 P4 development bounded run 结果

- Review 文档：
  - `docs/review/T15_frozen_smoke_review.md`
- Review verdict：
  - `PASS_WITH_WARNINGS`
  - Blocking issues: none
- 命令口径：
  - interpreter: `C:\ProgramData\anaconda3\python.exe`
  - config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
  - scenarios: `static_bias_theta`, `linear_ramp`
  - modes: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`
  - repeats: `2`
  - seed policy: `--paired-seeds`
- 新运行目录：
  - `runs/p4_benchmark/p4multis_20260508_221718_b82874_48280`
- 运行完整性：
  - `missing_runs = []`
  - 10 个 scenario/mode comparison rows 均 `coverage = 1.0`
  - `raw_rows` 共 20 行，即 `2 scenario x 5 mode x 2 repeat`
- 新 comparison 关键结果：
  - `static_bias_theta` winner: `hybrid_residual_b`
  - `static_bias_theta hybrid_residual_b final_ler_mean = 0.8109015277777778`
  - `static_bias_theta runner_up = ukf`
  - `static_bias_theta runner_up_gap = 0.014468888888888864`
  - `linear_ramp` winner: `hybrid_residual_b`
  - `linear_ramp hybrid_residual_b final_ler_mean = 0.7877551388888888`
  - `linear_ramp runner_up = ukf`
  - `linear_ramp runner_up_gap = 0.023445694444444554`
- 边界：
  - 该 run 是 `development bounded run`
  - 仍是 `mock-backed P4 wrapper over software HIL`
  - 不是 `real_board`
  - 不是 `.tflite` runtime 验收
  - 不是正式四场景 frozen benchmark 已恢复
- Review warning 需后续判断：
  - `hybrid_residual_b` 的 teacher diagnostics 全零，可能是指标收集缺口或 runner 指标路径 bug；不阻塞 LER 证据，但影响机制分析深度
  - `delta_rows` 对 `static_linear` / `cnn_fpga` 为 null 是预期，因为 strong-baseline config 不包含这两个 mode

## 4. 当前判断

项目当前判断已经从“是否还能退出 Recovery”切换为“在继续开发前，下一张 bounded 任务包应该优先补哪块正式证据或环境说明”：

1. `T6` 已确认最小 software HIL 路径可复验
2. `T7` 已确认最小 P4 benchmark 路径可复验
3. `T8` 已明确在 `T7` 证据下仍应继续 `Repair`
4. `T9` 已把 P4 recovery 证据扩到 `single-scenario + four-mode + repeats=1`
5. `T10` 已明确在 `T8 + T9` 证据下仍应继续 `Repair`
6. `T11` 已把 recovery 期最小依赖 manifest 收口到可接力状态
7. 真板 backend 仍是 placeholder，不能被写成已验收能力
8. `.tflite` 路径仍必须区分真实 runtime 与 stub 回退
9. `T12` 已把 bounded software HIL recovery smoke 收口到逐字一致复验
10. `T13` 已确认 recovery exit 条件满足，项目可进入受控继续开发
11. `T14` 已完成 P4 frozen benchmark protocol audit 和 bounded run plan
12. `T15` 已完成双场景、五模式、`repeats=2` 的 development bounded run
13. `T15` review 为 `PASS_WITH_WARNINGS`；当前没有 blocking issue，但 teacher diagnostics 全零需要 `T16` 判断
14. `T16` 已完成，结论为 `Conditional`
15. 当前更适合优先转向 `T17 / T18` 这类独立 manifest / boundary 任务，而不是继续扩大 P4 benchmark
16. `T17` 已完成，训练链环境说明现已独立收口，但训练链可移植性仍未锁定
17. `T18` 已完成，`.tflite` export/runtime 与 stub 边界现已独立收口，但真实 runtime 依赖仍未满足
18. `T18` review 已通过，真实 `.tflite` runtime 不可用继续保留为 R12
19. `T19` 已通过 review 并完成，tracked cache cleanup manifest 已就位，但物理 cleanup 仍未执行
20. `T20` 已完成并通过 adversarial review，real-board readiness checklist 已就位，但仍不是真板验证
21. `T21` 的 milestone gate 输出为 `Conditional`，Captain 已接受该结论并按 `PASS_WITH_WARNINGS` 收口
22. `T22` 已完成 real-board smoke execution plan，并由 Captain 接受为 `PASS_WITH_WARNINGS`
23. `T23` 已完成 protocol lock，并由 Captain 接受为 `PASS_WITH_WARNINGS`
24. 当前阶段仍为 `Phase 2: Controlled Development` / `Go`；T23 不改变 formal benchmark、`.tflite` runtime、cleanup 或 real-board validation 的证据等级
25. `docs/reference/进一步的深度研究结果.md` 已读；`T23` 已将强 classical / soft-information / calibration / learned baseline、更多 drift scenario、seed/CI/latency/commit/fallback 指标与 statcalib baseline 进入条件分类为 adopted / deferred / rejected
26. `T24` Worker 已完成四场景、五模式、`repeats=2` formal software revalidation：`missing_runs = []`，所有 `coverage = 1.0`，40 repeat-runs
27. T24 结果：
   - 四场景 winner 均为 `hybrid_residual_b`，runner-up 均为 `ukf`
   - 请求统计字段全部存在于 `comparison.csv`
   - `correction_saturation_rate_mean` 全为 0.0，teacher diagnostics 全零（与 T15 一致）
   - Mock-backed software HIL only
28. `T24` adversarial review 已完成，Captain 接受为 `PASS_WITH_WARNINGS`；T24 可标记完成，但 evidence boundary 仍限定为 `mock-backed` software HIL formal software revalidation
29. `correction_saturation_rate_mean` 全零与 teacher diagnostics header-only 是 deferred 机制证据缺口，不阻塞 T24 LER ranking，但必须进入 T25/T27 后续收口
30. `T25` gate review 已完成，Captain verdict = `PASS_WITH_WARNINGS`；T25 不执行新 benchmark，只判断 T24 证据等级、边界和下一任务优先级
31. `T27` 已完成，Captain verdict = `PASS_WITH_WARNINGS`；R10/R20 已缩窄但未全部关闭
32. `T28` 已完成，Captain verdict = `PASS_WITH_WARNINGS`；R21 对当前 writer 语义可关闭，但 R10 不关闭
33. `T29` 已通过 `PASS` 收口；P4 markdown report 重复表头已修复
34. `T26` 已通过 `PASS` 收口；gate verdict = `CONDITIONAL_GO`，statcalib 只能作为 separate comparator lane 后续推进
35. `T30` 已通过 `PASS` 收口；其结果是 interface-only statcalib contract 与 focused tests，不是 slow-loop integration、formal benchmark 或部署边界证据
36. `T36` 已完成，Captain verdict = `PASS`；它只读分析既有 `seed=20260429` 结果，未重跑 benchmark、未扩新分支、未改模型或部署边界
37. 在当时时点，当前唯一任务为 `T38`；T38 只允许做 `seed=20260429` single-seed trace-export probe，保持 Full vs Gated v5 语义与四场景边界不变

## 5. 已完成任务包

- `T1`：`docs/tasks/P0/T1_environment_and_min_entry.md`
- `T2`：`docs/tasks/P0/T2_smoke_reuse_and_bootstrap.md`
- `T3`：`docs/tasks/P0/T3_hil_p4_boundary_audit.md`
- `T4`：`docs/tasks/P0/T4_software_hil_bootstrap_and_smoke.md`
- `T5`：`docs/tasks/P0/T5_repo_noise_governance.md`
- `T6`：`docs/tasks/P0/T6_software_hil_reverification.md`
- `T7`：`docs/tasks/P0/T7_p4_benchmark_reverification.md`
- `T8`：`docs/tasks/P0/T8_gate_review_and_phase_decision.md`
- `T9`：`docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`
- `T10`：`docs/tasks/P0/T10_gate_review_after_t9.md`
- `T11`：`docs/tasks/P0/T11_recovery_dependency_manifest.md`
- `T12`：`docs/tasks/P0/T12_software_hil_determinism_recovery.md`
- `T13`：`docs/tasks/P0/T13_recovery_exit_and_closeout.md`
- `T14`：`docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`
- `T15`：`docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md`
- `T16`：`docs/tasks/Phase2/T16_p4_evidence_gate_review.md`
- `T17`：`docs/tasks/Phase2/T17_training_manifest_bootstrap.md`
- `T18`：`docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`
- `T19`：`docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`
- `T20`：`docs/tasks/Phase2/T20_real_board_readiness_checklist.md`
- `T21`：`docs/tasks/Phase2/T21_phase2_milestone_review.md`
- `T22`：`docs/tasks/Phase2/T22_real_board_smoke_execution_plan.md`
- `T23`：`docs/tasks/Phase2/T23_p4_formal_benchmark_protocol_lock.md`
- `T24`：`docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
- `T25`：`docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md`
- `T27`：`docs/tasks/Phase2/T27_teacher_diagnostics_path_audit.md`
- `T28`：`docs/tasks/Phase2/T28_teacher_diagnostics_semantics_repair.md`
- `T29`：`docs/tasks/Phase2/T29_p4_report_header_cleanup.md`
- `T26`：`docs/tasks/Phase2/T26_statcalib_feasibility_gate.md`
- `T30`：`docs/tasks/Phase2/T30_statcalib_interface_contract.md`
- `T36`：`docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md`
- `T38`：`docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md`

关键产出：

- `requirements-recovery.txt`
- `docs/recovery_bootstrap/P0_smoke_bootstrap.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/06_repo_noise_governance.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/review/T8_gate_review.md`
- `docs/review/T10_gate_review.md`
- `docs/review/T14_protocol_audit_review.md`
- `docs/review/T15_frozen_smoke_review.md`
- `docs/review/T16_p4_evidence_gate_review.md`
- `docs/review/T16_milestone_review.md`
- `docs/review/T17_review.md`
- `docs/review/T18_review.md`
- `docs/review/T19_review.md`
- `docs/review/T20_review.md`
- `docs/review/T21_phase2_milestone_review.md`
- `docs/review/T22_review.md`
- `docs/review/T23_review.md`
- `docs/review/T24_review.md`
- `docs/review/T25_p4_formal_evidence_gate_review.md`
- `docs/review/T27_teacher_diagnostics_path_audit.md`
- `docs/review/T28_review.md`
- `docs/review/T29_review.md`
- `docs/review/T30_review.md`
- `docs/review/T36_review.md`
- `docs/review/T38_review.md`
- `docs/review/Milestone2I_review.md`
- `docs/review/T31_review.md`
- `docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md`
- `docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md`
- `docs/protocols/benchmark/P4_benchmark_development_protocol.md`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md`
- `cnn_fpga/decoder/statcalib.py`
- `tests/test_statcalib_interface.py`
- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
- `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md`
- `docs/evidence_packs/training_reproducibility/training_chain_portable_dependency_lock_plan.md`
- `docs/evidence_packs/deployment_boundary/TFLite_runtime_bootstrap.md`
- `docs/evidence_packs/repo_hygiene/cleanup_tracked_cache_manifest.md`
- `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`

## 6. 当前唯一任务包摘要

当前唯一任务已切换为 `T54: Phase A multi-seed trace-only generalization probe`，任务包为 `docs/tasks/Phase2/T54_multi_seed_trace_only_generalization_probe.md`。

T54 任务边界：

- 允许做 bounded trace-only execution，但只限 `Full` vs `Gated v5`、冻结四场景、总 seed 数不超过 6
- 必须优先复用现有 `20260429` T38 trace 输出，并先 preflight `20260427/20260428` 是否可直接复用
- 不得运行 intervention variant、不得扩 baseline、不得扩 scenario、不得改 benchmark semantics
- 不得修改 source、benchmark code/config、`.tflite`、hardware 或 cleanup 相关文件
- 所有新增执行产物必须收敛到一个 T54-scoped run root
- 不得把 trace-only evidence 升级成 causal proof 或 paper-grade benchmark evidence

T46 历史收口事实：

- `docs/review/T46_review.md` verdict = `PASS`，blocking issues = none
- T46 只冻结 multi-seed / intervention evidence plan，不产生任何 execution evidence upgrade
- T46 保持单 seed diagnosis、multi-seed confirmation 与 causal evidence 三者分离
- T46 明确给出下一步应先做 Phase A trace-only probe，而不是直接冻结 paper ablation/material ledger

## 7. 下一步建议

当前建议交给 Worker 执行 `T54` Phase A multi-seed trace-only generalization probe。

建议优先级：

1. 先复用 `20260429` 的 T38 trace 输出，并对 `20260427/20260428` 做 field-complete preflight。
2. 对 `20260425`、`20260430`、`20260510` 做 bounded trace-only rerun，判断模式是否跨 seed 复现。
3. 先产出 cross-seed diagnostic summary，再决定是否值得开 intervention lane。
4. 不直接进入 `T47`，也不运行任何 intervention、`.tflite`、硬件或 cleanup。

## 8. 暂不继续的事项

在 T54 完成前，暂不继续：

1. 任何 intervention variant execution
2. 任何 benchmark scope 扩展或 paper-grade benchmark 叙事升级
3. 直接推进旧版 `T47` paper material ledger
4. 新的训练、`.tflite`、硬件或 cleanup
5. 任何对 `docs/02_experiment_plan.md` 的修改

## 8A. 2026-05-23 Captain Supersession For T55

This section supersedes the older T54-current-task wording in Sections 6-8 above.

### Current unique task

- Current unique task: `T55: Phase B multi-seed I1 residual-clip intervention probe`
- Task package: `docs/tasks/Phase2/T55_multi_seed_i1_residual_clip_intervention_probe.md`
- Task type: bounded intervention execution on the same mock-backed P4 wrapper over software HIL path

### T54 closeout facts

- `docs/review/T54_review.md` verdict = `PASS`; blocking issues = none
- T54 warning classification: all non-blocking items are `accepted`
- T54 conclusion: the committed-`b` instability pattern is broadly repeated with qualifications across the locked 6-seed pack
- The observed multi-seed picture now includes quiet / classic / universal categories
- `C4` remains `partial`; T54 does not provide causal proof and does not justify skipping intervention evidence

### T55 execution boundary

- Reuse the same 6 seeds: `20260425`, `20260427`, `20260428`, `20260429`, `20260430`, `20260510`
- Reuse the same frozen four scenarios and repeat count `2`
- Reuse existing model assets; do not retrain
- Run exactly one config-only intervention: lower Gated v5 `residual_clip_b` from `0.12` to `0.06`
- Keep generated config(s), helper script(s), outputs, and summaries inside one T55-scoped run root
- Do not edit source code, source-tree config, `.tflite`, hardware, cleanup, or paper-material scope

### Next recommended worker action

1. Build a seed/model reuse manifest for the locked 6-seed pack.
2. Generate task-scoped benchmark config(s) inside the T55 run root only.
3. Execute only the pure I1 clip-lowered intervention variant.
4. Compare intervention outputs against the reused T54 baseline references.
5. Produce a bounded intervention report that classifies the intervention as helpful, harmful, mixed, or no-clear-effect.

### Do not continue yet

1. Do not jump directly to `T47`.
2. Do not run any second intervention variant.
3. Do not expand benchmark scope, comparator scope, `.tflite`, hardware, or cleanup scope.
4. Do not edit `docs/02_experiment_plan.md`.

## 8B. 2026-05-24 Captain Supersession For T56

This section supersedes the older T55-current-task wording in Sections 1 and 8A above.

### Current unique task

- Current unique task: `T56: Post-I1 mechanism claim reframing gate`
- Task package: `docs/tasks/Phase2/T56_post_i1_mechanism_claim_reframing_gate.md`
- Task type: docs-only post-I1 evidence-interpretation and next-lane gate task

### T55 closeout facts

- `docs/review/T55_review.md` verdict = `PASS`; blocking issues = none
- T55 warning classification: all non-blocking items are `accepted`
- T55 intervention result: mixed and mostly harmful across the locked 6-seed pack
- The simple mechanism framing “high committed-b is harmful” is not supported as a general explanation
- `C4` remains `partial`

### T56 execution boundary

- Read and synthesize `T36`, `T38`, `T46`, `T54`, and `T55`
- Update claim wording only in bounded docs
- Do not run new benchmark, trace export, second intervention, `.tflite`, hardware, cleanup, or comparator execution
- Do not edit governance docs or `docs/02_experiment_plan.md`

### Next recommended worker action

1. Produce a retain / weaken / retire / reframe / still-open mechanism-claim table.
2. Update `docs/paper_materials/paper_claim_evidence_ledger.md` only where `T55` changes claim status or wording boundary.
3. State whether `T47` can proceed only under conditioned hedge wording.
4. State whether any second intervention lane is `no-go`, `deferred`, or `conditionally justified`.

### Do not continue yet

1. Do not jump directly to `T47` as unconditional next work.
2. Do not run a second intervention variant.
3. Do not reopen benchmark expansion, `.tflite`, hardware, or cleanup scope.

## 8C. 2026-05-24 Captain Supersession For T47

This section supersedes the older T56-current-task wording in Sections 1 and 8B above.

### Current unique task

- Current unique task: `T47: Paper ablation result-pack and material ledger`
- Task package: `docs/tasks/Phase2/T47_paper_ablation_result_pack_and_material_ledger.md`
- Task type: docs-only hedge-conditioned paper-material lane

### T56 closeout facts

- `docs/review/T56_review.md` verdict = `PASS`; blocking issues = none
- T56 warning classification: all non-blocking items are `accepted`
- T56 claim table has already reframed the mechanism story into retain / weaken / retire / reframe / still-open
- The simple “high committed-b is harmful” explanation remains unsupported as a general mechanism claim
- `C4` remains `partial`

### T47 execution boundary

- Read and synthesize `T56` plus the existing paper-assembly / recovery baseline docs
- Stay docs-only and preserve the T56 hedge wording
- Do not run new benchmark, trace export, intervention, `.tflite`, hardware, cleanup, or comparator execution
- Do not edit governance docs or `docs/02_experiment_plan.md`

### Next recommended worker action

1. Produce a ready / partial / missing ledger for figures, tables, and ablation/material items.
2. Keep every mechanism-facing statement bounded by the T56 claim table.
3. State clearly whether the paper can proceed with the current evidence pack or what is still missing.
4. Preserve the distinction between frozen paper materials and solved mechanism.

### Do not continue yet

1. Do not convert this lane into a new intervention or benchmark task.
2. Do not reopen `.tflite`, real-board, or cleanup scope.
3. Do not upgrade the mechanism claim beyond the T56 boundary.

## 9. 2026-05-16 Captain Supersession

This section supersedes older T38-next wording in sections above.

45. T38 reviewer warnings 如何处理？
   - 当前答案：
     - Verdict：`PASS`；blocking issues: none。
     - N1 unused imports：`accepted` as cosmetic。
     - N2 `missing_runs = 0` wording versus JSON `missing_runs: []`：`accepted`，未来文档应写精确数据格式。
     - N3 report 未显式解释 `sqrt(2) * 0.12` 常数：`accepted`，只影响说明清晰度。
     - N4 initial timeout and same-dir resume：`accepted`，最终 evidence 完整。
46. Milestone 2I 是否允许进入下一里程碑？
   - 当前答案：
     - 允许，但结论是 `Conditional Allow`。
     - 已完成 bounded mechanism-evidence hardening；仍未关闭 clean-environment reproducibility、mitigation、多 seed confirmation、真实 `.tflite` runtime 和真板 validation。
47. T31 是否可以交给 Worker？
   - 当前答案：
     - 可以。在当时时点，当前唯一任务是 `T31: Training-chain portable dependency lock plan`。
     - Allowed files: `docs/tasks/Phase2/T31_training_chain_portable_dependency_lock_plan.md`, `docs/evidence_packs/training_reproducibility/training_chain_portable_dependency_lock_plan.md`, `docs/review/T31_review.md`, `docs/for_human/T31_explanation.md`。
     - Worker 只允许做 read-only interpreter/package inventory 和 dependency-lock planning；不得安装依赖、运行训练、运行 benchmark、创建 `runs/` 或 `artifacts/`、修改 source/config/protocol/baseline/seed policy。

## 10. 2026-05-17 Captain Supersession

This section supersedes older T38/T31-next wording in sections above.

48. T31 reviewer warnings 如何处理？
   - 当前答案：
     - Verdict：`PASS`；blocking issues: none。
     - N1 markdown subsection numbering：`accepted` as cosmetic。
     - N2 `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md` later alignment：`accepted` as future alignment。
     - N3 worker self-review overwritten by adversarial review：`accepted`。
     - 没有 `deferred` warning，因此未新增 risk。
49. T31 是否允许进入下一 reproducibility step？
   - 当前答案：
     - 允许。
     - T31 已完成 plan-level dependency boundary，但未实际创建 clean environment 或 lockfile。
     - R11 仍 open but narrowed。
50. T39 review 结果如何处理？
   - 当前答案：
     - Verdict：`PASS`；blocking issues: none。
     - N1 exact version pins：`accepted`。
     - N2 `pip list` vs `pip freeze`：`accepted`。
     - N3 sandbox/escalation note：`accepted`。
     - 没有 `deferred` warning，因此未新增 risk。
51. T40 review 结果如何处理？
   - 当前答案：
     - Verdict：`PASS`；blocking issues: none。
     - N1 worker pre-review overlap：`accepted`。
     - N2 legacy macOS dataset-manifest paths：`accepted`。
     - N3 R11 narrowing governance sync：`deferred`，已由 Captain 写回 risks/governance。
52. T33 是否可以交给 Worker？
   - 当前答案：
     - 不再提交。`T33` 已完成并通过 Captain `PASS` 收口。
53. T33 review 结果如何处理？
   - 当前答案：
     - Verdict：`PASS`；blocking issues: none。
     - N1 Windows `index.lock` permission friction：`accepted`。
     - 没有 `deferred` warning，因此未因 T33 新增 risk。
     - `R4` 已缩窄，`R7` 对 tracked-cache lane 已收口。
54. 当前下一唯一任务是什么？
   - 当前答案：
     - `T42: Paper Background / Related Work scaffold and method-positioning calibration`。
     - 任务包为 `docs/tasks/Phase2/T42_paper_background_related_work_and_positioning.md`。
     - Worker 只允许做 docs-only 结构扩展与定位校准；不得运行新实验、不得升级 evidence level、不得改写阶段结论或 repo facts。
