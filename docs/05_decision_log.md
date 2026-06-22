# Decision Log

## D-2026-06-15-02

- 日期：`2026-06-15`
- 决策：接受 `T89` 为 `PASS`，并将当前唯一任务切换为 `T90: 训练链 clean-CPU 同机 repeated-run 一致性证据包`

### 背景

`T89` 的目标不是继续改主线 note，也不是把仓库写成投稿完成态，而是在 `T88` 已完成 mainline surface freeze 之后，把 frozen mainline 的唯一 handoff 入口、source-of-truth、post-freeze change-control 与 blocked-surface re-entry 条件整理成一套可交接、可维护、不可误写的答案。review 已确认这轮 docs-only 交付在 allowlist 范围内真实完成，同时也明确提醒：`T89` 只是 frozen-mainline handoff consolidation，不是 submission-ready completion，也不是任何 blocked surface 的解锁。

### 依据

1. `docs/review/T89_review.md` verdict = `PASS`，blocking issues = none。
2. review 明确确认本轮真实新增了 `paper_frozen_mainline_handoff_packet.md`、`paper_frozen_mainline_source_of_truth_map.md`、`paper_postfreeze_change_control.md` 与 `paper_blocked_surface_reentry_conditions.md` 四份主线 handoff / change-control 文档，并完成了两个 README 的正式登记。
3. review 明确确认 `T89` 保持了既有 guardrail：`T24` 仍是 frozen mock-backed software-HIL anchor，`FR6/FR7` 仍是 descriptive support，`FR8/statcalib` 仍是 extension lane / no-promotion，training/material 仍是 bounded rerun + no full closure，`.tflite` 仍是 isolated current-host runtime，real-board 仍是 read-only gate / regeneration / provenance with current-host `NO_GO`。
4. review 的 non-blocking notes 仅涉及 allowlist-scoped diff 纪律、dirty-worktree 噪声隔离以及“当前仍不是 submission-ready / blocked-surface unlocked”的 retelling 提醒；它们都不构成新的 risk opening。
5. 在 `T89` 完成后，`T51/T52` 这类更大范围 paper reopen 仍然过早，`T37` 仍受当前暂无 `Linux + FPGA` 硬件宿主的现实条件阻塞；当前最可执行的高价值非硬件 evidence-hardening lane 是 `R11` 的 same-host repeated-run consistency follow-up。

### 结论

1. 接受 `T89` 为 `PASS` 并标记完成。
2. 本轮不进入 `PASS_WITH_WARNINGS` 分类流；review 中的 non-blocking notes 统一按 operational reminder 接受处理。
3. 本轮不新增 `deferred/rejected` warning，也不新增风险项。
4. 当前唯一任务切换为 `T90`，只允许做 mainline-only、clean-CPU-only、same-host repeated-run consistency 的训练复现强化，不得改写 mainline note、理论分支或部署/板级边界。
5. `T90` 的 strongest intended truth 不是 full training reproducibility，而只是把 `T50` 的“一次 bounded rerun”推进成“一组可审计的同机 repeated-run consistency pack”。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T89 -> PASS` 与 `Current Unique Task -> T90`。
2. 新增 `docs/tasks/Phase2/T90_training_same_host_repeated_run_consistency_pack.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T89` 产物、Captain closeout 文档、治理同步与 `T90` 任务包；不要混入 theory 分支相关改动、无关编译噪声或未审阅 diff，更不要使用 blanket `git add -A`。

## D-2026-06-15-01

- 日期：`2026-06-15`
- 决策：接受 `T88` 为 `PASS`，并将当前唯一任务切换为 `T89: 主线 frozen-mainline handoff 包与 post-freeze change-control 收口`

### 背景

`T88` 的目标不是继续扩大主线 note，也不是宣布投稿完成，而是在 `T87` 已完成 author-final QA / pre-submission gate 之后，真实执行已经批准的 `MF01-MF05` bounded manual-finish 动作，并把 mainline surface 选择、编辑决策、blocked disclaimer 与唯一 handoff gate 固定下来。review 已确认这轮 docs-only 交付在允许范围内真实完成，同时也明确提醒：`GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY` 只能理解为“允许进入 frozen-mainline handoff / post-freeze change-control 任务”，不能回述成 submission-ready completed。

### 依据

1. `docs/review/T88_review.md` verdict = `PASS`，blocking issues = none。
2. review 明确确认本轮真实新增了 `paper_manual_finish_execution_log.md`、`paper_mainline_surface_freeze_manifest.md`、`paper_author_edit_decision_register.md`、`paper_blocked_surface_disclaimer_table.md` 与 `paper_frozen_mainline_handoff_gate.md` 五份主线收口台账，并完成了最小必要的 `% T88-MANUAL` note refresh。
3. review 明确确认 `T88` 保持了既有 guardrail：`T24` 仍是 frozen mock-backed software-HIL anchor，`FR6/FR7` 仍是 descriptive support，`FR8/statcalib` 仍是 extension lane / no-promotion，training/material 仍是 canonical chain intact + one clean CPU-only bounded rerun，`.tflite` 仍是 isolated current-host runtime，real-board 仍是 read-only gate / regeneration / provenance with current-host `NO_GO`。
4. review 的 non-blocking notes 仅涉及 dirty-worktree 下的 allowlist-scoped diff 纪律、`GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY` 的窄语义、compile 结论仅限当前宿主，以及 Windows Git/CRLF 宿主噪声；它们都不构成新的 risk opening。

### 结论

1. 接受 `T88` 为 `PASS` 并标记完成。
2. 本轮不进入 `PASS_WITH_WARNINGS` 分类流；review 中的 non-blocking notes 统一按 operational reminder 接受处理。
3. 本轮不新增 `deferred/rejected` warning，也不新增风险项。
4. 当前唯一任务切换为 `T89`，只允许做 mainline-only、docs-only、freeze-preserving 的 frozen-mainline handoff、source-of-truth mapping、post-freeze change-control 与 blocked-surface re-entry 条件固化。
5. `T89` 必须继续保留 `T24` frozen anchor、`FR6/FR7` descriptive-only、`FR8` no-promotion、training/material bounded rerun、`.tflite` isolated current-host only、real-board current-host `NO_GO`、hardware-dependent blocked surface，以及“不得混入独立 theory 分支内容”的 guardrail。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T88 -> PASS` 与 `Current Unique Task -> T89`。
2. 新增 `docs/tasks/Phase2/T89_mainline_frozen_handoff_packet_and_postfreeze_change_control.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T88` 产物、Captain closeout 文档与 `T89` 任务包；不要混入理论分支相关改动、无关编译噪声或未审阅 diff，更不要使用 blanket `git add -A`。

## D-2026-06-14-05

- 日期：`2026-06-14`
- 决策：接受 `T87` 为 `PASS`，并将当前唯一任务切换为 `T88: 主线 bounded manual finish 执行与 surface freeze 收口包`

### 背景

`T87` 的目标不是宣布投稿完成，而是在 `T86` 已完成 submission-facing assembly 之后，把当前主线 note/material 再压一层作者终检纪律：固定红旗表述、manual-finish 队列与唯一 pre-submission regression gate。review 已确认这轮 docs-only 交付在允许范围内真实完成，同时也明确提醒：`GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY` 只能理解为“允许开启下一张 manual-finish execution / surface-freeze 任务”，不能回述成 submission-ready completed。

### 依据

1. `docs/review/T87_review.md` verdict = `PASS`，blocking issues = none。
2. review 明确确认本轮真实新增了 `paper_author_final_qa_checklist.md`、`paper_presubmission_regression_gate.md`、`paper_submission_wording_redflag_register.md` 与 `paper_manual_finish_queue.md` 四份作者终检台账，并完成了主线 note 中 `Numerical Results`、`Discussion`、`Conclusion` 的最小 `% T87-QA` 刷新。
3. review 明确确认 `T87` 保持了既有 guardrail：`T24` 仍是 frozen mock-backed software-HIL anchor，`FR6/FR7` 仍是 descriptive support，`FR8/statcalib` 仍是 extension lane / no-promotion，training/material 仍是 canonical chain intact + one clean CPU-only bounded rerun，`.tflite` 仍是 isolated current-host runtime，real-board 仍是 read-only gate / regeneration / provenance with current-host `NO_GO`。
4. review 的 non-blocking notes 仅涉及 CRLF/git-ignore 宿主噪声与 `GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY` 的窄语义；它们都不构成新的 risk opening。

### 结论

1. 接受 `T87` 为 `PASS` 并标记完成。
2. 本轮不进入 `PASS_WITH_WARNINGS` 分类流；review 中的 non-blocking notes 统一按 operational reminder 接受处理。
3. 本轮不新增 `deferred/rejected` warning，也不新增风险项。
4. 当前唯一任务切换为 `T88`，只允许做 mainline-only、docs-only、bounded manual-finish execution、surface freeze 与 blocked-disclaimer 固化。
5. `T88` 必须继续保留 `T24` frozen anchor、`FR6/FR7` descriptive-only、`FR8` no-promotion、training/material bounded rerun、`.tflite` isolated current-host only、real-board current-host `NO_GO`、hardware-dependent blocked surface，以及“不得混入独立 theory 分支内容”的 guardrail。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T87 -> PASS` 与 `Current Unique Task -> T88`。
2. 新增 `docs/tasks/Phase2/T88_mainline_bounded_manual_finish_and_surface_freeze.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T87` 产物、Captain closeout 文档与 `T88` 任务包；不要混入理论分支相关改动、无关编译噪声或未审阅 diff，更不要使用 blanket `git add -A`。

## D-2026-06-14-04

- 日期：`2026-06-14`
- 决策：接受 `T86` 为 `PASS`，并将当前唯一任务切换为 `T87: 主线作者终检与 pre-submission QA 收口包`

### 背景

`T86` 的目标不是宣布 submission-ready pack 已完成，而是在 `T85` 已完成 submission-readiness preflight 之后，把当前主线 note / paper-material / blocker / exclusion 信息装配成一套 submission-facing 但仍显式有边界的 package。review 已确认这轮 docs-only 交付在允许范围内真实完成，同时也明确提醒：`T86` 的 assembly 台账只能证明“当前主线作者已经拿到一份可审计的装配/排除答案”，不能回述成 blocked surface 已解除、投稿闭环已完成或 theory 分支已并回主线。

### 依据

1. `docs/review/T86_review.md` verdict = `PASS`，blocking issues = none。
2. review 明确确认本轮真实新增了 `paper_submission_pack_assembly_manifest.md`、`paper_submission_surface_route_map.md`、`paper_submission_exclusion_register.md` 与 `paper_submission_author_handoff.md` 四份 submission-facing 台账，并完成了 mainline note 的最小 `% T86-ASSEMBLY` 装配刷新。
3. review 明确确认 `T86` 保持了既有 guardrail：`T24` 仍是 frozen mock-backed software-HIL anchor，`FR6/FR7` 仍是 descriptive support，`FR8/statcalib` 仍是 extension lane / no-promotion，training/material 仍是 canonical chain intact + one clean CPU-only bounded rerun，`.tflite` 仍是 isolated current-host runtime，real-board 仍是 read-only gate / regeneration / provenance with current-host `NO_GO`。
4. review 的 non-blocking notes 仅涉及 dirty-worktree 下的 allowlist-scoped diff 纪律、assembly 台账不等于 submission-ready 完成态、compile 结论仅限当前宿主，以及 Windows Git 噪声；它们都不构成新的 risk opening。

### 结论

1. 接受 `T86` 为 `PASS` 并标记完成。
2. 本轮不进入 `PASS_WITH_WARNINGS` 分类流；review 中的 non-blocking notes 统一按 operational reminder 接受处理。
3. 本轮不新增 `deferred/rejected` warning，也不新增风险项。
4. 当前唯一任务切换为 `T87`，只允许做 mainline-only、docs-only、QA-only 的作者终检 / 投稿前回归 gate。
5. `T87` 必须继续保留 `T24` frozen anchor、`FR6/FR7` descriptive-only、`FR8` no-promotion、training/material bounded rerun、`.tflite` isolated current-host only、real-board current-host `NO_GO`、hardware-dependent blocked surface，以及“不得混入独立 theory 分支内容”的 guardrail。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T86 -> PASS` 与 `Current Unique Task -> T87`。
2. 新增 `docs/tasks/Phase2/T87_mainline_author_final_qa_and_presubmission_gate.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T86` 产物、Captain closeout 文档与 `T87` 任务包；不要混入理论分支相关改动、无关编译噪声或未审阅 diff，更不要使用 blanket `git add -A`。

## D-2026-06-14-03

- 日期：`2026-06-14`
- 决策：接受 `T85` 为 `PASS`，关闭 `R36`，并将当前唯一任务切换为 `T86: 主线 bounded submission-pack assembly 与显式 exclusion route 收口`

### 背景

`T85` 的目标不是直接宣布 submission-ready pack，而是在 `T84` 已完成 bounded reader-facing final polish / assembly 之后，收掉唯一残余状态滞后句，并给出一份 honest 的 submission-readiness preflight / blocker matrix。review 已确认这轮 docs-only 交付在允许范围内真实完成，同时也明确提醒：`GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY` 只能理解为“允许开启下一张 docs-only、mainline-only、assembly-only 的 bounded task”，不能回述成 submission-ready pack 已完成。

### 依据

1. `docs/review/T85_review.md` verdict = `PASS`，blocking issues = none。
2. review 明确确认本轮真实新增了 `paper_submission_readiness_preflight_gate.md`、`paper_submission_blocker_matrix.md` 与 `paper_residual_state_lag_sweep.md` 三份台账，并完成了 note 中 residual wording-lag 清扫。
3. review 明确确认 `T85` 保持了既有 guardrail：`T24` 仍是 frozen mock-backed software-HIL anchor，`FR6/FR7` 仍是 descriptive support，`FR8/statcalib` 仍是 extension lane / no-promotion，`.tflite` 仍是 isolated current-host runtime，real-board 仍是 read-only gate / provenance。
4. review 的 non-blocking notes 仅涉及 dirty-worktree 下的 allowlist-scoped diff 纪律、`GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY` 的窄语义，以及 compile 结论仅限当前宿主；它们都不构成新的 risk opening。

### 结论

1. 接受 `T85` 为 `PASS` 并标记完成。
2. 本轮不进入 `PASS_WITH_WARNINGS` 分类流；review 中的 non-blocking notes 统一按 operational reminder 接受处理。
3. `R36` 已由 `T85` 收口关闭。
4. 当前唯一任务切换为 `T86`，只允许做 mainline-only、docs-only、assembly-only 的 submission-pack 装配与显式 exclusion route 收口。
5. `T86` 必须继续保留 `T24` frozen anchor、`FR6/FR7` descriptive-only、`FR8` no-promotion、training/material bounded rerun、`.tflite` isolated current-host only、real-board current-host `NO_GO`、hardware-dependent blocked surface，以及“不得混入独立 theory 分支内容”的 guardrail。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T85 -> PASS`、`R36` 关闭与 `Current Unique Task -> T86`。
2. 新增 `docs/tasks/Phase2/T86_mainline_bounded_submission_pack_assembly.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T84/T85` 产物、Captain closeout 文档与 `T86` 任务包，不要把同工作区内的无关或未审阅 diff 混入，更不要使用 blanket `git add -A`。

## D-2026-06-14-02

- 日期：`2026-06-14`
- 决策：接受 `T84` 为 `PASS_WITH_WARNINGS`，打开 `R36`，并将当前唯一任务切换为 `T85: 主线 submission-readiness preflight gate 与残余状态滞后清扫`

### 背景

`T84` 的目标不是恢复 full-manuscript reopen，也不是直接宣布 submission-ready pack，而是在 `T83` 已完成全文一致性 closeout gate 之后，对当前主线 note 做一轮有界的 reader-facing final polish、术语翻译与 appendix/supplement 装配。review 已确认这轮 docs-only 交付在允许范围内真实完成，但也明确指出 `Conclusion` 仍残留一处状态滞后句，会把本轮已经完成的 reader-facing polish 写成“后续仍待执行”的工作，因此 closeout 结论应为 `PASS_WITH_WARNINGS` 而不是 `PASS`。

### 依据

1. `docs/review/T84_review.md` verdict = `PASS_WITH_WARNINGS`，blocking issues = none。
2. review 明确确认本轮 diff 保持在 `T84` 允许的 note/material/README/summary 与编译产物范围内，且 `% T80-REOPEN`、`% T81-CALIBRATION`、`% T82-SUPPORT`、`% T83-CLOSEOUT`、`% T84-POLISH` 标记保留。
3. review 明确确认 `paper_bounded_final_polish_change_map.md`、`paper_reader_facing_term_translation_table.md` 与 `paper_appendix_supplement_reader_assembly_map.md` 已真实建立本轮 reader-facing final polish / assembly 台账。
4. review 明确指出 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 的 `Conclusion` 仍保留句子 `The remaining writing work is to translate these internal layers into a final reader-facing polish pass...`，与 `T84` 已完成的 strongest supported truth 存在轻微状态滞后。
5. review 同时提醒：当前 worktree 仍含 pre-existing docs dirty state，且 compile 结论只证明当前宿主上的 `TeX Live 2024 + latexmk` 路径可刷新 note；因此提交与后续 retelling 仍必须坚持 allowlist-scoped / precise-staging 纪律，不得把 compile 结果写成通用环境闭环。

### 结论

1. 接受 `T84` 为 `PASS_WITH_WARNINGS` 并标记完成。
2. warning 分类如下：
   - `N1` `Conclusion` 残留状态滞后句 = `deferred -> R36`
   - `N2` allowlist-scoped diff / precise staging discipline = `accepted`
   - `N3` compile 结论仅限当前宿主 `TeX Live 2024 + latexmk` = `accepted`
3. `T84` 只关闭了“当前 note 是否已经完成有界 reader-facing final polish / assembly”这一层问题，不关闭任何 submission-ready、deployment、`.tflite`、real-board 或 `statcalib` promotion 风险。
4. 当前唯一任务切换为 `T85`，只允许做 submission-readiness preflight gate、blocker matrix、残余 wording-lag sweep、README 登记与 compile-aware refresh。
5. `T85` 必须继续保留 `T24` frozen anchor、`FR6/FR7` descriptive-only、`FR8` no-promotion、training/material bounded rerun、`.tflite` isolated current-host only、real-board current-host `NO_GO` 与 hardware-dependent blocked surface 等 guardrail。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T84 -> PASS_WITH_WARNINGS`、`R36` 与 `Current Unique Task -> T85`。
2. 新增 `docs/tasks/Phase2/T85_mainline_submission_readiness_preflight_gate.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T84` 产物、Captain closeout 文档与 `T85` 任务包，不要把同工作区内的无关或未审阅 diff 混入，更不要使用 blanket `git add -A`。

## D-2026-06-14-01

- 日期：`2026-06-14`
- 决策：接受 `T83` 为 `PASS`，标记 `T83` 完成，并将当前唯一任务切换为 `T84: 主线 note 有界 final polish 与读者化装配包`

### 背景

`T83` 的目标不是恢复 full-manuscript reopen，而是在 `T80/T81/T82` 已完成 ready sections、contribution/methods 与 supporting-boundary 收口之后，判断当前整份 note 是否已经全文自洽，并把后续是否还能继续推进压成一份唯一 closeout gate。review 已确认这项工作已在允许范围内真实完成，同时也明确指出：`T83` 之后的剩余瓶颈已经不是“哪些 section 还互相打架”，而是“如何把当前一致性版本翻译成更读者友好的语言并完成有界装配，而不升级任何 blocked surface”。

### 依据

1. `docs/review/T83_review.md` verdict = `PASS`，blocking issues = none。
2. review 明确确认本轮 diff 保持在 `T83` 允许的 note/material/README/summary 与编译产物范围内，且 `% T80-REOPEN`、`% T81-CALIBRATION`、`% T82-SUPPORT` 标记全部保留。
3. review 明确确认 `paper_fullnote_consistency_crosswalk.md` 与 `paper_closeout_gate_and_blocker_register.md` 已真实建立全文一致性回链与唯一 gate 结论。
4. review 明确给出的唯一 closeout gate 是 `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`，并且强调这不等于 submission-ready pack、deployment closure、real-board success 或 blocked surface 解锁。
5. review 同时提醒：当前 worktree 已有 pre-existing dirty state，因此提交与后续验证必须继续采用 allowlist-scoped / precise-staging 纪律，而不能把全仓 `git diff --name-only` 直接当成 `T83` 变更清单。

### 结论

1. 接受 `T83` 为 `PASS` 并标记完成。
2. 由于 verdict 为 `PASS`，本轮不进入 warning 分类流程，也不新开 risk。
3. `T83` 只关闭了“全文是否已经自洽”的 gate，不关闭任何 deployment/runtime/board 风险。
4. 当前唯一任务切换为 `T84`，只允许做读者化术语翻译、结构压缩、appendix/supplement 装配、README 登记与 compile-aware refresh。
5. `T84` 必须继续保留 `T24` frozen anchor、`FR6/FR7` descriptive-only、`FR8` no-promotion、training/material bounded rerun、`.tflite` isolated current-host only、real-board current-host `NO_GO` 与 hardware-dependent blocked surface 等 guardrail。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T83 -> PASS` 与 `Current Unique Task -> T84`。
2. 新增 `docs/tasks/Phase2/T84_mainline_bounded_final_polish_and_reader_facing_assembly.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T82/T83` 产物、Captain closeout 文档与 `T84` 任务包，不要把同工作区内的无关或未审阅 diff 混入。

## D-2026-06-13-01

- 日期：`2026-06-13`
- 决策：接受 `T82` 为 `PASS`，标记 `T82` 完成，并将当前唯一任务切换为 `T83: 主线 note 全文一致性收口与 manuscript closeout gate`

### 背景

`T82` 的目标不是恢复 full-manuscript reopen，而是在 `T80/T81` 已完成 ready sections 与 contribution/methods calibration 之后，把 `FR8/statcalib`、training/material、isolated true `.tflite`、real-board `NO_GO` 等 supporting-boundary 材料压成一条 manuscript-facing closeout route。review 已确认这条 route 已在允许范围内真实完成，同时也明确指出：`T82` 之后的剩余瓶颈已经不是“某一块 supporting 材料缺失”，而是“整份 note 是否已经 section-by-section 自洽，以及 full-manuscript closeout 仍被哪些 blocker 阻塞”。

### 依据

1. `docs/review/T82_review.md` verdict = `PASS`，blocking issues = none。
2. review 明确确认本轮 diff 保持在 `T82` 允许的 note/material/README/summary 范围内，且 note 中存在 4 条 `% T82-SUPPORT` 标记。
3. review 明确确认 `paper_supporting_material_closeout_pack.md` 与 `paper_manuscript_closeout_readiness_matrix.md` 已真实建立 `main text / appendix / supplement / blocked` 四层收口结构。
4. review 明确确认 `T80` 的 `% T80-REOPEN` 与 `T81` 的 `% T81-CALIBRATION` 标记仍全部保留，说明 `T82` 没有把 scope 静默扩回已关闭的主文或方法章任务。
5. review 还明确提醒：`T82` 完成后仍不能自动回述成 full-manuscript closeout；后续若继续推进，必须由 Captain 开出一张新的、全文级别的一致性 sweep / closeout gate 任务。

### 结论

1. 接受 `T82` 为 `PASS` 并标记完成。
2. 由于 verdict 为 `PASS`，本轮不进入 warning 分类流程，也不新开 risk。
3. `T82` 只关闭了 supporting-material route 分散的问题，不关闭任何 deployment/runtime/board 风险。
4. 当前唯一任务切换为 `T83`，只允许做全文 section-by-section consistency sweep、受控 wording 收口、section-to-evidence crosswalk 与唯一 closeout gate / blocker register。
5. `T83` 必须继续保留 `T24` frozen anchor、`FR6/FR7` descriptive-only、`FR8` no-promotion、training/material bounded rerun、`.tflite` isolated current-host only、real-board current-host `NO_GO` 与 hardware-dependent blocked surface 等 guardrail。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T82 -> PASS` 与 `Current Unique Task -> T83`。
2. 新增 `docs/tasks/Phase2/T83_mainline_note_full_consistency_sweep_and_closeout_gate.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T82` 产物、Captain closeout 文档与 `T83` 任务包，不要把无关历史 diff 混入。

## D-2026-06-12-09

- 日期：`2026-06-12`
- 决策：接受 `T81` 为 `PASS`，标记 `T81` 完成，并将当前唯一任务切换为 `T82: supporting-material 收口与 appendix/supplement 边界整合包`

### 背景

`T81` 的目标不是恢复 full-manuscript reopen，而是在 `T80` 已完成 ready sections prose reopen 之后，把 `Summary of Contributions` 与三章 methods 压回到当前 strongest supported truth 上。review 已确认这 4 个 target sections 完整收口，同时没有扩大到 benchmark、deployment、real-board 或 `statcalib` promotion 叙事。

### 依据

1. `docs/review/T81_review.md` verdict = `PASS`，blocking issues = none。
2. review 明确确认本轮 diff 保持在 `T81` 允许的 section 与文件范围内，且 note 中存在 4 条 `% T81-CALIBRATION` 标记。
3. review 明确确认 `T80` 的 8 条 `% T80-REOPEN` 标记仍保留，说明 `T81` 没有把 scope 静默扩回到已关闭的 ready sections。
4. review 明确确认 `paper_methods_and_contribution_calibration_manifest.md` 已记录 4 个 changed sections、evidence anchors、guardrails 与 compile 状态。
5. review 也明确提醒：`T81` 完成后，当前 note 仍不等于 full-manuscript reopen；后续若继续推进 paper 主线，必须另开新任务决定 supporting-material closeout 还是更大范围的 prose gate。

### 结论

1. 接受 `T81` 为 `PASS` 并标记完成。
2. 由于 verdict 为 `PASS`，本轮不进入 warning 分类流程，也不新开 risk。
3. `T81` 只关闭了 contribution/methods calibration 缺口，不关闭任何 deployment/runtime/board 风险。
4. 当前唯一任务切换为 `T82`，只允许在 supporting-boundary 层面整合 `FR8/statcalib`、training/material、isolated true `.tflite`、real-board `NO_GO` 等 manuscript-facing closeout 材料。
5. `T82` 必须继续保留 `T24` frozen anchor、`FR6/FR7` descriptive-only、`FR8` no-promotion、`.tflite` isolated current-host only、real-board current-host `NO_GO` 等 guardrail。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T81 -> PASS` 与 `Current Unique Task -> T82`。
2. 新增 `docs/tasks/Phase2/T82_supporting_material_closeout_and_boundary_integration_pack.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T81` 产物、Captain closeout 文档与 `T82` 任务包，不要把无关历史 diff 混入。

## D-2026-06-12-08

- 日期：`2026-06-12`
- 决策：接受 `T80` 为 `PASS`，标记 `T80` 完成，并将当前唯一任务切换为 `T81: Summary of Contributions 与 methods-only calibration pack`

### 背景

`T80` 的目标不是恢复 full-manuscript reopen，而是在 `T79` 已完成 reopen gate 之后，只对当前 note 中已经 ready 的 8 个 narrative / result-facing sections 做一轮受控 prose reopen。review 已确认这层工作真实完成，同时也明确指出 `Summary of Contributions` 与三章 methods 仍刻意保持 untouched，因此它们成为下一轮主线校准的唯一集中缺口。

### 依据

1. `docs/review/T80_review.md` verdict = `PASS`，blocking issues = none。
2. review 已确认本轮 diff 保持在 `T80` 允许的 section 和文件范围内，且 `% T80-REOPEN` 标记只落在 8 个允许改写的 ready sections。
3. review 已确认 `Summary of Contributions`、`Brief Review of the GKP Code`、`Noise and Drift Model`、`Model Architecture` 仍保持 untouched，这与任务包边界一致，而不是遗漏实现。
4. review 已确认 `paper_bounded_prose_reopen_manifest.md` 已把 changed sections、guardrails 与 compile 状态记录清楚，并继续保留：
   - `T24` frozen anchor
   - `FR6/FR7` descriptive-only
   - `FR8` extension-lane / no-promotion
   - `.tflite` isolated current-host only
   - real-board current-host `NO_GO`
5. review 还明确提醒：`T80` 不是 full-manuscript reopen，也不是 methods-ready、deployment upgraded 或 real-board story upgraded。

### 结论

1. 接受 `T80` 为 `PASS` 并标记完成。
2. 由于 verdict 为 `PASS`，本轮不进入 warning 分类流程，也不新开 risk。
3. `T80` 只关闭了“ready narrative / result-facing sections 是否已完成有界重写”的问题，不关闭任何 deployment/runtime/board 风险。
4. 当前唯一任务切换为 `T81`，只允许在 `Summary of Contributions` 与三章 methods 内做受控 contribution/methods calibration。
5. `T81` 必须继续保留 `T24` frozen anchor、`FR6/FR7` descriptive-only、`FR8` no-promotion、training/material supporting boundary、`.tflite` isolated current-host only 和 real-board current-host `NO_GO` 等 guardrail。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T80 -> PASS` 与 `Current Unique Task -> T81`。
2. 新增 `docs/tasks/Phase2/T81_summary_and_methods_calibration_pack.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T80` 产物、Captain closeout 文档和 `T81` 任务包，不要把无关历史 diff 混入。

## D-2026-06-12-07

- 日期：`2026-06-12`
- 决策：接受 `T79` 为 `PASS`，标记 `T79` 完成，并将当前唯一任务切换为 `T80: 主线校准段落的 bounded prose reopen`

### 背景

`T79` 的目标不是写 prose，而是在 `T74`-`T78` 已经完成材料整理、authoring、rendered QA、note 结果层同步与非结果层校准之后，给出一个唯一的 reopen/readiness 判断：当前材料栈是否已经足够支持下一轮 bounded prose reopen。

### 依据

1. `docs/review/T79_review.md` verdict = `PASS`，blocking issues = none。
2. review 已确认 `paper_reopen_gate_and_prose_readiness_review.md` 只包含一个 gate verdict：`GO_FOR_BOUNDED_PROSE_REOPEN`。
3. review 已确认 `paper_reopen_gate_and_prose_readiness_review.md` 只推荐一个后续任务：`T80: 主线校准段落的 bounded prose reopen`。
4. review 已确认 readiness matrix 覆盖了要求的最小 area，gap matrix 把每个 gap 都绑定到现有 evidence，而没有把未来实验或硬件条件写成既成事实。
5. review 还明确提醒：`GO_FOR_BOUNDED_PROSE_REOPEN` 不能被 retell 成 full-manuscript reopen approved、methods-ready approved、deployment boundary upgraded 或 real-board story upgraded。

### 结论

1. 接受 `T79` 为 `PASS` 并标记完成。
2. 由于 verdict 为 `PASS`，本轮不进入 warning 分类流程，也不新开 risk。
3. `T79` 只关闭了“当前是否允许下一轮 bounded prose reopen”的 gate 问题，不关闭任何 deployment/runtime/board 风险。
4. 当前唯一任务切换为 `T80`，只允许在已经 ready 的 8 个 narrative / result-facing sections 内做受控 prose reopen。
5. `T80` 必须继续保留 methods chapters out-of-scope、`T24` frozen anchor、`FR6/FR7` descriptive-only、`FR8` no-promotion、`.tflite` isolated current-host only 和 real-board current-host `NO_GO` 等 guardrail。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T79 -> PASS` 与 `Current Unique Task -> T80`。
2. 新增 `docs/tasks/Phase2/T80_mainline_calibrated_sections_bounded_prose_reopen.md` 作为下一张 worker-facing 任务包。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T79` 产物、Captain closeout 文档和 `T80` 任务包，不要把无关历史 diff 混入。

## D-2026-06-12-06

- 日期：`2026-06-12`
- 决策：接受 `T78` 为 `PASS`，标记 `T78` 完成，关闭 `R35`，并将当前唯一任务切换为 `T79: 论文材料 reopen gate 与 bounded prose 扩写就绪性评审`

### 背景

`T78` 的目标不是新增实验，而是在 `T77` 已完成结果层同步与 traceability hardening 的前提下，补完当前 note-draft 的非结果层校准、`statcalib` 视觉层级降权、LaTeX 排版 warning 收口和更机械的 section-scope 审计。review 已确认这些工作在允许范围内真实完成，且没有扩成 benchmark、训练、`.tflite`、real-board 或治理越界任务。

### 依据

1. `docs/review/T78_review.md` verdict = `PASS`，blocking issues = none。
2. review 已确认本轮 diff 保持在允许 section 范围内，`statcalib` 的结果层级完成结构性降权，note `.log` 中 `Underfull \hbox` 已收口到 `0`，且没有 `runs/`、`artifacts/`、源码目录或治理文档漂移。
3. `docs/paper_materials/paper_note_alignment_and_layout_closeout.md` 已把“已校准范围”和“未校准范围”明确分开，因此 `T78` 没有把局部 note 收口误写成整份 manuscript 已 fully calibrated。
4. review 的 non-blocking notes 都属于未来 full-manuscript reopen 前的后续 gate 提醒，而不是当前 `T78` 的未收口风险。

### 结论

1. 接受 `T78` 为 `PASS` 并标记完成。
2. `T78` 不引入新的 deferred/rejected warning，因此本轮不新增风险项。
3. `R35` 已由 `T78` 收口关闭；今后是否具备 prose reopen 条件，不再通过开放 note 校准风险表述，而改由专门的 gate 任务判断。
4. 当前唯一任务切换为 `T79`，优先做论文材料的 reopen/readiness gate，而不是直接重开 prose 扩写。
5. `T79` 必须保持 docs-only；它是 gate 评审任务，不是 prose drafting、benchmark rerun、`.tflite` 扩展、real-board 执行或 theory 分支大范围改写。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T78 -> PASS`、`R35` 关闭与 `Current Unique Task -> T79`。
2. 新增 `docs/tasks/Phase2/T79_paper_reopen_gate_and_prose_readiness_review.md` 作为下一张 worker-facing 任务包，要求产出 gate verdict、section-level readiness matrix、gap-to-action matrix 与唯一后续任务建议。
3. main 分支在完成上述治理同步后可以继续提交，但提交前应精确暂存 `T78` 产物、Captain closeout 文档和 `T79` 任务包，不要把无关历史 diff 混入。

## D-2026-06-12-05

- 日期：`2026-06-12`
- 决策：接受 `T76` 为 `PASS_WITH_WARNINGS`，标记 `T76` 完成，打开 `R34`，并将当前唯一任务切换为 `T77: 论文 note-draft 结果层同步与 T76 traceability hardening`

### 背景

`T76` 的目标不是新增实验，而是在 `T75` 已锁定的主文 Results prose / figure authoring 边界内，补做真实 rendered preview、人工可读性 QA、contact sheet / PDF bundle 与 Results-section assembly。review 已确认这层工作完成，但也指出当前 preview-source schema 与逐图 stable-ID 绑定粒度仍不够干净，尚不适合直接作为后续 note-draft 或自动化复用的长期接口。

### 依据

1. `docs/review/T76_review.md` verdict = `PASS_WITH_WARNINGS`，blocking issues = none。
2. review 已确认三张 preview PNG、contact sheet 和 PDF bundle 都真实落地，且人工目视检查认为本轮排版修正后的主要裁切问题已消失。
3. review 同时指出两类 paper-facing traceability 缺口：
   - `preview_source_map.csv` 的聚合行把 `upstream_t74_ids` 字段复用于 `T76-PREVIEW-*` 自身 ID；
   - `paper_rendered_figure_qa.md` 的逐图 QA 结论没有把对应上游 `T74-*` stable ID 全部内联写全。
4. review 中关于 `.tmp_t76_*` 与 `.tmp_t76_fontcache/` 的提醒属于提交治理与临时噪声问题，不影响 `T76` 完成性，但不应带入正式提交。
5. `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 已存在 experiment/results 章节，因此主线下一步不必停留在抽象 reopen gate，而可以先做一张更强的 docs-only note-draft 同步任务。

### 结论

1. 接受 `T76` 为 `PASS_WITH_WARNINGS` 并标记完成。
2. warning 分类为：
   - `N1` preview-source 聚合行字段语义复用 = `deferred -> R34`
   - `N2` `.tmp_t76_*` 探针/缓存残留 = `accepted`
   - `N3` 逐图 QA 结论未内联完整上游 `T74-*` stable ID = `deferred -> R34`
3. `R34` 打开，主题是 paper-facing preview traceability schema 与逐图 stable-ID 绑定粒度仍然偏弱。
4. 当前唯一任务切换为 `T77`，优先做 note-draft 结果层同步、`T76` traceability hardening 与可选的本地 note 编译检查。
5. `T77` 必须保持 docs-only；它不是 benchmark rerun、`.tflite` 扩展、real-board 执行、theory 分支大范围改写或 full-manuscript reopen。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T76 -> PASS_WITH_WARNINGS`、`R34` 和 `Current Unique Task -> T77`。
2. 新增 `docs/tasks/Phase2/T77_paper_note_results_sync_and_traceability_hardening.md` 作为下一张 worker-facing 任务包，要求同步 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 的结果层、修补 `T76` traceability 粒度、并在本地工具可用时做受控编译检查。
3. main 分支可以继续沿论文材料主线推进，但提交前必须精确排除 `.tmp_t76_*` / `.tmp_t76_fontcache/` 这类过程性噪声。

## D-2026-06-12-04

- 日期：`2026-06-12`
- 决策：接受 `PSE1: sidecar main-controlled governance refresh`，将 sidecar 从长期 `.wt/*` 分支同步模式调整为 main 控制台治理模式

### 背景

旧 Wave A sidecar worktree 已在 `2026-06-08` 创建，但 main 分支随后完成了大量治理、paper-material 与 evidence-pack 整理。旧 `.wt/*` 分支相对 main 的文档差异持续扩大，而每条 sidecar 分支的独有资产主要只是 S0 设计文档。继续逐分支 rebase / 同步会把成本消耗在文档搬运上，而不是实验路线本身。

### 依据

1. `docs/sidecar/parallel_sidecar_worktree_plan.md` 已记录旧 `.wt/tcn`、`.wt/teach`、`.wt/bank`、`.wt/ctrl` 只是 Wave A 工作面，并未运行实验或创建 `runs/sidecar`。
2. main 当前已经有 `docs/sidecar/` 目录入口，适合承接更轻量的 route registry、execution protocol、artifact schema 和 promotion gate。
3. 用户明确希望后续 sidecar 可以基于 main 当前代码新增实验实现，但不得破坏原有逻辑，结果按 lane 写入 `runs/sidecar/<lane_id>/<run_id>/`。
4. 继续维护长期 sidecar 分支会与当前 main 的快速文档整理节奏冲突。

### 结论

1. `docs/sidecar/` 成为 sidecar 精简治理入口。
2. 旧 `.wt/*` 长期 worktree 退役为 read-only reference；不再要求 rebase 到 main。
3. 旧 S0 路线思路收编到 `docs/sidecar/lane_plans/`。
4. 后续 sidecar 默认允许在 main 代码基础上新增-only helper / standalone module / task-scoped config；如需修改已有入口，必须是 additive、default-off、explicit opt-in，并通过旧路径回归验证。
5. sidecar 结果继续只允许写入 `runs/sidecar/<lane_id>/<run_id>/`，不得写入或改写主线历史 run root。
6. 任何 sidecar 晋升仍必须通过 Captain promotion gate；本决策不授权运行实验、不创建 `runs/sidecar`，也不改变当前唯一主线任务。

### 直接影响

1. 新增 `docs/tasks/Phase2/PSE1_sidecar_main_controlled_governance_refresh.md`。
2. 新增 `docs/sidecar/00_sidecar_snapshot.md` 至 `docs/sidecar/04_sidecar_promotion_gate.md`。
3. 更新 `docs/sidecar/README.md`、`docs/sidecar/parallel_sidecar_extension_governance.md` 与 `docs/sidecar/parallel_sidecar_worktree_plan.md`。
4. 在 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md` 和 `docs/07_handoff.md` 中加入 PSE1 supersession，避免继续误读旧 PSE0 worktree 强制规则。

## D-2026-06-12-03

- 日期：`2026-06-12`
- 决策：接受 `T75` 为 `PASS`，标记 `T75` 完成，并将当前唯一任务切换为 `T76: Rendered figure QA and results-section assembly pack`

### 背景

`T75` 的目标不是新增实验，而是把 `T74` 的 stable-ID 结果材料进一步压缩成主文 Results prose、caption/placement lock、appendix bridge、do-not-write guardrail 和 publication-facing 最终成图资产。当前 review 已确认这一层已完成，但也指出如果后续要真正进入稿件排版或投稿整合，仍需要一张新的 bounded 任务来完成 rendered preview QA 与 manuscript-facing assembly，而不是继续在 `T75` 内隐式扩 scope。

### 依据

1. `docs/review/T75_review.md` verdict = `PASS`，blocking issues = none。
2. review 明确确认 `authoring_manifest.json`、`asset_source_map.csv`、三张 `T75-FIG-*` SVG、四份 authoring 文档、worker summary、for-human 文档都已落地，且 `T75` 资产与上游 `T74-*` stable IDs 的回链一致。
3. `git diff --name-only -- runs`、`artifacts`、`cnn_fpga physics benchmark tests` 仍为空，说明 `T75` 保持 docs-only 范围。
4. review 中的两条 non-blocking note 不构成 `BLOCK` 或新风险：其一是当前 worktree 仍需精确暂存；其二是下一步若进入排版整合，应补真实 rendered preview QA。

### 结论

1. 接受 `T75` 为 `PASS` 并标记完成。
2. `T75` 没有新的 deferred/rejected warning，因此本轮不新增风险项。
3. review 中的非阻塞意见按操作性 carry-forward note 处理：精确暂存属于提交纪律，rendered preview QA 被提升为下一张 bounded 任务的目标，而不是回写为 `T75` 缺陷。
4. 当前唯一任务切换为 `T76`，继续沿着 main 分支 paper-material-first、board-lowest 的主线推进。
5. `T76` 必须保持 docs-only；它是 rendered figure QA + Results-section assembly pack，不是 benchmark rerun、`.tflite` 扩展、real-board 执行、theory 分支写作或 full-manuscript reopen。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T75 -> PASS` 和 `Current Unique Task -> T76`。
2. 新增 `docs/tasks/Phase2/T76_rendered_figure_qa_and_results_section_assembly_pack.md` 作为下一张 worker-facing 任务包，要求真实 rendered QA、手工可读性检查、task-scoped 预览输出和 manuscript-facing Results assembly，而不是简单补一句说明。
3. 在当前暂无 `Linux + FPGA` 硬件宿主前，main 分支可以继续提交 `T75/T76` 这类 paper-material / authoring / QA 主线任务；不建议让 worker 切回真板 execution 准备。

## D-2026-06-12-02

- 日期：`2026-06-12`
- 决策：接受 `T74` 为 `PASS`，标记 `T74` 完成，并将当前唯一任务切换为 `T75: Main-text results prose and final figure authoring pack`

### 背景

`T74` 的目标不是新增实验，而是把 `T24/T48/T50/T57/T58/T70/T72` 的主线 simulation/material 证据整理成可直接服务论文写作的 stable-ID 结果表、caption、insertion map、traceability 资产和 gap checklist。当前 review 已确认这一层已经完成，因此主线下一步不应回到 benchmark 或真板，而应进入一个更强但仍受控的 authoring 层：基于 `T74` 稳定入口去写 Results 段落和最终成图。

### 依据

1. `docs/review/T74_review.md` verdict = `PASS`，blocking issues = none。
2. 独立只读核查表明：`figure_manifest.json`、`submission_bundle_manifest.json`、`result_source_map.csv`、`caption_source_map.csv`、`table_snapshot.csv` 之间的 stable ID 完整一致，`status_counts` 为 `ready=11 / partial=3 / blocked=1`，且所有直接/支撑证据路径都存在。
3. `git diff --name-only -- runs`、`artifacts`、`cnn_fpga physics benchmark tests` 均为空，说明 `T74` 维持了 docs-only 范围。
4. review 中唯一的 non-blocking issue 只是提醒当前工作区还存在 `T74` 之外的既有 captain-side diff；这影响提交策略，不影响 `T74` 任务完成性，也不构成新的风险或 scope drift。
5. `T74` 已把 simulation/material-first 路线整理好，但 repo 仍缺一个明确的 next-step authoring package，来冻结主文 Results 段落、最终成图和 caption/placement lock。

### 结论

1. 接受 `T74` 为 `PASS` 并标记完成。
2. `T74` 没有新的 deferred/rejected warning，因此本轮不新增风险项。
3. review 中的 non-blocking staging caution 作为 `accepted` 的提交注意事项处理，不进入风险台账。
4. 当前唯一任务切换为 `T75`，继续沿着 main 分支 simulation/material-first、board-lowest 的主线推进。
5. `T75` 必须保持 docs-only；它是 main-text Results prose + final figure authoring pack，不是 benchmark rerun、`.tflite` 扩展、real-board 执行、theory 分支写作或 full-manuscript reopen。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T74 -> PASS` 和 `Current Unique Task -> T75`。
2. 新增 `docs/tasks/Phase2/T75_maintext_results_prose_and_final_figure_authoring_pack.md` 作为下一张 worker-facing 任务包，要求 actual figure authoring、bounded Results prose、caption lock 和 appendix bridge，而不是简单补注释。
3. 在当前暂无 `Linux + FPGA` 硬件宿主前，main 分支可以继续提交 `T74/T75` 这类 paper-material / authoring 主线任务；不建议让 worker 切回真板 execution 准备。

## D-2026-06-12-01

- 日期：`2026-06-12`
- 决策：接受 `T73` 为 `PASS`，标记 `T73` 完成，并将当前唯一任务切换为 `T74: Paper-ready simulation result and figure pack`

### 背景

`T73` 的目标不是新增实验，而是把 `T48/T50/T57/T58/T70/T72` 等近期主线证据统一回写到 paper-facing claim/evidence、result/figure、risk 和 README 入口层。当前用户又明确要求真实硬件优先级继续降低，因此 `T73` 之后最合理的主线推进不是回到真板或 prose，而是继续把已有仿真证据压缩成论文可直接复用的材料包。

### 依据

1. `docs/review/T73_review.md` verdict = `PASS`，blocking issues = none，non-blocking issues = none。
2. `T73` 当前 diff 只落在允许的 paper-material / review / explanation / worker-summary 路径；没有源码、测试、`runs/`、`artifacts/` 或治理文档越界。
3. `paper_claim_evidence_ledger.md`、`paper_result_figure_ledger.md`、`paper_claim_risk_table.md`、`paper_ablation_result_pack.md` 与 `docs/paper_materials/README.md` 已形成统一的 post-`T72` 主线入口，而且没有把 `.tflite`、real-board、`statcalib` 或 FR6/FR7 边界静默升级。
4. `T74` 任务包已存在，且可以在不新增实验的前提下被增强为更强的 paper-material 打包任务：加入 stable IDs、main-text/appendix/supplement placement、insertion map、submission bundle manifest 与更严格的 traceability 约束。

### 结论

1. 接受 `T73` 为 `PASS` 并标记完成。
2. `T73` 没有新的 deferred/rejected warning，因此本轮不新增风险项。
3. 当前唯一任务切换为 `T74`，继续沿着 main 分支 paper-first / board-lowest 主线推进。
4. `T74` 仍然必须保持 docs-only；它不是 benchmark rerun、`.tflite` 扩展、real-board 执行、theory 分支写作或 sidecar promotion 入口。

### 直接影响

1. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/04_task_board.md`、`docs/06_repo_noise_governance.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 `T73 -> PASS` 和 `Current Unique Task -> T74`。
2. `docs/tasks/Phase2/T74_paper_ready_simulation_result_and_figure_pack.md` 增强为更完整的 paper-material bundle 任务包，而不是停留在简单图表整理层。
3. 在当前暂无 `Linux + FPGA` 硬件宿主前，main 分支可以继续提交 `T73/T74` 这类 paper-material 主线任务；不建议让 worker 切回真板 execution 准备。

## D-2026-06-11-03

- 日期：`2026-06-11`
- 决策：保持 `T73` 为当前唯一任务，把 real-board execution 路线降为最低优先级 backlog，并预先定义 `T74: Paper-ready simulation result and figure pack` 作为 `T73` 后的唯一推荐下一任务

### 背景

用户明确说明当前暂时没有可用的 `Linux + FPGA` 硬件宿主，因此真板执行路线在近期既缺资源也缺新增证据条件。与此同时，项目更紧迫的缺口已经转向“高质量论文所需的仿真结果、图表、caption、材料索引是否充足且可追溯”。

### 依据

1. `T49/T71/T72` 的 strongest supported truth 仍然只是 current-host `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`、checked-in regeneration 与 transfer-pack provenance hardening；没有任何真板 execution success。
2. 用户已明确要求把真实实验优先级放到最低，优先补论文质量所需的仿真结果与图表材料。
3. `T73` 当前正好是 post-`T72` 的主线 claim/evidence、result/figure、risk 三台账刷新入口；它完成后，最自然的下一步是把这些主线仿真证据整理成 paper-ready result/figure pack，而不是继续推进真板路线。
4. theory 分支已经单独存在，因此 main 分支可以继续专注实验主线与 paper-facing supporting materials，而不与 theory 工作混写。

### 结论

1. 维持 `T73` 作为当前唯一任务，不切回真板方向。
2. `T37` 继续保持 `blocked + lowest-priority backlog`，直到硬件宿主条件变化。
3. 预定义 `T74` 作为 `T73` 通过后的唯一推荐下一任务，目标是把已有主线仿真证据整理成论文可直接复用的结果表、图表、caption 与 traceability pack。
4. main 分支后续优先顺序调整为：主线台账刷新 -> 论文可直接复用的仿真材料打包 -> supporting-material follow-up；真板执行路线后置。

### 直接影响

1. `docs/02_experiment_plan.md` 的 Part II 优先级改为 paper-first / board-lowest。
2. `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 需要同步记录 `T73 -> T74` 的主线衔接和 `T37` 的资源阻塞状态。
3. 新增 `docs/tasks/Phase2/T74_paper_ready_simulation_result_and_figure_pack.md` 作为下一张 worker-facing 任务包。
4. 在硬件条件变化前，main 分支可以继续提交 `T73/T74` 这类 paper-material 主线任务；不建议让 worker 切回真板 execution 准备。

## D-2026-06-11-02

- 日期：`2026-06-11`
- 决策：接受 `T72` 为 `PASS_WITH_WARNINGS`，关闭 `R31`，打开 `R32`，并把主线当前唯一任务切换为 `T73: Mainline claim/evidence and result/figure/risk ledger refresh`

### 背景

`T72` 的任务目标不是推进真板执行，而是把 `T71` 留下的 transfer-pack provenance 问题收紧到 execution-derived / override-aware 边界，同时保持 current-host `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE` verdict 不漂移。

### 依据

1. `docs/review/T72_review.md` 明确给出 reviewer verdict = `PASS_WITH_WARNINGS`，blocking issues = none。
2. fresh verification 通过：
   - `python -m py_compile cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
   - `python -m unittest tests.test_t71_real_board_gate_regeneration_pack`
   - `python -m unittest tests.test_t72_real_board_transfer_pack_provenance_hardening`
3. `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md` 已证明：
   - `probe_limitations` 不再把未执行探测写成固定事实
   - default-config 与 CLI override provenance 已变为 execution-derived / override-aware
   - `expected_byte_count_basis` 已从固定文案改为运行时推导
   - `T49` replay 与 `T72` current-host regeneration verdict 仍一致
4. review 剩余问题不影响 `T72` 的主目标，但会在 future-host 最小 config 场景下留下更窄的 provenance 标签精确性问题：
   - path provenance 仍不能区分“YAML 明确提供字段”和“代码默认值补出字段”
   - 缺少对应 focused regression
   - 原始主报告路径曾短暂越过精确 allowed-files 边界，但当前 `HEAD` 已整理回允许目录

### 结论

1. `T72` 应收口为 `PASS_WITH_WARNINGS`，而不是 `BLOCK`
2. `R31` 关闭，因为它对应的默认-config / override / probe-limitations 主问题已由 `T72` 收口
3. 新的残余风险登记为 `R32`，主题是 future-host 最小 config 场景下 path provenance 的 `config_field` / `code_default` 标签仍不够精确
4. `T37` 继续 blocked；`T72` 不构成真板执行放行
5. 当前唯一任务切换为 `T73`，优先刷新主线 paper-facing claim/evidence、result/figure、risk 三套台账

### 直接影响

1. 可以把 `T72` closeout 与 `T73` handoff 一起提交到 `main`
2. Worker 不应继续留在 `T72`，下一步只能领取 `T73`
3. 后续 retelling 不得把 `T72` 写成 real-board execution success、hardware validation 或 fully provenance-clean future-host closure
4. theory 分支继续独立；`T73` 仅在 main 分支实验主线推进

## D-2026-06-11-01

- 日期：`2026-06-11`
- 决策：接受 `T71` 为 `PASS_WITH_WARNINGS`，关闭 `R30`，并把主线当前唯一任务切换为 `T72: Real-board transfer-pack provenance hardening`

### 背景

`T49` 已经诚实证明当前 host 的 real-board gate verdict 为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`，`T71` 的目标是把这个 current-host `NO_GO` 收敛成一个仓库内可 replay / regeneration 的 checked-in、read-only、role-aware gate 包，而不是直接推进 `T37` 真板执行。

### 依据

1. `build_t49_real_board_smoke_gate.py` 现已使用 role-aware `mmio + dma` 判定，而不是只按 openable path 数量计数。
2. 仓库内已新增 `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`，并补齐了 `T49` replay 与 current-host regeneration 的 focused tests。
3. fresh verification 显示：
   - `python -m unittest tests.test_t49_real_board_smoke_execution_gate` 通过
   - `python -m unittest tests.test_t71_real_board_gate_regeneration_pack` 通过
   - current-host regeneration verdict 仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
   - `T49` checked-in artifact replay verdict 仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
4. review 中的剩余问题不影响当前主 verdict，但会削弱 future-host transfer-pack 的 provenance 严谨度：
   - `probe_limitations` 有未实际探测却写成事实的固定文案
   - `source_records` / `expected_byte_count_basis` 仍偏默认 config 写死
   - 对 `--config` / `--mmio-path` / `--dma-path` 的 provenance/override 回归覆盖不足

### 结论

1. `T71` 应收口为 `PASS_WITH_WARNINGS`，而不是 `BLOCK`
2. `R30` 关闭，因为 role-aware gate、checked-in collector、replay/regeneration consistency 三个缺口都已补齐
3. 新的风险登记为 `R31`，主题是 transfer-pack provenance 仍不够 execution-derived / override-safe
4. 当前唯一任务切换为 `T72`，只做 provenance 去硬编码与 override 回归加固
5. theory 分支继续独立；`T72` 仅在 main 分支实验主线推进

### 直接影响

1. 可以把 `T71` 代码与文档收口提交到 `main`
2. Worker 不应继续留在 `T71`，下一步只能领取 `T72`
3. `T37` 继续 blocked；`T72` 不构成真板执行放行
4. 后续任何 retelling 仍不得把 `T71` 改写成 `real-board validated`

## D-2026-05-05-01

- 日期：`2026-05-05`
- 决策：项目进入 `Repair`，而不是 `Go`、`Salvage` 或 `Stop`

### 背景

项目已经积累了较多代码、配置和实验结果，但仓库治理文件缺失，默认运行环境也未恢复到可直接复现状态。

### 依据

1. `cnn_fpga/` 与 `physics/` 主代码真实存在
2. P0/P2/P3/P4 入口脚本存在
3. `board_backend.py` 仍是 placeholder 风格，说明项目不能被表述为“已真板完成”
4. 默认 `python 3.13.7` 下最小 benchmark 因缺少 `numpy` 失败，说明当前仍需做入口恢复

### 结论

当前最合理路线是：

- 保留现仓库
- 先恢复治理、依赖、入口和最小验证
- 暂不继续扩功能或扩 teacher-representation 分支

### 直接影响

1. 当前唯一任务切到 `T1: 确认依赖矩阵与最小入口`
2. 恢复期默认不新增实验主线
3. 后续所有结论更新都要以复验结果为前提

## D-2026-05-06-01

- 日期：`2026-05-06`
- 决策：将恢复期推荐最小 smoke 解释器固定为 `C:\ProgramData\anaconda3\python.exe`

### 背景

为完成 `T1`，本轮对当前机器上的多套解释器进行了最小依赖探测与 smoke 验证。

### 依据

1. `C:\Python313\python.exe`
   - 无 `numpy`
   - 不能运行最小 P0 benchmark
2. `C:\ProgramData\anaconda3\python.exe`
   - 有 `numpy + yaml`
   - 已成功跑通：
     - `benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda`
3. `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
   - 有 `torch`
   - 更适合作为后续训练环境候选，而不是恢复期最小 smoke 环境
4. 仓库内未找到 `.venvs/tf311`

### 结论

恢复期当前最稳妥的环境分工为：

1. 最小 smoke / P0：`C:\ProgramData\anaconda3\python.exe`
2. 后续 torch 训练候选：`C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
3. 默认 `python 3.13.7` 不再作为项目运行解释器

### 直接影响

1. `T1` 可标记完成
2. 当前唯一任务切换到 `T2`
3. 后续若补 bootstrap 文档，应以该分工为起点

## D-2026-05-06-02

- 日期：`2026-05-06`
- 决策：将恢复期 P0 smoke 的最终交接口径固定为“轻量解释器 smoke + DLEnv 训练候选”

### 背景

已确认：

1. `C:\ProgramData\anaconda3\python.exe`
   - 有 `numpy + yaml`
   - 已成功跑通最小 P0 smoke
2. `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
   - 有 `numpy + yaml + torch`
   - 更适合作为后续训练环境候选

### 结论

1. 恢复期最小 smoke 继续用轻量解释器，不切换到 DLEnv
2. `DLEnv` 保留为 legacy 开发常用训练环境候选
3. `docs/recovery_bootstrap/P0_smoke_bootstrap.md` 作为当前最小复用说明

### 直接影响

1. `T2` 可标记完成
2. 当前唯一任务切换到 `T3`

## D-2026-05-06-03

- 日期：`2026-05-06`
- 决策：固定 HIL / P4 的恢复期表述口径为“软件 HIL 已有主链，但必须显式标注 backend 与 artifact type；real-board 仍是 placeholder；`tflite_stub_v1` 不等于真实 TFLite 部署”

### 背景

完成 `T3` 边界审计后，已确认当前项目最容易被误写的，不是 P0 环境，而是 P3/P4 链路里的真实实现边界。

### 依据

1. `cnn_fpga/benchmark/run_hil_suite.py`
   - 通过 `hil.backend` 选择 backend
   - `backend == "mock"` 时构造 mock noise provider
   - 真实写出 `hil_events.json` 与 `hil_summary.json`
2. `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
   - 直接调用 `run_hil_session(...)`
   - P4 benchmark 没有绕开 HIL backend realism 限制
3. `cnn_fpga/hwio/board_backend.py`
   - 文件顶层直接声明为 `Placeholder real-board backend`
   - `schedule_commit(...)` 返回含多个 `None` 字段的占位元信息
   - `step(...)` 仅刷新状态并返回空事件
4. `cnn_fpga/model/export.py` 与 `cnn_fpga/runtime/inference_service.py`
   - 导出路径存在真实 `.tflite` 与 `tflite_stub_v1` 两条分支
   - 推理路径明确区分 `tflite_service` 与 `tflite_stub_service`

### 结论

恢复期后续所有涉及 P3/P4 的描述、报告和复验，都必须显式回答两个问题：

1. backend 是 `mock` 还是 `board`
2. inference artifact 是 `artifact_npz`、真实 `.tflite`，还是 `.tflite.json` stub

### 直接影响

1. `T3` 可标记完成
2. 当前唯一任务切换到 `T4`
3. 后续 `T4/T6/T7` 默认不能把 `real_board` 或 `tflite_stub_v1` 写成已完成真实部署

## D-2026-05-06-04

- 日期：`2026-05-06`
- 决策：将恢复期最小 software HIL bootstrap 固定为“`mock + model_artifact + artifact_npz + inproc`，并使用 `C:\ProgramData\anaconda3\python.exe` 运行 `hardware_hil_recovery_smoke.yaml`”

### 背景

为完成 `T4`，需要把“软件 HIL 可复验”从文档口径落成一条当前机器上实际可跑的最小路径，同时避免把 `.venvs/tf311`、`.tflite` 或 `real_board` 引入恢复期最小入口。

### 依据

1. `cnn_fpga/config/hardware_hil.yaml`
   - 默认 `inference_service.python_executable` 指向 `.venvs/tf311/bin/python`
   - 对当前 Windows 恢复期最小路径不稳妥
2. `artifacts/models/static_theta_v2/`
   - 已存在可直接读取的 `tiny_cnn_... .npz` artifact
3. `cnn_fpga/model/tiny_cnn.py`
   - `tiny_cnn_float` artifact 的推理路径可由 NumPy 实现
4. `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - 已显式固定：
     - `hil.backend=mock`
     - `slow_loop.mode=model_artifact`
     - `inference_service.mode=inproc`
     - `inference_service.backend=artifact_npz`
5. 实际运行结果：
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - 运行目录：`runs/hil_suite/hardware_hil_recovery_smoke_20260506_021326_3ae9f9176104`
   - 关键结果：
     - `backend = mock`
     - `n_slow_updates_finished = 2`
     - `n_commits_applied = 2`
     - `artifact_path = ...static_theta_v2...npz`
     - `inference_service_mode = inproc`

### 结论

恢复期最小 software HIL 入口已经恢复，但其结论边界必须保持为：

1. 已恢复的是 `mock-backed software HIL smoke`
2. 不是 `real_board HIL`
3. 不是 `.tflite` runtime 验收
4. 不是 P4 正式 benchmark 验收

### 直接影响

1. `T4` 可标记完成
2. 当前唯一任务切换到 `T5`
3. 后续 `T6` 若要做正式最小复验，应继承同一条 bootstrap 路径或在此基础上明确升级

## D-2026-05-07-01

- 日期：`2026-05-07`
- 决策：将仓库噪声治理口径固定为“先分类治理、暂不破坏性清理；缓存/字节码后续移出版本库；`runs/` 与 `artifacts/` 在恢复期暂作历史证据保留”

### 背景

`T4` 已恢复最小 software HIL bootstrap，但仓库内仍混有大量历史缓存文件与实验输出。如果不先固定这些内容在恢复期的语义边界，`T6/T7` 的复验结果会继续与历史产物混淆。

### 依据

1. `.gitignore` 已忽略：
   - `__pycache__/`
   - `runs/`
   - `artifacts/`
2. 但 Git 中仍已有大量历史已跟踪文件：
   - 已跟踪缓存/字节码文件：`116`
   - 已跟踪 `runs/` 文件：`1841`
   - 已跟踪 `artifacts/` 文件：`110`
3. 当前工作区仍可见 `9` 个 `__pycache__` 目录，`.pyc` 总数为 `133`
4. 当前最小 software HIL 路径仍依赖 `artifacts/models/static_theta_v2/...npz`，说明不能在恢复期直接粗暴清空历史产物

### 结论

恢复期对仓库噪声的统一口径固定为：

1. `__pycache__/` / `.pyc`
   - 属于应最终移出版本库的历史噪声
   - 但 `T5` 不直接做批量删除或 untrack
2. `runs/`
   - 视为历史运行证据
   - 后续文档必须引用具体 run dir，不能把整个目录写成新的事实来源
3. `artifacts/`
   - 先保留当前 bootstrap 所需 artifact
   - 后续再拆分“bootstrap 必需”与“历史归档”
4. 本地临时文档文件
   - 立即通过 `.gitignore` 继续约束
   - 例如 `*.drawio.dtmp`

### 直接影响

1. `T5` 可标记完成
2. 新增 `docs/06_repo_noise_governance.md`
3. 当前唯一任务切换到 `T6`
4. 在专门 cleanup 任务出现前，后续任务默认不做 `runs/`、`artifacts/`、`__pycache__/` 的破坏性清理

## D-2026-05-07-02

- 日期：`2026-05-07`
- 决策：将恢复期最小 software HIL 路径提升为“已二次复验通过，但暂按可复验而非逐字确定性复现表述”，并将当前唯一任务切换到 `T7`

### 背景

`T4` 已经恢复出一条最小 software HIL bootstrap 路径，但要进入 `Phase 1: Recovery`，还需要确认它不是一次性成功，也没有被历史结果语义掩盖。

### 依据

1. 使用同一命令再次运行：
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
2. 新运行目录：
   - `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`
3. 新 summary 结果再次确认：
   - `backend = mock`
   - `n_windows_ready = 2`
   - `n_slow_updates_started = 2`
   - `n_slow_updates_finished = 2`
   - `n_commits_applied = 2`
   - `artifact_path = ...static_theta_v2...npz`
   - `inference_service_mode = inproc`
4. 新 events 结果再次确认：
   - `window_ready = 2`
   - `slow_update_started = 2`
   - `slow_update_finished = 2`
   - `commit_applied = 2`
   - `commit_ack_asserted = 2`
   - `fast_budget_violation = 1`
5. 两次复验的 control-plane 字段一致，但 `final_ler` 与 `overflow_rate` 存在小幅差异
6. 代码侧仍可见一部分全局 `np.random` 路径，例如：
   - `physics/syndrome_measurement.py`

### 结论

恢复期当前可以把这条最小 software HIL 路径表述为：

1. `mock-backed software HIL` 已完成二次复验
2. `backend`、`artifact_path`、`inference_service_mode` 已再次固定
3. 当前更适合表述为“可复验”
4. 暂不把它表述为“逐字确定性复现”

### 直接影响

1. `T6` 可标记完成
2. `docs/recovery_bootstrap/P3_software_hil_bootstrap.md` 应更新为包含最新 run 证据
3. 当前唯一任务切换到 `T7`
4. `T7` 必须继承同一条 `mock + model_artifact + artifact_npz + inproc` 口径

## D-2026-05-08-01

- 日期：`2026-05-08`
- 决策：将恢复期最小 P4 benchmark 路径固定为“`mock-backed P4 recovery smoke`，显式复用 `T6` 的 software HIL 主链，并将当前唯一任务切换到 `T8`”

### 背景

`T6` 已把最小 software HIL 路径重新验收到“可复验”状态，但 `run_p4_multiscenario_benchmark.py` 还没有在同一套 `mock + model_artifact + artifact_npz + inproc` 口径下完成新的最小复验。

### 依据

1. 新增恢复期专用 P4 配置：
   - `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
   - 显式固定：
     - `hil.backend=mock`
     - `slow_loop.inference_service.mode=inproc`
     - `slow_loop.inference_service.backend=artifact_npz`
     - `slow_loop.model_artifact.path=...static_theta_v2...npz`
2. 新增恢复期专用 bootstrap 文档：
   - `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
3. 实际运行命令：
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds`
4. 新运行目录：
   - `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308`
5. 新 summary 结果确认：
   - `protocol_id = p4_hil_recovery_smoke_v1`
   - `scenario = static_bias_theta`
   - `modes = static_linear, cnn_fpga`
   - `seed_pairing = paired`
6. 新 comparison 结果确认：
   - `Static Linear final_ler = 1.00890625`
   - `CNN-FPGA final_ler = 0.72109375`
   - `CNN-FPGA artifact_path = ...static_theta_v2...npz`
7. 新 repeat HIL summary 再次确认：
   - `backend = mock`
   - `n_slow_updates_finished = 8`
   - `n_commits_applied = 8`
   - `cnn_fpga` repeat 中 `inference_service_mode = inproc`

### 结论

恢复期当前可以把这条 P4 路径表述为：

1. `mock-backed P4 benchmark minimal recovery path` 已完成一次新的复验
2. 它显式复用了 `T6` 已固定的软件 HIL 主链
3. 当前更适合表述为“P4 recovery smoke 已可复验”
4. 暂不把它表述为“正式四场景四模式 frozen benchmark 已恢复”

### 直接影响

1. `T7` 可标记完成
2. `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md` 应更新为包含最新 run 证据
3. 当前唯一任务切换到 `T8`
4. `T8` 需要基于 `T6 + T7` 的证据，决定项目继续 `Repair` 还是进入 `Go`

## D-2026-05-08-02

- 日期：`2026-05-08`
- 决策：基于 `T6 + T7` 的现有证据，项目继续保持 `Repair`，不立即进入 `Go`

### 背景

`T8` 的目标不是再跑一个新 benchmark，而是判断当前仓库是否已经从“恢复可信度”跨过了进入正常开发的门槛。

### 依据

1. `T6` 已确认最小 software HIL 路径可复验：
   - `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`
2. `T7` 已确认最小 P4 benchmark 路径可复验：
   - `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308`
3. 但 `T7` 当前仍只覆盖：
   - `single-scenario`
   - `two-mode`
   - `repeats = 1`
4. 根目录仍缺少最小依赖 manifest：
   - 无 `requirements.txt`
   - 无 `pyproject.toml`
   - 无 `environment.yml`
5. `T6` 仍保留“可复验而非逐字确定性复现”的随机性观察
6. `docs/review/T8_gate_review.md` 已明确给出 verdict：
   - `Continue Repair`

### 结论

当前最合理的 gate 判断是：

1. 项目已经进入 `Phase 1: Recovery`
2. 最小 P3/P4 recovery path 已经恢复
3. 但现阶段证据还不足以把仓库决策状态切到 `Go`

### 直接影响

1. `T8` 可标记完成
2. 当前决策状态继续保持 `Repair`
3. 当前唯一任务切换到 `T9`
4. `T9` 应优先扩大 P4 frozen baseline 的 recovery 证据，而不是重新发散到新功能或长跑

## D-2026-05-08-03

- 日期：`2026-05-08`
- 决策：将恢复期 `P4 frozen baseline` 证据提升为“`single-scenario + four-mode + repeats=1` smoke 已复验”，并将当前唯一任务切换到 `T10`

### 背景

`T8` 的 gate review 已明确指出，`T7` 的 `single-scenario + two-mode + repeats=1` 证据还不足以支撑进入 `Go`。因此先执行 `T9`，在不扩到正式多场景长跑的前提下，把单场景 mode 集补齐到冻结的四类 baseline。

### 依据

1. `T9` task package 已固定：
   - `docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`
2. 实际运行命令：
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds`
3. 新运行目录：
   - `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`
4. `summary.json` 已确认：
   - `protocol_id = p4_hil_recovery_smoke_v1`
   - `scenario = static_bias_theta`
   - `modes = static_linear, window_variance, ekf, cnn_fpga`
   - `seed_pairing = paired`
5. `comparison.csv` 已确认：
   - `Static Linear final_ler = 0.99575`
   - `Window Variance final_ler = 0.57440625`
   - `EKF final_ler = 0.6795`
   - `CNN-FPGA final_ler = 0.7248125`
   - scenario winner: `window_variance`
6. 各 mode 的 repeat HIL summary 已继续确认：
   - 四个 mode 都是 `backend = mock`
   - 四个 mode 都是 `inference_service_mode = inproc`
   - `static_linear / window_variance / ekf` 的 `artifact_path = null`
   - `cnn_fpga` 的 `artifact_path = ...static_theta_v2...npz`
7. 当前边界仍然成立：
   - 这仍是 `mock-backed P4 recovery smoke`
   - 不是 `real_board`
   - 不是 `.tflite` runtime 验收
   - 不是正式多场景 frozen benchmark

### 结论

当前最合理的仓库状态表述是：

1. `T9` 已完成
2. 恢复期 `P4` 证据已从“两模式最小路径”扩到“冻结 baseline 四模式单场景 smoke”
3. 但 `T9` 本身还不直接改变 `Repair / Go` 决策状态
4. 下一唯一任务应切换到 `T10`，对 `T8 + T9` 的组合证据重新做一次 gate review

### 直接影响

1. `T9` 可标记完成
2. `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md` 应更新为包含最新四模式 run 证据
3. 当前唯一任务切换到 `T10`
4. `T10` 应优先回答“`T9` 是否足以支撑进入 `Go`”，而不是继续扩 benchmark 或顺手扩功能

## D-2026-05-08-04

- 日期：`2026-05-08`
- 决策：基于 `T8 + T9` 的组合证据，项目继续保持 `Repair`，不进入 `Go`，并将当前唯一任务切换到 `T11`

### 背景

`T9` 已经把 `P4 frozen baseline` 的 recovery 证据从“两模式最小路径”扩到“单场景四模式 smoke”。`T10` 的目标就是正式判断：这组增强后的证据，是否已经足以把仓库从 `Repair` 切到 `Go`。

### 依据

1. `T10` gate review 文档已完成：
   - `docs/review/T10_gate_review.md`
2. `T9` 已确认：
   - `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`
   - `single-scenario + four-mode + repeats=1`
   - `backend = mock`
   - `cnn_fpga artifact_path = ...static_theta_v2...npz`
3. `T6` 已确认最小 software HIL 路径可复验，但仍不是逐字确定性复现：
   - `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`
4. 根目录当前仍无最小依赖 manifest：
   - `requirements.txt`
   - `pyproject.toml`
   - `environment.yml`
5. 因此当前仍存在三个关键缺口：
   - 环境可移植性未收口
   - software HIL 确定性复现未收口
   - P4 仍未从单场景 smoke 提升到正式多场景 frozen benchmark

### 结论

当前最合理的 gate 判断是：

1. 项目继续保持在 `Phase 1: Recovery`
2. 决策状态继续维持 `Repair`
3. `T9` 虽然显著增强了 P4 recovery 证据，但还不足以单独支撑进入 `Go`
4. 下一唯一任务应先切到 `T11`，补 recovery 期最小依赖 manifest

### 直接影响

1. `T10` 可标记完成
2. 当前决策状态继续保持 `Repair`
3. 当前唯一任务切换到 `T11`
4. `T11` 应优先收口 recovery smoke 的依赖 manifest，而不是继续扩 benchmark 长跑或新功能

## D-2026-05-08-05

- 日期：`2026-05-08`
- 决策：在根目录新增 `requirements-recovery.txt` 作为 recovery-scoped 最小依赖 manifest，并显式保持它不是完整仓库环境文件

### 背景

`T10` 的 gate review 已明确指出：当前最容易收口、也最影响接力效率的缺口，就是根目录仍没有一份作用域诚实的最小依赖说明。`T11` 的目标就是把已经复验过的 `P0/P3/P4 recovery smoke` 路径，收口成一个可直接引用的根级 manifest。

### 依据

1. 当前已复验的 recovery 入口只有三类：
   - `benchmark/compare_full_vs_simplified_ler.py --no-plot`
   - `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml ...`
2. 这些 recovery 入口的第三方依赖边界当前可收口到：
   - `numpy`
   - `PyYAML`
3. `benchmark/compare_full_vs_simplified_ler.py` 中的 `matplotlib` 只在去掉 `--no-plot` 时才会触发，因此不应被写成 recovery smoke 的必需依赖
4. `torch`、`tensorflow`、`tflite-runtime` 与 `real_board` backend 只属于训练链、`.tflite` 或真板路径，不属于当前已复验的 recovery smoke 默认范围
5. 当前推荐解释器分工仍成立：
   - recovery smoke：`C:\ProgramData\anaconda3\python.exe`
   - 训练候选：`C:\ProgramData\anaconda3\envs\DLEnv\python.exe`

### 结论

恢复期当前采用如下依赖口径：

1. 根目录新增 `requirements-recovery.txt`
2. 它只覆盖 `P0/P3/P4 recovery smoke`
3. 它故意不命名为 `requirements.txt`，避免被误读为“完整仓库环境已恢复”
4. 它故意不引入 `torch`、`tensorflow`、`tflite-runtime`、`matplotlib`
5. 训练链、`.tflite` 路径与真板路径仍需继续按恢复期边界单独说明

### 直接影响

1. `T11` 可标记完成
2. `README.md` 与 `docs/recovery_bootstrap/P0_smoke_bootstrap.md`、`docs/recovery_bootstrap/P3_software_hil_bootstrap.md`、`docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md` 应改为显式引用 `requirements-recovery.txt`
3. 根目录“完全没有 manifest”的风险已收口为“只有 recovery-scoped manifest，完整环境仍未统一”
4. 当前唯一任务切换到 `T12`
5. `T12` 应优先收口 software HIL recovery smoke 的随机源与确定性表述，而不是继续扩 benchmark 长跑或新功能

## D-2026-05-08-06

- 日期：`2026-05-08`
- 决策：将恢复期最小 software HIL recovery smoke 的表述从“可复验”提升为“在固定 seed 链路下已完成逐字一致复验”，但严格限定在 `mock + model_artifact + artifact_npz + inproc` 这条 bounded 路径内

### 背景

`T11` 已经补齐 recovery-scoped manifest，但 `T6/T10` 仍保留一个关键缺口：`hardware_hil_recovery_smoke` 的 control-plane 字段虽然稳定，`final_ler` 与 `overflow_rate` 仍存在小幅 run-to-run 差异。`T12` 的目标就是在不改 benchmark 主线语义的前提下，把这个 bounded recovery path 的随机源收口到更强的可复现状态。

### 依据

1. `physics/syndrome_measurement.py`
   - `RealisticSyndromeMeasurement` 现支持注入显式 `rng`
   - recovery 路径的测量噪声、shot noise 与 ancilla 扰动不再依赖全局 `np.random`
2. `cnn_fpga/runtime/fast_loop_emulator.py`
   - 快回路误差采样继续使用 `default_rng(seed)`
   - 综合征测量额外拆出 `default_rng(seed + 1)`
   - 两类随机源已显式分离且都挂在同一条 seed 链上
3. `run_hil_suite.py` 当前 recovery seed 链路已明确：
   - mock noise provider: `seed + 17`
   - physical noise bridge: `seed + 19`
   - driver / mock backend: `seed + 7`
   - slow loop runtime: `seed + 31`
   - host latency injector: `seed + 43`
4. 使用同一命令连续复验两次：
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
5. 两次新运行目录分别为：
   - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104`
   - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104`
6. 文件级对比结果：
   - `hil_summary.json` SHA256 完全一致
   - `hil_events.json` SHA256 完全一致
7. 两次 run 的共同关键结果：
   - `backend = mock`
   - `artifact_path = ...static_theta_v2...npz`
   - `inference_service_mode = inproc`
   - `n_slow_updates_finished = 2`
   - `n_commits_applied = 2`
   - `final_ler = 0.454375`
   - `overflow_rate = 0.002`

### 结论

恢复期当前可以把这条最小 software HIL 路径表述为：

1. `mock-backed software HIL recovery smoke` 已完成逐字一致复验
2. 它的确定性结论仅限于当前 bounded 路径与当前机器上的 seed 链
3. 这不等于 `real_board` 已恢复
4. 这也不等于 `.tflite` runtime 或正式多场景 P4 benchmark 已恢复

### 直接影响

1. `T12` 可标记完成
2. `docs/recovery_bootstrap/P3_software_hil_bootstrap.md` 应更新为包含新的确定性复验证据与随机源链路说明
3. `docs/01_legacy_audit.md`、`docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 应同步移除“该路径仍有 run-to-run 数值漂移”的旧表述
4. 当前唯一任务清空，等待下一张任务包

## D-2026-05-08-07

- 日期：`2026-05-08`
- 决策：第一轮 `Phase 1: Recovery` 收尾完成，项目退出恢复期，进入 `Phase 2: Controlled Development`，决策状态切换为 `Go`

### 背景

`T10` 时点仍不能进入 `Go`，因为当时还缺少 recovery-scoped manifest 与 bounded software HIL 的确定性收口。`T11` 和 `T12` 完成后，需要再做一次正式的 recovery exit review，判断当前仓库是否已经满足工作流里“Recovery-Ready”的退出条件。

### 依据

1. `docs/reference/AI_coding_workflow.md` 第 4 节迁移/恢复工作流给出的退出标准，当前已满足：
   - 最小路径可按文档运行
   - MVP 范围明确
   - task board / handoff / risks 稳定
   - smoke / 等价验证存在
   - mock / stub / placeholder 已标记
   - 不再依赖旧长 session 才能理解仓库状态
2. `T11` 已完成：
   - 根目录新增 `requirements-recovery.txt`
   - recovery 期最小依赖边界已固定
3. `T12` 已完成：
   - bounded software HIL recovery smoke 已逐字一致复验
4. `P0/P3/P4` 的 bounded recovery 入口都已有新的文档化证据：
   - `docs/recovery_bootstrap/P0_smoke_bootstrap.md`
   - `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
   - `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
5. `docs/review/T13_recovery_exit_review.md`
   - 已给出 verdict：`Allow`

### 结论

当前最合理的仓库状态表述是：

1. 第一轮恢复期目标已完成
2. 项目可以从 `Repair` 切换到 `Go`
3. 这里的 `Go` 只代表“允许继续做 bounded 开发任务”
4. 它不代表：
   - `real_board` 已恢复
   - 真实 `.tflite` runtime 已恢复
   - 正式多场景 frozen benchmark 已恢复

### 直接影响

1. `AGENTS.md` 与 `README.md` 应切换到新的阶段与决策状态
2. `docs/04_task_board.md`、`docs/07_handoff.md` 应标记 recovery 已收尾，当前唯一任务待重新定义
3. 后续任务默认从“恢复期修复”切换为“受控继续开发”，但仍保留真实边界与验证硬规则

## D-2026-05-08-08

- 日期：`2026-05-08`
- 决策：按 AI Coding 工作流重新初始化 Phase 2 任务板，并将当前唯一任务设为 `T14: P4 frozen benchmark protocol audit and bounded run plan`

### 背景

`T13` 已完成 recovery exit，项目进入 `Phase 2: Controlled Development`。但 `docs/04_task_board.md` 仍处于“下一任务待定义”状态，无法直接指导后续 Worker 会话。

### 依据

1. `docs/reference/AI_coding_workflow.md` 要求 Captain 明确：
   - 当前唯一任务
   - Worker 任务包
   - Allowed files
   - Forbidden scope
   - Verification
   - Docs to update
2. `docs/02_experiment_plan.md` 的后续优先级和禁止项同时表明：
   - P4 多场景证据重要
   - 但不应无准备启动长时间正式 benchmark
3. 当前已有证据边界：
   - `T9` 只覆盖 `single-scenario + four-mode + repeats=1`
   - `T12` 只覆盖 bounded software HIL deterministic recovery smoke
   - `real_board` 与 `.tflite` 路径仍不能被外推

### 结论

Phase 2 先按以下顺序推进：

1. `T14`: P4 frozen benchmark protocol audit
2. `T15`: P4 multi-scenario frozen baseline bounded smoke
3. `T16`: P4 evidence gate review
4. `T17` / `T18`: training 与 `.tflite` 独立 manifest
5. `T19` / `T20`: repo cleanup manifest 与 real-board readiness

当前唯一任务为 `T14`，本任务只做 protocol audit 和 bounded run plan，不启动 benchmark 长跑。

### 直接影响

1. `docs/04_task_board.md` 成为 Phase 2 Worker 会话的任务主状态。
2. `docs/tasks/Phase2/` 新增 `T14` 至 `T20` 的任务包。
3. `docs/07_handoff.md` 与 `docs/08_risks_and_open_questions.md` 同步更新为 `T14` 当前任务口径。
4. 后续 Worker 不应绕过 `T14` 直接执行 `T15` 或长跑 benchmark。

## D-2026-05-09-01

- 日期：`2026-05-09`
- 决策：接受 `T15` 的 `PASS_WITH_WARNINGS` review，标记 `T15` 完成，并将当前唯一任务切换为 `T16: P4 benchmark evidence review and next-gate decision`

### 背景

`T15` 已按 `T14` 固定的 bounded matrix 完成双场景、五模式、`repeats=2` 的 P4 development run。`docs/review/T15_frozen_smoke_review.md` 给出 verdict：`PASS_WITH_WARNINGS`，且没有 blocking issue。

### Warning 分类

1. N1: handoff 多个状态节未同步更新
   - 分类：`accepted`
   - 处理：Captain 在本次整合中更新 `docs/04_task_board.md` 与 `docs/07_handoff.md`
2. N2: `hybrid_residual_b` teacher diagnostics 全零
   - 分类：`deferred`
   - 处理：写入 `docs/08_risks_and_open_questions.md` 的 R10，并要求 `T16` gate review 判断
3. N3: `delta_rows` 全部为 null
   - 分类：`accepted`
   - 处理：记录为 strong-baseline config 不包含 `static_linear` / `cnn_fpga` 的预期后果，提醒 `T16` 不要误判

### 依据

1. `T15` matrix 与 `T14` protocol 完全匹配：
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`
   - repeats: `2`
   - seed policy: `paired`
2. 新 run dir：
   - `runs/p4_benchmark/p4multis_20260508_221718_b82874_48280`
3. Review 已确认：
   - `missing_runs = []`
   - 10 个 comparison rows 均 `coverage = 1.0`
   - no forbidden scope violation
   - no code / config / ParamMapper changes

### 结论

`T15` 可以标记完成，但其结果只升级为 `development bounded run` evidence，不升级为正式四场景 frozen benchmark 结论。

下一唯一任务为 `T16`。`T16` 只做 gate review，不运行新 benchmark。

### 直接影响

1. `docs/04_task_board.md` 标记 `T15` 完成，并切换 `Current Unique Task` 到 `T16`。
2. `docs/07_handoff.md` 补齐 `T14/T15` 完成记录、T15 evidence 与当前任务摘要。
3. `docs/08_risks_and_open_questions.md` 增加 R10，并记录 warning 分类。
4. 在 `T16` 完成前，不继续扩大 P4 benchmark。

## D-2026-05-09-02

- 日期：`2026-05-09`
- 决策：接受 `T16` milestone review 的 `PASS_WITH_WARNINGS`，保留 `T16` gate verdict = `Conditional`，并将当前唯一任务切换为 `T17: Training-chain independent manifest and bootstrap`

### 背景

`T16` gate review 已完成，verdict = `Conditional`。随后 `docs/review/T16_milestone_review.md` 给出 `PASS_WITH_WARNINGS`，且没有 blocking issue。

### Warning 分类

1. N1: `T16` review 深度和 handoff / task board 状态一致性问题
   - 分类：`accepted`
   - 处理：Captain 在 `docs/04_task_board.md` 与 `docs/07_handoff.md` 中修正状态口径
2. N2: `T16` review 未充分讨论 R5/R9 风险是否可降级
   - 分类：`deferred`
   - 处理：暂不降级 R5/R9，后续风险维护任务再判断

### 依据

1. `T16` gate verdict = `Conditional`，允许继续 Phase 2 受控开发，但不建议继续扩大 P4 benchmark。
2. `T16` milestone review 没有 blocking issue。
3. 当前更适合转向训练链、`.tflite` 等独立 manifest / boundary 任务。

### 结论

`T16` 可以标记完成，但 P4 证据仍保持 `Conditional` 边界；下一唯一任务切换为 `T17`。

### 直接影响

1. `docs/04_task_board.md` 切换当前唯一任务为 `T17`。
2. `docs/07_handoff.md` 同步 `T16` 完成记录与 `T17` 任务摘要。
3. `docs/08_risks_and_open_questions.md` 保留 R10，并继续把 R5/R9 作为未降级风险。

## D-2026-05-10-01

- 日期：`2026-05-10`
- 决策：接受 `T17` review 的 `PASS`，标记 `T17` 完成，并将当前唯一任务切换为 `T18: TFLite export/runtime manifest and boundary smoke plan`

### 背景

`T17` 已完成训练链独立 bootstrap。`docs/review/T17_review.md` 给出 verdict：`PASS`，没有 blocking issue。

### Non-blocking 观察

1. N1: `torch = 2.8.0.dev20250405+cu128` 是 dev build
   - 分类：`accepted`
   - 处理：当前 bootstrap 已明确它只是本机环境探测结果；后续不把该版本写成跨机器保证
2. N2: 未产出 `requirements-train.txt`
   - 分类：`accepted`
   - 处理：任务允许选择 `requirements-train.txt` 或 `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md`；在 dev torch 场景下先采用文档化 bootstrap 更诚实。若后续需要训练链可移植性，再单开依赖锁定任务

### 依据

1. `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md` 已独立说明训练链推荐解释器、入口、依赖边界与未覆盖项。
2. `requirements-recovery.txt` 仍只覆盖 `P0/P3/P4 recovery smoke`，没有被混写成训练链 manifest。
3. Worker 没有修改训练代码、没有启动训练长跑、没有改模型主线。
4. Verification 已达到任务包要求：DLEnv import 级检查与 `python -m cnn_fpga.model.train --help`。

### 结论

`T17` 可以标记完成。下一唯一任务为 `T18`，用于收口 `.tflite` export/runtime 路径的 manifest 与 boundary smoke plan。

### 直接影响

1. `docs/04_task_board.md` 标记 `T17` 完成，并切换 `Current Unique Task` 到 `T18`。
2. `docs/07_handoff.md` 补齐 `T17` review 判定、warning 处理和 `T18` 任务摘要。
3. `docs/08_risks_and_open_questions.md` 记录训练链 dev build / requirements-train 后续风险，并把当前下一任务更新为 `T18`。
4. `T18` 任务包已经存在：`docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`。

## D-2026-05-10-02

- 日期：`2026-05-10`
- 决策：接受 `T18` review 的 `PASS`，标记 `T18` 完成，并将当前唯一任务切换为 `T19: Bounded cleanup manifest for tracked cache files`

### 背景

`T18` 已完成 `.tflite` export/runtime manifest 与 boundary smoke plan。`docs/review/T18_review.md` 给出 verdict：`PASS`，没有 blocking issue。

### Warning 分类

1. N1: 推荐表述中的 Markdown 引号嵌套不够整洁
   - 分类：`accepted`
   - 处理：只作为文档排版提醒，不影响当前结论，也不写入 risks

### 依据

1. `docs/evidence_packs/deployment_boundary/TFLite_runtime_bootstrap.md` 已完成，真实 `.tflite` runtime 不可用被明确写成阻塞事实。
2. `T18` 没有修改导出/runtime 代码，也没有改 benchmark 口径。
3. 当前更适合进入 repo hygiene 的只读 cleanup manifest 阶段。

### 结论

`T18` 可以标记完成。下一唯一任务为 `T19`，用于为已跟踪缓存/字节码文件制定有界 cleanup manifest。

### 直接影响

1. `docs/04_task_board.md` 切换当前唯一任务为 `T19`。
2. `docs/07_handoff.md` 补齐 `T18` review 判定与 `T19` 任务摘要。
3. `docs/06_repo_noise_governance.md`、`docs/08_risks_and_open_questions.md` 进入 cleanup manifest 口径。
4. `T19` 任务包已经存在：`docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`。

## D-2026-05-10-03

- 日期：`2026-05-10`
- 决策：接受 `T19` review 的 `PASS`，标记 `T19` 完成，并将当前唯一任务切换为 `T20: Real-board HIL readiness checklist without implementation claims`

### 背景

`T19` 已完成 tracked cache cleanup manifest。`docs/review/T19_review.md` 给出 verdict：`PASS`，没有 blocking issue。

### Warning 分类

1. N1: `docs/evidence_packs/repo_hygiene/cleanup_tracked_cache_manifest.md` 第 4.1 节 preflight 命令中的 glob 模式在 PowerShell 下可能存在 shell 展开差异
   - 分类：`accepted`
   - 处理：作为后续 cleanup 执行任务的命令写法注意；当前 T19 是只读 manifest，不影响结论
2. N2: tracked `.pyc` = `116` 与工作区 `.pyc` 总数 `133` 的差异未在 manifest 中显式解释
   - 分类：`accepted`
   - 处理：差异来自未跟踪/忽略缓存；当前任务只处理已跟踪文件，不影响 manifest 结论

### 依据

1. `docs/evidence_packs/repo_hygiene/cleanup_tracked_cache_manifest.md` 已完成：
   - tracked `.pyc` files = `116`
   - tracked `__pycache__` directories = `9`
   - tracked standalone `.pyc` outside `__pycache__` = `0`
   - cleanup command draft、rollback plan 与 acceptance criteria
2. Reviewer 独立验证清点数字全部匹配。
3. 本轮未执行 `git rm`，未删除文件，未触碰 `runs/` 或 `artifacts/`。
4. `T20` 任务包已经存在，且下一步应转向真板边界 readiness，而不是顺手执行物理 cleanup。

### 结论

`T19` 可以标记完成。下一唯一任务为 `T20`，用于为 real-board HIL 路径补 readiness checklist 与缺口清单。

`T20` 只能做只读 readiness 审计，不实现真板 backend，不调用硬件命令，也不能把 `board_backend.py` 的 placeholder 语义写成真实板级完成。

### 直接影响

1. `docs/04_task_board.md` 标记 `T19` 完成，并切换 `Current Unique Task` 到 `T20`。
2. `docs/07_handoff.md` 补齐 `T19` review 判定、warning 处理和 `T20` 任务摘要。
3. `docs/08_risks_and_open_questions.md` 记录 `T19` warning 分类，并明确 `T20` 只读边界。
4. 不执行物理 cleanup；如需 untrack 缓存文件，必须后续单开 cleanup 执行任务。

## D-2026-05-10-04

- 日期：`2026-05-10`
- 决策：接受 `T20` adversarial review 的 `PASS`，按 `PASS_WITH_WARNINGS` 收口 warning，标记 `T20` 完成，并将当前唯一任务切换为 `T21: Phase 2 milestone review and next-phase decision`

### 背景

`T20` 已完成 real-board HIL readiness checklist。`docs/review/T20_review.md` 给出 verdict：`PASS`，没有 blocking issue。

### Warning 分类

1. N1: `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md` 第 3.3 节寄存器名称来源不透明
   - 分类：`deferred`
   - 处理：写入 risks/open questions；后续真板执行任务如引用这些寄存器名，必须直接审计 `axi_map.py` / DMA 相关代码和 RTL 地址表
2. N2: 第 4 节验收标准缺少量化阈值
   - 分类：`deferred`
   - 处理：写入 risks/open questions；后续真板 smoke execution plan 必须补充平台/bitstream 相关 timeout、shape、epoch 变化等阈值
3. N3: 第 3.2 节权限描述偏 Linux
   - 分类：`deferred`
   - 处理：写入 risks/open questions；后续真板任务必须先确认目标主机是 Linux 还是 Windows，并按平台更新权限/driver 说明

### 依据

1. `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md` 已完成：
   - 当前 placeholder 证据
   - 真板任务前置条件
   - 设备/地址/权限/日志需求
   - Layer A-D 最小 smoke 验收标准
   - 禁止表述
2. Reviewer 独立验证 `board_backend.py` / `fpga_driver.py` 的 placeholder 证据全部匹配。
3. 本轮未修改 `cnn_fpga/`，未调用硬件命令，未把 mock/software HIL 外推为真板完成。
4. `T14` 至 `T20` 已完成 Phase 2 原任务队列，下一步应先做 milestone review，而不是直接进入真板 smoke 或新 benchmark。

### 结论

`T20` 可以标记完成，但其产物只是 readiness checklist，不是 real-board validation。

下一唯一任务为 `T21`。`T21` 只做 Phase 2 milestone review 和 next-phase decision，不运行 benchmark、不执行 cleanup、不调用硬件。

### 直接影响

1. `docs/04_task_board.md` 标记 `T20` 完成，并切换 `Current Unique Task` 到 `T21`。
2. 新增 `docs/tasks/Phase2/T21_phase2_milestone_review.md`。
3. `docs/07_handoff.md` 补齐 `T20` review 判定、warning 分类和 `T21` 任务摘要。
4. `docs/08_risks_and_open_questions.md` 保留 R13，并新增/更新后续真板执行任务必须处理的 N1/N2/N3 deferred 风险。

## D-2026-05-10-05

- 日期：`2026-05-10`
- 决策：`T21` 的 Phase 2 milestone gate 输出为 `Conditional`；项目继续保持在 `Phase 2: Controlled Development` / `Go`，但不升级当前证据为 formal benchmark、真实 `.tflite` runtime、physical cleanup 或 real-board validation

### 背景

`T14` 至 `T20` 已完成本轮 Phase 2 队列，覆盖了：

1. P4 benchmark evidence hardening
2. training / `.tflite` manifest
3. tracked cache cleanup manifest
4. real-board readiness checklist

在直接进入真板 smoke、physical cleanup 或新的 formal benchmark 前，需要先做一次 milestone review。

### 依据

1. `docs/review/T21_phase2_milestone_review.md` 已完成，并给出 `Conditional`。
2. `T15` 仍只是 `development_smoke`，不是 formal four-scenario frozen benchmark。
3. `T18` 只确认 `.tflite` 代码路径与 stub 边界，真实 runtime 仍不可用。
4. `T19` 只形成 cleanup manifest，尚未执行物理 cleanup。
5. `T20` 只形成 readiness checklist，尚未进入真板 smoke / validation。
6. `R13/R14` 表明真板路径仍缺平台、权限、地址表来源与量化阈值。

### 结论

当前最合理的 gate 判断是：

1. 允许继续留在 `Phase 2: Controlled Development`
2. 允许继续开 bounded 下一任务
3. 不允许把当前阶段升级写成：
   - formal P4 benchmark 已恢复
   - 真实 `.tflite` runtime 已恢复
   - physical cleanup 已完成
   - real-board HIL 已验证

### Captain 收口判断

`T21` 本身是 milestone review 工作，本轮不再启用重复 reviewer。Captain 接受该 review 的 `Conditional` gate 输出，并按 `PASS_WITH_WARNINGS` 收口：

1. `Conditional` gate：`accepted`
   - 处理：项目继续保持 `Phase 2: Controlled Development` / `Go`
2. formal P4 benchmark 未恢复：`deferred`
   - 处理：继续保留 R5/R9，不升级 `T15` 证据等级
3. 真实 `.tflite` runtime 未恢复：`deferred`
   - 处理：继续保留 R12
4. physical cleanup 未执行：`deferred`
   - 处理：继续保留 R7
5. real-board HIL 未验证：`deferred`
   - 处理：继续保留 R13/R14

### 直接影响

1. `docs/04_task_board.md` 标记 `T21` 完成，并切换 `Current Unique Task` 到 `T22`。
2. 新增 `docs/tasks/Phase2/T22_real_board_smoke_execution_plan.md`。
3. `docs/07_handoff.md` 记录 `T21` 的 gate 输出、Captain 收口判断与 `T22` 任务摘要。
4. `docs/08_risks_and_open_questions.md` 保留 milestone gate 风险与下一任务边界。

## D-2026-05-10-06

- 日期：`2026-05-10`
- 决策：接受 `T22` adversarial review 的 `PASS_WITH_WARNINGS`，标记 `T22` 完成，并将当前唯一任务切换为 `T23: Paper claims evidence roadmap toward publishable results`

### 背景

`T22` 已产出 `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`，将 `T20` 后遗留的真板前置模糊点具体化：

1. host platform decision points：Linux / Windows / WSL / remote board host
2. AXI/register map 审计清单，直接对应 `cnn_fpga/hwio/axi_map.py`
3. DMA buffer 审计清单，直接对应 `cnn_fpga/hwio/dma_client.py`
4. Layer A-D 量化阈值草案和 fail-fast budget
5. future evidence pack 与 prohibited wording

`docs/review/T22_review.md` 给出 `PASS_WITH_WARNINGS`，blocking issues 为无。

### Warning 分类

1. N1：5 个 out-of-scope 文件被修改
   - 分类：`accepted`
   - Captain 判断：这些修改属于 Captain 在 `T21` gate / `T22` 初始化期间对 00/01/05/06 与 T21 任务包做的治理同步，不归为 T22 Worker 越界。后续任务包会更明确区分 Worker allowed files 与 Captain 整合文件。
2. N2：`AXI_REGISTER_MAP` preflight 直接 `print(...)` 只会输出 dataclass repr
   - 分类：`deferred`
   - 处理：写入后续执行风险。若未来进入真板执行任务，preflight 应输出格式化地址表，而不是只依赖 repr。
3. N3：`byte_count = 4096` 依赖 `32 x 32 float32` histogram 假设
   - 分类：`deferred`
   - 处理：写入后续执行风险。若未来进入真板执行任务，必须用实际 bitstream / DMA contract 确认 histogram shape 与 element dtype。

### 结论

1. `T22` 完成，但它仍只是 execution plan，不是 hardware validation。
2. `R13/R14` 继续有效，需在真实硬件任务中用设备、权限、寄存器活性、DMA 和 commit/ack 证据关闭。
3. 当前不应直接启动真板、formal benchmark、`.tflite` runtime 或 cleanup 执行任务。
4. 为了以高质量论文为目标继续推进，下一步优先建立论文 claim / evidence / gap / figure / task roadmap。

### 直接影响

1. `docs/04_task_board.md` 标记 `T22` 完成，并切换 `Current Unique Task` 到 `T23`。
2. 新增 `docs/tasks/Phase2/T23_paper_claims_evidence_roadmap.md`。
3. `docs/07_handoff.md` 记录 `T22` 的 review 判定、warning 分类与 `T23` 任务摘要。
4. `docs/08_risks_and_open_questions.md` 保留真板执行风险，并新增论文 claim 过度外推风险。

## D-2026-05-10-07

- 日期：`2026-05-10`
- 决策：根据 Project Manager 反馈，撤回“最近任务直接转向 paper claims roadmap”的安排；保留“发表论文”为远期目标，但把当前唯一任务改为 `T23: P4 formal benchmark protocol lock and evidence gap audit`

### 背景

Project Manager 明确指出：论文发表是最终目标，但当前仍应一步步推进，不应把最近任务直接安排成论文 claim / evidence roadmap。Captain 重新阅读 00~08 治理文档和 `docs/reference/AI_coding_workflow.md` 后，确认当前最关键缺口仍是证据等级，而不是论文写作本身。

### 依据

1. `docs/06_repo_noise_governance.md` 中 formal benchmark 定义要求预先冻结 protocol、baseline、seed、repeat，并通过 review。
2. `docs/review/T21_phase2_milestone_review.md` 已确认 `T15` 仍为 `development_smoke`，不能升级成 formal benchmark。
3. `docs/review/T22_review.md` 已确认 real-board execution plan 通过，但它不是 hardware validation。
4. `docs/02_experiment_plan.md` 的路线建议仍把 formal HIL / P4 paired benchmark 作为模型与论文结论的关键证据。

### 结论

1. 远期目标：形成可投稿论文所需的可信工程与实验链。
2. 中期路线：按 benchmark formalization、机制证据、复现/部署边界、真板 gate、论文收口逐步推进。
3. 当前唯一任务：`T23` 只锁定 P4 formal benchmark protocol，不运行 benchmark。
4. 后续 paper claim/evidence ledger 推迟到 formal P4 和机制证据更清楚之后。

### 直接影响

1. 删除/替换原 `docs/tasks/Phase2/T23_paper_claims_evidence_roadmap.md`。
2. 新增 `docs/tasks/Phase2/T23_p4_formal_benchmark_protocol_lock.md`。
3. `docs/04_task_board.md` 重写 T23 及 T24+ 后续大纲。
4. `docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步当前任务和风险口径。

## D-2026-05-10-08

- 日期：`2026-05-10`
- 决策：阅读 `docs/reference/进一步的深度研究结果.md` 与 `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md` 后，维持 `T23` 当前大方向，但增强 `T23` 的输入与验收要求，并小幅调整后续 pending roadmap

### 背景

新深度研究报告给出的关键结论是：

1. 目前未发现与“GKP syndrome histogram -> CNN slow loop -> FPGA-friendly linear fast path parameter update”完全同构的工作。
2. 但相邻方向已很接近，包括 GKP soft-information decoding、hardware-conditioned neural decoder、real-time/FPGA QEC decoder、drift-aware calibration / prior update。
3. 项目叙事应收窄到 teacher-anchored residual calibration / histogram-conditioned slow loop / fixed-point linear fast path / atomic commit / software-HIL-to-runtime boundary。
4. 后续优先级应是 formal benchmark protocol、机制诊断、statcalib baseline、true `.tflite` runtime，再考虑 real-board smoke。

### 判断

这份报告不要求推翻当前 `T23`。相反，它支持当前“先锁 P4 formal benchmark protocol”的方向。

需要调整的是任务细节：

1. `T23` 必须读取该深度研究报告和 paper-inspired 草案。
2. `T23` 的 protocol lock 必须显式评估：
   - strong classical / soft-information / calibration / learned baseline classes
   - static / drift / random-walk / sinusoidal / burst-reset scenario families
   - training-seed 与 evaluation-seed 分离
   - confidence interval 或 stopping rule
   - latency / commit / rollback / fallback metrics
   - statcalib baseline 是否必须先实现
   - true `.tflite` runtime 为什么应优先于 real-board smoke 支撑 deployment claim
3. 若 formal execution 条件未满足，`T23` 应建议 prerequisite closure task，而不是强行进入 `T24` execution。

### 直接影响

1. `docs/tasks/Phase2/T23_p4_formal_benchmark_protocol_lock.md` 增加深度研究报告和 paper-inspired 草案作为输入。
2. `docs/04_task_board.md` 保持当前唯一任务为 `T23`，但增强 expected output，并将 `T24` 改为由 T23 gate 决定“先补 prerequisite 还是执行”。
3. `docs/04_task_board.md` 的后续 pending roadmap 增加 calibration/statcalib baseline feasibility gate。
4. `docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步该判断。

## D-2026-05-10-09

- 日期：`2026-05-10`
- 决策：接受 `T23` adversarial review 的 `PASS_WITH_WARNINGS`，标记 `T23` 完成，并将当前唯一任务切换为 `T24: P4 bounded formal software revalidation execution`

### 背景

`T23` 已产出 `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`，锁定了下一步 P4 formal software revalidation 的证据等级、frozen matrix、baseline 集合、seed/repeat、统计报告、compute budget、evidence pack 与 go/no-go 条件。

`docs/review/T23_review.md` 给出 `PASS_WITH_WARNINGS`，blocking issues 为无。

### Warning 分类

1. N1：7 个文件超出 T23 Worker allowed list
   - 分类：`accepted`
   - Captain 判断：这些修改属于治理同步，与 T22 相同，不改变 T23 技术结论；后续任务包继续更明确区分 Worker allowed files 与 Captain 整合文件
2. N2：T23 protocol 未写出 T24 exact CLI shape
   - 分类：`deferred`
   - 处理：写入 risks，并在 `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md` 中固定 repeat-chunked CLI shape
3. N3：`histogram_input_saturation_rate_mean` 与 `correction_saturation_rate_mean` 未被 reviewer 逐项验证
   - 分类：`deferred`
   - 处理：写入 risks；T24 必须报告这些字段是否实际出现在 runner 输出中
4. N4：`fast_cycle_violation_rate_mean` 未被 reviewer 逐项验证
   - 分类：`deferred`
   - 处理：写入 risks；T24 必须报告该字段是否实际出现在 runner 输出中

### 依据

1. `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` 明确写出 `T23 did not run benchmark`。
2. Protocol 已把 `T24` gate 锁定为：
   - `GO_FOR_BOUNDED_FORMAL_SOFTWARE_REVALIDATION`
   - `NO_GO_FOR_SCOPE_EXPANSION_INSIDE_T24`
3. Frozen execution scope 为：
   - `static_bias_theta / linear_ramp / step_sigma_theta / periodic_drift`
   - `ekf / ukf / constant_residual_mu / rls_residual_b / hybrid_residual_b`
   - `paired_seeds`
   - `repeats=2`
4. `statcalib`、soft-information comparator、额外 drift family、CI-driven stopping、真实 `.tflite` runtime 与真板 smoke 均已被明确排除出 T24。

### 结论

`T23` 可以标记完成。下一唯一任务为 `T24`，用于执行 bounded formal software revalidation。

`T24` 可以运行 P4 benchmark，但只能在 mock-backed software HIL 边界内运行 frozen set；不得改源码、改 config、改 benchmark 语义或夹带部署/真板/机制扩展任务。

### 直接影响

1. `docs/04_task_board.md` 标记 `T23` 完成，并切换 `Current Unique Task` 到 `T24`。
2. 新增 `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`。
3. `docs/07_handoff.md` 记录 `T23` review 判定、warning 分类与 `T24` 任务摘要。
4. `docs/08_risks_and_open_questions.md` 新增 T24 exact CLI / metric availability 风险。

## D-2026-05-11-01

- 日期：`2026-05-11`
- 决策：接受 `T24` adversarial review 的 `PASS_WITH_WARNINGS`，标记 `T24` 完成，并将当前唯一任务切换为 `T25: P4 formal evidence gate review and result-boundary update`

### 背景

`T24` 已完成 `4 scenarios x 5 modes x repeats=2` frozen-set formal software revalidation。`docs/review/T24_review.md` 给出 `PASS_WITH_WARNINGS`，blocking issues 为无。

已验证事实：

1. Run dir：`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
2. `missing_runs = []`
3. 20/20 scenario/mode rows `coverage = 1.0`
4. 40 repeat-runs completed
5. 四场景 winner 均为 `hybrid_residual_b`，runner-up 均为 `ukf`
6. 结果仍为 `mock-backed` software HIL only

### Warning 分类

1. N1：`correction_saturation_rate_mean` 在所有 20 行中结构性为 0.0
   - 分类：`deferred`
   - 处理：写入 R20；后续机制审计需判断是 metric collection bug、真实零值，还是当前参数区间 not applicable
2. N2：`docs/04_task_board.md` 中出现一行环境提示，略超出 T24 execution result 口径
   - 分类：`accepted`
   - 处理：视为 Captain 治理同步提示，不影响 T24 结果，也不写入风险
3. N3：`teacher_scalar_diagnostics.csv` 只有 header，teacher diagnostics 全零
   - 分类：`deferred`
   - 处理：继续归入 R10；T25 后应优先安排 `T27` 或等价机制证据审计

### 结论

`T24` 可以标记完成，证据等级为 frozen-set formal software revalidation，但边界必须保持为 `mock-backed software HIL`。

不得将 T24 外推为 `.tflite` runtime、`real_board` validation、paper-grade expanded benchmark、statcalib comparator 结果或 CI-driven stopping 结果。

### 直接影响

1. `docs/04_task_board.md` 记录 T24 Captain verdict，并将 `Current Unique Task` 固定为 `T25`。
2. 新增 `docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md`。
3. `docs/07_handoff.md` 记录 T24 review 判定、warning 分类与 T25 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 R10 并新增 R20。

## D-2026-05-11-02

- 日期：`2026-05-11`
- 决策：记录 `T25` Worker gate-review draft 已产出，但在 Captain 手动收口前，不标记 `T25` 完成，也不切换当前唯一任务

### 背景

`T25` 的目标是对 `T24` evidence pack 做只读 adversarial gate review，并更新结果边界。用户已明确要求本轮不要直接标记 task 已结束，因为后续还会手动交给 claude code 审核。

### Worker draft 结论

1. `docs/review/T25_p4_formal_evidence_gate_review.md` 已形成草案，verdict = `PASS_WITH_WARNINGS`
2. `T24` 可被视为 completed frozen-set formal software revalidation
3. 该结论仍严格限定为 `mock-backed software HIL only`
4. `correction_saturation_rate_mean` structural zero 继续 `deferred` 到 `R20`
5. `teacher_scalar_diagnostics.csv` header-only / teacher diagnostics 全零继续 `deferred` 到 `R10`
6. `docs/04_task_board.md` 中 T24 task-board environment-note warning 继续 `accepted`
7. Worker 推荐的下一唯一任务是 `T27: Teacher diagnostics path audit and mechanism-evidence repair plan`

### 结论

当前最合理做法是：

1. 保留 `Current Unique Task = T25`
2. 把 T25 review draft、for-human explanation 和治理同步写入仓库
3. 不提前把 `T27` 写成已开始
4. 等 Captain 手动审阅和收口后，再决定是否正式切换下一唯一任务

### 直接影响

1. `docs/review/T25_p4_formal_evidence_gate_review.md` 新增 gate-review draft。
2. `docs/for_human/T25_explanation.md` 新增给人读的简明说明。
3. `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 同步 T25 draft 结论，但保持当前唯一任务不变。
4. `docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md` 需要补 Worker output，记录本轮只读检查和剩余风险。

## D-2026-05-11-03

- 日期：`2026-05-11`
- 决策：Captain 接受 `T25` gate review 的 `PASS_WITH_WARNINGS`，标记 `T25` 完成，并将当前唯一任务切换为 `T27: Teacher diagnostics path audit and mechanism-evidence repair plan`

### 背景

`T25` 本身是只读 review / gate 任务。用户已说明本轮不启用 Claude Code 重复 review，因此 Captain 直接按项目治理规则对 `docs/review/T25_p4_formal_evidence_gate_review.md` 做收口判断。

### Warning 分类

1. N1：`correction_saturation_rate_mean` 在 T24 所有 20 个 scenario/mode rows 中结构性为 `0.0`
   - 分类：`deferred`
   - 处理：继续挂 R20；后续机制审计需判断这是 metric collection dead path、真实零值，还是当前参数区间 not applicable
2. N2：T24 task-board environment-note warning
   - 分类：`accepted`
   - 处理：归为 Captain 治理同步提示，不影响 T24/T25 结论，不写入新风险
3. N3：`teacher_scalar_diagnostics.csv` header-only / teacher diagnostics 全零
   - 分类：`deferred`
   - 处理：继续挂 R10；优先进入 `T27` 做路径审计和机制证据修复计划

### 结论

`T25` 可以标记完成。T24 可作为 completed frozen-set formal software revalidation，但该证据等级只在 `mock-backed software HIL` 边界内成立。

不得将 T24/T25 外推为真实 `.tflite` runtime、`real_board` validation、paper-grade expanded benchmark、statcalib comparator 结果或 CI-driven stopping 结果。

下一唯一任务不切到 `T26`。虽然 `T26` 仍是路线图中的 pending 项，但 T25 gate 推荐先处理 deferred 链最长的 R10 teacher diagnostics 问题；Captain 接受该推荐，将当前唯一任务切换为 `T27`。

### 直接影响

1. `docs/04_task_board.md` 标记 `T25` 完成，并切换 `Current Unique Task` 到 `T27`。
2. 新增 `docs/tasks/Phase2/T27_teacher_diagnostics_path_audit.md`。
3. `docs/07_handoff.md` 记录 T25 Captain verdict、warning 分类与 T27 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 T25 状态、R10/R20 后续路径与 T26/T27 优先级判断。

## D-2026-05-11-04

- 日期：`2026-05-11`
- 决策：接受 `T27` teacher diagnostics path audit 的 `PASS_WITH_WARNINGS`，标记 `T27` 完成，并将当前唯一任务切换为 `T28: Teacher diagnostics missing-vs-zero semantics repair and minimal smoke`

### 背景

`T27` 对 `T15/T24` 的 teacher diagnostics 全零、`teacher_scalar_diagnostics.csv` header-only、以及 `correction_saturation_rate_mean` structural zero 做了只读路径审计。该任务未运行 benchmark、训练、`.tflite`、硬件或 cleanup，也未修改源码、config、run 或 artifact。

### Warning 分类

1. W1：`R10` 的主因已定位但未修复
   - 分类：`deferred`
   - 处理：继续挂 R10；T28 需要修复 missing-vs-zero 语义，并明确 broadcast teacher path 与 scalar diagnostics 的支持边界
2. W2：P4 writer / aggregation 层把缺失 teacher diagnostics 压成 `0.0`
   - 分类：`deferred`
   - 处理：新增/并入 R21；后续不得再把 `not generated` 静默写成 `true zero`
3. W3：`R20` 不共享 teacher diagnostics 死路径，当前 T24 零值更像当前参数区间下未触发 correction saturation
   - 分类：`accepted`
   - 处理：R20 从“疑似 metric dead path”缩窄为“独立 fast-loop path，当前参数区间未触发”；但不关闭全局 stress/edge 触发性问题

### 结论

`T27` 可以标记完成。它把 R10 从泛泛的“teacher diagnostics 全零”缩窄为：当前 `hybrid_residual_b` frozen config 使用 broadcast teacher features，而 `tiny_cnn.py::explain_from_loaded_artifact()` 只在 `scalar_feature_dim > 0` 时产出 teacher scalar diagnostics，因此当前 hybrid path 的 teacher diagnostics 是 `data not generated`。

`T27` 同时确认 R20 不与 teacher diagnostics 共享同一条空路径；`correction_saturation_rate_mean` 来自独立 fast-loop saturation counter。当前 T24 的 `0.0` 不应再简单归类为指标未写出，但也不能外推为所有参数区间都不会触发 saturation。

下一唯一任务不切到 T26/statcalib。当前必须先修复 teacher diagnostics 的可观察性和缺失语义，否则 statcalib、论文 claim 或机制解释都会继续继承混淆指标。

### 直接影响

1. `docs/04_task_board.md` 标记 `T27` 完成，并切换 `Current Unique Task` 到 `T28`。
2. 新增 `docs/tasks/Phase2/T28_teacher_diagnostics_semantics_repair.md`。
3. `docs/07_handoff.md` 记录 T27 Captain verdict、warning 分类与 T28 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 R10/R20，并新增 R21 记录 missing-vs-zero 语义风险。

## D-2026-05-12-01

- 日期：`2026-05-12`
- 决策：接受 `T28` independent review 的 `PASS_WITH_WARNINGS`，标记 `T28` 完成，并将当前唯一任务切换为 `T29: P4 markdown report header cleanup after T28`

### 背景

`T28` 已完成 teacher diagnostics missing-vs-zero 语义修复与最小 smoke。`docs/review/T28_review.md` 给出 `PASS_WITH_WARNINGS`，blocking issues 为无。Reviewer 确认：

1. `comparison.csv` 已显式区分 `not_applicable` 与 `not_generated`。
2. 缺失 teacher diagnostics 不再被机器可读输出静默压成 `0.0`。
3. `correction_saturation_rate_mean = 0.0` 仍保留为独立 fast-loop 指标的真实观察零值。
4. T24 历史 run 未被改写。
5. T28 smoke 输出隔离在 `runs/p4_benchmark/T28_teacher_diag_semantics_smoke_manual_20260511`。

### Warning 分类

1. N1：`_write_report()` 中 markdown report 旧 header 未删除，导致两行 header 列数不一致
   - 分类：`deferred`
   - 处理：写入 R22，并新开 `T29` 做一行级报告格式修复
2. N2：tracked `.pyc` 文件因执行产生 side-effect diff
   - 分类：`rejected as technical signal`
   - 处理：不作为有意义技术改动提交；按 `T19` / `docs/06_repo_noise_governance.md` 的 tracked-cache 口径排除或另行 cleanup
3. N3：`comparison.csv` column order changed
   - 分类：`accepted`
   - 处理：这是 T28 missing-vs-zero 语义修复的预期接口变化；后续 consumer 应按列名而非位置解析
4. Missing focused tests
   - 分类：`deferred`
   - 处理：写入 R23；T28 smoke 对本轮可接受，但后续再改 aggregation/report writer 时应补 focused unit test 或静态格式测试
5. S1/S2/S3 suspicious implementation details
   - 分类：`accepted`
   - 处理：均符合 T28 语义修复目标；未来若启用 scalar-branch generated path，再重新审查 `teacher_contribution_vector` 行为

### 结论

`T28` 可以标记完成。`R21` 对当前 writer 语义可关闭：当前输出已经能区分 `not_generated` / `not_applicable` / observed zero。`R10` 进一步缩窄但不关闭，因为这只是可观察性和输出语义修复，不是完整 teacher mechanism evidence repair。

下一唯一任务不切到 `T26` 或 `T36`。应先执行 `T29`，修复 T28 review 指出的 markdown report 重复表头，避免后续机制分析或 statcalib 任务继承破损的人读 report。

### 直接影响

1. `docs/04_task_board.md` 标记 `T28` 完成，并切换 `Current Unique Task` 到 `T29`。
2. 新增 `docs/tasks/Phase2/T29_p4_report_header_cleanup.md`。
3. `docs/07_handoff.md` 记录 T28 Captain verdict、warning 分类与 T29 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 R10/R21，并新增 R22/R23。

## D-2026-05-12-02

- 日期：`2026-05-12`
- 决策：接受 `T29` independent review 的 `PASS`，标记 `T29` 完成，并将当前唯一任务切换为 `T26: Calibration/statcalib baseline feasibility gate and minimal design plan`

### 背景

`T29` 只修复 `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py::_write_report()` 中重复旧 markdown header 的问题。`docs/review/T29_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. 代码 diff 是一行删除旧 11-column header。
2. 新 header 保留 `Teacher Diag`，header / separator / data row 均为 12 列。
3. 未运行 benchmark，未新增 run dir，未改变 teacher diagnostics 语义、CSV columns、aggregation、baseline/scenario、seed policy 或 formal protocol。

### Warning 分类

1. N1：tracked `.pyc` side-effect
   - 分类：`accepted as known repo-noise side effect / rejected as technical signal`
   - 处理：不写入新风险，不作为有意义技术改动提交；继续按 `T19` / `docs/06_repo_noise_governance.md` 的 tracked-cache 口径处理。

### 结论

`T29` 可标记完成。`R22` 可收口：T28 后遗留的人读 P4 markdown report 重复表头问题已修复，并通过最小格式验证。

下一唯一任务切换到 `T26`，原因是 T24/T25 已完成 frozen-set formal software revalidation 边界收口，T27/T28/T29 已先处理 teacher diagnostics 可观察性和 report 格式问题；现在可以做 calibration/statcalib baseline 的 feasibility gate。T26 仍是只读/设计门控任务，不实现 comparator、不运行 benchmark、不扩展 formal benchmark。

### 直接影响

1. `docs/04_task_board.md` 标记 `T29` 完成，并切换 `Current Unique Task` 到 `T26`。
2. 新增 `docs/tasks/Phase2/T26_statcalib_feasibility_gate.md`。
3. `docs/07_handoff.md` 记录 T29 Captain verdict、warning 分类与 T26 任务摘要。
4. `docs/08_risks_and_open_questions.md` 将 R22 标为已收口，并更新当前开放问题与下一任务口径。

## D-2026-05-12-03

- 日期：`2026-05-12`
- 决策：接受 `T26` independent review 的 `PASS`，标记 `T26` 完成，并将当前唯一任务切换为 `T30: Statcalib comparator interface contract and bounded implementation package`

### 背景

`T26` 是 docs-only/read-only feasibility gate。`docs/review/T26_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md` 已形成，并给出 `CONDITIONAL_GO`。
2. statcalib 当前仍未实现、未验证，不能从 T24-T29 结果中推断为已有 evidence。
3. statcalib 只能作为 separate comparator lane 推进，不得静默插入 T24 frozen benchmark set。
4. 本轮未修改 source/config/run/artifact，未运行 benchmark，未新增 run dir。

### Non-blocking comments

1. Worker self-review doc 较薄
   - 分类：`accepted`
   - 处理：对 docs-only gate 可接受；后续 implementation task 必须给出更完整 audit trail
2. For-human doc 很简短
   - 分类：`accepted`
   - 处理：不阻塞 T26；后续如进入实现/验证，应补更清晰的人读解释
3. `StatCalibInput` / `StatCalibOutput` 仍是概念级接口
   - 分类：`accepted as follow-up constraint`
   - 处理：写入 T30 任务包，要求先收紧 exact field names / types / status semantics

### 结论

`T26` 可标记完成。Gate verdict = `CONDITIONAL_GO`：允许后续做最小 statcalib comparator lane，但前提是保持 separate comparator boundary，不改 frozen benchmark semantics。

下一唯一任务切换到 `T30`。`T30` 不直接启动长跑、不扩展 formal benchmark、不触碰 `.tflite` 或真板范围；它只负责把 T26 的概念设计收紧为具体接口契约、最小实现边界和可审查的 bounded implementation package。

### 直接影响

1. `docs/04_task_board.md` 标记 `T26` 完成，并切换 `Current Unique Task` 到 `T30`。
2. 新增 `docs/tasks/Phase2/T30_statcalib_interface_contract.md`。
3. `docs/07_handoff.md` 记录 T26 Captain verdict、non-blocking comments 与 T30 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 R18 与当前开放问题，明确 statcalib 后续仍不得改写 frozen benchmark 边界。

## D-2026-05-13-01

- 日期：`2026-05-13`
- 决策：接受 `T30` independent adversarial review 的 `PASS`，标记 `T30` 完成，并将当前唯一任务切换为 `T36: seed=20260429 failure-mechanism diagnosis, bounded no-new-branch scope`

### 背景

`T30` 是 statcalib comparator 的接口契约与最小实现边界任务。`docs/review/T30_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. 新增 `cnn_fpga/decoder/statcalib.py`，定义 `StatCalibInput` / `StatCalibOutput`、status/reason、provenance 和 `DecoderRuntimeParams` conversion boundary。
2. 新增 `tests/test_statcalib_interface.py`，覆盖 generated / not_generated / not_applicable / invalid input 等关键接口语义。
3. `ParamMapper`、`SlowLoopRuntime`、P4 benchmark runner、config、formal protocol、baseline/scenario/seed/repeat policy 均未改变。
4. T30 未新增 benchmark run dir，未把 statcalib 加入 frozen ranked set，未触碰 `.tflite` 或真板范围。

### Warning 分类

1. N1：`docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md` 的 T26 非声明仍写着 no source code changed，T30 后已过期
   - 分类：`accepted`
   - 处理：Captain closeout 已在 gate 文档中按 T26/T30 时点修正，明确 T30 仅新增 interface-only source/tests，不等于 benchmark validation
2. N2：`tests/` 目录无 `__init__.py`
   - 分类：`accepted`
   - 处理：当前 `python -m unittest tests.test_statcalib_interface` 可发现并通过；若测试目录增长，再单开测试布局整理
3. N3：测试产生 `tests/__pycache__` side-effect
   - 分类：`rejected as technical signal`
   - 处理：按 T19/T28/T29 repo-noise 口径，不作为有意义技术改动提交
4. N4：`from_delta_b()` 使用 `prior.b + delta_b`，是最小 residual-b baseline
   - 分类：`deferred`
   - 处理：写入 R24；未来 statcalib integration/calibration logic 必须重新验证，不能把当前接口 helper 外推为完整 calibration comparator

### 结论

`T30` 可标记完成。它完成的是 interface-level contract 和 focused tests，不是 slow-loop runtime integration、formal benchmark evidence、`.tflite` runtime 或 real-board validation。

下一唯一任务切换到 `T36`。理由是 `docs/02_experiment_plan.md` 已把 `seed=20260429` 的 Gated v5 收益收缩列为第一优先机制诊断；当前可以只读取既有结果做 bounded diagnosis，不需要重跑 benchmark 或扩新分支。

### 直接影响

1. `docs/04_task_board.md` 标记 `T30` 完成，并切换 `Current Unique Task` 到 `T36`。
2. 新增 `docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md`。
3. `docs/07_handoff.md` 记录 T30 Captain verdict、warning 分类与 T36 任务摘要。
4. `docs/08_risks_and_open_questions.md` 新增 R24，并更新当前开放问题与下一任务口径。

## D-2026-05-13-02

- 日期：`2026-05-13`
- 决策：接受 `T36` adversarial review 的 `PASS`，标记 `T36` 完成，并将当前唯一任务切换为 `T38: seed=20260429 single-seed trace-export probe, bounded unchanged-semantics rerun`

### 背景

`T36` 是只读 failure-mechanism diagnosis。`docs/review/T36_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md` 已包含 artifact inventory、Full vs Gated v5 scenario summary、跨 seed comparison、机制矩阵和下一 bounded 建议。
2. `cnn_fpga/benchmark/analyze_seed20260429_failure.py` 是标准库 CSV/JSON 读取脚本，只打印 deterministic JSON，不导入 project runtime、不运行 simulation。
3. T36 未重跑 benchmark、未新增/改写 `runs/` 或 `artifacts/`、未改模型/config/formal protocol/baseline/scenario/seed policy。
4. 诊断没有把 hypothesis 写成因果证明，也没有改变 `.tflite`、real-board 或 statcalib 边界。

### Warning 分类

1. N1：analysis script 中 `Iterable` import 未使用
   - 分类：`accepted`
   - 处理：cosmetic，不要求返工
2. N2：script 中 scenario/mode folder mappings hardcoded
   - 分类：`accepted`
   - 处理：该脚本只面向既有 frozen artifacts；若未来做 reusable analysis tool，再要求动态发现
3. N3：Worker pre-review file 与 adversarial review 同名并被覆盖
   - 分类：`accepted`
   - 处理：任务包保留 Worker Output / Verification Record；review 文件作为 reviewer 输出即可

### 结论

`T36` 可标记完成。它把 `seed=20260429` 的 Gated-v5 收益收缩缩窄为 residual-amplitude / teacher-delta regime instability hypothesis；同时排除了 response lag、overflow/correction saturation、dead teacher branch 作为当前 artifacts 支持的主因。

但 T36 仍不是 causal proof。现有 artifacts 缺少 per-window / per-commit `teacher_b`、predicted `delta_b`、committed `b` trace，因此 sign offset、overshoot chronology、teacher-vs-CNN-vs-committed-b attribution 仍不能回答。

下一唯一任务切换到 `T38`。T38 是 T36 推荐的最小后续：允许一个 T38-scoped single-seed trace-export probe，但必须保持 benchmark 语义、baseline/scenario、seed/repeat policy 不变；不得扩新分支、不得改 formal benchmark、不得触碰 `.tflite` 或真板路径。

### 直接影响

1. `docs/04_task_board.md` 标记 `T36` 完成，并切换 `Current Unique Task` 到 `T38`。
2. 新增 `docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md`。
3. `docs/07_handoff.md` 记录 T36 Captain verdict、warning 分类与 T38 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 R10 和当前开放问题，明确 T38 只补 trace 证据缺口，不扩大 benchmark。

## D-2026-05-16-01

- 日期：`2026-05-16`
- 决策：接受 `T38` adversarial review 的 `PASS`，标记 `T38` 完成；完成 Milestone 2I comprehensive review，结论为 `Conditional Allow`；将当前唯一任务切换为 `T31: Training-chain portable dependency lock plan`

### 背景

`T38` 是 `seed=20260429` single-seed trace-export probe。`docs/review/T38_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. Required trace fields are present for `4798 / 4798` trace rows.
2. Required report sections are present in `docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md`.
3. Scope stayed bounded to unchanged benchmark semantics and a T38-scoped run root.
4. T38 did not expand baseline/scenario/seed policy and did not touch `.tflite`, real-board, or statcalib integration boundaries.

### Warning 分类

1. N1 unused imports in `cnn_fpga/benchmark/analyze_seed20260429_trace.py`: `accepted` as cosmetic.
2. N2 task package wording `missing_runs = 0` versus JSON `missing_runs: []`: `accepted`; semantically equivalent, but future task packages should state exact data format.
3. N3 report does not explicitly explain `max_abs_delta_b=0.169705627 = sqrt(2) * 0.12`: `accepted`; clarity improvement only.
4. N4 initial tool timeout followed by successful resume in the same run dir: `accepted`; final evidence has `missing_runs=[]`, `raw_rows=16`, `comparison_rows=8`.

No T38 warnings are `deferred`; no new risk entry is required from warning classification.

### Milestone 2I Review

`docs/review/Milestone2I_review.md` records the comprehensive milestone review. Verdict = `Conditional Allow`.

Milestone 2I is complete within its bounded scope: mechanism-evidence hardening and trace-level diagnosis improved confidence, but it does not close clean-environment reproducibility, mitigation, multi-seed confirmation, true `.tflite` runtime validation, or real-board validation.

### 结论

`T38` is complete and accepted as `PASS`. `R10` is narrowed but remains open. The next unique task is `T31`, because Milestone 2I review identifies training-chain portability and clean-environment reproducibility as the most appropriate next bounded milestone entry.

### 直接影响

1. `docs/04_task_board.md` marks `T38` complete and switches `Current Unique Task` to `T31`.
2. `docs/review/Milestone2I_review.md` is added as the milestone review output.
3. `docs/tasks/Phase2/T31_training_chain_portable_dependency_lock_plan.md` is added as the next task package.
4. `docs/07_handoff.md` and `docs/08_risks_and_open_questions.md` are updated to preserve the T38 warning classification and T31 execution boundary.

## D-2026-05-17-01

- 日期：`2026-05-17`
- 决策：接受 `T31` adversarial review 的 `PASS`，标记 `T31` 完成，并将当前唯一任务切换为 `T39: Training-chain CPU-only clean-environment draft lock and dry-run bootstrap`

### 背景

`T31` 是 Milestone 2J 的 training-chain portable dependency-lock planning task。`docs/review/T31_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `docs/evidence_packs/training_reproducibility/training_chain_portable_dependency_lock_plan.md` 覆盖了 T31 要求的 8 类输出：本机解释器 inventory、package evidence、static-theta / residual-b / Gated-v5 dependency map、CPU-vs-GPU lock strategy、可提交内容与 local-only evidence、clean-environment bootstrap proposal、explicit non-claims、下一 bounded task 建议。
2. 本轮只修改 T31 allowed files，未修改 source/config/protocol/baseline/scenario/seed policy。
3. 未安装、升级或删除依赖，未运行训练、benchmark、`.tflite`、硬件或 cleanup。
4. 未创建或修改 `runs/` / `artifacts/`，也未 repurpose `requirements-recovery.txt`。
5. T31 文档明确区分 local `DLEnv` / CUDA / dev-torch facts 与 portable guarantees。

### Warning 分类

1. N1：`docs/evidence_packs/training_reproducibility/training_chain_portable_dependency_lock_plan.md` 的 subsection heading 应为 `### 4.1` 等，而不是 `## 4.1`
   - 分类：`accepted`
   - 处理：cosmetic only，不影响 T31 结论；后续文档整理时可修正。
2. N2：`docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md` 仍把 `DLEnv` 写作训练推荐解释器，而 T31 形成了更细的 two-lane view
   - 分类：`accepted`
   - 处理：未来 alignment 可让 bootstrap doc 引用 T31 two-lane plan；不属于 T31 blocking scope。
3. N3：Worker self-review file 被 adversarial review 覆盖
   - 分类：`accepted`
   - 处理：符合当前 reviewer 输出流程；任务包保留 Worker Output / Verification Record 即可。

没有 T31 `deferred` warning；不因 T31 review 新增 risk。

### 结论

`T31` 可标记完成。它把训练链环境状态从“本机 bootstrap notes + dev torch facts”推进为“可审计的 portable dependency-lock plan”，但仍不等于 clean-environment rebuild proof，也不等于 full training reproducibility。

下一唯一任务切换到 `T39`。理由是 T31 已确认当前训练配置默认不强制 `torch` / CUDA，CPU-only training dependency lane 是最低风险的可复现性后续。T39 只允许创建或使用 clean Python `3.12` CPU-only 环境、产出 draft lock/spec，并运行 dry-run/import-level entrypoint checks；不得运行训练、benchmark、`.tflite`、真板、cleanup 或 GPU portability work。

### 直接影响

1. `docs/04_task_board.md` 标记 `T31` 完成，并切换 `Current Unique Task` 到 `T39`。
2. 新增 `docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md`。
3. `docs/07_handoff.md` 记录 T31 Captain verdict、warning 分类与 T39 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 R11：training-chain portability 已有 lock plan，但 clean-environment CPU lock 仍未实际创建/验证。

## D-2026-05-17-02

- 日期：`2026-05-17`
- 决策：接受 `T39` adversarial review 的 `PASS`，标记 `T39` 完成，并将当前唯一任务切换为 `T40: Training-chain CPU-only clean-environment minimal real-training smoke`

### 背景

`T39` 是 Milestone 2J 的 clean-environment draft lock and dry-run bootstrap task。`docs/review/T39_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `requirements-train-cpu-win-py312.txt`、`docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_bootstrap.md`、任务包 Worker Output、`docs/review/T39_review.md`、`docs/for_human/T39_explanation.md` 五项输出均到位。
2. clean environment 与 `DLEnv` 分离，且只安装了 `numpy==2.4.5` 与 `PyYAML==6.0.3`。
3. 只运行了 `dataset_builder --dry-run`、`runtime_dataset_builder --dry-run`、`train --help`；未运行 training、benchmark、`.tflite`、hardware 或 cleanup。
4. 未修改 source/config/protocol/baseline/scenario/seed policy，未改写 `runs/`、`artifacts/` 或 `requirements-recovery.txt`。

### Warning 分类

1. N1：exact version pins are resolved-on-date rather than compatibility-matrix analyzed
   - 分类：`accepted`
   - 处理：对 draft lock 范围合适，不构成阻塞
2. N2：bootstrap doc records `pip list` rather than `pip freeze`
   - 分类：`accepted`
   - 处理：clean env 仅 2 个直接包，artifact file 本身已相当于 freeze 记录
3. N3：initial sandbox/network failure is documented transparently
   - 分类：`accepted`
   - 处理：这是权限/执行轨迹透明记录，不是绕过边界

没有 T39 `deferred` warning，因此不新增 risk。

### 结论

`T39` 可标记完成。它把 `R11` 从“只有 plan”推进到“clean CPU-only environment、draft lock、dry-run/import-level bootstrap 已验证”，但仍不等于 real clean-environment training execution，更不等于 full training reproducibility。

下一唯一任务切换到 `T40`。理由是：`R11` 当前最短闭环缺口已不再是 lock planning 或 dry-run，而是一次最小 real-training smoke；同时 `train.py` 会按 config 中的 `model_dir` / `report_dir` 落盘，因此必须通过单独 task package 强制使用 task-scoped isolated output paths，避免污染 canonical historical artifacts。

### 直接影响

1. `docs/04_task_board.md` 标记 `T39` 完成，并切换 `Current Unique Task` 到 `T40`。
2. 新增 `docs/tasks/Phase2/T40_training_chain_cpu_cleanenv_minimal_train_smoke.md`。
3. `docs/07_handoff.md` 记录 T39 Captain verdict、warning 分类与 T40 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 `R11`：clean-env draft lock/dry-run 已验证，但 real clean-environment training execution 仍未验证。
5. `docs/06_repo_noise_governance.md` 补充 T40 的 isolated-output noise boundary，禁止改写 canonical `artifacts/models/*` / `artifacts/reports/*`。

## D-2026-05-17-03

- 日期：`2026-05-17`
- 决策：接受 `T40` adversarial review 的 `PASS`，标记 `T40` 完成，并将当前唯一任务切换为 `T33: Tracked cache physical cleanup execution, only within T19 manifest`

### 背景

`T40` 是 Milestone 2J 的 clean CPU-only minimal real-training smoke task。`docs/review/T40_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml` 正确继承 canonical config，并把 `model_dir` / `report_dir` 重定向到 `artifacts/t40_train_smoke/...`。
2. `.venvs/t39_train_cpu_py312/` 中的一次真实训练 smoke 已成功完成，训练报告记录 `training_backend = numpy`、`training_device = cpu`。
3. canonical historical `artifacts/models/*`、`artifacts/reports/*`、`runs/`、`requirements-recovery.txt` 与 source/config 主线均未被修改。
4. 文档诚实记录了 T40 能验证与不能验证的边界，没有把 smoke 夸写成 full reproducibility。

### Warning 分类

1. N1：worker pre-review file overlap
   - 分类：`accepted`
   - 处理：属 task-package design artifact，不构成 worker 错误
2. N2：dataset manifest contains legacy macOS paths
   - 分类：`accepted`
   - 处理：属于历史数据元信息，不是 T40 新引入功能风险
3. N3：R11 narrowing not yet recorded
   - 分类：`deferred`
   - 处理：由 Captain 同步进治理文档；不返工 worker

### 结论

`T40` 可标记完成。它把 `R11` 从“clean-env 只做过 draft lock + dry-run/import”推进到“clean-env 已完成 one real-training smoke”，但仍不等于 full training reproducibility、GPU/CUDA portability、Linux portability 或 production-scale training validation。

下一唯一任务切换到 `T33`。理由是：

1. `T32` 仍被 `R12` 阻塞：当前机器缺少 `tensorflow` / `tflite_runtime`，真实 `.tflite` runtime 不可执行。
2. `T37` 仍被硬件/bitstream/readiness 前提阻塞。
3. `T33` 已有 `T19` 只读 manifest 作为唯一执行清单，是当前唯一既有边界又可立即执行的 bounded task。

### 直接影响

1. `docs/04_task_board.md` 标记 `T40` 完成，并切换 `Current Unique Task` 到 `T33`。
2. 新增 `docs/tasks/Phase2/T33_tracked_cache_physical_cleanup_execution.md`。
3. `docs/07_handoff.md` 记录 T40 Captain verdict、warning 分类与 T33 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 `R11` 与当前唯一任务口径。

## D-2026-05-17-04

- 日期：`2026-05-17`
- 决策：接受 `T33` adversarial review 的 `PASS`，标记 `T33` 完成，并将当前唯一任务切换为 `T34: Paper claim/evidence ledger and figure-table outline`

### 背景

`T33` 是 Milestone 2J 的 bounded tracked-cache physical cleanup execution task。`docs/review/T33_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. 116 个 tracked `.pyc` 文件全部位于 `T19` manifest 固定的 9 个 `__pycache__` 目录中，并已按 manifest 通过 `git rm --cached` 从 Git index 中移除。
2. `git ls-files | rg "__pycache__|\\.pyc$"` 已归零。
3. `runs/`、`artifacts`、source、config、benchmark、`.tflite`、hardware scope 均未被触碰。
4. 唯一 non-blocking issue 是 Windows 下 `index.lock` 权限摩擦，最终结果一致且正确。

### Warning 分类

1. N1：`index.lock` permission issue
   - 分类：`accepted`
   - 处理：属环境摩擦，不构成 scope 或 correctness 问题

没有 T33 `deferred` warning，因此不因 T33 review 新增 risk。

### 结论

`T33` 可标记完成。它把 tracked-cache cleanup 从“manifest/plan 已就位”推进到“bounded physical execution 已完成”，但没有扩展为更大范围的 repo cleanup。

下一唯一任务切换到 `T34`。理由是：

1. `T32` 仍被 `R12` 阻塞：当前机器缺少 `tensorflow` / `tflite_runtime`，真实 `.tflite` runtime 仍不可执行。
2. `T37` 仍被硬件/bitstream/readiness 前提阻塞。
3. `T34` 是当前唯一未完成、且不依赖新环境、新硬件或新运行结果的 bounded docs-only task。
4. `T34` 可以为后续论文收口建立 claim/evidence 边界账本，但不会升级任何 evidence level。

### 直接影响

1. `docs/04_task_board.md` 标记 `T33` 完成，并切换 `Current Unique Task` 到 `T34`。
2. 新增 `docs/tasks/Phase2/T34_paper_claim_evidence_ledger.md`。
3. `docs/07_handoff.md` 记录 T33 Captain verdict、warning 分类与 T34 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 `R4`、`R7` 与当前唯一任务口径。
5. `docs/06_repo_noise_governance.md` 更新 tracked-cache lane 已执行完成的 repo-noise 事实边界。

## D-2026-05-17-05

- 日期：`2026-05-17`
- 决策：接受 `T34` adversarial review 的 `PASS`，标记 `T34` 完成，并将当前唯一任务切换为 `T35: Paper draft skeleton and reviewer-risk audit`

### 背景

`T34` 是 Milestone 2K 的 docs-only paper claim/evidence ledger and figure-table outline task。`docs/review/T34_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. claim ledger、figure outline、table outline 都已落地，且所有 `supported` / `partial` / `blocked` 标注与 concrete evidence paths、risk IDs 一致。
2. mock-backed software HIL、true `.tflite` runtime、real-board validation、clean-env one-run smoke、statcalib interface-contract evidence 等关键边界都被显式保留。
3. 本轮没有 source、config、`runs/`、`artifacts`、`docs/02_experiment_plan.md` 或治理文档越界修改。

### Review Notes

1. N1 `C9` direct evidence paths are one hop indirect
   - 处理：`accepted`
2. N2 float/int8 quantization-gap claim omitted from ledger
   - 处理：`accepted`
3. N3 historical ablation conclusions omitted from ledger
   - 处理：`accepted`
4. N4 worker pre-review overwritten by adversarial review
   - 处理：`accepted`

这些都不是 blocking issue，也不需要写入新的 risks。

### 结论

`T34` 可标记完成。它把 paper assembly 从“靠人工记忆边界”推进到“有显式 claim/evidence ledger 可引用”的状态，但没有把任何历史、mock、stub、smoke 或 readiness 证据升级成更强结论。

下一唯一任务切换到 `T35`。理由是：

1. `T35` 是 Milestone 2K 中与 `T34` 直接衔接的下一个 bounded docs-only task。
2. `T32` 与 `T37` 仍分别被 `.tflite` 运行时依赖和硬件/bitstream 前提阻塞。
3. 基于现有 ledger 先做 paper skeleton 与 reviewer-risk audit，能继续收紧写作边界，而不引入新的实验或环境前提。

### 直接影响

1. `docs/04_task_board.md` 标记 `T34` 完成，并切换 `Current Unique Task` 到 `T35`。
2. 新增 `docs/tasks/Phase2/T35_paper_draft_skeleton_and_reviewer_risk_audit.md`。
3. `docs/07_handoff.md` 记录 T34 Captain verdict 与 T35 任务摘要。
4. `docs/08_risks_and_open_questions.md` 仅更新当前唯一任务口径；不新增 risk。

## D-2026-05-19-01

- Date: `2026-05-19`
- Decision: accept `T45` review as `PASS`, classify all `T45` warnings as `accepted`, mark `T45` complete, and switch the current unique task to `T46: Multi-seed mechanism/intervention plan and trace pack`.

### Rationale

`T45` is a bounded docs-only benchmark-expansion protocol task reviewed in `docs/review/T45_review.md` with verdict `PASS` and no blocking issues. The reviewer comments are all non-blocking and were handled as `accepted`:
1. N1 worker self-review overwritten by adversarial review: accepted as standard review practice.
2. N2 `sinusoidal` rejection rationale could be stronger: accepted, because T45 is a protocol-lock task rather than an execution task.
3. N3 exact future drift grids remain intentionally unlocked: accepted, because later execution tasks must lock them before running.
4. N4 worker human-facing explanation filename mismatch: accepted, because worker-facing and reviewer-facing explanation files serve different purposes.

No `deferred` warnings remain from `T45`, so no new risk item is opened here.

### Consequences

1. `docs/04_task_board.md` marks `T45` complete and switches `Current Unique Task` to `T46`.
2. `docs/07_handoff.md` records T45 closeout and points Worker to `T46`.
3. `docs/08_risks_and_open_questions.md` is synchronized so `R17/R18` reflect that protocol lock now exists while mechanism evidence remains open.
4. `docs/tasks/Phase2/T46_multi_seed_mechanism_intervention_plan_and_trace_pack.md` already exists, so no brand-new task package is needed; it should instead be treated as the next active bounded task.

## D-2026-05-17-06

- 日期：`2026-05-17`
- 决策：接受 `T35` adversarial review 的 `PASS`，标记 `T35` 完成，并将当前唯一任务切换为 `T41: Milestone 2K paper-assembly gate review and next-phase decision`

### 背景

`T35` 是 Milestone 2K 的 docs-only paper draft skeleton and reviewer-risk audit task。`docs/review/T35_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `docs/paper_materials/paper_draft_skeleton.md` 已按 section-level scaffold 绑定 claim IDs 与 figure/table IDs。
2. `docs/paper_materials/paper_reviewer_risk_audit.md` 已把 novelty、evidence-grade、overclaim、reproducibility/deployment、ablation/mechanism 风险绑定到具体 claim/risk。
3. `C6`、`C7`、`C8`、`C10`、`C11` 等 blocked claims 没有被静默升级。
4. 本轮没有 source、config、`runs/`、`artifacts`、benchmark protocol、`.tflite`、hardware 或治理结论文档越界修改。

### Review Notes

1. N1 title candidates are unusually conservative
   - 处理：`accepted`
2. N2 skeleton omits Background / Related Work section
   - 处理：`accepted`
3. N3 section-by-section hotspot table uses generic labels
   - 处理：`accepted`
4. N4 worker pre-review overwritten by adversarial review
   - 处理：`accepted`

这些都不是 blocking issue，也不需要写入新的 risks。

### 结论

`T35` 可标记完成。它把 Milestone 2K 从“只有 ledger”推进到“ledger + draft skeleton + reviewer-risk audit”三件套齐备的状态，但没有把任何 mock、stub、smoke、readiness 或 partial evidence 升级为更强结论。

下一唯一任务切换到 `T41`。理由是：

1. Milestone 2K（`T34 + T35`）已经完成，下一步最合理的是先做里程碑 gate review，而不是直接跳入 prose expansion。
2. `T32` 与 `T37` 仍分别被 `.tflite` runtime 依赖和硬件/bitstream 前提阻塞。
3. `T41` 可以在不引入新实验的前提下，明确 paper positioning、Background / Related Work 是否必须先补、以及下一个真正应推进的 bounded task。

### 直接影响

1. `docs/04_task_board.md` 标记 `T35` 完成，并切换 `Current Unique Task` 到 `T41`。
2. 新增 `docs/tasks/Phase2/T41_paper_assembly_milestone_review.md`。
3. `docs/07_handoff.md` 记录 T35 Captain verdict、non-blocking note 处理与 T41 任务摘要。
4. `docs/08_risks_and_open_questions.md` 仅更新当前唯一任务口径；不新增 risk。

## D-2026-05-17-07

- 日期：`2026-05-17`
- 决策：接受 `T41` review 的 `PASS`，标记 `T41` 完成，并将当前唯一任务切换为 `T42: Paper Background / Related Work scaffold and method-positioning calibration`

### 背景

`T41` 是 Milestone 2K 的 read-only paper-assembly gate review task。`docs/review/T41_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `docs/review/Milestone2K_review.md` 给出 `Allow`，并明确 Milestone 2K 可关闭。
2. minimum safe paper positioning 与 `docs/paper_materials/paper_reviewer_risk_audit.md` 一致，没有升级 blocked claims。
3. Background / Related Work 必须在 prose expansion 前先补。
4. 推荐下一唯一任务为 `T42`，范围仍是 docs-only。

### Review Notes
## D-2026-05-18-08

- Date: `2026-05-18`
- Decision: accept `T42` review as `PASS`, mark `T42` complete, and switch the current unique task to `T43: Paper Background / Related Work bounded prose draft`.

### Rationale

`T42` is a bounded docs-only framing task reviewed in `docs/review/T42_review.md` with verdict `PASS` and no blocking issues. The reviewer comments are non-blocking and are treated as accepted framing guidance:
1. N1 accepted and integrated: subsection-6 wording should remain neutral survey language rather than self-justifying novelty language.
2. N2 accepted and integrated: the method-forward title direction is retained only as a working framing recommendation, not as a locked human-final title.
3. N3 accepted as normal workflow: worker pre-review being superseded by adversarial review does not create a repository or evidence risk.

No `deferred` warnings remain from `T42`, so no new risk item is opened here.

### Consequences

1. `docs/04_task_board.md` marks `T42` complete and switches `Current Unique Task` to `T43`.
2. `docs/07_handoff.md` records T42 closeout, clarifies that the current working framing is method-forward title plus evidence-bounded body text, and points Worker to `T43`.
3. `docs/08_risks_and_open_questions.md` is synchronized so T42 adds no new deferred risks and T43 is the next active task.
4. `docs/03_hil_p4_boundary_audit.md` and `docs/06_repo_noise_governance.md` are synchronized so T42/T43 remain explicitly docs-only and non-evidence-upgrading.
5. The next recommended Worker task is `docs/tasks/Phase2/T43_paper_background_related_work_prose_draft.md`, and it must not expand into full-manuscript drafting, experiments, `.tflite`, hardware, cleanup, or claim upgrades.

## D-2026-05-18-09

- Date: `2026-05-18`
- Decision: accept `T43` review as `PASS`, classify all `T43` warnings as `accepted`, mark `T43` complete, and switch the current unique task to `T44: Research Reality Recovery Mode setup and evidence-gap ledger`.

### Rationale

`T43` is a bounded docs-only prose draft reviewed in `docs/review/T43_review.md` with verdict `PASS` and no blocking issues. The reviewer comments are all non-blocking and were handled as `accepted`:
1. N1 subsection-6 neutrality: accepted. It is wording guidance only.
2. N2 placeholder citation markers: accepted. They are later bibliography work, not an evidence blocker.
3. N3 drafting annotations: accepted. They are drafting scaffolding, not repo facts.
4. N4 inline claim-format inconsistency: accepted. It is cosmetic cleanup for later assembly.

No `deferred` warnings remain from `T43`, so no new risk item is opened here. The user explicitly requested that paper expansion pause until evidence/material truth is re-frozen and audited, so the project now enters `Research Reality Recovery Mode`.

### Consequences

1. `docs/04_task_board.md` already carries `T44` as the current unique task and Milestone 2O.
2. `docs/07_handoff.md`, `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, and `docs/08_risks_and_open_questions.md` are synchronized to point to `T44`.
3. `docs/tasks/Phase2/T44_research_reality_recovery_mode_setup_and_evidence_gap_ledger.md` already exists, so no new task package is needed.
4. The next recommended Worker task is `T44`, and it must only build the recovery baseline: claim/evidence truth, reproducibility gaps, figure/result readiness, and paper-claim risk.

1. N1 T34 review path mis-cited in `docs/review/Milestone2K_review.md`
   - 处理：`accepted`
   - 说明：Captain integration 已修正为 `docs/review/T34_review.md`
2. N2 `docs/for_human/T41_explanation.md` 把 challenge-point count 写成 18，实际为 20
   - 处理：`accepted`
   - 说明：Captain integration 已修正计数

这些都不是 blocking issue，也不需要写入新的 risks。

### 结论

`T41` 可标记完成。它正式关闭 Milestone 2K，并把后续论文推进从“先做 gate”推进到“可以开始补 Background / Related Work 与 method-positioning calibration”，但没有把任何 mock、stub、smoke、readiness 或 partial evidence 升级为更强结论。

下一唯一任务切换到 `T42`。理由是：

1. `T41` 已明确要求在 prose expansion 前先补 Background / Related Work scaffold。
2. `T35` review 的 conservative-title 问题和 `T41` milestone review 的 method-forward compromise 需要一个独立的校准任务来收口。
3. `T32` 与 `T37` 仍分别被 `.tflite` runtime 依赖和硬件/bitstream 前提阻塞，不适合作为当前唯一任务。

### 直接影响

1. `docs/04_task_board.md` 标记 `T41` 完成，并切换 `Current Unique Task` 到 `T42`。
2. 新增 `docs/tasks/Phase2/T42_paper_background_related_work_and_positioning.md`。
3. `docs/07_handoff.md` 记录 T41 Captain verdict、non-blocking note 处理与 T42 任务摘要。
4. `docs/08_risks_and_open_questions.md` 仅更新当前唯一任务口径；不新增 risk。

## D-2026-05-22-01

- 日期：`2026-05-22`
- 决策：复核最近提交后，确认项目可以进入 `T46`，并将 `docs/tasks/Phase2/T46_multi_seed_mechanism_intervention_plan_and_trace_pack.md` 收紧为更规范的执行型任务包。

### 背景

用户指出最近对项目做了一轮未提前同步的改进，因此需要重新核对 `docs/04_task_board.md` 与最近 Git 提交，判断当前唯一任务是否仍应停在 `T46`，以及 `T46` 任务包是否已经足够清晰可执行。

### 依据

1. `2026-05-18` 的提交 `84c3fb7` 已完成 `T42/T43` 收口，并把项目切入 `Research Reality Recovery Mode`。
2. `2026-05-19` 的提交 `8b32318` 已完成 `T44` recovery baseline 与 `T53` 主线理论分析，补齐了 truth freeze、claim/evidence ledger 与当前主线理论说明。
3. `2026-05-20` 的提交 `4857bf5` 已完成 `T45` 收口，把 benchmark-expansion lane 锁成 protocol/gap-audit，而不是执行任务，并正式把当前唯一任务切到 `T46`。
4. 最近这些提交主要新增的是 recovery、theory、protocol 与 README 类文档，没有新增更强的多 seed 机制执行证据，也没有新增 `.tflite` runtime、real-board 或更广 benchmark 的已验证事实。
5. 因此当前最紧的缺口仍然是 `R10` 所代表的机制证据闭环缺口，而不是 benchmark 扩展、部署边界升级或继续 paper prose。

### 结论

可以进入 `T46`，而且现在进入比此前更合理：

1. `T44` 已把 paper-facing truth baseline 冻结清楚，避免 `T46` 在错误事实边界上做机制规划。
2. `T53` 已提供当前主线的理论解释，使 `T46` 可以把 trace / intervention 设计与主线公式语义对齐。
3. `T45` 已把 benchmark expansion 与 mechanism lane 分开，减少了 `T46` 被误写成 benchmark-scope expansion 的风险。
4. 但最近提交并没有新增 multi-seed causal evidence，所以不能跳过 `T46` 直接进入 `T47` 或更后续的 paper/material lane。

### 直接影响

1. `docs/tasks/Phase2/T46_multi_seed_mechanism_intervention_plan_and_trace_pack.md` 被补充了 `Docs To Update`、`Forbidden Scope`、更完整的 `Required Inputs`、输出结构、verification 项和 review no-go triggers。
2. `docs/07_handoff.md` 当前任务摘要同步提醒，以任务包中的新边界与 no-go 规则为准。
3. `docs/04_task_board.md` 现状判断保持不变：当前唯一任务仍是 `T46`。

## D-2026-05-24-01

- 日期：`2026-05-24`
- 决策：接受 `T55` review 的 `PASS`，标记 `T55` 完成，并将当前唯一任务切换为 `T56: Post-I1 mechanism claim reframing gate`。

### 背景

`T55` 是 `T54` 之后的 Phase B 单变体 intervention probe。`docs/review/T55_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. 任务保持在锁定的 6-seed pack、冻结四场景、`repeats=2` 和单一 pure I1 lower-clip intervention 范围内。
2. Worker 没有重训模型、没有修改 source-tree code/config，也没有打开 `.tflite`、真板、cleanup 或 benchmark-expansion 范围。
3. 报告的主要实证结论是 I1 intervention 为 mixed 且整体偏 harmful，不支持把 “high committed-b is harmful” 继续当成一般机制解释。
4. `C4` 仍保持 `partial`，并且 `T47` 不应被视为自动可推进的下一任务。

### Warning 处理

T55 review 的非阻塞评论全部按 `accepted` 处理：

1. N1 顶层 `summary.json` 在最终 resume 后一度变 stale，但底层 per-repeat data 完整且 reviewer 已重新聚合：`accepted`
2. N2 run root 中残留 `benchmark_test/` smoke 产物：`accepted`
3. N3 T54 baseline comparison 使用 per-seed 平均 LER 而非 per-repeat pairing：`accepted`
4. N4 缺少单独的 `intervention_trace_summary.csv` / `seed_model_reuse_manifest.json` 文件名，但信息已由现有 report / CSV / `manifest.json` 覆盖：`accepted`
5. N5 report 中 “expected I1 effect” 属于诊断性解释而非实测 trace comparison：`accepted`

Suspicious implementation details 也不升格为阻塞：

1. S1 未对 I1 run 产出独立 trace export：`accepted`
2. S2 I1 benchmark timing 继承 `n_slow_updates=900`，与部分 T54 旧 seed baseline 的 `300` 不完全同源：`accepted`

没有 `deferred` 或 `rejected` warning，因此这次不新增由 warning classification 触发的 risk。

### 结论

`T55` 可以标记完成，但当前项目不应直接切入 `T47`。

原因是：

1. `T55` 已经提供 targeted intervention evidence，因此 `R10` 的状态发生了变化：问题不再是“缺 intervention evidence”，而是“现有 intervention evidence 使原先的简单 harmful-instability 叙事站不住脚”。
2. 这意味着项目先需要一个 docs-only gate 来重写 mechanism claim boundary，而不是马上做 paper ablation/material packaging。
3. 因此下一唯一任务应是新增 `T56`：对 `T36/T38/T54/T55` 的 mechanism claims 做 retain / weaken / retire / reframe / still-open 收口，并明确 `T47` 是否可在带 hedge 的前提下推进。

### 直接影响

1. `docs/04_task_board.md` 标记 `T55` 完成，并切换 `Current Unique Task` 到 `T56`。
2. 新增 `docs/tasks/Phase2/T56_post_i1_mechanism_claim_reframing_gate.md`。
3. `docs/07_handoff.md` 记录 T55 Captain closeout，并把当前唯一任务包摘要切换到 `T56`。
4. `docs/08_risks_and_open_questions.md` 更新 `R10`：现在不是缺 intervention evidence，而是已有 intervention evidence 但需要机制叙事重构。

## D-2026-05-24-02

- 日期：`2026-05-24`
- 决策：接受 `T56` review 的 `PASS`，标记 `T56` 完成，并将当前唯一任务切换为 `T47: Paper ablation result-pack and material ledger`。

### 背景

`T56` 是一张 docs-only 的 post-I1 mechanism-claim reframing gate。`docs/review/T56_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `T36/T38/T54/T55` 的 evidence floor 已被诚实收口为 retain / weaken / retire / reframe / still-open。
2. 现有 intervention evidence 使 “high committed-`b` is harmful” 不能继续作为一般机制解释。
3. `T47` 只能作为 hedge-conditioned paper-material lane 推进，不能被写成自动的下一 intervention 或机制闭环。
4. Worker 没有越界修改 source、config、`runs/`、`artifacts` 或治理文档。

### Warning 处理

T56 review 的非阻塞评论全部按 `accepted` 处理：

1. N1 worker self-review was overwritten by adversarial review：`accepted`
2. N2 claim-table summary wording count inconsistency：`accepted`
3. N3 governance sync items in `docs/08_risks_and_open_questions.md`：`accepted`
4. N4 governance sync items in `docs/04_task_board.md`：`accepted`
5. N5 C4 boundary wording is long but justified：`accepted`

没有 `deferred` 或 `rejected` warning，因此不因 T56 review 新增 risk。

### 结论

`T56` 可标记完成。下一步可以进入 `T47`，但只能作为 docs-only 的 hedge-conditioned paper-material lane。

原因是：

1. `T56` 已经完成 mechanism-claim reframing，当前最小可继续推进的内容不是新 intervention，而是 paper-facing ablation/result/material 台账。
2. `T47` 可以冻结 ready / partial / missing 的 paper materials，而不打开 benchmark expansion、`.tflite`、真板、cleanup 或第二 intervention 范围。
3. `T47` 不能被写成“机制已解决后自然进入论文扩写”；它必须保留 `T56` 给出的 hedge wording 与 non-claims。

### 直接影响

1. `docs/04_task_board.md` 标记 `T56` 完成，并切换 `Current Unique Task` 到 `T47`。
2. `docs/07_handoff.md` 记录 T56 Captain closeout，并把当前唯一任务包摘要切换到 `T47`。
3. `docs/08_risks_and_open_questions.md` 更新 `R10`、`R18` 与当前唯一任务口径。

## D-2026-05-23-01

- 日期：`2026-05-23`
- 决策：接受 `T54` review 的 `PASS`，标记 `T54` 完成，并将当前唯一任务切换为 `T55: Phase B multi-seed I1 residual-clip intervention probe`。

### 背景

`T54` 是 `T46` 所定义的 Phase A multi-seed trace-only generalization probe。`docs/review/T54_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `docs/evidence_packs/mechanism_ablation/multi_seed_trace_generalization_probe.md` 结构完整，包含命令、复用/重跑矩阵、字段可用性、跨 seed 汇总、结论与 non-claims。
2. 任务保持在既有 `T38` trace/export 路径之内，只使用锁定的 6-seed pack、冻结四场景、`Full` vs `Gated v5`，没有运行 intervention。
3. 最终结论是 “broadly repeated with qualifications”，而不是把单 seed 结果直接升级成 causal proof 或 paper-grade benchmark evidence。
4. Worker 没有修改 source、benchmark code、source-tree config、tests、runtime、hardware、training 文件或治理文档。

### Warning 处理

T54 review 的非阻塞评论全部按 `accepted` 处理：

1. N1 新 benchmark run dirs 写在 T54 单一 run root 之外，但属于新增目录而不是历史覆盖：`accepted`
2. N2 新 rerun seeds 的 seed reuse manifest 元数据不完整：`accepted`
3. N3 复用 seeds 与新 rerun seeds 采用了不同 paired-seed conventions：`accepted`
4. N4 `cross_seed_analysis.py` 是 run-root-local helper script：`accepted`
5. N5 机制结论由 “broadly repeated” 精化为 “broadly repeated with qualifications”：`accepted`

没有 `deferred` 或 `rejected` warning，因此这次不新增由 warning classification 触发的 risk。

### 结论

`T54` 可以标记完成，但项目仍不应直接跳到旧版 `T47`。

原因是：

1. `T54` 只回答了 generalization 问题，把 `committed-b` instability 从单 seed 诊断升级为 bounded multi-seed diagnostic evidence。
2. `T54` 同时表明机制图景比单 seed 版本更复杂，存在 quiet / classic / universal 三类表现，因此 `C4` 仍保持 `partial`。
3. 当前最小仍未关闭的缺口是 targeted intervention evidence，而不是 paper ablation/material packaging。
4. 因此下一唯一任务应是新增 `T55`：在同一 6-seed pack 与冻结四场景下，只做一个 pure I1 residual-clip lower-bound intervention probe，判断该 intervention 是 helpful、harmful、mixed，还是 no-clear-effect。

### 直接影响

1. `docs/04_task_board.md` 标记 `T54` 完成，并切换 `Current Unique Task` 到 `T55`。
2. 新增 `docs/tasks/Phase2/T55_multi_seed_i1_residual_clip_intervention_probe.md`。
3. `docs/07_handoff.md` 记录 T54 Captain closeout，并把当前唯一任务包摘要、下一步建议、暂不继续事项切换到 `T55`。
4. `docs/08_risks_and_open_questions.md` 保持 `R10` / `R18` 指向 `T55` 作为下一条 bounded mechanism lane。

## D-2026-05-22-02

- 日期：`2026-05-22`
- 决策：接受 `T46` review 的 `PASS`，标记 `T46` 完成，并将当前唯一任务切换为 `T54: Phase A multi-seed trace-only generalization probe`。

### 背景

`T46` 是一张 docs-only 的机制证据规划任务。`docs/review/T46_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `docs/evidence_packs/mechanism_ablation/seed_mechanism_multi_seed_plan.md` 结构完整，包含 claim-boundary、seed-selection、trace-field inventory、future execution-pack 四张关键表。
2. `T36/T38` 的 single-seed evidence 没有被静默升级为 multi-seed confirmation 或 causal proof。
3. T46 保持了与 benchmark expansion、deployment validation、paper material freeze 的分离。
4. Worker 没有越界修改 source、config、`runs/`、`artifacts` 或治理文档。

### Warning 处理

T46 review 的非阻塞评论全部按 `accepted` 处理：

1. N1 相邻 seed `20260430` 与 seed-spacing wording tension：`accepted`
2. N2 working tree 中可见 Captain-level governance sync：`accepted`
3. N3 Phase A 未给 wall-clock estimate：`accepted`
4. N4 未直接回引 `docs/paper_materials/paper_claim_evidence_ledger.md`：`accepted`
5. N5 worker self-review accurate：`informational`

没有 `deferred` 或 `rejected` warning，因此不因 T46 review 新增 risk。

### 结论

`T46` 可标记完成。但下一步不应直接进入旧版 `T47`。

原因是：

1. `T46` 自己已经给出明确 phased recommendation：先做 Phase A multi-seed trace-only probe，再决定是否值得开 intervention lane。
2. 如果 `committed-b` instability 不能在极小多 seed 包里复现，那么直接进入 `T47` paper material ledger 会让 paper-facing material freeze 跑在机制证据前面。
3. 因此当前更合理的下一唯一任务是新增 `T54`，而不是让 Worker 立即执行 `T47`。

### 直接影响

1. `docs/04_task_board.md` 标记 `T46` 完成，并切换 `Current Unique Task` 到 `T54`。
2. 新增 `docs/tasks/Phase2/T54_multi_seed_trace_only_generalization_probe.md`。
3. `docs/07_handoff.md` 记录 T46 Captain closeout 和 T54 任务摘要。
4. `docs/08_risks_and_open_questions.md` 更新 `R10` 的下一收口路径与当前唯一任务口径。
## D-2026-05-24-03

- 日期：`2026-05-24`
- 决策：接受 `T47` review 的 `PASS`，标记 `T47` 完成，并将当前唯一任务切换为 `T57: FR7 feature/teacher ablation re-execution under locked T24 protocol`。

### 背景

`T47` 是 `T56` 之后的 hedge-conditioned docs-only paper-material lane。`docs/review/T47_review.md` 给出 `PASS`，blocking issues 为无。Reviewer 确认：

1. `docs/paper_materials/paper_ablation_result_pack.md` 已按边界要求冻结 ready / partial / missing 的 ablation/material ledger。
2. 当前输出没有把 `C4`、机制解释、`.tflite`、真板、formal benchmark expansion 或第二 intervention 写成已完成事实。
3. `FR7` 被明确保留为 `missing`，没有被历史 pre-T24 证据冒充为 formal protocol evidence。
4. 该任务没有触碰 source、config、历史 run-root、`.tflite`、硬件或 cleanup 范围。

### Warning 处理

T47 review 的 non-blocking items 全部按 `accepted` 处理：

1. N1 Worker Output 中 figure-entry count 误写：`accepted`
2. N2 worker summary 超出严格 Allowed Files：`accepted`
3. N3 F2 ready-without-drawn-figure classification：`accepted`
4. N4 F3 blocked carry-forward note：`accepted`
5. N5 conceptual regeneration paths instead of executable scripts：`accepted`

没有 `deferred` 或 `rejected` warning，因此这次不新增由 warning classification 触发的 risk。

### 结论

`T47` 可以标记完成，但它没有关闭 `FR7`，只是把这个 gap 更诚实地冻结了出来。当前最小且合理的下一步不是恢复更宽的 paper prose、`.tflite`、真板、benchmark expansion、comparator expansion 或新 intervention，而是新增 `T57`，在锁定 `T24` protocol 的前提下做 bounded FR7 re-execution。

### 直接影响

1. `docs/04_task_board.md` 记录 `T47 -> PASS` 并将当前唯一任务切换到 `T57`。
2. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `docs/tasks/Phase2/T57_fr7_feature_teacher_ablation_reexecution.md`。
3. `docs/08_risks_and_open_questions.md` 同步 T47 warning 分类与 `T57` 当前任务口径；不新增由 T47 warning 触发的 risk。
4. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T47` 收口和 `T57` 边界。

## D-2026-05-26-02

- 鏃ユ湡锛歚2026-05-26`
- 鍐崇瓥锛氭帴鍙?`T58` review 鐨?`PASS_WITH_WARNINGS`锛屾爣璁?`T58` 瀹屾垚锛屽苟灏嗗綋鍓嶅敮涓€浠诲姟鍒囨崲涓?`T59: Statcalib separate comparator lane integration and bounded smoke`銆?

### 鑳屾櫙

`T58` 鏄?`FR6` 鐨?docs-only figure-pack 鏀跺彛浠诲姟銆俙docs/review/T58_review.md` 缁欏嚭 `PASS_WITH_WARNINGS`锛宐locking issues 涓烘棤銆俁eviewer 纭锛?

1. `FR6` 鐨?figure-pack 浜х墿銆乧aption銆乫igure-data snapshot 鍜?paper-facing ledger 鏇存柊宸插叿浣撳苟涓?bounded form 瀛樺湪銆?
2. `T58` 娌℃湁鎶?`T54/T55/T56` 鐨勮瘉鎹鎺ㄤ负 causal proof銆乣.tflite` validation銆乺eal-board validation 鎴?expanded benchmark evidence銆?
3. `FR6` 鐜板湪鍙互鎸?ready` 鐨?bounded descriptive figure-pack 鎰忎箟鏀跺彛锛屼絾 `R10` 鍜?`C4` 鐨勮竟鐣屼笉鍙樸€?
4. review 鍚屾椂鎸囧嚭锛宍FR8` 鐩稿叧鐨?statcalib` 材料缂哄彛浠嶇劧瀛樺湪锛屼笖褰撳墠浠撳簱杩樻病鏈?integrated comparator lane锛屽洜姝や笉鑳界洿鎺ユ妸涓嬩竴姝ュ啓鎴?FR8 result table銆?

### Warning 澶勭悊

T58 review 鐨?non-blocking items 鎸夊備笅鍒嗙被锛?

1. N1 mixed governance diff provenance = `accepted`
2. N2 FR6 deliverable completeness note = `accepted`
3. N3 task-local seed-category derivation = `accepted`
4. N4 worker self-review is not final acceptance review = `accepted`

娌℃湁 `deferred` 鎴?`rejected` warning锛屽洜姝よ繖娆′笉鏂板鐢?warning classification 瑙﹀彂鐨?risk銆?

### 缁撹

`T58` 鍙互鏍囪瀹屾垚锛屼絾瀹冨叧闂殑鍙槸 `FR6` 鐨?paper-facing figure-pack gap锛屼笉鏄?mechanism closure锛屼笉鏄?`R10` causal closure锛屼篃涓嶆槸 `FR8` statcalib comparator evidence銆傚綋鍓嶆渶灏忎笖鍚堢悊鐨勪笅涓€姝ヤ笉鏄?`.tflite`銆佺湡鏉裤€乀50/T51/T52` 鎴栫洿鎺ョ敓鎴?`FR8` 琛ㄦ牸锛岃€屾槸鏂板 `T59`锛屽厛鎶?`statcalib` 浣滀负 separate comparator lane 鎺ュ叆 slow-loop / bounded P4 smoke锛岀敤鏈€灏忚瘉鎹湅瀹冩槸鍚︾湡姝ｅ彲鎵ц銆?

### 鐩存帴褰卞搷

1. `docs/04_task_board.md` 璁板綍 `T58 -> PASS_WITH_WARNINGS`锛屽苟灏?`Current Unique Task` 鍒囨崲鍒?`T59`銆?
2. 鏂板浠诲姟鍖?`docs/tasks/Phase2/T59_statcalib_comparator_lane_integration_and_smoke.md`銆?
3. `docs/07_handoff.md` 鍒囨崲褰撳墠 worker-facing task package 鍒?`T59`銆?
4. `docs/00_project_snapshot.md`銆乣docs/01_legacy_audit.md`銆乣docs/03_hil_p4_boundary_audit.md`銆乣docs/06_repo_noise_governance.md`銆乣docs/08_risks_and_open_questions.md` 鍚屾 `T58` 鏀跺彛鍜?`T59` 杈圭晫銆?

## D-2026-05-26-03

- 日期：`2026-05-26`
- 决策：接受 `T58_review.md` 的 `PASS_WITH_WARNINGS`，标记 `T58` 完成，并将当前唯一任务切换为 `T59: Statcalib separate comparator lane integration and bounded smoke`。

### 背景

`T58` 是 `FR6` 的 docs-only figure-pack 收口任务。依据 `docs/review/T58_review.md`：

1. blocking issues = none
2. `FR6` figure-pack、caption、figure data 与 paper-facing ledgers 已形成一致的 bounded descriptive package
3. 本次收口没有把 `T54/T55/T56` 证据升级成 causal proof、`.tflite` validation、real-board validation 或 expanded benchmark evidence
4. 当前真正仍缺的是 `FR8` 相关的 integrated `statcalib` comparator lane，而不是继续扩写 paper prose

### Warning 处理

本次 warning 分类如下：

1. `N1` mixed-governance diff provenance = `accepted`
2. `N2` FR6 deliverable completeness note = `accepted`
3. `N3` task-local seed-category derivation = `accepted`
4. `N4` worker self-review is not final acceptance review = `accepted`

没有新的 `deferred` 或 `rejected` warning，因此本次不新增由 warning classification 触发的 risk。

### 结论

`T58` 可以完成，但它关闭的只是 `FR6` 的 bounded descriptive figure-pack gap，不是 `R10` causal closure，也不会把 `C4` 升级为 `supported`。下一步不能直接写 `FR8` 正式结果表；更小且更诚实的步骤是新增 `T59`，先把 `statcalib` 作为 separate comparator lane 接入并完成一轮 bounded smoke。

### 直接影响

1. `docs/04_task_board.md` 记录 `T58 -> PASS_WITH_WARNINGS`，并切换 `Current Unique Task` 到 `T59`
2. 新增任务包 `docs/tasks/Phase2/T59_statcalib_comparator_lane_integration_and_smoke.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T59`
4. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md`、`docs/08_risks_and_open_questions.md` 同步 `T58` 收口与 `T59` 边界

## D-2026-05-26-01

- 日期：`2026-05-26`
- 决策：接受 `T57` review 的 `PASS`，标记 `T57` 完成，并将当前唯一任务切换为 `T58: FR6 multi-seed mechanism/intervention figure pack`。

### 背景

`T57` 是在锁定 `T24` formal benchmark 协议下，对 `FR7` 做的 bounded feature/teacher ablation re-execution。`docs/review/T57_review.md` 确认：

1. scope 仍严格锁定在 4 个 frozen scenarios、6 个固定模式和 `repeats=2`
2. coverage 为 `4 x 6 x 2 = 48`，完整覆盖 `coverage=1.0`
3. `FR7` 现在可以作为 frozen-set result table 收口
4. `hybrid_no_teacher_params` 在 4 个场景里都成为最佳模式，因此简单的 teacher-params-necessary 叙事不再安全
5. reviewer 建议下一个最小合理任务是 `FR6`，而不是继续追加 `FR7` 或提前进入 `FR8`

### Warning 处理

- `T57` review 没有新的 blocking issue。
- `T57` review 也没有新的 non-blocking item 需要再做 `accepted / deferred / rejected` 分类。
- 因此这次不会因为 warning classification 新增 risk。

### 结论

`T57` 可以标记完成，但它关闭的是 `FR7` 的 bounded 表格缺口，不是 `R10` 的 causal closure，也不是架构归因证明。后续 paper lane 必须继续避免把 teacher params 写成必要正贡献因素。当前最小且合理的下一步是新增 `T58`，仅利用既有 `T54/T55/T56` 证据完成 `FR6` 的 docs-only figure-pack 收口。

### 直接影响

1. `docs/04_task_board.md` 记录 `T57 -> PASS`，并将 `Current Unique Task` 切换到 `T58`
2. 新增任务包 `docs/tasks/Phase2/T58_fr6_multi_seed_mechanism_intervention_figure_pack.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T58`
4. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md`、`docs/08_risks_and_open_questions.md` 同步 `T57` 收口和 `T58` 边界

## D-2026-05-26-04

- 日期：`2026-05-26`
- 决策：接受 `T59_review.md` 的 `PASS_WITH_WARNINGS`，标记 `T59` 完成，并将当前唯一任务切换为 `T60: Statcalib lane isolation and regression hardening`。

### 背景

`T59` 的目标是把 `statcalib` 作为 separate comparator lane 接入 slow-loop/runtime/benchmark 链路，并用一个 bounded smoke 证明它能端到端运行。依据 `docs/review/T59_review.md` 和本轮 captain 复核：

1. blocking issues = none
2. focused tests 和 `py_compile` 均通过
3. `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740` 的 bounded smoke 真实存在，且 `comparison.csv` / `summary.json` 中已出现独立 `mode=statcalib` 与 `statcalib_status` / `statcalib_reason`
4. 但当前结果仍只是 integration evidence，不是 `FR8` formal comparator evidence

### Warning 处理

本次 warning 分类如下：

1. `W1` cross-mode `teacher_mode` fallback coupling = `deferred`
2. `W2` smoke-doc key-name mismatch = `accepted`
3. `W3` dirty-worktree smoke provenance weakness = `deferred`

另外，review 中的 missing-tests / suspicious-details 结论不单独再开新的 warning 编号，而是并入 `R26` / `R27` 继续跟踪。

### 结论

`T59` 可以完成，但它关闭的只是 "integrated lane can execute once" 这一层缺口，不是 `FR8` 正式结果表，也不是 formal comparator ranking。下一步最小且更诚实的动作不是直接开 `FR8`，而是先做 `T60`，把 cross-mode semantics 和 regression coverage 硬化到足以承接后续 fairness / robustness 审计。

### 直接影响

1. `docs/04_task_board.md` 记录 `T59 -> PASS_WITH_WARNINGS`，并切换 `Current Unique Task` 到 `T60`
2. 新增任务包 `docs/tasks/Phase2/T60_statcalib_lane_isolation_and_regression_hardening.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T60`
4. `docs/08_risks_and_open_questions.md` 新增 `R26` / `R27`，并同步 `T59` warning 分类
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T59` 收口与 `T60` 边界

## D-2026-05-27-01

- 日期：`2026-05-27`
- 决策：接受 `T60_review.md` 的 `PASS`，标记 `T60` 完成，并将当前唯一任务切换为 `T61: Statcalib clean-provenance fairness sanity rerun`。

### 背景

`T60` 的目标是收口 `T59` 留下的语义与回归覆盖问题，而不是继续扩 benchmark scope。依据 `docs/review/T60_review.md` 与本轮 captain 复核：

1. blocking issues = none
2. focused unit tests 与 `py_compile` 均通过
3. `slow_loop.statcalib.teacher_mode` 已不再泄漏到非 `statcalib` mode
4. 直接 estimator / aggregation / report 回归覆盖已补齐
5. 没有新 run root、没有 config 变更、没有 theory-branch 混写

### Warning 处理

- `T60` review 没有新的 warning 项需要做 `accepted / deferred / rejected` 分类。
- `T59` 遗留 warning 中：
  - `W1` cross-mode `teacher_mode` fallback coupling 现已由 `T60` 解决
  - `W2` 保持 `accepted`
  - `W3` 继续保留在 `R27`

### 结论

`T60` 可以完成，并且它关闭的是 pre-FR8 的语义/回归覆盖 blocker，不是 `FR8` 正式 comparator evidence。当前最小且更诚实的下一步不是直接开 `FR8`，而是新增 `T61`：在 clean committed worktree 上，对 `T59` 的 bounded statcalib smoke 做一轮 provenance-clean fairness sanity rerun，确认那个异常强的结果是否仍然成立。

### 直接影响

1. `docs/04_task_board.md` 记录 `T60 -> PASS`，并切换 `Current Unique Task` 到 `T61`
2. 新增任务包 `docs/tasks/Phase2/T61_statcalib_clean_provenance_fairness_sanity.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T61`
4. `docs/08_risks_and_open_questions.md` 关闭 `R26`，并把 `R27` 缩窄为 provenance/fairness blocker
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T60` 收口与 `T61` 边界
## D-2026-05-27-02

- 日期：`2026-05-27`
- 决策：接受 `T61_review.md` 的 `BLOCK`，不标记 `T61` 完成，并将当前唯一任务切换为 `T62: Statcalib provenance-isolated fairness rerun`

### 背景

`T61` 的目标不是一般性的 fairness sanity rerun，而是 **clean-provenance** fairness sanity rerun。它被创建出来，是为了关闭 `T59` dirty-worktree smoke 留下的 `R27`。

### 依据

1. `docs/review/T61_review.md` 的 blocking 判断与仓库事实一致：
   - clean launch `HEAD=9174065`
   - final `runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239/summary.json` records `git_commit=6058f42`
2. `git reflog --date=iso --all` shows a mid-run checkout away from `main`
3. `git diff --name-only 9174065 6058f42 -- cnn_fpga tests` is non-empty, so launch and finish identities are not interchangeable
4. `T61` 的 bounded result signal did persist:
   - same two scenarios
   - same three modes
   - `statcalib` still wins both scenarios
5. 但 persisted fairness signal 不等于 provenance blocker 已关闭

### 结论

`T61` 不能按 `PASS` 或 `PASS_WITH_WARNINGS` 收口，因为它没有完成自己被指派去关闭的 blocker。更准确地说：

- fairness sanity 这一半做到了
- clean provenance 这一半没有做到

因此：

- `T61` verdict = `BLOCK`
- `T61` 不完成
- `R27` 继续保持打开
- 下一步只允许修 blocking issue，不允许跳到 `FR8`

### 直接影响

1. `docs/04_task_board.md` 记录 `T61 -> BLOCK`，并将 `Current Unique Task` 切换到 `T62`
2. 新增任务包 `docs/tasks/Phase2/T62_statcalib_provenance_isolated_fairness_rerun.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T62`
4. `docs/08_risks_and_open_questions.md` 更新 `R27` 的具体 T61 证据与 `T62` 缓解动作
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T61` blocked closeout 与 `T62` 边界
6. 因为这是同一 provenance blocker 的单次自动重试，若 `T62` 仍然 `BLOCK`，Captain 应停止自动推进并交由用户裁决

## D-2026-05-27-03

- 日期：`2026-05-27`
- 决策：接受 `T62_review.md` 的 `PASS`，标记 `T62` 完成，关闭 `R27`，并将当前唯一任务切换为 `T63: FR8 statcalib comparator gate review`

### 背景

`T62` 的目标非常窄：它不是新的 comparator 扩张任务，而是 `T61` clean-provenance blocker 的单次 blocking-only 自动重试。只有当以下条件同时成立时，`T62` 才能被接受为 `PASS`：

1. clean committed `main` preflight
2. one uninterrupted invocation
3. one T62-scoped run root only
4. launch / finish / `summary.json` commit identity match
5. bounded fairness signal persists without widening matrix or touching source/config semantics

### 依据

依据 `docs/review/T62_review.md`、`docs/evidence_packs/statcalib_fr8/statcalib_provenance_isolated_fairness_rerun.md` 与 `runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943/summary.json`，当前仓库事实是：

1. `T62` verdict = `PASS`
2. blocking issues = none
3. launch branch = `main`，launch `HEAD = e2773d3`
4. finish branch = `main`，finish `HEAD = e2773d3`
5. `summary.json git_commit = e2773d3`
6. `progress.jsonl` 无 duplicate `running` repeat key
7. bounded `statcalib` winner signal 在两场景上继续存在

### 结论

`T62` 可以完成，因为它真正关闭了 `T61` 被创建来修复的那个 blocker：当前仓库已经拥有一份 provenance-clean 的 bounded fairness sanity rerun。

但这仍然 **不等于**：

- `FR8` 已打开
- formal comparator ranking 已成立
- `.tflite` runtime 已验证
- real-board 已验证

因此，最小且更诚实的下一步不是直接执行 `FR8`，而是新建 `T63`，做一个 docs-only gate review，决定是否应当开启一个 bounded FR8 task，或者是否还需要一个更小的前置 prerequisite。

### 直接影响

1. `docs/04_task_board.md` 记录 `T62 -> PASS`，并将 `Current Unique Task` 切换到 `T63`
2. 新增任务包 `docs/tasks/Phase2/T63_fr8_statcalib_comparator_gate_review.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T63`
4. `docs/08_risks_and_open_questions.md` 将 `R27` 标记为已收口，同时保留 `R24` 与 `FR8` gate 边界
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T62` 收口与 `T63` 边界

## D-2026-05-27-04

- 日期：`2026-05-27`
- 决策：接受 `T63_review.md` 的 `PASS`，标记 `T63` 完成，并将当前唯一任务切换为 `T64: FR8 statcalib extension-lane bounded benchmark`

### 背景

`T63` 的目标不是产生新 benchmark evidence，而是回答一个更窄的问题：在 `T59` 到 `T62` 之后，仓库是否已经具备打开一个有界 `FR8` 执行任务的前提，还是还需要再做一个更小的前置 prerequisite。

依据 `docs/review/T63_review.md`、`docs/evidence_packs/statcalib_fr8/fr8_statcalib_comparator_gate_review.md` 和已有 `T59/T60/T61/T62` 本地证据：

1. `statcalib` separate-lane integration 已由 `T59` 建立
2. cross-mode leakage 与 regression hardening 已由 `T60` 关闭
3. provenance-clean bounded fairness sanity blocker 已由 `T62` 关闭
4. 当前剩余 gap 已不再是“能不能开 FR8”，而是“如何在不改写 frozen table 的前提下做一个诚实的 extension-lane task”

### Warning 处理

- `T63` verdict = `PASS`
- blocking issues = none
- `T63` review 没有新的 warning 项需要做 `accepted / deferred / rejected` 分类
- 因此本轮不新增 warning-derived risk

### 结论

`T63` 可以完成，但它只关闭 pre-FR8 gate-discussion lane，不是 `FR8` evidence 本身。

当前最小且更诚实的下一步是新建 `T64`：

- 保持 `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` 的 frozen 四场景 / 五模式协议边界
- 只把 `statcalib` 作为 separately labeled extension lane 加入
- 保持 `--paired-seeds` 与 `--repeats 2`
- 如需分块，只允许按 repeat range 分块
- 保持 clean provenance
- 明确禁止把 `T64` 写成 `.tflite`、real-board、paper-grade expansion，或 historical `T24` rewrite

### 直接影响

1. `docs/04_task_board.md` 记录 `T63 -> PASS`，并切换 `Current Unique Task` 到 `T64`
2. 新增任务包 `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T64`
4. `docs/08_risks_and_open_questions.md` 保持 `R27` 已收口，并将 `R24` 明确为 `T64` 的 scope/reporting constraint
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T63` 收口与 `T64` 边界

## D-2026-05-29-01

- 日期：`2026-05-29`
- 决策：接受 `T64_review.md` 的 `PASS_WITH_WARNINGS`，标记 `T64` 完成，并将当前唯一任务切换为 `T65: FR8 extension-lane consistency guard and report closeout`

### 背景

`T64` 的目标不是重写历史 frozen benchmark，而是在 locked four-scenario protocol 下，新增一份 clean-provenance 的 `statcalib` extension-lane bounded benchmark，并明确它仍然只是 mock-backed software-HIL evidence。

依据 `docs/review/T64_review.md`、`docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`、`cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml` 与 `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/*`，当前仓库事实是：

1. blocking issues = none
2. T64 只新增了一个 task-scoped derived config 和一个 T64-scoped run root
3. launch / finish / `summary.json git_commit` 一致
4. locked four scenarios、frozen five-mode ordering、`statcalib` appended-last、`paired_seeds`、`repeats=2` 都被保留
5. T64 的 frozen five-mode subset 与历史 `T24` 在 20 个 frozen comparison rows 上完全一致
6. 但 T64 结果文档仍有两个明确的 report/provenance wording drift

### Warning 分类

本次 warning 分类如下：

1. `N1` execution-shape wording drift in `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md` = `deferred`
2. `N2` finish-timestamp provenance wording drift in `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md` = `deferred`
3. `N3` extension-lane over-interpretation boundary = `deferred`

对应风险处理：

1. `N1` 与 `N2` 写入 `R28`
2. `N3` 继续写入 `R24`

### 结论

`T64` 可以完成，但它关闭的是一份 bounded FR8 extension-lane benchmark 缺口，而不是 paper-grade expanded benchmark、`.tflite` validation、real-board validation 或完整 calibration comparator validation。

当前最小且更诚实的下一步不是继续跑新 benchmark，而是新增 `T65`：

1. 修正 T64 报告措辞与 provenance 归因
2. 增加一个轻量 audit helper
3. 增加 focused regression coverage
4. 产出一份显式 consistency-audit doc

### 直接影响

1. `docs/04_task_board.md` 记录 `T64 -> PASS_WITH_WARNINGS`，并切换 `Current Unique Task` 到 `T65`
2. 新增任务包 `docs/tasks/Phase2/T65_fr8_extension_lane_consistency_guard_and_closeout.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T65`
4. `docs/08_risks_and_open_questions.md` 保持 `R24` 打开，并新增 `R28`
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T64` 收口与 `T65` 边界
## D-2026-05-29-02

- 日期：`2026-05-29`
- 决策：接受 `T65_review.md` 的 `PASS_WITH_WARNINGS`，标记 `T65` 完成，并将当前唯一任务切换为 `T66: FR8 statcalib sensitivity bounded benchmark`

### 背景

`T65` 的目标不是新增实验，而是把 `T64` 的 bounded FR8 extension-lane result pack 做成一个更强的 self-audited artifact。根据 `docs/review/T65_review.md`、`docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`、`cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py` 与 `tests/test_fr8_extension_lane_consistency.py`：

1. T64 报告中的 execution-shape wording drift 已修正
2. finish-timestamp provenance wording drift 已修正
3. 仓库现在有一个轻量 audit helper，可对照 `summary.json`、`launch_plan.json`、`progress.jsonl` 与 T24 frozen-subset anchor 做一致性检查
4. focused regression coverage 已补上
5. T65 本身未创建新的 run root，也未改写任何历史 `runs/` 文件

### Warning 分类

本次 warning 分类如下：

1. `N1` mixed-diff scope acceptance depends on explicit user/captain clarification = `accepted`
2. `N2` T64-specific audit helper is intentionally narrow, not generic FR8 framework = `accepted`
3. `N3` review wording should have stated the clarification dependency more explicitly = `accepted`

因此：

1. 无新的 `deferred`
2. 无新的 `rejected`
3. 不新增由 warning classification 触发的新 risk

### 结论

`T65` 可以完成，并且它关闭的是 `R28`：T64 result pack 的 report/artifact consistency gap 已通过显式 audit helper、focused tests 与 bounded audit doc 收紧。

但 `T65` 不关闭 `R24`。当前更重要的剩余问题不再是报告措辞，而是 T64 中 `statcalib` 的 bounded 优势是否具有最小程度的 robustness，还是仅仅是一个狭窄 heuristic point。

因此当前最小且更诚实的下一步不是继续做 docs-only 修补，也不是直接跳到 `.tflite`、real-board 或 theory lane，而是新增 `T66`：

1. 保持 T24 frozen table 作为锚点
2. 保持 locked four-scenario mainline protocol
3. 仅对 statcalib 做一个预声明的小型 sensitivity grid
4. 保持 clean provenance
5. 不允许改动 statcalib/runtime 语义

### 直接影响

1. `docs/04_task_board.md` 记录 `T65 -> PASS_WITH_WARNINGS`，并切换 `Current Unique Task` 到 `T66`
2. 新增任务包 `docs/tasks/Phase2/T66_fr8_statcalib_sensitivity_bounded_benchmark.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T66`
4. `docs/08_risks_and_open_questions.md` 关闭 `R28`，并保持 `R24` 为当前主导风险
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T65` 收口与 `T66` 边界

## D-2026-06-01-01

- 日期：`2026-06-01`
- 决策：接受 `T66_review.md` 的 `PASS_WITH_WARNINGS`，标记 `T66` 完成，并将当前唯一任务切换为 `T67: FR8 statcalib teacher-anchor dependence bounded benchmark`

### 背景

`T66` 的目标不是继续做 report wording hardening，而是在不改 statcalib/runtime semantics 的前提下，验证 `T64` extension-lane gain 是否能穿过一个预声明的局部 sensitivity grid。

依据 `docs/review/T66_review.md`、`docs/evidence_packs/statcalib_fr8/statcalib_sensitivity_bounded_benchmark.md`、`cnn_fpga/config/p4_multiscenario_statcalib_sensitivity.yaml` 与 `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906/*`，当前仓库事实是：

1. blocking issues = none
2. `T66` 只有一个新 run root，且 `launch commit = finish commit = summary.json git_commit = ad981bb`
3. `4 scenarios x 7 modes x repeats=2` 的 bounded matrix 完整收口，`missing_runs = []`
4. `T66` 证明 `T64` 的 statcalib gain 不是单一点 fluke，但它没有把 statcalib 变成 mature calibration comparator
5. review 中的主要剩余问题已经从 local sensitivity 转成 teacher-anchor dependence

### Warning 分类

本次 warning 分类如下：

1. `N1` duplicate-running progress-log artifact after same-run-root timeout relaunch = `accepted`
2. `N2` aggregate-best vs stability-best split = `deferred -> R24`
3. `N3` `static_bias_theta / statcalib_high_threshold` best row still carries aggregate `statcalib_status = mixed` = `deferred -> R24`

因此：

1. 没有新的 `rejected`
2. 没有新的独立 risk 编号
3. `R24` 继续保留并被 `T66` 进一步细化

### 结论

`T66` 可以完成，但它关闭的是一个 bounded local-grid robustness gap，而不是 `R24` 本身。

当前最小且更诚实的下一步不应再做一个简单 docs follow-up，也不应跳到 `.tflite`、real-board、training 或 theory lane，而应新增 `T67`：

1. 保持 `T24` frozen table 为锚点
2. 保持 locked four-scenario mainline protocol
3. 只复用 `T66` 最强的两个 statcalib parameter points
4. 只改变 `teacher_mode`
5. 保持 clean provenance 与 `--paired-seeds`
6. 用一个 grouped summary pack 回答 teacher-anchor dependence 问题

### 直接影响

1. `docs/04_task_board.md` 记录 `T66 -> PASS_WITH_WARNINGS`，并切换 `Current Unique Task` 到 `T67`
2. 新增任务包 `docs/tasks/Phase2/T67_fr8_statcalib_teacher_anchor_dependence_bounded_benchmark.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T67`
4. `docs/08_risks_and_open_questions.md` 保持 `R24` 打开，并写入 `T66` 的 deferred warning 细节
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T66` 收口与 `T67` 边界
## D-2026-06-05-01

- 日期：`2026-06-05`
- 决策：接受 `T67_review.md` 的 `PASS_WITH_WARNINGS`，标记 `T67` 完成，并将当前唯一任务切换为 `T68: FR8 statcalib generated-only robustness bounded benchmark`

### 背景

`T67` 的目标不是扩写主表，也不是继续做 teacher breadth，而是回答一个更具体的问题：`T64/T66` 的 bounded statcalib gain 是否只是绑在 `teacher_mode=ukf` 上成立。

依据 `docs/review/T67_review.md`、`docs/evidence_packs/statcalib_fr8/statcalib_teacher_anchor_bounded_benchmark.md` 与 `runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718/*`，当前仓库事实是：

1. blocking issues = none
2. `T67` 只产生了一个新 run root，且 `launch commit = finish commit = summary.json git_commit = 84f4468`
3. 所有 6 条 statcalib teacher-anchor 变体都在 4 个锁定场景里同时优于两个 frozen anchors
4. 非 `ukf` teacher 在两个参数点内都整体排在 `ukf` teacher 前面
5. 但仍有两条 comparison row 保持 `mixed`

### Warning 分类

1. `N1` source-worktree scope-external PDF diff but clean-clone launch preserved provenance = `accepted`
2. `N2` helper tie-label nuance for equal mean LER = `accepted`
3. `N3` two comparison rows remain `mixed` = `deferred -> R24`

### 结论

`T67` 可以完成，但它关闭的是 gross teacher-anchor dependence gap，而不是 `R24` 本身。

当前最小且更诚实的下一步不应是 docs-only follow-up，也不应跳到 `.tflite`、real-board、training 或 theory lane，而应新增 `T68`：

1. 保持 `T24` frozen table 为锚点
2. 保持 locked four-scenario mainline protocol
3. 只聚焦最强的非 `ukf` teacher anchors
4. 只跑一个预声明 threshold-only candidate matrix，专门回答 generated-only robustness 问题
5. 保持 clean provenance 与 `--paired-seeds`

### 直接影响

1. `docs/04_task_board.md` 记录 `T67 -> PASS_WITH_WARNINGS`，并切换 `Current Unique Task` 到 `T68`
2. 新增任务包 `docs/tasks/Phase2/T68_fr8_statcalib_generated_only_robustness_bounded_benchmark.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T68`
4. `docs/08_risks_and_open_questions.md` 保持 `R24` 打开，并把 `T67` 的 deferred warning 继续并入其中
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T67` 收口与 `T68` 边界

## 2026-06-08 Captain Decision: accept T68 and open T69

- 决策：接受 `T68_review.md` 的 `PASS_WITH_WARNINGS`，标记 `T68` 完成，并将当前唯一任务切换为 `T69: FR8 statcalib clean-winner tie-break bounded benchmark`

### 事实基础

依据 `docs/review/T68_review.md`、`docs/evidence_packs/statcalib_fr8/statcalib_generated_only_robustness_bounded_benchmark.md`、`docs/worker_summary/T68_worker_summary.md`、`runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723/*`，以及本轮重新执行的：

1. `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_generated_only.py`
2. `python -m unittest tests.test_statcalib_generated_only_summary`
3. `python -m cnn_fpga.benchmark.summarize_statcalib_generated_only --run-dir runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723`

当前仓库事实是：

1. `T68` 只产生了一个新 run root：`runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723`
2. provenance 闭合：`launch HEAD = finish HEAD = summary.json git_commit = bda8f2b`
3. `missing_runs == []`
4. comparison rows = `40`
5. 所有 comparison rows 都满足 `coverage = 1.0` 且 `completed_repeats = 2`
6. 存在 4 个 full generated-only winners：
   - `statcalib_window_variance_t001`
   - `statcalib_window_variance_t003`
   - `statcalib_window_variance_t005`
   - `statcalib_ekf_t001`
7. strongest clean answer 不是唯一 winner，而是 `window_variance_t001 = t003 = t005`
8. `worst-case-best` 集合还额外包含 `window_variance_t010`
9. 预声明小网格里仍有部分候选带 `mixed` 行，因此 T68 关闭的是 existence question，不是全网格 clean closure

### Warning 分类

1. `N1` full generated-only winner set remains a tie, not a unique final threshold = `deferred -> R24`
2. `N2` some predeclared candidates remain `mixed` even though the bounded existence question is closed = `deferred -> R24`
3. `N3` clean short-path clone launch boundary must remain visible in downstream retellings = `accepted`

### 结论

`T68` 可以完成，但它关闭的是 bounded generated-only existence gap，而不是 `R24` 本身。

当前最小且更诚实的下一步不应回到 docs-only follow-up，也不应跳到 `.tflite`、real-board、training 或 theory lane，而应新增 `T69`：

1. 保持 `T24` frozen table 为锚点
2. 保持 locked four-scenario mainline protocol
3. 只聚焦 `T68` 的 4 个 full generated-only winners 与 2 个 frozen anchors
4. 用略强于 `T68` 的 repeat budget 去回答 clean-winner tie set 是否会收缩为一个更稳定的 reference point
5. 如果 tie 仍然存在，则把“tie set 就是最终诚实答案”写实，而不是强行压扁成单 winner

### 直接影响

1. `docs/04_task_board.md` 记录 `T68 -> PASS_WITH_WARNINGS`，并切换 `Current Unique Task` 到 `T69`
2. 新增任务包 `docs/tasks/Phase2/T69_fr8_statcalib_clean_winner_tiebreak_bounded_benchmark.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T69`
4. `docs/08_risks_and_open_questions.md` 保持 `R24` 打开，并把 `T68` 的 deferred warning 继续并入其中
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T68` 收口与 `T69` 边界

## D-2026-06-08-02

- 日期：`2026-06-08`
- 决策：接受并落地 `PSE0：并行 sidecar 扩展实验治理设置`，允许后续规划多个 sidecar worktree，但不在本任务中创建 worktree 或执行实验

### 背景

用户明确表示可以接受多个 worktree，甚至十几个，并希望在 `T69` 这类 2-4 天主线长跑之外并行准备更完备的扩展实验路线。

当前仓库事实是：

1. `T69` 是当前唯一主线任务，且任务包已经存在。
2. `T69` 尚未执行。
3. `T24` 仍是 authoritative frozen ranked table。
4. `T64/T65/T66/T67/T68` 都只是 mock-backed software-HIL extension-lane evidence。
5. `R24` 仍打开，当前由 `T69` 聚焦 clean-winner tie-break 问题。
6. `docs/deep_research_reports/GPT-Pro有关扩展实验的建议.md` 建议可以并行开扩展路线，但必须作为 sidecar extension lane，而不是重写主线。

### 依据

1. `docs/04_task_board.md` 要求后续 Worker 只能领取 `Current Unique Task` 指向的单个任务包，且每个任务包必须有 Allowed files / Forbidden scope / Verification / Docs to update。
2. `docs/06_repo_noise_governance.md` 要求历史 run roots 不得被重写、重标或 cleanup。
3. `docs/08_risks_and_open_questions.md` 的 `R24` 要求 statcalib 继续作为 separately labeled extension lane，不得升级为 mature comparator。
4. GPT-Pro 扩展建议明确要求 frozen anchor、sidecar artifact、promotion gate、mock-backed boundary、`.tflite` boundary 和 statcalib boundary。

### 结论

项目可以开始规划多个 sidecar worktree，但必须满足：

1. `T69` 仍是当前唯一主线任务。
2. sidecar worktree 不得写入 `T69` run root。
3. sidecar 输出必须写入 `runs/sidecar/<lane_id>/...`。
4. sidecar 分支使用 `codex/sidecar-*`。
5. sidecar 输出只能作为候选证据，不能进入主线事实口径。
6. 从 sidecar 晋升到主线候选必须经过新的 Captain gate 和任务包。

### 直接影响

1. 新增 `docs/tasks/Phase2/PSE0_parallel_sidecar_extension_governance_setup.md`
2. 新增 `docs/sidecar/parallel_sidecar_extension_governance.md`
3. 新增 `docs/sidecar/parallel_sidecar_worktree_plan.md`
4. `docs/04_task_board.md` 增加 PSE0 记录，但不改变 `Current Unique Task = T69`
5. `docs/06_repo_noise_governance.md` 增加 sidecar run-root 规则
6. `docs/07_handoff.md` 增加 sidecar governance handoff note
7. `docs/08_risks_and_open_questions.md` 增加 `R29`，用于追踪 sidecar worktree 污染主线事实口径的风险

## D-2026-06-08-03

- 日期：`2026-06-08`
- 决策：按 Wave A 创建四个 sidecar worktree，并在各自 worktree 内先写中文 `S0_design` 任务包；main 分支 `T69` 执行继续保持独立

### 背景

用户确认可以接受多个 worktree，并明确要求在主线 `T69` 独立保留的前提下，创建 Wave A 的多个 worktree，并分别先写 `S0_design` 任务包。

### 结论

Wave A 采用短路径项目内 worktree：

1. `.wt/tcn` -> `codex/sidecar-temporal-tcn-residual`
2. `.wt/teach` -> `codex/sidecar-adaptive-teacher-replay`
3. `.wt/bank` -> `codex/sidecar-gain-scheduled-bank-sim`
4. `.wt/ctrl` -> `codex/sidecar-atomic-commit-rollback`

第一次尝试使用 `.worktrees/<long-name>` 时，完整 checkout 触发 Windows `Filename too long`；最终使用 `.wt/<short>` 并配合 `core.longpaths=true` 创建。`.wt/` 与 `.worktrees/` 均加入 `.gitignore`。

### 直接影响

1. Wave A worktree 已创建，但均未运行实验。
2. 四个 `S0_design` 任务包已写入各自 worktree。
3. 未创建 `runs/sidecar`。
4. 未改变 `T69` 当前唯一主线任务，也未执行 `T69`。
5. 后续 sidecar 执行必须先从对应 `S0_design` 升级为独立 `S1` 或 contract-test 任务包。

## 2026-06-10 Captain Decision: accept T69 and open T70

- 决策：接受 `docs/review/T69_review.md` 的 `PASS_WITH_WARNINGS`，标记 `T69` 完成，并将当前唯一任务切换为 `T70: FR8 statcalib bounded closure pack and promotion gate`

### 事实基础

依据 `docs/review/T69_review.md`、`docs/evidence_packs/statcalib_fr8/statcalib_clean_winner_tiebreak_bounded_benchmark.md`、`docs/worker_summary/T69_worker_summary.md`、`runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358/*`，以及本轮重新执行的：
1. `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/benchmark/summarize_statcalib_clean_winner_tiebreak.py`
2. `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_clean_winner_tiebreak_summary`
3. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.summarize_statcalib_clean_winner_tiebreak --run-dir runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358`

当前仓库事实是：

1. `T69` 只产生了一个新 run root：`runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358`
2. provenance 闭合：launch `HEAD = finish HEAD = summary.json git_commit = 1dbfbc3`
3. `missing_runs == []`
4. comparison rows = `24`
5. 所有 comparison rows 都满足 `coverage = 1.0` 且 `completed_repeats = 4`
6. 四个冻结 clean-winner candidates 都仍然是 full generated-only：
   - `statcalib_window_variance_t001`
   - `statcalib_window_variance_t003`
   - `statcalib_window_variance_t005`
   - `statcalib_ekf_t001`
7. 三个 `window_variance` 候选在四个锁定场景中全部精确打平
8. 三个 `window_variance` 候选在四个锁定场景中全部优于 `statcalib_ekf_t001`
9. final classification = `persistent_clean_tie_set`
10. unique clean reference point = `False`

### Warning 分类

1. `N1` persistent clean tie set is the honest bounded answer, not a unique final threshold = `accepted`
2. `N2` bounded-matrix-only conclusion must remain explicit in downstream retellings = `accepted`

### 结论

`T69` 可以完成，但它关闭的是 bounded clean-winner tie-break execution question，而不是 `R24` 本身。
当前最小且更诚实的下一步不应继续做另一个 threshold rerun，也不应跳到 `.tflite`、real-board、training、theory lane 或 sidecar promotion，而应新增 `T70`：
1. 只读复用 `T24/T64/T66/T67/T68/T69` 的历史 artifact 与结果文档
2. 构建一个 task-scoped closure helper 与 focused tests，而不是手写拼接结论
3. 产出一份单独的 FR8 bounded closure pack，统一写清 extension-lane 证据链到底证明了什么、没有证明什么
4. 明确给出 promotion / no-promotion gate：`T24` 不改写，`statcalib` 不升格为 mature comparator，不发明 unique threshold
5. 如果未来真的需要选一个单一阈值，必须另开一项预声明 selection-criterion task，而不是从 T69 静默外推

### 直接影响

1. `docs/04_task_board.md` 记录 `T69 -> PASS_WITH_WARNINGS`，并切换 `Current Unique Task` 到 `T70`
2. 新增任务包 `docs/tasks/Phase2/T70_fr8_statcalib_bounded_closure_pack_and_promotion_gate.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T70`
4. `docs/08_risks_and_open_questions.md` 保持 `R24` 打开，但把它缩窄为 overclaim/promotion 边界而非未完成的 tie-break 执行问题
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T69` 收口与 `T70` 边界

## 2026-06-10 Captain Decision：accept T70 and open T50

- 决策：接受 `docs/review/T70_review.md` 的 `PASS`，标记 `T70` 完成，并将当前唯一任务切换为 `T50: Training reproducibility and material-regeneration pack`

### 事实基础

依据 `docs/review/T70_review.md`、`docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`、`docs/worker_summary/T70_worker_summary.md`、`cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py`、`tests/test_fr8_statcalib_bounded_closure_pack.py`，以及本轮重新执行的：

1. `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py`
2. `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_fr8_statcalib_bounded_closure_pack`
3. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.build_fr8_statcalib_bounded_closure_pack`
4. `@(Get-ChildItem -LiteralPath 'runs\\p4_benchmark' -Directory | Where-Object { $_.Name -like 'T70*' }).Count`
5. `git diff --name-only -- runs`

当前仓库事实是：

1. `T70` 没有新建 run root，也没有重跑 benchmark。
2. helper、tests、closure pack 文档三者结论一致。
3. 当前最强 clean answer 仍然是：
   - `statcalib_window_variance_t001`
   - `statcalib_window_variance_t003`
   - `statcalib_window_variance_t005`
4. `unique_clean_reference_point_exists = False`
5. `promotion_gate = no_promotion_keep_extension_lane_only`
6. `unique_threshold_gate = future_selection_task_required`

### 结论

`T70` 可以完成，而且不需要 `PASS_WITH_WARNINGS` 收口，因为本轮没有发现需要继续保留到 verdict 级别的 warning 分类项。

在 `T70` 之后，主线最小而诚实的下一步不应继续做 FR8 自由转述，也不应直接跳到 `.tflite` / 真板：

1. `.tflite` runtime 仍受当前机器环境前提限制
2. 真板 smoke 仍受硬件宿主与 bitstream 前提限制
3. 当前更适合在 main 上继续补强的是训练复现与材料再生证据

因此新增 `T50`：

1. 只在 clean CPU-only 训练 lane 内工作
2. 构建 task-scoped training reproducibility pack helper 与 focused tests
3. 产出一个统一的训练材料台账与 bounded rerun/eval evidence pack
4. 强化训练链可复用性与 paper-material 证据，而不碰 `.tflite`、真板、benchmark 或 theory branch

### 直接影响

1. `docs/04_task_board.md` 记录 `T70 -> PASS`，并切换 `Current Unique Task` 到 `T50`
2. 新增任务包 `docs/tasks/Phase2/T50_training_reproducibility_and_material_regeneration_pack.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T50`
4. `docs/08_risks_and_open_questions.md` 保持 `R24` 打开，但把 carry-forward 口径切换为引用 `T70` closure pack，而不是继续自由转述 FR8 链
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T70` 收口与 `T50` 边界

## 2026-06-10 Captain Decision：accept T50 and open T48

- 决策：接受 `docs/review/T50_review.md` 的 `PASS`，标记 `T50` 完成，并将当前唯一任务切换为 `T48: True .tflite runtime smoke gate`

### 事实基础

依据 `docs/review/T50_review.md`、`docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`、`docs/worker_summary/T50_worker_summary.md`、`docs/for_human/T50_explanation.md`、`cnn_fpga/model/build_training_reproducibility_pack.py`、`tests/test_training_reproducibility_pack.py`，以及本轮重新执行的：

1. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m py_compile cnn_fpga/model/build_training_reproducibility_pack.py`
2. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m unittest tests.test_training_reproducibility_pack`
3. `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe -m cnn_fpga.model.build_training_reproducibility_pack --rerun-train-report artifacts/t50_training_repro_pack/reports/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c_train_report.json --rerun-eval-report artifacts/t50_training_repro_pack/reports/static_theta_v2/eval_test_20260610_195030.json`
4. `git diff --name-only -- runs`
5. `git diff --name-only -- artifacts/models/static_theta_v2 artifacts/reports/static_theta_v2 artifacts/models/runtime_b_residual_v1 artifacts/reports/runtime_b_residual_v1`
6. `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt`

当前仓库事实是：

1. helper、tests、主报告和 pack JSON 彼此一致。
2. `T50` 没有改写 `runs/`、canonical historical training artifacts、`requirements-recovery.txt` 或 `requirements-train-cpu-win-py312.txt`。
3. 当前仓库已经有一份 code-backed training/material pack，可统一引用：
   - canonical `static_theta_v2`
   - canonical `runtime_b_residual_v1`
   - preserved mainline model references
   - one clean CPU-only bounded train+eval rerun
4. 本轮 Captain 没有重跑完整训练，而是以 fresh `py_compile` / `unittest` / helper / diff evidence 复核现有 train+eval artifacts 与文档边界。

### 结论

`T50` 可以完成，而且不需要 `PASS_WITH_WARNINGS` 收口，因为本轮没有发现需要继续保留到 verdict 级别的 warning 分类项。

在 `T50` 之后，主线最小而诚实的下一步不应回到训练补缝或直接跳到真板：

1. `T50` 已经把主线训练材料与 clean CPU-only 训练边界收成一个可引用的 pack。
2. `T49` 仍受硬件宿主、bitstream 与 DMA contract 前提约束。
3. `T51/T52` 仍不应在 true `.tflite` runtime truth 未核验前重开 paper-positioning / prose gate。
4. 当前更适合在 main 上继续补强的是 `.tflite` runtime 真值：当前机器到底有没有真实解释器环境、preserved `.tflite` artifacts 到底能不能真实执行、artifact-vs-`.tflite` 偏差是否仍在可接受量级。

因此新增 `T48`：

1. 在 preserved `static_theta_v2` 材料边界内工作，而不是重训或扩 benchmark
2. 构建 task-scoped `.tflite` runtime gate helper 与 focused tests
3. 要求真实环境探测、preserved artifact 选择、bounded `evaluate_tflite` / `validate_export` 和显式 gate verdict
4. 即便结论是 `NO_GO`，也必须以真实 runtime 证据收口，而不能用 stub 或文档推测代替

### 直接影响

1. `docs/04_task_board.md` 记录 `T50 -> PASS`，并切换 `Current Unique Task` 到 `T48`
2. 新增任务包 `docs/tasks/Phase2/T48_true_tflite_runtime_smoke_gate.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T48`
4. `docs/08_risks_and_open_questions.md` 保持 `R11`、`R12` 打开，但将 carry-forward 口径分别升级为引用 `T50` training/material pack 与 `T48` true-runtime gate
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T50` 收口与 `T48` 边界

## 2026-06-10 Captain Decision：accept T48 and open T49

- 决策：接受 `docs/review/T48_review.md` 的 `PASS`，标记 `T48` 完成，并将当前唯一任务切换为 `T49: Real-board smoke execution gate`
- 说明：`T48` review 中的 non-blocking notes 不上升为 `PASS_WITH_WARNINGS`；它们被接受为 advisory carry-forward，仅提醒后续不要把 helper 默认参数错当成“无参直接复跑入口”，也不影响当前 gate 结果成立

依据 `docs/review/T48_review.md`、`docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`、`docs/worker_summary/T48_worker_summary.md`、`cnn_fpga/model/build_t48_true_tflite_runtime_gate.py`、`tests/test_t48_true_tflite_runtime_gate.py`，以及本轮重新执行的：

1. `.\.venvs\t39_train_cpu_py312\Scripts\python.exe -m py_compile cnn_fpga\model\build_t48_true_tflite_runtime_gate.py`
2. `.\.venvs\t39_train_cpu_py312\Scripts\python.exe -m unittest tests.test_t48_true_tflite_runtime_gate`
3. `.\.venvs\t39_train_cpu_py312\Scripts\python.exe -m cnn_fpga.model.build_t48_true_tflite_runtime_gate --env-probe-json artifacts\t48_true_tflite_runtime_gate\runtime_env_probe_tf221.json --compatibility-probe-json artifacts\t48_true_tflite_runtime_gate\preserved_tflite_load_probe_tf221.json --float-eval-report artifacts\t48_true_tflite_runtime_gate\reports\static_theta_v2\eval_tflite_test_20260610_211759.json --float-validate-report artifacts\t48_true_tflite_runtime_gate\reports\static_theta_v2\validate_export_tiny_cnn_20260319_151717_b87c6c227b57_20260610_211815.json --int8-eval-report artifacts\t48_true_tflite_runtime_gate\reports\static_theta_v2\eval_tflite_test_20260610_211830.json --int8-validate-report artifacts\t48_true_tflite_runtime_gate\reports\static_theta_v2\validate_export_tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_20260610_211845.json`
4. `.venvs\t48_tf221\Scripts\python.exe` 下对 selected float / int8 preserved `.tflite` 重新执行 `tf.lite.Interpreter(...).allocate_tensors()`
5. `git diff --name-only -- runs`
6. `git diff --name-only -- artifacts/models/static_theta_v2 artifacts/reports/static_theta_v2 artifacts/models/runtime_b_residual_v1 artifacts/reports/runtime_b_residual_v1`
7. `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt`

得出：

1. `T48` 的核心结论成立：当前机器上确实存在一个 isolated `tensorflow==2.21.0` 真 `.tflite` 运行时环境，可真实加载并执行 preserved `static_theta_v2` float / int8 `.tflite`。
2. `T48` 没有改写 `runs/`、canonical historical artifacts、`requirements-recovery.txt` 或 `requirements-train-cpu-win-py312.txt`。
3. `T48` 的 strongest supported claim 仍然是“isolated current-host true `.tflite` runtime truth”，而不是默认环境恢复、HIL closure、real-board validation 或 deployment closure。

`T48` 可以完成，而且不需要 `PASS_WITH_WARNINGS` 收口，因为当前 non-blocking notes 不影响 gate verdict 成立，也不要求在同一任务内继续 blocker-only 修补。

在 `T48` 之后，主线最小而诚实的下一步不应直接跳到 paper re-open，也不应直接把真板写成 ready：

1. `T48` 已经去掉了当前机器的软件侧 `.tflite` 运行时不确定性。
2. 剩余 deployment-boundary 大缺口已经切换为“当前宿主是否具备 defensible 的 real-board host / device / bitstream / AXI / DMA 前提”。
3. `T51/T52` 论文 reopen 仍然过早，因为 hardware-side truth 仍未过 gate。

因此新增 `T49`：

- 任务名：`T49: Real-board smoke execution gate`
- 任务包：`docs/tasks/Phase2/T49_real_board_smoke_execution_gate.md`
- 任务类型：有界当前宿主真板前提 gate 任务
- 允许结果：`GO_REAL_BOARD_SMOKE_EXECUTION_PRECONDITIONS_READY` 或三个显式 `NO_GO`
- 强边界：只允许 host fact probe、read-only device-path probe、AXI/DMA/placeholder 审计、task-scoped helper、focused tests 和显式 gate verdict；不得执行 MMIO/DMA/register 写入，不得跑 board benchmark，不得改写 `board_backend.py`

对应治理同步：

1. `docs/04_task_board.md` 记录 `T48 -> PASS`，并切换 `Current Unique Task` 到 `T49`
2. 新增任务包 `docs/tasks/Phase2/T49_real_board_smoke_execution_gate.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T49`
4. `docs/08_risks_and_open_questions.md` 保持 `R13/R14` 打开，并将 `R12` 收窄为“isolated current-host true runtime 已确认，但 default env / portability / deployment 未闭环”
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T48` 收口与 `T49` 边界
## 2026-06-10 Captain Decision：accept T49 and open T71

- 决策：接受 `docs/review/T49_review.md` 的 `PASS_WITH_WARNINGS`，标记 `T49` 完成，并将当前唯一任务切换为 `T71: Real-board gate regeneration and host-transfer pack`

依据 `docs/review/T49_review.md`、`docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`、`docs/worker_summary/T49_worker_summary.md`、`cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`、`tests/test_t49_real_board_smoke_execution_gate.py`，以及本轮重新执行的：

1. `python -m py_compile cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
2. `python -m unittest tests.test_t49_real_board_smoke_execution_gate`
3. `python cnn_fpga/hwio/build_t49_real_board_smoke_gate.py --output-json artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.recheck.json`

得到以下收口判断：

1. `T49` 的核心结论成立：当前机器上的诚实 gate verdict 是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。
2. `T49` 没有执行任何 real-board smoke，也没有越界修改 `runs/`、`board_backend.py`、`axi_map.py`、`dma_client.py` 或治理文档。
3. `T49` 最强可支持表述仍然只是“当前宿主的只读 real-board 前提 gate 已被显式核查并得到 `NO_GO`”，而不是 real-board validation、HIL closure 或 deployment closure。

`T49` 之所以不是 `PASS`，而是 `PASS_WITH_WARNINGS`，是因为当前 current-host `NO_GO` 虽然成立，但 review 指出的三个问题确实应该继续保留：

- `W1`：`device_path_truth` 当前只按 openable path 总数计数，没有强制 `mmio + dma` 角色同时满足
- `W2`：缺少 role-aware regression 与 `T49` checked-in artifact replay regression
- `W3`：host/device/code-side probe 结果还没有收敛成一个单一 checked-in 的只读再生成入口

这些问题都不推翻 `T49` 当前宿主 `NO_GO` 的真实性，因此不构成 blocker；但它们会影响 future-host 复用该 gate 的可靠性，因此统一按 `deferred -> R30` 收口。

在 `T49` 之后，主线不应直接打开 `T37`，也不应跳回 paper re-open：

1. `T37` 仍被当前宿主缺少可读设备路径、缺少绑定到当前宿主的 bitstream/RTL/DMA contract 事实、以及 placeholder repo execution path 所阻塞。
2. `T51/T52` 仍然过早；在 real-board lane 尚未形成可再生成的 checked-in gate 入口前，继续 paper re-open 只会再次压扁 deployment boundary 诚实性。

因此新增 `T71`：

- 任务名：`T71: Real-board gate regeneration and host-transfer pack`
- 任务包：`docs/tasks/Phase2/T71_real_board_gate_regeneration_and_host_transfer_pack.md`
- 任务目标：
  - 把 `device_path_truth` 改成 role-aware `mmio + dma` ready 判定
  - 增加 checked-in 的只读 artifact 再生成入口
  - 补 `T49` artifact replay 与 current-host regeneration regression
  - 形成 future-host 可直接复用的 host-transfer pack

直接影响：

1. `docs/04_task_board.md` 记录 `T49 -> PASS_WITH_WARNINGS`，并将当前唯一任务切换到 `T71`
2. 新增任务包 `docs/tasks/Phase2/T71_real_board_gate_regeneration_and_host_transfer_pack.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T71`
4. `docs/08_risks_and_open_questions.md` 新增 `R30`，并把 `W1/W2/W3` 统一收口到该风险
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T49` 收口与 `T71` 边界

## 2026-06-12 Captain Decision：accept T77 and open T78

- 决策：接受 `docs/review/T77_review.md` 的 `PASS_WITH_WARNINGS`，标记 `T77` 完成，关闭 `R34`，新增 `R35`，并将当前唯一任务切换为 `T78: 论文 note-draft 非结果层校准、statcalib 层级降权与排版 warning 收口`

依据 `docs/review/T77_review.md`、`docs/paper_materials/paper_note_results_sync_manifest.md`、`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`、`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log`、`docs/paper_notes/README.md`、`docs/paper_materials/README.md`，以及 `T77` 已产出的 note 编译刷新与 traceability 文档，可得到以下收口判断：

1. `T77` 的核心目标已经成立：当前仓库已经把 `T74/T75/T76` 锁定的结果层材料同步进现有 note-draft，并把 `T76` preview-source / stable-ID traceability 粒度补干净。
2. `T77` 没有越界成 benchmark、训练、`.tflite`、真板、治理文档或 theory 分支大范围改写任务。
3. `R34` 的原始问题已经被解决，因此不应继续把旧 traceability/schema 缺口挂成开放风险。

`T77` 之所以不是 `PASS`，而是 `PASS_WITH_WARNINGS`，是因为 review 指出的四条问题都应该继续保留：

- `N1`：整份 `.tex` 仍含未校准的非结果层历史段落
- `N2`：`statcalib` 在 `Numerical Results` 中的视觉层级仍偏高
- `N3`：note `.log` 仍有 `Underfull \hbox` 排版 warning
- `N4`：section-scope 证明仍主要依赖 manifest / `% T77-SOURCE` 注释而非更机械的审计

这些问题都不推翻 `T77` 当前“结果层已同步、traceability 已补强”的真实性，因此不构成 blocker；但它们会直接影响 note 是否能被当作更完整、对外可复用的 paper-facing 草稿，因此统一按 `deferred -> R35` 收口。

在 `T77` 之后，主线不应直接打开 full-manuscript reopen，也不应跳回 real-board：

1. 当前暂无 `Linux + FPGA` 硬件宿主，`T37` 仍受 `R33` 约束。
2. `T77` 只证明了结果层同步和 traceability hardening，不等于整份 note fully calibrated。
3. 如果现在直接重开 prose 扩写，会把局部收口误讲成整体校准完成。

因此新增 `T78`：

- 任务名：`T78: 论文 note-draft 非结果层校准、statcalib 层级降权与排版 warning 收口`
- 任务包：`docs/tasks/Phase2/T78_paper_note_alignment_statcalib_hierarchy_and_layout_closeout.md`
- 任务类型：高强度 docs-only note 校准 / hierarchy / layout closeout 任务
- 强边界：只允许在 note / paper-material 边界内调整非结果层 wording、`statcalib` 视觉层级、LaTeX 排版与 section-scope 审计；不得重开 benchmark、`.tflite`、真板、治理文档或 theory 分支大范围改写

直接影响：

1. `docs/04_task_board.md` 记录 `T77 -> PASS_WITH_WARNINGS`，关闭 `R34`，并切换 `Current Unique Task` 到 `T78`
2. 新增任务包 `docs/tasks/Phase2/T78_paper_note_alignment_statcalib_hierarchy_and_layout_closeout.md`
3. `docs/07_handoff.md` 切换当前 worker-facing task package 到 `T78`
4. `docs/08_risks_and_open_questions.md` 关闭 `R34`、新增 `R35`，并同步当前唯一任务口径
5. `docs/00_project_snapshot.md`、`docs/01_legacy_audit.md`、`docs/02_experiment_plan.md`、`docs/03_hil_p4_boundary_audit.md`、`docs/06_repo_noise_governance.md` 同步 `T77` 收口与 `T78` 边界
