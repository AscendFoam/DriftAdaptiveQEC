# Reality Recovery Retired

> **状态提示：** 本目录已经退役；下面的 T72 和旧治理链只描述 2026-06-11 附近状态。当前入口见 [`../README.md`](../README.md)。

本目录原用于 `T44: Research Reality Recovery Mode` 附近的真实性冻结与证据复核。随着后续主线已经继续完成 `T45`、`T48`、`T49`、`T57`、`T58`、`T64`-`T70`、`T71` 并切换到 `T72`，原始 reality-recovery 文件中的许多状态已经过时。

因此，本目录自 2026-06-11 起退役，不再作为当前项目事实来源。

## 当前权威入口

以下是本目录退役时使用的历史治理入口：

- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/02_experiment_plan.md`（Part II 为后续开发计划唯一入口）

特别注意：

- 当时的唯一任务以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为准；当前任务改从 `docs/new_task_board.md` 读取。
- `T48` 已改变旧文件中关于 true `.tflite` runtime 的 current-host 状态，但不等于默认环境或部署闭环完成。
- `T49/T71/T72` 已改变旧文件中关于 real-board gate 的状态，但 current-host verdict 仍是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`，不是真板验证完成。
- `T64`-`T70` 已改变旧文件中关于 `statcalib` 的状态，但 `statcalib` 仍是 mock-backed software-HIL extension lane，不是成熟主线 comparator。

## 归档位置

原文件已整体移动到：

- `docs/legacy_context/reality_recovery_2026-05/00_freeze_snapshot.md`
- `docs/legacy_context/reality_recovery_2026-05/01_claim_evidence_table.md`
- `docs/legacy_context/reality_recovery_2026-05/02_code_truth_audit.md`
- `docs/legacy_context/reality_recovery_2026-05/03_experiment_reproducibility_audit.md`
- `docs/legacy_context/reality_recovery_2026-05/04_figure_and_result_ledger.md`
- `docs/legacy_context/reality_recovery_2026-05/05_paper_claim_risk_table.md`
- `docs/legacy_context/reality_recovery_2026-05/06_human_brief.md`

这些文件仍可用于理解 2026-05 的真实性复核起点、T44/T45/T46/T47/T57/T58 等历史任务语境，但不能直接用于描述当前完成态。
