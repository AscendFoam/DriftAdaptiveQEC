# T84 Review

审查方式：只读审查 `T84` allowlist 范围内的 docs diff，复核 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md` 与任务包边界；检查主 note 中 `% T80-REOPEN`、`% T81-CALIBRATION`、`% T82-SUPPORT`、`% T83-CLOSEOUT`、`% T84-POLISH` 标记链，三份 `T84` 台账、两份 README 登记，以及当前主机上的 LaTeX 编译日志。未重跑任何 benchmark、训练、`.tflite` smoke 或 real-board 执行。

Verdict: `PASS_WITH_WARNINGS`

## Blocking issues

- 无。

## Non-blocking issues

- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 的 `Conclusion` 段仍保留一句旧状态口径：`The remaining writing work is to translate these internal layers into a final reader-facing polish pass...`。而 `T84` 本身就是这轮 bounded final polish，这句话会把“本轮已做完的读者化润色”写成“后续仍待执行的工作”，与 `paper_bounded_final_polish_change_map.md`、`T84_worker_summary.md` 以及本轮 closeout 口径存在轻微状态滞后。
- 当前 worktree 在进入 `T84` 前已经存在与本任务无关的脏状态，因此本轮与后续审查都必须继续使用 allowlist-scoped status/diff，不能把整仓 `git status` 直接当成 `T84` 改动清单。Worker 这次实际采用了 scoped 检查，这一点本身不是违规，但需要持续保留该审查习惯。
- 本轮 compile 结论只证明当前主机上的 `TeX Live 2024 + latexmk` 可以刷新这份 note；它不是所有宿主机、所有 LaTeX 路径或所有后续写作分支的普适结论。

## Missing tests

- 无阻塞性缺口。对 `T84` 这类 docs-only reader-facing polish 任务，关键验证已经覆盖：
  - allowlist-scoped diff/status；
  - `% T80-REOPEN` / `% T81-CALIBRATION` / `% T82-SUPPORT` / `% T83-CLOSEOUT` 保留；
  - `% T84-POLISH` 与 `paper_bounded_final_polish_change_map.md` 的 `touched_sections` 对齐；
  - 三份 `paper_materials` 台账的必需字段齐全；
  - 本地 `latexmk -g` 编译成功，`.log` 未检出 `Underfull`、`Overfull`、`LaTeX Warning`、`undefined`、`Citation`。
- 若后续还要继续做同类 note closeout，可补一个很小的机械检查：同时核对 `% T84-POLISH` 标记、change map 列表，以及被标记 section 内是否还残留把本轮任务写成“未来待做”的句子。

## Suspicious implementation details

- 未发现伪实现、mock、stub、hardcode，亦未发现对历史 `runs/` / `artifacts/` 事实的改写。
- 未发现越过 `T84` guardrail 的写强行为。以下边界仍被保留：
  - 冻结四场景 benchmark 仍只是 mainline frozen reference；
  - `FR6/FR7` 仍只是 descriptive support；
  - `FR8/statcalib` 仍只是 extension lane / no-promotion / no unique clean threshold；
  - training/material 仍只是 canonical chain intact + one clean CPU-only bounded rerun；
  - `.tflite` 仍只是 isolated current-host true runtime；
  - real-board 仍只是 read-only gate / regeneration / provenance，且当前 host 仍不能进入 board execution；
  - 缺失 `Linux + FPGA` host 的 hardware-dependent surface 仍显式 `blocked`。

## Recommended next action

- 以 `PASS_WITH_WARNINGS` 接受 `T84`，不要 block。
- 若 Captain 认为值得消化这条 warning，只开一个极小的 docs-only cleanup：把 `Conclusion` 中“remaining writing work is to translate ... into a final reader-facing polish pass”改写成“本轮已完成读者化翻译/装配，后续只剩更高层人工终修或独立 bounded task”，且不得顺手扩写 submission-ready、deployment closure 或 hardware-ready 叙事。
