# T86 Review

审查方式：只读审查 `T86` allowlist 范围内的 docs diff 与 note 编译产物，复核 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md` 的当前唯一任务口径；检查主 note 中 `% T80-REOPEN`、`% T81-CALIBRATION`、`% T82-SUPPORT`、`% T83-CLOSEOUT`、`% T84-POLISH`、`% T85-PREFLIGHT` 保留情况与 `% T86-ASSEMBLY` 新增位置；检查 4 份 `T86` submission-pack 台账、2 份 README 登记，以及当前主机上的 LaTeX 编译与 `.log` 关键字扫描结果。未重跑任何 benchmark、训练、`.tflite` smoke 或 real-board 流程。

Verdict: `PASS`

## Blocking issues

- 无。

## Non-blocking issues

- 当前 worktree 在进入 `T86` 前已经存在与本轮无关的脏状态，而且相对 `HEAD` 的 note / README diff 仍混有部分前序 `T84/T85` 未提交内容。因此，`T86` 的真实范围必须继续通过 allowlist-scoped status/diff、当前文件内容与 `% T86-ASSEMBLY` 标记链来判断，不能把整仓 diff 直接当成 `T86` 输出。
- `paper_submission_pack_assembly_manifest.md`、`paper_submission_surface_route_map.md`、`paper_submission_exclusion_register.md`、`paper_submission_author_handoff.md` 只是在当前主线证据边界内把可装配 surface、路由位置和显式排除项写清楚，不等于 submission-ready pack 已完成，也不等于 blocked surface 已解除。
- 本轮 note 编译成功只说明当前主机上的 `TeX Live 2024 + latexmk` 可以刷新这份 note；它不是其他宿主环境、其他写作分支或期刊模板环境的普适编译结论。
- allowlist `git diff --check` 已无内容级报错；剩余输出只有 Windows 当前 working-copy 行尾提示（`LF will be replaced by CRLF`）和 `C:\\Users\\26410/.config/git/ignore` 访问告警，属于宿主机 Git 配置噪声，不构成 `T86` 文档缺陷。

## Missing tests

- 无阻塞性缺口。对 `T86` 这类 docs-only submission-pack assembly 任务，关键验证已经覆盖：
  - allowlist-scoped status/diff 核对；
  - `% T80-REOPEN` / `% T81-CALIBRATION` / `% T82-SUPPORT` / `% T83-CLOSEOUT` / `% T84-POLISH` / `% T85-PREFLIGHT` 标记保留；
  - `% T86-ASSEMBLY` 已覆盖 `Numerical Results`、`Discussion`、`Conclusion` 三个实际修改 section；
  - `paper_submission_pack_assembly_manifest.md`、`paper_submission_surface_route_map.md`、`paper_submission_exclusion_register.md` 具备任务包要求的字段；
  - `paper_submission_author_handoff.md` 明确列出了仍不可升级的四类边界；
  - 本地 `latexmk -g -pdf -synctex=1 -interaction=nonstopmode -halt-on-error CNN_FPGA_GKP_theory_note_draft.tex` 编译成功；
  - `.log` 关键字扫描未检出 `Underfull`、`Overfull`、`LaTeX Warning`、`undefined`、`Citation`。
- 若后续还要继续做同类 assembly 任务，可补一个很小的机械检查：同时核对 `% T86-ASSEMBLY` 标记、surface route map 中实际触达的 section，以及 author handoff 中列出的 touched sections 是否完全一致。

## Suspicious implementation details

- 未发现伪实现、mock、stub、hardcode，亦未发现对历史 `runs/` / `artifacts/` 事实的改写。
- `T86` 的核心动作真实发生且保持边界：
  - note 只做了最小 route / exclusion 装配刷新，没有新建第二份 manuscript，也没有把独立 theory 分支内容拉回 main；
  - 4 份新台账把 main text / appendix / supplement / exclusion 的 submission-facing 装配边界写成可审计事实；
  - 两个 README 已登记 `T86` 入口与 `% T86-ASSEMBLY` 使用规则；
  - 编译产物已随 note 刷新。
- 以下 guardrail 仍被保留，没有被 `T86` 偷偷升级：
  - `T24` 仍只是 frozen four-scenario mock-backed software-HIL reference；
  - `FR6/FR7` 仍只可写作 descriptive support；
  - `FR8/statcalib` 仍只可写作 extension lane / no-promotion / no unique clean threshold；
  - training/material 仍只可写作 canonical chain intact + one clean CPU-only bounded rerun；
  - `.tflite` 仍只可写作 isolated current-host true runtime；
  - real-board 仍只可写作 read-only gate / regeneration / provenance，当前 host 仍是 `NO_GO`；
  - 缺失 `Linux + FPGA` host 的 hardware-dependent surface 仍显式 blocked。

## Recommended next action

- 以 `PASS` 接受 `T86`。
- 后续若继续推进，只应开启更小的作者终检 / 投稿前 QA 类 docs-only 任务，或由 Captain 明确继续保持 `NO_GO_SUBMISSION_READY_COMPLETION`；不要把 `T86` 的 assembly 台账回述成 submission-ready pack 已完成、已完成 portability、已完成 real-board，或已完成 `statcalib` comparator promotion 的事实。
