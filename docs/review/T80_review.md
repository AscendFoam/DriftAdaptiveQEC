# T80 Review

审查方式：只读审查本次 `T80` diff；未重跑任何长实验。主要依据为 `git diff` 范围检查、`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 的 section 标记、`docs/paper_materials/paper_bounded_prose_reopen_manifest.md`、两份 README 登记，以及已存在的 LaTeX 编译产物与 `.log` 关键字扫描结果。

Verdict: `PASS`

## Blocking issues

- 无。

## Non-blocking issues

- `T80` 故意没有触碰 `Summary of Contributions`，也没有改写三章 methods：`Brief Review of the GKP Code`、`Noise and Drift Model`、`Model Architecture`。这与任务包边界一致，但也意味着当前 note 仍然不是 full-manuscript reopen。证据见 note 中 `% T80-REOPEN` 只出现在第 11、20、61、536、643、687、993、1056 行，而 `Summary of Contributions` 与三章 methods 仍位于第 142、274、363、405 行的 untouched 区域。
- 当前 compile 结论应理解为“这份 note 在当前主机上已经通过 `TeX Live 2024 + latexmk` 刷新过一次”，不应外推成“仓库默认 LaTeX 工具链完全无条件可用”。这不是缺陷，但需要保持口径克制。

## Missing tests

- 无阻塞性缺口。对 `T80` 这类 docs-only prose reopen 任务，关键验证点已经覆盖：
  - 变更文件范围落在任务包 `Allowed Files` 内。
  - `% T80-REOPEN` 标记只落在允许改写的 8 个 section 入口。
  - `Summary of Contributions` 与三章 methods 保持 untouched。
  - `paper_bounded_prose_reopen_manifest.md` 已记录 section ledger、guardrail 与 compile 状态。
  - `.log` 关键字扫描未检出 `Underfull`、`Overfull`、`LaTeX Warning`、`undefined`、`Citation`。
- 可选增强但非本任务必需：后续若还会继续做 prose-bound task，可以增加一个小型 grep/check helper，自动核对 manifest 中的 changed sections 是否与 `% T80-REOPEN` 标记一一对应。

## Suspicious implementation details

- 未发现伪实现、mock、stub 或 hardcode。`T80` 本质是 note prose 与 traceability manifest 收口任务，不涉及源码路径、benchmark 执行或结果伪造。
- 未发现把计划写成事实的越界表述。`paper_bounded_prose_reopen_manifest.md` 将 scope verdict 明确限定为 `SECTION_BOUNDED_REOPEN_COMPLETED`，并在 `Numerical Results` / boundary checklist 中继续保持：
  - `T24` 仍是主线主锚点，而不是 expanded benchmark closure；
  - `FR6/FR7` 仍是 descriptive support，而不是 causal closure；
  - `FR8/statcalib` 仍是 extension lane / no-promotion；
  - `.tflite` 仍只写 isolated current-host true runtime；
  - real-board 仍只写 read-only gate / regeneration / provenance with current-host `NO_GO`。
- README 登记口径一致：`docs/paper_notes/README.md` 明确说 `% T80-REOPEN` 链路只覆盖 8 个 ready sections；`docs/paper_materials/README.md` 明确说该 manifest 不是 full-manuscript reopen 批准。

## Recommended next action

- 按 `PASS` 接受 `T80`。
- 后续若继续推进 paper prose，应新开有界任务处理下一层未收口内容，例如：
  - 单独处理 `Summary of Contributions` 的一致性校准；
  - 单独处理 methods-only calibration；
  - 或继续做 paper-ready closeout，但不得把 `T80` 视为 full-manuscript reopen、deployment closure 或 `statcalib` promotion 的授权。
