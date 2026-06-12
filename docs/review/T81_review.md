# T81 Review

审查方式：只读审查本次 `T81` diff；未重跑任何 benchmark、训练、`.tflite` 或 real-board 长实验。主要依据为：`T81` 任务包、`T81_worker_summary`、`git status --short --untracked-files=all`、`git diff`、note 源文件中的 `% T81-CALIBRATION` / `% T80-REOPEN` 标记、`paper_methods_and_contribution_calibration_manifest.md`、两份 README 登记，以及现有 LaTeX 产物与 `.log` 关键字扫描结果。

Verdict: `PASS`

## Blocking issues

- 无。

## Non-blocking issues

- `docs/paper_materials/README.md` 的改动幅度略大于“只登记 `T81` 入口”的最低需要，包含了一轮更宽的措辞压缩与目录整理。当前内容仍然与既有边界一致，所以不构成阻断，但它确实扩大了 review 面。
- 当前 compile 结论应理解为“这份 note 在当前主机上已通过 `TeX Live 2024 + latexmk` 成功刷新”，而不是“仓库默认所有 LaTeX 路径都已完全稳定可用”。bundled `tectonic` 的 doctor smoke 仍然报告 `os error 5`，这不是本轮 blocker，但需要继续诚实表述。
- `T81` 完成后，当前 note 仍然不是 full-manuscript reopen。这个结果与任务包边界一致，不是缺陷；但后续若继续推进 manuscript closeout，仍必须新开受控任务，不能把 `T81` 当作自动放行。

## Missing tests

- 无阻塞性缺口。对 `T81` 这类 docs-only contribution/methods calibration 任务，关键验证点已经覆盖：
  - 变更文件范围仍落在 `Allowed Files` 内。
  - note 中确实存在 4 条 `% T81-CALIBRATION` section-level 标记。
  - `T80` 的 8 条 `% T80-REOPEN` 标记仍保留，且本轮未把 `T81` 扩成对这些 ready sections 的再次大改。
  - `paper_methods_and_contribution_calibration_manifest.md` 已记录 4 个 changed sections、evidence anchors、guardrails 与 compile 状态。
  - `.log` 关键字扫描未检出 `Underfull`、`Overfull`、`LaTeX Warning`、`undefined`、`Citation`。
- 可选增强但非本任务必需：后续若还会继续做 note 校准，可增加一个小型检查脚本，同时核对 `% T80-REOPEN`、`% T81-CALIBRATION` 与两个 manifest 的 section ledger 是否一一对应。

## Suspicious implementation details

- 未发现伪实现、mock、stub 或 hardcode。`T81` 是 note prose / manifest / README 收口任务，不涉及源码、运行配置、benchmark harness 或历史结果改写。
- 未发现把计划写成事实的越界表述。`Summary of Contributions` 与三章 methods 虽然被重写，但新文本仍然保持以下边界：
  - `T24` 仍是主线 frozen-set main anchor，而不是 expanded benchmark closure；
  - `FR6/FR7` 仍是 descriptive support，而不是 causal closure；
  - `FR8/statcalib` 仍是 separately labeled extension lane / no-promotion / no unique clean threshold；
  - `.tflite` 仍只写 isolated current-host true runtime；
  - real-board 仍只写 read-only gate / regeneration / provenance with current-host `NO_GO`；
  - `Noise and Drift Model` 明确写成 effective model / control-oriented abstraction，而不是 full circuit-level 或 hardware-validated noise closure。
- 未发现对 `T80` 已关闭 section 的静默重开。diff 的主要正文变化集中在：
  - `Summary of Contributions`
  - `Brief Review of the GKP Code`
  - `Noise and Drift Model`
  - `Model Architecture`

## Recommended next action

- 按 `PASS` 接受 `T81`。
- 后续若继续推进 paper 主线，应由 Captain 另开且只开一张新的有界任务卡；不要把 `T81` 回述成 full-manuscript reopen、deployment closure、benchmark 扩写完成或 `statcalib` promotion。
- 如果下一步要继续做 manuscript-facing 收口，建议优先让 Captain 先决定唯一后续任务到底是 supporting-material gap closeout，还是更大范围的 prose gate，而不是让 worker 自行外推。
