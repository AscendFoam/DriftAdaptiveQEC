# T87 Review

审查方式：只读复核 `T87` allowlist 内的 note、README 与 4 份新台账；检查 `% T80-REOPEN`、`% T81-CALIBRATION`、`% T82-SUPPORT`、`% T83-CLOSEOUT`、`% T84-POLISH`、`% T85-PREFLIGHT`、`% T86-ASSEMBLY` 保留情况，以及 `% T87-QA` 是否只覆盖本轮实际修改的 `Numerical Results`、`Discussion`、`Conclusion`；执行 allowlist-scoped `git diff --check`、red-flag grep 与本地 LaTeX 编译/日志扫描。未运行任何 benchmark、训练、`.tflite` smoke 或 real-board 流程。

Verdict: `PASS`

## Blocking issues

- 无。

## Non-blocking issues

- `git diff --check` 现已无内容级报错，但当前 Windows 宿主机仍会输出 `LF will be replaced by CRLF` 提示；这是 working-copy 行尾噪声，不构成 `T87` 内容缺陷。
- `git status` 仍会输出 `C:\\Users\\26410/.config/git/ignore` 访问告警；这是当前宿主机 Git 配置读取噪声，不影响 `T87` allowlist 范围内文档事实。
- `paper_presubmission_regression_gate.md` 的 verdict 是 `GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY`，它只允许 bounded manual finish，并不等于 submission-ready completed。

## Missing tests

- 对 `T87` 这类 docs-only 作者终检任务，没有额外的自动化测试缺口；本轮关键验证已覆盖：
  - allowlist-scoped `git diff --check`
  - `% T80` 至 `% T86` 标记保留，以及 `% T87-QA` 三处新增标记
  - 四份新台账字段完整性检查
  - red-flag grep 扫描
  - `latexmk -g -pdf -synctex=1 -interaction=nonstopmode -halt-on-error CNN_FPGA_GKP_theory_note_draft.tex`
  - `.log` 关键字扫描未检出 `Underfull`、`Overfull`、`LaTeX Warning`、`undefined`、`Citation`

## Suspicious implementation details

- 未发现越出 allowlist 的修改，也未发现对 `runs/`、`artifacts/`、治理文档或源码树的触碰。
- `T87` 的 3 处 note 修改是最小 QA 定向刷新：它们只把“下一步”从 `T86` 的 assembly 口径进一步收紧为 author-final QA / bounded manual finish，没有引入新的 figure/table/claim，也没有重开 theory 扩写。
- red-flag 扫描结果与台账一致：危险表述仍然会在禁写清单、排除表、风险说明中以“不可这样写”的形式出现，但没有被写进作者主叙述句作为完成态结论。
- 编译刷新真实发生在当前主机的 `TeX Live 2024 + latexmk` 环境中，但这个结果只约束本机 note 产物，不外推到其他宿主或投稿模板链路。

## Recommended next action

- 以 `PASS` 接受 `T87`。
- 后续如果继续推进，只能做 `paper_manual_finish_queue.md` 中列出的 bounded manual finish，不应再把任务升级成 submission-ready completion、real-board success、default-env `.tflite` portability closure、full reproducibility closure 或 `statcalib` comparator promotion。
