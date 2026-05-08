# Task Package: T5

Task ID:
`T5`

Goal:
清点仓库中已经被跟踪的缓存、生成物与目录噪声，形成恢复期可执行的治理策略，并明确“立即保留 / 继续忽略 / 后续清理”的边界。

Why now:
`T4` 已恢复最小 software HIL bootstrap，但仓库仍混有大量历史 `.pyc`、`runs/` 与 `artifacts/` 结果。若不先固定治理口径，`T6/T7` 的复验结果会继续与历史噪声混在一起，也容易误把历史产物当成当前事实来源。

Allowed files:
- `.gitignore`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/06_repo_noise_governance.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/tasks/P0/T5_repo_noise_governance.md`

Forbidden scope:
- `runs/`、`artifacts/`、`__pycache__/`、`*.pyc` 的批量删除或 `git rm --cached`
- `cnn_fpga/`、`physics/` 核心逻辑修改
- 改写历史 benchmark 结果含义
- 把治理任务写成“仓库已经清理完成”

Inputs to read:
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `.gitignore`
- `docs/reference/AI_coding_workflow.md`
- `git ls-files "*__pycache__*" "*.pyc"`
- `git ls-files runs`
- `git ls-files artifacts`

Expected output:
1. 一份仓库噪声治理文档，明确分类、边界与后续动作
2. 一份 `T5` 任务包
3. 同步后的 task board / decision log / handoff / legacy audit / risks
4. 如有必要，仅补充非破坏性的忽略规则，不执行实际清理

Verification:
- `Get-Content -Raw -Encoding UTF8 "docs/06_repo_noise_governance.md"`
- `(git ls-files "*__pycache__*" "*.pyc" | Measure-Object).Count`
- `(git ls-files runs | Measure-Object).Count`
- `(git ls-files artifacts | Measure-Object).Count`
- `Get-ChildItem -Recurse -Directory -Filter "__pycache__"`

Docs to update:
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/06_repo_noise_governance.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Reviewer type:
`milestone`
