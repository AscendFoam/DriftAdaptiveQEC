# Task Package: T2

Task ID:
`T2`

Goal:
把已跑通的 P0 smoke 结果正式收束为恢复期最小验证闭环，并决定是否需要补一份最小 bootstrap 说明或脚本，方便后续会话稳定复用。

Why now:
`T1` 已经证明最小入口可跑，但恢复期还缺少把这条路径稳定交给后续会话复用的落地说明；如果不收口，环境结论会再次退化成口头结论。

Allowed files:
- `README.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/tasks/P0/T2_smoke_reuse_and_bootstrap.md`
- 新增的 bootstrap / smoke 说明文档

Forbidden scope:
- `physics/` 核心算法
- `cnn_fpga/` 主逻辑代码
- P3/P4 benchmark 口径
- teacher-representation 分支实验逻辑

Inputs to read:
- `README.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/02_experiment_plan.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/tasks/P0/T1_environment_and_min_entry.md`
- `runs/smoke_test_anaconda/n10_r2_s0.250_summary.json`
- `runs/smoke_test_anaconda/n10_r2_s0.250_ler_curve_compare.csv`

Expected output:
1. 最小 smoke 路径的复用说明
2. 明确是否需要补 bootstrap 文档，若需要则补一份最小版本
3. 明确后续任务 `T3` 是否继续审计 mock/stub/placeholder 边界
4. 把 `DLEnv` 和轻量解释器的分工写清

Verification:
- 读取 smoke summary：
  - `Get-Content -Raw -Encoding UTF8 "runs/smoke_test_anaconda/n10_r2_s0.250_summary.json"`
- 读取 smoke CSV：
  - `Get-Content -Raw -Encoding UTF8 "runs/smoke_test_anaconda/n10_r2_s0.250_ler_curve_compare.csv"`
- 复核当前推荐解释器分工：
  - `C:\ProgramData\anaconda3\python.exe`
  - `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
  - `C:\ProgramData\anaconda3\envs\QuantumEnv\python.exe`

Docs to update:
- `README.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Reviewer type:
`normal`
