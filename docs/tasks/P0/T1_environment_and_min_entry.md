# Task Package: T1

Task ID:
`T1`

Goal:
确认当前仓库可用的依赖矩阵与最小入口，至少让一个 smoke 级入口在当前机器上真实跑通，并把结果回写治理文档。

Why now:
默认 `python 3.13.7` 无法运行最小 benchmark；如果不先固定可用解释器和最小入口，后续任何 P3/P4 或 teacher-representation 工作都没有可靠执行面。

Allowed files:
- `README.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/tasks/P0/T1_environment_and_min_entry.md`
- 与环境/bootstrap 说明直接相关的新文档

Forbidden scope:
- `physics/` 核心算法实现
- `cnn_fpga/runtime/` 控制语义
- `cnn_fpga/decoder/` 参数映射主线
- 正式 benchmark 口径
- teacher-representation 分支实验逻辑

Inputs to read:
- `README.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/02_experiment_plan.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/legacy_context/2026-05-06_CNN_FPGA_GKP_legacy_handoff.md`
- `cnn_fpga/config/hardware_hil.yaml`
- `cnn_fpga/config/p4_multiscenario_hybrid_b_long.yaml`

Expected output:
1. 明确当前机器上的可用解释器/环境分工
2. 明确恢复期推荐的最小 smoke 命令
3. 至少一个 smoke 入口真实跑通
4. 把阻塞环境和不推荐环境写清

Verification:
- 解释器探测：
  - `where.exe python`
  - `py -0p`
- 依赖探测：
  - `C:\Python313\python.exe`
  - `C:\ProgramData\anaconda3\python.exe`
  - `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
  - `C:\ProgramData\anaconda3\envs\QuantumEnv\python.exe`
  - `C:\ProgramData\anaconda3\envs\TF1_14\python.exe`
- smoke 命令：
  - `C:\ProgramData\anaconda3\python.exe benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda`

Docs to update:
- `README.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Reviewer type:
`normal`
