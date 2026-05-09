# T17: Training Manifest Bootstrap

Task ID: `T17`

Goal: 为训练链补独立 manifest 与 bootstrap，不把它混入 `requirements-recovery.txt`。

Why now: `requirements-recovery.txt` 只覆盖 P0/P3/P4 recovery smoke；训练链当前依赖 `DLEnv / torch`，需要独立说明才能让后续 Worker 安全接力。

Allowed files:

- `docs/tasks/Phase2/T17_training_manifest_bootstrap.md`
- `requirements-train.txt` 或 `docs/training_chain_bootstrap.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Forbidden scope:

- 不改训练代码
- 不启动训练长跑
- 不改模型主线
- 不把 DLEnv 探测结果写成跨机器保证

Inputs to read:

- `docs/02_experiment_plan.md`
- `docs/07_handoff.md`
- `cnn_fpga/model/train.py`
- 相关 `cnn_fpga/config/experiment_*.yaml`

Expected output:

- 训练链 bootstrap 文档或最小 manifest
- 明确解释器、依赖边界、可运行 smoke 命令与未覆盖项

Verification:

- 至少运行只读环境探测或 `--help` / import 级检查。
- 不要求完整训练。

Docs to update:

- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/04_task_board.md`（Captain 整合阶段）

Reviewer type: `normal`

## Worker Output Summary

- Output type: `docs/training_chain_bootstrap.md`
- No training code changed
- No long training run started
- Verification used import-level and `--help` checks only
- Updated docs:
  - `docs/training_chain_bootstrap.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
