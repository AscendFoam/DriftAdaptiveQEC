# PSE1：Sidecar main-controlled governance refresh

## Status

- Type: Captain docs-only setup
- Date: `2026-06-12`
- Execution: no experiment, no benchmark, no training, no `.tflite`, no real-board

## Goal

把 sidecar 治理从旧的长期 `.wt/*` 分支同步模式，调整为 main 分支内的精简治理控制台：

1. 在 `docs/sidecar/` 下建立类似主线 `00~08` 但更轻量的治理文档。
2. 明确后续 sidecar 可以在 main 代码基础上做新增-only实验实现，但不得破坏原有主线逻辑。
3. 统一规定结果目录为 `runs/sidecar/<lane_id>/<run_id>/`。
4. 将旧 Wave A worktree 标记为 retired/read-only，并把 S0 思路收编到 main。

## Allowed Files

- `docs/sidecar/README.md`
- `docs/sidecar/parallel_sidecar_extension_governance.md`
- `docs/sidecar/parallel_sidecar_worktree_plan.md`
- `docs/sidecar/00_sidecar_snapshot.md`
- `docs/sidecar/01_sidecar_lane_registry.md`
- `docs/sidecar/02_sidecar_execution_protocol.md`
- `docs/sidecar/03_sidecar_artifact_schema.md`
- `docs/sidecar/04_sidecar_promotion_gate.md`
- `docs/sidecar/lane_plans/*.md`
- `docs/tasks/Phase2/PSE1_sidecar_main_controlled_governance_refresh.md`

## Forbidden Scope

- 不运行任何实验。
- 不创建 `runs/sidecar`。
- 不修改 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/`。
- 不修改 `runs/` 或 `artifacts/`。
- 不改写当前唯一主线任务。
- 不把 sidecar 输出写成主线事实、论文 claim、`.tflite` runtime 或 real-board validation。
- 不执行旧 `.wt/*` worktree cleanup。

## Verification

1. `git diff -- docs/sidecar docs/tasks/Phase2/PSE1_sidecar_main_controlled_governance_refresh.md`
2. `rg -n "runs/sidecar|新增-only|retired|promotion gate|real-board validated|tflite deployed|mature calibration comparator" docs/sidecar docs/tasks/Phase2/PSE1_sidecar_main_controlled_governance_refresh.md`
3. `git diff --name-only -- cnn_fpga physics benchmark tests runs artifacts`

## Expected Output

- `docs/sidecar/` 成为 sidecar 精简治理入口。
- 旧 Wave A 长期 worktree 计划退役为参考。
- 后续 sidecar 代码和结果的新增-only、隔离目录、晋升规则清楚。

