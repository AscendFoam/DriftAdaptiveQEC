# PSE0：并行 sidecar 扩展实验治理设置

## 状态

- Captain 于 `2026-06-08` 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：docs-only Captain 治理设置
- 与主线关系：本任务不替代、不执行、不修改 `T69`

## 任务背景

主线 `T69` benchmark 已经作为当前唯一任务准备好，但预计会长跑数天。项目可以利用这段墙钟时间准备若干互相独立的扩展路线，但前提是这些路线必须与主线证据隔离，不能改写已经冻结的 benchmark 叙事。

本任务补齐让并行 sidecar 实验安全展开的规则：

1. 明确 frozen anchor manifest
2. 统一 sidecar artifact schema
3. 设置从 sidecar 晋升到主线候选的 promotion gate
4. 规定 run dir 与 worktree 隔离规则
5. 划出 sidecar 输出禁止合入主线事实口径的红线

## 目标

创建后续开启多个 sidecar worktree 所需的治理与计划文档；本任务不运行任何实验，也不改变任何 benchmark 语义。

## Allowed Files（允许文件）

Captain 只允许修改：

- `docs/tasks/Phase2/PSE0_parallel_sidecar_extension_governance_setup.md`
- `docs/parallel_sidecar_extension_governance.md`
- `docs/parallel_sidecar_worktree_plan.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/06_repo_noise_governance.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

## Docs To Update（需更新文档）

Captain 必须更新：

- `docs/parallel_sidecar_extension_governance.md`
- `docs/parallel_sidecar_worktree_plan.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/06_repo_noise_governance.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

## Forbidden Scope（禁止范围）

Captain 不得：

- 执行 `T69`
- 创建新的 Git worktree
- 创建分支
- 启动 benchmark、训练、`.tflite` smoke、real-board smoke 或 toy experiment
- 修改源码、测试、配置、`runs/` 或 `artifacts/`
- 修改 `docs/02_experiment_plan.md`
- 改变 `T69` matrix、repeat budget、candidate set、allowed files 或 verification 规则
- 把 `T24`、`T64`、`T65`、`T66`、`T67`、`T68` 或未来 `T69` 结果改写成成熟 comparator、`.tflite`、real-board 或 paper-grade 证据
- 把任何 sidecar 路线标记为已晋升、已接受或 mainline-ready

## Required Inputs（必读输入）

Captain 必须使用：

- `README.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/02_experiment_plan.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/06_repo_noise_governance.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/tasks/Phase2/T69_fr8_statcalib_clean_winner_tiebreak_bounded_benchmark.md`
- `docs/reference/GPT-Pro有关扩展实验的建议.md`

## Required Governance Outputs（必需治理输出）

治理文档必须包含：

1. frozen anchor manifest
2. sidecar artifact schema
3. promotion gate
4. run directory 与 worktree 隔离规则
5. 禁止 sidecar 输出进入主线事实口径的红线
6. sidecar 路线与 `T69`、`R24`、`.tflite`、real-board 边界的关系

worktree 计划必须包含：

1. 拟议分支名
2. 拟议 worktree root
3. lane 目标
4. 第一批允许输出
5. 验证形态
6. 证据边界
7. lane 是 `recommended_now`、`pilot_only` 还是 `research_only`

## Verification（验证）

Captain 必须运行并报告：

1. `git status --short --branch`
2. 文本检索确认 sidecar 治理与 worktree 计划提到：
   - `T24`
   - `T69`
   - `sidecar`
   - `promotion gate`
   - `runs/sidecar`
   - `real-board`
   - `.tflite`
3. 文本检索确认 sidecar 文档没有宣称：
   - `real-board validated`
   - `tflite deployed`
   - `mature calibration comparator`
4. 明确确认本任务未创建实验、分支或 worktree

## Review No-Go Triggers（审查 BLOCK 条件）

Reviewer 如发现以下任一情况，应返回 `BLOCK`：

1. 本任务改变 `T69` 执行语义
2. 本任务创建 run root、分支或 worktree
3. sidecar 规则允许结果改写 `T24` 或 `T69`
4. sidecar 规则允许在没有独立任务时声明 `.tflite` 或 real-board 证据
5. worktree 计划把 research-only code-family 变化放入近期 benchmark 队列
6. 治理同步把 sidecar 路线标记为当前唯一任务

## Captain Output（Captain 输出）

Captain 必须报告：

1. 修改了哪些文件
2. `T69` 是否仍是当前唯一主线任务
3. sidecar 治理摘要
4. worktree 计划摘要
5. verification 命令与结果
6. 剩余风险
