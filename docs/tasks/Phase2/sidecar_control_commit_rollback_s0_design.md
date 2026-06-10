# sidecar_control_commit_rollback：S0_design 任务包

## 状态

- 创建日期：`2026-06-08`
- 所属 wave：`Wave A`
- lane ID：`sidecar_control_commit_rollback`
- 分支：`codex/sidecar-atomic-commit-rollback`
- worktree：`.wt/ctrl`
- 当前级别：`S0_design`
- lane 分类：`recommended_now`
- 与主线关系：不替代、不执行、不阻塞 `T69`

## 当前唯一任务

本 worktree 的当前唯一任务是：

```text
sidecar_control_commit_rollback S0_design
```

main 分支的当前唯一主线任务仍是：

```text
T69: FR8 statcalib clean-winner tie-break bounded benchmark
```

本 lane 只设计 atomic commit / rollback control sidecar 路线，不得启动 `T69`，不得写入 `T69` run root，也不得把控制安全设计写成 real-board 完成态。

## 目标

设计一条 control-safety sidecar 路线：围绕 slow-loop 参数提交、bank 切换、版本确认、失败回滚和异常冻结，定义后续 contract tests 的最小范围。`S0_design` 不实现控制逻辑，不跑 mock-HIL contract test。

## Allowed Files（允许文件）

本任务只允许修改：

- `docs/tasks/Phase2/sidecar_control_commit_rollback_s0_design.md`

后续若进入 contract tests，必须另开任务包再声明允许的代码、测试、manifest 和 run root。

## Docs To Update（需更新文档）

本任务只新增当前任务包。不得同步修改主线治理文档、阶段结论、review 文档或 paper 叙事。

## Forbidden Scope（禁止范围）

不得：

- 运行 benchmark、训练、toy simulation、cached replay 或 contract test
- 修改 `cnn_fpga/` 源码、测试、配置、scheduler 或 board backend
- 把 `board_backend.py` placeholder 写成真实板级完成
- 修改 `ParamMapper`、`SlowLoopRuntime` 或 fast-loop contract
- 写入 `runs/`、`artifacts/` 或任何历史 run root
- 写入 `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_*`
- 宣称 `.tflite` runtime、real-board 或 mature calibration comparator 已验证
- 将本 lane 输出写入 `T24`、`T64`、`T65`、`T66`、`T67`、`T68` 或 `T69` 事实口径

## Required Inputs（必读输入）

设计时必须阅读或引用：

- `README.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/reference/GPT-Pro有关扩展实验的建议.md`
- `docs/tasks/Phase2/T69_fr8_statcalib_clean_winner_tiebreak_bounded_benchmark.md`

若本 branch 暂未包含最新 sidecar 治理文档，必须按 main checkout 中的 PSE0 中文治理规则执行，不得以 branch 缺文档为由放宽边界。

## S0 设计必须回答的问题

1. commit 边界：什么算 pending、committed、acked、failed、rolled back。
2. rollback 触发：版本不一致、timeout、range violation、bank thrash、confidence drop 如何处理。
3. mock-HIL contract：后续只允许先做 deterministic mock contract tests，不做 real-board。
4. 与 placeholder 边界关系：如何避免把 `board_backend.py` 写成真实板级完成。
5. 状态记录：后续 manifest 或 event log 必须记录哪些字段。
6. 后续 contract-test 任务的最小 test matrix。
7. 与 `T24` frozen anchor、`T69` tie-break、`R24` statcalib overclaim 边界的关系。

## Run Directory Policy（运行目录规则）

`S0_design` 不创建 run dir。

后续若进入执行，run root 必须位于：

```text
runs/sidecar/sidecar_control_commit_rollback/<timestamp_or_run_id>/
```

不得写入 `runs/p4_benchmark/` 下的主线任务目录。

## Verification（验证）

完成本 S0 任务时必须运行：

1. `git status --short --branch`
2. `rg -n "sidecar_control_commit_rollback|S0_design|T69|T24|runs/sidecar|real-board|\\.tflite|board_backend" docs/tasks/Phase2/sidecar_control_commit_rollback_s0_design.md`
3. `rg -n "real-board validated|tflite deployed|mature calibration comparator" docs/tasks/Phase2/sidecar_control_commit_rollback_s0_design.md`，预期只允许出现在禁止声明语境中

## Promotion Status（晋升状态）

- 当前：`not_requested`
- 允许的下一级：`S1_toy_or_replay` 或 contract-test 任务
- 晋升条件：另开 Captain 任务包，明确 allowed files、mock-HIL contract test 范围、run root、manifest schema 和验证命令

## Worker Output Requirements（Worker 输出要求）

Worker 完成时必须汇报：

1. 设计了什么
2. 没有运行什么
3. 是否改变 fast-loop 或 board backend 语义
4. 是否读取历史 run root
5. 后续 contract-test 的最小任务建议
6. 剩余风险
