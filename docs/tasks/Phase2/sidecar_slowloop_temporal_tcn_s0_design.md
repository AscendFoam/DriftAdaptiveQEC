# sidecar_slowloop_temporal_tcn：S0_design 任务包

## 状态

- 创建日期：`2026-06-08`
- 所属 wave：`Wave A`
- lane ID：`sidecar_slowloop_temporal_tcn`
- 分支：`codex/sidecar-temporal-tcn-residual`
- worktree：`.wt/tcn`
- 当前级别：`S0_design`
- lane 分类：`recommended_now`
- 与主线关系：不替代、不执行、不阻塞 `T69`

## 当前唯一任务

本 worktree 的当前唯一任务是：

```text
sidecar_slowloop_temporal_tcn S0_design
```

main 分支的当前唯一主线任务仍是：

```text
T69: FR8 statcalib clean-winner tie-break bounded benchmark
```

两者必须保持独立。此 sidecar 任务不得启动 `T69`，不得写入 `T69` run root，也不得把任何 sidecar 判断写成主线事实。

## 目标

设计一条 slow-loop temporal modeling sidecar 路线：在不改变 fast-loop input/output contract 的前提下，把历史 syndrome histogram 或窗口统计整理成时间序列特征，评估是否值得在后续 `S1_toy_or_replay` 中尝试一个 tiny TCN residual-`b` head。

`S0_design` 只回答路线是否定义清楚，不产生实验结果。

## Allowed Files（允许文件）

本任务只允许修改：

- `docs/tasks/Phase2/sidecar_slowloop_temporal_tcn_s0_design.md`

后续若进入 `S1_toy_or_replay`，必须另开任务包再声明允许的代码、配置、manifest 和 run root。

## Docs To Update（需更新文档）

本任务只新增当前任务包。不得同步修改主线治理文档、阶段结论、review 文档或 paper 叙事。

## Forbidden Scope（禁止范围）

不得：

- 运行 benchmark、训练、toy simulation 或 cached replay
- 修改 `cnn_fpga/` 源码、测试、配置或 benchmark runner
- 修改 `ParamMapper`、`SlowLoopRuntime` 或 fast-loop contract
- 写入 `runs/`、`artifacts/` 或任何历史 run root
- 写入 `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_*`
- 宣称 `.tflite` runtime、real-board 或 mature calibration comparator 已验证
- 将本 lane 输出写入 `T24`、`T64`、`T65`、`T66`、`T67`、`T68` 或 `T69` 事实口径

## Required Inputs（必读输入）

设计时必须阅读或引用：

- `README.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/reference/GPT-Pro有关扩展实验的建议.md`
- `docs/tasks/Phase2/T69_fr8_statcalib_clean_winner_tiebreak_bounded_benchmark.md`

若本 branch 暂未包含最新 sidecar 治理文档，必须按 main checkout 中的 PSE0 中文治理规则执行，不得以 branch 缺文档为由放宽边界。

## S0 设计必须回答的问题

1. 输入特征：使用哪些 syndrome histogram、窗口统计、EWMA、delta 或 entropy 字段。
2. 时间尺度：窗口长度、stride、warm-up、missing-window 处理。
3. 输出目标：residual-`b` head 的目标定义，以及是否只作为 correction residual candidate。
4. 边界：是否保持 fast-loop contract 不变；如需要 contract change，必须停止并另开任务。
5. 复用数据：是否读取历史 run root；若读取，必须列出精确路径和只读理由。
6. 后续 `S1_toy_or_replay` 的最小验证形态。
7. 与 `T24` frozen anchor、`T69` tie-break、`R24` statcalib overclaim 边界的关系。

## Run Directory Policy（运行目录规则）

`S0_design` 不创建 run dir。

后续若进入执行，run root 必须位于：

```text
runs/sidecar/sidecar_slowloop_temporal_tcn/<timestamp_or_run_id>/
```

不得写入 `runs/p4_benchmark/` 下的主线任务目录。

## Verification（验证）

完成本 S0 任务时必须运行：

1. `git status --short --branch`
2. `rg -n "sidecar_slowloop_temporal_tcn|S0_design|T69|T24|runs/sidecar|real-board|\\.tflite" docs/tasks/Phase2/sidecar_slowloop_temporal_tcn_s0_design.md`
3. `rg -n "real-board validated|tflite deployed|mature calibration comparator" docs/tasks/Phase2/sidecar_slowloop_temporal_tcn_s0_design.md`，预期只允许出现在禁止声明语境中

## Promotion Status（晋升状态）

- 当前：`not_requested`
- 允许的下一级：`S1_toy_or_replay`
- 晋升条件：另开 Captain 任务包，明确 allowed files、run root、manifest schema、只读历史输入和验证命令

## Worker Output Requirements（Worker 输出要求）

Worker 完成时必须汇报：

1. 设计了什么
2. 没有运行什么
3. 是否改变 fast-loop contract
4. 是否读取历史 run root
5. 后续 `S1_toy_or_replay` 的最小任务建议
6. 剩余风险
