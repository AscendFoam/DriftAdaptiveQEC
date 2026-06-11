# T70 任务与本次 Review 的解释

## 1. 先用通俗话解释：T70 到底在做什么

`T70` 不是再跑一次实验。

前面的 `T64/T66/T67/T68/T69` 已经把 `statcalib` 这条 `FR8` extension lane 能回答的几个关键问题都回答过一遍了：

1. 它确实能赢
2. 这个赢法不是一个参数点偶然撞出来的
3. 它不是粗略依赖某一个 `teacher_mode`
4. 预声明网格里确实存在 full generated-only winners
5. 即使再做更强的 tie-break，最强 clean answer 也还是三路并列，而不是唯一阈值

所以 `T70` 想解决的，已经不是“还要不要再跑一个 benchmark”，而是：

- 现在仓库里到底应该怎样用一句统一、不会夸大的话来描述这条 `statcalib` 证据链？

更直白地说，`T70` 是把前面几轮 FR8 结果收口成一个“统一结论包”，避免后面的人在写治理文档、论文材料或分支说明时：

1. 把 `T24` 冻结主表偷偷改写掉
2. 把 extension lane 说成已经成为正式主线 comparator
3. 把三路并列硬说成已经选出了唯一阈值
4. 把 mock-backed software-HIL 证据说成 `.tflite`、真板或更强等级证据

## 2. 这个任务具体是怎么实现的

### 2.1 任务目标

结合 `docs/02_experiment_plan.md`、`docs/04_task_board.md` 和 `docs/07_handoff.md`，T70 的真正目标是：

1. 不再新增任何执行事实
2. 只读复用 `T24/T64/T66/T67/T68/T69` 的历史 artifact 和已接受 review
3. 生成一个可复算的 closure/gate 结果
4. 明确回答两个治理问题：
   - 当前 FR8 证据能不能 promotion？
   - 当前 FR8 证据能不能支持“唯一阈值已确定”？

这和前面几轮 benchmark 任务不一样。前面几轮是在做“跑出来什么”；T70 是在做“已经跑出来的东西，允许怎么说，不允许怎么说”。

### 2.2 改了哪些文件

这次变更是典型的 task-scoped helper + task-scoped tests + task-scoped docs：

1. `cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py`
   - 新增 closure helper
   - 负责只读读取历史 FR8 artifact 和 review 链
   - 输出统一 closure pack JSON

2. `tests/test_fr8_statcalib_bounded_closure_pack.py`
   - 新增聚焦测试
   - 确认当前 preserved artifact 链会得到预期 gate 结论
   - 也确认如果 `T69` tie-set 结果被篡改，helper 会拒绝继续收口

3. `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
   - 这是给人看的最终 closure 文档
   - 它把 FR8 这条证据链统一翻译成可复用的治理结论

4. `docs/review/T70_review.md`
   - reviewer 视角的正式审查记录

5. `docs/for_human/T70_explanation.md`
   - 也就是你现在看到的解释文档

6. `docs/worker_summary/T70_worker_summary.md`
   - worker 的交付总结

### 2.3 helper 在做什么

`build_fr8_statcalib_bounded_closure_pack.py` 做的不是 generic benchmark，而是一个非常明确的 task-scoped consolidator。

它的输入是固定的：

1. `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
2. `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
3. `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`
4. `docs/evidence_packs/statcalib_fr8/statcalib_sensitivity_bounded_benchmark.md`
5. `docs/evidence_packs/statcalib_fr8/statcalib_teacher_anchor_bounded_benchmark.md`
6. `docs/evidence_packs/statcalib_fr8/statcalib_generated_only_robustness_bounded_benchmark.md`
7. `docs/evidence_packs/statcalib_fr8/statcalib_clean_winner_tiebreak_bounded_benchmark.md`
8. `docs/review/T64_review.md` 到 `docs/review/T69_review.md`
9. `runs/p4_benchmark/T24/...`
10. `runs/p4_benchmark/T64/...`
11. `runs/p4_benchmark/T66/...`
12. `runs/p4_benchmark/T67/...`
13. `runs/p4_benchmark/T68/...`
14. `runs/p4_benchmark/T69/...`

然后它会逐段重建这条证据链：

1. 先确认 `T24` 主表仍然成立
   - 四个冻结场景 winner 仍都是 `hybrid_residual_b`
   - runner-up 仍都是 `ukf`

2. 再确认 `T64`
   - `statcalib` 第六 lane 的加入没有改写 `T24` 冻结五模式子表
   - 同时 `statcalib` 在四个场景里都赢

3. 再确认 `T66`
   - 这个优势不是单点参数偶然值

4. 再确认 `T67`
   - 这个优势不是粗略依赖 `teacher_mode=ukf`

5. 再确认 `T68`
   - full generated-only winners 确实存在

6. 最后确认 `T69`
   - 最强 clean answer 仍然是
     - `statcalib_window_variance_t001`
     - `statcalib_window_variance_t003`
     - `statcalib_window_variance_t005`
   - 不存在 unique clean reference point

在这个基础上，它给出两个显式 gate：

1. `promotion_gate`
2. `unique_threshold_gate`

### 2.4 为什么这对后续开发有意义

从 `docs/04_task_board.md` 和 `docs/07_handoff.md` 的最新状态看，`T69` 已经把“tie-break execution question”做完了。

所以 T70 的意义在于：

1. 它把 `R24` 从“还要不要再跑一个 tie-break”收缩成“现在允许怎么宣传、不允许怎么宣传”
2. 它让后续 Captain 或论文整理者有一个统一引用点，而不是每次重新解读 `T64/T66/T67/T68/T69`
3. 它防止主线叙述滑坡
   - 不会把 extension lane 写成 frozen main table
   - 不会把 persistent tie 写成 unique threshold
   - 不会把 mock-backed software-HIL 写成更强证据

这对 Phase 2 非常关键，因为当前阶段的核心不是“尽量讲得更好看”，而是“不要破坏已经恢复出来的可信边界”。

## 3. T70 的核心结果是什么

T70 的关键结果其实就两个 gate：

### 3.1 No-Promotion Gate

结论是：

- `no_promotion_keep_extension_lane_only`

翻译成中文就是：

- 现在还不能把 `statcalib` 这条 FR8 证据链升格成可以改写 `T24` 主表的正式 comparator

理由很简单：

1. `T24` 仍是 authoritative frozen ranked table
2. `T64-T69` 全部仍是 mock-backed software-HIL extension-lane evidence
3. broader predeclared grid 还不是 uniformly clean
4. 当前 accepted review chain 从来没有批准 promotion

### 3.2 Unique-Threshold Gate

结论是：

- `future_selection_task_required`

这不是说“快要支持唯一阈值了”，而是说：

- 如果以后业务上必须选出一个唯一阈值，那必须先单开一个新任务，预先讲清楚到底按什么标准选

因为当前 `T69` 的真实答案仍然是：

- `persistent_clean_tie_set`

也就是：

- `t001 = t003 = t005` 仍然并列

## 4. 为什么我的 Review 结论是 `PASS`

### 4.1 为什么不是 `BLOCK`

因为任务包要求的主体都完成了，而且没有越界。

我实际核对到的关键点包括：

1. 没有新建 `T70` run root
   - `T70*` run-root 计数为 `0`

2. 没有重跑 benchmark
   - 本轮只运行了 `py_compile`、`unittest`、closure helper 和只读 shell 检查

3. 没有改写 `runs/`
   - `git diff --name-only -- runs` 为空

4. helper 和文档结论一致
   - helper 输出的最强 clean answer set 仍是 `t001/t003/t005`
   - `unique_clean_reference_point_exists = false`
   - `promotion_gate = no_promotion_keep_extension_lane_only`
   - `unique_threshold_gate = future_selection_task_required`

5. 没有改主线语义文件
   - 没改 `cnn_fpga/decoder/statcalib.py`
   - 没改 `cnn_fpga/runtime/slow_loop_runtime.py`
   - 没改 `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

6. 没有把 sidecar 或 theory-only 材料混入主线 closure pack

因此，这不是伪实现，也不是“只写文档不校验”的任务。

### 4.2 为什么不是 `PASS_WITH_WARNINGS`

因为这次没有发现那种会影响任务完成判定、但需要额外保留在 verdict 级别的风险。

`T70` 的 gate 结论虽然保守，但那不是 warning，而正是这个任务应该产出的东西：

1. 它没有假装 promotion 已经成立
2. 它没有假装唯一阈值已经成立
3. 它也没有把负面边界藏起来

换句话说，`T70` 的结果本来就应该是“更克制的统一结论”，而不是“更激进的宣传结果”。

## 5. 这次我对 worker 原文档做了哪些补充

worker 原来的方向基本是对的：

1. 它已经抓住了 T70 的本质不是新 benchmark，而是 closure/gate
2. 它也已经给出了两个最关键 gate 结论

我这次主要补了三类东西：

1. 把 T70 放回 `docs/04_task_board.md` 和 `docs/07_handoff.md` 的治理语境里重新解释
2. 把“为什么这是 PASS，而不是 BLOCK / PASS_WITH_WARNINGS”写得更清楚
3. 明确指出一个实现细节：
   - helper 里有几项“只读完整性”字段是任务内声明值
   - 真正的 run-root / runs-diff 可信检查仍然来自外部验证命令

这不推翻 worker 的主结论，只是让 reviewer 视角下的边界更清晰。

## 6. 一句话总结

`T70` 的价值不在于再证明一次 `statcalib` 很强，而在于把前面 `T64-T69` 的 bounded FR8 证据链收成一个不会越界的统一结论：

- 可以说它强、而且不是偶然
- 但现在还不能说它已经 promotion，也不能说唯一阈值已经确定
