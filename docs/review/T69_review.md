# T69 Review

## Verdict

`PASS_WITH_WARNINGS`

我按 T69 任务包要求做了只读复核，并额外重跑了三个轻量验证项：

1. `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_clean_winner_tiebreak.py`
2. `python -m unittest tests.test_statcalib_clean_winner_tiebreak_summary`
3. `python -m cnn_fpga.benchmark.summarize_statcalib_clean_winner_tiebreak --run-dir runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358`

结论是：T69 的主体任务确实完成了，且不是手写结论或伪实现。唯一保留 `WARNINGS` 的原因不是实现失败，而是结果边界和后续转述风险仍需明确保留。

## Blocking issues

- None.

## Non-blocking issues

1. `T69` 没有得到唯一的 clean reference point。
   - 最终结论仍然是 `statcalib_window_variance_t001 = statcalib_window_variance_t003 = statcalib_window_variance_t005`
   - helper 也把它明确分类为 `persistent_clean_tie_set`
   - 所以后续任何治理文档、阶段结论或口头转述，都不能把 T69 改写成“已经找到了唯一最优阈值”

2. `T69` 的更强结论只成立在任务包锁定的四候选 bounded matrix 内。
   - 这次只比较了 `T68` 留下的四个 full generated-only clean winners
   - 因此它支持“clean tie 在更强 repeats=4 下仍然存在”
   - 但它不支持被扩写成“已经完成通用阈值优化”或“主线 comparator 已经成熟定型”

## Missing tests

- 没有发现会影响 T69 结论成立的阻断性缺测。

建议未来若复用这类 helper，再补几类负向测试：

1. 重复 comparison row 的拒绝测试
2. `repeats != 4` 的拒绝测试
3. `coverage != 1.0` 或 `completed_repeats != 4` 的拒绝测试
4. `T68` 输入 tie-set 不匹配时的拒绝测试

## Suspicious implementation details

1. `cnn_fpga/benchmark/summarize_statcalib_clean_winner_tiebreak.py` 明确硬编码了：
   - 四个场景
   - 两个 frozen anchors
   - 四个 frozen candidates
   - `T68` clean tie-set 对照关系
   这看起来像 hardcode，但这里是任务包明确要求的 task-scoped helper，不是伪实现，也不应被误当成通用基础设施。

2. helper 读取 `T68` summary pack 只用于 tie-set 对照，没有回写历史结果。
   - 这符合任务包要求
   - 我没有发现它重写 `T68` 或其它历史 run root 的行为

3. benchmark 是从干净短路径 clone `C:\t69c_1dbfbc3` 发起，而不是直接从当前脏工作区发起。
   - 这不是异常，反而是本仓库当前治理下的正确 provenance 保护做法
   - `host_launch_meta.json`、`launch_plan.json`、`summary.json` 三者能够互相对上

## Recommended next action

接受 `T69` 为 `PASS_WITH_WARNINGS`。

后续推荐动作：

1. Captain 可以把 `T69` 作为对 `R24` 窄问题的一次有效收口证据
2. 治理文档应明确写成：
   - `T24` 仍是冻结主表
   - `statcalib` 仍是单独标注的 extension lane
   - 当前最强 clean answer 仍是三路 persistent tie，而不是唯一阈值
3. 如果后续业务上必须强行选一个唯一阈值，应另开新任务，并先定义新的选择准则；不能把 T69 现有证据硬解释成唯一答案
