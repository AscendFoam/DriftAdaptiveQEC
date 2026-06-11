# T38 人话版说明

## 1. 这个任务在做什么（通俗解释）

上一轮 T36 做的事情是"只看已有考试成绩分析为什么没考好"。

这一轮 T38 做的事情更像是"让这个学生用原来的方式再考一次，但这次全程录像"。

具体来说：

- 用和之前完全一样的实验设置（同一个 seed、同一组场景、Full 和 Gated v5 两个模式），重新跑一次 benchmark
- 但这次把每个慢更新窗口里的关键数据都导出来，形成一条完整的时间序列
- 然后沿着这条时间线去看：teacher 给了多少、CNN 又修了多少、最终提交的 b 是多少、那一刻误码率怎么样

这就像从"只有总分和最后一题答案"升级到"有每道题的解题过程"。

## 2. 任务实现详解

### 2.1 任务目标

T36 的结论是：`seed=20260429` 上 Gated v5 的收益收缩更像是"残差幅度不稳定"，但 T36 只能看到总结数据和最终快照，看不到中间过程。所以无法判断：

- 到底是 teacher 给的 b 本身就不稳？
- 还是 CNN 学出来的 delta_b 太大？
- 还是两者叠加后 committed b 被推得太远？
- 还是简单的符号偏了？

T38 的目标就是填补这个 gap：导出 per-window trace，直接回答上面这些问题。

### 2.2 任务流程

1. **阅读 T36 产出**：把 T36 的诊断报告和 review 当作待验证的假说
2. **确认不需要改代码**：检查 `hil_events.json` 发现，需要的 trace 字段（teacher_b、delta_b、committed_b、window_ler 等）已经在事件流里了，不需要改 runtime 或 benchmark runner
3. **跑一次 bounded rerun**：用原来的 Full vs Gated v5 语义，seed=20260429，四个场景，每个场景 2 个 repeat
4. **写一个 trace 导出脚本**：`analyze_seed20260429_trace.py`，从 `hil_events.json` 提取 per-window 数据
5. **分析 trace 并写诊断报告**

### 2.3 代码和配置变化

这次任务**没有**修改任何源码或配置文件。

任务包允许修改 `slow_loop_runtime.py`、`run_hil_suite.py`、`run_p4_teacher_representation_paired.py` 和新增一个 config，但 Worker 检查后发现需要的字段已经存在于 `hil_events.json` 中，所以选择了更小的改动路径——只加了一个纯读取的分析脚本。

| 文件 | 性质 |
|------|------|
| `cnn_fpga/benchmark/analyze_seed20260429_trace.py` | 新增：只读 trace 导出脚本（标准库 only） |
| `docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md` | 新增：trace 级诊断报告 |
| `docs/review/T38_review.md` | 新增：审查文件 |
| `docs/for_human/T38_explanation.md` | 新增：本文件 |
| `docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md` | 更新：Worker Output / Verification Record |
| `runs/T38_seed20260429_trace_probe_20260513/` | 新增：唯一的 T38 scoped 运行目录 |

### 2.4 Trace 导出的核心发现

**数据规模**：

- 4798 条 per-window trace rows
- 2399 条 Full + 2399 条 Gated v5
- 4 场景 x 2 repeat x ~300 windows = 结构一致

**关键发现 1：delta_b 幅度差距是最大的分离信号**

| Mode | max abs delta_b |
|------|----------------|
| Full | ~0.025 - 0.028 |
| Gated v5 | 0.1697（恒定） |

Gated v5 的残差修正幅度是 Full 的约 6-7 倍。而且这个 0.1697 不是巧合——它等于 `sqrt(2) * 0.12`，正好是两个分量都触碰到 0.12 clip 边界时的 L2 范数。说明 Gated v5 在很多窗口里都触发了 clip。

**关键发现 2：符号翻转不是简单偏移**

Gated v5 在所有场景中都有大量的 sign flip（delta_b_q 翻转 288-325 次/场景），但翻转在赢的场景和输的场景里都存在。所以不能把失败简单归因于"符号搞反了"。

**关键发现 3：committed b 的幅度才是决定性信号**

输的 repeat 有一个共同模式：
- teacher_b 比 Full 大
- delta_b 比 Full 大
- 两者叠加后的 committed b 比 Full 大很多
- 然后 window_ler 被推高

**结论升级**：

- T36 说："像是残差幅度不稳"
- T38 进一步收紧为："teacher_b + delta_b 合成后的 committed b 幅度和抖动过大"

这个结论有 4798 条 trace rows 做支撑，比 T36 的 summary/final-snapshot 级别证据强得多。

### 2.5 对后续开发的意义

1. **下一步很明确**：不是继续观察，而是试一个有界的缓解动作。比如降低 Gated v5 的 residual clip 或 residual scale，或者做一个 teacher-delta 衰减变体。

2. **R10 被显著缩窄但未关闭**：T38 把 teacher mechanism evidence 从 "summary 级假说"升级到了 "trace 级支持"，但还没有完全隔离上游原因是 teacher 本身还是 CNN 残差。不过对于缓解来说，这已经足够——不管上游是谁，只要把 committed b 的幅度压住就有可能改善。

3. **不影响 formal benchmark**：T38 仍然是单 seed 诊断，不是正式 benchmark 证据。

4. **验证了不需要改 runtime**：需要的 trace 字段已经在 hil_events.json 里了，说明之前的 T27/T28/T29 的 observability 改善工作已经到位。

## 3. 为什么 review 给了 PASS

### 3.1 任务确实完成了

- 任务包要求的 7 组 trace 字段全部 100% 可用（4798/4798）
- 任务包要求的 6 个报告章节全部到位
- 有界重跑完成（missing_runs = []，16 raw rows，8 comparison rows）
- Trace 导出完成（4798 rows，5 个 CSV + 2 个 JSON）

### 3.2 没有越界

- 代码/文档改动只落在 T38 allowed files
- Worker 甚至没有修改任务包允许修改的 runtime/benchmark 源文件（因为不需要）
- 历史 `runs/` / `artifacts/` 没有被改写
- 只有一个 T38 scoped 的新 run 目录
- 没有碰 `.tflite`、真板、训练、协议、baseline/scenario

### 3.3 没有伪实现

- Trace 数据来自 `hil_events.json` 的真实事件流，不是从 final snapshot 反推的
- 4798 条 trace rows 的数字经过 spot-check 验证，与底层 artifact 一致
- 分析脚本只用了标准库，没有导入项目运行时代码
- 没有硬编码结果、没有 mock、没有 stub

### 3.4 没有过度声称

- 报告正确标注为"单 seed 诊断 trace probe"
- 没有把结果写成 formal benchmark 或 paper-grade 证据
- 明确标注了限制（不测缓解、不隔离上游原因、只有一个 seed）
- 没有试图改写 formal benchmark boundary 或 statcalib 范围

### 3.5 发现的小问题（不阻塞）

1. 脚本有两个未使用的 import（`Iterable`, `Mapping`）——纯粹美观问题
2. Verification Record 写了 `missing_runs = 0`，实际 JSON 是 `missing_runs: []`——语义等价，但文档应精确匹配 artifact 格式
3. 报告中 Gated v5 的恒定 `max_abs_delta_b = 0.1697` 其实就是 clip 边界的对角线 `sqrt(2) * 0.12`，但报告没有显式给出这个几何解释——不构成正确性问题，但可能让读者疑惑为什么这个值跨所有场景/repeat 完全一样
4. 第一次长跑命令被工具的 1 小时 timeout 截断了，后续在同一个 run dir 上 resume 完成——这是正确的 resumable 行为，Worker 也做了透明记录

这些问题都不影响诊断的正确性和完整性，所以最终给出了 PASS。
