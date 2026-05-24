# T55 Reviewer Explanation

## 1. T55 任务通俗解释

### 任务目标

T54 发现了一个现象：Gated v5（一种 CNN 解码器变体）会产生高幅度的 committed-b（合成控制偏置），这种"不稳定性"在 6 个测试 seed 中的 5 个上都存在。

T55 想回答一个简单问题：**如果把 Gated v5 的输出幅度限制（residual_clip_b）从 0.12 降低一半到 0.06，会改善结果吗？**

### 为什么这个问题重要

如果降低 clip 能稳定 committed-b 并改善 LER，那就说明高幅 committed-b 是问题，干预方向正确。这会直接支持论文的机制叙事。

如果降低 clip 让事情变差，那就说明高幅 committed-b 可能不是问题——它可能是 Gated v5 表现好的原因。

### 怎么做

用 T54 已经训练好的 6 个 seed 的 Gated v5 模型，不改模型、不改代码、只改一个配置参数（clip 0.12 → 0.06）。在全部 48 个 HIL 会话（6 seeds × 4 scenarios × 2 repeats）上测试。

---

## 2. 任务实现详解

### 任务目标

根据 T46 制定的 multi-seed plan：
- Phase A（T54）完成了：committed-b 不稳定性跨 seed 的现象确认
- Phase B（T55）要完成：干预测试——降低 clip 是否能改善结果

### 任务流程

1. **阅读必要上下文**：T54 报告、T54 review、T46 plan、benchmark runner 代码、配置文件等
2. **创建 T55 运行目录**：`runs/T55_multi_seed_i1_probe_20260523/`
3. **生成 6 个 seed 专属配置文件**：每个 config 继承自 `p4_multiscenario_strong_baselines.yaml`，覆盖 `residual_clip_b: 0.06`，指向 T54 训练好的 Gated v5 模型路径
4. **执行 benchmark**：用 `run_p4_multiscenario_benchmark.py` 对每个 seed 跑 4 个场景 × 2 个 repeat = 8 个 HIL 会话
5. **跨 seed 分析**：将 T55 I1 结果与 T54 Gated v5 基线（clip=0.12）对比
6. **产出文档**：主报告、review、中文说明

### 配置变化

整个 T55 任务只改了一个参数：

```yaml
# Gated v5 baseline
slow_loop.hybrid_residual_b.residual_clip_b: 0.12

# I1 intervention (only change)
slow_loop.hybrid_residual_b.residual_clip_b: 0.06
```

不重新训练、不改其他参数、不改代码。

### 执行结果

| Seed | 原 Gv5 LER | I1 LER | 差距 | 效果 |
| --- | ---: | ---: | ---: | --- |
| 20260425 (安静) | 0.664 | 0.827 | +0.163 | ❌ 变差 |
| 20260427 (经典) | 0.502 | 0.816 | +0.314 | ❌ 变差 |
| 20260428 (经典) | 0.522 | 0.579 | +0.057 | ❌ 变差 |
| 20260429 (经典) | 0.504 | 0.826 | +0.322 | ❌ 变差 |
| 20260430 (经典) | 0.492 | 0.468 | -0.024 | ✅ 变好 |
| 20260510 (普遍不稳) | 0.489 | 0.453 | -0.036 | ✅ 变好 |

**核心结论**：降低 clip 在 4/6 的 seed 上让 Gated v5 变差，整体有害（平均差距 +0.128）。

### 对后续开发的意义

1. **C4 应保持 `partial`**——干预测试没有证明"高 committed-b 是问题"
2. **T47 论文 ablation 打包不应在机制未闭合时推进**
3. **需要重新思考机制叙事**——证据显示 instability 在多数 seed 上是 Gated v5 的优势来源

---

## 3. 为什么给出 PASS 审核结果

### 任务完成度 ✅

- 48/48 HIL 会话全部完成
- 跨 seed 对比表完整（24 行，6 seeds × 4 scenarios）
- 所有 required sections 和 tables 都存在

### 边界纪律 ✅

- 没有修改任何源码或配置
- 没有重训练模型
- 只跑了一个干预变体（pure I1）
- 没有超出 6 seed 包、4 个场景、repeats=2 的固定边界
- 没有打开 TFLite、real-board、benchmark expansion 等被禁止的范围

### 证据级别诚实 ✅

- 报告使用了 bounded intervention evidence 语言
- 明确声明不包含因果证明
- C4 保持 `partial` 的结论被清晰支持

### 发现的问题（全部非阻塞）

审查过程中发现并修复了一个问题：
- N1：种子 20260430 和 20260510 的 `summary.json` 在最后 resume 运行后被不完整覆盖（只剩 periodic_drift 场景），影响了 summary.json 的完整性。底层的 per-repeat `hil_summary.json` 文件都是完整和正确的。已在审查期间重新聚合（resume-only）修复。

其他非阻塞问题：
- 运行根目录有 `benchmark_test/` 测试残留（约 225 MB）
- trace export 未单独运行（但原始事件数据存在）
- 配置链的 timing 差异（900 vs 300 窗口）导致跨 seed 的 baseline 比较窗口数不完全一致

---

## 4. 与 Worker 自行审查的对比

Worker 已经写了一篇 self-review（`docs/review/T55_review.md`）和中文说明（`docs/for_human/T55_explanation.md`），质量良好。我的 adversarial review 做了以下补充：

### 我的 review 增加的发现

| 项目 | Worker self-review | Adversarial review |
|------|-------------------|-------------------|
| N1 summary.json stale | 未提及 | 发现并修复 |
| N2 benchmark_test 残留 | 未提及 | 记录为非阻塞问题 |
| N3 timing 差异 (900 vs 300) | 提及 baseline 窗口数差异但未分析根因 | 明确指出是配置链继承差异 |
| N4 trace export 未单独运行 | 未提及 | 记录为怀疑但不阻塞 |
| S2 配置链差异 | 未提及 | 详细说明 n_slow_updates=900 继承路径 |

### 结论

Worker 的 self-review 是诚实的、边界清晰的。我的 adversarial review 在数据完整性检查和配置链分析上更深入，但结论一致：**PASS**。没有需要 BLOCK 的问题。
