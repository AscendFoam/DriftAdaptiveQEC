# T54: Multi-Seed Trace-Only Generalization Probe —— 给人类的说明

## 1. T54 做了什么

T54 执行了 T46 计划推荐的 Phase A：用 6 个 seed（3 个已有 + 3 个新 seed）做 trace-only 诊断，判断 committed-`b` 不稳定性是否只出现在 `seed=20260429` 上。

**全部 6 个 seed 的 trace 导出已完成**，cross-seed 分析已产出。

## 2. 核心发现

**committed-`b` 不稳定性在大多数 seed 上出现，但形态并不单一。**

6 个 seed 分为三类：

### 2.1 "安静" seed（20260425，1/6）

- Full 和 Gated v5 都没有 instability
- delta-b 幅度极低（< 0.04），committed-b 极低（< 0.02）
- 两者性能几乎相同（LER ≈ 0.66）
- 原因：该 seed 训练出的模型 teacher_b 本身就很低，gating 机制基本不激活

### 2.2 "经典" seeds（20260427, 20260428, 20260429, 20260430，4/6）

- Full 稳定（低 delta-b），Gated v5 不稳定（高 delta-b > 0.12，高 committed-b > 0.63）
- 这就是 T36/T38 在 seed 20260429 上看到的模式
- Gated v5 在 3/4 的 seed 上大幅领先
- seed 20260429 是唯一在 static_bias_theta 上 Gv5 略差于 Full 的 seed

### 2.3 "普遍不稳定" seed（20260510，1/6）

- **Full 和 Gated v5 都不稳定**（delta-b ≈ 0.17，committed-b ≈ 0.87）
- Full 变体——之前在所有其他 seed 上都稳定——在这个 seed 上也出现了高幅 delta-b
- 两者性能几乎相同（LER ≈ 0.49），因为两个模式的行为几乎一样

## 3. 机制泛化结论

**结论：广泛复现，但有重要差异（broadly repeated with qualifications）**

committed-`b` 不稳定性在 6 个 seed 中的 5 个上出现（不仅是 seed 20260429），所以它不是 seed 特异性的异常。但：

1. 有 1 个 seed 完全没有 instability（20260425）
2. 有 1 个 seed 上 Full 也不稳定（20260510）——不稳定性不是 Gated v5 独有的
3. instability 在大多数 seed 上**帮助了** Gated v5 的性能，而不是损害它

## 4. 对后续任务的建议

1. **Phase B 干预（I1: 降低 residual_clip_b）仍然有理由做**，但预期需要调低。高幅 delta-b 是系统性的，干预会测试一个真实的机制。但干预不应该被期望在所有 seed 上都改善结果。
2. **干预必须在所有 6 个 seed 上测试**，特别关注 seed 20260425（安静）和 20260510（普遍不稳定）。
3. **C4 应保持 `partial`**。机制故事比简单的"高 committed-b = 坏"更复杂。

## 5. 没有改变什么

- 没有运行任何干预变体
- 没有修改源码、config、benchmark 语义
- 没有重新打开 T45 冻结的 benchmark 边界
- 没有把 trace 证据升级为因果证明
- 所有新执行产物都在一个 T54-scoped run root 内（benchmark 输出目录除外）
- 总计 57,586 条 trace 行，6 个 seed source
