# T36 人话版说明

## 1. 这个任务在做什么（通俗解释）

这个项目有一个 CNN 模型叫 "Gated v5"，它做的事情是：在量子纠错的慢回路中，用 CNN 学一个残差修正来帮助解码器更好地跟踪噪声漂移。

实验发现一个奇怪的现象：

- 在大多数随机种子下，Gated v5 明显比不带 gated 分支的 "Full" 方案好。
- 但在 `seed=20260429` 这个种子下，这个优势消失了，甚至微微反转为略差。

T36 的任务就是：只看已有的实验数据（不重跑实验），搞清楚 `seed=20260429` 上到底发生了什么。

打个比方：就像一个学生平时考试都进步明显，但某次考试突然没考好。我们要做的是把他之前的考卷翻出来分析，看看是哪几科拖了后腿，是粗心丢分、还是题目类型不擅长、还是状态不好——但不去重新出题让他再考一次。

## 2. 任务实现详解

### 2.1 任务目标

从现有实验产出（`runs/` 目录下的 CSV 和 JSON 文件）中，对 `seed=20260429` 的 Gated v5 vs Full 结果做一次只读的失败机理诊断。

具体需要回答：
1. 20260429 上 Gated v5 到底差了多少？
2. 哪些场景拖了后腿？
3. 可能的原因是什么？（符号偏了？幅度太大？响应太慢？teacher 本身不稳？分支太保守？）
4. 哪些原因可以被现有证据排除？
5. 哪些问题现有证据回答不了？

### 2.2 任务流程

1. **读取实验计划**（`docs/02_experiment_plan.md` 等），了解 Gated v5 的设计和历史实验背景
2. **定位并读取关键实验产出**：
   - 配对/分块重跑的汇总摘要（`paired_20260427_220702/summary.csv`）
   - 三个种子的逐场景对比（`comparison.csv`）
   - 三个种子的 teacher 诊断数据（`teacher_scalar_diagnostics.csv`）
   - 20260429 的逐 repeat 最终快照（`hil_summary.json`）
   - 更早的非分块历史运行（仅作参考）
3. **编写只读分析脚本**（`cnn_fpga/benchmark/analyze_seed20260429_failure.py`）：
   - 只使用 Python 标准库（csv、json、math、pathlib）
   - 不导入项目运行时代码
   - 不写入任何文件，只向 stdout 输出 JSON 摘要
4. **基于数据和脚本输出，撰写诊断报告**（`docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md`）
5. **撰写预审输入和人话说明**

### 2.3 代码和配置变化

本次任务**没有**改变任何代码、配置、模型或 benchmark 语义。所有新增文件都是纯文档或只读分析工具：

| 文件 | 性质 |
|------|------|
| `docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md` | 新增：诊断报告 |
| `cnn_fpga/benchmark/analyze_seed20260429_failure.py` | 新增：只读分析脚本 |
| `docs/review/T36_review.md` | 新增：审查文件 |
| `docs/for_human/T36_explanation.md` | 新增：本文件 |
| `docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md` | 更新：补充 Worker Output 和 Verification Record |

### 2.4 诊断的核心发现

**数据事实：**

| 种子 | Full 平均 LER | Gated v5 平均 LER | 差距 |
|------|--------------|------------------|------|
| 20260427 | 0.8066 | 0.6205 | Gated v5 明显更好 (-0.186) |
| 20260428 | 0.8326 | 0.5944 | Gated v5 明显更好 (-0.238) |
| 20260429 | 0.6374 | 0.6397 | 基本持平，Gated v5 微差 (+0.002) |

20260429 是唯一一个 Gated v5 没有明显赢的种子。但不是全面崩溃：
- `static_bias_theta` 和 `step_sigma_theta`：Full 赢
- `linear_ramp`：Gated v5 仍略好
- `periodic_drift`：几乎完全打平

**能排除的原因：**
- 响应滞后：调度统计和 commit 数量完全一致，不是主因
- overflow / 修正饱和：Gated v5 的 overflow 甚至更低
- teacher 分支死亡：teacher 贡献和 gate 活跃度都是非零的
- 分支太保守：Gated v5 最终的 |b| 更大，不是不够激进

**最可能的原因（假说级别，非因果证明）：**
- `teacher_delta_b_q` 和 `teacher_delta_b_p` 在 20260429 上进入了一个更强、更不稳的 regime
- 在这个 regime 下，Gated v5 的残差修正有时修得更好，有时末端偏得更大
- 本质是"残差幅度的稳定性不够"，而不是"分支没起作用"

**现有证据回答不了的问题：**
- 到底是符号偏了还是幅度过了？没有逐窗口的 b 追踪数据
- 不稳是从哪个环节开始的？teacher 预测？CNN 残差输出？还是合成后的 commit？
- 换个 loss 或更强的 clipping 能不能修好？

### 2.5 对后续开发的意义

1. **明确了下一步该做什么**：如果后续要解决这个问题，最值的做法是给 `seed=20260429` 做一次有界重跑，只加一个"逐窗口追踪导出"，不改 benchmark 语义。这能直接回答"到底是符号偏了、幅度过了、还是后段漂了"。

2. **不影响当前主线结论**：Gated v5 在 3 个种子中的 2 个上明显更好，在 1 个上基本持平。这仍然支持"Gated v5 方向正确"的结论，只是暴露了一个种子特有的稳定性弱点。

3. **不影响 formal benchmark、statcalib、`.tflite`、真板等任何其他路线**：本次任务完全没有触碰这些边界。

4. **R10 和 R20 仍然开着**：teacher 机制证据缺口和 correction saturation 的 structural zero 问题都没有被这次诊断关闭。

## 3. 为什么 review 给了 PASS

### 3.1 任务确实完成了

任务包要求的 6 项产出全部到位：证据清单、逐场景汇总、跨种子对比、机理矩阵（5 个候选机制都有证据标签）、三分式结论（支持/假说/无法回答）、下一步建议。

### 3.2 没有越界

- 修改的 5 个文件全部在允许范围内
- 没有触碰 `runs/`、`artifacts/`、模型代码、配置、benchmark 语义、formal protocol
- 分析脚本只用了标准库，不导入项目运行时
- `git diff --name-only -- runs artifacts` 为空

### 3.3 没有伪实现

- 诊断报告中引用的 11+ 条 artifact 路径全部实际存在于磁盘上
- 报告中的数值经过 spot-check 验证，与底层 CSV 数据一致
- 分析脚本成功运行，输出确定性的 JSON 摘要
- 没有硬编码的结果、没有 mock 数据、没有 stub

### 3.4 没有过度声称

报告正确地：
- 把 `sign offset` 标记为"无法回答"（因为缺少逐窗口追踪）
- 把 `magnitude overshoot` 标记为"假说 / 部分支持"（只有最终快照）
- 明确说这不是因果证明
- 没有试图改写 formal benchmark 边界或 statcalib 范围

### 3.5 发现的小问题（不阻塞）

1. 分析脚本中有一个未使用的 `Iterable` import — 纯粹是代码美观问题
2. 脚本硬编码了场景名到目录名的映射 — 对于这个有界的诊断脚本来说是可接受的，但如果要复用需要注意
3. Worker 的预审文件和正式审查文件同名 — 信息已经在任务包中保留，不影响审查质量

这些问题都不影响诊断的正确性和完整性，所以最终给出了 PASS。
