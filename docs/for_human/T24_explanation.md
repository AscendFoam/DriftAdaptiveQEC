# T24: P4 Formal Software Revalidation — 解释文档

## 一、这个任务在做什么？（通俗解释）

### 背景：量子纠错里的"考试"

想象你在做一个"自适应控制系统"——系统里有快回路（每 5 微秒做一次解码）和慢回路（每几十毫秒估计一次环境参数变化，然后更新快回路的配置）。

问题是：你的慢回路有好几种方案（经典方法 EKF/UKF、CNN 辅助的混合方案等），到底哪个最好？

要回答这个问题，需要在"模拟的量子噪声漂移场景"里跑对比 benchmark——就像在不同天气条件下测试不同轮胎的性能。

### T24 做了什么

T24 就是**正式重跑**了一遍这些对比测试。具体来说：

- **4 种漂移场景**：静态偏置、线性斜坡、阶跃变化、周期漂移
- **5 种解码方案**：EKF、UKF、常数残差、RLS 残差、混合残差-B（CNN 辅助）
- **每种组合跑 2 次**（用配对种子确保公平对比）
- 总共 **40 次** 独立实验

之前 T15 跑过类似的测试，但只覆盖了 2 个场景。T24 补齐了全部 4 个场景，形成了完整的正式证据包。

### 结果一句话

**所有 4 个场景里，混合残差-B（CNN 辅助方案）都赢了**，第二名都是 UKF。优势幅度约 1.5%~2.3%。

但需要注意：这只是**软件模拟**的结果，不是真 FPGA 板卡的实测结果。

---

## 二、详细技术解释

### 2.1 任务目标

T24 的正式目标是：按照 T23 锁定的正式协议（`docs/protocols/benchmark/P4_benchmark_formal_protocol.md`），执行一次完整的 frozen-set formal software revalidation。

"Frozen-set"意味着场景集合和方案集合是提前冻结的，不允许在执行中改动——这保证了实验的可复现性和公平性。

"Revalidation"意味着这不是首次实验，而是用恢复后的代码链路重跑历史实验，确认结论仍然成立。

### 2.2 任务流程

```
T23 (锁定协议) → T24 (执行正式重跑) → T25 (对抗性审查)
```

T24 的执行流程：

1. **Chunk 1**：跑全部 4 场景 x 5 方案的 repeat 0（20 次实验）
2. **Chunk 2**：跑全部 4 场景 x 5 方案的 repeat 1（20 次实验）
3. **Resume-only**：让 runner 在同一个 run dir 上汇总所有结果

为什么要分 chunk？因为 40 次实验总耗时约 20 小时，一次跑完可能超时。分 chunk 时只按 repeat 编号切分，不按场景切分——因为场景过滤会改变 seed 构造逻辑，导致种子不一致。

### 2.3 代码和配置变化

**没有改动任何源码或配置文件语义。** T24 是纯执行任务。

具体修改的文件：

| 文件 | 变化 |
| --- | --- |
| `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` | 新增 Section 15（T24 execution record），记录执行细节 |
| `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md` | 新增 Worker Output 段，记录命令、验证结果和风险 |
| `docs/04_task_board.md` | T24 标记为 `[x]`，当前任务切换到 T25 |
| `docs/07_handoff.md` | 新增 item 39，更新状态描述 |
| `docs/08_risks_and_open_questions.md` | R5 降级（正式重跑完成）、R9 收窄、R19 关闭 |

运行配置：
- 解释器：`C:\ProgramData\anaconda3\python.exe`
- 配置文件：`cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Run dir：`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`

### 2.4 关键结果

#### Per-scenario 结果

| 场景 | 赢家 | LER | 第二名 | 差距 |
| --- | --- | ---: | ---: | ---: |
| 静态偏置+旋转角 | hybrid_residual_b | 0.811 | ukf | 0.014 |
| 线性斜坡 | hybrid_residual_b | 0.788 | ukf | 0.023 |
| 阶跃变化 | hybrid_residual_b | 0.789 | ukf | 0.023 |
| 周期漂移 | hybrid_residual_b | 0.806 | ukf | 0.015 |

#### 与 T15 的一致性

T24 与 T15 有 20 次重叠实验（static_bias_theta 和 linear_ramp 的全部 repeat）。这些重叠实验的 final_ler 值完全一致（精确到小数点后 6 位），确认了 seed 一致性和实验可复现性。

#### 证据缺口

1. `correction_saturation_rate_mean` 始终为 0 — 可能是指标收集限制或确实没有修正饱和
2. Teacher diagnostics（hybrid_residual_b 的 teacher 贡献指标）全零 — 自 T15 以来的已知缺口，影响机制分析但不影响 LER 排名
3. `delta.csv` 全空 — 因为 strong-baseline 配置不包含 `static_linear` 和 `cnn_fpga`，这是设计预期

### 2.5 对后续开发的意义

1. **正式证据链完成**：T24 补齐了从 T15 的 2 场景到 4 场景的 gap，形成了完整的 frozen-set formal software revalidation。后续论文可以引用这个完整证据包。

2. **结论稳健**：hybrid_residual_b 在全部 4 个场景中一致最优，且与 T15 的部分结果完全一致，排除了"某次偶然跑出好结果"的风险。

3. **边界仍然清晰**：这只是 mock-backed software HIL 的结果。真 FPGA 部署、.tflite runtime 和更大量 repeat 的置信区间分析都还留到后续任务。

4. **下一步**：T25 会对 T24 做对抗性审查。之后项目路线分为几个方向：
   - 机制证据（teacher diagnostics 全零的原因分析、T27）
   - 部署边界（.tflite runtime 恢复、T31）
   - 论文准备（claim/evidence 整理、T34）

---

## 三、为什么给出 PASS_WITH_WARNINGS 的 review 结果

### 总体判断

T24 Worker 的工作质量很高，核心原因是：

1. **严格执行了任务包的所有约束**：没有改源码，没有改配置语义，没有扩大范围。所有 9 项验证检查全部通过。

2. **证据包完整**：summary.json、comparison.csv、launch_plan.json 等 9 类文件全部存在且内容正确。40 次实验全部完成，没有 missing runs。

3. **与 T15 结果完全一致**：重叠的 20 次实验 bit-for-bit 相同，确认了 seed 完整性和实验可复现性。

4. **边界声明准确**：所有文档都明确标注这是 mock-backed software HIL，没有越界声称。

### 为什么不是 PASS 而是 PASS_WITH_WARNINGS

有 3 个非阻塞 warning：

**N1（correction_saturation_rate_mean 全零）**：所有 20 个 scenario/mode 行的这个指标都是 0.0。Worker 正确地报告了这一点，但我们不确定这是因为指标收集代码有死分支，还是这些参数下确实没有修正饱和。未来需要一个机制审计任务来区分这两种情况。

**N2（04_task_board.md 的一条全局建议）**：diff 中加了一行关于 DLEnv 环境的建议，这不属于 T24 的 allowed scope（它不是 T24 的执行结果）。这是一个小的治理越界，不影响实质结论。

**N3（teacher diagnostics 已 deferred 四个任务）**：从 T15 到 T16 到 T23 到 T24，teacher diagnostics 全零的问题已经连续被 deferred 了四次。每次都正确地判定为"非阻塞"，但延迟链越来越长，增加了被遗忘的风险。我建议 T25 之后优先安排机制审计。

### 为什么不是 BLOCK

没有任何阻塞问题。所有验证通过，范围合规，数据真实，结论准确。上述 3 个 warning 都是"值得注意但不阻止项目继续推进"的问题。
