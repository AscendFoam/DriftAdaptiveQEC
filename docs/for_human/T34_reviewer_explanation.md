# T34 Reviewer 说明

## 1. 这个 Task 在做什么（通俗解释）

这个项目做的是一个"用 CNN+经典估计器混合方案来自适应纠正量子纠错中的漂移"的工程系统。项目从 Phase 0 到现在经历了大量的恢复、验证、benchmark 跑分、机制诊断等工作，积累了各种文档和运行结果。

现在还没开始写论文正文，但有一个很现实的问题：**如果我们明天要写论文，哪些话我们真的能说？哪些话只能带条件地说？哪些话现在绝对不能说？**

T34 的任务就是做一份"claim 和证据台账"——把所有可能的论文陈述列出来，逐条标明它背后有没有具体证据、证据在哪里、有什么限制条件。这样后续真正写论文时，就不会因为记忆模糊而把"mock 模拟跑出来的结果"写成"真实硬件验证结果"，也不会把"一次 CPU 训练冒烟测试"写成"全平台可复现"。

## 2. 实现细节

### 2.1 任务目标

T34 的目标是纯文档任务：

- 不写论文正文
- 不跑新实验
- 不改代码、配置、benchmark protocol
- 只做一份 claim/evidence 台账，把"可说 / 带条件说 / 不能说"的三类论文陈述整理清楚

### 2.2 Worker 做了什么

Worker 产出了 4 个文件（全部在 T34 允许范围内）：

1. **`docs/paper_claim_evidence_ledger.md`** — 主台账，包含：
   - **Scope and Non-Claims**：明确这份台账不做什么（不升级证据等级、不把 stub 写成真 runtime 等）
   - **Claim Ledger（C1-C11）**：11 条论文可能涉及的 claim，逐条标明 `supported` / `partial` / `blocked`，附带具体证据路径和措辞边界
   - **Figure Outline（F1-F3）**：3 个论文可能用的图，标明各自的证据支撑状态
   - **Table Outline（T1-T5）**：5 个论文可能用的表，同上
   - **Blocked Claims Summary**：汇总当前所有被阻塞的 claim 和具体阻塞原因
   - **Wording Guardrails**：7 条措辞护栏，告诉后续论文作者"用什么词替换什么词"

2. **`docs/review/T34_review.md`** — Worker 自审笔记（后被我的 adversarial review 覆盖）

3. **`docs/for_human/T34_explanation.md`** — 面向人类的中文简短说明

4. **`docs/tasks/Phase2/T34_paper_claim_evidence_ledger.md`** — 任务包更新（Worker Output + Verification Record）

### 2.3 五条关键边界的拆分

台账的核心价值在于把 5 条最容易被误写强的边界钉死了：

| 边界 | 台账中怎么拆的 |
|------|---------------|
| mock-backed software HIL vs real-board | C1（supported，mock 路径已恢复）vs C8（blocked，真板仍是 placeholder） |
| true `.tflite` runtime vs stub/fallback | C7（blocked，真 runtime 依赖不在当前机器上） |
| frozen-set formal software revalidation vs paper-grade expanded benchmark | C2/C3（supported，T24 已完成）vs C11（blocked，当前证据不能写成论文级扩展 benchmark） |
| clean CPU-only one-run smoke vs full training reproducibility | C5（supported，T40 已完成一次 CPU 训练冒烟测试）vs C6（blocked，全平台复现和 GPU 移植性未验证） |
| statcalib interface contract vs integrated comparator evidence | C9（supported，接口和测试存在）vs C10（blocked，没有集成到 benchmark 的结果） |

### 2.4 对后续开发的意义

这份台账直接服务于 Milestone 2K 的下一个任务 `T35: Paper draft skeleton and reviewer-risk audit`。当 T35 开始搭建论文骨架时，可以：

- 用 `supported` 的 claim（C1-C3, C5, C9）直接写进论文结果部分
- 用 `partial` 的 claim（C4, F1, T5）写进论文但必须带限定条件
- 对 `blocked` 的 claim（C6-C8, C10-C11）只能写在"future work"或"limitations"中
- 用 wording guardrails 的 7 条规则校对措辞

台账还帮助后续任务排优先级：如果某个 claim 对论文论证很关键但当前 `blocked`，那解除对应的 blocker 就是高优先级。

## 3. 为什么给出这个 Review 结果

我的 review 结果是 **PASS**，即"任务完成，没有阻塞问题"。

### 3.1 任务是否真的完成了

是的。T34 任务包要求产出 6 个 section（scope、claim ledger、figure outline、table outline、blocked claims、wording guardrails），Worker 全部产出了，且：

- 每条 claim 都带了 claim ID、短文本、状态、具体证据路径、边界措辞和关联风险
- 台账用了稳定的 ID 系统（C1-C11, F1-F3, T1-T5），后续论文可以直接引用
- 5 条关键边界全部被正确拆分
- 没有碰任何不允许碰的文件

### 3.2 有没有伪实现、mock、stub、hardcode

没有。T34 是纯文档任务，Worker 只改了 4 个 markdown 文件。没有代码改动，没有 mock/stub/hardcode。

### 3.3 有没有缺测试或验证

没有缺测试。T34 是文档任务，验证方式是文档结构和证据可追溯性检查，Worker 在 Verification Record 中记录了：

- 每个 supported/partial claim 都引用了具体的现有证据路径
- 每个 blocked claim 都引用了具体的 blocker 路径和对应的风险编号
- 台账显式保留了所有硬边界
- 本次没有引入代码/config/runs/artifacts 改动

我独立验证了以上 4 条：所有 23 个文档路径和 9 个 run/artifact 路径都确实存在于磁盘上；所有引用的风险 ID（R5, R8-R14, R24）都在 `docs/08_risks_and_open_questions.md` 中存在。

### 3.4 有没有过度工程

没有。台账结构简洁（一张 claim 表 + 一张 figure 表 + 一张 table 表 + 一张 blocked 汇总 + 7 条措辞护栏），没有引入不必要的抽象或模板。

### 3.5 有没有破坏已有功能

没有。T34 只改了 4 个文档文件，不影响任何代码、配置、benchmark、训练或 HIL 路径。

### 3.6 文档有没有把计划写成事实

没有。这是本次审查最重点检查的维度。具体来说：

- 所有 `supported` claim 的证据路径都经过核实确实存在
- 所有 `blocked` claim 都明确标注为不可写，并给出了具体的阻塞原因
- 所有 `partial` claim 都带了显式的边界措辞限制
- wording guardrails 明确列出了"用 X 替换 Y"的规则

### 3.7 我发现的非阻塞问题

我标记了 4 个非阻塞问题（N1-N4）：

1. **N1**：C9 的证据路径只引用了 review 文档，没有直接引用 `cnn_fpga/decoder/statcalib.py` 和 `tests/test_statcalib_interface.py`。claim 本身是准确的，但后续论文作者溯源需要多跳一步。
2. **N2**：台账没有包含"float/int8 量化退化 < 1%"的 claim。这个结论在 P1/P3 阶段已经建立，但因为是 Phase 2 恢复期之前的历史结果，所以没有被纳入。后续论文可能需要这条 claim。
3. **N3**：台账没有包含 features ablation 的结论（histogram delta 是关键通道等）。同理，这些是历史结果。
4. **N4**：Worker 的自审文件被我覆盖了。信息已保留在任务包的 Verification Record 中，不影响完整性。

这些问题不阻塞 T34 收口，但 Captain 在安排 T35 时应该考虑是否需要补充 N2/N3 对应的 claim 行。
