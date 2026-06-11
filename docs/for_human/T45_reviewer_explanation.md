# T45 Reviewer Explanation

## 1. 这个 task 在做什么（通俗版）

项目正在准备一篇论文。论文里需要一个 benchmark（基准测试）结果来证明"我们的方法比别人好"。

目前仓库里有一个已经跑完的 benchmark（T24），包含了 4 种漂移场景、5 种解码方法、每种跑 2 次重复。这个结果是真的，`hybrid_residual_b` 确实在 4 个场景里都赢了。

问题是：**这个 benchmark 够不够撑起一篇论文？**

T45 不跑新的实验，不改代码。它做的是一件"定规则"的事情——先想清楚：

- 当前 benchmark 能老老实实支撑什么样的论文说法？
- 如果想让论文 benchmark 更强，可以往哪些方向扩？
- 哪些扩展现在可以做，哪些先不做，哪些坚决不做？

用一个比喻：T45 不是在建房子，而是在画一张"如果以后要扩建，应该怎么建才不会搞乱现有结构"的施工规划图。

## 2. Worker 具体做了什么

### 2.1 任务目标

T45 的核心目标是回答一个二选一的问题：

- 论文能不能就靠现有 frozen-set benchmark（保守路线），还是
- 必须先开一个单独的扩展车道，把 benchmark 做强了再投稿

### 2.2 Worker 的工作流程

1. **阅读了 10 份输入文档**，包括任务板、交接文档、风险清单、P4 正式协议、论文 claim/evidence 台账、审稿人风险审计、恢复冻结快照、恢复 claim/evidence 表、以及两份参考研究文档
2. **创建了 1 份协议文档**：`docs/protocols/benchmark/paper_benchmark_expansion_protocol.md`
3. **创建了 1 份 self-review**：`docs/review/T45_review.md`
4. **创建了 1 份人类说明**：`docs/for_human/T45_explanation.md`
5. **在任务包里补了 verification record**

### 2.3 文件变化

只变了 4 个文件，全部是文档：

- `docs/protocols/benchmark/paper_benchmark_expansion_protocol.md`（新建）— 协议主体
- `docs/review/T45_review.md`（新建）— worker self-review
- `docs/for_human/T45_explanation.md`（新建）— 面向人类的通俗说明
- `docs/tasks/Phase2/T45_...md`（修改）— 追加了 verification record

**没有改任何代码、配置、benchmark 运行、训练、`.tflite` 或硬件相关的东西。**

### 2.4 协议文档的核心内容

协议分了 10 个章节，关键决策如下：

**当前 benchmark 能支撑什么：**
- 支撑一个保守、有界的论文说法（frozen-set only, software-HIL only）
- 不能支撑"广泛优势"或"paper-grade expanded benchmark"这种更强的说法

**哪些扩展被采纳（以后可以做）：**
- 保留 T24 作为锚点，不重写历史
- 新增 `random_walk`、`burst_reset`、unseen holdout 等 drift family
- `statcalib` 作为单独标注的 comparator lane
- 要求 learned modes 区分 training seed 和 evaluation seed
- 要求继续报告 commit/violation/saturation 等系统约束指标

**哪些扩展被延迟（先不做）：**
- soft-information / correlation-aware comparator（仓库里还没准备好）
- CI-driven stopping（以后可以加，现在不锁）
- rollback/fallback 作为硬性验收字段

**哪些扩展被拒绝（坚决不塞进主线 benchmark 扩展）：**
- 把 Gated v5、FiLM-style 等分支搜索并进当前 mainline benchmark
- 把 `.tflite` runtime 或 real_board 验证混进同一个任务
- 把参考文档里的设想当成已经完成的事实

### 2.5 对后续开发的意义

T45 把后续路线分成了两条合法走法：

- **路线 A（保守）**：用 frozen-set only 投稿，定位为有界的系统论文。现在就能走。
- **路线 B（更强 benchmark）**：新开一个 bounded expansion lane，按协议里的预声明规则扩展。需要额外任务。

无论走哪条路，T45 协议都锁住了一个关键约束：**T24 frozen-set 结果不能被静默改写或重新解释。**

这与 task board 中 `T46`（multi-seed mechanism/intervention plan）和 `T47`（paper ablation result-pack）是互补关系——T45 锁的是 benchmark 扩展协议，T46 补的是多 seed 机制证据，T47 冻结的是论文图表/ablation 材料。

## 3. 为什么我给了 PASS

### 3.1 任务确实完成了

T45 任务包要求：
1. 产出 benchmark-expansion protocol note — 完成
2. 分类 candidate expansion items 为 adopted/deferred/rejected — 完成
3. 判断论文能不能保持 frozen-set only — 完成（结论：可以，但只能保守定位）
4. 保持 docs-only — 完成
5. 把参考文档当 reference-only — 完成
6. 不静默升级 evidence level — 完成

### 3.2 没有伪实现、mock、stub、hardcode

T45 是纯文档任务，不涉及代码实现。所有结论都是分类和规则，没有假装任何 benchmark 已经跑过。

### 3.3 没有缺测试或验证的问题

docs-only 任务不需要代码测试。Worker 做了文档级验证：
- 确认 `git status --short` 只有 4 个允许文件
- 确认没有启动任何 benchmark/training/`.tflite`/hardware 执行
- 确认协议文档里明确写了 reference-only 约束

我做了一个额外的交叉验证：逐一检查了 worker 引用的 C2、C3、C11、E3 等标记是否在源文档中真实存在，以及参考文档里是否确实包含 drift families、soft-information、statcalib、learned branches 等概念。全部准确。

### 3.4 没有过度工程

协议文档长度适中（约 200 行），只包含必要的分类、规则和缺口清单。没有包含代码、配置或执行指令。

### 3.5 没有破坏已有功能

没有修改任何代码、配置或 benchmark 相关文件。

### 3.6 没有把计划写成事实

协议的 Section 9（Explicit Non-Claims）明确列出了 7 条"本文不做以下声明"，包括不声称扩展 evidence 已经存在、不声称参考文档是当前主线 truth、不声称 `statcalib` 已经集成。

### 3.7 几个被接受的非阻塞问题

- Worker 的 self-review 被这份 adversarial review 覆盖，这是项目惯例
- `sinusoidal` 被拒绝作为新 drift family 的理由可以更强，但在 protocol-lock 层面是可接受的
- 精确的 drift parameter grid 还没锁，这是故意的，留给未来 expansion task
- 文件命名惯例没有问题（worker explanation 和 reviewer explanation 是两个独立文件）

## 4. 对 Worker 已有 review 和 explanation 的评价

### Worker self-review（`docs/review/T45_review.md`）

Worker 的 self-review 给出了 PASS，列出了 3 个 non-blocking issues（N1-N3），都是 `accepted` 分类。

**评价：**
- Scope confirmation 部分准确，5 条确认全部正确
- 3 个 N-issues 分类合理，理由简洁
- 缺少一个重要的检查：没有交叉验证 C2/C3/C11/E3 是否在源文档中真实存在
- Review summary 部分的 adopted/deferred/rejected 归纳准确，但 `rejected` 列表可以更完整（缺少"把 `.tflite`/real_board 混入 benchmark 扩展"这一条）
- 推荐的 next task 描述合理但过于模糊（"a bounded execution task"），可以更具体地指向 T46

**总体：** Worker self-review 诚实、范围准确，但深度不如 adversarial review。这是预期内的。

### Worker explanation（`docs/for_human/T45_explanation.md`）

Worker 的 explanation 面向人类读者，用通俗中文写成。

**评价：**
- 结构清晰，6 个章节覆盖了"做了什么""结论""采纳/延迟/拒绝分类""后续路线""没改变什么""一句话总结"
- 对 frozen-set only 路线和 expansion lane 路线的区分解释得很清楚
- 没有误导性表述，没有把计划写成事实
- 可以补充的内容：
  - 缺少对 worker self-review 和 reviewer review 之间关系的说明
  - 缺少对参考文档（`延伸改进思路.md`、`进一步的深度研究结果.md`）为什么被当作 reference-only 的简要解释
  - "一句话总结"可以更明确地说明 T45 在整个 T44→T45→T46→T47 链条中的位置

**总体：** 质量合格，核心结论准确，个别细节可补充但不影响正确性。
