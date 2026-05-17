# T35 Reviewer 说明

## 1. 这个 Task 在做什么（通俗解释）

T34 刚做完一份"claim 和证据台账"——把所有论文可能涉及的陈述分成"可以说"、"带条件说"、"不能说"三类。T35 在此基础上再做两件事：

1. **搭论文写作骨架**：把论文的每一节（摘要、引言、方法、实验、结果、局限、结论）都标明"这一节只能引用哪些 claim、图、表，哪些 claim 是 blocked 绝对不能写成完成态"。后续真正写正文时，只要对着这个骨架填内容，就不会越界。

2. **做审稿人风险审计**：站在一个挑剔的审稿人视角，把论文最可能被质疑的点全部列出来——新颖性够不够？证据等级够不够？有没有可能被误读为过度宣称？训练复现性够不够？机制解释够不够硬？每个质疑点都标明"仅靠改措辞能不能缓解"还是"必须补实验才能缓解"。

通俗地说，T35 不是写论文正文，而是在写正文之前先建一个"防越界模板"和一个"审稿风险预检清单"。

## 2. 实现细节

### 2.1 任务目标

T35 的目标是纯文档任务：

- 不写论文正文
- 不跑新实验
- 不改代码、配置、benchmark protocol
- 只搭一个 section-level 的论文骨架 + 做一份审稿人风险审计

### 2.2 Worker 做了什么

Worker 产出了 5 个文件（全部在 T35 允许范围内）：

1. **`docs/paper_draft_skeleton.md`** — 论文骨架，包含：
   - **Title Candidates**：4 个候选标题
   - **Global Guardrails**：全局护栏（哪些 claim 可写、哪些 partial、哪些 blocked）
   - **8 个 section skeleton**：Abstract / Introduction / Method / Experiment / Results / Limitations / Conclusion / Appendix Planning
   - 每个 section 都包含：
     - 预期的子标题
     - 允许引用的 claim/figure/table IDs
     - 明确列出不能出现的 blocked claims
     - 起草注意事项

2. **`docs/paper_reviewer_risk_audit.md`** — 审稿人风险审计，包含：
   - **Novelty Challenge Points (N1-N3)**：新颖性质疑
   - **Evidence-Grade Challenge Points (E1-E5)**：证据等级质疑
   - **Overclaim Wording Traps (W1-W6)**：过度宣称措辞陷阱
   - **Reproducibility/Deployment Challenge Points (R1-R3)**：复现/部署质疑
   - **Ablation/Mechanism Challenge Points (A1-A3)**：消融/机制质疑
   - **Section-by-Section Reviewer Hotspots**：逐节审稿热点
   - **Minimum Safe Paper Positioning**：当前最安全的论文定位
   - **Do-Not-Publish-As-Claimed List**：绝对不能发的论文定位

3. **`docs/review/T35_review.md`** — Worker 自审笔记（后被我的 adversarial review 覆盖）

4. **`docs/for_human/T35_explanation.md`** — 面向人类的中文简短说明

5. **`docs/tasks/Phase2/T35_paper_draft_skeleton_and_reviewer_risk_audit.md`** — 任务包更新（Worker Output + Verification Record）

### 2.3 代码/配置变化

没有任何代码或配置文件变化。5 个文件全部是纯 markdown 文档。

### 2.4 对后续开发的意义

T35 的产出直接服务于论文撰写的下一步：

1. **骨架控制论文写作**：后续只要按骨架逐节填内容，每节只引用允许的 claim IDs，就能确保论文不越界。骨架中有 46 处 C/F/T/R 交叉引用，相当于 46 个"护栏桩"。

2. **风险审计控制措辞**：审计中有 6 个措辞陷阱（W1-W6），每个都给出了"不安全措辞 → 安全替换措辞"的映射。后续论文作者可以直接用这些替换规则校对初稿。

3. **风险分层指导后续实验优先级**：审计把每个质疑分成"仅靠改措辞可缓解"和"必须补实验"两类。如果后续要升级论文证据等级，只需看"evidence-upgrade-needed"列就知道该补什么实验。

4. **Milestone 2K 收口**：T34 + T35 完成后，Milestone 2K（Paper Assembly Readiness）的两个任务全部完成。后续可以进入论文写作阶段或进入下一个开发里程碑。

## 3. 为什么给出这个 Review 结果

我的 review 结果是 **PASS**，即"任务完成，没有阻塞问题"。

### 3.1 任务是否真的完成了

是的。T35 任务包要求产出：
- 论文骨架（至少 8 个 section：title candidates, abstract, introduction, method, experiment, results, limitations, conclusion）— 全部产出 ✅
- 每个 section 要列出 subsection headings、允许引用的 claim/figure/table IDs、blocked claims — 全部包含 ✅
- 审稿人风险审计（至少 6 类：novelty, evidence-grade, overclaim, reproducibility/deployment, ablation/mechanism, mitigation options）— 全部产出 ✅
- 每个 objection 要绑定具体 blocker、risk 或 wording hazard — 全部绑定 ✅

### 3.2 有没有伪实现、mock、stub、hardcode

没有。T35 是纯文档任务，Worker 只改了 5 个 markdown 文件。没有代码改动，没有 mock/stub/hardcode。

### 3.3 有没有缺测试或验证

没有缺测试。验证方式是文档结构和边界一致性检查。我独立验证了：
- 骨架中有 46 处 C/F/T/R 交叉引用，审计中有 23 处 — 引用密度足够
- 所有 5 个 blocked claims（C6/C7/C8/C10/C11）在骨架每个 section 中都被明确标记为不允许作为完成态 — 与 T34 台账一致
- 所有风险 ID（R5/R9/R10/R11/R12/R13/R14/R20/R24）都在 `docs/08_risks_and_open_questions.md` 中存在
- 没有任何越界文件被修改（git diff 确认）

### 3.4 有没有过度工程

基本没有。骨架结构清晰，每个 section 只有 4 个子部分（headings、evidence map、blocked claims、drafting notes）。审计用了表格格式，每个 objection 都有 6 列（ID、objection、trigger、reference、wording mitigation、evidence mitigation），信息密度高但不冗余。

唯一可以讨论的是标题候选：4 个标题都偏保守（主打"recovery/revalidation/boundary audit"），可能不太匹配目标投稿会议 QCE 的风格。但标题选择不是 Worker 的决定范围，标记为非阻塞评论。

### 3.5 有没有破坏已有功能

没有。T35 只改了 5 个文档文件，且没有修改 T34 的台账文件。

### 3.6 文档有没有把计划写成事实

没有。这是最重点检查的维度：
- 骨架每个 section 都明确列出了 blocked claims 不能出现的区域
- 审计的"Minimum Safe Paper Positioning"给出了当前最安全的论文定位，没有夸大
- "Do-Not-Publish-As-Claimed List"列出了 5 种当前证据不支持的论文定位

### 3.7 我发现的非阻塞问题

我标记了 4 个非阻塞问题（N1-N4）：

1. **N1 标题候选偏保守**：4 个标题都主打"恢复/复验/边界审计"，可能太窄了。`docs/02_experiment_plan_simplified.md` 推荐的标题更偏方法/系统型，更适合目标投稿会议。但标题选择是 Captain/人类的决定。
2. **N2 缺少 Background/Related Work section**：骨架满足了任务最低要求的 8 个 section，但实验方案推荐了 Background section 来介绍 GKP syndrome、快慢回路时间尺度等背景。后续起草前应补上。
3. **N3 逐节审稿热点表偏泛化**：前面的详细表格（N1-N3, E1-E5, W1-W6, R1-R3, A1-A3）都绑定了具体 claim IDs，但最后的 section-by-section 表只给了高层描述，没有交叉引用前面的 table IDs。
4. **N4 Worker 自审被覆盖**：和 T34 一样的模式，信息已保留在任务包中。

这些问题都不阻塞 T35 收口，但 Captain 在安排后续论文写作任务时应考虑 N1 和 N2。
