# 00d AI Coding 工作流

本文件用来指导后续项目开发中的 AI coding 会话管理。

定位：这是操作手册，不是新的调度系统。项目状态仍以仓库文件为准，AI 会话只是临时执行者。

## 0. 总原则

1. 仓库文件是主状态，session 不是主状态。
2. 每轮只推进一个当前唯一任务。
3. Captain 负责拆解、调度、整合和更新文档，Worker 负责单任务实现，Reviewer 负责只读审查。
4. 不让两个 agent 同时修改同一批文件。
5. 不把计划、mock、stub、未来能力写成已完成事实。
6. 每个任务完成后必须有可验证结果、风险记录和下一步唯一任务。
7. 中枢项目只管理项目状态、证据和简历素材，不吞并子项目源码。

## 1. 角色分工

### 你

最终裁决者。只做 4 件事：

1. 批准是否立项。
2. 选择当前唯一任务。
3. 判断 review 中哪些意见必须修、哪些可暂缓。
4. 决定项目进入 Continue / Narrow / Pause / Archive / Resume-ready。

### Codex Captain

项目开发的主控会话。职责：

1. 阅读项目治理文件。
2. 维护 `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md`。
3. 把任务拆成 worker 可执行的任务包。
4. 整合 worker 结果和 reviewer 意见。
5. 不直接进行大规模实现。

Captain 每轮必须输出：

```text
1. 当前唯一任务
2. 为什么现在做它
3. Worker 任务包
4. 允许修改的文件范围
5. 禁止做的事
6. 验证命令或验收标准
7. 完成后需要更新的治理文件
```

### Codex Worker

单任务实现会话。职责：

1. 只完成 Captain 指定的任务。
2. 只改任务包允许的文件。
3. 运行相关验证。
4. 汇报改动、验证、风险。
5. 不自动领取下一任务。

### Claude Code Reviewer

审查会话。默认只读，不改代码。职责：

1. 审查 diff 是否完成任务。
2. 找 bug、伪实现、mock、hardcode、缺测试、过度工程。
3. 判断是否 PASS / PASS_WITH_WARNINGS / BLOCK。
4. 给出最小修复建议。

如果已安装 Codex 调 Claude 的插件，可以显式使用：

```text
$claude-review
$claude-adversarial-review
```

如果插件不稳定，就用单独的 Claude Code 会话手动 review。不要把插件或 hook 当成当前项目的核心基础设施。

### ChatGPT / Gemini

适合做前期研究和高层判断：

1. raw idea 初筛。
2. 文献调研和查重。
3. 理论路线比较。
4. 实验设计审稿。
5. 项目是否值得继续的外部视角判断。

## 2. 所有项目都应维护的文件

新项目建议采用：

```text
README.md
AGENTS.md
CLAUDE.md
docs/
  00_raw_idea.md
  01_feasibility_report.md
  02_experiment_plan.md
  03_architecture.md
  04_task_board.md
  05_decision_log.md
  06_eval_protocol.md
  07_handoff.md
  08_risks_and_open_questions.md
```

最小启动时也至少要有：

```text
README.md
AGENTS.md
docs/02_experiment_plan.md
docs/04_task_board.md
docs/07_handoff.md
```

`qcy_project_hub` 负责引用子项目路径和证据，不替代子项目自己的这些文件。

## 3. 未正式开发项目的工作流

适用于只有 idea、调研报告或空仓库的项目。

### A0. Raw Idea 初筛

工具：ChatGPT / Gemini。

输出：`docs/00_raw_idea.md`。

必须回答：

```text
1. 解决什么问题
2. 为什么现在值得做
3. 最小可验证实验
4. 最相似已有工作
5. 失败标准
```

决策：

```text
Go: 进入调研
Revise: 缩小 idea 后重审
Stop: 不建项目
```

### A1. 可行性调研

工具：ChatGPT / Gemini 深度研究。

输出：`docs/01_feasibility_report.md`。

必须包含：

```text
1. 问题定义
2. 相关工作矩阵
3. 最像的 5 个已有工作
4. 可差异化点
5. MVP 实验
6. 风险
7. Go / No-Go 判断
```

硬规则：不能只收集支持 idea 的材料，必须找最可能撞车的工作。

### A2. 工程实验方案

工具：Claude Code 主写，Codex 或 ChatGPT/Gemini 审稿。

输出：`docs/02_experiment_plan.md`、`docs/06_eval_protocol.md`。

必须写清：

```text
1. MVP 目标
2. 不做什么
3. 输入输出
4. 目录结构
5. 核心模块
6. 实验流程
7. 评价指标
8. 失败判据
9. 3 个里程碑
10. 每个里程碑验收标准
```

### A3. Captain 初始化项目

工具：Codex Captain。

操作：

1. 创建或校正 `AGENTS.md`、`CLAUDE.md`、`README.md`。
2. 建立 `docs/03_architecture.md`。
3. 建立 `docs/04_task_board.md`。
4. 写入 `docs/07_handoff.md`。
5. 把第一个任务设为 `Current Unique Task`。

任务板格式：

```markdown
# Task Board

## Milestone 1: MVP

- [ ] T1: ...
- [ ] T2: ...

## Current Unique Task

T1: ...
```

### A4. 单任务开发循环

每个任务都按下面顺序执行。

#### 1. Captain 生成任务包

任务包模板：

```text
Task ID:
Goal:
Why now:
Allowed files:
Forbidden scope:
Inputs to read:
Expected output:
Verification:
Docs to update:
Reviewer type: normal / adversarial / milestone
```

#### 2. Worker 执行

Worker prompt：

```text
你是 Codex worker。

请先阅读：
- README.md
- AGENTS.md
- docs/02_experiment_plan.md
- docs/04_task_board.md
- docs/07_handoff.md

本轮只完成以下任务：
[粘贴 Task ID 和任务包]

规则：
1. 只改 Allowed files。
2. 不做 Forbidden scope。
3. 不领取下一任务。
4. 完成后运行 Verification。
5. 更新任务包指定的 docs。
6. 最后报告：改了什么、如何验证、剩余风险。
```

#### 3. Reviewer 审查

Reviewer prompt：

```text
你是 Claude Code reviewer。

只读审查本次 diff，不要修改文件。

重点检查：
1. 是否真的完成任务
2. 是否有伪实现、mock、stub、hardcode
3. 是否缺测试或验证
4. 是否过度工程
5. 是否破坏已有功能
6. 文档是否把计划写成事实

输出：
- Verdict: PASS / PASS_WITH_WARNINGS / BLOCK
- Blocking issues
- Non-blocking issues
- Missing tests
- Suspicious implementation details
- Recommended next action
```

高风险任务用 adversarial review：

```text
核心算法、实验指标、数据 pipeline、旧项目迁移、架构变更、简历素材生成。
```

#### 4. Captain 整合

Captain 根据 review 做判断：

```text
PASS:
  标记任务完成，更新 handoff，推荐下一任务但不执行。

PASS_WITH_WARNINGS:
  把 warning 分类为 accepted / deferred / rejected。
  deferred 写入 risks。

BLOCK:
  让 worker 只修 blocking issue。
  同一任务最多自动复审一次。
  第二次仍 BLOCK，则停止，交给用户裁决。
```

### A5. 里程碑闸门

每个 milestone 结束后暂停开发，做一次里程碑审查。

输出：`reviews/` 下的 milestone review。

审查问题：

```text
1. 当前功能是否真的完成
2. 是否能从干净环境运行
3. 是否有测试、demo 或实验结果
4. 是否存在伪完成
5. 是否允许进入下一里程碑
6. 对简历证据等级有什么影响
```

结论只能是：

```text
Allow
Conditional
Block
```

### A6. 同步到中枢项目

只有当项目状态或证据发生变化时，才更新 `qcy_project_hub`：

1. `data/projects.yaml`
2. `projects/<project_id>.md`
3. `resume_fragments/<project_id>.yaml`
4. `reviews/`

简历规则：

```text
L0: 不写
L1: 一般不写，最多内部候选
L2: 可保守写入候选
L3: 可进入正式项目经历候选
L4: 重点项目
L5: 核心项目
```

## 4. 已开发一部分项目的工作流

适用于长 session 开发过、开发到一半、快完成但缺审查、文档和代码可能不一致的项目。

核心目标：先恢复可信度，再继续开发。

### B0. 冻结旧状态

不要继续加功能。先做：

```text
1. git status
2. 记录当前分支和未提交改动
3. 如果可以，提交 checkpoint
4. 收集旧 session summary 到 docs/legacy_context/
```

旧 session summary 只能当线索，不能当事实。

### B1. 只读 Legacy Audit

工具：Claude Code 或 Codex 只读会话。

输出建议：

```text
docs/01_legacy_audit.md
docs/00_project_snapshot.md
```

如果项目已经采用九件套，就把审计结论写入现有 `docs/07_handoff.md` 和 `docs/08_risks_and_open_questions.md`。

Audit prompt：

```text
当前项目是一个已经由 coding agent 开发过的项目。

本轮只读审计，不要改代码。

请检查：
1. 项目目标和当前 MVP 是否清楚
2. README/docs 声称完成了什么
3. 代码实际完成了什么
4. 入口命令和依赖是否清楚
5. 测试覆盖情况
6. 疑似 mock/stub/hardcode/placeholder
7. 文档与代码不一致处
8. 下一步唯一恢复任务

所有判断必须引用文件路径。
```

建议建立 Feature Reality Matrix：

```markdown
| Feature | Claimed status | Evidence path | Verified? | Risk |
|---|---|---|---|---|
| ... | ... | ... | yes/no | ... |
```

### B2. 重建治理文件

如果旧项目没有清晰治理文件，先补最小集合：

```text
AGENTS.md
CLAUDE.md
docs/02_experiment_plan.md 或 docs/00_project_snapshot.md
docs/04_task_board.md
docs/07_handoff.md
docs/08_risks_and_open_questions.md
```

迁移期任务板应以恢复可信度为主：

```markdown
## Phase 0: Stabilization

- [ ] T0: 冻结 legacy 状态并完成只读审计
- [ ] T1: 确认依赖和最小入口
- [ ] T2: 跑通最小 demo 或记录阻塞
- [ ] T3: 审计核心算法或核心 pipeline
- [ ] T4: 补核心路径 smoke test
- [ ] T5: 标记 mock/stub/hardcode

## Current Unique Task

T1
```

迁移期禁止：

```text
新增功能
大规模重构
改技术栈
补复杂 UI
把旧文档改成完成态宣传稿
```

### B3. 最小运行验证

工具：Codex Worker。

任务目标：找出项目能否跑起来。

Worker 只做：

```text
1. 找依赖文件
2. 找入口命令
3. 运行最小 demo / test / pipeline
4. 记录真实输出或报错
5. 只修阻塞运行的最小问题
6. 更新 handoff
```

分流：

```text
能跑: 进入测试补强
不能跑但阻塞明确: 先修环境或入口
不能跑且结构混乱: 进入 Salvage
```

### B4. 核心可信度审查

工具：Claude Code adversarial reviewer。

重点审：

```text
1. 核心算法是否真实实现
2. 是否只是 baseline 换名
3. 是否有 hardcoded 示例输入
4. 是否有 mock 结果伪装真实结果
5. 指标是否真实计算
6. 是否有数据泄漏
7. baseline 是否公平
8. 随机性和结果是否可复现
```

结论：

```text
Allow: 可恢复正常开发
Conditional: 先补验证或修核心问题
Block: 不允许继续加功能
```

### B5. Recovery 开发循环

迁移期每个任务都应是修复型任务：

```text
验证...
补齐...测试
删除或标记...伪实现
替换...mock
固定...入口
复现...实验
```

不要写：

```text
继续优化...
顺手增加...
全面重构...
```

当满足以下条件，迁移才算完成：

```text
1. 能按文档跑最小路径
2. MVP 范围明确
3. task board 明确
4. handoff 明确
5. 核心功能可信等级至少 L2
6. 有 smoke test 或等价验证
7. mock/stub/hardcode 已标记
8. 不再依赖旧长 session
```

### B6. Go / Repair / Salvage / Stop

迁移审查后做一次决策：

```text
Go:
  项目能跑，核心功能可信，继续开发成本合理。

Repair:
  方向对，代码有价值，但入口、测试、文档不足。

Salvage:
  局部模块有价值，整体结构混乱。新建 clean branch/repo，只搬可验证模块。

Stop:
  idea 不成立、撞车严重、结果不可信或继续收益低。
```

这个决策要写入 `docs/05_decision_log.md` 或对应 review 记录。

## 5. 多 Worker 使用规则

默认顺序执行。只有满足以下条件才开多个 worker：

1. Captain 已拆出互不依赖的任务。
2. 每个 worker 的允许修改文件不重叠。
3. 每个 worker 使用独立分支或 worktree。
4. Captain 明确合并顺序。
5. Reviewer 分别审查每个 diff。

不满足这些条件时，不要并行。

推荐分支命名：

```text
agent/codex-T12-feature-name
agent/claude-review-T12
recovery/salvage-mvp
```

## 6. 每天实际操作顺序

### 开始

打开 Captain 会话：

```text
请作为 Codex Captain 工作。

先阅读：
- README.md
- AGENTS.md
- docs/02_experiment_plan.md
- docs/04_task_board.md
- docs/07_handoff.md
- docs/08_risks_and_open_questions.md

不要直接实现。
请根据 Current Unique Task 生成一个 worker 任务包。
```

### 执行

把任务包交给 Worker。Worker 完成后停下。

### 审查

把 diff 交给 Claude Code Reviewer，或者显式运行：

```text
$claude-review
```

高风险任务运行：

```text
$claude-adversarial-review
```

### 收口

Captain 更新：

```text
docs/04_task_board.md
docs/07_handoff.md
docs/08_risks_and_open_questions.md
docs/05_decision_log.md（如有关键决策）
```

结束时必须能回答：

```text
1. 今天完成了什么
2. 哪些已验证
3. 哪些未验证
4. 当前证据等级是否变化
5. 下一步唯一任务是什么
```

## 7. 完成标准

一个 AI coding 任务只有同时满足下面条件，才算完成：

1. 任务包目标完成。
2. 未越界修改。
3. 验证已运行，或明确说明无法验证。
4. Reviewer 没有未处理的 BLOCK。
5. 文档状态与代码状态一致。
6. `handoff` 已更新。
7. 下一步唯一任务已明确，但未自动执行。
