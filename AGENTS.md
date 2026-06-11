# AGENTS

本仓库已完成第一轮恢复期治理收尾。当前目标是在不破坏已恢复可信度的前提下，围绕已验证路径继续受控开发。

## 当前阶段

- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 当前唯一任务以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为准

## 开始任何工作前必须阅读

1. `README.md`
2. `docs/00_project_snapshot.md`
3. `docs/01_legacy_audit.md`
4. `docs/02_experiment_plan.md`
5. `docs/04_task_board.md`
6. `docs/07_handoff.md`
7. `docs/08_risks_and_open_questions.md`

如果任务涉及 HIL / P4 / `.tflite` / 真板边界，再补读：

- `docs/03_hil_p4_boundary_audit.md`
- `docs/recovery_bootstrap/README.md`
- `docs/evidence_packs/README.md`

如果任务涉及研究背景、阶段结论或后续计划，再补读：

- `docs/02_experiment_plan.md` Part I / Part II
- `docs/deep_research_reports/README.md`
- `docs/paper_materials/README.md`
- `docs/paper_notes/README.md`

说明：

- `docs/progress_summary/CNN_FPGA_GKP_阶段结论.md` 已退役为索引；当前阶段结论统一维护在 `docs/02_experiment_plan.md` Part I。
- `docs/follow-up_plan/README.md` 已退役为索引；当前后续计划统一维护在 `docs/02_experiment_plan.md` Part II。

## 角色约束

### Captain

- 负责拆任务、控范围、收口文档
- 不直接顺手扩功能
- 每轮必须明确：
  - 当前唯一任务
  - Allowed files
  - Forbidden scope
  - Verification
  - Docs to update

### Worker

- 只完成当前任务包
- 只改 Allowed files
- 不自动领取下一任务
- 修改代码时必须同步登记到对应 README / 文档索引；如果任务包没有允许文档文件，必须在完成汇报中明确“代码登记缺口”，不能偷偷越界修改
- 完成后必须汇报：
  - 改了什么
  - 怎么验证
  - 剩余风险

### Reviewer

- 默认只读
- 优先查：
  - 文档与代码是否一致
  - 是否把 mock / stub / placeholder 写成完成态
  - benchmark 是否公平
  - 环境假设是否被偷偷省略
  - 结果是否可复现

### 自定义角色

用户可以显式指定临时自定义角色，例如“文档整理员”“只做路径迁移”“只回答问题”“只按我给的清单修改”。自定义角色只根据用户本轮明确指令完成特定任务，不要求套用 Captain / Worker / Reviewer 的任务拆分、verdict、Allowed files 模板或 review 流程。

自定义角色的边界：

- 只有用户明确指定时才启用；默认仍按 Captain / Worker / Reviewer 语义理解仓库工作。
- 不自动推进 `Current Unique Task`，不自动创建下一任务包，不自动开关风险项。
- 不要求维护 `docs/04_task_board.md` / `docs/07_handoff.md`，除非用户明确要求或本轮工作本身就是治理同步。
- 仍必须遵守仓库硬规则：不得 overclaim `.tflite`、real-board、benchmark、statcalib、paper-grade 结论；不得改写历史 `runs/` / `artifacts/` 为新事实；不得无验证更新阶段结论。
- 如果自定义角色要修改代码，仍必须执行“代码修改与 README 登记规则”。

## 代码修改与 README 登记规则

本仓库要求通过文档记录代码结构和行为变化。任何修改 `cnn_fpga/`、`physics/`、`benchmark/` 或其他源码目录的任务，都必须考虑 README / 文档索引同步。

登记优先级：

1. 优先更新被修改目录内最近的 `README.md`。
2. 如果同目录没有 README，更新最近的上级模块 README。
3. 如果源码树内没有合适 README，更新负责代码结构说明的 `docs/codebase_overview/README.md` 或对应模块说明文档。
4. 如果变更属于 benchmark、runtime、HIL、`.tflite`、real-board、training reproducibility、statcalib 或 paper evidence，还要同步对应的 `docs/evidence_packs/**/README.md`、`docs/protocols/**/README.md` 或任务产物 README。
5. 如果当前任务包没有允许 README / 文档索引文件，Worker 不得擅自扩大 Allowed files；必须在交付中写明“需要后续 Captain 添加 README 登记任务”。

登记内容应简短但可追溯：

- 修改了哪个模块或入口；
- 行为、接口、配置或证据边界发生了什么变化；
- 对应 task package / review / evidence pack / run root / helper；
- 本次验证命令或未验证原因；
- 不能外推的边界，例如 mock-backed、isolated current-host、read-only gate、extension lane 等。

不要把 README 登记写成宣传文案。README 只记录当前可验证事实、代码入口、维护边界和阅读路径。

## 00-08 治理文档更新逻辑

治理文档不是每轮都全量重写。Captain 在任务包创建、review closeout 或当前唯一任务切换时，应按下面逻辑判断是否同步。

| 文档 | 角色 | 何时更新 |
| --- | --- | --- |
| `README.md` | 面向新读者的仓库入口 | 阶段、当前主入口、目录结构或关键边界发生稳定变化时更新 |
| `docs/00_project_snapshot.md` | 高层项目快照 | 当前阶段、Go/No-Go、当前唯一任务或高层证据边界变化时更新；不记录每个任务细节 |
| `docs/01_legacy_audit.md` | legacy 真实性审计底稿 | 只在发现会改变 legacy truth matrix 的事实时更新；普通任务 closeout 不应滚动追加 |
| `docs/02_experiment_plan.md` | 历史演进 + 后续计划唯一入口 | 阶段结论、P0-P4/T 系列高层转折、后续路线、候选任务池变化时更新 |
| `docs/03_hil_p4_boundary_audit.md` | HIL/P4/mock/`.tflite`/真板边界 | 证据等级或边界口径变化时更新；不得无验证改写 |
| `docs/04_task_board.md` | 当前任务主状态 | 创建/关闭任务、切换 Current Unique Task、记录 Captain verdict 时必须更新 |
| `docs/05_decision_log.md` | 决策日志 | Go/No-Go、gate verdict、warning 处理、重大范围决策变化时更新 |
| `docs/06_repo_noise_governance.md` | 仓库噪声与产物治理 | cleanup、tracked cache、`runs/` / `artifacts/` 口径变化时更新 |
| `docs/07_handoff.md` | 交接与下一步 | 每次 Captain 切换当前唯一任务时必须更新，且必须写明最新 worker-facing task package |
| `docs/08_risks_and_open_questions.md` | 风险台账 | review warning 被 deferred/rejected、风险打开/关闭/缩窄、边界风险变化时更新 |

Captain closeout 推荐顺序：

1. 读取 review，并分类 blocking / non-blocking / warning。
2. 决定 verdict：`PASS`、`PASS_WITH_WARNINGS`、`BLOCK` 或其他明确状态。
3. 先更新 `docs/04_task_board.md` 和 `docs/07_handoff.md`，保证当前唯一任务不漂移。
4. 再更新 `docs/05_decision_log.md` 和 `docs/08_risks_and_open_questions.md`，记录 warning 与风险处理。
5. 仅当高层快照变化时更新 `README.md` / `docs/00_project_snapshot.md`。
6. 仅当阶段结论、历史演进或后续计划变化时更新 `docs/02_experiment_plan.md`。
7. 仅当 legacy 真实性判断变化时更新 `docs/01_legacy_audit.md`。
8. 按任务产物所属目录同步对应 README 或索引文档。

Worker 不应在任务包未授权时修改 00-08 治理文档。Reviewer 默认只读，除非用户明确要求修正文档。

## docs 子目录登记

| 目录 | 当前用途 | 维护规则 |
| --- | --- | --- |
| `docs/tasks/` | task package | Worker 只能按当前任务包 allowed files 执行；历史任务包不代表当前任务 |
| `docs/review/` | Reviewer / Captain review 记录 | review 结果必须保留 verdict、blocking issues、warning classification |
| `docs/worker_summary/` | Worker 输出摘要 | 只记录任务执行摘要，不替代治理文档 |
| `docs/for_human/` | 面向人的解释材料 | 可解释任务结果，但不得升级证据等级 |
| `docs/recovery_bootstrap/` | P0/P3/P4 recovery smoke 复用入口 | 只维护已验证的最小 recovery 路径 |
| `docs/protocols/` | benchmark / execution protocol | protocol 先于执行任务；不得事后改写 benchmark 口径 |
| `docs/evidence_packs/` | 已完成任务证据包与边界说明 | 新证据必须绑定 task/review/run/artifact/helper |
| `docs/codebase_overview/` | 代码结构阅读说明 | 只作理解辅助，不是当前任务状态来源 |
| `docs/paper_materials/` | 论文 claim/evidence、草稿、风险审计 | 论文 claim 必须回指证据包和 review |
| `docs/paper_notes/` | 论文 note / LaTeX 草稿 | note 不等于当前计划或完成事实 |
| `docs/deep_research_reports/` | 深度调研报告 | README 标注可复用、过时、不匹配、已完成内容 |
| `docs/reference/` | 可参考但不承担当前计划状态的材料 | 不放当前任务状态；可转化建议必须先进入 `docs/02_experiment_plan.md` |
| `docs/sidecar/` | sidecar 扩展实验治理 | sidecar 输出不自动晋升主线，需 Captain promotion gate |
| `docs/legacy_context/` | 退役、归档、历史上下文 | 只作历史参考，不作为当前事实入口 |
| `docs/follow-up_plan/` | 已退役后续计划索引 | 不再维护新计划，统一转到 `docs/02_experiment_plan.md` Part II |
| `docs/progress_summary/` | 已退役阶段结论索引 | 不再维护阶段结论，统一转到 `docs/02_experiment_plan.md` Part I |
| `docs/reality_recovery/` | 已退役真实性复核说明 | 当前只保留退役说明，历史文件在 `docs/legacy_context/` |
| `docs/prompts/` | 外部模型/研究提示词 | prompt 不是证据来源，不能直接改写当前结论 |
| `docs/figure_assets/` | 图表素材 | 图表必须回指 result/figure ledger 或对应证据 |
| `docs/汇报用/` | 汇报材料与历史 presentation 产物 | 汇报材料不替代治理文档；历史构建 manifest 只作本地记录 |

## 当前仓库的特殊硬规则

1. 不得把 `P3-软件 HIL` 写成 `P3-真板 HIL 已完成`
2. 不得把 `board_backend.py` 的 placeholder 语义写成真实板级完成
3. 不得静默改动正式 benchmark 口径、baseline 集合、ParamMapper 主线语义
4. 不得把 `runs/`、`artifacts/` 中的历史结果改写为新的“事实来源”
5. 不得无任务包顺手启动新的 teacher-representation 长跑或正式长跑 benchmark
6. 不得跳过验证就更新阶段结论类文档
7. 不得把 `T48` 写成默认环境、HIL 或 deployment closure
8. 不得把 `T49/T71/T72` 写成 real-board execution success
9. 不得把 `T64`-`T70` 写成 mature `statcalib` comparator promotion

## 当前阶段默认允许的任务类型

- 在已验证路径上做有界开发
- 补更强的 benchmark / benchmark 边界证据
- 为训练链、`.tflite` 或真板路径补独立 manifest / bootstrap / smoke
- 补有界 cleanup 任务
- 补最小测试、回归验证或治理文件

## 当前阶段默认禁止的任务类型

- 无任务包的大规模重构
- 无验证支撑的新模型主线切换
- 无边界说明的新论文分支大规模展开
- 真板联调语义扩写成既成事实
- 在无验证前提下重写阶段结论文档
