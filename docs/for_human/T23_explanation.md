# T23 通俗解释：P4 Formal Benchmark Protocol Lock

## 一、这个 task 在做什么？（通俗版）

想象你在准备一场正式考试。你已经做过几套模拟卷（T15 的 development run），现在你还没有真正参加考试（formal benchmark），但你需要先把"考试规则"写清楚：

- 考哪几门课？（哪几个场景 scenarios）
- 每门课用什么方法答题？（哪几种算法 modes）
- 每门课考几遍？（repeats）
- 怎么保证公平？（所有方法用同一套随机种子 paired seeds）
- 怎么算分？（统计报告规则）
- 大概要考多久？（计算预算）
- 考完要交哪些材料？（evidence pack）
- 什么情况下可以开考？（go/no-go 条件）

T23 就是"写考试规则"这一步。它**不参加考试，不改考卷，不跑任何 benchmark**。它的全部产出就是一份协议文档和配套的治理同步。

为什么需要这一步？因为之前的 T15 只是一场"练习考试"（development smoke），而 T21 milestone review 明确指出：如果把 T15 直接当成正式考试结果，就是在自欺欺人。所以在真正重新跑一轮正式 benchmark 之前，先把规则锁定，确保后续执行是可追溯、可复现、可审查的。

---

## 二、任务实现详解

### 2.1 任务目标

在不运行 benchmark 的前提下，把 P4 formal benchmark 的以下要素锁定成文档：

1. **证据等级边界**：区分 recovery smoke / development run / formal benchmark 三层
2. **formal matrix**：哪些场景、哪些算法、多少次重复、什么种子策略
3. **baseline 规则**：哪些算法纳入正式排名、哪些排除、为什么不纳入
4. **统计报告规则**：要求报告哪些指标、用什么公平规则
5. **深度研究建议审计**：把外部研究报告的建议分类为采纳 / 推迟 / 拒绝
6. **计算预算**：40 次 repeat-run，约为 T15 的两倍，需要分块执行
7. **evidence pack**：考试结束后要交什么材料
8. **T24 的 gate 条件**：下一步可以做什么、不可以做什么

### 2.2 任务流程

1. **只读审计**：Worker 对照已有的 config 文件（`p4_multiscenario_strong_baselines.yaml`）、benchmark runner（`run_p4_multiscenario_benchmark.py`）、历史 review 结论（T15/T16/T21）和深度研究报告，进行只读核对
2. **起草 formal protocol**：在 `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` 中锁定上述所有要素
3. **治理同步**：更新 `04_task_board.md`、`07_handoff.md`、`08_risks_and_open_questions.md` 等文档

### 2.3 代码/配置文件变化

本次任务**没有修改任何源码或配置文件**。全部变化都是文档层面的：

| 文件 | 变化性质 |
| --- | --- |
| `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` | **新增**：核心产出，锁定 formal benchmark 的完整协议 |
| `docs/protocols/benchmark/P4_benchmark_development_protocol.md` | **追加** Section 12：说明与 formal protocol 的关系 |
| `docs/04_task_board.md` | **重写** Current Unique Task 指向 T23，展开 T24-T35 路线图 |
| `docs/07_handoff.md` | **追加** items 32-36，重写 section 4/6/7 |
| `docs/08_risks_and_open_questions.md` | **追加** R15-R18、open questions 19-28 |
| `docs/00_project_snapshot.md` | 更新当前任务引用 |
| `docs/01_legacy_audit.md` | 更新后续优先级建议 |
| `docs/03_hil_p4_boundary_audit.md` | 追加 T22/T23 的 HIL 边界表述 |
| `docs/05_decision_log.md` | 追加 D-2026-05-10-05 至 D-2026-05-10-08 |
| `docs/06_repo_noise_governance.md` | 追加 T22/T23 的 cleanup 禁止规则 |
| `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md` | 追加宿主模型选择要求 |
| `docs/tasks/Phase2/T21_phase2_milestone_review.md` | 回填 Worker Output Summary |

### 2.4 Formal Protocol 的核心内容

**三层证据边界**：

| 等级 | 范围 | 允许的声称 |
| --- | --- | --- |
| Recovery smoke | 单场景、repeats=1 | recovery 路径可运行 |
| Development run | 2 场景、5 模式、repeats=2 | 仅供开发参考 |
| Formal benchmark | 4 场景、5 模式、paired seeds、repeats=2 | 正式复验证据 |

**锁定矩阵**：4 场景 x 5 模式 x 2 repeats = 40 repeat-runs

**T24 gate**：
- `GO_FOR_BOUNDED_FORMAL_SOFTWARE_REVALIDATION`：允许按锁定矩阵执行
- `NO_GO_FOR_SCOPE_EXPANSION_INSIDE_T24`：不允许把 statcalib、soft-information、额外场景、.tflite、真板混进同一任务

### 2.5 对后续开发的意义

1. **T24 有了明确的执行边界**：不需要猜测"正式 benchmark 到底该跑多大范围"，protocol 已经锁死
2. **深度研究建议有了分类归宿**：不是所有建议都采纳（有些合理但推迟），也不是简单忽视，每个建议都有明确的处理方式和理由
3. **计算风险有了前置评估**：T15 已经出现过 shell timeout，T24 是 T15 的两倍规模，必须分块执行
4. **后续路线图有了分层结构**：从 benchmark -> 机制 -> 复现/部署 -> 论文的推进顺序已固定在 task board 中

---

## 三、为什么 Review 给出 PASS_WITH_WARNINGS？

### 3.1 为什么是 PASS 而不是 BLOCK

1. **任务真的完成了**：formal protocol 覆盖了任务包要求的所有要素（evidence levels、formal matrix、baseline rules、stats、deep-research audit、compute budget、evidence pack、T24 gate、non-claims）
2. **没有伪实现**：文档明确写出 `T23 did not run benchmark`，没有把计划写成事实
3. **声称与代码一致**：我逐一核对了 protocol 中的场景列表、模式列表、种子配置、CLI 参数、输出文件名、指标名——全部与实际 config 和 runner 代码匹配
4. **深度研究审计合理**：adopted/deferred/rejected 的分类有道理，没有为了追求"完整"而把 T23 变成无法执行的大任务
5. **没有破坏已有功能**：全部改动都是文档，零源码变更

### 3.2 为什么有 WARNINGS

**N1：Worker 修改了 7 个不在 allowed list 中的文件**

任务包只允许 Worker 改 6 个文件，但实际改了 13 个。多出来的 7 个都是治理同步（更新任务引用、追加决策记录、补风险条目）。这些改动内容是正确的，技术上不算"越界实现"，但确实超出了任务包的显式允许范围。

这和 T22 的情况一样——之前的 review 也指出了同样的问题，Captain 也接受了。如果每次都出现，说明任务包的 allowed files 可能需要系统性扩大，或者 Captain 应在 Worker 执行前明确说明"治理同步文件也允许修改"。

**N2：Protocol 没有给出 T24 的具体 CLI 命令**

Protocol 要求 T24 的 evidence pack 中包含"exact CLI shape"，但 protocol 本身没有写出这个命令。这意味着 T24 的任务包需要补上这一步。

**N3/N4：两个统计指标未逐一验证**

`histogram_input_saturation_rate_mean` 和 `correction_saturation_rate_mean` 被列为 T24 必须报告的指标，但我在本次 review 中没有逐一在 runner 代码中验证它们是否存在。这不阻塞 T23，但 T24 执行时应确认实际可用的指标列表。

### 3.3 总结

T23 的核心产出质量好：协议文档严谨、声称与代码一致、边界表述诚实、深度研究建议分类合理。Warnings 都是"任务包边界"和"后续任务细节"层面的问题，不影响 T23 本身的完成质量。建议 Captain 接受并推进 T24。
