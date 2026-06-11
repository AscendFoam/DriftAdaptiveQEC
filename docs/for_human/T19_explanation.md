# T19: Tracked Cache Cleanup Manifest — 通俗解释与 Review 说明

## 一、这个 Task 在做什么？（通俗版）

### 背景

你平时用 Python 写代码时，Python 会自动在 `__pycache__/` 目录里生成 `.pyc` 文件（编译后的字节码缓存）。这些文件是自动生成的，不应该放进 Git 版本库。

但是这个项目在早期没有做好 `.gitignore`，结果把 `116` 个 `.pyc` 文件和 `9` 个 `__pycache__` 目录都提交进了 Git 历史。虽然现在 `.gitignore` 已经配置正确（不会再提交新的缓存文件），但那些已经提交的旧缓存文件仍然"跟踪"在 Git 里。

这就像：门已经锁了（`.gitignore` 生效），但屋里原来就有的杂物还在（历史已提交的缓存文件）。

### 问题

这些缓存文件虽然不影响代码运行，但会造成以下困扰：

1. 每次 `git status` 或 `git diff` 都会看到一堆无关文件
2. 克隆仓库时多出不必要的文件
3. 搜索代码时容易被 `.pyc` 文件干扰

另外，仓库里还有更大的"杂物"——`1841` 个 `runs/` 文件和 `110` 个 `artifacts/` 文件，但这些不是本任务的处理范围。

### T19 做了什么

T19 的任务就是：**先不做清理，而是写一份详细的"清理计划书"**。具体来说：

- 清点仓库里到底有多少个已跟踪的缓存文件（答案：116 个，分布在 9 个目录）
- 写出将来要执行的清理命令（`git rm --cached`）
- 写出如果出问题怎么回滚（`git restore --staged`）
- 写出清理成功的验收标准（`git ls-files` 不再返回缓存文件）
- 明确列出**不碰的东西**（`runs/`、`artifacts/`、源码）

用一句话总结：**T19 是一份"仓库大扫除的施工方案"，只写计划，不动手。**

## 二、任务实现详细解释

### 2.1 任务目标

为已跟踪的 `__pycache__/` 与 `.pyc` 文件制定一份有界的 cleanup manifest，包含清点结果、执行命令草案、回滚方案和验收标准。

关键约束：**只做只读清点和计划，不执行任何删除操作。**

### 2.2 任务流程

1. **只读清点**：Worker 使用 `git ls-files` 统计了所有被 Git 跟踪的 `__pycache__/` 和 `.pyc` 文件。
   - 结果：`116` 个 `.pyc` 文件，全部位于 `9` 个 `__pycache__` 目录中
   - 额外散落在 `__pycache__` 之外的已跟踪 `.pyc`：`0` 个

2. **撰写 cleanup manifest**：产出 `docs/evidence_packs/repo_hygiene/cleanup_tracked_cache_manifest.md`，包含 8 个章节：
   - Scope（范围界定）
   - Read-Only Inventory（只读清点结果）
   - Target Paths（9 个目标目录清单）
   - Cleanup Command Draft（`git rm --cached` 命令草案）
   - Rollback Plan（`git restore --staged` 回滚方案）
   - Acceptance Criteria（验收标准）
   - Non-Touch Boundaries（不触碰范围）
   - Recommended Next Step（后续建议）

3. **更新治理文档**：同步更新了 `06_repo_noise_governance.md`、`04_task_board.md`、`07_handoff.md`、`08_risks_and_open_questions.md`。

4. **刻意不做的事**：
   - 没有勾选 T19 为完成（等审核后才勾）
   - 没有执行 `git rm`
   - 没有碰 `runs/` 或 `artifacts/`
   - 没有改源码

### 2.3 代码/配置文件的变化

**没有代码变更。** 没有配置文件变更。所有改动都在 `docs/` 目录内。

具体文件变化：

| 文件 | 变化类型 | 内容 |
|------|----------|------|
| `docs/evidence_packs/repo_hygiene/cleanup_tracked_cache_manifest.md` | 新增 | cleanup manifest 全文（8 节） |
| `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md` | 追加 | Worker Output Summary 段落 |
| `docs/06_repo_noise_governance.md` | 修改 | 第 3/7/8 节同步 T19 清点结论 |
| `docs/04_task_board.md` | 修改 | Current Unique Task 区域追加 Worker 只读产出 |
| `docs/07_handoff.md` | 修改 | 追加 T19 Worker 只读结果，更新当前状态描述 |
| `docs/08_risks_and_open_questions.md` | 修改 | R4/R7 更新，Q8/Q9 更新 |

### 2.4 对后续开发的意义

1. **为物理 cleanup 铺路**：现在有一份精确的施工方案，Captain 可以在任何时候批准一个执行任务，按 manifest 严格落地。不需要重新清点。

2. **回滚方案就绪**：`git restore --staged` 回滚命令已经写好，如果执行后发现问题，可以快速恢复。且明确禁止 `git reset --hard` 这种危险操作。

3. **边界清晰**：manifest 第 7 节列出了 6 项不触碰范围（`runs/`、`artifacts/`、benchmark 口径、`.tflite`、`real_board`、源码），防止后续执行时顺手做多余的事。

4. **与 T5/T20 形成互补**：T5 在 Phase 0 建立了噪声治理口径（"先治理后清理"），T19 把清理方案落到了可执行的纸上，T20 将处理真板准备清单。三个任务共同覆盖了仓库从"治理"到"清理"到"扩展"的路径。

5. **风险文档同步**：R4（缓存噪声）和 R7（cleanup 执行窗口）都更新为反映"T19 已有方案但未执行"的状态，保持了风险清单与实际进度的同步。

## 三、为什么给出 PASS 的 Review 结果？

### 审查逻辑

我从 6 个维度进行了检查：

1. **任务是否真的完成** — 是。T19 目标是制定 cleanup manifest，不是执行 cleanup。manifest 已产出，清点数字经独立验证全部准确（116 个文件、9 个目录、0 个散落文件）。

2. **是否有伪实现 / mock / stub / hardcode** — 无。整个任务是只读文档任务，没有执行任何删除。文档多处明确写"不等于已执行 cleanup"、"本轮不执行"。Task board 上 T19 仍为 `[ ]`，没有被提前勾选。

3. **是否缺测试或验证** — 不缺。任务包限定的验证是"只读清点"，Worker 已执行。我还用独立命令复验了所有数字，全部吻合。

4. **是否过度工程** — 否。8 节文档全部服务于单一目标（清点 + 方案 + 回滚 + 验收），没有多余抽象。命令草案是标准 git 命令，没有自定义脚本。

5. **是否破坏已有功能** — 否。无代码变更，无文件删除。

6. **文档是否把计划写成事实** — 否。关键表述全部诚实：
   - "本 manifest 是 `T19` 的只读产物，不等于已执行 cleanup"
   - "以下命令只是后续执行任务的草案，本轮不执行"
   - 第 7 节列出 6 项不触碰范围

### 两个非阻塞提醒

- **N1**：preflight 命令中使用了 glob 模式（`"cnn_fpga/__pycache__/*"`），在 Windows PowerShell 下可能有不同的行为。不影响当前只读任务，但后续执行时应注意。
- **N2**：tracked `.pyc`（116）与 workspace `.pyc`（133）的差异未在 manifest 中显式说明。差异来自未跟踪的新缓存文件，不影响结论。

### 结论

Worker 完全按任务包执行，清点数字准确，没有越界，没有伪实现，边界表述诚实。**PASS**。
