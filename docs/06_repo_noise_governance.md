# Evaluation Protocol And Repo Noise Governance

本文件对应 `docs/reference/AI_coding_workflow.md` 中 `06_eval_protocol.md` 的当前项目版本，同时保留恢复期形成的 repo noise governance。Phase 2 的所有 Worker 应先按本文件确定“什么算验证通过”，再记录或引用新结果。

## 1. 目的

本文件有两个目的：

1. 固定当前项目的评估协议，避免 recovery smoke、development smoke、正式 benchmark 被混写。
2. 继续保留 `T5` 形成的缓存、生成物与历史运行噪声治理口径。

本文件不宣称仓库已经完成物理清理。

## 2. Evaluation Protocol

### 2.1 Evidence Levels

| Level | 含义 | 可写结论 |
| --- | --- | --- |
| `code_exists` | 入口或模块存在，未在当前阶段复验 | “代码存在，待复验” |
| `recovery_smoke` | 用 recovery-scoped config / manifest 跑通 | “bounded recovery path 可复验” |
| `deterministic_recovery_smoke` | recovery smoke 在固定 seed 链下逐字一致 | “bounded path 已逐字一致复验” |
| `development_smoke` | 在受控开发任务中扩展场景、mode 或环境，但仍非正式长跑 | “development evidence，需附边界” |
| `formal_benchmark` | 预先冻结 protocol、baseline、seed、repeat，并通过 review | “正式 benchmark 结果” |
| `hardware_validated` | 真实板级 backend、设备、日志与验收齐备 | “real-board validated” |

当前最高已确认等级：

- P3 software HIL recovery path：`deterministic_recovery_smoke`
- P4 frozen baseline：`recovery_smoke`
- real-board HIL：未达到 `hardware_validated`
- `.tflite` runtime：当前 Phase 2 尚未重新复验

### 2.2 必须记录的字段

任何新运行结果都必须记录：

1. 命令
2. 解释器
3. config path
4. output run dir
5. scenario / mode / repeats / seed pairing
6. backend
7. inference mode 与 artifact type
8. 关键 summary / comparison 字段
9. 结论边界

### 2.3 当前推荐验证入口

- P0 recovery smoke：`docs/P0_smoke_bootstrap.md`
- P3 software HIL recovery smoke：`docs/P3_software_hil_bootstrap.md`
- P4 recovery smoke：`docs/P4_benchmark_recovery_bootstrap.md`
- Phase 2 P4 development protocol：`docs/P4_benchmark_development_protocol.md`（由 `T14` 产出）
- Phase 2 P4 formal software protocol：`docs/P4_benchmark_formal_protocol.md`（由 `T23` 产出，并由 `T24` 写入 frozen-set formal software revalidation execution record）

### 2.4 禁止的评估写法

1. 不把 `single-scenario + repeats=1` 写成正式多场景 benchmark。
2. 不把 `mock` backend 写成真板验证。
3. 不把 `.tflite.json` stub manifest 写成真实 `.tflite` runtime。
4. 不把整个 `runs/` 或 `artifacts/` 目录写成事实来源；必须引用具体 run dir 或 artifact path。
5. 不在未跑验证时更新阶段结论文档。

## 3. 当前噪声清点

基于 `2026-05-07` 的只读清点，当前可确认的事实如下：

- `.gitignore` 已经忽略：
  - `__pycache__/`
  - `runs/`
  - `artifacts/`
- 但 Git 历史中仍已有大量已跟踪噪声，这说明当前问题主要是“历史已跟踪文件”，而不是“缺少忽略规则”。
- 已跟踪缓存/字节码文件：`116`
  - 分布于 `9` 个 `__pycache__` 目录
  - `T19` 只读清点确认：上述 `116` 个文件全部位于 `__pycache__` 目录内，额外散落的已跟踪 `.pyc` 文件数为 `0`
- 当前工作区 `.pyc` 文件总数：`133`
- 已跟踪 `runs/` 文件数：`1841`
- 已跟踪 `artifacts/` 文件数：`110`

代表性内容：

- `runs/`
  - 包含 `drift_suite`、`hardware_emulation`、`hil_mode_benchmark`、`hil_suite`、`p4_*` 等历史运行输出
- `artifacts/`
  - 包含数据集 `.npz`
  - 模型 `.npz`
  - `.tflite`
  - `.tflite.json`
  - 训练、导出、评估报告
- 当前恢复期最小 software HIL 路径仍依赖：
  - `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`

## 4. 噪声分类与治理口径

| 类别 | 路径模式 | 当前状态 | Phase 0 是否保留 | 当前治理口径 | 后续动作 |
| --- | --- | --- | --- | --- | --- |
| A | `**/__pycache__/`, `*.pyc` | 已忽略，但历史上已有已跟踪文件 | 是 | `T5` 不做批量删除或 untrack；后续禁止再把新缓存文件纳入版本库 | 未来单开有界 cleanup 任务，将其清理出版本库 |
| B | `runs/` | 已忽略，但历史运行结果大量已跟踪 | 是 | 视为历史证据，不视为新的事实来源；`T6/T7` 必须引用具体 run dir，而不是笼统引用 `runs/` | 后续区分“保留摘要”与“归档移出” |
| C | `artifacts/` | 已忽略，但历史模型/数据/报告已跟踪 | 是 | 暂保留，因为当前恢复期最小路径仍依赖部分 `.npz` artifact；不可默认把整个目录当作源码一部分 | 后续拆分为“bootstrap 必需”和“历史归档”两类 |
| D | 本地临时文档文件，例如 `*.drawio.dtmp` | 容易制造工作树噪声 | 否 | 立即通过 `.gitignore` 忽略，不进入治理结论文档 | 保持忽略即可 |

## 5. 当前立即执行的规则

1. 在专门 cleanup 任务出现前，不做破坏性清理。
2. 在专门 cleanup 任务出现前，不执行：
   - `git rm --cached` 批量移除 `__pycache__/`
   - `git rm --cached` 批量移除 `runs/`
   - `git rm --cached` 批量移除 `artifacts/`
   - 批量删除历史 `.pyc`
3. 后续任务若产生新的运行结果，允许本地落盘，但必须：
   - 在文档里写清解释器、命令、run dir、backend、artifact type
   - 不把历史结果目录直接写成“当前事实来源”
4. 当前恢复期最小路径依赖的历史 artifact，可以继续被引用，但要显式写清具体文件路径。
5. `board_backend.py` 仍是 placeholder real-board backend；`.tflite` 真路径与 stub 路径仍按 `T3/T4` 的既有口径区分，不能被噪声治理任务改写语义。

## 6. T5 实际落地动作

本轮实际执行了以下非破坏性动作：

1. 新增本治理文档，固定噪声分类与阶段口径。
2. 新增 `T5` 任务包，限制本轮作用域只在治理文档侧。
3. 给 `.gitignore` 增补 `*.drawio.dtmp`，避免 Draw.io 临时文件继续制造工作树噪声。
4. 同步更新 task board、decision log、handoff、legacy audit 与风险文档。

本轮没有执行：

- `runs/` 清理
- `artifacts/` 清理
- `__pycache__/` / `.pyc` untrack
- 任意 benchmark 口径改写

## 7. 后续清理建议

物理 cleanup 应在后续单独任务中进行，并至少满足以下条件：

1. 先完成 `T6/T7`，避免在最小复验前误伤当前可复用入口。
2. cleanup 任务必须提供明确 manifest：
   - 哪些 `__pycache__/` / `.pyc` 会被移出版本库
   - 哪些 `runs/` 只保留摘要
   - 哪些 `artifacts/` 被认定为 bootstrap 必需
3. cleanup 前必须明确回滚方式和验收标准。

`T19` 已补出：

- `docs/cleanup_tracked_cache_manifest.md`
- 显式目标目录清单（9 个 `__pycache__` 目录）
- `git rm --cached -r -- ...` 草案
- `git restore --staged -- ...` 回滚草案
- 验收标准：`git ls-files | rg "__pycache__|\\.pyc$"` 归零

但 `T19` 仍未执行物理 cleanup。

## 8. 对后续任务的约束

- `T14/T15` 若进入 P4 证据增强，仍必须显式写清 backend 与 inference artifact type。
- `T17/T18` 只能补 manifest / bootstrap，不应把环境说明写成完整验证完成。
- `T19` 只允许先处理 tracked cache cleanup manifest；`runs/` 和 `artifacts/` 另行拆分。
- `T19` 不执行物理删除，不执行 `git rm`，只做只读清点与 cleanup 方案。
- 若后续进入执行任务，作用域仍应限制在 `docs/cleanup_tracked_cache_manifest.md` 列出的 9 个目录内。
- `T19` review verdict = `PASS`；后续如要执行 physical untrack，仍必须单开任务，不得借 `T20` 或真板 readiness 任务顺手 cleanup。
- `T21` milestone review 也不得执行 physical untrack；cleanup 执行仍必须在独立任务中按 `docs/cleanup_tracked_cache_manifest.md` 落地。
- `T22` real-board smoke execution plan 不得顺手执行 cleanup，也不得把 `runs/` / `artifacts/` 纳入物理清理。
- `T23` P4 formal protocol lock 不得顺手执行 cleanup，也不得把历史 `runs/` / `artifacts/` 改写成新的 formal 事实来源；只能引用具体路径来定义 evidence pack 和缺口。
- `T24` P4 formal software revalidation 可以新增一个 `runs/p4_benchmark/T24_formal_software_revalidation_*` 运行目录，但不得清理、改写或重标历史 `runs/` / `artifacts/`；所有新结论必须引用该具体 run dir。
- `T25` gate review 不得新增 run dir，也不得把 T24 run 外推为 `.tflite` runtime、真板验证或 paper-grade expanded benchmark；若需要后续机制审计或部署验证，必须新开任务包。
- `T27` teacher diagnostics audit 只能只读引用既有 T15/T24 run outputs 和源码路径；不得新增 run dir，不得把 header-only diagnostics 改写成机制证据已完成。
- `T28` teacher diagnostics 语义修复若需要最小 smoke，只能新增 T28 专属 run dir；不得改写 T15/T24 历史 outputs，也不得把最小 smoke 结果升级为新的 formal benchmark。
- `T28` 执行产生的 tracked `.pyc` 改动不应作为有意义结果提交；若提交前需要处理，应按 T19 tracked-cache governance 排除或后续单开 cleanup，不得混入技术结论。
- `T29` P4 markdown report cleanup 不得新增 run dir，也不得清理、改写或重标 T28 smoke outputs；只允许修复报告表头格式和必要文档记录。
