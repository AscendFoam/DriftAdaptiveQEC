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
- 当前 Git index 中已跟踪缓存/字节码文件：`0`
  - `T33` 已按 `T19` manifest 将先前 `116` 个 tracked `.pyc` 文件从 Git index 中移除
  - 上述 tracked cache 仅涉及 `9` 个 `__pycache__` 目录；未扩展到任何非 manifest 路径
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
| A | `**/__pycache__/`, `*.pyc` | 已忽略；`T33` 已完成历史 tracked cache 的 bounded untrack | 是 | 不再把新的缓存文件纳入版本库；不得借机扩展到其他 cleanup 范围 | 对 tracked cache 本身无需继续执行；其他噪声类型仍需独立任务 |
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

随后 `T33` 已按该 manifest 执行 bounded physical cleanup，并通过 Captain `PASS` 收口。

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
- `T29` review 发现的 tracked `.pyc` side-effect 继续按已知 repo-noise 处理：不作为技术改动提交，不因此开启临时 cleanup。
- `T26` statcalib feasibility gate 只能做 docs-only/read-only 审计；不得借机清理 tracked cache、`runs/` 或 `artifacts/`。
- `T30` statcalib comparator interface/implementation package 不得借机执行 tracked-cache cleanup、改写历史 `runs/` / `artifacts/`、或把接口 smoke 输出升级为 formal benchmark evidence；其测试产生的 `tests/__pycache__/` side-effect 只能按 repo-noise 处理，不作为任务输出提交。
- `T36` seed failure diagnosis 已完成，只读取既有 `runs/teachrepr*` 结果并输出诊断文档/小型分析脚本；不得把该诊断输出重标为新的 benchmark run evidence。
- `T38` seed trace-export probe 允许创建一个 T38-scoped run directory，但不得重写历史 `runs/` / `artifacts/`、不得借 trace probe 执行 cleanup、不得把单 seed trace 输出升级为 formal benchmark evidence。
- `T38` 已完成；其 run root 只能作为 T38-scoped trace evidence 引用，不能重标为 formal benchmark、clean-environment proof、runtime validation 或 real-board validation。
- `T31` 不允许创建或修改 `runs/` / `artifacts/`，不允许执行 cleanup，不允许安装/升级/删除依赖；如需读取本机 package evidence，只能作为 read-only inventory 写入文档并明确其非跨机器保证。
- `T31` 已完成并通过 Captain `PASS` 收口；它不是 clean-environment execution，也不改变 `requirements-recovery.txt` 的 recovery-smoke scope。
- `T39` 允许创建 ignored local environment `.venvs/t39_train_cpu_py312/` 和一个 CPU-only training dependency spec，但不得创建或修改 `runs/` / `artifacts/`，不得执行 cleanup，不得运行真实训练/benchmark/`.tflite`/hardware。若 dry-run flag 不存在，必须记录 blocker 或使用不会落盘的 import/help-level check，不得静默替换成真实数据生成。
- `T39` 已完成并通过 Captain `PASS` 收口；其 clean environment、draft lock 与 bootstrap docs 可提交，但 `.venvs/t39_train_cpu_py312/` 仍是 ignored local state，不得作为仓库产物提交。
- `T40` 允许复用 `.venvs/t39_train_cpu_py312/` 并创建 task-scoped derived config 与 task-scoped output directories，但只允许把真实训练输出写到 T40-isolated paths；不得改写 canonical historical `artifacts/models/*`、`artifacts/reports/*`、`runs/`，不得借机执行 cleanup、benchmark、`.tflite` 或 hardware 操作。
- `T40` 已完成并通过 Captain `PASS` 收口；其 T40-isolated outputs 只能作为 smoke evidence 引用，不得重标为 canonical historical model/report facts。
- `T33` 已完成并通过 Captain `PASS` 收口；其结论仅限“manifest-listed tracked cache 已从 Git index 中移除”，不得外推为 `runs/`、`artifacts`、`.pytest_cache/`、`.mypy_cache/` 或其他 repo-noise 已被清理。
- `T34` 只允许做 docs-only 的 claim/evidence ledger 与 figure-table outline；不得借论文收口任务顺手触碰任何 cleanup、运行结果或 evidence-level 语义。
- `T34` 已完成并通过 Captain `PASS` 收口；其输出不改变任何 repo-noise 事实。
- `T35` 已完成并通过 Captain `PASS` 收口；其输出只是 docs-only 的 paper draft skeleton 与 reviewer-risk audit，不改变任何 repo-noise 事实。
- `T41` 已完成并通过 Captain `PASS` 收口；其输出只是 Milestone 2K paper-assembly gate review，不改变任何 repo-noise 事实。
- `T42` 若推进，只允许做 docs-only 的 Background / Related Work scaffold 与 method-positioning calibration；不得借写作结构任务顺手触碰任何 cleanup、运行结果或 evidence-level 语义。
- `T42` 已由 Captain 以 `PASS` 收口；其结论只代表论文结构与 framing 校准完成，不代表任何代码、benchmark、`.tflite`、硬件、`runs/`、`artifacts` 或 cleanup 事实发生升级。
- `T43` 仍是 docs-only 的 Background / Related Work prose drafting 任务；其输出只是草稿文本，不是新的仓库事实源，也不得把 framing 语言静默升级成更强验证结论。
- `T43` 已由 Captain 以 `PASS` 收口；即使 prose draft 存在，也不得把它作为新的事实源覆盖 `runs/`、`artifacts`、review 文档或风险文档中的既有边界。
- `T44` 进入 `Research Reality Recovery Mode` 后，只允许新增 recovery baseline 文档；不得借 recovery 文档任务顺手清理、重命名、重标记或重解释 `runs/`、`artifacts`、`.pyc`、`.tflite`、benchmark outputs 或历史报告。
- `T45` 已由 Captain 以 `PASS` 收口；其输出只是 benchmark-expansion protocol lock，不是 benchmark execution，也不改变任何 repo-noise 事实。
- `T46` 已由 Captain 以 `PASS` 收口；其输出只是 docs-only 的 multi-seed mechanism/intervention plan 与 trace pack，不改变任何 repo-noise 事实。
- `T54` 已完成并由 Captain 按 `PASS` 收口；其 `runs/T54_multi_seed_trace_phase_a_20260522/` run root 与新建的 `runs/teachrepr/p4_benchmark/trp60425_resume`、`trp60430_resume`、`trp60510_resume` 都只可视为新输出目录，不得回写为历史事实重标或 cleanup 依据。
- `T55` 允许创建一个 T55-scoped run root，并且任务内生成的 benchmark config、helper script 与 benchmark output 都必须收敛在这个单一 run root 内；不得再向 run root 外分散写出新 benchmark 目录、不得改写历史 `runs/` / `artifacts/`、不得借机执行 cleanup，也不得把 intervention outputs 重标为 formal benchmark、`.tflite` runtime 或 real-board validation 事实。
- `T55` 已完成并由 Captain 按 `PASS` 收口；其 `runs/T55_multi_seed_i1_probe_20260523/` 只可视为 T55-scoped intervention evidence 目录。run root 内的 `benchmark_test/` 残留属于 accepted debris，不得被误写成正式 deliverable、也不得在无新任务包前提下顺手 cleanup。
- `T56` 已由 Captain 以 `PASS` 收口；其 docs-only 边界已完成，不允许被回写成 run root、analysis artifact 或 benchmark output 目录操作。
- `T47` 若推进，也只能是 hedge-conditioned docs-only paper-material lane，默认不允许创建、修改或清理任何 run root、analysis artifact 或 benchmark output 目录。
## 2026-05-24 Captain Update (T47/T57 repo-noise supersession)

- `T47` 已由 Captain 以 `PASS` 收口；其输出只限 docs-only paper-material ledger，不得被回写成新的 run-root、analysis artifact 或 benchmark output 目录操作。
- 当前唯一任务 `T57` 只允许创建一个 `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_*` 作用域内的 run root。
- 除该单一 T57-scoped run root 外，不得新建、改写或清理任何其他 `runs/`、`artifacts/`、analysis artifact 或 benchmark output 目录，也不得借 `T57` 触碰 source-tree code/config。

## 2026-05-26 Captain Update (T58/T59 repo-noise supersession)

- `T58` 已由 Captain 接受为 `PASS_WITH_WARNINGS`；其新增 figure assets 现在只应被视为 `T58` 的 task-scoped paper-material output，不得被重写成新的 benchmark/run 事实。
- 当前唯一任务切换为 `T59: Statcalib separate comparator lane integration and bounded smoke`。
- `T59` 允许的新增落盘范围只包括：
  - task-scoped docs
  - focused tests
  - one statcalib-specific config file
  - one task-scoped run root: `runs/p4_benchmark/T59_statcalib_lane_smoke_*`
- 除上述单一 T59-scoped run root 外，不得新建、改写或 cleanup 任何其他 `runs/`、`artifacts/`、analysis-output 或 benchmark-output 目录。
- `T59` 不得借机执行 repo cleanup，也不得把 mainline experiment evidence 与 theory-only branch materials 混写。

## 2026-05-26 Captain Update (T57/T58 repo-noise supersession)

- `T57` 已由 Captain 接受为 `PASS`；其 task-scoped run root 现在只应被视为历史证据，不得被重写成可变当前事实。
- 当前唯一任务 `T58` 为 docs-only，禁止创建或修改任何 `runs/`、`artifacts/`、benchmark output 或 analysis-output 目录。
- 如果 `T58` 需要生成 figure assets，只允许写入 `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/`。
- `T58` 不得触碰 source-tree code/config，不得触发 cleanup，也不得把 mainline experiment evidence 与 theory-only branch materials 混写。

## 2026-05-26 Captain Update (T59/T60 repo-noise supersession)

- `T59` has been accepted as `PASS_WITH_WARNINGS`; its run root `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740` must now be treated as frozen historical evidence and must not be regenerated or rewritten.
- The current unique task is now `T60: Statcalib lane isolation and regression hardening`.
- `T60` is not allowed to create any new run root, benchmark output, analysis-output, or artifact directory.
- `T60` may modify only the allowed source/test/docs files and may add only task-scoped test modules.
- `T60` must not use repo cleanup as a side goal, and it must keep mainline experiment evidence separate from theory-only branch materials.

## 2026-05-27 Captain Update (T60/T61 repo-noise supersession)

- `T60` has been accepted as `PASS`; it created no new run root and did not modify any historical benchmark artifact.
- The current unique task is now `T61: Statcalib clean-provenance fairness sanity rerun`.
- `T61` may create exactly one new run root under `runs/p4_benchmark/T61_statcalib_fairness_sanity_*`.
- `T61` must not regenerate, resume into, or rewrite `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740` or any other historical run root.
- `T61` must not modify source code, source-tree config, tracked governance docs, theory-only branch materials, or use repo cleanup as a side goal.
- Because `T61` exists partly to repair provenance weakness, the rerun must start from a clean committed worktree before any task-local docs are written.
