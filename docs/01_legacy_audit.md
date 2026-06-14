# Feasibility And Legacy Audit

**维护状态**：legacy 真实性审计底稿
**最后维护说明更新**：2026-06-15
**当前事实入口**：`docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md`

## 1. 审计目的

本文件对应 `docs/reference/AI_coding_workflow.md` 中 `01_feasibility_report.md` 的角色，并保留 legacy audit 结果。它回答两个问题：

1. 项目是否值得继续
2. 继续时哪些能力是真实代码、哪些只是 `mock` / `stub` / `placeholder`

本轮审计默认只读核查，目标是回答：

1. 这个项目到底已经有什么
2. 哪些是代码已实现
3. 哪些只是未来扩展或 placeholder
4. 恢复期当时如何判断下一步接力任务

## 1A. 维护边界

本文件解释的是 legacy 仓库“原本有哪些真实代码、哪些只是 mock / stub / placeholder”。它不是当前任务流水账，也不是后续计划入口。

因此，`docs/01_legacy_audit.md` 不应随每个 T 系列任务滚动追加。只有当新证据会改变 legacy reality matrix 时才更新，例如：

- 发现某条早期代码路径此前被误判为 placeholder 或真实实现；
- `.tflite`、HIL、real-board、benchmark 的证据等级发生会反向影响 legacy 审计判断的变化；
- 需要修正本文件中仍指向旧路径或旧事实入口的内容。

普通 Captain closeout、当前唯一任务切换、后续计划更新，应优先同步 `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 和必要时的 `docs/02_experiment_plan.md`，不应把本文件变成第二份任务日志。

### 1A-补充. 2026-06-15 Captain 维护说明

- `T88` 已按 `PASS` 收口，但它是 docs-only 的主线 paper-material closeout，不改变 legacy truth matrix。
- `T88` 没有把任何 `mock` / `stub` / `placeholder` 提升成真实实现，也没有改变 `.tflite`、real-board、benchmark 或 training reproducibility 的 legacy 证据等级判断。
- 当前唯一任务已切到 `T89`，但 `T89` 仍只是 frozen-mainline handoff / change-control 任务；当前任务状态仍应以 `docs/04_task_board.md` 与 `docs/07_handoff.md` 为准，而不是由本文件维护。

## 2. 总体结论

### 2.1 结论摘要

- 项目主体代码真实存在，且明显不是最小脚手架
- P0/P2/P3/P4 的 benchmark 入口与对应配置存在
- 软件 HIL 主线存在，但真板 backend 仍未到“已完成”状态
- 默认环境不可直接复现实验，当前最明显阻塞是缺依赖与入口说明
- `.tflite` 部署链路同时存在真实导出/runtime 与 stub 回退路径
- 仓库此前缺少治理文件，导致“代码真实状态”和“开发流程状态”没有被固定
- 截至 `2026-05-08`，第一轮恢复期收尾已完成，仓库可退出 `Phase 1: Recovery`，进入受控继续开发

### 2.2 恢复推进状态

- 初始恢复任务建议是 `T1: 确认依赖矩阵与最小入口`
- 截至 `2026-05-08` 已完成：
  - `T1`
  - `T2`
  - `T3`
  - `T4`
  - `T5`
  - `T6`
  - `T7`
  - `T8`
  - `T9`
  - `T10`
  - `T11`
  - `T12`
  - `T13`
- 历史恢复期下一任务建议曾多次被后续 Captain closeout supersede；当前唯一任务不由本节维护，必须以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为准。

说明：

- `T3` 已把 `mock` / `stub` / `placeholder` 边界固定到 `docs/03_hil_p4_boundary_audit.md`
- `T4` 已把恢复期最小 software HIL 路径固定到 `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `T5` 已把缓存/生成物噪声治理口径固定到 `docs/06_repo_noise_governance.md`
- `T6` 已对最小 software HIL 路径完成新的复验，并再次确认 `mock + model_artifact + artifact_npz + inproc`
- `T7` 已对最小 P4 benchmark 路径完成新的复验，并把 recovery 级 P4 配置固定到 `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
- `T8` 已完成 gate review，并明确结论为 `Continue Repair`
- `T9` 已把 `P4 frozen baseline` recovery 证据从两种 mode 扩到四种正式 baseline，但仍限制在 `single-scenario + repeats=1`
- `T10` 已完成二次 gate review，并继续给出 `Continue Repair`
- `T11` 已在根目录补入 `requirements-recovery.txt`，并把它接入 `P0/P3/P4 recovery smoke` 的 bootstrap 文档
- `T12` 已把最小 software HIL recovery smoke 的随机源链路收口到逐字一致复验
- `T13` 已通过 recovery exit review，项目进入 `Phase 2: Controlled Development`
- `T24` 已完成 frozen-set formal software revalidation，并由 Captain 接受为 `PASS_WITH_WARNINGS`
- `T25` 已完成 result-boundary gate review，并确认 T24 只能作为 `mock-backed` software HIL formal software revalidation
- 后续 P3/P4 文档与复验结果都应沿用同一套 backend / artifact type 表述口径

### 2026-06-14 Captain Update (T86 closeout)

- `T86` 已被 Captain 接受为 `PASS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T86` 只是 main 分支 docs-only 的 submission-facing assembly / exclusion 收口，不是新的实验、不是新的板级证据，也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- `T86` review 中的 4 条 non-blocking notes 全部按 accepted operational reminder 处理；它们只约束后续精确暂存、retelling 与 current-host compile 口径，不改变 truth matrix。
- 当前唯一任务已切换为 `T87: 主线作者终检与 pre-submission QA 收口包`。`T87` 仍是 main 分支 docs-only 的 QA/gate 任务，不触碰 theory 分支的大范围改写，也不改变 legacy truth matrix。

### 2026-06-14 Captain Update (T87 closeout)

- `T87` 已被 Captain 接受为 `PASS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T87` 只是 main 分支 docs-only 的作者终检、manual-finish queue 固化与 pre-submission regression gate 收口，不是新的实验、不是新的板级证据，也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- `T87` review 中的 3 条 non-blocking notes 全部按 accepted operational reminder 处理；它们只约束主机噪声隔离与 gate retelling 口径，不改变 truth matrix。
- 当前唯一任务已切换为 `T88: 主线 bounded manual finish 执行与 surface freeze 收口包`。`T88` 仍是 main 分支 docs-only 的 manual-finish execution / freeze 任务，不触碰 theory 分支的大范围改写，也不改变 legacy truth matrix。

### 2026-06-14 Captain Update (T85 closeout)

- `T85` 已被 Captain 接受为 `PASS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T85` 只是 main 分支 docs-only 的 residual wording-lag 清扫、submission-readiness preflight 与 blocker matrix 收口，不是新的实验、不是新的板级证据，也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- `R36` 已由 `T85` 收口关闭；这只是 paper-note 文字状态问题的解决，不改变 truth matrix。
- 当前唯一任务已切换为 `T86: 主线 bounded submission-pack assembly 与显式 exclusion route 收口`。`T86` 仍是 main 分支 docs-only 的装配/排除项收口任务，不触碰 theory 分支的大范围改写，也不改变 legacy truth matrix。

### 2026-06-14 Captain Update (T84 closeout)

- `T84` 已被 Captain 接受为 `PASS_WITH_WARNINGS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T84` 只是 main 分支 docs-only 的有界 final polish / reader-facing assembly，不是新的实验、不是新的板级证据，也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- 新的 carry-forward 风险变为 `R36`，它只约束主线 note `Conclusion` 中残留的一处状态滞后句，不改变 truth matrix。
- 当前唯一任务已切换为 `T85: 主线 submission-readiness preflight gate 与残余状态滞后清扫`。`T85` 仍是 main 分支 docs-only 的预检与残余措辞清扫任务，不触碰 theory 分支的大范围改写，也不改变 legacy truth matrix。

### 2026-06-14 Captain Update (T83 closeout)

- `T83` 已被 Captain 接受为 `PASS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T83` 只是 docs-only 的全文一致性 sweep / closeout gate，不是新的实验、不是新的板级证据、也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- 当前唯一任务已切换为 `T84: 主线 note 有界 final polish 与读者化装配包`。`T84` 仍是 main 分支 docs-only 的读者化润色与装配任务，不触碰 theory 分支的大范围改写，也不改变 legacy truth matrix。

### 2026-06-13 Captain Update (T82 closeout)

- `T82` 已被 Captain 接受为 `PASS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T82` 只是 docs-only 的 supporting-boundary closeout，不是新的实验、不是新的板级证据、也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- 当前唯一任务已切换为 `T83: 主线 note 全文一致性收口与 manuscript closeout gate`。`T83` 仍是 main 分支 docs-only 的全文一致性与 closeout gate 任务，不触碰 theory 分支的大范围改写，也不改变 legacy truth matrix。

### 2026-06-12 Captain Update (T80 closeout)

- `T80` 已被 Captain 接受为 `PASS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T80` 只是 docs-only 的 section-bounded prose reopen，不是新的实验、不是新的板级证据、也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- 当前唯一任务已切换为 `T81: Summary of Contributions 与 methods-only calibration pack`。`T81` 仍是 main 分支 docs-only 的受控方法章/贡献段校准任务，不触碰 theory 分支的大范围改写，也不改变 legacy truth matrix。

### 2026-06-12 Captain Update (T79 closeout)

- `T79` 已被 Captain 接受为 `PASS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T79` 只是 docs-only 的 reopen gate / readiness 评审，不是新的实验、不是新的板级证据、也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- 当前唯一任务已切换为 `T80: 主线校准段落的 bounded prose reopen`。`T80` 仍是 main 分支 docs-only 的受控 prose 任务，不触碰 theory 分支的大范围改写，也不改变 legacy truth matrix。

### 2026-06-12 Captain Update (T78 closeout)

- `T78` 已被 Captain 接受为 `PASS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T78` 只是 docs-only 的 note 非结果层校准、`statcalib` 层级降权、排版 warning 收口与 section-scope 审计，不是新的实验、不是新的板级证据、也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- `R35` 已由 `T78` 收口；当前唯一任务已切换为 `T79: 论文材料 reopen gate 与 bounded prose 扩写就绪性评审`。`T79` 仍是 main 分支 docs-only 的 gate 评审任务，不触碰 theory 分支的大范围改写，也不改变 legacy truth matrix。

### 2026-06-12 Captain Update (T77 closeout)

- `T77` 已被 Captain 接受为 `PASS_WITH_WARNINGS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T77` 只是 docs-only 的 note 结果层同步、traceability hardening 与本地编译刷新，不是新的实验、不是新的板级证据、也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- `R34` 已由 `T77` 收口；新的 carry-forward 风险变为 `R35`，它只约束 note 的非结果层校准、`statcalib` 视觉层级与排版 warning，不改变 truth matrix。
- 当前唯一任务已切换为 `T78: 论文 note-draft 非结果层校准、statcalib 层级降权与排版 warning 收口`；它仍是 main 分支 docs-only 的论文 note 质量收口任务，不触碰 theory 分支的大范围改写，也不改变 legacy truth matrix。

### 2026-06-12 Captain Update (T76 closeout)

- `T76` 已被 Captain 接受为 `PASS_WITH_WARNINGS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T76` 只是 docs-only 的 rendered QA / Results assembly 收口，不是新的实验、不是新的板级证据、也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- `T76` 的 `N1/N3` 被收敛为新的 paper-facing traceability 风险 `R34`；`N2` 只是提交前的过程性噪声提醒，不改变 truth matrix。
- 当前唯一任务已切换为 `T77: 论文 note-draft 结果层同步与 T76 traceability hardening`；它仍是 main 分支 docs-only 的论文结果层同步任务，不触碰 theory 分支的大范围内容，也不改变 legacy truth matrix。

### 2026-06-12 Captain Update (T75 closeout)

- `T75` 已被 Captain 接受为 `PASS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`T75` 只是 docs-only 的主线 authoring 资产与写作边界收口，不是新的实验、不是新的板级证据、也不是任何 `.tflite` / real-board / mature comparator 事实升级。
- 当前唯一任务已切换为 `T76: Rendered figure QA and results-section assembly pack`；它仍是 main 分支 docs-only 的论文材料质量控制与装配任务，不触碰 theory 分支，也不改变 legacy truth matrix。

### 2026-06-12 Captain Update (T74 closeout)

- `T74` 已被 Captain 接受为 `PASS`。
- 这次 closeout 仍然没有改变 legacy audit 的真实性判断：`board_backend.py` 依旧是 placeholder-only，真实板级执行证据仍然缺失，mainline 仍不得把 paper-ready 结果表/图表打包改写成 `real-board validated`、`deployment closed` 或 `mature comparator promoted`。
- `T74` 只是在 docs-only 范围内把 post-`T73` 的 paper-facing simulation/material 包整理成作者可直接复用的入口，不改变 legacy truth matrix。
- 当前唯一任务已切换为 `T75: Main-text results prose and final figure authoring pack`；它仍是 main 分支 docs-only 实验结果 authoring 任务，不触碰 theory 分支。

### 2026-06-12 Captain Update

- `T73` 已被 Captain 接受为 `PASS`。
- 这次 closeout 没有改变 legacy audit 的真实性判断：`board_backend.py` 仍是 placeholder-only，真实板级执行证据仍然缺失，mainline 仍不得把 read-only gate / transfer-pack / paper-facing 台账刷新改写成 `real-board validated`。
- `T73` 只是在 docs-only 范围内把主线 paper-facing claim/result/risk 入口刷新到 post-`T72` 状态，不改变 legacy truth matrix。
- 当前唯一任务已切换为 `T74: Paper-ready simulation result and figure pack`；它仍是 main 分支 docs-only paper-material 任务，不触碰 theory 分支。

### 2026-06-11 Captain Update

- `T72` 已被接受为 `PASS_WITH_WARNINGS`。
- 这次 closeout 没有改变 legacy audit 的底线判断：`board_backend.py` 仍是 placeholder-only，真实板级执行证据仍然缺失，mainline 仍不得把 read-only gate / transfer-pack 改写成 `real-board validated`。
- `T72` 收紧的是 transfer-pack provenance 与 future-host 入口严谨度，不是 `T37` 真板执行前提本身。
- `R31` 已关闭，因为默认-config / override / probe-limitations 这组三个主 provenance 问题都已补上。
- 新的后续风险为 `R32`：最小 config 场景下 path provenance 仍不能区分“YAML 明确字段”与“代码默认值回退”，所以 future-host 入口还不能写成 fully provenance-clean。
- 当前唯一任务已切换为 `T73: Mainline claim/evidence and result/figure/risk ledger refresh`；它是 main 分支 docs-only 主线台账刷新任务，不触碰 theory 分支。

### 2026-05-16 Captain Update

- `T38` has been accepted as `PASS`; it produced bounded seed=20260429 trace-export evidence without changing benchmark semantics.
- Milestone 2I is closed as `Conditional Allow`, not as full causal proof, true `.tflite` runtime validation, real-board validation, or paper-grade benchmark completion.
- `R10` is narrowed by T38 trace evidence but remains open because mitigation, multi-seed confirmation, and upstream root-cause isolation are still absent.
- Current unique task is `T31`, focused on training-chain portable dependency-lock planning and clean-environment reproducibility boundaries.

### 2026-05-17 Captain Update

- `T42` has been accepted as `PASS`; it produced a bounded Background / Related Work scaffold and method-positioning calibration without touching code, configs, `runs/`, `artifacts`, `.tflite`, or hardware scope.
- T42 review comments are non-blocking and remain framing-guidance notes only; they do not introduce new repository or evidence risks.
- The current paper framing is now calibrated to a working method-forward title with evidence-bounded body text; this is a drafting choice, not an evidence upgrade.
- At that time, the current unique task was `T43`, focused on a docs-only bounded prose draft for Background / Related Work, not on new experiments or evidence upgrades.

### 2026-05-18 Captain Update

- `T43` has been accepted as `PASS`.
- T43 warnings are all classified as `accepted`; no `deferred` warning remains from this review.
- The project now enters `Research Reality Recovery Mode` because the user explicitly requested that paper expansion pause until evidence/material truth is re-frozen and audited.
- Current unique task is `T44`, focused on a docs-only recovery baseline: claim/evidence truth, reproducibility gaps, figure/result readiness, and paper-claim risk.

### 2026-05-19 Captain Update

- `T45` has been accepted as `PASS`.
- T45 warnings are all classified as `accepted`; there are no `deferred` warnings from this review.
- `T45` locks the benchmark-expansion protocol only; it does not widen the executed benchmark set or upgrade evidence.
- Current unique task is now `T46: Multi-seed mechanism/intervention plan and trace pack`.
- `T46` is docs-only and remains a mechanism-evidence planning gate, not an execution or benchmark task.

### 2026-05-22 Captain Update

- `T46` has been accepted as `PASS`.
- T46 non-blocking comments are all treated as `accepted`; no `deferred` warning remains from this review.
- T46 successfully freezes the mechanism-evidence execution boundary: current evidence is still single-seed diagnostic, and any stronger claim needs a bounded multi-seed trace lane before any intervention or paper-material freeze.
- Current unique task is now `T54: Phase A multi-seed trace-only generalization probe`.
- `T54` is not benchmark expansion and not paper assembly. It is a bounded trace-only execution task that reuses the existing T38 path, stays inside the frozen four scenarios and Full vs Gated v5 comparison, and does not yet run any intervention.

### 2026-05-23 Captain Update

- `T54` has been accepted as `PASS`.
- T54 non-blocking comments are all treated as `accepted`; no `deferred` warning remains from this review.
- T54 confirms that the committed-`b` instability pattern generalizes beyond `seed=20260429`, but only as bounded diagnostic evidence with important quiet/classic/universal qualifications; it does not close `C4`.
- The next active task is not `T47`. The current unique task is now `T55: Phase B multi-seed I1 residual-clip intervention probe`.
- `T55` must stay on the same mock-backed P4 wrapper over software HIL path, reuse existing model assets, and test only one config-only intervention variant before any paper-material freeze is revisited.

### 2026-05-24 Captain Update

- `T55` has been accepted as `PASS`.
- T55 non-blocking comments are all treated as `accepted`; no `deferred` warning remains from this review.
- T55 provides the first bounded targeted intervention evidence, and that evidence is mixed and mostly harmful rather than a clean mitigation success.
- The simple harmful-instability framing is no longer a defensible general mechanism claim. `C4` remains `partial`.
- `T56` is complete and accepted as `PASS`.
- T56 warnings are all classified as `accepted`; no `deferred` warning remains from this review.
- The current unique task is now `T47: Paper ablation result-pack and material ledger`.
- `T47` is docs-only and may proceed only under the hedge boundary defined by `T56`; it is not unconditional paper expansion or mechanism closure.

### 2026-06-10 Captain Update

- `T49` has been accepted as `PASS_WITH_WARNINGS`.
- `T49` closes one honest current-host real-board gate pack with verdict `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`; no real-board smoke was executed.
- `T49` warning handling is `W1/W2/W3 = deferred -> R30`, so the current-host `NO_GO` stands, but the gate path is not yet future-host hard enough.
- This does not change the legacy audit bottom line: `board_backend.py` is still placeholder-backed, real-board validation is still absent, and mainline may not retell readiness scaffolding as completed hardware execution.
- `R13/R14` remain open but narrower because the current-host truth is now explicit rather than unknown.
- The current unique task is now `T71: Real-board gate regeneration and host-transfer pack`.
- `T71` stays on the same deployment-boundary lane and exists to harden reproducibility/portability of the gate itself, not to open `T37` or claim board execution readiness.
- Any older T55-next wording later in this file is superseded by this status block and `docs/04_task_board.md`.

## 3. 可行性判断

### 3.1 问题定义

项目目标是在 GKP 纠错中验证一种工程可落地的快慢回路架构：FPGA 侧 fast loop 执行低延迟线性控制，CPU/CNN 侧 slow loop 根据窗口统计更新控制参数。

### 3.2 可差异化点

1. 不是纯离线 decoder accuracy 项目，而是带 HIL / latency / commit / overflow 指标的闭环工程实验。
2. learned module 的角色被限制为 residual / calibration，而不是完全替代解码器。
3. 当前文档已经明确区分 mock-backed HIL、真实 `.tflite`、stub manifest 与真板 placeholder。

### 3.3 MVP 实验

当前 Phase 2 MVP 不是重新训练模型，而是增强 P4 evidence：

1. 审计 P4 frozen benchmark protocol。
2. 在 bounded matrix 下扩展 P4 多场景 smoke。
3. 经 gate review 决定是否进入更正式 benchmark。

### 3.4 主要风险

1. 把 recovery smoke 误写成正式 benchmark。
2. 把 `board_backend.py` placeholder 误写成真板完成。
3. 把 `.tflite.json` stub manifest 误写成真实 `.tflite` runtime。
4. 在未固定环境和 run matrix 前启动长跑。

### 3.5 Go / No-Go 判断

当前判断：`Go`，但只允许 bounded development。

理由：

1. Recovery exit review 已给出 `Allow`。
2. 最小 P3 software HIL path 已逐字一致复验。
3. P4 recovery smoke 已覆盖单场景四模式。
4. 仍有明确边界和风险文档，适合继续受控推进。

## 4. Feature Reality Matrix

| Feature | Claimed status | Evidence path | Verified? | Risk |
| --- | --- | --- | --- | --- |
| P0 full vs simplified 基线脚本 | 已存在最小对比脚本 | `benchmark/compare_full_vs_simplified_ler.py` | 部分验证 | 默认环境缺 `numpy`，当前无法在系统 Python 下直接运行 |
| P1 数据与训练链 | 已有数据集构建、训练、评估、量化入口 | `cnn_fpga/data/dataset_builder.py`, `cnn_fpga/model/train.py`, `cnn_fpga/model/evaluate.py`, `cnn_fpga/model/quantize.py` | 代码存在 | 依赖矩阵未确认，当前未复跑 |
| P2 行为级硬件仿真 | 已有硬件行为仿真与模式 benchmark | `cnn_fpga/benchmark/run_hardware_emulation.py`, `cnn_fpga/benchmark/run_p2_mode_benchmark.py` | 代码存在 | 当前环境未复验 |
| P3 软件 HIL | 已有 HIL 主流程、mock backend、驱动抽象、推理服务 | `cnn_fpga/benchmark/run_hil_suite.py`, `cnn_fpga/hwio/mock_fpga.py`, `cnn_fpga/runtime/inference_service.py`, `docs/03_hil_p4_boundary_audit.md`, `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`, `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104/hil_summary.json`, `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104/hil_summary.json` | bounded 最小路径已逐字一致复验 | 结论仅限 `mock + model_artifact + artifact_npz + inproc`，不等于真板或 `.tflite` 路径已恢复 |
| P3 真板 HIL | 当前是 placeholder / gate / provenance 层证据，尚非真实板级完成 | `cnn_fpga/hwio/board_backend.py`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/evidence_packs/deployment_boundary/` | 是 | 真实设备节点、bitstream / RTL / DMA contract 和地址表证据仍缺失，不能写成已完成 |
| P4 多场景 benchmark | 已有统一 benchmark 汇总脚本；最小 recovery path、frozen baseline 单场景全模式 smoke、以及 T24 frozen-set formal software revalidation 都已复验 | `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`, `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`, `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`, `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json` | `mock-backed` software HIL formal revalidation 已完成 | 当前证据仍不是 `.tflite` runtime、真板验证或 paper-grade expanded benchmark；T28 已修复 teacher diagnostics missing-vs-zero writer 语义，但 R10 机制证据仍未完全修复 |
| teacher-representation 多版本分支 | 已有 v2-v9 配置与配对 benchmark 入口 | `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v*.yaml`, `cnn_fpga/benchmark/run_p4_teacher_representation_paired.py` | 代码存在 | 当前不应继续扩分支，先恢复可信度 |
| 真板 backend 语义 | placeholder/骨架状态 | `cnn_fpga/hwio/board_backend.py` | 是 | 若表述不严谨，极易误导项目完成度判断 |
| `.tflite` 真导出路径 | 代码支持真导出与 stub 回退双路径 | `cnn_fpga/model/export.py`, `cnn_fpga/runtime/inference_service.py`, `docs/03_hil_p4_boundary_audit.md` | 边界已审计 | 必须明确区分真实 `.tflite`、artifact 与 stub manifest |
| recovery 期根级依赖 manifest | 已新增 recovery-scoped 最小 manifest | `requirements-recovery.txt`, `docs/recovery_bootstrap/P0_smoke_bootstrap.md`, `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`, `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md` | 是 | 只覆盖 `P0/P3/P4 recovery smoke`，不等于完整训练链、`.tflite` 或真板环境 |
| 根级治理文件 | 恢复前缺失 | 根目录与 `docs/` | 是 | 高，直接影响后续接力与审查 |

## 5. 关键证据

### 4.1 代码主干不是空壳

关键证据：

- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/model/train.py`
- `physics/logical_tracking.py`

### 4.2 真板路径尚未完成

关键证据：

- `cnn_fpga/hwio/board_backend.py`
  - 文件顶层注释直接标注为 placeholder real-board backend
- `docs/02_experiment_plan.md`
  - 明确写到 board backend 仍是 placeholder / gate / readiness / provenance 级，不能写成真实板级完成

### 4.3 `.tflite` 路径不能默认视为真实部署

关键证据：

- `cnn_fpga/model/export.py`
  - 优先导出真实 `.tflite`
  - 失败时回退为 `tflite_stub_v1`
- `cnn_fpga/runtime/inference_service.py`
  - 真实路径使用 `tflite_service`
  - stub manifest 路径使用 `tflite_stub_service`

### 4.4 默认环境不可信

关键证据：

- 根目录现已新增：
  - `requirements-recovery.txt`
- 但它只覆盖：
  - `P0/P3/P4 recovery smoke`
- 根目录仍没有完整仓库环境文件：
  - `requirements.txt`
  - `pyproject.toml`
  - `environment.yml`
- 最小 benchmark 在默认 `python 3.13.7` 下因缺少 `numpy` 失败

### 4.5 T3 边界澄清已完成

关键证据：

- `docs/03_hil_p4_boundary_audit.md`
  - 已把以下边界固定写清：
    - `software_hil_orchestrator`
    - `mock_backend`
    - `placeholder_real_board_backend`
    - `true_tflite_or_stub_export`
    - `true_tflite_or_stub_runtime`

### 4.6 T4 / T6 最小 software HIL 路径已恢复并二次复验

关键证据：

- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
  - 显式固定 `hil.backend=mock`
  - 显式固定 `inference_service.mode=inproc`
  - 显式固定 `inference_service.backend=artifact_npz`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
  - 固定恢复期最小 software HIL 复用命令
- `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`
  - `backend = mock`
  - `n_windows_ready = 2`
  - `n_slow_updates_finished = 2`
  - `n_commits_applied = 2`
  - `artifact_path` 指向 `static_theta_v2` 下的 `.npz`
  - `inference_service_mode = inproc`

### 4.7 T7 最小 P4 benchmark 路径已复验

关键证据：

- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
  - 显式固定 `hil.backend=mock`
  - 显式固定 `slow_loop.inference_service.mode=inproc`
  - 显式固定 `slow_loop.inference_service.backend=artifact_npz`
  - 显式固定 `slow_loop.model_artifact.path`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
  - 固定恢复期最小 P4 benchmark 复用命令与过滤条件
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/summary.json`
  - `protocol_id = p4_hil_recovery_smoke_v1`
  - `scenario = static_bias_theta`
  - `modes = static_linear, cnn_fpga`
  - `seed_pairing = paired`
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/comparison.csv`
  - `static_linear final_ler = 1.00890625`
  - `cnn_fpga final_ler = 0.72109375`
  - `cnn_fpga artifact_path = ...static_theta_v2...npz`
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/static/cnnfpg/repeat_00/hil_summary.json`
  - `backend = mock`
  - `n_slow_updates_finished = 8`
  - `n_commits_applied = 8`
  - `inference_service_mode = inproc`
  - `artifact_path = ...static_theta_v2...npz`

### 4.8 T9 单场景全模式 frozen baseline smoke 已复验

关键证据：

- `docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`
  - 已把 `T9` 的目标、边界、验证命令与文档更新范围固定
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/summary.json`
  - `protocol_id = p4_hil_recovery_smoke_v1`
  - `scenario = static_bias_theta`
  - `modes = static_linear, window_variance, ekf, cnn_fpga`
  - `seed_pairing = paired`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/comparison.csv`
  - `Static Linear final_ler = 0.99575`
  - `Window Variance final_ler = 0.57440625`
  - `EKF final_ler = 0.6795`
  - `CNN-FPGA final_ler = 0.7248125`
  - scenario winner: `window_variance`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/static/static/repeat_00/hil_summary.json`
  - `backend = mock`
  - `artifact_path = null`
  - `inference_service_mode = inproc`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/static/window/repeat_00/hil_summary.json`
  - `backend = mock`
  - `artifact_path = null`
  - `inference_service_mode = inproc`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/static/ekf/repeat_00/hil_summary.json`
  - `backend = mock`
  - `artifact_path = null`
  - `inference_service_mode = inproc`
- `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/static/cnnfpg/repeat_00/hil_summary.json`
  - `backend = mock`
  - `artifact_path = ...static_theta_v2...npz`
  - `inference_service_mode = inproc`

### 4.9 T10 gate review 已完成

关键证据：

- `docs/review/T10_gate_review.md`
  - 已明确给出 verdict：
    - `Continue Repair`
  - 已明确下一唯一任务：
    - `T11: 补一份恢复期最小依赖 manifest（优先覆盖 P0/P3/P4 recovery smoke）`
- 根目录当前仍无：
  - `requirements.txt`
  - `pyproject.toml`
  - `environment.yml`
- 这是 `T10` 时点的历史背景：
  - 当时 `T6` 的“可复验而非逐字确定性复现”观察仍成立
  - 对应 run 为 `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`
- `T9` 的 P4 证据虽然增强，但仍只覆盖：
  - `single-scenario`
  - `four-mode`
  - `repeats = 1`

### 4.10 T11 recovery 期最小依赖 manifest 已完成

关键证据：

- `docs/tasks/P0/T11_recovery_dependency_manifest.md`
  - 已把 `T11` 的作用域、验证命令与文档更新范围固定
- `requirements-recovery.txt`
  - 当前 manifest 仅包含：
    - `numpy`
    - `PyYAML`
  - 已明确覆盖：
    - `benchmark/compare_full_vs_simplified_ler.py --no-plot`
    - `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
    - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml ...`
  - 已明确不覆盖：
    - `torch` 训练链
    - `tensorflow` / `tflite-runtime`
    - `.tflite` export/runtime
    - `real_board` HIL backend
- `README.md`
  - 已改为显式引用 `requirements-recovery.txt`
- `docs/recovery_bootstrap/P0_smoke_bootstrap.md`、`docs/recovery_bootstrap/P3_software_hil_bootstrap.md`、`docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
  - 已改为显式引用同一份 root manifest

### 4.11 T12 software HIL recovery smoke 确定性收口已完成

关键证据：

- `docs/tasks/P0/T12_software_hil_determinism_recovery.md`
  - 已把 `T12` 的作用域、验证命令与文档更新范围固定
- `physics/syndrome_measurement.py`
  - `RealisticSyndromeMeasurement` 已支持显式 `rng`
  - recovery 路径的测量噪声不再依赖全局 `np.random`
- `cnn_fpga/runtime/fast_loop_emulator.py`
  - 已把快回路误差 RNG 与测量噪声 RNG 分离
- `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104/hil_summary.json`
- `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104/hil_summary.json`
  - `final_ler = 0.454375`
  - `overflow_rate = 0.002`
- 两次 run 的文件级对比
  - `hil_summary.json` SHA256 一致
  - `hil_events.json` SHA256 一致

## 6. 疑似需要后续标记或治理的问题

1. `docs/06_repo_noise_governance.md` 已确认仓库中存在 `116` 个已跟踪缓存/字节码文件；物理 cleanup 仍未执行
2. `runs/` 当前有 `1841` 个已跟踪文件，只能暂作历史证据，不能自动视作当前事实来源
3. `artifacts/` 当前有 `110` 个已跟踪文件，且 `T4/T6/T7` 的最小复验路径仍依赖其中的 `static_theta_v2` `.npz`
4. `T12` 已把 bounded software HIL recovery smoke 收口到逐字一致复验，但这不外推到 `real_board` 或 `.tflite`
5. `T9` 已确认 P4 recovery smoke 可以把 frozen baseline 集扩到四种正式 baseline，但目前还不等于正式多场景 frozen benchmark 已恢复
6. `cnn_fpga/model/export.py` 同时支持真实 `.tflite` 与 stub 回退，后续文档必须持续严区分
7. `requirements-recovery.txt` 已经补齐 recovery 期最小 manifest，但完整训练链、`.tflite` 与真板路径仍没有统一根级环境文件

## 7. 审计建议

基于 `T13` recovery exit review，当前更合理的建议已从 `Repair` 更新为“受控 `Go`”：

- 原因 1：核心算法、runtime、benchmark 资产都已经存在
- 原因 2：当前最小 P3/P4 路径都已恢复，其中 software HIL bounded path 已做到逐字一致复验
- 原因 3：`T9` 已把 P4 recovery 证据扩到 `single-scenario + four-mode + repeats=1`，但仍不是正式多场景 frozen benchmark
- 原因 4：`T11/T12` 已分别收口 recovery 期 manifest 与 software HIL 确定性，剩余缺口已经从“阻止接力”降为“下一阶段的 bounded 开发任务”

后续优先级建议（历史审计口径，已由后续任务与 `docs/02_experiment_plan.md` supersede）：

1. `T14` 至 `T56`、`T53`、`T54`、`T55` 等历史任务已经被后续 Captain closeout 串联推进；当前唯一任务必须以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为准。本节只保留“不要把 docs-only paper/material lane 直接扩写成 `.tflite` runtime 或真板范围”的边界建议。
2. 继续保持 `mock` / `.tflite` / `real_board` 边界表述诚实
3. `T26` gate 结论为 `CONDITIONAL_GO`，且 `T30` 已把 statcalib 收紧为 interface-only separate comparator contract；后续仍不得把 statcalib 静默并入 T24 frozen benchmark set，不得扩展 formal benchmark、baseline/scenario、`.tflite` 或真板范围。
4. `T36/T38` 已把 `seed=20260429` 诊断推进到 single-seed trace-supported mechanism evidence，但仍不是 mitigation 或 multi-seed causal proof。
5. `T31` 已把训练链依赖边界从本机 bootstrap 推进到 portable dependency-lock plan；`T39` 已把它推进到 clean-environment draft lock + dry-run/import-level bootstrap；`T40` 已把它推进到 one real clean-environment training smoke；但仍不得把本机 `DLEnv`、dev torch、GPU/CUDA 或 smoke-scale结果写成跨机器保证或完整训练可复现结论。
### 2026-05-24 Captain Update (T47 closeout supersession)

- `T47` has now been accepted as `PASS`.
- T47 warnings `N1-N5` are all classified as `accepted`; there is no `deferred` warning from this review.
- `T47` closes the docs-only paper-material freeze honestly and keeps `FR7` explicit as the largest remaining ablation gap.
- The current unique task is now `T57: FR7 feature/teacher ablation re-execution under locked T24 protocol`.
- `T57` is a bounded execution task only. It must not retrain, modify source-tree code/config, or reopen `.tflite`, real-board, cleanup, comparator-expansion, or benchmark-expansion scope.

### 2026-05-26 Captain Update (T57 closeout supersession)

- `T57` has now been accepted as `PASS`.
- `T57` review contains no blocking issue and no new non-blocking item that needs separate `accepted / deferred / rejected` handling.
- No new risk item is opened by warning classification for `T57`.
- `T57` closes `FR7` only as a bounded frozen-set result-table gap. It does not close `R10`, and it does not justify causal or architecture-attribution upgrades.
- The most important legacy-boundary correction from `T57` is that `hybrid_no_teacher_params` becomes best in all 4 frozen scenarios, which weakens any simple teacher-parameter-necessity story.
- The current unique task is now `T58: FR6 multi-seed mechanism/intervention figure pack`.
- `T58` remains a docs-only mainline paper-material task and must not be mixed with theory-only branch materials.

### 2026-05-26 Captain Update (T58 closeout supersession)

- `T58` has been accepted as `PASS_WITH_WARNINGS`.
- T58 warnings are classified as: `N1 accepted`, `N2 accepted`, `N3 accepted`, `N4 accepted`.
- No `deferred` or `rejected` warning remains from this review, so no new risk item is opened by warning classification.
- `T58` does not change any HIL / P4 / `.tflite` / real-board truth boundary. It only closes `FR6` as a bounded descriptive figure pack built from existing `T54/T55/T56` evidence.
- The current unique task is now `T59: Statcalib separate comparator lane integration and bounded smoke`.
- `T59` is the smallest honest next step toward `FR8`: integrate a separate `statcalib` comparator lane and prove bounded end-to-end executability without mixing with theory-only branch materials or altering frozen `T24` semantics.

### 2026-05-26 Captain Update (T59 closeout supersession)

- `T59` has been accepted as `PASS_WITH_WARNINGS`.
- T59 warnings are classified as: `W1 deferred`, `W2 accepted`, `W3 deferred`.
- The deferred items are now written into risks and remain open before any `FR8` task.
- `T59` does not change any HIL / P4 / `.tflite` / real-board truth boundary. It only closes the first integrated `statcalib` lane smoke gap.
- The current unique task is now `T60: Statcalib lane isolation and regression hardening`.
- `T60` is the smallest honest next step after T59: isolate cross-mode semantics and harden regression coverage without reopening benchmark execution or mixing with theory-only branch materials.

### 2026-05-27 Captain Update (T60 closeout supersession)

- `T60` has been accepted as `PASS`.
- `T60` review adds no new warning item that needs separate `accepted / deferred / rejected` handling.
- `T60` closes the cross-mode leakage concern from `T59`; `R26` should now be treated as closed.
- `R27` remains open, but only as a provenance/fairness sanity blocker before any `FR8` task; the regression-coverage part has already been closed by `T60`.
- `T60` does not change any HIL / P4 / `.tflite` / real-board truth boundary. It only hardens semantics and tests on the existing mainline lane.
- The current unique task is now `T61: Statcalib clean-provenance fairness sanity rerun`.
- `T61` is the smallest honest next step: rerun the bounded T59 matrix from a clean committed worktree, with no source/config edits and no theory-branch mixing.

### 2026-05-27 Captain Update (T61 closeout supersession)

- `T61` has been judged `BLOCK`.
- The bounded fairness result persisted, but the task failed its own clean-provenance objective because the final artifact commit anchor drifted away from the clean launch commit during execution.
- `T61` therefore does not close `R27` and does not open `FR8`.
- No new warning-classification-driven risk item is opened from T61 because the verdict is `BLOCK`, not `PASS_WITH_WARNINGS`.
- The current unique task is now `T62: Statcalib provenance-isolated fairness rerun`.
- `T62` is the smallest honest next step: rerun the exact same bounded matrix in a provenance-isolated way, with no source/config edits, no same-run resume, and no theory-branch mixing.

### 2026-05-27 Captain Update (T62 closeout supersession)

- `T62` has now been accepted as `PASS`.
- `T62` closes the specific provenance blocker that caused `T61` to fail, so `R27` should now be treated as closed.
- `T62` does not change any HIL / P4 / `.tflite` / real-board truth boundary. It still remains bounded mock-backed software-HIL sanity evidence only.
- `T62` does not open `FR8` automatically and does not close `R24`.
- The current unique task is now `T63: FR8 statcalib comparator gate review`.
- `T63` is the smallest honest next step: a docs-only gate discussion that decides whether a bounded FR8 task should exist at all, using only already existing repository evidence.

### 2026-05-27 Captain Update (T63 closeout supersession)

- `T63` has now been accepted as `PASS`.
- `T63` changes no code, no benchmark artifact, and no deployment-boundary fact. It only closes the question of whether one bounded FR8 task may be opened next.
- The legacy/truth boundary remains unchanged: current `statcalib` evidence is still mock-backed software-HIL only, and `.tflite` / real-board validation remain outside the mainline evidence pack.
- `T24` remains the authoritative historical frozen-set formal software revalidation record. `T64` must not relabel or silently rewrite that record.
- The current unique task is now `T64: FR8 statcalib extension-lane bounded benchmark`.

### 2026-05-29 Captain Update (T64 closeout supersession)

- `T64` has now been accepted as `PASS_WITH_WARNINGS`.
- `T64` does not change any legacy truth boundary about `.tflite`, real-board, or paper-grade expanded benchmark evidence.
- What `T64` adds is narrower: one clean-provenance bounded extension-lane result pack in which the historical `T24` frozen five-mode subset is preserved exactly and `statcalib` is reported only as a sixth separately labeled lane.
- The remaining legacy-risk correction is also now clearer:
  - `R24` remains open because the lane is still a minimal comparator semantics path, not a mature validated calibration comparator
  - `R28` is newly open because the T64 report wording is not yet strongly self-audited against its artifacts
- The current unique task is now `T65: FR8 extension-lane consistency guard and report closeout`.
- `T65` is bounded hardening only. It must not create new experimental evidence or touch theory-only branch materials.

## 2026-05-29 Captain Update (T65 closeout supersession)

- `T65` has now been accepted as `PASS_WITH_WARNINGS`.
- T65 warning classification is: `N1 accepted`, `N2 accepted`, `N3 accepted`.
- No new warning-derived risk item is opened by `T65`.
- `R28` should now be treated as closed by `T65`.
- `R24` remains open because the repository still has only a minimal statcalib comparator semantics path, not a mature validated calibration comparator.
- The current unique task is now `T66: FR8 statcalib sensitivity bounded benchmark`.
- `T66` is the next bounded mainline task because the remaining question is substantive robustness, not report wording: the repository still needs to know whether the T64 statcalib win survives a small predeclared heuristic sensitivity grid.

## 2026-06-01 Captain Update (T66 closeout supersession)

- `T66` has now been accepted as `PASS_WITH_WARNINGS`.
- T66 warning classification is: `N1 accepted`, `N2 deferred -> R24`, `N3 deferred -> R24`.
- `T66` corrects the remaining legacy ambiguity about local heuristic fragility: inside one bounded five-point grid, the statcalib extension-lane gain is not a single-point fluke.
- `T66` does not turn the lane into a mature calibration comparator. `R24` remains open because the repo still has not answered whether the gain is structurally teacher-anchor-dependent.
- The current unique task is now `T67: FR8 statcalib teacher-anchor dependence bounded benchmark`.
- `T67` is the next bounded mainline task because the remaining honest question is cross-anchor dependence, not more local parameter tuning.
## 2026-06-05 Captain Update (T67 closeout supersession)

- `T67` has now been accepted as `PASS_WITH_WARNINGS`.
- T67 warning classification is: `N1 accepted`, `N2 accepted`, `N3 deferred -> R24`.
- `T67` resolves the legacy ambiguity about gross teacher-anchor dependence: inside one bounded four-scenario teacher-anchor matrix, the current statcalib gain is not narrowly tied to `teacher_mode=ukf`.
- `T67` still does not turn the lane into a mature calibration comparator. `R24` remains open because two comparison rows remain `mixed`, and the current strongest aggregate lane still is not a clean generated-only result pack.
- The current unique task is now `T68: FR8 statcalib generated-only robustness bounded benchmark`.
- `T68` is the next bounded mainline task because the remaining honest question is generated-only robustness under the strongest non-`ukf` teachers, not more teacher-anchor breadth.

## 2026-06-08 Captain Update (T68 closeout supersession)

- `T68` has now been accepted as `PASS_WITH_WARNINGS`.
- T68 warning classification is: `N1 deferred -> R24`, `N2 deferred -> R24`, `N3 accepted`.
- `T68` removes one remaining legacy-style ambiguity: the statcalib extension lane is no longer blocked by "maybe no full generated-only winner exists in the bounded grid".
- `T68` still does not turn the lane into a mature calibration comparator. `R24` remains open because the strongest clean answer is still a tie set and the broader predeclared grid is still not uniformly clean.
- The current unique task is now `T69: FR8 statcalib clean-winner tie-break bounded benchmark`.
- `T69` is the next bounded mainline task because the remaining honest question is no longer existence, but whether the clean-winner tie set narrows under a stronger bounded repeat budget or remains the right final answer.

## 2026-06-10 Captain Update (T69 closeout supersession)

- `T69` has now been accepted as `PASS_WITH_WARNINGS`.
- T69 warning classification is: `N1 accepted`, `N2 accepted`.
- `T69` resolves the remaining legacy ambiguity about the bounded clean-winner tie-break question: inside the stronger `repeats=4` matrix, the `window_variance_t001 = t003 = t005` tie set persists exactly and no unique clean reference point emerges.
- `T69` still does not turn the lane into a mature calibration comparator. `R24` remains open because the lane is still extension-only and the broader predeclared grid is still not uniformly clean.
- The current unique task is now `T70: FR8 statcalib bounded closure pack and promotion gate`.
- `T70` is the next bounded mainline task because the repo now needs one authoritative closure pack and promotion/no-promotion gate built from read-only historical artifacts, not another threshold rerun.

## 2026-06-10 Captain Update (T70 closeout supersession)

- `T70` has now been accepted as `PASS`.
- `T70` closes the remaining legacy ambiguity about how the FR8 lane may be retold: there is now one accepted closure pack with explicit no-promotion and no-unique-threshold gates.
- `T70` still does not turn the lane into a mature calibration comparator, and it still does not upgrade the evidence beyond mock-backed software-HIL extension-lane status.
- `R24` remains open, but its carry-forward shape is now narrower: the dominant remaining risk is overclaiming or promotion drift, not missing closure-pack infrastructure.
- The current unique task is now `T50: Training reproducibility and material-regeneration pack`.
- `T50` is the next bounded mainline task because the next honest repository gap is no longer FR8 closure logic but training reproducibility/material evidence that can be strengthened on the current machine without assuming `.tflite` or hardware readiness.

## 2026-06-10 Captain Update (T50 closeout supersession)

- `T50` has now been accepted as `PASS`.
- `T50` does not resolve the remaining legacy ambiguity about full reproducibility or portability, but it does remove one missing infrastructure gap: the repo now has a code-backed training/material pack instead of scattered historical references only.
- `R11` remains open, but its carry-forward shape is now narrower: the dominant remaining issue is not whether canonical materials still exist, but whether reproducibility can be strengthened beyond one bounded clean CPU-only rerun.
- The current unique task is now `T48: True .tflite runtime smoke gate`.
- `T48` is the next bounded mainline task because the next honest repository gap is no longer training-material bookkeeping, but true `.tflite` runtime truth on the current machine under preserved canonical artifacts.

## 2026-06-10 Captain Update (T48 closeout supersession)

- `T48` has now been accepted as `PASS`.
- `T48` removes one legacy-style ambiguity: the repository no longer has to speak vaguely about whether preserved canonical `.tflite` artifacts can really execute on the current machine. Under one isolated `tensorflow==2.21.0` environment, the answer is now explicitly yes for the selected float / int8 preserved pair.
- The legacy caution still remains: this is an isolated current-host truth only, not a restored default environment, not a cross-host portability claim, and not a real-board claim.
- `board_backend.py` remains a placeholder real-board backend, so the historical software-vs-real-board boundary is unchanged.
- The current unique task is now `T49: Real-board smoke execution gate`.
- `T49` is the next bounded mainline task because the remaining deployment-boundary ambiguity has shifted from software-side `.tflite` runtime truth to current-host real-board preconditions and repo execution-path truth.
