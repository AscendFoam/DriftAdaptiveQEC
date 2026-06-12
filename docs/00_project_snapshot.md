# Project Snapshot / Raw Idea Record

## 1. 快照目的

本文件对应 `docs/reference/AI_coding_workflow.md` 中的 `00_raw_idea.md` / 项目快照角色：用最短事实说明项目解决什么问题、为什么值得继续、当前最小可验证实验是什么，以及当前阶段边界。

原始恢复期快照日期为 `2026-05-05`。截至 `2026-05-08`，第一轮 Recovery 已完成，本文件现在作为 Phase 2 继续开发的入口快照。

## 2. 基本信息

- 快照日期：`2026-05-05`
- 最近更新：`2026-06-12`
- 当前分支：`main`
- 工作流依据：`docs/reference/AI_coding_workflow.md`
- 当前阶段：`Phase 2: Controlled Development`
- 当前决策状态：`Go`
- 当前唯一任务来源：`docs/04_task_board.md`

## 2026-06-12 Captain Update (T76 closeout)

- `T76` 已由 Captain 判定为 `PASS_WITH_WARNINGS`。
- `T76` 在不改动源码、测试、`runs/`、`artifacts/` 或任何证据等级的前提下，完成了真实 rendered preview、人工可读性 QA、contact sheet / PDF bundle 与 Results-section assembly。
- warning 分类为：
  - `N1` preview-source 聚合行字段语义复用 = `deferred -> R34`
  - `N2` `.tmp_t76_*` 探针/缓存残留 = `accepted`
  - `N3` 逐图 QA 结论未内联完整上游 `T74-*` stable ID = `deferred -> R34`
- 当前唯一任务切换为 `T77: 论文 note-draft 结果层同步与 T76 traceability hardening`。
- `T77` 仍留在 main 分支 paper-material 主线，只处理 note-draft 结果层同步、preview-source schema 清理、逐图 stable-ID 显式绑定和可选编译检查；不与 theory 分支大范围改写混做，也不允许扩成 benchmark、`.tflite`、真板执行、sidecar promotion 或 full-manuscript reopen。

## 2026-06-12 Captain Update (T75 closeout)

- `T75` 已由 Captain 判定为 `PASS`。
- `T75` 在不改动源码、测试、`runs/`、`artifacts/` 或任何证据等级的前提下，完成了主线 Results prose、caption/placement lock、appendix bridge、do-not-write guardrail 和三张 publication-facing `T75-FIG-*` 资产。
- 本次 closeout 没有引入新的 deferred/rejected warning，也没有新增风险项；review 中的两条非阻塞意见仅作为后续提交与排版 QA 的操作提醒。
- 当前唯一任务切换为 `T76: Rendered figure QA and results-section assembly pack`。
- `T76` 仍留在 main 分支 paper-material 主线，只处理 rendered figure QA 与 Results-section assembly，不与 theory 分支混做，也不允许扩成 benchmark、`.tflite`、真板执行、sidecar promotion 或 full-manuscript reopen。

## 2026-06-12 Captain Update (T74 closeout)

- `T74` 已由 Captain 判定为 `PASS`。
- `T74` 已把主线 simulation/material-first 路线整理成一套 paper-ready 结果表、caption、insertion map、traceability 资产和 gap checklist，但没有改动源码、测试、`runs/`、`artifacts/` 或任何证据等级。
- 这次 closeout 没有引入新的 deferred/rejected warning，也没有新增风险项；review 里唯一的 non-blocking issue 只是提醒提交时要精确暂存已有 diff。
- 当前唯一任务切换为 `T75: Main-text results prose and final figure authoring pack`。
- `T75` 仍留在 main 分支实验主线，只处理实验结果写作与成图 authoring，不与 theory 分支混做，也不允许扩成 benchmark、`.tflite`、真板执行、sidecar promotion 或 full-manuscript reopen。

## 2026-06-12 Captain Update

- `T73` 已由 Captain 判定为 `PASS`。
- `T73` 完成了 post-`T72` 主线 paper-facing claim/evidence、result/figure、risk 和 README 入口的统一回写，但没有改动源码、测试、`runs/`、`artifacts/` 或治理边界。
- 这次 closeout 没有引入新的 deferred/rejected warning，也没有新增风险项；现有主风险仍是 `R13/R14/R32/R33` 和受其约束的 `T37`。
- 当前唯一任务切换为 `T74: Paper-ready simulation result and figure pack`。
- `T74` 仍留在 main 分支实验主线，不与单独的 theory 分支混做，也不允许扩成 benchmark、`.tflite`、真板执行、sidecar promotion 或 paper prose reopen。

## 2026-06-11 Captain Update

- `T72` 已由 Captain 判定为 `PASS_WITH_WARNINGS`。
- `T72` 解决了 `T71` 留下的主 provenance 问题：`probe_limitations` 不再把未执行探测写成固定事实，default-config / CLI override provenance 变为 execution-derived / override-aware，`expected_byte_count_basis` 改为运行时推导，且 `T49` replay 与 `T72` current-host regeneration verdict 继续一致。
- 这次通过不代表真板执行成功，也不代表 `T37` 解锁；当前最强结论仍然只是 current-host / regenerated verdict 继续为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。
- `T72` warning 分类为：
  - `N1` 最小 config 场景下 path provenance 仍会把代码默认值写成 `source_kind=config_field` = `deferred -> R32`
  - `N2` Worker 原始主报告路径曾短暂落在精确 allowed files 之外，但当前 `HEAD` 已整理回允许目录 = `accepted`
  - `N3` 缺少覆盖 path 字段缺省回退标签的 focused regression = `deferred -> R32`
- `R31` 已由 `T72` 收口；新的后续缺口收敛为 `R32`，它只针对 future-host 最小 config 场景下的 provenance 标签精确性。
- 当前唯一任务切换为 `T73: Mainline claim/evidence and result/figure/risk ledger refresh`。
- `T73` 仍留在 main 分支实验主线，不与单独的 theory 分支混做，也不允许扩成 benchmark、`.tflite`、真板执行或 paper prose reopen。

## 2026-06-11 Captain 优先级调整

- 用户已明确：当前暂无可用的 `Linux + FPGA` 硬件宿主，因此 real-board execution 路线不再作为近期主线。
- main 分支当前优先补论文所需的仿真结果、图表、caption、claim/result/risk 台账与 supporting materials。
- `T37` 继续保持 `blocked + lowest-priority backlog`；在硬件条件变化前，不让真板执行任务抢占 `T73` 及其后的 paper-material 主线任务。
- `T73` 之后，唯一推荐的下一张主线任务是 `T74` 一类“论文可直接复用的仿真结果与图表打包”任务，而不是继续推进真板执行准备。

## 2026-05-16 Captain Update

- `T38` reviewer verdict accepted by Captain as `PASS`.
- `T38` warnings are all classified as `accepted`; there are no `deferred` or `rejected` warnings from this review.
- Milestone 2I review is recorded in `docs/review/Milestone2I_review.md`; verdict = `Conditional Allow`.
- Current unique task is now `T31: Training-chain portable dependency lock plan`.
- `T31` is a documentation/environment-boundary task only. It must not install packages, run training, run benchmark, create new `runs/` or `artifacts/`, or modify `docs/02_experiment_plan.md`.

## 2026-05-17 Captain Update

- `T42` reviewer verdict accepted by Captain as `PASS`.
- T42 blocking issues: none.
- T42 non-blocking comments were accepted as framing guidance; the subsection-6 novelty wording was neutralized and the method-forward title was kept as a working recommendation pending any later human override.
- `T42` completes the Background / Related Work scaffold and method-positioning calibration without upgrading any mock-backed, stub, readiness, or smoke evidence level.
- At that time, the current unique task was `T43: Paper Background / Related Work bounded prose draft`.
- `T43` is docs-only. It may draft only the Background / Related Work prose under the already locked evidence boundaries and working method-forward framing; it must not run new experiments, rewrite evidence levels, or overclaim `.tflite` / real-board / full reproducibility completion.

## 2026-05-18 Captain Update

- `T43` reviewer verdict accepted by Captain as `PASS`.
- T43 warnings are all classified as `accepted`; there are no `deferred` warnings from this review.
- The project now enters `Research Reality Recovery Mode` before any further prose expansion.
- Current unique task is now `T44: Research Reality Recovery Mode setup and evidence-gap ledger`.
- `T44` is docs-only. It must freeze claim/evidence/material truth and build the recovery baseline before any more manuscript drafting, experiments, or evidence upgrades.

## 2026-05-19 Captain Update

- `T45` reviewer verdict accepted by Captain as `PASS`.
- T45 warnings are all classified as `accepted`; there are no `deferred` warnings from this review.
- `T45` freezes the benchmark-expansion protocol at the policy level only; it does not execute a broader benchmark lane.
- Current unique task is now `T46: Multi-seed mechanism/intervention plan and trace pack`.
- `T46` is docs-only. It must stay on the mechanism-evidence boundary and must not become a new benchmark, training, `.tflite`, hardware, or cleanup task.

## 2026-05-22 Captain Update

- `T46` reviewer verdict accepted by Captain as `PASS`.
- T46 non-blocking comments are all treated as `accepted`; there are no `deferred` or `rejected` warning items from this review.
- `T46` successfully freezes the smallest believable multi-seed / intervention evidence plan without upgrading any single-seed evidence into multi-seed confirmation or causal proof.
- The project does not jump directly to the old `T47` paper-material lane after T46.
- Current unique task is now `T54: Phase A multi-seed trace-only generalization probe`.
- `T54` is a bounded execution task that stays inside the existing Full vs Gated v5, frozen-four-scenario, trace-only mechanism lane. It must not run interventions, widen the benchmark set, or touch `.tflite`, hardware, or cleanup scope.

## 2026-05-23 Captain Update

- `T54` reviewer verdict accepted by Captain as `PASS`.
- T54 non-blocking comments are all treated as `accepted`; there are no `deferred` or `rejected` warning items from this review.
- `T54` upgrades the mechanism story from single-seed diagnostic evidence to bounded multi-seed diagnostic generalization: the committed-`b` instability pattern is broadly repeated with qualifications across the 6-seed pack, but `C4` still remains `partial`.
- The project still does not jump directly to `T47`.
- Current unique task is now `T55: Phase B multi-seed I1 residual-clip intervention probe`.
- `T55` is a bounded config-only intervention task: it reuses existing seed/model assets, tests only one Gated v5 intervention (`residual_clip_b: 0.06`) on the same 6 seeds and frozen four scenarios, and does not reopen training, `.tflite`, hardware, cleanup, or benchmark-expansion scope.

## 2026-05-24 Captain Update

- `T55` reviewer verdict accepted by Captain as `PASS`.
- T55 non-blocking comments are all treated as `accepted`; there are no `deferred` or `rejected` warning items from this review.
- `T55` adds the first bounded multi-seed intervention evidence, and the result is mixed and mostly harmful: the pure I1 lower-clip intervention harms 4/6 seeds and helps 2/6.
- The simple mechanism framing “high committed-`b` is harmful” is not supported as a general explanation; `C4` remains `partial`.
- The project still does not jump directly to `T47`.
- Current unique task is now `T56: Post-I1 mechanism claim reframing gate`.
- `T56` is docs-only. It must freeze which mechanism claims remain valid after `T55`, which claims must be weakened or reframed, and whether any later `T47` or second-intervention lane is still justified.

## 2026-05-24 Captain Update (T56 closeout)

- `T56` reviewer verdict accepted by Captain as `PASS`.
- T56 non-blocking comments are all treated as `accepted`; there are no `deferred` or `rejected` warning items from this review.
- T56 freezes the current mechanism-claim boundary into retain / weaken / retire / reframe / still-open and confirms that the simple harmful-instability story cannot be kept as a general explanation.
- A second intervention lane is not auto-approved; it remains deferred pending a better question.
- Current unique task is now `T47: Paper ablation result-pack and material ledger`.
- `T47` is docs-only and may proceed only under the mechanism-hedge boundary defined by `T56`; it is not unconditional paper expansion or mechanism closure.

## 3. 解决什么问题

该项目围绕 “CNN + FPGA 快慢回路进行近似 GKP 码解码” 展开，当前主代码与实验材料主要分布在：

- `physics/`
- `cnn_fpga/`
- `benchmark/`
- `docs/`

核心问题：

1. 在 GKP 误差校正中，用 fast loop 承担低延迟线性控制。
2. 用 slow loop 的 CNN / teacher-guided 模块周期性更新 `(K, b)` 等控制参数。
3. 用软件 HIL、benchmark 与后续真板路径证明该闭环不仅有算法效果，也能落入工程约束。

从文档与代码交叉读取后的判断是：

- 这不是空壳仓库
- 也不是只停留在 idea 或设计稿
- 它已经积累了较完整的 P0-P4 代码路径、实验配置和结果目录
- 但仓库缺少统一治理层，默认环境也尚未恢复到“开箱可跑”

截至 Phase 2，治理层与最小可复验入口已经恢复；后续工作转为受控证据增强与环境边界补齐。

## 4. 为什么现在值得继续

当前继续推进的理由：

1. P0/P1/P2/P3/P4 的代码与历史实验资产都存在，不是从零立项。
2. Recovery 已恢复 P0/P3/P4 的最小可复验路径。
3. `T12` 已将 bounded software HIL recovery smoke 收口到逐字一致复验。
4. `T9` 已完成 `single-scenario + four-mode + repeats=1` 的 P4 frozen baseline recovery smoke。
5. 主要剩余风险已经从“仓库是否可信”转为“如何有边界地增强 benchmark、训练、TFLite 与真板证据”。

## 5. 最小可验证实验

当前推荐的最小可验证入口均以 `C:\ProgramData\anaconda3\python.exe` 和 `requirements-recovery.txt` 为基准。

### 5.1 P0 smoke

```powershell
& 'C:\ProgramData\anaconda3\python.exe' benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda
```

### 5.2 P3 software HIL recovery smoke

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml
```

边界：`mock + model_artifact + artifact_npz + inproc`，不是真板或 `.tflite` runtime。

### 5.3 P4 frozen baseline single-scenario smoke

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds
```

边界：`mock-backed P4 recovery smoke`，不是正式多场景 frozen benchmark。

## 6. 最相似已有工作 / 内部证据

外部论文与路线参考集中在 `docs/02_experiment_plan.md` 和相关背景文档中；本快照只列内部事实源：

- `docs/02_experiment_plan.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/recovery_bootstrap/P0_smoke_bootstrap.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/review/T13_recovery_exit_review.md`

## 7. 失败标准

Phase 2 任一任务若出现以下情况，应进入 `Narrow` 或 `Pause` 判断：

1. 需要隐式修改 benchmark 口径才能得到结论。
2. 需要把 `mock`、`stub`、`placeholder` 结果写成真实部署完成。
3. 无法在文档中复现命令、环境、run dir 与边界。
4. Reviewer 给出 `BLOCK` 且二次修复后仍未解除。
5. 任务越界修改 Allowed files 之外的关键代码或历史结果。

## 8. 恢复前观察到的仓库现实

### 8.1 治理文件缺失

恢复前，仓库根目录中没有以下最小治理文件：

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

这些文件现在已经补齐，但仍应由 Captain 持续维护。

### 8.2 代码主干存在

已确认存在以下主模块：

- `cnn_fpga/data/`
- `cnn_fpga/model/`
- `cnn_fpga/decoder/`
- `cnn_fpga/runtime/`
- `cnn_fpga/hwio/`
- `cnn_fpga/benchmark/`
- `physics/`

### 8.3 结果与生成物很多

仓库内已有大量：

- `runs/`
- `artifacts/`
- 自动生成配置
- `__pycache__/` / `.pyc`

这说明项目确实跑过很多轮实验，但也意味着仓库噪声较大，恢复期需要明确区分“源码、治理文件、实验产物、缓存文件”。

## 9. 恢复前已做的最小验证

### 9.1 成功

命令：

```powershell
python --version
```

结果：

- 默认解释器为 `Python 3.13.7`

命令：

```powershell
python -c "import cnn_fpga, physics; print(cnn_fpga.__version__)"
```

结果：

- 可导入本地包
- `cnn_fpga.__version__ = 0.1.0`

### 9.2 失败

命令：

```powershell
python benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test
```

结果：

- 失败
- 报错：`ModuleNotFoundError: No module named 'numpy'`

### 9.3 当前解释

这说明：

1. 默认 Python 解释器不是完全空的，但也不是项目可运行环境
2. 当前最优先阻塞项不是算法实现，而是依赖矩阵与最小入口恢复

Phase 2 当前解释：默认 Python 仍不作为推荐入口；recovery smoke 统一使用 `C:\ProgramData\anaconda3\python.exe`。

## 10. 当前已确认的关键边界

### 10.1 软件 HIL 与真板 HIL 边界

- `cnn_fpga/benchmark/run_hil_suite.py` 已支持 mock/backend 抽象
- `cnn_fpga/hwio/board_backend.py` 仍然是 placeholder 风格的真板后端骨架
- 文档中也已明确：当前更准确状态是 `P3-软件 HIL 完成，P3-真板 HIL 待完成`

### 10.2 benchmark 代码主线存在

已确认存在：

- `benchmark/compare_full_vs_simplified_ler.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

这说明恢复期不需要从零设计实验入口，而要优先确认它们在当前环境下是否还能真实跑通。

## 11. 继续开发的当前唯一任务

当前唯一任务由 `docs/04_task_board.md` 定义：

- `T77: 论文 note-draft 结果层同步与 T76 traceability hardening`

## 12. 快照结论

当前项目的核心问题已经从“有没有可靠治理层和入口”切换为“如何在不夸大完成度的前提下继续增强证据”。Phase 2 的默认策略是：

1. P4 benchmark protocol 与 bounded evidence 已完成第一轮受控增强；`T24` 已完成 frozen-set formal software revalidation，`T25` gate review 已接受该结果边界。
2. 训练链独立 bootstrap 与 current-host isolated true `.tflite` runtime 已完成，但都不等于跨机器完整环境、default-env runtime 或部署闭环已恢复。
3. `T24` 证据等级仍限定为 `mock-backed` software HIL formal benchmark；不得外推为 `.tflite` runtime、真板验证或 paper-grade expanded benchmark。
4. `T27` 已把 teacher diagnostics 缺口缩窄为 broadcast teacher 布局与 scalar explain 机制不匹配；`R20` 已缩窄为独立 fast-loop saturation 路径。
5. `T28` 已完成 teacher diagnostics missing-vs-zero 语义修复与最小 smoke；`R21` 对当前 writer 语义可关闭，但 `R10` 机制证据仍未完全修复。
6. `T29` 已修复 T28 review 指出的 P4 markdown report 重复表头问题，并通过 independent review，Captain verdict = `PASS`。
7. `T26` 已完成 calibration/statcalib baseline feasibility gate，并通过 independent review，Captain verdict = `PASS`；gate 结论为 `CONDITIONAL_GO`，只允许后续作为 separate comparator lane 推进。
8. `T30` 已完成 statcalib comparator 的 concrete interface contract 与 interface-level tests，并通过 independent review，Captain verdict = `PASS`；该结论不等于 statcalib 已接入 slow loop、formal benchmark、`.tflite` runtime 或真板路径。
9. `T36` 已完成并通过 adversarial review，Captain verdict = `PASS`；其结论将 `seed=20260429` 的收益收缩缩窄为 residual-amplitude / teacher-delta regime instability hypothesis，但不是 causal proof。
10. `T38` 已完成并通过 Captain `PASS` 收口；它只提供 single-seed trace-level mechanism evidence，不是 mitigation success、formal benchmark、`.tflite` runtime 或 real-board validation。
11. `T31` 已完成并通过 adversarial review，Captain verdict = `PASS`；产物是 training-chain portable dependency-lock plan，不是 clean-environment rebuild proof。
12. `T39` 已完成并通过 adversarial review，Captain verdict = `PASS`；它证明了 clean CPU-only environment、draft lock 和 dry-run/import-level bootstrap 可复现，但不等于 real training reproducibility。
13. `T40` 已完成并通过 adversarial review，Captain verdict = `PASS`；它证明了 clean CPU-only environment 已能完成一次真实的最小训练 smoke，但不等于 full training reproducibility、GPU/CUDA portability、Linux portability、`.tflite` runtime 或 benchmark readiness。
14. `T41` 已完成并通过 Captain `PASS` 收口；Milestone 2K 已正式由 gate review 关闭，并为后续 `T42-T46` 的 paper-boundary 与 mechanism-planning 任务提供了边界前置。
15. `T46` 已完成并通过 Captain `PASS` 收口；它把 `seed=20260429` 的单 seed 机制诊断收束成了一个明确的多 seed / intervention 计划，但没有升级任何 evidence level。
16. `T70` 已完成并通过 Captain `PASS` 收口；FR8 当前已有一个 code-backed closure pack，可明确阻止 promotion 和唯一阈值外推，但这不升级为成熟 comparator、`.tflite` 或真板证据。
17. `T50` 已完成并通过 Captain `PASS` 收口；仓库现在已有一份 code-backed 训练复现与材料再生证据包，可统一引用 canonical 训练材料、主线 preserved references 和一次 clean CPU-only bounded train+eval rerun。
18. 当前唯一任务已切换为 `T75`，用于在 `T74` 已形成的 stable-ID 材料包基础上，继续冻结主文 Results 段落、最终成图资产和 caption/placement lock；它比在当前硬件条件下重开真板执行或直接恢复 full-manuscript 扩写更符合现阶段主线。

## 13. T45 后的拟议路线图（非当前任务）

T45 结束后，基于 recovery 结论形成的下一轮 bounded task 建议是：

1. `T47`：paper ablation result-pack and material ledger
2. `T48`：true `.tflite` runtime booster, only if environment is available
3. `T49`：real-board smoke execution gate, only if hardware host and bitstream evidence are ready

更后面的 milestone 方向可粗略分为：

- benchmark hardening
- deployment boundary boosters
- reproducibility and material pack
- paper re-open gate

这些都只是后续路线图，不是当前唯一任务，也不是已执行事实。
## 2026-05-24 Captain Update (T47 closeout supersession)

- `T47` has now been accepted as `PASS`.
- `T47` warnings `N1-N5` are all `accepted`; there are no `deferred` or `rejected` items.
- `T47` only froze the paper-facing ablation/material ledger honestly; it did not close `FR7`.
- The current unique task is now `T57: FR7 feature/teacher ablation re-execution under locked T24 protocol`.
- `T57` must stay bounded to the locked `T24` protocol and must not reopen `.tflite`, real-board, cleanup, benchmark-expansion, comparator-expansion, or intervention scope.

## 2026-05-26 Captain Update (T57 closeout supersession)

- `T57` has now been accepted as `PASS`.
- `T57` review introduces no blocking issue and no new warning item that needs `accepted / deferred / rejected` classification beyond the verdict itself.
- Therefore no new risk item is opened by warning classification for `T57`.
- `T57` closes `FR7` as a bounded frozen-set result-table gap, but it does not upgrade causal interpretation, mechanism closure, expanded benchmark evidence, `.tflite` validation, or real-board validation.
- The strongest boundary update from `T57` is that `hybrid_no_teacher_params` is best in all 4 frozen scenarios, so the paper must not claim teacher params are a necessary positive contributor to the win.
- The current unique task is now `T58: FR6 multi-seed mechanism/intervention figure pack`.
- `T58` is docs-only, must reuse existing `T54/T55/T56` evidence, must not create new `runs/` or `artifacts/`, and must stay separate from theory-only branch materials.

## 2026-05-26 Captain Update (T58 closeout supersession)

- `T58` has now been accepted as `PASS_WITH_WARNINGS`.
- T58 warning classification is: `N1 accepted`, `N2 accepted`, `N3 accepted`, `N4 accepted`.
- There is no `deferred` or `rejected` warning item from `T58`, so no new risk is opened by warning classification.
- `T58` closes `FR6` only as a bounded descriptive figure pack. It does not close `R10`, and it does not upgrade `C4` beyond `partial`.
- The largest remaining mainline paper-material gap is now `FR8`, but the repository still lacks an integrated `statcalib` comparator lane.
- The current unique task is now `T59: Statcalib separate comparator lane integration and bounded smoke`.
- `T59` stays on the mainline experiment-evidence lane, must remain isolated from theory-only branch materials, and must not rewrite frozen `T24` benchmark semantics.

## 2026-05-26 Captain Update (T59 closeout supersession)

- `T59` has now been accepted as `PASS_WITH_WARNINGS`.
- T59 warning classification is: `W1 deferred`, `W2 accepted`, `W3 deferred`.
- The deferred items have been written into the risk ledger and must be treated as pre-FR8 blockers.
- `T59` closes separate-lane integration, status propagation, and one bounded smoke only. It does not open `FR8`, and it does not upgrade the evidence to formal comparator ranking.
- The current unique task is now `T60: Statcalib lane isolation and regression hardening`.
- `T60` stays on the mainline experiment-evidence lane, must create no new run root, must remain isolated from theory-only branch materials, and must not rewrite frozen `T24` semantics.

## 2026-05-27 Captain Update (T60 closeout supersession)

- `T60` has now been accepted as `PASS`.
- `T60` review introduces no blocking issue and no new warning item that needs `accepted / deferred / rejected` classification.
- `T60` closes the T59 cross-mode semantics blocker; `W1` is resolved and `R26` should now be treated as closed.
- `R27` remains open but narrower: the regression-coverage gap is closed, while provenance-clean fairness/robustness sanity is still missing before any `FR8` task.
- `T60` hardens semantics and tests only; it does not upgrade the lane into formal comparator evidence.
- The current unique task is now `T61: Statcalib clean-provenance fairness sanity rerun`.
- `T61` must stay on the mainline experiment branch, start from a clean committed worktree, create at most one T61-scoped run root, and remain separate from theory-only branch materials.

## 2026-05-27 Captain Update (T61 closeout supersession)

- `T61` has been judged `BLOCK`.
- `T61` preserved the bounded fairness signal, but it did not close the clean-provenance blocker it was created to repair.
- The concrete blocker is now explicit: clean launch `HEAD=9174065`, final `summary.json git_commit=6058f42`, and mid-run branch movement means the run lacks one defensible commit identity.
- `R26` remains closed by `T60`.
- `R27` remains open and is now a pure provenance-isolation blocker before any later `FR8` gate discussion.
- The current unique task is now `T62: Statcalib provenance-isolated fairness rerun`.
- `T62` is blocking-only, must stay on the mainline experiment branch, must not touch theory-only materials, and is the single automatic retry for this blocker.

## 2026-05-27 Captain Update (T62 closeout supersession)

- `T62` has now been accepted as `PASS`.
- `T62` closes the specific blocker that caused `T61` to fail: the repository now has one provenance-clean bounded fairness sanity rerun for the current `statcalib` smoke lane.
- `R26` remains closed by `T60`.
- `R27` should now be treated as closed by `T62`.
- `T62` still does not upgrade the evidence beyond mock-backed software-HIL bounded sanity evidence and does not open `FR8` automatically.
- The current unique task is now `T63: FR8 statcalib comparator gate review`.
- `T63` is docs-only and exists to decide whether a bounded FR8 task should exist at all, without mixing mainline experiment evidence with theory-only branch materials.

## 2026-05-27 Captain Update (T63 closeout supersession)

- `T63` has now been accepted as `PASS`.
- `T63` is a docs-only gate review. It does not itself create new comparator evidence and it does not upgrade any `.tflite`, real-board, or paper-grade benchmark claim.
- `T63` concludes that `R27` remains closed by `T62` and that no further pre-FR8 prerequisite is needed before one bounded extension-lane execution task.
- `R24` remains open, but after `T63` it is treated as the main scope/reporting constraint for the next task rather than as a blocker that requires another gate loop.
- The current unique task is now `T64: FR8 statcalib extension-lane bounded benchmark`.
- `T64` must remain on the mainline experiment branch, keep `statcalib` as a separately labeled extension lane, preserve the historical frozen five-mode table, and stay inside mock-backed software-HIL scope only.

## 2026-05-29 Captain Update (T64 closeout supersession)

- `T64` has now been accepted as `PASS_WITH_WARNINGS`.
- T64 warning classification is: `N1 deferred`, `N2 deferred`, `N3 deferred`.
- `N1` and `N2` open `R28`: the current T64 result pack is experimentally acceptable, but its report/artifact consistency still depends too much on manual wording discipline.
- `N3` remains under `R24`: even after T64, `statcalib` is still only a separately labeled bounded extension lane, not a mature validated calibration comparator.
- `T64` closes one clean-provenance bounded FR8 extension-lane benchmark on the locked four-scenario protocol and preserves the historical `T24` frozen subset exactly.
- The current unique task is now `T65: FR8 extension-lane consistency guard and report closeout`.
- `T65` is a code/test/docs hardening task only. It must create no new run root and must stay separate from theory-only branch materials.

## 2026-05-29 Captain Update (T65 closeout supersession)

- `T65` has now been accepted as `PASS_WITH_WARNINGS`.
- T65 warning classification is: `N1 accepted`, `N2 accepted`, `N3 accepted`.
- No new risk item is opened by warning classification for `T65`.
- `T65` closes `R28`: the T64 result pack is now backed by an explicit audit helper, focused regression coverage, and a bounded consistency-audit document.
- `R24` remains open as the dominant mainline comparator-scope risk.
- The current unique task is now `T66: FR8 statcalib sensitivity bounded benchmark`.
- `T66` is a bounded execution-and-summary task only. It may probe a small predeclared statcalib sensitivity grid, but it must not rewrite `T24`, change statcalib/runtime semantics, widen into `.tflite` or real-board scope, or mix with theory-only branch materials.

## 2026-06-01 Captain Update (T66 closeout supersession)

- `T66` has now been accepted as `PASS_WITH_WARNINGS`.
- T66 warning classification is: `N1 accepted`, `N2 deferred -> R24`, `N3 deferred -> R24`.
- `T66` closes one bounded local-grid robustness gap: the T64 extension-lane win survives a predeclared five-point statcalib sensitivity grid under clean provenance.
- `T66` does not close `R24`: aggregate-best and stability-best variants still differ, and the `static_bias_theta / statcalib_high_threshold` scenario-best row still carries aggregate `statcalib_status = mixed`.
- The current unique task is now `T67: FR8 statcalib teacher-anchor dependence bounded benchmark`.
- `T67` stays on the mainline experiment-evidence lane, must keep `T24` authoritative, must keep statcalib as a separate extension lane, and must stay separate from theory-only branch materials.
## 2026-06-05 Captain Update (T67 closeout supersession)

- `T67` has now been accepted as `PASS_WITH_WARNINGS`.
- T67 warning classification is: `N1 accepted`, `N2 accepted`, `N3 deferred -> R24`.
- `T67` closes the gross teacher-anchor dependence question honestly: non-`ukf` teachers remain competitive and still beat both frozen anchors across all four locked scenarios.
- `T67` does not close `R24`: the strongest aggregate statcalib lane still is not a clean generated-only result pack.
- The current unique task is now `T68: FR8 statcalib generated-only robustness bounded benchmark`.
- `T68` stays on the mainline experiment-evidence lane, must keep `T24` authoritative, must keep statcalib as a separate extension lane, and must stay separate from theory-only branch materials.

## 2026-06-08 Captain Update (T68 closeout supersession)

- `T68` has now been accepted as `PASS_WITH_WARNINGS`.
- T68 warning classification is: `N1 deferred -> R24`, `N2 deferred -> R24`, `N3 accepted`.
- `T68` closes the bounded generated-only existence question honestly: full generated-only winners now exist inside the predeclared grid, and the strongest clean answer is the tied `window_variance_t001 = t003 = t005` set.
- `T68` does not close `R24`: the clean winner set is still not unique, and some other predeclared candidates remain `mixed`.
- The current unique task is now `T69: FR8 statcalib clean-winner tie-break bounded benchmark`.
- `T69` stays on the mainline experiment-evidence lane, must keep `T24` authoritative, must keep statcalib as a separate extension lane, and must stay separate from theory-only branch materials.

## 2026-06-10 Captain Update (T69 closeout supersession)

- `T69` has now been accepted as `PASS_WITH_WARNINGS`.
- T69 warning classification is: `N1 accepted`, `N2 accepted`.
- `T69` closes the last bounded execution question in the current FR8 statcalib lane: under the locked four-scenario matrix and `repeats=4`, the strongest clean answer remains the persistent `window_variance_t001 = t003 = t005` tie set and no unique clean reference point emerges.
- `T69` does not change any truth boundary about `.tflite`, real-board, or paper-grade expanded benchmark evidence. The lane remains mock-backed software-HIL extension-lane evidence only.
- The current unique task is now `T70: FR8 statcalib bounded closure pack and promotion gate`.
- `T70` is the smallest honest next step because the repo now needs one consolidated, code-backed FR8 closure pack and explicit promotion/no-promotion gate, not another threshold rerun.

## 2026-06-10 Captain Update (T70 closeout supersession)

- `T70` has now been accepted as `PASS`.
- `T70` introduces no warning-derived risk item from review classification.
- `T70` closes the FR8 closure-pack gap honestly: the repository now has one code-backed closure artifact that keeps `T24` authoritative, keeps `statcalib` as an extension lane, blocks promotion, and blocks any unique-threshold retelling without a new task.
- `T70` does not change any truth boundary about `.tflite`, real-board, paper-grade expanded benchmark evidence, or mature calibration-comparator validation.
- The current unique task is now `T50: Training reproducibility and material-regeneration pack`.
- `T50` is the smallest honest next step because `.tflite` / 真板前提仍未满足，而训练复现与材料再生证据仍缺一个统一、代码驱动的主线证据包。

## 2026-06-10 Captain Update (T50 closeout supersession)

- `T50` has now been accepted as `PASS`.
- `T50` introduces no warning-derived risk item from review classification.
- `T50` closes one mainline training-material gap honestly: the repository now has one code-backed pack that ties canonical training materials, preserved mainline model references, and one clean CPU-only bounded train+eval rerun together.
- `T50` still does not prove full training reproducibility, repeated-run stability, cross-host portability, GPU/CUDA portability, Linux portability, `.tflite` correctness, or real-board validation.
- The current unique task is now `T48: True .tflite runtime smoke gate`.
- `T48` is the smallest honest next step because the repo now has both preserved `.tflite` candidates and one authoritative training/material boundary artifact, while real-board execution and paper re-open still require stronger runtime/deployment truth.

## 2026-06-10 Captain Update (T48 closeout supersession)

- `T48` has now been accepted as `PASS`.
- `T48` closes one narrow current-host true `.tflite` runtime gap honestly: the repository now has one isolated `tensorflow==2.21.0` environment on this machine that can real-load and real-execute preserved `static_theta_v2` float / int8 `.tflite` artifacts and can run bounded source-vs-`.tflite` consistency checks.
- `T48` does not restore default-environment compatibility and does not upgrade the result to HIL closure, real-board validation, or deployment closure.
- `R12` remains open, but its carry-forward shape is now narrower: the dominant remaining `.tflite` issue is no longer “current-host true runtime unconfirmed”, but “default env / portability / deployment still not closed”.
- The current unique task is now `T49: Real-board smoke execution gate`.
- `T49` is the next bounded mainline task because the remaining deployment-boundary question is now host/device/bitstream/AXI/DMA truth on the current machine, not software-side `.tflite` runtime truth.
- `T49` has now been accepted as `PASS_WITH_WARNINGS`.
- `T49` closes one honest current-host real-board gate pack with verdict `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`; no real-board smoke was executed.
- `T49` warning classification is now: `W1 deferred -> R30`, `W2 deferred -> R30`, `W3 deferred -> R30`.
- `T49` does not upgrade the repository to real-board validation, HIL closure, or deployment closure.
- `R13/R14` remain open but narrower: the current-host truth is no longer unknown; it is explicitly blocked by missing openable device paths, missing bound bitstream/RTL/DMA contract evidence, and a placeholder repo execution path.
- The current unique task is now `T71: Real-board gate regeneration and host-transfer pack`.
- `T71` is the next bounded mainline task because the repo now needs a checked-in, role-aware, read-only gate regeneration path for future candidate hosts before any later real-board execution task can be opened honestly.
