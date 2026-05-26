# Project Snapshot / Raw Idea Record

## 1. 快照目的

本文件对应 `docs/reference/AI_coding_workflow.md` 中的 `00_raw_idea.md` / 项目快照角色：用最短事实说明项目解决什么问题、为什么值得继续、当前最小可验证实验是什么，以及当前阶段边界。

原始恢复期快照日期为 `2026-05-05`。截至 `2026-05-08`，第一轮 Recovery 已完成，本文件现在作为 Phase 2 继续开发的入口快照。

## 2. 基本信息

- 快照日期：`2026-05-05`
- 最近更新：`2026-05-26`
- 当前分支：`main`
- 工作流依据：`docs/reference/AI_coding_workflow.md`
- 当前阶段：`Phase 2: Controlled Development`
- 当前决策状态：`Go`
- 当前唯一任务来源：`docs/04_task_board.md`

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
- `docs/P0_smoke_bootstrap.md`
- `docs/P3_software_hil_bootstrap.md`
- `docs/P4_benchmark_recovery_bootstrap.md`
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

- `T47: Paper ablation result-pack and material ledger`

## 12. 快照结论

当前项目的核心问题已经从“有没有可靠治理层和入口”切换为“如何在不夸大完成度的前提下继续增强证据”。Phase 2 的默认策略是：

1. P4 benchmark protocol 与 bounded evidence 已完成第一轮受控增强；`T24` 已完成 frozen-set formal software revalidation，`T25` gate review 已接受该结果边界。
2. 训练链与 `.tflite` 独立 bootstrap 已完成，但都不等于跨机器完整环境或真实 `.tflite` runtime 已恢复。
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
16. `T55` 已完成并表明 pure I1 lower-clip intervention 为 mixed 且整体偏 harmful；随后 `T56` 已完成机制 claim 重构收口，因此当前唯一任务切到 `T47`，但它只能作为带 hedge 的 docs-only paper-material lane 推进。

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
