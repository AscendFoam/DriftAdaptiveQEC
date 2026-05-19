# Project Snapshot / Raw Idea Record

## 1. 快照目的

本文件对应 `docs/reference/AI_coding_workflow.md` 中的 `00_raw_idea.md` / 项目快照角色：用最短事实说明项目解决什么问题、为什么值得继续、当前最小可验证实验是什么，以及当前阶段边界。

原始恢复期快照日期为 `2026-05-05`。截至 `2026-05-08`，第一轮 Recovery 已完成，本文件现在作为 Phase 2 继续开发的入口快照。

## 2. 基本信息

- 快照日期：`2026-05-05`
- 最近更新：`2026-05-18`
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

- `T44: Research Reality Recovery Mode setup and evidence-gap ledger`

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
14. `T41` 已完成并通过 Captain `PASS` 收口；Milestone 2K 已正式由 gate review 关闭，但当前唯一任务 `T42` 只允许做 docs-only 的 Background / Related Work scaffold 与 method-positioning calibration，不得触碰代码、`runs/`、`artifacts`、benchmark、`.tflite`、真板或阶段结论文档。

## 13. T44 后的拟议路线图（非当前任务）

T44 结束后，基于 recovery 结论形成的下一轮 bounded task 建议是：

1. `T45`：paper-grade benchmark expansion protocol lock and gap audit
2. `T46`：multi-seed mechanism/intervention plan and trace pack
3. `T47`：paper ablation result-pack and material ledger

更后面的 milestone 方向可粗略分为：

- benchmark hardening
- deployment boundary boosters
- reproducibility and material pack
- paper re-open gate

这些都只是后续路线图，不是当前唯一任务，也不是已执行事实。
