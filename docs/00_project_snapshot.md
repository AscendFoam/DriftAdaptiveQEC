# Project Snapshot / Raw Idea Record

## 1. 快照目的

本文件对应 `docs/reference/AI_coding_workflow.md` 中的 `00_raw_idea.md` / 项目快照角色：用最短事实说明项目解决什么问题、为什么值得继续、当前最小可验证实验是什么，以及当前阶段边界。

原始恢复期快照日期为 `2026-05-05`。截至 `2026-05-08`，第一轮 Recovery 已完成，本文件现在作为 Phase 2 继续开发的入口快照。

## 2. 基本信息

- 快照日期：`2026-05-05`
- 最近更新：`2026-05-08`
- 当前分支：`main`
- 工作流依据：`docs/reference/AI_coding_workflow.md`
- 当前阶段：`Phase 2: Controlled Development`
- 当前决策状态：`Go`
- 当前唯一任务来源：`docs/04_task_board.md`

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

- `T18: TFLite export/runtime manifest and boundary smoke plan`

## 12. 快照结论

当前项目的核心问题已经从“有没有可靠治理层和入口”切换为“如何在不夸大完成度的前提下继续增强证据”。Phase 2 的默认策略是：

1. P4 benchmark protocol 与 bounded evidence 已完成第一轮受控增强，当前 gate 结论为 `Conditional`。
2. 训练链独立 bootstrap 已完成，当前优先补 `.tflite` export/runtime manifest 与 boundary smoke plan。
3. 对 repo noise 做有界 cleanup，而不是大规模重写历史产物。
